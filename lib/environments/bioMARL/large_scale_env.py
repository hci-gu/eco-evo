import numpy as np
import scipy.special
import scipy.stats

from .actions import LowEnergyMortality, ReproduceAction
from .utilities import get_rnd_gen, rnd_init_h

np.set_printoptions(suppress=True, precision=3)
np.seterr(all="raise", under="warn")


class LargeScaleRLEnv:
    def __init__(
        self,
        grid_shape=(50, 50),
        functional_groups=10,
        h=None,
        energy_table=None,
        padding=1,
        migrate_fg=None,
        feed_fg=None,
        reproduce_fg: ReproduceAction = None,
        low_energy_mortality: LowEnergyMortality = None,
        carrying_capacity=None,  # (fg, x, y)
        max_energy=100,
        seed=None,
        mask=None,
    ):
        self.grid_shape = grid_shape
        self.functional_groups = functional_groups
        self.seed = seed
        self.rs = get_rnd_gen(seed=seed)
        self.padding = padding

        if h is None:
            h = rnd_init_h(self.grid_shape, self.functional_groups, seed=self.seed)

        if self.padding:
            n = np.zeros(
                (
                    self.functional_groups,
                    self.grid_shape[0] + self.padding * 2,
                    self.grid_shape[1] + self.padding * 2,
                )
            )
            n[:, self.padding : -self.padding, self.padding : -self.padding] = h
        self._h = n if self.padding else h

        if energy_table is not None:
            pad_width = ((0,), (self.padding,), (self.padding,))
            self._energy_table = np.pad(energy_table, pad_width=pad_width)
        else:
            self._energy_table = np.ones(self._h.shape) * max_energy * (self._h > 0)

        self.low_energy_mortality = low_energy_mortality

        if carrying_capacity is not None:
            n = np.zeros(
                (
                    self.functional_groups,
                    self.grid_shape[0] + self.padding * 2,
                    self.grid_shape[1] + self.padding * 2,
                )
            )
            n[:, self.padding : -self.padding, self.padding : -self.padding] = (
                carrying_capacity
            )
        else:
            n = None
        self.carrying_capacity = n
        self.max_energy = max_energy

        self.set_mask(mask)
        self.mask = mask.copy() if mask is not None else None
        self.migrate_fg = migrate_fg
        self.feed_fg = feed_fg
        self.reproduce_fg = reproduce_fg
        self.reproduce_fg.reset_counter()
        self.history = {"pop_sizes": [], "avg_energy": [], "range_sizes": []}

        # Choose hide_n_feed version.
        self.hide_n_feed = self.hide_n_feed_no_loop

    def set_mask(self, mask):
        dispersal_masks = []
        if mask is None:
            mask = np.zeros(self.grid_shape)
        mask = np.pad(mask, self.padding, mode="constant", constant_values=True)
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                dispersal_masks.append(1 - mask[i : i + 3, j : j + 3].copy())
        self.dispersal_masks = dispersal_masks

    # MIGRATION STEP
    def old_migrate(self, vec, current_h=None, current_e=None):
        # current distribution of FGs
        if current_h is None:
            current_h = self.h + 0
        if current_e is None:
            current_e = self.energy_table + 0
        new_h = np.zeros(current_h.shape)
        new_e = np.zeros(current_h.shape)
        cell_counter = 0
        for x in range(0, self.grid_shape[0]):
            for y in range(0, self.grid_shape[1]):
                xy_mask = self.dispersal_masks[cell_counter]
                # loop over FGs
                for i, fg in enumerate(self.migrate_fg):
                    h_i = current_h[i, x + self.padding, y + self.padding]
                    m = (
                        fg.migration_action(
                            m=vec[cell_counter, i, :2],
                            s=float(vec[cell_counter, i, 2]),
                            mask=xy_mask,
                        )
                        * h_i
                    )
                    new_h[i, (x) : (x + 3), (y) : (y + 3)] += m
                    # energy costs
                    en_i = current_e[i, x + self.padding, y + self.padding]
                    new_e[i, (x) : (x + 3), (y) : (y + 3)] += m * (
                        en_i + fg.migration_energy_cost
                    )
                cell_counter += 1

        new_h_r = np.floor(new_h + 1e-6)
        new_h_r = self.trim_array(new_h)
        # TODO: fix rounding error
        delta_h = (current_h.sum(axis=(1, 2)) - new_h_r.sum(axis=(1, 2))).astype(int)
        # print("delta_h", delta_h)

        # get indices of non zeros
        for fg_i in range(self.functional_groups):
            x, y = np.where((current_h[fg_i] + new_h_r[fg_i]) > 0)
            if x.size:
                r = len(x)
                rnd_i = self.rs.choice(range(r), size=delta_h[fg_i], replace=True)
                unique_indx, counts = np.unique(rnd_i, return_counts=True)
                new_h_r[fg_i, x[unique_indx], y[unique_indx]] += counts

        # print(new_h_r.sum() == current_h.sum())

        den = new_h + 0
        den[den == 0] = 1.0
        new_energy_tbl_r = new_e / den
        return new_h_r, new_energy_tbl_r

    @staticmethod
    def vec_migrate(actions, habitat, masks, max_speed=1.0):
        #     print("actions.shape", actions.shape)
        assert actions.shape[-1] == 3, "actions should have last shape=3."
        actions = actions.copy()
        action_shape = actions.shape
        actions = actions.reshape(-1, 3)
        # Sigma clipping, in order to behave the same as original.
        actions[:, 2] = np.maximum(actions[:, 2], 0.1)
        # Extract m and s and extra dimensions for the broadcasting to work.
        m = actions[..., :2, None]
        m *= max_speed
        s = actions[..., 2, None, None]
        # x is used instead of a mesh grid
        x = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
        pdf_x = scipy.stats.norm.logpdf(x=x[None, None], loc=m, scale=np.sqrt(s))
        #
        # Here we do the mesh grid operation. We compute the "outer addition" of the
        # columns of x. This is because the logpdf of the multivariate normal is
        # just the additions of the marginal logpdfs since they are independent.
        pdf2 = pdf_x[..., 1, :][..., None] + pdf_x[..., 0, :][..., None, :]
        # print("pdf2.shape:", pdf2.shape)
        s = scipy.special.softmax(pdf2, axis=(-1, -2)).reshape(*action_shape[:-1], 3, 3)
        # print("s.shape", s.shape, "masks.shape", masks.shape)
        if masks is not None:
            s = s * masks
            # Only normalize over the last two axis ie 3x3 grid.
            sums = s.sum(axis=(-1, -2))
            sums[sums == 0] = 1  # To avoid division with 0.
            s = s / sums[..., None, None]
        return s

    def migrate(self, vec, current_h=None, current_e=None):
        DEBUG = False
        if DEBUG:
            # Just to be able to compare with the old implementation.
            _state = self.rs.bit_generator.state
        # current distribution of FGs
        if current_h is None:
            current_h = self.h + 0
        if current_e is None:
            current_e = self.energy_table + 0

        new_h = np.zeros(current_h.shape)
        new_e = np.zeros(current_h.shape)
        dispersal_masks = np.array(self.dispersal_masks)

        for i, fg in enumerate(self.migrate_fg):
            # Vectorize `vec_migrate` computation.
            hvm = (
                self.vec_migrate(
                    vec[:, i], None, dispersal_masks, max_speed=fg.max_speed
                )
                * current_h[i, 1:-1, 1:-1].flatten()[..., None, None]
            )

            # Reshape `hvm` into a grid form for broadcasting.
            hvm_grid = hvm.reshape(self.grid_shape[0], self.grid_shape[1], 3, 3)
            # Do the same with energy.
            energy_grid = hvm_grid * current_e[i, 1:-1, 1:-1][..., None, None]
            # Deduct migration costs.
            energy_grid += np.einsum(
                "ijkl, kl -> ijkl", hvm_grid, fg.migration_energy_cost
            )

            # Update `new_h` and `new_e` with broadcasting
            for dx in range(3):
                for dy in range(3):
                    # Update the habitat grid
                    _slice = (
                        i,
                        slice(dx, dx + self.grid_shape[0]),
                        slice(dy, dy + self.grid_shape[1]),
                    )

                    new_h[_slice] += hvm_grid[:, :, dx, dy]
                    new_e[_slice] += energy_grid[:, :, dx, dy]

        new_h_r = np.floor(new_h + 1e-6)
        new_h_r = self.trim_array(new_h)

        # TODO: fix rounding error
        delta_h = (current_h.sum(axis=(1, 2)) - new_h_r.sum(axis=(1, 2))).astype(int)
        assert (delta_h == 0).all(), f"Hmmm {delta_h}"

        # get indices of non zeros
        for fg_i in range(self.functional_groups):
            x, y = np.where((current_h[fg_i] + new_h_r[fg_i]) > 0)
            if x.size:
                r = len(x)
                rnd_i = self.rs.choice(range(r), size=delta_h[fg_i], replace=True)
                unique_indx, counts = np.unique(rnd_i, return_counts=True)
                new_h_r[fg_i, x[unique_indx], y[unique_indx]] += counts

        den = new_h + 0
        den[den == 0] = 1.0
        new_energy_tbl_r = new_e / den

        if DEBUG:
            new_state = self.rs.bit_generator.state
            self.rs.bit_generator.state = _state
            _h, _e = self.old_migrate(vec, current_h, current_e)
            diff_h = _h - new_h_r
            diff_e = _e - new_energy_tbl_r
            assert np.allclose(diff_h, 0) and np.allclose(diff_e, 0), (
                np.linalg.norm(diff_h),
                np.linalg.norm(diff_e),
            )
            self.rs.bit_generator.state = new_state

        return new_h_r, new_energy_tbl_r

    # FEEDING STEP
    def old_hide_n_feed(self, vec, current_h=None, current_e=None):
        if current_h is None:
            current_h = self.h + 0
        if current_e is None:
            current_e = self.energy_table + 0
        new_h = np.zeros(current_h.shape)
        new_e = np.zeros(current_h.shape)
        cell_counter = 0
        for x in range(self.grid_shape[0]):
            for y in range(self.grid_shape[1]):
                h_ini = current_h[:, x + self.padding, y + self.padding]
                e_ini = current_e[:, x + self.padding, y + self.padding]

                h_tmp, e_tmp = self.feed_fg.feeding_hiding_action(
                    cell_populations=h_ini,
                    fg_actions=vec[cell_counter, :, :],
                    energy_level=e_ini,
                )

                new_h[:, x + self.padding, y + self.padding] = h_tmp
                new_e[:, x + self.padding, y + self.padding] = e_tmp
                cell_counter += 1

        return new_h, new_e

    def hide_n_feed_no_loop(self, vec, current_h=None, current_e=None):
        if current_h is None:
            current_h = self.h
        if current_e is None:
            current_e = self.energy_table

        _slice = slice(self.padding, -self.padding)
        _slice = (slice(None), _slice, _slice)
        h = current_h[_slice]
        e = current_e[_slice]

        _h, _e = self.feed_fg.grid_feeding_hiding_action(vec, h, e)
        new_h = np.zeros_like(current_h)
        new_h[_slice] = _h
        new_e = np.zeros_like(current_e)
        new_e[_slice] = _e

        return new_h, new_e

    def reproduce(self, current_h=None, current_e=None):
        if current_h is None:
            current_h = self.h + 0
        if current_e is None:
            current_e = self.energy_table + 0
        h = self.reproduce_fg.reproduce_action(current_h)

        current_e_tmp = current_e + 0
        delta_h = h - current_h
        delta_h_p = delta_h[h > 0] / h[h > 0]
        e_tmp = self.max_energy * delta_h_p + current_e[h > 0] * (1 - delta_h_p)
        current_e_tmp[h > 0] = e_tmp
        # new born with max energy only for primary producers
        current_e[self.reproduce_fg.seed_bank_fg > 0] = current_e_tmp[
            self.reproduce_fg.seed_bank_fg > 0
        ]

        return h, current_e

    def die(self, current_h=None, current_e=None):
        if current_h is None:
            current_h = self.h + 0
        if current_e is None:
            current_e = self.energy_table + 0

        if self.low_energy_mortality is not None:
            delta = self.low_energy_mortality.fg_mortality_function(energy=current_e)

            new_h = np.round(current_h * (1 - delta)).astype(int)
        else:
            new_h = current_h.copy()

        if self.carrying_capacity is not None:
            new_h = np.round(np.minimum(new_h, self.carrying_capacity)).astype(int)

        return new_h

    def observe(self):
        pass

    def update_history(self):
        self.history["pop_sizes"].append(self.total_pop)
        self.history["avg_energy"].append(self.avg_energy)
        self.history["range_sizes"].append(self.geo_range)

    def step(self, action):
        # print(f"Max energy {self.energy_table.max(axis=(-2,-1))}")
        current_h = self.h + 0
        current_e = (
            self.energy_table
            - self.feed_fg.get_base_energy_cost[:, np.newaxis, np.newaxis]
        )
        if action is not None:
            new_h, new_e = self.migrate(
                action["migrate"], current_h=current_h, current_e=current_e
            )
            DEBUG = False
            if DEBUG:
                h_2, e_2 = self.hide_n_feed2(
                    action["hide_n_feed"], current_h=new_h, current_e=new_e
                )
            new_h, new_e = self.hide_n_feed(
                action["hide_n_feed"], current_h=new_h, current_e=new_e
            )
            if DEBUG:
                print(
                    f"h diff: {np.linalg.norm(h_2 - new_h):.3f}, "
                    f"e diff: {np.linalg.norm(e_2 - new_e):.3f}"
                )
            new_h, new_e = self.reproduce(current_h=new_h, current_e=new_e)
            new_h = self.die(current_h=new_h, current_e=new_e)
        else:
            new_h, new_e = self.reproduce(current_h=current_h, current_e=current_e)

        self.reset_h(new_h)
        self.reset_e(new_e)
        self.update_history()
        # Make a very simple energy related death criteria, if energy < 0, then die.
        self.h[self.energy_table <= 0] = 0
        self.energy_table[self.energy_table < 0] = 0

    def reset(self):
        pass

    def render(self):
        pass

    def agent_iter(self):
        pass

    def reset_h(self, h):
        self._h = h

    def reset_e(self, e):
        self._energy_table = e

    @property
    def h(self):
        return self._h

    @property
    def energy_table(self):
        return self._energy_table

    @property
    def total_pop(self):
        return np.sum(self._h, axis=(1, 2)).astype(int)

    @property
    def geo_range(self):
        return np.sum(self._h > 0, axis=(1, 2)).astype(int)

    @property
    def avg_energy(self):
        en = self._energy_table + 0
        en[en == 0] = np.nan
        return np.nanmean(en, axis=(1, 2))

    def adapt_action(action):
        pass

    def trim_array(self, h):
        """Trim the (habitat) arrays so that they are all integers but the
        fractions are distributed randomly over the other cells."""
        bs = []
        for fg in h:
            # To be sure we push the number that should be an int but it is too small
            # due to rounding errors, above the floor.
            # fg = (fg + 1e-6).ravel()
            # We skip this, since it is very likely to be chosen anway.
            fg = fg.ravel()
            b = np.floor(fg)
            fracs = fg - b
            n_extra = int(np.round(fracs.sum()))
            if n_extra:  # Only do this if there is something to distribute.
                b[
                    self.rs.choice(
                        fracs.size, p=fracs / fracs.sum(), replace=False, size=n_extra
                    )
                ] += 1
                # print((b.reshape(fg.shape)- np.floor(fg)).sum())
            bs.append(b)

        bs = np.array(bs).reshape(h.shape)
        # print((bs - h).sum(axis=(1,2)))
        return bs
