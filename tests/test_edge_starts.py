"""Section 308.6: edge starts of the GPU training worlds (train_gpu --edge-starts)."""
import pytest

torch = pytest.importorskip("torch")


def _trainer(edge, depleted=0.0, migration=True, warmup=0, worlds=8):
    from lib.gpu.config import EnvironmentBuilder, ProjectSpec
    from lib.gpu.trainer import TensorARSTrainer
    torch.set_num_threads(1)
    spec = ProjectSpec(EnvironmentBuilder(project_path="mareld2.yaml", grid=(12, 12),
                                          migration=migration))
    extra = {} if edge is None else {"edge_starts": edge}
    return TensorARSTrainer(spec, device="cpu", execution="eager", n_deltas=2, hidden_dim=7,
                            worlds=worlds, depleted_patches=depleted, warmup_ticks=warmup,
                            **extra)


def test_closed_edge_rows_are_the_mobile_dms_with_closed_edges():
    on = _trainer(0.0)
    fish = {"benthivorous_fish", "mackerel", "planktivorous_fish", "predatory_fish", "whiting"}
    assert {on.model.ids[r] for r in on.edge_rows} == fish
    off = _trainer(0.0, migration=False)
    rows = {off.model.ids[r] for r in off.edge_rows}
    # Migration off closes every edge; zooplankton cannot move at all.
    assert fish | {"porpoises", "seals"} == rows


def test_share_zero_leaves_the_worlds_unchanged():
    # The default (no edge starts) and an explicit 0 draw the same worlds.
    a, b = _trainer(0.0, depleted=0.5), _trainer(None, depleted=0.5)
    a.refresh_worlds()
    b.refresh_worlds()
    torch.testing.assert_close(a.world_biomass, b.world_biomass, rtol=0, atol=0)
    torch.testing.assert_close(a.world_depletion, b.world_depletion, rtol=0, atol=0)
    assert (a.world_edge == -1).all()
    assert not _trainer(0.0).depletes


def test_edge_worlds_keep_totals_and_move_only_the_drawn_groups_to_their_place():
    on, off = _trainer(1.0), _trainer(0.0)
    on.refresh_worlds()
    off.refresh_worlds()
    m = on.model
    torch.testing.assert_close(on.world_biomass.sum(-1), off.world_biomass.sum(-1), rtol=1e-5, atol=1e-3)
    moved = on.world_edge >= 0
    assert moved.any() and (~moved).any()
    d = on.spawner.edge_distances().reshape(8, -1)
    for k, g in enumerate(on.edge_rows):
        for w in range(on.worlds):
            new, old = on.world_biomass[w, g].double(), off.world_biomass[w, g].double()
            if not bool(moved[w, k]):
                torch.testing.assert_close(new, old)
                continue
            dist = d[int(on.world_edge[w, k])].double()
            # Mixed at m >= 0.5 towards exp(-d / lambda), lambda <= 3 cells.
            assert (new * dist).sum() / new.sum() < (old * dist).sum() / old.sum()
    others = [g for g in range(m.G) if g not in on.edge_rows]
    torch.testing.assert_close(on.world_biomass[:, others], off.world_biomass[:, others])


def test_prey_drawn_down_under_the_bands_in_benthic_groups_only():
    t = _trainer(1.0)
    t.refresh_worlds()
    f = t.world_depletion
    assert t.depletes
    assert float(f.min()) >= 0.3 - 1e-6 and float(f.max()) <= 1.0
    no_move = (t.world_edge < 0).all(-1)
    assert (f[no_move] == 1.0).all()
    assert (f < 0.999).any()
    keys = torch.zeros(t.worlds, dtype=torch.int64)
    b = t.world_biomass
    r = t.spawner.reserves(b, keys)
    nb, _ = t.model.deplete(b, r, f)
    for j in range(t.model.G):
        if not bool(t.model.benthic_mask[0, j, 0]):
            torch.testing.assert_close(nb[:, j], b[:, j])


def test_edge_starts_with_patches_multiply_the_factors():
    both, patches = _trainer(1.0, depleted=1.0), _trainer(0.0, depleted=1.0)
    both.refresh_worlds()
    patches.refresh_worlds()
    assert (both.world_depletion <= patches.world_depletion + 1e-6).all()


def test_same_world_same_edge_start():
    a, b = _trainer(0.5), _trainer(0.5)
    a.refresh_worlds(epoch=3)
    b.refresh_worlds(epoch=3)
    torch.testing.assert_close(a.world_biomass, b.world_biomass, rtol=0, atol=0)
    assert torch.equal(a.world_edge, b.world_edge)


@pytest.mark.parametrize("warmup", [0, 3])
def test_train_step_logs_the_places(warmup):
    t = _trainer(0.5, depleted=0.5, warmup=warmup, worlds=2)
    t.train_step(n_eval_ticks=3)
    places = t.metrics()["edge_start_places"]
    assert len(places) == 2 and len(places[0]) == len(t.edge_rows)
    assert all(-1 <= p <= 7 for row in places for p in row)


def test_edge_points_lie_on_their_side_or_corner():
    t = _trainer(0.0)
    s = t.spawner
    H1, W1 = s.H - 1, s.W - 1
    place = torch.arange(8, device=s.device)
    pos = torch.full((8,), 0.25, dtype=s.wide, device=s.device)
    r, c = s.edge_points(place, pos)
    expect = [(0, 0.25 * W1), (0.25 * H1, W1), (H1, 0.25 * W1), (0.25 * H1, 0),
              (0, 0), (0, W1), (H1, 0), (H1, W1)]
    for k, (er, ec) in enumerate(expect):
        assert float(r[k]) == pytest.approx(er) and float(c[k]) == pytest.approx(ec)


def test_short_bands_are_dense_patches_at_points_along_the_edge():
    # Section 308.8: a side start is a patch at a drawn point, not a line.
    t = _trainer(0.0, worlds=64)
    s = t.spawner
    t.refresh_worlds()
    keys = torch.arange(64, dtype=torch.int64, device=s.device)
    b = t.world_biomass
    out, _, places = s.edge_starts(b, keys, 1.0, t.edge_rows, length=(0.7, 0.7), mix=(1.0, 1.0))
    sides = 0
    columns = set()
    for k, g in enumerate(t.edge_rows):
        for w in range(64):
            p = int(places[w, k])
            if p < 0:
                continue
            cell = out[w, g].double()
            top = int(cell.argmax())
            # A point between two cells splits its top between them (~0.23 each).
            assert float(cell.max() / cell.sum()) > 0.2
            row, col = divmod(top, s.W)
            assert row in (0, s.H - 1) or col in (0, s.W - 1)
            if p in (0, 2):
                sides += 1
                columns.add(col)
    assert sides > 0 and len(columns) > 2
