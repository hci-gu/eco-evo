"""Reward ledger of a migrating decision maker (section 246).

A DM with ``outside_density`` exchanges biomass and reserve with the
outside every tick (``open_pool.migrate``), and so does an open-population
group through its recruit import and emigration (section 250,
``stages.exchange``). Its reward must not score that
migration, or the seasonal swing (x ~20) and the relaxation towards the
same outside target would swamp what the policy does. The ledger keeps
local indices of the group's biomass, reserve and energy,

    Q*_{t+1} = Q*_t (Q_{t+1} - X_t) / Q_t,   Q*_0 = Q_0,

with ``X_t`` the tick's net exchange of that quantity
(``env.tick_exchange_biomass`` / ``_reserve`` / ``_energy``): the
multiplicative change the group made itself. Every reward form reads the
indices instead of the raw totals for a migrating group (the log-energy
reward, the legacy biomass/reserve reward, population stability and the
survival tick); every other group is passed through untouched. The GPU
rollout keeps the same indices (``RolloutRunner``).

``--local_reward`` (per cell) is refused for a migrating group: its source
tracking has no cell for the outside.
"""


def migrating(env, fids):
    """The DMs among ``fids`` that exchange with the outside: the migrating
    ones (section 246) and the open-population groups with a recruit import
    or emigration (section 250, ``stages.exchange``)."""
    open_ids = set(getattr(env, "open_dm_ids", ()) or ())
    open_ids |= set(getattr(env, "stage_exchange_ids", ()) or ())
    return [fid for fid in fids if fid in open_ids]


def check_modes(env, fids, local_reward=None):
    """Refuse the reward mode that cannot carry the ledger."""
    moving = migrating(env, fids)
    if moving and local_reward is not None:
        raise ValueError(f"{moving} exchange with the outside (outside_density or an open "
                         "population): --local_reward has no source cell for the outside and "
                         "is not supported (sections 246, 250)")


_BOOKS = ("tick_exchange_biomass", "tick_exchange_reserve", "tick_exchange_energy")


class EnergyLedger:
    """Per-group local indices of (biomass, reserve, energy)."""

    def __init__(self, env, start):
        """``start`` maps fid -> (b0, r0, e0)."""
        self._env = env
        moving = migrating(env, list(start))
        self._prev = {fid: [float(v) for v in start[fid]] for fid in moving}
        self._index = {fid: [float(v) for v in start[fid]] for fid in moving}

    def tracks(self, fid):
        return fid in self._index

    def update(self, fid, b_now, r_now, e_now):
        """This tick's (biomass, reserve, energy) as the reward sees them;
        call once per tick, after the step."""
        if fid not in self._index:
            return b_now, r_now, e_now
        now = (float(b_now), float(r_now), float(e_now))
        for k, book in enumerate(_BOOKS):
            x = float(getattr(self._env, book, {}).get(fid, 0.0))
            if self._prev[fid][k] > 0.0:
                self._index[fid][k] *= (now[k] - x) / self._prev[fid][k]
            self._prev[fid][k] = now[k]
        return tuple(self._index[fid])

    def final(self, fid, b_now, r_now, e_now):
        """The end-of-rollout values (after the last ``update``)."""
        return tuple(self._index[fid]) if fid in self._index else (b_now, r_now, e_now)


__all__ = ["EnergyLedger", "check_modes", "migrating"]
