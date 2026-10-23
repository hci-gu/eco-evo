"""Regenerated nutrients from consumer losses (section 202).

The nutrient pools of section 172 were fed by new nitrogen (the
reservoir below the subsurface layer, section 180) and by the grazers'
excretion (``grazing_regeneration`` on a producer, the NDM grazers'
``excretion_regeneration``); everything a consumer lost to its own M1
left the system, so the model's production ran at an f-ratio of ~0.9
(new / total; field summer f < 0.1, resume 179.2, 201.3).

Two species keys close that leak:

``mortality_regeneration`` (0-1, default 0)
    Share of the group's M1 loss returned to the pools - the dead and
    respired matter remineralised within the water column. The detritus
    pool of the section-201 scratch test (k 0.4 /d at 10 C, sinking
    0.5 m/d) remineralised 8 of 8.4 g C m-2 yr-1 within days in its
    own layer, so the share is returned in the same tick, with no new
    state. Zooplankton and microzooplankton: 0.5 (ERSEM pe_R1, 179.2).
``egestion_regeneration`` (0-1, default 0, a grazing NDM)
    Share of its egesta (1 - growth_efficiency) x (1 -
    excretion_regeneration) of the intake that returns to the grazed
    producer's own pool, beside the excretion. Microzooplankton: 1.0.

The pools hold producer-biomass equivalents of N, and a tonne of
consumer is taken to carry the N of a tonne of phytoplankton (0.017 vs
0.018 g N per g WW). With the two-layer column a group's M1 return is
split between the mixed and the subsurface pool by where it fed this
tick (its intake from each producer, a grazer prey counted by that
grazer's own split); a group that fed on neither falls back to the
layers' thickness shares M(t) / bottom. Without the subsurface layer
everything goes to the one producer's pool. The return is added at the
end of the population step, after every group's growth, on both
engines, so it feeds the next tick's production. The growth budget is
unchanged: this is nitrogen, not the consumer's mass.
"""
import math

import numpy as np

from lib.world import nutrients


def species_share(params, key):
    """``key`` of a species entry as a share in [0, 1] (0 when unset)."""
    raw = (params or {}).get(key)
    if raw in (None, ""):
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number in [0, 1], got {raw!r}")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1], got {raw!r}")
    return value


def build(env):
    """Set ``env.regen`` (or None) and ``env._has_regeneration``.

    Needs the nutrient tables (``build_nutrient_tables``) and the grazing
    NDMs (``build_ndm_grazing``). Inert without nutrient pools.
    """
    env.regen = None
    env._has_regeneration = False
    env._regen_loss = {}
    shares = {}
    for fid in env.global_fg_order:
        share = species_share(env.fgs[fid].params, "mortality_regeneration")
        if share > 0.0:
            shares[fid] = share
    if not shares or not getattr(env, "_has_nutrients", False):
        return
    ex = getattr(env, "layer_exchange", None)
    if ex:
        mixed, deep = ex["mixed"], ex["subsurface"]
        cfg = env.nutrient_config
        hours = float(cfg["tick_hours"])
        bottom = float(cfg["subsurface"]["bottom_m"])
        fallback = np.array([
            float(nutrients.layer_thickness(cfg, t * hours + 0.5 * hours))
            / bottom for t in range(env.light_period)], dtype=np.float64)
    elif len(env.nutrient_ids) == 1:
        mixed, deep = env.nutrient_ids[0], None
        fallback = np.ones(env.light_period, dtype=np.float64)
    else:
        return
    env.regen = {"shares": shares, "mixed": mixed, "subsurface": deep,
                 "fallback": fallback,
                 "grazers": sorted({e["grazer"] for e in env.ndm_grazers})}
    env._has_regeneration = True


def record_m1(env, fg_id, before, after):
    """Remember ``fg_id``'s M1 loss per cell this tick (a no-op unless the
    group returns a share of it)."""
    if getattr(env, "_has_regeneration", False) and fg_id in env.regen["shares"]:
        env._regen_loss[fg_id] = (np.asarray(before, dtype=np.float64)
                                  - np.asarray(after, dtype=np.float64))


def _phi(env, prey_id, memo):
    """Mixed-layer share of what a predator gets from ``prey_id``, or None
    when the prey is no plankton the split can follow."""
    r = env.regen
    if prey_id == r["mixed"]:
        return 1.0
    if prey_id == r["subsurface"]:
        return 0.0
    if prey_id in r["grazers"]:
        return fraction(env, prey_id, memo)
    return None


def fraction(env, fg_id, memo=None):
    """Mixed-layer share of ``fg_id``'s feeding this tick (section 202)."""
    memo = {} if memo is None else memo
    if fg_id in memo:
        return memo[fg_id]
    r = env.regen
    if r["subsurface"] is None:
        memo[fg_id] = 1.0
        return 1.0
    num = den = 0.0
    if env.fgs[fg_id].is_decision_maker:
        rows = getattr(env, "_tick_dm_intake", None)
        if rows is not None:
            row = rows[env.dm_ids.index(fg_id)]
            for j, pid in enumerate(env.global_fg_order):
                value = float(row[j])
                if value <= 0.0:
                    continue
                phi = _phi(env, pid, memo)
                if phi is None:
                    continue
                num += value * phi
                den += value
    else:
        for (grazer, prey), value in getattr(env, "_tick_ndm_intake", {}).items():
            if grazer != fg_id or value <= 0.0:
                continue
            phi = 1.0 if prey == r["mixed"] else 0.0 if prey == r["subsurface"] else None
            if phi is None:
                continue
            num += value * phi
            den += value
    if den > 0.0:
        f = num / den
    else:
        from lib.environments.ecosystem_env import interactions
        f = float(r["fallback"][interactions.light_index(env)])
    memo[fg_id] = f
    return f


def apply(env):
    """Return this tick's recorded M1 shares to the pools, then clear."""
    if not getattr(env, "_has_regeneration", False) or not env._regen_loss:
        env._regen_loss = {}
        return
    r = env.regen
    memo = {}
    mixed = np.zeros(env.fgs[r["mixed"]].biomass.shape, dtype=np.float64)
    deep = np.zeros_like(mixed)
    for fg_id in env.global_fg_order:
        loss = env._regen_loss.get(fg_id)
        if loss is None:
            continue
        back = r["shares"][fg_id] * loss
        f = fraction(env, fg_id, memo)
        mixed += f * back
        deep += (1.0 - f) * back
    top = env.fgs[r["mixed"]]
    top.energy_reserve = (top.energy_reserve + mixed).astype(env.dtype, copy=False)
    if r["subsurface"] is not None:
        low = env.fgs[r["subsurface"]]
        low.energy_reserve = (low.energy_reserve + deep).astype(env.dtype, copy=False)
    env._regen_loss = {}
