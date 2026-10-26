"""Benthic refuge with exchange (resume 223.11-223.12).

A non decision maker with ``exposed_share`` phi in (0, 1) splits its
biomass B per cell into an accessible part A and a refuge R = B - A:
infauna in the surface layer versus deep-burrowing animals and the parts
(arm tips, siphons) that regrow out of reach. A rides in the group's
unused ``energy_reserve`` slot (tonnes), as a producer's nutrient pool
does, so B stays the group's biomass everywhere (plots, K, trawl, totals).

Per tick:
- predation removes from A only, and the Holling term, the harvest cap
  and the observations see A instead of B;
- trawl, M1, the habitat mask and the extinction sweep scale A with B;
- the exchange relaxes A towards phi B,
  A <- phi B + (A - phi B) exp(-k), k = ``refuge_exchange_rate`` per tick
  (exact for any tick length; the loader rescales k as a flux);
- then growth is logistic per compartment on its own capacity (variant
  S, Krivan 1998): A on phi K, R on (1 - phi) K; at A = phi B the sum is
  exactly the group's logistic, so the derived growth rate holds.
  Exchange before growth makes the fast limit (k -> infinity) exactly
  today's fixed accessible fraction phi with a logistic on B.

research_notes/"Refugium och utbyte i bentos" (Swedish letters).
"""
import numpy as np


def settings(fg):
    """``(phi, k)`` for a refuge group, else None."""
    phi = getattr(fg, "exposed_share", None)
    if phi is None or getattr(fg, "is_decision_maker", False):
        return None
    return float(phi), float(getattr(fg, "refuge_exchange_rate", 0.0) or 0.0)


def build_tables(env):
    """``env.refuge_index`` (prey columns in ``global_fg_order``) and
    ``env._has_refuge``."""
    env.refuge_index = np.array(
        [j for j, fid in enumerate(env.global_fg_order)
         if settings(env.fgs[fid]) is not None], dtype=np.int64)
    env._has_refuge = bool(env.refuge_index.size)


def accessible_rows(env, prey_biomass):
    """``prey_biomass`` [N_all, H, W] with every refuge group's row
    replaced by its accessible part A. Unchanged (no copy) without one."""
    if not getattr(env, "_has_refuge", False):
        return prey_biomass
    out = prey_biomass.copy()
    for j in env.refuge_index:
        fg = env.fgs[env.global_fg_order[j]]
        out[j] = np.clip(fg.energy_reserve, 0.0, fg.biomass)
    return out


def grow(fg, rate, capacity, seed_delta, dtype):
    """The exchange, then logistic growth per compartment. Updates
    ``fg.biomass`` and ``fg.energy_reserve`` (A) in place."""
    phi, k = settings(fg)
    b = fg.biomass
    a = np.clip(fg.energy_reserve, 0.0, b)
    a = phi * b + (a - phi * b) * np.float32(np.exp(-k))
    r = b - a
    cap_a = phi * capacity + 1e-9
    cap_r = (1.0 - phi) * capacity + 1e-9
    a_new = a + rate * a * (1.0 - a / cap_a) + phi * seed_delta
    r_new = r + rate * r * (1.0 - r / cap_r) + (1.0 - phi) * seed_delta
    a_new = np.clip(a_new, 0.0, phi * capacity)
    r_new = np.clip(r_new, 0.0, (1.0 - phi) * capacity)
    b_new = a_new + r_new
    fg.biomass = b_new.astype(dtype, copy=False)
    fg.energy_reserve = np.clip(a_new, 0.0, b_new).astype(dtype, copy=False)
