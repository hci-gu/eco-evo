"""Neighbour (and own-cell) weight with memory and persistence (reading C).

Reading B (section 273, research_notes/Observabilitetsparen) counts only the
current 6 h track:
    w_B = P(track enters a given neighbour) x E[1 - exp(-n) | entered].
Reading C lets the observer remember earlier visits, discounted by how much of
the target's distribution there has persisted since:

    w_C = E over tracks and neighbours of
          sum_i rho(age_i) p_i prod_{j<i} (1 - p_j)

over the visits i to that neighbour within the memory span M, ordered from the
most recent (age = time from the end of the visit to the decision), with
p_i = 1 - exp(-n_i) the chance the visit met the target (n_i as in reading B,
from the path inside that visit) and rho(age) the target's persistence. The
most recent visit that met the target is what the observer knows. With M = one
tick and rho = 1 this is reading B (checked by --check).

The track is a correlated random walk at constant speed L / 6 h, calibrated as
in reading B to the median 6 h net displacement D, run for M ticks before the
decision; the decision cell is where it ends. Optional TETHER (km): a home-range
pull towards the start that keeps the multi-day spread at the measured daily
displacement (0 = free CRW). The script prints the implied 24 h median net
displacement so it can be held against the field value.

Persistence: rho(t) = a exp(-t / T1) + (1 - a) exp(-t / T2) (t in hours).
"Within the current tick" ages count like any other (rho < 1 even there),
unless --tick-perfect (reading B's implicit rho = 1 for the current tick).

Encounter inputs (densities, units, radii, observers' L and D, who eats whom)
are imported from research_notes/Observabilitetsparen/scripts/neighbour_weight.py
(central case) so readings B and C differ only in memory and persistence.

Run: python3 -I memory_weight.py --check
     python3 -I memory_weight.py --pair porpoises planktivores --memory 8 \
         --rho 0.5 12 168
"""
import argparse
import importlib.util
import sys

import numpy as np

NW_PATH = ('/home/simon/eco-evo/research_notes/Observabilitetsparen/scripts/'
           'neighbour_weight.py')
_argv, sys.argv = sys.argv, ['neighbour_weight.py', 'central']
_spec = importlib.util.spec_from_file_location('nw', NW_PATH)
nw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nw)
sys.argv = _argv

TICK_H = 6.0
STEPS_PER_TICK = 180          # 2-min steps
N_TRACKS = 2000
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _rng(*key):
    return np.random.default_rng(abs(hash(key)) % (2 ** 32))


def turn_sd(L, D):
    """Turning sd (per 2-min step) giving median 6 h net displacement D."""
    if D >= 0.97 * L:
        return 0.0
    rng = _rng('sd', L, D)
    step = L / STEPS_PER_TICK
    lo, hi = 0.0, 3.0
    for _ in range(30):
        sd = 0.5 * (lo + hi)
        ang = rng.random((800, 1)) * 2 * np.pi + np.cumsum(
            rng.normal(0.0, sd, (800, STEPS_PER_TICK)), axis=1)
        net = np.hypot((np.cos(ang) * step).sum(1), (np.sin(ang) * step).sum(1))
        if np.median(net) > D:
            lo = sd
        else:
            hi = sd
    return sd


_TRACK_CACHE = {}
_VISIT_CACHE = {}


def tracks(L, D, ticks, tether=0.0):
    """Positions (N, steps+1, 2) in km; start uniform in [0, 1]^2."""
    key = (L, D, ticks, tether)
    if key in _TRACK_CACHE:
        return _TRACK_CACHE[key]
    rng = _rng('trk', *key)
    sd = turn_sd(L, D)
    n_steps = ticks * STEPS_PER_TICK
    step = L / STEPS_PER_TICK
    pos = np.empty((N_TRACKS, n_steps + 1, 2), dtype=np.float32)
    pos[:, 0] = rng.random((N_TRACKS, 2))
    origin = pos[:, 0].copy()
    head = rng.random(N_TRACKS) * 2 * np.pi
    turns = rng.normal(0.0, sd, (N_TRACKS, n_steps)).astype(np.float32)
    for k in range(n_steps):
        head = head + turns[:, k]
        if tether > 0:
            # Turn a little towards the start when far from it (home range).
            off = pos[:, k] - origin
            dist = np.hypot(off[:, 0], off[:, 1])
            home = np.arctan2(-off[:, 1], -off[:, 0])
            pull = np.clip(dist / tether, 0.0, 1.0) * 0.05
            dang = np.angle(np.exp(1j * (home - head)))
            head = head + pull * dang
        pos[:, k + 1, 0] = pos[:, k, 0] + step * np.cos(head)
        pos[:, k + 1, 1] = pos[:, k, 1] + step * np.sin(head)
    _TRACK_CACHE[key] = pos
    return pos


def net_24h(L, D, tether=0.0):
    pos = tracks(L, D, 4, tether)
    return float(np.median(np.hypot(*(pos[:, -1] - pos[:, 0]).T)))


def visits(L, D, ticks, tether=0.0, own=False):
    """Per track and target cell: list of (age_h, path_km) visits, newest first.

    Target cells: the four neighbours of the cell the track ends in (or, with
    own=True, that cell itself). Steps are 2 min; a visit is a run of steps
    inside the cell; age = hours from its last step to the decision.
    """
    vkey = (L, D, ticks, tether, own)
    if vkey in _VISIT_CACHE:
        return _VISIT_CACHE[vkey]
    pos = tracks(L, D, ticks, tether)
    cells = np.floor(pos[:, 1:]).astype(np.int32)
    end = cells[:, -1]
    step_km = L / STEPS_PER_TICK
    step_h = TICK_H / STEPS_PER_TICK
    n_steps = cells.shape[1]
    out = []
    targets = ((0, 0),) if own else NBRS
    for t in range(N_TRACKS):
        for dx, dy in targets:
            inside = ((cells[t, :, 0] == end[t, 0] + dx)
                      & (cells[t, :, 1] == end[t, 1] + dy))
            if not inside.any():
                out.append([])
                continue
            idx = np.flatnonzero(inside)
            breaks = np.flatnonzero(np.diff(idx) > 1)
            starts = np.concatenate(([idx[0]], idx[breaks + 1]))
            ends = np.concatenate((idx[breaks], [idx[-1]]))
            vis = [((n_steps - 1 - e) * step_h, (e - s + 1) * step_km)
                   for s, e in zip(starts, ends)]
            out.append(vis[::-1])
    _VISIT_CACHE[vkey] = out
    return out


def rho_fn(a, t1, t2):
    def rho(age_h):
        return a * np.exp(-age_h / t1) + (1.0 - a) * np.exp(-age_h / t2)
    return rho


def encounter_rate(observer, target, L):
    """n per km of path inside a cell, per unit class: [(share, n_per_km)]."""
    if target == 'diffuse':
        return [(1.0, np.inf)]
    units = [(1.0, nw.UNIT[target])]
    if target == 'planktivores':
        units = [(0.5, nw.UNIT[target]), (0.5, nw.NIGHT_UNIT)]
    out = []
    for share, m in units:
        dens = nw.DENS[target] / m
        k = dens * 2 * nw.radius(observer, target)
        if observer in nw.PREYS_ON.get(target, set()):
            k = max(k, dens * 2 * nw.radius(target, observer) * nw.PATH[target] / L)
        out.append((share, k))
    return out


def weight(observer, target, memory_ticks, rho, tick_perfect=False,
           tether=0.0, own=False, density_factor=1.0):
    total = 0.0
    for share, L, D in nw.OBSERVERS[observer]:
        vis = visits(L, D, memory_ticks, tether, own)
        for ushare, k in encounter_rate(observer, target, L):
            k = k * density_factor
            vals = np.empty(len(vis))
            for q, vs in enumerate(vis):
                acc, miss = 0.0, 1.0
                for age, path in vs:
                    p = 1.0 if np.isinf(k) else 1.0 - np.exp(-k * path)
                    r = 1.0 if (tick_perfect and age < TICK_H) else float(rho(age))
                    acc += miss * p * r
                    miss *= 1.0 - p
                vals[q] = acc
            total += share * ushare * float(vals.mean())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='reading B limit (M = 1 tick, rho = 1) vs neighbour_weight.py')
    ap.add_argument('--pair', nargs=2)
    ap.add_argument('--memory', type=int, default=1, help='memory span in ticks')
    ap.add_argument('--rho', nargs=3, type=float, default=(1.0, 1e9, 1e9),
                    metavar=('A', 'T1_H', 'T2_H'))
    ap.add_argument('--tether', type=float, default=0.0)
    ap.add_argument('--own', action='store_true')
    ap.add_argument('--tick-perfect', action='store_true')
    ap.add_argument('--density', type=float, default=1.0)
    args = ap.parse_args()
    if args.check:
        for o, t in (('porpoises', 'planktivores'), ('porpoises', 'diffuse'),
                     ('planktivores', 'porpoises'), ('benthivores', 'diffuse'),
                     ('whiting', 'planktivores')):
            wb = nw.w(o, t)
            wc = weight(o, t, 1, rho_fn(1.0, 1e9, 1e9), tick_perfect=True)
            print(f'{o:13s} -> {t:13s} B {wb:.3f}  C(M=1, rho=1) {wc:.3f}')
        for o in ('porpoises', 'planktivores', 'benthivores'):
            for s, L, D in nw.OBSERVERS[o]:
                print(f'{o} member L {L} D {D}: 24 h median net {net_24h(L, D):.1f} km')
        return
    o, t = args.pair
    v = weight(o, t, args.memory, rho_fn(*args.rho), args.tick_perfect,
               args.tether, args.own, args.density)
    print(f'{o} -> {t} M {args.memory} rho {tuple(args.rho)} tether {args.tether} '
          f'own {args.own} density x{args.density}: w {v:.3f}')


if __name__ == '__main__':
    main()
