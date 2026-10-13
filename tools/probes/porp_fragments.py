"""Porpoise fragment probe: where do 5-50 kg cells come from, and what of
the 5 kg sweep traces back to splits of >= 50 kg cells?

    python3 tools/probes/porp_fragments.py CKPT_DIR TICKS [--fractional-splits]
        [--rule-a] [--rule-d] [--seed S]

Frozen policies from CKPT_DIR, the deterministic probe world (60x60,
mortality, migration, currents 0.1/20/0/12). --rule-a cancels in-grid
porpoise outflows below one animal (min_split_biomass) unless the
destination's pre-move stay already holds one animal. --rule-d reassigns,
in the decision step, every spatial fraction (stay, N, E, S, W) of a
>= 1-animal cell that holds less than one animal (B * p < min_split) to
the fractions that hold at least one, proportionally; with none such the
whole cell goes to the largest fraction (the sub-animal argmax collapse).
Applied to every DM with min_split_biomass > 0. The engine applies D by
default since section 159 (--rule-d is the pre-159 probe variant and now
redundant); --fractional-splits sets env.unit_splits = False, the pre-159
continuous split that section 159.2's baseline was measured on.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import train as train_mod
from inference import load_policies_and_stats
from lib.environments.ecosystem_env import decisions, movement, population_change
from lib.environments.ecosystem_env.constants import MOVE_SLICE, REST_INDEX
from lib.environments.ecosystem_env.constants import NORTH, EAST, SOUTH, WEST
from lib.environments.ecosystem_env.currents import CurrentConfig

ap = argparse.ArgumentParser()
ap.add_argument('ckpt')
ap.add_argument('ticks', type=int)
ap.add_argument('--rule-a', action='store_true')
ap.add_argument('--rule-d', action='store_true')
ap.add_argument('--fractional-splits', action='store_true')
ap.add_argument('--seed', type=int, default=20260530)
ap.add_argument('--out', default=None)
args = ap.parse_args()

for f in sorted(os.listdir(args.ckpt)):
    if f.endswith('.pth'):
        h = hashlib.md5(open(os.path.join(args.ckpt, f), 'rb').read())
        print(f"  {h.hexdigest()}  {f}")

builder = train_mod._ProbeEnvBuilder(
    project_path='mareld2.yaml', grid_size=(60, 60),
    apply_natural_mortality=True, migration=True,
    currents=CurrentConfig(0.1, 20, 0, 12.0))
env = builder()
env.unit_splits = not args.fractional_splits
policies, mean, var = load_policies_and_stats(env, args.ckpt, verbose=False)
env.policies = policies
env.obs_mean = mean
env.obs_var = var
env._batched_ready = False
for attr in ('_setup_batched_policies', '_prepare_batched_policies',
             '_build_batched_policies'):
    fn = getattr(env, attr, None)
    if callable(fn):
        fn()
        break

FID = 'porpoises'
IP = env.dm_ids.index(FID)
fg = env.fgs[FID]
UNIT = float(fg.min_split_biomass)                       # t (loader converts kg), one animal
THR = float(env._dm_split_thr[IP])                       # t, sweep line
H, W = fg.biomass.shape
print(f"unit={UNIT*1000:.0f} kg  sweep<{THR*1000:.1f} kg  "
      f"current_response={getattr(fg, 'current_response', None)}  "
      f"rule_a={args.rule_a} rule_d={args.rule_d}  B0={fg.biomass.sum():.3f} t")


def shift_in(a):
    """(4,H,W) outflows aligned on the source -> (4,H,W) inflow per
    direction aligned on the destination."""
    out = np.zeros_like(a)
    out[NORTH, :-1, :] = a[NORTH, 1:, :]
    out[EAST, :, 1:] = a[EAST, :, :-1]
    out[SOUTH, 1:, :] = a[SOUTH, :-1, :]
    out[WEST, :, :-1] = a[WEST, :, 1:]
    return out


def dest_of(a):
    """(H,W) field -> (4,H,W) value at each direction's destination,
    aligned on the source; inf off-grid."""
    d = np.full((4,) + a.shape, np.inf, dtype=a.dtype)
    d[NORTH, 1:, :] = a[:-1, :]
    d[EAST, :, :-1] = a[:, 1:]
    d[SOUTH, :-1, :] = a[1:, :]
    d[WEST, :, 1:] = a[:, :-1]
    return d


# --------------------------------------------------------------- hooks
cap = {}
_orig_transfer = movement._transfer_to_neighbours
_orig_suppress = movement.suppress_subthreshold_splits
_orig_follow = movement.follow_subthreshold_residues
_orig_move = movement.apply_movement
_orig_sweep = population_change.apply_extinction_threshold


def transfer_hook(b_stay, r_stay, b_out, r_out):
    if cap.get('in_move'):                      # apply_currents uses it too
        cap['b_stay'] = b_stay[IP].copy()
    return _orig_transfer(b_stay, r_stay, b_out, r_out)


def suppress_hook(env_, b_out, r_out, b_total, r_total, return_cancelled=False):
    b_total, r_total, b_c, r_c = _orig_suppress(
        env_, b_out, r_out, b_total, r_total, return_cancelled=True)
    if args.rule_a:
        eff = b_out[IP] - b_c[IP]
        dstay = dest_of(cap['b_stay'])          # inf off-grid -> never blocked
        blk = (eff > 0.0) & (eff < UNIT) & (dstay < UNIT)
        if np.any(blk):
            bc = np.where(blk, eff, np.float32(0.0))
            frac = np.where(blk, bc / np.maximum(b_out[IP], 1e-30), 0.0)
            rc = (r_out[IP] - r_c[IP]) * frac
            b_total[IP] += bc.sum(axis=0)
            r_total[IP] += rc.sum(axis=0)
            b_total[IP] -= shift_in(bc).sum(axis=0)
            r_total[IP] -= shift_in(rc).sum(axis=0)
            np.maximum(b_total, 0.0, out=b_total)
            np.maximum(r_total, 0.0, out=r_total)
            b_c = b_c.copy(); r_c = r_c.copy()
            b_c[IP] += bc
            r_c[IP] += rc.astype(r_c.dtype)
            stats['a_cancel_n'] += int(blk.sum())
            stats['a_cancel_t'] += float(bc.sum())
    if return_cancelled:
        return b_total, r_total, b_c, r_c
    return b_total, r_total


def follow_hook(env_, b_out, b_total, r_total):
    res = _orig_follow(env_, b_out, b_total, r_total)
    cap['b_out_final'] = (b_out[IP] + res[3][IP]).copy()
    return res


stats = dict(
    ticks=0,
    piece_alone_n=0, piece_alone_t=0.0,      # split piece lands in a <1-animal cell
    piece_join_n=0, piece_join_t=0.0,        # split piece joins a >=1-animal cell
    remainder_n=0, remainder_t=0.0,          # >=1-animal source left with < 1 animal
    shrink_entry_n=0, shrink_entry_t=0.0,    # cell shrinks through 1 animal (no move)
    grow_exit_n=0, grow_exit_t=0.0,          # sub-animal cell grows past 1 animal
    sweep_n=0, sweep_t=0.0, sweep_split_t=0.0, sweep_other_t=0.0,
    a_cancel_n=0, a_cancel_t=0.0,
    d_cells=0, d_reassigned_t=0.0, d_collapsed=0,
    sub_cells=0, occ_cells=0, sub_mass=0.0, tot_mass=0.0,
    split_mass_in_sub=0.0,
)
# Mass-weighted share of each cell's biomass that came out of a split
# of a >= 1-animal cell; only meaningful while the cell holds < 1 animal.
frac = np.zeros((H, W), dtype=np.float64)
snap = {}


def shrink_check(before, after):
    ent = (before >= UNIT) & (after > 0.0) & (after < UNIT)
    ext = (before > 0.0) & (before < UNIT) & (after >= UNIT)
    stats['shrink_entry_n'] += int(ent.sum())
    stats['shrink_entry_t'] += float(after[ent].sum())
    stats['grow_exit_n'] += int(ext.sum())
    stats['grow_exit_t'] += float(after[ext].sum())


def move_hook(env_, settlement):
    global frac
    pre = fg.biomass.astype(np.float64).copy()
    if 'end' in snap:
        shrink_check(snap['end'], pre)
    cap.pop('b_out_final', None)
    cap['in_move'] = True
    _orig_move(env_, settlement)
    cap['in_move'] = False
    post = fg.biomass.astype(np.float64)
    bo = cap.get('b_out_final')
    if bo is None:
        # No split rules ran (no thr): nothing to attribute.
        snap['post'] = post.copy()
        return
    bo = bo.astype(np.float64)
    big = pre >= UNIT
    stay = np.maximum(pre - bo.sum(axis=0), 0.0)
    # Creation accounting, sources holding >= 1 animal.
    dpost = dest_of(post)
    ingrid = np.isfinite(dpost)
    piece = big[None] & ingrid & (bo > 0.0) & (bo < UNIT)
    alone = piece & (dpost < UNIT)
    join = piece & (dpost >= UNIT)
    stats['piece_alone_n'] += int(alone.sum())
    stats['piece_alone_t'] += float(bo[alone].sum())
    stats['piece_join_n'] += int(join.sum())
    stats['piece_join_t'] += float(bo[join].sum())
    rem = big & (stay > 0.0) & (stay < UNIT)
    stats['remainder_n'] += int(rem.sum())
    stats['remainder_t'] += float(stay[rem].sum())
    # Lineage mixture into the post-move cells.
    f_src = np.where(big, 1.0, frac)
    split_in = shift_in(bo * f_src[None]).sum(axis=0) + stay * f_src
    new = np.where(post > 0.0, np.minimum(split_in / np.maximum(post, 1e-30), 1.0), 0.0)
    frac = np.where((post > 0.0) & (post < UNIT), new, 0.0)
    snap['post'] = post.copy()


def sweep_hook(env_):
    global frac
    before = fg.biomass.astype(np.float64).copy()
    if 'post' in snap:
        shrink_check(snap['post'], before)
    _orig_sweep(env_)
    after = fg.biomass.astype(np.float64)
    lost = np.maximum(before - after, 0.0)
    gone = lost > 1e-6                          # >= 1 g; skip float dust
    stats['sweep_n'] += int(gone.sum())
    stats['sweep_t'] += float(lost.sum())
    stats['sweep_split_t'] += float((lost * frac).sum())
    stats['sweep_other_t'] += float((lost * (1.0 - frac)).sum())
    frac = np.where((after > 0.0) & (after < UNIT), frac, 0.0)
    occ = after > 0.0
    sub = occ & (after < UNIT)
    stats['occ_cells'] += int(occ.sum())
    stats['sub_cells'] += int(sub.sum())
    stats['tot_mass'] += float(after.sum())
    stats['sub_mass'] += float(after[sub].sum())
    stats['split_mass_in_sub'] += float((after * frac)[sub].sum())
    snap['end'] = after.copy()
    stats['ticks'] += 1


_orig_collapse = decisions.collapse_subthreshold_actions


def collapse_hook(probs, subthreshold_mask):
    probs = _orig_collapse(probs, subthreshold_mask)
    if not args.rule_d:
        return probs
    unit = env.dm_min_split[:, None, None]                     # (N_dm,1,1)
    B = np.stack([env.fgs[f].biomass for f in env.dm_ids]).astype(np.float64)
    act = (unit > 0) & (B >= unit)
    if not np.any(act):
        return probs
    p = probs.astype(np.float64)
    P = np.concatenate([p[:, MOVE_SLICE],
                        np.clip(1.0 - p[:, MOVE_SLICE].sum(axis=1, keepdims=True),
                                0.0, 1.0)], axis=1)            # (N_dm,5,H,W)
    keep = (B[:, None] * P >= unit[:, None]) & (P > 0)
    none = ~np.any(keep, axis=1)
    top = np.argmax(P, axis=1)
    onehot = np.arange(5)[None, :, None, None] == top[:, None]
    keep = np.where(none[:, None], onehot, keep)
    Pk = np.where(keep, P, 0.0)
    Pn = Pk / np.maximum(Pk.sum(axis=1, keepdims=True), 1e-30)
    new = p.copy()
    new[:, MOVE_SLICE] = Pn[:, :4]
    scale = Pn[:, 4] / np.maximum(P[:, 4], 1e-30)               # stationary group
    new[:, REST_INDEX:] = p[:, REST_INDEX:] * scale[:, None]
    out = np.where(act[:, None], new, p).astype(probs.dtype)
    m = act[IP]
    stats['d_cells'] += int(m.sum())
    stats['d_reassigned_t'] += float((B[IP] * np.abs(Pn[IP] - P[IP]).sum(axis=0) / 2)[m].sum())
    stats['d_collapsed'] += int((none[IP] & m).sum())
    return out


decisions.collapse_subthreshold_actions = collapse_hook
movement._transfer_to_neighbours = transfer_hook
movement.suppress_subthreshold_splits = suppress_hook
movement.follow_subthreshold_residues = follow_hook
movement.apply_movement = move_hook
population_change.apply_extinction_threshold = sweep_hook

np.random.seed(args.seed)
torch.manual_seed(args.seed)

YEAR = 1460
traj = []
for t in range(args.ticks):
    env.step()
    if (t + 1) % 365 == 0 or t + 1 == args.ticks:
        traj.append((t + 1, float(fg.biomass.sum()),
                     {k: float(env.fgs[k].biomass.sum()) for k in env.fgs}))
        print(f"  tick {t+1:6d}  porpoise {traj[-1][1]:.3f} t  "
              f"sweep so far {stats['sweep_t']:.3f} t", flush=True)

yrs = stats['ticks'] / YEAR
n = max(stats['ticks'], 1)
print(f"\n== per year over {yrs:.2f} yr (rule_a={args.rule_a}, rule_d={args.rule_d}) ==")
for k in ('piece_alone', 'piece_join', 'remainder', 'shrink_entry',
          'grow_exit', 'a_cancel'):
    print(f"  {k:14s} {stats[k+'_n']/yrs:10.0f} /yr   {stats[k+'_t']/yrs:8.3f} t/yr")
print(f"  sweep          {stats['sweep_n']/yrs:10.0f} /yr   {stats['sweep_t']/yrs:8.3f} t/yr"
      f"   (split-origin {stats['sweep_split_t']/yrs:.3f}, other {stats['sweep_other_t']/yrs:.3f})")
print(f"  rule D: {stats['d_reassigned_t']/yrs:.3f} t/yr reassigned over "
      f"{stats['d_cells']/n:.0f} cells/tick, {stats['d_collapsed']/max(stats['d_cells'],1):.1%} collapsed whole")
print(f"  sub-animal share of occupied cells {stats['sub_cells']/max(stats['occ_cells'],1):.1%}"
      f", of mass {stats['sub_mass']/max(stats['tot_mass'],1e-30):.1%}"
      f", split-origin share of sub-animal mass {stats['split_mass_in_sub']/max(stats['sub_mass'],1e-30):.1%}")
print(f"  mean porpoise mass {stats['tot_mass']/n:.3f} t, mean occupied cells {stats['occ_cells']/n:.0f}")
print(f"  final: " + ", ".join(f"{k} {v:.1f}" for k, v in traj[-1][2].items()))
if args.out:
    json.dump(dict(stats=stats, traj=traj, rule_a=args.rule_a, rule_d=args.rule_d, seed=args.seed),
              open(args.out, 'w'))
