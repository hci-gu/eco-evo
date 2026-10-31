"""Re-derivation of the rest water column -> benthivores stage access [f_J, f_A] (section 223.32) with
two bookkeeping corrections, and the effective access at several J/B.

Usage: python3 -I stage_access_check.py <Arsungarnas_andel_dir>

Imports research_notes/"Arsungarnas andel" (A-ring) scripts/stage_access.py as a module (its windows,
member a-shares, box length mixes and prey stage tables, unchanged) and evaluates three variants:

  lib   the library method (reproduces [0.431, 0.027]): per member, the share of each identity
        species' stage biomass inside the window, averaged over the identity species with the
        species' WHOLE-GROUP weight (BEN_W) and normalised by the identity species only.
  bw    stage-biomass weights: species weighted by w_sp x (stage share of that species' biomass), so a
        species with few juveniles weighs little in f_J (the group's J is ~3/4 haddock by mass).
  bw+all  as bw, and normalised by ALL benthivore species of the stage (a species outside the member's
        diet identity counts as 0 inside): f is then the share of the GROUP's stage biomass the member
        can take, which is what the engine multiplies with the group's J or A tonnes.

  aw    as bw+all, and each member's window averaged over its box length mix with weights
        biomass(L) x a(L), a(L) per tonne ~ L^-2 (L^-3 for the ambushing anglerfish), because the pair's a
        is dominated by the small members of each species (appended block at the end).

Species without a length key (long rough dab, dab, lemon sole, starry ray, grey gurnard) are adults
only (stage_access.py convention). Their adult biomass = BEN_W; keyed species' stage shares come from
the stage_length_biomass table per quarter.
"""
import importlib.util
import os
import sys

import numpy as np

AD = os.path.abspath(sys.argv[1])
spec = importlib.util.spec_from_file_location("stage_access", os.path.join(AD, "scripts", "stage_access.py"))
sys.argv = [os.path.join(AD, "scripts", "stage_access.py"), os.path.join(AD, "tables", "stage_length_biomass.csv"),
            "/nonexistent_out_not_used"]
sa = importlib.util.module_from_spec(spec)
_mk = os.makedirs
os.makedirs = lambda *a, **k: None  # the module makes its out dir at import; nothing is written
spec.loader.exec_module(sa)
os.makedirs = _mk

SL, BEN_W, PERIOD = sa.SL, sa.BEN_W, sa.PERIOD
members = sa.PAIRS["water column -> benthivores"][1]


def stage_share(q, sp, stage):
    d = SL[(SL.period == PERIOD[q]) & (SL.quarter == q) & (SL.species == sp)]
    if d.empty:
        return 0.0 if stage == "J" else 1.0  # no length key: adults only
    J, A = d.J.sum(), d.A.sum()
    return (J if stage == "J" else A) / (J + A) if J + A > 0 else 0.0


def inside(q, sp, window, stage):
    d = SL[(SL.period == PERIOD[q]) & (SL.quarter == q) & (SL.species == sp)]
    if d.empty or d[stage].sum() <= 0:
        return None
    L = d.L_cm.values + 0.5
    v = d[stage].values / d[stage].sum()
    if window[0] == "ratio":
        Ls, ws = sa.PRED_LEN[window[2]]
        return sum(wp * v[L <= window[1] * Lp].sum() for Lp, wp in zip(Ls, ws))
    return v[(L >= window[1]) & (L <= window[2])].sum()


def member_f(q, window, ident, stage, variant):
    num = den = 0.0
    for sp, w in BEN_W.items():
        ss = stage_share(q, sp, stage)
        wt = w if variant == "lib" else w * ss
        if wt <= 0:
            continue
        fin = inside(q, sp, window, stage)
        in_ident = ident is None or sp in ident
        if variant in ("lib", "bw"):
            if not in_ident or fin is None:
                continue
            num += wt * fin
            den += wt
        else:  # bw+all
            if fin is None:
                # adults of unkeyed species: no length table, so no window test possible; count them in
                # the denominator only (they are 20-40 cm flatfish / rays, largely outside saithe's diet
                # identity; for spurdog/hake/anglerfish this is a slight understatement, J)
                den += wt
                continue
            num += wt * (fin if in_ident else 0.0)
            den += wt
    return num / den if den > 0 else np.nan


print("Rest water column -> benthivores stage access [f_J, f_A], annual mean of Q1-Q4")
print("members (share of the pair's a): " + ", ".join(f"{m} {s}" for m, s, _ in members))
res = {}
for variant in ("lib", "bw", "bw+all"):
    out = {}
    for stage in ("J", "A"):
        fq = []
        for q in (1, 2, 3, 4):
            fq.append(sum(s * member_f(q, sa.WINDOWS[m], ident, stage, variant) for m, s, ident in members))
        out[stage] = (float(np.nanmean(fq)), fq)
    res[variant] = out
    print(f"  {variant:7s} f_J {out['J'][0]:.3f} (Q {', '.join(f'{x:.3f}' for x in out['J'][1])})   "
          f"f_A {out['A'][0]:.4f} (Q {', '.join(f'{x:.4f}' for x in out['A'][1])})")
print("\nPer member (bw+all, annual mean): f_J / f_A")
for m, s, ident in members:
    fj = np.nanmean([member_f(q, sa.WINDOWS[m], ident, "J", "bw+all") for q in (1, 2, 3, 4)])
    fa = np.nanmean([member_f(q, sa.WINDOWS[m], ident, "A", "bw+all") for q in (1, 2, 3, 4)])
    fjl = np.nanmean([member_f(q, sa.WINDOWS[m], ident, "J", "lib") for q in (1, 2, 3, 4)])
    fal = np.nanmean([member_f(q, sa.WINDOWS[m], ident, "A", "lib") for q in (1, 2, 3, 4)])
    print(f"  {m:13s} a-share {s:.2f}: lib [{fjl:.3f}, {fal:.4f}]  bw+all [{fj:.3f}, {fa:.4f}]")
print("\nGroup J composition by mass (w x J share, annual mean):")
for sp, w in BEN_W.items():
    js = np.mean([w * stage_share(q, sp, "J") for q in (1, 2, 3, 4)])
    print(f"  {sp:15s} {js:.2f}", end="")
print()
print("\nEffective access f_eff = JB f_J + (1 - JB) f_A:")
for jb in (0.05, 0.08, 0.10, 0.15):
    print(f"  J/B {jb:.2f}: " + "; ".join(
        f"{v} {jb * res[v]['J'][0] + (1 - jb) * res[v]['A'][0]:.4f}" for v in res))

# ------------------------------------------------------------------ predator-length weighting by a
# The pair's a per tonne of a member falls with predator length: N per tonne ~ L^-3 and the search
# speed ~ L (R fixed: dim bed, section 220), so a(L) per tonne ~ L^-2 (anglerfish, an ambusher whose
# encounter rate is set by the prey's speed: ~ L^-3). The library window is averaged over the member's
# box length mix by BIOMASS; the a it multiplies is dominated by the smaller fish. Variant "aw": the
# window averaged with weights biomass(L) x a(L).
EXP = {"saithe": 2.0, "spurdog": 2.0, "hake": 2.0, "anglerfish": 3.0}
orig = dict(sa.PRED_LEN)
for m, k in EXP.items():
    Ls, ws = orig[m]
    w2 = ws * Ls ** (-k)
    sa.PRED_LEN[m] = (Ls, w2 / w2.sum())
print("\nVariant aw (bw+all with the window weighted by each length's share of the member's a):")
aw = {}
for stage in ("J", "A"):
    fq = [sum(s * member_f(q, sa.WINDOWS[m], ident, stage, "bw+all") for m, s, ident in members) for q in (1, 2, 3, 4)]
    aw[stage] = (float(np.nanmean(fq)), fq)
print(f"  aw      f_J {aw['J'][0]:.3f} (Q {', '.join(f'{x:.3f}' for x in aw['J'][1])})   "
      f"f_A {aw['A'][0]:.4f} (Q {', '.join(f'{x:.4f}' for x in aw['A'][1])})")
for m, s, ident in members:
    fj = np.nanmean([member_f(q, sa.WINDOWS[m], ident, "J", "bw+all") for q in (1, 2, 3, 4)])
    fa = np.nanmean([member_f(q, sa.WINDOWS[m], ident, "A", "bw+all") for q in (1, 2, 3, 4)])
    print(f"  {m:13s} aw [{fj:.3f}, {fa:.4f}]")
for jb in (0.05, 0.08, 0.10, 0.15):
    print(f"  J/B {jb:.2f}: aw f_eff {jb * aw['J'][0] + (1 - jb) * aw['A'][0]:.4f}")
sa.PRED_LEN.update(orig)
# saithe's own access by quarter (bw+all, biomass windows and aw windows), for the quarterly test
print("\nSaithe alone, f_J / f_A by quarter (bw+all; biomass-weighted window | a-weighted window):")
for q in (1, 2, 3, 4):
    fjb = member_f(q, sa.WINDOWS["saithe"], sa.GADOIDS, "J", "bw+all")
    fab = member_f(q, sa.WINDOWS["saithe"], sa.GADOIDS, "A", "bw+all")
    Ls, ws = orig["saithe"]
    w2 = ws * Ls ** -2.0
    sa.PRED_LEN["saithe"] = (Ls, w2 / w2.sum())
    fja = member_f(q, sa.WINDOWS["saithe"], sa.GADOIDS, "J", "bw+all")
    faa = member_f(q, sa.WINDOWS["saithe"], sa.GADOIDS, "A", "bw+all")
    sa.PRED_LEN.update(orig)
    print(f"  Q{q}: [{fjb:.3f}, {fab:.4f}] | [{fja:.3f}, {faa:.4f}]")
