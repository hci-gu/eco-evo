"""COPY of research_notes/"Bottenfiskens hojdprofil"/scripts/wc_benth_window.py (2026-11-11), changed to read the
member shares of THIS note's band_hs.json variants, to recompute the a(L) exponent per variant, and to add a
saithe identity variant (haddock only: cod is a trace in saithe stomachs, 1:10 by number in 1981, absent in the
1991 tables; research_notes/"Sejens diet pa unga torskfiskar" summary section 3). Usage:
python3 -I window_hs.py <repo_root> <scratch_dir> <band_hs.json> <out_json>

Original docstring follows.
Water column -> benthivores stage_access re-weighted on the band a's member shares (research_notes/"Overlapp i
fiskparen" finding 7), with the section 232.2 / 243 method (research_notes/"Fonsterviktning", o-umlaut):
each member's window averaged over its box length mix with weights biomass x membership x a(L), normalised
over all benthivore tonnes ("norm" + "aw"), then members combined by their shares of the NEW a.

Usage: python3 -I wc_benth_window.py <repo_root> <scratch_dir>

research_notes/"Fonsterviktning"/scripts/aw_stage_access.py is exec'd up to its run block (unchanged), so
pair_access / lweights / the windows / the prey length tables are the ones the library windows came from.

a(L) within a member: the 243 windows used a(L) of the section 220 in-layer encounter (fixed R; a per tonne
~ L^-2.1 for saithe). Under the band overlap with rule A the near-bed R_sat = 2 L binds for every saithe
length (R0 on a 15 cm fish 3.2 m), so in light a per tonne ~ L^-2.1 x L^2 ~ L^-0.1, in the dark (R_nv 0.3 m)
still ~ L^-2.1. kexp multiplies a(L) by L^kexp. The central kexp is DERIVED from the band-A a itself: saithe adult
(43 cm) and large (70 cm) carry 0.601 / 0.291 of the band a (band_pairs.json) at stage weights W (STG_T), so the
per-tonne ratio a(43)/a(70) gives the band-A exponent; kexp = that minus the 220 form's fitted -2.11 (Fonsterviktning
stdout; -1.92 from its per-stage 2.55). 0 (243 form), +1, +1.5, +2 are printed as brackets.
Members: saithe, hake, anglerfish have windows; pollack takes saithe's [J]; ling has no window in the
library tables and is mapped to anglerfish's (a large demersal gape) or excluded and renormalised [J].
"""
import contextlib
import io
import json
import math
import os
import sys

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
RN = os.path.join(ROOT, "research_notes")
AW = os.path.join(RN, "Fönsterviktning", "scripts", "aw_stage_access.py")
src = open(AW).read()
head = src[:src.index('print("a(L) shape check')]
ARGV = list(sys.argv)
sys.argv = [AW, RN, SCR]
NS = {"__name__": "aw_head", "__file__": AW}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, AW, "exec"), NS)
sys.argv = ARGV
pair_access, sa = NS["pair_access"], NS["sa"]
IDENT = {m: ident for m, _, ident in NS["WC_BEN"]}
print("windows available:", sorted(k for k in sa.WINDOWS))

BH = json.load(open(sys.argv[3]))
OUTW = os.path.abspath(sys.argv[4])
BAND = {k: {"fish": v["fish"]} for k, v in BH.items()}


def shares(var):
    mem = BAND[var]["fish"]["predatory_fish->benthivorous_fish"]["members"]
    out = {}
    for k, v in mem.items():
        sp = k.split("|")[0]
        out[sp] = out.get(sp, 0.0) + v
    return out


STG = {(r.sp, r.st): r for r in NS["GE"]["STG_T"]}
mA = BAND["central_repro"]["fish"]["predatory_fish->benthivorous_fish"]["members"]
ra = (mA["saithe|adult"] / STG[("saithe", "adult")].W) / (mA["saithe|large"] / STG[("saithe", "large")].W)
lr = math.log(STG[("saithe", "adult")].L / STG[("saithe", "large")].L)
k_band = math.log(ra) / lr
K_C = round(k_band + 2.11, 1)
print(f"saithe band-A a per tonne adult/large {ra:.3f} (L {STG[('saithe', 'adult')].L:.3f} / {STG[('saithe', 'large')].L:.3f} m):"
      f" exponent {k_band:.2f}; 220 form -2.11 (fitted) / {math.log(2.55) / lr:.2f} (per-stage 2.55) -> kexp"
      f" {k_band + 2.11:.2f} / {k_band - math.log(2.55) / lr:.2f}; central kexp {K_C}")
KS = (0.0, 1.0, 1.5, K_C, 2.0)
PER = {}
for k in KS:
    for m in ("saithe", "hake", "anglerfish", "spurdog", "velvet belly"):
        fj, fa, _ = pair_access("ben", [(m, 1.0, IDENT[m])], "norm", "aw", kexp=k)
        PER[(m, k)] = (fj, fa)
print("\nper-member windows [f_J, f_A] (norm + aw), a(L) x L^kexp:")
for m in ("saithe", "hake", "anglerfish", "spurdog", "velvet belly"):
    print(f"  {m:13s} " + "  ".join(f"k {k:+.1f}: [{PER[(m, k)][0]:.3f}, {PER[(m, k)][1]:.4f}]" for k in KS))
print("  (k 0 reproduces Fonsterviktning's central aw: saithe [0.251, 0.0003], hake [0.386, 0.1330], anglerfish"
      " [0.868, 0.4445], spurdog [0.715, 0.2392], velvet belly [0.044, 0.0028])")

MAP = {"saithe": "saithe", "pollack": "saithe", "hake": "hake", "anglerfish": "anglerfish", "ling": "anglerfish",
       "spurdog": "spurdog", "velvet belly": "velvet belly"}
HAD = {"haddock"}
PER_H = {}
RES = {}


def kexp_of(var):
    m = BAND[var]["fish"]["predatory_fish->benthivorous_fish"]["members"]
    if "saithe|adult" not in m or "saithe|large" not in m:
        return K_C
    r = (m["saithe|adult"] / STG[("saithe", "adult")].W) / (m["saithe|large"] / STG[("saithe", "large")].W)
    return round(math.log(r) / lr + 2.11, 2)


print("\nre-weighted on each variant's band-A member shares (ling as anglerfish; JB 0.15 for f_eff);"
      " 'had' = saithe identity haddock only")
for var in BAND:
    sh = shares(var)
    k = kexp_of(var)
    for ident in ("gad", "had"):
        num_j = num_a = den = 0.0
        for sp, s_ in sh.items():
            mm = MAP[sp]
            if mm == "saithe" and ident == "had":
                key = ("saithe_had", k)
                if key not in PER_H:
                    PER_H[key] = pair_access("ben", [("saithe", 1.0, HAD)], "norm", "aw", kexp=k)[:2]
                fj, fa = PER_H[key]
            else:
                key = (mm, k)
                if key not in PER_H:
                    PER_H[key] = pair_access("ben", [(mm, 1.0, IDENT[mm])], "norm", "aw", kexp=k)[:2]
                fj, fa = PER_H[key]
            num_j += s_ * fj
            num_a += s_ * fa
            den += s_
        fj, fa = num_j / den, num_a / den
        RES[f"{var}|{ident}"] = (fj, fa, k)
        print(f"  {var:24s} {ident} kexp {k:+.2f}: [{fj:.3f}, {fa:.4f}]  f_eff {0.15 * fj + 0.85 * fa:.4f}   saithe share "
              f"{sh.get('saithe', 0) + sh.get('pollack', 0):.3f}")
kc = kexp_of("central_repro")
print(f"  saithe alone at kexp {kc}: gadoids {[round(x, 4) for x in PER_H[('saithe', kc)]]}, haddock only "
      f"{[round(x, 4) for x in PER_H[('saithe_had', kc)]]}")
print("\n  library [0.392, 0.0388] = central_repro|gad (check above)")
json.dump({k: list(v) for k, v in RES.items()}, open(OUTW, "w"), indent=1)
