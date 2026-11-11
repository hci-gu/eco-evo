"""Water column -> benthivores stage_access re-weighted on the band a's member shares (research_notes/"Overlapp i
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

BAND = json.load(open(os.path.join(RN, "Overlapp i fiskparen", "tables", "band_pairs.json")))["variants"]


def shares(var):
    mem = BAND[var]["fish"]["predatory_fish->benthivorous_fish"]["members"]
    out = {}
    for k, v in mem.items():
        sp = k.split("|")[0]
        out[sp] = out.get(sp, 0.0) + v
    return out


STG = {(r.sp, r.st): r for r in NS["GE"]["STG_T"]}
mA = BAND["A_band"]["fish"]["predatory_fish->benthivorous_fish"]["members"]
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
RES = {}
print("\nre-weighted on the band-A member shares (JB 0.15 for f_eff):")
for var in ("A_band", "A_band_Hs5", "A_band_Hs40"):
    sh = shares(var)
    for ling in ("anglerfish", "excluded"):
        for k in KS:
            num_j = num_a = den = 0.0
            for sp, s in sh.items():
                if sp == "ling" and ling == "excluded":
                    continue
                fj, fa = PER[(MAP[sp], k)]
                num_j += s * fj
                num_a += s * fa
                den += s
            fj, fa = num_j / den, num_a / den
            RES[f"{var}|ling {ling}|k{k:+.1f}"] = (fj, fa)
            print(f"  {var:12s} ling {ling:10s} k {k:+.1f}: [{fj:.3f}, {fa:.4f}]  f_eff {0.15 * fj + 0.85 * fa:.4f}"
                  f"   shares " + ", ".join(f"{a} {b:.3f}" for a, b in sh.items()))
print("\n  243 library [0.42, 0.12] (shares saithe 0.45, spurdog 0.18, hake 0.21, velvet belly 0.05, anglerfish 0.12),"
      " f_eff 0.165")
RES["k_central"] = (K_C, k_band)
json.dump({k: list(v) for k, v in RES.items()}, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                                     "tables", "wc_benth_window.json"), "w"), indent=1)
