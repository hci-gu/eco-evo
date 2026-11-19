"""Per-bed-class attack rates of the water-column (WC) pairs, decomposed into overlap geometry, vision (light) and
prey-member terms, under saithe search profiles that differ over deep beds.

Usage: python3 -I class_terms.py <repo_root> <scratch_dir> <out_json>

research_notes/"Overlapp i fiskparen"/scripts/band_pairs.py is exec'd UNCHANGED up to its section 0 block (as
research_notes/"Sejens hojd over botten"/scripts/band_hs.py does), so band_fish_pair, run_krill, the member list,
rule A, the section 253 light, the section 262 krill layer, slab clipping and every other factor are the ones the
library's section 269 class tables came from. Replaced here, per variant:
  * the saithe and pollack search profile (day and night; krill members re-pointed, PCACHE cleared);
  * the vision term, via the gadoid eye constant Ke and the deep attenuation k_deep (the "dark" run, Ke = 1e9 lx,
    puts every visual member at R_nv in every class: geometry and member mix only);
  * the planktivore pair split by prey member (herring + sprat / Norway pout / blue whiting) through the pair's
    own `subset` slot.
Nothing is fitted; no model output is read. Classes: bed < 100 / 100-150 / 150-250 / >= 250 m, equal-cell means
(class 0 of the krill pair presence-weighted, as in the library).

Saithe profiles (day; night = f_up in 0-30 m + (1 - f_up) x the day shape at scale x Hn_ratio unless stated):
  exp15        library: exp(-h/15 m) above the bed, never above 10 m depth
  expHsz       Hs grows with the bed: Hs = 15 m x (bed + 10)/(140 + 10) (the swim-bladder free range, a fixed
               share of absolute pressure, scales with bed + 10 m; 140 m = median bed of the HERAS hauls the
               15 m fit came from) [J on the form]
  freerange    uniform from the bed up to the swim-bladder free ascent, 0.25 (bed + 10) m above it (Armannsson &
               Jonsson 2012 adopt the cod limits +50 % / -25 % of pressure) [J]
  mixK<m>      (1 - m) exp15 + m x the krill day layer of that bed (search time inside the krill layer) [J, bracket]
  nightcol     day exp15; night uniform 10 m .. bed (FishMap: "hundreds of metres above the bottom during night
               time") [S/J]
"""
import contextlib
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
OUTJ = os.path.abspath(sys.argv[3])
os.makedirs(SCR, exist_ok=True)
BP = os.path.join(ROOT, "research_notes", "Overlapp i fiskparen", "scripts", "band_pairs.py")
src = open(BP).read()
head = src[:src.index("# ================================================================== 0. overlap geometry")]
ARGV = list(sys.argv)
sys.argv = [BP, ROOT, SCR, os.path.join(SCR, "_unused_band.json")]
B = {"__name__": "band_pairs_head", "__file__": BP}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, BP, "exec"), B)
sys.argv = ARGV
K, qk0, qf, ZG = B["K"], B["qk0"], B["qf"], B["ZG"]
uniform, expbed = K["uniform"], K["expbed"]
BEDS, WCELL, CLS_IDX = B["BEDS"], B["WCELL"], B["CLS_IDX"]
SHARE = np.array([WCELL[i].sum() for i in CLS_IDX])
REF = json.load(open(os.path.join(ROOT, "research_notes", "Overlapp i fiskparen", "tables", "band_pairs.json")))["variants"]
SAITHE0 = K["saithe_field"]


def krill_day(bed):
    top, bot = K["krill_day_layer"](bed, qk0)
    return uniform(top, bot)


def day_shape(bed, kind, q):
    if kind == "exp15":
        return expbed(bed, q["Hs_saithe"])
    if kind == "expHsz":
        return expbed(bed, q["Hs_saithe"] * (bed + 10.0) / 150.0)
    if kind == "freerange":
        h = 0.25 * (bed + 10.0)
        return uniform(max(10.0, bed - h), bed)
    if kind.startswith("mixK"):
        m = float(kind[4:])
        return (1 - m) * expbed(bed, q["Hs_saithe"]) + m * krill_day(bed)
    raise ValueError(kind)


def make_prof(kind, night_kind="lib"):
    def f(bed, q, night):
        if not night:
            return day_shape(bed, kind, q)
        if night_kind == "col":
            return uniform(10.0, bed) if bed > 11 else uniform(bed - 1, bed)
        up = uniform(0.0, min(30.0, bed - 1.0))
        if kind == "exp15" or kind.startswith("mixK"):
            base = expbed(bed, q["Hs_saithe"] * q["Hn_ratio"])
        elif kind == "expHsz":
            base = expbed(bed, q["Hs_saithe"] * q["Hn_ratio"] * (bed + 10.0) / 150.0)
        else:
            base = day_shape(bed, kind, q)
        return q["f_up_saithe"] * up + (1 - q["f_up_saithe"]) * base
    return f


PRED_PROF = B["PRED_PROF"]
FP = B["FISH_PAIRS"]
base = FP["predatory_fish->planktivorous_fish"]
SUB = {"herring+sprat": ("herring", "sprat"), "pout": ("pout",), "blue whiting": ("blue whiting",)}
for k, s in SUB.items():
    FP[f"plk[{k}]"] = (base[0], base[1], s) + tuple(base[3:])
PAIRS = ["predatory_fish->planktivorous_fish", "plk[herring+sprat]", "plk[pout]", "plk[blue whiting]",
         "predatory_fish->benthivorous_fish", "predatory_fish->whiting"]

PROFILES = {"exp15": make_prof("exp15"), "expHsz": make_prof("expHsz"), "freerange": make_prof("freerange"),
            "mixK0.1": make_prof("mixK0.1"), "mixK0.25": make_prof("mixK0.25"),
            "nightcol": make_prof("exp15", "col")}
VISION = {"A": (0.02, 0.06), "dark": (1e9, 0.06), "Ke2e-4": (2e-4, 0.06), "kdeep0.05": (0.02, 0.05),
          "kdeep0.09": (0.02, 0.09)}
RUNS = [(p, "A") for p in PROFILES] + [("exp15", v) for v in VISION if v != "A"] + \
       [("freerange", "dark"), ("mixK0.25", "dark"), ("expHsz", "dark")]


def set_profile(fn):
    PRED_PROF["saithe"] = fn
    PRED_PROF["pollack"] = fn
    for m in K["GROUPS"]["predatory_fish"]["members"]:
        if m["name"].startswith(("saithe", "pollack")):
            m["prof"] = fn
    B["PCACHE"].clear()


RES = {}
B["RULE"].update(rule="A", RD=1.2, Lk=None)
for prof, vis in RUNS:
    Ke, kd = VISION[vis]
    set_profile(PROFILES[prof])
    with contextlib.redirect_stdout(io.StringIO()):
        kr = B["run_krill"](dict(K["level"](1), Ke_gad=Ke, k_deep=kd))["predatory_fish"]
    out = {"krill": dict(eq=kr["eq"], classes=list(kr["classes"]))}
    for name in PAIRS:
        r = B["band_fish_pair"](name, dict(qk0), "A", Ke, kd, True)
        out[name] = dict(eq=r["eq"], classes=list(r["classes"]), ratio_lib=r["ratio"], nd=r["nd"])
    RES[f"{prof}|{vis}"] = out
    print(f"{prof}|{vis} done", flush=True)
set_profile(SAITHE0)

c = RES["exp15|A"]
print("\nReproduction (exp15|A vs band_pairs.json A_band): ratio new/library a")
for name in ["predatory_fish->planktivorous_fish", "predatory_fish->benthivorous_fish", "predatory_fish->whiting"]:
    print(f"  {name:40s} here {c[name]['ratio_lib']:.4f}  file {REF['A_band']['fish'][name]['ratio']:.4f}  "
          f"cls_rel here " + " / ".join(f"{x / c[name]['eq']:.3f}" for x in c[name]['classes'])
          + "  file " + " / ".join(f"{x:.3f}" for x in REF['A_band']['fish'][name]['cls_rel']))
kref = REF["A_band"]["krill"]["predatory_fish"]
print(f"  krill classes here " + " / ".join(f"{x * 0.026 / 0.0260:.4g}" for x in c['krill']['classes'])
      + "  (file, library units " + " / ".join(f"{x:.4g}" for x in kref['classes_lib_units']) + ")")


def shallow(cls):
    w = SHARE[:2] / SHARE[:2].sum()
    return float(np.dot(w, cls[:2]))


def summary(run, name):
    r = RES[run][name]
    cls = np.array(r["classes"])
    eq = float(SHARE @ cls) if name == "krill" else r["eq"]
    return cls / eq, shallow(cls) / cls[2], cls[1] / cls[2], eq


print("\nClass a / equal-cell a (< 100 / 100-150 / 150-250 / >= 250 m); G = a(< 150 m, cell-weighted) / a(150-250 m);"
      " G1 = a(100-150) / a(150-250); eq relative to exp15|A")
ALLP = ["krill"] + PAIRS
for name in ALLP:
    print(f"\n  {name}")
    for run in RES:
        rel, G, G1, eq = summary(run, name)
        _, _, _, eq0 = summary("exp15|A", name)
        print(f"    {run:20s} " + " / ".join(f"{x:6.3f}" for x in rel) + f"   G {G:7.2f}  G1 {G1:6.2f}  eq x{eq / eq0:.3f}")

print("\nDecomposition of the central (exp15|A) class shape: vision factor = (A / dark) per class, normalised")
for name in ALLP:
    a = np.array(RES["exp15|A"][name]["classes"])
    d = np.array(RES["exp15|dark"][name]["classes"])
    geo = d / (SHARE @ d)
    vis = (a / d) / ((a / d) @ SHARE)
    print(f"  {name:38s} geometry " + " / ".join(f"{x:6.3f}" for x in geo) + f" (G {shallow(geo) / geo[2]:6.2f});"
          f"  vision " + " / ".join(f"{x:6.3f}" for x in vis) + f" (G {shallow(vis) / vis[2]:6.2f})")

print("\nPlanktivore pair: each prey member's share of the pair's class a (central)")
tot = np.array(RES["exp15|A"]["predatory_fish->planktivorous_fish"]["classes"])
for k in SUB:
    part = np.array(RES["exp15|A"][f"plk[{k}]"]["classes"])
    print(f"  {k:14s} " + " / ".join(f"{x:5.3f}" for x in part / tot))
json.dump(dict(runs=RES, share=SHARE.tolist()), open(OUTJ, "w"), indent=1, default=float)
print(f"\nwritten {os.path.basename(OUTJ)}")
