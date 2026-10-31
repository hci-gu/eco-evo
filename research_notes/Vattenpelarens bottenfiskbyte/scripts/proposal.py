"""Candidate correction of the rest water column -> benthivores pair under the vertical-overlap
convention (the one section 232 applied to whiting's bed pair), with ranges, and what it implies.

Usage: python3 -I proposal.py <Arsungarnas_andel_dir>

Per member m: a_m (the section 220 in-layer a, its share of the pair a 0.0651) x O_m, O_m = the share of
the member's foraging time inside the ~2-m layer of settled 0-1 group gadoids (evidence_literature.md):
  saithe       0.05 (0.02-0.15)  [D/J: dispersed feeding saithe 30-100 m above the bed (Aglen 1996),
                                  depth tags spread over ~100 m (Skilbrei & Ottera 2016); bed-referenced
                                  echo shares only for spawning pollock aggregations]
  spurdog      0.15 (0.05-0.30)  [J: Klocker 2024 off the bottom much of the time; near-bed by day
                                  0.1-0.3, ~0 at night]
  hake         0.25 (0.10-0.50)  [J: near the bed by day, up through the thermocline at night]
  velvet belly 0.10 (0.05-0.30)  [J: Skagerrak diet krill, Pasiphaea, blue whiting; no gadoids]
  anglerfish   1.0               [F: on the bed, inactive 95 % of the time]
New pair a = 0.0651 x sum_m share_m O_m; the member shares of the new a re-weight the stage access
(access_lib "aw": stage-biomass species weights, group-wide normalisation, windows weighted by each
length's share of the a). Two variants for the other pairs: (b) they are unchanged (M2-like), (c) the
saithe part of the planktivore, zooplankton and whiting pairs takes saithe's own overlaps with those
layers [J]: herring/sprat layer 0.85 and pout/blue whiting 0.3 -> planktivores 0.78; zooplankton 0.8;
whiting (5-20 m above the bed) 0.25.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from access_lib import Access  # noqa: E402

ACC = Access(sys.argv[1])
sa = ACC.sa
F = json.load(open(os.path.join(HERE, "..", "tables", "field_diets.json")))
P = ["plk", "zoo", "ben", "wht", "mack"]
B = {"plk": 3.33, "zoo": 9.72, "ben": 1.53, "wht": 0.45, "mack": 0.85}
H = {"plk": 239.2, "zoo": 151.1, "ben": 151.1, "wht": 161.1, "mack": 265.1}
SH = {"saithe": 0.77, "spurdog": 0.10, "hake": 0.07, "velvet belly": 0.04, "anglerfish": 0.01}
IDENT = {"saithe": sa.GADOIDS, "spurdog": sa.FLAT | {"haddock"}}
O = {"central": {"saithe": 0.05, "spurdog": 0.15, "hake": 0.25, "velvet belly": 0.10, "anglerfish": 1.0},
     "low": {"saithe": 0.02, "spurdog": 0.05, "hake": 0.10, "velvet belly": 0.05, "anglerfish": 1.0},
     "high": {"saithe": 0.15, "spurdog": 0.30, "hake": 0.50, "velvet belly": 0.30, "anglerfish": 1.0}}
JB = {"plk": 0.38, "wht": 0.165}


def mean(x):
    return sum(x) / len(x)


def feff(pair, variant, jb):
    J, A = ACC.access(pair, variant)
    return jb * mean(J) + (1 - jb) * mean(A), (mean(J), mean(A))


fplk, _ = feff("plk", "aw", JB["plk"])
fwht, _ = feff("wht", "aw", JB["wht"])
print("Rest -> benthivores under the overlap convention (a x O per member)")
res = {}
for lev, o in O.items():
    k = sum(SH[m] * o[m] for m in SH)
    a_new = 0.0651 * k
    shares = {m: SH[m] * o[m] / k for m in SH}
    ACC.pairs["ben"] = ("ben", [(m, shares[m], IDENT.get(m)) for m in SH])
    J, A = ACC.access("ben", "aw")
    fj, fa = mean(J), mean(A)
    ACC.pairs["ben"] = ("ben", [(m, SH[m], IDENT.get(m)) for m in SH])
    res[lev] = dict(a=a_new, fJ=fj, fA=fa, shares=shares)
    print(f"  {lev:7s}: overlap factor {k:.4f} -> a {a_new:.5f}; member shares of the new a "
          + ", ".join(f"{m} {v:.2f}" for m, v in shares.items()) + f"; stage access (aw) [{fj:.3f}, {fa:.4f}]")
    for jb in (0.05, 0.08, 0.15):
        print(f"           J/B {jb:.2f}: f_eff {jb * fj + (1 - jb) * fa:.4f}, a x f_eff {a_new * (jb * fj + (1 - jb) * fa):.6f}"
              f" (library {0.0651 * (jb * 0.431 + (1 - jb) * 0.027):.6f})")

print("\nConcurrent-search diet shares at J/B 0.15 (all prey visible) vs field; and single-pair saturation x/(1+x):")
FIELD = {"V1 booked": F["V1"], "V2 CRR164": F["V2"], "V3 CRR219": F["V3"]}
LIB = {"plk": 0.00559 * 0.4331, "zoo": 0.000241, "ben": 0.0651 * 0.0877, "wht": 0.013 * 0.2160, "mack": 0.00088 * 0.29}


def line(name, m):
    t = sum(m[k] * B[k] for k in P)
    sat = {k: m[k] * H[k] * B[k] / (1 + m[k] * H[k] * B[k]) for k in P}
    print(f"  {name:40s} " + " ".join(f"{k} {m[k] * B[k] / t:.3f}" for k in P)
          + "  | sat " + " ".join(f"{k} {sat[k]:.2f}" for k in P))


line("library", LIB)
OTH = {"plk": 0.84 * 0.78 + 0.16, "zoo": 0.86 * 0.8 + 0.14, "wht": 0.77 * 0.25 + 0.23, "mack": 1.0}
for lev, r in res.items():
    fe = 0.15 * r["fJ"] + 0.85 * r["fA"]
    mb = {"plk": 0.00559 * fplk, "zoo": 0.000241, "ben": r["a"] * fe, "wht": 0.013 * fwht, "mack": 0.00088 * 0.29}
    line(f"(b) {lev}, other pairs aw access", mb)
    mc = {k: mb[k] * (OTH[k] if k != "ben" else 1.0) for k in P}
    line(f"(c) {lev}, + saithe overlaps on the others", mc)
for n, d in FIELD.items():
    print(f"  field {n:34s} " + " ".join(f"{k} {d[k]:.3f}" for k in P))
print("\nClearance ratio c_ben / c_plk, model / field (b, central): " + ", ".join(
    f"{n} {(res['central']['a'] * (0.15 * res['central']['fJ'] + 0.85 * res['central']['fA']) / (0.00559 * fplk)) / ((d['ben'] / B['ben']) / (d['plk'] / B['plk'])):.2f}"
    for n, d in FIELD.items()))
json.dump(res, open(os.path.join(HERE, "..", "tables", "proposal.json"), "w"), indent=1)
