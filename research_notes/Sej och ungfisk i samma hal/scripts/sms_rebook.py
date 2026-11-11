"""Re-booking of water column -> benthivores (predation_mortality 0.0833) from the members' field diets and rations,
split J / A, against SMS 2023 at the box's saithe density.

Usage: python3 -I sms_rebook.py <sms dir> <out_json>
  <sms dir> = research_notes/"Fiskgrupper i rutan"/predators/data/population (sms2023_summary.out,
              sms2023_partial_m2.out, sms2023_species_names.in; ICES WGSAM 2024 key run 2023). Parsed only.

Parts
 1. The booking's origin, member by member (research_notes/"Rovfiskens uppdelning" tables/split_fields_223_stdout.txt
    lines 54-62: Q per member x its "Bf" share, the pre-228 water column), and what each term is today.
 2. The field route: member B (library weights x 0.617 t/km2) x Q/B x diet share ON THE MODEL'S BENTHIVORE MEMBERS
    (haddock, cod, flatfish, long rough dab, ...; research_notes/"Vattenpelarens vuxna bottenfisk" evidence E1-E19
    and field_vs_model.py FIELD) x J / A split. Hake excluded (user decision 2026-11-11); its value is shown.
    Range: Monte Carlo over triangular (lo, central, hi) per input, 5-95 %; and the all-lo / all-hi ends.
 3. SMS 2023 (2001-22), saithe on haddock (SMS saithe eat no cod): predator side = haddock eaten per tonne of
    saithe x the box's saithe (0.617 x 0.797 t/km2); prey side = SMS partial M2 on haddock J / A x (box saithe /
    North Sea saithe density, 575 000 km2 [J]) x the box's haddock J / A tonnes. J / A by length at sea (cut 27 cm,
    CV 0.10, as sms_adult_share.py). SMS overshot the observed stomach haddock share 1980-91 (9.1 vs 4.1 %): x0.45
    is the stomach-level bracket.
 Booking = flow / B_ben (1.53 t/km2), the library convention M2 = Q x share / prey B.
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

D, OUTJ = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(20261111)
B_WC, B_BEN, JS = 0.617, 1.53, 0.155            # rest group, benthivores, annual J/B (start_share mean)
HAD_SHARE_B, HAD_SHARE_J = 0.386, 0.87          # haddock of B (member weight) and of J (GOV, Rekryternas bottendjup)
M2_LIB = 0.0833
W = {"saithe": 0.797, "spurdog": 0.097, "hake": 0.048, "anglerfish": 0.027, "ling": 0.014, "velvet belly": 0.009,
     "pollack": 0.008}

# ---------------------------------------------------------------- 1. origin
print("1. Origin of 0.0833 (split_fields_223, the pre-228 water column of 0.754 t/km2; Q t/km2/yr x 'Bf' share):")
ORIG = {  # member: (Q, Bf share used, what Bf held, today's group)
    "saithe": (1.4739, 0.05, "0-1 group haddock", "rest"),
    "mackerel": (0.2117, 0.01, "trace", "mackerel (no benthivore pair)"),
    "spurdog": (0.1201, 0.22, "Ecopath 'Bf' incl. small gadoids (poor cod, bib, rocklings), gurnards, rays", "rest"),
    "horse mackerel": (0.1489, 0.07, "haddock in 20-29 cm horse mackerel, Dahl & Kirkegaard 1985", "mackerel (no pair)"),
    "hake": (0.1044, 0.01, "juveniles", "rest (out of this pair, user 2026-11-11)"),
    "anglerfish": (0.0296, 0.30, "Ecopath-like 'Bf' incl. small gadoids", "rest"),
    "ling": (0.0167, 0.30, "anglerfish proxy", "rest"),
    "velvet belly": (0.0114, 0.07, "horse mackerel proxy", "rest"),
    "pollack": (0.0149, 0.05, "saithe proxy", "rest"),
}
tot = 0.0
for sp, (q, s, what, grp) in ORIG.items():
    f = q * s
    tot += f
    print(f"   {sp:15s} Q {q:.4f} x {s:.2f} = {f:.4f}  ({what}; today: {grp})")
print(f"   sum {tot:.4f} t/km2/yr -> / {B_BEN} = {tot / B_BEN:.4f} /yr raw; x 0.9874 (library-sum scale) = "
      f"{tot / B_BEN * 0.9874:.4f} -> library {M2_LIB}; booked flow {M2_LIB * B_BEN:.4f} t/km2/yr")
moved = ORIG["mackerel"][0] * ORIG["mackerel"][1] + ORIG["horse mackerel"][0] * ORIG["horse mackerel"][1]
print(f"   of which by members now in the mackerel group (228 kept the M2 on the rest): {moved:.4f} t/km2/yr "
      f"({moved / tot:.2f} of the flow)")

# ---------------------------------------------------------------- 2. field route
# (Q/B central, lo, hi), (benthivore share on the model's members: central, lo, hi), (A fraction: central, lo, hi)
FIELD = {
    "saithe": ((3.0, 2.78, 3.6), (0.05, 0.02, 0.09), (0.08, 0.0, 0.17)),
    "spurdog": ((2.0, 1.5, 2.5), (0.06, 0.02, 0.15), (0.85, 0.6, 1.0)),
    "anglerfish": ((1.8, 1.5, 2.2), (0.19, 0.08, 0.34), (0.85, 0.5, 0.95)),
    "ling": ((2.0, 1.5, 3.0), (0.10, 0.03, 0.20), (0.5, 0.2, 0.9)),
    "pollack": ((3.0, 2.5, 3.6), (0.0, 0.0, 0.02), (0.0, 0.0, 0.5)),
    "velvet belly": ((2.0, 1.5, 3.0), (0.0, 0.0, 0.02), (0.0, 0.0, 0.5)),
    "hake": ((3.5, 2.2, 5.8), (0.01, 0.0, 0.03), (0.0, 0.0, 0.1)),
}
SRC = {"saithe": "E5 stomachs 1980-91 + Mehl 2005; Q/B SMS 2.78 - Ecopath 3.6",
       "spurdog": "Ecopath 1991 on the model's members 0.068; ICES dab",
       "anglerfish": "Ecopath 0.186 on members; ICES n 34; Iceland, Shetland",
       "ling": "[J] Q/B; MFRI Iceland haddock + cod ~15 %; in-area anecdotal",
       "pollack": "ICES n 30: none", "velvet belly": "[J]", "hake": "284 ICES stomachs; excluded"}


def tri(lo, c, hi, n):
    if hi <= lo:
        return np.full(n, c)
    return rng.triangular(lo, min(max(c, lo), hi), hi, n)


N = 20000
print("\n2. Field route (member B x Q/B x share on the model's benthivores x J/A); t/km2/yr")
print(f"   {'member':13s} {'B':>6s} {'flow':>7s} {'J':>7s} {'A':>7s}   source")
cen = {"J": 0.0, "A": 0.0}
mc = {"J": np.zeros(N), "A": np.zeros(N)}
lo_end = hi_end = 0.0
rows = {}
for sp, ((qb, qbl, qbh), (d, dl, dh), (fa, fal, fah)) in FIELD.items():
    b = W[sp] * B_WC
    fj, fa_ = b * qb * d * (1 - fa), b * qb * d * fa
    rows[sp] = dict(B=b, J=fj, A=fa_, flow=fj + fa_)
    print(f"   {sp:13s} {b:6.4f} {fj + fa_:7.4f} {fj:7.4f} {fa_:7.4f}   {SRC[sp]}{' (EXCLUDED)' if sp == 'hake' else ''}")
    if sp == "hake":
        continue
    cen["J"] += fj
    cen["A"] += fa_
    qd = b * tri(qbl, qb, qbh, N) * tri(dl, d, dh, N)
    a = tri(fal, fa, fah, N)
    mc["J"] += qd * (1 - a)
    mc["A"] += qd * a
    lo_end += b * qbl * dl
    hi_end += b * qbh * dh
flow_c = cen["J"] + cen["A"]
mct = mc["J"] + mc["A"]
p5, p95 = np.percentile(mct, [5, 95])
print(f"   total (hake out): {flow_c:.4f} (J {cen['J']:.4f}, A {cen['A']:.4f}; A share {cen['A'] / flow_c:.2f});"
      f" MC 5-95 % {p5:.4f}-{p95:.4f}; all-lo / all-hi ends {lo_end:.4f} / {hi_end:.4f}")
print(f"   booking = flow / {B_BEN}: {flow_c / B_BEN:.4f} /yr (MC 5-95 % {p5 / B_BEN:.4f}-{p95 / B_BEN:.4f}; ends "
      f"{lo_end / B_BEN:.4f}-{hi_end / B_BEN:.4f}); with hake {(flow_c + rows['hake']['flow']) / B_BEN:.4f}")
JT, AT = JS * B_BEN, (1 - JS) * B_BEN
print(f"   the library 0.0833 sits at the {(mct / B_BEN <= M2_LIB).mean() * 100:.0f}th percentile of the MC; J share of"
      f" the MC flow 5-95 % {np.percentile(mc['J'] / mct, 5):.2f}-{np.percentile(mc['J'] / mct, 95):.2f}")
print(f"   M2 on J {cen['J'] / JT:.3f} /yr (J {JT:.3f} t/km2), on A {cen['A'] / AT:.4f} /yr (A {AT:.3f} t/km2)")
saithe_share = rows["saithe"]["flow"] / flow_c
print(f"   saithe {saithe_share:.2f} of the field flow; the saithe share 0.05 (0.02-0.09) alone spans "
      f"{(flow_c - rows['saithe']['flow'] * (1 - 0.02 / 0.05)) / B_BEN:.4f}-"
      f"{(flow_c + rows['saithe']['flow'] * (0.09 / 0.05 - 1)) / B_BEN:.4f} /yr")
print(f"   the field share of the rest's ration: {flow_c / (B_WC * 2.83):.3f} at the group Q/B 2.83 "
      f"(booked today {M2_LIB * B_BEN / (B_WC * 2.83):.3f})")

# ---------------------------------------------------------------- 3. SMS
names = [l.strip().strip("_") for l in open(os.path.join(D, "sms2023_species_names.in"))][:27]
SP = {n: i + 1 for i, n in enumerate(names)}
SAI, HAD, COD = SP["Saithe"], SP["Haddock"], SP["Cod"]
summ = {}
hdr = None
for line in open(os.path.join(D, "sms2023_summary.out")):
    p = line.split()
    if not p:
        continue
    if p[0] == "Year":
        hdr = p
        continue
    r = dict(zip(hdr, p))
    summ[(int(r["Year"]), int(r["Quarter"]), int(r["Species.n"]), int(r["Age"]))] = r


def adult_frac(L_mm, cut=270.0, cv=0.10):
    if L_mm <= 0:
        return 0.0
    z = (math.log(cut) - math.log(L_mm)) / cv
    return 0.5 * math.erfc(z / math.sqrt(2))


YRS = range(2001, 2023)
sai_B = defaultdict(float)      # (y) sum over quarters of mean biomass / 4
sai_C = defaultdict(float)      # consumption t/yr
hadJ = defaultdict(float)
hadA = defaultdict(float)
for (y, q, s, a), r in summ.items():
    if y not in YRS:
        continue
    nb, w = float(r["N.bar"]), float(r["west"])
    if s == SAI:
        sai_B[y] += nb * w / 4.0
        sai_C[y] += nb * float(r["ration"])
    if s == HAD:
        fa = adult_frac(float(r["Lsea"]))
        hadA[y] += nb * w * fa / 4.0
        hadJ[y] += nb * w * (1 - fa) / 4.0
eatJ = defaultdict(float)
eatA = defaultdict(float)
eatcod = defaultdict(float)
for line in open(os.path.join(D, "sms2023_partial_m2.out")):
    p = line.split()
    if not p or not p[0].isdigit():
        continue
    y, q, pr, pa, py, ya, m2 = int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), float(p[6])
    if y not in YRS or pr != SAI or py not in (HAD, COD):
        continue
    r = summ.get((y, q, py, ya))
    if r is None:
        continue
    t = m2 * float(r["N.bar"]) * float(r["west"])
    if py == COD:
        eatcod[y] += t
        continue
    fa = adult_frac(float(r["Lsea"]))
    eatA[y] += t * fa
    eatJ[y] += t * (1 - fa)
Y = list(YRS)
aB = np.array([sai_B[y] for y in Y])
aC = np.array([sai_C[y] for y in Y])
eJ, eA = np.array([eatJ[y] for y in Y]), np.array([eatA[y] for y in Y])
hJ, hA = np.array([hadJ[y] for y in Y]), np.array([hadA[y] for y in Y])
AREA = 575000.0
S_NS = aB.mean() / AREA
S_BOX = W["saithe"] * B_WC
print(f"\n3. SMS 2023 key run 2001-22 (annual means): saithe biomass {aB.mean():.0f} t ({S_NS:.3f} t/km2), Q/B "
      f"{(aC / aB).mean():.2f}; haddock share of saithe food {((eJ + eA) / aC).mean() * 100:.1f} % (J "
      f"{(eJ / aC).mean() * 100:.1f}, A {(eA / aC).mean() * 100:.1f}); cod eaten by saithe {sum(eatcod.values()):.0f} t")
print(f"   haddock J (< 27 cm) {hJ.mean():.0f} t, A {hA.mean():.0f} t; M2 by saithe on J {(eJ / hJ).mean():.3f}, on A "
      f"{(eA / hA).mean():.4f} /yr at {S_NS:.3f} t/km2 of saithe")
pred_J = S_BOX * (eJ / aB).mean()
pred_A = S_BOX * (eA / aB).mean()
hJ_box, hA_box = HAD_SHARE_J * JS * B_BEN, HAD_SHARE_B * B_BEN - HAD_SHARE_J * JS * B_BEN
m2J_box = (eJ / hJ).mean() * S_BOX / S_NS
m2A_box = (eA / hA).mean() * S_BOX / S_NS
prey_J, prey_A = m2J_box * hJ_box, m2A_box * hA_box
other = sum(rows[s]["flow"] for s in ("spurdog", "anglerfish", "ling", "pollack", "velvet belly"))
otherJ = sum(rows[s]["J"] for s in ("spurdog", "anglerfish", "ling", "pollack", "velvet belly"))
print(f"   box saithe {S_BOX:.3f} t/km2 ({S_BOX / S_NS:.2f} x the North Sea mean)")
print(f"   predator side: {pred_J + pred_A:.4f} t/km2/yr (J {pred_J:.4f}, A {pred_A:.4f}) = box saithe x SMS haddock "
      f"eaten per saithe tonne {(eJ + eA).sum() / aB.sum():.4f}")
print(f"   prey side: M2 on haddock J {m2J_box:.3f}, A {m2A_box:.4f} /yr x box haddock J {hJ_box:.3f}, A {hA_box:.3f}"
      f" t/km2 = {prey_J + prey_A:.4f} (J {prey_J:.4f}, A {prey_A:.4f})")
print(f"   box haddock J per saithe tonne {hJ_box / S_BOX:.3f} vs SMS North Sea {(hJ / aB).mean():.3f} (the two sides"
      f" differ by this ratio)")
res_sms = {}
for lab, f in (("predator side", pred_J + pred_A), ("prey side", prey_J + prey_A)):
    for lvl, k in (("SMS level", 1.0), ("stomach level x0.45", 0.45)):
        t = f * k + other
        res_sms[f"{lab}|{lvl}"] = t / B_BEN
        print(f"   {lab:13s} {lvl:19s}: saithe {f * k:.4f} + other members (field) {other:.4f} = {t:.4f} t/km2/yr"
              f" -> booking {t / B_BEN:.4f} /yr")
print(f"   M2 on the model's J from saithe (SMS, prey side) {prey_J / (JS * B_BEN):.3f} /yr; field route saithe J "
      f"{rows['saithe']['J'] / (JS * B_BEN):.3f} /yr")

out = dict(origin_flow=tot, origin_moved_to_mackerel=moved, field=dict(flow=flow_c, J=cen["J"], A=cen["A"],
           mc5=p5, mc95=p95, lo_end=lo_end, hi_end=hi_end, booking=flow_c / B_BEN, booking_mc=[p5 / B_BEN, p95 / B_BEN],
           booking_ends=[lo_end / B_BEN, hi_end / B_BEN], members=rows),
           sms=dict(pred=[pred_J, pred_A], prey=[prey_J, prey_A], booking=res_sms, S_NS=S_NS, S_BOX=S_BOX))
json.dump(out, open(OUTJ, "w"), indent=1)
print(f"\nwrote {OUTJ}")
