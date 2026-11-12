"""Winter (Nov-Mar) and summer stomach contents and gastric-evacuation rations of whiting and
Norway pout from the public ICES stomach database, as compiled per predator record by the sibling
note `Krill över grunda bottnar/stomachs` (records_linked.csv.gz; one row per stomach or pooled
sample of `Number` stomachs, prey mass per category in g).

Usage: python3 -I stomach_rations.py <records_linked.csv.gz> <ices_stomach_dir>

Ration (g/h per fish) by the ICES generic cylinder model of gastric evacuation (WGSAM 2024, ICES Sci.
Rep. 6:13, Table 3-7; Andersen & Beyer 2005):
    R = rho exp(delta T) L^lambda E^-mu (N/F)^(alpha-1) S^alpha,
S = mean content over all N stomachs of the record (empties included), F = fed stomachs, E = the
record's mass-weighted prey energy density (kJ/g wet), armament 1 (krill digests like fish of equal
energy density, Andersen 1999 [A], earlier note). Whiting parameters rho 0.00171, lambda 1.35,
delta 0.081, mu 0.85, alpha 0.5. Norway pout has no GER parameters; the whiting (gadoid) set is used
as a proxy [J]. Daily ration = 24 R. Rations are pooled as sum(24 R) / sum(N W) over records
(stomach-number weighted, no catch weighting, as the sibling notes).

Predator weight: whiting W = a L^b fitted here to the 2022 uploads' IndWgt; Norway pout W = 0.0065
L^3.17 [J; a 15 cm pout ~34 g, cf. the SMS adult 25 g]. Length = class midpoint for pooled 1980-91
records (classes 5 cm; pout 7/10/15/20/25 as lower bounds; whiting 30 -> 30-39.9 and 40 -> 40-49.9),
L + 0.5 for 1-cm records.

Prey energy density, kJ/g wet [J, from the sibling notes]: krill 5.0, fish 5.0, pelagic shrimp 4.0,
mysid 4.0, copepods 3.6, hyperiids 3.5, other zooplankton 3.0, chaetognaths 2.5, other (benthos) 3.0,
crustacea unid. 4.0, unidentified 4.0.

Temperature: bottom water in winter 6.5-8.0 C in the box (mareld2.yaml `bottom`: Jan 7.8, Feb 7.0,
Mar 6.6, Nov 8.4, Dec 8.5); central 7.4, sensitivity 6 and 8.5. The NE North Sea in Q1 is ~6-7 C [J].
"""
import csv
import gzip
import math
import os
import sys
from collections import defaultdict

import numpy as np

rec_path, ices_dir = sys.argv[1], sys.argv[2]

E = dict(krill=5.0, fish=5.0, pshrimp=4.0, mysid=4.0, cop=3.6, hyp=3.5, ozoo=3.0, chaet=2.5,
         other=3.0, crustu=4.0, U=4.0)
CATS = list(E)
RHO, LAM, DEL, MU = 0.00171, 1.35, 0.081, 0.85

# --- whiting length-weight from the 2022 single-stomach uploads ---------------------------------
Ls, Ws = [], []
with open(os.path.join(ices_dir, "x2022", "PredatorInformation.csv"), encoding="utf-8", errors="replace") as f:
    for r in csv.DictReader(f):
        if r["AphiaIDPredator"] != "126438":
            continue
        try:
            L, w = float(r["Length"]) + 0.5, float(r["IndWgt"])
        except ValueError:
            continue
        # IndWgt is in kg in some uploads and g in others: keep the reading with a Fulton K
        # (100 W / L^3) of 0.4-1.6
        for W in (w * 1000.0, w):
            if 8 <= L <= 50 and 0.4 <= 100 * W / L ** 3 <= 1.6:
                Ls.append(L)
                Ws.append(W)
                break
b, lna = np.polyfit(np.log(Ls), np.log(Ws), 1)
WA, WB = math.exp(lna), b
print(f"whiting L-W from 2022 IndWgt: W = {WA:.5f} L^{WB:.3f}  (n {len(Ls)}); 25 cm -> {WA*25.5**WB:.0f} g")
PA, PB = 0.0065, 3.17


def weight(sp, L):
    return WA * L ** WB if sp == "whiting" else PA * L ** PB


def midpoint(sp, year, L):
    if year >= 2000:
        return L + 0.5
    if sp == "Norway pout":
        return {7: 8.5, 10: 12.5, 15: 17.5, 20: 22.5, 25: 27.5}.get(int(L), L + 2.5)
    if int(L) == 30 and year < 2000:
        return 35.0 if year == 1981 else 32.5
    if int(L) == 40 and year == 1981:
        return 45.0
    return L + 2.5


def season(m):
    return "winter" if m in (11, 12, 1, 2, 3) else ("Q3" if m in (7, 8, 9) else "other")


def region(area, inbox):
    if inbox == "True":
        return "box"
    if area in ("Skagerrak", "Kattegat"):
        return "Sk+Kat"
    if area == "NE North Sea":
        return "NE NS"
    return None


acc = defaultdict(lambda: defaultdict(float))
flag = []
with gzip.open(rec_path, "rt") as f:
    for r in csv.DictReader(f):
        sp = r["sp"]
        if sp not in ("whiting", "Norway pout"):
            continue
        m, y = int(r["month"]), int(r["year"])
        s = season(m)
        reg = region(r["area"], r["inbox"])
        if s == "other" or reg is None:
            continue
        N = float(r["Number"] or 1)
        empty = float(r["StomachEmpty"] or 0)
        if y < 2000:
            F = max(N - empty, 0.0)
        else:
            F = 0.0 if empty >= 1 else 1.0
        g = {c: float(r.get("g_" + c) or 0.0) for c in CATS}
        Wtot = sum(g.values())
        if Wtot > 0 and F == 0:
            F = N  # prey present but flagged empty: treat as fed
        L = midpoint(sp, y, float(r["L_cm"]))
        Wf = weight(sp, L)
        S_mean = Wtot / N
        if S_mean / Wf > 0.15:
            flag.append((sp, y, m, r["area"], L, N, round(Wtot, 1), round(Wf, 1)))
            continue
        Erec = sum(g[c] * E[c] for c in CATS) / Wtot if Wtot > 0 else 4.5
        regs = [reg] + (["Sk+Kat"] if reg == "box" else [])
        for T in (6.0, 7.4, 8.5):
            core = RHO * math.exp(DEL * T) * L ** LAM * Erec ** (-MU)
            Rh = core * math.sqrt(Wtot * F) / N if F > 0 else 0.0  # mean g/h per fish
            Rd = 24.0 * Rh * N  # g/d summed over the record's fish
            for rg in regs:
                for yy in ("all", "1980-91" if y < 2000 else "2000+"):
                    k = (sp, rg, s, yy, T)
                    acc[k]["ration_g"] += Rd
                    acc[k]["krill_ration_g"] += Rd * (g["krill"] / Wtot if Wtot > 0 else 0.0)
                    if T == 7.4:
                        acc[k]["kJ"] += Rd * Erec
        for rg in regs:
            for yy in ("all", "1980-91" if y < 2000 else "2000+"):
                k = (sp, rg, s, yy)
                a = acc[k]
                a["N"] += N
                a["F"] += F
                a["BW"] += N * Wf
                a["L"] += N * L
                a["S"] += Wtot
                a["krill_g"] += g["krill"]
                a["krill_FO"] += N if g["krill"] > 0 else 0.0  # pooled: record-level occurrence
                a["krill_n"] += float(r["krill_count"] or 0.0)
                a["records"] += 1

print("\nRecords dropped as implausible (mean content > 15 % BW; likely a whole-fish or pooled-weight artefact):")
for x in flag:
    print("  ", x)

hdr = ("species", "region", "season", "years", "stom", "fed%", "L", "W g", "S %BW", "krill %W", "krillFO%",
       "krill/stom", "ration %BW/d 6/7.4/8.5C", "krill-only %BW/d @7.4", "kJ/kg/d gross @7.4")
print("\n" + " | ".join(hdr))
for k in sorted(k for k in acc if len(k) == 4):
    a = acc[k]
    if a["N"] < 10:
        continue
    sp, rg, s, yy = k
    BW = a["BW"]
    rat = [acc[k + (T,)]["ration_g"] / BW * 100 for T in (6.0, 7.4, 8.5)]
    kr = acc[k + (7.4,)]["krill_ration_g"] / BW * 100
    kj = acc[k + (7.4,)]["kJ"] / BW * 1000
    print(f"{sp} | {rg} | {s} | {yy} | {a['N']:.0f} | {100*a['F']/a['N']:.0f} | {a['L']/a['N']:.1f} | "
          f"{BW/a['N']:.0f} | {100*a['S']/BW:.2f} | {100*a['krill_g']/a['S'] if a['S'] else 0:.0f} | "
          f"{100*a['krill_FO']/a['N']:.0f} | {a['krill_n']/a['N']:.2f} | "
          f"{rat[0]:.2f}/{rat[1]:.2f}/{rat[2]:.2f} | {kr:.2f} | {kj:.0f}")

# GER Cmax at a satiation fill (S = 2.5 % BW; 5 % upper) for reference
print("\nGER 'Cmax' (intake = evacuation at a satiation fill S), on krill E 5.0 kJ/g:")
for sp, L in (("whiting", 20.5), ("whiting", 25.5), ("whiting", 30.5), ("Norway pout", 12.5), ("Norway pout", 17.5)):
    Wf = weight(sp, L)
    for T in (6.0, 7.4, 8.5):
        out = []
        for fill in (0.025, 0.05):
            R = 24 * RHO * math.exp(DEL * T) * L ** LAM * 5.0 ** (-MU) * math.sqrt(fill * Wf)
            out.append(100 * R / Wf)
        print(f"  {sp:12s} L {L:4.1f} W {Wf:5.0f} g T {T}: {out[0]:.2f} (S 2.5 %) / {out[1]:.2f} (S 5 %) %BW/d")

# --- count-based krill ration for records without prey weights (1981-91 whiting) -----------------
# The 1981 whiting uploads give prey counts, not weights. Krill content S_k = krill count x the
# fresh mass of one adult M. norvegica (0.15 / 0.23 / 0.30 g WW; `Krill i Skagerrak`: 0.23 g at
# 35 mm). Partly digested krill counted as whole ones overstate S_k. The GER on S_k alone,
# rho e^(dT) L^l E^-mu sqrt(S_k F)/N, is an upper bound on the krill part of the evacuation
# (sqrt is concave) [D/J].
print("\nCount-based krill ration (whiting, records without weights), T 7.4 C:")
acc2 = defaultdict(lambda: defaultdict(float))
with gzip.open(rec_path, "rt") as f:
    for r in csv.DictReader(f):
        if r["sp"] != "whiting":
            continue
        m, y = int(r["month"]), int(r["year"])
        s = season(m)
        if y >= 2000 or s == "other":
            continue
        reg = {"Skagerrak": "Sk", "Kattegat": "Sk", "NE North Sea": "NE NS", "NW North Sea": "NW NS"}.get(r["area"])
        if reg is None:
            continue
        N = float(r["Number"] or 1)
        F = max(N - float(r["StomachEmpty"] or 0), 0.0)
        kc = float(r["krill_count"] or 0.0)
        L = midpoint("whiting", y, float(r["L_cm"]))
        Wf = weight("whiting", L)
        a = acc2[(reg, s)]
        a["N"] += N
        a["F"] += F
        a["BW"] += N * Wf
        a["kc"] += kc
        a["L"] += N * L
        for wk in (0.15, 0.23, 0.30):
            Sk = kc * wk
            R = 24 * RHO * math.exp(DEL * 7.4) * L ** LAM * 5.0 ** (-MU) * (math.sqrt(Sk * F) if F > 0 else 0.0)
            a[f"R{wk}"] += R
            a[f"S{wk}"] += Sk
for k in sorted(acc2):
    a = acc2[k]
    if a["N"] < 10:
        continue
    print(f"  whiting {k[0]:6s} {k[1]:6s} 1981-91: stomachs {a['N']:.0f}, fed {100*a['F']/a['N']:.0f} %, "
          f"L {a['L']/a['N']:.1f}, krill/stomach {a['kc']/a['N']:.2f}, krill content "
          + " / ".join(f"{100*a[f'S{w}']/a['BW']:.2f}" for w in (0.15, 0.23, 0.30))
          + " %BW, krill ration <= "
          + " / ".join(f"{100*a[f'R{w}']/a['BW']:.2f}" for w in (0.15, 0.23, 0.30))
          + " %BW/d (0.15 / 0.23 / 0.30 g per krill)")
