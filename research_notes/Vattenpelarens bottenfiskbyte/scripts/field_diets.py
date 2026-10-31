"""Field diets of the rest water-column group over the model's five checked prey of the group
(planktivores, zooplankton, benthivores, whiting, mackerel), by saithe size and quarter, and member-
weighted to the group. No model output is read.

Usage: python3 -I field_diets.py

Saithe (0.83 of the group's consumption) - hand transcriptions, % of stomach content wet weight:
  CRR 219 (ICES 1997, Tables 8-E-1..4, saithe 1991, Total North Sea, 94 % roundfish area 1), by age and
    quarter; Q1-Q3 as in research_notes/"Sejens diet pa unga torskfiskar" (Swedish letters)
    data/excerpts/crr219_table8E_saithe_1991.txt, Q4 read here from the PDF (printed p. 124):
    Melanogrammus 8.01 5.69 3.09 4.85 3.53 1.30 1.53; Merlangius 1.30 1.16 0.46 0.95 0.72 0.20 0.42;
    Trisopterus esmarki 80.41 77.79 49.15 26.67 11.87 7.99 8.05; Clupea 0.40 5.20 37.70 57.78 76.88 86.01
    85.15; Limanda (9+) 1.94; Scomber 0.05 1.67 3.69 1.51; unid. Gadidae 6.23 3.10 0.95 0.48 0.15 0.01;
    unid. Clupeidae 0.00 0.07 1.56 1.99 3.18 4.12 2.83; non-comm. Pleuronectidae 0.03; non-comm. Gadidae
    0.47 1.68 1.56 2.44 1.47 0.12; Euphausiacea 0.39 0.53 0.34 0.38 0.25 0.09 0.05; Hyperiidea 0.01 0.01.
  CRR 164 (Daan 1989, Table 8-E, saithe 1980-82 northern North Sea, half-years), read from the page image
    data/excerpts/crr164_p72_table8E_saithe.png of the same note.
  SMS 2023 key-run observed stomachs (1981, 1986, 1987, 1991; haul-weighted, from the same note's
    evidence_saithe.md entry 3): haddock 4.1, Norway pout 44, herring 7, whiting 0.7, other 43 %; used for
    the benthivore / planktivore ratio only (krill and blue whiting sit in "other").

Prey mapping (model groups): plk = Norway pout, herring, sprat, unid. Clupeidae, non-commercial Gadidae
(mostly blue whiting at ages 6-8 in Q2 per CRR 219 [J]; variant: 0); ben = haddock, cod, flatfish;
wht = whiting; mack = Scomber; zoo = Euphausiacea, Calanoida/Copepoda, Hyperiidea, Amphipoda (CRR 164);
unid. Gadidae pro rata over the identified gadoids (pout, haddock, whiting, cod, non-comm. Gadidae)
[J; variant: dropped]. Sandeel, cephalopods, Caridea/Pandalus/Crangon and unidentified fish are not
checked prey of the group (dropped before renormalising); CRR 164 unidentified fish are dropped (they
are prorated over identified fish in a variant, which leaves the shares unchanged).

Box weighting [J, as the saithe note]: saithe biomass 40-59 cm 0.71, >= 60 cm 0.29; CRR 219 ages 3-4 =
40-59 cm, ages 5-9+ = >= 60 cm (equal weights within); CRR 164 40-59 = mean(400, 500 mm), >= 60 =
mean(500, 700 mm). Quarters (half-years) weighted by mean content x (1 - empty) (CRR 219) or mean
content (CRR 164), a consumption proxy.

Other members (shares of the group's consumption: biomass x Q/B, saithe 0.797 x 3.0, spurdog 0.097 x
2.0, hake 0.048 x 3.5, anglerfish 0.027 x 1.8, ling 0.014 x 2.0 [J], velvet belly 0.009 x 3.0 [J],
pollack 0.008 x 3.0 [J]); their diets over the checked prey:
  spurdog    Ecopath North Sea 1991 (Mackinson & Daskalov 2007, EcoBase 457; the saithe note's
             evidence_spurdog_anglerfish.md 1.1): plk 0.187 + 0.099 + 0.041 + 0.022, zoo 0, ben 0.067,
             wht 0.119, mack 0.082 + 0.019
  hake       ICES stomach DB, Skagerrak 2000+ (115 stomachs, 811 g; tables/rest_diet_ices_stdout.txt):
             plk 81.2, zoo 0.1, ben 0.4, wht 13.0, mack 0
  anglerfish Ecopath 1991: plk 0.132 + 0.119, ben 0.193, wht 0.117, mack 0.068, zoo 0
  ling       ICES DB Skagerrak 2000+ (16 stomachs): plk 80.8, unid. gadoids 19.2 -> plk 0.9, ben 0.1 [J]
  velvet belly  shrimps and small fish [J]: plk 0.2, zoo 0.8
  pollack    as saithe [J]
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
P = ["plk", "zoo", "ben", "wht", "mack"]

# ------------------------------------------------------------------ CRR 219, 1991, by quarter x age 3..9+
AGES = [3, 4, 5, 6, 7, 8, 9]
Z7 = [0.0] * 7
C219 = {
    1: dict(e=[26.15, 26.32, 26.09, 25.00, 33.33, 37.50, 50.00], w=[3.26, 5.19, 12.91, 17.26, 21.35, 24.08, 165.92],
            had=Z7, wht=Z7, pout=[50.38, 41.06, 25.20, 34.42, 33.79, 28.64, 1.32],
            her=[0.99, 23.77, 55.00, 47.87, 51.22, 55.17, 72.75], spr=Z7,
            uclu=[0, 0.09, 1.04, 2.12, 1.76, 1.22, 20.96], ncg=[1.77, 2.05, 0.57, 0.21, 0.08, 0.03, 0.0],
            ug=[4.77, 6.05, 3.46, 1.45, 0.45, 0, 0], flat=Z7, mack=[0.08, 0.10, 0.01, 0.0, 0.0, 1.65, 0.37],
            zoo=[7.80 + 0.63 + 1.59, 5.87 + 0.05 + 0.85, 2.09 + 0.08, 2.06 + 0.01, 1.49, 0.69, 0.0]),
    2: dict(e=[2.22, 2.33, 2.86, 1.75, 0.0, 1.52, 0.0], w=[14.11, 19.04, 44.05, 72.36, 100.00, 137.31, 177.02],
            had=[0.01, 0.06, 0.13, 0.03, 3.65, 8.64, 9.06], wht=[0, 0, 0.01, 0.05, 2.81, 6.55, 15.59],
            pout=[14.04, 22.22, 32.29, 24.32, 17.37, 11.75, 7.15], her=[0.0, 0.24, 4.31, 11.66, 21.48, 31.29, 37.58],
            spr=Z7, uclu=[0, 0, 0, 0, 0.19, 0.44, 0.71], ncg=[0.48, 0.66, 3.22, 11.09, 14.93, 15.28, 10.74],
            ug=[2.46, 3.21, 3.28, 1.67, 0.70, 0.43, 2.37], flat=[0.03, 0.01, 0, 0, 0, 0, 0], mack=Z7,
            zoo=[49.13 + 5.63, 42.66 + 2.90, 24.32 + 0.98, 25.94 + 0.26, 21.59 + 0.13, 12.83 + 0.18, 6.97 + 0.13]),
    3: dict(e=[16.12, 13.89, 17.86, 20.00, 15.38, 14.29, 8.33], w=[9.31, 15.25, 16.88, 17.05, 138.32, 196.12, 300.34],
            had=[23.34, 10.73, 5.14, 3.36, 1.21, 1.13, 1.06], wht=[4.40, 2.91, 3.03, 3.40, 0.18, 0.06, 0.03],
            pout=[39.15, 59.69, 64.61, 59.89, 13.18, 10.61, 6.76], her=[0.08, 2.14, 8.19, 20.59, 75.31, 78.00, 81.40],
            spr=[0.05, 0.01, 0, 0, 0, 0, 0], uclu=[0, 0, 0, 0, 0.35, 0.37, 0.40], ncg=[0.52, 2.39, 2.33, 1.04, 0.38, 0.37, 0.40],
            ug=[5.87, 8.44, 7.49, 4.45, 1.33, 1.19, 1.04], flat=[0, 0, 0, 0, 1.04, 1.10, 1.20], mack=Z7,
            zoo=[8.85 + 0.07, 2.76 + 0.03, 2.54 + 0.01, 3.51, 2.63, 2.66, 2.87]),
    4: dict(e=[10.99, 9.70, 7.41, 12.50, 20.00, 33.33, 7.69], w=[13.57, 19.45, 40.02, 56.22, 88.33, 133.44, 154.43],
            had=[8.01, 5.69, 3.09, 4.85, 3.53, 1.30, 1.53], wht=[1.30, 1.16, 0.46, 0.95, 0.72, 0.20, 0.42],
            pout=[80.41, 77.79, 49.15, 26.67, 11.87, 7.99, 8.05], her=[0.40, 5.20, 37.70, 57.78, 76.88, 86.01, 85.15],
            spr=Z7, uclu=[0.0, 0.07, 1.56, 1.99, 3.18, 4.12, 2.83], ncg=[0.47, 1.68, 1.56, 2.44, 1.47, 0.12, 0.0],
            ug=[6.23, 3.10, 0.95, 0.48, 0.15, 0.01, 0.0], flat=[0.03, 0, 0, 0, 0, 0, 1.94], mack=[0.05, 1.67, 3.69, 1.51, 0, 0, 0],
            zoo=[0.40, 0.54, 0.34, 0.38, 0.25, 0.09, 0.05]),
}


def groups219(q, i, ncg_plk=True, ug_prorata=True):
    d = C219[q]
    g = dict(plk=d["pout"][i] + d["her"][i] + d["spr"][i] + d["uclu"][i] + (d["ncg"][i] if ncg_plk else 0.0),
             zoo=d["zoo"][i], ben=d["had"][i] + d["flat"][i], wht=d["wht"][i], mack=d["mack"][i])
    if ug_prorata and d["ug"][i] > 0:
        gad = d["pout"][i] + d["had"][i] + d["wht"][i] + d["ncg"][i]
        if gad > 0:
            g["plk"] += d["ug"][i] * (d["pout"][i] + (d["ncg"][i] if ncg_plk else 0.0)) / gad
            g["ben"] += d["ug"][i] * d["had"][i] / gad
            g["wht"] += d["ug"][i] * d["wht"][i] / gad
    return g


def norm(g):
    t = sum(g[k] for k in P)
    return {k: g[k] / t for k in P}, t


def crr219(quarters=(1, 2, 3, 4), **kw):
    """Box-weighted saithe diet over the checked prey (shares) and the checked prey's share of the food."""
    out = {}
    for cls, idx in (("40-59", [0, 1]), (">=60", [2, 3, 4, 5, 6])):
        acc = {k: 0.0 for k in P}
        tot_w = 0.0
        for i in idx:
            for q in quarters:
                wq = C219[q]["w"][i] * (1 - C219[q]["e"][i] / 100)
                g = groups219(q, i, **kw)
                for k in P:
                    acc[k] += wq * g[k] / 100 / len(idx)
                tot_w += wq / len(idx)
        out[cls] = {k: acc[k] / tot_w for k in P}  # share of all food, by class
    mix = {k: 0.71 * out["40-59"][k] + 0.29 * out[">=60"][k] for k in P}
    sh, checked = norm(mix)
    return sh, checked, out


# ------------------------------------------------------------------ CRR 164, 1980-82, half-years x size
C164 = {  # content g, plk (pout, herring, sprat), zoo (euph + copepoda + amphipoda), ben (had, cod, dab), wht, mack
    "Q1+4": {400: (11.395, 11.47 + 7.52, 76.43, 2.84, 0.0, 0.0), 500: (21.916, 67.44 + 0.70, 21.10, 5.92, 0.0, 0.0),
             700: (42.032, 75.49 + 1.88 + 0.32, 9.35 + 0.01, 3.67 + 0.34, 1.01, 0.0)},
    "Q2+3": {400: (9.562, 39.55, 37.13 + 0.73, 3.89, 0.46, 0.0), 500: (16.044, 21.72 + 1.11, 45.66 + 0.07, 5.36 + 0.78, 0.30, 0.0),
             700: (46.844, 29.27 + 0.44, 33.56 + 0.01, 12.75 + 0.39, 0.0, 0.0)},
}


def crr164(halves=("Q1+4", "Q2+3")):
    cls = {}
    for sc in (400, 500, 700):
        acc = {k: 0.0 for k in P}
        tw = 0.0
        for h in halves:
            w, plk, zoo, ben, wht, mack = C164[h][sc]
            for k, v in zip(P, (plk, zoo, ben, wht, mack)):
                acc[k] += w * v / 100
            tw += w
        cls[sc] = {k: acc[k] / tw for k in P}
    a = {k: 0.5 * (cls[400][k] + cls[500][k]) for k in P}
    b = {k: 0.5 * (cls[500][k] + cls[700][k]) for k in P}
    mix = {k: 0.71 * a[k] + 0.29 * b[k] for k in P}
    return norm(mix)


# ------------------------------------------------------------------ other members and the group
CONS = {"saithe": 0.797 * 3.0, "spurdog": 0.097 * 2.0, "hake": 0.048 * 3.5, "anglerfish": 0.027 * 1.8,
        "ling": 0.014 * 2.0, "velvet belly": 0.009 * 3.0, "pollack": 0.008 * 3.0}
ct = sum(CONS.values())
CONS = {k: v / ct for k, v in CONS.items()}
OTHERS = {
    "spurdog": norm(dict(plk=0.187 + 0.099 + 0.041 + 0.022, zoo=0.0, ben=0.067, wht=0.119, mack=0.082 + 0.019))[0],
    "hake": norm(dict(plk=81.2, zoo=0.1, ben=0.4, wht=13.0, mack=0.0))[0],
    "anglerfish": norm(dict(plk=0.132 + 0.119, zoo=0.0, ben=0.193, wht=0.117, mack=0.068))[0],
    "ling": dict(plk=0.9, zoo=0.0, ben=0.1, wht=0.0, mack=0.0),
    "velvet belly": dict(plk=0.2, zoo=0.8, ben=0.0, wht=0.0, mack=0.0),
}


def group(saithe):
    g = {k: CONS["saithe"] * saithe[k] + CONS["pollack"] * saithe[k] for k in P}
    for m, d in OTHERS.items():
        for k in P:
            g[k] += CONS[m] * d[k]
    return norm(g)[0]


def fmt(d):
    return " ".join(f"{k} {d[k]:.3f}" for k in P)


print("Consumption shares of the rest group's members: " + ", ".join(f"{k} {v:.3f}" for k, v in CONS.items()))
res = {}
s219, chk219, cls219 = crr219()
s219b, _, _ = crr219(ncg_plk=False)
s219c, _, _ = crr219(ug_prorata=False)
s164, chk164 = crr164()
print("\nSaithe, shares of the five checked prey (box length mix 40-59 0.71 / >=60 0.29):")
print(f"  CRR 219 (1991)                       {fmt(s219)}   (checked prey = {chk219:.2f} of the food)")
for c, d in cls219.items():
    print(f"     {c:6s} (share of all food)        {fmt(d)}")
print(f"  CRR 219, non-comm. Gadidae not plk   {fmt(s219b)}")
print(f"  CRR 219, unid. Gadidae dropped       {fmt(s219c)}")
print(f"  CRR 164 (1980-82)                    {fmt(s164)}   (checked prey = {chk164:.2f} of the identified food)")
sms_ratio = 4.1 / (44 + 7)
print(f"  SMS observed stomachs: ben / (pout + herring) = 4.1 / 51 = {sms_ratio:.3f} (by mass)")
print("\nSaithe by quarter, CRR 219 (box-weighted, shares of the checked prey):")
byq = {}
for q in (1, 2, 3, 4):
    sq, cq, _ = crr219(quarters=(q,))
    byq[q] = sq
    print(f"  Q{q}: {fmt(sq)}   ben/plk {sq['ben'] / sq['plk']:.4f}")
print("\nRest group (member-weighted by consumption):")
g219, g164 = group(s219), group(s164)
print(f"  V2 with CRR 164 saithe  {fmt(g164)}")
print(f"  V3 with CRR 219 saithe  {fmt(g219)}")
B = {"plk": 3.33, "zoo": 9.72, "ben": 1.53, "wht": 0.45, "mack": 0.85}
M2 = {"plk": 0.247, "zoo": 0.0521, "ben": 0.0833, "wht": 0.157, "mack": 0.021}
v1 = norm({k: M2[k] * B[k] for k in P})[0]
print(f"  V1 booked M2 x B        {fmt(v1)}")
for m, d in OTHERS.items():
    print(f"  member {m:12s}     {fmt(d)}")
json.dump(dict(V1=v1, V2=g164, V3=g219, saithe219=s219, saithe164=s164, saithe219_q={str(k): v for k, v in byq.items()},
               sms_ben_over_poutherring=sms_ratio, cons=CONS, others=OTHERS),
          open(os.path.join(OUT, "field_diets.json"), "w"), indent=1)
