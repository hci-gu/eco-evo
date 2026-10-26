"""Nephrops share of the benthivorous_fish group's consumption and the M2 it implies on norway_lobster.

Usage:
  python3 -I nephrops_group_share.py <ices_nephrops_shares.csv> <length_biomass_longrun.csv> \
      <qb_species.csv> <out_dir>

Inputs
  ices_nephrops_shares.csv   ../tables (ices_nephrops_scan.py): ICES stomach DB, Nephrops % of total food
                             mass per predator x area x period x size class.
  length_biomass_longrun.csv ../../Fiskgrupper i rutan/benthivores/tables: NS-IBTS GOV biomass per 1-cm
                             class in the Mareld grid region, 2001-25 and 2021-25 shares, mean weight.
  qb_species.csv             ../../Fiskgrupper i rutan/benthivores/inputs: biomass share and central Q/B.

Steps
  1. Cod Nephrops share s(L) per size class from each stomach data set (Dorman et al. 1992 Feb 1991
     Skagerrak and Kattegat, Table 2/3 primary weights transcribed below; ICES DB NNS Q1 1981, NNS Q1 2023,
     Skagerrak 2023-25, Kattegat 2021-25).
  2. Weight s(L) by the grid's cod biomass by length (2001-25 or 2021-25), optionally times a ration
     per unit mass ~ W^-0.2 (consumption weights rather than biomass weights).
  3. Stomach-to-intake and season levers (stated in LEVERS; judgement "J", sources in the report).
  4. Group share p = sum_i b_i QB_i s_i / sum_i b_i QB_i over all members (qb_species.csv).
  5. p as a share of the measured benthos-epifauna part (0.2182 of consumption, 221/223.10, the base
     of the 0.003 / 0.118 figures) and of the whole epifauna part incl. booked prey (0.3143, E1).
  6. M2 = QB 4.46 x 1.53 x p / 4.572 (E1 grid-mean Nephrops stock). Q/B 3.4 and 5.0 as range.
Outputs: cod_size_profiles.csv, cod_share_by_mix.csv, group_share_scenarios.csv (+ stdout).
"""
import csv, os, sys

shares_f, lb_f, qb_f, outd = sys.argv[1:5]
lcp_f = sys.argv[5] if len(sys.argv) > 5 else None   # optional: Torskfiskarnas bytesandelar length_class_pct.csv
CLS = ["00-20", "20-30", "30-40", "40-50", "50-70", "70-"]
EDGES = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 70), (70, 999)]

# ---- 1. Dorman, Johnsson & Larsson 1992 (ICES CM 1992/G:70), Feb 1991, IYFS/IBTS GOV ----------------
# Columns: class lower bound (mm), total stomach content (g, "Total" row), Nephrops (g), n sampled, n empty.
# Transcribed from the layout text of the PDF (../../Refugium och utbyte i bentos/data/vliz_ocrd_274761);
# the "Total" rows sum to 4538.8 g (Skagerrak) and 9344.1 g (Kattegat) as printed.
DORMAN = {
    "Skagerrak": [(100, 1.3, 0, 11, 3), (120, 14, 0, 39, 12), (150, 69.1, 0, 91, 13), (200, 105, 0, 54, 6),
                  (250, 52, 0, 10, 1), (300, 48.9, 0, 12, 0), (350, 121, 11.1, 13, 1), (400, 607, 6.4, 43, 2),
                  (500, 607, 11.2, 24, 0), (600, 1261, 70.0, 25, 4), (700, 768, 59.4, 8, 0),
                  (800, 792, 0, 5, 0), (900, 91.6, 0, 1, 0)],
    "Kattegat": [(100, 0.5, 0, 3, 0), (120, 5.65, 0, 15, 1), (150, 41.3, 2.2, 78, 10), (200, 104, 2.5, 82, 12),
                 (250, 303, 0, 81, 10), (300, 427, 0, 82, 10), (350, 895, 5.7, 91, 8), (400, 1629, 37.0, 96, 6),
                 (500, 1628, 52.3, 40, 1), (600, 2047, 41.6, 41, 3), (700, 1326, 42.6, 27, 3),
                 (800, 751, 0, 13, 3), (900, 186, 0, 4, 0)],
}


def cls_of(L_cm):
    for (a, b), c in zip(EDGES, CLS):
        if a <= L_cm < b:
            return c
    return None


profiles = {}   # name -> {cls: (food_g, neph_g, n_fed)}
for area, rows in DORMAN.items():
    d = {c: [0.0, 0.0, 0] for c in CLS}
    for lo, tot, neph, n, ne in rows:
        c = cls_of(lo / 10.0)
        d[c][0] += tot; d[c][1] += neph; d[c][2] += n - ne
    profiles[f"Dorman1992_{area}_Feb1991"] = d

ICES_SETS = {
    "ICES_NNS_Q1_1981": ("cod", "NNS", "1981"),
    "ICES_NNS_Q1_2023": ("cod", "NNS", "2023"),
    "ICES_NNS_100-200m_Q1_2023": ("cod", "NNS_d100-200", "2023"),
    "ICES_Skagerrak_2023-25": ("cod", "Skagerrak", "2000+"),
    "ICES_Kattegat_2021-25": ("cod", "Kattegat", "2000+"),
    "ICES_CSNS_50-100m_2023": ("cod", "CSNS_d50-100", "2023"),
}
srows = list(csv.DictReader(open(shares_f)))
for name, (sp, ar, per) in ICES_SETS.items():
    d = {c: [0.0, 0.0, 0] for c in CLS}
    for r in srows:
        if r["predator"] == sp and r["area"] == ar and r["period"] == per and r["size_cm"] in CLS:
            d[r["size_cm"]] = [float(r["food_g"]), float(r["nephrops_g"]), int(r["fed_records"])]
    profiles[name] = d

# ---- 2. cod biomass by length in the grid region, and mean weight per class --------------------------
lb = [r for r in csv.DictReader(open(lb_f)) if r["species"] == "cod"]
mix, wmean = {}, {}
for col in ("share_2001_25", "share_2021_25"):
    m = {c: 0.0 for c in CLS}; ww = {c: 0.0 for c in CLS}
    for r in lb:
        c = cls_of(float(r["L_cm"]))
        s = float(r[col])
        m[c] += s; ww[c] += s / float(r["w_g"])          # biomass-weighted harmonic mean weight
    mix[col] = m
    wmean[col] = {c: (m[c] / ww[c]) if ww[c] > 0 else 1.0 for c in CLS}
if lcp_f:   # box length mix used by the 221 scripts (variant A_floor, quarters pooled), mean weights from 2001-25
    for r in csv.DictReader(open(lcp_f)):
        if r["variant"] == "A_floor" and r["quarter"] == "pooled" and r["species"] == "cod":
            g = lambda k: float(r[k])
            m = {"00-20": g("[0, 10)") + g("[10, 15)") + g("[15, 20)"), "20-30": g("[20, 25)") + g("[25, 30)"),
                 "30-40": g("[30, 40)"), "40-50": g("[40, 50)"), "50-70": g("[50, 60)") + g("[60, 70)"),
                 "70-": g("[70, 200)")}
            t = sum(m.values())
            mix["box_A_floor"] = {k: v / t for k, v in m.items()}
            wmean["box_A_floor"] = wmean["share_2001_25"]


def share_of(profile, mixcol, cons_weight, min_fed=5):
    """size-weighted Nephrops % of total food; classes with < min_fed fed fish borrow the pooled
    share of the neighbouring classes (next larger, then smaller)."""
    m = mix[mixcol]
    num = den = 0.0
    for i, c in enumerate(CLS):
        food, neph, nf = profile[c]
        if nf < min_fed or food <= 0:
            f2 = n2 = 0.0; k = 0
            for j in (i + 1, i - 1, i + 2, i - 2):
                if 0 <= j < len(CLS) and profile[CLS[j]][2] > 0:
                    f2 += profile[CLS[j]][0]; n2 += profile[CLS[j]][1]; k += profile[CLS[j]][2]
                    if k >= min_fed:
                        break
            s = n2 / f2 if f2 > 0 else 0.0
        else:
            s = neph / food
        w = m[c] * (wmean[mixcol][c] ** -0.2 if cons_weight else 1.0)
        num += w * s; den += w
    return num / den


os.makedirs(outd, exist_ok=True)
with open(os.path.join(outd, "cod_size_profiles.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["data_set", "size_cm", "fed_fish", "food_g", "nephrops_g", "nephrops_pct_mass"])
    for name, d in profiles.items():
        for c in CLS:
            food, neph, nf = d[c]
            wr.writerow([name, c, nf, round(food, 2), round(neph, 2),
                         round(100 * neph / food, 2) if food > 0 else ""])
        tf = sum(v[0] for v in d.values()); tn = sum(v[1] for v in d.values())
        wr.writerow([name, "pooled", sum(v[2] for v in d.values()), round(tf, 1), round(tn, 1),
                     round(100 * tn / tf, 2) if tf else ""])

print("Cod Nephrops % of total stomach mass by size class (fed fish in brackets):")
for name, d in profiles.items():
    tf = sum(v[0] for v in d.values()); tn = sum(v[1] for v in d.values())
    print(f"  {name:28s} " + "  ".join(f"{c}: {100*d[c][1]/d[c][0]:5.2f} ({d[c][2]})" if d[c][0] > 0
                                         else f"{c}:   -   ({d[c][2]})" for c in CLS)
          + f"   pooled {100*tn/tf:5.2f}")
print("\nGrid cod biomass by length class (NS-IBTS, length_biomass_longrun.csv):")
for col in mix:
    print(f"  {col}: " + "  ".join(f"{c} {100*mix[col][c]:4.1f} %" for c in CLS))

cod_tab = []
for name, d in profiles.items():
    row = [name]
    for col in ("share_2001_25", "share_2021_25", "box_A_floor"):
        if col not in mix:
            row += ["", ""]; continue
        for cw in (False, True):
            row.append(round(100 * share_of(d, col, cw), 2))
    cod_tab.append(row)
with open(os.path.join(outd, "cod_share_by_mix.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["data_set", "pct_mix2001_25_biomass", "pct_mix2001_25_consumption_W-0.2",
                 "pct_mix2021_25_biomass", "pct_mix2021_25_consumption_W-0.2",
                 "pct_boxAfloor_biomass", "pct_boxAfloor_consumption_W-0.2"])
    wr.writerows(cod_tab)
print("\nCod Nephrops % of total diet, weighted by the grid's size structure "
      "(2001-25 biomass | 2001-25 W^-0.2 | 2021-25 biomass | 2021-25 W^-0.2 | box A_floor biomass | W^-0.2):")
for r in cod_tab:
    print(f"  {r[0]:28s} " + " | ".join(f"{x:5.2f}" if x != "" else "  -  " for x in r[1:]))

# ---- 3-6. scenarios -------------------------------------------------------------------------------------
qb = {r["species"]: (float(r["share_pct"]) / 100, float(r["qb_central"])) for r in csv.DictReader(open(qb_f))}
WSUM = sum(b * q for b, q in qb.values())
cons_share = {sp: b * q / WSUM for sp, (b, q) in qb.items()}
print("\nConsumption weights (biomass share x Q/B, normalised): " +
      ", ".join(f"{k} {v:.3f}" for k, v in cons_share.items()))

QB_GROUP, B_GROUP, B_NEPH = 4.46, 1.53, 4.572
EPI_MEASURED, EPI_ALL = 0.2182, 0.3143
P_B_NEPH = 0.34      # E1 r_max / P/B for Nephrops (Hiddink 2019, M 0.2-0.3, Ecopath 0.35-0.4)

# Per-species Nephrops share of total diet (fraction), stomach-mass basis BEFORE levers, per scenario.
# cod: from the profiles above (2021-25 biomass mix, the grid today); others: data or J.
c = {r[0]: r for r in cod_tab}
cod_skag91 = c["Dorman1992_Skagerrak_Feb1991"][3] / 100
cod_nns81 = c["ICES_NNS_Q1_1981"][3] / 100
cod_nns23 = c["ICES_NNS_Q1_2023"][3] / 100
cod_nns23_grounds = c["ICES_NNS_100-200m_Q1_2023"][3] / 100
cod_skag23 = c["ICES_Skagerrak_2023-25"][3] / 100
cod_kat91 = c["Dorman1992_Kattegat_Feb1991"][3] / 100

# Levers on the COD share only (multiplicative on its Q1 stomach-mass share):
#  digestion: stomach mass -> intake. Armstrong et al. 1991 (Irish Sea cod): Nephrops 20-40 % of stomach
#             mass = 12-28 % of daily intake once the slow digestion of prawns is allowed for; ratio 0.6-0.7 [F].
#  season:    Q1 stomach share -> ration-weighted annual mean. season_factor.py on the North Sea 1981 and
#             1991 stomach projects (CRR 164 / CRR 219 Table 8, cod ages 2+): median 2.06 (1991 1.72,
#             1981 3.28; range 1.49-4.62) [F, computed].
# Other species: annual intake shares given directly (no levers), see the report:
#  haddock  0.003 (0-0.01): ICES-DB NNS Q1 2025 0/306 fed; CRR 164 1981 none in ~17 500 stomachs; CRR 219
#           1991 0.3-1.1 % annual by age (Q1/Q4 > Q2/Q3); DAPSTOM IVa FO 0.27 % (5/1834) [F/J]
#  long rough dab 0.005 (0.002-0.014): ICES-DB 1991 CSNS 0.4 %, NNS 0; Mackinson & Daskalov 2007 1.4 % [F/J]
#  witch    0 (0-0.01): Bergstad et al. 2003 decapods 10 % unresolved, DAPSTOM IVa 0/43 [F/J]
#  plaice   0 (0-0.003): ICES-DB 2024 0/396 fed Skagerrak/Kattegat/NNS; DAPSTOM IVa FO 1/81 [F/J]
CENTRAL_SEASON, CENTRAL_DIG = 2.06, 0.65
SCEN = {
    #  name: (cod Q1 stomach share, haddock, long_rough_dab, dab, plaice, witch, lemon_sole, others, digestion, season)
    "A_221_note (Skagerrak 2023-25 ICES only, no levers)": (cod_skag23, 0.0, 0.002, 0, 0, 0, 0, 0, 1.0, 1.0),
    "G_221 codnns recomputed (NNS 2023 own B share, no levers)": (cod_nns23, 0.0, 0.002, 0, 0, 0, 0, 0, 1.0, 1.0),
    "LOW (Kattegat Feb 1991 profile, season 1.5, dig 0.6)": (cod_kat91, 0.0, 0.002, 0, 0, 0, 0, 0, 0.6, 1.5),
    "CENTRAL (Skagerrak Feb 1991 profile)": (cod_skag91, 0.003, 0.005, 0, 0, 0, 0, 0, CENTRAL_DIG, CENTRAL_SEASON),
    "alt NNS Q1 1981 profile": (cod_nns81, 0.003, 0.005, 0, 0, 0, 0, 0, CENTRAL_DIG, CENTRAL_SEASON),
    "alt NNS Q1 2023 profile": (cod_nns23, 0.003, 0.005, 0, 0, 0, 0, 0, CENTRAL_DIG, CENTRAL_SEASON),
    "HIGH (NNS 2023 100-200 m profile, dig 0.7)": (cod_nns23_grounds, 0.01, 0.014, 0, 0.003, 0.01, 0, 0,
                                                    0.7, CENTRAL_SEASON),
    "EXTREME corner (HIGH with season 3.3 = 1981)": (cod_nns23_grounds, 0.01, 0.014, 0, 0.003, 0.01, 0, 0, 0.7, 3.28),
}
SP = ["cod", "haddock", "long_rough_dab", "dab", "plaice", "witch", "lemon_sole", "others"]
out = []
print("\nScenario: cod share (stomach) -> cod intake share; group p; p/epi_measured; p/epi_all; "
      "flow t/km2/yr; M2 (QB 3.4 / 4.46 / 5.0); fish flow / Nephrops production (P/B 0.34)")
for name, v in SCEN.items():
    sp_share = dict(zip(SP, v[:8])); dig, sea = v[8], v[9]
    intake = dict(sp_share); intake["cod"] = sp_share["cod"] * dig * sea
    p = sum(cons_share[k] * intake[k] for k in SP if k in cons_share)
    flow = QB_GROUP * B_GROUP * p
    m2 = {q: q * B_GROUP * p / B_NEPH for q in (3.4, 4.46, 5.0)}
    ratio = flow / (P_B_NEPH * B_NEPH)
    out.append([name, round(100 * sp_share["cod"], 2), dig, sea, round(100 * intake["cod"], 2),
                round(100 * intake["haddock"], 2), round(100 * p, 3), round(p / EPI_MEASURED, 4),
                round(p / EPI_ALL, 4), round(flow, 4), round(m2[3.4], 4), round(m2[4.46], 4), round(m2[5.0], 4),
                round(100 * ratio, 1)])
    print(f"  {name:48s} cod {100*sp_share['cod']:5.2f} % -> {100*intake['cod']:5.2f} %;  p {100*p:5.3f} %;  "
          f"{p/EPI_MEASURED:6.4f}; {p/EPI_ALL:6.4f};  flow {flow:6.4f};  M2 {m2[3.4]:6.4f} / {m2[4.46]:6.4f} / "
          f"{m2[5.0]:6.4f};  {100*ratio:5.1f} % of P")
with open(os.path.join(outd, "group_share_scenarios.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["scenario", "cod_pct_stomach_mass_sizeweighted_2021_25", "digestion_factor", "season_factor",
                 "cod_pct_intake", "haddock_pct_intake", "group_pct_of_consumption",
                 "share_of_measured_benthos_epifauna_0.2182", "share_of_all_epifauna_0.3143",
                 "flow_t_km2_yr", "M2_QB3.4", "M2_QB4.46", "M2_QB5.0", "fish_flow_pct_of_production_PB0.34"])
    wr.writerows(out)
