"""Model side of the winter krill test: the library's krill pairs in winter, the field-implied a,
and the fix97 ledger's winter krill flow. Reads only: fgconfig/fg_library.yaml (today's, = 270; the
krill pairs are unchanged since 269), mareld2.yaml, maps/depth.npz, and the main agent's
`Ransonen i energi/tables/ledger_fix97.json` (frozen fix97, probe world, 269 library).

    python3 scripts/model_side.py   (run from the repository root or anywhere; paths are absolute)

Conventions (section 220 / 245): a is per t predator per (t/km2 of prey in the 1 km2 cell) per 6-h
tick, Type II, single-prey intake I = a B / (1 + a h B) t/t/tick. ingestion_q10 scales a and 1/h
together (annual mean 1); the daylight lever m(t) scales a (planktivores -> krill dark_ratio 0.57 at
0 deg; whiting none). The pair's depth_access_table and the krill pool's outside_depth_table are each
renormalised to an equal-cell mean of 1 on the depth map. Krill target density = 5 t/km2 x the
quarterly season x the depth weight (the pool sits ~0.89 of target in fix97).
"""
import json
import math
import os
import sys

import numpy as np
import yaml

ROOT = "/home/simon/eco-evo"
sys.path.insert(0, ROOT)
from lib.environments.ecosystem_env import open_pool  # noqa: E402
from lib.world import habitat  # noqa: E402

lib = yaml.safe_load(open(os.path.join(ROOT, "fgconfig/fg_library.yaml")))
man = yaml.safe_load(open(os.path.join(ROOT, "mareld2.yaml")))
SP, PAIRS = lib["species_definitions"], lib["interaction_definitions"]
depth = np.load(os.path.join(ROOT, "maps/depth.npz"))["depth"].astype(np.float64)
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
layers = man["simulation_settings"]["temperature"]["layers"]
glayer = man["simulation_settings"]["temperature"]["group_layers"]
# daylight multiplier m(t), monthly means printed by tools/probes/daylight_table.py (2026-11-12)
M_PLK_KRILL = [0.88, 0.93, 0.99, 1.05, 1.10, 1.14, 1.12, 1.07, 1.01, 0.95, 0.89, 0.86]
M_WHI_KRILL = [1.0] * 12

krill = SP["krill"]
wmap = open_pool.depth_weight_map(open_pool.parse_depth_table(krill), depth, depth.shape)
season = krill["outside_season"]
classes = [("<100", depth < 100), ("100-150", (depth >= 100) & (depth < 150)),
           ("150-250", (depth >= 150) & (depth < 250)), (">=250", depth >= 250)]
print(f"depth map {depth.shape}: median bed {np.median(depth):.0f} m; class shares "
      + ", ".join(f"{n} {m.mean():.2f}" for n, m in classes))


def qmult(g):
    sp = SP[g]
    q10 = float(sp["ingestion_q10"])
    T = np.array(layers[glayer[g]], dtype=float)
    raw = q10 ** (T / 10.0)
    return raw / raw.mean(), T


out = {}
for g, pair, mlist in (("planktivorous_fish", "planktivorous_fish_preys_on_krill", M_PLK_KRILL),
                       ("whiting", "whiting_preys_on_krill", M_WHI_KRILL)):
    P = PAIRS[pair]
    a, h = float(P["max_intake_rate"]), float(P["handling_time"])
    f = habitat.pair_depth_map(P, depth, pair)
    q, T = qmult(g)
    print(f"\n=== {pair}: a {a}, h {h} (1/h = {400/h:.2f} %BW/d at q 1), ingestion_q10 "
          f"{SP[g]['ingestion_q10']} on '{glayer[g]}' (T Jan {T[0]}); q: "
          + " ".join(f"{MON[i]} {q[i]:.2f}" for i in range(12)))
    print("    month  ceiling %BW/d | class: a_eff (x lib a) B_krill t/km2 aB h-sat  encounter-limited I %BW/d")
    res = {}
    for i in (10, 11, 0, 1, 2, 6):
        s = season[i]
        line = f"    {MON[i]:5s} {q[i]*400/h:6.2f}        |"
        for n, msk in classes:
            fc = float(f[msk].mean())
            Bc = 5.0 * s * float(wmap[msk].mean())
            aeff = a * fc * mlist[i] * q[i]
            heff = h / q[i]
            I = aeff * Bc / (1 + aeff * heff * Bc)
            sat = aeff * heff * Bc / (1 + aeff * heff * Bc)
            res[(MON[i], n)] = dict(aeff=aeff, heff=heff, B=Bc, I=I)
            line += f" {n}: {aeff:.4f} ({fc*mlist[i]*q[i]:.2f}) {Bc:.1f} {aeff*Bc*400:.0f}% {sat:.2f} {I*400:.2f} |"
        print(line)
    out[g] = res

# --- field-implied a ------------------------------------------------------------------------------
# a_field = R / (D (1 - h_eff R)), R krill ration t/t/tick, at the January h_eff, compared with the
# library's January a_eff in the field site's bed class. D brackets: 3 (winter floor), 8 (box winter
# column central), 20 and 72 t/km2 (Gullmarsfjord 4-16, Kattegat Alkor Deep 33-160: hotspot analogues).
FIELD = {
    "planktivorous_fish": [("group central (herring 0.554 x 0.15, pout 0.35 x 0.45)", 0.24, "100-150"),
                           ("group low", 0.10, "100-150"), ("group high", 0.46, "100-150"),
                           ("pout NE North Sea 1980 (ICES DB, GER)", 0.20, "100-150"),
                           ("pout Oslofjord Mar (Onsrud, GER)", 0.75, "100-150"),
                           ("herring Oslofjord Nov-Mar (Onsrud)", 0.15, "100-150")],
    "whiting": [("whiting NE+NW North Sea 1981 Q1 (counts, GER <=)", 0.45, "100-150"),
                ("whiting Skagerrak 1981 Q1 (counts, GER <=)", 0.14, "100-150"),
                ("whiting Oslofjord Mar (Onsrud, GER)", 0.75, "100-150")],
}
print("\n=== field-implied a against the library a (January, class 100-150 m; x = library / field)")
for g, rows in FIELD.items():
    r = out[g][("Jan", "100-150")]
    for name, Rpct, cls in rows:
        R = Rpct / 400.0
        line = f"  {g:18s} {name:52s} R {Rpct:.2f} %BW/d:"
        for D in (3.0, 8.0, 20.0, 72.0):
            af = R / (D * (1 - r["heff"] * R))
            line += f" D{D:.0f} a {af:.1e} (x{r['aeff']/af:.0f})"
        print(line)
    print(f"  {g:18s} library a_eff Jan 100-150 m = {r['aeff']:.4f}; model krill B there {r['B']:.1f} t/km2; "
          f"encounter-limited I {r['I']*400:.2f} %BW/d")

# --- in-layer encounter from the section 220 terms (order of magnitude) ---------------------------
print("\n=== in-layer encounter per fish, section 220 terms (scan 0.5, p_cap 0.5), krill 0.23 g WW")
for C in (0.2, 1.0, 5.0):
    for R_m in (0.1, 0.3):
        for v in (0.10, 0.15):
            caught = 0.5 * 0.5 * math.pi * R_m ** 2 * v * C * 86400
            print(f"  C {C} ind/m3, R {R_m} m, v {v} m/s: {caught:7.0f} krill/d = {caught*0.23:6.0f} g/d "
                  f"= {caught*0.23/100*100:5.0f} %BW/d of a 100 g fish; time in layer for 0.25 %BW/d: "
                  f"{0.25/(caught*0.23/100*100)*100:.2f} %")

# --- fix97 ledger: winter rations, krill flow, counterfactual -------------------------------------
L = json.load(open(os.path.join(ROOT, "research_notes/Ransonen i energi/tables/ledger_fix97.json")))
FISH = ["planktivorous_fish", "benthivorous_fish", "whiting", "predatory_fish", "mackerel"]
mon = {}
for row in L["rows"]:
    if row["year"] not in (1, 2):
        continue
    for g in FISH:
        d = row["groups"][g]
        m = mon.setdefault((g, row["month"]), dict(B=0.0, t=0.0, k=0.0, cost=0.0, gain=0.0, n=0))
        m["B"] += d["B"]
        m["t"] += sum(v for k, v in d.items() if k.startswith("t:"))
        m["k"] += d.get("t:krill", 0.0)
        m["cost"] += d["c_rest"] + d["c_eat"] + d["c_move"]
        m["gain"] += d["gain"]
        m["n"] += 1
print("\n=== fix97 ledger (years 2-3, per month): ration %BW/d total / krill, krill share of tonnes,"
      " assimilated gain and cost kJ/kg/d")
for g in ("planktivorous_fish", "whiting"):
    print(f"  {g}")
    for mo in MON:
        m = mon[(g, mo)]
        tot, kr = 400 * m["t"] / m["B"], 400 * m["k"] / m["B"]
        print(f"    {mo}: {tot:.2f} / {kr:.2f}  share {m['k']/m['t']:.2f}  gain {4*m['gain']/m['B']:.0f}"
              f"  cost {4*m['cost']/m['B']:.0f}")

WIN = ["Nov", "Dec", "Jan", "Feb", "Mar"]
ann = {g: sum(mon[(g, mo)]["k"] for mo in MON) / 2 for g in FISH}
win = {g: sum(mon[(g, mo)]["k"] for mo in WIN) / 2 for g in FISH}
tot = sum(ann.values())
print(f"\n=== krill eaten, t/yr (mean of years 2-3): five groups {tot:.0f}")
for g in FISH:
    print(f"  {g:18s} annual {ann[g]:7.0f}  Nov-Mar {win[g]:7.0f} ({win[g]/ann[g]:.2f} of its own; "
          f"{win[g]/tot:.3f} of the five-group total)")

# counterfactual: Nov-Mar krill intake of planktivores and whiting at the field krill ration
FIELD_R = {"planktivorous_fish": (0.10, 0.24, 0.46), "whiting": (0.14, 0.30, 0.50)}
print("\n=== Nov-Mar krill intake at the field krill ration (biomass and other months as realised)")
for lab, j in (("low", 0), ("central", 1), ("high", 2)):
    new_tot = tot
    parts = []
    for g, rr in FIELD_R.items():
        Bdays = sum(mon[(g, mo)]["B"] / 4.0 for mo in WIN) / 2  # tonne-days per year
        new = rr[j] / 100.0 * Bdays
        new_tot += new - win[g]
        parts.append(f"{g} {win[g]:.0f} -> {new:.0f} t (x{new/win[g]:.2f})")
    print(f"  {lab:7s}: " + "; ".join(parts) + f"; five-group krill flow {tot:.0f} -> {new_tot:.0f} "
          f"(-{100*(1-new_tot/tot):.0f} %)")

# fill: Dec-Mar energy balance of a planktivore eating the field ration
ME = float(SP["planktivorous_fish"]["max_energy_reserve"])
sda = float(SP["planktivorous_fish"]["sda_fraction"])
net_per_pct = 0.01 * 5000 * 0.8 * (1 - sda)  # MJ/t/d per 1 %BW/d of krill
print(f"\n=== planktivore Dec-Mar fill change at the field total ration (krill-equivalent, "
      f"{net_per_pct:.1f} kJ/kg/d per 1 %BW/d), ME {ME:.0f} MJ/t")
for cost_lab, cost in (("model realised cost", None), ("field routine 22 kJ/kg/d", 22.0)):
    for Rt in (0.45, 0.7, 1.0):
        dfill = 0.0
        for mo in ["Dec", "Jan", "Feb", "Mar"]:
            m = mon[("planktivorous_fish", mo)]
            c = 4 * m["cost"] / m["B"] if cost is None else cost
            dfill += (Rt * net_per_pct - c) * DAYS[MON.index(mo)] / ME
        print(f"  {cost_lab:26s} ration {Rt:.2f} %BW/d: fill change Dec-Mar {dfill:+.2f}"
              f"  (field start_fill Dec 0.68 -> Apr 0.20: -0.48)")
