"""Fields, diets and pair M2 of a mackerel group split out of today's water column (resume 223.26 G1_wc).

Usage: python3 -I split_fields_mack.py <fg_predators_dir> <out_dir> [densities]

[densities] = "mackerel=0.80,horse mackerel=0.05": field levels (t/km2, box annual means) instead of
the GOV shares (research_notes/"Makrillens tathet" (Swedish letters: a-umlaut); GOV q for mackerel
~0.1). The mackerel group's B is their sum and its member weights their ratio. Its own M2 on its prey
follow its members' biomass: zooplankton by the members' consumption at the same calibration as the
pooled pair (library 0.0756 / the raw pooled diet M2), planktivores by the SMS partial M2 per member
scaled to the member's biomass (223.37 box scaling); the rest's diet flow on it is kept, so that M2 is
flow / B. The rest water column is unchanged. The prey totals then move by design (printed). Outputs
get a "_field" suffix.

LOCAL F (user 2026-10-30, resume 228.1): with densities, mackerel carries the LOCAL fishing mortality of
3.a.20 instead of the stock F: catch / mean biomass = 343 t/yr (ICES official 2016-24: Sweden 182,
Denmark 55, Norway 105) / 25 945 t (the survey-based Skagerrak annual mean) = 0.0132 /yr
(research_notes/"Makrillens tathet" tables/annual_mean_stdout.txt, catch_notes.md). Gear split: the EU
part (FDI 222 t/yr) is gillnets 0.45, hooks 0.32, bottom trawls 0.17, midwater / pelagic trawls 0.038
(fdi_eu_3a20_mac_hom_stdout.txt); Norway's part is trolling and handlines. Only the midwater-trawl
share goes on the pelagic_trawling map (0.038 x 222/343 = 0.0246); the rest on the bottom_trawling map,
which carries the static gears (gillnet, longline; 223.26) - J for the hook catch. Horse mackerel keeps
its stock values (0.003 + 0.035), which match its local 2020-24 catch / field biomass (~0.03).

Construction as split_fields_223.py / section 209 weighted_fields.py: biomass-weighted species values
(predators/inputs/species_values.csv) with the long-run (2001-25) NS-IBTS GOV shares; B* = 1.20 t/km2 x
the members' share (top of the long-run span, user's choice). No model output is read.

  mackerel group: mackerel, horse mackerel                      (layer mean_0_50m, user 2026-10-30)
  rest water column (`predatory_fish`): saithe, spurdog, hake, anglerfish, ling, velvet belly, pollack
                                                                 (layer at_50m, unchanged)

Reproduction gate: today's water column pooled (G1_wc, the nine members at at_50m) must land on the
live library block (r 0.374, M1 0.261, F 0.2128 + 0.0291, RM 3.31, ceiling 32.71 with Bjornsson x1.30,
lean 4240, reserve 2600, fill 0.5). Pair M2 (today's water-column pairs, the targets the split must add
back to exactly, so no prey's derived growth moves):
  planktivores 0.257 (223.37: SMS partial M2 scaled to the box, by SMS predator: saithe 0.209, hake
      0.037, horse mackerel 0.008, mackerel 0.002 -> mackerel group 0.010, rest the remainder 0.247);
  zooplankton 0.0756 (split by the members' zooplankton consumption Q/B x B x Z);
  benthivores 0.0833 and whiting 0.157: all on the rest (mackerel and horse mackerel have capability
      score 0.01 / 0.05 < 0.1 on fish of the benthivore pool: layer a 0, split_encounter_mack.py).
Cmax and SMR in species_values.csv are at the at_50m annual mean; each group's values are moved to
its layer's annual mean (Cmax with the ICES GER temperature term: gadoids 0.078, mackerel 0.080 per C;
SMR with the member Q10).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

d, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
TICKS = 1460.0
B_TOTAL = 1.20
BJORNSSON_X = 1.30
V = pd.read_csv(os.path.join(d, "inputs", "species_values.csv")).set_index("species")
S = pd.read_csv(os.path.join(d, "tables", "pred_season.csv")).set_index("common")
W = S.loc[V.index, "pooled"]
W = W / W.sum()
DC = pd.read_csv(os.path.join(d, "inputs", "diet_consumption_central.csv"))
DC["species"] = DC["species"].str.replace("_", " ")
DC = DC.set_index("species")
PROXY = {"ling": "anglerfish", "pollack": "saithe", "velvet belly": "horse mackerel"}  # as split_fields_223
T_LAYER = {"at_50m": 8.367, "mean_0_50m": 9.308}  # mareld2.yaml annual means
GER_K = {"mackerel": 0.080, "horse mackerel": 0.080}  # ICES WGSAM Table 3-7 (mackerel 0.080); others 0.078
# Trawl tables are linear in effort at today's mean (library 223.26): biomass_factor = F x c
C_BOT = 0.0048361803927 / 0.2128243650582523      # per F at value 100
C_PEL = 0.0079827776176 / 0.029092991818223186    # per F at value 10

WC = ["saithe", "mackerel", "spurdog", "horse mackerel", "hake", "anglerfish", "ling", "velvet belly", "pollack"]
MACK = ["mackerel", "horse mackerel"]
REST = [s for s in WC if s not in MACK]


# ---- mackerel group physiology with explicit reference temperatures (resume 228.2) ----
# research_notes/"Fiskgrupper i rutan"/predators/evidence_A_physiology.md. The species_values.csv
# rows for mackerel and horse mackerel carry values at 13 C (SMR, routine) or a judgement (Cmax 150)
# that split_fields_223 read as at_50m (8.37 C) values; with densities the group is rebuilt here.
#  * rest cost: mackerel has no swim bladder and cannot rest: its floor is the routine rate at 1 BL/s,
#    Bachiller et al. 2018 R = 0.00264 W^-0.217 e^(0.06818 T) e^(0.0234 SW): 74 mg O2/kg/h = 6.07
#    MJ/t/6h at 400 g, 13 C [F, D]; Q10 e^0.6818 = 1.98. Its true SMR (chub mackerel proxy, U = 0)
#    3.37 at 13 C. Horse mackerel has a swim bladder: rest = SMR 3.4 at 13 C (mackerel proxy), Q10 2.2.
#  * routine (feeding baseline): mackerel = its floor (already swimming); horse mackerel 1.5 x SMR.
#  * moving one 1-km cell per 6 h needs ~0.05 m/s, far below the routine 1 BL/s (~7.6 km per tick):
#    directed swimming costs mackerel nothing above its floor (J) -> 1.0 x floor; horse mackerel the
#    group convention 2.0 x SMR (cruising at Uopt).
#  * Cmax anchored on the field, not on the GER construction: the GER ceiling at S 2.5 % is 109-118
#    kJ/kg/d at 12-13 C (x1.30 Bjornsson 142-153) and fails both field checks - the fattening budget
#    (fat 5 -> 25 % in 100 d) needs >= 180 kJ/kg/d, and the Norwegian Sea feeding season ration is
#    0.08 g prey/g/d = 8 % BW/d of ~3.6 kJ/g zooplankton ~ 290 kJ/kg/d (Bachiller et al. 2018 [F],
#    a bioenergetics inversion of measured growth; feeding grounds ~8-12 C, J 10 C). A ceiling cannot
#    sit below the mean ration: Cmax = 290 at 10 C (the lowest ceiling consistent with it), no
#    Bjornsson factor (field-anchored). Horse mackerel: Cape horse mackerel ration 3.8 % BW/d of
#    crustaceans (~4 kJ/g) = 152 kJ/kg/d [A], J 13 C. Temperature with the mackerel GER term 0.080/C.
MACK_PHYS = {"mackerel": dict(rest=6.07, rest_T=13.0, q10=1.98, routine_x=1.0, move_x=1.0,
                              cmax=290.0, cmax_T=10.0),
             "horse mackerel": dict(rest=3.4, rest_T=13.0, q10=2.2, routine_x=1.5, move_x=2.0,
                                    cmax=152.0, cmax_T=13.0)}
GER_K_MACK = 0.080


def mackerel_physiology(w, T):
    """Group rest cost, routine, movement cost, Cmax and ceiling at layer temperature T (biomass weights w)."""
    rest = {sp: p["rest"] * p["q10"] ** ((T - p["rest_T"]) / 10.0) for sp, p in MACK_PHYS.items()}
    RM = float(sum(w[sp] * rest[sp] for sp in MACK_PHYS))
    routine = float(sum(w[sp] * rest[sp] * MACK_PHYS[sp]["routine_x"] for sp in MACK_PHYS))
    move = float(sum(w[sp] * rest[sp] * MACK_PHYS[sp]["move_x"] for sp in MACK_PHYS))
    cm = {sp: p["cmax"] * np.exp(GER_K_MACK * (T - p["cmax_T"])) for sp, p in MACK_PHYS.items()}
    cmax = float(sum(w[sp] * cm[sp] for sp in MACK_PHYS))
    ceil = cmax / 4.0
    return dict(rest_by_member=rest, cmax_by_member=cm, smr=RM, routine=routine, routine_x=routine / RM,
                movement_cost=move / RM, cmax=cmax, ceiling_field=ceil,
                feeding_cost=routine / RM + 0.17 * 0.5 * ceil / RM, starve_rate=routine / 0.15)


LOCAL_F_MACKEREL = 343.0 / 25945.0
PEL_SHARE_MACKEREL = 0.038 * 222.0 / 343.0


def fields(members, layer, dens=None):
    w = (pd.Series(dens) / sum(dens.values()))[members] if dens else W[members] / W[members].sum()
    v = V.loc[members]
    wm = lambda c: float((w * v[c]).sum())
    dT = T_LAYER[layer] - T_LAYER["at_50m"]
    t_cmax = float(sum(w[s] * np.exp(GER_K.get(s, 0.078) * dT) for s in members))
    q10 = wm("q10")
    t_smr = float(q10 ** (dT / 10.0))
    B = sum(dens.values()) if dens else B_TOTAL * float(W[members].sum())
    cm = wm("cmax_kJkgd") * t_cmax
    ceil = cm / 4.0 * BJORNSSON_X
    smr = wm("smr") * t_smr
    el, rmx, rmean = wm("e_lean"), wm("res_max"), wm("res_mean")
    Fb, Fp = wm("F_bot"), wm("F_pel")
    if dens and "mackerel" in members:
        loc = {"mackerel": (LOCAL_F_MACKEREL * (1 - PEL_SHARE_MACKEREL), LOCAL_F_MACKEREL * PEL_SHARE_MACKEREL)}
        Fb = float(sum(w[sp] * (loc[sp][0] if sp in loc else v.loc[sp, "F_bot"]) for sp in members))
        Fp = float(sum(w[sp] * (loc[sp][1] if sp in loc else v.loc[sp, "F_pel"]) for sp in members))
    phys = {}
    if dens and "mackerel" in members:
        phys = mackerel_physiology(w, T_LAYER[layer])
        phys["starve_rate"] = phys["starve_rate"] / wm("e_lean")
    out = dict(
        members=members, weights={k: round(float(x), 4) for k, x in w.items()}, layer=layer,
        T_layer=T_LAYER[layer], t_cmax=t_cmax, t_smr=t_smr, share_of_pooled=float(W[members].sum()),
        B=B, tonnes=B * 3600, r=wm("r_now"), r_lo=wm("r_lo"), r_hi=wm("r_hi"), F_bot=Fb, F_pel=Fp,
        M1=wm("natmort"), M1_lo=wm("nm_lo"), M1_hi=wm("nm_hi"), cmax=cm, ceiling_bjornsson=ceil,
        smr=smr, smr_lo=wm("smr_lo") * t_smr, smr_hi=wm("smr_hi") * t_smr, q10=q10,
        e_lean=el, res_max=rmx, res_mean=rmean, fill=rmean / rmx, E_prey=el + rmean,
        qb=wm("qb"), feeding_cost=1.5 + 0.17 * 0.5 * ceil / smr, starve_rate=1.5 * smr / el / 0.15,
        bf_bottom_100=Fb * C_BOT, bf_pelagic_10=Fp * C_PEL)
    out.update(phys)
    return out


def zoo_consumption(members):
    return {sp: float(V.loc[sp, "qb"] * W[sp] * B_TOTAL * DC.loc[PROXY.get(sp, sp), "Z"]) for sp in members}


DENS = dict((k, float(v)) for k, v in (kv.split("=") for kv in sys.argv[3].split(","))) if len(sys.argv) > 3 else None
SUFFIX = "_field" if DENS else ""
res = {"wc_pooled": fields(WC, "at_50m"), "mackerel": fields(MACK, "mean_0_50m", DENS),
       "predatory_fish": fields(REST, "at_50m")}
p = res["wc_pooled"]
print("REPRODUCTION GATE: today's water column (nine members, at_50m) vs the live library")
lib = dict(r=0.374, M1=0.261, F_bot=0.2128, F_pel=0.0291, smr=3.31, ceiling_bjornsson=32.71, e_lean=4240,
           res_max=2600, fill=0.5, feeding_cost=2.34, starve_rate=0.0078)
ok = True
for k, x in lib.items():
    y = p[k]
    good = abs(y - x) <= 0.01 * abs(x) + 0.006 * (k in ("fill",)) + 0.5 * (k in ("e_lean", "res_max"))
    good = good or (k in ("e_lean", "res_max") and abs(y - x) / x < 0.01)
    ok &= good
    print(f"  {k:18s} derived {y:10.4f}  library {x:10.4f}  {'ok' if good else 'MISMATCH'}")
print(f"  gate {'PASS' if ok else 'FAIL'}")

# ---- pair M2 ----
LIB_M2 = {"planktivorous_fish": 0.257, "zooplankton": 0.0756, "benthivorous_fish": 0.0833, "whiting": 0.157}
SMS_PLK = {"saithe": 0.209, "hake": 0.037, "horse mackerel": 0.008, "mackerel": 0.002}  # 223.37 comment
zc = zoo_consumption(WC)
zm = sum(zc[s] for s in MACK) / sum(zc.values())
m2 = {"mackerel": {}, "predatory_fish": {}}
m2["mackerel"]["planktivorous_fish"] = SMS_PLK["mackerel"] + SMS_PLK["horse mackerel"]
m2["predatory_fish"]["planktivorous_fish"] = LIB_M2["planktivorous_fish"] - m2["mackerel"]["planktivorous_fish"]
m2["mackerel"]["zooplankton"] = LIB_M2["zooplankton"] * zm
m2["predatory_fish"]["zooplankton"] = LIB_M2["zooplankton"] * (1 - zm)
m2["predatory_fish"]["benthivorous_fish"] = LIB_M2["benthivorous_fish"]
m2["predatory_fish"]["whiting"] = LIB_M2["whiting"]
print("\nZOOPLANKTON consumption shares (Q/B x B x Z): " + ", ".join(f"{s} {zc[s] / sum(zc.values()):.3f}" for s in WC))
print("\nPAIR M2 (conservation gate: mackerel + rest = today's water-column pair)")
gate = True
for prey, tgt in LIB_M2.items():
    s = m2["mackerel"].get(prey, 0.0) + m2["predatory_fish"].get(prey, 0.0)
    good = abs(s - tgt) < 1e-12
    gate &= good
    print(f"  {prey:20s} mackerel {m2['mackerel'].get(prey, 0.0):.5f} + rest {m2['predatory_fish'].get(prey, 0.0):.5f}"
          f" = {s:.5f} vs {tgt:.5f}  {'PASS' if good else 'FAIL'}")
print(f"  conservation {'PASS' if gate else 'FAIL'}")
if DENS:
    ZOO_B = 35000 / 3600
    raw = lambda sp, b: float(V.loc[sp, "qb"] * b * DC.loc[PROXY.get(sp, sp), "Z"]) / ZOO_B
    k_zoo = LIB_M2["zooplankton"] / sum(raw(sp, W[sp] * B_TOTAL) for sp in WC)
    m2["mackerel"]["zooplankton"] = k_zoo * sum(raw(sp, DENS[sp]) for sp in MACK)
    m2["mackerel"]["planktivorous_fish"] = sum(SMS_PLK[sp] * DENS[sp] / (W[sp] * B_TOTAL) for sp in MACK)
    res["mackerel"].update(B_gov=float(B_TOTAL * W[MACK].sum()),
                           dens_over_gov={sp: DENS[sp] / float(W[sp] * B_TOTAL) for sp in MACK})
    print(f"\nFIELD DENSITIES {DENS} (GOV " + ", ".join(f"{sp} {W[sp] * B_TOTAL:.4f}" for sp in MACK)
          + f"): calibration k_zoo {k_zoo:.3f}; the mackerel group's M2 and the prey totals now:")
    for prey in ("planktivorous_fish", "zooplankton"):
        tot = m2["mackerel"][prey] + m2["predatory_fish"][prey]
        print(f"  {prey:20s} mackerel {m2['mackerel'][prey]:.5f} + rest {m2['predatory_fish'][prey]:.5f} = {tot:.5f}"
              f" (today {LIB_M2[prey]:.5f}, +{tot - LIB_M2[prey]:.5f})")
for g in ("mackerel", "predatory_fish"):
    res[g]["M2"] = m2[g]

# ---- the new cross pair: rest water column on the mackerel group (spurdog, hake, anglerfish) ----
# split_fields_223 WCOMP: spurdog W 0.22 x (mackerel 0.33 + horse mackerel 0.17); hake W 0.21 x
# (mackerel 0.05 + horse mackerel 0.05); anglerfish and ling W x horse mackerel 0.2.
WSH = {"spurdog": 0.50, "hake": 0.10, "anglerfish": 0.2, "ling": 0.2, "velvet belly": 0.88}
flow = 0.0
for sp, sh in WSH.items():
    flow += float(V.loc[sp, "qb"] * W[sp] * B_TOTAL * DC.loc[PROXY.get(sp, sp), "W"]) * sh
res["predatory_fish"]["M2"]["mackerel"] = flow / res["mackerel"]["B"]
print(f"\nREST -> MACKEREL GROUP: flow {flow:.5f} t/km2/yr, M2 {flow / res['mackerel']['B']:.4f} /yr "
      "(velvet belly's horse mackerel share is cannibal-like: small; kept as the diet says)")

for g in ("mackerel", "predatory_fish"):
    f = res[g]
    print(f"\n--- {g}: {f['members']}  weights {f['weights']}")
    print(f"  B* {f['B']:.4f} t/km2 = {f['tonnes']:.0f} t (share of pooled 1.20: {f['share_of_pooled']:.4f})")
    print(f"  r_max {f['r']:.3f} ({f['r_lo']:.3f}-{f['r_hi']:.3f}); F bottom {f['F_bot']:.4f}, pelagic {f['F_pel']:.4f};"
          f" M1 {f['M1']:.3f} ({f['M1_lo']:.3f}-{f['M1_hi']:.3f}) = {f['M1'] / TICKS:.6g} /tick")
    print(f"  layer {f['layer']} ({f['T_layer']:.2f} C): Cmax x{f['t_cmax']:.4f}, SMR x{f['t_smr']:.4f}; Cmax {f['cmax']:.1f}"
          f" kJ/kg/d -> ceiling x{BJORNSSON_X} {f['ceiling_bjornsson']:.2f} MJ/t/tick")
    print(f"  SMR {f['smr']:.3f} ({f['smr_lo']:.2f}-{f['smr_hi']:.2f}), Q10 {f['q10']:.2f}, feeding_cost {f['feeding_cost']:.2f},"
          f" starve_rate {f['starve_rate']:.4f}")
    print(f"  lean {f['e_lean']:.0f}, max reserve {f['res_max']:.0f}, mean reserve {f['res_mean']:.0f}, fill {f['fill']:.3f},"
          f" prey quality {f['E_prey']:.0f} MJ/t; Q/B {f['qb']:.2f}")
    print(f"  trawl tables: bottom bf at 100 {f['bf_bottom_100']:.11g}, pelagic bf at 10 {f['bf_pelagic_10']:.11g}")
    print("  M2 booked on prey: " + ", ".join(f"{k} {x:.5f}" for k, x in f["M2"].items()))
json.dump(res, open(os.path.join(out, f"split_fields_mack{SUFFIX}.json"), "w"), indent=1, default=float)
