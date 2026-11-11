"""Ling's encounter-implied flows per water-column pair (field spread) against its field ration and diet, per gate
run of gate_ling.py; and the WC pair ratios (implied / booked M2, prey side) for every prey ling eats.

Usage: python3 -I ling_flow_check.py <decompose_ling.json> <gate_ling.json>

Model side, per pair and run: the pair's implied / booked M2 at the field spread (gate prey side) x the booked flow
(library predation_mortality x prey biomass), split over members in proportion to s_m g_m f_m (benthivores: J and A
apart, with each member's stage p relative to the pair p and its A-profile factor, as pair_edit; whiting and
planktivores: f = js f_J + (1 - js) f_A with the prey J/B). Benthivore J / A split of the pair flow with the WC F_eff
J 0.655 / A 0.882 (research_notes/"Rekryternas bottendjup", as field_vs_model.py).
Field side: ling biomass 0.014 x 0.617 t/km2 x Q/B x diet shares (evidence.md); every field value is an input
listed below with its grade; nothing is fitted.
"""
import json
import sys

DM = json.load(open(sys.argv[1]))
GA = json.load(open(sys.argv[2]))
B_WC, W_LING = 0.617, 0.014
B_L = B_WC * W_LING
BOOK = {"benthivorous_fish": (0.0833, 1.53, 0.155), "whiting": (0.157, 0.446, 0.165),
        "planktivorous_fish": (0.174, 4.72, 0.38), "mackerel": (0.021, 0.85, None)}
PAIRK = {"benthivorous_fish": "predatory_fish->benthivorous_fish", "whiting": "predatory_fish->whiting",
         "planktivorous_fish": "predatory_fish->planktivorous_fish", "mackerel": "predatory_fish->mackerel"}
GRP = {"whiting": "wht", "planktivorous_fish": "plk"}
P0 = DM["p0"]
# field inputs: (central, lo, hi)
QB = (2.0, 1.3, 3.8)          # [J] central inside the two proxies: Palomares-Pauly 1.2-1.6 [D], SMS cod ration at ling weights 3.8 [S proxy]
BEN = (0.15, 0.05, 0.30)      # benthivore members in ling's diet by mass: Iceland haddock + cod ~15 % [F figure], flatfish not
                              # counted there; North Sea in-area 0-0.39 on 8-13 fed fish [F data, anecdotal]; Hywind demersal
                              # teleosts a minor share of identified fish items [F figure, counts]
AFR = (0.5, 0.2, 0.9)         # adult fraction of that share [J]: no in-area adult item; Iceland prey sizes not given
RUNS = ["prev_central", "win0.30", "act0.35", "act0.35s1", "central", "central+night0", "scan0.5+win0.30",
        "idle0.35+win0.30", "act0.50s1+win0.40", "act0.35s1+prof2+win0.30", "ling_out"]
VAR = {"prev_central": (None, None), "win0.30": (None, "ling@0.3"), "act0.35": ("act0.35", None),
       "act0.35s1": ("act0.35s1", None), "central": ("act0.35s1", "ling@0.3"),
       "central+night0": ("act0.35s1+night0", "ling@0.3"), "scan0.5+win0.30": ("act0.35", "ling@0.3"),
       "idle0.35+win0.30": ("idle0.35", "ling@0.3"), "act0.50s1+win0.40": ("act0.50s1", "ling@0.4"),
       "act0.35s1+prof2+win0.30": ("act0.35s1+prof2", "ling@0.3"), "ling_out": ("OUT", None)}
APROF = {}
for k, v in DM["members"]["predatory_fish->benthivorous_fish"].items():
    sp = k.split("|")[0]
    APROF.setdefault(sp, []).append((v["share"], v["Aprof"]))
APROF = {sp: sum(s * f for s, f in l) / sum(s for s, f in l) for sp, l in APROF.items()}


def ratio(run, prey, which):
    r = GA["summary"][run]["prey"][prey]["predatory_fish"]
    return (r[1] if which == "eq" else r[2]) / r[0]


print(f"ling biomass {B_L:.4f} t/km2; field ration {B_L * QB[0]:.4f} ({B_L * QB[1]:.4f}-{B_L * QB[2]:.4f}) t/km2/yr at Q/B "
      f"{QB[0]} ({QB[1]}-{QB[2]})")
fA = B_L * QB[0] * BEN[0] * AFR[0]
fJ = B_L * QB[0] * BEN[0] * (1 - AFR[0])
fA_lo, fA_hi = B_L * QB[1] * BEN[1] * AFR[1], B_L * QB[2] * BEN[2] * AFR[2]
print(f"field ling flow on benthivores: J {fJ:.5f}, A {fA:.5f} ({fA_lo:.5f}-{fA_hi:.5f}) t/km2/yr\n")
print("WC pair, implied / booked M2 (prey side), equal cells / field spread:")
print(f"  {'run':24s} " + " ".join(f"{p[:13]:>15s}" for p in BOOK))
for run in RUNS:
    print(f"  {run:24s} " + " ".join(f"{ratio(run, p, 'eq'):6.2f} /{ratio(run, p, 'field'):5.2f}" for p in BOOK))
print("\nLing's implied flow per prey at the field spread (t/km2/yr) and against the field:")
print(f"  {'run':24s} {'ben J':>7s} {'ben A':>7s} {'whiting':>8s} {'plankt.':>8s} {'mack':>7s} {'total':>7s} |"
      f" {'tot/ration':>10s} {'A/ration':>8s} {'A/field A':>9s} {'ben share':>9s}")
OUT = {}
for run in RUNS:
    lv, lw = VAR[run]
    flows = {}
    for prey, (m2, b, js) in BOOK.items():
        pair = PAIRK[prey]
        tot = ratio(run, prey, "field") * m2 * b
        mem = {}
        for k, v in DM["members"][pair].items():
            sp = k.split("|")[0]
            mem[sp] = mem.get(sp, 0.0) + v["share"]
        g = {"hake": 0.0} if prey == "benthivorous_fish" else {}
        if lv == "OUT":
            g["ling"] = 0.0
        elif lv:
            g["ling"] = DM["ling_variants"][lv][pair]["g"]
        if prey == "benthivorous_fish":
            wJ, wA = {}, {}
            for sp, r in DM["flows"].items():
                fj, fa, pj, pa = r["fJ"], r["fA"], r["pJ"], r["pA"]
                ap = APROF.get(sp, 1.0)
                if sp == "ling":
                    if lw:
                        w = DM["windows"][lw]
                        fj, fa, pj, pa = w["fJ"], w["fA"], w["pJ"], w["pA"]
                    if lv and lv != "OUT":
                        ap = DM["ling_variants"][lv][pair]["Aprof"]
                s = r["a_share"] * g.get(sp, 1.0)
                wJ[sp] = s * fj * (pj / P0 if pj == pj else 1.0)
                wA[sp] = s * fa * (pa / P0 if pa == pa else 1.0) * ap
            eJ, eA = sum(wJ.values()) * 0.655 * js, sum(wA.values()) * 0.882 * (1 - js)
            TJ, TA = tot * eJ / (eJ + eA), tot * eA / (eJ + eA)
            flows["ben J"] = TJ * wJ["ling"] / sum(wJ.values())
            flows["ben A"] = TA * wA["ling"] / sum(wA.values())
            flows["_TA"] = TA
        else:
            wt = {}
            for sp, s in mem.items():
                if js is None:
                    f = 1.0
                else:
                    key = f"{'saithe' if sp == 'pollack' else ('anglerfish' if sp == 'ling' else sp)}|{GRP[prey]}"
                    fj, fa = DM["member_windows"][key]
                    if sp == "ling" and lw:
                        fj, fa = DM["ling_windows"][f"{lw}|{GRP[prey]}"]
                    f = js * fj + (1 - js) * fa
                wt[sp] = s * g.get(sp, 1.0) * f
            flows[prey] = tot * wt["ling"] / sum(wt.values())
    t = sum(v for k, v in flows.items() if not k.startswith("_"))
    q = B_L * QB[0]
    OUT[run] = flows
    print(f"  {run:24s} {flows['ben J']:7.4f} {flows['ben A']:7.4f} {flows['whiting']:8.4f} {flows['planktivorous_fish']:8.4f}"
          f" {flows['mackerel']:7.5f} {t:7.4f} | {t / q:10.2f} {flows['ben A'] / q:8.2f} {flows['ben A'] / fA if fA else 0:9.1f}"
          f" {(flows['ben J'] + flows['ben A']) / t if t else 0:9.2f}")
print(f"\n  ration ranges: total / ration at Q/B {QB[1]}-{QB[2]} = x{QB[0] / QB[1]:.2f}-x{QB[0] / QB[2]:.2f} of the column;"
      f" A / field A range: column x {fA / fA_hi:.2f}-{fA / fA_lo:.1f}")
print("\nPair A flow (t/km2/yr, field spread) and the field A of all members (field_vs_model.py: 0.0177, 0.0022-0.0669):")
for run in RUNS:
    print(f"  {run:24s} A {OUT[run]['_TA']:.4f} (x{OUT[run]['_TA'] / 0.0177:.1f} the field A); ling's part "
          f"{OUT[run]['ben A'] / OUT[run]['_TA'] if OUT[run]['_TA'] else 0:.2f}")
