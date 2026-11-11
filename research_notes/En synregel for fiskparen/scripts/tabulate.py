"""Compact tables from gate_variants.json (and one_rule_pairs.json): encounter / booked per pair, OFF flags
(the budget_gate band: ratio outside 1/3-3 on a pair with either share >= 0.05; prey side: a pair with
>= 0.10 of its prey's booked M2), and the per-class water-column krill share.

Usage: python3 -I tabulate.py <repo_root> <tables_dir>
"""
import json
import os
import sys

ROOT = sys.argv[1]
TAB = sys.argv[2]
G = json.load(open(os.path.join(TAB, "gate_variants.json")))
BAND, MINS, PMIN = 3.0, 0.05, 0.10
PRED = ["predatory_fish", "whiting", "benthivorous_fish", "mackerel", "planktivorous_fish"]
SHORT = {"predatory_fish": "WC", "whiting": "whi", "benthivorous_fish": "ben", "mackerel": "mac",
         "planktivorous_fish": "plk", "krill": "krill", "zooplankton": "zoo", "mobile_epifauna": "epi",
         "benthic_infauna": "inf", "norway_lobster": "neph", "porpoises": "por", "seals": "seal"}
LABELS = list(G)


def off(e, k):
    r = e / k if k > 0 else float("inf")
    return max(e, k) >= MINS and not (1 / BAND <= r <= BAND)


# booked shares (equal) from the library run: book_sum and rows hold enc shares only, so take booked from
# class gate's 'book' (same M2 x density)
def booked(label, pid):
    c = G[label]["class"].get(pid)
    return {p: v["book"] for p, v in c[0].items()} if c else None


print("1. Predator side: encounter share (ratio to booked) at equal cells | field spread; * = OFF")
for pid in PRED:
    bk = booked("lib", pid)
    prey = [p for p in bk if bk[p] > 0 or True]
    print(f"\n  {pid}: booked " + ", ".join(f"{SHORT.get(p, p)} {bk[p]:.3f}" for p in prey))
    for lab in LABELS:
        eq = G[lab]["equal"].get(pid)
        fl = G[lab]["field"].get(pid)
        if eq is None:
            continue
        cells = []
        for p in prey:
            e = eq[1].get(p, 0.0)
            s = f"{SHORT.get(p, p)} {e:.3f}({e / bk[p]:.2f}){'*' if off(e, bk[p]) else ''}" if bk[p] > 0 else ""
            cells.append(s)
        line = f"    {lab:16s} eq: " + " ".join(c for c in cells if c)
        if fl:
            fc = []
            for p in prey:
                e = fl[1].get(p, 0.0)
                if bk[p] > 0:
                    fc.append(f"{SHORT.get(p, p)} {e:.3f}({e / bk[p]:.2f}){'*' if off(e, bk[p]) else ''}")
            line += " | field: " + " ".join(fc)
        print(line)

# prey side
print("\n2. Prey side: implied M2 / booked M2 per pair (equal cells; field spread in brackets where the predator has"
      " one); * = OFF (pair carries >= 0.10 of the prey's booked M2). Mammal and seabird pairs (out of scope,"
      " unchanged) enter the total at their booked M2, not at budget_gate's implied value.")
FISHPREY = ["krill", "planktivorous_fish", "benthivorous_fish", "whiting", "mackerel"]
import yaml  # noqa: E402
lib = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml")))["interaction_definitions"]
M2 = {}
for k, v in lib.items():
    if "_preys_on_" not in k or not isinstance(v, dict) or not v.get("preys_on"):
        continue
    pred, prey = k.split("_preys_on_")
    if prey in FISHPREY and v.get("predation_mortality"):
        M2.setdefault(prey, {})[pred] = float(v["predation_mortality"])
for prey in FISHPREY:
    tot = sum(M2[prey].values())
    print(f"\n  {prey}: booked M2 total {tot:.4g}; pairs " + ", ".join(f"{SHORT.get(p, p)} {m:.4g} ({m / tot:.2f})"
                                                                     for p, m in sorted(M2[prey].items(), key=lambda kv: -kv[1])))
    for lab in LABELS:
        parts = []
        tot_imp = 0.0
        for pred, m in sorted(M2[prey].items(), key=lambda kv: -kv[1]):
            eq = G[lab]["equal"].get(pred)
            if eq is None or pred not in PRED:   # mammals, seabirds: out of scope, counted at booked M2
                tot_imp += m
                continue
            bk = booked("lib", pred)
            r = eq[1].get(prey, 0.0) / bk[prey]
            tot_imp += m * r
            s = f"{SHORT.get(pred, pred)} {r:.2f}{'*' if (m / tot >= PMIN and not (1 / BAND <= r <= BAND)) else ''}"
            fl = G[lab]["field"].get(pred)
            if fl:
                rf = fl[1].get(prey, 0.0) / bk[prey]
                s += f"({rf:.2f}{'*' if (m / tot >= PMIN and not (1 / BAND <= rf <= BAND)) else ''})"
            parts.append(s)
        print(f"    {lab:16s} total {tot_imp / tot:.2f}: " + ", ".join(parts))

print("\n3. Per bed class (16 x 16 probe grid, predator uniform within the class): krill share of each predator's"
      " encounter; class < 100 / 100-150 / 150-250 / > 250 m. 'flat' = fish a's as the engine has them (no table);"
      " 'cls' = fish a's class-resolved by the rule")
for pid in PRED:
    bk = booked("lib", pid)
    print(f"\n  {pid} (booked krill share {bk.get('krill', 0):.3f})")
    for lab in LABELS:
        fl = G[lab]["class_flat"][pid]
        cl = G[lab]["class"][pid]
        print(f"    {lab:16s} flat " + " / ".join(f"{c['krill']['enc']:.3f}" for c in fl) + "   cls "
              + " / ".join(f"{c['krill']['enc']:.3f}" for c in cl))
print("\n4. Water column at 150-250 m (class 2), full diet: share (ratio to booked); flat reading | cls reading")
bk = booked("lib", "predatory_fish")
for lab in LABELS:
    for flav in ("class_flat", "class"):
        c = G[lab][flav]["predatory_fish"][2]
        print(f"    {lab:16s} {'flat' if flav == 'class_flat' else 'cls ':4s} "
              + ", ".join(f"{SHORT.get(p, p)} {v['enc']:.3f}({v['enc'] / bk[p]:.2f})" for p, v in c.items()))
