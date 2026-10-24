"""North Sea 1991 Ecopath (Mackinson & Daskalov 2007, Table 3.4): diets of the predatory_fish
members mapped to the split model pools, plus Q/B (Table 3.3).

Usage: python3 -I ecopath_predator_diets.py <ecopath_ns1991 dir>
Reads diet.json (keys 'prey,pred') and params.json parsed by
../../../Litteraturbaserad growth rate per FG/ecopath_ns1991 (same column corrections as there).
"""
import json, os, sys
d = sys.argv[1]
P = {int(k): v for k, v in json.load(open(os.path.join(d, "params.json"))).items()}
P[15].update(B=0.222, PB=2.36, QB=6.58); P[17].update(B=0.284, PB=2.0, QB=5.39)
D = {tuple(map(int, k.split(","))): v for k, v in json.load(open(os.path.join(d, "diet.json"))).items()}
POOL = {}
for i in (22, 23, 28, 29, 30, 33, 48): POOL[i] = "P"
for i in (9, 10, 11, 12, 13, 14, 17, 18, 25, 27, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47): POOL[i] = "Bf"
for i in (5, 6, 7, 8, 15, 16, 19, 20, 21, 24, 26, 31, 32, 42, 43): POOL[i] = "W"
for i in (51, 52): POOL[i] = "Z"
POOL[50] = "larvae"
for i in range(54, 62): POOL[i] = "B"
for i in (49,): POOL[i] = "ceph"
for i in (53, 62, 63, 64, 65, 66, 67, 68): POOL[i] = "O"
PRED = {6: "spurdog", 15: "whiting juv", 16: "whiting adult", 19: "saithe juv", 20: "saithe adult",
        21: "hake", 26: "monkfish", 31: "mackerel", 32: "horse mackerel"}
cols = ["P", "Bf", "W", "Z", "larvae", "B", "ceph", "O"]
print("pred B QB " + " ".join(cols) + " sum | top items")
for j, nm in PRED.items():
    s = {c: 0.0 for c in cols}; items = []
    for (i, k), v in D.items():
        if k != j: continue
        s[POOL.get(i, "O")] += v
        items.append((v, i))
    items.sort(reverse=True)
    top = ", ".join(f"{P.get(i, {}).get('name', i)} {v:.3f}" for v, i in items[:6])
    print(f"{nm}: B {P[j]['B']} QB {P[j]['QB']} " + " ".join(f"{c} {s[c]:.3f}" for c in cols) + f" sum {sum(s.values()):.3f} | {top}")
