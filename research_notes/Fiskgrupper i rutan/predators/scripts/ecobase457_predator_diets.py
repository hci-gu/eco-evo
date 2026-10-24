"""EcoBase model 457 (North Sea 1991, Mackinson & Daskalov): diets of the predatory_fish members
mapped to the split model pools, and Q/B. Parses the XML as text (regex), executes nothing.

Usage: python3 -I ecobase457_predator_diets.py <model_457.xml>
"""
import re, sys
s = open(sys.argv[1], encoding="utf-8").read()
G = {}
for g in s.split("<group>")[1:]:
    seq = int(re.search(r"<group_seq>(\d+)</group_seq>", g).group(1))
    name = re.search(r"<group_name>([^<]*)</group_name>", g).group(1).replace("&amp;amp;", "&")
    qb = float(re.search(r"<qb>([-\d.eE]+)</qb>", g).group(1))
    b = float(re.search(r"<biomass>([-\d.eE]+)</biomass>", g).group(1))
    diet = {int(a): float(p) for a, p in re.findall(r"<prey_seq>(\d+)</prey_seq><proportion>([-\d.eE]+)</proportion>", g)}
    G[seq] = dict(name=name, qb=qb, b=b, diet=diet)
POOL = {}
for i in (22, 23, 28, 29, 30, 33, 48): POOL[i] = "P"
for i in (9, 10, 11, 12, 13, 14, 17, 18, 25, 27, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47): POOL[i] = "Bf"
for i in (5, 6, 7, 8, 15, 16, 19, 20, 21, 24, 26, 31, 32, 42, 43): POOL[i] = "W"
for i in (51, 52): POOL[i] = "Z"
POOL[50] = "larvae"
for i in range(54, 62): POOL[i] = "B"
POOL[49] = "ceph"
cols = ["P", "Bf", "W", "Z", "larvae", "B", "ceph", "O"]
for j in (6, 15, 16, 19, 20, 21, 26, 31, 32):
    g = G[j]; sh = {c: 0.0 for c in cols}
    for i, p in g["diet"].items():
        sh[POOL.get(i, "O")] += p
    top = sorted(((p, G[i]["name"]) for i, p in g["diet"].items() if p > 0), reverse=True)[:7]
    print(f"{j} {g['name']}: B {g['b']:.3f} QB {g['qb']:.2f} | " + " ".join(f"{c} {sh[c]:.3f}" for c in cols)
          + f" | sum {sum(sh.values()):.3f}")
    print("    top: " + "; ".join(f"{n} {p:.3f}" for p, n in top))
