"""EcoBase model 457 (North Sea 1991, Mackinson & Daskalov 2007): the benthivore-pool items (haddock, cod,
flatfish by Ecopath stage) in the diets of the water-column members that carry the adult part of the
WC -> benthivores pair (spurdog, hake, monkfish = anglerfish; saithe for contrast). Parses the XML as text.
No ling group exists in the model.

Usage: python3 -I ecopath_bf_items.py <model_457.xml>
"""
import re
import sys

s = open(sys.argv[1], encoding="utf-8").read()
G = {}
for g in s.split("<group>")[1:]:
    seq = int(re.search(r"<group_seq>(\d+)</group_seq>", g).group(1))
    name = re.search(r"<group_name>([^<]*)</group_name>", g).group(1).replace("&amp;amp;", "&")
    qb = float(re.search(r"<qb>([-\d.eE]+)</qb>", g).group(1))
    b = float(re.search(r"<biomass>([-\d.eE]+)</biomass>", g).group(1))
    diet = {int(a): float(p) for a, p in re.findall(r"<prey_seq>(\d+)</prey_seq><proportion>([-\d.eE]+)</proportion>", g)}
    G[seq] = dict(name=name, qb=qb, b=b, diet=diet)
print("Groups 1-62:")
for i in sorted(G):
    if i <= 62:
        print(f"  {i:3d} {G[i]['name']}  B {G[i]['b']:.3f}  QB {G[i]['qb']:.2f}")
MODEL_BEN = [i for i in G if re.search(r"cod|haddock|plaice|sole|dab|flounder|flatfish|witch|lemon", G[i]["name"], re.I)
             and not re.search(r"saithe", G[i]["name"], re.I)]
print("\nModel benthivore members (Ecopath groups):", [(i, G[i]["name"]) for i in sorted(MODEL_BEN)])
for j in sorted(G):
    if not re.search(r"spurdog|hake|monkfish|saithe|starry|ray|gurnard", G[j]["name"], re.I):
        continue
    g = G[j]
    items = [(G[i]["name"], p) for i, p in sorted(g["diet"].items()) if i in MODEL_BEN and p > 0]
    tot = sum(p for _, p in items)
    print(f"\n{j} {g['name']}: B {g['b']:.3f} QB {g['qb']:.2f}; model-benthivore items total {tot:.3f}")
    for n, p in items:
        print(f"    {n:40s} {p:.4f}")
