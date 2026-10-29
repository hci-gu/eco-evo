"""EcoBase model 457 (North Sea 1991 Ecopath, Mackinson & Daskalov 2007): every diet item of
spurdog (6), juvenile saithe (19), adult saithe (20), monkfish (26), with the items that map to
the model's benthivores (cod, haddock, plaice, dab, long rough dab, witch, starry ray etc.) flagged.
Parses the XML as text (regex), executes nothing.
Usage: python3 -I ecopath1991_benthivore_prey.py <model_457.xml>
"""
import re, sys
s = open(sys.argv[1], encoding="utf-8").read()
G = {}
for g in s.split("<group>")[1:]:
    seq = int(re.search(r"<group_seq>(\d+)</group_seq>", g).group(1))
    name = re.search(r"<group_name>([^<]*)</group_name>", g).group(1).replace("&amp;amp;", "&")
    diet = {int(a): float(p) for a, p in re.findall(r"<prey_seq>(\d+)</prey_seq><proportion>([-\d.eE]+)</proportion>", g)}
    G[seq] = dict(name=name, diet=diet)
print("All groups:", "; ".join(f"{k} {v['name']}" for k, v in sorted(G.items()) if k < 50))
KEY = ("cod", "haddock", "plaice", "dab", "flounder", "flatfish", "sole", "ray", "skate", "gurnard", "other gadoids")
for j in (6, 19, 20, 26):
    print(f"\n== {j} {G[j]['name']}")
    tot = 0.0
    for i, p in sorted(G[j]["diet"].items(), key=lambda x: -x[1]):
        if p <= 0: continue
        flag = any(k in G[i]["name"].lower() for k in KEY)
        if flag: tot += p
        print(f"   {p:.4f}  {G[i]['name']}{'   <-- benthivore-like' if flag else ''}")
    print(f"   sum flagged = {tot:.4f}")
