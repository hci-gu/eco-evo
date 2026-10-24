"""EcoBase model 457 (North Sea 1991 Ecopath, Mackinson & Daskalov 2007): diets of Norway pout and blue whiting by
model pool, and M2 on them by predator (Q/B x B_pred x diet share / B_prey), mapped to the new model groups.
Parses the XML as text only. Usage: python3 -I ecobase457_pout_whb.py model_457.xml"""
import re, sys
s = open(sys.argv[1], encoding="utf-8").read()
G = {}
for g in s.split("<group>")[1:]:
    seq = int(re.search(r"<group_seq>(\d+)</group_seq>", g).group(1))
    name = re.search(r"<group_name>([^<]*)</group_name>", g).group(1).replace("&amp;amp;", "&").replace("&amp;", "&")
    num = lambda tag: float((re.search(rf"<{tag}>([-\d.eE]+)</{tag}>", g) or [0, "nan"])[1])
    diet = {int(a): float(p) for a, p in re.findall(r"<prey_seq>(\d+)</prey_seq><proportion>([-\d.eE]+)</proportion>", g)}
    G[seq] = dict(name=name, qb=num("qb"), b=num("biomass"), pb=num("pb"), ee=num("ee"), diet=diet)
name2seq = {v["name"]: k for k, v in G.items()}
def find(sub): return [k for k, v in G.items() if sub.lower() in v["name"].lower()]
POOL = {}
for k, v in G.items():
    n = v["name"].lower()
    if any(x in n for x in ("zooplankton",)) and "gelatinous" not in n: POOL[k] = "zooplankton"
    elif any(x in n for x in ("crab", "nephrops", "macrobenthos", "shrimp", "infauna", "epifauna", "benthos")): POOL[k] = "benthic_community"
    elif any(x in n for x in ("herring", "sprat", "norway pout", "blue whiting", "sandeel", "filterfeeding")): POOL[k] = "fish:planktivore"
    elif "larvae" in n: POOL[k] = "fish_larvae"
    elif any(x in n for x in ("gadoid", "cod", "haddock", "whiting", "saithe", "demersal", "plaice", "sole", "dab", "flounder", "ray", "skate", "gurnard", "turbot", "lemon", "witch", "megrim", "halibut", "dragonet", "catfish", "monkfish", "hake", "mackerel", "shark", "spurdog", "dogfish")): POOL[k] = "fish:other"
    else: POOL[k] = "other:" + v["name"]
PRED = {}
for k, v in G.items():
    n = v["name"].lower()
    if any(x in n for x in ("cod", "haddock", "plaice", "sole", "dab", "flounder", "ray", "skate", "gurnard", "turbot", "lemon", "witch", "dragonet", "catfish", "small demersal", "large demersal")): PRED[k] = "benthivorous_fish"
    if any(x in n for x in ("whiting (adult)", "juvenile whiting", "saithe", "hake", "mackerel", "spurdog", "monkfish", "large gadoid", "other gadoids (large)", "megrim", "halibut", "shark")): PRED[k] = "predatory_fish"
    if "blue whiting" in n: PRED[k] = "planktivorous_fish (within group)"
    if any(x in n for x in ("herring", "sprat", "norway pout", "sandeel", "filterfeeding")): PRED[k] = "planktivorous_fish (within group)"
    if "seal" in n: PRED[k] = "seals"
    if "toothed" in n: PRED[k] = "porpoises (toothed whales)"
    if "bird" in n: PRED[k] = "seabirds"
for target in ("Norway pout", "Blue whiting"):
    j = find(target)[0]; g = G[j]
    print(f"== {g['name']} (seq {j}) B {g['b']} t/km2  P/B {g['pb']}  Q/B {g['qb']}  EE {g['ee']}")
    sh = {}
    for i, p in sorted(g["diet"].items(), key=lambda x: -x[1]):
        if p > 0:
            print(f"   prey {G[i]['name']:45s} {p:.3f}  -> {POOL[i]}")
            sh[POOL[i]] = sh.get(POOL[i], 0) + p
    print("   by pool:", {k: round(v, 3) for k, v in sorted(sh.items(), key=lambda x: -x[1])}, "sum", round(sum(sh.values()), 3))
    m2 = {}
    for k, v in G.items():
        p = v["diet"].get(j, 0)
        if p > 0 and v["qb"] == v["qb"]:
            m = p * v["qb"] * v["b"] / g["b"]
            m2.setdefault(PRED.get(k, "not_in_model"), []).append((v["name"], m))
    tot = sum(m for l in m2.values() for _, m in l)
    print(f"   M2 total {tot:.3f}; M0 = PB(1-EE) = {g['pb']*(1-g['ee']):.3f}")
    for grp, l in sorted(m2.items(), key=lambda x: -sum(m for _, m in x[1])):
        print(f"     {grp:34s} {sum(m for _, m in l):.3f}  " + ", ".join(f"{n} {m:.3f}" for n, m in sorted(l, key=lambda x: -x[1]) if m >= 0.004))
