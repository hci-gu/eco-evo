"""Split the predatory_fish (and planktivore members') benthos diet into infauna and mobile epifauna.

Usage: python3 -I pred_benthos_split.py <ices_diet_pools_taxa.csv>

Input: research_notes/"Fiskgrupper i rutan"/predators/tables/ices_diet_pools_taxa.csv (ICES stomach DB, g of
prey per taxon, pool B = benthos as in the parent note). Every B taxon is classified by name into the
two new groups; the brief's rule: polychaetes, echinoderms, molluscs, amphipods, cumaceans, isopods and
other small burrowing crustaceans (Calocaris) are infauna; Nephrops, Pandalus, crabs, hermit crabs,
Munida/Galathea, Crangon and other carideans, mysids and unidentified decapods are mobile epifauna.
Prints g and shares per species/area/period and the pooled whiting (box + Skagerrak) split.
"""
import csv
import sys
from collections import defaultdict

EPI_KEYS = {
    "pandalid": ["Pandalus", "Pandalidae", "Dichelopandalus", "Pandalina"],
    "nephrops": ["Nephrops"],
    "crab_hermit": ["Brachyura", "Pagurus", "Paguridae", "Anapagurus", "Polybius", "Liocarcinus", "Portunidae",
                    "Polybiidae", "Reptantia", "Munida", "Galathea", "Hyas", "Cancer", "Atelecyclus", "Goneplax"],
    "small_shrimp": ["Crangon", "Crangonidae", "Caridea", "Processa", "Processidae", "Palaemon", "Leontocaris",
                     "Philocheras", "Eualus", "Spirontocaris", "Pasiphaea", "Decapoda"],
    "mysid": ["Mysida", "Mysidae", "Mysis", "Erythrops", "Gastrosaccus", "Praunus", "Schistomysis"],
}
INF_KEYS = ["Polychaeta", "Nereid", "Annelida", "Aphrodit", "Arenicola", "Lagis", "Bivalvia", "Ensis",
            "Mollusca", "Gastropoda", "Prosobranchia", "Amphipoda", "Gammaridea", "Cumacea", "Cumopsis",
            "Isopoda", "Peracarida", "Echinidea", "Echinodermata", "Ophiur", "Calocaris"]


def classify(t):
    for k, keys in EPI_KEYS.items():
        if any(t.startswith(x) or x in t for x in keys):
            return "epi", k
    if any(x in t for x in INF_KEYS):
        return "inf", "infauna"
    return "inf", "infauna(other:" + t + ")"


g = defaultdict(float)
unk = defaultdict(float)
for r in csv.DictReader(open(sys.argv[1])):
    if r["pool"] != "B":
        continue
    part, sub = classify(r["taxon"])
    g[(r["species"], r["area"], r["period"], part, sub)] += float(r["g"])
    if "other:" in sub:
        unk[r["taxon"]] += float(r["g"])
keys = sorted({k[:3] for k in g})
for k in keys:
    tot = sum(v for kk, v in g.items() if kk[:3] == k)
    epi = sum(v for kk, v in g.items() if kk[:3] == k and kk[3] == "epi")
    subs = defaultdict(float)
    for kk, v in g.items():
        if kk[:3] == k:
            subs[kk[4].split("(")[0]] += v
    print(f"{' '.join(k):<34} B {tot:7.1f} g  epifauna share {epi / tot:5.2f}   " +
          ", ".join(f"{s} {v / tot:.2f}" for s, v in sorted(subs.items(), key=lambda x: -x[1])))
print("unclassified taxa counted as infauna:", dict(unk))


def pooled(sp, areas, periods):
    subs = defaultdict(float)
    for kk, v in g.items():
        if kk[0] == sp and kk[1] in areas and kk[2] in periods:
            subs[kk[4].split("(")[0]] += v
    tot = sum(subs.values())
    epi = sum(v for s, v in subs.items() if s != "infauna")
    return tot, epi / tot, {s: v / tot for s, v in subs.items()}


for lab, sp, ar, per in [("whiting box+Skagerrak, all periods", "whiting", ("Box", "Skagerrak"), ("pre2000", "2022")),
                         ("whiting box+Skagerrak pre-2000", "whiting", ("Box", "Skagerrak"), ("pre2000",)),
                         ("whiting box+Skagerrak 2022", "whiting", ("Box", "Skagerrak"), ("2022",)),
                         ("whiting all areas 2022-24", "whiting", ("Box", "Skagerrak", "CSNS", "NNS", "Kattegat", "other"),
                          ("2022", "2023", "2024")),
                         ("hake box+Skagerrak 2024", "hake", ("Box", "Skagerrak"), ("2024",))]:
    tot, e, subs = pooled(sp, ar, per)
    print(f"POOLED {lab:<40} B {tot:6.1f} g  epifauna {e:.2f}  " +
          ", ".join(f"{s} {v:.2f}" for s, v in sorted(subs.items(), key=lambda x: -x[1])))
