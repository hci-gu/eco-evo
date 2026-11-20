"""Inventory of weighed vs counted ICES stomach records per species, year, quarter and area.

Input: research_notes/"Krill over grunda bottnar"/stomachs/tables/records_linked.csv.gz (o-umlaut in the real
folder name). Usage: python3 -I ices_inventory.py <records_linked.csv.gz>
A record is 'weighed' when any prey group weight g_* > 0; 'krill-bearing' when krill_count > 0 or g_krill > 0.
"""
import csv, gzip, sys
from collections import defaultdict

G = ["g_U", "g_chaet", "g_cop", "g_crustu", "g_fish", "g_hyp", "g_krill", "g_mysid", "g_other", "g_ozoo", "g_pshrimp"]
SP = {"whiting", "saithe", "norway pout", "haddock", "cod", "hake"}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


inv = defaultdict(lambda: [0, 0, 0, 0, 0])  # records, stomachs, weighed recs, krill recs, weighed krill recs
with gzip.open(sys.argv[1], "rt") as fh:
    for r in csv.DictReader(fh):
        sp = r["sp"].strip().lower()
        if sp not in SP:
            continue
        N = f(r["Number"]) or 1.0
        w = sum((f(r[g]) or 0.0) for g in G) > 0
        k = (f(r["krill_count"]) or 0) > 0 or (f(r["g_krill"]) or 0) > 0
        key = (sp, r["year"], r["quarter"], r["area"])
        v = inv[key]
        v[0] += 1; v[1] += N; v[2] += w; v[3] += k; v[4] += (w and k)
print("species year Q area: records, stomachs, weighed records, krill-bearing records, weighed krill-bearing")
for key in sorted(inv):
    v = inv[key]
    if v[3] == 0 and key[0] not in ("whiting", "norway pout"):
        continue
    print(f"{key[0]:12s} {key[1]} Q{key[2]} {key[3]:14s} rec {v[0]:5d} stom {int(v[1]):6d} weighed {v[2]:5d} krill {v[3]:4d} weighed+krill {v[4]:4d}")
