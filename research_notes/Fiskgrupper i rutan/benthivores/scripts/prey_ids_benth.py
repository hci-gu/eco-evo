"""List prey AphiaIDs of the benthivorous_fish members that lack a cached WoRMS lineage.
Usage: python3 -I prey_ids_benth.py <ices_stomach_dir> <worms_dir> [<worms_dir2> ...]
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benth_common import PRED
root = sys.argv[1]; wdirs = sys.argv[2:]
ids = set()
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d, "PreyInformation.csv")
    if not os.path.exists(p):
        continue
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r["AphiaIDPredator"].strip('"') in PRED and r["AphiaIDPrey"].strip('"'):
                ids.add(r["AphiaIDPrey"].strip('"'))
have = set()
for w in wdirs:
    if os.path.isdir(w):
        have |= {fn[4:-5] for fn in os.listdir(w) if fn.startswith("cls_")}
miss = sorted(ids - have, key=int)
sys.stderr.write(f"{len(ids)} prey ids, {len(miss)} missing\n")
print("\n".join(miss))
