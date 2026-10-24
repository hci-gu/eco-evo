"""List prey AphiaIDs of the predatory_fish members and which lack a cached WoRMS lineage.
Usage: python3 -I prey_ids.py <ices_stomach_dir> <worms_dir> [<worms_dir2>]
Prints missing ids, one per line, to stdout.
"""
import csv, os, sys
root = sys.argv[1]; wdirs = sys.argv[2:]
PRED = {"126441", "126438", "127023", "126822", "105923", "126555", "126484"}
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
