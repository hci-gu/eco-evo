"""Count benthivorous_fish predator records in the public ICES stomach download, by year and area.
Usage: python3 -I benth_pred_counts.py <ices_stomach_dir>
Prints species, year, area, records, records with prey rows (weight > 0).
"""
import csv, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benth_common import PRED, area_of, fnum
root = sys.argv[1]
rec = collections.Counter(); fed = collections.Counter()
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d)
    if not os.path.isdir(p):
        continue
    hauls = {}
    with open(os.path.join(p, "HaulInformation.csv"), newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            hauls[r["tblHaulID"]] = r
    preds = {}
    with open(os.path.join(p, "PredatorInformation.csv"), newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            sp = PRED.get(r["AphiaIDPredator"].strip('"'))
            if not sp:
                continue
            h = hauls.get(r["tblHaulID"], {})
            a = area_of(fnum(h.get("ShootLat")), fnum(h.get("ShootLong")))
            k = (sp, r["Year"], a)
            rec[k] += 1
            preds[r["tblPredatorInformationID"]] = k
    seen = set()
    with open(os.path.join(p, "PreyInformation.csv"), newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            pid = r["tblPredatorInformationID"]
            if pid in preds and pid not in seen and fnum(r["Weight"]) > 0:
                seen.add(pid); fed[preds[pid]] += 1
for k in sorted(rec):
    print(",".join(k), rec[k], fed[k])
