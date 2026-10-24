"""Count predator records per species, year, area in the ICES stomach public download.
Usage: python3 -I ices_pred_counts.py <ices_stomach_dir>
"""
import csv, os, sys, collections
root = sys.argv[1]
PRED = {"126441": "saithe", "126438": "whiting", "127023": "mackerel", "126822": "horse_mackerel",
        "105923": "spurdog", "126555": "anglerfish", "126484": "hake"}
cnt = collections.Counter()
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d)
    if not os.path.isdir(p):
        continue
    hauls = {}
    with open(os.path.join(p, "HaulInformation.csv"), newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            hauls[r["tblHaulID"]] = r
    with open(os.path.join(p, "PredatorInformation.csv"), newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            sp = PRED.get(r["AphiaIDPredator"].strip('"'))
            if not sp:
                continue
            h = hauls.get(r["tblHaulID"], {})
            try:
                lat = float(h.get("ShootLat") or "nan"); lon = float(h.get("ShootLong") or "nan")
            except ValueError:
                lat = lon = float("nan")
            area = "IIIa" if (lat >= 57.0 and lon >= 7.0 and lon <= 13 and lat <= 59.5) else ("NNS" if lat >= 56 else "other")
            cnt[(sp, r["Year"], area)] += 1
for k in sorted(cnt):
    print(*k, cnt[k])
