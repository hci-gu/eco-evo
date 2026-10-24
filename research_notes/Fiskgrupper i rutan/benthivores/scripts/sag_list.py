"""List SAG assessments matching stock-label prefixes. Usage: python3 -I sag_list.py file.json [file2.json ...]"""
import json, sys
pref = ("ple.27", "wit.", "lem.", "dab.", "lrd", "gug.", "rjr.", "rja", "had.27.46a20", "cod.27.46a7d20", "cod.27.21", "wee", "hag", "fle.", "gur", "rjc.", "sol.27.4", "sol.27.20")
for fn in sys.argv[1:]:
    for r in json.load(open(fn)):
        lab = r.get("StockKeyLabel") or ""
        if lab.startswith(pref):
            print(r["AssessmentYear"], r["AssessmentKey"], lab, r.get("Purpose"), r.get("Status"), r.get("ModelType"), r.get("ModelName"), sep=" | ")
