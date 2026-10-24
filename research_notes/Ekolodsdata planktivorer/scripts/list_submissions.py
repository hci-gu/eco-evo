"""List ICES acoustic database submissions (untrusted JSON) as TSV.
Usage: python3 -I list_submissions.py uploadlist.json > submissions.tsv"""
import json, sys
d = json.load(open(sys.argv[1]))
print("id\tsurvey\tcountry\tplatform\tstart\tend\tlocal_id")
for e in d:
    v = e["Vocabulary"]; c = e["Cruise"]
    s = ",".join(x["Value"] for x in v.get("Survey", []))
    co = ",".join(x["Value"] for x in v.get("Country", []))
    pl = ",".join(x["Value"] for x in v.get("Platform", []))
    print(f"{c.get('ID')}\t{s}\t{co}\t{pl}\t{c.get('StartDate')}\t{c.get('EndDate')}\t{c.get('LocalID')}")
