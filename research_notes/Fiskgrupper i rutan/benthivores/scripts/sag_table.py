"""Print a SAG SummaryTable compactly. Usage: python3 -I sag_table.py summary_<key>.json"""
import json, sys
d = json.load(open(sys.argv[1]))
print(d["FishStock"], d["AssessmentYear"], "FAge", d.get("FAge"), "RecAge", d.get("RecruitmentAge"),
      "|", d.get("StockSizeDescription"), d.get("StockSizeUnits"), "|", d.get("FishingPressureDescription"),
      "| custom:", [d.get(f"CustomName{i}") for i in range(1, 6) if d.get(f"CustomName{i}")])
keys = ["Year", "Recruitment", "SSB", "TBiomass", "Catches", "Landings", "Discards", "IBC", "Unallocated_Removals",
        "F", "FishingPressure", "StockSize", "CustomSeries1", "CustomSeries2", "CustomSeries3"]
lines = d["Lines"]
present = [k for k in keys if any(l.get(k) is not None for l in lines)]
print("\t".join(present))
for l in lines:
    print("\t".join("" if l.get(k) is None else (f"{l[k]:.4g}" if isinstance(l[k], float) else str(l[k])) for k in present))
