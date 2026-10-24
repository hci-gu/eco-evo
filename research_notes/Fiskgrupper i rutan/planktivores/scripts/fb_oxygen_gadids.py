"""List FishBase oxygen-consumption rows for Gadiformes-relevant genera and sprat.
Usage: fb_oxygen_gadids.py SPECIES_PARQUET OXYGEN_PARQUET"""
import sys
import pyarrow.parquet as pq
sp = pq.read_table(sys.argv[1], columns=["SpecCode", "Genus", "Species"]).to_pylist()
names = {r["SpecCode"]: f'{r["Genus"]} {r["Species"]}' for r in sp}
want = {"Trisopterus", "Micromesistius", "Sprattus", "Merlangius", "Melanogrammus", "Pollachius", "Gadiculus", "Boreogadus", "Gadus"}
codes = {c for c, n in names.items() if n.split()[0] in want}
print({c: names[c] for c in sorted(codes) if names[c].split()[0] in ("Trisopterus", "Micromesistius", "Sprattus")})
ox = pq.read_table(sys.argv[2]).to_pylist()
for r in ox:
    if r["SpecCode"] in codes:
        print(names[r["SpecCode"]], "| W", r["Weight"], "g | T", r["Temp"], "| O2cons", r["OxygenCons"], "| level", r["MetabolicLevel"], "| ref", r["OxygenRefNo"], "|", (r["Comment"] or "")[:80])
