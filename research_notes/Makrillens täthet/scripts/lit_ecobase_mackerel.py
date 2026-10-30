"""Extract mackerel / horse mackerel (and herring, sprat for context) biomass from EcoBase model XMLs.
Parses the XML as text (regex), executes nothing from the files.

Usage: python3 -I lit_ecobase_mackerel.py <model_*.xml> [...]
"""
import re, sys
PAT = re.compile(r"(?i)mackerel|scomber|trachurus|herring|sprat|pelagic|scad")
for path in sys.argv[1:]:
    s = open(path, encoding="utf-8").read()
    if "<group>" not in s:
        print(f"{path}: no group data"); continue
    hdr = {t: (re.search(rf"<{t}>([^<]*)</{t}>", s) or [None, ""])[1]
           for t in ("model_name", "model_year", "area", "author")}
    print(f"== {path}: {hdr}")
    for g in s.split("<group>")[1:]:
        name = re.search(r"<group_name>([^<]*)</group_name>", g).group(1)
        if not PAT.search(name):
            continue
        f = {t: (re.search(rf"<{t}>([-\d.eE]+)</{t}>", g) or [None, "nan"])[1]
             for t in ("biomass", "pb", "qb", "ee", "habitat_area")}
        print(f"  {name}: B {f['biomass']} t/km2 | P/B {f['pb']} | Q/B {f['qb']} | EE {f['ee']} | habitat_area {f['habitat_area']}")
