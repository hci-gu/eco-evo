"""Fetch DATRAS NS-IBTS HH and HL (Q1 1980-1987, 2026) and keep only the fields and taxa needed here.
The raw XML is parsed with iterparse (no entity expansion of note) and discarded; filtered CSVs go to <out>.
ICES data policy (DATRAS). Usage: python3 -I fetch_datras.py <out_dir> [years to keep unfiltered ...]
"""
import csv, os, sys, time, urllib.request, xml.etree.ElementTree as ET
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
TAXA = {"126444": "Norway pout", "126417": "herring", "126425": "sprat", "126439": "blue whiting",
        "126441": "saithe", "126438": "whiting", "126437": "haddock", "126436": "cod", "126751": "sandeel",
        "125909": "sandeels (Ammodytidae)", "126484": "hake"}
HHF = ["Year", "Quarter", "Country", "Ship", "Gear", "StNo", "HaulNo", "Month", "Day", "TimeShot", "HaulDur",
       "DayNight", "ShootLat", "ShootLong", "StatRec", "Depth", "HaulVal", "DataType", "Distance", "WingSpread",
       "DoorSpread", "GroundSpeed", "BotTemp"]
HLF = ["Year", "Quarter", "Country", "Ship", "Gear", "StNo", "HaulNo", "SpecCodeType", "SpecCode", "ValidAphiaID",
       "SpecVal", "Sex", "TotalNo", "CatIdentifier", "NoMeas", "SubFactor", "SubWgt", "CatCatchWgt", "LngtCode",
       "LngtClass", "HLNoAtLngt"]
YEARS = [2026]   # 1980-87 Q1 were fetched too (the 1980s stomach years), not used, and deleted
KEEP_ALL = set(int(a) for a in sys.argv[2:])   # years whose HL uses NODC codes without ValidAphiaID: keep every row
for rt, fields in (("HH", HHF), ("HL", HLF)):
    for y in YEARS:
        p = os.path.join(out, f"{rt}_NS-IBTS_{y}_Q1.csv")
        if os.path.exists(p):
            continue
        url = f"https://datras.ices.dk/WebServices/DATRASWebService.asmx/get{rt}data?survey=NS-IBTS&year={y}&quarter=1"
        tmp = p + ".xml.part"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-client/1.0"})
            with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
        except Exception as e:
            print("fail", rt, y, e, flush=True)
            continue
        n = k = 0
        with open(p + ".tmp", "w", newline="") as fo:
            w = csv.writer(fo)
            w.writerow(fields)
            row = {}
            for ev, el in ET.iterparse(tmp, events=("end",)):
                tag = el.tag.split("}")[-1]
                if tag.startswith("Cls_"):
                    n += 1
                    sid = (row.get("ValidAphiaID") or row.get("SpecCode") or "").strip()
                    if rt == "HH" or sid in TAXA or y in KEEP_ALL:
                        w.writerow([(row.get(c) or "").strip() for c in fields])
                        k += 1
                    row = {}
                    el.clear()
                else:
                    row[tag] = el.text
        os.replace(p + ".tmp", p)
        os.remove(tmp)
        print(rt, y, "records", n, "kept", k, flush=True)
        time.sleep(0.5)
