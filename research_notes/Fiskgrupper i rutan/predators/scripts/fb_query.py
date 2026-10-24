"""Print FishBase parquet rows (oxygen, ration, popqb, speed, swimming) for target species.
Usage: <venv-python> -I fb_query.py <fishbase_dir>
"""
import sys, os
import pyarrow.parquet as pq
d = sys.argv[1]
targets = {("Pollachius", "virens"), ("Merlangius", "merlangus"), ("Scomber", "scombrus"),
           ("Squalus", "acanthias"), ("Trachurus", "trachurus"), ("Merluccius", "merluccius"),
           ("Lophius", "piscatorius"), ("Molva", "molva"), ("Pollachius", "pollachius")}
sp = pq.read_table(os.path.join(d, "species.parquet"), columns=["SpecCode", "Genus", "Species"]).to_pylist()
codes = {r["SpecCode"]: r["Genus"] + " " + r["Species"] for r in sp if (r["Genus"], r["Species"]) in targets}
print("codes", codes)
for t in ["oxygen", "ration", "popqb", "speed", "swimming"]:
    tab = pq.read_table(os.path.join(d, t + ".parquet"))
    cols = tab.column_names
    rows = [r for r in tab.to_pylist() if r.get("SpecCode") in codes or r.get("Speccode") in codes]
    print("\n#####", t, len(rows), "rows; cols:", cols)
    for r in rows:
        sc = r.get("SpecCode") or r.get("Speccode")
        keep = {k: v for k, v in r.items() if v not in (None, "", 0) and k not in ("Entered", "DateEntered", "Modified", "DateModified", "Expert", "DateChecked", "TS")}
        print(codes[sc], "|", keep)
