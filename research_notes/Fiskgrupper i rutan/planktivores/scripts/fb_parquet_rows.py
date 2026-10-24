"""Print FishBase parquet rows for given SpecCodes. Usage: fb_parquet_rows.py FILE SPECCODE..."""
import sys
import pyarrow.parquet as pq
t = pq.read_table(sys.argv[1]).to_pylist()
codes = {int(c) for c in sys.argv[2:]}
print("columns:", list(t[0].keys()) if t else [])
for r in t:
    if r.get("SpecCode") in codes or r.get("Speccode") in codes:
        print({k: v for k, v in r.items() if v not in (None, "", 0)})
