"""Parse Unpaywall JSON responses (one per file, paths given as arguments)
and print title, year, OA status and OA locations. Network access is done
by curl on the command line, not here. Run with python3 -I."""
import json
import sys

for path in sys.argv[1:]:
    try:
        with open(path, encoding="utf8") as fh:
            j = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(path, "unreadable:", exc)
        continue
    print("==", j.get("doi"), "|", j.get("title"), "|", j.get("year"), "| OA:", j.get("is_oa"))
    for loc in j.get("oa_locations") or []:
        print("   ", loc.get("url_for_pdf"), "|", loc.get("url"))
