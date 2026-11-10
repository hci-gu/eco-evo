"""Print title, venue, year, DOI and the reconstructed abstract of OpenAlex
work records (JSON files downloaded into data/openalex/).

Usage: python3 -I scripts/openalex_abstract.py data/openalex/*.json
Untrusted input: parsed with json only. (Copied from
../Bioenergetik for alla fiskgrupper/scripts/openalex_abstract.py.)
"""
import json
import sys


def abstract(inv):
    if not inv:
        return "(no abstract in record)"
    pos = []
    for word, idx in inv.items():
        for i in idx:
            pos.append((i, word))
    return " ".join(w for _, w in sorted(pos))


for path in sys.argv[1:]:
    try:
        rec = json.load(open(path, encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"== {path}: unreadable ({exc})")
        continue
    venue = ((rec.get("primary_location") or {}).get("source") or {}).get("display_name")
    oa = (rec.get("open_access") or {}).get("oa_url")
    print(f"== {path}")
    print(f"   {rec.get('title')} | {venue} | {rec.get('publication_year')} | {rec.get('doi')} | oa: {oa}")
    print("   " + abstract(rec.get("abstract_inverted_index")))
    print()
