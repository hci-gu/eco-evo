"""Print title and the de-inverted abstract of OpenAlex work records (JSON files given as arguments).

Usage: python3 -I sea_openalex_abstract.py <record.json> [...]
Reads the files as data only.
"""
import json
import sys

for path in sys.argv[1:]:
    try:
        rec = json.load(open(path, encoding="utf-8"))
    except Exception as e:  # not a work record
        print(path, "unreadable:", e)
        continue
    print("==", path)
    print("title:", rec.get("title"), "| year:", rec.get("publication_year"), "| doi:", rec.get("doi"))
    inv = rec.get("abstract_inverted_index")
    if not inv:
        print("abstract: none in record")
        continue
    pos = {}
    for word, idx in inv.items():
        for i in idx:
            pos[i] = word
    print("abstract:", " ".join(pos[i] for i in sorted(pos)))
