"""Rebuild the abstract from an OpenAlex work JSON (abstract_inverted_index). Usage: python3 -I scripts/ov_oa_abstract.py <json> ..."""
import json, sys
for f in sys.argv[1:]:
    d = json.load(open(f))
    inv = d.get("abstract_inverted_index") or {}
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    print("==", d.get("display_name"), d.get("publication_year"), d.get("doi"))
    print(" ".join(w for _, w in pos)); print()
