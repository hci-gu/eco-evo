"""Print title, biblio and reconstructed abstract from OpenAlex JSON files. Usage: python3 -I q2_openalex_abs.py <files...>"""
import json, sys
for f in sys.argv[1:]:
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f, "unreadable", e); continue
    if "message" in d:  # crossref
        for i in d["message"]["items"]:
            print("CROSSREF", i["DOI"], "|", (i.get("title") or [""])[0], "|", (i.get("container-title") or [""])[0], i["issued"]["date-parts"][0])
        continue
    inv = d.get("abstract_inverted_index") or {}
    w = sorted((p, k) for k, v in inv.items() for p in v)
    print("==", f, "|", d.get("doi"), "|", d.get("title"), "|", d.get("publication_year"), d.get("biblio"))
    print(" ".join(k for _, k in w) or "(no abstract)")
