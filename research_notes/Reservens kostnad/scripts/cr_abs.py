"""Print Crossref title and abstract for each DOI."""
import json, re, sys, time, urllib.request, urllib.parse
for doi in sys.argv[1:]:
    req = urllib.request.Request("https://api.crossref.org/works/" + urllib.parse.quote(doi), headers={"User-Agent": "lit-review-script"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            m = json.load(r)["message"]
        ab = re.sub(r"<[^>]+>", "", m.get("abstract") or "(no abstract in Crossref)")
        print("###", doi, "|", (m.get("title") or [""])[0], "|", (m.get("container-title") or [""])[0], "\n", ab.strip(), "\n")
    except Exception as e:
        print("###", doi, "ERR", e)
    time.sleep(1)
