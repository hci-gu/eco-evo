#!/usr/bin/env python3
"""Download SSB / recruitment summary tables and reference points from the ICES
Stock Assessment Graphs (SAG) API for the member stocks of the Mareld fish groups.

Usage:
    python3 -I scripts/sr_fetch.py OUT_DIR
    e.g. python3 -I research_notes/Beståndsreglering/scripts/sr_fetch.py \
             research_notes/Beståndsreglering/data/sag

Writes OUT_DIR/summary_<key>.json (SAG SummaryTable), OUT_DIR/refpts_<key>.json
(SAG FishStockReferencePoints) and OUT_DIR/stockdownload_<key>.json (SAG
StockDownload, for the recruitment units) for each assessment key below. The keys are the
latest published "Advice" assessments in the SAG StockList of 2024-2026 (looked up
2026-10-29 with https://sag.ices.dk/SAG_API/api/StockList?year=YYYY).
No credentials, no personal data.
"""
import json
import pathlib
import sys
import time
import urllib.request

API = "https://sag.ices.dk/SAG_API/api"

# label, assessment key, note
KEYS = [
    ("her.27.3a47d", 22468, "NSAS herring, SAG 2026"),
    ("her.27.20-24", 21294, "WBSS herring, SAG 2026"),
    ("spr.27.3a4", 21311, "North Sea sprat, SMS, SAG 2026"),
    ("nop.27.3a4", 26011, "Norway pout, seasonal model, SAG 2026"),
    ("whb.27.1-91214", 25908, "blue whiting, SAG 2026"),
    ("had.27.46a20", 22417, "haddock, SAG 2026"),
    ("cod.27.46a7d20", 26010, "North Sea cod, whole stock, SAG 2026"),
    ("cod.27.46a7d20_viking", 26001, "North Sea cod Viking substock, SAG 2026"),
    ("cod.27.46a7d20_southern", 26002, "North Sea cod Southern substock, SAG 2026"),
    ("cod.27.46a7d20_northwestern", 26003, "North Sea cod NW substock, SAG 2026"),
    ("cod.27.47d20", 17652, "North Sea cod, old single-stock SAM, SAG 2022 (long series)"),
    ("cod.27.21", 22381, "Kattegat cod, SAG 2026"),
    ("ple.27.420", 22489, "North Sea plaice, SAG 2026"),
    ("ple.27.21-32", 22470, "Baltic + Kattegat plaice, SAG 2026"),
    ("ple.27.21-23", 18631, "Kattegat/Belts plaice, SAG 2024 (pre-merge)"),
    ("wit.27.3a47d", 22534, "witch, SAG 2026"),
    ("pok.27.3a46", 22504, "saithe, SAG 2026"),
    ("mac.27.nea", 25875, "NEA mackerel, SAG 2026"),
    ("hom.27.4bc7d", 25921, "North Sea horse mackerel, SAG 2026"),
    ("hom.27.2a3a4a5b6a7a-ce-k8", 25892, "western horse mackerel, SAG 2026"),
    ("hke.27.3a46-8abd", 22627, "northern hake, SS3, SAG 2026"),
    ("dgs.27.nea", 25850, "spurdog, SAG 2026"),
    ("whg.27.47d", 22488, "North Sea whiting, SAG 2026"),
    ("anf.27.3a46", 23766, "anglerfish 3a46, SS3, SAG 2026"),
]


def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception as exc:  # network hiccup: retry
            print("retry", url, exc, file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("failed: " + url)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for label, key, note in KEYS:
        for kind, ep in (("summary", "SummaryTable"), ("refpts", "FishStockReferencePoints"),
                         ("stockdownload", "StockDownload")):
            path = out / f"{kind}_{key}.json"
            if path.exists():
                continue
            raw = get(f"{API}/{ep}?assessmentKey={key}")
            json.loads(raw)  # must parse
            path.write_bytes(raw)
            print("saved", path, label, note)
    (out / "keys.json").write_text(json.dumps(KEYS, indent=1))


if __name__ == "__main__":
    main()
