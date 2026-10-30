"""Fetch ICES SAG summary tables and reference points for the member stocks (WP3, box_share.md).

Usage: python3 -I share_fetch_sag.py <out_dir>
Writes summary_<key>.json and refpts_<key>.json (untrusted data; parsed only by share_sag.py).
"""
import os
import sys
import time
import urllib.request

KEYS = {
    22468: "her.27.3a47d", 21294: "her.27.20-24", 21311: "spr.27.3a4", 26011: "nop.27.3a4",
    25908: "whb.27.1-91214", 22417: "had.27.46a20", 26010: "cod.27.46a7d20", 26001: "cod viking",
    26002: "cod southern", 26003: "cod northwestern", 22381: "cod.27.21", 22489: "ple.27.420",
    22470: "ple.27.21-32", 22534: "wit.27.3a47d", 19480: "dab.27.3a4", 22504: "pok.27.3a46",
    25875: "mac.27.nea", 25892: "hom.27.2a3a4a5b6a7a-ce-k8", 25921: "hom.27.4bc7d",
    22627: "hke.27.3a46-8abd", 25850: "dgs.27.nea", 22563: "whg.27.3a", 22488: "whg.27.47d",
}
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
base = "https://sag.ices.dk/SAG_API/api/"
for k in KEYS:
    for kind, ep in (("summary", "SummaryTable"), ("refpts", "FishStockReferencePoints")):
        fn = os.path.join(out, f"{kind}_{k}.json")
        if os.path.exists(fn):
            continue
        with urllib.request.urlopen(f"{base}{ep}?assessmentKey={k}", timeout=60) as r:
            open(fn, "wb").write(r.read())
        time.sleep(0.3)
    print(k, KEYS[k])
