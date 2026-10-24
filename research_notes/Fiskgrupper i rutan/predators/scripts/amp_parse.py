"""Parse AmP *_par.html pages (downloaded, untrusted; text only) and print key DEB parameters.
Usage: python3 -I amp_parse.py <dir>
"""
import sys, re, html, os
keys = ["T_A", "p_Am", "kap_X", "v", "kap", "p_M", "E_G", "del_M", "z", "F_m", "p_T", "k_J"]
d = sys.argv[1]
for f in sorted(os.listdir(d)):
    if not f.endswith("_par.html"):
        continue
    t = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    t = re.sub(r"\s+", " ", t)
    m = re.search(r"Model: (\w+)", t)
    out = {}
    for k in keys:
        mm = re.search(r" %s ([-0-9.e+]+) " % re.escape(k), t)
        if mm:
            out[k] = mm.group(1)
    dm = re.findall(r" (del_M\w*) ([0-9.e+-]+) ", t)
    print(f[:-9], m.group(1) if m else "?", out, "shape:", dm)
