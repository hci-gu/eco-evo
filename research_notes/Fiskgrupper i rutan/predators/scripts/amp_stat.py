"""Parse AmP *_stat.html (implied properties) pages: text only. Usage: python3 -I amp_stat.py <dir>"""
import sys, re, html, os
keys = ["z", "c_T", "s_M", "Ww_p", "Ww_i", "E_m", "t_starve", "J_Op", "J_Oi", "xi_WE", "del_V", "r_B"]
d = sys.argv[1]
for f in sorted(os.listdir(d)):
    if not f.endswith("_stat.html"):
        continue
    t = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
    t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))
    tt = re.search(r"typical temperature \(([-0-9.]+) deg", t)
    out = {}
    for k in keys:
        mm = re.search(r" %s ([-0-9.e+]+) " % re.escape(k), t)
        if mm:
            out[k] = float(mm.group(1))
    print(f[:-10], "Ttyp", tt.group(1) if tt else "?", out)
