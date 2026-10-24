"""Parse EcoBase model input XML files (B, P/B, Q/B, EE, diet) into one CSV per call.
Usage: python3 -I ecobase_groups.py <out_csv> <model_xml> [<model_xml> ...]
Rows: model, seq, group, B, PB, QB, EE, prey_seq, prey_name, proportion (one row per diet entry;
a group with no diet gets one row with empty prey fields).
"""
import re, sys, csv, os, html
out = sys.argv[1]
rows = []
for path in sys.argv[2:]:
    s = open(path, encoding="utf-8", errors="replace").read()
    model = os.path.basename(path).replace("model_", "").replace(".xml", "")
    G = {}
    for m in re.finditer(r"<group>(.*?)</group>", s, re.S):
        b = m.group(1)
        seq = int(re.search(r"<group_seq>([\d.]+)", b).group(1).split(".")[0])
        name = html.unescape(html.unescape(re.search(r"<group_name>(.*?)</group_name>", b, re.S).group(1))).strip()
        def f(t):
            x = re.search(r"<%s>([-\d.eE]+)</%s>" % (t, t), b)
            return float(x.group(1)) if x else None
        diet = re.findall(r"<prey_seq>([\d.]+)</prey_seq>\s*<proportion>([-\d.eE]+)</proportion>", b)
        G[seq] = (name, f("biomass"), f("pb"), f("qb"), f("ee"), [(int(p.split(".")[0]), float(q)) for p, q in diet])
    for seq, (name, B, PB, QB, EE, diet) in sorted(G.items()):
        d = [(p, q) for p, q in diet if q > 0]
        if not d:
            rows.append([model, seq, name, B, PB, QB, EE, "", "", ""])
        for p, q in d:
            rows.append([model, seq, name, B, PB, QB, EE, p, G.get(p, ("?",))[0], q])
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "seq", "group", "B", "PB", "QB", "EE", "prey_seq", "prey", "proportion"])
    w.writerows(rows)
print(len(rows))
