"""Parse stock-assessment numbers-at-age, F, M and weights into one long table.

Usage: python3 -I stock_parse.py <data_stock_dir> <out_tables_dir>

Inputs (untrusted downloads, parsed only, never executed):
  <data_stock_dir>/sam_<run>/res/xxx-00-00.00.00_tab2.html (F at age), _tab3.html (N at age, Jan 1),
  <data_stock_dir>/sam_<run>/data/{sw,cw,nm}.dat (Lowestoft format: title, type, years, ages, fmt, rows),
  <data_stock_dir>/wgnssk2026/WGNSSK2026_cod.txt and _saithe3a46.txt (pdftotext -layout of the
  WGNSSK 2026 report sections): cod Tables 4.8 (F), 4.9 (N), 4.11 (stock weights), 4.13 (M) per
  substock; saithe Tables 14.4.2 (F), 14.4.3 (N), 14.3.11 (stock weights) and the M-at-age line.
Output: <out>/stock_naa.csv  stock,run,year,age,N_thousands,F,M,sw_kg,cw_kg (plus-group = last age).
"""
import os
import re
import sys

import pandas as pd

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
ROW = re.compile(r"^\s*((?:19|20)\d\d)((?:\s+-?[\d.]+)+)\s*$")


def html_table(path):
    t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", open(path, encoding="utf-8", errors="replace").read()))
    m = re.search(r"Year Age ((?:\d{1,2} )+)", t)
    ages = [int(a) for a in m.group(1).split()]
    body = t[m.end():].split()
    rows, i = [], 0
    while i + len(ages) < len(body) and re.fullmatch(r"(19|20)\d\d", body[i]):
        try:
            vals = [float(x) for x in body[i + 1:i + 1 + len(ages)]]
        except ValueError:
            break
        rows.append([int(body[i])] + vals)
        i += 1 + len(ages)
    return pd.DataFrame(rows, columns=["year"] + ages).set_index("year")


def lowestoft(path):
    ln = open(path, encoding="utf-8", errors="replace").read().splitlines()
    y0, y1 = [int(x) for x in ln[2].split()[:2]]
    a0, a1 = [int(x) for x in ln[3].split()[:2]]
    vals = []
    for s in ln[5:]:
        s = s.split("#")[0].strip()
        if s:
            vals.append([float("nan") if x == "NA" else float(x) for x in s.split()[:a1 - a0 + 1]])
    df = pd.DataFrame(vals[:y1 - y0 + 1], index=range(y0, y0 + len(vals[:y1 - y0 + 1])),
                      columns=range(a0, a1 + 1))
    return df.ffill()  # trailing NA years (e.g. whiting 2026n M) take the last given year


def text_block(lines, caption, sub=None, ages=None):
    i = next(k for k, s in enumerate(lines) if caption in s)
    if sub:
        i = next(k for k in range(i + 1, len(lines)) if lines[k].strip() == sub)
    rows = []
    for k in range(i + 1, len(lines)):
        s = lines[k]
        if k > i + 1 and (re.search(r"Table \d+\.\d", s) or s.strip() in ("Northwestern", "Southern", "Viking")):
            break
        m = ROW.match(s)
        if m:
            v = [float(x) for x in m.group(2).split()]
            rows.append([int(m.group(1))] + v[:len(ages)])
    return pd.DataFrame(rows, columns=["year"] + ages).set_index("year")


def long(stock, run, N, F, M, sw, cw=None):
    recs = []
    for y in N.index:
        for a in N.columns:
            g = lambda d: (float(d.loc[y, a]) if d is not None and y in d.index and a in d.columns else float("nan"))
            recs.append([stock, run, y, a, g(N), g(F), g(M), g(sw), g(cw)])
    return recs


recs = []
sam = {"had.27.46a20": ["NShaddock_WGNSSK2026_Run1", "NShaddock_WGNSSK2024_Run2"],
       "whg.27.47d": ["NSwhiting_2026n", "NSwhiting_2025"],
       "cod.27.47d20_old": ["nscod20_ass02"]}
for stock, runs in sam.items():
    for run in runs:
        d = os.path.join(src, "sam_" + run)
        N = html_table(os.path.join(d, "res", "xxx-00-00.00.00_tab3.html"))
        F = html_table(os.path.join(d, "res", "xxx-00-00.00.00_tab2.html"))
        sw = lowestoft(os.path.join(d, "data", "sw.dat"))
        cw = lowestoft(os.path.join(d, "data", "cw.dat"))
        M = lowestoft(os.path.join(d, "data", "nm.dat"))
        recs += long(stock, run, N, F, M, sw, cw)

cod = open(os.path.join(src, "wgnssk2026", "WGNSSK2026_cod.txt"), encoding="utf-8").read().splitlines()
A = list(range(1, 8))
for sub in ["Viking", "Northwestern", "Southern"]:
    N = text_block(cod, "Table 4.9. Cod", sub, A + ["tot"])[A]
    F = text_block(cod, "Table 4.8. Cod", sub, A + ["fbar"])[A]
    sw = text_block(cod, "Table 4.11. Cod", sub, A)
    M = text_block(cod, "Table 4.13. Cod", sub, A)
    recs += long("cod.27.46a7d20_" + sub, "WGNSSK2026_multistockSAM", N, F, M, sw)

sai = open(os.path.join(src, "wgnssk2026", "WGNSSK2026_saithe3a46.txt"), encoding="utf-8").read().splitlines()
A = list(range(3, 11))
N = text_block(sai, "Table 14.4.3. Saithe", None, A)
F = text_block(sai, "Table 14.4.2. Saithe", None, A[:-1])
F[10] = F[9]  # F at 9 and 10+ are coupled (9+)
sw = text_block(sai, "Table 14.3.11. Saithe", None, A)
mline = next(s for s in sai if s.strip().startswith("Natural mortality") and "0.384" in s)
mv = [float(x) for x in mline.split()[2:]]
M = pd.DataFrame([mv] * len(N), index=N.index, columns=A)
recs += long("pok.27.3a46", "WGNSSK2026_SAM", N, F, M, sw)

df = pd.DataFrame(recs, columns=["stock", "run", "year", "age", "N_thousands", "F", "M", "sw_kg", "cw_kg"])
df.to_csv(os.path.join(out, "stock_naa.csv"), index=False)
chk = df[df.year.between(2019, 2025)].groupby(["stock", "run"]).agg(
    years=("year", lambda s: f"{s.min()}-{s.max()}"), ages=("age", lambda s: f"{s.min()}-{s.max()}"),
    nN=("N_thousands", "count"), nF=("F", "count"), nM=("M", "count"), nsw=("sw_kg", "count"))
print(chk)
