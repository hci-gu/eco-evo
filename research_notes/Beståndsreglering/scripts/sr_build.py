#!/usr/bin/env python3
"""Build lagged stock-recruitment pairs from the SAG summary tables.

Usage:
    python3 -I scripts/sr_build.py SAG_DIR OUT_CSV [SERIES_CSV]
    e.g. python3 -I research_notes/Beståndsreglering/scripts/sr_build.py \
             research_notes/Beståndsreglering/data/sag \
             research_notes/Beståndsreglering/tables/sr_pairs.csv \
             research_notes/Beståndsreglering/tables/sr_ssb_series.csv

For each stock: recruitment at age a in year y + a is paired with SSB in year y
(year class y). Recruitment years from the terminal year on are dropped
(terminal year = SAG TerminalYear, capped at assessment year - 1; later years are
forecast / geometric-mean assumptions, and the terminal year's recruitment rests on
one survey point and is the least determined).
North Sea cod: the whole stock is the sum of the three 2026 substocks
(SSB and age-1 recruitment are additive).
Columns: stock, year_class, ssb_t, rec_n (numbers as in SAG, usually thousands),
rec_year, rec_age, rec_cv (from the SAG 95 % interval, log scale, if given),
terminal_year, assessment_key, use (1/0), drop_reason.
use = 0 when the recruitment's 95 % interval gives a log-scale CV < 0.05 (the
assessment imposes deterministic recruitment there: anglerfish before 1981,
spurdog before 1975) or > 0.5 (undetermined, e.g. North Sea horse mackerel
2022-23).
SERIES_CSV (optional): stock, year, ssb, rec, terminal_year - the unlagged SSB
series used for "today's SSB".
"""
import csv
import json
import math
import pathlib
import sys

SUBSTOCKS = {26001: "viking", 26002: "southern", 26003: "northwestern"}


def lines(path):
    s = json.loads(path.read_text())
    return s, {int(l["Year"]): l for l in s["Lines"]}


def cv(lo, hi):
    if lo and hi and lo > 0 and hi > 0:
        return math.log(hi / lo) / (2 * 1.96)
    return ""


def pairs_for(label, key, s, by_year):
    a = int(s["RecruitmentAge"])
    ay = int(s["AssessmentYear"])
    term = min(int(s["TerminalYear"] or ay - 1), ay - 1)
    rows = []
    for y, l in sorted(by_year.items()):
        ry = y + a
        if ry >= term or ry not in by_year:
            continue
        r = by_year[ry].get("Recruitment")
        ssb = l.get("SSB")
        if not r or not ssb or r <= 0 or ssb <= 0:
            continue
        c = cv(by_year[ry].get("Low_Recruitment"), by_year[ry].get("High_Recruitment"))
        reason = ""
        if c != "" and c < 0.05:
            reason = "deterministic recruitment (cv<0.05)"
        elif c != "" and c > 0.5:
            reason = "undetermined recruitment (cv>0.5)"
        rows.append(dict(stock=label, year_class=y, ssb_t=ssb, rec_n=r, rec_year=ry,
                         rec_age=a, rec_cv=round(c, 4) if c != "" else "",
                         terminal_year=term, assessment_key=key,
                         use=0 if reason else 1, drop_reason=reason))
    return rows


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    d = pathlib.Path(sys.argv[1])
    keys = json.loads((d / "keys.json").read_text())
    out = []
    series = []
    sub = {}
    for label, key, _note in keys:
        s, by = lines(d / f"summary_{key}.json")
        if key in SUBSTOCKS:
            sub[key] = (s, by)
        if not by or all(not l.get("SSB") for l in by.values()):
            continue
        out += pairs_for(label, key, s, by)
        ay = int(s["AssessmentYear"])
        term = min(int(s["TerminalYear"] or ay - 1), ay - 1)
        for y, l in sorted(by.items()):
            series.append(dict(stock=label, year=y, ssb=l.get("SSB") or "",
                               rec=l.get("Recruitment") or "", terminal_year=term))
    # whole North Sea cod = sum of the substocks
    if len(sub) == 3:
        s0 = sub[26001][0]
        years = set.intersection(*[set(v[1]) for v in sub.values()])
        tot = {}
        for y in years:
            ls = [v[1][y] for v in sub.values()]
            if all(l.get("SSB") and l.get("Recruitment") for l in ls):
                tot[y] = dict(SSB=sum(l["SSB"] for l in ls),
                              Recruitment=sum(l["Recruitment"] for l in ls))
        out += pairs_for("cod.27.46a7d20_sum", "26001+26002+26003", s0, tot)
        for y, l in sorted(tot.items()):
            series.append(dict(stock="cod.27.46a7d20_sum", year=y, ssb=l["SSB"],
                               rec=l["Recruitment"], terminal_year=2025))
    p = pathlib.Path(sys.argv[2])
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("wrote", len(out), "pairs to", p)
    if len(sys.argv) > 3:
        q = pathlib.Path(sys.argv[3])
        with q.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(series[0]))
            w.writeheader()
            w.writerows(series)
        print("wrote", len(series), "series rows to", q)


if __name__ == "__main__":
    main()
