"""s_J per species and per model group: SMS quarterly F at age 0 and 1 weighted by the BOX age-0 / age-1
biomass shares (tables/juvenile_share_by_species.csv), divided by the stock's biomass-weighted F on all ages.

Usage: python3 -I box_weighted_sj.py <sms summary.out> <juvenile_share_by_species.csv>

Per species s:  F_J,box = 4 * sum_q (F_q0 b_q0 + F_q1 b_q1) / sum_q (b_q0 + b_q1)   (F per quarter, SMS 2013-22)
                r_s = F_J,box / F_all,stock (SMS 2013-22, all ages, N.bar * west weights)
Box shares b_qa = share of age a in the species' box biomass in quarter q (q-corrected):
  period 2001-25: Q1 and Q3 measured, Q2 := Q1 and Q4 := Q3 (no survey);   period 1991-96: all four.
Group: F_J,group = sum_s W_s j_s r_s F_s / sum_s W_s j_s, with W_s the model member weight, j_s the
annual-mean J share and F_s the library's species F (catch / total biomass); s_J = F_J,group / F_group,
F_group = sum_s W_s F_s / sum_s W_s. Also s_A = (1 - s_J p_J) / (1 - p_J) and the cap 1 / p_J.
"""
import csv
import sys
from collections import defaultdict

SPN = {16: "cod", 18: "haddock", 21: "herring", 24: "Norway pout", 25: "sprat", 26: "plaice"}
sms, shares = sys.argv[1], sys.argv[2]

Fq = defaultdict(float)          # (species, q, age) -> mean quarterly F
fb = defaultdict(float); bb = defaultdict(float)
cfb = defaultdict(float); cbb = defaultdict(float)
n = 0
with open(sms) as fh:
    head = fh.readline().split(); ix = {k: i for i, k in enumerate(head)}
    for line in fh:
        p = line.split()
        if len(p) < len(head) - 1:
            continue
        yr, q, sp, a = int(p[ix["Year"]]), int(p[ix["Quarter"]]), int(p[ix["Species.n"]]), int(p[ix["Age"]])
        if sp not in SPN or not 2013 <= yr <= 2022:
            continue
        F = float(p[ix["F"]]); b = float(p[ix["N.bar"]]) * float(p[ix["west"]])
        if a <= 1:
            Fq[(SPN[sp], q, a)] += F / 10
        cfb[(SPN[sp], q, a)] += F * b; cbb[(SPN[sp], q, a)] += b
        fb[SPN[sp]] += F * b; bb[SPN[sp]] += b
Fall = {s: 4 * fb[s] / bb[s] for s in fb}

sh = defaultdict(dict)
for r in csv.DictReader(open(shares)):
    if r["age0"] == "":
        continue
    sh[(r["period"], r["species"])][int(r["quarter"])] = (float(r["age0"]), float(r["age1"]))

# J variants: quarters in which age 1 still counts as J (age 0 is J in Q3-Q4; it is absent in Q1-Q2).
VARIANTS = {"A (age 1 all year)": {1, 2, 3, 4}, "B (age 1 leaves at Q3)": {1, 2}, "C (age 1 leaves at Q4)": {1, 2, 3}}


def injs(v, q, a):
    return a == 0 or (a == 1 and q in VARIANTS[v])


r_s = {}
for v in VARIANTS:
    for period in ("2001-25", "1991-96"):
        if v != "A (age 1 all year)" and period == "1991-96":
            continue
        print(f"--- {v}, box shares {period}")
        for s in SPN.values():
            d = dict(sh.get((period, s), {}))
            if not d:
                continue
            if period == "2001-25":
                d[2] = d[1]; d[4] = d[3]
            num = sum(Fq[(s, q, a)] * d[q][a] for q in d for a in (0, 1) if injs(v, q, a))
            den = sum(d[q][a] for q in d for a in (0, 1) if injs(v, q, a))
            j = den / 4
            fj = 4 * num / den
            # stock F on the non-J cells (stock weights), relative to the stock F_all
            fa = 4 * sum(cfb[(s, q, a)] for (ss, q, a) in cfb if ss == s and not injs(v, q, a)) / \
                sum(cbb[(s, q, a)] for (ss, q, a) in cbb if ss == s and not injs(v, q, a))
            r_s[(v, period, s)] = (fj / Fall[s], j, fa / Fall[s])
            print(f"{s:12s} F_J,box {fj:.3f}  F_all,stock {Fall[s]:.3f}  r_J {fj / Fall[s]:.2f}  "
                  f"r_A(stock) {fa / Fall[s]:.2f}  annual-mean J share {j:.2f}")

GROUPS = {
    # member: (model weight, library F = catch / total biomass)
    "planktivores": {"herring": (0.788, 0.101), "blue whiting": (0.106, 0.18), "Norway pout": (0.076, 0.03),
                     "sprat": (0.030, 0.45)},
    "benthivores": {"haddock": (38.6, 0.06), "cod": (31.4, 0.39), "long rough dab": (12.6, 0.10), "dab": (4.5, 0.18),
                    "plaice": (3.8, 0.04), "witch": (1.7, 0.12), "starry ray": (1.5, 0.03), "grey gurnard": (0.9, 0.03),
                    "lemon sole": (0.9, 0.09)},
}
# Herring: box J is ~92 % NSAS by weight (HAWG 2026 Table 3.2.4: WBSS 5 % of 0-wr, 12 % of 1-wr in 3.a.20 Q3),
# so F_J = r_J x (0.92 x 0.145 + 0.08 x 0.0087); adults ~0.63 NSAS / 0.37 WBSS (Q3 2-6 wr), F_A = r_A x that mix.
HJ = 0.145 * 0.92 + 0.0087 * 0.08
HA = 0.145 * 0.63 + 0.0087 * 0.37
for v in VARIANTS:
    for period in ("2001-25", "1991-96"):
        if (v, period, "herring") not in r_s:
            continue
        for g, mem in GROUPS.items():
            num = den = fw = w = fbox = 0.0
            for s, (W, F) in mem.items():
                fw += W * F; w += W
                if (v, period, s) not in r_s:
                    fbox += W * F       # no age key: all tonnes adult at the species F
                    continue
                r, j, ra = r_s[(v, period, s)]
                fjs = r * (HJ if s == "herring" else F)
                fas = ra * (HA if s == "herring" else F)
                num += W * j * fjs; den += W * j
                fbox += W * (j * fjs + (1 - j) * fas)
            fg = fw / w; pj = den / w; fj = num / den; sj = fj / fg
            print(f"{v:24s} {period} {g:12s} F_group {fg:.3f}  F_all,box {fbox / w:.3f}  p_J {pj:.2f}  "
                  f"F_J {fj:.4f}  s_J {sj:.2f}  s_A {(1 - sj * pj) / (1 - pj):.2f}  cap {1 / pj:.1f}")

# Per-quarter F on the box J tonnes (per year), 2001-25 shares (Q2 := Q1, Q4 := Q3), relative to F_all,stock.
print("--- per quarter, F_J,box(q) / F_all,stock (2001-25 shares)")
for s in SPN.values():
    d = dict(sh.get(("2001-25", s), {}))
    if not d:
        continue
    d[2] = d[1]; d[4] = d[3]
    vals = []
    for q in (1, 2, 3, 4):
        b0, b1 = d[q]
        vals.append(4 * (Fq[(s, q, 0)] * b0 + Fq[(s, q, 1)] * b1) / (b0 + b1) / Fall[s])
    print(f"{s:12s} " + " ".join(f"Q{q} {v:.2f}" for q, v in zip((1, 2, 3, 4), vals)))
