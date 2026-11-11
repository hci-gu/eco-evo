"""Capture probability per ENCOUNTER (prey inside the rule-A detection disc, predator hungry) for the
fish-prey pairs, built from per-strike capture success as a function of prey:predator length ratio (PPR).

Usage: python3 -I p_ppr.py <out_json>

No model output is read. Every number is a literature value (source tag in the comment) or a stated
judgement [J]. See ../evidence.md for the sources E1..

Steps
1. Per-strike success p_s(PPR): least-squares line through the juvenile bluefish data of Juanes &
   Conover 1994 (E5; per-trial means of single-size trials and pooled counts of choice trials), the
   only open data set with predator and prey sizes. Scharf et al. 1998 (E6, abstract) state the same
   form (linear decline over PPR 0.20-0.65) for bluefish on two prey species.
2. Cross-species spread at PPR 0.25-0.33 (single or dispersed prey, per strike): percids/esocids
   (E1, E8), cod (E3), whiting on shrimp (E4), esocids on minnow/shad (E7). Used as the low/high
   per-strike bracket around the line.
3. Per encounter: p = q p_s / (1 - r (1 - p_s))
      q = P(strike | detected) for a hungry predator, i.e. 1 - pre-strike escape (prey sees the
          predator first and leaves R);
      r = P(another strike on the same prey | miss), i.e. the follow-up inside one encounter.
   Tank values bound r from above (prey cannot leave R): perch 0.91, pike 0.47, pikeperch 0.09
   (inverted from E1's per-strike and per-sequence numbers); bluefish "unsuccessful attacks were not
   usually followed by pursuits" (E5); whiting "either rapidly reorientate, or simply turn away" (E4);
   cod "would often continue to pursue prey that escaped" (E3, 60 x 33 cm arena).
4. PPR per pair: the encounter-weighted prey length inside the stage-access window, from the
   realised stomach ratios and window p95 of research_notes/"Arsungarnas andel" (A-ring)
   evidence_prey_size.md [D/J].
"""
import json
import sys

import numpy as np

OUT = sys.argv[1]

# ---------------------------------------------------------------- 1. bluefish per-strike data (E5)
# bluefish TL class midpoints (mm): small 105, medium 145, large 185; silversides 30, 60, 90
BF = {"S": 105.0, "M": 145.0, "L": 185.0}
SV = {"s": 30.0, "m": 60.0, "l": 90.0}
# single-size trials: mean percent success per trial (text of E5; small x small / small x medium
# only in the figure -> omitted)
single = [("M", "s", 0.631), ("M", "m", 0.408), ("M", "l", 0.229),
          ("L", "s", 0.868), ("L", "m", 0.512), ("L", "l", 0.504)]
# choice trials: successful / unsuccessful attacks (Table 2 of E5)
choice = [("S", "s", 11, 4), ("S", "m", 6, 20),
          ("M", "s", 23, 3), ("M", "m", 9, 27),
          ("M", "s", 15, 5), ("M", "l", 6, 15),
          ("L", "s", 3, 2), ("L", "m", 4, 2),
          ("L", "s", 13, 3), ("L", "m", 13, 14), ("L", "l", 8, 15)]
x, y, w = [], [], []
for b, s, v in single:
    x.append(SV[s] / BF[b]); y.append(v); w.append(1.0)
for b, s, k, n in choice:
    x.append(SV[s] / BF[b]); y.append(k / (k + n)); w.append(1.0)
x, y, w = map(np.asarray, (x, y, w))
A = np.vstack([np.ones_like(x), x]).T
coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
res = y - A @ coef
s2 = float(res @ res) / (len(x) - 2)
cov = s2 * np.linalg.inv(A.T @ A)
print("1. Bluefish per-strike success vs PPR (Juanes & Conover 1994, E5): n points", len(x))
for xi, yi in sorted(zip(x, y)):
    print(f"   PPR {xi:.3f}  success {yi:.3f}")
print(f"   line: p_s = {coef[0]:.3f} {coef[1]:+.3f} PPR   (residual SD {np.sqrt(s2):.3f})")
# pooled counts only (choice trials), for a count-weighted check
kk = np.array([c[2] for c in choice], float); nn = np.array([c[2] + c[3] for c in choice], float)
xc = np.array([SV[c[1]] / BF[c[0]] for c in choice])
Ac = np.vstack([np.ones_like(xc), xc]).T
cc, *_ = np.linalg.lstsq(Ac * np.sqrt(nn)[:, None], (kk / nn) * np.sqrt(nn), rcond=None)
print(f"   count-weighted (choice trials only): p_s = {cc[0]:.3f} {cc[1]:+.3f} PPR")


def p_line(ppr):
    return float(np.clip(coef[0] + coef[1] * ppr, 0.02, 0.98))


def p_line_se(ppr):
    v = np.array([1.0, ppr])
    return float(np.sqrt(v @ cov @ v))


# ---------------------------------------------------------------- 2. cross-species spread
print("\n2. Per-strike capture success on single / dispersed fish prey, other predators (PPR ~0.2-0.35)")
OTHERS = [
    ("perch / roach, single prey, pool 2.44 m (E1)", 0.27, 34 / 168),
    ("pike pursuit / roach (E1)", 0.30, 17 / 144),
    ("pike sit-and-wait / roach (E1)", 0.30, 8 / 22),
    ("pikeperch / roach, dusk (E1)", 0.30, 0.085),
    ("0-group cod / flatfish, PPR 0.20-0.33, most trials 1.0 (E3)", 0.27, 0.8),
    ("0-group cod / flatfish, PPR 0.35-0.45 (E3)", 0.40, 0.17),
    ("whiting 23-26 cm / shrimp, empty (E4)", 0.26, 0.8),
    ("esocids / gizzard shad, 'optimal size' (E7, abstract)", float("nan"), 0.78),
    ("esocids / fathead minnow (E7, abstract)", float("nan"), 0.67),
    ("esocids / bluegill, spiny (E7, abstract)", float("nan"), 0.14),
]
for name, r, v in OTHERS:
    line = f"line {p_line(r):.2f}" if r == r else "line -"
    print(f"   {name:62s} PPR {r:4.2f}  p_s {v:.2f}   ({line})")
vals = np.array([v for _, r, v in OTHERS if r == r and r < 0.35])
print(f"   PPR < 0.35 data: median {np.median(vals):.2f}, range {vals.min():.2f}-{vals.max():.2f}")

# tank follow-up probability r inverted from E1: p_seq = p_s / (1 - r (1 - p_s))
print("\n3. Follow-up probability r inverted from per-strike and per-sequence success (E1, tanks)")
for name, ps, pseq in [("perch", 0.176, 35 / 47), ("pike", 0.151, 0.25), ("pikeperch", 0.085, 0.093)]:
    r = (1 - ps / pseq) / (1 - ps)
    print(f"   {name:10s} p_s {ps:.3f}  p_seq {pseq:.3f}  -> r {r:.2f}")

# ---------------------------------------------------------------- 4. per pair
# PPR (central, low, high) of the encountered prey inside the window [D/J from Arsungarnas andel]
PAIRS = {
    # pair: (p_library, PPR central, low, high, note)
    "predatory_fish->planktivorous_fish": (0.2, 0.25, 0.19, 0.33, "saithe 0.9 of a: pollock mean 0.19, Mehl median 0.27, window p95 0.30"),
    "predatory_fish->benthivorous_fish": (0.2, 0.25, 0.19, 0.33, "saithe on settled 0-group gadoids, window 16 cm at 52 cm"),
    "predatory_fish->whiting": (0.2, 0.25, 0.19, 0.33, "saithe on juvenile whiting (gape 0.47 share)"),
    "predatory_fish->mackerel": (0.2, 0.30, 0.20, 0.40, "spurdog 0.76 / hake 0.22 of a on 30 cm mackerel [J]"),
    "whiting->planktivorous_fish": (0.2, 0.28, 0.26, 0.40, "whiting 20-30 cm p50 0.26-0.27, p95 0.38-0.46"),
    "whiting->benthivorous_fish": (0.2, 0.33, 0.27, 0.42, "benthivore prey p50 0.31, p95 0.45"),
    "mackerel->planktivorous_fish": (0.2, 0.22, 0.15, 0.31, "age-0 sprat/herring, window p95 0.31-0.33 [J]"),
    "benthivorous_fish->planktivorous_fish": (0.3, 0.20, 0.16, 0.30, "cod 45-65 cm p50 0.16-0.21, p95 0.30"),
    "benthivorous_fish->whiting": (0.3, 0.22, 0.16, 0.33, "cod on whiting [J]"),
}
# readings: (q, r, per-strike source)
READ = {
    "low": dict(q=0.6, r=0.0, ps="low"),         # percid per strike, no follow-up, 40 % pre-strike escape
    "percid": dict(q=0.8, r=0.3, ps="low"),      # percid per strike with the central q, r
    "central": dict(q=0.8, r=0.3, ps="line"),    # bluefish line with the central q, r
    "central_full": dict(q=0.8, r=0.3, ps="line_full"),  # central at field gut fullness (E4 slope)
    "per_strike": dict(q=1.0, r=0.0, ps="line"),  # bluefish line, per strike as per encounter
    "high": dict(q=1.0, r=0.6, ps="high"),       # gadoid tank per strike, frequent follow-up
    "tank_sequence": dict(q=1.0, r=0.9, ps="line"),  # confined-tank sequence (perch-like r)
}
PS_LOW, PS_HIGH = 0.2, 0.8   # cross-species bracket at PPR 0.25-0.33 (perch single prey / cod, whiting, esocids)
# Fullness: E4 small whiting p_s = 0.8 + s x stomach g (s -0.10 small, -0.17 large shrimp; max 8 g). At a field
# feeding level of 0.22-0.53 of Cmax (AGENTS.md, benthic pairs), read as mean stomach 0.22-0.53 of 8 g = 1.8-4.2 g:
# small shrimp 0.62-0.38 (x0.78-0.48 of 0.8), large 0.49-0.09 (x0.61-0.11). Central x0.6 [L one species, J].
FULL = 0.6
print("\nFullness factor on p_s (E4 slopes at stomach 1.8-4.2 g of 8 g):")
for g in (1.8, 3.0, 4.2):
    print(f"   {g:.1f} g: small shrimp x{(0.8 - 0.10 * g) / 0.8:.2f}, large shrimp x{max(0.8 - 0.17 * g, 0) / 0.8:.2f}")


def p_enc(ps, q, r):
    return q * ps / (1.0 - r * (1.0 - ps))


out = {"line": {"b0": float(coef[0]), "b1": float(coef[1]), "sd": float(np.sqrt(s2))},
       "readings": {k: {kk: v for kk, v in d.items()} for k, d in READ.items()}, "pairs": {}}
print("\n4. p per encounter by pair (p = q p_s / (1 - r (1 - p_s))); factor = p / p_library")
hdr = "   " + f"{'pair':40s} {'p_lib':>5s} {'PPR':>5s} {'p_s':>5s} " + " ".join(f"{k:>14s}" for k in READ)
print(hdr)
for pair, (plib, r0, rlo, rhi, note) in PAIRS.items():
    ps_c = p_line(r0)
    row, d = [], {"p_lib": plib, "PPR": [r0, rlo, rhi], "p_s_line": ps_c,
                  "p_s_line_range": [p_line(rhi), p_line(rlo)], "note": note}
    for k, rd in READ.items():
        if rd["ps"] == "line":
            ps = ps_c
        elif rd["ps"] == "line_full":
            ps = FULL * ps_c
        elif rd["ps"] == "low":
            ps = PS_LOW
        else:
            ps = max(PS_HIGH, p_line(rlo))
        p = p_enc(ps, rd["q"], rd["r"])
        d[k] = {"p_s": ps, "p": p, "factor": p / plib}
        row.append(f"{p:5.2f} (x{p / plib:4.2f})")
    out["pairs"][pair] = d
    print(f"   {pair:40s} {plib:5.2f} {r0:5.2f} {ps_c:5.2f} " + " ".join(f"{s:>14s}" for s in row))
    print(f"      {note}; line p_s over PPR {rlo}-{rhi}: {p_line(rhi):.2f}-{p_line(rlo):.2f} (+-SE {p_line_se(r0):.2f})")

# group values (a-weighted members: saithe ~0.9 of the WC fish pairs, whiting 1.0, mackerel 0.96, cod)
print("\n5. Group values: mean over the group's fish-prey pairs (central reading; low and high over its pairs)")
GROUPS = {
    "predatory_fish": ["predatory_fish->planktivorous_fish", "predatory_fish->benthivorous_fish", "predatory_fish->whiting"],
    "whiting": ["whiting->planktivorous_fish", "whiting->benthivorous_fish"],
    "mackerel": ["mackerel->planktivorous_fish"],
    "benthivorous_fish": ["benthivorous_fish->planktivorous_fish", "benthivorous_fish->whiting"],
}
for g, ps in GROUPS.items():
    vals = [out["pairs"][p]["central"]["p"] for p in ps]
    lo = [out["pairs"][p]["low"]["p"] for p in ps]
    hi = [out["pairs"][p]["high"]["p"] for p in ps]
    print(f"   {g:20s} central {np.mean(vals):.2f}  (low {min(lo):.2f} - high {max(hi):.2f})")
json.dump(out, open(OUT, "w"), indent=1)
print(f"\nwritten {OUT}")
