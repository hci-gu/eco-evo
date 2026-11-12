"""Arithmetic for the note "Ljusstyrd aptit" (light-cued appetite).

Every number printed here is either read from a source (marked READ) or
derived from those readings (marked D). Run: python3 -I light_appetite_numbers.py
"""
import math
import statistics

print("=" * 72)
print("1. Berg et al. 2024 Fig. 6: seasonal swing of somatic condition")
print("   (GAM line of the ln(weight) residual at length; READ off the figure,")
print("   +-0.005 reading error; quarterly samples, phase resolution +-1.5 mo)")
print("=" * 72)
# (panel, min, season of min, max, season of max); seasons are the
# EXPERIENCED light season (Table 2 assigns each regime its own season).
fig6 = [
    ("10 C natural", -0.050, "winter", +0.030, "late summer-autumn"),
    ("7 C natural",  -0.035, "winter", +0.000, "summer"),
    ("7 C offset",   -0.030, "spring", +0.045, "autumn"),
    ("10 C offset",  -0.040, "winter/summer (inconclusive)", +0.005, "autumn"),
]
swings = []
for name, lo, slo, hi, shi in fig6:
    sw = hi - lo
    swings.append(sw)
    pct = (math.exp(sw) - 1) * 100
    print(f"  {name:13s} min {lo:+.3f} ({slo}), max {hi:+.3f} ({shi}): "
          f"swing {sw:.3f} ln units = {pct:.1f} % of weight at length")
print(f"  range over the three conclusive panels: "
      f"{min(swings[:3]):.3f}-{max(swings[:3]):.3f} ln units")

print()
print("Individual scatter (Fig. S5, READ): residuals span about -0.3..+0.3 in")
print("every panel, i.e. the seasonal swing is ~1/8-1/4 of the individual range.")

print()
print("-" * 72)
print("1b. Convert the weight swing to a reserve-fill bracket (D)")
print("-" * 72)
ME = 4990.0          # MJ/t, planktivores max_energy_reserve (library, 243.3)
FAT_E = 39.5         # MJ/kg lipid (library comment, start_fill derivation)
# Lower bound: lost mass is lean-like wet tissue (~5 MJ/kg) or fat replaced
# by water; upper bound: all lost mass is lipid (39.5 MJ/kg). Herring replace
# lost fat with water (fat% + water% ~ constant), so the weight-at-length
# residual UNDERSTATES fat loss; the upper bound is not a ceiling on the fat
# loss, only on what the weight change can show.
for sw in (0.035, 0.080):
    frac = math.exp(sw) - 1
    lo = frac * 1000 * 5.0 / ME
    hi = frac * 1000 * FAT_E / ME
    print(f"  swing {sw:.3f}: {frac*100:.1f} % of weight -> fill change "
          f"{lo:.2f} (5 MJ/kg tissue) to {hi:.2f} (pure lipid)")
field_drop = 0.68 - 0.20   # start_fill Dec -> Apr (library)
field_swing = 0.96 - 0.20  # start_fill Aug -> Apr
print(f"  field planktivore fill (start_fill): Dec->Apr drop {field_drop:.2f}, "
      f"annual swing {field_swing:.2f} (Apr 0.20 -> Aug 0.96)")

print()
print("=" * 72)
print("2. dos Santos Schmidt et al. 2022 Table 1 (sister experiment, Matre,")
print("   8.9 +- 0.2 C, natural vs 6-mo offset light): Kn by EXPERIENCED month")
print("=" * 72)
# (experienced month, regime, sex, N, Kn mean, Kn sd, GSI mean) READ
t1 = [
    ("Oct", "nat", "F", 3, 1.05, 0.06, 0.41), ("Oct", "nat", "M", 7, 1.05, 0.07, 0.23),
    ("Apr", "off", "F", 3, 1.07, 0.25, 1.26), ("Apr", "off", "M", 7, 1.04, 0.14, 1.15),
    ("Feb", "nat", "F", 9, 1.00, 0.17, 1.92), ("Feb", "nat", "M", 2, 0.87, 0.09, 1.05),
    ("Aug", "off", "F", 4, 0.89, 0.09, 0.68), ("Aug", "off", "M", 6, 0.98, 0.14, 0.25),
    ("May", "nat", "F", 6, 1.01, 0.11, 2.72), ("May", "nat", "M", 4, 0.96, 0.14, 3.63),
    ("Nov", "off", "F", 5, 0.96, 0.05, 1.14), ("Nov", "off", "M", 5, 0.99, 0.05, 0.64),
    ("Oct", "nat", "F", 28, 0.96, 0.12, 1.57), ("Oct", "nat", "M", 22, 1.03, 0.15, 2.41),
    ("Apr", "off", "F", 37, 1.01, 0.08, 4.13), ("Apr", "off", "M", 29, 1.04, 0.08, 7.43),
]
season = {"Nov": "winter", "Feb": "winter", "May": "spring", "Apr": "spring",
          "Aug": "summer", "Oct": "autumn"}
by = {}
for m, r, s, n, kn, sd, gsi in t1:
    by.setdefault(season[m], []).append((n, kn, sd, m, r, s))
for se in ("winter", "spring", "summer", "autumn"):
    rows = by[se]
    ntot = sum(r[0] for r in rows)
    wmean = sum(r[0] * r[1] for r in rows) / ntot
    det = ", ".join(f"{m}/{r}/{s} n{n} {kn:.2f}+-{sd:.2f}" for n, kn, sd, m, r, s in rows)
    print(f"  {se:6s} n-weighted Kn {wmean:.3f} (n {ntot}): {det}")
# standard error of a mean of n=10 at sd 0.10
for n, sd in ((10, 0.10), (10, 0.15), (5, 0.10)):
    print(f"  SE of a sample mean, n {n}, sd {sd:.2f}: {sd/math.sqrt(n):.3f} "
          f"(95 % CI half-width {1.96*sd/math.sqrt(n):.3f})")
print("  Berg's swing 0.035-0.08 ln units (~3.5-8 %) is of the order of 1-2 SE")
print("  of a 10-fish sample: the sister experiment can neither show nor exclude it.")
gsis = [g for *_, g in t1]
print(f"  GSI range over all samples: {min(gsis):.2f}-{max(gsis):.2f} %; natural-group")
print("  winter (Feb) 1.05-1.92 %, spring (May) 2.72-3.63 %.")
print("  A gonad term of <= 2-4 % of weight cannot by itself produce a 3.5-8 % swing,")
print("  and in spring spawners heavier winter gonads would RAISE winter condition.")

print()
print("=" * 72)
print("3. Bull 1996 thesis ch. 3.4 (salmon parr, excess food 24 h, ambient T,")
print("   simulated natural photoperiod): lipid trajectory of controls (READ)")
print("=" * 72)
sep, apr = 5.32, 2.63
days = 7 * 30.4   # mid-Sep to mid-Apr, ~7 months
print(f"  lipid {sep} % BW (Sep) -> {apr} % BW (Apr): ratio {apr/sep:.2f}, "
      f"loss {sep-apr:.2f} % BW over ~{days:.0f} d = {(sep-apr)/days*100:.2f} % BW per 100 d")
print("  Deprivation losses (3 wk each): early 1.12, mid 1.00, late 1.94 % BW;")
print("  appetite rose above controls only after the EARLY (Oct) deprivation, for")
print("  14 d, then fell back to control level; fat gain during refeeding +0.49 % BW")
print("  (early), -0.05 (mid, Dec), -0.37 (late, Feb): only the early deficit was")
print("  partly restored (fig. 3.12, 3.14).")
print("  ch. 3.5 (photoperiod, ad lib 24 h, same ambient T): after the Nov deficit")
print("  restoration +1.02 (acc.), +1.27 (del.), +1.91 (control) % BW in 3 wk;")
print("  after the Jan deficit the group perceiving April did NOT restore,")
print("  the group perceiving October and the controls did (in this second year the")
print("  controls restored a Jan deficit; in ch. 3.4 a Dec deficit was not restored).")

print()
print("=" * 72)
print("4. Group weighting of where a light-cued winter appetite could apply (D)")
print("=" * 72)
members = {"herring": 0.554, "Norway pout": 0.350, "blue whiting": 0.075, "sprat": 0.021}
cases = {
    "herring only (pout contradicts, blue whiting absent, sprat no krill)":
        members["herring"],
    "herring + sprat": members["herring"] + members["sprat"],
}
for k, v in cases.items():
    print(f"  {k}: {v:.3f} of planktivore biomass")
print("  The krill-note group krill ration 0.24 %BW/d (herring+pout weighted) is")
print("  not a herring-only appetite target; a herring-only mechanism leaves the")
print("  pout's 0.35 share on its encounter and reserve room.")
