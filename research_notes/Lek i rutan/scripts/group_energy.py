"""Group-weighted reproductive energy and the plausibility comparison with start_fill.

Usage: python3 -I group_energy.py ../tables/species_inputs.csv

species_inputs.csv columns (one row per member species; every value from the literature files or
the DATRAS tables, never from model output):
  group, species, share (biomass share in the group), in_box (spawning weight 0-1: 1 spawns in the
  box, 0.5 builds gonads in the box but leaves / unresolved, 0 immature or no evidence),
  mature (mature share of the species' biomass in the box), G_lo, G_mid, G_hi (reproductive energy,
  MJ per t of MATURE fish per year, sex-averaged), res_lo, res_mid, res_hi (share of the gonad
  energy drawn from the stored reserve rather than concurrent food), cut (fractional intake
  reduction of mature fish around spawning), cut_days (its duration), fmr (field metabolism of
  the species, MJ/t/day, the deficit a full fast draws from the reserve), source.

Prints per group: the share of group biomass that spawns in the box, reproductive energy per t of
group per year (lo/mid/hi), in units of the group's ME, the reserve-funded part, and the
reserve drawdown a feeding cut adds (mature share x cut x fmr x days), all in fill units. The
comparison with start_fill is printed beside it; start_fill is a TEST, not an input.
"""
import csv
import sys

ME = {"benthivorous_fish": 1800.0, "whiting": 2000.0, "predatory_fish": 2000.0,
      "planktivorous_fish": 4990.0, "mackerel": 6980.0}
START_FILL = {
    "benthivorous_fish": [0.76, 0.76, 0.62, 0.47, 0.33, 0.33, 0.33, 0.57, 0.81, 0.81, 0.81, 0.79],
    "whiting": [0.69, 0.69, 0.56, 0.43, 0.3, 0.3, 0.3, 0.52, 0.73, 0.73, 0.73, 0.71],
    "predatory_fish": [0.71] * 12,
    "planktivorous_fish": [0.5, 0.36, 0.26, 0.2, 0.32, 0.81, 0.94, 0.96, 0.88, 0.88, 0.76, 0.68],
    "mackerel": None,
}

rows = list(csv.DictReader(open(sys.argv[1])))
groups = []
for r in rows:
    if r["group"] not in groups:
        groups.append(r["group"])
for g in groups:
    rs = [r for r in rows if r["group"] == g]
    tot = sum(float(r["share"]) for r in rs)
    spawn_share = sum(float(r["share"]) * float(r["in_box"]) * float(r["mature"]) for r in rs)
    E = {k: sum(float(r["share"]) * float(r["in_box"]) * float(r["mature"]) * float(r["G_" + k]) for r in rs)
         for k in ("lo", "mid", "hi")}
    R = {k: sum(float(r["share"]) * float(r["in_box"]) * float(r["mature"]) * float(r["G_" + k]) *
                float(r["res_" + k]) for r in rs) for k in ("lo", "mid", "hi")}
    F = sum(float(r["share"]) * float(r["in_box"]) * float(r["mature"]) * float(r["cut"]) *
            float(r["fmr"]) * float(r["cut_days"]) for r in rs)
    me = ME[g]
    print(f"===== {g} (member shares listed {tot:.3f}; ME {me:.0f} MJ/t)")
    for r in rs:
        m = float(r["share"]) * float(r["in_box"]) * float(r["mature"])
        print(f"  {r['species']:16s} share {float(r['share']):.3f} in_box {float(r['in_box']):.2f} mature "
              f"{float(r['mature']):.2f} -> spawning share of group {m:.3f}; G {r['G_lo']}-{r['G_mid']}-"
              f"{r['G_hi']} MJ/t mature; reserve share {r['res_lo']}-{r['res_mid']}-{r['res_hi']}; "
              f"cut {r['cut']} x {r['cut_days']} d at {r['fmr']} MJ/t/d")
    print(f"  share of group biomass that is mature and spawns in the box: {spawn_share:.3f}")
    print("  reproductive energy, MJ per t of group per yr (lo / mid / hi): "
          f"{E['lo']:.0f} / {E['mid']:.0f} / {E['hi']:.0f}  = {E['lo'] / me:.3f} / {E['mid'] / me:.3f} / "
          f"{E['hi'] / me:.3f} ME")
    print("  of it from the stored reserve: "
          f"{R['lo']:.0f} / {R['mid']:.0f} / {R['hi']:.0f} MJ/t = {R['lo'] / me:.3f} / {R['mid'] / me:.3f} / "
          f"{R['hi'] / me:.3f} ME")
    print(f"  feeding cut around spawning (fast-equivalent deficit): {F:.0f} MJ/t = {F / me:.3f} ME")
    print(f"  total reserve drawdown (reserve-funded gonad + cut): {(R['lo'] + F) / me:.3f} / "
          f"{(R['mid'] + F) / me:.3f} / {(R['hi'] + F) / me:.3f} ME")
    sf = START_FILL[g]
    if sf:
        hi_w = max(sf[0:3])
        lo_s = min(sf[3:8])
        print(f"  TEST start_fill: winter max {hi_w:.2f} -> spring/summer min {lo_s:.2f}: drop {hi_w - lo_s:.2f} ME; "
              f"monthly {sf}")
