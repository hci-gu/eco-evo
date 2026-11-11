"""Seal and porpoise visibility floors: the derived pieces (research_notes/"Sal och tumlare golv").

Usage:
    python3 -I mammal_floors.py <repo> > ../scripts/mammal_floors_stdout.txt

Reads (never writes) the repository's fgconfig/fg_library.yaml, maps/depth.npz and
research_notes/"Arsungarnas andel"/tables/stage_length_biomass.csv (the A-ring spelling on disk).
Nothing in the repository is changed. Every input that is not read from those files is a literature
number (evidence.md, E-numbers in the comments) or a judgement marked [J].

Parts
  1. Box bed depths (all cells; the east third, where the coast-anchored seals live).
  2. Can a harbour seal rest ON THE BED in the box? Resting dives 4.0-5.3 min (E3), vertical speed.
  3. Vertical reach of a cue released at the surface within one 6 h tick (diffusion length).
  4. Porpoise clicks as a cue for cod: received level vs the cod ultrasound threshold (E8-E10).
  5. Porpoise biosonar bed dead zone c*tau/2 (E11, E13).
  6. Species floors of the mammals (A): per-observer factor, two observer weightings.
  7. Pair floors on fish prey (B): member weights inside the mammal prey-size windows, per-member
     judgement floors, weighted value and range.
  8. What a change does at today's learned rest shares (Gom och sinnen table, fix97).
"""
import math
import os
import sys

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

repo = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()

lib = yaml.safe_load(open(os.path.join(repo, "fgconfig", "fg_library.yaml")))
inter = lib["interaction_definitions"]
spec = lib["species_definitions"]

# Start densities (t/km2) from AGENTS.md section 4 / the library notes (box means).
B = {"benthivorous_fish": 1.53, "whiting": 0.446, "predatory_fish": 0.617,
     "planktivorous_fish": 4.72, "mackerel": 0.85}

print("=" * 78)
print("1. Box bed depths (maps/depth.npz, 60 x 60 cells of 1 km2)")
depth = np.load(os.path.join(repo, "maps", "depth.npz"))["depth"].astype(float)
east = depth[:, 40:]
for name, d in (("all cells", depth), ("east third (cols 40-59)", east)):
    d = d[np.isfinite(d)]
    qs = np.percentile(d, [5, 25, 50, 75, 95])
    print(f"  {name:24s} n={d.size:4d} min {d.min():5.0f} m  p5/25/50/75/95 "
          + " / ".join(f"{q:4.0f}" for q in qs))
    for lim in (35, 50, 100, 150, 180):
        print(f"      share of cells with bed <= {lim:3d} m: {np.mean(d <= lim):.3f}")

print()
print("=" * 78)
print("2. Seal rest on the bed: deepest bed reachable with >= t_bot of bottom time")
print("   D_max = v (T_dive - t_bot) / 2; T_dive = mean resting-dive submersion 4.0-5.3 min (E3),")
print("   v = vertical speed: resting dives have slow descents (E2); 0.6-1.4 m/s spans slow")
print("   drift to the 1.4 m/s typical forward speed used in E2. t_bot >= 1 min [J].")
for T in (4.0, 5.3):
    for v in (0.6, 1.0, 1.4):
        Dmax = v * (T - 1.0) * 60.0 / 2.0
        share_all = np.mean(depth <= Dmax)
        share_east = np.mean(east <= Dmax)
        print(f"   T {T:.1f} min v {v:.1f} m/s -> D_max {Dmax:5.0f} m; cells with bed <= D_max: "
              f"all {share_all:.3f}, east {share_east:.3f}")
print("   Observed deepest bottom rest: ~35 m (E3); no box cell is that shallow.")

print()
print("=" * 78)
print("3. Vertical reach of a surface cue in one 6 h tick: L = sqrt(2 Kz t)")
t = 6 * 3600.0
for kz in (1e-5, 1e-4, 1e-3, 1e-2):
    print(f"   Kz {kz:7.0e} m2/s -> L {math.sqrt(2 * kz * t):6.1f} m")
print("   Kz 1e-5 to 1e-4 below a pycnocline, up to 1e-2 in a wind-mixed layer [J, textbook range].")
print("   A seal or porpoise resting at the surface leaves no cue at a 100-300 m bed within a tick")
print("   except by advection along isopycnals, which does not carry it downward [D].")

print()
print("=" * 78)
print("4. Porpoise click received level vs cod's conditioned ultrasound threshold")
print("   SL 178-205 dB re 1 uPa pp @1 m, mean 191 (E11); cod threshold 194.4 dB re 1 uPa")
print("   (38 kHz, conditioned, E9; cited as 203-204 dB pp by later work, E8); unconditioned")
print("   cod show no response at 204-213 dB (E8). Spherical spreading + absorption ~0.04 dB/m")
print("   at 130 kHz [J, textbook].")
for SL in (191.0, 205.0):
    for thr in (194.4, 203.0):
        # solve SL - 20 log10 R - a R = thr
        a = 0.04
        lo, hi = 0.01, 1000.0
        if SL - thr <= 0:
            print(f"   SL {SL:.0f}, threshold {thr:.1f}: never reached (SL below threshold)")
            continue
        for _ in range(100):
            mid = math.sqrt(lo * hi)
            rl = SL - 20 * math.log10(mid) - a * mid
            if rl > thr:
                lo = mid
            else:
                hi = mid
        print(f"   SL {SL:.0f}, threshold {thr:.1f}: detectable within {lo:5.2f} m (on axis)")
print("   Herring (Atlantic): 50 % startle threshold at 4 kHz, nothing above (E10); Pacific")
print("   herring no ultrasound response up to 185 dB (E12). At km scale the click is no cue.")

print()
print("=" * 78)
print("5. Porpoise biosonar bed dead zone: a target closer to the bed than c tau / 2 merges")
print("   with the bed echo (normal incidence). The beam footprint adds R (1 - cos(half-beam)):")
print("   porpoise -3 dB beam ~16 deg [S, Miller & Wahlberg 2013], half-beam 8 deg.")
for tau_us in (44, 77, 113, 175):
    print(f"   click duration {tau_us:3d} us -> {1500.0 * tau_us * 1e-6 / 2 * 100:4.1f} cm")
for R in (1.0, 2.0, 5.0):
    print(f"   beam term at range {R:.0f} m: +{R * (1 - math.cos(math.radians(8.0))) * 100:4.1f} cm")
print("   Dead zone: a few cm close in, up to ~1-2 dm at 5 m.")
print("   A cod, haddock or whiting lying on the bed is 5-15 cm high and carries a swimbladder")
print("   (90-95 % of the backscatter, E13): above the dead zone in the final approach (< 2 m),")
print("   at its edge from ~5 m. A flatfish has no swimbladder and, buried, lies inside it.")

print()
print("=" * 78)
print("6. Species floors of the mammals (A): observers and weights")


def flow(pred, prey):
    d = inter.get(f"{pred}_preys_on_{prey}", {})
    if not d.get("preys_on", False):
        return 0.0
    return float(d.get("predation_mortality", 0.0)) * B[prey]


observers = ["benthivorous_fish", "whiting", "predatory_fish", "planktivorous_fish", "mackerel"]
layer = {"benthivorous_fish": "bed", "whiting": "bed", "predatory_fish": "pelagic",
         "planktivorous_fish": "pelagic", "mackerel": "pelagic"}
for m in ("seals", "porpoises"):
    obs_m = [o for o in observers if inter.get(f"{o}_observes_{m}", {}).get("observes", False)]
    print(f"   {m}: species floor today {spec[m]['visibility_floor']}; observed by {obs_m}")
print("   Mammals also observe each other (porpoises <-> seals) at the other's species floor;")
print("   nobody eats a mammal, so the species floor touches observation only.")

# Per-observer factor f_o = share of a resting mammal's km-scale cue that still reaches observer o,
# relative to the same animal foraging. Central and range.
#  seals: rest is at the surface or on the bed (E2, E3); in the box bed rest is possible only where
#   the bed is shallow enough (part 2); foraging is benthic (E2, E4). For bed observers the factor is
#   r_b x P(bed <= D_max): r_b = bed share of at-sea rest. E3 (bed rest 8.1-12.4 % of at-sea time)
#   over E2 (all rest 14.3 % of at-sea time, Wadden Sea) gives 0.57-0.87; two studies [D].
#   For pelagic observers the resting seal is in their layer at night (herring rise) but 50-150 m
#   above them by day (herring and Norway pout deep or near the bed by day); odour stays in the
#   water (smell note) but does not cross that gap in a tick (part 3): factor 0.75 (0.5-1) [J].
#  porpoises: rest = shallow, quiet 'parabolic' dives and surface time (E5, E6); foraging is mostly
#   pelagic in a 100-300 m box (max dives 132-152 m in Skagerrak, porpoise note). Bed observers: a
#   resting porpoise is never at the bed, a foraging one only over beds < ~150 m [J, 0-0.1].
#   Pelagic observers: same layer at night, apart by day as for seals; odour persists; clicks
#   inaudible in both states (part 4): 0.75 (0.5-1) [J].
d_east = east[np.isfinite(east)]
p_rest_bed = {"low": float(np.mean(d_east <= 0.6 * 3.0 * 60 / 2)),   # T 4.0, v 0.6
              "high": float(np.mean(d_east <= 1.4 * 4.3 * 60 / 2))}  # T 5.3, v 1.4
f_bed_seal = (0.57 * p_rest_bed["low"], 0.87 * p_rest_bed["high"])
f_bed_seal_c = 0.72 * float(np.mean(d_east <= 1.0 * 3.65 * 60 / 2))
fac = {
    "seals": {"bed": (f_bed_seal_c, f_bed_seal[0], f_bed_seal[1]), "pelagic": (0.75, 0.5, 1.0)},
    "porpoises": {"bed": (0.05, 0.0, 0.1), "pelagic": (0.75, 0.5, 1.0)},
}
print("   P(bed rest feasible) in the east third: "
      f"{p_rest_bed['low']:.3f} (T 4.0, v 0.6) - {p_rest_bed['high']:.3f} (T 5.3, v 1.4)")
for m in ("seals", "porpoises"):
    print(f"\n   {m}: factor per observer layer (central, low, high): "
          f"bed {tuple(round(x, 3) for x in fac[m]['bed'])}, "
          f"pelagic {tuple(round(x, 3) for x in fac[m]['pelagic'])}")
    w_flow = {o: flow(m, o) for o in observers}
    w_bio = {o: B[o] for o in observers}
    for wname, w in (("booked flow M2 x B (t/km2/yr) of the mammal on o", w_flow),
                     ("observer biomass B (t/km2)", w_bio)):
        tot = sum(w.values())
        vals = []
        for k in range(3):
            vals.append(sum(w[o] * fac[m][layer[o]][k] for o in observers) / tot)
        shares = ", ".join(f"{o.split('_')[0]} {w[o] / tot:.2f}" for o in observers)
        print(f"     weights = {wname}: {shares}")
        print(f"       -> floor central {vals[0]:.2f} (range {min(vals[1:]):.2f}-{max(vals[1:]):.2f})")

print()
print("=" * 78)
print("7. Pair floors on fish prey (B): member weights inside the mammals' prey-size windows")
import csv  # noqa: E402

slb = os.path.join(repo, "research_notes", "Årsungarnas andel", "tables",
                   "stage_length_biomass.csv")
rows = [r for r in csv.DictReader(open(slb, encoding="utf-8")) if r["period"] == "2001-25"]


def window_share(sp, lo, hi):
    tot = inside = 0.0
    for r in rows:
        if r["species"] != sp:
            continue
        b = float(r["J"]) + float(r["A"])
        tot += b
        if lo <= int(r["L_cm"]) <= hi:
            inside += b
    return inside / tot if tot > 0 else float("nan")


# Benthivore members (Fiskgrupper i rutan, long-run biomass shares %).
ben_members = {
    "haddock": (38.62, "gadoid", "haddock"),
    "cod": (31.38, "gadoid", "cod"),
    "long rough dab": (12.63, "flatfish", None),
    "dab": (4.49, "flatfish", None),
    "plaice": (3.80, "flatfish", "plaice"),
    "witch": (1.67, "flatfish", "witch"),
    "lemon sole": (0.87, "flatfish", "plaice"),
    "flounder+sole+small flatfish": (0.44, "flatfish", "plaice"),
}
# Long rough dab and dab: no length table; Lmax ~ 30-40 cm, nearly all box biomass < 35 cm [J] -> 1.
windows = {"seals": (7, 35), "porpoises": (0, 40)}
# Per-member floor judgements (central, low, high) and what 'hide' means (summary.md section B).
floor_member = {
    "seals": {"gadoid": (0.85, 0.75, 0.95), "flatfish": (0.95, 0.85, 1.0)},
    "porpoises": {"gadoid": (0.9, 0.75, 1.0), "flatfish": (0.5, 0.3, 0.8)},
}
for m, (lo, hi) in windows.items():
    print(f"\n   {m} -> benthivores, window {lo}-{hi} cm")
    w = {}
    for name, (share, kind, proxy) in ben_members.items():
        ws = 1.0 if proxy is None else window_share(proxy, lo, hi)
        w[name] = (share * ws, kind, ws)
        print(f"     {name:30s} group share {share:5.2f} %  in window {ws:.3f}  -> {share * ws:6.2f}")
    tot = sum(v[0] for v in w.values())
    gad = sum(v[0] for v in w.values() if v[1] == "gadoid") / tot
    print(f"     inside the window: gadoids {gad:.3f}, flatfish {1 - gad:.3f} "
          "(rays, gurnard, hagfish and the rest, ~6.5 % of the group, left out)")
    vals = []
    for k in range(3):
        vals.append(gad * floor_member[m]["gadoid"][k] + (1 - gad) * floor_member[m]["flatfish"][k])
    print(f"     member floors gadoid {floor_member[m]['gadoid']}, flatfish {floor_member[m]['flatfish']}")
    print(f"     -> pair floor central {vals[0]:.2f} (range {vals[1]:.2f}-{vals[2]:.2f}); "
          f"library {inter[m + '_preys_on_benthivorous_fish']['visibility_floor']}")

# Water column: accessible composition from research_notes/"Tumlarens byten" tables/wc_window_stdout.txt
# (< 35 cm: saithe 0.70, hake 0.30; < 40 cm: saithe 0.73, hake 0.27).
wc = {"seals": {"saithe": 0.70, "hake": 0.30}, "porpoises": {"saithe": 0.73, "hake": 0.27}}
wc_floor = {
    "seals": {"saithe": (0.6, 0.35, 0.75), "hake": (0.85, 0.75, 0.95)},
    "porpoises": {"saithe": (0.95, 0.85, 1.0), "hake": (0.9, 0.75, 1.0)},
}
for m in ("seals", "porpoises"):
    vals = [sum(wc[m][s] * wc_floor[m][s][k] for s in wc[m]) for k in range(3)]
    print(f"\n   {m} -> water column: accessible part {wc[m]}; member floors {wc_floor[m]}")
    print(f"     -> pair floor central {vals[0]:.2f} (range {vals[1]:.2f}-{vals[2]:.2f}); "
          f"library {inter[m + '_preys_on_predatory_fish']['visibility_floor']}")
wh = {"seals": (0.85, 0.6, 0.95), "porpoises": (0.9, 0.75, 1.0)}
for m in ("seals", "porpoises"):
    print(f"\n   {m} -> whiting (one member): {wh[m]}; library "
          f"{inter[m + '_preys_on_whiting']['visibility_floor']}")

print()
print("=" * 78)
print("8. Effect at today's learned rest shares (fix97 annual means, Gom och sinnen section 4)")
rest = {"benthivorous_fish": 0.026, "whiting": 0.066, "predatory_fish": 0.032,
        "seals": 0.020, "porpoises": 0.016}
changes = [
    ("fish observe seals (species floor)", "seals", 0.0, 0.40),
    ("fish observe porpoises (species floor)", "porpoises", 0.0, 0.55),
    ("seals -> benthivores", "benthivorous_fish", 0.85, 0.90),
    ("seals -> water column", "predatory_fish", 0.75, 0.70),
    ("porpoises -> whiting", "whiting", 0.75, 0.90),
]
for name, hider, old, new in changes:
    r = rest[hider]
    vo, vn = 1 - r * (1 - old), 1 - r * (1 - new)
    print(f"   {name:40s} rest {r:.3f}: visible share {vo:.4f} -> {vn:.4f} "
          f"(x{vn / vo:.4f})")
print("   Annual means; hiding is seasonal and diel, so single ticks differ more.")
