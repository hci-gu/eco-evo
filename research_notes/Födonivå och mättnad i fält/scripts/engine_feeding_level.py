#!/usr/bin/env python3
"""Engine feeding level versus field feeding level, per demersal pair.

Research diagnostic only (no library, manifest or code change). Reads the
live library and manifest and prints, per checked fish pair:

  * the ceiling 1/h (t prey / t predator / tick) and Cmax in kJ/kg/d
    (section 148 energy convention: Cmax = E_prey / h per tick, x 4 per day
    at the library tick of 6 h; 1 MJ/t/tick = 4 kJ/kg/d);
  * the half-saturation in whole-pool units (1 + w B_pred) / (a h f);
  * the saturation index x = a h f N / (1 + w B_pred) at today's grid-mean
    density N and at 30, 10, 3 and 1 % of it, and the share of the ceiling
    a foraging fish gets there, f/(1/h) = x / (1 + x) (Type II) or the
    Type III analogue;
  * "minutes to fill": how long a foraging fish needs at today's density to
    collect one tick's ceiling, 360 min / x (Type II, prey not depleted).

Then the feeding level per group: the field ration / Cmax (from the notes
cited below) against the engine's eat x hunger (hunger = (1 - s)/(1 - u)
at the probe fills of resume 223; diagnostic values, NOT inputs to any
parameter).

Usage: python3 engine_feeding_level.py [LIBRARY] [MANIFEST]
"""
import sys
import os

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LIB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "fgconfig", "fg_library.yaml")
MAN = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "mareld2.yaml")

lib = yaml.safe_load(open(LIB))
man = yaml.safe_load(open(MAN))
spec = lib["species_definitions"]
inter = lib["interaction_definitions"]

# Grid area (1 km cells).
area = 60 * 60
start = {}


def walk(node):
    if isinstance(node, dict):
        if "group_id" in node and "inference_initial_biomass" in node:
            start[node["group_id"]] = float(node["inference_initial_biomass"]) / area
        for v in node.values():
            walk(v)
    elif isinstance(node, list):
        for v in node:
            walk(v)


walk(man)


def prey_energy(sp):
    e = float(sp.get("energy_content") or 0.0)
    if sp.get("prey_includes_reserve"):
        e += float(sp.get("reserve_reference_fill", 0.0)) * float(sp.get("max_energy_reserve", 0.0))
    return e


PREDS = ["benthivorous_fish", "predatory_fish", "planktivorous_fish"]
FRACS = [1.0, 0.3, 0.1, 0.03, 0.01]

print(f"library: {LIB}")
print("today's grid-mean densities (t/km2):",
      {k: round(v, 3) for k, v in start.items()
       if k in ("benthivorous_fish", "predatory_fish", "planktivorous_fish",
                "benthic_infauna", "mobile_epifauna", "zooplankton")})
print()
rows = []
for pred in PREDS:
    psp = spec[pred]
    w = float(psp.get("interference", 0.0))
    bpred = start.get(pred, 0.0)
    I = w * bpred
    for key, d in inter.items():
        if not key.startswith(pred + "_preys_on_") or not d.get("preys_on"):
            continue
        prey = key[len(pred + "_preys_on_"):]
        h = float(d.get("handling_time", 0.0))
        if h <= 0:
            continue
        a = float(d.get("max_intake_rate", psp.get("max_intake_rate", 0.04)))
        f = float(d.get("accessible_fraction", 1.0))
        typ = int(d.get("functional_response", 2))
        N = start.get(prey, 0.0)
        E = prey_energy(spec[prey])
        cmax_kj = E / h * 4.0  # kJ/kg/d
        if typ == 2:
            half = (1.0 + I) / (a * h * f)
        else:
            half = ((1.0 + I) / (a * h)) ** 0.5 / f
        shares = []
        for fr in FRACS:
            Nv = N * fr * f
            if typ == 2:
                x = a * h * Nv / (1.0 + I)
            else:
                x = a * h * Nv * Nv / (1.0 + I)
            shares.append((x, x / (1.0 + x)))
        x0 = shares[0][0]
        rows.append((pred, prey, a, h, f, typ, cmax_kj, half, N, shares, 360.0 / x0 if x0 > 0 else float("inf"),
                     d.get("dark_ratio")))

print("Per pair (visible = whole pool x accessible_fraction; hiding of DM prey ignored, i.e. upper bound):")
hdr = (f"{'pair':44s} {'a':>6s} {'h':>6s} {'f':>5s} T {'Cmax':>5s} {'half':>7s} {'N':>6s} "
       + " ".join(f"{'x@' + str(int(fr * 100)) + '%':>9s}" for fr in FRACS)
       + "   share of ceiling @100/10/1 %   min-to-fill")
print(hdr)
for (pred, prey, a, h, f, typ, cmax, half, N, shares, minfill, dark) in rows:
    xs = " ".join(f"{x:9.2f}" for x, _ in shares)
    sh = f"{shares[0][1]:.3f}/{shares[2][1]:.3f}/{shares[4][1]:.3f}"
    print(f"{pred + ' -> ' + prey:44s} {a:6.3f} {h:6.1f} {f:5.2f} {typ} {cmax:5.0f} {half:7.3f} {N:6.2f} {xs}   {sh:>22s}   {minfill:8.1f}")
print()
print("Cmax column is kJ per kg predator per day on that prey (energy ceiling x 4).")
print("half = half-saturation in t/km2 of the WHOLE pool (accessible share and interference included).")
print("min-to-fill = minutes of foraging a fish needs at today's density to collect one 6-h tick's ceiling.")
print()

# Light multiplier: does a dark ratio bind on a saturated pair?
print("Night ticks: share of the ceiling at today's density if a is multiplied by the dark ratio")
for (pred, prey, a, h, f, typ, cmax, half, N, shares, minfill, dark) in rows:
    if dark is None:
        continue
    x0 = shares[0][0]
    xd = x0 * float(dark)
    print(f"  {pred} -> {prey}: dark_ratio {dark}: x {x0:.2f} -> {xd:.3f}, share {x0 / (1 + x0):.3f} -> {xd / (1 + xd):.3f}")
print("  (pairs without dark_ratio, e.g. the benthic pairs, have no night term at all)")
print()

# Feeding level, field vs engine.
print("Feeding level = consumption / Cmax")
u = {p: float(spec[p].get("maintenance_level", 0.0)) for p in PREDS}


def hunger(s, uu):
    return min(1.0, max(0.0, (1.0 - s) / (1.0 - uu)))


# Engine probe values (resume 223.3, 223.7, 223.8; diagnostic only).
probe = {
    "benthivorous_fish": dict(fill=0.72, eat=0.96, note="fix76 gen 9/10 year 1: fill 0.72, eat 0.957 (223.8)"),
    "predatory_fish": dict(fill=None, eat=None, measured=(0.43, 0.50),
                           note="share of the Holling ceiling eaten 0.43-0.50 (223.7 instrumented run)"),
}
for p, d in probe.items():
    if d.get("fill") is not None:
        hg = hunger(d["fill"], u[p])
        print(f"  engine {p}: hunger at fill {d['fill']} (u {u[p]}) = {hg:.2f}; x eat {d['eat']} = {hg * d['eat']:.2f} of the ceiling ({d['note']})")
    else:
        print(f"  engine {p}: {d['measured'][0]:.2f}-{d['measured'][1]:.2f} of the ceiling ({d['note']})")

# Field: group Cmax (kJ/kg/d) and field rations (notes: Fiskgrupper i rutan,
# Bentivorernas energibudget, Bentivorernas energi och diet).
field = [
    # group, Cmax kJ/kg/d (range), Q/B /yr, diet-mean energy MJ/t as eaten, label
    ("benthivorous_fish", 108.0, (82.0, 133.0), 3.48, 3440.0, "published Q/B 3.48 (2.5-4.7) on the old mixed diet (3440 MJ/t)"),
    ("benthivorous_fish", 108.0, (82.0, 133.0), 3.6, 3440.0, "SMS cod + haddock 3.6 (WGSAM 2024)"),
    ("benthivorous_fish", 108.0, (82.0, 133.0), 4.46, 2472.0 / 0.746, "energy-budget Q/B 4.46 on the 223.10 diet (2472 MJ/t assimilated / mean AE 0.746 = 3314 MJ/t as eaten)"),
    ("predatory_fish", 102.0, (78.0, 154.0), 3.34, 4180.0, "SMS 2023 / Ecopath rations 3.34 /yr at 4.18 kJ/g"),
]
for g, cm, (lo, hi), qb, e, label in field:
    kj = qb * e / 365.0  # MJ/t/yr -> kJ/kg/d (1 MJ/t = 1 kJ/kg)
    print(f"  field {g}: {label}: {kj:.0f} kJ/kg/d = {kj / cm:.2f} of Cmax {cm:.0f} ({kj / hi:.2f}-{kj / lo:.2f} over the Cmax range)")
print()

# Engine Q/B at today's ceiling and level, versus the budget.
print("Engine mass Q/B implied by level x ceiling on a single prey (per yr):")
for (pred, prey, a, h, f, typ, cmax, half, N, shares, minfill, dark) in rows:
    print(f"  {pred} -> {prey}: ceiling {1460.0 / h:6.2f} /yr; x0.54 = {0.54 * 1460.0 / h:5.2f}; x0.30 = {0.30 * 1460.0 / h:5.2f}")
