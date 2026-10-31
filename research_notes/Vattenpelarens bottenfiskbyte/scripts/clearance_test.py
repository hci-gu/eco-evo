"""Clearance-ratio test of the rest water column's five checked pairs against field diets, and the
candidate corrections of the benthivore pair (no model output read; the budget gate's B_vis are the
library's inference-start densities and accesses).

Usage: python3 -I clearance_test.py <Arsungarnas_andel_dir> [H_saithe_day phi_bed]

Method (as research_notes/"Vitlingens fiskbyten"/scripts/clearance_test.py): under concurrent search
with one handling budget the diet share is D_i = a_i f_i B_i / sum_j a_j f_j B_j at any saturation, so
the FIELD clearance c_i = D_i / B_i (one common factor) and c_i / c_anchor is free of Q/B, feeding
level and saturation. It is compared with the model (a_i f_eff,i) / (a_anchor f_eff,anchor). Two
anchors: planktivores (same weight tables as the haddock share in every saithe diet; the pair's access
derived from field prey sizes, 228) and zooplankton (no stage access, but no saithe/krill overlap
factor either).

Field diets (shares of the five checked prey, tables/field_diets.json from field_diets.py):
  V1 booked M2 x B; V2 group with CRR 164 saithe (1980-82); V3 group with CRR 219 saithe (1991);
  S219 / S164 saithe alone; SMS observed stomachs give ben/(pout + herring) = 0.080 by mass (saithe).
Densities (t/km2, inference start): planktivores 3.33, zooplankton 9.72, benthivores 1.53, whiting
0.45 (budget gate: 0.427 visible of 0.45 B? the gate lists B 0.45), mackerel 0.85.

Model variants (a x f_eff, km2/t/tick):
  M0 library: plk 0.00559 [0.69, 0.27] at J/B 0.38; zoo 0.000241; ben 0.0651 [0.43, 0.027] at J/B
     0.15; whiting 0.013 [0.57, 0.15] at J/B 0.165; mackerel 0.00088 x f 0.29.
  M0t the same at the trained runs' benthivore J/B 0.05-0.08.
  M1 the stage accesses with the corrected bookkeeping ("aw": species by stage biomass, group-wide
     normalisation, member windows weighted by each length's share of the a; access_lib.py).
  M2 M1 + bed overlap on the benthivore pair only: x O_bed = phi_bed x H_juv / H_s (saithe's share of
     the pair a 0.77 takes it; the other members keep O 1 [J]).
  M3 M1 + the overlap convention on every pair of the saithe-dominated pairs (section "layers" below).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from access_lib import Access  # noqa: E402

AD = sys.argv[1]
H_S = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0      # saithe benthopelagic layer by day (m above bed)
PHI_BED = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5   # share of saithe foraging time in that layer
TAB = os.path.join(HERE, "..", "tables")
F = json.load(open(os.path.join(TAB, "field_diets.json")))
P = ["plk", "zoo", "ben", "wht", "mack"]
B = {"plk": 3.33, "zoo": 9.72, "ben": 1.53, "wht": 0.45, "mack": 0.85}
H = {"plk": 239.2, "zoo": 151.1, "ben": 151.1, "wht": 161.1, "mack": 265.1}
A_LIB = {"plk": 0.00559, "zoo": 0.000241, "ben": 0.0651, "wht": 0.013, "mack": 0.00088}
JB = {"plk": 0.38, "ben": 0.15, "wht": 0.165}
ACC = Access(AD)


def mean(x):
    return sum(x) / len(x)


def feff(pair, variant, jb):
    J, A = ACC.access(pair, variant)
    return jb * mean(J) + (1 - jb) * mean(A)


def model(variant="lib", jb_ben=0.15, over=None):
    f = {"plk": feff("plk", variant, JB["plk"]), "zoo": 1.0, "ben": feff("ben", variant, jb_ben),
         "wht": feff("wht", variant, JB["wht"]), "mack": 0.29}
    m = {k: A_LIB[k] * f[k] * (over[k] if over else 1.0) for k in P}
    return m, f


# ------------------------------------------------------------------ layers for M2 / M3 [J unless F]
# Heights above the bed (m) in the box (median depth 142 m, mean 167, 5-95 % 72-357; maps/depth.npz).
H_JUV = 2.0     # settled 0-1 group gadoids, 1-5 m (section 220 [J]; Bromley & Kell 1999 [A])
# saithe's share of each pair's a (split_encounter_mack_field_stdout.txt, 'layer' variant):
SAITHE_SHARE = {"plk": 0.84, "zoo": 0.86, "ben": 0.77, "wht": 0.77, "mack": 0.0}
# Inside the planktivore a of saithe, the prey types' shares (section 220 central, dispersed herring):
# herring + sprat ~0.88 (day dispersed 70 m layer, night upper 40 m), Norway pout ~0.08 (15 m above the
# bed), blue whiting ~0.04 (lowest 50 m on the slope) [D from predators_encounter.md 2.7].
PLK_TYPES = {"herring": 0.88, "pout": 0.08, "bluewhiting": 0.04}


def overlaps(h_s, phi_bed):
    """Fraction of saithe foraging time inside each prey layer, two-state saithe [J]:
    phi_bed of the time uniformly in the lowest h_s m above the bed (benthopelagic, mostly by day),
    1 - phi_bed in the pelagic water column above (mostly at night).
      juvenile gadoids (0-H_JUV m): phi_bed x H_JUV / h_s
      whiting (5-20 m above the bed): phi_bed x min(1, 15 / h_s)
      Norway pout (5-20 m): as whiting;  blue whiting (lowest 50 m): phi_bed x min(1, 50 / h_s)
      herring (pelagic by day 10-100 m depth, upper 30-50 m at night): 1 - phi_bed
      zooplankton (krill below ~70 m by day, upper 50-70 m at night; copepods upper 30-100 m): 1 - phi_bed
        + phi_bed x 0.5 (deep krill partly near the bed by day in a 100-300 m trench) [J]
      mackerel (upper 0-50 m): 1 - phi_bed
    Returns the per-pair factor applied to the saithe part of each pair's a (the other members keep 1)."""
    o_juv = phi_bed * H_JUV / h_s
    o_bp15 = phi_bed * min(1.0, 15.0 / h_s)
    o_bw = phi_bed * min(1.0, 50.0 / h_s)
    o_pel = 1.0 - phi_bed
    o_plk = PLK_TYPES["herring"] * o_pel + PLK_TYPES["pout"] * o_bp15 + PLK_TYPES["bluewhiting"] * o_bw
    o_zoo = o_pel + phi_bed * 0.5
    o = {"ben": o_juv, "wht": o_bp15, "plk": o_plk, "zoo": o_zoo, "mack": o_pel}
    return {k: SAITHE_SHARE[k] * o[k] + (1 - SAITHE_SHARE[k]) * 1.0 for k in P}, o


def shares(m, b=B):
    t = sum(m[k] * b[k] for k in P)
    return {k: m[k] * b[k] / t for k in P}


def rr(num, den):
    return f"{num / den:7.2f}" if den > 0 else "    n/a"


def fmt(d, nd=3):
    return " ".join(f"{k} {d[k]:.{nd}f}" for k in P)


FIELD = {"V1 booked": F["V1"], "V2 grp CRR164": F["V2"], "V3 grp CRR219": F["V3"],
         "S219 saithe": F["saithe219"], "S164 saithe": F["saithe164"]}
print("Field diets (shares of the five checked prey) and clearance c = D/B relative to planktivores | zooplankton:")
fc = {}
for n, d in FIELD.items():
    c = {k: d[k] / B[k] for k in P}
    fc[n] = c
    print(f"  {n:14s} {fmt(d)}")
    print(f"  {'':14s} c/c_plk " + " ".join(f"{k} {c[k] / c['plk']:.3f}" for k in P)
          + "  | c/c_zoo " + " ".join(f"{k} {c[k] / c['zoo']:.3f}" for k in P))
sms = F["sms_ben_over_poutherring"] * B["plk"] / B["ben"]
print(f"  SMS observed saithe: c_ben / c_plk = 0.080 x B_plk / B_ben = {sms:.3f} (pout + herring as the plk prey)")

over2, o2 = overlaps(H_S, PHI_BED)
over2 = {k: (over2[k] if k == "ben" else 1.0) for k in P}
over3, o3 = overlaps(H_S, PHI_BED)
MODELS = {
    "M0 library (ben J/B 0.15)": model("lib"),
    "M0t library, ben J/B 0.05": model("lib", 0.05),
    "M0t library, ben J/B 0.08": model("lib", 0.08),
    "M1 access bookkeeping (aw)": model("aw"),
    f"M2 M1 + bed overlap (H_s {H_S:g}, phi {PHI_BED:g})": model("aw", over=over2),
    f"M3 M1 + all-pair overlap (H_s {H_S:g}, phi {PHI_BED:g})": model("aw", over=over3),
}
print(f"\nSaithe layer overlaps used in M2/M3 (H_s {H_S:g} m, phi_bed {PHI_BED:g}): " + fmt(o3)
      + "\n  pair factors (saithe part only): " + fmt(over3))
print("\nModel a x f_eff and clearance relative to planktivores | zooplankton:")
for n, (m, f) in MODELS.items():
    print(f"  {n:44s} f_eff " + " ".join(f"{k} {f[k]:.4f}" for k in P))
    print(f"  {'':44s} rel plk " + " ".join(f"{k} {m[k] / m['plk']:.3f}" for k in P)
          + "  | rel zoo " + " ".join(f"{k} {m[k] / m['zoo']:.2f}" for k in P))
print("\nModel / field of the clearance ratio (1 = consistent; > 1 model too high). Anchor planktivores:")
for n, (m, f) in MODELS.items():
    print(f"  {n}")
    for fn, c in fc.items():
        print(f"     vs {fn:14s} " + " ".join(f"{k} {rr(m[k] / m['plk'], c[k] / c['plk'])}" for k in ("zoo", "ben", "wht", "mack")))
    print(f"     vs SMS ben/plk   ben {(m['ben'] / m['plk']) / sms:7.2f}")
print("Anchor zooplankton:")
for n, (m, f) in MODELS.items():
    print(f"  {n}")
    for fn, c in fc.items():
        print(f"     vs {fn:14s} " + " ".join(f"{k} {rr(m[k] / m['zoo'], c[k] / c['zoo'])}" for k in ("plk", "ben", "wht", "mack")))
print("\nConcurrent-search diet shares (all prey visible) vs field:")
for n, (m, f) in MODELS.items():
    print(f"  {n:44s} {fmt(shares(m))}")
for fn, d in FIELD.items():
    print(f"  field {fn:38s} {fmt(d)}")
print("\nSingle-pair return at today's density, x = a f h B; share of the ceiling x/(1+x) (the time-allocation"
      " check: no pair should pay far more per unit eat share than the others where the field diet is mixed).")
print("  planktivores also at the fish-pair visibility floor 0.25 (hidden herring), zooplankton at its floor 0.5:")
for n, (m, f) in MODELS.items():
    xs = {k: m[k] * H[k] * B[k] for k in P}
    xh = {"plk": m["plk"] * H["plk"] * B["plk"] * 0.25, "zoo": m["zoo"] * H["zoo"] * B["zoo"] * 0.5}
    print(f"  {n:44s} " + " ".join(f"{k} {xs[k] / (1 + xs[k]):.2f}" for k in P)
          + f"   | plk hidden {xh['plk'] / (1 + xh['plk']):.2f}, zoo floor {xh['zoo'] / (1 + xh['zoo']):.2f}")
print("\nFactor on the benthivore a x f_eff that reproduces the field share at fixed other pairs (concurrent search):")
for n, (m, f) in MODELS.items():
    rest = sum(m[k] * B[k] for k in P if k != "ben")
    out = []
    for fn, d in FIELD.items():
        need = d["ben"] / (1 - d["ben"]) * rest / B["ben"]
        out.append(f"{fn.split()[0]} {need / m['ben']:.3f}")
    print(f"  {n:44s} " + ", ".join(out))

# ------------------------------------------------------------------ quarterly test (saithe alone)
print("\nQuarterly test, saithe alone: clearance ratio c_ben / c_plk, model vs CRR 219 (1991).")
print("Model: saithe per-stage a (benthivores 10.6 x planktivores per tonne, 'Per-stage central a', both stages),")
print("saithe's own access by quarter x the compartment's J/B cycle (benthivores 0.15/0.14/0.17/0.16, planktivores")
print("0.37/0.35/0.40/0.39 for Feb/May/Aug/Nov, parameters_B.md 2); field: mass ratio x B_plk / B_ben.")
A_RATIO = (0.2588 * 0.08048 + 0.1506 * 0.03158) / (0.2588 * 0.007607 + 0.1506 * 0.002889)
JBQ = {"ben": [0.15, 0.14, 0.17, 0.16], "plk": [0.37, 0.35, 0.40, 0.39]}
print(f"  saithe a_ben / a_plk (stage-weighted) {A_RATIO:.2f}")
for variant in ("lib", "aw"):
    Jb, Ab = ACC.access("saithe_ben", variant)
    Jp, Ap = ACC.access("saithe_plk", variant)
    row = []
    for qi in range(4):
        fb = JBQ["ben"][qi] * Jb[qi] + (1 - JBQ["ben"][qi]) * Ab[qi]
        fp = JBQ["plk"][qi] * Jp[qi] + (1 - JBQ["plk"][qi]) * Ap[qi]
        row.append(A_RATIO * fb / fp)
    print(f"  model ({variant:3s}) Q1-Q4: " + " ".join(f"{x:.3f}" for x in row) + f"   Q1/Q3 {row[0] / row[2]:.2f}")
fq = F["saithe219_q"]
frow = [fq[str(q)]["ben"] / fq[str(q)]["plk"] * B["plk"] / B["ben"] for q in (1, 2, 3, 4)]
print("  field CRR 219  Q1-Q4: " + " ".join(f"{x:.3f}" for x in frow) + f"   Q1/Q3 {frow[0] / frow[2]:.2f}")

# ------------------------------------------------------------------ overlap grid
print("\nOverlap grid (M1 access): model/field clearance ratio of the benthivore pair, anchor planktivores, against")
print("V1 / V3 / S219, and the concurrent-search benthivore share (field 0.064-0.082). M2 = bed overlap only,")
print("M3 = all saithe-dominated pairs overlapped. phi_bed = share of saithe foraging time in the benthopelagic")
print("layer of thickness H_s above the bed:")
for h_s in (5.0, 10.0, 20.0, 30.0, 50.0):
    for phi in (0.3, 0.5, 0.7, 1.0):
        row = []
        for tag in ("M2", "M3"):
            ov, _ = overlaps(h_s, phi)
            if tag == "M2":
                ov = {k: (ov[k] if k == "ben" else 1.0) for k in P}
            m, _ = model("aw", over=ov)
            r = [(m["ben"] / m["plk"]) / (fc[n]["ben"] / fc[n]["plk"]) for n in ("V1 booked", "V3 grp CRR219", "S219 saithe")]
            row.append(f"{tag} ratio {r[0]:.2f}/{r[1]:.2f}/{r[2]:.2f} share {shares(m)['ben']:.3f}")
        print(f"  H_s {h_s:4.0f} phi {phi:.1f}: " + " | ".join(row))

# ------------------------------------------------------------------ prey hiding
# Every prey at its floor (pair floor where set: rest -> planktivores 0.25; zooplankton species floor 0.2;
# else the prey's species floor: benthivores 0.55, whiting 0.5, mackerel 0.7; fg_library.yaml).
FLOOR = {"plk": 0.25, "zoo": 0.2, "ben": 0.55, "wht": 0.5, "mack": 0.7}
print("\nConcurrent-search shares with every prey hidden at its visibility floor (the trained runs' herring hide):")
for n, (m, f) in MODELS.items():
    bh = {k: B[k] * FLOOR[k] for k in P}
    print(f"  {n:44s} {fmt(shares(m, bh))}   ben/plk clearance x{FLOOR['ben'] / FLOOR['plk']:.1f} vs all visible")
