"""Field / chain factor on a, anchor by anchor, against TODAY's library (fgconfig/fg_library.yaml).
Reads only the library, mareld2.yaml and maps/depth.npz; no model run, no model output, no FIELD_QB.

    python3 anchors.py > ../tables/anchors_stdout.txt

Factor = a implied by a field whole-product measurement / the chain's a under the anchor's conditions.
Engine conventions (section 220/245): a per t predator per (t/km2 prey) per 6-h tick, Type II,
I = a B / (1 + a h B); ingestion_q10 scales a and 1/h together (annual mean 1); the daylight lever m(t)
scales a (annual mean 1); depth_access_table per bed class (equal-cell mean 1); stage_access scales the
prey biomass (a is per t/km2 of ACCESSIBLE prey).

A. Krill pairs (stomach-based): a_field = R / (D (1 - h_eff R)), R the field krill ration t/t/tick,
   D the krill density the fish meet. D is NOT measured at any site: the self-consistent reading uses the
   library's own krill target in that class and month (5 t/km2 x quarterly season x depth weight), so the
   factor is then the factor on the PRODUCT a x D (it absorbs any error in the library's krill density);
   brackets 3 and 20 t/km2 as in "Krillparen mot vintermagar".
B. Whiting on fish (Rindorf & Gislason 2005, stomach series + trawl density, North Sea, September):
   field initial slope = plateau K x Wmax / half-saturation (in t/km2 of vulnerable prey), against the
   library a (per t/km2 of accessible prey) x the September light and temperature multipliers.
C. Planktivores on zooplankton (clearance-based): Arrhenius & Hansson 1999 (YOY herring) and Kulke 2018
   (field sprat) against the 243.3 encounter chain ("Kolmulens biomassa"/zoo, the library's 0.0122).
"""
import math, os, sys
import numpy as np, yaml
ROOT = "/home/simon/eco-evo"
sys.path.insert(0, ROOT)
from lib.environments.ecosystem_env import open_pool  # noqa: E402
from lib.world import habitat  # noqa: E402

lib = yaml.safe_load(open(os.path.join(ROOT, "fgconfig/fg_library.yaml")))
man = yaml.safe_load(open(os.path.join(ROOT, "mareld2.yaml")))
SP, PAIRS = lib["species_definitions"], lib["interaction_definitions"]
depth = np.load(os.path.join(ROOT, "maps/depth.npz"))["depth"].astype(np.float64)
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
layers = man["simulation_settings"]["temperature"]["layers"]
glayer = man["simulation_settings"]["temperature"]["group_layers"]
# monthly daylight multipliers m(t) of the krill pairs (tools/probes/daylight_table.py, as used in
# "Krillparen mot vintermagar"/scripts/model_side.py; planktivores -> krill dark_ratio 0.57 unchanged since)
M_PLK_KRILL = [0.88, 0.93, 0.99, 1.05, 1.10, 1.14, 1.12, 1.07, 1.01, 0.95, 0.89, 0.86]
M_WHI_KRILL = [1.0] * 12
CLS = [("<100", depth < 100), ("100-150", (depth >= 100) & (depth < 150)),
       ("150-250", (depth >= 150) & (depth < 250)), (">=250", depth >= 250)]


def qmult(g):
    q10 = float(SP[g]["ingestion_q10"])
    T = np.array(layers[glayer[g]], dtype=float)
    raw = q10 ** (T / 10.0)
    return raw / raw.mean()


krill = SP["krill"]
wmap = open_pool.depth_weight_map(open_pool.parse_depth_table(krill), depth, depth.shape)
season = krill["outside_season"]
print("=== A. Krill pairs: field-implied a against today's a_eff (class 100-150 m unless stated)")
print("    factor = a_field / a_eff; D_lib = library krill target in the class and month")
KRILL_ROWS = {
    # (label, month index, R %BW/d (low, central, high), class, tag, D brackets)
    "whiting": [
        ("NE+NW North Sea Q1 1981, 25-29 cm (ICES DB counts, GER upper)", 0, (0.34, 0.45, 0.52), "100-150", "D on F"),
        ("Skagerrak Q1 1981, 21 cm (ICES DB counts, GER upper)", 0, (0.11, 0.14, 0.16), "100-150", "D on F"),
        ("Skagerrak Apr 1991 (0.30 of the model ceiling)", 3, None, "100-150", "D on F"),
        ("Oslofjord Nov (Onsrud 2004, 0.09-0.22 of ceiling)", 10, None, "100-150", "D on F, fjord"),
        ("Oslofjord Mar (Onsrud 2004, 0.26-0.60 of ceiling)", 2, None, "100-150", "D on F, fjord"),
        ("Oslofjord Jun (Onsrud 2004, 0.08-0.18 of ceiling)", 5, None, "100-150", "D on F, fjord"),
    ],
    "planktivorous_fish": [
        ("group Nov-Mar central (herring 0.554, pout 0.35)", 0, (0.10, 0.24, 0.46), "100-150", "D/J"),
        ("pout NE North Sea 1980 (ICES DB, GER)", 0, (0.20, 0.20, 0.20), "100-150", "D on F"),
        ("pout Oslofjord Mar (Onsrud, GER)", 2, (0.51, 0.75, 1.14), "100-150", "D on F, fjord"),
        ("herring Oslofjord Nov-Mar (Onsrud)", 0, (0.03, 0.15, 0.53), "100-150", "D on F, fjord"),
    ],
}
CEIL_SHARE = {"Skagerrak Apr 1991 (0.30 of the model ceiling)": (0.25, 0.30, 0.35),
              "Oslofjord Nov (Onsrud 2004, 0.09-0.22 of ceiling)": (0.09, 0.15, 0.22),
              "Oslofjord Mar (Onsrud 2004, 0.26-0.60 of ceiling)": (0.26, 0.40, 0.60),
              "Oslofjord Jun (Onsrud 2004, 0.08-0.18 of ceiling)": (0.08, 0.12, 0.18)}
krill_factor = {}
for g, pair, mlist in (("whiting", "whiting_preys_on_krill", M_WHI_KRILL),
                       ("planktivorous_fish", "planktivorous_fish_preys_on_krill", M_PLK_KRILL)):
    P = PAIRS[pair]
    a, h = float(P["max_intake_rate"]), float(P["handling_time"])
    fmap = habitat.pair_depth_map(P, depth, pair)
    q = qmult(g)
    print(f"\n  {pair}: library a {a}, h {h}")
    for lab, mi, Rr, cls, tag in KRILL_ROWS[g]:
        msk = dict(CLS)[cls]
        fc = float(fmap[msk].mean())
        aeff = a * fc * mlist[mi] * q[mi]
        heff = h / q[mi]
        Dlib = 5.0 * season[mi] * float(wmap[msk].mean())
        if Rr is None:
            Rr = tuple(s * 400.0 / heff for s in CEIL_SHARE[lab])
        fac = {}
        for D in (Dlib, 3.0, 8.0, 20.0):
            fac[D] = [(R / 400.0) / (D * (1 - heff * R / 400.0)) / aeff for R in Rr]
        krill_factor.setdefault(g, []).append((lab, fac[Dlib][1], fac[Dlib][0], fac[Dlib][2], "fjord" in tag))
        print(f"    {lab:62s} {MON[mi]} R {Rr[0]:.2f}/{Rr[1]:.2f}/{Rr[2]:.2f} %BW/d; a_eff {aeff:.4f} "
              f"(x{fc * mlist[mi] * q[mi]:.2f}); D_lib {Dlib:.1f}")
        print(f"      factor at D_lib: {fac[Dlib][1]:.4f} ({fac[Dlib][0]:.4f}-{fac[Dlib][2]:.4f});  "
              f"at D 3: {fac[3.0][1]:.4f}; D 8: {fac[8.0][1]:.4f}; D 20: {fac[20.0][1]:.4f}   [{tag}]")
for g, rows in krill_factor.items():
    shelf = [r[1] for r in rows if not r[4]]
    allr = [r[1] for r in rows]
    gm = lambda v: float(np.exp(np.mean(np.log(v))))
    print(f"  {g}: geometric mean of central factors, shelf rows {gm(shelf):.4f} (n {len(shelf)}), "
          f"all rows {gm(allr):.4f}; span of centrals {min(allr):.4f}-{max(allr):.4f}")

# ------------------------------------------------------------------------------------------------
print("\n=== B. Whiting on fish prey: Rindorf & Gislason 2005 (North Sea, Sep 1988-89, 20-35 cm whiting)")
print("    field initial slope = K x Wmax / half-saturation; half-saturation t/km2 = q rho x w / (A x c)")
WMAX = (2.7, 3.3, 6.1)               # %BW/d, Table 1 range and mean [F]
rows_b = {
    # prey, K (plateau share of Wmax), q rho (catch/haul) central and corners, w g, c, library pair
    "clupeids (Type II, 25-35 cm)": dict(K=(0.10, 0.135, 0.17), qr=(59, 75, 95), w=15.0, c=(0.05, 0.2, 1.0),
                                         pair="whiting_preys_on_planktivorous_fish"),
    "gadoids (Type II, 25-35 cm)": dict(K=(0.063, 0.09, 0.12), qr=(965, 1675, 2907), w=5.0, c=(0.3, 0.5, 1.0),
                                        pair="whiting_preys_on_benthivorous_fish"),
}
A_HAUL = (0.05, 0.09, 0.18)          # km2 per haul (J; gear not obtained)
rindorf = {}
for lab, r in rows_b.items():
    P = PAIRS[r["pair"]]
    a_lib = float(P["max_intake_rate"])
    fmap = habitat.pair_depth_map(P, depth, r["pair"])
    shallow = float(fmap[depth < 100].mean())
    mid = float(fmap[(depth >= 100) & (depth < 150)].mean())
    q = qmult("whiting")[8]
    half_c = r["qr"][1] * r["w"] / (A_HAUL[1] * r["c"][1]) / 1e6
    half_lo = r["qr"][0] * r["w"] / (A_HAUL[2] * r["c"][2]) / 1e6   # smallest half-saturation
    half_hi = r["qr"][2] * r["w"] / (A_HAUL[0] * r["c"][0]) / 1e6   # largest
    slope = lambda K, W, H: K * W / 100.0 / 4.0 / H                 # t/t/tick per t/km2
    s_c = slope(r["K"][1], WMAX[1], half_c)
    s_hi = slope(r["K"][2], WMAX[2], half_lo)
    s_lo = slope(r["K"][0], WMAX[0], half_hi)
    # the engine's Sep a: library a x September daylight (~1, annual mean) x Sep q; Rindorf's beds:
    # four of five 44-73 m (< 100 m class), one 137 m
    chain_eq = a_lib * q
    chain_bed = a_lib * q * (0.8 * shallow + 0.2 * mid)
    rindorf[r["pair"]] = (s_c / chain_eq, s_c / chain_bed, s_lo / chain_bed, s_hi / chain_eq)
    print(f"  {lab}: half-saturation {half_c:.3f} t/km2 ({half_lo:.4f}-{half_hi:.2f}); field slope "
          f"{s_c:.4f} ({s_lo:.5f}-{s_hi:.3f}) km2/t/tick")
    print(f"    library {r['pair']} a {a_lib} x Sep q {q:.2f}; bed classes <100 x{shallow:.2f}, 100-150 x{mid:.2f}")
    print(f"    factor vs equal-cell a: {s_c / chain_eq:.3f}; vs Rindorf's bed mix (0.8 <100, 0.2 100-150): "
          f"{s_c / chain_bed:.3f}; corner span {s_lo / chain_bed:.4f}-{s_hi / chain_eq:.2f}   [D on F, 3 J]")
    # the old reading (half-saturation only, engine h): what a the field half-saturation implies at h
    h = float(P["handling_time"])
    print(f"    (half-saturation reading at today's h {h}: a = 1/(half x h) = {1 / (half_c * h):.4f}, "
          f"x{1 / (half_c * h) / chain_eq:.2f} of the library - keeps a, puts the field plateau in a ceiling)")

# ------------------------------------------------------------------------------------------------
print("\n=== C. Planktivores on zooplankton (clearance-based), against the 243.3 chain (library 0.0122)")
# chain numbers from "Kolmulens biomassa"/zoo/planktivores_a_pout164_foraging_stdout.txt (size-free R,
# k_v 115 herring / 100 sprat, f 0.5, foraging-fish overlap O 1): herring day a 0.0400 per t herring,
# sprat day a 0.0569; group day 0.0236, annual 0.0122; herring 0.928 of the group a.
# V1 Arrhenius & Hansson 1999 (YOY 4-8 cm, Baltic, Jul-Nov, daylight, small copepods): 46 L/g/h.
v1_free = (437, 864, 2258)      # chain L/g/h at 8 / 6 / 4 cm, size-free R (as in the library)
v1_prop = (70, 78, 90)          # R proportional to length, 20 cm anchor
print("  V1 herring YOY: field 46 L/g/h (daylight initial slope; WSC from stomachs vs Z 0-30 m)")
print(f"    vs chain, size-free R (the library's convention): factor {46 / v1_free[1]:.3f} "
      f"({46 / v1_free[2]:.3f}-{46 / v1_free[0]:.3f})")
print(f"    vs chain, R proportional to L: factor {46 / v1_prop[1]:.2f} ({46 / v1_prop[2]:.2f}-{46 / v1_prop[0]:.2f})")
print("    transfer to the box adults [J]: size-free -> the YOY factor applies as is; R ~ L -> the chain is"
      " x0.78 at the box size structure (group 0.0096 vs 0.0122), so the library factor is"
      f" {0.78 * 46 / v1_prop[2]:.2f}-{0.78 * 46 / v1_prop[0]:.2f}")
# V2 Kulke 2018 field sprat 12 cm, prey ~32 ug WW (small copepods, cladocerans): 0.5-1.0 m3/h per fish
L, W = 0.12, 12.5           # m, g (sprat 12 cm, survey L-W fits 11.8-19.8 g at 12-14 cm; J)
kv, v_bl, p, f = 100.0, 0.6, 0.7, 0.5
A_SMALL = 0.75 * 1.0e-3 * 0.4e-3
R = kv * math.sqrt(A_SMALL)
chain_m3h = f * math.pi * R ** 2 * v_bl * L * p * 3600
for lab, fld in (("low", 0.5), ("high", 1.0)):
    print(f"  V2 sprat 12 cm: field {fld} m3/h per fish vs chain {chain_m3h:.2f} m3/h (R {100 * R:.1f} cm, f 0.5, "
          f"v 0.6 BL/s, p 0.7): factor {fld / chain_m3h:.2f}")
print("    (f 0.5 was chosen with this datum (planktivores_encounter V2), so V2 is not independent of the"
      " chain's scan term)")
print("  Engine temperature: the library a is at the annual-mean temperature and the engine multiplies it by"
      " q(T); planktivore q Jul-Nov 1.07-1.36 (summer ~1.3): at the anchors' summer temperatures the factor"
      " on the library a is x0.75-0.85 lower [D on J]")
