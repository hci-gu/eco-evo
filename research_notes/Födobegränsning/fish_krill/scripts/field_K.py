"""Field half-saturations for fish -> fish and fish -> krill/zooplankton pairs,
converted to the engine's units, and compared with the library pairs.

Research only. Every input is a field/lab value or a stated judgement range;
nothing is taken from model output. Model values are copied from the task
brief (library 2026-11-07) for comparison only.

Units: densities t/km2 wet, box mean; a in km2 per t of predator per 6 h tick;
h in ticks (E_prey / ceiling).

Run: python3 field_K.py   (no arguments, no file input)
"""
import math
import itertools

print("== 1. Rindorf & Gislason 2005, whiting on fish prey (North Sea, Sept) ==")
# [F] Table 2, Type II fits for 25-35 cm whiting: half-saturation q*rho in
# prey caught per trawl hour; plateau K_R as a share of Wmax.
# [F] Rindorf 2002 (via search abstract): EXPO bottom trawl, 1 h at 3.5 kn.
# [J] wing spread 15-25 m (EXPO not obtained); catch efficiency c of a
# bottom trawl for clupeids in a 44-137 m column 0.05-1 (central 0.2), for
# demersal juvenile gadoids 0.3-1 (central 0.5).
tow_km = 3.5 * 1.852 * 1.0
wing = {"low": 0.015, "central": 0.020, "high": 0.025}  # km
A = {k: tow_km * v for k, v in wing.items()}
print("  swept area per trawl hour km2:", {k: round(v, 3) for k, v in A.items()})

def K_t_km2(qrho, w_g, area, c):
    return qrho * w_g * 1e-6 / (area * c)

cases = {
    # name: (qrho low, central, high), (w_g low, central, high), (c low, central, high), plateau share of Wmax
    "clupeids (herring+sprat <= 14 cm)": ((59, 75, 95), (10, 15, 20), (0.05, 0.2, 1.0), (0.10, 0.17)),
    "gadoids (whiting, haddock, pout <= 11 cm)": ((965, 1675, 2907), (3, 5, 8), (0.3, 0.5, 1.0), (0.06, 0.12)),
}
field = {}
for name, (q, w, c, plat) in cases.items():
    central = K_t_km2(q[1], w[1], A["central"], c[1])
    lo = K_t_km2(q[0], w[0], A["high"], c[2])
    hi = K_t_km2(q[2], w[2], A["low"], c[0])
    field[name] = (lo, central, hi)
    print(f"  {name}: K = {central:.3f} t/km2 of vulnerable prey ({lo:.4f} - {hi:.3f}); plateau {plat[0]}-{plat[1]} of Wmax")

print()
print("== 2. Engine pairs at today's densities (visible share from stage_access, J/B 0.417) ==")
jb = 0.417
B_plank = 4.72
def vis(fJ, fA):
    return fJ * jb + fA * (1 - jb)

pairs = [
    # name, a, h, B_visible, field K (lo, c, hi) or None, note
    ("whiting -> planktivores", 0.0212, 207, B_plank * vis(0.18, 0.0073), field["clupeids (herring+sprat <= 14 cm)"], "Rindorf clupeid fit"),
    ("water column -> planktivores", 0.0060, 6487 / 29.79, B_plank * vis(0.65, 0.18), None, "no local fit; stock-scale growth link (Cormon 2016)"),
    ("benthivores -> planktivores", 0.0198, 238, B_plank * vis(0.62, 0.18), (0.11, 0.6, 1.57), "cod-capelin proxy: Fall 2021 Holling fit 0.11 / end of main rise 0.6 (2 km), MULTSPEC 1.57 (~100 km)"),
    ("mackerel -> planktivores", 0.0266, 6487 / 66.22, B_plank * vis(0.020, 0.0029), None, "no field fit"),
    ("whiting -> benthivores (bonus)", 0.21, 4500 / 31.34, None, field["gadoids (whiting, haddock, pout <= 11 cm)"], "Rindorf gadoid fit"),
    ("whiting -> krill", 0.043, 159.5, 5.0, None, "no field response"),
    ("water column -> krill", 0.0241, 167.8, 5.0, None, "no field response"),
    ("planktivores -> krill", 0.018, 57.8, 5.0, None, "no field response"),
    ("benthivores -> krill", 0.013, 182.7, 5.0, None, "no field response"),
    ("mackerel -> krill", 0.0025, 75.5, 5.0, None, "no field response"),
    ("planktivores -> zooplankton", 0.0122, 52, 9.7, None, "no field K; Type II, floor 0.5"),
    ("mackerel -> zooplankton", 0.00084, 68.0, 9.7, None, "a from Macy 1998 clearance (already)"),
]
print(f"  {'pair':32s} {'a':>8s} {'h':>6s} {'K_mod':>7s} {'B_vis':>6s} {'x_mod':>6s} {'sat':>5s} | field K (lo/c/hi) -> a_impl c (lo-hi), model/field a")
for name, a, h, Bv, fK, note in pairs:
    K = 1.0 / (a * h)
    line = f"  {name:32s} {a:8.5f} {h:6.1f} {K:7.3f} "
    if Bv is not None:
        x = Bv / K
        line += f"{Bv:6.2f} {x:6.2f} {x/(1+x):5.2f} | "
    else:
        line += f"{'-':>6s} {'-':>6s} {'-':>5s} | "
    if fK is not None:
        ai = [1.0 / (k * h) for k in fK]
        line += f"{fK[0]:.3f}/{fK[1]:.3f}/{fK[2]:.3f} -> {ai[1]:.4f} ({ai[2]:.4f}-{ai[0]:.3f}), ratio {a/ai[1]:.2f} ({a/ai[0]:.2f}-{a/ai[2]:.2f})"
        if Bv is not None:
            xf = [Bv / k for k in fK]
            line += f"; x_field {xf[1]:.2f} ({xf[2]:.2f}-{xf[0]:.1f})"
    line += f"  [{note}]"
    print(line)

print()
print("  Reading of the Rindorf K under two placements of the plateau share tau:")
print("   A) tau on the whole response (time/overlap window = eat share): I = tau a N/(1 + a h N) -> a = 1/(K h)")
print("   B) tau on 1/h only (capture-limited ceiling h/tau): I = a N/(1 + a (h/tau) N) -> a = tau/(K h)")
for name, h, a_mod, key, taus in [
    ("whiting -> planktivores", 207, 0.0212, "clupeids (herring+sprat <= 14 cm)", (0.10, 0.17)),
    ("whiting -> benthivores", 4500 / 31.34, 0.21, "gadoids (whiting, haddock, pout <= 11 cm)", (0.06, 0.12)),
]:
    lo, c, hi = field[key]
    aA = 1 / (c * h)
    aB = [t / (c * h) for t in taus]
    print(f"   {name}: A a = {aA:.4f} (model/field {a_mod/aA:.2f}); B a = {aB[0]:.4f}-{aB[1]:.4f} (model/field {a_mod/aB[1]:.2f}-{a_mod/aB[0]:.2f})")

print()
print("== 3. Lauerburg et al. 2018: stock-scale elasticity of whiting intake to forage fish ==")
# [F] forage fish 2000-07 vs 2008-14: IBTS 285 -> 353 N/h; SMS recruits 200 -> 382 million.
# [F] growth-implied stomach content 2008-14 vs 2000-07: +16.8 % (F), +12.9 % (M);
#     temperature-corrected +7.6 % (F), +3.9 % (M).
dN = [math.log(353 / 285), math.log(382 / 200)]
for label, dC in [("uncorrected", (0.129, 0.168)), ("T-corrected", (0.039, 0.076))]:
    el = [math.log(1 + c) / n for c in dC for n in dN]
    lo, hi = min(el), max(el)
    # Holling II elasticity of intake to prey = 1/(1 + x), x = N/K
    print(f"  {label}: elasticity {lo:.2f}-{hi:.2f} -> x = N/K {1/hi - 1:.1f}-{1/lo - 1:.1f}, saturation {1 - hi:.2f}-{1 - lo:.2f}")

print()
print("== 4. Fall et al. 2021: Barents cod on capelin at 2 km (Fig. 4a, 5 digitised by eye) ==")
# capelin density in ind/km2 and partial effect on the log link (deviation from mean)
pts = [(5e3, -0.60), (1e4, -0.40), (2e4, -0.25), (5e4, -0.12), (1e5, -0.05), (1.8e5, 0.0),
       (1e7, 0.30), (2e7, 0.36), (4e7, 0.42), (5.5e7, 0.45)]
best = None
for K in [10 ** (k / 20) for k in range(60, 160)]:
    # log C = c0 + log(N/(K+N)); choose c0 by least squares
    r = [y - math.log(n / (K + n)) for n, y in pts]
    c0 = sum(r) / len(r)
    sse = sum((ri - c0) ** 2 for ri in r)
    if best is None or sse < best[0]:
        best = (sse, K, c0)
sse, K, c0 = best
print(f"  best Holling II on all points: K = {K:.3g} ind/km2, SSE {sse:.3f} (poor: the field curve is log-like)")
# The curve rises 0.6 log units over 5e3-1.8e5 and 0.45 over 1.8e5-5.5e7: far flatter than Holling.
for w in (8, 12, 20):
    print(f"   capelin {w} g: median 1.8e5 ind/km2 = {1.8e5*w*1e-6:.2f} t/km2; 5e4 (end of main rise) = {5e4*w*1e-6:.2f}; fitted K = {K*w*1e-6:.2f} t/km2")
print("  share of the far-right (300x median) intake reached at the median: exp(-0.45) =", round(math.exp(-0.45), 2))

print()
print("== 5. WKCAPELIN 2023: Barents cod on maturing capelin, Jan-Mar, Type III ==")
# [F] levels out at ~500 000 t of capelin. [J] area of maturing capelin overlapping immature cod 5e4 - 2e5 km2.
for area in (5e4, 1e5, 2e5):
    print(f"   area {area:.0e} km2: levelling density {5e5/area:.1f} t/km2")

print()
print("== 6. Zooplankton concentrations vs lab feeding modes ==")
# Model pool 9.7 t/km2 over 0-25 m (A17 WP2).  [J] large Calanus C5-C6 300 ug DW, DW/WW 0.2 (0.15-0.25)
conc = 9.7 / 25  # g WW m-3
for dw_ww in (0.15, 0.2, 0.25):
    ww = 300e-6 / dw_ww  # g per animal
    print(f"   pool 0-25 m = {conc:.2f} g WW/m3; if all were C5-C6 Calanus ({ww*1e3:.1f} mg WW): {conc/ww/1000:.2f} ind/L")
print("   Macy 1998 mackerel: feeding stops below 0.4 /L, starts at 4.4-14.7 /L; Ivlev half-rate 11.5 /L; filter switch 24-64 /L")
B1 = math.log(10) / (37.3 - 0.4)
print(f"   (Ivlev B1 = {B1:.4f} L per copepod; C50 = 0.4 + ln2/B1 = {0.4 + math.log(2)/B1:.1f} /L)")
print("   Gibson & Ezzi 1990 herring: biting -> filtering switch at 18 /L (10-34)")
