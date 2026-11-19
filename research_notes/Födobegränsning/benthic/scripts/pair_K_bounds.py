"""Field half-saturation bounds for the benthivores' bed pairs, and the model's shared-handling
elasticities. Pure arithmetic, no downloads read except the earlier note's station table.

Run: python3 -I scripts/pair_K_bounds.py <path to Mättnad note tables/field_density_x.csv>

Units: B in t/km2 wet (box-mean convention), a in km2 per t predator per 6 h tick, h in ticks,
x = a h B_vis, X = sum_k p_k x_k (shared handling; p = 1 for every searched prey here).
Intake per tick while foraging = T/h_eff * X/(1 + X) for a one-prey menu (T = 1 tick).
tau = foraging share of the time (a_day = a_model * tau when a is read from daily rations).
"""
import csv
import math
import sys

# ---- model numbers (the brief; fg_library.yaml as of resume 247) ----
PAIRS = {  # a, h, B, accessible/reachable share
    'infauna': (0.036, 112.0, 94.0, 0.2),
    'epifauna': (0.15, 258.0, 3.78, 0.62),
    'nephrops': (0.44, 286.0, 4.58, 0.02),
}
# Non-bed prey of the benthivores (approximate, J: stage_access mixes and seasonal/depth
# multipliers set to their means; used only to show they are minor in X).
OTHER = {
    'planktivores': (0.0198, 238.0, 4.72, 0.3),     # stage_access mix J ~0.3
    'whiting': (0.0026, 177.0, 0.446, 0.1),
    'krill': (0.013, 183.0, 5.0, 1.0),              # h = 5000/(26.18*(5000/4184)**0.25)
}

print('== 1. Model, shared handling (p_k = 1 on every prey; elasticity of pair i to its own B_i '
      '= (1 + X - x_i)/(1 + X))')
xs = {}
for k, (a, h, B, f) in {**PAIRS, **OTHER}.items():
    xs[k] = a * h * B * f
    print(f'  {k:12s} a {a:<7} h {h:5.0f}  B_vis {B * f:7.3f} t/km2  x = a h B_vis = {xs[k]:7.2f}  '
          f'single-pair K = 1/(a h) = {1 / (a * h):.4f} t/km2 reachable = {1 / (a * h) / f:.3f} t/km2 of B')
for menu, keys in (('bed only', list(PAIRS)), ('bed + fish + krill', list(xs))):
    X = sum(xs[k] for k in keys)
    print(f'  menu {menu}: X = {X:.1f}; total-menu intake at 0.5x all prey / at today = '
          f'{(0.5 * X / (1 + 0.5 * X)) / (X / (1 + X)):.3f}; elasticity of total intake = {1 / (1 + X):.4f}')
    for k in keys:
        print(f'     {k:12s} own-B elasticity {(1 + X - xs[k]) / (1 + X):.3f}; '
              f'intake at 0.1 x B_i / today {(0.1 * xs[k] / (1 + X - 0.9 * xs[k])) / (xs[k] / (1 + X)):.3f}')

print('\n== 2. Irish Sea gradient (Johnson et al. 2015 [F]; Hinz station data via the earlier note [D])')
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['area'] == 'IrishSea']
sw = [float(r['swept_per_yr']) for r in rows]
lnB = [math.log(float(r['B_prey_t_km2'])) for r in rows]
n = len(rows)
ms, mb = sum(sw) / n, sum(lnB) / n
sxx = sum((s - ms) ** 2 for s in sw)
b_lnB = sum((s - ms) * (y - mb) for s, y in zip(sw, lnB)) / sxx
Bgm = math.exp(mb)
print(f'  {n} stations, B prey (grab, excl. large) gm {Bgm:.1f} t/km2, range '
      f'{min(math.exp(v) for v in lnB):.1f}-{max(math.exp(v) for v in lnB):.1f}; '
      f'slope ln B on trawl {b_lnB:+.3f} per pass/yr; trawl range {min(sw)}-{max(sw)}')
print('  observed: plaice fullness ~ trawl F(1,13) = 0.004, p 0.949; stomach energy F < 0.001, '
      'p 0.981; dab fullness p 0.76 [F]. Prey:fish ratio unchanged (p 0.65) [F].')

# candidate a for benthivores -> infauna (per t predator per tick)
CAND = [('Exposure note / model', 0.036), ('Exposure 5 %', 0.016), ('Exposure 95 %', 0.092),
        ('Feeding-level note e 0.01', 0.25 * 0.01), ('Feeding-level note e 0.002', 0.25 * 0.002),
        ('Feeding-level tau=1 inversion', 0.00015)]
H = PAIRS['infauna'][1]
print('  predicted intake (one-prey menu: plaice is an infauna specialist [F]) at the gradient ends '
      'and the predicted slope of ln intake on trawl (to compare with ~0)')
print('  candidate                         a        share   x at gm B   I(2.4)/I(18.5)  slope ln I/pass')
for share in (0.2, 1.0):
    for lab, a in CAND:
        x = lambda B: a * H * share * B
        f = lambda B: x(B) / (1 + x(B))
        lnI = [math.log(f(math.exp(v))) for v in lnB]
        mi = sum(lnI) / n
        b_I = sum((s - ms) * (y - mi) for s, y in zip(sw, lnI)) / sxx
        print(f'  {lab:32s} {a:<8.5f} {share:4.1f}  {x(Bgm):9.2f}   {f(2.4) / f(18.5):8.3f}     {b_I:+.4f}')

print('\n  Detectability: SE of the fullness slope = sigma / sqrt(Sxx); sigma = between-station SD of '
      'ln mean fullness (J 0.3-0.6;\n  from Johnson: plaice stomach energy 3.22 +- 0.54 J/g s.e., ~200 fish -> '
      'CV ~2.4 per fish, ~13 fish/station -> ~0.6).')
print('  e_max = largest elasticity of intake to prey compatible with a flat result (95 %, t 2.16, 13 df, '
      'Johnson n 15; Sxx scaled to 15 stations)')
sxx15 = sxx * 15 / n
for sig in (0.3, 0.45, 0.6):
    se = sig / math.sqrt(sxx15)
    emax = 2.16 * se / abs(b_lnB)
    if emax >= 1:
        print(f'  sigma {sig}: e_max {emax:.2f} >= 1 -> no bound')
        continue
    Xmin = 1 / emax - 1
    Kmax = Bgm / Xmin                                # K in units of total grab B (box-mean)
    for share in (0.2, 1.0):
        amin = Xmin / (H * share * Bgm)
        print(f'  sigma {sig}: e_max {emax:.2f} -> X >= {Xmin:.2f} at B {Bgm:.1f}; K_menu <= {Kmax:.2f} t/km2 '
              f'of total B (= {Kmax * share:.2f} reachable at share {share}); a >= {amin:.5f} at h {H:.0f} '
              f'(model a / a_min = {0.036 / amin:.1f})')

print('\n  Caveat (J): with tau < 1 the daily ration is tau x the foraging-time response; a flat ration '
      'still bounds the foraging-time x from below, so the bound on a_model holds for any tau.')
print('  Caveat (J): fish declined with prey (ratio constant), so a RATIO-dependent food limitation '
      'also predicts flat fullness;\n  the test falsifies only prey-dependent food limitation (the '
      "model's form with w = 0).")

print('\n== 3. Kattegat (Dryad [D], tables/kattegat_fishfood_patchiness_stdout.txt)')
print('  fish food (0.01-0.2 g individuals) 11.6-77.9 t/km2 over 14-15 fine stations (7x); plaice '
      'condition ~ ln food r2 0.00, LRD 0.05.')
for lab, a in CAND[:1] + CAND[3:]:
    for share in (0.2, 1.0):
        x = lambda B: a * H * share * B
        f = lambda B: x(B) / (1 + x(B))
        print(f'  {lab:32s} share {share}: predicted intake I(11.6)/I(77.9) = {f(11.6) / f(77.9):.3f}')

print('\n== 4. Supply vs demand on the exposed infauna (model box, J feeding level)')
Bb, QB, inf_share = 1.53, 4.46, 0.54
cons = Bb * QB * inf_share
A = 94.0 * 0.2
print(f'  benthivore infauna consumption at the field Q/B {QB}: {cons:.2f} t/km2/yr = '
      f'{cons / A:.2f} /yr of the exposed stock {A:.1f} t/km2; refuge exchange 2.5 /yr; '
      f'infauna production ~r_max 1.0 x 94 = 94 t/km2/yr')
print(f'  encounter-limited removal if fish searched all the time: a P 1460 = '
      f'{0.036 * Bb * 1460:.0f} /yr (gut-capped in practice)')

print('\n== 5. Epifauna: lab a (Persson et al. 2012 [F], juvenile cod 30-60 g) scaled to adults by ~1/L (J)')
for lab, lo, hi in (('Crangon (sand)', 0.065, 0.13), ('Palaemon', 0.76, 1.5)):
    for L in (2.5, 3.0):
        alo, ahi = lo / L, hi / L
        h = PAIRS['epifauna'][1]
        Bv = PAIRS['epifauna'][2] * PAIRS['epifauna'][3]
        print(f'  {lab:15s} /{L}: a {alo:.3f}-{ahi:.3f}; K = 1/(a h) {1 / (ahi * h):.4f}-{1 / (alo * h):.4f} '
              f't/km2 accessible; x at today {alo * h * Bv:.0f}-{ahi * h * Bv:.0f}')
print('  Hvingel & Kingsley 2006 (cod -> Pandalus, stock scale, Type III): P50 3.6-3.8 x B_MSY at '
      'B/B_MSY ~1-1.7 -> x ~0.27-0.47 per pair;\n  in the shared form a minority prey has own-B '
      'elasticity ~1 (section 1), so a linear per-prey stock response is the shared denominator, not a low a (J).')

print('\n== 6. Irish Sea: likelihood of the observed flat slope (b_obs ~ 0, |t| 0.06) under each candidate')
print('  z = |predicted slope| / SE; LR vs the model a = exp((z_cand^2 - z_model^2)/2) (Gaussian, J)')
sxx15 = sxx * 15 / n
for sig in (0.3, 0.45, 0.6):
    se = sig / math.sqrt(sxx15)
    out = []
    zs = {}
    for lab, a in CAND:
        x = lambda B: a * H * 0.2 * B
        f = lambda B: x(B) / (1 + x(B))
        lnI = [math.log(f(math.exp(v))) for v in lnB]
        mi = sum(lnI) / n
        b_I = sum((s - ms) * (y - mi) for s, y in zip(sw, lnI)) / sxx
        zs[lab] = abs(b_I) / se
    zm = zs[CAND[0][0]]
    print(f'  sigma {sig} (SE {se:.3f}): ' + '; '.join(
        f'{lab} z {z:.2f} LR {math.exp((z * z - zm * zm) / 2):.1f}' for lab, z in zs.items()))
