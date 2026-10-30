"""Annual-mean mackerel and horse mackerel density (box, Skagerrak 3.a.20, Division 3.a) from a
peak-season density and monthly presence profiles; catch-based bounds; implied GOV catchability.

Usage: python3 -I annual_mean.py
Every input number is typed here and quoted with its source in ../summary.md. Profiles are
judgement (J) built from the quarterly/monthly evidence listed in summary.md section 3.
"""
MONTHS = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()
A20, A21 = 34041.0, 20536.0          # km2, ICES areas shapefile (catch_ices_areas_km2.csv)
KG_H_TO_T_KM2 = 0.00704             # GOV kg/h -> t/km2 (project convention, acoustic note)


def pres(p):
    return sum(p) / 12


def show(p):
    return ' '.join(f'{m}:{v:g}' for m, v in zip(MONTHS, p))


# =========================== MACKEREL ===========================
# Peak (July) density, t/km2, IESSNS surface trawl q = 1 (tables/iessns_box_stdout.txt):
#   box hauls median 2.0 (n 6), box-depth (100-320 m) hauls median 1.83, NS stratum 2018-25 mean
#   1.83, Degnbol 1987 Skagerrak 1.6-1.8 (Aug-Sep), Argos 1989 1.9 (Aug).
# Depth weighting over the box (maps/depth.npz, 3600 cells): <250 m 85.5 %, 250-300 m 5.9 %,
# >=300 m 8.7 %. July density by depth: in-box hauls 83-207 m median 2.0; hauls at 313/480 m next
# to the box 0.24/0.18 and Trench >=400 m median 0.49 -> 0.35; 250-300 m assumed 1.0 (J).
DEPTH_W = [(0.855, 2.0), (0.059, 1.0), (0.087, 0.35)]
JULY_DW = sum(w * v for w, v in DEPTH_W) / sum(w for w, _ in DEPTH_W)
print(f'depth-weighted July box density {JULY_DW:.2f} t/km2')
JULY_BOX = {'low': 0.9, 'central': round(JULY_DW, 1), 'high': 3.0}
JULY_SKAG = {'low': 0.8, 'central': 1.7, 'high': 2.7}   # whole-Skagerrak hauls median 1.78, geomean 1.12
KATT_REL = {'low': 0.05, 'central': 0.2, 'high': 0.5}  # J: Kattegat density relative to Skagerrak

# Monthly density relative to July. Evidence: GOV Q1/Q3 = 0.052 (box, 2001-25); NO 3.a hook catch
# by month 2015-24 Apr 4.5 %, May 18, Jun 23, Jul 19, Aug 16, Sep 12, Oct 7, Nov-Mar 1.3 %; EU
# 3.a.20 Q1 1 %, Q2 48, Q3 43, Q4 7 %; Degnbol Aug-Sep 1987 ~ July level; Revheim 1969 autumn
# withdrawal to the Trench, purse-seine peak Oct; SMS North Sea Q1-Q4 shares 4.9/5.8/21.2/27.7 %.
MAC = {
    'narrow': [0.03, 0.03, 0.03, 0.1, 0.4, 0.9, 1.0, 0.9, 0.6, 0.25, 0.08, 0.04],
    'central': [0.05, 0.05, 0.05, 0.15, 0.5, 1.0, 1.0, 1.0, 0.8, 0.5, 0.2, 0.08],
    'wide': [0.1, 0.1, 0.1, 0.25, 0.7, 1.0, 1.0, 1.0, 1.0, 0.8, 0.5, 0.2],
}
print('== Mackerel monthly profiles (density / July) ==')
for k, p in MAC.items():
    print(f'{k:8s} presence {pres(p):.3f}  ({sum(p):.2f} July-months)  {show(p)}')

fc, fl, fh = pres(MAC['central']), pres(MAC['narrow']), pres(MAC['wide'])
box = (JULY_BOX['central'] * fc, JULY_BOX['low'] * fl, JULY_BOX['high'] * fh)
sk = (JULY_SKAG['central'] * fc, JULY_SKAG['low'] * fl, JULY_SKAG['high'] * fh)
lik_box = (JULY_BOX['central'] * fl, JULY_BOX['central'] * fh)
print('\n== Mackerel annual mean, t/km2 (central; envelope = low July x narrow .. high July x wide) ==')
print(f'box        {box[0]:.2f}  envelope {box[1]:.2f}-{box[2]:.2f}  '
      f'(central July, profile spread {lik_box[0]:.2f}-{lik_box[1]:.2f})')
print(f'Skagerrak  {sk[0]:.2f}  envelope {sk[1]:.2f}-{sk[2]:.2f}')
k_c = sk[0] * KATT_REL['central']
k_l, k_h = sk[1] * KATT_REL['low'], sk[2] * KATT_REL['high']
d3a = [(s * A20 + k * A21) / (A20 + A21) for s, k in ((sk[0], k_c), (sk[1], k_l), (sk[2], k_h))]
print(f'Kattegat   {k_c:.2f}  envelope {k_l:.3f}-{k_h:.2f}  (J: x{KATT_REL["central"]} of Skagerrak)')
print(f'3.a        {d3a[0]:.2f}  envelope {d3a[1]:.2f}-{d3a[2]:.2f}  (area-weighted, {A20:.0f} + {A21:.0f} km2)')
print(f'summer (Jun-Aug) box {JULY_BOX["central"]:.1f}, Skagerrak {JULY_SKAG["central"]:.1f} t/km2; '
      f'Sep {JULY_BOX["central"] * MAC["central"][8]:.2f}, Oct {JULY_BOX["central"] * MAC["central"][9]:.2f}')

# standing stock in tonnes (model box 3600 km2)
print(f'box 3600 km2: annual mean {box[0] * 3600:.0f} t, July {JULY_BOX["central"] * 3600:.0f} t')

# GOV comparison
gov_q3_long, gov_q3_recent = 22.43 * KG_H_TO_T_KM2, 0.29
model_mac = 0.138 * 0.70
print(f'\nGOV Q3 box: long-run 2001-25 {gov_q3_long:.3f}, 2021-25 {gov_q3_recent:.2f} t/km2 -> '
      f'implied GOV q vs July {gov_q3_long / JULY_BOX["central"]:.2f}-{gov_q3_recent / JULY_BOX["central"]:.2f}')
print(f'model now: mackerel 0.138 x 0.70 = {model_mac:.3f} t/km2; central box is x{box[0] / model_mac:.1f}')

# Catch route: annual-mean B = C / F_local (Baranov, catch = F x mean biomass, any residence)
print('\n== Catch route (lower bound with stock F; implied local F with the survey B) ==')
C = {'3.a.20': 343.4, '3.a.21': 211.1, '3.a': 554.5}   # t/yr 2016-24, ICES official (+NK to 3.a.20)
AREA = {'3.a.20': A20, '3.a.21': A21, '3.a': A20 + A21}
for F in (0.215, 0.285):
    print(f'F_stock {F}: ' + ', '.join(f'{a} B >= {C[a] / F:.0f} t = {C[a] / F / AREA[a]:.3f} t/km2' for a in C))
B20 = sk[0] * A20
print(f'survey-based 3.a.20 annual mean {B20:.0f} t -> F_local = {C["3.a.20"] / B20:.3f} /yr '
      f'(stock F 0.215-0.285)')

# =========================== HORSE MACKEREL ===========================
print('\n== Horse mackerel ==')
argos = 7442 / 8345
print(f'Argos 1-4 Aug 1989 eastern Skagerrak 20-300 m: {argos:.2f} t/km2 at TS -71.2 dB; '
      f'at -67.5 dB {argos * 10 ** (-3.7 / 10):.2f}')
gq3, gq1, gpool = 9.939 * KG_H_TO_T_KM2, 0.204 * KG_H_TO_T_KM2, 5.071 * KG_H_TO_T_KM2
print(f'GOV box 2001-25: Q3 {gq3:.3f}, Q1 {gq1:.4f}, pooled {gpool:.3f} t/km2 (x1.18 = {gpool * 1.18:.3f}; '
      f'model now 0.138 x 0.30 = {0.138 * 0.30:.3f})')
# peak (Sep-Oct) density: GOV Q3 / q, q 0.3-1 (semi-demersal by day; Argos 1989 an upper, big-stock year)
PEAK_HOM = {'low': gq3, 'central': 0.17, 'high': 0.40}
# Catch: EU 3.a.20 Q3 19 %, Q4 80 % (FDI 2016-24); NO all Aug-Nov; WGWIDE 3.a 2023-25 Q3 46/13/11,
# Q4 3/11/43 t; IESSNS early July 0; HERAS July acoustic box ~0.01 t/km2.
HOM = {
    'narrow': [0.05, 0, 0, 0, 0, 0, 0.02, 0.3, 0.8, 1.0, 0.6, 0.15],
    'central': [0.1, 0.03, 0, 0, 0, 0.02, 0.1, 0.5, 0.9, 1.0, 0.8, 0.3],
    'wide': [0.2, 0.1, 0.05, 0, 0, 0.05, 0.2, 0.7, 1.0, 1.0, 1.0, 0.5],
}
for k, p in HOM.items():
    print(f'{k:8s} presence {pres(p):.3f}  {show(p)}')
h = (PEAK_HOM['central'] * pres(HOM['central']), PEAK_HOM['low'] * pres(HOM['narrow']),
     PEAK_HOM['high'] * pres(HOM['wide']))
print(f'horse mackerel annual mean box/Skagerrak {h[0]:.3f} t/km2, envelope {h[1]:.3f}-{h[2]:.3f}; '
      f'autumn peak {PEAK_HOM["central"]} ({PEAK_HOM["low"]:.2f}-{PEAK_HOM["high"]})')
CH = {'3.a.20 2016-24': 210.5, '3.a.20 2020-24': 55.3}
for F in (0.038, 0.065):
    print(f'F_stock {F}: ' + ', '.join(f'{a} {c / F:.0f} t = {c / F / A20:.3f} t/km2' for a, c in CH.items()))
print('Kattegat horse mackerel catch 3.4 t/yr (3.a.21) -> treat as ~0; 3.a mean = Skagerrak x '
      f'{A20 / (A20 + A21):.2f} = {h[0] * A20 / (A20 + A21):.3f} t/km2')
