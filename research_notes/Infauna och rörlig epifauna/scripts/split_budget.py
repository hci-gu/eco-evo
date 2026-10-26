"""Library numbers for the infauna / mobile_epifauna split of benthic_community (no model output used).

Usage: python3 -I split_budget.py <repo> <epifauna_note_dir> [variant]
  <repo>              /home/simon/eco-evo (read-only: maps/depth.npz, maps/fishing_bottom_otter_trawls.npz,
                      tools/make_fishing_maps.py for the grid projection, lib/world/habitat.py formulas)
  <epifauna_note_dir> research_notes/"Bottensamhallets epifauna" (Swedish letters): data/uwtv_survey_share_per_cell.npy
  [variant] [d_inf]   d_inf: infauna d per pass (default 0.06; 0.10 = the library today)
  [variant]           central (default) | bottomup (crab stock at the evidence-B proxy, no mass-balance raise)
                      | codnns (benthivore epifauna diet with cod from the northern North Sea)

Conventions (resume 207/218, lib/world/growth_budget.py, lib/world/habitat.py):
- t WW/km2, rates per year; 1460 ticks per year at the library's 6 h.
- NDM growth is DERIVED from r_max: g = (r_max + sum M2)/1460, so the engine's logistic r is
  r_gross = r_max + M2 and today's equilibrium share is s = 1 - (F + M2)/r_gross = (r_max - F)/(r_max + M2).
- K table rows = today's stock at the anchor / s (box-mean F and M2), as in resume 218.
- Trawl loss F(c) = 1460 x slope x E(c), slope = F_mean / E_mean (linear table from 0), E(c) the bottom
  otter trawl baseline map (MW-h/km2/yr). Library biomass_factor at table value 100 = slope x 100 / 1460... per
  tick: bf100 = F_mean x 100 / (1460 x E_mean).
- Engine start = sum over cells of K_table(depth) x max(0, 1 - (F(c) + M2)/r_gross).
"""
import sys

import numpy as np

repo, epi_dir = sys.argv[1], sys.argv[2]
variant = sys.argv[3] if len(sys.argv) > 3 else "central"
sys.path.insert(0, repo + "/tools")
from make_fishing_maps import wgs84_to_sweref99tm, E0, N0, CELL, GRID  # noqa: E402

TPY = 1460.0
dep = np.load(repo + "/maps/depth.npz")["depth"].astype(float)
E = np.load(repo + "/maps/fishing_bottom_otter_trawls.npz")["bottom_trawling"].astype(float)
Em = E.mean()
sv = np.nan_to_num(np.load(epi_dir + "/data/uwtv_survey_share_per_cell.npy"))
K_SAR = 0.75          # SAR per MW-h/km2 (swept_area.md)


def lin(a, b):
    return np.interp(dep, [100, 300], [a, b])


# ---------------------------------------------------------------------------------------------------
# today's stocks per cell
# ---------------------------------------------------------------------------------------------------
LO, LA = np.meshgrid(np.linspace(10.0, 11.25, 1000), np.linspace(57.75, 58.36, 1000))
Ex, Nx = wgs84_to_sweref99tm(LO.ravel(), LA.ravel())
c = np.floor((Ex - E0) / CELL).astype(int)
r = GRID - 1 - np.floor((Nx - N0) / CELL).astype(int)
ok = (c >= 0) & (c < 60) & (r >= 0) & (r < 60)
n = np.zeros((60, 60)); clon = np.zeros((60, 60)); clat = np.zeros((60, 60))
np.add.at(n, (r[ok], c[ok]), 1); np.add.at(clon, (r[ok], c[ok]), LO.ravel()[ok]); np.add.at(clat, (r[ok], c[ok]), LA.ravel()[ok])
clon /= n; clat /= n
S = np.where(clon < 10.71, 1, np.where(clat > 57.83 + (clon - 10.71) * (0.17 / 0.49), 8, 9))
dens = {1: 0.26, 8: 0.09, 9: 0.55}                       # combine.py central (UWTV 2024 strata)
stock = {
    "infauna": lin(110.0, 55.0),                          # Josefson 1990 anchors (resume 207)
    "nephrops": sum(sv * (S == s) * d for s, d in dens.items()) * 30.0,
    "pandalus": np.select([dep < 200, dep < 300, dep < 500], [0.12, 0.57, 0.35], 0.0) / 0.40,
    "crabs": lin(0.9, 0.45),                              # brachyurans + hermits + Munida, evidence B central
    "small": lin(0.5, 0.5),                               # small shrimps 0.3 + mysids/hyperbenthos 0.2
}
EPI = ["nephrops", "pandalus", "crabs", "small"]

# ---------------------------------------------------------------------------------------------------
# consumption flows by the checked predators (t/km2/yr), from the diet scripts in ../tables
# ---------------------------------------------------------------------------------------------------
QB_B, BB, PB_B = 3.48, 1.53, 0.72          # benthivores: Q/B, biomass, benthos share (group_diet_consumption.csv)
QB_P, BP, PB_P = 3.34, 1.20, 0.11          # predators (params_predatory_fish.md)
e_benth = {"central": 0.290, "bottomup": 0.290, "codnns": 0.338}[variant]
comp_benth = {"central": {"crabs": 0.872, "pandalus": 0.051, "nephrops": 0.003, "small": 0.073},
              "bottomup": {"crabs": 0.872, "pandalus": 0.051, "nephrops": 0.003, "small": 0.073},
              "codnns": {"crabs": 0.548, "pandalus": 0.264, "nephrops": 0.118, "small": 0.073}}[variant]
e_pred = 0.65                               # whiting box + Skagerrak 0.61, all areas 2022-24 0.71
comp_pred = {"pandalus": 0.5, "small": 0.4, "crabs": 0.1, "nephrops": 0.0}
C_benth = QB_B * BB * PB_B
C_pred = QB_P * BP * PB_P
flow = {"infauna": {"benthivorous_fish": C_benth * (1 - e_benth), "predatory_fish": C_pred * (1 - e_pred)}}
for k in EPI:
    flow[k] = {"benthivorous_fish": C_benth * e_benth * comp_benth[k], "predatory_fish": C_pred * e_pred * comp_pred[k]}
# unresolved (unchecked pair, inside r_max): planktivores' mysids / small shrimps, judgement
UNRES_SMALL = 0.5

# ---------------------------------------------------------------------------------------------------
# trawl: catch (t/km2/yr at mean effort, summary_trawl_removal.md) + kill of the uncaught part d_u x SAR
# ---------------------------------------------------------------------------------------------------
catch = {"infauna": 0.0, "nephrops": 0.296, "pandalus": 0.336, "crabs": 0.006 + 0.031, "small": 0.005}
# 0.044 dead discards of other invertebrates: ~0.031 crabs/hermits/Munida, ~0.005 small shrimps,
# ~0.008 echinoderms (infauna, inside d) [J]
d_u = {"infauna": float(sys.argv[4]) if len(sys.argv) > 4 else 0.06, "nephrops": 0.01, "pandalus": 0.01, "crabs": 0.05, "small": 0.02}
r_max = {"infauna": 1.0, "nephrops": 0.34, "pandalus": 1.2, "crabs": 0.9, "small": 1.86}
PB_LIT = {"nephrops": (0.3, 0.4), "pandalus": (1.2, 1.7), "crabs": (0.4, 0.9), "small": (1.4, 2.5)}

# mass-balance (Ecopath-style) minimum stock: B (EE x P/B - F_kill) >= C + catch
EE = 0.9
print(f"variant {variant}; mean effort {Em:.4f}; SAR at mean effort {K_SAR * Em:.3f}")
print("-" * 100)
print("Per-taxon flows (t/km2/yr) and the mass-balance check B_min = (C + catch) / (EE P/B - d_u SAR)")
for k in EPI:
    Cf = sum(flow[k].values()) + (UNRES_SMALL if k == "small" else 0.0)
    out = []
    for pb in PB_LIT[k]:
        den = EE * pb - d_u[k] * K_SAR * Em
        out.append((Cf + catch[k]) / den if den > 0 else float("inf"))
    print(f"  {k:<9} stock {stock[k].mean():5.2f}  consumption {Cf:5.3f} (benth {flow[k]['benthivorous_fish']:.3f},"
          f" pred {flow[k]['predatory_fish']:.3f}{', unresolved %.2f' % UNRES_SMALL if k == 'small' else ''})"
          f"  catch {catch[k]:.3f}  B_min at P/B {PB_LIT[k][1]}-{PB_LIT[k][0]}: {out[1]:.2f}-{out[0]:.2f}")
if variant != "bottomup":
    # raise the crab stock to the mass-balance estimate at P/B 0.7 (Brey 2001 runs: hermits 0.66-0.70,
    # Munida 0.42-0.54, Atelecyclus/Goneplax 0.44-0.46, Pisidia 0.94; M&D large crabs 0.55)
    Cf = sum(flow["crabs"].values())
    target = (Cf + catch["crabs"]) / (EE * 0.7 - d_u["crabs"] * K_SAR * Em)
    stock["crabs"] = stock["crabs"] * target / stock["crabs"].mean()
    print(f"  crabs raised to the mass-balance stock {stock['crabs'].mean():.2f} t/km2 (x {target / lin(0.9, 0.45).mean():.2f}"
          f" the evidence-B central; evidence-B high ~2.3)")
print("-" * 100)


def F_mean(k):
    return catch[k] / stock[k].mean() + d_u[k] * K_SAR * Em


tax = {}
print(f"{'taxon':<9}{'B*':>7}{'F':>7}{'M2':>8}{'r_max':>7}{'r_gross':>8}{'share s':>8}{'K':>8}")
for k in ["infauna"] + EPI:
    B = stock[k].mean()
    F = F_mean(k)
    M2 = sum(flow[k].values()) / B
    rg = r_max[k] + M2
    s = 1 - (F + M2) / rg
    tax[k] = dict(B=B, F=F, M2=M2, r=r_max[k], s=s, K=B / s)
    print(f"{k:<9}{B:7.2f}{F:7.3f}{M2:8.4f}{r_max[k]:7.2f}{rg:8.3f}{s:8.3f}{B / s:8.2f}")
print("-" * 100)


def group(keys, label, rmax_choice=None, pairs=("benthivorous_fish", "predatory_fish")):
    Bc = sum(stock[k] for k in keys)                       # today's stock per cell
    Bm = Bc.mean()
    loss_t = sum(catch[k] + d_u[k] * K_SAR * Em * stock[k].mean() for k in keys)   # t/km2/yr at mean effort
    Fg = loss_t / Bm
    M2p = {p: sum(flow[k][p] for k in keys) / Bm for p in pairs}
    M2g = sum(M2p.values())
    Ksum = sum(tax[k]["K"] for k in keys)
    s_sum = Bm / Ksum
    r_self = (Fg + M2g) / (1 - s_sum) - M2g                # r_max that reproduces sum K_i
    r_bw = sum(r_max[k] * stock[k].mean() for k in keys) / Bm
    rm = r_self if rmax_choice is None else rmax_choice
    rg = rm + M2g
    s = 1 - (Fg + M2g) / rg
    Kc = Bc / s                                            # K per cell at the box-mean share
    x = np.clip(dep, 100, 300).ravel()
    t = (x - 100) / 200
    A = np.vstack([1 - t, t]).T
    k100, k300 = np.linalg.lstsq(A, Kc.ravel(), rcond=None)[0]
    b100, b300 = np.linalg.lstsq(A, Bc.ravel(), rcond=None)[0]
    Ktab = np.interp(dep, [100, 300], [k100, k300])
    slope_tick = Fg / (TPY * Em)                           # per tick per unit effort
    bf100 = slope_tick * 100
    # iterate M2 on the engine's own equilibrium map once (resume 218 practice)
    for it in range(3):
        Fc = TPY * slope_tick * E
        Bstar = Ktab * np.clip(1 - (Fc + M2g) / rg, 0, 1)
        M2p = {p: sum(flow[k][p] for k in keys) / Bstar.mean() for p in pairs}
        M2g = sum(M2p.values())
        rg = rm + M2g
    Fc = TPY * slope_tick * E
    Bstar = Ktab * np.clip(1 - (Fc + M2g) / rg, 0, 1)
    print(f"== {label}  (members: {', '.join(keys)})")
    print(f"   today's stock: grid mean {Bm:.2f}; LSQ depth anchors 100 m {b100:.2f}, 300 m {b300:.2f}")
    print(f"   trawl loss at mean effort F {Fg:.4f} /yr  -> slope {slope_tick:.4e} per tick per unit,"
          f" biomass_factor at table value 100 = {bf100:.6f}")
    print(f"   M2 by pair (final, on the engine B* mean): " + ", ".join(f"{p} {v:.4f}" for p, v in M2p.items())
          + f"  (sum {M2g:.4f})")
    print(f"   r_max: self-consistent with sum K_i {r_self:.3f}; biomass-weighted {r_bw:.3f}; used {rm:.3f}"
          f" -> engine r_gross {rg:.3f} (growth_rate {rg / TPY:.9f} per tick)")
    print(f"   equilibrium share s = 1 - (F + M2)/r_gross = {1 - (Fg + M2g) / rg:.3f}")
    print(f"   K table: 100 m {k100:.2f}, 300 m {k300:.2f}; box mean of table {Ktab.mean():.2f}"
          f" (sum of taxon K {Ksum:.2f})")
    print(f"   engine B*(c): mean {Bstar.mean():.2f}, min {Bstar.min():.2f}, max {Bstar.max():.2f};"
          f" START TOTAL sum B* = {Bstar.sum():.0f} t  (range +/-25 %: {0.75 * Bstar.sum():.0f}-{1.25 * Bstar.sum():.0f});"
          f" zero cells {(Bstar <= 0).sum()}")
    for lo, hi in [(0, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 600)]:
        m = (dep >= lo) & (dep < hi)
        print(f"     {lo:>3}-{hi:<3} m ({m.sum():4d} cells): today {Bc[m].mean():7.2f}  engine B* {Bstar[m].mean():7.2f}"
              f"  K {Ktab[m].mean():7.2f}  mean effort {E[m].mean():5.2f}")
    # optional multi-row table: one row per depth band at the band's mean depth, K_j = today's band
    # mean / band-mean equilibrium share at the engine's per-cell F (keeps the depth structure of B*)
    share_c = np.clip(1 - (Fc + M2g) / rg, 0, 1)
    rows = []
    for lo, hi in [(0, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 600)]:
        m = (dep >= lo) & (dep < hi)
        rows.append((dep[m].mean(), Bc[m].mean() / share_c[m].mean()))
    Kmulti = np.interp(dep, [x for x, _ in rows], [k for _, k in rows])
    Bm2 = Kmulti * share_c
    print("   OPTIONAL multi-row table (depth_m: capacity): " + ", ".join(f"{d:.0f}: {k:.2f}" for d, k in rows)
          + f"; box mean K {Kmulti.mean():.2f}; engine B* mean {Bm2.mean():.2f}; START {Bm2.sum():.0f} t")
    for lo, hi in [(0, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 600)]:
        m = (dep >= lo) & (dep < hi)
        print(f"     multi-row {lo:>3}-{hi:<3} m: today {Bc[m].mean():7.2f}  engine B* {Bm2[m].mean():7.2f}")
    return dict(k100=k100, k300=k300, Kmean=Ktab.mean(), start=Bstar.sum(), F=Fg, M2=M2p, r=rm, rg=rg, bf100=bf100)


g_inf_pred = group(["infauna"], "INFAUNA with predators -> infauna checked (alternative)", rmax_choice=1.0)
g_inf_only = group(["infauna"], "INFAUNA without predators -> infauna (recommended)", rmax_choice=1.0,
                   pairs=("benthivorous_fish",))
g_epi = group(EPI, "mobile_epifauna (r_max self-consistent)")
g_epi_bw = group(EPI, "mobile_epifauna (r_max biomass-weighted, comparison)",
                 rmax_choice=sum(r_max[k] * stock[k].mean() for k in EPI) / sum(stock[k].mean() for k in EPI))
# accessible fraction for benthivores by taxon (evidence A1, A2, B)
f_b = {"nephrops": 0.10, "pandalus": 0.7, "crabs": 0.6, "small": 0.6}
f_p = {"nephrops": 0.0, "pandalus": 0.7, "crabs": 0.15, "small": 0.6}   # predators: no Nephrops, small crabs only [J]
Bt = sum(stock[k].mean() for k in EPI)
print("-" * 100)
print(f"accessible_fraction benthivores -> mobile_epifauna = {sum(f_b[k] * stock[k].mean() for k in EPI) / Bt:.3f}"
      f" (accessible {sum(f_b[k] * stock[k].mean() for k in EPI):.2f} t/km2)")
print(f"accessible_fraction predators   -> mobile_epifauna = {sum(f_p[k] * stock[k].mean() for k in EPI) / Bt:.3f}"
      f" (accessible {sum(f_p[k] * stock[k].mean() for k in EPI):.2f} t/km2)")
print(f"taxon shares of the epifauna stock: " + ", ".join(f"{k} {stock[k].mean() / Bt:.3f}" for k in EPI))
