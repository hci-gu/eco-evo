"""E1 split of mobile_epifauna: exposed taxa (mobile_epifauna) and Nephrops (norway_lobster).

Usage (from the repository root; read-only on everything outside this note):
  python3 -I "research_notes/Havskräfta och exponerad epifauna/scripts/split_e1.py" \
      <repo> <epifauna_note_dir> <variant> <out_json>
  <repo>              the repository root (maps/depth.npz, maps/fishing_bottom_otter_trawls.npz,
                      tools/make_fishing_maps.py for the grid projection)
  <epifauna_note_dir> research_notes/"Bottensamhallets epifauna" (Swedish letters):
                      data/uwtv_survey_share_per_cell.npy
  <variant>           central  : 221 taxon stocks; the benthivores' 223.10 diet (epifauna 0.314 of the
                                 consumption at Q/B 4.46) booked on the existing taxa; the 0.096 share of
                                 "redistributed" prey without a stock (cephalopods, krill/hyperbenthos,
                                 small demersal fish, the epifauna slices of unidentified fish and carrion)
                                 is booked on the exposed compartment, stocks unchanged (223.10 convention)
                      massbal  : the 221 convention applied to the 223.10 ration: a stock that cannot
                                 produce its measured consumption is raised by mass balance (crabs at P/B
                                 0.7, small shrimps/mysids at 1.95), and the booked prey get their own
                                 member "cephalopods and small demersal fish" with B = flow / (EE P/B)
                      codnns   : as central, but the benthivores' measured epifauna diet with the cod of
                                 the northern North Sea (Nephrops 0.118 of it; 221 variant), total epifauna
                                 share kept at 0.314

Conventions (resume 207/218/221/223.10; lib/world/growth_budget.py; lib/world/habitat.py):
- t WW/km2, rates per year; 1460 ticks per year at the library's 6 h.
- NDM growth DERIVED: g = (r_max + sum M2)/1460; r_gross = r_max + M2; equilibrium share
  s = 1 - (F + M2)/r_gross.
- M2 of a pair = flow (t/km2/yr) / the engine's equilibrium-map mean B* of the group (iterated).
- Trawl: linear table from 0; F(c) = F_group x E(c)/E_mean; bf100 = F_group x 100 / (1460 E_mean).
- K table: six rows at the band-mean depths (81, 123, 173, 221, 272, 387 m), fitted by NON-NEGATIVE
  least squares so B*(c) = K(c) keep(c) matches today's stock map of the group's members;
  keep(c) = clip(1 - (F(c) + M2)/r_gross, 0, 1). Start total = sum B*(c).
"""
import json
import sys

import numpy as np
from scipy.optimize import nnls

repo, epi_dir, variant, out_json = sys.argv[1:5]
assert variant in ("central", "massbal", "codnns", "massbal_neph"), variant
# massbal_neph (user, resume 223.14-223.15): massbal with the Nephrops share of the measured
# benthos-epifauna diet from research_notes/"Havskrafta i fiskdieter" (central 0.069; it replaces part
# of the crab/Pagurus/Aphrodita share, the epifauna total unchanged)
MASSBAL = variant.startswith("massbal")
sys.path.insert(0, repo + "/tools")
from make_fishing_maps import wgs84_to_sweref99tm, E0, N0, CELL, GRID  # noqa: E402

TPY = 1460.0
EE = 0.9                      # Ecopath ecotrophic efficiency convention (221)
dep = np.load(repo + "/maps/depth.npz")["depth"].astype(float)
E = np.load(repo + "/maps/fishing_bottom_otter_trawls.npz")["bottom_trawling"].astype(float)
Em = E.mean()
sv = np.nan_to_num(np.load(epi_dir + "/data/uwtv_survey_share_per_cell.npy"))
K_SAR = 0.75                  # SAR per MW-h/km2 (../Bottentralning och bentos/swept_area.md)
SAR = K_SAR * Em


def lin(a, b):
    return np.interp(dep, [100, 300], [a, b])


# ---------------------------------------------------------------------------------------------------
# 1. Today's stocks per cell: exactly split_budget.py of "Infauna och rorlig epifauna" (221)
# ---------------------------------------------------------------------------------------------------
LO, LA = np.meshgrid(np.linspace(10.0, 11.25, 1000), np.linspace(57.75, 58.36, 1000))
Ex, Nx = wgs84_to_sweref99tm(LO.ravel(), LA.ravel())
c = np.floor((Ex - E0) / CELL).astype(int)
r = GRID - 1 - np.floor((Nx - N0) / CELL).astype(int)
ok = (c >= 0) & (c < 60) & (r >= 0) & (r < 60)
n = np.zeros((60, 60)); clon = np.zeros((60, 60)); clat = np.zeros((60, 60))
np.add.at(n, (r[ok], c[ok]), 1); np.add.at(clon, (r[ok], c[ok]), LO.ravel()[ok])
np.add.at(clat, (r[ok], c[ok]), LA.ravel()[ok])
clon /= n; clat /= n
S = np.where(clon < 10.71, 1, np.where(clat > 57.83 + (clon - 10.71) * (0.17 / 0.49), 8, 9))
dens = {1: 0.26, 8: 0.09, 9: 0.55}                       # UWTV 2024 strata, animals/m2 (combine.py)
stock = {
    "nephrops": sum(sv * (S == s) * d for s, d in dens.items()) * 30.0,   # x 30 g per animal
    "pandalus": np.select([dep < 200, dep < 300, dep < 500], [0.12, 0.57, 0.35], 0.0) / 0.40,
    "crabs": lin(0.9, 0.45),                              # evidence B central, raised below (221)
    "small": lin(0.5, 0.5),                               # small shrimps 0.3 + mysids/hyperbenthos 0.2
}
# 221 mass-balance raise of the crabs (at the 221 ration, Q/B 3.48): reproduces the library's 2.0
_c221 = 3.48 * 1.53 * 0.72 * 0.290 * 0.872 + 3.34 * 1.20 * 0.11 * 0.65 * 0.1
_t221 = (_c221 + 0.037) / (EE * 0.7 - 0.05 * SAR)
stock["crabs"] = stock["crabs"] * _t221 / stock["crabs"].mean()

# ---------------------------------------------------------------------------------------------------
# 2. Flows (t/km2/yr) at the library as it stands (223.10)
# ---------------------------------------------------------------------------------------------------
QB_B, BB = 4.46, 1.53         # benthivores: energy-equivalent Q/B (223.10), biomass (209)
QB_P, BP = 3.34, 1.20         # predators (params_predatory_fish.md; 221)
C_B, C_P = QB_B * BB, QB_P * BP
# research_notes/"Bentivorernas energi och diet"/tables/diet_categories.csv, diet_m2_stdout.txt
DIET_B = 0.720                # benthos share of the consumption (209 central)
E_BENTHOS = 0.290             # epifauna share of the benthos diet (221 central)
NORM = 1.0 - 0.04325          # within-group fish dropped and renormalised (223.10)
BOOKED = {                    # 223.10 central: no-pool items sent to the epifauna (share of consumption)
    "cephalopod": 0.03185,                    # sepiolids/octopods on the bed
    "krill_hyperbenthos": 0.02508,            # mysids, small shrimps
    "small_demersal_fish": 0.02376,           # gobies, dragonets, sculpins
    "unid_fish_E": 0.03906 * 0.19932,         # the small-demersal-fish slice of unidentified fish
    "carrion_E": 0.01189 * 0.29,              # hagfish's live prey, epifauna part
}
measured_E = DIET_B * E_BENTHOS / NORM       # share of consumption, measured benthos-epifauna
share_E = measured_E + sum(BOOKED.values()) / NORM
comp_benth = {"crabs": 0.872, "pandalus": 0.051, "nephrops": 0.003, "small": 0.073}   # 221 central
if variant == "codnns":
    comp_benth = {"crabs": 0.548, "pandalus": 0.264, "nephrops": 0.118, "small": 0.073}
if variant == "massbal_neph":
    comp_benth = {"crabs": 0.872 - (0.069 - 0.003), "pandalus": 0.051, "nephrops": 0.069, "small": 0.073}
comp_pred = {"pandalus": 0.5, "small": 0.4, "crabs": 0.1, "nephrops": 0.0}          # 221 J
SHARE_P = 0.11 * 0.65         # predators: benthos 0.11 x epifauna 0.65 = 0.072 (221)

flow = {k: {"benthivorous_fish": C_B * measured_E * comp_benth[k],
            "predatory_fish": C_P * SHARE_P * comp_pred[k]} for k in comp_benth}
booked_flow = {k: C_B * v / NORM for k, v in BOOKED.items()}
UNRES_SMALL = 0.5             # planktivores on mysids/small shrimps, unresolved (inside r_max), J (221)

catch = {"nephrops": 0.296, "pandalus": 0.336, "crabs": 0.006 + 0.031, "small": 0.005}  # 218/221
d_u = {"nephrops": 0.01, "pandalus": 0.01, "crabs": 0.05, "small": 0.02}               # 221 J
r_max = {"nephrops": 0.34, "pandalus": 1.2, "crabs": 0.9, "small": 1.86}               # 221
PB = {"nephrops": (0.3, 0.35, 0.4), "pandalus": (1.2, 1.45, 1.7), "crabs": (0.4, 0.7, 0.9),
      "small": (1.4, 1.95, 2.5)}
# exposure shares (refuge note, tables/refuge_parameters.csv); predators: small crabs only (x 0.25)
f_b = {"nephrops": 0.07, "pandalus": 0.75, "crabs": 0.6, "small": 0.6}
f_p = {"nephrops": 0.0, "pandalus": 0.75, "crabs": 0.6 * 0.25, "small": 0.6}

lines = []


def say(s=""):
    print(s)
    lines.append(s)


say(f"variant {variant}; mean effort {Em:.4f} MW-h/km2/yr; SAR at mean effort {SAR:.4f}")
say(f"benthivore consumption {C_B:.4f} t/km2/yr, epifauna share {share_E:.4f} "
    f"(measured {measured_E:.4f} + booked {sum(BOOKED.values()) / NORM:.4f}); predators {C_P:.4f}, "
    f"epifauna share {SHARE_P:.4f}")
say(f"crabs raised (221, ration Q/B 3.48) to {stock['crabs'].mean():.3f} t/km2")

# where the booked prey go
if MASSBAL:
    # krill/hyperbenthos and the carrion slice are small shrimps/mysids in kind (J)
    for k in ("krill_hyperbenthos", "carrion_E"):
        flow["small"]["benthivorous_fish"] += booked_flow[k]
    # cephalopods + small demersal fish (+ unid. fish slice): own member, B = flow / (EE P/B)
    # P/B: Mackinson & Daskalov 2007 Table 11.x North Sea: squid & cuttlefish 4.5 (4), small demersal
    # fish 1.42, dragonets 1.5 (1.44). Stocks there: squid 0.08, small demersal fish 0.34, dragonets 0.045.
    fl_ceph = booked_flow["cephalopod"]
    fl_sdf = booked_flow["small_demersal_fish"] + booked_flow["unid_fish_E"]
    B_ceph, B_sdf = fl_ceph / (EE * 4.5), fl_sdf / (EE * 1.45)
    Bb = B_ceph + B_sdf
    stock["booked"] = np.full_like(dep, Bb)            # flat with depth (J)
    flow["booked"] = {"benthivorous_fish": fl_ceph + fl_sdf, "predatory_fish": 0.0}
    catch["booked"] = 0.0
    d_u["booked"] = 0.02                                 # J, as small shrimps
    r_max["booked"] = (B_ceph * 3.0 + B_sdf * 1.4) / Bb  # J: annual squid r ~3, small fish ~1.4
    PB["booked"] = (1.4, (B_ceph * 4.5 + B_sdf * 1.45) / Bb, 4.5)
    f_b["booked"] = 0.6                                  # J
    f_p["booked"] = 0.0                                  # predators' diets do not show them on the bed (J)
    say(f"massbal: booked cephalopods {fl_ceph:.3f} t/km2/yr -> B {B_ceph:.3f} at P/B 4.5; small demersal fish "
        f"{fl_sdf:.3f} -> B {B_sdf:.3f} at P/B 1.45 (M&D 2007 North Sea stocks 0.08 / 0.34 + 0.045)")
    # raise crabs and small shrimps/mysids to the mass-balance stock at the central P/B if needed
    for k in ("crabs", "small"):
        Cf = sum(flow[k].values()) + (UNRES_SMALL if k == "small" else 0.0)
        target = (Cf + catch[k]) / (EE * PB[k][1] - d_u[k] * SAR)
        if target > stock[k].mean():
            say(f"massbal: {k} raised {stock[k].mean():.3f} -> {target:.3f} t/km2 (consumption {Cf:.3f} + catch "
                f"{catch[k]:.3f}, P/B {PB[k][1]})")
            stock[k] = stock[k] * target / stock[k].mean()
EXPOSED = ["crabs", "pandalus", "small"] + (["booked"] if MASSBAL else [])
GROUPS = {"mobile_epifauna": EXPOSED, "norway_lobster": ["nephrops"]}
extra_exposed = 0.0 if MASSBAL else sum(booked_flow.values())   # booked without stock

say("-" * 100)
say("Per-taxon flows and mass balance B_min = (C + catch)/(EE P/B - d_u SAR) at P/B high-central-low")
for k in EXPOSED + ["nephrops"]:
    Cf = sum(flow[k].values()) + (UNRES_SMALL if k == "small" else 0.0)
    bm = []
    for pb in (PB[k][2], PB[k][1], PB[k][0]):
        den = EE * pb - d_u[k] * SAR
        bm.append((Cf + catch[k]) / den if den > 0 else float("inf"))
    say(f"  {k:<9} stock {stock[k].mean():6.3f}  consumption {Cf:6.3f} (benth {flow[k]['benthivorous_fish']:.4f}, "
        f"pred {flow[k]['predatory_fish']:.4f}{', unresolved %.2f' % UNRES_SMALL if k == 'small' else ''})  "
        f"catch {catch[k]:.3f}  B_min {bm[0]:.2f} / {bm[1]:.2f} / {bm[2]:.2f}")
if extra_exposed:
    say(f"  booked prey without a stock, on the exposed compartment: {extra_exposed:.4f} t/km2/yr ("
        + ", ".join(f"{k} {v:.4f}" for k, v in booked_flow.items()) + ")")

# ---------------------------------------------------------------------------------------------------
# 3. Groups: F, M2, r, K table (NNLS), start totals
# ---------------------------------------------------------------------------------------------------
# rows at the band-mean depths (221); Nephrops gets an extra row at 300 m, where the UWTV grounds end
# ("flat to ~250 m, ~0 below 300 m", evidence_A2) - without it the 272 -> 387 m interpolation spreads
# Nephrops over the 300-400 m cells; the table holds its last row (300 m) below it
ROWS_G = {"mobile_epifauna": [81.0, 123.0, 173.0, 221.0, 272.0, 387.0],
          "norway_lobster": [81.0, 123.0, 173.0, 221.0, 272.0, 300.0]}
BANDS = [(0, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 600)]
W_TOTAL = 1.0e3               # weight of the extra equation that keeps sum B*(c) = today's total


def taxon_K(k, M2_k):
    B = stock[k].mean()
    F = catch[k] / B + d_u[k] * SAR
    rg = r_max[k] + M2_k
    s = 1 - (F + M2_k) / rg
    return B / s, s, F


res = {"variant": variant, "groups": {}, "pairs": {}}
for g, keys in GROUPS.items():
    ROWS = ROWS_G[g]
    basis = np.stack([np.interp(dep, ROWS, np.eye(len(ROWS))[i]).ravel() for i in range(len(ROWS))], axis=1)
    Bc = sum(stock[k] for k in keys)
    Bm = Bc.mean()
    loss_t = sum(catch[k] + d_u[k] * SAR * stock[k].mean() for k in keys)
    Fg = loss_t / Bm
    bf100 = Fg * 100.0 / (TPY * Em)
    Fc = Fg * E / Em
    pf = {p: sum(flow[k][p] for k in keys) for p in ("benthivorous_fish", "predatory_fish")}
    if g == "mobile_epifauna":
        pf["benthivorous_fish"] += extra_exposed
    r_bw = sum(r_max[k] * stock[k].mean() for k in keys) / Bm
    # taxon K at the taxon M2 (booked prey without a stock spread over the exposed taxa by stock, J)
    Ksum = 0.0
    for k in keys:
        fk = sum(flow[k].values()) + (extra_exposed * stock[k].mean() / Bm if g == "mobile_epifauna" else 0.0)
        Ksum += taxon_K(k, fk / stock[k].mean())[0]
    M2g0 = sum(pf.values()) / Bm
    r_self = (Fg + M2g0) / (1 - Bm / Ksum) - M2g0
    rm = r_bw                                     # literature, stock-weighted (central choice, J)
    Bstar_mean = Bm
    for it in range(20):
        M2p = {p: v / Bstar_mean for p, v in pf.items()}
        M2g = sum(M2p.values())
        rg = rm + M2g
        keep = np.clip(1 - (Fc + M2g) / rg, 0, 1)
        A = basis * keep.ravel()[:, None]
        # NNLS (a negative row is rejected by habitat.parse_capacity_table), plus one heavily weighted
        # equation for the total, so the start total stays today's stock (the pattern is least squares)
        A2 = np.vstack([A, W_TOTAL * A.sum(axis=0) / A.shape[0]])
        b2 = np.concatenate([Bc.ravel(), [W_TOTAL * Bc.sum() / A.shape[0]]])
        rows, _ = nnls(A2, b2)
        Kc = np.interp(dep, ROWS, rows)
        Bstar = Kc * keep
        if abs(Bstar.mean() - Bstar_mean) < 1e-9:
            break
        Bstar_mean = Bstar.mean()
    rms = float(np.sqrt(np.mean((Bstar - Bc) ** 2)) / Bc.mean())
    share = 1 - (Fg + M2g) / rg
    say("=" * 100)
    say(f"== {g}  (members: {', '.join(keys)})")
    say("   members (share of today's stock): " + ", ".join(f"{k} {stock[k].mean() / Bm:.3f} ({stock[k].mean():.3f})"
                                                        for k in keys))
    say(f"   today's stock grid mean {Bm:.4f} t/km2; total {Bc.sum():.0f} t")
    say(f"   trawl F at mean effort {Fg:.4f} /yr (catch {sum(catch[k] for k in keys):.3f} + kill "
        f"{sum(d_u[k] * SAR * stock[k].mean() for k in keys):.3f} t/km2/yr); bf100 {bf100:.6f}; F(c) max "
        f"{Fc.max():.3f}")
    say(f"   flows: " + ", ".join(f"{p} {v:.4f}" for p, v in pf.items()) + " t/km2/yr")
    say(f"   M2 on the engine B* mean {Bstar.mean():.4f}: " + ", ".join(f"{p} {v:.4f}" for p, v in M2p.items())
        + f" (sum {M2g:.4f})")
    say(f"   r_max: stock-weighted {r_bw:.4f} (used); self-consistent with sum of taxon K {r_self:.4f} "
        f"(sum K {Ksum:.3f})")
    say(f"   r_gross {rg:.4f} -> growth_rate {rg / TPY:.11g} per tick; equilibrium share {share:.4f}; "
        f"keep(c) min {keep.min():.3f}")
    say(f"   K rows (NNLS): " + ", ".join(f"{d:.0f}: {k:.2f}" for d, k in zip(ROWS, rows))
        + f"; box-mean K {Kc.mean():.3f}")
    say(f"   engine B*(c): mean {Bstar.mean():.4f}, min {Bstar.min():.3f}, max {Bstar.max():.3f}; START TOTAL "
        f"{Bstar.sum():.0f} t (+/-25 %: {0.75 * Bstar.sum():.0f}-{1.25 * Bstar.sum():.0f}); rms deviation "
        f"{100 * rms:.1f} % of the mean; zero cells {(Bstar <= 0).sum()}")
    for lo, hi in BANDS:
        m = (dep >= lo) & (dep < hi)
        say(f"     {lo:>3}-{hi:<3} m ({m.sum():4d} cells): today {Bc[m].mean():7.3f}  engine B* {Bstar[m].mean():7.3f}"
            f"  K {Kc[m].mean():7.3f}  effort {E[m].mean():5.2f}")
    np.save(out_json.replace(".json", f"_{g}_today.npy"), Bc)    # today's stock map, the K-fit target
    res["groups"][g] = dict(members={k: float(stock[k].mean()) for k in keys}, B_today=float(Bm),
                            F=float(Fg), bf100=float(bf100), flows=pf, M2=M2p, M2_sum=float(M2g),
                            r_max=float(rm), r_max_bw=float(r_bw), r_max_self=float(r_self),
                            r_gross=float(rg), growth_rate=float(rg / TPY), share=float(share),
                            K_rows=[[d, float(k)] for d, k in zip(ROWS, rows)], K_mean=float(Kc.mean()),
                            Bstar_mean=float(Bstar.mean()), start=float(Bstar.sum()), rms=rms,
                            Fc_max=float(Fc.max()), keep_min=float(keep.min()))

# ---------------------------------------------------------------------------------------------------
# 4. Pairs: f, h, a, saturation
# ---------------------------------------------------------------------------------------------------
CEIL = {"benthivorous_fish": 27.1, "predatory_fish": 25.5}          # MJ/t/tick (section 148)
E_GROUP = {"mobile_epifauna": 3750.0, "norway_lobster": 3600.0}     # energy_content.py of this note
A = {"benthivorous_fish": 0.44, "predatory_fish": 0.33}             # per predator tonne, 221 encounter
say("=" * 100)
say("Pairs (f from the taxon exposure shares weighted by stock; h = E / ceiling; x = a h f B*)")
for g, keys in GROUPS.items():
    Bm = sum(stock[k].mean() for k in keys)
    Bs = res["groups"][g]["Bstar_mean"]
    for p, fd in (("benthivorous_fish", f_b), ("predatory_fish", f_p)):
        f = sum(fd[k] * stock[k].mean() for k in keys) / Bm
        if f <= 0:
            say(f"  {p} -> {g}: no access (unchecked)")
            continue
        variants = [("", f)]
        if g == "norway_lobster":
            variants.append((" x size window 0.29", f * 0.29))
        for lab, ff in variants:
            h = E_GROUP[g] / CEIL[p]
            x = A[p] * h * ff * Bs
            say(f"  {p} -> {g}{lab}: f {ff:.4f}, h {h:.2f}, a {A[p]}, M2 {res['groups'][g]['M2'][p]:.5f}; "
                f"half-saturation 1/(a h f) {1 / (A[p] * h * ff):.4f} t/km2 of the group; x at B* {x:.1f} -> "
                f"{x / (1 + x):.3f} of the ceiling; at 10 % of B* {x / 10 / (1 + x / 10):.3f}")
            # demand at the ceiling vs the group's max surplus
            rg, K = res["groups"][g]["r_gross"], res["groups"][g]["K_mean"]
            dem = (1.0 / h) * TPY * {"benthivorous_fish": BB, "predatory_fish": BP}[p]
            say(f"      demand at the ceiling {dem:.1f} t/km2/yr vs max surplus r_gross K/4 {rg * K / 4:.2f}, net of "
                f"trawl (r-F)^2 K/(4 r) {(rg - res['groups'][g]['F']) ** 2 * K / (4 * rg):.2f}; budget flow "
                f"{res['groups'][g]['flows'][p]:.3f}")
            res["pairs"][f"{p}_preys_on_{g}{lab.replace(' ', '_')}"] = dict(
                f=float(ff), h=float(h), a=A[p], M2=float(res["groups"][g]["M2"][p]), x=float(x))

# ---------------------------------------------------------------------------------------------------
# 5. Independent checks (not used to set values)
# ---------------------------------------------------------------------------------------------------
say("=" * 100)
say("Share of production eaten by the modelled fish = fish flow / (P/B_lit x B), P/B central (low-high):")
say("  bounds (refuge note, M&D 2007 recomputed): Nephrops 65-94 %, large crabs 60-92 %, shrimp 15-39 %")
for k in EXPOSED + ["nephrops"]:
    Cf = flow[k]["benthivorous_fish"] + flow[k]["predatory_fish"]
    B = stock[k].mean()
    say(f"  {k:<9} {100 * Cf / (PB[k][1] * B):5.1f} % ({100 * Cf / (PB[k][2] * B):.1f}-{100 * Cf / (PB[k][0] * B):.1f}); "
        f"taxon M2 {Cf / B:.4f} /yr")
gE = res["groups"]["mobile_epifauna"]
Cex = sum(gE["flows"].values())
pb_bw = sum(PB[k][1] * stock[k].mean() for k in EXPOSED) / gE["B_today"]
say(f"  exposed group: fish {Cex:.3f} t/km2/yr incl. booked prey; P/B stock-weighted {pb_bw:.3f}; "
    f"fish/P {100 * Cex / (pb_bw * gE['B_today']):.0f} %; r_gross {gE['r_gross']:.3f} vs P/B {pb_bw:.3f} "
    f"(x {gE['r_gross'] / pb_bw:.2f}); M2/(F+M2) {gE['M2_sum'] / (gE['F'] + gE['M2_sum']):.3f}")
gN = res["groups"]["norway_lobster"]
say(f"  Nephrops M2 by benthivores {gN['M2']['benthivorous_fish']:.4f} /yr vs Irish Sea cod 0.04-0.25 (western "
    f"Irish Sea; refuge note part D); M2/(F+M2) {gN['M2_sum'] / (gN['F'] + gN['M2_sum']):.3f}")
res["checks"] = dict(pb_bw_exposed=float(pb_bw), fish_flow_exposed=float(Cex),
                     booked_flow=booked_flow, extra_exposed=float(extra_exposed))
res["stdout"] = lines
json.dump(res, open(out_json, "w"), indent=1, default=float)
