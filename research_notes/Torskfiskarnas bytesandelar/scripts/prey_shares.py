"""Stage-resolved gadoid prey-pool shares phi_benthos, phi_fish, phi_zooplankton for the Mareld box.

Usage: python3 -I prey_shares.py <note_dir> <species_mix_dir> [n_draws] [fixed]
("fixed" = no haul bootstrap; writes tables/phi_summary_fixedhauls.csv only)

<note_dir>        research_notes/Torskfiskarnas bytesandelar (inputs/, tables/ written by
                  length_biomass.py; results are written to tables/)
<species_mix_dir> research_notes/Torskfiskarnas artsammansättning (read only: minor-species shares in
                  tables/share_table.csv and tables/q_<variant>/share_table.csv)

phi_j = sum_i w_i [ (1 - p0_i) sum_k s_ik score_ikj + p0_i score_i,pel0,j ] + sum_m w_m score_mj
  w_i      box biomass share of main species i (haddock, whiting, saithe, cod, hake), depth-stratified
           NS-IBTS GOV Q1/Q3 2021-25 (same estimator as the species-mix note), per catchability variant;
  s_ik     share of species i's (survey) biomass in stage k, from the 1-cm length biomass and the stage
           boundaries in inputs/boundaries.csv (linear within a 1-cm class);
  p0_i     share of species i's biomass in the pelagic 0-group, which the GOV does not catch
           (inputs/pelagic0.csv, per quarter; 'pooled' = annual mean);
  score    = access x capability (inputs/scores.csv), each min / central / max in [0, 1], from
           morphology and habitat only;
  w_m      minor species (ling, pollack, rocklings, ...) at their share in the species-mix note's
           depth-stratified table, scored per species (inputs/minor_scores.csv).

Pools: benthos, fish (pelagic_fish), zoo (krill-sized prey), zoo_cop (copepod-sized prey), and two
sensitivity pools fish_g_bio / fish_g_num = fish access x mouth factor (inputs/gape.csv mode_*) x the
share of today's near-bottom pelagic prey (tables/pelagic_prey_length_biomass.csv, by biomass or by
numbers) a stage can swallow, prey length <= k x predator length (inputs/gape.csv k_*). The central
fish pool uses the stepped gape rule folded into C in inputs/scores.csv; the scores.csv "gape" column
(1 = multiply the central fish score by the computed gape share) is kept but set to 0 everywhere.

Uncertainty: Monte Carlo over triangular(min, c, max) access and capability factors, triangular
boundaries, triangular p0, and a haul bootstrap within quarter x depth bin; per catchability variant
(raw = central; Fraser A_floor and A_sw_floor). Corner bounds (all factors at min / at max) are also
given. One-at-a-time swings rank the inputs that move each phi most.
"""
import os
import sys

import numpy as np
import pandas as pd

note, mix = sys.argv[1], sys.argv[2]
ND = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
FIXED = len(sys.argv) > 4 and sys.argv[4] == "fixed"  # no haul bootstrap: width from scores/boundaries only
SUF = "_fixedhauls" if FIXED else ""
inp, tab = os.path.join(note, "inputs"), os.path.join(note, "tables")
rng = np.random.default_rng(208)

MAIN = ["haddock", "whiting", "saithe", "cod", "hake"]
POOLS = ["benthos", "fish", "zoo", "zoo_cop", "fish_g_bio", "fish_g_num"]
GPOOLS = {"fish_g_bio": "kg_h", "fish_g_num": "n_h"}
BINS = ["<100", "100-200", "200-300", ">=300"]
VARS = {"raw": "kg_h", "A_floor": "kg_A_floor", "A_sw_floor": "kg_A_sw_floor"}
QDIR = {"raw": "tables", "A_floor": "tables/q_A_floor", "A_sw_floor": "tables/q_A_sw_floor"}

BND = pd.read_csv(os.path.join(inp, "boundaries.csv"), index_col=0)
STG = pd.read_csv(os.path.join(inp, "stages.csv"))
SC = pd.read_csv(os.path.join(inp, "scores.csv"))
MIN = pd.read_csv(os.path.join(inp, "minor_scores.csv"))
P0 = pd.read_csv(os.path.join(inp, "pelagic0.csv"))
GP = pd.read_csv(os.path.join(inp, "gape.csv"), index_col=0)
GAPE = {(r.species, r.stage) for r in SC.itertuples() if r.pool == "fish" and r.gape == 1}
FISHA = SC[(SC.pool == "fish") & (SC.stage != "pelagic0")]
# cumulative biomass share of the pelagic_fish pool (herring, sprat, mackerel, horse mackerel) by prey
# length, near-bottom GOV catches, depth-stratified (pelagic_prey_sizes.py)
PL = pd.read_csv(os.path.join(tab, "pelagic_prey_length_biomass.csv"))
PREY = {}
for col in ["kg_h", "n_h"]:
    for q in [1, 3]:
        d = PL[PL.quarter == q].set_index("L_cm")[col].reindex(range(0, 200), fill_value=0.0).values
        PREY[(col, q)] = d / d.sum()
    PREY[(col, "pooled")] = 0.5 * (PREY[(col, 1)] + PREY[(col, 3)])
PREY_MID = np.arange(200) + 0.5


def gape_share(vec_stage, q, k, col="kg_h"):
    """biomass-weighted share of the pool's prey biomass a predator stage can swallow (prey <= k L)."""
    L = np.arange(NL) + 0.5
    n = np.clip(np.floor(k * L - 0.5).astype(int) + 1, 0, 200)  # prey classes with midpoint <= k L
    cum = np.concatenate([[0.0], np.cumsum(PREY[(col, q)])])[n]
    tot = vec_stage.sum()
    return (vec_stage * cum).sum() / tot if tot > 0 else 0.0

area = pd.read_csv(os.path.join(mix, "tables", "box_area_by_depth.csv"), index_col=0).iloc[:, 0].reindex(BINS)

HL = pd.read_csv(os.path.join(tab, "haul_length_kgh.csv.gz"))
HU = pd.read_csv(os.path.join(tab, "hauls_used.csv"))
HU["hid"] = np.arange(len(HU))
HL = HL.merge(HU[["Year", "Quarter", "Ship", "HaulNo", "hid"]], on=["Year", "Quarter", "Ship", "HaulNo"])
LMAX = 130
NL = LMAX + 1
COLS = [(s, l) for s in MAIN for l in range(NL)]


def haul_matrix(col):
    """hauls x (species, 1-cm class) kg/h matrix."""
    M = np.zeros((len(HU), len(MAIN) * NL))
    si = HL.species.map({s: i for i, s in enumerate(MAIN)}).values
    li = np.clip(HL.Lc.values, 0, LMAX)
    np.add.at(M, (HL.hid.values, si * NL + li), HL[col].values)
    return M


MAT = {v: haul_matrix(c) for v, c in VARS.items()}
STRATA = {(q, d): HU.index[(HU.Quarter == q) & (HU.dbin == d)].values for q in [1, 3] for d in BINS}


def stratified(M, idx=None):
    """dict quarter -> (species, L) kg/h vector, depth-stratified; >=300 m takes 200-300 m."""
    out = {}
    for q in [1, 3]:
        acc = np.zeros(M.shape[1])
        for d in BINS:
            dd = "200-300" if d == ">=300" else d
            rows = STRATA[(q, dd)] if idx is None else idx[(q, dd)]
            acc += area[d] * M[rows].mean(axis=0)
        out[q] = acc
    out["pooled"] = 0.5 * (out[1] + out[3])
    return out


def minor_shares(var):
    t = pd.read_csv(os.path.join(mix, QDIR[var], "share_table.csv"))
    t = t[(t.estimator == "depth") & (t["group"] == "core") & (t["corr"] == "raw")]
    res = {}
    for q in ["1", "3", "pooled"]:
        r = t[t.quarter.astype(str) == q].iloc[0]
        res[q] = {m: r[m] / 100.0 for m in MIN.species.unique()}
    return res


MINS = {v: minor_shares(v) for v in VARS}


def stage_frac(lo, hi):
    edges = np.arange(NL)
    return np.clip(np.minimum(edges + 1, hi) - np.maximum(edges, lo), 0, 1)


def stage_share(vec, lo, hi):
    """share of a species' biomass (1-cm vector) between lengths lo and hi (cm), linear in class."""
    tot = vec.sum()
    return (vec * stage_frac(lo, hi)).sum() / tot if tot > 0 else 0.0


def tri(lo, c, hi, n):
    lo, c, hi = np.broadcast_arrays(np.asarray(lo, float), np.asarray(c, float), np.asarray(hi, float))
    out = np.where(hi > lo, 0.0, c)
    m = hi > lo
    if m.any():
        out = out.copy()
        out[m] = rng.triangular(lo[m], np.clip(c[m], lo[m], hi[m]), hi[m])
    return out


def bound(name, b):
    if isinstance(name, str) and name in b:
        return b[name]
    return float(name)


def compute(strat, q, var, bvals, scores, p0, mscore, kv):
    """phi per pool and the contribution table for one quarter key."""
    vec = strat[q]
    tot = np.array([vec[i * NL:(i + 1) * NL].sum() for i in range(len(MAIN))])
    mn = MINS[var][str(q)]
    msum = sum(mn.values())
    w = (1 - msum) * tot / tot.sum()
    phi = dict.fromkeys(POOLS, 0.0)
    contrib = []
    for i, s in enumerate(MAIN):
        v = vec[i * NL:(i + 1) * NL]
        pz = p0[(s, str(q))]
        for _, r in STG[STG.species == s].iterrows():
            fr = stage_frac(bound(r.lo, bvals), bound(r.hi, bvals))
            sh = (v * fr).sum() / v.sum() if v.sum() > 0 else 0.0
            g = gape_share(v * fr, q, kv[s]) if (s, r.stage) in GAPE else 1.0
            for p in POOLS:
                f = g if p == "fish" else 1.0
                if p in GPOOLS:
                    f = gape_share(v * fr, q, kv[s], GPOOLS[p])
                c = w[i] * (1 - pz) * sh * scores[(s, r.stage, p)] * f
                phi[p] += c
                contrib.append((s, r.stage, p, w[i] * (1 - pz) * sh, c))
        for p in POOLS:
            c = w[i] * pz * scores[(s, "pelagic0", p)]
            phi[p] += c
            contrib.append((s, "pelagic0", p, w[i] * pz, c))
    for m, wm in mn.items():
        for p in POOLS:
            c = wm * mscore[(m, p)]
            phi[p] += c
            contrib.append((m, "all", p, wm, c))
    return phi, contrib


def add_gpools(sc, ms, idx, a_vals, mode):
    """sensitivity pools: access x mode factor x computed gape share (biomass or numbers basis)."""
    for j, (s, st) in enumerate(idx.values()):
        for p in GPOOLS:
            sc[(s, st, p)] = a_vals[j] * mode[s]
    for s in MAIN:
        for p in GPOOLS:
            sc[(s, "pelagic0", p)] = 0.0
    for m in MIN.species.unique():
        for p in GPOOLS:
            ms[(m, p)] = ms[(m, "fish")]


def central_inputs(mode="c"):
    b = {k: BND.loc[k, mode] for k in BND.index}
    k = {"c": "c", "min": "min", "max": "max"}[mode]
    sc = {(r.species, r.stage, r.pool): r["A_" + k] * r["C_" + k] for _, r in SC.iterrows()}
    p0 = {(r.species, str(r.quarter)): r["share_" + k] for _, r in P0.iterrows()}
    ms = {(r.species, r.pool): r["s_" + k] for _, r in MIN.iterrows()}
    kv = {s: GP.loc[s, "k_" + k] for s in GP.index}
    add_gpools(sc, ms, {r.Index: (r.species, r.stage) for r in FISHA.itertuples()},
               FISHA["A_" + k].values, GP["mode_" + k])
    return b, sc, p0, ms, kv


QS = [1, 3, "pooled"]
rows, contrib_rows, stage_rows = [], [], []
base = {v: stratified(MAT[v]) for v in VARS}
for v in VARS:
    for mode in ["c", "min", "max"]:
        b, sc, p0, ms, kv = central_inputs(mode)
        for q in QS:
            phi, con = compute(base[v], q, v, b, sc, p0, ms, kv)
            rows.append(dict(variant=v, quarter=q, kind="corner_" + mode if mode != "c" else "central", **phi))
            if mode == "c":
                for s, st, p, wst, c in con:
                    contrib_rows.append(dict(variant=v, quarter=q, species=s, stage=st, pool=p,
                                             biomass_share=wst, contribution=c))
# Monte Carlo
mc = []
for v in VARS:
    for it in range(ND):
        idx = {key: rng.choice(rows_, size=len(rows_), replace=True) for key, rows_ in STRATA.items()}
        st = base[v] if FIXED else stratified(MAT[v], idx)
        b = {k: float(tri(BND.loc[k, "min"], BND.loc[k, "c"], BND.loc[k, "max"], 1)) for k in BND.index}
        A = tri(SC.A_min.values, SC.A_c.values, SC.A_max.values, len(SC))
        C = tri(SC.C_min.values, SC.C_c.values, SC.C_max.values, len(SC))
        sc = {(r.species, r.stage, r.pool): A[j] * C[j] for j, r in enumerate(SC.itertuples())}
        pz = tri(P0.share_min.values, P0.share_c.values, P0.share_max.values, len(P0))
        p0 = {(r.species, str(r.quarter)): pz[j] for j, r in enumerate(P0.itertuples())}
        ms_ = tri(MIN.s_min.values, MIN.s_c.values, MIN.s_max.values, len(MIN))
        ms = {(r.species, r.pool): ms_[j] for j, r in enumerate(MIN.itertuples())}
        kd = tri(GP.k_min.values, GP.k_c.values, GP.k_max.values, len(GP))
        kv = {s: kd[j] for j, s in enumerate(GP.index)}
        md = tri(GP.mode_min.values, GP.mode_c.values, GP.mode_max.values, len(GP))
        fa = A[((SC.pool == "fish") & (SC.stage != "pelagic0")).values]
        add_gpools(sc, ms, {r.Index: (r.species, r.stage) for r in FISHA.itertuples()}, fa,
                   pd.Series(md, index=GP.index))
        for q in QS:
            phi, _ = compute(st, q, v, b, sc, p0, ms, kv)
            mc.append(dict(variant=v, quarter=q, **phi))
MC = pd.DataFrame(mc)
for (v, q), g in MC.groupby(["variant", MC.quarter.astype(str)]):
    for kind, f in [("p05", 0.05), ("p50", 0.5), ("p95", 0.95)]:
        rows.append(dict(variant=v, quarter=q, kind=kind, **g[POOLS].quantile(f).to_dict()))
R = pd.DataFrame(rows)
R["quarter"] = R.quarter.astype(str)
R.to_csv(os.path.join(tab, f"phi_summary{SUF}.csv"), index=False)
if FIXED:
    print(R.pivot_table(index=["variant", "quarter"], columns="kind", values=POOLS).round(3).to_string())
    sys.exit(0)
CT = pd.DataFrame(contrib_rows)
CT.to_csv(os.path.join(tab, "phi_contributions.csv"), index=False)

# stage shares (central boundaries), all variants and quarters
b, *_ = central_inputs("c")
for v in VARS:
    for q in QS:
        vec = base[v][q]
        for i, s in enumerate(MAIN):
            vv = vec[i * NL:(i + 1) * NL]
            for _, r in STG[STG.species == s].iterrows():
                stage_rows.append(dict(variant=v, quarter=q, species=s, stage=r.stage,
                                       share=stage_share(vv, bound(r.lo, b), bound(r.hi, b))))
pd.DataFrame(stage_rows).to_csv(os.path.join(tab, "stage_shares.csv"), index=False)

# one-at-a-time swings (raw, pooled): each access / capability factor, boundary, p0, minor score
b0, sc0, p00, ms0, kv0 = central_inputs("c")
phi0, _ = compute(base["raw"], "pooled", "raw", b0, sc0, p00, ms0, kv0)
sw = []
for _, r in SC.iterrows():
    key = (r.species, r.stage, r.pool)
    for f in ["A", "C"]:
        other = r["C_c"] if f == "A" else r["A_c"]
        lo, hi = dict(sc0), dict(sc0)
        lo[key], hi[key] = r[f + "_min"] * other, r[f + "_max"] * other
        pl, _ = compute(base["raw"], "pooled", "raw", b0, lo, p00, ms0, kv0)
        ph, _ = compute(base["raw"], "pooled", "raw", b0, hi, p00, ms0, kv0)
        sw.append(dict(input=f"{f} {r.species}/{r.stage}/{r.pool}", pool=r.pool, low=pl[r.pool] - phi0[r.pool],
                       high=ph[r.pool] - phi0[r.pool]))
for k in BND.index:
    for p in POOLS:
        bl, bh = dict(b0), dict(b0)
        bl[k], bh[k] = BND.loc[k, "min"], BND.loc[k, "max"]
        pl, _ = compute(base["raw"], "pooled", "raw", bl, sc0, p00, ms0, kv0)
        ph, _ = compute(base["raw"], "pooled", "raw", bh, sc0, p00, ms0, kv0)
        sw.append(dict(input=f"boundary {k}", pool=p, low=pl[p] - phi0[p], high=ph[p] - phi0[p]))
for sp in GP.index:
    kl, kh = dict(kv0), dict(kv0)
    kl[sp], kh[sp] = GP.loc[sp, "k_min"], GP.loc[sp, "k_max"]
    pl, _ = compute(base["raw"], "pooled", "raw", b0, sc0, p00, ms0, kl)
    ph, _ = compute(base["raw"], "pooled", "raw", b0, sc0, p00, ms0, kh)
    sw.append(dict(input=f"gape k {sp}", pool="fish_g_bio", low=pl["fish_g_bio"] - phi0["fish_g_bio"],
                   high=ph["fish_g_bio"] - phi0["fish_g_bio"]))
# gape shares per stage (central), for the record
grow = []
for q in QS:
    for i, s_ in enumerate(MAIN):
        v_ = base["raw"][q][i * NL:(i + 1) * NL]
        for _, r in STG[STG.species == s_].iterrows():
            fr = stage_frac(bound(r.lo, b0), bound(r.hi, b0))
            grow.append(dict(quarter=q, species=s_, stage=r.stage,
                             gape_share_biomass=gape_share(v_ * fr, q, kv0[s_], "kg_h"),
                             gape_share_numbers=gape_share(v_ * fr, q, kv0[s_], "n_h")))
pd.DataFrame(grow).to_csv(os.path.join(tab, "gape_shares.csv"), index=False)
for v in ["A_floor", "A_sw_floor"]:
    pv, _ = compute(base[v], "pooled", v, b0, sc0, p00, ms0, kv0)
    for p in POOLS:
        sw.append(dict(input=f"catchability {v}", pool=p, low=pv[p] - phi0[p], high=pv[p] - phi0[p]))
for q in [1, 3]:
    pv, _ = compute(base["raw"], q, "raw", b0, sc0, p00, ms0, kv0)
    for p in POOLS:
        sw.append(dict(input=f"season Q{q}", pool=p, low=pv[p] - phi0[p], high=pv[p] - phi0[p]))
SW = pd.DataFrame(sw)
SW["span"] = (SW.high - SW.low).abs().where(~SW.input.str.startswith(("catch", "season")), SW.high.abs())
SW = SW.sort_values(["pool", "span"], ascending=[True, False])
SW.to_csv(os.path.join(tab, "phi_swings.csv"), index=False)

pd.set_option("display.width", 250, "display.max_rows", 400, "display.max_columns", 20)
print("central phi (raw, pooled):", {k: round(x, 3) for k, x in phi0.items()})
print(R.pivot_table(index=["variant", "quarter"], columns="kind", values=POOLS).round(3).to_string())
print("\nstage shares, raw / A_floor, pooled")
S = pd.DataFrame(stage_rows)
print(S[S.quarter.astype(str).isin(["1", "3", "pooled"])].pivot_table(index=["species", "stage"],
                                                                         columns=["variant", "quarter"],
                                                                         values="share").round(3).to_string())
print("\ncontributions (raw, pooled, central)")
c = CT[(CT.variant == "raw") & (CT.quarter.astype(str) == "pooled")]
print(c.pivot_table(index=["species", "stage"], columns="pool", values="contribution", aggfunc="sum").round(3)
      .to_string())
print("\nlargest swings per pool")
for p in POOLS:
    print(SW[SW.pool == p].head(8)[["input", "low", "high"]].round(3).to_string(index=False))
