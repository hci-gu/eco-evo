"""Derived monthly attack rate (vision rule A on the band overlap, section 269 form) of every pair with a dark_ratio,
against the engine's daylight multiplier m(t), under several readings of the winter light.

Usage: python3 light_pairs.py <repo_root> <scratch_dir> <out_dir>      (~15-30 min)

Nothing is fitted and no model output is read. Nothing in the repository is written.

Machinery (exec'd, unchanged): band_pairs_copy.py = research_notes/"Overlapp i fiskparen"/scripts/band_pairs.py
(identical copy), head up to its overlap-geometry block; it execs research_notes/"En synregel for fiskparen"/
scripts/one_rule_pairs.py (head) from its original place, which execs krill_pairs_wc.py ("Krill i Skagerrak") and
split_encounter_223.py ("Rovfiskens uppdelning"). Same state as band_pairs.py's A_band run and seasonal_pairs.py
("Sillens vinterprofil").

What changes per light reading (and nothing else):
  one_rule_pairs.trans_month  -> trans_reading (monthly K(2-20) from the reading's Secchi table via K = 1.56/Secchi x
                                 k2_mult, K(20-50) per month, K(50-65) 0.091, K_deep 0.06, as the original)
  krill_pairs_wc.E0 (in place) -> the reading's surface illuminance (lx) per time row
Caches of the light (one_rule_pairs GCACHE / ACACHE) are cleared between readings.

Series per pair (cell-weighted over the box's bed depths, equal cells; per time row of the derivation grid: 2 days
per month, 15-min steps at 58.15 N):
  fish pairs (7)        band_fish_pair (vision A), returning the per-time series (copy of the function below)
  benthivore fish (2)   one_rule_pairs.benth_V(pair, "A", Ke 0.02, K_deep 0.06) = sum_L w_L R_L^2(E at the bed)
  krill pairs (5)       krill_pairs_wc.evaluate under rule A (one_rule_pairs.run_krill state), a(t, cell)
  zooplankton pairs (4) not rule-A pairs: the engine's own form made intensity-aware,
                        rho + (1 - rho) g(E(t, z)), g the predator's eye (clupeid Ke 1 lx, biting threshold 0.016 lx;
                        gadoid Ke 0.02 lx) averaged over a judged layer [J]: planktivores at 15 m (planktivores_a.py
                        z_forage central; 8 / 25 m bracket), mackerel over its 0-40 m profile, whiting over its
                        bottom 10 m, water column over the saithe field profile.
Night/day: each derived series is scaled on the engine's night rows (sun <= the pair's light_threshold_deg) by one
factor so that its night/day ratio at that threshold equals the LIBRARY dark_ratio (the engine's), isolating the
within-day and seasonal intensity effect; the unmatched (rule's own night/day) residual is printed too.
Engine: lib/world/daylight.attack_multiplier(58.15, 6 h, rho, thr), monthly means binned by daylight.month_of_day.
"""
import contextlib
import io
import json
import math
import os
import sys

import numpy as np
import yaml

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
OUTD = os.path.abspath(sys.argv[3])
os.makedirs(SCR, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.join(ROOT, "research_notes")
MN = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
MDAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], float)

# ------------------------------------------------------------------ 1. machinery
BP = os.path.join(HERE, "band_pairs_copy.py")
src = open(BP).read()
head = src[:src.index("# ================================================================== 0. overlap geometry")]
save = list(sys.argv)
sys.argv = [os.path.join(NOTES, "Overlapp i fiskparen", "scripts", "band_pairs.py"), ROOT, SCR,
            os.path.join(SCR, "_unused_band.json")]
G = {"__name__": "band_pairs_head", "__file__": sys.argv[0]}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, BP, "exec"), G)
sys.argv = save
O, K = G["O"], G["K"]
ZG, ELEV, MONTH, BEDS, WCELL, CLS_IDX = G["ZG"], G["ELEV"], G["MONTH"], G["BEDS"], G["WCELL"], G["CLS_IDX"]
IS_NIGHT, STG_T, terms, sc, qf = G["IS_NIGHT"], G["STG_T"], G["terms"], G["sc"], G["qf"]
area_in_slab, gmat, rsat_fish, FISH_PAIRS = G["area_in_slab"], G["gmat"], G["rsat_fish"], G["FISH_PAIRS"]
WMUL, REST, EATERS, SLAB, SUMMER = G["WMUL"], G["REST"], G["EATERS"], G["SLAB"], G["SUMMER"]
NB, RNV_FISH, slab_series, point_series = G["NB"], G["RNV_FISH"], G["slab_series"], G["point_series"]
PRED_PROF, SEASONAL, pmat = G["PRED_PROF"], G["SEASONAL"], G["pmat"]
qk0 = dict(G["qk0"])
E0 = O["E0"]
assert E0 is K["E0"]
E0_BASE = E0.copy()

with contextlib.redirect_stdout(io.StringIO()):
    G["RULE"].update(rule="lib", RD=1.2, Lk=None)
    G["run_krill"](dict(K["level"](1)))
    G["RULE"].update(rule="A", RD=1.2, Lk=None)
    G["run_krill"](dict(K["level"](1), Ke_gad=0.02, k_deep=0.06))

# ------------------------------------------------------------------ 2. light readings
DERIV_SECCHI = list(O["SECCHI"])
WL = json.load(open(os.path.join(OUTD, "winter_light.json")))
SHAPED = WL["shaped_secchi"]
# manifest cloud transmission (SMHI Nordkoster global radiation / Haurwitz clear sky, resume 138.3)
MAN = yaml.safe_load(open(os.path.join(ROOT, "mareld2.yaml")))
CLOUD = np.array(MAN["simulation_settings"]["daylight"]["cloud_transmission"], float)
LAT = float(MAN["simulation_settings"]["daylight"]["latitude_deg"])
SEC_CAST_WINTER = 1.56 / 0.191          # Nov-Feb cast median K(2-20) 0.191 (winter_light_stdout, thr 0.5) -> 8.2 m


def secchi_minfix():
    s = list(DERIV_SECCHI)
    s[0] = 1.56 / 0.192                  # Jan: cast median K(2-20) 0.192 (n 10) replaces the n = 1 Secchi 5.0 m
    s[11] = SEC_CAST_WINTER              # Dec: no cast; n = 1 + 1 Secchi 11.5 m replaced by the Nov-Feb cast median
    return s


def surf_haurwitz():
    """surface lx = the derivation's 118.4 lm/W x Haurwitz (air mass) x the manifest's monthly cloud transmission;
    the twilight branch (sun < 5 deg) scaled continuously at 5 deg."""
    h = np.maximum(ELEV, 5.0)
    air = np.exp(-0.057 / np.sin(np.radians(h)))
    return E0_BASE * air * CLOUD[MONTH] / 0.5


def surf_castdark():
    """bracket: R3 surface, Nov-Feb darkened by the casts' PAR(2 m)/clear-sky ratio, winter over summer
    (0.067 / 0.207 = 0.32), relative to the manifest's own winter/summer cloud ratio [J, weak; ship shadow]."""
    e = surf_haurwitz()
    win = np.isin(MONTH, (10, 11, 0, 1))
    rel = (0.067 / 0.207) / (CLOUD[[10, 11, 0, 1]].mean() / CLOUD[[3, 4, 5, 6, 7, 8]].mean())
    e = e.copy()
    e[win] *= rel
    return e


K2050 = np.full(12, 0.109)
K2050_WIN = K2050.copy()
K2050_WIN[[11, 0, 1]] = 0.132           # Jan-Feb cast median K(20-50) (n 3), Dec-Feb

READINGS = {
    "R0 derivation as is": dict(secchi=DERIV_SECCHI, k2050=K2050, surf=lambda: E0_BASE),
    "R1 Jan/Dec K from casts": dict(secchi=secchi_minfix(), k2050=K2050, surf=lambda: E0_BASE),
    "R2 Slaggo-shaped Secchi": dict(secchi=SHAPED, k2050=K2050, surf=lambda: E0_BASE),
    "R3 R1 + Haurwitz x cloud": dict(secchi=secchi_minfix(), k2050=K2050, surf=surf_haurwitz),
    "R4 R3 + winter K20-50": dict(secchi=secchi_minfix(), k2050=K2050_WIN, surf=surf_haurwitz),
    "R5 R3 + dark winter surface": dict(secchi=secchi_minfix(), k2050=K2050, surf=surf_castdark),
    "R6 R2 + Haurwitz x cloud": dict(secchi=SHAPED, k2050=K2050, surf=surf_haurwitz),
}
CUR = {}


def trans_reading(q):
    z = ZG
    out = np.empty((12, len(z)))
    for m in range(12):
        k2 = 1.56 / CUR["secchi"][m] * q["k2_mult"]
        tau = (0.30 * np.minimum(z, 2) + k2 * np.clip(np.minimum(z, 20) - 2, 0, None)
               + CUR["k2050"][m] * np.clip(np.minimum(z, 50) - 20, 0, None)
               + q["k50_65"] * np.clip(np.minimum(z, 65) - 50, 0, None) + q["k_deep"] * np.clip(z - 65, 0, None))
        out[m] = np.exp(-tau)
    return out


O["trans_month"] = trans_reading


def set_reading(name):
    r = READINGS[name]
    CUR.update(secchi=list(r["secchi"]), k2050=np.asarray(r["k2050"]))
    E0[:] = r["surf"]()
    O["GCACHE"].clear()
    O["ACACHE"].clear()
    O["TRM"].clear()


# ------------------------------------------------------------------ 3. series per pair
def band_series(name, q=qk0, Ke=0.02, kdeep=0.06, clip=True):
    """band_pairs.band_fish_pair (vision 'A') returning the cell-weighted per-time series."""
    members, pt, subset, lightprey, ovl, field, a_lib, nd_lib, thr = FISH_PAIRS[name]
    den = 0.0
    ser = np.zeros((len(ELEV), NB))
    for r in STG_T:
        if r.sp not in members:
            continue
        w = r.W * (WMUL.get(r.sp, 1.0) if field else 1.0)
        den += w
        if sc(r, pt) < 0.1:
            continue
        seasons = [("all", np.ones(len(ELEV), bool))]
        if r.sp in SEASONAL:
            seasons = [("all", ~SUMMER), ("summer", SUMMER)]
        for t in terms(r, pt, qf, subset):
            k = t["a"] / (area_in_slab(t["R"], t["H"]) / t["H"])
            prey = lightprey or t["prey"]
            for sname, smask in seasons:
                rows = (IS_NIGHT if t["night"] else ~IS_NIGHT) & smask
                if not rows.any():
                    continue
                pp = pmat("pred", r.sp, q, t["night"], sname)
                ph = pmat("prey", prey, q, t["night"])
                if t["eye"] in ("fixed", "none"):
                    Rfix = t["R"] if t["eye"] == "fixed" else qf["R_fish_elasmo"]
                    rn2 = rs2 = Rfix ** 2
                    g_rows = None
                else:
                    rs = max(float(rsat_fish(qk0, r.L)), RNV_FISH)
                    rn2, rs2 = RNV_FISH ** 2, rs ** 2
                    g_rows = gmat(t["eye"], Ke, kdeep)[rows]
                if clip and prey in SLAB:
                    H = SLAB[prey](q)
                    if g_rows is None:
                        val = slab_series(pp, H, lambda iz: np.full((1, NB), math.sqrt(rn2)), 1)
                    else:
                        Gm = g_rows

                        def R_of(iz, Gm=Gm, rn2=rn2, rs2=rs2):
                            gg = Gm[:, iz]
                            return np.sqrt(rn2 * (1 - gg) + rs2 * gg)
                        val = slab_series(pp, H, R_of, int(rows.sum()))
                else:
                    val = point_series(pp * ph, rn2, rs2, g_rows)
                ser[rows] += w * k * val
    if members is EATERS:
        den = sum(r.W for r in STG_T if r.sp in REST)
    return (ser / den) @ WCELL


def benth_series(name):
    pair = O["BENTH_PAIRS"][name][0]
    return O["benth_V"](pair, "A", 0.02, 0.06) @ WCELL


def krill_series():
    q = O["meas"](dict(K["level"](1), Ke_gad=0.02, k_deep=0.06))
    ev = K["evaluate"](q)
    return {g: r["a"] @ WCELL / WCELL.sum() for g, r in ev.items()}


def zoo_series(name, rho):
    """intensity-aware engine form rho + (1 - rho) g over a judged layer [J]."""
    if name.startswith("planktivorous_fish"):
        gm = gmat("clupeid", 0.02, 0.06)                       # clupeid branch ignores Ke (Ke_clup 1 lx)
        g = gm[:, int(np.argmin(abs(ZG - ZOO_Z)))]
    else:
        eye, prof = {"mackerel": ("clupeid", "mackerel"), "whiting": ("gadoid", "whiting"),
                     "predatory_fish": ("gadoid", "saithe")}[name.split("->")[0]]
        gm = gmat(eye, 0.02, 0.06)
        g = np.zeros(len(ELEV))
        for night in (False, True):
            rows = IS_NIGHT if night else ~IS_NIGHT
            f = pmat("pred", prof, qk0, night)                 # (n_z, n_b)
            g[rows] = (gm[rows] @ f) @ WCELL
    return rho + (1 - rho) * g


ZOO_Z = 15.0

# ------------------------------------------------------------------ 4. pairs, engine, matching
LIB = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml"), encoding="utf-8-sig"))
INTER = LIB["interaction_definitions"]
ALL = {}
for k, v in INTER.items():
    if isinstance(v, dict) and v.get("dark_ratio") not in (None, ""):
        p, q = k.split("_preys_on_")
        ALL[f"{p}->{q}"] = dict(rho=float(v["dark_ratio"]), thr=(-6.0 if v.get("light_threshold_deg") in (None, "") else float(v["light_threshold_deg"])),
                                M2=float(v.get("predation_mortality") or 0.0), a=v.get("max_intake_rate"))
ALL["whiting->krill"] = dict(rho=1.0, thr=0.0, M2=float(INTER["whiting_preys_on_krill"]["predation_mortality"]),
                             a=INTER["whiting_preys_on_krill"]["max_intake_rate"], note="no dark_ratio; matched to 1")
KIND = {}
for n in ALL:
    p, q = n.split("->")
    if q == "krill":
        KIND[n] = "krill"
    elif q == "zooplankton":
        KIND[n] = "zoo"
    elif n in O["BENTH_PAIRS"]:
        KIND[n] = "benth"
    elif n in FISH_PAIRS:
        KIND[n] = "fish"
    else:
        raise SystemExit(f"no derivation for {n}")

sys.path.insert(0, ROOT)
from lib.world import daylight as DL          # noqa: E402  (read-only)
sys.path.pop(0)
NT = DL.ticks_per_year(6)
TMON = np.array([DL.month_of_day((k * 6) // 24) for k in range(NT)])


def engine_month(rho, thr):
    m = DL.attack_multiplier(LAT, 6, rho, thr)
    return np.array([m[TMON == k].mean() for k in range(12)])


def nightday(s, thr):
    d = s[ELEV > thr].mean()
    return float(s[ELEV <= thr].mean() / d)


def match(s, rho, thr):
    """scale the ENGINE's night rows (sun <= thr) so night/day at thr equals the library rho (exact, one factor)."""
    c = rho / nightday(s, thr)
    return np.where(ELEV <= thr, s * c, s), c


def monthly(s):
    mon = np.array([s[MONTH == k].mean() for k in range(12)])
    return mon / ((mon * MDAYS).sum() / MDAYS.sum())


# ------------------------------------------------------------------ 5. run
RES = {}
for rn in READINGS:
    set_reading(rn)
    S = {}
    kr = krill_series()
    for n, d in ALL.items():
        kind = KIND[n]
        if kind == "fish":
            S[n] = band_series(n)
        elif kind == "benth":
            S[n] = benth_series(n)
        elif kind == "krill":
            S[n] = kr[n.split("->")[0]]
        else:
            S[n] = zoo_series(n, d["rho"])
    out = {}
    for n, d in ALL.items():
        s = S[n]
        sm, c = match(s, d["rho"], d["thr"])
        eng = engine_month(d["rho"], d["thr"])
        der = monthly(sm)
        raw = monthly(s)
        nd_raw = nightday(s, d["thr"])
        eng_raw = engine_month(min(nd_raw, 0.999), d["thr"])
        out[n] = dict(der=der.tolist(), eng=eng.tolist(), res=(der / eng).tolist(), nd_raw=nd_raw, c=c,
                      eq_raw=float(np.array([s[MONTH == k].mean() for k in range(12)]) @ MDAYS / MDAYS.sum()),
                      der_raw=raw.tolist(), eng_raw=eng_raw.tolist(), res_raw=(raw / eng_raw).tolist())
    RES[rn] = out
    print(f"{rn}: done", flush=True)

# planktivores -> zooplankton depth bracket (R0 and R3)
ZB = {}
for rn in ("R0 derivation as is", "R3 R1 + Haurwitz x cloud"):
    set_reading(rn)
    for z in (8.0, 25.0):
        ZOO_Z = z
        n = "planktivorous_fish->zooplankton"
        d = ALL[n]
        sm, _ = match(zoo_series(n, d["rho"]), d["rho"], d["thr"])
        ZB[f"{rn}|{z}"] = (monthly(sm) / engine_month(d["rho"], d["thr"])).tolist()
    ZOO_Z = 15.0

# daytime light at 100 m and 50 m, January vs July, per reading (median over day rows, sun > 0)
LIGHT = {}
for rn in READINGS:
    set_reading(rn)
    E = E0[:, None] * trans_reading(dict(qk0, k_deep=0.06))[MONTH]
    row = {}
    for z in (15, 50, 100, 150):
        iz = int(np.argmin(abs(ZG - (z - 0.5))))
        for m in (0, 6, 11):
            sel = (MONTH == m) & (ELEV > 0)
            row[f"{z}m_{MN[m]}"] = float(np.median(E[sel, iz]))
            row[f"{z}m_{MN[m]}_noon"] = float(E[sel, iz].max())
    LIGHT[rn] = row

# ------------------------------------------------------------------ 6. print
print("\nPairs with a dark_ratio (library) and the derivation used; whiting->krill added (no dark_ratio, stomachs)")
for n, d in ALL.items():
    print(f"  {n:38s} rho {d['rho']:.4g}  thr {d['thr']:+.0f}  M2 {d['M2']:.4g}  kind {KIND[n]}"
          f"  rule's own night/day (R0) {RES['R0 derivation as is'][n]['nd_raw']:.3f}"
          f"  night-row factor to library rho {RES['R0 derivation as is'][n]['c']:.3f}")

print("\nEngine m(t) monthly mean (annual mean 1), per pair:")
for n in ALL:
    print(f"  {n:38s} " + " ".join(f"{x:5.2f}" for x in RES['R0 derivation as is'][n]['eng']))
print("  (months: " + " ".join(MN) + ")")

for rn in READINGS:
    print(f"\n[{rn}] residual = derived / engine per month (night/day matched to the library)")
    print(f"  {'pair':38s} " + " ".join(f"{m:>5s}" for m in MN) + "  NovMar AprOct")
    for n in ALL:
        r = np.array(RES[rn][n]["res"])
        print(f"  {n:38s} " + " ".join(f"{x:5.2f}" for x in r) +
              f"  {r[[10, 11, 0, 1, 2]].mean():5.2f}  {r[3:10].mean():5.2f}")
print("\n[R0 derivation as is] residual with the rule's OWN night/day (engine at that night/day), for comparison:")
for n in ALL:
    r = np.array(RES["R0 derivation as is"][n]["res_raw"])
    print(f"  {n:38s} " + " ".join(f"{x:5.2f}" for x in r))
print("\nAnnual equal-time mean of the derived series (rule's own night/day, before matching), relative to R0:")
print(f"  {'pair':38s} " + " ".join(f"{rn[:2]:>6s}" for rn in READINGS))
for n in ALL:
    e0 = RES["R0 derivation as is"][n]["eq_raw"]
    print(f"  {n:38s} " + " ".join(f"{RES[rn][n]['eq_raw'] / e0:6.3f}" for rn in READINGS))
print("\nplanktivores -> zooplankton, layer depth bracket (residual):")
for k, v in ZB.items():
    print(f"  {k:40s} " + " ".join(f"{x:5.2f}" for x in v))
print("\nDaytime light (lx): median over sun > 0 rows / noon maximum; Jan, Jul, Dec")
for rn, row in LIGHT.items():
    print(f"  {rn:30s} " + "  ".join(f"{z}m {row[f'{z}m_Jan']:.3g}/{row[f'{z}m_Jan_noon']:.3g} | "
                                     f"{row[f'{z}m_Jul']:.3g}/{row[f'{z}m_Jul_noon']:.3g} | "
                                     f"{row[f'{z}m_Dec']:.3g}/{row[f'{z}m_Dec_noon']:.3g}" for z in (50, 100, 150)))
json.dump(dict(pairs=ALL, kind=KIND, readings={k: {kk: (vv if not callable(vv) else None) for kk, vv in v.items()
                                                    if kk != "surf"} for k, v in READINGS.items()},
               res=RES, zoo_bracket=ZB, light=LIGHT, lat=LAT),
          open(os.path.join(OUTD, "light_pairs.json"), "w"), indent=1, default=lambda x: np.asarray(x).tolist())
print("\nwritten light_pairs.json")
