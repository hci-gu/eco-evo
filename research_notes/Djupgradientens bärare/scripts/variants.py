"""Class tables of the water column's (WC) krill and planktivore pairs under field-tied variants, at the CURRENT
library conventions (rule A + band overlap of section 269, K_deep 0.09 of section 280).

Usage: python3 -I variants.py <repo_root> <scratch_dir> <out_json> [variant ...]

research_notes/"Overlapp i fiskparen"/scripts/band_pairs.py is exec'd UNCHANGED up to its section 0 block (as
research_notes/"Sejens profil over djupa bottnar"/scripts/class_terms.py does), so band_fish_pair, run_krill, the
member list, rule A, the section 253 light, the section 262 krill layer, slab clipping and every other factor are
the ones the library's class tables came from. K_deep is passed as 0.09 (section 280) in every run, as
research_notes/"Djupljus i synparen"/scripts/s1_band.py does. Library units follow final_gate.py:
  krill:        classes_lib_units = class a x KRILL_LIB a / KR_REPRO eq   (KR_REPRO = the library-rule krill run at
                level(1), K_deep 0.06, the section 262 derivation: a ratio denominator, kept at 0.06 as in s1_band)
  planktivores: a_new x cls_rel, a_new = a_lib x eq / a_enc_lib.
The gate: variant "library" must give 0.0302 / 0.0129 / 0.0011 / 6.8e-05 (krill) and 0.038 / 0.0167 / 0.00368 /
0.00284 (planktivores) at 3 significant figures, and a 0.0107 / 0.0148 (pre-270, pre-296).

What a variant may replace (each tied to a named source in ../evidence.md; [J] where the form is a judgement):
  krill_layer   the krill day layer over beds in [b_lo, b_hi): its BOTTOM edge put d_bot m above the bed, same
                thickness rule; outside the range the library layer
  saithe        the saithe / pollack search profile: a share m of the day (or of a twilight window) spent in the
                krill layer of that cell, the rest the library exp 15 m profile
  vision        Ke (gadoid eye, lx) and R_nv (dark detection distance on krill, m; on fish, m)
Nothing is fitted; no model output is read.
"""
import contextlib
import copy
import io
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
OUTJ = os.path.abspath(sys.argv[3])
ONLY = sys.argv[4:]
os.makedirs(SCR, exist_ok=True)
BP = os.path.join(ROOT, "research_notes", "Overlapp i fiskparen", "scripts", "band_pairs.py")
src = open(BP).read()
head = src[:src.index("# ================================================================== 0. overlap geometry")]
ARGV = list(sys.argv)
sys.argv = [BP, ROOT, SCR, os.path.join(SCR, "_unused_band.json")]
B = {"__name__": "band_pairs_head", "__file__": BP}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, BP, "exec"), B)
sys.argv = ARGV
K, qk0, qf, ZG = B["K"], B["qk0"], B["qf"], B["ZG"]
uniform, expbed = K["uniform"], K["expbed"]
BEDS, WCELL, CLS_IDX, ELEV = B["BEDS"], B["WCELL"], B["CLS_IDX"], B["ELEV"]
SHARE = np.array([WCELL[i].sum() for i in CLS_IDX])
KD = 0.09
SAITHE0 = K["saithe_field"]
LAYER0 = K["krill_day_layer"]
PRED_PROF = B["PRED_PROF"]
FP = B["FISH_PAIRS"]
KRILL_LIB = B["KRILL_LIB"]
PLK = "predatory_fish->planktivorous_fish"
RNV_FISH0 = B["RNV_FISH"]

# library reproduction denominators (as s1_band.py: KR_REPRO at K_deep 0.06, library rule)
B["RULE"].update(rule="lib", RD=1.2, Lk=None)
with contextlib.redirect_stdout(io.StringIO()):
    KR_REPRO = B["run_krill"](dict(K["level"](1)))
B["RULE"].update(rule="A", RD=1.2, Lk=None)


MONTH = B["MONTH"]
_OG = B["run_krill"].__globals__          # one_rule_pairs namespace: meas, krill_classes, night_day, SHARE_CLS
FISH = {"plk": PLK, "benth": "predatory_fish->benthivorous_fish", "whit": "predatory_fish->whiting"}


def layer_anchor(anchors, H=None):
    """krill day layer whose BOTTOM edge lies gap(bed) m above the bed, gap linear between the anchors
    [(bed, gap), ...] (field rows in ../tables/q1_krill_bed_rows.csv); thickness H (m) or the library thickness
    rule; never above 60 m (library ztop_min). Beds outside [first, last anchor bed]: the library layer."""
    bs = [float(a[0]) for a in anchors]
    gs = [float(a[1]) for a in anchors]

    def f(bed, q):
        if not (bs[0] <= bed <= bs[-1]):
            return LAYER0(bed, q)
        Hk = H if H is not None else q["Hk"] * (1.0 + (2.0 / 3.0) * np.clip((bed - 150.0) / 100.0, 0, 1))
        bot = min(bed - float(np.interp(bed, bs, gs)), bed - 1.0)
        top = max(bot - Hk, q["ztop_min"])
        if bot - top < 5.0:
            top = bot - 5.0
        return max(top, 0.0), bot
    return f


def lib_gap(bed):
    return bed - LAYER0(bed, dict(K["level"](1)))[1]


def krill_day_density(bed, q):
    top, bot = K["krill_day_layer"](bed, q)
    return uniform(top, bot)


def saithe_mix(m_day, m_night=0.0):
    """saithe/pollack: share m_day of the day search time inside the krill DAY layer of the cell, m_night of the
    night time inside the krill NIGHT profile; the rest the library profile (exp 15 m day; night 0.07 up)."""
    def f(bed, q, night):
        base = SAITHE0(bed, q, night)
        if not night:
            return (1 - m_day) * base + m_day * krill_day_density(bed, q) if m_day > 0 else base
        if m_night > 0:
            return (1 - m_night) * base + m_night * K["krill_profile"](bed, q, True)
        return base
    return f


def set_profile(fn):
    PRED_PROF["saithe"] = fn
    PRED_PROF["pollack"] = fn
    for m in K["GROUPS"]["predatory_fish"]["members"]:
        if m["name"].startswith(("saithe", "pollack")):
            m["prof"] = fn
    B["PCACHE"].clear()


def run_krill_v(q, layer):
    """run_krill of one_rule_pairs with the layer given; layer may be seasonal: dict(summer=f, winter=f,
    months=[0-based months of the summer layer]); a(t, bed) is then taken from each season's run by MONTH."""
    qm = _OG["meas"](q)
    if isinstance(layer, dict):
        K["krill_day_layer"] = layer["summer"]
        evs = K["evaluate"](qm)
        K["krill_day_layer"] = layer["winter"]
        evw = K["evaluate"](qm)
        msk = np.isin(MONTH, layer["months"])[:, None]
        ev = {g: dict(a=np.where(msk, evs[g]["a"], evw[g]["a"])) for g in evs}
    else:
        K["krill_day_layer"] = layer
        ev = K["evaluate"](qm)
    K["krill_day_layer"] = LAYER0
    out = {}
    for g, r in ev.items():
        cls = _OG["krill_classes"](r["a"])
        s = r["a"] @ WCELL / WCELL.sum()
        out[g] = dict(classes=cls.tolist(), eq=float(_OG["SHARE_CLS"] @ cls), nd=_OG["night_day"](s, KRILL_LIB[g][3]))
    return out


def run(v):
    """one variant: every krill pair and the three WC fish pairs, class tables in library units, a, night/day."""
    set_profile(v.get("saithe", SAITHE0))
    B["RNV_FISH"] = v.get("rnv_fish", RNV_FISH0)
    Ke = v.get("Ke", 0.02)
    qx = dict(K["level"](1), Ke_gad=Ke, k_deep=KD)
    if "rnv_krill" in v:
        qx["R_nv"] = v["rnv_krill"]
    with contextlib.redirect_stdout(io.StringIO()):
        kr = run_krill_v(qx, v.get("layer", LAYER0))
    out = {"krill": {}, "fish": {}}
    for g, r in kr.items():
        lib = KRILL_LIB[g]
        rep = KR_REPRO[g]
        out["krill"][g] = dict(a=lib[0] * r["eq"] / rep["eq"], classes=[c * lib[0] / rep["eq"] for c in r["classes"]],
                               nd=r["nd"])
    if not isinstance(v.get("layer", LAYER0), dict):
        K["krill_day_layer"] = v.get("layer", LAYER0)       # the saithe-mix profiles read the cell's krill layer
    for key, name in FISH.items():
        r = B["band_fish_pair"](name, dict(qk0), "A", Ke, KD, True)
        a_new = r["a_lib"] * r["ratio"]
        out["fish"][key] = dict(a=a_new, classes=[a_new * c for c in r["cls_rel"]], nd=r["nd"])
    K["krill_day_layer"] = LAYER0
    set_profile(SAITHE0)
    B["RNV_FISH"] = RNV_FISH0
    return out


def G(cls):
    c = np.asarray(cls, float)
    w = SHARE[:2] / SHARE[:2].sum()
    return float(w @ c[:2] / c[2]), float(c[1] / c[2]), (float(c[0] / c[2]) if c[2] > 0 else float("inf"))


def g3(x):
    return float(f"{x:.3g}") if x > 0 else 0.0


H = dict(layer_anchor=layer_anchor, lib_gap=lib_gap, saithe_mix=saithe_mix, LAYER0=LAYER0, K=K, uniform=uniform,
         expbed=expbed, SAITHE0=SAITHE0)
VARIANTS = {"library": {}}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variant_defs  # noqa: E402
VARIANTS.update(variant_defs.defs(H))
if ONLY:
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in ONLY or k == "library"}

print("Library krill day layer (section 262) by bed: bed / layer / gap of its bottom edge above the bed")
for b in (100, 120, 140, 150, 155, 160, 166, 170, 180, 200, 213, 230, 250, 300, 350):
    t_, b_ = LAYER0(float(b), dict(K["level"](1)))
    print(f"  {b:4d} m: {t_:6.1f}-{b_:6.1f} m, gap {b - b_:6.1f} m")

RES = {}
for name, v in VARIANTS.items():
    RES[name] = run(v)
    print(f"{name} done: {v.get('src', '')}", flush=True)

L = RES["library"]
LIBV = {("krill", "predatory_fish"): ([0.0302, 0.0129, 0.0011, 6.8e-05], 0.0107, 0.0257),
        ("krill", "whiting"): ([0.146, 0.0317, 0.0006, 0.0], 0.041, None),
        ("krill", "planktivorous_fish"): ([0.0689, 0.018, 0.00281, 0.00166], 0.0214, None),
        ("krill", "mackerel"): ([0.0053, 0.0042, 0.00297, 0.00284], 0.00382, None),
        ("fish", "plk"): ([0.038, 0.0167, 0.00368, 0.00284], 0.0148, None),
        ("fish", "benth"): ([0.0676, 0.0426, 0.0141, 0.0127], None, None),
        ("fish", "whit"): ([0.0588, 0.0379, 0.0112, 0.00966], None, None)}
print("\nGate: variant 'library' (K_deep 0.09) against the live library (pre-296 units; g3 rounding)")
ok_all = True
for (kind, g), (cls, a, nd) in LIBV.items():
    r = L[kind][g]
    got = [g3(x) for x in r["classes"]]
    ok = got == cls and (a is None or g3(r["a"]) == a) and (nd is None or round(r["nd"], 4) == nd)
    ok_all &= ok
    print(f"  {kind:5s} {g:20s} " + " / ".join(f"{x}" for x in got) + f"  a {g3(r['a'])}  n/d {r['nd']:.4f}"
          + f"   live " + " / ".join(f"{x}" for x in cls) + f"  {'OK' if ok else 'MISMATCH'}")
print(f"  gate {'PASSED' if ok_all else 'FAILED'}")
print("  (benthivores -> krill is derived in research_notes/\"Bottenfiskens hojdprofil\"/scripts/benth_profile.py, not by"
      " this band run: not gated, printed for its direction only, never written to a library copy)")
print(f"\nclass cell shares < 100 / 100-150 / 150-250 / >= 250 m: " + " / ".join(f"{x:.3f}" for x in SHARE))
print("\nVariants: class a in library units (pre-296; the loader renormalises to an equal-cell mean 1); a x = equal-"
      "cell a relative to the library variant; G = a(< 150 m, cell-weighted) / a(150-250 m); G1 = a(100-150) / "
      "a(150-250); R0 = a(< 100) / a(150-250) (class 0 presence-weighted for krill)")
ROWS = [("krill", "predatory_fish", "WC -> krill"), ("fish", "plk", "WC -> planktivores"),
        ("fish", "benth", "WC -> benthivores"), ("fish", "whit", "WC -> whiting"),
        ("krill", "whiting", "whiting -> krill"), ("krill", "planktivorous_fish", "planktivores -> krill"),
        ("krill", "benthivorous_fish", "benthivores -> krill"), ("krill", "mackerel", "mackerel -> krill")]
for kind, g, lab in ROWS:
    print(f"\n  {lab}")
    for name, r in RES.items():
        c = r[kind][g]["classes"]
        gg, g1, r0 = G(c)
        print(f"    {name:30s} " + " / ".join(f"{g3(x):9.3g}" for x in c)
              + f"   a x{r[kind][g]['a'] / L[kind][g]['a']:.3f}   G {gg:8.2f}  G1 {g1:7.2f}  R0 {r0:8.2f}"
              f"  n/d {r[kind][g]['nd']:.4f}")
json.dump(dict(K_deep=KD, share=SHARE.tolist(), runs=RES, src={k: v.get("src", "") for k, v in VARIANTS.items()}), open(OUTJ, "w"), indent=1, default=float)
print(f"\nwritten {os.path.basename(OUTJ)}")
