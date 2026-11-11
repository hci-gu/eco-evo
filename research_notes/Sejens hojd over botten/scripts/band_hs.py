"""The water-column (WC) fish pairs and WC -> krill on the band overlap + rule A (section 269 form) with the saithe
(and pollack) search profile varied over the field evidence of this note; the juvenile prey layer varied too.

Usage: python3 -I band_hs.py <repo_root> <scratch_dir> <out_json>

research_notes/"Overlapp i fiskparen"/scripts/band_pairs.py is exec'd (unchanged) up to its section 0 block, so
band_fish_pair, run_krill, the member list, rule A, light, slab clipping and every other factor are the ones the
library's section 269 a's came from. Only the saithe/pollack profile function and (in named variants) the
juvenile layer are replaced. Every result is reported as a RATIO to this script's own reproduction of the
central A_band (exp 15 m), which must reproduce band_pairs.json A_band (printed check).

Saithe profile shapes (day; night = f_up in 0-30 m + (1 - f_up) x the same shape at scale x Hn_ratio):
  exp      exp(-h/Hs) above the bed, never above 10 m depth (the library's form, krill_pairs_wc.saithe_field)
  gamma2   h exp(-h/s)/s^2: mode s above the bed, little in the bottom metres (the HERAS pelagic-haul hump)
  uniform  uniform over the bottom Hb metres
Nothing is fitted to model output.
"""
import contextlib
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
SCR = os.path.abspath(sys.argv[2])
OUTJ = os.path.abspath(sys.argv[3])
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
REF = json.load(open(os.path.join(ROOT, "research_notes", "Overlapp i fiskparen", "tables", "band_pairs.json")))["variants"]


def shape_prof(bed, shape, s):
    z = ZG
    h = bed - z
    if shape == "exp":
        return expbed(bed, s)
    if shape == "gamma2":
        out = np.where((z >= 10.0) & (z < bed), h * np.exp(-h / max(s, 0.5)), 0.0)
    elif shape == "uniform":
        out = np.where((z >= max(10.0, bed - s)) & (z < bed), 1.0, 0.0)
    else:
        raise ValueError(shape)
    tot = out.sum()
    return out / tot if tot > 0 else uniform(bed - 1, bed)


def saithe_var(bed, q, night):
    shape = q.get("shape_saithe", "exp")
    s = q["Hs_saithe"]
    if not night:
        return shape_prof(bed, shape, s)
    up = uniform(0.0, min(30.0, bed - 1.0))
    return q["f_up_saithe"] * up + (1 - q["f_up_saithe"]) * shape_prof(bed, shape, s * q["Hn_ratio"])


B["PRED_PROF"]["saithe"] = saithe_var
B["PRED_PROF"]["pollack"] = saithe_var

PREY_ORIG = B["prey_prof"]


def juv_night_rise(share_up, top):
    """juveniles: day bottom H_juv; night (1 - share_up) bottom H_juv + share_up uniform H_juv..top m above the bed
    (Lough et al. 1989: 'some fish were several meters up', 3-5 m; Aglen et al. 1997: small haddock lifted 30-40 m)."""
    def pp(name, q):
        f = PREY_ORIG(name, q)
        if name != "juvenile benthivores":
            return f

        def g(b, qq, n):
            base = uniform(b - qf["H_juv"], b)
            if not n:
                return base
            return (1 - share_up) * base + share_up * uniform(b - top, b - qf["H_juv"])
        return g
    return pp


VARS = {
    # name: (q overrides, options)
    "central_repro": ({}, {}),
    "exp_Hs5": ({"Hs_saithe": 5.0}, {}),
    "exp_Hs7": ({"Hs_saithe": 7.0}, {}),
    "exp_Hs10": ({"Hs_saithe": 10.0}, {}),
    "exp_Hs20": ({"Hs_saithe": 20.0}, {}),
    "exp_Hs30": ({"Hs_saithe": 30.0}, {}),
    "exp_Hs40": ({"Hs_saithe": 40.0}, {}),
    "exp_Hs15_night_same": ({"Hn_ratio": 1.0}, {}),
    "exp_Hs15_night_up25": ({"f_up_saithe": 0.25}, {}),
    "gamma2_s6": ({"shape_saithe": "gamma2", "Hs_saithe": 6.0}, {}),
    "gamma2_s10": ({"shape_saithe": "gamma2", "Hs_saithe": 10.0}, {}),
    "uniform_35": ({"shape_saithe": "uniform", "Hs_saithe": 35.0}, {}),
    "uniform_57": ({"shape_saithe": "uniform", "Hs_saithe": 57.0}, {}),
    "juv_H0.5": ({}, {"H_juv": 0.5}),
    "exp_Hs10_juvH0.5": ({"Hs_saithe": 10.0}, {"H_juv": 0.5}),
    "gamma2_s6_juvH0.5": ({"shape_saithe": "gamma2", "Hs_saithe": 6.0}, {"H_juv": 0.5}),
    "noclip": ({}, {"clip": False}),
    "noclip_juv_night_3_5m": ({}, {"clip": False, "juv": (0.3, 5.0)}),
    "noclip_juv_night_aglen": ({}, {"clip": False, "juv": (0.4, 35.0)}),
}
WC_FISH = ["predatory_fish->planktivorous_fish", "predatory_fish->benthivorous_fish", "predatory_fish->whiting",
           "predatory_fish->mackerel"]
RES = {}
H_JUV0 = qf["H_juv"]
B["RULE"].update(rule="lib", RD=1.2, Lk=None)
with contextlib.redirect_stdout(io.StringIO()):
    KR_REP = B["run_krill"](dict(K["level"](1)))["predatory_fish"]["eq"]      # library-rule reproduction (0.026)
for vn, (qo, opt) in VARS.items():
    B["PCACHE"].clear()
    qv = dict(qk0, **qo)
    qf["H_juv"] = opt.get("H_juv", H_JUV0)
    B["prey_prof"] = juv_night_rise(*opt["juv"]) if "juv" in opt else PREY_ORIG
    B["RULE"].update(rule="A", RD=1.2, Lk=None)
    # krill: the krill machinery's saithe / pollack members carry the profile bound at load: point them here
    for m in K["GROUPS"]["predatory_fish"]["members"]:
        if m["name"].startswith(("saithe", "pollack")):
            m["prof"] = saithe_var
    with contextlib.redirect_stdout(io.StringIO()):
        kr = B["run_krill"](dict(K["level"](1), Ke_gad=0.02, k_deep=0.06, **qo))
    out = {"krill": {"eq": kr["predatory_fish"]["eq"], "classes": list(kr["predatory_fish"]["classes"])}, "fish": {}}
    for name in WC_FISH:
        r = B["band_fish_pair"](name, qv, "A", 0.02, 0.06, opt.get("clip", True))
        out["fish"][name] = dict(eq=r["eq"], ratio_lib=r["ratio"], classes=r["classes"], nd=r["nd"],
                                 members=r["members"])
    RES[vn] = out
    print(f"{vn}: done", flush=True)
qf["H_juv"] = H_JUV0
B["prey_prof"] = PREY_ORIG

c = RES["central_repro"]
print("\nReproduction check vs band_pairs.json A_band (ratio new a / library a):")
for name in WC_FISH:
    print(f"  {name:40s} here {c['fish'][name]['ratio_lib']:.4f}  band_pairs.json {REF['A_band']['fish'][name]['ratio']:.4f}")
print(f"  predatory_fish->krill: here a {0.026 * c['krill']['eq'] / KR_REP:.4g}  band_pairs.json "
      f"{REF['A_band']['krill']['predatory_fish']['a_new']:.4g} (library 0.0117)")

print("\nRatio to the central (exp 15 m) A_band, equal-cell annual mean a; night/day of the pair")
hdr = ["krill"] + [n.split("->")[1][:12] for n in WC_FISH]
print(f"  {'variant':26s} " + " ".join(f"{h:>13s}" for h in hdr) + "   ben n/d   saithe share of ben a")
for vn, r in RES.items():
    base = RES["noclip"] if vn.startswith("noclip_") else c
    row = [r["krill"]["eq"] / base["krill"]["eq"]] + [r["fish"][n]["eq"] / base["fish"][n]["eq"] for n in WC_FISH]
    mem = r["fish"]["predatory_fish->benthivorous_fish"]["members"]
    sai = sum(v for k, v in mem.items() if k.startswith("saithe") or k.startswith("pollack"))
    print(f"  {vn:26s} " + " ".join(f"{x:13.3f}" for x in row)
          + f"   {r['fish']['predatory_fish->benthivorous_fish']['nd']:.3f}     {sai:.3f}")
print("  (noclip_* variants are relative to 'noclip'; noclip itself relative to the central)")

print("\nSaithe day profile: share within 2 m / 10 m / 35 m of the bed and density at the bed (/m), equal-cell mean over beds")
for vn, (qo, opt) in VARS.items():
    if vn.startswith(("juv", "noclip")) or "juvH" in vn:
        continue
    qv = dict(qk0, **qo)
    s2 = s10 = s35 = 0.0
    for ib, b in enumerate(B["BEDS"]):
        f = saithe_var(b, qv, False)
        h = b - ZG
        s2 += B["WCELL"][ib] * f[(h > 0) & (h <= 2)].sum()
        s10 += B["WCELL"][ib] * f[(h > 0) & (h <= 10)].sum()
        s35 += B["WCELL"][ib] * f[(h > 0) & (h <= 35)].sum()
    print(f"  {vn:26s} {s2:.3f} / {s10:.3f} / {s35:.3f};  f(bed) ~ {s2 / 2:.4f} /m")
json.dump(RES, open(OUTJ, "w"), indent=1, default=float)
print(f"\nwritten {os.path.basename(OUTJ)}")
