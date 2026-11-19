"""Step 6 (section 270 capture p x schooling F) at a given K_deep: the factor loop of research_notes/"Fangst och
stim i fiskparen"/scripts/combined.py ("central" reading), on
  * research_notes/"Vitlinglojans stim"/scripts/school_factor_pout.py's school-factor algebra (head up to its
    VARIANTS, executed from its own file) fed with step 5's term shares and R_eff at this K_deep;
  * research_notes/"Fangstsannolikhet for fiskbyten"/scripts/p_ppr.json (capture p: prey/predator length only, no
    light term; read unchanged);
  * the section 269 a and night/day of step 4 at this K_deep (combined.py read them from the then-live library).
factor = (p_new / p_library) x F(p_new); night/day' = night/day x F_night / F_day (stomach ratio kept).

The schooling note hard-codes the benthivore pairs' rows (they have no in-layer term split): night/day 0.141 /
0.168 (the section 269 values), pout 0.85 of benthivores -> planktivores, and the rule-A radius at the bed 0.6 m by
day / 0.3 m at night ([J], also the default for the water column's bed-slab terms). Here they follow K_deep:
night/day = step 4's value; pout share = step 2's derived pout share of the a (0.85 at 0.06); radii x R_eff(K_deep)
/ R_eff(0.06) of benth_pair's light-weighted radius at the bed (step 1 jsons; the benthivores -> planktivores
radius ratio also scales the bed-slab default). At K_deep 0.06 every one of these is the original number.

Usage: python3 s6_combined.py <repo_root> <K_deep> <step5 json> <step4 json> <step1 json> <step1 json at 0.06>
                              <step2 json> <scratch_dir> <out_json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k280_common import run_patched  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
KD = float(sys.argv[2])
TERMS = os.path.abspath(sys.argv[3])
ROWS = json.load(open(sys.argv[4]))["rows"]
BAND = json.load(open(sys.argv[5]))["variants"]["A_band"]["fish"]
BAND0 = json.load(open(sys.argv[6]))["variants"]["A_band"]["fish"]
BEN = json.load(open(sys.argv[7]))["variants"]["central"]
SCR = os.path.abspath(sys.argv[8])
OUTJ = os.path.abspath(sys.argv[9])
os.makedirs(SCR, exist_ok=True)
RN = os.path.join(ROOT, "research_notes")
POUT_PY = os.path.join(RN, "Vitlinglojans stim", "scripts", "school_factor_pout.py")
P_JSON = os.path.join(RN, "Fangstsannolikhet for fiskbyten", "scripts", "p_ppr.json")
PP = json.load(open(P_JSON))["pairs"]

BP, BW = "benthivorous_fish->planktivorous_fish", "benthivorous_fish->whiting"


def rscale(name, per):
    k = "R_eff_new_day" if per == "day" else "R_eff_new_night"
    return BAND[name][k] / BAND0[name][k]


nd_p = ROWS["benthivorous_fish_preys_on_planktivorous_fish"]["dark_ratio"]
nd_w = ROWS["benthivorous_fish_preys_on_whiting"]["dark_ratio"]
pout = round(BEN[BP]["species"]["Norway pout"], 2)
RPD, RPN = 0.6 * rscale(BP, "day"), 0.3 * rscale(BP, "night")
RWD, RWN = 0.6 * rscale(BW, "day"), 0.3 * rscale(BW, "night")
OLD_ROWS = '''for name, nd, plk in (("benthivorous_fish->planktivorous_fish", 0.141, True),
                      ("benthivorous_fish->whiting", 0.168, False)):
    dshare = 1.0 / (1.0 + nd)
    if plk:
        rows = [("pout", "day", 0.85 * dshare, 0.6), ("pout", "night", 0.85 * (1 - dshare), 0.3),
                ("herring", "day", 0.15 * dshare, 0.6), ("herring", "night", 0.15 * (1 - dshare), 0.3)]
    else:
        rows = [("whiting", "day", dshare, 0.6), ("whiting", "night", 1 - dshare, 0.3)]'''
NEW_ROWS = f'''for name, nd, plk in (("benthivorous_fish->planktivorous_fish", {nd_p!r}, True),
                      ("benthivorous_fish->whiting", {nd_w!r}, False)):
    dshare = 1.0 / (1.0 + nd)
    if plk:
        rows = [("pout", "day", {pout!r} * dshare, {RPD!r}), ("pout", "night", {pout!r} * (1 - dshare), {RPN!r}),
                ("herring", "day", {1 - pout!r} * dshare, {RPD!r}), ("herring", "night", {1 - pout!r} * (1 - dshare), {RPN!r})]
    else:
        rows = [("whiting", "day", dshare, {RWD!r}), ("whiting", "night", 1 - dshare, {RWN!r})]'''
OLD_SLAB = '            R = 0.6 if per == "day" else 0.3\n'
NEW_SLAB = f'            R = {RPD!r} if per == "day" else {RPN!r}\n'
NS = run_patched(POUT_PY, [(OLD_ROWS, NEW_ROWS, 1), (OLD_SLAB, NEW_SLAB, 1)], [POUT_PY, TERMS, os.path.join(SCR, "_unused.json")],
                 name="pout_head", upto="VARIANTS = {", quiet=True)
PAR, SCHOOL_PAIRS = NS["PAR_NEW"], NS["pairs"]
STOMACH_ND = {"mackerel->planktivorous_fish"}

print(f"K_deep {KD}: benthivore rows night/day {nd_p} / {nd_w}, pout share {pout}, bed radii day/night "
      f"planktivores {RPD:.3f}/{RPN:.3f} m, whiting {RWD:.3f}/{RWN:.3f} m (0.6 / 0.3 at K_deep 0.06)")
out = {}
for name, pp in PP.items():
    pred, prey = name.split("->")
    key = f"{pred}_preys_on_{prey}"
    a0 = ROWS[key]["max_intake_rate"]
    nd0 = ROWS[key].get("dark_ratio")
    p_lib, p_new = pp["p_lib"], pp["central"]["p"]
    pf = p_new / p_lib
    NS["P_I"] = p_new
    f = NS["pair_factor"](SCHOOL_PAIRS[name], PAR, 1, None)
    tot = pf * f["F"]
    nd_new = nd0 if (nd0 is None or name in STOMACH_ND) else nd0 * f["F_night"] / f["F_day"]
    out[name] = dict(a269=a0, nd269=nd0, p_lib=p_lib, p_new=p_new, p_factor=pf, F=f["F"], F_day=f["F_day"],
                     F_night=f["F_night"], factor=tot, a_new=a0 * tot, dark_ratio_new=nd_new)
    print(f"  {name:40s} a269 {a0:.4g} p x{pf:.3f} F {f['F']:.3f} (day {f['F_day']:.3f}, night {f['F_night']:.3f}) "
          f"factor x{tot:.4f} -> a {a0 * tot:.4g}; night/day {nd0} -> {nd_new if nd_new is None else round(nd_new, 4)}")
json.dump(dict(K_deep=KD, pairs=out, factors={k: v["factor"] for k, v in out.items()},
               bed_rows=dict(nd_p=nd_p, nd_w=nd_w, pout=pout, RPD=RPD, RPN=RPN, RWD=RWD, RWN=RWN)),
          open(OUTJ, "w"), indent=1)
