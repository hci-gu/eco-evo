"""Whiting search speed: factor on each whiting pair's a (and night/day) from the chain's own encounter
machinery (band_pairs.py definitions, rule A, K_deep 0.09, Ke 0.02), comparing the pair's ABSOLUTE band a
(eq) between speeds. (The chain's own ratio eq / a_enc_lib has the speed in both terms and cancels.)
Readings of Onsrud et al. 2005 (deep fish 'mainly Norway pout and whiting', day modes 14-16 cm/s, night
10-11 cm/s; species and lengths not separated, abstract only):
  base       chain 0.8 BL/s, night x1.0
  bl_day     0.6 BL/s every stage, night x1.0            (day speed read as whiting at the box length)
  bl_night   0.6 BL/s, night x0.7                         (also the night slowdown)
  abs_day    15 cm/s every stage (juveniles 11.8 cm faster than now), night x1.0
  abs_night  15 cm/s, night 10.5 cm/s
  night_only chain 0.8 BL/s, night x0.7
Krill members carry no night speed in the chain (whiting -> krill night/day is stomach-given): the night
factor is not applied to krill. Epifauna (bed pair, 232) is outside the chain: vrel ratio per stage, weights
nt x vrel (2-D swath, R the same for both stages), u_bed_inv.
usage: speed_factors.py <scratch>
"""
import os, sys, json, math
ROOT = "/home/simon/eco-evo"
SCR = os.path.abspath(sys.argv[1]); os.makedirs(SCR, exist_ok=True)
BP = os.path.join(ROOT, "research_notes", "Overlapp i fiskparen", "scripts", "band_pairs.py")
src = open(BP).read()
src = src[:src.index("# ================================================================== 0. overlap geometry")]
sys.argv = [BP, ROOT, SCR, os.path.join(SCR, "_unused.json")]
B = {"__name__": "band_defs", "__file__": BP}
exec(compile(src, BP, "exec"), B)
O, K = B["O"], B["K"]
RULE, run_krill, band_fish_pair, qk0, qf = B["RULE"], B["run_krill"], B["band_fish_pair"], B["qk0"], B["qf"]
L_J, L_A = 0.1183, 0.2516
READ = {"base": (0.8, 1.0, None), "bl_day": (0.6, 1.0, None), "bl_night": (0.6, 0.7, None),
        "abs_day": (None, 1.0, 0.15), "abs_night": (None, 0.7, 0.15), "night_only": (0.8, 0.7, None)}
v_of0 = O["v_of"]
mems = K["GROUPS"]["whiting"]["members"]
assert [m["name"] for m in mems] == ["whiting juvenile", "whiting adult"]

def setup(bl, nv, ab):
    def v_of(r, q, night):
        if r.sp != "whiting":
            return v_of0(r, q, night)
        v = (ab if ab is not None else bl * r.L)
        return v * (nv if night else 1.0)
    O["v_of"] = v_of
    # krill members: per-stage keys (no night speed in the chain's krill members)
    mems[0]["vkey"], mems[1]["vkey"] = "v_wj", "v_wa"
    return {"v_wj": (ab / L_J if ab is not None else bl), "v_wa": (ab / L_A if ab is not None else bl)}

vrel = K["vrel"]
res = {}
for name, (bl, nv, ab) in READ.items():
    extra = setup(bl, nv, ab)
    RULE.update(rule="A", RD=1.2, Lk=None)
    kr = run_krill(dict(K["level"](1), Ke_gad=0.02, k_deep=0.09, **extra))["whiting"]
    out = {"whiting->krill": dict(eq=kr["eq"], nd=kr["nd"])}
    for pn in ("whiting->planktivorous_fish", "whiting->benthivorous_fish"):
        r = band_fish_pair(pn, dict(qk0), "A", 0.02, 0.09, True)
        out[pn] = dict(eq=r["eq"], nd=r["nd"], members=r["members"])
    # epifauna: 2-D swath, a ~ sum_stage W nt vrel(v, u); day and night alike weighted 0.5 / 0.5
    u = qf["u_bed_inv"]; tot = 0.0
    for m, L in zip(mems, (L_J, L_A)):
        v = ab if ab is not None else bl * L
        tot += m["W"] * m["nt"] * 0.5 * (vrel(v, u) + vrel(v * nv, u))
    out["whiting->mobile_epifauna"] = dict(eq=float(tot), nd=None)
    res[name] = out
    print(f"{name} done", flush=True)
O["v_of"] = v_of0

base = res["base"]
print("\nFactor on each whiting pair's a (absolute band a vs the chain's 0.8 BL/s); night/day new (base)")
print(f"  {'pair':30s} " + " ".join(f"{n:>11s}" for n in READ if n != "base"))
fac = {}
for pn in base:
    row = []
    for n in READ:
        if n == "base":
            continue
        f = res[n][pn]["eq"] / base[pn]["eq"]
        fac.setdefault(n, {})[pn] = f
        row.append(f"{f:11.3f}")
    print(f"  {pn:30s} " + " ".join(row))
print("\nnight/day (rule A day/night of the pair's band a): base -> each reading")
for pn in base:
    if base[pn]["nd"] is None:
        continue
    print(f"  {pn:30s} {base[pn]['nd']:.3f} -> " + " / ".join(f"{res[n][pn]['nd']:.3f}" for n in READ if n != "base"))
print("\nstage shares of the base band a (fish pairs):")
for pn in ("whiting->planktivorous_fish", "whiting->benthivorous_fish"):
    print(f"  {pn}: {base[pn]['members']}")
# krill stage shares at base: run with only one stage (W of the other 0)
w0 = [m["W"] for m in mems]
extra = setup(0.8, 1.0, None)
sh = []
for k in (0, 1):
    for j, m in enumerate(mems):
        m["W"] = w0[j] if j == k else 0.0
    RULE.update(rule="A", RD=1.2, Lk=None)
    sh.append(run_krill(dict(K["level"](1), Ke_gad=0.02, k_deep=0.09, **extra))["whiting"]["eq"])
for j, m in enumerate(mems):
    m["W"] = w0[j]
print(f"  whiting->krill: juvenile {sh[0] / sum(sh):.2f}, adult {sh[1] / sum(sh):.2f} (sum check {sum(sh) / base['whiting->krill']['eq']:.3f})")
json.dump(dict(res=res, factors=fac), open(os.path.join(SCR, "speed_factors.json"), "w"), indent=1, default=float)
