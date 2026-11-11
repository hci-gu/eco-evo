"""Schooling factor on the encounter a of the fish-prey pairs (research only; nothing fitted, no model output read).

Usage: python3 -I school_factor.py <pair_terms.json> <out_json>

Per-capita capture rate of one prey individual, relative to the same individual dispersed (S):

    dispersed (library, rule A):  rate_i = scan pi R_i^2 v_rel,i p_i n          (n = prey number density)
    schooled:                     rate_s = pi R_s^2 v_rel,s (n / N) k            (schools of N, k caught per school met)
    S = rate_s / rate_i = (R_s^2 / (scan R_i^2)) (v_rel,s / v_rel,i) k / (N p_i)

R_i = the term's own light-weighted rule-A radius (pair_terms.json, R_eff), so S is computed against the radius the
library a actually uses (no double count with rule A); the band overlap is unchanged (schools and dispersed fish
share the prey's biomass profile). R_s = school half-length + school detection distance; scan 1 for a school
(detected all round, as in the section 220 school variant). k = prey caught per school encounter, set by the
per-attack success on a school (confusion) and the number of attacks before the predator leaves or is sated,
capped by one meal. Satiation per encounter is in k; handling h is unchanged.

Share of the prey biomass in schools in the term's period: sigma (day / night). Term factor F = 1 - sigma (1 - S).
Pair factor = sum over terms of (term share of today's a) x F. New night/day = old night/day x F_night / F_day.

Three readings: "low" protection (sigma low, S high), "central", "high" protection (sigma high, S low). Each factor's
bracket is listed with its grade in evidence.md.
"""
import json
import math
import sys

TERMS = json.load(open(sys.argv[1]))
OUT = sys.argv[2]

P_I = 0.2          # library capture success on a single fish (Turesson & Bronmark 2004 single prey, 220)
SCAN = 0.5         # library half-field scan
VREL = 1.1         # v_rel school / v_rel dispersed prey (u 0.3 vs 0.2 m/s, section 220 script) [D]

# (low protection, central, high protection)
PAR = {
    # herring and sprat: day schools, dispersed at night
    "R_s": {"herring": (30.0, 15.0, 10.0), "sprat": (30.0, 15.0, 10.0), "pout": (15.0, 8.0, 5.0),
            "blue whiting": (15.0, 8.0, 5.0), "juv": (8.0, 4.0, 2.0), "whiting": (10.0, 5.0, 3.0),
            "mackerel": (50.0, 25.0, 15.0)},
    "N": {"herring": (2000, 25000, 100000), "sprat": (2000, 25000, 100000), "pout": (500, 5000, 50000),
          "blue whiting": (200, 2000, 20000), "juv": (20, 300, 1000), "whiting": (50, 300, 3000),
          "mackerel": (3000, 300000, 3000000)},
    "k": (3.0, 1.0, 0.3),
    "sigma_day": {"herring": (0.6, 0.9, 1.0), "sprat": (0.6, 0.9, 1.0), "pout": (0.0, 0.5, 0.9),
                  "blue whiting": (0.0, 0.2, 0.5), "juv": (0.0, 0.0, 0.3), "whiting": (0.0, 0.1, 0.5),
                  "mackerel": (0.8, 1.0, 1.0)},
    "sigma_night": {"herring": (0.0, 0.0, 0.2), "sprat": (0.0, 0.0, 0.2), "pout": (0.0, 0.25, 0.9),
                    "blue whiting": (0.0, 0.1, 0.5), "juv": (0.0, 0.0, 0.15), "whiting": (0.0, 0.05, 0.25),
                    "mackerel": (0.8, 1.0, 1.0)},
}
READ = ("low", "central", "high")


def S(sp, R_i, i):
    Rs, N, k = PAR["R_s"][sp][i], PAR["N"][sp][i], PAR["k"][i]
    return (Rs * Rs / (SCAN * R_i * R_i)) * VREL * k / (N * P_I)


# Terms per pair. In-layer-derived pairs from pair_terms.json (rule A, band). Mackerel as prey: the derivation
# reuses the planktivore term template (one_rule_pairs caveat 9), so every term of that pair is mackerel.
# Benthivore fish pairs (benth_pair, already band; not in pair_terms.json): prey split from the library comment
# (pout 0.85 of benthivores -> planktivores, section 269 line) and the day share 1 / (1 + night/day) of the
# library dark_ratio (0.141, 0.168) at equal day and night time [D, approximation]; R_i = 2 L-capped cod / haddock
# day radius at the bed under rule A, 0.6 m central [J], night 0.3 m (R_nv fish).
PREYMAP = {"herring": "herring", "sprat": "sprat", "pout": "pout", "blue whiting": "blue whiting",
           "juvenile benthivores": "juv", "whiting": "whiting"}
pairs = {}
for name, d in TERMS.items():
    rows = []
    for key, share in d["shares"].items():
        prey, per = key.split("|")
        sp = "mackerel" if name.endswith("->mackerel") else PREYMAP[prey]
        R = d["reff"].get(key)
        if R is None:                       # bed slab terms (juvenile benthivores, whiting): rule A at the bed
            R = 0.6 if per == "day" else 0.3
        rows.append((sp, per, share, R))
    pairs[name] = dict(rows=rows, nd=d["nd"])
for name, nd, plk in (("benthivorous_fish->planktivorous_fish", 0.141, True),
                      ("benthivorous_fish->whiting", 0.168, False)):
    dshare = 1.0 / (1.0 + nd)
    if plk:
        rows = [("pout", "day", 0.85 * dshare, 0.6), ("pout", "night", 0.85 * (1 - dshare), 0.3),
                ("herring", "day", 0.15 * dshare, 0.6), ("herring", "night", 0.15 * (1 - dshare), 0.3)]
    else:
        rows = [("whiting", "day", dshare, 0.6), ("whiting", "night", 1 - dshare, 0.3)]
    pairs[name] = dict(rows=rows, nd=nd)

CAP = True   # S capped at 1: an aggregation whose geometry gives S > 1 is one the predator does not meet as a unit
             # (loose aggregation, R_s ~ R_i), and the field data (Andersen 2017, Rindorf & Gislason 2005,
             # Turesson & Bronmark 2004 groups 0.03 vs single 0.20) put day-schooled clupeids on the protected side [J]


def S_eff(sp, R, i, over=None):
    if over and sp in over.get("S", {}):
        return over["S"][sp]
    v = S(sp, R, i)
    return min(v, 1.0) if CAP else v


def sigma(sp, per, i, over=None):
    if over and sp in over.get("sigma", {}):
        return over["sigma"][sp][0 if per == "day" else 1]
    return PAR["sigma_day" if per == "day" else "sigma_night"][sp][i]


# Variants (i = the reading index used for every factor not overridden)
VARIANTS = {
    "low": (0, None),
    "central": (1, None),
    "high": (2, None),
    # herring and sprat only: pout and blue whiting not schooling (their schooling evidence is [A]/[J])
    "central_clupeids_only": (1, {"sigma": {"pout": (0.0, 0.0), "blue whiting": (0.0, 0.0)}}),
    # field-bounded day risk for schooled clupeids: S = 0.01 (Andersen 2017 day/twilight 0.001, day/night 0.04)
    "central_fieldS": (1, {"S": {"herring": 0.01, "sprat": 0.01}}),
}

res = {}
print("S = per-capita capture rate in a school / dispersed (rule-A radius of the term), and term factor F = 1 - sigma (1 - S)")
print("Per term: S uncapped (theory) low / central / high protection; F with S capped at 1.\n")
for name, d in pairs.items():
    res[name] = {}
    print(f"{name}  (today's night/day {d['nd']:.3f})")
    for sp, per, share, R in d["rows"]:
        if share < 0.005:
            continue
        sv = [S(sp, R, i) for i in range(3)]
        sig = [sigma(sp, per, i) for i in range(3)]
        F = [1 - sig[i] * (1 - S_eff(sp, R, i)) for i in range(3)]
        print(f"   {sp:13s} {per:5s} share {share:5.3f} R_i {R:4.2f} m  S {sv[0]:7.3g} / {sv[1]:7.3g} / {sv[2]:7.3g}"
              f"   sigma {sig[0]:.2f} / {sig[1]:.2f} / {sig[2]:.2f}   F {F[0]:6.3f} / {F[1]:6.3f} / {F[2]:6.3f}")
    for vn, (i, over) in VARIANTS.items():
        tot = day = night = dsh = nsh = 0.0
        for sp, per, share, R in d["rows"]:
            f = 1 - sigma(sp, per, i, over) * (1 - S_eff(sp, R, i, over))
            tot += share * f
            if per == "day":
                day += share * f
                dsh += share
            else:
                night += share * f
                nsh += share
        Fd = day / dsh if dsh else 1.0
        Fn = night / nsh if nsh else 1.0
        nd_new = d["nd"] * Fn / Fd
        res[name][vn] = dict(F=tot, F_day=Fd, F_night=Fn, nd_new=nd_new)
        print(f"   -> {vn:22s}: pair factor {tot:6.3f}  (day {Fd:6.3f}, night {Fn:6.3f})  night/day {d['nd']:.3f} -> {nd_new:.3f}"
              + ("  (>= 1: no light dependence in the engine)" if nd_new >= 1 else ""))
    print()

# field cross-check (Andersen et al. 2017, Bornholm Basin, late winter, cod on sprat): 194 sprat ingestion times
n_tot, n_day, f_dusk, f_dawn, f_night = 194, 1, 0.397, 0.474, 0.124
h_twi = 2.0          # dusk 17:00-18:00 and dawn 04:25-05:25 (their echogram windows)
h_night = 24 - 11.0 - h_twi
h_day = 11.0         # [J] day length late winter at 55 N, ~11 h between the windows
r_day = n_day / n_tot / h_day
r_twi = (f_dusk + f_dawn) / h_twi
r_night = f_night / h_night
print("Field cross-check, Andersen et al. 2017 (cod on sprat; sprat schools at the cod's depth by day):")
print(f"   ingestion per hour (share of 194 sprat): day {r_day:.2e}, twilight {r_twi:.3f}, night {r_night:.4f}")
print(f"   day / twilight {r_day / r_twi:.4f}   day / night {r_day / r_night:.3f}")
print("   (confounded: cod satiation after the dawn meal, night overlap; an upper bound on the day per-capita risk"
      " relative to dissolving schools)")
res["_field_andersen2017"] = dict(day_over_twilight=r_day / r_twi, day_over_night=r_day / r_night)

# Rindorf & Gislason 2005: clupeid plateau K 0.10-0.17 of Wmax with intake 'almost exclusively' at dawn and dusk
for K in (0.10, 0.17):
    print(f"Rindorf & Gislason 2005 plateau K {K:.2f} of Wmax = a feeding window of {24 * K:.1f} h per day at the digestive"
          " rate [D reading]")
json.dump(dict(PAR={k: (v if not isinstance(v, dict) else v) for k, v in PAR.items()}, pairs=res), open(OUT, "w"), indent=1)
print(f"\nwritten {OUT}")
