"""Schooling factor on the encounter a, re-run with Norway pout (and blue whiting) parameters from the field
evidence of research_notes/"Vitlinglojans stim"/evidence.md. Research only; nothing fitted, no model output read.

Copied from research_notes/"Stimskydd i motes-a"/scripts/school_factor.py (the original is unchanged). The algebra,
the term shares (pair_terms.json from that note, copied here unchanged), the capture p, scan, v_rel ratio, k and the
benthivore pair's prey split are the original's. What changed:
  - PAR_NOTE = the original note's parameters (reproduced, so its readings can be printed next to the new ones);
  - PAR_NEW  = the same, except pout and blue whiting (sigma day / night, N, R_s) from this note's evidence;
  - a printed derivation of pout N and R_s from school geometry and acoustic packing (E5, E6, E14, E15);
  - a sweep of pout sigma_day x (N, R_s) for the three pairs pout matters to.

Usage: python3 -I school_factor_pout.py <pair_terms.json> <out_json>

S = (R_s^2 / (scan R_i^2)) (v_rel,s / v_rel,i) k / (N p_i), capped at 1; term factor F = 1 - sigma (1 - S);
pair factor = sum over terms of (term share of today's a) x F.
"""
import copy
import json
import math
import sys

TERMS = json.load(open(sys.argv[1]))
OUT = sys.argv[2]

P_I = 0.2          # library capture success on a single fish (section 220/269)
SCAN = 0.5         # library half-field scan
VREL = 1.1         # v_rel school / v_rel dispersed prey (original note) [D]

# ---------------------------------------------------------------------------------------------------------------
# 1. Pout school geometry and packing, printed with every assumption [D]
# ---------------------------------------------------------------------------------------------------------------
print("1. Norway pout day-school size from acoustic evidence [D]")
# E6 (Fassler 2010 thesis ch. 2 / Fassler et al. 2007): pout and herring echotraces near the seabed, summer, NW North
# Sea, daytime; "densely packed schools, which are similar in shape and size"; mean Sv at 38 kHz of both ~ -45 dB
# (Fig. 2.3b, eroded echotraces, read from the figure, +-2 dB); pout echotraces mostly 5-15 cm fish (Table 2.2).
SV38 = -45.0
SV38_P05, SV38_P95 = -50.0, -37.0      # 5th / 95th percentile of the pout echotraces (Fig. 2.3b, read)
# ICES standard TS-length relations at 38 kHz: physoclist gadoids 20 log L - 67.5, herring 20 log L - 71.2 [S]
def ts_gadoid(L):
    return 20 * math.log10(L) - 67.5
def ts_herring(L):
    return 20 * math.log10(L) - 71.2
def density(sv_db, ts_db):
    return 10 ** ((sv_db - ts_db) / 10)
L_POUT_ECHO = 10.0     # cm, mid of the 5-15 cm pout echotraces (E6)
L_POUT_BOX = 12.0      # cm, Skagerrak modal length, IBTS Q1 (E3)
L_HER = 27.0           # cm, modal herring of the same echotraces (E6, Table 2.2: 25-30 cm)
n_pout = density(SV38, ts_gadoid(L_POUT_ECHO))
n_pout_lo, n_pout_hi = density(SV38_P05, ts_gadoid(L_POUT_ECHO)), density(SV38_P95, ts_gadoid(L_POUT_ECHO))
n_her = density(SV38, ts_herring(L_HER))
ratio = 10 ** ((ts_herring(L_HER) - ts_gadoid(L_POUT_ECHO)) / 10)
print(f"   TS38 pout {L_POUT_ECHO:.0f} cm {ts_gadoid(L_POUT_ECHO):.1f} dB, box pout {L_POUT_BOX:.0f} cm {ts_gadoid(L_POUT_BOX):.1f} dB,"
      f" herring {L_HER:.0f} cm {ts_herring(L_HER):.1f} dB")
print(f"   packing from the echotrace mean Sv {SV38:.0f} dB: pout {n_pout:.1f} fish/m3 (5-95 %: {n_pout_lo:.1f}-{n_pout_hi:.1f}),"
      f" herring {n_her:.2f} fish/m3")
print(f"   equal Sv => pout / herring number density = {ratio:.1f}")
# Two volume anchors for a herring-like school:
#  (a) Fernandes et al. 2001 (note 'Stimskydd' E22): day herring schools 21-27 m long, 3.5-5.2 m high; volume of an
#      ellipsoid with width = length: pi/6 L W H
#  (b) Misund et al. 1995 (note 'Stimskydd' E21): ~25 000 herring at 4.25 fish/m3 => volume 25 000 / 4.25
V_a = math.pi / 6 * 24.0 * 24.0 * 4.3
V_b = 25000 / 4.25
N_a = V_a * n_pout                    # absolute packing from Sv, Fernandes volume
N_b = 25000 * ratio                   # Misund herring school, pout packed `ratio` times denser
print(f"   (a) volume {V_a:.0f} m3 x pout {n_pout:.1f}/m3 -> N {N_a:.0f}   (5-95 % packing: {V_a * n_pout_lo:.0f}-{V_a * n_pout_hi:.0f})")
print(f"   (b) Misund herring school ({V_b:.0f} m3, 25 000 fish) x pout/herring ratio {ratio:.1f} -> N {N_b:.0f}")
N_central = math.sqrt(N_a * N_b)
print(f"   geometric mid of (a) and (b): N {N_central:.0f}; used: central 10 000, range 2 500 - 50 000")
print("   R_s = half-length 10.5-13.5 m (herring-like, 21-27 m) + detection distance 0-5 m near the bed at 100-300 m by day")
print("        [J]: central 13 m, range 10-18 m\n")

# ---------------------------------------------------------------------------------------------------------------
# 2. Parameter sets. (low protection, central, high protection)
# ---------------------------------------------------------------------------------------------------------------
PAR_NOTE = {
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
PAR_NEW = copy.deepcopy(PAR_NOTE)
# Norway pout (evidence.md E1-E17):
#  night: a homogeneous near-bed layer 5-10 m thick over long distances, resolved single targets [F] -> sigma 0 (0-0.1)
#  day: below the 1.5 m detection height in the box (Albert 1994) - state unobserved; schools exist elsewhere
#       (Fassler, summer, juveniles, NW North Sea); "no organized schools in the way herring do" (Raitt 1968);
#       "non-schooling" prey, whiting ingestion peak ~15 h (Rindorf & Gislason 2005) -> sigma 0.4 (0.15-0.7) [J]
#  N, R_s from section 1 [D/J]
PAR_NEW["sigma_day"]["pout"] = (0.15, 0.4, 0.7)
PAR_NEW["sigma_night"]["pout"] = (0.0, 0.0, 0.1)
PAR_NEW["N"]["pout"] = (2500, 10000, 50000)
PAR_NEW["R_s"]["pout"] = (18.0, 13.0, 10.0)
# Blue whiting (E18-E19, abstracts): more dispersed and higher at night over < 350 m; near-bottom concentrations at
# the shelf break day and night -> day kept at the note's 0.2 (0-0.5) [J/A], night 0 (0-0.1) [A]
PAR_NEW["sigma_night"]["blue whiting"] = (0.0, 0.0, 0.1)

READ = ("low", "central", "high")
CAP = True


def S(par, sp, R_i, i, Rs=None, N=None):
    Rs = par["R_s"][sp][i] if Rs is None else Rs
    N = par["N"][sp][i] if N is None else N
    v = (Rs * Rs / (SCAN * R_i * R_i)) * VREL * par["k"][i] / (N * P_I)
    return v


def S_eff(par, sp, R, i, over=None):
    over = over or {}
    if sp in over.get("S", {}):
        return over["S"][sp]
    v = S(par, sp, R, i, Rs=over.get("R_s", {}).get(sp), N=over.get("N", {}).get(sp))
    return min(v, 1.0) if CAP else v


def sigma(par, sp, per, i, over=None):
    over = over or {}
    if sp in over.get("sigma", {}):
        return over["sigma"][sp][0 if per == "day" else 1]
    return par["sigma_day" if per == "day" else "sigma_night"][sp][i]


# Terms per pair, exactly as the original script builds them
PREYMAP = {"herring": "herring", "sprat": "sprat", "pout": "pout", "blue whiting": "blue whiting",
           "juvenile benthivores": "juv", "whiting": "whiting"}
pairs = {}
for name, d in TERMS.items():
    rows = []
    for key, share in d["shares"].items():
        prey, per = key.split("|")
        sp = "mackerel" if name.endswith("->mackerel") else PREYMAP[prey]
        R = d["reff"].get(key)
        if R is None:
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


def pair_factor(d, par, i, over=None):
    tot = day = night = dsh = nsh = 0.0
    for sp, per, share, R in d["rows"]:
        f = 1 - sigma(par, sp, per, i, over) * (1 - S_eff(par, sp, R, i, over))
        tot += share * f
        if per == "day":
            day += share * f
            dsh += share
        else:
            night += share * f
            nsh += share
    Fd = day / dsh if dsh else 1.0
    Fn = night / nsh if nsh else 1.0
    return dict(F=tot, F_day=Fd, F_night=Fn, nd_new=d["nd"] * Fn / Fd)


VARIANTS = {
    # the original note's readings, reproduced
    "note_central": (PAR_NOTE, 1, None),
    "note_clupeids_only": (PAR_NOTE, 1, {"sigma": {"pout": (0.0, 0.0), "blue whiting": (0.0, 0.0)}}),
    # this note: pout and blue whiting from evidence; herring, sprat and everything else as the note's central
    "pout_low": (PAR_NEW, 1, {"sigma": {"pout": (0.15, 0.0)}, "N": {"pout": 2500}, "R_s": {"pout": 18.0}}),
    "pout_central": (PAR_NEW, 1, None),
    "pout_high": (PAR_NEW, 1, {"sigma": {"pout": (0.7, 0.1)}, "N": {"pout": 50000}, "R_s": {"pout": 10.0}}),
    # pout and blue whiting not schooling at all (clupeids only), on PAR_NEW (= note_clupeids_only)
    "pout_none": (PAR_NEW, 1, {"sigma": {"pout": (0.0, 0.0), "blue whiting": (0.0, 0.0)}}),
    # every factor at its high-protection end (herring, k, ... too) - the note's "high" with the new pout
    "all_high": (PAR_NEW, 2, None),
}

print("2. Per-term S and F (pout and blue whiting terms; PAR_NEW low / central / high, S uncapped shown)")
for name, d in pairs.items():
    shown = [r for r in d["rows"] if r[0] in ("pout", "blue whiting") and r[2] >= 0.005]
    if not shown:
        continue
    print(f"   {name}")
    for sp, per, share, R in shown:
        sv = [S(PAR_NEW, sp, R, i) for i in range(3)]
        sig = [sigma(PAR_NEW, sp, per, i) for i in range(3)]
        F = [1 - sig[i] * (1 - min(sv[i], 1.0)) for i in range(3)]
        print(f"      {sp:13s} {per:5s} share {share:5.3f} R_i {R:4.2f} m  S {sv[0]:7.3g} / {sv[1]:7.3g} / {sv[2]:7.3g}"
              f"  sigma {sig[0]:.2f} / {sig[1]:.2f} / {sig[2]:.2f}  F {F[0]:5.3f} / {F[1]:5.3f} / {F[2]:5.3f}")
print("   (the per-term 'low'/'high' columns use the reading index for every factor incl. k; the pout_low / pout_high"
      " variants below move only pout)\n")

res = {"pairs": {}}
print("3. Pair factors F (night/day today -> new)")
hdr = "   pair".ljust(42) + "".join(f"{v:>20s}" for v in VARIANTS)
print(hdr)
for name, d in pairs.items():
    res["pairs"][name] = {}
    line = f"   {name:39s}"
    for vn, (par, i, over) in VARIANTS.items():
        r = pair_factor(d, par, i, over)
        res["pairs"][name][vn] = r
        line += f"{r['F']:>20.3f}"
    print(line)
print()
for name in ("predatory_fish->planktivorous_fish", "whiting->planktivorous_fish", "mackerel->planktivorous_fish",
             "benthivorous_fish->planktivorous_fish"):
    d = pairs[name]
    print(f"   {name}: night/day today {d['nd']:.3f}")
    for vn in ("note_central", "note_clupeids_only", "pout_low", "pout_central", "pout_high"):
        r = res["pairs"][name][vn]
        print(f"      {vn:20s} F {r['F']:.3f} (day {r['F_day']:.3f}, night {r['F_night']:.3f}) night/day -> {r['nd_new']:.3f}"
              + ("  (>= 1: no light dependence in the engine; stomach ratio kept)" if r['nd_new'] >= 1 else ""))
print()

print("4. Sweep: pout sigma_day x (N, R_s); pout sigma_night 0; everything else PAR_NEW central")
combos = [(2500, 13.0), (10000, 13.0), (50000, 13.0), (5000, 8.0), (10000, 18.0), (10000, 10.0)]
sweep_pairs = ("predatory_fish->planktivorous_fish", "whiting->planktivorous_fish",
               "benthivorous_fish->planktivorous_fish")
res["sweep"] = []
print("   sigma_day   N      R_s  " + "".join(f"{p.split('->')[0][:13]:>16s}" for p in sweep_pairs))
for sd in (0.0, 0.25, 0.4, 0.5, 0.75):
    for N, Rs in combos:
        over = {"sigma": {"pout": (sd, 0.0)}, "N": {"pout": N}, "R_s": {"pout": Rs}}
        Fs = [pair_factor(pairs[p], PAR_NEW, 1, over)["F"] for p in sweep_pairs]
        res["sweep"].append(dict(sigma_day=sd, N=N, R_s=Rs, F=dict(zip(sweep_pairs, Fs))))
        print(f"   {sd:5.2f}  {N:7d}  {Rs:5.1f}  " + "".join(f"{f:16.3f}" for f in Fs))
    if sd == 0.0:
        print("   (sigma_day 0: N and R_s inert)")
print()
res["derivation"] = dict(n_pout=n_pout, n_pout_p05=n_pout_lo, n_pout_p95=n_pout_hi, n_her=n_her, ratio=ratio,
                         V_a=V_a, V_b=V_b, N_a=N_a, N_b=N_b, N_geo=N_central)
res["PAR_NEW"] = PAR_NEW
json.dump(res, open(OUT, "w"), indent=1)
print(f"written {OUT}")
