"""Size of the January light residual in consumption terms: monthly share of each pair's annual flow under the
engine's m(t) and under the derived monthly a (light_pairs.json), weighted by the presence arrays the engine
runs; and each predator's / prey's monthly intake / M2 change if the derived timing replaced m(t).

Usage: python3 -I flow_shift.py <repo_root> ../tables/light_pairs.json > ../tables/flow_shift_stdout.txt

Linear (unsaturated) regime: flow_j(m) ~ a_j m_j(m) w_j(m) with w = predator presence x prey presence
(library outside_season of predatory_fish, mackerel, planktivorous_fish, krill; resident groups 1); booked
annual flow_j = M2_j x B_prey (inference_initial_biomass, mareld2.yaml). Saturation (Holling, shared handling)
damps every number here; these are upper bounds of the timing error. Nothing fitted, no model output read.
"""
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.abspath(sys.argv[1])
R = json.load(open(sys.argv[2]))
LIB = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml"), encoding="utf-8-sig"))
MAN = yaml.safe_load(open(os.path.join(ROOT, "mareld2.yaml")))
SP, INTER = LIB["species_definitions"], LIB["interaction_definitions"]
MN = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
MD = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], float)

B = {}
for sec in ("decision_makers", "non_decision_makers"):
    for e in MAN.get(sec, []):
        B[e["group_id"]] = float(e.get("inference_initial_biomass", 0.0))
DM = [e["group_id"] for e in MAN["decision_makers"] if not e.get("muted")]


def presence(g):
    s = SP.get(g, {}).get("outside_season")
    return np.array(s, float) if s else np.ones(12)


def w_pair(n):
    p, q = n.split("->")
    return presence(p) * presence(q)


READS = list(R["res"])
PAIRS = list(R["pairs"])
print("Presence arrays used (library outside_season; 1 = resident or none):")
for g in ("predatory_fish", "mackerel", "planktivorous_fish", "krill"):
    print(f"  {g:20s} " + " ".join(f"{x:.2f}" for x in presence(g)))

# ------------------------------------------------------------------ 1. per pair: monthly flow share, engine vs derived
print("\n1. Per pair: share of the annual flow per month (%), engine m(t) vs derived a, presence-weighted; and the")
print("   annual flow ratio sum(der w) / sum(eng w) (the annual mean of a is equal-time 1 in both; presence moves it)")
TAB = {}
for rn in READS:
    print(f"\n  [{rn}]")
    print(f"  {'pair':38s} {'':4s} " + " ".join(f"{m:>5s}" for m in MN) + "  annual  |shift| %")
    TAB[rn] = {}
    for n in PAIRS:
        eng = np.array(R["res"][rn][n]["eng"])
        der = np.array(R["res"][rn][n]["der"])
        w = w_pair(n) * MD
        fe = eng * w / (eng * w).sum() * 100
        fd = der * w / (der * w).sum() * 100
        ann = (der * w).sum() / (eng * w).sum()
        shift = 0.5 * np.abs(fd - fe).sum()
        TAB[rn][n] = dict(fe=fe.tolist(), fd=fd.tolist(), annual=ann, shift=shift)
        if rn.startswith(("R0", "R3")):
            print(f"  {n:38s} eng  " + " ".join(f"{x:5.1f}" for x in fe))
            print(f"  {'':38s} der  " + " ".join(f"{x:5.1f}" for x in fd) + f"  x{ann:.3f}  {shift:5.1f}")
        else:
            print(f"  {n:38s} d-e  " + " ".join(f"{x:+5.1f}" for x in fd - fe) + f"  x{ann:.3f}  {shift:5.1f}")

# ------------------------------------------------------------------ 1b. Dec-Jan share and tonnes moved
FL0 = {}
for k, v in INTER.items():
    if isinstance(v, dict) and "_preys_on_" in k and v.get("predation_mortality"):
        p, q = k.split("_preys_on_")
        if q in B:
            FL0[f"{p}->{q}"] = float(v["predation_mortality"]) * B[q]
print("\n1b. Dec + Jan share of each pair's annual flow (%), engine -> derived, and the booked tonnes per year moved out")
print("    of (-) or into (+) Dec-Jan at the booked annual flow (M2 x inference biomass), per reading")
print(f"  {'pair':38s} {'flow t/yr':>9s}  " + "  ".join(f"{rn[:2]:>16s}" for rn in READS))
for n in PAIRS:
    f = FL0.get(n, 0.0)
    cells = []
    for rn in READS:
        t = TAB[rn][n]
        e = t["fe"][11] + t["fe"][0]
        d = t["fd"][11] + t["fd"][0]
        cells.append(f"{e:4.1f}->{d:4.1f} {f * (d - e) / 100:+6.0f}")
    print(f"  {n:38s} {f:9.0f}  " + "  ".join(f"{c:>16s}" for c in cells))

# ------------------------------------------------------------------ 2. per predator: monthly intake change
print("\n2. Per predator: monthly encounter-weighted intake under the derived timing / under m(t) (all its booked pairs;")
print("   pairs without a dark_ratio flat), linear regime; booked diet shares of its light pairs in brackets")
FLOWS = {}
for k, v in INTER.items():
    if not isinstance(v, dict) or "_preys_on_" not in k:
        continue
    p, q = k.split("_preys_on_")
    m2 = v.get("predation_mortality")
    if p in DM and m2 and q in B:
        FLOWS[f"{p}->{q}"] = float(m2) * B[q]
PRED = {}
for rn in READS:
    PRED[rn] = {}
    print(f"\n  [{rn}]")
    for p in ("predatory_fish", "whiting", "benthivorous_fish", "mackerel", "planktivorous_fish"):
        num = np.zeros(12)
        den = np.zeros(12)
        light_share = 0.0
        tot = sum(f for n, f in FLOWS.items() if n.startswith(p + "->"))
        for n, f in FLOWS.items():
            if not n.startswith(p + "->"):
                continue
            w = w_pair(n)
            if n in R["res"][rn] and R["pairs"][n]["rho"] < 1:          # whiting->krill (no dark_ratio) stays flat
                eng = np.array(R["res"][rn][n]["eng"])
                der = np.array(R["res"][rn][n]["der"])
                light_share += f / tot
            else:
                eng = der = np.ones(12)
            # flow_j(m) = F_j x (shape) with the annual booked flow F_j fixed under the engine timing
            fe = f * eng * w / (eng * w * MD).sum() * MD.sum()
            fd = f * der * w / (eng * w * MD).sum() * MD.sum()
            num += fd
            den += fe
        r = num / den
        PRED[rn][p] = r.tolist()
        print(f"  {p:20s} (light pairs {light_share:.2f} of booked diet) " + " ".join(f"{x:5.2f}" for x in r)
              + f"   Dec-Jan {(num[[11, 0]].sum() / den[[11, 0]].sum()):.3f}  Nov-Feb "
              f"{(num[[10, 11, 0, 1]].sum() / den[[10, 11, 0, 1]].sum()):.3f}")

# ------------------------------------------------------------------ 3. per prey: monthly M2 change from the light pairs
print("\n3. Per prey: monthly predation (all booked DM predators; light pairs re-timed) derived / engine")
for rn in READS:
    print(f"\n  [{rn}]")
    for q in ("planktivorous_fish", "krill", "benthivorous_fish", "whiting", "zooplankton", "mackerel"):
        num = np.zeros(12)
        den = np.zeros(12)
        for n, f in FLOWS.items():
            if not n.endswith("->" + q):
                continue
            w = w_pair(n)
            if n in R["res"][rn] and R["pairs"][n]["rho"] < 1:
                eng = np.array(R["res"][rn][n]["eng"])
                der = np.array(R["res"][rn][n]["der"])
            else:
                eng = der = np.ones(12)
            num += f * der * w / (eng * w * MD).sum()
            den += f * eng * w / (eng * w * MD).sum()
        r = num / den
        print(f"  {q:20s} " + " ".join(f"{x:5.2f}" for x in r) + f"   Dec-Jan {(num[[11, 0]].sum() / den[[11, 0]].sum()):.3f}")
json.dump(dict(pairs=TAB, predators=PRED, flows=FLOWS),
          open(os.path.join(os.path.dirname(sys.argv[2]), "flow_shift.json"), "w"), indent=1)
