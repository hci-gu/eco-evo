"""Candidate rest-payoff forms: implied pair floors and the payoff they would give (Q3).

Usage: python3 -I candidate_forms.py <repo_root> > ../tables/candidate_forms_stdout.txt

Library read: booked M2, today's floors (fgconfig/fg_library.yaml). Inputs below are tagged; nothing is
taken from model output except the fix105 yardstick rho (rest share a full fish could take; sizing only,
see rest_payoff.py). A floor f is the engine's form: a resting share is exposed at f (r = 1/f).

Forms (each a per-pair floor; none caps T, none is keyed on season):
  V  motionless rest: the resting prey does not swim, so the encounter falls from v_rel(v, u) to v_rel(v, 0)
     (Gerritsen-Strickler, the a's own convention). Per member: f = v_rel(v, 0)/v_rel(v, u); ambushers
     f = 0. Obligate swimmers (herring, sprat, blue whiting schools, mackerel) do not stop: f = 1.
     Pair floor = sum over prey members (take share) x predator members (share of a, day/night).
  M  motion detection: a resting prey is detected over a smaller width/area by a motion-dependent sense.
     f = 1 - m (1 - d), m the share of the predator's detections made by that sense, d the still/moving
     detection ratio. Fish predators: [J] from PROXY ratios (lit_activity_risk.md); mammals: the mammal
     subagent's r per pair (lit_mammals.md, [J] on F/L components).
  Q  vigilance split: q_rest / q_feed at a fixed mean q. Sign evidence both ways (foraging fish react later,
     Krause & Godin 1996 [A]; resting/sleep-like fish have raised thresholds, Yokogawa 2007 [F] PROXY);
     no magnitude for any model species -> f = 1 (0.8-1.25) [J]; not computed further.
  VM V x M combined (independent factors: encounter speed vs detection width).
"""
import os
import sys

import yaml

ROOT = os.path.abspath(sys.argv[1])
LIB = yaml.safe_load(open(os.path.join(ROOT, "fgconfig", "fg_library.yaml")))
S = LIB["species_definitions"]
I = LIB["interaction_definitions"]

RHO = {"planktivorous_fish": 0.55, "benthivorous_fish": 0.35, "whiting": 0.53, "predatory_fish": 0.49,
       "mackerel": 0.51}  # rest_payoff_stdout.txt part B (fix105 yardstick)


def vrel(v, u):
    return (u * u + 3 * v * v) / (3 * v) if v >= u else (v * v + 3 * u * u) / (3 * u)


def fv(v, u):
    """still/moving encounter ratio for a cruising predator at v (m/s) and prey at u (m/s)."""
    return vrel(v, 0.0) / vrel(v, u)


def day_share(dark_ratio):
    """share of a pair's daily a carried by day, from the pair's night/day ratio (12 h each) [D]."""
    return 1.0 / (1.0 + dark_ratio)


# predator members: (share of the pair's a [S/J], day speed m/s, night speed m/s or None = same, ambusher)
# speeds: research_notes/"Attackrater ur motesteori" central BL/s x member length [S]; shares [J/S] from
# research_notes/"Bottenfiskens och vattenpelarens golv" D1 (saithe 0.89 of WC -> benthivores a; hake 0.093;
# anglerfish 0.044 of the take) and "Rovfiskens uppdelning" (saithe 0.80 of the WC biomass).
WC = [(0.85, 0.50, 0.30, False), (0.08, 0.25, None, False), (0.04, 0.35, None, False), (0.03, 0.0, None, True)]
COD = [(1.0, 0.16, None, False)]
WHI = [(1.0, 0.224, None, False)]
MAC = [(1.0, 0.576, None, False)]

# prey members: (take share [J/S], cruising u m/s, stops at rest?)
# planktivores: Norway pout carries 0.52-0.75 of the fish pairs' planktivore a (research_notes/"Planktivorernas
# golv" Gaps) -> 0.6 central [S/J]; whether pout rest motionless is unknown [J]; herring/sprat/blue whiting
# school and keep swimming (0.25-0.42 m/s when not feeding, "Planktivorernas golv" E8 [A]).
PLANK = [(0.6, 0.10, True), (0.4, 0.20, False)]
PLANK_MAC = [(1.0, 0.20, False)]  # mackerel -> herring and sprat only
JUVGAD = [(1.0, 0.10, True)]       # juvenile haddock/cod on the bed (cod adults ~0.70 of days quiet on the
                                   # bed, Hobson et al. 2007 [F, via lit_activity_budgets.md])
WHITING = [(1.0, 0.10, None)]      # whiting hover with a swim bladder; 'rest = motionless' unobserved [J]


def pair_fv(pred, prey, dark_ratio, whiting_stops):
    ds = day_share(dark_ratio) if dark_ratio is not None else 1.0
    tot = 0.0
    for ws, u, stops in prey:
        if stops is None:
            stops = whiting_stops
        if not stops:
            tot += ws
            continue
        sub = 0.0
        for wp, vd, vn, amb in pred:
            if amb:
                f = 0.0
            else:
                f = ds * fv(vd, u) + (1 - ds) * fv(vn if vn else vd, u)
            sub += wp * f
        tot += ws * sub
    return tot


# Form M inputs [J]
# fish predators: m = share of detections by a motion-dependent sense on a still bed/near-bed prey, d = still /
# moving detection. PROXY ratios: merganser on coho 10.6 per look (Martel & Dill 1995 [F]), catfish wake 5.3
# per time in darkness (Pohlmann et al. 2001 [F]), crappie 3x distance on copepods (Wright & O'Brien 1984 [S]),
# stickleback: motion matters only on a matching background (Ioannou & Krause 2009 [F]). A gadoid seen in
# silhouette or by smell at 100-300 m is not background-matched [J] -> d 0.5 (0.2-1.0) for the share m 0.2
# (0-0.5) of detections that depend on motion (lateral line, wake; vision in the dark) [J].
M_FISH = {"central": (0.2, 0.5), "low": (0.0, 1.0), "high": (0.5, 0.2)}
# mammals: floors from lit_mammals.md (the subagent's r per pair; central, low-floor end, high-floor end)
M_MAM = {("seals", "benthivorous_fish"): (0.61, 0.3, 0.95),   # 0.57 gadoids r 2 / 0.43 flatfish r 1.3
         ("seals", "whiting"): (0.7, 0.25, 1.0),
         ("seals", "predatory_fish"): (0.75, 0.33, 1.0),
         ("seals", "planktivorous_fish"): (0.9, 0.5, 1.0),       # rest only; 0.35 today stands for schooling
         ("porpoises", "benthivorous_fish"): (0.76, 0.6, 0.93),   # 0.64 gadoids 0.9 / 0.36 flatfish 0.5
         ("porpoises", "whiting"): (0.95, 0.77, 1.0),
         ("porpoises", "predatory_fish"): (0.95, 0.77, 1.0),
         ("porpoises", "planktivorous_fish"): (1.0, 0.83, 1.0)}   # r 1 (0.6-1.2): rest may be riskier

FISHPAIRS = [("predatory_fish", "planktivorous_fish", WC, PLANK), ("benthivorous_fish", "planktivorous_fish", COD, PLANK),
             ("whiting", "planktivorous_fish", WHI, PLANK), ("mackerel", "planktivorous_fish", MAC, PLANK_MAC),
             ("predatory_fish", "benthivorous_fish", WC, JUVGAD), ("whiting", "benthivorous_fish", WHI, JUVGAD),
             ("predatory_fish", "whiting", WC, WHITING), ("benthivorous_fish", "whiting", COD, WHITING),
             ("predatory_fish", "mackerel", WC, [(1.0, 0.6, False)])]

SH = {"planktivorous_fish": "planktivores", "benthivorous_fish": "benthivores", "whiting": "whiting",
      "predatory_fish": "water column", "mackerel": "mackerel", "seals": "seals", "porpoises": "porpoises"}


def today_floor(pred, prey):
    v = I[f"{pred}_preys_on_{prey}"]
    return float(v.get("visibility_floor", S[prey].get("visibility_floor", 1.0)))


rows = []
print("Per-pair floors by form (V: motionless rest; M: motion detection; VM: both). Today = library floor.")
print(f"{'pair':36s} {'M2':>6s} {'today':>6s} {'V(wst)':>7s} {'V(whv)':>7s} {'M c':>6s} {'M lo-hi':>11s} "
      f"{'VM c':>6s} {'VM lo':>6s}")
for pred, prey, P, Q in FISHPAIRS:
    d = I[f"{pred}_preys_on_{prey}"]
    m2 = float(d["predation_mortality"])
    dr = d.get("dark_ratio")
    v_stop = pair_fv(P, Q, dr, True)    # whiting stops at rest
    v_hover = pair_fv(P, Q, dr, False)  # whiting hovers (does not stop)
    mc, dc = M_FISH["central"]; ml, dl = M_FISH["high"]
    stop_share = sum(w for w, u, s in Q if s is not False)
    f_mc = 1 - stop_share * mc * (1 - dc)
    f_mlo = 1 - stop_share * ml * (1 - dl)
    vm_c = v_stop * f_mc
    vm_lo = v_stop * f_mlo
    rows.append((pred, prey, m2, today_floor(pred, prey), v_stop, f_mc, vm_c, vm_lo, v_hover))
    print(f"{SH[pred] + ' -> ' + SH[prey]:36s} {m2:6.4f} {today_floor(pred, prey):6.2f} {v_stop:7.3f} {v_hover:7.3f} "
          f"{f_mc:6.3f} {f_mlo:5.3f}-1.000 {vm_c:6.3f} {vm_lo:6.3f}")
for (pred, prey), (c, lo, hi) in M_MAM.items():
    d = I[f"{pred}_preys_on_{prey}"]
    m2 = float(d["predation_mortality"])
    rows.append((pred, prey, m2, today_floor(pred, prey), 1.0, c, c, lo, 1.0))
    print(f"{SH[pred] + ' -> ' + SH[prey]:36s} {m2:6.4f} {today_floor(pred, prey):6.2f} {'~1.0':>7s} {'~1.0':>7s} "
          f"{c:6.2f} {lo:5.2f}-{hi:5.2f} {c:6.3f} {lo:6.3f}")
print("V(wst) = whiting stops at rest; V(whv) = whiting hovers at rest (f = 1 for whiting as prey).")
print("V for mammals ~0.99-1.0 (search 1.0-1.2 m/s vs prey 0.1-0.2 m/s; rest_payoff_stdout.txt part D).")

print("\nPayoff per unit rest share, sum_p M2_p (1 - f_p) (/yr), and at the yardstick rho (%/yr of biomass)")
print(f"{'prey':14s} {'today':>14s} {'V only':>14s} {'VM central':>14s} {'VM low-floor':>14s}  of which mammals (VM c)")
for prey in ["planktivorous_fish", "benthivorous_fish", "whiting", "predatory_fish", "mackerel"]:
    rs = [r for r in rows if r[1] == prey]
    t = sum(r[2] * (1 - r[3]) for r in rs)
    # V only: fish pairs at V, mammal pairs keep today's floors (V ~1 for mammals)
    v = sum(r[2] * (1 - (r[4] if r[0] not in ("seals", "porpoises") else r[3])) for r in rs)
    c = sum(r[2] * (1 - r[6]) for r in rs)
    lo = sum(r[2] * (1 - r[7]) for r in rs)
    mam = sum(r[2] * (1 - r[6]) for r in rs if r[0] in ("seals", "porpoises"))
    rho = RHO[prey]
    print(f"{SH[prey]:14s} {t:6.4f} {100 * rho * t:5.2f}% {v:6.4f} {100 * rho * v:5.2f}% {c:6.4f} {100 * rho * c:5.2f}% "
          f"{lo:6.4f} {100 * rho * lo:5.2f}%  {mam:.4f}")
print("\nSeals -> planktivores: today's 0.35 is the schooling stand-in of the seal a (no F), not a rest value;")
print("the VM columns replace it with 0.9 for rest only, which drops 0.0028 of today's planktivore payoff.")
print("All M2 are BOOKED (upper bounds); realised fish-pair M2 is lower for several pairs.")
