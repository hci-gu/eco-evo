"""Vertical refuge as rest: per-pair floors from a refuge profile vs a foraging profile (scope addition).

Usage: python3 -I vertical_refuge.py > ../tables/vertical_refuge_stdout.txt

No file is read; every input is listed and tagged here. No model output.

Form R: 'rest' = the prey's refuge profile phi_R(h), 'forage' = its foraging profile phi_F(h) (h = height
above the bed). Per predator p with time-at-height density f_p(h) and vision area A_p(h) (rule A form):
    floor_p = sum_h f_p phi_R A_p / sum_h f_p phi_F A_p
This can exceed 1: a refuge from one predator can raise exposure to another. The engine's floor is a share
<= 1 (visible = B (1 - rest (1 - floor))), so a value > 1 needs the floor generalised to an exposure
multiplier (or a second, rest-state band profile per pair, a_rest / a_forage).

Only one box prey group has a NAMED refuge vs foraging layer in the field: Norway pout in the Norwegian
Deep (Albert 1994, doi:10.1006/jmsc.1994.1005 [F, via lit_vertical_refuge.md]): by day in the ~1.5 m
near-bottom zone, at night homogeneous in the lowest 5-10 m, feeding more intense at night; the author
offers the antipredator reading only as a hypothesis, the mix share is not identified. Second, a PROXY stage
and area: settled 0-group cod and haddock (Lough et al. 1989, doi:10.3354/meps056001 [F]): within cm of the
bed by day, 'part of the population' rises 3-5 m at night to feed.

Predator profiles [S/J] (research_notes/"Overlapp i fiskparen" band_pairs.py docstring, "Bottenfiskens
hojdprofil", "Sal och tumlare golv"): saithe exp(15 m) above the bed by day; hake exp(8 m); cod 0.70 of
time in the bottom 2 m (Hobson et al. 2007 SB class [F via lit_activity_budgets.md]) + 0.30 uniform 2-20 m
[J]; whiting uniform bottom 10 m; harbour seal at a 140 m bed: 0.7 in the bottom 5 m, 0.3 uniform [J];
harbour porpoise at a 140 m bed: 0.25 in the bottom 5 m, 0.75 uniform [J] (E6 of "Sal och tumlare golv":
25 % of diving time on the bottom). Mammals: no light term (model convention).
Light at a 140 m bed by day [J]: E(bed) 0.024 lx (5e4 lx at the surface, K 0.12 to 65 m, K_deep 0.09
below), E(h) = E(bed) exp(0.09 h); night 0 (g = 0). Rule A: R^2 = R_nv^2 (1 - g) + R_sat^2 g,
g = E / (E + 0.02 lx), R_nv 0.3 m, R_sat = min(R0, 2 L): saithe 1.0, hake 1.0, cod 0.7, whiting 0.5 m [J].
"""
import math

DH = 0.05
H = [DH * (i + 0.5) for i in range(int(140 / DH))]   # 0-140 m above a 140 m bed


def norm(w):
    s = sum(w)
    return [x / s for x in w]


def uni(a, b):
    return norm([1.0 if a <= h < b else 0.0 for h in H])


def expo(L):
    return norm([math.exp(-h / L) for h in H])


def mix(*pairs):
    out = [0.0] * len(H)
    for w, prof in pairs:
        out = [o + w * p for o, p in zip(out, prof)]
    return out


PRED = {
    "saithe (WC)": (expo(15.0), 1.0, True),
    "hake (WC)": (expo(8.0), 1.0, True),
    "cod (benthivores)": (mix((0.7, uni(0, 2)), (0.3, uni(2, 20))), 0.7, True),
    "whiting": (uni(0, 10), 0.5, True),
    "harbour seal": (mix((0.7, uni(0, 5)), (0.3, uni(0, 140))), None, False),
    "harbour porpoise": (mix((0.25, uni(0, 5)), (0.75, uni(0, 140))), None, False),
}
E_BED = 0.024


def area(h, rsat, day):
    if rsat is None:
        return 1.0
    e = E_BED * math.exp(0.09 * h) if day else 0.0
    g = e / (e + 0.02)
    return 0.3 ** 2 * (1 - g) + rsat ** 2 * g


def exposure(prey, pred, rsat, vis, day):
    return sum(f * p * (area(h, rsat, day) if vis else 1.0) for h, f, p in zip(H, pred, prey))


CASES = [
    ("Norway pout: refuge 0-1.5 m vs forage uniform 0-10 m (Albert 1994 layers)", uni(0, 1.5), uni(0, 10)),
    ("Norway pout: refuge 0-1.5 m vs forage uniform 5-10 m (strict reading)", uni(0, 1.5), uni(5, 10)),
    ("0-group cod/haddock: refuge 0-0.5 m vs forage 3-5 m (Lough 1989, PROXY stage)", uni(0, 0.5), uni(3, 5)),
]
for lab, ref, fora in CASES:
    print(lab)
    for day in (True, False):
        print(f"  {'day (rule A light)' if day else 'night (dark: vision at R_nv)'}")
        for name, (prof, rsat, vis) in PRED.items():
            er = exposure(ref, prof, rsat, vis, day)
            ef = exposure(fora, prof, rsat, vis, day)
            print(f"    {name:20s} floor = refuge/forage exposure {er / ef:6.3f}")
    print()

print("Check (not a fit): can a mix w x refuge + (1 - w) x forage reproduce the library's pout share in the")
print("bottom 10 m (0.7 by day, 0.4 by night, [J], krill_pairs_wc.py pout_prof)? Both Albert layers lie inside")
print("the bottom 10 m, so every mix gives a bottom-10-m share of 1.0:")
for w in (0.0, 0.5, 1.0):
    m = mix((w, uni(0, 1.5)), (1 - w, uni(0, 10)))
    print(f"  w {w:.1f}: bottom-10-m share {sum(p for h, p in zip(H, m) if h < 10):.2f}")
print("-> the library's pout profile is not a mix of these two layers (it spreads the remainder over midwater")
print("   after Onsrud et al. 2004, Oslofjord, where pout sit in midwater at night with no fullness difference).")
