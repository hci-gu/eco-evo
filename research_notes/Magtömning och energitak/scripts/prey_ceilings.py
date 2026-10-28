"""Per-prey energy ceiling and handling time for the benthivore pairs under an
evacuation exponent b (mass evacuation ~ E^-b) and an armour factor A (time per g
relative to soft fish prey of equal E).

  ceiling_j (MJ gross / t predator / tick) = C_ref * (E_j / E_ref)^(1-b) / A_j
  h_j (ticks per t prey per t predator)     = E_j / ceiling_j

C_ref = 26.18 MJ/t/tick is the library's benthivore ceiling (bottom layer ~7.4 C).
E_ref = 4.184 kJ/g: the FB4 cod Cmax (Hansson et al. 1996) is a wet-mass ration of
mostly fish food, converted to energy at 4.184 kJ/g in the benthivore note (evidence
A); the Jobling 1988 cod Cmax was measured on capelin (~5.5 kJ/g; sensitivity below).
b = 1 reproduces the library exactly (h = E / 26.18), whatever E_ref.
'floor': Andersen's reported floor (no E effect below ~3.5 kJ/g [S]) applied to infauna.

Run: python3 -I prey_ceilings.py > ../tables/prey_ceilings_stdout.txt
"""
C_REF = 26.18

# name: (E kJ/g as eaten, library, assimilation, armour central, low, high)
PREY = {
    "planktivores (herring)": (7.124, 0.8, 1.0, 1.0, 1.0),
    "whiting (lean fish)":    (4.800, 0.8, 1.0, 1.0, 1.0),
    "mobile epifauna":        (3.750, 0.8, 1.75, 1.4, 2.3),
    "Nephrops":               (3.600, 0.8, 2.0, 1.5, 2.5),
    "infauna":                (2.600, 0.7, 1.0, 0.6, 1.5),
}


def ceiling(e, b, a, e_ref, floor=None):
    """Gross energy per tick at a full gut. With a floor, the mass rate below
    the floor stays at the floor value (rate ~ max(E, floor)^-b)."""
    if floor is not None and e < floor:
        mass_rate = (floor / e_ref) ** (-b)
    else:
        mass_rate = (e / e_ref) ** (-b)
    return C_REF * (e / e_ref) * mass_rate / a


def table(b, e_ref=4.184, floor=None, armour="central"):
    idx = {"central": 2, "low": 3, "high": 4, "none": None}[armour]
    rows = []
    for k, v in PREY.items():
        e, ass = v[0], v[1]
        a = 1.0 if idx is None else v[idx]
        c = ceiling(e, b, a, e_ref, floor if k == "infauna" else None)
        rows.append((k, e, a, c, e * 1000 / c, c * ass))
    return rows


def show(title, b, **kw):
    rows = table(b, **kw)
    print(f"\n{title}")
    print(f"  {'prey':24s} {'E':>6s} {'A':>5s} {'ceiling':>8s} {'h':>7s} {'assim.':>7s} {'h/lib':>6s}")
    lib = {k: v[0] * 1000 / C_REF for k, v in PREY.items()}
    for k, e, a, c, h, ca in rows:
        print(f"  {k:24s} {e:6.3f} {a:5.2f} {c:8.2f} {h:7.1f} {ca:7.2f} {h / lib[k]:6.2f}")
    d = {r[0]: r for r in rows}
    hi, inf = d["planktivores (herring)"], d["infauna"]
    print(f"  herring/infauna gross {hi[3] / inf[3]:.2f}, assimilated {hi[5] / inf[5]:.2f}; "
          f"herring/epifauna assim. {hi[5] / d['mobile epifauna'][5]:.2f}; "
          f"herring/whiting {hi[3] / d['whiting (lean fish)'][3]:.2f}")


print("Library today (b = 1, no armour): every prey ceiling 26.18 gross; assimilated 20.94 (0.8) / 18.33 (0.7)")
for b in (1.0, 0.85, 0.75, 0.7, 0.6, 0.45, 0.37, 0.16):
    show(f"b = {b} (armour central, no floor)", b)
print("\n--- Andersen branch with the 3.5 kJ/g floor on infauna ---")
for b in (0.85, 0.875):
    show(f"b = {b}, floor 3.5", b, floor=3.5)
print("\n--- Armour sensitivity at b = 0.7 ---")
for arm in ("none", "low", "high"):
    show(f"b = 0.7, armour {arm}", 0.7, armour=arm)
print("\n--- E_ref sensitivity (capelin 5.5 kJ/g as the Cmax reference) ---")
show("b = 0.7, E_ref 5.5", 0.7, e_ref=5.5)
show("b = 0.37, E_ref 5.5", 0.37, e_ref=5.5)

print("\n--- Group-weighted b (J): haddock 0.39 and others 0.10 at Andersen 0.875, flatfish 0.20 at 1.0, cod 0.31 at x ---")
for bc in (0.16, 0.37, 0.45, 0.6, 0.85):
    bg = 0.39 * 0.875 + 0.10 * 0.875 + 0.20 * 1.0 + 0.31 * bc
    print(f"  cod b {bc:4.2f} -> group b {bg:.2f}")
