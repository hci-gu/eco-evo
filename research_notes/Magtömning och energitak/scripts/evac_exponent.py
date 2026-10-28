"""Energy-density exponent b of gastric evacuation in gadoids (mass rate ~ E^-b).

Data are typed in from the sources (see summary.md); nothing is downloaded here.
Run: python3 -I evac_exponent.py  (prints to stdout; saved in ../tables/evac_exponent_stdout.txt)

b = 1 -> energy evacuated per hour independent of E (energy ceiling, the library's h = E/Cmax)
b = 0 -> mass evacuated per hour independent of E (mass ceiling)
"""
import math


def fit_loglog(pts):
    """Least squares slope of ln(rate) on ln(E). Returns slope, intercept."""
    xs = [math.log(e) for e, _ in pts]
    ys = [math.log(r) for _, r in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return b, my - b * mx


print("=" * 78)
print("A. dos Santos & Jobling 1995 (ICES JMS 52:209), cod, Table 1 (E, kJ/g wet)")
print("   and Table 3 restricted form, BW: half-life scale alpha (h); mass rate ~ 1/alpha")
print("=" * 78)
ds = {  # name: (E kJ/g, alpha h, class, item size g)
    "Lugworm (Arenicola)": (2.5, 30, "polychaete", "5-12"),
    "Squid mantle": (4.4, 78, "mollusc", "10-73"),
    "Krill": (4.0, 41, "crustacean", "0.4"),
    "Prawn (Pandalus)": (4.9, 103, "crustacean", "2-8"),
    "Herring": (6.8, 88, "fish", "4-48"),
    "Capelin": (5.5, 58, "fish", "4-28"),
    "Redfish": (3.5, 68, "fish", "7-29"),
    "Long rough dab": (3.0, 62, "fish", "6-17"),
    "Haddock": (3.2, 84, "fish", "16-40"),
    "Polar cod": (3.3, 59, "fish", "14-28"),
}
ref = ds["Capelin"]
print(f"{'prey':22s} {'E':>5s} {'alpha':>6s} {'mass rate':>10s} {'energy rate':>12s}  (rel. capelin)")
for k, (e, a, c, sz) in ds.items():
    mr = (1 / a) / (1 / ref[1])
    er = (e / a) / (ref[0] / ref[1])
    print(f"{k:22s} {e:5.1f} {a:6.0f} {mr:10.2f} {er:12.2f}   {c}, items {sz} g")
fish = [(e, 1 / a) for k, (e, a, c, _) in ds.items() if c == "fish"]
b, c0 = fit_loglog(fish)
print(f"\nFish prey only (n=6): slope of ln(mass rate) on ln(E) = {b:+.2f} -> b = {-b:.2f}")
fish_nr = [(e, 1 / a) for k, (e, a, c, _) in ds.items() if c == "fish" and k != "Redfish"]
b2, _ = fit_loglog(fish_nr)
print(f"Fish prey without redfish (spiny; n=5): b = {-b2:.2f}")
fat = [ds["Herring"][:2], ds["Capelin"][:2]]
lean = [ds[k][:2] for k in ("Redfish", "Long rough dab", "Haddock", "Polar cod")]
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
Ef, af = gm([e for e, _ in fat]), gm([a for _, a in fat])
El, al = gm([e for e, _ in lean]), gm([a for _, a in lean])
print(f"Fatty (herring, capelin) geo-mean E {Ef:.2f}, alpha {af:.0f}; lean four E {El:.2f}, alpha {al:.0f}")
print(f"  two-point b = {math.log(af / al) / math.log(Ef / El):.2f}; "
      f"energy per gut-hour fatty/lean = {(Ef / af) / (El / al):.2f}")
print(f"  herring/haddock energy per gut-hour = {(6.8 / 88) / (3.2 / 84):.2f}; "
      f"capelin/LRD = {(5.5 / 58) / (3.0 / 62):.2f}")
print(f"Shrimp armour (cod): Pandalus vs fish fit at E 4.9: alpha 103 vs "
      f"{1 / math.exp(c0 + b * math.log(4.9)):.0f} -> x{103 * math.exp(c0 + b * math.log(4.9)):.2f} slower")
print(f"Krill vs fish fit at 4.0: x{41 * math.exp(c0 + b * math.log(4.0)):.2f} (1 = like fish)")

print()
print("=" * 78)
print("B. Temming & Herrmann 2003 (Fish Res 63:21) cod rho (g^0.5/h, wet), via the")
print("   cluster table (Appendix S7, Table G1) of the 2023 Ecol Evol cod paper (PMC10667610)")
print("=" * 78)
th = {  # name: (E kJ/g, rho)
    "Annelids (Arenicola)": (2.50, 0.01414),
    "Flatfish (LRD)": (3.00, 0.00677),
    "Other fish (whiting+goby)": (4.23, 0.00597),
    "Clupeids (herring+sprat)": (6.58, 0.00507),
    "Crangon+Pandalus": (4.58, 0.00445),
    "Liocarcinus (fit to Andersen 2016)": (3.54, 0.0034),
    "Chionoecetes (fit to Andersen 2016)": (4.53, 0.0028),
    "Carcinus (regression est.)": (2.81, 0.00381),
}
rc = th["Clupeids (herring+sprat)"]
print(f"{'prey':36s} {'E':>5s} {'rho':>8s} {'mass':>6s} {'energy':>7s}  (rel. clupeids, equal S)")
for k, (e, r) in th.items():
    print(f"{k:36s} {e:5.2f} {r:8.5f} {r / rc[1]:6.2f} {e * r / (rc[0] * rc[1]):7.2f}")
fish = [th[k] for k in ("Flatfish (LRD)", "Other fish (whiting+goby)", "Clupeids (herring+sprat)")]
b3, c3 = fit_loglog(fish)
print(f"\nFish clusters (n=3): b = {-b3:.2f} (T&H's own fitted exponent reported as 0.45 [S])")
fit_at = lambda e: math.exp(c3 + b3 * math.log(e))
print(f"Shrimp armour: fish-fit rho at 4.58 = {fit_at(4.58):.5f} -> x{fit_at(4.58) / 0.00445:.2f} slower")
print(f"Crab armour: Liocarcinus x{fit_at(3.54) / 0.0034:.2f}, Chionoecetes x{fit_at(4.53) / 0.0028:.2f}")
print(f"Polychaete vs fish-fit at 2.5: x{0.01414 / fit_at(2.5):.2f} FASTER (soft body, no bones)")

print()
print("=" * 78)
print("C. Andersen's gadoid (cylinder / square-root) models")
print("=" * 78)
print("Whiting, Andersen 1999 abstract: evacuation time x1.6-1.7 for E 3.4 -> 5.6 kJ/g")
for f in (1.6, 1.65, 1.7):
    print(f"  time ratio {f}: b = {math.log(f) / math.log(5.6 / 3.4):.2f}")
print("Gadoids (whiting form, Andersen pers. comm. in Velasco & Olaso 2000): E^-0.875 -> b 0.875")
print("Cod, Andersen 2012 (Andersen et al. 2017 suppl. eq. S2): E^-0.85 -> b 0.85")
print("Plaice, Jobling 1980 abstract: energy evacuation rate independent of diet energy -> b 1")

print()
print("=" * 78)
print("D. Herring : infauna energy per gut-hour at a full stomach, (E_h/E_i)^(1-b)")
print("   E_h = 7.124 (library, herring with reserve at fill), E_i = 2.6 (infauna)")
print("   'floor' = Andersen's E floor 3.5 kJ/g applied to the infauna (mass rate capped")
print("   at the 3.5 kJ/g value), meaningful only in the Andersen branch")
print("=" * 78)
Eh, Ei = 7.124, 2.6
for bb in (0.0, 0.35, 0.45, 0.5, 0.6, 0.7, 0.85, 0.875, 1.0):
    r = (Eh / Ei) ** (1 - bb)
    rf = (Eh ** (1 - bb)) / (Ei * 3.5 ** (-bb))
    print(f"  b={bb:5.3f}: ratio {r:4.2f}   with floor {rf:4.2f}")
