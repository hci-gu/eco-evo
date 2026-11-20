"""Does a foraging-time share act on the ceiling? Engine form vs a gut-buffered fish [D].

Usage: python3 -I s3_gut_window.py

Engine (Holling with h = E/Cmax, i.e. the SSS limit in which ingestion while foraging can never outrun digestion):
    daily feeding level  D_eng = w X / (1 + X)            (w = share of the day spent foraging)
A fish with a stomach of capacity G (in days of Cmax) that keeps digesting at Cmax while it rests: during a
foraging window of w x 24 h it ingests at its encounter rate X (Cmax units; X = sum a B_vis h) until the stomach
is full; over a periodic day
    D_gut = min(1, w X, G + w)  (the gut fills in the window, empties over the day; digestion during the window
                                 adds w).  First-order bound, item handling neglected (pursuit seconds, E1 of
                                 "Intag under fodosok").
A rest spell of n days (no foraging) with the same G: over a cycle of n + 1 days with one foraging day,
    D_spell = min(n + 1, X, G + 1) / (n + 1)
G (days of Cmax in one satiation meal) [D on F/S]: ceiling 104.7 kJ/kg/d at 7.4 C ("Exponering av bottenbyten"),
i.e. 2.1 %BW/d on fish (5 kJ/g) and 2.8 %BW/d on shrimp-like epifauna (3.75 kJ/g); satiation meals of cod
17.8 %BW on fish, 9.1 % on Crangon, 2.8 % on crabs (Arnott & Pihl 2000 [F], via "Mattnad hos bottenbyten")
-> G 8.5 / 3.3 / 1.0 days. Group X_eff from s1 (frozen fix105 yardstick, rough).
"""
X_GROUPS = {"planktivores": 1.34, "whiting": 5.13, "water column": 3.53, "mackerel": 1.30, "benthivores": 36.3}
G_CASES = {"crabs 1.0 d": 1.0, "Crangon 3.3 d": 3.3, "fish 8.5 d": 8.5}
W = (1.0, 0.75, 0.5, 0.25, 0.125)

print("Daily feeding level (share of Cmax) with foraging share w of each day")
print(f"{'group':14s} {'X':>6s} {'w':>6s} {'engine wX/(1+X)':>16s} " + " ".join(f"{k:>14s}" for k in G_CASES))
for g, X in X_GROUPS.items():
    for w in W:
        eng = w * X / (1 + X)
        gut = [min(1.0, w * X, G + w) for G in G_CASES.values()]
        print(f"{g:14s} {X:6.2f} {w:6.3f} {eng:16.3f} " + " ".join(f"{v:14.3f}" for v in gut))
print("\nRest spells of n whole days between single foraging days (benthivores X 36.3)")
for n in (0, 1, 2, 4, 8):
    print(f"  n {n}: engine {1 / (n + 1) * 36.3 / 37.3:.3f}; gut "
          + ", ".join(f"{k} {min(n + 1, 36.3, G + 1) / (n + 1):.3f}" for k, G in G_CASES.items()))
print("\nReading: under a gut buffer a daily foraging window lowers the daily ration only where w X < 1, i.e. where")
print("the in-window encounter is itself low; the engine's T share lowers it linearly at any X. A time share")
print("therefore stands for the field only if the non-foraging spells outlast the gut buffer, or if it is read")
print("together with a low in-window encounter (an a-side error).")
