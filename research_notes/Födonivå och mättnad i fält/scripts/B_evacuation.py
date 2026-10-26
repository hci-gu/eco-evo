"""B_evacuation.py - gastric evacuation times vs the 6-h tick, meal size vs
daily Cmax, and what a feeding-window (bout) schedule does to daily intake
when the stomach buffers.

Evacuation: ICES WGSAM 2024 (ICES Sci. Rep. 6:13) Table 3-7, generic
cylinder model for one stomach:
    dS/dt = -R' S^0.5,  R' = rho M exp(delta T) L^lambda E^-0.85   (g/h)
rho, lambda, delta per species from Table 3-7; M = prey armament (1 for
fish and soft prey; 0.5 is a judgement for armoured crustaceans/brittle
stars after Andersen 1999 "almost twice as long" for brown shrimp in
whiting); the table's K = 0.85 is not used here.
Integrated:  sqrt(S(t)) = sqrt(S0) - R' t / 2
    t_empty = 2 sqrt(S0) / R'
    t_half  = 2 (sqrt(S0) - sqrt(S0/2)) / R'
Steady-state ration at a mean fill S: 24 R' sqrt(S) g/d.
Weights: W = 0.01 L^3 g (Ursin et al. 1985 "1% of length cubed").
Cmax check: Hansson et al. 1996 cod (Fish Bioenergetics 4 row:
CA 0.099, CB -0.195, CQ 2.41, CTO 13.7, CTM 21; Kitchell f(T)).

Bout schedule: n feeding windows per day; the fish fills to S_max in each
window (encounter not limiting), then evacuates for 24/n h. Daily intake
= n (S_max - S(24/n)); compared with continuous feeding at S_max.

Run: python3 -I B_evacuation.py > ../tables/B_evacuation_stdout.txt
"""
import math

GER = {  # species: (rho, lambda, delta)  WGSAM 2024 Table 3-7
    "cod": (0.00224, 1.30, 0.083),
    "haddock": (0.00191, 1.30, 0.083),
    "whiting": (0.00171, 1.35, 0.081),
    "saithe": (0.00171, 1.35, 0.081),
}


def r_prime(sp, L, T, E, M=1.0):
    rho, lam, dl = GER[sp]
    return rho * M * math.exp(dl * T) * L**lam * E**-0.85


def kitchell_f(T, CQ=2.41, CTO=13.7, CTM=21.0):
    V = (CTM - T) / (CTM - CTO)
    Z = math.log(CQ) * (CTM - CTO)
    Y = math.log(CQ) * (CTM - CTO + 2)
    X = Z**2 * (1 + (1 + 40 / Y) ** 0.5) ** 2 / 400
    return V**X * math.exp(X * (1 - V))


def cmax_cod(W, T):
    return 0.099 * W**-0.195 * kitchell_f(T)  # g prey / g fish / d


def main():
    T = 7.5
    print(f"# T = {T} C (bottom water 100-300 m, 7-8 C)")
    cases = [("cod", 40), ("cod", 60), ("haddock", 40), ("whiting", 30)]
    preys = [("fish/soft, E 4.2", 4.2, 1.0), ("benthos soft, E 3.5", 3.5, 1.0),
             ("benthos armoured, E 3.5, M 0.5 (J)", 3.5, 0.5)]
    print("\n## evacuation of a single meal S0 (% BW); times in hours")
    print("species L(cm) W(g) prey                                 S0%  R'(g/h) t_half t_empty ticks_half ticks_empty")
    for sp, L in cases:
        W = 0.01 * L**3
        for pname, E, M in preys:
            R = r_prime(sp, L, T, E, M)
            for s0p in (1.0, 2.0, 5.0):
                S0 = s0p / 100 * W
                te = 2 * math.sqrt(S0) / R
                th = 2 * (math.sqrt(S0) - math.sqrt(S0 / 2)) / R
                print(f"{sp:<8s}{L:>4d} {W:6.0f} {pname:<36s} {s0p:4.1f} {R:7.4f} {th:6.1f} {te:7.1f} {th / 6:9.1f} {te / 6:10.1f}")

    print("\n## steady-state ration (Cmax-equivalent) at mean fill S, %BW/d")
    print("species L  prey                      S=1%BW  S=2.5%BW  S=5%BW   FB4 cod Cmax (if cod)")
    for sp, L in cases:
        W = 0.01 * L**3
        for pname, E, M in preys:
            R = r_prime(sp, L, T, E, M)
            vals = [100 * 24 * R * math.sqrt(s / 100 * W) / W for s in (1.0, 2.5, 5.0)]
            cm = f"{100 * cmax_cod(W, T):.2f}" if sp == "cod" else "-"
            print(f"{sp:<8s}{L:>3d} {pname[:24]:<24s} " + " ".join(f"{v:8.2f}" for v in vals) + f"   {cm}")
    print("# FB4 Cmax is in g of 4.184 kJ/g prey; compare with the fish/soft E 4.2 rows")

    print("\n## fill needed to sustain FB4 cod Cmax (fish prey E 4.2)")
    for L in (40, 60):
        W = 0.01 * L**3
        R = r_prime("cod", L, T, 4.2)
        C = cmax_cod(W, T) * W  # g/d
        S = (C / 24 / R) ** 2
        print(f"cod {L} cm, W {W:.0f} g: Cmax {100 * C / W:.2f} %BW/d -> mean fill {100 * S / W:.2f} %BW")

    print("\n## bout schedule: fill to S_max in each of n windows/day, evacuate between")
    print("species L prey               S_max%BW  n  intake/continuous  intake %BW/d  continuous %BW/d")
    for sp, L in (("cod", 40), ("haddock", 40), ("whiting", 30)):
        W = 0.01 * L**3
        for pname, E, M in preys[:3:2]:
            R = r_prime(sp, L, T, E, M)
            for smax_p in (2.5, 5.0):
                Smax = smax_p / 100 * W
                cont = 24 * R * math.sqrt(Smax)
                for n in (1, 2, 4):
                    dt = 24 / n
                    root = max(0.0, math.sqrt(Smax) - R * dt / 2)
                    intake = n * (Smax - root**2)
                    print(f"{sp:<8s}{L:>3d} {pname[:18]:<18s} {smax_p:8.1f} {n:2d} {intake / cont:12.3f} {100 * intake / W:13.2f} {100 * cont / W:15.2f}")

    print("\n## daily intake if the fish eats ONLY what one window's encounter allows")
    print("# feeding level = n_windows x meal / Cmax; e.g. whiting 250 g eating one 9-15 g clupeid")
    for W, prey_g, cm in ((250, 9, 0.033), (250, 15, 0.033)):
        for days_per_capture in (1, 2, 4):
            fl = prey_g / days_per_capture / (cm * W)
            print(f"whiting {W} g, prey {prey_g} g, one capture per {days_per_capture} d: feeding level {fl:.2f} of Wmax 3.3 %BW/d")


if __name__ == "__main__":
    main()
