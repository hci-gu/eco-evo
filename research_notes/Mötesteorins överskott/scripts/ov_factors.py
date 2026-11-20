"""Arithmetic for lit_overlap.md (prints only).
1) Factor on a from a covariance: F = mean(P N)/(mean P mean N) = 1 + r CV_P CV_N.
2) Two-zone co-location (Bertrand et al. 2014 densities in/out of structures).
3) Saturated-intake loss for a predator that does not track a patchy prey (occupancy p, in-patch density N/p):
   R = p*f(x/p)/f(x), f(x) = x/(1+x), x = a h N  ->  R = (1+x)/(1+x/p).
4) r needed for a target factor at given CVs."""
def two_zone(s, kp, kn):
    num = s*kp*kn + (1-s); den = (s*kp + (1-s))*(s*kn + (1-s)); return num/den
print("1) F = 1 + r CVp CVn examples")
for r in (-0.3, -0.5, -0.7, -0.9):
    for cv in (0.5, 1.0, 1.5):
        print(f"   r {r:+.1f} CVp=CVn={cv}: F {1 + r*cv*cv:.2f}")
print("2) Bertrand 2014 two-zone, fish x1.92, zoo x1.088 inside structures (density):")
for s in (0.1, 0.2, 0.3, 0.5):
    print(f"   volume share s {s}: F {two_zone(s,1.92,1.088):.3f}")
print("   column biomass x10.5 fish, x2.5 zoo (upper bound, includes vertical stretching):")
for s in (0.1, 0.2, 0.3):
    print(f"   s {s}: F {two_zone(s,10.5,2.5):.3f}")
print("3) untracked patchy prey, intake ratio R = (1+x)/(1+x/p); S = x/(1+x) mean-field saturation")
for S in (0.1, 0.2, 0.5):
    x = S/(1-S)
    print(f"   S {S}: " + ", ".join(f"p {p}: R {(1+x)/(1+x/p):.2f}" for p in (0.05, 0.1, 0.3, 0.5, 0.8)))
print("4) r needed for F target (CVp, CVn):")
for Ft in (0.3, 0.5, 0.7):
    for cv in (0.5, 1.0, 1.5):
        r = (Ft-1)/(cv*cv)
        print(f"   F {Ft} CV {cv}: r {r:+.2f}" + ("  (impossible, |r|>1)" if r < -1 else ""))
print("5) Christiansen 2024 krill -68% under schools >10 m by day: local factor 0.32;")
for share_sch in (0.3, 0.6, 0.9):
    for day in (0.33, 0.5):
        f = 1 - day*share_sch*0.55*(1-0.32)
        print(f"   schooled share of herring {share_sch}, daytime share of encounter {day}: group factor {f:.2f}")
