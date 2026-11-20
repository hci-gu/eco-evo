"""Derived numbers [D] for lit_mammals.md (activity-risk ratio, mammals).

Run: python3 -I mammal_activity_risk.py
No inputs; every constant is a sourced literature value or a stated assumption.

D1  Gas-model encounter ratio, moving vs still prey, for a searcher at v
    and prey at u (random directions; 3D and 2D bed plane).
D2  Seal vibrissal detection width / area: a moving fish's trail of
    length L = u * T_age against a still fish's breathing-current radius.
D3  Bowen et al. 2002 per-VSU captures (attempts x success).
D4  Porpoise hearing threshold below 16 kHz extrapolated from Kastelein 2002.
"""
import math


def enc_3d(v, u):
    # Gerritsen & Strickler 1977: mean relative speed for v >= u
    if v >= u:
        return (u * u + 3 * v * v) / (3 * v)
    return (v * v + 3 * u * u) / (3 * u)


def enc_2d(v, u, n=20000):
    # mean |v_vec - u_vec| over a uniform angle in the plane
    s = 0.0
    for i in range(n):
        th = 2 * math.pi * (i + 0.5) / n
        s += math.sqrt(v * v + u * u - 2 * v * u * math.cos(th))
    return s / n


print("D1 gas-model encounter ratio active/rest = <|v-u|> / v")
for v in (0.5, 1.0, 1.4, 1.9):
    row = []
    for u in (0.2, 0.5, 1.0):
        row.append("u=%.1f: 3D %.2f 2D %.2f" % (u, enc_3d(v, u) / v, enc_2d(v, u) / v))
    print("  v=%.1f m/s  " % v + "; ".join(row))

print("\nD2 seal vibrissae: detection width (2D bed) and area (3D) moving vs still")
# still fish: breathing-current radius R_b where peak velocity > threshold
# (Niesterok 2017 jeb.158055: 16 cm flounder current 2-2.5 cm/s at the
# 36 cm path distance, threshold 3.7-4.2 cm/s -> R_b < ~0.25-0.36 m; take
# 0.1 / 0.25 / 0.4 m).  moving fish: trail radius a (paddle wake 0.1-0.25 m
# half-width at 5-35 s, Wieskotten 2010a; seal wake ~0.8-1 m half-width,
# Schulte-Pelkum 2007), detectable age T (15 s glide-impaired, 30-35 s lab
# max; Wieskotten 2010a,b), fish speed u.
cases = [
    ("low", 0.4, 0.10, 0.2, 10.0),
    ("central", 0.25, 0.20, 0.3, 20.0),
    ("high", 0.10, 0.25, 0.5, 35.0),
]
for name, Rb, a, u, T in cases:
    L = u * T
    w_move = 2 * (Rb + a) + (2 / math.pi) * L      # randomly oriented segment, 2D
    w_rest = 2 * Rb
    # 3D: mean projected area of a sausage (cylinder radius r=Rb+a, length L)
    r = Rb + a
    A_move = math.pi * r * L / 2 + math.pi * r * r
    A_rest = math.pi * Rb * Rb
    print("  %-7s R_b %.2f a %.2f u %.1f T %4.0f -> L %.1f m; 2D width %.2f/%.2f = %.1fx; 3D area %.2f/%.3f = %.1fx"
          % (name, Rb, a, u, T, L, w_move, w_rest, w_move / w_rest, A_move, A_rest, A_move / A_rest))

print("\nD3 Bowen et al. 2002 Table 2, per video-sampling unit (VSU, 10 min)")
for lab, att, succ, search in (("cryptic sand lance", 5.5, 63.7, 235), ("flounder", 1.9, 79.9, 180),
                               ("schooling sand lance", 9.2, 37.5, 135)):
    print("  %-21s attempts %.1f x success %.3f = captures %.2f; search %d s" % (lab, att, succ / 100, att * succ / 100, search))
print("  capture-success ratio schooling/cryptic sand lance = %.2f; schooling/flounder = %.2f"
      % (37.5 / 63.7, 37.5 / 79.9))
print("  search-time ratio cryptic/schooling sand lance = %.2f" % (235 / 135))

print("\nD4 porpoise threshold below 16 kHz (best ~33 dB at 100-140 kHz; within 10 dB over 16-140 kHz;")
print("   -10 dB/octave below 16 kHz; Kastelein et al. 2002 abstract)")
for f in (16000, 8000, 4000, 2000, 1000, 500, 250):
    oct_ = math.log2(16000 / f)
    print("  %6d Hz: ~%.0f dB re 1 uPa (taking 43 dB at 16 kHz)" % (f, 43 + 10 * oct_))
