"""Mackerel density (t/km2) per stratum from Degnbol, Iversen, Kirkegaard & Lewy 1988 (ICES CM 1988/H:40),
acoustic survey July-August 1987. Numbers transcribed from Table 1 (RV Dana stratum areas, nm2),
Table 4 (RV Dana biomass, t) and Tables 7/8 (RV Eldjarn sub-area areas nm2 and biomass 1000 t).
Stratum geography read from Fig. 2 / Fig. 4 of the paper (judgement, see lit_notes.md).

Usage: python3 -I lit_degnbol1988_density.py
"""
NM2_KM2 = 1.852 ** 2
dana_area = dict(A=2317, B=2911, C=988, D=1837, E=5228, F=5724, I=3516, K=4350, L=2172, M=4949,
                 O=3072, P=3871, Q=3983, R=4075, S=4199, T=3767, U=3873, V=3980, X=4100, Y=4192, Z=3654)
dana_bio = dict(A=9460, B=15861, C=2634, D=16620, E=188, F=0, I=16739, K=881, L=909, M=783271,
                O=24214, P=10558, Q=11359, R=2918, S=14068, T=64937, U=55722, V=14150, Y=91080, Z=21586)
groups = {
    "Skagerrak (Dana A+B+C+D; A straddles the 3.a line)": "ABCD",
    "Skagerrak without A (B+C+D)": "BCD",
    "Kattegat north (Dana E)": "E",
    "Belts/Sound (Dana F)": "F",
    "Norwegian Trench south slope / NE North Sea (Dana O+I)": "OI",
    "All Dana strata": "".join(dana_area),
    "All Dana strata except M": "".join(k for k in dana_area if k != "M"),
}
print("RV Dana, 31 Jul - 9 Sep 1987 (stratum X biomass not printed in Table 4 -> taken as 0)")
for name, ks in groups.items():
    a = sum(dana_area[k] for k in ks) * NM2_KM2
    b = sum(dana_bio.get(k, 0) for k in ks)
    print(f"  {name:58s} area {a:9.0f} km2  biomass {b:9.0f} t  density {b / a:6.2f} t/km2")
eld = {1: (785, 19), 2: (1500, 29), 3: (1830, 39), 4: (1120, 10), 5: (6500, 239), 6: (9250, 224)}
print("RV Eldjarn, 31 Jul - 11 Aug 1987 (sub-areas Fig. 4; 1000 t)")
for k, (nm2, kt) in eld.items():
    a = nm2 * NM2_KM2
    print(f"  sub-area {k}: area {a:8.0f} km2  biomass {kt * 1000:8.0f} t  density {kt * 1000 / a:6.2f} t/km2")
a = sum(v[0] for v in eld.values()) * NM2_KM2
print(f"  all sub-areas: area {a:8.0f} km2  biomass {560000:8.0f} t  density {560000 / a:6.2f} t/km2")
