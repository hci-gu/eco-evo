"""Q2: day length (sunrise-sunset, sun centre at -0.833 deg) and civil twilight (-6 deg)
for the sampling dates/latitudes of the diel stomach studies, so clock windows can be
turned into shares of 24 h. Pure astronomy (NOAA-style declination formula); no model output.

Usage: python3 -I q2_daylength.py
"""
import math


def declination(doy):
    g = 2 * math.pi / 365.0 * (doy - 1)
    return (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))


def hours_above(lat_deg, doy, elev_deg):
    lat = math.radians(lat_deg)
    d = declination(doy)
    c = (math.sin(math.radians(elev_deg)) - math.sin(lat) * math.sin(d)) / (math.cos(lat) * math.cos(d))
    if c <= -1:
        return 24.0
    if c >= 1:
        return 0.0
    return 2 * math.degrees(math.acos(c)) / 15.0


CASES = [
    ("Mergardt & Temming 1997, 30 May-1 Jun, 57.53 N", 57.53, 151),
    ("Modica 2011, 27 Jun, Catalan Sea ~41 N", 41.0, 178),
    ("Box (manifest latitude_deg 58.15 N), 21 Mar", 58.15, 80),
    ("Box 58.15 N, 21 Jun", 58.15, 172),
    ("Box 58.15 N, 5 Sep", 58.15, 248),
    ("Box 58.15 N, 21 Dec", 58.15, 355),
]

# Rindorf & Gislason 2005, Table 1 (hh.mm, as printed): sunrise, sunset, civil dawn, civil dusk.
RG_TABLE1 = {
    1: (4.55, 18.17, 4.18, 18.54),
    2: (4.44, 18.27, 4.06, 19.05),
    3: (4.42, 18.32, 4.03, 19.12),
    4: (5.01, 18.33, 4.23, 19.11),
    5: (5.10, 18.37, 4.32, 19.17),
}


def hhmm(x):
    h = int(x)
    return h + round((x - h) * 100) / 60.0


if __name__ == "__main__":
    print("Rindorf & Gislason 2005 Table 1 (whiting sites, early Sept): windows from the printed clock times")
    print(f"{'loc':>3s} {'day h':>6s} {'day/24':>6s} {'night h':>7s} {'night/24':>8s} {'twilight h':>10s}")
    for loc, (sr, ss, dawn, dusk) in RG_TABLE1.items():
        day = hhmm(ss) - hhmm(sr)
        night = 24 - (hhmm(dusk) - hhmm(dawn))
        tw = (hhmm(sr) - hhmm(dawn)) + (hhmm(dusk) - hhmm(ss))
        print(f"{loc:3d} {day:6.2f} {day/24:6.3f} {night:7.2f} {night/24:8.3f} {tw:10.2f}")
    print("Fish window bounds with 4-h sampling: twilight only vs two 4-h bins = 8 h ->",
          f"{1.23/24:.3f} - {8/24:.3f} of 24 h")
    print()
    print(f"{'case':70s} {'day h':>6s} {'civil h':>7s} {'day/24':>6s} {'night/24':>8s} {'twilight h (both)':>17s}")
    for name, lat, doy in CASES:
        day = hours_above(lat, doy, -0.833)
        civ = hours_above(lat, doy, -6.0)
        print(f"{name:70s} {day:6.2f} {civ:7.2f} {day/24:6.3f} {(24-civ)/24:8.3f} {civ-day:17.2f}")
