"""Share of the 24 h in twilight (sun between 0 and -6 deg, civil; 0 and -12 deg, nautical), in daylight and at night,
for Rindorf's whiting sites (early September, 54-58 N) and the box (58.1 N) by month. A check of the size of a
twilight capture window for fish prey against Rindorf & Gislason's clupeid plateau, not a fit [D].
Standard solar geometry (declination from day of year; refraction ignored). Usage: python3 -I twilight_share.py
"""
import math


def elev(lat, doy, hour):
    d = math.radians(-23.44) * math.cos(2 * math.pi * (doy + 10) / 365.0)
    h = math.radians(15.0 * (hour - 12.0))
    la = math.radians(lat)
    return math.degrees(math.asin(math.sin(la) * math.sin(d) + math.cos(la) * math.cos(d) * math.cos(h)))


def shares(lat, doy, n=2880):
    e = [elev(lat, doy, 24.0 * i / n) for i in range(n)]
    day = sum(x > 0 for x in e) / n
    civ = sum(-6 < x <= 0 for x in e) / n
    nau = sum(-12 < x <= 0 for x in e) / n
    return day, civ, nau


print("site            lat  doy   day   civil-twilight  nautical-twilight  (shares of 24 h)")
for name, lat, doy in [("Rindorf loc 1-2", 54.5, 247), ("Rindorf loc 3-4", 56.5, 247), ("Rindorf loc 5", 58.0, 247)]:
    d, c, n = shares(lat, doy)
    print(f"{name:15s} {lat:4.1f} {doy:4d}  {d:5.3f}  {c:5.3f} ({24*c:4.2f} h)  {n:5.3f} ({24*n:4.2f} h)")
print("box 58.1 N by month (15th):")
for m, doy in zip(range(1, 13), [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]):
    d, c, n = shares(58.1, doy)
    print(f"  month {m:2d}: day {d:5.3f}  civil {c:5.3f} ({24*c:4.2f} h)  nautical {n:5.3f} ({24*n:4.2f} h)")
