"""Day length at 58.15 N (the box) from the repo's daylight calendar, read-only import.

Usage: python3 day_length.py  (run from the repo root; prints the 1st and 15th of each month with
the day length in hours and the sign of its change: + rising, - falling).
"""
import sys

sys.path.insert(0, ".")
from lib.world.daylight import day_length_hours  # noqa: E402

LAT = 58.15
d = day_length_hours(LAT)
cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
print(f"day length at {LAT} N, hours; sign = d(day length)/dt")
for m, c in enumerate(cum):
    for day in (1, 15):
        i = c + day - 1
        sign = "+" if d[(i + 1) % 365] > d[i - 1] else "-"
        print(f"{names[m]} {day:2d}  {d[i]:5.2f} h  {sign}")
print(f"min {d.min():.2f} h (day {d.argmin() + 1}), max {d.max():.2f} h (day {d.argmax() + 1})")
