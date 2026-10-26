"""Day/night catch ratios from Smidt 1976 (ICNAF Res. Doc. 76/XII/149, Table 2:
commercial P. borealis catches, West Greenland Div. 1B, Oct-Dec 1975, kg/h in
3-h intervals, local time as printed). Values typed in by hand from the PDF
text extraction (Table 2 is cleanly legible; Table 1 is not and is not used).
J: 'night' = 21-03 h (two intervals 21-24, 0-3), 'day' = 09-15 h."""
t2 = {
    "October":  [76, 66, 275, 466, 500, 675, 420, 300],
    "November": [244, 496, 370, 632, 1050, 761, 537, 460],
    "December": [187, 445, 383, 508, 562, 600, 473, 427],
}
for m, v in t2.items():
    night = (v[7] + v[0]) / 2
    day = (v[3] + v[4]) / 2
    mean = sum(v) / len(v)
    print(f"{m:9s} max/min {max(v)/min(v):5.2f}  day/night {day/night:5.2f}  "
          f"24h mean/max {mean/max(v):.2f}  night/max {night/max(v):.2f}")
