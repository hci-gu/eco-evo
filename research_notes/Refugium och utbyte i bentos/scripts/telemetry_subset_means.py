"""Recompute per-rhythm means of Aguzzi et al. 2023 (Rev Fish Biol Fish 33:1465,
doi:10.1007/s11160-023-09787-2) Table 2, typed in by hand from the extracted text
(data/aguzzi2023_rfbf_telemetry/rfbf2023.txt). Rhythm labels from Table 1.
Columns: door-keeping (<=0.5 m/h, %), in burrow (0 detections, %), full emergence (>0.5 m/h, %).
Also converts the shares to hours per day and computes the exposure-weighted
time scale. Run: python3 -I telemetry_subset_means.py
"""

TABLE2 = {  # id: (door_le05, in_burrow, full_gt05, rhythm, displacement per day from Table 1)
    14456: (2.05, 23.34, 74.61, "arrhythmic", 229.6),
    14457: (1.93, 1.58, 96.49, "inertial", 290.1),
    14458: (10.82, 42.63, 46.55, "day-night", 137.6),
    14459: (11.70, 47.11, 41.19, "day-night", 247.2),
    14461: (0.24, 13.60, 86.16, "other", 240.4),
    14462: (7.54, 62.61, 29.85, "day-night", 124.1),
    14463: (16.80, 30.10, 53.10, "inertial", 236.9),
    14464: (17.61, 39.68, 42.71, "arrhythmic", 128.8),
    14465: (23.36, 50.65, 25.99, "day-night", 333.3),
    14467: (24.97, 33.00, 42.03, "arrhythmic", 155.6),
    14468: (25.45, 25.00, 49.55, "day-night", 424.7),
    14470: (9.25, 84.67, 6.08, "arrhythmic", 340.1),
    14471: (15.26, 54.31, 30.43, "day-night", 269.4),
    14472: (14.24, 72.97, 12.79, "other", 110.4),
    14473: (10.17, 15.99, 73.84, "day-night", 102.7),
    14474: (13.10, 80.45, 6.45, "day-night", 90.7),
    14475: (27.02, 25.63, 47.35, "inertial", 143.1),
    14476: (2.29, 7.19, 90.52, "arrhythmic", 193.4),
    14479: (11.82, 21.41, 66.77, "day-night", 205.6),
    14482: (1.82, 11.18, 87.00, "arrhythmic", 243.0),
    14484: (18.26, 69.31, 12.43, "day-night", 94.0),
    14485: (37.44, 37.30, 25.26, "day-night", 185.9),
    15829: (9.29, 81.22, 9.49, "day-night", 285.6),
    15830: (10.05, 21.25, 68.50, "arrhythmic", 178.1),
    15831: (0.41, 5.54, 94.05, "arrhythmic", 410.5),
}


def summarise(label, rows):
    n = len(rows)
    door = sorted(r[0] for r in rows)
    inb = sorted(r[1] for r in rows)
    full = sorted(r[2] for r in rows)
    mean = lambda v: sum(v) / len(v)
    med = lambda v: (v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2]))
    print(f"{label}: n={n}")
    for name, v in (("door-keeping", door), ("in burrow", inb), ("full emergence", full)):
        print(f"  {name:15s} mean {mean(v):6.2f} %  median {med(v):6.2f} %  range {v[0]:.2f}-{v[-1]:.2f} %"
              f"  -> mean {mean(v) * 24 / 100:5.2f} h/day")
    sums = [r[0] + r[1] + r[2] for r in rows]
    print(f"  row sums (door+in+full) range {min(sums):.2f}-{max(sums):.2f} %")


summarise("All 25 animals", list(TABLE2.values()))
summarise("Day-night rhythmic (burrow-holding subset)", [r for r in TABLE2.values() if r[3] == "day-night"])
summarise("Arrhythmic + inertial + other", [r for r in TABLE2.values() if r[3] != "day-night"])

# Exposure share used in the model sense: fully emerged = exposed; door-keeping = partly exposed.
dn = [r for r in TABLE2.values() if r[3] == "day-night"]
full = sum(r[2] for r in dn) / len(dn) / 100
door = sum(r[0] for r in dn) / len(dn) / 100
print(f"\nDay-night subset: fully exposed share {full:.3f}; with door-keepers {full + door:.3f}")
# If emergence is one bout per day (single midday peak at this depth), the mean bout length is
# full*24 h and the mean hidden spell (24 - bout) h; a two-state switching model with one
# cycle per day has a cycle time of 24 h regardless; the bout length sets the exposed share.
print(f"Implied single daily bout length {full * 24:.1f} h (door-keeping adds {door * 24:.1f} h)")
