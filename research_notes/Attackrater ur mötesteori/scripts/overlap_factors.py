"""Encounter-weighted vertical overlap O of herring-dominated pelagic fish with the 0-25 m
zooplankton pool: O = sum_bins p_fish(bin) * r_prey(bin), r = prey concentration in the bin
relative to the 0-25 m mean (= pool B / 25 m). H_eff = 25 m / O is the thickness that turns the
pool's depth-integrated biomass into the concentration the average fish meets.

Inputs are the stdout numbers of heras_herring_depth.py (box, July 2010-20, day > 6 deg and
twilight -12..0) and skagerrak_copepod_strata.py (herring prey set, b = 3, five-strata stations;
mean, min, max), copied here. r below 100 m = 0.5 x r(60-100) (judgement).
Usage: python3 -I overlap_factors.py
"""
fish = {"day (July, box)": [0.03, 0.39, 0.36, 0.21, 0.01],
        "twilight (July, box)": [0.03, 0.53, 0.41, 0.01, 0.01],
        "day (July, region)": [0.04, 0.46, 0.27, 0.22, 0.02],
        "twilight (July, region)": [0.08, 0.64, 0.28, 0.01, 0.00]}
prey = {"spring mean": [1.483, 0.678, 0.203, 0.074], "spring min": [0.604, 0.241, 0.029, 0.010],
        "spring max": [2.138, 1.264, 0.423, 0.149], "autumn mean": [1.959, 0.361, 0.117, 0.088],
        "autumn min": [1.696, 0.278, 0.031, 0.016], "autumn max": [2.083, 0.536, 0.190, 0.288]}
print("O and H_eff = 25/O (m); fish bins 0-10 (from integration start ~7-10 m), 10-25, 25-50, 50-100, >100 m")
for fk, p in fish.items():
    cells = []
    for pk, r in prey.items():
        rr = r + [0.5 * r[3]]
        o = sum(a * b for a, b in zip(p, rr))
        cells.append(f"{pk}: O {o:.2f} H {25 / o:4.0f}")
    print(f"{fk:24s} " + " | ".join(cells))
print("\nIf the unsampled top 0-7 m held a share s of the fish at r(0-10): O rises by about s x (r0_10 - O)."
      " s = 0.1 -> +0.10 to +0.17.")
