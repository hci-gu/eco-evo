"""Predation mortality on mackerel in the North Sea Ecopath model (1991 base),
Mackinson & Daskalov 2007, Cefas Sci. Ser. Tech. Rep. 142, Table 3.3 (B, Q/B)
and the diet matrix (mackerel row, predator columns aligned by character
position in the pdftotext output; columns checked by character offset). M2 = sum_j B_j (Q/B)_j DC_j / B_mackerel."""
# predator: (B t/km2, Q/B /yr, DC mackerel, DC horse mackerel)
preds = {
    "baleen whales":            (0.067, 9.9,    0.094, 0.0),
    "toothed whales":           (0.017, 17.63,  0.015, 0.010),
    "seals":                    (0.008, 26.842, 0.0,   0.012),
    "seabirds":                 (0.003, 216.56, 0.015, 0.007),
    "juvenile sharks":          (0.001, 2.5,    0.014, 0.0),
    "spurdog":                  (0.0167, 2.0,   0.078, 0.018),
    "large piscivorous sharks": (0.002, 1.6,    0.153, 0.0),
    "small sharks":             (0.002, 2.96,   0.005, 0.0),
    "cod (adult)":              (0.161, 3.5,    0.012, 0.003),
    "saithe (adult)":           (0.22, 3.6,     0.004, 0.0),
    "hake":                     (0.014, 2.2,    0.164, 0.180),
    "monkfish":                 (0.042, 1.9,    0.053, 0.0),
    "gurnards":                 (0.077, 3.2,    0.001, 0.001),
    "juvenile cod":             (0.079, 5.96,   0.0,   0.001),
    "juvenile whiting":         (0.222, 6.58,   0.0,   0.002),
    "whiting (adult)":          (0.352, 5.46,   0.0,   0.010),
    "mackerel":                 (1.72, 1.73,    0.0,   0.006),
    "squid & cuttlefish":       (0.08, 20.0,    0.011, 0.0),
}
B_mac, B_hom = 1.72, 0.579
tot_m = tot_h = 0.0
print(f"{'predator':26s} {'flow mac t/km2/yr':>18s} {'M2 mac':>8s}")
for k, (b, qb, dm, dh) in preds.items():
    fm, fh = b * qb * dm, b * qb * dh
    tot_m += fm; tot_h += fh
    print(f"{k:26s} {fm:18.4f} {fm / B_mac:8.4f}")
print(f"total mackerel flow {tot_m:.4f} t/km2/yr, M2 = {tot_m / B_mac:.4f} /yr")
print(f"total horse mackerel flow {tot_h:.4f}, M2 = {tot_h / B_hom:.4f} /yr ")
w = 0.94 * tot_m / B_mac + 0.06 * tot_h / B_hom
print(f"group (0.94/0.06) M2 = {w:.4f} /yr")
fish = ["juvenile sharks", "spurdog", "large piscivorous sharks", "small sharks",
        "cod (adult)", "saithe (adult)", "hake", "monkfish", "gurnards"]
fm = sum(preds[k][0] * preds[k][1] * preds[k][2] for k in fish)
print(f"fish-predator subtotal on mackerel: flow {fm:.4f} t/km2/yr, M2 = {fm / B_mac:.4f} /yr")
mm = sum(preds[k][0] * preds[k][1] * preds[k][2] for k in ("baleen whales", "toothed whales", "seals"))
print(f"marine mammals on mackerel: M2 = {mm / B_mac:.4f} /yr; seabirds {preds['seabirds'][0]*preds['seabirds'][1]*preds['seabirds'][2]/B_mac:.4f}")
