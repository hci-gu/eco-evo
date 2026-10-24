"""Biomass-weighted bottom-line values [D] for evidence_A_physiology.md. Usage: python3 -I calc_weighted.py"""
w = {"saithe": 41, "whiting": 37, "mackerel": 8, "spurdog": 5, "horse_mackerel": 3.4, "hake": 2.5, "anglerfish": 1.4}
cmax = {"saithe": 97, "whiting": 104, "mackerel": 150, "spurdog": 50, "horse_mackerel": 120, "hake": 114, "anglerfish": 45}  # kJ/kg/d
smr = {"saithe": 3.5, "whiting": 4.2, "mackerel": 3.4, "spurdog": 2.1, "horse_mackerel": 3.4, "hake": 3.5, "anglerfish": 1.2}  # MJ/t/6h
smr_mack_routine = dict(smr, mackerel=6.1)
q10 = {"saithe": 2.3, "whiting": 2.3, "mackerel": 2.0, "spurdog": 2.5, "horse_mackerel": 2.2, "hake": 2.3, "anglerfish": 2.3}
tot = sum(w.values())
def wm(d): return sum(w[k] * d[k] for k in w) / tot
c = wm(cmax)
print("weights sum", tot)
print("Cmax %.0f kJ/kg/d = %.2f %%BW/d at 4.184 kJ/g = %.1f MJ/t/6h" % (c, 100 * c / 4184, c / 4))
print("SMR %.2f MJ/t/6h (mackerel at routine 6.1: %.2f)" % (wm(smr), wm(smr_mack_routine)))
print("Q10 %.2f" % wm(q10))
# range: low/high
lo = {"saithe": 75, "whiting": 80, "mackerel": 110, "spurdog": 40, "horse_mackerel": 90, "hake": 90, "anglerfish": 30}
hi = {"saithe": 140, "whiting": 150, "mackerel": 300, "spurdog": 70, "horse_mackerel": 170, "hake": 170, "anglerfish": 60}
print("Cmax range %.0f-%.0f kJ/kg/d" % (wm(lo), wm(hi)))
slo = {"saithe": 2.9, "whiting": 3.3, "mackerel": 3.0, "spurdog": 1.7, "horse_mackerel": 2.5, "hake": 2.5, "anglerfish": 0.8}
shi = {"saithe": 5.0, "whiting": 5.0, "mackerel": 6.5, "spurdog": 2.7, "horse_mackerel": 5.0, "hake": 5.0, "anglerfish": 2.0}
print("SMR range %.2f-%.2f MJ/t/6h" % (wm(slo), wm(shi)))
