"""Derived numbers for lit_costs_sinks.md. No downloaded data read; inputs typed from sources."""
import math
MG = 0.0816            # MJ/t/tick per mg O2/kg/h (13.6 kJ/g O2, 6 h)
def vo2(c):            # Chung et al. 2019 cod calibration: C = 0.243 (1 - exp(-0.0088 VO2))
    return -math.log(1 - c / 0.243) / 0.0088
def tcorr(v, t0, t1, q10):
    return v * q10 ** ((t1 - t0) / 10)
model = {"whiting": (5.8, 7.4, 3.87), "saithe/WC": (7.04, 8.4, 3.29)}
otol = {"whiting": (0.162, 0.139, 0.216, 17.2), "saithe/WC": (0.143, 0.106, 0.164, 11.1)}
print("1. Otolith FMR (Alewijnse 2022 dataset, n=5 each) -> model temperature")
for k, (c, lo, hi, t) in otol.items():
    tot, tm, rm = model[k]
    for q in (2.3, 2.7):
        v = [tcorr(vo2(x), t, tm, q) for x in (c, lo, hi)]
        m = [x * MG for x in v]
        print(f"  {k:9s} Q10 {q}: VO2 at {t} C = {vo2(c):.1f} ({vo2(lo):.1f}-{vo2(hi):.1f}); at {tm} C = {v[0]:.1f} ({v[1]:.1f}-{v[2]:.1f}) mg/kg/h"
              f" = {m[0]:.2f} ({m[1]:.2f}-{m[2]:.2f}) MJ/t/tick; /model non-SDA {tot}: {m[0]/tot:.2f} ({m[1]/tot:.2f}-{m[2]/tot:.2f});"
              f" FMR/model RM {m[0]/rm:.2f}")
print("2. SDA on top of the model cost at the SMS saithe ration")
ration = [0.44, 1.27, 0.67, 0.79]   # %BW/d Q1-Q4 (SMS)
E = 5.5                             # MJ/kg diet energy [J, prior note]
for i, r in enumerate(ration):
    intake = r / 100 * 1000 * E / 4  # MJ/t/tick
    print(f"  Q{i+1}: ration {r} %BW/d -> intake {intake:.2f} MJ/t/tick, SDA 0.12 = {0.12*intake:.2f}, model total {7.04+0.12*intake:.2f}")
mean = sum(ration) / 4
intake = mean / 100 * 1000 * E / 4
print(f"  annual mean ration {mean:.3f} %BW/d -> intake {intake:.2f} MJ/t/tick = {intake*1460:.0f} MJ/t/yr; model cost+SDA {7.04+0.12*intake:.2f}")
print("3. Andersen & Riis-Vestergaard 2004 activity multiplier on the model's SMR (saithe 3.29 at 8.4 C)")
for a in (1.44, 2.0, 2.5, 3.27):
    print(f"  x{a}: {3.29*a:.2f} MJ/t/tick, /7.04 = {3.29*a/7.04:.2f}")
print("4. Cruising at Uopt (COT, Steinhausen et al. 2005 [S]) for assumed lab sizes")
g = 9.81
for sp, cot, bl, L in (("saithe", 0.14, 1.4, 0.30), ("saithe", 0.14, 1.4, 0.50), ("whiting", 0.15, 1.0, 0.25)):
    U = bl * L
    W = cot * g * U                  # W/kg gross
    mg = W * 3600 / 13.6             # mg O2/kg/h
    print(f"  {sp} L {L} m: U {U:.2f} m/s, {W:.3f} W/kg = {mg:.0f} mg O2/kg/h = {mg*MG:.1f} MJ/t/tick if cruising all day")
print("5. Saithe spawning round trip box -> northern North Sea shelf edge")
def gc(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    return 6371 * 2 * math.asin(math.sqrt(math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2))
box = (58.05, 10.6)
for name, p in (("Tampen 61.0N 2.0E", (61.0, 2.0)), ("Viking/NT edge 60.0N 3.0E", (60.0, 3.0)), ("NT edge 59.0N 4.0E", (59.0, 4.0))):
    d = gc(box, p)
    print(f"  to {name}: {d:.0f} km one way, {2*d:.0f} km round trip")
for cot in (0.06, 0.10, 0.14):     # gross J/N/m; lower for 60-80 cm adults [J]
    for d in (800, 1100):
        for U in (0.5, 0.84):
            gross = cot * g * d * 1e3 / 1e3          # MJ/t
            days = d * 1e3 / U / 86400
            routine = 28.2 * days                    # model non-SDA cost MJ/t over those days
            print(f"  COT {cot} d {d} km U {U}: gross {gross:.0f} MJ/t over {days:.1f} d, minus model routine {routine:.0f} -> extra {gross-routine:.0f} MJ/t migrant")
print("6. Seasonal temperature at_50m 6.9-11.7 C, Q10 2.3, relative to 8.4 C")
for t in (6.9, 11.7):
    print(f"  {t} C: x{2.3**((t-8.4)/10):.2f}")
print("7. Inline arithmetic used in the notes")
print(f"  saithe FMR/SMR: 6.58 / 3.5 = {6.58/3.5:.2f}")
print(f"  SMS Q1 shortfall: (0.79-0.44)/100*1000 kg/t * 90 d * 5.5 MJ/kg = {(0.79-0.44)/100*1000*90*5.5:.0f} MJ/t = {(0.79-0.44)/100*1000*90*5.5/15909:.3f} of annual ration")
for lo, hi in ((0.25, 0.6),):
    for x in (0, 652, 1083):
        print(f"  migration {x} MJ/t migrant x mature {lo}-{hi} x 0.797 = {x*lo*0.797:.0f}-{x*hi*0.797:.0f} MJ/t group/yr ({x*hi*0.797/15909*100:.1f} % of ration at the top)")
print(f"  cruising multiples vs model 7.04: 12.5 -> {12.5/7.04:.2f}, 20.8 -> {20.8/7.04:.2f}; whiting 7.9 vs 5.8 -> {7.9/5.8:.2f}")
print("8. Ration-route closure, water column: R = A * ration energy - P * body energy (gonads ignored)")
for A in (0.7, 0.8):
    R = A * 15909 - 0.42 * 5080
    print(f"  A {A}: R = {A}*15909 - 0.42*5080 = {R:.0f} MJ/t/yr = {R/1460:.2f} MJ/t/tick (otolith 6.58, model+SDA 8.35)")
