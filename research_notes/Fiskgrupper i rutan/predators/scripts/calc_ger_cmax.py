"""GER-based sustained Cmax [D]: WGSAM 2024 Table 3-7 cylinder model, fish prey (armament M=1),
R = rho e^(delta T) L^lambda E^-0.85 S^0.5 (g/h). S_max calibrated so that cod reproduces the
FB4 Hansson cod Cmax (500 g, 36 cm, 7 C, 4.184 kJ/g = 2.13 %BW/d). Usage: python3 -I calc_ger_cmax.py
"""
import math
par = {"cod": (0.00224, 0.083, 1.30), "saithe": (0.00171, 0.081, 1.35), "whiting": (0.00171, 0.081, 1.35),
       "mackerel": (0.00174, 0.080, 1.30)}
def R(sp, L, T, E, S):
    rho, de, la = par[sp]
    return rho * math.exp(de * T) * L**la * E**-0.85 * S**0.5
def kit(T, CQ=2.41, CTO=13.7, CTM=21):
    V=(CTM-T)/(CTM-CTO); Z=math.log(CQ)*(CTM-CTO); Y=math.log(CQ)*(CTM-CTO+2)
    X=(Z**2*(1+(1+40/Y)**0.5)**2)/400; return V**X*math.exp(X*(1-V))
target = 0.099 * 500**-0.195 * kit(7.0) * 500   # g/d
k = 24 * R("cod", 36, 7.0, 4.184, 1.0)          # g/d per sqrt(g)
S = (target / k) ** 2
print("FB4 cod 500 g 7 C Cmax %.2f g/d (%.2f %%BW/d); calibrated S_max = %.1f g = %.2f %%BW" % (target, target / 5, S, S / 5))
sfrac = S / 500
cases = [("whiting", 28, 190, 8.4), ("whiting", 32, 280, 8.4), ("saithe", 40, 578, 8.4), ("saithe", 50, 1066, 8.4),
         ("saithe", 65, 2180, 8.4), ("whiting", 40, 415, 8.4), ("mackerel", 35, 400, 13.0), ("mackerel", 35, 400, 12.0),
         ("mackerel", 35, 400, 8.4), ("cod", 36, 500, 8.4)]
print("label: hake uses whiting/saithe parameters at 40 cm/415 g")
for sp, L, W, T in cases:
    for E in (4.184, 5.5):
        for f in (sfrac, 0.05):
            c = 24 * R(sp, L, T, E, f * W)
            print("%-8s L %2d W %4d T %4.1f E %.2f Smax %.1f%%BW -> Cmax %.2f %%BW/d = %4.0f kJ/kg/d" % (sp, L, W, T, E, 100 * f, 100 * c / W, 1000 * c / W * E))
print("GER Q10 gadoid e^0.81 = %.2f, mackerel e^0.80 = %.2f, cod e^0.83 = %.2f" % (math.exp(0.81), math.exp(0.80), math.exp(0.83)))
