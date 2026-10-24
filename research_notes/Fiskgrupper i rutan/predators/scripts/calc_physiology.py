"""Arithmetic for evidence_A_physiology.md ([D] values). No downloaded code is run.
Usage: python3 -I calc_physiology.py <amp_dir>
"""
import math, sys, os, re, html

OXY = 13.6          # J per mg O2 (task convention; 13.56 in Elliott & Davison)
def mgO2_to_kJkgd(x):   # mg O2/kg/h -> kJ/kg/d
    return x * OXY * 24 / 1000
def kJkgd_to_MJt6h(x):  # 1 MJ/t per 6 h tick = 4 kJ/kg/d
    return x / 4

def kitchell(T, CQ, CTO, CTM):
    V = (CTM - T) / (CTM - CTO)
    Z = math.log(CQ) * (CTM - CTO)
    Y = math.log(CQ) * (CTM - CTO + 2)
    X = (Z**2 * (1 + (1 + 40 / Y) ** 0.5) ** 2) / 400
    return V**X * math.exp(X * (1 - V))

print("== A. FB4 Cmax (g prey / g fish / d) ==")
def cod_cmax(W, T): return 0.099 * W**-0.195 * kitchell(T, 2.41, 13.7, 21)
def pollock_ceq1(W, T): return 0.02878 * W**-0.6292 * math.exp(0.3763 * T)
def pollock_mason(W, T): return 0.3 * W**-0.5875 * kitchell(T, 3.5, 8, 15)
for W in (200, 500, 1000, 2000):
    for T in (7.0, 8.4, 10.0, 13.0):
        print("W %5d T %4.1f  cod %.4f  pollockB&L %.4f  pollockMason %.4f" % (W, T, cod_cmax(W, T), pollock_ceq1(W, T), pollock_mason(W, T)))
print("cod f(T) 7, 8.4, 13.7:", [round(kitchell(t, 2.41, 13.7, 21), 3) for t in (7, 8.4, 13.7)])
print("cod Cmax Q10 7->8.4 eff:", round((kitchell(8.4,2.41,13.7,21)/kitchell(7,2.41,13.7,21))**(10/1.4),2))

print("\n== FB4 cod respiration RA 0.008 W^-0.172 f(T; RQ 1.88, RTO 21, RTM 24) x ACT 1.25 ==")
def cod_resp(W, T, act=1.0): return 0.008 * W**-0.172 * kitchell(T, 1.88, 21, 24) * act
for W in (200, 1000):
    for T in (8.4,):
        r = cod_resp(W, T)
        mg = r * 1000 / 24 * 1000  # g O2/g/d -> mg O2/kg/h
        print("W %d T %.1f resting %.5f gO2/g/d = %.1f mgO2/kg/h = %.1f kJ/kg/d = %.2f MJ/t/6h; x1.25 = %.2f" % (
            W, T, r, mg, mgO2_to_kJkgd(mg), kJkgd_to_MJt6h(mgO2_to_kJkgd(mg)), 1.25 * kJkgd_to_MJt6h(mgO2_to_kJkgd(mg))))

print("\n== B. Andersen gadoid GER (Velasco & Olaso Eq. 3): dS/dt = 0.00132 L^1.44 e^0.078T E^-0.875 S^0.5 g/h ==")
def ger(L, T, E, S): return 0.00132 * L**1.44 * math.exp(0.078 * T) * E**-0.875 * S**0.5
for name, L, W, E in (("whiting", 28, 200, 5.0), ("whiting", 35, 400, 5.0), ("saithe", 50, 1066, 5.0), ("hake", 40, 415, 5.0)):
    for T in (8.4,):
        for sfrac in (0.03, 0.05, 0.08):
            c = 24 * ger(L, T, E, sfrac * W)
            print("%-8s L %d W %d T %.1f E %.1f S=%.0f%%BW -> %.2f %%BW/d = %.0f kJ/kg/d" % (name, L, W, T, E, sfrac * 100, 100 * c / W, 1000 * c / W * E))
print("GER Q10 (e^0.78):", round(math.exp(0.78), 2))

print("\n== C. AmP DEB implied max ingestion and somatic maintenance per kg ==")
d = sys.argv[1]
def grab(fn, keys):
    t = open(fn, encoding="utf-8", errors="replace").read()
    t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))
    out = {}
    for k in keys:
        m = re.search(r" %s ([-0-9.e+]+) " % re.escape(k), t)
        if m: out[k] = float(m.group(1))
    return out
cases = {"Pollachius_virens": (1066, 8.4), "Merlangius_merlangus": (200, 8.4), "Scomber_scombrus": (400, 13.0),
         "Squalus_acanthias": (2000, 8.4), "Trachurus_trachurus": (250, 13.0), "Merluccius_merluccius": (415, 8.4),
         "Lophius_piscatorius": (4000, 8.4), "Gadus_morhua": (1000, 8.4)}
for sp, (W, T) in cases.items():
    p = grab(os.path.join(d, sp + "_par.html"), ["T_A", "p_Am", "kap_X", "p_M"])
    s = grab(os.path.join(d, sp + "_stat.html"), ["s_M", "del_V", "t_starve", "c_T", "E_m"])
    sM = s.get("s_M", 1.0)
    TC = math.exp(p["T_A"] / 293.15 - p["T_A"] / (273.15 + T))
    V = s["del_V"] * W
    JX = sM * p["p_Am"] / p["kap_X"] * V ** (2 / 3) * TC   # J/d food
    pM = p["p_M"] * V * TC
    q10 = math.exp(10 * p["T_A"] / ((273.15 + 3) * (273.15 + 13)))
    print("%-22s W %5d T %4.1f TC %.3f  maxFood %.0f kJ/kg/d  somMaint %.1f kJ/kg/d (%.2f MJ/t/6h)  Q10(3-13C) %.2f  t_starve(Ttyp) %.0f d  del_V %.3f" % (
        sp, W, T, TC, JX / W, pM / W, kJkgd_to_MJt6h(pM / W), q10, s.get("t_starve", float("nan")), s["del_V"]))

print("\n== D. Mackerel respiration ==")
def bach(W, T, SW=1.0): return 0.00264 * W**-0.217 * math.exp(0.06818 * T) * math.exp(0.0234 * SW)
def chub(W, T, U): return 0.0103 * W**-0.490 * math.exp(0.0457 * T) * math.exp(0.0235 * U)
for W in (300, 400, 500):
    for T in (12.0, 13.0, 14.0):
        r = bach(W, T); mg = r * 1e6 / 24
        r0 = chub(W, T, 0); mg0 = r0 * 1e6 / 24
        r1 = chub(W, T, 35); mg1 = r1 * 1e6 / 24
        print("W %d T %.0f  Bachiller(SW=1BL/s) %.1f mgO2/kg/h = %.1f kJ/kg/d = %.2f MJ/t/6h | chub U=0 %.1f (%.2f MJ/t/6h), U=35cm/s %.1f (%.2f)" % (
            W, T, mg, mgO2_to_kJkgd(mg), kJkgd_to_MJt6h(mgO2_to_kJkgd(mg)), mg0, kJkgd_to_MJt6h(mgO2_to_kJkgd(mg0)), mg1, kJkgd_to_MJt6h(mgO2_to_kJkgd(mg1))))
print("Bachiller Q10 e^0.6818 = %.2f; chub Q10 e^0.457 = %.2f" % (math.exp(0.6818), math.exp(0.457)))

print("\n== E. Conversions of measured MO2 ==")
for lab, x in (("dogfish resting 10C (B&B)", 32.4), ("dogfish routine 10C", 49.2), ("dogfish active 10C", 88.4),
               ("S. suckleyi SMR 9C", 21.4), ("S. suckleyi SMR 13C mean", 50.0), ("saithe/whiting SMR (Steinhausen)", 59.2),
               ("cod SMR 5C", 35.5), ("cod SMR 10C", 57.0), ("cod SMR 8.4C interp", 35.5 * (57.0 / 35.5) ** (3.4 / 5))):
    k = mgO2_to_kJkgd(x)
    print("%-34s %6.1f mgO2/kg/h = %5.1f kJ/kg/d = %.2f MJ/t/6h" % (lab, x, k, kJkgd_to_MJt6h(k)))
for lab, q10, T in (("dogfish 10->8.4 Q10 2.0", 2.0, 8.4), ("dogfish 10->8.4 Q10 2.5", 2.5, 8.4)):
    x = 32.4 * q10 ** ((T - 10) / 10)
    print(lab, round(x, 1), "mg/kg/h", round(kJkgd_to_MJt6h(mgO2_to_kJkgd(x)), 2), "MJ/t/6h")

print("\n== F. Ecopath Q/B (Mackinson & Daskalov 2007) as %BW/d ==")
for lab, qb in (("spurdog", 2), ("whiting adult", 5.46), ("saithe adult", 3.6), ("hake", 2.2), ("monkfish", 1.9), ("mackerel", 1.73), ("horse mackerel", 3.51)):
    print("%-15s Q/B %.2f /yr = %.2f %%BW/d" % (lab, qb, 100 * qb / 365))
