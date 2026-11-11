"""The planktivores' energy budget, term by term, against member field metabolism and field consumption.

Library-only arithmetic (no environment, no policy, no rollout). Values are the library on 2026-11-11
(fgconfig/fg_library.yaml `planktivorous_fish`, its pairs, and mareld2.yaml temperatures), hard-coded
below so the table is reproducible. Units: MJ per tonne; tick = 6 h; 1460 ticks per year;
1 MJ/t/tick = 4 kJ/kg/d; 1 mg O2/g/h = 13.56 J/mg x 24 = 325.4 kJ/kg/d.

Sections
 1. library cost terms by month and the implied annual metabolism at several eating shares
 2. the policy-free closure (Krillens flykt och energibudget, check 4a) re-done with M1 per year
    (closure.py read natural_mortality as a per-year rate; it is per tick)
 3. member metabolism: herring (Rose/Rudstam resting at the box sizes, activity at 1 BL/s, Rudstam's own
    swimming speed, Johnstone et al. 1993 routine, Bernreuther 2007 fasting), sprat (Meskendahl routine and
    standard), Norway pout (cod SMR proxy, Hansson 1996 term, blue-whiting otolith proxy), blue whiting
    (otolith, tables/otolith_bw_stdout.txt)
 4. biomass-weighted field metabolism (low / central / high) next to the library
 5. field consumption: Q/B sources, energy at the field prey energy density, respiration R = A C - P
 6. Q/B needed (field closure) and e* per lever
 7. feeding-time share by member (daylight at 58.2 N)
Usage: python3 -I budget_terms.py
"""
import math

TPY = 1460.0
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
T_0_50 = [6.3, 5.4, 5.2, 6.1, 7.8, 9.9, 12.1, 13.4, 13.8, 12.8, 10.5, 8.4]       # mareld2.yaml mean_0_50m
T_50 = [7.7, 7.2, 6.6, 6.6, 6.8, 7.2, 8.0, 8.5, 10.2, 11.7, 10.6, 9.3]          # at_50m
T_BOT = [7.80, 6.99, 6.59, 6.58, 6.69, 6.96, 7.22, 7.41, 7.68, 8.11, 8.27, 8.07]  # bottom (annual 7.40)
O2 = 325.44  # kJ/kg/d per mg O2/g/h

# ---- library (planktivorous_fish) ----
RM, Q10, TREF = 8.0, 1.5, 14.0
FC, RC, MC = 1.65, 1.0, 1.75
SDA, EC, ME, U = 0.11, 4000.0, 4990.0, 0.079
P_SOM = 0.66
M1_TICK = 0.000186986
M1 = M1_TICK * TPY
M2_ON = 0.1023 + 0.174 + 0.0186 + 0.0545 + 0.00946 + 0.00501  # checked predators, seabirds muted
START_FILL = [0.50, 0.36, 0.26, 0.20, 0.32, 0.81, 0.94, 0.96, 0.88, 0.88, 0.76, 0.68]
S_FIELD = sum(START_FILL) / 12
A_PAIR = 0.8
E_ZOO, E_KRILL = 4500.0, 5000.0
BOOKED = {"zooplankton": 0.6021704631721533, "krill": 0.39782953682784666}  # closure.json (M2 x B at start)
QB_FIELD = 5.0
W_MEMBER = {"herring": 0.554, "pout": 0.350, "blue whiting": 0.075, "sprat": 0.021}


def q_net(e_zoo, e_krill, a=A_PAIR, sda=SDA):
    """Assimilated energy net of SDA per tonne eaten over the booked diet (the engine's energy_gain)."""
    return BOOKED["zooplankton"] * (a * e_zoo - sda * e_zoo) + BOOKED["krill"] * (a * e_krill - sda * e_krill)


Q_MODEL = q_net(E_ZOO, E_KRILL)


def mmean(x):
    return sum(v * d for v, d in zip(x, DAYS)) / 365.0


print("=" * 100)
print("1. Library cost terms")
mult = [Q10 ** ((t - TREF) / 10) for t in T_0_50]
rm_m = [RM * m for m in mult]
print("  month   T0-50  Q10 mult  RM MJ/t/tick  kJ/kg/d   eat tick (x1.65) kJ/kg/d")
for i in range(12):
    print(f"  {MON[i]}   {T_0_50[i]:5.1f}   {mult[i]:.3f}     {rm_m[i]:6.2f}     {4 * rm_m[i]:6.1f}     {4 * rm_m[i] * FC:6.1f}")
RM_YR = mmean(rm_m) * TPY
print(f"  annual-mean multiplier {mmean(mult):.3f}; RM {mmean(rm_m):.2f} MJ/t/tick = {4 * mmean(rm_m):.1f} kJ/kg/d = {RM_YR:.0f} MJ/t/yr")
print(f"  annual mean T0-50 = {mmean(T_0_50):.2f} C")


def cost(e_eat, e_move=0.0):
    e_rest = 1.0 - e_eat - e_move
    return RM_YR * (e_rest * RC + e_eat * FC + e_move * MC)


for lab, ee, em in (("trained split rest/eat/move 0.08/0.81/0.11", 0.81, 0.11), ("all eat (e = 1)", 1.0, 0.0),
                    ("field feeding-time share e = 0.57", 0.57, 0.0), ("e = 0.5", 0.5, 0.0), ("all rest", 0.0, 0.0)):
    c = cost(ee, em)
    print(f"  costs at {lab:44s}: {c:7.0f} MJ/t/yr = {c / 365:5.1f} kJ/kg/d = {c / RM_YR:.2f} x RM")
print(f"  q (assimilated net of SDA per tonne of booked diet; zoo 4500, krill 5000, A 0.8, SDA 0.11) = {Q_MODEL:.0f} MJ/t")
print(f"  field I = Q/B {QB_FIELD} x q = {QB_FIELD * Q_MODEL:.0f} MJ/t/yr")

print()
print("=" * 100)
print("2. Policy-free closure, corrected (closure.py used natural_mortality 1.87e-4 as /yr; it is /tick)")
print(f"  M1 = {M1:.3f} /yr; booked M2 on the group = {M2_ON:.3f} /yr; s_field (mean start_fill) = {S_FIELD:.3f}")
for tag, s in (("field fill", S_FIELD), ("fill 1", 1.0)):
    G = P_SOM * EC
    ref_old = M2_ON * s * ME
    ref_new = (M1 + M2_ON) * s * ME
    ref_p = P_SOM * s * ME
    I = QB_FIELD * Q_MODEL
    for lab, ref in (("closure.py (M2 only)", ref_old), ("M1 + M2 per year", ref_new), ("P x s ME (steady state)", ref_p)):
        es = (I - G - ref - RM_YR * RC) / (RM_YR * (FC - RC))
        print(f"  {tag:10s} refill {lab:24s} {ref:6.0f}; G {G:.0f}; e* = {es:5.2f}")
print(f"  largest FMR the field Q/B 5.0 pays: field fill {QB_FIELD * Q_MODEL - P_SOM * (EC + S_FIELD * ME):.0f}, "
      f"fill 1 {QB_FIELD * Q_MODEL - P_SOM * (EC + ME):.0f} MJ/t/yr")
print("  (P x s ME: in steady state with imports arriving at the resident fill, the reserve the diet must"
      " supply is the reserve on the year's new tonnes; (M1 + M2) x s ME is the same within 4 %)")


def need_qb(c, s, q=Q_MODEL, p=P_SOM, extra=0.0):
    return (c + p * (EC + s * ME) + extra) / q


print("  Q/B needed (model tonnes at q) = (costs + P (ec + s ME)) / q:")
for lab, c in (("trained split", cost(0.81, 0.11)), ("e = 1", cost(1.0)), ("e = 0.57", cost(0.57)), ("e = 0.5", cost(0.5)),
               ("all rest", cost(0.0))):
    print(f"    {lab:14s} costs {c:6.0f}: field fill {need_qb(c, S_FIELD):.2f}, fill 1 {need_qb(c, 1.0):.2f}")

print()
print("=" * 100)
print("3. Member metabolism (annual means over the box months)")

# Herring size mix in the box (Sillens vinterbudget/winter_budget.md section 4, IBTS site rectangles):
Q1MIX = [(14.0, 16.6, 0.42), (20.9, 56.6, 0.47), (25.2, 110.0, 0.11)]
Q3MIX = [(11.9, 10.9, 0.28), (18.6, 45.3, 0.36), (23.8, 107.7, 0.25), (27.0, 161.9, 0.12)]


def mix(i):
    m = Q1MIX if i < 6 else Q3MIX  # Jan-Jun winter/spring mix, Jul-Dec with the 0-group
    s = sum(x[2] for x in m)
    return [(L, W, w / s) for L, W, w in m]


def rose_rest(W, T):
    """Rose et al. 2008 / Rudstam 1988 herring resting term, kJ/kg/d (0.0033 W^-0.227 e^0.0548T g O2/g/d)."""
    return 0.0033 * W ** -0.227 * math.exp(0.0548 * T) * 13560.0


def rud_vel(W, T):
    return 15.0 * W ** 0.13 if T <= 9.0 else 3.9 * W ** 0.13 * math.exp(0.149 * T)


print(f"  check: Rose resting at 129 g, 14 C = {rose_rest(129, 14):.1f} kJ/kg/d = {rose_rest(129, 14) / 4:.2f} MJ/t/tick (library RM 8.0)")
LAT = 58.2


def daylight_frac(i):
    n = sum(DAYS[:i]) + DAYS[i] / 2
    dec = math.radians(23.44) * math.sin(2 * math.pi * (284 + n) / 365)
    phi = math.radians(LAT)
    ch = (math.sin(math.radians(-0.833)) - math.sin(phi) * math.sin(dec)) / (math.cos(phi) * math.cos(dec))
    ch = max(-1.0, min(1.0, ch))
    return 2 * math.degrees(math.acos(ch)) / 15.0 / 24.0


DAY = [daylight_frac(i) for i in range(12)]
rows = {k: [] for k in ("rest129", "restbox", "act1", "act_diel", "rudstam", "john", "john_m", "bern")}
for i in range(12):
    T = T_0_50[i]
    mx = mix(i)
    rows["rest129"].append(rose_rest(129, T))
    rb = sum(w * rose_rest(W, T) for L, W, w in mx)
    rows["restbox"].append(rb)
    rows["act1"].append(sum(w * rose_rest(W, T) * math.exp(0.03 * L) for L, W, w in mx))
    # day 1 BL/s, night 0.7 BL/s (gadoid night/day speed ratio 10.5/15, Onsrud et al. 2005, used for herring [J])
    rows["act_diel"].append(sum(w * rose_rest(W, T) * (DAY[i] * math.exp(0.03 * L) + (1 - DAY[i]) * math.exp(0.03 * 0.7 * L))
                                for L, W, w in mx))
    rows["rudstam"].append(sum(w * rose_rest(W, T) * math.exp(0.03 * rud_vel(W, T)) for L, W, w in mx))
    # Johnstone et al. 1993: adults 25.5-31 cm (~160 g), 0.093 mg O2/g/h at 9.3 C, "low activity"; T by RQ 0.0548
    rows["john"].append(0.093 * O2 * math.exp(0.0548 * (T - 9.3)))
    rows["john_m"].append(sum(w * 0.093 * O2 * math.exp(0.0548 * (T - 9.3)) * (W / 160.0) ** -0.227 for L, W, w in mx))
    # Bernreuther 2007 ms 3: starved 6.6 g juveniles, 0.076-0.133 mg O2/g/h at 9.7 C; b 0.080 per C (per fish)
    rows["bern"].append(0.1045 * O2 * math.exp(0.080 * (T - 9.7)))
print("  Herring (kJ/kg/d) by month:")
print("  month  daylight  rest129  rest_box  x1BL/s  diel(1/0.7BL)  Rudstam VEL  Johnstone  Johnstone_m  Bernreuther")
for i in range(12):
    print(f"  {MON[i]}    {DAY[i]:.2f}    {rows['rest129'][i]:6.1f}   {rows['restbox'][i]:6.1f}   {rows['act1'][i]:6.1f}"
          f"     {rows['act_diel'][i]:6.1f}       {rows['rudstam'][i]:6.1f}     {rows['john'][i]:6.1f}     {rows['john_m'][i]:6.1f}"
          f"      {rows['bern'][i]:6.1f}")
H = {k: mmean(v) * 365 for k, v in rows.items()}
print("  annual MJ/t/yr: " + ", ".join(f"{k} {v:.0f}" for k, v in H.items()))
print(f"  box-size factor on the resting term (rest_box / rest129) = {H['restbox'] / H['rest129']:.2f}; "
      f"activity at 1 BL/s = {H['act1'] / H['restbox']:.2f}; diel = {H['act_diel'] / H['restbox']:.2f}; "
      f"Rudstam VEL = {H['rudstam'] / H['restbox']:.2f}")
print(f"  Johnstone low-activity routine / Rose resting at 160 g and 9.3 C = {0.093 * O2 / rose_rest(160, 9.3):.2f}; "
      f"Bernreuther fasting / Rose resting at 6.6 g and 9.7 C = {0.1045 * O2 / rose_rest(6.6, 9.7):.2f} "
      f"(range {0.076 * O2 / rose_rest(6.6, 9.7):.2f}-{0.133 * O2 / rose_rest(6.6, 9.7):.2f})")
print(f"  library charged on the herring part (8.0 x 1.5^((T-14)/10)): resting {RM_YR:.0f}, eat tick {RM_YR * FC:.0f} MJ/t/yr")

# Sprat: Meskendahl et al. 2010 (thesis ms 1): RR = 0.074 W^1.077 e^0.080T, RS = 0.069 W^1.073 e^0.078T mg O2/ind/h
sp_rr = [0.074 * 8.0 ** 0.077 * math.exp(0.080 * t) * O2 for t in T_0_50]
sp_rs = [0.069 * 8.0 ** 0.073 * math.exp(0.078 * t) * O2 for t in T_0_50]
SPRAT = {"routine": mmean(sp_rr) * 365, "standard": mmean(sp_rs) * 365}
print(f"  Sprat 8 g, 0-50 m (extrapolated below 9 C): routine {SPRAT['routine']:.0f}, standard {SPRAT['standard']:.0f} MJ/t/yr "
      f"({mmean(sp_rr):.0f} / {mmean(sp_rs):.0f} kJ/kg/d)")


# Norway pout (no respirometry): cod SMR (Schurmann & Steffensen 1997: 35.5 / 57.0 / 78.2 mg O2/kg/h at 5/10/15 C,
# juveniles of unstated mass, taken as 100-400 g, central 250 g) scaled with exponent -0.13 to the pout mass;
# bottom-layer temperatures (Kolmulens biomassa put pout on `bottom`).
def cod_smr(T):
    pts = [(5, 35.5), (10, 57.0), (15, 78.2)]
    if T <= 10:
        (t0, v0), (t1, v1) = pts[0], pts[1]
    else:
        (t0, v0), (t1, v1) = pts[1], pts[2]
    return v0 * (v1 / v0) ** ((T - t0) / (t1 - t0))  # mg O2/kg/h


def hansson(W, T):
    """Hansson et al. 1996 cod respiration (FB4 REQ 2, Kitchell): RA 0.008, RB -0.172, RQ 1.88, RTO 21, RTM 24."""
    RA, RB, RQ, RTO, RTM = 0.008, -0.172, 1.88, 21.0, 24.0
    V = (RTM - T) / (RTM - RTO)
    Z = math.log(RQ) * (RTM - RTO)
    Y = math.log(RQ) * (RTM - RTO + 2)
    X = Z ** 2 * (1 + (1 + 40 / Y) ** 0.5) ** 2 / 400
    f = V ** X * math.exp(X * (1 - V))
    return RA * W ** RB * f * 13560.0  # kJ/kg/d


POUT = {}
for W in (12.0, 25.0):
    smr = [cod_smr(t) * 0.3264 * (W / 250.0) ** -0.13 for t in T_BOT]  # kJ/kg/d (1 mg/kg/h = 0.3264 kJ/kg/d)
    smr_lo = [cod_smr(t) * 0.3264 * (W / 100.0) ** -0.13 for t in T_BOT]
    smr_hi = [cod_smr(t) * 0.3264 * (W / 400.0) ** -0.13 for t in T_BOT]
    hs = [hansson(W, t) for t in T_BOT]
    POUT[W] = dict(smr=mmean(smr) * 365, smr_lo=mmean(smr_lo) * 365, smr_hi=mmean(smr_hi) * 365, hansson=mmean(hs) * 365)
    print(f"  Norway pout {W:.0f} g, bottom layer: cod-SMR proxy {POUT[W]['smr']:.0f} ({POUT[W]['smr_lo']:.0f}-{POUT[W]['smr_hi']:.0f}) "
          f"MJ/t/yr = {mmean(smr):.1f} kJ/kg/d; Hansson resting term {POUT[W]['hansson']:.0f} MJ/t/yr = {mmean(hs):.1f} kJ/kg/d "
          f"(= {POUT[W]['hansson'] / POUT[W]['smr']:.2f} x the SMR proxy)")
# Blue whiting otolith FMR (tables/otolith_bw_stdout.txt): cod-anchored at 7.5 C, 100 g: 12 270 (fit, cod+bw),
# 13 135 (cod+bw+haddock), 14 705 (Q10 2), 12 691 (Q10 3); CI over the fits 9 230-17 740.
BW_OTO = {"low": 9232.0, "central": 12700.0, "high": 17736.0}
# Box blue whiting: Jul-Sep at 50 m (8.0-10.2 C, mean 8.9): Q10 2 from 7.5 C
bw_t = (T_50[6] + T_50[7] + T_50[8]) / 3
bw_f = 2.0 ** ((bw_t - 7.5) / 10)
print(f"  Blue whiting otolith (cod-anchored, 7.5 C, 100 g): {BW_OTO['central']:.0f} ({BW_OTO['low']:.0f}-{BW_OTO['high']:.0f}); "
      f"x {bw_f:.2f} to the Jul-Sep 50 m mean {bw_t:.1f} C")

print()
print("=" * 100)
print("4. Biomass-weighted field metabolism without SDA (MJ/t/yr)")
# Herring readings (annual MJ/t/yr): a measured low-activity routine (Johnstone), the same mass-scaled to the box,
# a seasonal reading (Wisconsin diel activity Apr-Oct, field fasting loss Nov-Mar), and the Wisconsin lineage.
# Field fasting loss in winter: NSS herring lose ~2 fat-points Sep-Dec (Slotte 1996 via Sillens vinterbudget),
# mass-scaled to the site mix: 10.8-14.4 kJ/kg/d; Bernreuther starved juveniles ~17 at 5.85 C (vinterkostnad).
WINTER = (12.6 + 17.2) / 2  # kJ/kg/d, Nov-Mar
seas = [rows["act_diel"][i] if 3 <= i <= 9 else WINTER for i in range(12)]
H["seasonal"] = mmean(seas) * 365
HERR = {"a Johnstone routine (adult, low activity)": H["john"], "b Johnstone mass-scaled to box sizes": H["john_m"],
        "c seasonal: Wisconsin diel Apr-Oct, fasting Nov-Mar": H["seasonal"],
        "d Wisconsin diel (1 BL/s day, 0.7 night)": H["act_diel"], "e Wisconsin 1 BL/s all day": H["act1"],
        "f Rudstam's own swimming speed (upper)": H["rudstam"]}
for k, v in HERR.items():
    print(f"  herring {k:52s} {v:7.0f} MJ/t/yr = {v / 365:5.1f} kJ/kg/d = {v / RM_YR:.2f} x library annual RM")
pout = {"low": POUT[25.0]["smr"] * 1.5, "central": BW_OTO["central"], "high": max(POUT[12.0]["smr"] * 2.2, POUT[25.0]["hansson"] * 1.25)}
bw = {k: v * bw_f for k, v in BW_OTO.items()}
sprat = {"low": SPRAT["standard"], "central": SPRAT["routine"], "high": SPRAT["routine"] * 1.3}
print(f"  pout      low {pout['low']:7.0f} / central {pout['central']:7.0f} / high {pout['high']:7.0f}: cod SMR (25 g) x 1.5 / "
      f"blue-whiting otolith FMR (cod-anchored, 100 g, 7.5 C) / max(cod SMR (12 g) x 2.2, Hansson (25 g) x 1.25)")
print(f"  bw        low {bw['low']:7.0f} / central {bw['central']:7.0f} / high {bw['high']:7.0f}: otolith (cod-anchored), x {bw_f:.2f} to Jul-Sep 50 m")
print(f"  sprat     low {sprat['low']:7.0f} / central {sprat['central']:7.0f} / high {sprat['high']:7.0f}: Meskendahl standard / routine / x 1.3")
CASES = {"low (a, pout low, bw low, sprat low)": (H["john"], "low"),
         "measured-routine (b, central others)": (H["john_m"], "central"),
         "seasonal (c, central others)": (H["seasonal"], "central"),
         "Wisconsin diel (d, central others)": (H["act_diel"], "central"),
         "high (e, pout high, bw high, sprat high)": (H["act1"], "high")}
FMR = {}
for lab, (h, k) in CASES.items():
    v = W_MEMBER["herring"] * h + W_MEMBER["pout"] * pout[k] + W_MEMBER["blue whiting"] * bw[k] + W_MEMBER["sprat"] * sprat[k]
    FMR[lab] = v
    print(f"  group {lab:44s}: {v:7.0f} MJ/t/yr = {v / 365:5.1f} kJ/kg/d = {v / RM_YR:.2f} x RM -> library-equivalent "
          f"eat share e_eq = {(v / RM_YR - RC) / (FC - RC):.2f}")
print(f"  library: all rest {RM_YR:.0f}; trained split {cost(0.81, 0.11):.0f}; e = 1 {cost(1.0):.0f}")
# Annual resting term the library would carry with the herring at the box sizes and pout at the cod-SMR proxy
rest_alt = (W_MEMBER["herring"] * H["restbox"] + W_MEMBER["pout"] * POUT[25.0]["smr"] + W_MEMBER["blue whiting"] * 7.42 * TPY
            * 0.8 + W_MEMBER["sprat"] * SPRAT["standard"])
print(f"  resting-term check: herring Rose at box sizes {H['restbox']:.0f}, pout cod-SMR proxy (25 g) {POUT[25.0]['smr']:.0f}, "
      f"blue whiting Hansson 7.42 MJ/t/tick x 0.8 (cod SMR / Hansson [J]), sprat standard -> weighted {rest_alt:.0f} "
      f"= {rest_alt / RM_YR:.2f} x the library's annual RM {RM_YR:.0f}")
# Winter check against Planktivorernas vinterkostnad (Jan-Feb group routine 22.4 kJ/kg/d, herring 17.2, pout 29.3)
jan_feb_h_j = (rows["john"][0] + rows["john"][1]) / 2
print(f"  Jan-Feb herring: Johnstone routine {jan_feb_h_j:.1f} kJ/kg/d, Rose resting at box sizes "
      f"{(rows['restbox'][0] + rows['restbox'][1]) / 2:.1f}, library resting {(4 * rm_m[0] + 4 * rm_m[1]) / 2:.1f} kJ/kg/d "
      f"(vinterkostnad central 17.2 from Bernreuther fasting)")

print()
print("=" * 100)
print("5. Field consumption and the respiration it implies")
# Greenstreet et al. 1997 Tables 1 and 3, pelagic planktivore guild (North Sea 1983-85; herring, sprat, sandeel,
# Norway pout 97-100 % of the biomass); consumption mgWW/m2/d and biomass gWW/m2 per quarter.
gc = [69.56, 124.80, 145.11, 115.60]
gb = [6.40, 7.52, 8.67, 9.84]
dr = [c / 1000 / b for c, b in zip(gc, gb)]
qb_g = sum(dr) / 4 * 365
qb_g2 = sum(gc) / 4 * 365 / 1000 / (sum(gb) / 4)
print(f"  Greenstreet guild daily ration by quarter: " + ", ".join(f"Q{i + 1} {100 * d:.2f} %/d" for i, d in enumerate(dr)) +
      f"; annual Q/B {qb_g:.2f} (mean of rations) / {qb_g2:.2f} (flow / mean B); Q1 / annual {dr[0] / (sum(dr) / 4):.2f}")
qb_members = {"herring": 4.7, "pout": 4.5, "blue whiting": 7.5, "sprat": 6.0}
qb_w = sum(W_MEMBER[k] * v for k, v in qb_members.items())
print(f"  params_planktivorous_fish member Q/B (herring 4.7, pout 4.5, bw 7.5, sprat 6.0) at today's weights: {qb_w:.2f} "
      f"(library 5.0 at the 209 weights herring 0.79)")
qb_box = 0.554 * (0.8 * 5.63 + 0.2 * 4.34) + 0.35 * 4.5 + 0.075 * 7.5 + 0.021 * 6.0
print(f"  same with the box's juvenile-heavy herring (Ecopath juvenile 5.63 / adult 4.34 at 0.8 / 0.2): {qb_box:.2f}")
# Prey energy: model zooplankton 4500, krill 5000; field copepods 3600, euphausiids 4000 (Bachiller 2018);
# Baltic zooplankton 2850 (Arrhenius & Hansson 1993); Greenstreet / Steele food 1 kcal/g = 4184.
for lab, ez, ek in (("model prey 4500/5000", 4500, 5000), ("Bachiller 3600/4000", 3600, 4000),
                    ("Greenstreet 4184/4184", 4184, 4184), ("Baltic 2850/4000", 2850, 4000)):
    q = q_net(ez, ek)
    print(f"  q at {lab:22s} = {q:6.0f} MJ/t net; I at Q/B 5.0 = {5 * q:6.0f}; = {5 * q / Q_MODEL:.2f} model-tonne Q/B")


# Respiration R = A C - P E_body (whole-body energy of the new tonnes at the field fill), FB4 herring A:
# egestion 0.16, excretion 0.10 of assimilated, SDA 0.175 of assimilated -> 0.609 of ingested left for R + G.
def resp(qb, ed, frac=0.84 * (1 - 0.10 - 0.175), p=P_SOM, s=S_FIELD):
    return qb * ed * frac - p * (EC + s * ME)


print(f"  R = Q/B x ED x 0.609 (FB4 herring losses) - P (ec + s ME), P {P_SOM}, s {S_FIELD:.2f} -> MJ/t/yr:")
for qb in (4.0, 4.87, 5.0, 5.11, 6.0, 7.0):
    print(f"    Q/B {qb:4.2f}: R at 3600 {resp(qb, 3600):6.0f}, at 4184 {resp(qb, 4184):6.0f}, at 4600 {resp(qb, 4600):6.0f}"
          f"   | model losses (A 0.8 - SDA 0.11 = 0.69): {resp(qb, 3600, 0.69):6.0f} / {resp(qb, 4184, 0.69):6.0f} / "
          f"{resp(qb, 4600, 0.69):6.0f}")
# Norway pout: Albert 1995, 1.1 % BW/d wet ration (July, 140 m, 7 C, 11-19 cm); FB4 cod losses (F 0.17, U 0.09,
# SDA 0.17 of assimilated -> 0.83 x 0.74 = 0.614); prey 4000 (copepods, krill, shrimps); P pout 1.2-1.4 /yr (SMS).
for p in (0.66, 1.2, 1.4):
    r = 0.011 * 365 * 4000 * 0.614 - p * 4600
    print(f"    pout at 1.1 %/d all year, P {p}: R {r:6.0f} MJ/t/yr = {r / 365:5.1f} kJ/kg/d "
          f"(cod-SMR proxy 12 g {POUT[12.0]['smr'] / 365:.1f}, 25 g {POUT[25.0]['smr'] / 365:.1f} kJ/kg/d)")

# 5b. The bioenergetic lineage's own consumption, in ingested energy per tonne-year
mean_e = BOOKED["zooplankton"] * E_ZOO + BOOKED["krill"] * E_KRILL
print(f"  library need in ingested energy (booked diet, mean prey {mean_e:.0f} MJ/t): trained split, field fill "
      f"{need_qb(cost(0.81, 0.11), S_FIELD) * mean_e:.0f}; e = 1, fill 1 {need_qb(cost(1.0), 1.0) * mean_e:.0f} MJ/t/yr; "
      f"field Q/B 5.0 at 3600-4184: {5 * 3600:.0f}-{5 * 4184:.0f}; Greenstreet 5.11 x 4184 = {5.11 * 4184:.0f}")
print(f"  Bachiller 2018 NSS herring (adults, Rudstam respiration at 1 BL/s): C/B 9.5-12 in 168 d at 3600-4000 J/g "
      f"= {9.5 * 3600:.0f}-{12 * 4000:.0f} MJ/t/yr ingested; blue whiting 4.7-6 in 183 d = {4.7 * 3600:.0f}-{6 * 4000:.0f}")
# Arrhenius & Hansson 1993 (Rudstam model, Baltic 1983-90): individual conversion efficiency (growth / consumption,
# 1 Jan to 31 Dec, wet mass) age 1: 5.2 %, age 2: 4.0 %; weights at 1 Jan from Table 2 (three populations).
for lab, w0, w1, eff in (("age 1, SD 22-24", 12.7, 23.9, 0.052), ("age 1, SD 25-27 coastal", 12.1, 28.6, 0.052),
                         ("age 1, SD 25-27 open sea", 10.9, 29.5, 0.052), ("age 2, SD 22-24", 23.9, 48.3, 0.040)):
    qb = (w1 - w0) / ((w0 + w1) / 2) / eff
    print(f"  Arrhenius & Hansson 1993 {lab:26s}: Q/B ~ (dW / W_mean) / eff = {qb:5.1f} /yr wet prey at 2850-3720 J/g "
          f"= {qb * 2850:.0f}-{qb * 3720:.0f} MJ/t/yr ingested")

print()
print("=" * 100)
print("6. Q/B needed (model tonnes, q = %.0f) and e* at Q/B 5.0 per lever" % Q_MODEL)


def estar(I, c_rest_mult=RC, c_eat_mult=FC, rm=RM_YR, s=S_FIELD, p=P_SOM, extra=0.0):
    return (I - p * (EC + s * ME) - extra - rm * c_rest_mult) / (rm * (c_eat_mult - c_rest_mult))


I5 = QB_FIELD * Q_MODEL
levers = []
levers.append(("library, field fill", cost(1.0), estar(I5)))
levers.append(("library, fill 1", cost(1.0), estar(I5, s=1.0)))
f_size = H["restbox"] / H["rest129"]
levers.append((f"herring RM at box sizes only (x{f_size:.2f} on 0.554)", cost(1.0) * (1 + 0.554 * (f_size - 1)),
               estar(I5, rm=RM_YR * (1 + 0.554 * (f_size - 1)))))
f_rest = rest_alt / RM_YR
levers.append((f"RM rebuilt per member (x{f_rest:.2f})", cost(1.0) * f_rest, estar(I5, rm=RM_YR * f_rest)))
r_rout = 0.093 * O2 / rose_rest(160, 9.3)
levers.append((f"rest = routine (x{r_rout:.2f} RM)", cost(1.0), estar(I5, c_rest_mult=r_rout)))
for k, v in FMR.items():
    levers.append((f"field FMR {k.split(' (')[0]}", v, (v / RM_YR - 1) / (FC - 1)))
gon = 0.58 * 0.15 * 6500  # adults ~0.58 of the biomass x GSI ~0.15 x roe/milt ~6.5 kJ/g [J]
levers.append((f"+ gonads ~{gon:.0f} MJ/t/yr (not in P)", cost(1.0) + gon, estar(I5, extra=gon)))
print("  lever                                         costs/FMR   Q/B need fill field / fill 1   e* at Q/B 5 (field fill)")
for lab, c, es in levers:
    print(f"  {lab:45s} {c:8.0f}      {need_qb(c, S_FIELD):5.2f} / {need_qb(c, 1.0):5.2f}            {es:6.2f}")
print("  (for the 'field FMR' rows the last column is the library-equivalent eat share e_eq, not e*)")
print(f"  e* with the field Q/B read at Bachiller prey energy (q {q_net(3600, 4000):.0f}): "
      f"{estar(5 * q_net(3600, 4000)):.2f} (field fill)")
for qb in (qb_w, qb_box):
    print(f"  e* at Q/B {qb:.2f} (model tonnes): {estar(qb * Q_MODEL):.2f} field fill / {estar(qb * Q_MODEL, s=1.0):.2f} fill 1")

print()
print("=" * 100)
print("7. Feeding-time share (share of biomass-time spent feeding), by member [J from the cited stomach evidence]")
herring_feed = [DAY[i] * (0.5 if i in (10, 11, 0, 1) else 1.0) for i in range(12)]  # daylight; winter at half rate [J]
pout_feed = [DAY[i] + 0.5 * (1 - DAY[i]) for i in range(12)]  # day and night, night stomach weight half (Albert 1995)
bw_feed = [0.5] * 12  # mainly night (Skagerrak), Jul-Sep only in the box; annual-mean weight already
sprat_feed = DAY[:]
fe = {"herring": mmean(herring_feed), "pout": mmean(pout_feed), "blue whiting": 0.5, "sprat": mmean(sprat_feed)}
grp = sum(W_MEMBER[k] * v for k, v in fe.items())
print("  " + ", ".join(f"{k} {v:.2f}" for k, v in fe.items()) + f"; group {grp:.2f}")
print(f"  daylight fraction at {LAT} N: " + " ".join(f"{d:.2f}" for d in DAY) + f"; annual {mmean(DAY):.2f}")
