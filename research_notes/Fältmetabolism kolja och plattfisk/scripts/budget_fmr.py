"""What the otolith-based group FMR does to the benthivore budget of `Bottenfiskens budget`.

Mirrors research_notes/"Bottenfiskens budget"/scripts/levers.py (same library constants, read there;
that script is not edited) and re-runs (1) its finding-3 subtraction, (2) the 223.9 field closure that
gives the comparison Q/B, (3) the engine need at a feeding_cost re-derived from the new FMR, for each
growth cost c. Group FMR values from tables/group_fmr_stdout.txt (Route B) and the prior 7 260 / 8 200.
Per species (haddock, plaice) the same subtraction with species FMR and RM = 1.2 x species SMR.
No model output enters any number.  Usage: python3 -I budget_fmr.py
"""
RM = 4862.0            # resting_metabolism 3.33 MJ/t/tick x 1460 (levers.py)
SDA = 1.05 * 1460      # SDA at the field ration, 1.05 MJ/t/tick (levers.py; 240.2)
EC, ME = 3600.0, 1800.0
Q = 2197.0             # assimilated net SDA per tonne of the booked diet (levers.py)
LOSS, S_FIELD = 0.694, 0.62   # M1 0.29 /yr added (unit bug in closure.py, fixed 2026-11-11)
P = 0.74               # somatic production, section 250
E_WHOLE = 4500.0       # whole-body energy of production in the 223.9 closure
A_PER_T = 2472.0       # assimilated energy per t of the redistributed diet behind Q/B 4.46/4.66 (223.9-223.10)
FMR = {"prior W.2 (1.85 x SMR)": 7260, "223.9 note central (2.0 x SMR)": 8200,
       "otolith low": 7922, "otolith central": 8891, "otolith high": 9979,
       "otolith central, LRD = plaice": 8479}
C_CASES = [("library 0.69", 0.69), ("cod own 0.49", 0.49), ("split, haddock/flatfish synthesis 0.33 -> 0.38", 0.38),
           ("none", 0.0)]


def synth(c, s=S_FIELD, p=P):
    return c * p * (EC + s * ME)


print("1. Finding-3 subtraction: activity left = (FMR - SDA - synthesis c P (ec + s ME)) / RM, s 0.62")
print(f"   SDA {SDA:.0f}, RM {RM:.0f} MJ/t/yr; synthesis per unit c at P 0.74: {synth(1):.0f}")
print(f"   {'FMR case':36s} {'FMR':>6s} " + " ".join(f"{lab[:14]:>14s}" for lab, _ in C_CASES) + "   c_max (activity = RM)")
for name, f in FMR.items():
    cells = " ".join(f"{(f - SDA - synth(c)) / RM:14.2f}" for _, c in C_CASES)
    cmax = (f - SDA - RM) / synth(1)
    print(f"   {name:36s} {f:6.0f} {cells}   {cmax:.2f}")
print()

print("2. Field closure (223.9 form): assimilated need = FMR (incl. SDA and synthesis) + P x 4500")
for name, f in FMR.items():
    for p in (0.64, 0.74):
        need = f + p * E_WHOLE
        print(f"   {name:36s} P {p:.2f}: {need:6.0f} MJ/t/yr -> Q/B {need / A_PER_T:.2f} (redistributed diet)")
print()

print("3. Engine need with feeding_cost re-derived from the FMR: FC = (FMR - SDA - synthesis(c)) / RM,")
print("   need = RM (e FC + (1 - e)) + P (ec + c (ec + s ME)) + LOSS s ME, at e 0.95 (levers.py form)")
for name, f in FMR.items():
    for lab, c in C_CASES:
        fc = (f - SDA - synth(c)) / RM
        for s in (1.0, S_FIELD):
            costs = RM * (0.95 * fc + 0.05)
            g = P * (EC + c * (EC + s * ME))
            refill = LOSS * s * ME
            n = costs + g + refill
            flag = "" if fc >= 1.0 else "  (FC < 1: eating cheaper than resting)"
            print(f"   {name:36s} c {c:.2f} fill {s:.2f}: FC {fc:.2f} need {n:6.0f} -> Q/B {n / Q:.2f}{flag}")
    print()

print("4. Per species, FMR from tables/group_fmr_stdout.txt (Route B central, low-high) and RM = 1.2 x SMR,")
print("   own lab residual c, group P 0.74 (species P not known), SDA at the group value")
SP = {"haddock": (7937, 6803, 9297, 2.50, 0.84), "plaice": (6431, 4487, 7977, 2.85, 0.75),
      "cod": (11338, 11338, 11338, 3.53, 0.49)}
for sp, (fc_, lo, hi, smr, c) in SP.items():
    rm = 1.2 * smr * 1460
    for lab, f in [("central", fc_), ("low", lo), ("high", hi)]:
        act = (f - SDA - synth(c)) / rm
        cmax = (f - SDA - rm) / synth(1)
        print(f"   {sp:8s} {lab:7s} FMR {f:6.0f} RM {rm:5.0f}: activity left at own c {c:.2f} = {act:.2f} x RM;"
              f" c_max (activity = RM) {cmax:.2f}")

print()
print("5. Per species with own field P (eggs excluded, as in the engine) and both FMR routes:")
print("   haddock P 0.43-0.61 /yr (Jones 1978 Table 2, 217-362 g); plaice P ~0.34 /yr (Rijnsdorp & Ibelings")
print("   1989: somatic 543 kJ/yr on a ~1 600 kJ spent female). Otolith = Route B; ration = Q3 (E8, E11).")
ROWS = [("haddock", 4380, 0.84, 0.43, 0.61, [("otolith low", 6803), ("otolith central", 7937), ("otolith high", 9297),
                                              ("ration single-meal", 4900), ("ration continuous 450 g", 9400),
                                              ("ration continuous 600 g", 11200)]),
        ("plaice", 4993, 0.75, 0.34, 0.34, [("otolith low", 4487), ("otolith central", 6431), ("otolith high", 7977),
                                             ("ration annual (weak)", 7900)])]
for sp, rm, c_own, p_lo, p_hi, cases in ROWS:
    for lab, f in cases:
        out = []
        for pp in sorted({p_lo, p_hi}):
            cmax = (f - SDA - rm) / synth(1, p=pp)
            act = (f - SDA - synth(c_own, p=pp)) / rm
            out.append(f"P {pp:.2f}: c_max {cmax:5.2f}, activity at own c {c_own} {act:4.2f} x RM")
        print(f"   {sp:8s} {lab:24s} FMR {f:6.0f}: " + "; ".join(out))
print()
print("6. Group c_max at the central otolith FMR with a realistic-meal SDA 0.13 of intake (x1.3 of 1 533):",
      f"{(8891 - 1.3 * SDA - RM) / synth(1):.2f}")
