"""Convert the measured krill rates (evidence.md) into the model's units.

Usage: python3 -I krill_rates.py

Units of the model: MJ per tonne wet weight (WW) per 6-h tick, and per year. Energy density of krill
5000 MJ/t WW (sibling note "Krill i Skagerrak", annual mean). Every input below is quoted in
evidence.md (E-number in the comment); the conversion factors are stated and are the only assumptions.
"""
OXY = 0.44          # J per umol O2 (lipid, Ikeda et al. 2000 via E35 Huenerlage 2015); 0.45-0.48 for mixed fuel
DW_WW = (0.20, 0.22, 0.27)   # dry/wet weight, summer / central / winter (E28)
E_KRILL = 5.0       # kJ/g WW = MJ/kg... = 5000 MJ/t (E28)
TICKS_PER_DAY = 4

print("1. Respiration (E21 Saborowski 2002; E33 Spicer & Saborowski 2010 Table 4.4)")
for label, r in (("acclimatised at ambient T (30-35)", 32.5), ("Kosterfjorden 5-6 C low", 21.9),
                 ("Kosterfjorden 5-6 C high", 40.6), ("Gullmarsfjord 6.5 C", 46.9),
                 ("Kattegat 4 C", 33.3), ("Clyde 4 C", 23.2), ("Clyde 16 C", 58.9)):
    for dw in DW_WW:
        j_gww_d = r * OXY * 24 * dw                  # J per g WW per day = MJ/t/d... (1 J/g = 1 MJ/t)
        frac = j_gww_d / (E_KRILL * 1000)            # share of body energy per day
        if dw == DW_WW[1]:
            print(f"  {label:36s} {r:5.1f} umol/gDW/h -> {j_gww_d:6.1f} MJ/t/d "
                  f"({j_gww_d / TICKS_PER_DAY:5.1f} MJ/t/tick) = {100 * frac:4.2f} %/d of body energy "
                  f"[DW/WW {DW_WW[0]}-{DW_WW[2]}: {r * OXY * 24 * DW_WW[0]:.0f}-{r * OXY * 24 * DW_WW[2]:.0f} MJ/t/d]")

print("\n2. T. inermis starved (E35 Huenerlage et al. 2015)")
fw_mg = 120.0
demand = 7.6                                         # J/d per animal (paper)
print(f"  {demand} J/d per {fw_mg:.0f} mg FW -> {demand / (fw_mg / 1000):.0f} MJ/t/d "
      f"(6 umol O2/h/g FW x 0.44 x 24 = {6 * OXY * 24:.0f} J/g FW/d)")
print(f"  dry weight 30.0 -> 20.5 mg in 28 d: {100 * (1 - 20.5 / 30.0):.0f} % DW lost, "
      f"{100 * (1 - (20.5 / 30.0) ** (1 / 28)):.2f} %/d")

print("\n3. Maximum ingestion (E34 McClatchie 1985 / PANGAEA 835086), 10 C, 'cultures isolated from Digby Neck' (food type not stated in the record)")
body_c = 14.58       # mg C per animal (.1458E5 ug)
ir = 33.8            # ug C per animal per h (.338E5 ng)
cr = 59.1            # ml per animal per h
print(f"  {ir} ug C/h = {ir * 24 / 1000:.3f} mg C/d = {100 * ir * 24 / 1000 / body_c:.1f} % body C/d; "
      f"max clearance {cr} ml/h = {cr * 24 / 1000:.2f} l/d per animal")

print("\n4. Copepod ration (E22 Bamstedt & Karlson 1998), % of body DW per day")
for lab, dr in (("lab mean", 12.7), ("lab max", 33.0), ("field station means low", 1.0),
                ("field station means high", 20.0)):
    print(f"  {lab:26s} {dr:5.1f} %DW/d (copepod energy/DW ~ krill energy/DW assumed)")

print("\n5. Tarling 2003 (E23), Clyde summer model: assimilated 25 (male) / 35 (female) J/d")
for ww in (0.15, 0.23, 0.30):
    print(f"  per {ww:.2f} g WW adult: {25 / ww:.0f}-{35 / ww:.0f} MJ/t/d = "
          f"{100 * 25 / ww / 5000:.1f}-{100 * 35 / ww / 5000:.1f} %/d of body energy")

print("\n6. Mortality (E23 Tarling 2003): adult beta 0.003 /d")
for b in (0.002, 0.003, 0.004):
    print(f"  {b} /d = {b * 365:.2f} /yr")

print("\n7. Clearance per tonne of krill per 6-h tick [I from M]")
# E22: 1.2 copepods ind-1 h-1 at 1.2-2 prey l-1 (lab tanks) -> clearance 0.6-1.0 l ind-1 h-1
for prey_l in (2.0, 1.2):
    cl = 1.2 / prey_l                                  # l per ind per h
    for ww_g in (0.15, 0.23):
        n_t = 1e6 / ww_g                               # ind per tonne WW
        print(f"  copepods at {prey_l} l-1: {cl:.2f} l/ind/h; adult {ww_g} g WW -> "
              f"{cl * 6 * n_t / 1000:,.0f} m3/t/tick")
# E34: maximum clearance 59.1 ml/h for an animal of 14.58 mg C; C/DW 0.45 [assumed], DW/WW 0.22 (E28)
ww = 14.58 / 0.45 / 0.22 / 1000
print(f"  McClatchie max 59.1 ml/h, animal ~{ww:.2f} g WW (C/DW 0.45 assumed) -> "
      f"{0.0591 * 6 * 1e6 / ww / 1000:,.0f} m3/t/tick")
