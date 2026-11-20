"""Derived numbers [D] for lit_cost.md. No inputs; all constants are stated
with their source tags in lit_cost.md. Run: python3 -I scripts/derive.py"""
RHO_W = 1.026      # seawater g/ml (Strand et al. 2005 use 1026 kg/m3)
RHO_L = 0.93       # lipid (triacylglycerol) g/ml, task convention
E_L = 39.5         # MJ/kg lipid
g = 9.81
lift_per_kg_lipid = RHO_W / RHO_L - 1           # kgf upward per kg lipid
print('upward force per kg lipid: %.4f kgf' % lift_per_kg_lipid)

# --- lean tissue density -------------------------------------------------
# herring: Brawn 1969 [S]: >12 % oil keeps swim bladder 2.6-3.0 % of body
# volume at sinking factor 1000-1008 (taken as density ratio x 1000).
for sb, sf in ((0.026, 1.000), (0.028, 1.004), (0.030, 1.008)):
    d_fish = sf * RHO_W
    d_tissue = d_fish / (1 - sb)            # body without the gas
    f = 0.12                                # oil mass fraction
    d0 = (1 - f) / (1 / d_tissue - f / RHO_L)
    print('herring: bladder %.3f sinking %.3f -> tissue %.4f, lean (0 %% oil) %.4f'
          % (sb, sf, d_tissue, d0))
D_LEAN = {'gadoid': 1.081, 'herring': 1.080, 'mackerel': 1.085}
# full-fill reserve lipid per tonne lean (task conventions) and floor lipid
FULL_MJ = {'gadoid': 2000., 'herring': 4990., 'mackerel': 6980.}
FLOOR_KG = {'gadoid': 4., 'herring': 20., 'mackerel': 10.}
print()
for k in ('gadoid', 'herring', 'mackerel'):
    excess = 1000 * (1 - RHO_W / D_LEAN[k])          # kgf per tonne lean
    lip = FULL_MJ[k] / E_L
    w0 = excess - FLOOR_KG[k] * lift_per_kg_lipid
    w1 = excess - (FLOOR_KG[k] + lip) * lift_per_kg_lipid
    vol0 = 1000 / D_LEAN[k] + FLOOR_KG[k] / RHO_L
    vol1 = vol0 + lip / RHO_L
    dv = vol1 / vol0 - 1
    print('%-8s lean excess %.1f kgf/t; reserve lipid %.0f kg/t; excess empty %.1f -> full %.1f '
          '(cut %.0f %%, quadratic %.0f %%); volume +%.1f %%, wetted area (girth only) +%.1f %%'
          % (k, excess, lip, w0, w1, 100 * (1 - w1 / w0), 100 * (1 - (w1 / w0) ** 2),
             100 * dv, 100 * ((1 + dv) ** 0.5 - 1)))
    print('          lipid needed for neutral buoyancy without gas: %.0f kg/t lean (%.0f %% of body)'
          % (excess / lift_per_kg_lipid, 100 * (excess / lift_per_kg_lipid) / (1000 + excess / lift_per_kg_lipid)))

# --- hydrodynamic lift cost (per kg lean), P_met = W U / (L/D * eta) -------
print()
def lift_cost(excess_frac, U, LD, eta):
    W = excess_frac * g            # N per kg
    return W * U / (LD * eta) * 86400 / 1000   # kJ/kg/d = MJ/t/d
for name, frac, U, rm in (('herring deep (bladder compressed), 0.30 m/s', 0.0479, 0.30, 32.),
                          ('mackerel 35 cm at 1 BL/s', 0.0534, 0.35, 18.4)):
    for LD, eta in ((10, 0.2), (5, 0.2), (5, 0.1), (3, 0.1)):
        c = lift_cost(frac, U, LD, eta)
        print('%-45s L/D %2d eta %.2f: lift %.1f MJ/t/d = %.2f RM' % (name, LD, eta, c, c / rm))

# --- gadoid liver: non-lipid liver mass vs fill (Davie 2005 regression) ----
print()
def liver(hsi, lip):
    moist = (95.775 - lip) / 1.1536
    other = 100 - lip - moist
    return hsi * (1 - lip / 100), hsi * other / 100, moist, other
for hsi, lip in ((2, 20), (4, 40), (8, 60)):
    nl, prot, m, o = liver(hsi, lip)
    print('HSI %d %% lipid %d %%: moisture %.1f %%, other %.1f %%; non-lipid liver %.2f %% BW, '
          'non-lipid-non-water %.2f %% BW; lipid %.0f kg/t lean -> %.0f MJ/t'
          % (hsi, lip, m, o, nl, prot, 10 * hsi * lip / 100 / (1 - hsi * lip / 1e4),
             39.5 * 10 * hsi * lip / 100 / (1 - hsi * lip / 1e4)))
# porgy (Oikawa & Itazawa 2003) at W = 1000 g, umol O2 / g / h
W = 1000.
q_liver = 13.39 * W ** -0.130
q_muscle = 0.80 * W ** -0.035
whole = 5.30 * W ** 0.816 / W
m_liver = 0.0170 * W ** 1.008
print('porgy 1 kg: liver %.2f, white muscle %.2f, whole body %.2f umol/g/h; liver mass %.1f g '
      '(%.1f %% BW) = %.1f %% of whole O2; liver/whole per g %.2f'
      % (q_liver, q_muscle, whole, m_liver, m_liver / 10, 100 * m_liver * q_liver / (whole * W),
         q_liver / whole))
r = q_liver / whole
add_nonlipid = liver(8, 60)[0] - liver(2, 20)[0]
add_other = liver(8, 60)[1] - liver(2, 20)[1]
print('added non-lipid liver %.2f %% BW x %.2f (lean liver/whole per g, /0.9 for its lipid) = +%.1f %% SMR'
      % (add_nonlipid, r / 0.9, add_nonlipid * r / 0.9))
print('added liver protein-etc %.2f %% BW = %.2f %% BW lean-liver equivalent (14.3 %% other) -> +%.1f %% SMR'
      % (add_other, add_other / 0.143, add_other / 0.143 * r))
for rm in (13., 16.):
    print('  gadoid RM %.0f MJ/t/d: +5 %% = %.2f MJ/t/d per full fill (2000 MJ/t) = %.4f %% of the store per day'
          % (rm, 0.05 * rm, 100 * 0.05 * rm / 2000))

# --- adipose (muscle-lipid fish): human Elia values as proxy ------------
print()
ad_vs_muscle = 4.5 / 13.
muscle_vs_whole = q_muscle / whole
for k, lip in (('herring', 4990 / E_L), ('mackerel', 6980 / E_L)):
    tissue = lip / 0.85
    print('%-8s adipose %.0f kg/t x (%.2f x %.2f) = +%.1f %% RM at full fill (upper; carp: fat negligible)'
          % (k, tissue, ad_vs_muscle, muscle_vs_whole, 100 * tissue / 1000 * ad_vs_muscle * muscle_vs_whole))

# --- storage efficiency ----------------------------------------------------
print()
for inv in (1.10, 1.31):
    print('lipid deposition 1/kL %.2f -> kL %.2f, loss %.0f %% of deposited energy' % (inv, 1 / inv, 100 * (inv - 1)))
print('mobilisation: 2 ATP (FA activation) / 106 ATP (palmitate) = %.1f %%' % (100 * 2 / 106.))
