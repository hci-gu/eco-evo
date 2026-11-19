"""Order-of-magnitude light at the capture depth of the field samples.
Assumptions (stated, not measured at the fish): E(z) = E0 exp(-K z), no moon,
no bioluminescence; 54 lx per umol m-2 s-1 (35 for 490 nm light).
Polar night: E0 = 1.5e-5 umol m-2 s-1 at noon in air (Cohen et al. 2015,
Kongsfjorden, January); K 0.03-0.10 /m judged for winter fjord water.
Skaret et al. 2020: K = 0.064 /m from their CTD PAR cast (Great Bank, October);
surface at civil dawn ~3 lx (~0.06 umol) and at noon taken from their isolume
reading (0.001 umol at ~140-155 m), back-computed.
Usage: python3 -I field_light_at_depth.py
"""
import math
LX = (35.0, 54.0)
print("Polar night, noon, E0 = 1.5e-5 umol (air):")
for K in (0.03, 0.05, 0.10):
    row = []
    for z in (90, 150, 250, 350):
        e = 1.5e-5 * math.exp(-K * z)
        row.append("%d m %.1e umol (%.0e lx)" % (z, e, e * LX[1]))
    print("  K %.2f: " % K + "; ".join(row))
print("Skaret 2020, K 0.064, bed 161-175 m:")
for lab, e0 in (("civil dawn, 0.06 umol at surface", 0.06),):
    for z in (161, 175):
        e = e0 * math.exp(-0.064 * z)
        print("  %s, %d m: %.1e umol (%.1e - %.1e lx)" % (lab, z, e, e * LX[0], e * LX[1]))
# noon: isolume 0.001 umol at 140-155 m -> bed
for ziso in (140, 155):
    for z in (161, 175):
        e = 1e-3 * math.exp(-0.064 * (z - ziso))
        print("  noon (0.001 umol isolume at %d m), %d m: %.1e umol (%.1e - %.1e lx)"
              % (ziso, z, e, e * LX[0], e * LX[1]))
print("Polar night with a full moon up: ~0.2 lx in air (prior note; moon elevation and cloud ignored):")
for K in (0.03, 0.05, 0.10):
    print("  K %.2f: " % K + "; ".join("%d m %.0e lx" % (z, 0.2 * math.exp(-K * z)) for z in (90, 150, 250, 350)))
