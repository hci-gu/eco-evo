"""Group Q10 multiplier for planktivorous_fish when the deep members (blue whiting, Norway pout)
sit in the 50 m layer while herring and sprat follow the 0-50 m mean (mareld2.yaml tables).
m_i(t) = Q10 ** ((T_i(t) - t_ref) / 10); group m = sum w_i(t) m_i(t) with month-specific shares
(blue whiting only Jul-Sep). Usage: python3 planktivores_temperature.py
"""
T0 = [6.3, 5.4, 5.2, 6.1, 7.8, 9.9, 12.1, 13.4, 13.8, 12.8, 10.5, 8.4]   # mean_0_50m
T50 = [7.7, 7.2, 6.6, 6.6, 6.8, 7.2, 8.0, 8.5, 10.2, 11.7, 10.6, 9.3]  # at_50m
Q10, TREF = 2.2, 14.0
# monthly biomass (t/km2): herring July 4.3 scaled to an annual mean of 2.6 with a winter low;
# pout 0.25 all year; sprat 0.1; blue whiting 0.35 annual mean, all of it in Jul-Sep (1.4 t/km2)
her = [2.0, 1.9, 1.9, 2.0, 2.4, 3.0, 3.6, 3.6, 3.3, 2.9, 2.5, 2.1]
bw = [0, 0, 0, 0, 0, 0, 1.4, 1.4, 1.4, 0, 0, 0]
pout = [0.25] * 12
spr = [0.1] * 12
M = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()
print('herring annual mean %.2f, bw %.2f' % (sum(her) / 12, sum(bw) / 12))
num_h = num_g = 0
for k in range(12):
    mh = Q10 ** ((T0[k] - TREF) / 10); md = Q10 ** ((T50[k] - TREF) / 10)
    tot = her[k] + bw[k] + pout[k] + spr[k]
    mg = ((her[k] + spr[k]) * mh + (bw[k] + pout[k]) * md) / tot
    num_h += mh; num_g += mg
    print(f"{M[k]} T0 {T0[k]:4.1f} T50 {T50[k]:4.1f}  m(0-50) {mh:.3f}  m(50) {md:.3f}  group {mg:.3f}  ratio {mg/mh:.3f}  deep share {(bw[k]+pout[k])/tot:.2f}")
print('annual mean multiplier: 0-50 m %.3f, group %.3f' % (num_h / 12, num_g / 12))
