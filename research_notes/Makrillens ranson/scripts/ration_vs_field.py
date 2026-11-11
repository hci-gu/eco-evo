"""Mackerel's realised ration in the box against field values, per month.
Inputs (no new run): research_notes/"Krillens flykt och energibudget"/diagnostics/tables/
ration_budget_pzbpgp248nsrpfix96.json (year 2: ration, I, C, G per month, frozen fix96, probe world) and
hide_pattern_pzbpgp248nsrpfix96.json (year 2: mean biomass per month = box presence).
Field values: (1) the SMS / Ecopath annual Q/B 2.2 spread over quarters by the ration shares per fish present
in research_notes/"Makrillens sasong" (a-umlaut) tables/feeding_season_stdout.txt (mean of the four phi rows:
0.0475 / 0.3575 / 0.3325 / 0.2575; SMS age-1 row 0.01 / 0.36 / 0.54 / 0.09); (2) the energetic need of a fish
that follows the field fat cycle (the library's outside_fill, monthly fills of the arriving fish) with the
model's own costs C and growth G: need = max(0, C + G + d fill x ME 6980 x 12) / q, q = presence-weighted I /
ration. Everything presence-weighted by the box biomass. usage: python3 -I ration_vs_field.py"""
import json, os
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                 'Krillens flykt och energibudget', 'diagnostics', 'tables')
rb = json.load(open(os.path.join(D, 'ration_budget_pzbpgp248nsrpfix96.json')))[1]
hp = json.load(open(os.path.join(D, 'hide_pattern_pzbpgp248nsrpfix96.json')))[1]
i = hp['dm'].index('mackerel'); B = np.array([hp['biomass'][m][i] for m in range(12)])
mon = rb['monthly']['mackerel']
get = lambda k: np.array([mon[m][k] for m in range(12)])
R, I, C, G, F = get('ration'), get('I'), get('C'), get('G'), get('fill')
w = lambda x: float((x * B).sum() / B.sum())
q = float((I * B).sum() / (R * B).sum())
fill = np.array([0.70, 0.58, 0.45, 0.37, 0.13, 0.51, 0.57, 0.91, 1.00, 1.00, 0.94, 0.85])
dR = (np.roll(fill, -1) - fill) * 6980 * 12
need = np.maximum(0, C + G + dR) / q
sms = {k: np.repeat(np.array(v) * 2.2 * 4, 3) for k, v in
       {'phi-mean': [0.0475, 0.3575, 0.3325, 0.2575], 'SMS age 1': [0.01, 0.36, 0.54, 0.09]}.items()}
print(f"q {q:.0f} MJ assimilated (net SDA) per t eaten")
print("month   B   model fill  field fill  model  need  SMS-quarters   (t/t/yr)")
for m in range(12):
    print(f"{m+1:4d} {B[m]:5.2f} {F[m]:8.2f} {fill[m]:9.2f} {R[m]:7.2f} {need[m]:5.2f} {sms['phi-mean'][m]:6.2f}")
print(f"presence-weighted: model {w(R):.2f}, energetic need on the field fat cycle {w(need):.2f}, "
      + ", ".join(f"field {k} {w(v):.2f}" for k, v in sms.items()) + ", annual stock Q/B 2.20")
for lab, s in (('Jan-Apr', slice(0, 4)), ('May-Sep', slice(4, 9)), ('Oct-Dec', slice(9, 12))):
    print(f"  {lab}: presence share {B[s].sum() / B.sum():.2f}; model {float((R[s]*B[s]).sum()/B[s].sum()):.2f}, "
          f"need {float((need[s]*B[s]).sum()/B[s].sum()):.2f}")
