# Mackerel's ration against the field Q/B (2026-11-11)

Main agent. Question: is mackerel's realised ration in the box (6.2-6.9 t/t/yr in fix96's long-run probe,
~3x the field Q/B 2.2) a model excess, or a comparison error? Nothing changed in the library or code.
Script `scripts/ration_vs_field.py` -> `ration_vs_field_stdout.txt` (no new run: it reads the frozen-fix96
tables of `Krillens flykt och energibudget/diagnostics`). Tags: [S] source / note, [D] computed here, [J]
judgement.

## The field value

Q/B 2.2 is the annual STOCK value (SMS 2023 and Ecopath 1991; `Fiskgrupper i rutan`
params_predatory_fish.md), including the non-feeding quarters ("mackerel Q1 C/B ~0.05"); that note says
not to put feeding-season rations on an annual-mean share. Since section 246 mackerel migrates, so the
box holds it mainly while it feeds; the box ration per box-tonne must be compared with the field rate
of the months it is present [S].

## Results [D]

| | t/t/yr, presence-weighted |
|---|---|
| model (frozen fix96, probe world, year 2) | 6.15 |
| field, SMS / Ecopath Q/B 2.2 spread by the quarterly ration shares per fish present (`Makrillens säsong`) | 2.75 (3.37 with the SMS age-1 shares) |
| energetic need of a fish that follows the FIELD fat cycle (outside_fill 0.13 in May -> 1.00 in Sep, 6 073 MJ/t deposited) with the model's own costs and growth | 4.99 |

1. **The annual stock Q/B is the wrong yardstick**: weighted by presence it becomes 2.75-3.37.
2. **The SMS ration cannot pay the measured fattening.** Laying down 0.87 of a 6 980 MJ/t reserve in
   May-September needs ~11 t/t/yr (2.5-3.1 % BW/d) in May and July with the model's costs; the SMS
   quarters give 2.9-3.2 (0.8-0.9 % BW/d). The field values disagree with each other: the library's
   own Cmax anchor is Bachiller et al. 2018's Norwegian Sea feeding-season ration of ~8 % BW/d
   (`Rovfiskens uppdelning` params_split_mack.md). On the energetic need, the model is x1.23, not x3.
3. **Where the remaining x1.23 sits** (presence share; model vs need):
   - May-Sep (0.73): 6.97 vs 6.57, close; the model fattens earlier (fill 0.76 in April, 1.00 by
     August) than the field (0.37 in April, 1.00 in September).
   - Oct-Dec (0.22): 3.50 vs 0.99. The field fish draw on their fat (1.00 -> 0.85) and feed less
     (adult stomachs with content 0.63-0.68 in Jun-Aug, 0.27 in Sep, 0.22 in Oct, `Makrillens säsong`);
     the model's fish stay at fill 1.00 and eat to their costs.
   - Jan-Apr (0.06): 5.79 vs ~0; the small winter remnant eats to keep a full reserve, and the
     April immigrants are topped up at once.
   So the excess is the autumn-winter appetite: at a full reserve with food available (the krill pool
   delivers all year) the reserve_room fish never switch off, and costs are free in the reward at a
   full reserve, so ARS has no reason to learn the field's autumn drawdown [J].

## Consequences for the krill excess [D, J]

- Mackerel's booked krill M2 (0.10) rests on the annual Q/B 2.2. If the field-consistent ration while
  present is ~5 (the energetic need), the booking itself is ~x1.8 low (4.99 / 2.75) and the realised /
  booked krill ratio 7-8 shrinks to ~4; the rest is the diet share (krill 0.58-0.67 of the diet vs booked
  0.39, x1.5-1.7; the pair is unsaturated, 0.48, so its a matters) and the x1.23 autumn appetite.
- Nothing here is a parameter fix. Re-booking the M2s from a presence-weighted, energetically consistent
  ration would be a library decision (it moves the krill's and planktivores' derived g); the autumn
  appetite is a behaviour/appetite question (field mackerel stop feeding in autumn with food still
  present, i.e. a seasonal appetite that the reserve_room rule does not have and the reward does not
  teach).

## Caveats

- One seed, frozen policies, the probe world; the need uses the model's own costs (RM 4.59 obligate
  swimming floor, feeding_cost 1.02, Q10 2.0), so it checks the field fat cycle against the field ration,
  not the costs themselves.
- outside_fill is the fat of the arriving (outside) population by month; using it as the box fish's fat
  trajectory is an approximation [J].
- Bachiller's 8 % BW/d and SMS's ~0.9 % BW/d differ ~9x; neither was re-read here.
