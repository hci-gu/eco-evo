# The krill pairs against winter stomachs (2026-11-12)

Research only, by a research subagent; saved by the main agent. Nothing in the library, code, tests, manifest,
resume or AGENTS.md was changed. No training or probe was run, and no model output set or selected a value.
Trail with quotes: `evidence.md`. Arithmetic: `scripts/` -> `tables/` (stdout saved). Papers: `papers/` (not
to be committed).

Tags: [F] full text, [L] lab, [A] abstract only, [S] secondary (an earlier note or a review), [D] my
arithmetic, [J] judgement.

## Question

In fix97 the planktivores (herring 0.554, Norway pout 0.35, blue whiting 0.075, sprat 0.021) and whiting eat
krill in Nov-Mar at 1.7-3.1x (planktivores) and 1.25-2.1x (whiting) what their reserve room can take.

Is the fish -> krill encounter consistent with what these fish are measured to eat in winter? If not, which
term is responsible:
- (A) the encounter a;
- (B) the winter Holling ceiling;
- (C) the fish meeting krill and not taking them?

## Model facts checked [D] (`tables/model_side_stdout.txt`)

**Library (270).**
- Planktivores -> krill: a 0.023, h 57.8, dark_ratio 0.57 at 0 deg.
- Whiting -> krill: a 0.043, h 159.5, no dark ratio.
- Each pair's depth_access_table and the krill depth shape are renormalised to an equal-cell mean of 1 on
  maps/depth.npz. The median bed is 143 m. The bed classes < 100 / 100-150 / 150-250 / >= 250 m hold 0.23 /
  0.30 / 0.32 / 0.15 of the cells.

**January, 100-150 m beds.**
- Planktivores:
  - a_eff = 0.023 x depth access 0.69 x daylight 0.88 x ingestion Q10 0.77 = 0.0135.
  - Krill target 5.0 t/km2; a h B = 5.3, saturation 0.84.
  - Encounter-limited single-prey intake 4.4 %BW/d.
  - Ceiling 1/h x Q10 = 5.3 %BW/d (6.9 at q 1). The Q10 here is normalised over a 12-month mean; the
    engine's calendar gives ~5.5.
- Whiting: a_eff 0.0345, saturation 0.96, intake 2.5 %BW/d; ceiling 2.6 %BW/d.
- Over < 100 m beds both pairs are as saturated. The encounter falls well below the ceiling only over beds
  >= 150 m (planktivores) or >= 250 m (whiting).

**fix97 ledger** (years 2-3, `Ransonen i energi/tables/ledger_fix97.json`):

| month | planktivores ration / krill %BW/d (krill share) | whiting ration / krill %BW/d (share) |
|---|---|---|
| Nov | 1.80 / 1.52 (0.84) | 1.04 / 0.94 (0.90) |
| Dec | 1.67 / 1.58 (0.95) | 1.03 / 0.93 (0.90) |
| Jan | 1.55 / 1.46 (0.94) | 1.00 / 0.88 (0.88) |
| Feb | 1.51 / 1.26 (0.83) | 0.95 / 0.81 (0.85) |
| Mar | 1.58 / 0.93 (0.59) | 0.92 / 0.76 (0.83) |

In Nov-Mar the planktivores' assimilated gain is 51-61 kJ/kg/d against a cost of 37-46. Whiting's is 31-35
against 22-25.

## Field winter rations and Cmax

| datum | site, months, T | fish | total ration %BW/d | krill ration %BW/d (share) | field Cmax %BW/d at that T | ration / Cmax | tag |
|---|---|---|---|---|---|---|---|
| herring, Lohmeyer & Hempel 1977 | W Baltic, Jan-Feb, 1-3 C | mostly < 25 cm | 0.4-1.0 (12-25 kJ/kg/d assim.) | 0 (gobies, Temora) | lab ad lib 3.5-5.0 at 6.5 C (6-19 g); ~1.7-2.4 krill-equiv. at box size | <= 0.2-0.5 | [S: F in `Sillens vinterföda`] |
| herring, Bernreuther 2007 | Bornholm Basin, winter | adults | <= 0.5, "more or less similar to maintenance" | 0 | as above | <= 0.2-0.3 | [S: F] |
| herring, Onsrud et al. 2004 | Oslofjord 120 m, Nov / Mar, deep water ~6-8 C | > 25 cm | fullness 2.3-3.4 of 5, 0-6 % empty | 0.03-0.18 (Nov), 0.08-0.53 (Mar) | as above | krill <= 0.1-0.3 | [D from F] |
| Norway pout, ICES stomach DB 1980 | NE North Sea, Nov-Mar | 13 cm, 29 g, 433 stomachs | 0.53-0.65 (GER, whiting set) | 0.20 (krill 34 % by mass) | GER at S 2.5-5 %BW: 1.8-2.9 | 0.2-0.35 | [D] |
| Norway pout, Onsrud 2004 | Oslofjord, Nov / Mar | 14-18 cm | fullness 2.9-4.0 | 0.36-0.82 (Nov), 0.51-1.14 (Mar) | 1.8-2.9 (GER); cod proxy ~3.5 at 6 C | 0.15-0.6 | [D from F] |
| Norway pout, Albert 1994 | Norwegian Deep incl. Skagerrak, Jan-Apr | < 14 / >= 14 cm | 0.4-1.3 (content 0.36-1.3 %BW, one turnover/d [J]) | euphausiids 79-81 % by mass | as above | 0.15-0.7 | [S: F in `Planktivorernas vinterkostnad`] |
| whiting, ICES DB 1981 (counts, no weights) | NE + NW North Sea, Q1 | 25-29 cm | - | <= 0.34-0.52 (3.7-8.8 krill per stomach) | GER 1.8-2.2 (S 2.5 %), 2.6-3.1 (S 5 %) at 6-8.5 C | krill <= 0.15-0.3 | [D] |
| whiting, ICES DB 1981 | Skagerrak, Q1, 51 stomachs | 21 cm | - | <= 0.11-0.16 (0.29 krill per stomach) | as above | <= 0.08 | [D] |
| whiting, SMS key run 2023 | North Sea, Q1 | 0.2 kg | 0.78 | - | as above | ~0.35-0.4 | [S: F] |
| whiting, Pedersen 2000 | North Sea, 1992-93 | - | 0.41-1.75 (Andersen GER) | - | - | - | [A] |
| whiting, Onsrud 2004 | Oslofjord, Nov / Mar | 21-28 cm | fullness 2.7-3.3, krill ~82 % by mass | 0.23-0.54 (Nov), 0.65-1.51 (Mar) | 2.0-3.1 | 0.1-0.7 | [D from F] |

**Group values [D/J].** These are biomass-weighted over herring 0.554 and pout 0.35. Blue whiting is not on
the bank in winter, and sprat eat no krill.

- **Planktivores: Nov-Mar krill ration 0.24 %BW/d (0.10-0.46); total 0.45-1.0 %BW/d.**
  - The model eats 0.93-1.58 krill and 1.5-1.8 in total: **x6 on krill (x3-15), x1.5-4 in total.**
  - Caveat: the Oslofjord herring stomachs are from adults > 25 cm, while the box's herring are mostly 15-25 cm
    juveniles. Juveniles feed more in late autumn (0.6-2 %BW/d, Arrhenius & Hansson via `Sillens vinterföda`).
    That would push the herring krill term up; the 0.10-0.46 range brackets it.
- **Whiting: krill 0.30 %BW/d (0.14-0.50) on the open shelf, 0.23-1.5 in the Oslofjord hotspot; total ~0.8.**
  - The model eats 0.76-0.94 krill and 0.92-1.04 in total.
  - Against the shelf that is **x3 on krill (x1.8-6), x1.3 in total**. It sits inside the hotspot range.

## Is the field winter fish limited by encounter, by its ceiling, or by itself?

### (B) The ceiling is not what limits the field fish [D]

- Every field winter ration is 0.1-0.5 (at most 0.7) of the field Cmax at that temperature (table above).
- The model's whiting ceiling (2.6 %BW/d in January) equals the GER Cmax (2.0-3.1).
- The planktivores' ceiling (5.3) is ~2-3x the lab-based herring Cmax at box sizes (~1.7-2.4). It does not
  bind: the reserve room cuts January intake to 1.55, below either value.
- Lowering h to the lab herring Cmax would cap the pre-cap demand (2.8) at ~2. The realised intake would be
  nearly unchanged.
- **Effect of (B) on the winter krill intake: 0 to -10 %.**

### (A) The encounter cannot be the field limit at any plausible krill density [D]

The field-implied a is a = R / (D (1 - h R)), with the library's January h. It lies far below the library's
January a_eff on 100-150 m beds. Each site is given its own density bracket:

| datum | R krill %BW/d | D bracket (t/km2) | library a / field a |
|---|---|---|---|
| planktivores, group central (shelf) | 0.24 | 3 / 8 (box winter floor / central) | 65 / 172 |
| pout, North Sea 1980 (shelf) | 0.20 | 3 / 8 | 78 / 208 |
| whiting, North Sea Q1 1981 (shelf) | 0.45 | 3 / 8 | 76 / 203 |
| pout, Oslofjord March (fjord) | 0.75 | 8 / 20 (fjord 4-16) | 50 / 124 |
| whiting, Oslofjord March (fjord) | 0.75 | 8 / 20 | 104 / 261 |
| any of the above at the Kattegat extreme | - | 72 (Alkor Deep 33-160) | 450-1900 |

The same holds from the bottom up, with the section 220 terms (scan 0.5, p_cap 0.5):
- A fish swimming at 0.10-0.15 m/s in a krill layer of 0.2-5 ind/m3 catches 3-1000 %BW/d at a reaction
  distance of 0.1-0.3 m. The dark floor R_nv is 0.1 m.
- Eating the field 0.25 %BW/d needs only 0.02-8 % of the time in the layer.
- Yet Onsrud et al. found whiting and pout "distributed together with the krill" day and night, and herring
  schools "above and in the upper part of the krill layer" by day [F]. The fish were in the layer.
- They ate 1.5-6 krill per stomach, with stomachs half full (fullness 2.3-4.0 of 5).

What (A) can still contribute inside the settled conventions:
- **Capture probability.** p_cap 0.5 replaced by the literature range 0.4-0.75 gives a x0.8-1.5
  (`Krillens flykt och energibudget`) [S].
- **Krill avoiding herring schools by day.** Christiansen et al. 2024 [S: F there] measured a median -68 % in
  krill density under schools > 10 m, ~-12 % under 5-10 m, and none at night.
  - In Dec-Jan about half of the planktivores' krill encounter falls in the daytime ticks (dark_ratio 0.57,
    ~1/3 of ticks lit) [J].
  - Herring is 0.55 of the biomass.
  - So the effect on the group a is at most x0.6-0.85.
- **Herring's own night term.** It is 0.11, against the group's 0.57, and would lower the group's winter a by
  ~x0.6-0.8. The vision rule and January light are settled; this only notes what the group value averages
  over.
- **Combined:** a x0.3-0.7 on a. On a pair at saturation 0.84 that moves the intake by only x0.75-0.95.
- Reaching the field krill ration through a alone needs a x0.01-0.05, i.e. a p_cap of 0.005-0.025. No
  measurement supports that: the only proxy, a lunging fish on mysid swarms, is 0.39-0.73 [S].
- **Effect of (A): -5 to -25 % on the planktivores' winter krill intake; ~0 to -5 % for whiting (saturation
  0.96).**

### (C) The field fish meet krill and take about their maintenance [D/F/J]

Two sources establish (C), because in both the food is shown to be present:
- **Onsrud et al. 2004 [F].** The fish were in the krill layer day and night. They had half-full stomachs,
  0-25 % empty, 1.5-6 krill per stomach, at rations of 0.1-0.5 of Cmax.
- **Berg, Seljestad & Folkvord 2024 [F].** Herring were reared for 3.5 yr at a constant 7 or 10 C, with "food
  provided in excess" and seasonal light. "The somatic condition of herring followed the daylength, with best
  conditions during summer and poorest during winter." Under a 6-month offset light regime the minimum moved
  with the light. The mechanism (feeding hours or an endogenous light-cued appetite) is not resolved.

Two sources give the winter level only. They do not show food being declined, since the Bornholm Basin is a
poor feeding area and the western Baltic has no krill:
- Bernreuther 2007: "The daily rations during winter were more or less similar to maintenance rations MRA."
- Lohmeyer & Hempel 1977: intake "might even be sufficient to cover the basic and locomotory metabolism ...
  without drawing on the fat reserves" [S: F].

Supporting: Norway pout lose condition and liver weight over winter while their stomach contents relative to
body weight are higher than in summer, and they spawn in Feb-Apr (Albert 1994; Raitt 1968: low feeding
intensity in January and in March-April in adults) [S: F].

**Effect of (C).** If the planktivores' Nov-Mar krill intake were the field krill ration, January would fall
from 1.46 to 0.24 %BW/d (x0.18; x0.07-0.34). Whiting would fall to x0.35 (x0.16-0.58).

### Verdict [J]

**(C).** The library's encounter a is far above any field-implied a. But no plausible correction to its terms
can turn the field ration into an encounter-limited Type II. In winter the field fish are limited neither by
their ceiling nor by encounter: they eat about their maintenance at food levels where they could eat more.
- (A) adds a small, herring-only daytime term.
- (B) is not a lever.

## Effect sizes on the model (fix97 ledger, years 2-3) [D]

**Krill flow.**
- The five fish groups eat 19 570 t/yr of krill.
- Planktivores take 6 980 t in Nov-Mar. That is 0.52 of their own annual krill and **0.36 of the five-group
  total**.
- Whiting takes 683 t in Nov-Mar, 0.035 of the total.
- Mackerel, the water column and the benthivores together take 0.04 of the total in Nov-Mar.

**Field-ration counterfactual.** Here the planktivores' and whiting's Nov-Mar krill intake is set to the field
krill ration, with biomass and the other months as realised.
- The five-group krill flow falls **-32 % (-25 to -36 %)**, from 19 570 to 13 400 t/yr.
- Planktivores go from 6 980 to 1 240 t; whiting from 683 to 237 t.
- This is larger than the main agent's "about a sixth" (`prey_timing.py`: -17 %). That estimate applied the
  field energy need to the model's own diet. Here the field winter diet also has much less krill than the
  model's krill share of 0.84-0.95.

**Fill.** A planktivore eating the field total ration (0.45-1.0 %BW/d, 16-36 kJ/kg/d net) changes fill over
Dec-Mar by:
- **-0.08 to -0.55** at the model's own costs (37-42 kJ/kg/d), against the field start_fill drop from 0.68 in
  December to 0.20 in April (-0.48);
- -0.15 to +0.33 at the field routine cost (22 kJ/kg/d).

So the field fill cycle follows from field rations only together with the model's higher winter cost. That
cost is almost all eat cost, from foraging on 97 % of ticks.

## Candidate ranges for the user's decision (none applied)

1. **The missing term is (C): a winter intake at about maintenance while food is met.**
   - The field data give the target: planktivores 0.45-1.0 %BW/d total and 0.10-0.46 krill; whiting ~0.8
     total and 0.14-0.5 krill.
   - They do not give the mechanism. The candidates are already in `Ransonen i energi`: the hunger gate
     (lipostat) instead of `reserve_room` (section 240 history), or a cap on reserve deposition.
   - Berg et al. 2024 adds a light-cued candidate: condition follows daylength with excess food.
   - A daylength-keyed appetite would be a season switch unless written as a physical payoff, and the project
     rules (behaviour learned, no season-keyed switches) may exclude it. The user decides.
2. **(A), herring only, by day: krill avoidance of schools (Christiansen 2024), x0.6-0.85 on the group's
   daytime krill a.**
   - It cannot be quantified without the share of planktivore biomass in schools > 5-10 m by day.
   - It has a small effect on a saturated pair.
3. **(B): no change is supported.** The planktivores' krill h could be re-derived on the lab herring Cmax (a
   ~2x lower ceiling), but the ceiling does not bind in winter.
4. **Not a krill-pair change, but it carries the fill:** the planktivores' winter cost (37-42 vs a field 22
   kJ/kg/d) is the eat cost of near-continuous foraging (`Planktivorernas vinterkostnad`).

## What I could not obtain

- **Herring winter stomachs with weights in the box.** The public ICES DB has no herring record with a position
  (`Krill över grunda bottnar/stomachs`).
- **Whiting winter stomach weights in the Skagerrak.**
  - 1981 has counts only.
  - 2022 IBTS Q1 is unusable as a ration: of 387 whiting, 352 have a blank StomachEmpty and only 43 have prey
    rows (mean content 0.05 %BW).
  - The "own whiting 68-92 %" rows of the earlier whiting table come from 4 whole-fish prey rows in tiny
    samples.
- **A measured krill density at any stomach site.** Onsrud 2004 gives none, so D is bracketed.
- **A Norway pout gastric evacuation model.** The whiting set is used as a proxy.
- **A measured fish capture probability on euphausiids** (G1 of the sibling note).
- **Huse & Ona 1996**, overwintering herring: only a search snippet saying they barely feed, with Calanus out
  of reach. Not used.
- **Fish lengths of the Onsrud stomach fish.** Assumed: whiting 21-28 cm, pout 14-18 cm, herring 25-30 cm.
