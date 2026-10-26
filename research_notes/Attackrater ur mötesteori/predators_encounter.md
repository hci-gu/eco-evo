# Attack rate a of the `predatory_fish` pairs from encounter theory

Group: saithe 0.41, whiting 0.37, mackerel 0.08, spurdog 0.05, horse mackerel 0.034, hake 0.025,
anglerfish 0.014, ling, velvet belly, pollack (long-run NS-IBTS biomass shares). Grid: Skagerrak box,
3600 cells of 1 km2, depth 44-529 m (mean 167 m; 23 % of cells shallower than 100 m, 53 % shallower than
150 m; `maps/depth.npz`).

Method: a is built bottom-up from detection radius, search speed, capture probability, the thickness of
the prey layer and individuals per tonne. Nothing was tuned to model output; no `results/` file was read,
the simulator was not run, and no code, library, manifest or resume file was edited. Tags: [F] full
text / data, [A] abstract, [S] secondary, [D] computed here, J = judgement. Quotes, DOIs and tags of
every source are in `predators_encounter_evidence.md` (E1-E6). Scripts in `scripts/`, printed numbers in
`tables/` (`pred_encounter_a_stdout.txt` is the master output).

## 1. Bottom line

### 1a. Pairs (km2 per tonne of foraging predator per 6-h tick; annual mean = library a)

| Pair (engine h) | a low corner | **a central** | a high corner | MC 5-50-95 % | 1/(a h) central (MC 90 %) | engine B for the Holling term | x = a h B_vis, central | night/day (MC 90 %) | Library now |
|---|---|---|---|---|---|---|---|---|---|
| -> planktivorous_fish (279.4) | 7.6e-5 | **0.0056** | 0.29 | 0.0019 / 0.0058 / 0.018 | **0.64 t/km2** (0.20-1.8) | 3.33 (visible 1.17-3.33) | 1.8-5.2 (0.65-0.84 of ceiling) | **1.25** (0.13-2.4) | a 0.04, dark_ratio 0.6 |
| same, schools NOT inside a (variant, see 4.1) | 2.6e-4 | 0.012 | 0.33 | 0.0041 / 0.013 / 0.032 | 0.30 | | 3.9-11 | 0.35 | |
| -> zooplankton (176.5) | 6.0e-5 | **0.0015** | 0.097 | 0.00087 / 0.0021 / 0.0054 | **3.8 t/km2** (1.1-6.5) | 9.72 (A17 winter 0.2-0.9) | 0.13 (B 0.5) - 2.6 (B 9.72) | **0.51** (0.21-0.67) | a 0.04, none |
| -> benthic_community (117.6) | 4.1e-4 | **0.030** | 2.4 | 0.0095 / 0.036 / 0.14 | 0.29 t/km2 of B_acc (0.06-0.89) = ~0.015 t/km2 of edible prey | 20.3 accessible (0.2 x 101.6) | 70 (0.99 of ceiling) | **0.99** (0.78-0.98) | a 0.04, none |
| -> benthivorous_fish (176.5) | 2.9e-3 | **0.18** | 9.4 | 0.058 / 0.20 / 0.65 | **0.032 t/km2** of B_acc (0.009-0.098) | 0.23 accessible (0.15 x 1.53) | 4.3-7.2 (0.81-0.88 of ceiling) | **0.62** (0.09-0.94) | a 0.04, none |

- Corners: every factor at its low-a or high-a end at once. MC: each factor drawn independently,
  log-triangular between its ends with the central value as mode, one draw shared by all members.
  The corners span 3.5-4 decades and are not confidence limits; the MC 90 % band (about one decade) is the
  better working range, given independence.
- Annual mean = 0.5 a_day + 0.5 a_night. A lit fraction 0.55-0.6 (58 N, twilight counted) moves the
  annual a by <= 5 % (zooplankton, benthivores) and < 2 % (planktivores).
- Variant with detection radius proportional to predator length (R x L/0.30 m): 0.0068, 0.0010, 0.013,
  0.19. Size structure is a second-order lever (<= 2.3x).

**Against the library 0.04:** planktivores 7x above the central and above the MC 95 % (inside the
corners); **zooplankton 27x above the central and above the MC 95 %** (the one pair where the engine is
saturated, half-saturation 0.14 t/km2, while encounter biology puts it at 3.8 t/km2: linear-to-moderate
in summer, linear in winter); benthos about at the central, saturated either way (x 70 here, 96 at 0.04);
benthivores 4.5x below the central and below the MC 5 %, saturated either way (0.81 here vs 0.62 of the
ceiling at 0.04).

### 1b. Factors per member (central; low-high in section 2)

N/t and L_num from `tables/pred_numbers_per_tonne_stdout.txt` (L_num = number-weighted mean length, the
length at which a clearance linear in L gives the stage's clearance per tonne exactly). v = BL/s x L_num.

| Member, stage (weight in group) | N/t | L_num cm | v m/s (BL/s) | R fish prey day / dark (m) | R krill / copepod (m) | R bed prey (m) | p fish / krill / bed | a P / Z / B / Bf (central, annual) |
|---|---|---|---|---|---|---|---|---|
| saithe adult < 60 cm (0.26) | 1241 | 43.0 | 0.43 (1.0); night x 0.6 | school 15 / 0.2 | 0.1 / 0.02 | 0.1; fish 0.25 / 0.2 | 0.2 / 0.5 / 0.4 | 0.0028 / 0.00034 / 0.0058 / 0.081 |
| saithe large >= 60 cm (0.15) | 304 | 70.0 | 0.70 (1.0); night x 0.6 | same | same | same | same | 0.0011 / 0.00013 / 0.0023 / 0.032 |
| whiting juvenile < 20 cm (0.09) | 58545 | 11.8 | 0.094 (0.8) | school 15 / 0.2 (day R capped at 2 L) | 0.1 / 0.02 | 0.1; 0.25 / 0.2 | same | 0.077 / 0.0044 / 0.078 / 1.3 |
| whiting adult (0.28) | 7330 | 25.2 | 0.20 (0.8) | school 15 / 0.2 | 0.1 / 0.02 | same | same | 0.012 / 0.0011 / 0.020 / 0.28 |
| mackerel (0.08) | 6035 | 26.6 | 0.48 (1.8) | school 15 / 0.2 | krill 0.2 day (x 0.5 overlap) / filter gape 5 cm2 x v x 0.6 | - | same; night feeding x 0.2 | 0.019 / 0.0023 / - / 0.51 (score 0.01) |
| spurdog (0.05) | 628 | 69.9 | 0.35 (0.5) | school 15 / 0.3 non-visual | 0.1 / - | 0.1; 0.3 | same | 0.0029 / 0.00013 / 0.0030 / 0.069 |
| horse mackerel (0.034) | 16596 | 12.2 | 0.16 (1.3) | school 15 / 0.2 | 0.1 / 0.02 | 0.1 | same | 0.025 / 0.0020 / 0.036 / 0.52 |
| hake (0.025) | 2100 adult | 36.3 | 0.18 (0.5) | 0.4 (ambush, midwater) / 0.2 | 0.1 / - | 0.1; 0.25 / 0.2 | same | 0.0034 / 0.00023 / 0.0052 / 0.073 (adult) |
| anglerfish (0.014) | 247 | 58.4 | ambush: prey 0.1 m/s past it | strike 0.4 | - | strike 0.4 | same | 0.00038 / - / 0.0024 / 0.027 |
| ling, velvet belly, pollack (0.016) | 550 / 4825 / 471 | 63 / 38 / 56 | hake / spurdog / saithe proxies | | | | | see stdout |

Prey layer thickness H (central, low-a to high-a end): herring and sprat schools by day 70 m (100-40;
July HERAS box profile), dispersed at night 40 m (60-30); blue whiting 50 m (100-30); Norway pout 15 m
(30-10); krill 30 m by day (40-20), 40 m at night (60-30); copepods 50 m (100-30); bed invertebrates 0.5 m
(1.0-0.2); settled juvenile gadoids 2 m (5-1). School encounter: radius 15 m (10-30), school 3 t (10-1),
1 prey caught per encounter (0.3-3), 50-g herring. Capture on single fish 0.2 (0.1-0.36). Shares of the
engine's Holling biomass: planktivore pool herring 0.79, blue whiting 0.105, pout 0.075, sprat 0.03 (held
at central); krill 0.12 (0.08-0.25) of the zooplankton pool; edible exposed bed prey 0.05 (0.02-0.12) of
the accessible benthos.

**Who sets a.** Clearance per tonne scales with N/t x L (v ~ L, R set by the prey): the planktivore pair
a is 46 % adult whiting, 9 % juvenile whiting and 22 % saithe, although saithe is 62 % of the pair's
predator_share weight; the zooplankton pair is 39 % juvenile whiting, 23 % mackerel, 19 % adult whiting,
10 % saithe; benthos 67 % juvenile whiting; benthivores 68 % adult whiting.

## 2. Derivation

### 2.1 Unit chain (`scripts/pred_encounter_a.py`)

The engine's intake per tick is min(eat, phi) x hunger x a B / (1 + a h B + w B_pred), B the visible
(and accessible) prey biomass, depth-integrated t/km2 (`interactions.holling_a_eff`). Each pair is its own
Holling term with its own ceiling 1/h, applied to the eat share the policy gives that prey
(`predation.apply_predation`). So a is the area cleared per tonne of predator per tick while foraging on
that prey, inside the prey's layer: the time budget is the eat share, the capable share is phi, both
outside a. With T = 21 600 s:

- Dispersed prey in a layer of thickness H: a = scan x A(R, H) x v_rel x T x N_t x p / H / 1e6 x s.
  A(R, H) is the detection disc clipped to the slab (pi R^2 when 2R <= H); in a thin near-bed layer
  (2R > H) it becomes the strip 2RH, so the same formula covers bed prey (areal density = engine t/km2).
- Schools: a = A(R_s, H) x v_rel x T x N_t x (k w / M_s) / H / 1e6 x s (encounter with schools of M_s
  tonnes, k prey of w tonnes caught per encounter).
- v_rel = Gerritsen-Strickler mean speed (u^2 + 3v^2)/(3v); prey speeds 0.005 (copepods) to 0.3 m/s
  (schools); it adds < 10 %.
- scan 0.5 (0.5-1): half of the visual disc is searched (Varpe & Fiksen 2010; Langbehn et al. 2019 [F]).
- Ambush (anglerfish): v is replaced by the prey's speed past the predator.
- s = share of the engine's Holling biomass that this prey type makes up (prey-side composition).
- Pair a = sum_i W_i sum_k score_ik a_ik / sum_i W_i score_i,pair: W_i is the stage's group biomass
  share, score its central capability score from `../Fiskgrupper i rutan/predators/inputs/pred_scores.csv`.
  The denominator equals the engine's phi (0.533, 0.478, 0.145, 0.290), so a is per tonne of the share
  that can take the prey, as the engine applies it.

### 2.2 Individuals per tonne [D]

Long-run GOV length biomass (`../Fiskgrupper i rutan/predators/tables/pred_length_biomass.csv`) x the
survey length-weight fits (`../Fiskgrupper i rutan/tables/length_weight_all.csv`), N/t = sum_c b_c / W(L_c)
x 1e6 with mid-class lengths (`scripts/pred_numbers_per_tonne.py`). Stage cuts as in the parent note:
whiting < 20 cm, saithe >= 60 cm, hake < 20 cm. Range 247 (anglerfish) to 58 545 (juvenile whiting) per
tonne. Horse mackerel N/t (16 596) is dominated in numbers by small fish.

### 2.3 Search speed

Saithe 1.0 BL/s (0.5-1.4): Uopt 1.4 BL/s, in-situ gadoids mostly < 1 BL/s (EN parent evidence A Q4).
Night x 0.6 (0.3-1.0): saithe reduce activity at night (Skilbrei & Ottera 2016, Armannsson & Jonsson 2012
[F]); Dolly Varden fall to 0.14 of day speed (Beauchamp 1999 via Eckmann [S]); J. Whiting 0.8 BL/s
(0.5-1.3), no night reduction (pelagic whiting feed at night, Pedersen 2000 [A]). Mackerel 1.8 BL/s
(1.25-3.0; Macy et al. 1998 feeding speed [F]). Horse mackerel 1.3 (0.5-1.8; Mapukata 2002 [F],
juveniles). Spurdog 0.5 (0.3-0.8; Sulikowski et al. 2010 net speed, a lower bound [F]). Hake 0.5 (0.2-1.0;
J, an ambusher in midwater, Carpentieri & Colloca 2005 [F]).

### 2.4 Detection radius R and light

No measured R exists for any member on any prey (E1). The pieces:
- Visual R on fish prey: lake trout RD = 0.255 I^0.475, cap 1.0 m at 17.8 lx (Beauchamp lab equation
  [S/F]); RDmax 1.0-1.9 m for trout and Chinook [A]. At the herring layer's depths (script table, Kd
  0.15/0.20, cloud 0.7): summer noon 1.0 m at 30-50 m and 0.06-0.33 m at 70 m; winter noon 0.15-0.5 m at
  50 m, <= 0.12 m at 70 m; R < 0.05 m at 100 m in every case. Central 0.5 m (0.15-1.5) for the cruisers
  by day, capped at 2 body lengths (J; models cap at ~1 BL).
- Dark / deep R on fish prey: 0.2 m (0.05-0.5): non-visual floor 0.1 m (Langbehn 2019 model [F]), Chinook
  low-light asymptote ~0.5 m (Rohan 2021 [F]), RD(0.01 lx) 0.03 m. Used at night for every cruiser and
  by day for blue whiting (slope, 150-300 m) and Norway pout (5-10 m above the bed), where the light is
  far below the cod scotopic range even at noon (vision note: < 0.005 lx below ~100 m at Kd 0.15-0.2).
- Spurdog: 0.3 m (0.1-0.5) day and night (J; olfactory, electro and lateral-line search, light
  independent). Hake by day 0.4 (0.15-1.0; J, dimmer than the cruisers).
- Herring schools by day: encounter radius 15 m (10-30): school half-length 10-14 m (day schools 21-27 m
  long, 3.5-5 m high, Fernandes et al. 2001 [F]) plus a few metres of detection; detection grows only ~2x
  for a 100x larger group (Krause & Ruxton via Pertzelan 2022 [S]). J.
- Krill: 0.1 m (0.05-0.2) day and night (J). Rohan's pollock model gives 11.6 cm on Calanus at saturating
  light; scaled by the square root of the image area to a 3-cm krill that is ~0.6 m in bright light, but
  krill sit below 70 m by day (Onsrud & Kaartvedt 1998 [F]) where vision is near its floor; bioluminescence
  and the non-visual floor (0.1 m) carry it. Whiting eat krill with "no systematic" day/night difference
  (Onsrud et al. 2004 [F]), consistent with a light-independent R. Mackerel near the surface by day 0.2 m
  (0.1-0.4), but only 0.5 (0.3-1.0) of its day foraging overlaps the krill layer (mackerel in the upper
  40 m by day, Diaz 2013 [F]).
- Copepods: 0.02 m (0.01-0.05) for particulate gadoids: Macy's mackerel clearance implies R 1.7-1.8 cm
  [D]; Rohan's 9.6-11.6 cm needs bright light. Mackerel filter instead: gape 5.0 cm2 (3.3-7.8; MacKay via
  Macy [F]; Hamann 2023 [F]) x v x retention 0.6 (0.3-0.9; Macy efficiencies 18-45 % at filtering
  concentrations [F]).
- Bed invertebrates: 0.1 m (0.05-0.2), J, non-visual at the bed; the gadoid swept-width assumption was
  0.05-0.3 m (EN `../Bentos åtkomlighet för torskfiskar/evidence_B_gadoid_foraging.md`).
- Juvenile gadoids on the bed: day 0.25 m (0.1-0.5; J, the bed in 77 % of the grid is below 100 m),
  night 0.2 (dark value).

### 2.5 Capture probability and school yield

Single fish 0.2 (0.1-0.36): perch on single roach 0.20, pike pursuit 0.12 and sit-and-wait 0.36,
pikeperch 0.085 and feeding in the dark (Turesson & Bronmark 2004 [F]); sailfish 0.24 on schools
(Herbert-Read 2016 [F]). Krill 0.5 (0.3-0.7; J, Langbehn's piscivore pc 0.7 [F]); copepods 0.7 (0.5-0.9);
bed invertebrates 0.4 (0.2-0.7; J). Schools: perch success on schools 1/31 = 0.03 per attack [F]; gadoid
attacks on herring schools are "frequent" but leave the school intact (Pitcher 1996 [A], Nottestad 2002
[F]); 85 % of Vestfjord saithe had eaten herring (Nottestad 1998 [F]). Yield k = 1 (0.3-3) herring per
school encounter (J). School 3 t (10-1): 25 000 fish per school on the Trench slope (Misund 1995 [F]) x
the box's 50-g juvenile herring = 1.25 t; 5.7-10 t per school as an upper bound from Petitgas 2001 [D];
attacked Vestfjord schools 4.5-18 t [F].

### 2.6 Vertical overlap

Divide by the prey layer's thickness, with the predator foraging inside it (its eat share). Herring by day
70 m (40-100; the July HERAS backscatter in the box is mostly 10-100 m, EN acoustic note); at night
dispersed in the upper 30-50 m (Onsrud 2004 [F], Falk 1968 [F]); night overlap 1 (0.5-1; J, saithe
reduce activity and do not all rise). Blue whiting: near the bottom on the slope day and night (Torgersen
1997 [F]); thickness not found, 50 m (30-100; J). Norway pout 5-10 m above the bed (Albert 1994 [A]):
15 m (10-30). Krill: 20-40 m thick layer below 70 m by day, 10-30 m to ~60 m at night (Onsrud 1998,
2004 [F]). Copepods 50 m (30-100; J). Suprabenthos 76-97 % in the lowest 0.5 m by day (Cunha 1997 via
EN [A]): 0.5 m (0.2-1.0). Settled juvenile gadoids within 1-5 m of the bed (cod > 8 cm fully demersal,
Bromley & Kell 1999 [A]): 2 m (J).

### 2.7 Prey-side shares of the engine biomass

- Planktivore pool (3.33 t/km2): herring 0.79, blue whiting 0.105, pout 0.075, sprat 0.03
  (`../Fiskgrupper i rutan/params_planktivorous_fish.md`). By prey type at the central (saithe adult):
  herring schools by day 0.0018, the same herring dispersed by day 0.011, at night 0.0021; pout 0.0007,
  blue whiting 0.0003. Schooling cuts the day herring term ~6x.
- Zooplankton pool (9.72 t/km2, A17 mesozooplankton): krill 1.17 (0.78-2.4) g/m2 in the North Sea
  (Mackinson & Daskalov 2007 via EN [F]) -> share 0.12 (0.08-0.25) [D]. No Skagerrak krill biomass was
  found; the Fladen "more than 90 %" (second-hand) is a spring peak, not a mean.
- Benthos: the engine's accessible biomass 0.2 x 101.6 = 20.3 t/km2 is mostly exposed infauna. What these
  fish eat is exposed mobile epifauna: Pandalus 0.56 x f 0.7 = 0.39, small shrimps / mysids / small crabs
  ~half of 1.27 x 0.6, hyperbenthos 0.2 (EN `../Bottensamhällets epifauna/summary_standing_stock.md`) ->
  ~1.0 (0.4-2.5) t/km2 = 0.05 (0.02-0.12) of B_acc [D].
- Benthivores: accessible fraction 0.15 is already the size-accessible share (< ~20 cm), so s = 1.

### 2.8 Night/day ratio

- Planktivores 1.25 (0.13-2.4): by day the herring are in schools (encounter diluted ~6x) and the deep prey
  are in the dark anyway; at night herring disperse into the upper 40 m but R falls from 0.5 to 0.2 m and
  saithe slow down. The two effects nearly cancel; the encounter model cannot tell 0.6 from 1.0. If
  schooling is left to the engine's hide action (variant), the ratio is 0.35.
- Zooplankton 0.51 (0.21-0.67): mackerel feed by day (night x 0.2; Dahl & Kirkegaard 1986), krill spread
  over a thicker layer at night (40 vs 30 m), saithe slow down. Without mackerel the gadoid ratio is 0.8.
- Benthos 0.99: non-visual at the bed day and night.
- Benthivores 0.62 (0.09-0.94): the small daytime R advantage at the bed and saithe's slower night search.

## 3. Validation (not used to set a)

| Data | What it tests | Result against the encounter a |
|---|---|---|
| Macy et al. 1998 [F]: mackerel 184 g, particulate 19.8 L/min (3.7 copepods/L), filter 6.3-7.5, small copepods 2.6 | the mackerel copepod term | per tonne over a 50-m layer: 7.8e-4 (particulate peak), 2.7e-4 (filter), 1.0e-4 (small copepods) km2/t/tick [D]; the model's mackerel term is 3.3e-4 by day. Consistent. Macy: Georges Bank densities are "too low to stimulate feeding" - mackerel feed in patches, i.e. the pool is in the linear range for them |
| Engelhard et al. 2008 [F]: whiting intake vs local sandeel, log-log slope 0.08 | the local half-saturation (Type II: slope = 1/(1+x)) | x = 11.5 -> a = 0.012-0.035 at B_vis 1.2-3.3 t/km2 [D]. Above the central 0.0056, at the MC 95 % / the no-school variant. Points to the upper half for fish prey. Different prey (sandeel, emerging at dawn), aggregation at the bank |
| Lopez-Lopez et al. 2015 [F]: hake fullness rises from 100-1000 blue whiting per km2 | the density at which feeding starts | 0.005-0.1 t/km2 [D]; read as 1/(a h) it gives a 0.036-0.7. A Type III threshold, not a half-saturation; order only. Supports high a on dispersed deep prey |
| Fall et al. 2021 [F]: cod on capelin, "main increase ... below median density" | half-saturation below the typical local density | central 1/(a h) 0.64 t/km2 < 3.33 (planktivores): consistent; zooplankton 3.8 vs 9.72: consistent in summer only |
| Kempf et al. 2008 [F]: stomach content plateaus from relative abundance ~0.1-0.2 | early saturation at moderate densities | consistent with x 1.8-5 on planktivores |
| Feeding level 0.49 of Cmax (parent note: SMS / Ecopath rations) | whether full-time foraging on one pair would overshoot | 0.49 from planktivores alone needs a 0.001-0.003; from zooplankton alone 0.0006-0.011 [D]. The encounter a gives more than that at full-time foraging, so the eat share (time budget) and hunger must carry the rest: as expected, not a contradiction |
| Lauerburg et al. 2018 [F]: whiting stomach content -60 % when the forage index fell 21 % | stock-scale elasticity 1.7-3.8 [D] | Type II cannot exceed 1: a stock-scale overlap / Type III effect, not a test of a |
| Rindorf, Gislason & Lewy 2006 [F] | the switching exponent b (0.15-0.35), not a | tests the pair shapes / caps (run the b regression on model diets); says nothing about the level of a |
| Rindorf 2002 [F]: satiation acts on search activity | where satiation sits | supports the engine's hunger gate on the encounter term |
| WKCAPELIN 2023 [F]: cod-capelin Type III levels out at ~500 kt | stock-scale ceiling | no area given; not converted |

Not found: any lab functional response with absolute a for a gadoid on fish or krill; saithe stomach
fullness vs pout, krill or herring density; Rindorf & Gislason 2005 parameters (403; worth retrieving by
hand: whiting intake vs local prey density at five sites).

## 4. Judgements, forks and the dominant uncertainty

### 4.1 Forks that conflict with engine conventions (the caller's decision)

1. **Schooling inside a vs the prey's hide action.** Here schooling is encounter geometry inside a (central
   0.0056, night/day 1.25). The engine also protects the Rest (hide) fraction of the planktivores, with a
   pair visibility floor 0.25. If that hide action is meant to stand for the schooling refuge, schooling
   must not also sit in a: use the dispersed variant (0.012, night/day 0.35). Engelhard's 0.012-0.035
   lands on the variant; this is a test outcome, not a reason to move the central.
2. **Krill share: phi and a disagree.** phi_zoo 0.478 was built with the pool half krill
   (`pred_prey_shares.py`, f = 0.5); the evidence here supports 0.08-0.25. At 0.12, phi_zoo = 0.12 x 0.650
   + 0.88 x 0.304 = 0.35 [D, from the species contributions in `pred_phi_stdout.txt`]. If phi is
   recomputed, the per-capable-tonne a rises by 0.478/0.35 = 1.38x (0.0021); the product phi x a, which
   is what the engine feeds, is unchanged.
3. **Benthos: a and accessible_fraction are one product.** f 0.2 was derived for benthivore access to
   infauna; for this group the edible exposed share of total B is ~0.01. a 0.030 on B_acc 20.3 equals
   a ~0.6 on f 0.01. Both are saturated (x ~ 70): as concluded for the gadoids
   (`../Bentos åtkomlighet för torskfiskar/benthos_accessibility.md`), the levers are phi and h, not a.
   The 1/(a h) 0.29 t/km2 is in B_acc units (~0.015 t/km2 of real edible prey).
4. **Night/day above 1 cannot be expressed.** `daylight.attack_multiplier` treats dark_ratio >= 1 as no
   light effect. The planktivore ratio is 1.25 (0.13-2.4): the mechanics are indifferent between 0.6 and
   unset. The diel stomach evidence (cod on sprat 87 % at dusk and dawn, Andersen 2017; saithe activity at
   sunrise) shows a crepuscular peak, which a logistic in sun elevation cannot represent either. Keep 0.6
   from the diel-stomach weighting (parent note) or unset it; the encounter model does not decide. The
   zooplankton pair (0.51; 0.8 without mackerel) could carry a dark_ratio of ~0.5-0.8 where today none is
   set; the benthivore pair 0.6-1.0.
5. **One a per pooled group is whiting-set.** The a per tonne is dominated by the many small whiting (N/t
   x L); saithe, 62 % of the planktivore pair's weight, gives 22 % of its a. A change in the size
   structure or the whiting share moves a more than any saithe parameter.
6. **Seasonal layers are not modelled.** Herring H is July HERAS; wintering herring are deeper (Falk 1968);
   krill and Calanus overwinter deep. The composition of the planktivore pool is held at its central shares
   (the 0.67-0.90 herring sensitivity is not in the corners).

### 4.2 Dominant uncertainty

**The detection radius in dim and dark water**, for which no member has a measurement: one at a time
(others central) R_fish_dark spans 1.07 decades of the planktivore a, R_krill 0.84 of the zooplankton a,
R_bed_inv 1.20 of the benthos a and 0.85 of the benthivore a. Second: the prey-type share of the engine's
biomass (krill share 0.45 decades; edible bed share 0.78). Third, for planktivores: the school
parameters (R_s, k, M_s, each ~0.3 decades) and the schooling fork 4.1.1 (2.1x).

### 4.3 What the numbers say, stated plainly

- Planktivores: half-saturation 0.64 t/km2 (0.2-1.8) against 1.2-3.3 t/km2 visible: partly saturated,
  with a density response. 0.04 makes it fully saturated (0.089 t/km2).
- Zooplankton: the clearest result. Copepods are too small for particulate gadoids, and krill are ~12 %
  of the pool. Half-saturation 3.8 t/km2: the pair responds to prey density all year and is nearly linear
  in winter. 0.04 is 27x above the central.
- Benthos and juvenile benthivores: saturated under every defensible combination; a is not identifiable
  from the encounter side and not the lever.

## 5. References

Full citations, DOIs, quotes and access tags: `predators_encounter_evidence.md`. Main sources: Albert 1994
ICES JMS 51:45; Andersen et al. 2017 MEPS 583:195; Bamstedt & Karlson 1998 MEPS 172:149; Beauchamp et al.
1999 CJFAS 56 S1:128 (via Eckmann et al. 2016 Ecol Freshw Fish); Bromley & Kell 1999 Acta Adriatica
40:7; Carpentieri & Colloca 2005 Fish Bull 103:411; Cunha 1997 MEPS 157:119; Diaz et al. 2013 ICES CM
2013/N:05; Engelhard et al. 2008 MEPS 369:213; Falk 1968 ICES CM 1968/H:17; Fall et al. 2021 Ecography
44:802; Fernandes et al. 2001 ICES CM 2001/Q:08; Hamann et al. 2023 Front Mar Sci 10:1253083; Hansen et al.
2013 TAFS 142:854; Herbert-Read et al. 2016 Proc R Soc B 283:20161671; Kaartvedt et al. 1996 MEPS 136:51;
Kaartvedt et al. 2005 MEPS 299:1; Kempf et al. 2008 MEPS 367:295; Klocker et al. 2024 Mov Ecol 12:62;
Langbehn et al. 2019 MEPS 623:161; Lauerburg et al. 2018 MEPS 594:213; Laurenson et al. 2004 J Fish Biol
65:947; Lopez-Lopez et al. 2015 MEPS 541:165; Macy et al. 1998 MEPS 172:89; Mapukata 2002 MSc UCT; Meager
et al. 2005 CJFAS 62:1978; Meager et al. 2010 MFBP 43:99; Misund et al. 1995 ICES JMS 52:11; Nottestad 1998
ICES CM 1998/J:20; Nottestad et al. 2002 ICES JMS 59:393; Onsrud & Kaartvedt 1998 MEPS 171:209; Onsrud et
al. 2004 ICES JMS 61:1278; Petitgas et al. 2001 ICES JMS 58:1150; Pitcher et al. 1996 ICES JMS 53:449;
Rindorf 2002 J Fish Biol 61:579; Rindorf, Gislason & Lewy 2006 MEPS 325:243; Rohan 2021 PhD Univ.
Washington; Skilbrei & Ottera 2016 ICES JMS 73:1186; Armannsson & Jonsson 2012 ICES JMS 69:1372;
Sulikowski et al. 2010 MEPS 418:249; Torgersen et al. 1997 Sarsia 82:87; Turesson & Bronmark 2004 J Fish
Biol 65:363; Varpe & Fiksen 2010 Ecology 91:311; Vogel & Beauchamp 1999 CJFAS 56:1293. Repository notes:
`../Fiskgrupper i rutan/params_predatory_fish.md` and `predators/`, `../Torskfiskars ljusberoende vid jakt/`,
`../Ekolodsdata planktivorer/`, `../Bottensamhällets epifauna/`, `../Bentos åtkomlighet för torskfiskar/`,
`../Sillens vinterföda/`.

## Files

- `scripts/pred_numbers_per_tonne.py` -> `tables/pred_numbers_per_tonne.csv`, `_stdout.txt`.
- `scripts/pred_encounter_a.py` (all factors with sources inline) -> `tables/pred_encounter_a_stdout.txt`,
  `pred_encounter_summary.csv`, `pred_encounter_stage_detail.csv`.
- `scripts/pred_validation_arith.py` -> `tables/pred_validation_arith_stdout.txt`.
- Run: `python3 -I scripts/pred_encounter_a.py tables/pred_numbers_per_tonne.csv
  "../Fiskgrupper i rutan/predators/inputs/pred_scores.csv"
  "../Fiskgrupper i rutan/predators/tables/pred_season.csv" tables` (from this folder).
