# Evidence: the rest water column's benthivore pair - field diets, access, clearance tests (2026-10-31)

Tags: [F] stated in a source, [D] computed here from data, [J] judgement. Scripts in `scripts/` (run with
`python3 -I` from this folder), stdout in `tables/`. Literature on layers (saithe height above the bed,
juvenile gadoid layer, new diet sources) in `evidence_literature.md` (literature agent), downloads in
`data/lit/`. No model output is read anywhere; the budget gate's densities are the library's start values.

## 1. The pair as built (sections 220, 223.26, 228, 223.32)

- a 0.0651 km2 per rest tonne per tick = the section 220 encounter, "layer" variant
  (`../Rovfiskens uppdelning/tables/split_encounter_mack_field_stdout.txt`, G3_rest_wc) [D, reproduced
  there]: every stage with a capability score >= 0.1 counts in full (saithe adult s 0.20, large 0.30,
  velvet belly 0.10, hake juvenile 0.15 all pass); member shares of the a saithe 0.77, spurdog 0.10, hake
  0.07, velvet belly 0.04, anglerfish 0.01.
- Geometry (`../Attackrater ur mötesteori/scripts/pred_encounter_a.py`, `benth_fish`): a3d(R, v, u 0.1,
  H_juv 2 m, N/t, p 0.2, scan 0.5, prey share 1), R = 0.25 m by day / 0.2 m at night (elasmobranchs 0.3);
  i.e. the predator searches INSIDE the 2-m near-bed slab that holds the settled 0-1 group for all of
  its foraging time ("Divide by the prey layer's thickness, with the predator foraging inside it (its eat
  share)", predators_encounter.md 2.6 [F]). Per member tonne: saithe < 60 cm 0.0805, >= 60 cm 0.0316,
  spurdog 0.069, hake adult 0.073, anglerfish 0.027 (ambush, prey speed), velvet belly 0.31.
- Saithe's ben/plk a ratio per tonne is 10.6 [D]; the planktivore a of saithe is ~0.88 herring (dispersed
  in a 70-m day / 40-m night layer), the rest pout (15 m above the bed) and blue whiting (50 m).
- **No size factor inside the a**: prey share 1, no gape term (the score only gates membership at 0.1).
  The stage access therefore does not repeat a size restriction of the a [D]. The whiting layer pair uses
  the same R, v, p with H_whiting 10 m.
- Stage access [0.43, 0.027] = sum over members of (share of a) x (share of the stage's biomass inside
  the member's window), windows p95 prey/predator length saithe 0.30, spurdog 0.34, hake 0.50, anglerfish
  0.57, velvet belly 11 cm, averaged over each member's box length mix BY BIOMASS
  (`../Årsungarnas andel/scripts/stage_access.py`) [F library].

## 2. Access bookkeeping (`scripts/stage_access_check.py`, `tables/stage_access_check_stdout.txt`) [D]

| variant | f_J | f_A | f_eff at J/B 0.15 / 0.08 / 0.05 |
|---|---|---|---|
| library method (reproduced) | 0.431 | 0.0272 | 0.088 / 0.059 / 0.047 |
| species by stage biomass (bw) | 0.446 | 0.0289 | 0.092 |
| bw + normalised over the whole group (bw+all) | 0.431 | 0.0163 | 0.079 / 0.050 / 0.037 |
| bw+all + window weighted by each length's share of the a (aw) | **0.312** | **0.0070** | **0.053** / 0.031 / 0.022 |

- bw+all: the library normalised each member's window share over its diet-identity species only
  (saithe: haddock + cod), but the engine multiplies f_A by ALL adult benthivore tonnes (30 % flatfish,
  rays, gurnard): f_A 0.027 -> 0.016.
- aw: a per member tonne ~ N/t x v ~ L^-3 x L = L^-2 (R fixed in the dark at the bed; anglerfish ~L^-3);
  the per-stage a values confirm it (saithe 43 vs 70 cm: 0.0805 / 0.0316 = 2.55, (70/43)^2 = 2.65). The
  pair's take is sum_L w_L a(L) f(L) B, so the window belongs averaged with weights w_L a(L), not w_L.
  Small saithe (40-55 cm) carry ~80 % of saithe's a and have 12-16 cm windows. f_J 0.43 -> 0.31,
  f_A 0.027 -> 0.007 (hake 0.20 -> 0.04, anglerfish 0.57 -> 0.20).
- The same weighting applies to the rest's planktivore pair ([0.69, 0.27] -> [0.55, 0.13], f_eff 0.43 ->
  0.29) and whiting pair ([0.57, 0.15] -> [0.42, 0.08], 0.22 -> 0.14): ~common mode, so relative to the
  planktivores the benthivore pair moves only x0.89.
- Group J composition by mass: haddock 21.4, cod 4.4, flatfish ~0 (of 100 group units) [D]: the
  juveniles the window acts on are haddock.

## 3. Field diets over the five checked prey (`scripts/field_diets.py`, `tables/field_diets_stdout.txt`)

Saithe = 0.83 of the group's consumption (biomass x Q/B) [D]. Shares of plk / zoo / ben / wht / mack:

| diet | plk | zoo | ben | wht | mack | basis |
|---|---|---|---|---|---|---|
| saithe CRR 219 (1991, NW North Sea), box length mix | 0.729 | 0.184 | 0.064 | 0.020 | 0.003 | [D] from ICES CRR 219 Tables 8-E-1..4 (Q4 read here, PDF p. 129) |
| saithe CRR 164 (1980-82, N North Sea) | 0.481 | 0.454 | 0.062 | 0.002 | 0.000 | [D] from CRR 164 Table 8-E |
| saithe SMS observed (1981, 86, 87, 91) | ben / (pout + herring) = 0.080 | | | | | [F] saithe note entry 3 |
| group V1 booked M2 x B | 0.532 | 0.328 | 0.082 | 0.046 | 0.012 | library |
| group V2 (CRR 164 saithe + members) | 0.507 | 0.389 | 0.066 | 0.026 | 0.013 | [D] |
| group V3 (CRR 219 saithe + members) | 0.715 | 0.162 | 0.067 | 0.040 | 0.015 | [D] |

Members (`tables/rest_diet_ices_stdout.txt`, ICES stomach DB, mass pooled):
- hake, Skagerrak 2000+: 115 stomachs (111 fed), 811 g: plk 81.2, whiting 13.0, ben 0.4 % [D].
- anglerfish, Skagerrak 2000+: 8 stomachs, 97 g: ben 63 % [D] (n tiny); Ecopath 1991: ben 19 % [F].
- spurdog: Ecopath 1991 ben 6.7 % (mostly plaice) [F]; Skagerrak DB 1.8 g of prey (dropped).
- saithe: no in-area diet by mass exists. The DB holds 408 northern North Sea saithe stomachs with 21 g
  of weighed prey (1981/1991 uploads are counts) [D]. All saithe weight tables are 1980-91, NW/N North Sea.

Saithe by quarter (CRR 219, box-weighted): ben share Q1 0.000, Q2 0.024, Q3 0.143, Q4 0.060 [D].

## 4. Clearance-ratio test (`scripts/clearance_test.py`, `tables/clearance_test_stdout.txt`) [D]

Model / field of c_ben / c_anchor (1 = consistent):

| model | vs plk: V1 / V2 / V3 / S219 / SMS | vs zoo: V1 / V2 / V3 / S219 |
|---|---|---|
| library, J/B 0.15 | 7.0 / 8.4 / 11.5 / 12.3 / 13.5 | 14.8 / 22.0 / 9.0 / 10.7 |
| library, trained J/B 0.05 | 3.8 / 4.5 / 6.2 / 6.7 / 7.3 | 8.0 / 11.9 / 4.8 / 5.8 |
| aw access | 6.2 / 7.4 / 10.2 / 10.9 / 12.0 | 8.9 / 13.2 / 5.4 / 6.4 |

The two anchors agree with each other within the field spread (zoo vs plk model/field 0.4-1.9) [D].
The whiting pair is 1.7-3.1x high against plk (V1-V3), mackerel 1.1-1.9x.

Per member (`scripts/member_test.py`, `tables/member_test_stdout.txt`), model/field of c_ben/c_plk (aw):
saithe 8.2 (CRR 219) / 5.6 (CRR 164) / 9.0 (SMS); spurdog 8.5 (Ecopath); hake 269 (Skagerrak 2000+);
anglerfish 17 (Ecopath; its planktivore a meets only pout and blue whiting, the saithe note's point).
Saithe's whiting/plk ratio is also high (3.3 / 18.6 / 6.7) while its ben/whiting ratio is near field
(0.25-2.5): both of saithe's near-bed pairs are inflated against the pelagic anchors by a similar factor.

Quarterly (saithe alone, c_ben/c_plk): model (library) Q1-Q4 1.16 / 0.97 / 2.13 / 1.93, (aw) 0.85 / 0.70 /
3.04 / 1.71; field CRR 219 0 / 0.13 / 0.40 / 0.14. Model/field Q3 5-8, Q2 5-8, Q4 12-13, Q1 infinite
(1991; 1981 winter has 3-10 % haddock, saithe note).

Concurrent-search diet shares (all prey visible): library ben 0.42 (budget gate 0.43), trained J/B 0.05
0.28, aw 0.37; field 0.062-0.082. With every prey at its visibility floor (herring pair 0.25, zooplankton
0.2, benthivores 0.55, whiting 0.5, mackerel 0.7) the library ben share is 0.60: hiding moves c_ben/c_plk
x2.2 against the planktivores.

Single-pair saturation x/(1+x) at today's density (the time-allocation check of the whiting audit):
library ben 0.57, plk 0.66 (hidden 0.33), zoo 0.26, whiting 0.17, mackerel 0.05. Unlike whiting's
epifauna pair (0.97 vs 0.44-0.76), the benthivore pair does not out-pay the herring pair at full
visibility; it does when the herring hide (x1.7).

Factor on a x f_eff (at fixed other pairs) that reproduces the field share under concurrent search:
library 0.09-0.12; at trained J/B 0.05 0.17-0.23; aw 0.11-0.15 [D].

## 5. In-layer factors (section 220 MC)

The pair's MC 5-95 % (rest group, layer variant) is 0.021-0.20 around 0.065
(`../Rovfiskens uppdelning/tables/split_encounter_mack_field_stdout.txt`): the 5 % end is x0.32. The
field needs x0.09-0.15 on a x f_eff: below the in-layer range, so no combination of R, p, speed or
H_juv inside their section 220 ranges reaches the field without a structural factor [D].

## 6. Overlap candidate (`scripts/proposal.py`, `tables/proposal_stdout.txt`) [D on J inputs]

Member shares of foraging time inside the ~2-m juvenile layer, from `evidence_literature.md` (no model
output; but NOT independent of the diets: the saithe foraging share 0.05 was argued partly from CRR
219's Q1 haddock zero, the same series it is tested against here. The layer-only alternative, saithe
biomass within 2 m of the bed 0.05-0.10 from H_saithe ~50 m plus the spawning-aggregation ceiling,
gives a ~0.0055-0.008 (other members as below), inside the low/high bracket):
saithe 0.05 (0.02-0.15), spurdog 0.15 (0.05-0.30), hake 0.25 (0.10-0.50), velvet belly 0.10 (0.05-0.30),
anglerfish 1. Caveat on the strongest-sounding source: Aglen 1996's 30-100 m best-match height holds
for large haddock and cod as well as saithe, so it is a trawl-acoustic calibration height, not a
position; the saithe-specific evidence is the depth tags (Skilbrei & Ottera 2016, spread over ~100 m),
Kaartvedt 2005 and the near-bed-only spawning aggregations (Neilson 2003, DFO 2003/075).

| | overlap factor | a | stage access (aw, new member shares) | f_eff J/B 0.15 | a x f_eff |
|---|---|---|---|---|---|
| library | 1 | 0.0651 | [0.431, 0.027] | 0.088 | 0.0057 |
| central | 0.085 | 0.0055 | [0.437, 0.035] | 0.095 | 0.00053 |
| low | 0.039 | 0.0026 | [0.488, 0.059] | 0.124 | 0.00032 |
| high | 0.20 | 0.013 | [0.375, 0.020] | 0.073 | 0.00096 |

New member shares of the a (central): saithe 0.45, spurdog 0.18, hake 0.21, velvet belly 0.05,
anglerfish 0.12. Concurrent-search benthivore share 0.083 (0.052-0.143; field 0.062-0.082); with
saithe's own overlaps also on its planktivore (0.78), zooplankton (0.8) and whiting (0.25) pairs 0.104
(0.065-0.174). Clearance ratio to planktivores, model/field: 0.95 (V1) / 1.14 (V2) / 1.57 (V3).
Single-pair saturation at today's density falls 0.57 -> 0.11 (0.07-0.18), below the herring pair 0.57.
Not removed by it: hake (field 0.4 % benthivores in the Skagerrak, 2000+) keeps 0.21 of the new a; the
quarterly shape (model Q1 take ~0.3-0.55 of Q3, field 0 in 1991).
