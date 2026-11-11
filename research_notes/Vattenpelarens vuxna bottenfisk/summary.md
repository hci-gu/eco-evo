# The adult part of water column -> benthivores against the members' field diets (2026-11-11)

Research subagent; the main agent saved this text. Nothing changed in the library, code, tests, manifest,
AGENTS.md, resume or existing notes; no training; library copies only in the session scratchpad; budget_gate.py
and field_diet.py were patched in memory only (as research_notes/"Rekryternas bottendjup"). Every value below is
read from field data, lab data or the exec'd derivations; the gate is a check and chose nothing. Sources E1-E19
and model facts M1-M6: `evidence.md`. Grades: [F] field data or full text, [L] lab, [A] abstract only, [S]
secondary / via another note, [D] derived here, [J] judgement.

## Files

Run from `scripts/`; each script's stdout is saved beside it.

- `decompose_members.py` -> `decompose_members_stdout.txt`, `../tables/decompose_members.json`: execs
  research_notes/"Overlapp i fiskparen"/scripts/band_pairs.py (head, unchanged) and re-states band_fish_pair's loop
  with every member kept (reproduces eq exactly for WC -> benthivores, whiting, planktivores, mackerel); member
  a shares, a per member tonne, R_sat, k per fish, time within the slab; execs research_notes/"Fonsterviktning"
  (o-umlaut)/scripts/aw_stage_access.py (head, unchanged) for the member windows (kexp 1.73, as window_hs.py) and
  adds the capture p of the window's own (prey length, predator length) pairs; the adult-prey vertical profile
  factor; an encounter-implied fish diet per member.
- `ices_member_diets.py` -> `ices_member_diets_stdout.txt`: ICES stomach database on disk, hake / anglerfish / ling
  / spurdog / pollack / saithe in the North Sea + Skagerrak, prey mass by J / A (prey length or back-calculated
  from weight).
- `ecopath_bf_items.py` -> `ecopath_bf_items_stdout.txt`: Ecopath 1991 (EcoBase 457) diet items on the model's
  benthivore members.
- `sms_adult_share.py` -> `sms_adult_share_stdout.txt`: SMS 2023 key run, each predator's flow to cod and haddock
  split J / A by length at sea.
- `gate_adult.py` -> `gate_adult_stdout.txt`, `../tables/gates/budget_gate_<run>.txt`, `../tables/gates/gate_adult.json`:
  a copy of research_notes/"Rekryternas bottendjup"/scripts/gate_ben_shape.py; every run carries the combined
  capture-p x schooling factors (central), J on the proposed recruit_depth_table, A on the GOV annual shape; the
  variants change only the WC -> benthivores a and stage_access through member factors. `baseline` reproduces
  that note's 2.00 / 1.90.
- `field_vs_model.py` -> `field_vs_model_stdout.txt`: field flow per member vs the model's implied flow per member.
- `papers/README.md`: Unpaywall lookups only; no open PDF for the ling / hake sources.

```
R=<repo>/research_notes; S=<session scratchpad>
python3 -I decompose_members.py <repo> $S/tmp ../tables/decompose_members.json                    # ~20 s
python3 -I ices_member_diets.py "$R/Torskfiskarnas bytesandelar/data/diet/ices_stomach"            # a-umlaut in the real name
python3 -I ecopath_bf_items.py "$R/Fiskgrupper i rutan/benthivores/data/ecobase/model_457.xml"
python3 -I sms_adult_share.py "$R/Fiskgrupper i rutan/predators/data/population"
python3 gate_adult.py <repo> "$R/Rekryternas bottendjup/tables/gov_juv_bed.json" ../tables/decompose_members.json $S/libs ../tables/gates
python3 -I field_vs_model.py ../tables/decompose_members.json ../tables/gates/gate_adult.json
```

## Question

Under the combined capture-p x schooling derivation with the proposed recruit bed-depth shape, budget_gate puts
water column (WC) -> benthivores at 2.00 (equal cells) / 1.90 (field spread) of its booking. About a third to a
half of the pair's encounter flow is on ADULT benthivores (>= 27 cm haddock / cod, >= 16 cm plaice, all dab and
long rough dab), carried by hake, anglerfish and ling, which hold ~0.10 of the pair's a. Is that adult part
supported by those members' field diets, and if not, which factor in their a makes it large?

## Model (M1-M6)

- One pair a for the group (0.0418; x2.96 combined = 0.1238), built as the sum over member stages of an encounter
  rate (section 220 terms on the section 269 band + rule A); every benthivore stage is met as prey in a uniform 2 m
  bed slab at 0.1 m/s; the stage window [f_J, f_A] then splits the encounter by stage. Windows: p95 prey/predator
  length over each member's box length mix, a(L)-weighted, normalised over all benthivore tonnes; ling has no
  window and takes anglerfish's [J].
- Membership is a 0/1 switch (capability score >= 0.1). Hake's benth_fish score 0.55 rests on "gadoids 16-48 % of
  the Skagerrak/Kattegat 2024 hake diet (whiting the main one)"; whiting is a separate prey group with its own pair.
- Hake and ling are "midambush" members cruising at v_hake 0.5 BL/s day and night ("J: no measurement"); ling's
  profile exp 5 m above the bed day and night ("[J] no field profile"); anglerfish a bed ambusher (strike 0.4 m,
  prey 0.1 m/s, [J]).
- Combined capture p: one pair value 0.592 at PPR 0.25 (bluefish line, q 0.8, r 0.3) on both stages.

## Findings

### 1. Decomposition at the combined central [D] (`decompose_members_stdout.txt`)

| member | biomass share | share of the a | a per tonne / group mean | time in the 2 m slab, day / night | R_sat m | window [f_J, f_A] | share of J flow | share of A flow | mean p inside, J / A | A-profile factor |
|---|---|---|---|---|---|---|---|---|---|---|
| saithe | 0.797 | 0.892 | 1.12 | 0.125 / 0.161 | 0.86 (43 cm), 1.40 (70 cm) | [0.354, 0.0008] | 0.796 | 0.018 | 0.61 / 0.56 | x0.89-0.91 |
| hake | 0.048 | 0.055 | 1.13 | 0.221 / 0.029 | 0.73 | [0.664, 0.284] | 0.092 | 0.382 | 0.53 / 0.50 | x0.84 |
| ling | 0.014 | 0.025 | 1.84 | 0.33 / 0.33 | 1.26 | [0.970, 0.587] (anglerfish's) | 0.061 | 0.359 | 0.56 / 0.48 | x0.79 |
| anglerfish | 0.027 | 0.015 | 0.57 | 1.0 / 1.0 | 0.40 (strike, fixed) | [0.970, 0.587] | 0.037 | 0.218 | 0.56 / 0.48 | x0.74 |
| spurdog | 0.097 | 0.004 | 0.04 | 0.074 / 0.009 (Nov-May), 0 (summer) | 0.30 (non-visual) | [0.759, 0.259] | 0.007 | 0.024 | 0.60 / 0.59 | x1.02 |
| pollack | 0.008 | 0.007 | 0.86 | 0.125 / 0.161 | 1.11 | saithe's | 0.006 | 0.000 | - | x0.90 |
| velvet belly | 0.009 | 0.002 | 0.22 | 0.032 / 0.032 | 0.30 (non-visual) | [0.044, 0.003] | 0.000 | 0.000 | - | x1.01 |

Capture p from the bluefish line within its fitted PPR range (0.16-0.62); every window's p95 is <= 0.57.

- The member-weighted window reproduces the library ([0.396, 0.0408] vs [0.392, 0.0388]); J share of the
  encounter flow 0.64 at uniform prey, 0.57 at the field spread with the recruit shape.
- Hake + ling + anglerfish carry 0.96 of the A flow with 0.095 of the a; hake + ling alone 0.74.
- Why: the windows (p95 PPR 0.50-0.57 over predators of 36-63 cm mean, up to 150 cm ling) take in 0.28-0.59 of
  the adult tonnes, and these members spend 0.22 (hake, by day) to 1.0 (anglerfish) of their time in the 2 m slab
  (saithe 0.125). Ling's a per tonne (1.84 x the group mean) comes from that slab time (0.33 day and night, exp 5 m
  [J]), R_sat 2 L = 1.26 m and continuous 0.5 BL/s search [J]; spurdog's near-zero a from its profile (0.074 of
  its non-summer day in the slab, none in summer) and the non-visual 0.3 m.
- Member encounter fish diets (each member as its own predator; krill and zooplankton out on both sides): hake
  benthivores J 0.12 / A 0.27, planktivores 0.48, whiting 0.13 (field fish diet: planktivores 0.78, whiting 0.21,
  benthivores 0.01, all juveniles); anglerfish A 0.60, J 0.21, planktivores 0.11 (field: benthivores ~0.21, A
  ~0.19, planktivores 0.39-0.49); ling A 0.37, J 0.13 (field: see 2).

### 2. Field diets [F/S/A] (E1-E6, E11-E19)

| member | benthivores in the diet by mass | adult part | basis |
|---|---|---|---|
| hake | 0.01 (0-0.03) | ~0: no adult benthivore in 284 North Sea + Skagerrak stomachs; the items are 24-25 cm haddock, a 12 cm cod, 13 cm plaice | E1 ICES 2022-24 [F data]; E2 Ecopath 0 [F]; E4 SMS: hake eats no cod or haddock [F]; E3 Bergstad 1991 blue whiting 95-98 % [S]; E16 Biscay/Celtic 405 stomachs: no haddock, cod or pleuronectids, prey 3-24 cm [F] |
| anglerfish | 0.19 (0.08-0.34) | ~0.85 (0.5-0.95) of it (32 cm haddock, 17-22 cm plaice and flatfish; Ecopath haddock adult 0.066 + flatfish 0.094) | E1 n 34 [F data]; E2 0.186 [F]; E6 Iceland ~21 %, Shetland 8 %, Irish Sea > 60 cm 8 % [S]; E11 Skagerrak 8 fed: flatfish/ray/gurnard 0.59 [F data] |
| ling | 0.10 (0.03-0.20) [J] | unknown; in-area items are a 6 cm cod and one 17 cm flatfish | E12 Iceland haddock + cod ~15 % [F figure]; E1/E11 in-area 0-6 % on ~140 g of prey (8-25 fed fish, mostly < 40 cm) [F data, anecdotal]; E13 Bergstad: cod juveniles and adults named [S] |
| spurdog | 0.06 (0.02-0.15) | ~0.85 (plaice, dab, witch) | E2 Ecopath 0.068 [F]; E1 dab 10-15 cm [F data]; E6 [S] |
| saithe | 0.05 (0.02-0.09) | 0-0.17: stomachs "almost all 0-1 group haddock"; SMS puts 0.17 of saithe's haddock flow on >= 27 cm (age 2, 25-30 cm) | E4, E5 |

Ration Q/B (E3): saithe 3.0, hake 3.5, anglerfish 1.8, spurdog 2.0; ling not found anywhere (2.0, 1.5-3.0 [J]).
Partial M2 (E4): SMS 2023 has no hake M2 on cod or haddock; anglerfish, ling and spurdog are not SMS predators.
Ling behaviour (E14 [A], E15 [F, weak]): crepuscular activity peaks, small core area, visual, takes mobile prey;
not documented as a crevice ambusher. Hake (E17 [A/S], E18 [A]): near the bed by day, off-bottom at night,
juvenile feeding at night with fullness peaking at dawn; 80 % empty stomachs in daylight bottom hauls (E16).

### 3. Field flow vs the model's implied flow per member [D] (`field_vs_model_stdout.txt`)

Model: baseline gate (combined + recruit shape), field spread 1.90 x booked 0.1274 = 0.243 t/km2/yr, J 0.138 /
A 0.105 (M2 on J 0.58, on A 0.081 /yr). Field: B x Q/B x benthivore share x J/A split (ranges multiply the ends).

| member | B t/km2 | field ration t/km2/yr | model J | model A | model A / member's whole field ration | field J | field A | model A / field A |
|---|---|---|---|---|---|---|---|---|
| saithe | 0.492 | 1.475 | 0.110 | 0.002 | 0.00 | 0.068 (0.023-0.159) | 0.006 (0-0.027) | 0.3 |
| spurdog | 0.060 | 0.120 | 0.001 | 0.003 | 0.02 | 0.001 | 0.006 (0.001-0.022) | 0.4 |
| hake | 0.030 | 0.104 | 0.013 | 0.040 | 0.38 | 0.001 (0-0.005) | 0.000 (0-0.0005) | > 80 |
| anglerfish | 0.017 | 0.030 | 0.005 | 0.023 | 0.76 | 0.001 | 0.005 (0.001-0.012) | 4.7 |
| ling | 0.009 | 0.017 ([J] Q/B 2.0) | 0.008 | 0.038 | 2.17 (1.4-2.9 at Q/B 1.5-3.0) | 0.001 | 0.001 (0.0001-0.005) | ~40 |
| total | | | 0.138 | 0.105 | | 0.072 (0.024-0.18) | 0.018 (0.002-0.067) | 5.9 (1.6-49) |

- The field A share of the field flow is 0.20; the model's 0.43. The model's A is 5.9x the field A (1.6-49).
- Wrong member as well as too large: the field A is carried by spurdog (0.006), saithe (0.006 by SMS, ~0 by
  stomachs) and anglerfish (0.005); the model puts 0.74 of its A on hake and ling, whose field A is ~0 and
  ~0.001. The model's hake J (0.013) is also ~10x hake's field J.
- Ration check, diet-independent: the model credits ling with 2.2x (1.4-2.9 over a [J] Q/B of 1.5-3.0; no ling
  Q/B was found) its WHOLE field ration on adult benthivores alone, hake with 0.38 and anglerfish with 0.76 of
  theirs. No field diet share can make ling's credited flow consistent; at the field share its a on this pair is
  ~40x too high.

### 4. Which factor makes the adult part large

| factor | effect on the A flow | evidence | grade |
|---|---|---|---|
| hake's membership: score 0.55 counted whiting (another group) as benthivore prey | hake 0.38 of A, 0.09 of J; field hake A 0, J ~0.01 of its diet | 284 in-area stomachs, Ecopath, SMS, Bergstad, Biscay: benthivores 0-0.03, no adult | [F] for the diet; [J] on reading the score from it |
| one pair p 0.592 (PPR 0.25) applied to adult encounters at PPR ~0.4-0.5 | x0.83 on A (J x1.00) | the bluefish line inside each window | [L analog / D] |
| adults placed in the juveniles' 2 m bed slab | x0.74 (anglerfish), x0.79 (ling), x0.84 (hake), x0.89 (saithe) on A | adult bottom-2-m share 0.71 day / 0.76 night (Hobson 2007, Aglen 1997 via "Bottenfiskens hojdprofil"); off-bed part uniform 2-20 m [J] | [S/J] |
| ling: exp 5 m profile, 0.5 BL/s cruising day and night, anglerfish's window, score 0.6 | ling 0.36 of A at 1.84 x the group a per tonne; 2.2x its ration | every input is [J]; tracking says crepuscular, core-area, visual (E14 [A]); no ling PPR, Q/B or in-area adult item | [J]; not correctable from evidence today |
| anglerfish: its pelagic a tiny (planktivores 0.06, whiting 0.14 of the group mean per tonne) against field P 0.35-0.44, W 0.21-0.25 | its encounter diet is 0.81 benthivores vs field ~0.2 | the 2 m slab concentrates its benthic prey, pout / blue-whiting layers dilute the pelagic ones | [D]; flagged, no correction |
| spurdog (mirror error): field A ~0.006 t/km2/yr, model 0.0025 | the field's largest adult eater gets ~0.004 of the a | its profile (bed only over beds < 182 m, Nov-May; pelagic in summer) and non-visual R | open |
| 2 L cap, rule A R | not separable: R_sat = 2 L binds for every member on 15 cm prey; it sets the per-tonne a but the window and slab time decide the stage | - | - |

Not causes: saithe's profile (Hs stands, sibling note) and saithe's window (0.0008 of A; SMS age-2 haddock would
ADD ~0.006 t/km2/yr of A on saithe, a "right flow, wrong member" item, not an overshoot).

### 5. Gate [D] (`gate_adult_stdout.txt`)

All runs: combined factors (central), J on the recruit_depth_table, A on the GOV annual shape. Columns: WC ->
benthivores implied / booked M2, equal cells | field spread | predator uniform with the prey shaped (P2); the
benthivores' prey-side total equal / field; the J and A flows at the field spread (t/km2/yr) against the field
J 0.072 and A 0.018.

| run | WC -> ben a, stage_access | eq / field / P2 | ben total eq / field | J / A flow (x field) |
|---|---|---|---|---|
| baseline | 0.1238, [0.392, 0.0388] | 2.00 / 1.90 / 2.66 | 1.06 / 1.05 | 0.138 / 0.105 (1.9 / 5.9) |
| stage p | 0.1238, [0.393, 0.0323] | 1.91 / 1.79 / 2.56 | 1.02 / 1.01 | |
| stage p + A profile | 0.1238, [0.393, 0.0260] | 1.81 / 1.68 / 2.45 | 0.99 / 0.97 | 0.144 / 0.070 (2.0 / 4.0) |
| hake out (alone) | 0.117, [0.377, 0.0254] | 1.67 / 1.55 / 2.26 | 0.94 / 0.92 | |
| hake A only 0 (alone) | 0.1238, [0.392, 0.0240] | 1.77 / 1.64 / 2.41 | 0.98 / 0.95 | |
| **central: p + A profile + hake out** | **0.117, [0.382, 0.0165]** | **1.54 / 1.41 / 2.13** | **0.89 / 0.87** | 0.136 / 0.043 (1.9 / 2.4) |
| bracket: + ling own length mix at p95 0.40 [J] | 0.117, [0.382, 0.0151] | 1.52 / 1.38 / 2.10 | 0.89 / 0.86 | 0.137 / 0.040 (1.9 / 2.2) |
| bracket: + ling out | 0.114, [0.367, 0.0072] | 1.32 / 1.17 / 1.86 | 0.81 / 0.79 | 0.131 / 0.019 (1.8 / 1.1) |
| bracket: no bed member on A | 0.1238, [0.392, 0.0007] | 1.38 / 1.21 / 1.98 | 0.84 / 0.80 | 0.152 / 0.002 (2.1 / 0.1) |

- Other prey barely move (equal / field): planktivores 0.89 / 0.86 -> 0.90 / 0.88, whiting 1.03 / 0.89 -> 1.05 /
  0.90, krill 1.01 / 1.02, epifauna, Nephrops, infauna unchanged. Whiting -> benthivores 0.29 / 0.34, seals ->
  benthivores 0.90, porpoises 2.44 unchanged.
- OFF pairs unchanged in every run: WC -> mackerel (prey), seals -> WC (prey), whiting -> benthivores (equal cells
  and prey side), whiting -> mobile_epifauna (field). No pair becomes OFF.
- The hake re-read is pair-specific: hake stays in WC -> whiting (whiting 0.21 of its Skagerrak diet) and WC ->
  planktivores. The member factors were applied to this pair only; hake's a share in WC -> whiting is 0.047 and
  its windows there are gape-only, not checked here.

## If applied (not applied; the user's call)

- `predatory_fish_preys_on_benthivorous_fish`: max_intake_rate 0.0418 x 0.945 = 0.0395 before the combined factor
  (0.117 after x2.96, as gated); stage_access [0.382, 0.0165] (relative to the section 269 central [0.392, 0.0388]).
  The comment must say that f_A now carries two encounter multipliers on top of the gape window: the member /
  stage capture p relative to the pair p (A x0.83) and the adult bottom-2-m share (0.71 day / 0.76 night, x0.74-0.89
  by member), so the 232.2 / 243 meaning of the window (share of tonnes inside the gape) is no longer the whole
  value.
- Hake: a pred_scores.csv basis correction for the benth_fish pool only (whiting is the whiting pool); hake's
  benth_fish score below 0.1 removes it from this pair's band a and window. Hake's a shares in WC -> whiting
  (0.047) and WC -> planktivores (0.046) are untouched.
- Ling: no change from evidence; brackets as in section 5.

## Gaps

1. Ling: no in-area adult-benthivore item, no prey/predator length ratio, no Q/B, behaviour from one tracking
   abstract and one weak tagging paper; its profile, speed, window and score are all judgements. The ration check
   bounds it (a at least x1.4-2.9 too high over a [J] Q/B of 1.5-3.0) but gives no value.
2. Anglerfish's and spurdog's pelagic and bed profiles are judgements; their residuals are flagged, not fixed.
3. Every in-area diet besides hake rests on 8-35 stomachs per species; anglerfish's adult share leans on one
   haddock and Ecopath 1991. Prey lengths of half the items are back-calculated from weight (lower bounds).
4. The A-prey off-bed part (uniform 2-20 m) is a judgement; only the bottom-2-m shares are measured (S).
5. SMS's 0.17 adult share of saithe's haddock flow depends on the 27 cm cut against age-2 haddock at 25-30 cm.
6. The member diets compare encounter shares over fish prey only; krill would lower every member's fish shares
   alike.
7. The residual after the A corrections is the J part (1.8-2.0x the field J): saithe on juveniles, the haul-level
   co-occurrence question of research_notes/"Sejens hojd over botten"; outside this note.

## Verdict

- The adult part is NOT supported by the members' field diets [F/D; high for hake, moderate for anglerfish and
  spurdog, low for ling]: at the field spread it is 0.43 of the pair's flow and ~6x (1.6-49) the field adult take
  (~0.018 t/km2/yr, 0.20 of the field flow), and 0.74 of it sits on hake and ling, whose field adult take is ~0
  and ~0.001, while the field's adult eaters (spurdog, anglerfish, saithe by SMS) carry little of it in the model.
- Evidence-supported factors:
  (1) hake out of WC -> benthivores [F diet; J on the score re-read]: its score 0.55 counted whiting; on the
      benthivore diet (0-0.03, never an adult, 284 in-area stomachs; SMS and Ecopath 0) it falls below the 0.1
      membership. Removing hake beats setting only its f_A to 0 (its J take is ~10x the field too).
  (2) capture p per member and stage inside the windows [L analog / D]: x0.83 on A (PPR 0.4-0.5 instead of 0.25).
  (3) adults off the juveniles' 2 m slab [S/J]: bottom-2-m share 0.71 / 0.76; x0.74-0.89 on A by member.
  Together: stage_access [0.382, 0.0165], a x0.945 (0.117 after the combined factor); gate 1.54 (equal) / 1.41
  (field spread) / 2.13 (P2); benthivore prey total 0.89 / 0.87; the adult flow 2.4x the field instead of 5.9x.
- Ling is the remaining adult excess [J]: its a on this pair is at least x1.4-2.9 too high whatever it eats (x2.2 at
  a [J] Q/B 2.0; ~x40 at the field diet), but every input is a judgement and no field value replaces one. Brackets: ling on its own length
  mix at p95 0.40, 1.38 field spread; ling out, 1.17 (A then on the field A, 1.1x). Not chosen here: ling out
  would follow the gate, not evidence.
- What remains after (1)-(3) at the field spread (1.4) is mostly the juvenile part at ~1.9x the field J; the booking
  (0.0833) is already above the field flow (field J + A 0.089, 0.026-0.25, vs booked 0.127).

## User decisions

- 2026-11-11: all three corrections into the combined section: hake out of water
  column -> benthivores (kept in the whiting and planktivore pairs), capture p per
  member and stage, adults off the juveniles' 2 m slab: stage_access [0.382,
  0.0165], a x0.945. Ling flagged open (every input a judgement). Investigate the
  haul-level co-occurrence of saithe and juveniles and a re-booking of the pair
  against the field flow; judge the rest with the long-run probe after training.
