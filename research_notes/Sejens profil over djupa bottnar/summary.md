# Saithe's profile over deep beds, and what carries the water column's shallow-bed gradient (2026-11-18)

Research subagent; the main agent saved this text. Nothing in the library, code, tests, manifest, AGENTS.md,
resume or existing notes was changed; no training; the library was read only; scratch files in the session
scratchpad. No value was set from or fitted to model output. Sources E1-E17 and model facts M1-M4: `evidence.md`.
Grades: [F] field data or full text, [L] lab, [A] abstract only, [S] secondary / via another note or a summarising
fetch, [D] derived here, [J] judgement.

## Files

Run from `scripts/`; stdout in `tables/`. S = session scratchpad, R = <repo>/research_notes.

    python3 -I class_terms.py <repo> $S/tmp ../tables/class_terms.json > ../tables/class_terms_stdout.txt     # ~4 min
    python3 class_diet.py <repo> ../tables/class_terms.json > ../tables/class_diet_stdout.txt                # ~1 min
    python3 -I light_at_bed.py <repo> $S/tmp > ../tables/light_at_bed_stdout.txt
    python3 -I prey_layer_time.py <repo> $S/tmp > ../tables/prey_layer_time_stdout.txt
    D="$R/Ekolodsdata planktivorer/data/derived"
    python3 -I heras_deep_windows.py "$D/biotic_region_hauls.tsv" "$D/biotic_region_catch.tsv" \
        "$D/biotic_region_individuals.tsv" "$R/Krill i Skagerrak/wc_depth/tables/heras_haul_meta.tsv" > ../tables/heras_deep_windows_stdout.txt
    python3 -I saithe_stomach_bed.py "$R/Krill över grunda bottnar/stomachs/tables/records_linked.csv.gz" \
        "$R/Födonivå och mättnad i fält/data/A_ices_stomach/all" ../tables/saithe_stomach_bed.csv > ../tables/saithe_stomach_bed_stdout.txt   # ~15 min
    python3 -I field_vs_model.py ../tables/saithe_stomach_bed.csv ../tables/class_terms.json ../tables/class_diet.json > ../tables/field_vs_model_stdout.txt
    python3 -I condition_bed.py "$R/Krill över grunda bottnar/stomachs/tables/records_linked.csv.gz" "$R/Födonivå och mättnad i fält/data/A_ices_stomach/all" > ../tables/condition_bed_stdout.txt

- class_terms.py: research_notes/"Overlapp i fiskparen"/scripts/band_pairs.py exec'd unchanged (as
  "Sejens hojd over botten"/band_hs.py does). It gives the per-class a of WC -> krill, planktivores (and its
  herring / pout / blue whiting terms), benthivores and whiting under saithe-profile and vision variants, and
  reproduces the library tables exactly (M1).
- class_diet.py: budget_gate's env (16 x 16, library read only); per-class diet and saturation of each variant,
  with no policy.
- heras_deep_windows.py (E1), saithe_stomach_bed.py (E9), field_vs_model.py (E11), light_at_bed.py (E13),
  prey_layer_time.py (E16), condition_bed.py (E17).
- papers/: Stensholt et al. 2001 PDF (ICES CM; check the licence before committing). Nothing else is stored there.

## Question

The water column's per-cell depth_access_table favours shallow beds on every pair (M1), and the trained policy
stands mostly over beds < 150 m, while GOV puts box saithe 0.78 over 150-250 m beds. Is the shallow-bed gradient
supported by field evidence? If not, which term of its derivation carries it?

## Model facts

- Class shapes, equal-cell mean 1, beds < 100 / 100-150 / 150-250 / >= 250 m (section 269 A_band, reproduced
  exactly) (M1):
  - krill 2.62 / 1.21 / 0.22 / 0.006
  - planktivores 2.21 / 1.21 / 0.41 / 0.16
  - benthivores 1.66 / 1.36 / 0.56 / 0.31
  - whiting 1.67 / 1.38 / 0.55 / 0.27
- Positions (M2):
  - Saithe searches exp(-h/15 m) above the bed by day.
  - The krill day layer is 50-90 m above 200-250 m beds.
  - Herring are in the top 100 m by day.
  - Pout, blue whiting and juvenile benthivores are near the bed.
  - Pool shares are fixed in every cell: herring 0.554, pout 0.35.
- Rule A (M3): Ke 0.02 lx and R_nv 0.3 m on fish, both judged. Light below ~65 m is the judged K_deep 0.06.
- Saturation (M4): at start densities the WC is at 0.88-0.98 of its Holling ceiling over beds < 250 m. Field Q/B
  2.83 is 0.33-0.42 of that ceiling.

## Findings

### 1. Saithe's vertical position over deep beds

HERAS pelagic hauls, June-July (E1) [F data, D]:
- Over 150-250 m beds by day, adults were in 6 of 8 hauls fished within 30 m of the bed (595 kg/h at 3-30 m above
  ~171 m beds). Blue whiting, pout and (in 2 of 4 hauls) euphausiids were caught in the same windows.
- 0 of 63 surface hauls held adults by day over beds >= 150 m.
- The 60-160 m layer over deep beds was fished only once: a gap, not an absence.
- At night 11 of 82 surface hauls held 70-106 cm adults: 3 of 26 over 150-250 m beds and 8 of 56 over > 250 m.

Stensholt, Aglen, Mehl & Stensholt 2001, ICES CM 2001/Q:24 (E2) [F]:
- Norwegian coast: "For depth less than 200m the saithe stay close to seabed". About 60-75 % of saithe are within
  0.1 (bed + 10) m and ~95 % within 0.2. That is Hs ~13-20 m at a 175 m bed [D].
- Over beds 300-600 m they "spread relatively higher" (Hs ~22-61 m at a 300 m bed) [D].
- North Sea: 69-96 % of the echo is within 10 m of the bed.
- Day/night difference "not clear". Survey Oct-Nov, acoustic dead zone excluded.

Other sources (E3-E7):
- Armannsson & Jonsson 2012 (Iceland, data storage tags, depth only): the daily vertical range rises about 1:1
  with depth; saithe feed by day [F].
- Kaartvedt et al. 2005: saithe in midwater beneath krill swarms over very deep water, eating fish [F].
- Nedreaas 1997: saithe dive at ship noise, which biases acoustics and trawls towards the bed [F].
- Skilbrei & Ottera 2016: farm-associated fish use about 100 m of the column; nothing is referenced to the bed [S].
- ICES FishMap: adults are far up at night [F].

Search time inside each prey's layer over 150-250 m beds by day (E16) [D]:

| profile | juvenile 2 m slab | pout layer | blue whiting layer | krill layer | herring layer |
|---|---|---|---|---|---|
| exp 15 m (library) | 0.13 | 0.87 | 0.86 | 0.15 | ~0 (only the herring profile's ~5 % bottom share) |
| depth-scaled (Stensholt) | 0.10 | 0.78 | not tabulated | 0.19 | not tabulated |

Reading:
- By day over 150-250 m beds the bed-anchored exp 15 m profile is supported, and the North Sea acoustics place
  saithe even nearer the bed.
- Over > 250-300 m beds the spread grows with bed depth (the depth-scaled variant, Hs = 15 x (bed + 10)/150).
- At night some large adults rise; the library's night share near the surface (0.07) has this in kind.
- Confidence: moderate for summer and autumn days; low for winter and night, which have no bed-referenced data.

### 2. Field test by bed class (E9-E11, E17)

Adult saithe, ICES stomach database, 1980-87 records at rectangle resolution. These records carry no prey weights,
so the measure is FO (share of fed records with the prey). NE North Sea + Skagerrak, 100-150 m against 150-250 m
beds:

| prey | Q3 | all quarters |
|---|---|---|
| krill | 0.72 -> 0.79 | 0.65 -> 0.76 |
| pout | 0.37 -> 0.29 | 0.42 -> 0.41 |
| identified juvenile gadoids | 0.20 -> 0.08 | 0.19 -> 0.18 |

Over > 250 m beds: krill 0.73-0.79, pout 0.32-0.44.

Occurrence ratio 150-250 / 100-150 m, field against model. Densities are GOV Q3 for fish; for krill they are the
library depth shape.

| prey | field | model, saturated (start densities) | model, a-limited (bracket) |
|---|---|---|---|
| krill | 1.23-1.44 | 0.79 | 0.23 |
| pout | 0.74-1.08 | 0.73 | 0.21 |
| juvenile gadoids | 0.40-0.93 | 0.13 | 0.04 |

Beyond 250 m: model 0.07-0.24 saturated, 0.01-0.02 a-limited, against field 0.8-2.6 for krill and pout.

The check that does not depend on saturation, the krill : pout ratio, is 1.3-1.7 in the field and ~1.1 in the
model. It does not discriminate.

Caveats [J]:
- The 1980s records are pooled samples, which compresses FO.
- The capture bed is not the feeding bed, which pulls every field ratio towards 1.
- The rectangle spread is ecological inference.
- The stomachs are from the 1980s and the densities from 2000-25.
- Krill density by class is a model input.

True-position data exist only for Q1 2020s, 5 stations at 150-250 m. Fullness there is 2.2 against 8.1 per mille
over 100-150 m: the model's direction, but thin and in the spawning season.

Condition data: 15 weighed fish, none over 150-250 m (E17).

Reading:
- Between 100-150 and 150-250 m the stomachs cannot test the gradient.
- Beyond 250 m they contradict it by one to two orders.
- Juvenile gadoids disagree at 150-250 m as well (prey-size and dilution caveats).

### 3. Which term carries the gradient (E12-E15)

Decomposition. "Geometry" is a run with every visual member at its dark distance (overlap and member mix only).
"Vision" is central divided by that run.

| pair | geometry G | vision G | total G |
|---|---|---|---|
| benthivores (J and A alike), whiting | 1.00 | 2.65-2.70 | 2.65-2.70 |
| pout term | 1.03 | 2.75 | 2.83 |
| blue whiting term | 1.49 | 2.25 | 3.35 |
| krill | 5.2 | 1.6 | 8.0 (x200 to >= 250 m) |
| herring term | 4.0 | 3.0 | 12.4 |
| planktivore pair | 1.4 | 2.8 | 3.9 |

G = a over < 150 m beds divided by a over 150-250 m beds.

- Bed prey: the gradient is all vision, i.e. rule A at the light at the bed. Daytime g at the bed is
  0.94 / 0.75 / 0.17 / 0.00 by class; 0.006 lx at a 150-250 m bed. That rests on the unmeasured K_deep below 65 m
  (0.0003-0.015 lx for K_deep 0.09-0.05) and on the judged Ke and R_nv.
- Planktivore pair: over 150-250 m beds its a is 0.77 pout. The fixed pool mix is a minor term.
- J and A share the class shape only by engine convention. The adult part rides on bed members with fixed R
  (anglerfish, ling), whose own class shape would be flat.

Profile variants (G; a(100-150)/a(150-250) in brackets):

| variant | krill | benthivores | whiting | planktivores |
|---|---|---|---|---|
| library exp 15 m | 8.0 (5.4) | 2.65 (2.4) | 2.7 (2.5) | 3.9 (3.0) |
| depth-scaled (Stensholt) | 7.7 (4.5) | 4.2 (3.3) | 3.7 (3.1) | 5.0 (3.4) |
| free range | 5.5 (4.1) | 4.1 (3.2) | 4.4 (3.4) | 5.8 (3.9) |
| krill-layer mix 0.10-0.25 [J] | 5.6-4.2 | 2.7-2.8 | 2.8-3.1 | 4.2-4.6 |
| night over the whole column | 7.4 | 3.0 | 3.0 | 4.1 |

Raising saithe off deep beds lowers its density at the bed there, which steepens the bed-prey gradients. No
allowed profile removes any gradient.

Vision variants:
- Ke 2e-4 lx flattens the bed prey to G 1.3 (100-150/150-250 ratio 1.25), keeps the fall to >= 250 m, and leaves
  krill at G 5.2.
- K_deep 0.09 steepens everything: benthivores 3.8, krill 18.

Effect size at start densities (E15), fixed attack and no policy:
- Intake per tonne over < 150 m beds is only 1.09-1.11 times that over 150-250 m beds; over >= 250 m it is 0.75.
  The reason is saturation.
- Diet moves more than ration: benthivore share 0.20 / 0.12 / 0.05 / 0.04 by class, whiting 0.14 / 0.04 /
  0.007 / 0.001.
- At the GOV saithe spread: benthivores 0.10, krill 0.39, planktivores 0.46, whiting 0.04, against booked
  0.055 / 0.37 / 0.51 / 0.043.
- Prey depletion, which would raise the payoff, is not modelled here.

## Verdict

1. Profile. Over 150-250 m beds by day saithe stand near the bed, as in the library (exp, Hs ~15 m) or nearer
   [F/D; moderate for summer and autumn days, low for winter and night]. Over > 250-300 m beds they spread higher,
   roughly in proportion to bed depth + 10 m [F; low-moderate]. Large adults partly rise to the surface at night
   [F data]. "Vertical room" holds only over the deepest beds.
2. Gradient against the stomachs. Between 100-150 and 150-250 m beds the stomachs cannot test the gradient, and
   the saturated model prediction is near the field. Beyond 250 m they contradict it by one to two orders.
   Juvenile gadoids disagree at 150-250 m too. Moderate in direction, weak in magnitude.
3. The carrier is not saithe's profile.
   - Bed prey (benthivores J and A, whiting, pout, blue whiting): the vision term at the light at the bed, i.e. the
     extrapolated K_deep below 65 m plus the judged Ke and R_nv.
   - Krill: mostly geometry, a bed-anchored saithe against a krill layer lifted over deep beds (G 5.2), plus vision
     (G 1.6).
   - The supported profiles keep or steepen every gradient.
4. Multipliers under the supported profile. The library's for beds < 250 m. Under the depth-scaled (Stensholt)
   form:
   - krill 2.83 / 1.05 / 0.23 / 0.011
   - planktivores 2.45 / 1.15 / 0.34 / 0.10
   - benthivores 2.13 / 1.30 / 0.39 / 0.14
   - whiting 1.98 / 1.36 / 0.43 / 0.14
   Both fall 2.4-5.4 times from 100-150 to 150-250 m.
5. Confidence. The decomposition is exact [D]. Which vision input is wrong is not identified, and no measurement of
   box light below 65 m or of gadoid detection in near-darkness was found.

## Gaps

1. Box light below 65 m (the sensor floor); K_deep sets g at the 150-250 m bed between 0.02 and 0.31.
2. Gadoid detection distance below ~0.01 lx (R_nv; Ke bracket 2e-4 to 0.1 lx).
3. True-position Q3 saithe stomachs with prey weights by bed class.
4. Bed-referenced night profiles, and the 60-160 m layer over deep beds by day.
5. Field krill density by bed class.
6. Capture bed against feeding bed (tags).
7. Not obtained: Stensholt et al. 2002 full text, Bergstad 1990/1991, Rosen & Holst 2013 (saithe near the bottom
   at 150-250 m), Thunen WH III 414.

## If applied (not applied; the user's call)

- The profile needs no change for beds < 250 m. The depth-scaled profile for > 250 m beds is supported, but it
  steepens the gradients there.
- A field-supported change would act on the vision term at depth for every visual pair, not only the water
  column: a measured or bracketed K_deep, a lower Ke, or a larger R_nv in near-darkness. None is measured here. The
  Ke 2e-4 bracket is the existing variant closest to the stomachs; it is not derived from them.
- For krill, the geometry would need field krill positions over 150-250 m beds, or saithe's search time in the
  krill layer.
- Any change is to be judged on the like-for-like field-spread gate and the long-run probe. Under fixed attack the
  class tables mainly move diets.
- The near-ceiling saturation at start densities (M4) is a separate question.
