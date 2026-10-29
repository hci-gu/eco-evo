# Prey-size windows of the predators on the planktivores and benthivores (2026-10-29)

Question (caller): for the juvenile compartment J (age 0 + age 1; age 0 9-13 cm in August, 14-16 cm
in November; age 1 ~15-25 cm), the prey-size window of each predator pair, so that f_J and f_A can be
derived from the prey's length distribution. Nothing in the library or code was changed.

Grades: **D** computed here from data, **F** stated in a source, **A** analogy (other species/area),
**J** judgement.

Scripts (`scripts/`) and outputs (`tables/`):
- `sms_prey_size.py` -> `sms_prey_size_stdout.txt`: SMS 2023 key run stomach file
  (`../Sejens diet pa unga torskfiskar/data/sms_keyrun2023/summary_stom.out`; North Sea 1981, 1985-87,
  1991, a few later years), fish prey only, weight = stomach-content share x hauls, the same weighting
  and step percentile as `../Makrillens bytesstorlek/scripts/mackerel_planktivore_gape.py`
  (it reproduces mackerel p95 0.33 and horse mackerel 0.31). 29 duplicate saithe rows (1986-87 Q3,
  repeated under two predator length classes with the same mean length) are dropped; they had
  pulled the 45-65 cm saithe p95 down from 0.30 to 0.21. Prey are SMS species only (herring,
  sprat, Norway pout, sandeels, cod, haddock, whiting, plaice, sole): blue whiting, poor cod, long
  rough dab, dab, gobies are "other food" with no length. Prey length classes are coarse (5.5, 6, 7.5,
  8.5, 9, 11, 12.5, 13.5, 17.5, 22.5, 27.5 cm mid-points).
- `ices_prey_size.py` -> `ices_prey_size_stdout.txt`: ICES Stomach Content database (public
  download, `../Torskfiskarnas bytesandelar/data/diet/ices_stomach/x*`), individual fish-prey
  lengths, by number and by mass, all areas and Skagerrak + northern North Sea.
- `stage_admission.py` -> `stage_admission_stdout.txt`: the window (ratio x length, ratio by
  predator-length band) over the box predator length mix (NS-IBTS GOV, biomass by length:
  `../Fiskgrupper i rutan/predators/tables/pred_length_biomass.csv`; cod
  `../Fiskgrupper i rutan/benthivores/tables/length_biomass_longrun.csv`), and the share of predator
  biomass whose window reaches 13 cm (age 0, Aug), 16 cm (age 0, Nov) and 25 cm (top of age 1).

Reading rule [J]: the realised stomach percentiles are the window (p95 central, p99 as the upper
case); morphological gape (Scharf-type lines, SMS minimum weight ratios) is only an upper bound -
predators eat well inside it (Floeter & Temming 2005: preference narrows before gape closes). All
ratios are prey total length / predator total length.

## 1. Box predator sizes [D]

NS-IBTS GOV in the box, biomass-weighted length, p10 / p50 / p90 (cm), pooled Q1 + Q3 unless stated:
saithe 42 / 52 / 78 (mean 58); spurdog 68 / 78 / 82; hake Q1 28 / 32 / 42, Q3 38 / 58 / 88
(bimodal by season); anglerfish 52 / 78 / 112; velvet belly 32 / 42 / 48; ling 52 / 78 / 142;
mackerel Q3 22 / 28 / 38; horse mackerel Q3 28 / 32 / 38; whiting Q1 18 / 28 / 38, Q3 12 / 22 / 32;
cod (2001-25) 30 / 54 / 80 (76 % of the biomass >= 40 cm); haddock 20 / 34 / 44. These are the
predator-side sizes; the cm figures in the earlier notes (saithe 19, spurdog 25-30, hake 28,
anglerfish > 40, whiting 12) are maximum PREY lengths and are checked against the windows below.

## 2. Fish predators

### 2.1 Water column -> planktivores and -> benthivores

Member shares of the pair's a (`../Rovfiskens uppdelning/tables/split_encounter_223_stdout.txt`,
G1_wc): on planktivores mackerel 0.49, saithe 0.33, horse mackerel 0.12, hake 0.03, spurdog 0.02,
velvet belly 0.01; on benthivores saithe 0.77, spurdog 0.10, hake 0.07, velvet belly 0.04,
anglerfish 0.01; ling < 0.01 in both.

| Member | Source, n fish-prey records | p50 | p90 | p95 | p99 | max | Box size (p50) | Window at p50: p95 (p99) | Grade |
|---|---|---|---|---|---|---|---|---|---|
| **saithe** 45-65 cm | SMS, 71 | 0.14 | 0.20 | **0.30** | **0.41** | 0.50 | 52 cm | **16 (21) cm** | D |
| saithe 30-45 / 65-130 cm | SMS, 20 / 74 | 0.23 / 0.14 | 0.24 / 0.26 | 0.24 / 0.30 | 0.29 / 0.31 | 0.42 / 0.37 | | | D |
| saithe, pooled (weighted pred mean ~61 cm, box mean 58) | SMS, 165 | 0.14 | 0.24 | 0.30 | 0.41 | 0.50 | 52 cm | 16 (21) cm | D |
| saithe on haddock + cod + plaice, 45-65 cm | SMS, 16 | 0.19 | 0.21 | 0.28 | 0.28 | 0.32 | 52 cm | 15 cm | D |
| **hake** 15-40 cm | ICES-DB 2024 (Skagerrak 48 of 69 items), 22 | 0.38 | 0.48 | 0.50 | 0.52 | 0.52 | Q1 32 cm | **16 cm** | D |
| hake 40-60 cm | ICES-DB 2024, 23 | 0.27 | 0.44 | 0.44 (mass 0.52) | 0.52 | 0.52 | Q3 58 cm | **26-28 cm** | D |
| hake 60-130 cm | ICES-DB 2024, 24 | 0.14 | 0.33 | 0.35 (mass 0.39) | 0.39 | 0.39 | | | D |
| hake (check) 20-40 / 40-110 cm | SMS 2013, 6 / 12 | 0.34 / 0.20 | 0.56 / 0.31 | 0.56 / 0.32 | | | | | D |
| **spurdog** 70-130 cm | ICES-DB 1991 (central N Sea), 16 records / 25 fish | 0.19 | 0.25 (mass 0.38) | 0.31 (mass 0.38) | 0.38 | 0.38 | 78 cm | **24-29 cm** | D (thin) |
| **anglerfish** 50-150 cm | ICES-DB 2022 (NNS, Skag, CSNS), 11 | 0.35 (mass 0.52) | 0.52 | 0.57 | 0.57 | 0.71 (all sizes) | 78 cm | **44 cm** | D (thin) |
| mackerel | SMS, 94 | 0.20 | 0.31 | 0.33 | 0.56 | 0.56 | Q3 28-30 cm | ~10 cm | D (`../Makrillens bytesstorlek`) |
| mackerel (check) 30-50 cm | ICES-DB 2022 (central N Sea), 300 | 0.23 | 0.26 | 0.27-0.28 | 0.29-0.32 | 0.32 | | | D |
| horse mackerel | SMS, 27 | 0.24 | 0.29 | 0.31 | 0.34 | 0.34 | Q3 32 cm | ~10 cm | D (`../Makrillens bytesstorlek`) |
| velvet belly | no prey lengths found | | | 0.25 | 0.30 | | 42 cm | ~11 (13) cm | J (small shark of shrimps and small fish) |
| ling | ICES-DB, 5 records | 0.20 | | 0.28 | | 0.28 | 78 cm | ~22 (27) cm | J (D on 5 items) |

Notes per member:
- **Saithe.** Realised prey are mostly small: 83 % of the fish-prey mass of 45-65 cm saithe is <= 8.5 cm
  (0-group Norway pout and sandeel; SMS, D). Q3 is the small-prey quarter (40-70 cm: p95 0.19, n 44),
  Q1 the large one (p90-p99 0.41, n 11). Field upper ends [F]: Mehl 2005 (NEA saithe, Oct-Nov,
  5865 stomachs) "haddock (10-24 cm), Norway pout (5-19 cm) and blue whiting (10-19 cm)" in 40-60 cm
  saithe, i.e. ratio up to ~0.4-0.48; Floeter & Temming 2005 "Saithe of twice the size of whiting
  preferred the same absolute prey sizes" and the highest predator/prey weight ratios of the three
  species; preferred length ratio ~0.27-0.38 (J, read from their Fig. 5b in
  `../Sejens diet pa unga torskfiskar/evidence_settlement.md` entry 12). ICES-DB has one saithe
  item. Literature check (section 4): Scharf et al. 2000 pollock (same species, NW Atlantic) mean
  0.19, 95th quantile 0.36 at 52.5 cm [F/J]; Mehl 2005 Table 5 by mass, 40-60 cm saithe, median
  0.27, p95 0.40 [J from F]. The low end is the Q3-Q4 diet (p95 0.19-0.20, 0-group pout and
  sandeel). **Central [D/F]: p95 0.30 (range 0.20-0.40) -> window 16 cm (11-21) at the box
  median; upper case 0.40-0.41 -> 21 cm.** The earlier 19 cm (ratio 0.36) sits inside the range,
  near its upper end: keep it as the upper-mid case, not the centre.
- **Hake.** The best member data: 2024 Skagerrak stomachs, herring 11-18 cm in 34-56 cm hake and
  22-25 cm herring and haddock in 48-78 cm hake. Window **16 cm in Q1** (small hake), **26-28 cm in
  Q3**: the earlier 28 cm holds for the summer hake only.
- **Spurdog.** 16 records (herring 20-30 cm, whiting 20 cm, flatfish 10-15 cm in 80 cm spurdog),
  not Skagerrak. Window ~24-29 cm confirms the earlier 25-30 cm. Scharf et al. 2000 (spiny
  dogfish, n 3253) agrees: mean 0.14, 99th quantile 0.41 at 75-85 cm (section 4). The window is for
  prey swallowed whole; spurdog also tear pieces from large prey (S, not quantified).
- **Anglerfish.** 11-16 items, prey 8-33 cm, ratio p50 0.35, p95 0.57, max 0.71: window ~44 cm at
  78 cm. The earlier "> 40 cm" holds; every J fish and most adults fit. Scharf et al. 2000 goosefish: 95th quantile 0.56 at 78 cm,
  prey up to 0.8-0.9 of own length (section 4) - agrees.
- **Velvet belly, ling.** Share of a 0.01-0.04 and < 0.01: judgement values are adequate.

### 2.2 Whiting -> benthivores (and whiting on planktivores)

| Whiting length | Source, n | p50 | p90 | p95 | p99 | max | Grade |
|---|---|---|---|---|---|---|---|
| 10-20 cm | SMS, 95 | 0.43 | 0.51 | 0.68 | 0.71 | 0.82 | D |
| **20-30 cm** | SMS, 191 | 0.27 | 0.38 | **0.38** | **0.49** | 0.56 | D |
| 20-30 cm, benthivore prey (haddock, cod, plaice) | SMS, 29 | 0.31 | 0.45 | 0.45 | 0.49 | 0.56 | D |
| 30-50 cm | SMS, 233 | 0.20 | 0.28 | 0.33 | 0.38 | 0.50 | D |
| 20-30 cm | ICES-DB (CSNS 2022 mostly), 120 | 0.26 (mass 0.35) | 0.44 | 0.46 (mass 0.48) | 0.55 | 0.65 | D |
| all sizes, Skagerrak + NNS | ICES-DB, 25 | 0.30 | 0.45 | 0.50 (mass 0.45) | 0.53 | 0.53 | D |

Box whiting is small (Q1 median 28 cm, Q3 22 cm). Window at the median: **Q1 11 cm (p99 13.5),
Q3 9 cm (p99 11)**; with the ICES-DB p95 0.46-0.48 it is 10-13 cm. The earlier ~12 cm is the upper
case (p99 at Q1, or ICES p95). Whiting therefore takes only the small end of the age-0 cohort:
settled haddock and cod in Jul-Sep (8-12 cm), hardly anything from November on (age 0 at 15-16 cm).
Field identity [F]: Floeter & Temming 2005: "Whiting larger than 30 cm feed almost entirely on fish";
its preferred predator/prey weight ratio rises with size beyond mid-size whiting. One ICES-DB
whiting item (84 cm prey in a ~32 cm whiting) is an obvious error; it touches only the 30-60 cm
p99.

### 2.3 Benthivores -> planktivores: cod (and haddock)

| Cod length | Source, n | p50 | p90 | p95 | p99 | max | Grade |
|---|---|---|---|---|---|---|---|
| 15-30 cm | SMS planktivore prey, 44 | 0.31 | 0.38 | 0.40 | 0.56 | 0.56 | D |
| 30-45 cm | SMS planktivore prey, 53 | 0.23 | 0.29 | 0.34 | 0.36 | 0.60 | D |
| **45-65 cm** | SMS planktivore prey, 78 | 0.16 | 0.25 | **0.30** | **0.39** | 0.50 | D |
| 45-65 cm | ICES-DB Skagerrak + NNS, planktivore prey, 40 records / 121 fish | 0.20 | 0.30 | 0.30 | 0.30 | 0.31 | D |
| 45-65 cm | ICES-DB all areas (mostly Kattegat/Baltic), planktivore prey, 1983 | 0.21 (mass 0.24) | 0.32 | 0.36 (mass 0.40) | 0.43 (mass 0.49) | | D |
| 65-120 cm | SMS planktivore prey, 102 | 0.13 | 0.19 | 0.27 | 0.35 | 0.35 | D |
| gape (morphology) | Scharf et al. 2000 line, k | | | | | 0.47 (0.44-0.50) | F, upper bound (`../Fiskgrupper i rutan/benthivores/evidence_benthivores_C_diets_consumption.md`) |

Box cod 30 / 54 / 80 cm: window p95 **10 / 16 / 22 cm** (p99 at the median 21 cm). Cod therefore
takes all age-0 herring, sprat and pout and the lower half of age 1 at the median size, and almost
all of age 1 only at >= 70-80 cm. Field [F, earlier note]: herring eaten by cod (ICES-DB, 1110
items, mostly Kattegat/Baltic) median 15 cm, ratio median 0.31; in the 2023 North Sea samples
herring was found only in 80-90 cm cod (CINEA 2024).

**Haddock eats little fish.** Fish ~3 % of its diet (P ~3 %, `../Fiskgrupper i rutan/benthivores/
evidence_benthivores_C_diets_consumption.md` section 3) [F/D]; SMS haddock fish prey (n 80: sandeel,
pout, herring) ratio p95 0.33 pooled, 0.28 at 30-45 cm, prey max 13.5-17.5 cm; ICES-DB 2025 six
items, max ratio 0.21. Window at the box median 34.5 cm ~10-12 cm [D]. For the pair the fish eater
is cod; haddock adds a small age-0-only term [J].

### 2.4 Share of predator biomass whose window reaches each stage [D]

`stage_admission.py`: ratio by length band (p95; p99 in brackets) x the box predator length mix.

| Predator | >= 13 cm (age 0, Aug) | >= 16 cm (age 0, Nov) | >= 25 cm (top of age 1) |
|---|---|---|---|
| saithe (p95 0.30, p99 0.41) | 0.83 (0.98) | 0.47 (0.95) | 0.07 (0.37) |
| cod | 0.80 (0.90) | 0.53 (0.70) | 0.04 (0.21) |
| whiting Q1 / Q3 | 0.03 / 0.02 (0.39 / 0.23) | 0.00 (0.03 / 0.02) | 0 |
| hake Q1 / Q3 | 0.91 / 0.97 | 0.71 / 0.94 | 0.03 / 0.38 (0.04 / 0.55) |
| spurdog | 1.00 | 0.99 | 0.87 (0.94) |
| anglerfish | 1.00 | 1.00 | 0.96 (0.99) |
| mackerel Q3 | 0.04 (p99 = one record) | 0 | 0 |
| horse mackerel Q3 | 0.04 | 0 | 0 |
| velvet belly (J) | 0 (0.12) | 0 | 0 |
| ling (J) | 0.99 | 0.89 | 0.35 (0.61) |

Saithe sensitivity (one ratio for all sizes), share of saithe biomass reaching 13 / 16 / 25 cm:
0.20 (Q3-Q4 diet, low case) -> ~0.35 / 0.15 / 0; **0.30 (central) -> 0.83 / 0.47 / 0.07**; 0.36 (the
earlier 19 cm) -> 0.97 / 0.83 / 0.19; 0.40 -> 0.98 / 0.95 / 0.37. Mackerel's p99 (0.56) is one
record (`../Makrillens bytesstorlek/summary.md`); do not carry it forward.

## 3. Marine mammals (prey lengths in cm; swallowed whole)

### 3.1 Harbour porpoise -> planktivores (and gadoids)

| Source | Area, n | Herring | Other prey | Grade |
|---|---|---|---|---|
| Börjesson et al. 2003 (MMS 19:38) | Kattegat-Skagerrak 1989-96, 112 stomachs | mean **26 cm**; "prey primarily on size classes containing mature or maturing individuals" | cod 28.1 +/- 5.8, whiting 19, gobies 4.5 | F (abstract) / S (means via Sveegaard 2011 p. 93, Wallin 2017); full text not accessed |
| Aarefjord et al. 1995 | Scandinavian waters | | 74 % of prey <= 25 cm, largest 49 cm | S |
| Sveegaard 2011 Paper VII | the Sound 1987-2010, 53 stomachs, 1442 fish | 174 herring: 20-25 cm 50 % (high season), 15-20 cm 21-31 %, < 15 cm 14-21 %, >= 25 cm 8-11 % by number; by mass < 15 cm ~3.5 %, >= 25 cm ~21 % | all prey 16.6 +/- 9.0 | F (Fig. 5a read) / D |
| Andreasen et al. 2017 (MMS 33:1053) | W Baltic + S Kattegat 1980-2011, 339 stomachs | mean 17.7 (2.6-30.6), n 756; by number 1-9 cm 4 %, 10-19 cm 64 %, >= 20 cm 32 % | sprat 10.3 (3-16), cod 23 (2.6-57), whiting 14.9, pout 8.2, sandeel 15.1, gobies 3.7 | F / D |
| Santos et al. 2004 (MMS 20:1) | Scotland 1992-2003, 188 stomachs | mode 11 cm, max 29 | whiting mode 12-13, max 35.5; Trisopterus mode 7 | F |
| SMS key run (porpoise, North Sea) | 22 records (1985, 1995, 2005) | p50 11, p90 17.5, max 32.5 (8 records) | all fish p50 7.5, p95 11, p99 17.5 | D (thin) |
| MacLeod et al. 2006 (MEPS 326:295) | 187 stomachs | | prey/porpoise length mode 0.05-0.06, max 0.39 (~51 cm) | F |
| Santos & Pierce 2003; Ransijn 2021 citing Aarefjord | | | prey mostly < 30 cm, practical limit ~40 cm; teeth hold, do not cut | F/S |

Window [J]: porpoises take the whole herring size range up to ~30 cm, upper limit ~40 cm. In the
Kattegat-Skagerrak they eat mainly adult herring (mean 26 cm), but 10-19 cm herring dominate where
small herring are common (Andreasen) and 9-16 cm age 0 is eaten in the Sound. The window admits
both stages: **f_J = f_A = 1 on length grounds [F/J]**. The Skagerrak adult preference is selection
inside the window (availability and payoff), so it belongs to validation of the trained porpoises
(share of adult herring in their diet), not to f. Weak:
Börjesson's distribution is second-hand; no local data after 1996.

### 3.2 Harbour seal -> benthivores (and -> planktivores)

| Source | Area, n | Lengths (median [quartiles] or mean) | Grade |
|---|---|---|---|
| **Sørlie et al. 2020** (Mar Biol Res 16:298) | Norwegian Skagerrak 2015-16, 121 scats, 757 otoliths, no digestion correction | whiting 15.1 [13.1-18.8] max ~25; haddock 23.2 [18.4-29.5] max 40; cod 28.6 [27.5-31.2] (n 6); long rough dab 16.3 [13.1-20.7]; witch 14.4 [11.2-21.8]; dab 23.7 [19.5-25.4]; plaice 24.6 [23.8-25.0]; poor cod 11.7; Norway pout 9.9 [7.7-10.8]; saithe 12.6 [11.4-15.2]; herring 26.3 [22.6-28.7]; range 1.2-54.8 | F (Fig. 3 read +/- 0.8 cm) |
| Teittinen 2024 (GU MSc, Hvaler) | Aug 2019, 43 scats, 774 otoliths, no correction | mean 15.9 (3.5-55); cod 15.3 (7-40); herring 18 (9-25); plaice 22.7 (11-41); 58 % of fish < 15 cm, 95 % < 30 cm by number | F |
| Harding 2022 | Swedish Skagerrak | most fish 10-30 cm historically, now ~7-20 cm; no cod > 40 cm | S |
| Wilson & Hammond 2016 (SMFS 7(21)) | UK 2010-11, 1976 scats, corrected | means: large gadoids 17.4-24.3, Trisopterus 12.4-14.0, flatfish 14.2-17.4, pelagics 26.6-32.6; 95 % < 30 cm | F (A for the box) |
| Brown & Pierce 1998 (MEPS 167:275) | Shetland, corrected | whiting bimodal 16-20 and 32-36; cod < 50, mode 20-30; haddock mode 30-35; herring mode 25-30 | F (A) |
| Aarts et al. 2019; Andersen et al. 2007 | Wadden Sea; Limfjord | mostly < 25 cm, peak 10-20; herring 21.5 +/- 5.5, other species below minimum landing size | F (A) |
| SMS key run, **grey seal** | North Sea, 232 records | all fish p50 11, p95 17.5, p99 22.5; benthivore prey p50 17.5, p95 22.5 | D, A (other seal) |

Biomass split by prey length (Sørlie Fig. 3 readings, lognormal fits, mass ~ L^3; agent's
calculation, `scripts/seal_prey_size_bins.py`, `seal_prey_size_trunc.py`) [J]: gadoids < 16 cm
18-24 %, 16-25 cm 27-33 %, > 25 cm 43-55 %; flatfish 5-6 / 37-44 / 50-58 %. The > 25 cm gadoid part
rests on ~45 haddock/pollack/cod otoliths; lengths are not digestion-corrected (true sizes larger).

Window [J/F]: seals take benthivores from ~7 cm to ~40 cm, centre 15-25 cm, few above 30-40 cm
(no large cod). On the benthivores the age-1 band (15-25 cm) and small adults (25-35 cm haddock,
dab, plaice) carry the diet; age 0 (9-13 cm) is taken mainly as poor cod/pout-sized prey (planktivore
side) and small whiting. Central for the benthivore pair: **f_J ~1, f_A ~ share of adult benthivore
biomass below ~35 cm (J)** - in the box most haddock (median 34.5 cm) and flatfish, little cod
(median 54.5 cm). For the planktivores: herring medians 18-26 cm and pout 10 cm: no exclusion
of either stage.

## 4. Literature checks for the fish windows

| Predator | Source | Statement / numbers | Ratio at box size | Grade |
|---|---|---|---|---|
| saithe | Scharf et al. 2000 (MEPS 208:229), pollock n 271 | 95th-quantile max PY = 0.259 PD + 51.9 mm; mean PY = 0.135 PD + 30.9; 61.6 % of prey < 0.20 PD | 52.5 cm: mean 0.19, q95 0.36 | F / J |
| saithe | Mehl 2005 Table 5 (NEA, Oct-Nov), by mass | 40-60 cm: median ~13.5 cm, p95 ~20 cm, max class 25-29; 60-80 cm: 0.20 / 0.34 | 0.27 / 0.40 | J from F |
| saithe | Floeter & Temming 2005 | preferred weight ratio 18-80 -> preferred length ratio 0.23-0.38 (electivity peak, not the diet) | ~0.29 | F / J |
| hake | Scharf 2000 silver hake n 1391; Guichet 1995 (Biscay, n 1096 fish prey); Mahé et al. 2007; Mellon-Duval et al. 2017 | silver hake mean 0.30-0.35, q99 0.63-0.67 at 32-57 cm; Guichet mean 0.39 (32 cm) to 0.31 (57 cm), largest prey 27 cm; Mahé mean PY = 0.34 PD + 0.9 cm; Mellon-Duval q95 ~0.55 PD | p50 0.30-0.35, p95 0.50-0.55, max 0.65-0.70 | F / J (A for silver hake) |
| spurdog | Scharf 2000 spiny dogfish n 3253 | q99 max PY = 0.324 PD + 70.6 mm; mean 0.112 PD + 23.1; 75 % < 0.20 PD | 80 cm: mean 0.14, q99 0.41 | F / J (A, conspecific NW Atlantic) |
| anglerfish | Scharf 2000 goosefish n 409; Johnson et al. 2008 | q95 max PY = 0.323 PD + 185 mm (0.56 at 78 cm); 41 % of prey > 0.40 PD; prey up to 0.8-0.9 PD; 82 cm cod in a 97 cm goosefish | p50 ~0.30, p95 0.56-0.6, max 0.85 | F / J (A, L. americanus) |

Not accessed: Börjesson 2003 full text, Aarefjord 1995, Härkönen 1987/1988, Härkönen &
Heide-Jørgensen 1991, Laurenson & Priede 2005 (L. piscatorius), Ellis et al. 1996, Du Buit 1991,
Homrum et al. 2012. Kappa et al. 2026 (MMS, doi 10.1111/mms.70069; likely the 1977/78 vs 2020
seal prey-length shift) is open access, but the automated download was blocked; it needs a manual
fetch.

## 5. Summary per pair

| Pair | Central window (p95 ratio x box median) | Upper case | Stages inside | Grade | Weakest point |
|---|---|---|---|---|---|
| water column -> planktivores | mackerel/horse mackerel ~10 cm (0.31-0.33); saithe 16 cm (0.30; 11-21); hake 16 (Q1) / 27 (Q3) cm; spurdog ~27 cm | saithe 21 cm, mackerel p99 15 cm | mackerel: small age 0 only; saithe: age 0 + lower age 1; hake Q3, spurdog: all of J | D/F | saithe ratio spread 0.20-0.40 (season) |
| water column -> benthivores | saithe 16 cm (11-21); spurdog 24-29; hake 16 / 27; anglerfish ~44; velvet belly ~11 (J) | saithe 21 | saithe: settled age 0 + small age 1; spurdog, anglerfish: all J + small adults | D/F | spurdog n 16, anglerfish n 11 (Scharf analogues agree) |
| whiting -> benthivores | 9 (Q3) - 11 (Q1) cm (0.38-0.40; ICES 0.46 -> 10-13) | 13.5 cm (p99) | small age 0 in Jul-Oct only | D | Q3 box whiting are 12-32 cm; window sensitive to the small fish |
| benthivores (cod) -> planktivores | 16 cm at 54 cm (0.30); 10 at 30 cm, 22 at 80 cm | 21 cm (0.39) | all age 0, lower age 1; age 1 fully only for cod >= 70-80 cm | D | box cod length mix (31 % of the group, 2001-25 level); haddock adds a small <= 12 cm term |
| porpoises -> planktivores | whole herring range to ~30 cm; Kattegat-Skagerrak mean 26 cm | ~40 cm | both (f_J = f_A = 1); adult preference is a validation target, not f | F/S | Börjesson second-hand; data 1985-96 |
| seals -> benthivores | 7-35 cm, centre 15-25 cm | ~40 cm (no large cod) | all J; adults below ~35 cm (haddock, flatfish), little cod | F (Sørlie) / J | otoliths uncorrected; > 25 cm share on ~45 otoliths; Härkönen not accessed |

## 6. Addendum: Kappa et al. 2026 (supplied by the user, 2026-10-29)

Kappa, Harkonen, Freitas, Carroll & Harding 2026, Mar. Mamm. Sci. 42: e70069 (open access).
Harbour seals, Hvaler (northern Skagerrak, 58.98 N 10.83 E), 73 scats, September 2020; historical
Koster samples 1977/78 and 1989 (Q3). The paper gives species shares of otoliths and of biomass, not
length distributions (Table S1, not seen, holds the same shares); otolith sizes are not corrected for
erosion. Mean prey weight fell from 57 g (1977/78) to 13 g (2020). Mean weights and lengths per
species from the totals (`scripts/seal_kappa2026.py`, `tables/seal_kappa2026_stdout.txt`; whiting and
blue whiting length-weight constants are J) [D from F]:

| species (2020) | otoliths % | biomass % | mean weight g | mean length cm (x1.25 erosion) |
|---|---|---|---|---|
| haddock/pollack/saithe | 12.8 | 18.7 | 18 | 13 (17) |
| cod | 1.7 | 10.7 | 79 | 21 (26) |
| whiting | 3.6 | 4.8 | 17 | 13 (16) |
| Norway pout | 21.8 | 11.3 | 6.5 | 10 (12) |
| herring | 0.5 | 1.4 | 35 | 18 (22) |
| long rough dab | 0.6 | 1.2 | 25 | 12.5 (16) |

Sandeel 39.6 % of otoliths (6.9 % of biomass) and ballan wrasse 13.5 % of biomass have no model pool.

Reading for the stage access: the seals' benthivore take is mostly juvenile, 0.74-0.86 by biomass (cod
by a lognormal with CV 0.3 around its mean length; haddock taken as half of h/p/s, all below 27 cm).
With f_J = 1 at the box J/B 0.155 that implies f_A 0.03-0.06, against 0.34 from the 7-35 cm window.
The two are not in conflict in kind: the window is what a seal can swallow; the take reflects where it
forages (coastal Hvaler, where juveniles are denser than in the 100 m box) and what it selects. Treat
it like the porpoise preference: f from the window (capability), the Kappa juvenile share 0.74-0.86 as
the validation target for the trained seals' realised take. Decision (user, 2026-10-29): the window
stays in f ([1, 0.34]); the Kappa juvenile share 0.74-0.86 is the validation target.
