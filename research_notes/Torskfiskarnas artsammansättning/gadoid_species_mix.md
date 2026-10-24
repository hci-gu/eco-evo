# Gadoid species composition in the Mareld box, and the benthivore share phi

Purpose: the model's single `gadoids` group needs the share of its biomass that forages on benthos
(phi, `../Bentos åtkomlighet för torskfiskar/benthos_accessibility.md` section 3). This note takes
the biomass composition from survey data for **every gadiform species caught**, not only the five
main ones, corrects it for gear catchability where evidence exists, and recomputes phi. Scores come
from morphology and habitat; diet is only a check.

Tags: **[F]** full text read or data downloaded, **[A]** abstract/snippet, **[S]** secondary,
**[D]** computed here from downloaded data. Detail and quotes are in:
- `evidence_A_survey.md`: DATRAS NS-IBTS hauls, coverage, length-weight relations, CPUE tables [D].
- `evidence_B_catchability.md`: Fraser et al. 2007 GOV q, ling/tusk indicators, landings [F]/[D].
- `evidence_C_diet_morphology.md`: morphology, habitat and diets of the added species [F]/[S].
- `scripts/` (run with `python3 -I`; order extract_datras, gadoid_mix, catchability, phi_mix), `tables/` (raw outputs; Fraser-corrected in `tables/q_<variant>/`), `inputs/` (scores; unity q file), `data/` (raw
  downloads, untrusted, see `data/README.md`).

## Bottom line

| Quantity | Central | Range | Basis |
|---|---|---|---|
| phi, box, without Norway pout | **0.53** | **0.32-0.65** | depth-stratified NS-IBTS Q1+Q3 2021-25, no q correction; low end = Fraser q (A_floor), high end = Q1 / strict box |
| phi, with Norway pout in the group | 0.50 | 0.29-0.62 | same; Norway pout scored 0.1 |
| phi, Fraser-q corrected (depth, pooled) | 0.32-0.49 | 0.21-0.66 (bootstrap) | depends on how the saithe q is read (A_floor vs A_sw_floor) |
| phi, seasonal (raw, depth) | Q1 0.64, Q3 0.48 | | saithe is a Q3 visitor |
| Box share of the ten "minor" gadiforms incl. ling (raw GOV) | 0.7 % | 0.5-1.2 % | ling 0.3 %, rocklings 0.2 %, poor cod, pollack, forkbeard, silvery pout |
| Box share of ling + tusk, catchability allowed for | **1.5-2 %** | 0.3-5 % | GOV raw 0.3 %, Fraser-corrected ~1 %, landings 2-3 %, shrimp survey 5 % |
| Effect of all added species on phi | < 0.01 | | under any score in their ranges |

**The added species do not move phi.** Ling is 0.2-0.4 % of the GOV gadoid biomass in the box,
all ten minor species together ~0.7 % (max 1.2 % in Q1), and tusk was not caught in any of 265 hauls. Even at the
catchability-corrected upper bound (ling + tusk ~5 %, scored 0.4-0.75) phi changes by < 0.01.
The levers are the haddock : whiting : saithe proportions, and those depend on three choices:
1. **Depth representativeness** (largest systematic effect on the raw estimate): the box-rectangle
   hauls are shallow (median 68 m against a grid mean of 168 m) and whiting-rich; the rectangle mean
   gives phi 0.40, the depth-stratified and strict-box estimates 0.53 and 0.56.
2. **Season**: saithe is ~1 % of the gadoids in Q1 and 30-40 % in Q3 (raw); phi is 0.64 in Q1 and
   0.48 in Q3.
3. **Catchability**: Fraser et al.'s GOV q lowers phi to 0.32-0.49, almost entirely through
   saithe, whose q is not measured but averaged from three other species (section 4).

Recommendation: **phi = 0.53, range 0.32-0.65.** Treat the Fraser-corrected value as the low
bound, not the central (section 4). The earlier note's 0.60 used Skagerrak-wide weights (haddock
54 %). The same species at the box's depths give 0.53; Skagerrak-wide they give 0.58 here.

## 1. Haul coverage (task 1) [D]

| Haul set | Hauls Q1 / Q3 | <100 m | 100-200 m | 200-300 m | >=300 m |
|---|---|---|---|---|---|
| Box rectangles 44G0-45G1 | 58 / 62 | 76 | 37 | 7 | 0 |
| Strict box (57.81-58.31 N, 10.07-11.15 E) | 25 / 16 | 12 | 22 | 7 | 0 |
| Region (Skagerrak + box rectangles) | 125 / 140 | 177 | 66 | 22 | 0 |
| **Grid area share** (`maps/depth.npz`) | | **23 %** | **49 %** | **19 %** | **9 %** |

- By rectangle: 44G0 27 / 6 / 0, 44G1 34 / 2 / 0, 45G0 0 / 20 / 7, 45G1 15 / 9 / 0 (<100 / 100-200 /
  200-300 m). The box is 98 % Swedish hauls, so national recording differences do not matter.
- No valid GOV haul in the region is deeper than 260 m. The NS-IBTS manual caps standard stations at
  250 m [F]. The 300-530 m trench slope (9 % of the grid) is unsampled; in the depth-stratified
  estimate it takes the 200-300 m composition. That is a floor for ling, hake, forkbeard, silvery
  pout and blue whiting, which all rise with depth.
- 44G1 is mostly the Kattegat mouth and the Swedish coast, outside the grid. Hence "depth-stratified"
  (region hauls in depth bins, weighted by the grid's area per bin) and "strict box" are the
  preferred estimators, and the rectangle mean is the shallow sensitivity.

## 2. Species found and length-weight [D]

Gadiformes were picked by WoRMS classification of all 116 taxa with catches in the region, not
from a hand list. Sixteen species occur. **Tusk, blue ling (one fish), threebeard rockling (one
fish), *Ciliata*, *Raniceps* and Macrouridae have no catch in the Skagerrak or box hauls
2021-2025.** The 2021-25 NS-IBTS tusk records are seven Norwegian fish from the northern North Sea.
DATRAS records silvery pout as *Gadiculus argenteus*. The Skagerrak form is *G. thori*
(evidence C).

Length-weight: fitted to NS-IBTS individual weights 2021-25 for 10 species (n 107-64 816).
FishBase Bayesian LWR [F] was used for pollack, tusk, forkbeard, silvery pout, rocklings and blue
ling (evidence A, section 3). Recomputed with these, the earlier five-species shares (box rectangles,
120 hauls) agree within 0.6 points: haddock 34.3, whiting 38.8, saithe 20.1, cod 5.0, hake 1.2 %.

## 3. Biomass shares, raw GOV (task 1) [D]

Shares of the 15 non-zooplanktivore gadiforms (%). "Pooled" = mean of the Q1 and Q3 kg/h, then
shares; this weights toward Q3, which has 2-3 times the biomass. Minor = ling, pollack, forkbeard,
poor cod, bib, silvery pout, rocklings, blue ling.

| Estimator | Quarter | Haddock | Whiting | Saithe | Cod | Hake | Ling | Other minor | phi |
|---|---|---|---|---|---|---|---|---|---|
| Depth-stratified (265 hauls) | Q1 | 54.3 | 32.3 | 2.4 | 9.5 | 0.3 | 0.3 | 0.9 | 0.64 |
| | Q3 | 43.6 | 11.2 | 39.3 | 4.3 | 1.0 | 0.3 | 0.4 | 0.48 |
| | **Pooled** | **46.6** | **17.0** | **29.1** | **5.7** | **0.85** | **0.27** | **0.5** | **0.53** |
| Strict box (41) | Pooled | 48.7 | 19.3 | 23.0 | 6.8 | 1.5 | 0.35 | 0.4 | 0.56 |
| Box rectangles (120) | Q1 / Q3 | 48.7 / 28.6 | 40.7 / 38.1 | 0.5 / 27.9 | 9.1 / 3.5 | 0.2 / 1.6 | 0.2 / 0.2 | 0.6 / 0.1 | 0.58 / 0.32 |
| | Pooled | 34.6 | 38.9 | 19.8 | 5.1 | 1.2 | 0.19 | 0.3 | 0.40 |
| Skagerrak (232) | Pooled | 52.1 | 27.3 | 13.3 | 5.8 | 1.1 | 0.2 | 0.3 | 0.58 |

The rectangle mean over all 120 hauls taken together (the earlier note's convention) gives the same
shares within 1 point. Rectangle-mean-then-mean gives totals within 10 %.

**With the zooplanktivores** (depth-stratified, pooled): Norway pout is 6.1 % of gadoids + pout.
Of all gadiforms, Norway pout is 5.2 % and blue whiting 13.9 %. Blue whiting is absent in Q1 and in
water < 100 m, and reaches ~200 kg/h at 100-300 m in Q3 (one haul = 45 % of the box sum). Box
strict: Norway pout 8.5 %, blue whiting 13.8 % of all gadiforms.

**Sampling error** (bootstrap over hauls, 5-95 %): saithe 3-37 % of the box-rectangle mix (one haul
= 54 % of the saithe sum), haddock 25-45 %, whiting 28-51 %, cod 3.8-6.6 %. phi, depth-stratified:
0.41-0.65; strict box 0.39-0.73; rectangles 0.30-0.51.

## 4. Catchability (task 2)

**GOV q, Fraser, Greenstreet & Piet 2007** [F, web render; evidence B, section A1]. q per 1-cm class
is from the Q3 NS-IBTS calibrated against VPA numbers. Haddock and whiting q are per door-spread
swept area, all other species per wing (net) swept area. Converted to the wing basis with each
haul's door/wing ratio (median 3.75), the biomass-weighted effective q in the region is: whiting
1.70, haddock 0.75, cod 0.21, saithe 0.17 (Fraser's rule: mean of cod, haddock and whiting q as
printed) or 0.43 (same mean on the wing basis), Norway pout 0.12, ling 0.105, hake 0.091,
fourbeard rockling 0.04, poor cod 0.011 [D]. Tusk, pollack and forkbeard get a "roundfish average"
whose value was not found; ling's 0.105 was used. Walker et al. 2017 efficiencies sit in a
supplement that was not obtained [A].

| Variant (depth-stratified, pooled) | Haddock | Whiting | Saithe | Cod | Hake | Ling | phi (min-max scores) | phi +NoP | phi Q1 / Q3 |
|---|---|---|---|---|---|---|---|---|---|
| Raw | 46.6 | 17.0 | 29.1 | 5.7 | 0.9 | 0.3 | 0.53 (0.50-0.58) | 0.50 | 0.64 / 0.48 |
| A_floor: Fraser, saithe rule as printed, q >= 0.10 for small species | 21.1 | 4.1 | 59.6 | 9.3 | 3.3 | 0.9 | 0.32 (0.27-0.33) | 0.29 | 0.70 / 0.25 |
| A_sw_floor: saithe from wing-basis q | 32.3 | 6.3 | 38.0 | 14.3 | 5.1 | 1.4 | 0.49 (0.41-0.51) | 0.41 | 0.75 / 0.42 |

- **Why the Fraser value is the low bound, not the central.**
  - The correction acts almost entirely through saithe, and saithe's q is a convention (an average
    of three species on mixed swept-area bases), not a measurement. Cod (up) and hake (up) move phi
    in opposite directions and roughly cancel.
  - Fraser's q is fitted against North-Sea-wide stock numbers, so it folds availability (fish
    outside the surveyed depths, area and season) into gear efficiency. Within-haul shares in the box
    need only gear efficiency. Saithe's low q is largely availability (the shelf edge, deep water).
  - A local check points the other way. The earlier note's Skagerrak catch / q=1 swept-area
    biomass gave u of haddock 0.11, saithe 0.13, whiting 0.03, cod 0.68, hake 0.89, against
    stock-level rates of 0.06 / 0.30 / 0.08 / 0.39 / 0.27. That implies q relative to haddock of
    about 1 for cod, 0.5 for hake and > 1 for saithe and whiting. It is rough (stock F is not local
    F), but it does not support up-weighting saithe [D, from `ices_gadoid_f.md`].
  - Fraser's poor cod q (0.011) would make poor cod 17-24 % of the Q1 biomass, against nil landings
    and absence from the shrimp survey. Hence the floor of 0.10 for the small species. The unfloored
    variants are kept in `tables/q_A`, `q_A_sw`.
  - Other caveats: Fraser is Q3 only, and Q1 uses 110 m sweeps below 70 m, so herding differs. The
    printed cod polynomial runs ~5 % low at 60-96 cm. The Q3 200-300 m bin is 8 hauls with 459 kg/h
    saithe, and that leverage survives every correction.
- **Ling and tusk** (evidence B, sections B-C):
  - GOV raw: ling 0.3 % of the box gadoids, tusk 0. Fraser-corrected ling: 0.9-1.4 % [D].
  - Landings, 27.3.a (incl. Norway's unsplit 3.a), 2022-24 [F/D]: ling 203 t, tusk 10.5 t, against
    10 390 t of all gadoid landings (excluding Norway pout and blue whiting). That is 2.1 %
    (2019-23: 2.8 %). It equals the biomass share only if ling's exploitation rate equals the group's
    (~0.12/yr). Ling F is "above FMSY proxy" (LF=M/Lmean ~1.04), so the share is probably lower. The
    Skagerrak ling catch comes largely from the trench (Danish trawl 144 t in 3.a.20), deeper than
    most of the box.
  - Norwegian shrimp survey (Campelen trawl, 111-540 m, 3.a + 4.a East, Q1 2017-21, no q) [F/D]:
    ling 4.3 %, tusk 0.7 % of demersal gadoids. This is an upper bound for the box, because the
    survey is deeper than the box.
  - Trends: ling up and tusk strongly down in Skagerrak 2013-22 (Nordic Council 2024) [F]. No
    biomass, F or density estimate exists for either stock in the area (category 3) [F].
  - **Box share of ling + tusk: central 1.5-2 %, range 0.3-5 %; tusk <= 0.3 %.** Uncertainty: a
    factor of ~3 either way. No independent density exists, and every handle has a different gear,
    area or depth.

## 5. Scores of the added species (task 3)

Morphology and habitat from the FAO gadiform catalogue (Cohen et al. 1990) and FishBase
morphology pages [F]. Diet is a check only (evidence C; most diet values are FishBase encodings of
papers not read, [S]).

| Species | Mouth / barbel / habitat | Score min / central / max | Diet benthos share by mass (check) |
|---|---|---|---|
| Tusk | terminal, lower jaw shorter, barbel; rough ground 150-450 m | 0.5 / 0.75 / 1.0 | 22 % Norwegian Deep, n = 4 [S]; 38-89 % NW Atlantic [S] |
| Ling | terminal, subequal jaws, long barbel, strong teeth; rock | 0.2 / 0.4 / 0.6 | no % found; "primarily fish eaters" [A] |
| Pollack | superior, no barbel; pelagic | 0 / 0 / 0.1 | ~6 % (+30 % Pasiphaea/Pandalus), Deep [S] |
| Greater forkbeard | terminal, barbel, long pelvic rays; benthopelagic on mud | 0.4 / 0.6 / 0.9 | 63 % by volume, Biscay [F] |
| Poor cod (bib alike) | terminal, lower jaw shorter, barbel; mud 15-200 m | 0.5 / 0.7 / 1.0 | 54-88 % Irish Sea [S] |
| Fourbeard rockling | 4 barbels; sedentary on mud 20-650 m | 0.7 / 0.9 / 1.0 | polychaetes, crustaceans (Deep, no %) [A]; 70 % NW Atlantic [S] |
| Threebeard rockling | 3 barbels; <= 120 m | 0.7 / 0.9 / 1.0 | 48-96 % Mediterranean [S] |
| Silvery pout (*G. thori*) | superior, no barbel; mesopelagic schools | 0 / 0 / 0.1 | 0 % by weight, Deep, n = 97-208 [S] |
| Blue ling | superior, lower jaw longer; mud 350-500 m | 0 / 0.1 / 0.2 | 0.1-0.5 %, Deep [S] |
| Norway pout | superior; pelagic feeder | 0 / 0.1 / 0.2 | ~0-1 % (+1-9 % shrimps), Albert 1994, Deep [S] |
| Blue whiting | lower jaw longer, no barbel; oceanic | 0 / 0 / 0.1 | <= 0.9 % by weight [S] |

The main species keep the earlier scores: haddock 1, cod 0.5 / 1 / 1, whiting 0 / 0 / 0.3, saithe 0,
hake 0. Diet agrees with the morphology scores except tusk, where the in-area sample is n = 4.

**Norway pout and blue whiting exclusion.** It holds for this box: both have superior or projecting
lower jaws, and their in-area diets are ~0-1 % benthos. Their biomass, though, is not small.
Raw depth-stratified, Norway pout is 5 % and blue whiting 14 % of all gadiforms; Fraser-corrected,
Norway pout is 11-19 %. Blue whiting is a Q3, >100 m visitor of saithe size. Note: **Norway pout is
in neither model pool.** The pelagic_fish definition (growth-rate report section 3.4) does not list
it either.

## 6. Group total against the survey (task 4, order of magnitude)

Swept-area density of the 15 non-zooplanktivore gadiforms in the box (t/km2, depth-stratified,
pooled Q1/Q3) [D]:

| Scaling | Mean | Median | Q1 mean |
|---|---|---|---|
| Wing spread, q = 1 | 3.1 | 1.6 | 1.7 |
| Door spread, q = 1 (/3.75) | 0.8 | 0.4 | 0.46 |
| Fraser q (A_floor / A_sw_floor) | 8.5 / 6.0 | 3.4 / 3.2 | 2.5 / 2.9 |

- Norway pout adds 0.2 (raw) to 1.4 (Fraser), and blue whiting 0.5-2.7 t/km2.
- The extra species add < 1 % to the raw total and ~2-4 % after correction. They do not change
  the comparison.
- **The model's 0.69 t/km2 matches only door-spread scaling at q = 1.** It is 2-5 times below the
  wing-spread survey density, and ~5-12 times below the Fraser-corrected density.
- Whether that is catchability, the unsampled trench, seasonal visitors (saithe, blue whiting) or a
  model choice is outside this note. The Q3 means are carried by a few saithe hauls; medians are
  40-60 % of means.

## 7. Main uncertainties, in order

1. **The saithe weight**: season (Q1 1 %, Q3 30-40 %), one-haul leverage (54 %), and its
   catchability convention. Alone it spans phi 0.32-0.64.
2. **Depth coverage**: no GOV haul > 260 m. 9 % of the grid is filled by assumption, and the box
   rectangles over-sample < 100 m.
3. **Catchability**: q is available only from Fraser 2007 (availability mixed in, mixed swept-area
   bases, Q3 only). Walker 2017 per-species efficiencies were not obtained.
4. **Scores**: cod 0.5-1 and whiting 0-0.3 give +-0.03-0.05. The added species give < 0.01.
5. **Ling/tusk**: factor ~3, but immaterial for phi.

## Source status

| Source | Use | Status |
|---|---|---|
| [DATRAS web services](https://datras.ices.dk/WebServices/DATRASWebService.asmx) getHHdata, getCPUELength, getCAdata, NS-IBTS Q1/Q3 2021-25 | hauls, CPUE, L-W | Fetched full data (calc. date 20260625) |
| [WoRMS REST](https://www.marinespecies.org/rest/) AphiaClassificationByAphiaID | gadiform list | Fetched |
| FishBase summary pages | Bayesian L-W, morphology | Fetched |
| Fraser, Greenstreet & Piet 2007, ICES JMS 64:1800 | GOV q at length | Read via web render (no PDF) |
| Walker et al. 2017, ICES JMS 74:1448 | gear efficiency | Abstract only, supplement not obtained |
| NS-IBTS manual SISP 10 rev. 11; ICES advice ling/tusk 2025; WGDEEP 2026; NAFO SCR shrimp-survey reports 2015-21; Nordic Council Nord 2024:032; ICES Official Nominal Catches 2006-24; Mackinson & Daskalov 2007 | evidence B | Fetched, read |
| FAO Species Catalogue vol. 10 (Cohen et al. 1990); Sorbe 1977 | morphology, forkbeard diet | Fetched, read |
| Bergstad 1991, Albert 1993/1994, Du Buit 1978, Mattson 1981, Bergstad & Hareide 1996 | in-area diets | Abstracts or FishBase encodings only |
