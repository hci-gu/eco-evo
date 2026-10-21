# Gadoid-herring predator-prey cycles: real or a model artefact? And how large is gadoid predation on herring?

Scope note:
- **Literature** comes from a sub-search. Hislop 1996, Dickey-Collas et al. 2010, Lynam et al. 2017, WGSAM 2024, HAWG 2026
  and ICES CRR 219 were fetched in full text. Others were read as abstracts or snippets, as marked.
- **New computation**: I re-downloaded the ICES WGSAM North Sea SMS 2023 key-run outputs from GitHub (`partial_m2.out`,
  `summary.out`, `species_names.in`) and computed, for 1974-2022:
  1. the biomass-weighted M2 on herring by each predator;
  2. herring eaten per unit predator biomass;
  3. each gadoid's ration.
- Method of the computation:
  - Biomass eaten = sum over quarters and ages of Part.M2 × N.bar × west.
  - The M2 rate is that sum divided by the annual mean herring BIO.
  - Ration is the `ration` column × N.bar.
  - North Sea area 570 000 km2, as in the sibling notes.
  - Species index check: numbers 15-19 = hake, cod, whiting, haddock, saithe and 21 = herring, by line order in
    `species_names.in`. Confirmed from the Q3 2015 weights at age: herring ages 0-3 weigh 0.007 / 0.063 / 0.134 / 0.194 kg,
    cod age 3 weighs 2.27 kg, haddock age 3 0.59 kg, saithe age 5 1.71 kg.
  - The script was a one-off in the session scratchpad and was not retained.
- Model numbers come from `mareld_resume.txt` 184.12-184.13 and from `fgconfig/fg_library.yaml`
  (`gadoids_preys_on_pelagic_fish`: budget M2 0.2997 /yr, Type III). Both were read only.

## Takeaway
- **No gadoid-herring overshoot cycles are documented for the North Sea or Skagerrak.**
  - North Sea herring collapsed in the 1970s through recruitment overfishing.
  - Its 2000s slump was a larval-overwintering recruitment failure tied to the plankton regime shift.
  - Fishing is the main predictor of herring biomass in a 1964-2010 food-web model that includes the gadoid piscivores
    (Lynam et al. 2017).
  - The 1960s-70s "gadoid outburst" was hypothesised to run the *other* way, with herring suppressing gadoid recruitment,
    and even that hypothesis is not supported by the timing (Hislop 1996).
  - The western Baltic spring spawners, which are part of the Skagerrak herring, are "driven by recruitment" (HAWG 2026).
  - Where gadoids do regulate clupeids (Baltic cod-sprat, Barents cod-capelin), the switches came from fishing out the
    cod or from herring predation on capelin larvae. They did not come from a gadoid numerical response that overshoots.
- **Gadoid M2 on herring in SMS is 0.15 /yr (2013-2022), about 0.11-0.20 in other periods, with no trend.**
  - Saithe takes 60 % of it (0.09), whiting 0.03, cod 0.02, hake 0.007 and haddock 0.
  - This is at a gadoid density of about 1.9 t/km2 and herring 3.2-5.2 t/km2.
  - All predators together give 0.20-0.24 /yr.
  - The model's M2 on herring from gadoids is 0.5-1.2 /yr at 1-2 t/km2 gadoids: **3-8x SMS**. Per unit gadoid density it
    is about 7x.
- **The model's gadoid diet is the main quantitative misfit.** Herring is 40-52 % of the model's gadoid diet
  (1.8-2.45 BW/yr of herring).
  - In SMS, herring is about 5 % of the pooled ration with today's Skagerrak species mix (W1 weights): 0.13-0.15 BW/yr of
    herring out of a ration of 3.3-3.4 BW/yr.
  - With the box weights (W3) it is 6.5 %, or 0.18-0.22 BW/yr.
  - So the model's gadoids eat about **10-16x more herring per unit biomass** than SMS gadoids of the same mix.
  - Only saithe comes close, at 24 % of ration.
  - The 1991 stomach data give cod 2-50 % herring by weight depending on age and quarter, whiting 0-21 %, saithe 0-86 %
    (mostly ages 5+), and haddock 0-25 % in Q1 only.
- **Inference.** The overshoot needs two things: a pool that grows at r_max - F whenever food is ideal (no K;
  `density_dependence.md`), and a diet that makes herring the main ideal food. The literature supports neither for a
  haddock- and whiting-dominated pool. The herring decline of about 20 %/yr is the symptom.

## Cited Findings

### Herring regulation in the North Sea
- **Hislop JRG 1996.** "Changes in North Sea gadoid stocks". ICES J. Mar. Sci. 53:1146-1156, doi
  [10.1006/jmsc.1996.0140](https://doi.org/10.1006/jmsc.1996.0140); [VLIZ PDF](https://www.vliz.be/imisdocs/publications/ocrd/248341.pdf).
  (Fetched full text.)
  - "The large decrease in the abundance and biomass of North Sea herring and mackerel that occurred in the 1960s and 1970s
    could therefore have opened a 'window of opportunity' for the gadoids ... However, the timing of the various events
    does not fully support this hypothesis."
  - "the first very large haddock year class in 1962 cannot be explained by a reduction in predation by, or competition
    with, herring and mackerel."
  - "it is generally accepted that overfishing was ultimately responsible for the collapse of the herring stocks".
- **Cushing DH 1980.** "The decline of the herring stocks and the gadoid outburst". J. Cons. int. Explor. Mer 39:70-81,
  doi [10.1093/icesjms/39.1.70](https://doi.org/10.1093/icesjms/39.1.70). (Snippet only.) It is cited via Hislop for the
  point that the timing of gadoid recruitment "did not exactly coincide with the decline in herring recruitment".
- **Dickey-Collas M et al. 2010.** "Lessons learned from stock collapse and recovery of North Sea
  herring: a review". ICES J. Mar. Sci. 67:1875-1886, doi [10.1093/icesjms/fsq033](https://doi.org/10.1093/icesjms/fsq033);
  [VLIZ PDF](https://www.vliz.be/imisdocs/publications/ocrd/256670.pdf). (Fetched full text.)
  - The collapse "presents a clear example of recruitment overfishing resulting in stock collapse".
  - "Older herring (age 2+) are mainly eaten by gadoids and birds ... M2 has been very stable and only immediately after
    the collapse and in recent years has predation removed more 2+ herring than the fishery did."
  - Juvenile M2 is mainly from mackerel and horse mackerel.
  - "Recruitment is intrinsically variable and, in combination with variability in predation mortality on adults, can
    yield natural variation in stock abundance by at least a factor of 4."
- **Lynam CP et al. 2017.** "Interaction between top-down and
  bottom-up control in marine food webs". PNAS 114:1952-1957, doi [10.1073/pnas.1621037114](https://doi.org/10.1073/pnas.1621037114).
  (Fetched full text, PMC5338359.)
  - "Fishing mortality (F) was the most important predictor of sandeel biomass and, after the population delay term,
    also for herring biomass."
  - "top-down impacts of fishing have dominated change in the biomass of commercially exploited fish."
  - Herring appears as a driver of others: "Without fishing mortality constraining the biomass of herring, this stock
    would increase to the detriment of sprat, Norway pout, and, thus, saithe."
- **Payne MR et al. 2009.** "Recruitment in a changing environment: the 2000s North Sea herring recruitment failure".
  ICES J. Mar. Sci. 66:272-277, doi [10.1093/icesjms/fsn211](https://doi.org/10.1093/icesjms/fsn211). (Abstract.)
  - "an unprecedented 6 sequential years of poor juvenile production"; "the poor recruitment arises during the larval
    overwintering phase."
- **Corten A 2013.** "Recruitment depressions in North Sea herring". ICES J. Mar. Sci. 70:1-15, doi
  [10.1093/icesjms/fss187](https://doi.org/10.1093/icesjms/fss187). (Abstract.)
  - The 2002-2010 depression may be "predation by the large stock of adult herring on its own larvae". That is
    cannibalism, not gadoids.
- **ICES 2026, HAWG report.** ICES Scientific Reports 8:16, doi [10.17895/ices.pub.31424315](https://doi.org/10.17895/ices.pub.31424315).
  (Fetched full text.)
  - North Sea herring uses smoothed SMS M. Input M for 2025, ages 0-8+: 0.759, 0.549, 0.319, 0.285, 0.259, 0.240, 0.233,
    0.228, 0.217.
  - Western Baltic spring spawners: "This stock has been shown to be driven by recruitment (Trijoulet et al. 2025)".
- **ICES 2024, WGSAM, ICES Scientific Reports 6:13** (North Sea key run 2023).
  [Archimer PDF](https://archimer.ifremer.fr/doc/00873/98452/107912.pdf). (Fetched full text.)
  - "Herring M2 has been variable over the years but with no clear temporal trend."
  - For age 4, a higher M2 "is due to a higher partial M2 from saithe and harbour porpoise".
  - Saithe "otherfood is approx. 60% in 1981 but around 30% in 1991 because of the increasing contribution of herring
    and/or a decrease in available otherfood like Euphausiids."
- **Skagerrak / Kattegat.** "No multispecies model is currently available for this extended area of transition between
  the Baltic and the North Sea" (ICES 2023, Stock Annex her.27.20-24, doi
  [10.17895/ices.pub.23898678](https://doi.org/10.17895/ices.pub.23898678); fetched full text).
  - I found no study attributing any Skagerrak herring decline to gadoid predation.

### Contrasts where gadoids do act on forage fish
- **Casini M, Hjelm J, Molinero J-C, Lövgren J, et al. 2009.** "Trophic
  cascades promote threshold-like shifts in pelagic marine ecosystems". PNAS 106:197-202, doi
  [10.1073/pnas.0806649105](https://doi.org/10.1073/pnas.0806649105). (Abstract.)
  - In the Baltic, sprat was released after cod was fished out. That is a cascade caused by fishing, not a cycle.
- **Hjermann DØ, Ottersen G, Stenseth NC 2004.** "Competition among fishermen and fish causes the collapse of Barents Sea
  capelin". PNAS 101:11679-11684, doi [10.1073/pnas.0402904101](https://doi.org/10.1073/pnas.0402904101). (Abstract.)
  - "Both overexploitation and predation by herring ... can cause the population to collapse, whereas predation by cod
    ... [causes] a delay in the stock's recovery."
- **Gjøsæter H, Bogstad B, Tjelmeland S 2009.** "Ecosystem effects of the three capelin stock collapses in the Barents
  Sea". Mar. Biol. Res. 5:40-53, doi [10.1080/17451000802454866](https://doi.org/10.1080/17451000802454866). (Abstract.)
  - During capelin collapses cod showed more cannibalism and slower growth. The effects were milder later because
    predators switched to alternative food.
  - Inference: this is the nearest documented analogue of the model's overshoot. Even there the predator response was
    reduced growth and switching, not a starvation crash of the predator stock.

### Gadoid diets: herring share
- **ICES 1997.** "Database report of the stomach sampling project, 1991". ICES Cooperative Research Report 219, doi
  [10.17895/ices.pub.4626](https://doi.org/10.17895/ices.pub.4626). Fetched full text. Read as the weight % of
  *Clupea harengus* in total stomach contents; that reading of the denominator is the sub-search's, not verified.
  - **Cod**, ages 2-6+: 6-16 % in Q1-Q2, 15-47 % in Q3, 2-51 % in Q4. Age 1 is 0-1.5 %.
  - **Whiting**, ages 2-6+: 1-4 % in Q1, Q2 and Q4; 7-21 % in Q3.
  - **Saithe**:
    - ages 3-4: 0-24 %;
    - ages 5-9+: 4-86 % (Q1 48-73 %, Q3 8-81 %). Several cells rest on fewer than 30 stomachs.
    - Q1 Norway pout is 25-50 % of the diet.
  - **Haddock**:
    - Q1, ages 3-6+: 6-25 %.
    - Q2-Q4: 0-3 %.
    - This conflicts with SMS, which has haddock eating no herring. The cause is unknown; the Q1 signal may be herring
      spawn or a sampling artefact (speculation).
- Kappa F, Härkönen TJ, Freitas C, Carroll D, Hårding KC 2025. "Shifting Harbor Seal (*Phoca vitulina*) Diet May Reflect
  Ecosystem Changes in Skagerrak". Marine Mammal Science, doi [10.1111/mms.70069](https://doi.org/10.1111/mms.70069).
  (Abstract.)
  - Seal diet shows "a notable decline in cod and herring consumption" since 1977-1989.
  - This is the only Skagerrak-specific diet source found. No Skagerrak cod or saithe stomach study was found.

## Computed: SMS 2023 key run, herring as prey of the gadoids

**M2 on herring (/yr; biomass-weighted over ages, annual).**

| Year(s) | Herring B (kt; t/km2) | Cod | Whiting | Haddock | Saithe | Hake | Gadoids | All predators |
|---|---|---|---|---|---|---|---|---|
| 1974 | 352; 0.62 | 0.045 | 0.015 | 0 | 0.143 | 0.001 | 0.204 | 0.290 |
| 1980 | 320; 0.56 | 0.074 | 0.055 | 0 | 0.092 | 0.001 | 0.223 | 0.369 |
| 1991 | 1896; 3.33 | 0.027 | 0.021 | 0 | 0.063 | 0.001 | 0.111 | 0.165 |
| 2005 | 2988; 5.24 | 0.013 | 0.012 | 0 | 0.151 | 0.001 | 0.177 | 0.221 |
| 2015 | 2856; 5.01 | 0.028 | 0.024 | 0 | 0.087 | 0.007 | 0.147 | 0.193 |
| 2022 | 1819; 3.19 | 0.017 | 0.051 | 0 | 0.105 | 0.009 | 0.182 | 0.240 |
| mean 1974-1990 | | 0.056 | 0.027 | 0 | 0.110 | 0.001 | **0.194** | 0.281 |
| mean 1991-2005 | | 0.025 | 0.024 | 0 | 0.122 | 0.001 | **0.172** | 0.240 |
| mean 2013-2022 | | 0.020 | 0.029 | 0 | 0.092 | 0.007 | **0.149** | 0.206 |

Other predators of herring, 2013-2022: N. horse mackerel 0.01-0.02, harbour porpoise 0.011-0.014, mackerel 0.006-0.015,
and seabirds about 0.01-0.02 together.

Gadoid M2 on herring stays within 0.11-0.22 even though herring biomass varied 9-fold, 0.56-5.2 t/km2. Predation was
proportional to herring biomass rather than overshooting it.

**Predator side: herring eaten and ration per unit predator biomass (BW/yr), 2013-2022 mean.**

| Predator | Mean biomass (kt) | Herring eaten / B | Ration / B | Herring share of ration |
|---|---|---|---|---|
| Cod | 126 | 0.374 | 4.10 | 9.1 % |
| Whiting | 377 | 0.180 | 3.99 | 4.5 % |
| Haddock | 221 | 0 | 2.94 | 0 % |
| Saithe | 321 | 0.667 | 2.72 | 24.5 % |
| Hake | 59 | 0.291 | 5.63 | 5.2 % |

In 2018-2022 the herring eaten / B values were cod 0.305, whiting 0.141, saithe 0.562 and hake 0.231.

Pooled with the Skagerrak weights of `ices_gadoid_f.md`:
- W1 (cod 0.058, haddock 0.543, whiting 0.267, saithe 0.121, hake 0.011): herring 0.154 BW/yr out of a ration of
  3.29 BW/yr, a share of **4.7 %**. 2018-2022: 0.126 of 3.38, 3.7 %.
- W3, box rectangles (0.051 / 0.356 / 0.387 / 0.192 / 0.012): 0.220 of 3.39, **6.5 %**. 2018-2022: 0.181 of 3.45, 5.2 %.
- Pooled SMS gadoid density in 2013-2022 was 1104 kt / 570 000 km2 = 1.94 t/km2. Gadoid M2 on herring per unit gadoid
  density was 0.149 / 1.94 = **0.077 /yr per (t/km2)**.

**Model, for comparison** (resume 184.13, and the task statement):
- Herring 2.45 / 2.30 / 1.83 BW/yr, benthos 1.42 / 1.09 / 1.12, zooplankton 0.86 / 1.16 / 1.63.
- Herring share 52 % / 50 % / 40 %. Total 4.6-4.7 BW/yr.
- M2 on herring from gadoids 0.5-1.2 /yr at 1-2 t/km2, i.e. about 0.5-0.6 /yr per (t/km2).
- Library budget value: 0.2997 /yr (`gadoids_preys_on_pelagic_fish`, from Ecopath NS 1991).

| Quantity | SMS 2013-2022 | Model | Ratio model/SMS |
|---|---|---|---|
| Herring eaten per gadoid biomass | 0.15-0.22 BW/yr (W1-W3) | 1.8-2.45 | 8-16x |
| Herring share of ration | 4.7-6.5 % | 40-52 % | 7-10x |
| Total ration | 3.3-3.4 BW/yr | 4.6-4.7 | 1.4x |
| Gadoid M2 on herring per (t/km2) gadoids | 0.077 | 0.5-0.6 | 7x |

## Inferences

1. **Is the cycle real? Not in the available evidence.** Confidence moderate-high.
   - The evidence against a gadoid-driven herring cycle in the North Sea is consistent across sources:
     - herring M2 has no trend over 49 years;
     - predation tracks herring biomass;
     - recruitment and fishing explain herring;
     - gadoid and herring stock histories since 1960 are explained by fishing and recruitment regimes.
   - The Skagerrak has no multispecies model. Absence of evidence there is weaker evidence than in the North Sea.
2. **Herring in the model should be regulated mainly by its own recruitment and the fishery.** Gadoid predation should
   take about 0.15 /yr from it, or about 0.2 /yr from all fish predators, at roughly today's relative densities.
   - The library's 0.30 is the Ecopath-1991 value. It is 2x SMS but the right order.
   - The realised 0.5-1.2 is 3-8x SMS.
3. **Why so much herring? This is my reading of the library, not a literature result.**
   - Since resume sections 147-148 the gadoid Holling ceilings are energy-limited (`h = 130 E_prey / 4500`), so at
     saturation every prey pays about the same energy per tick. That change was made to *reduce* the herring preference,
     and herring is no longer worth more per stomach.
   - Two things remain that favour herring:
     1. Herring is abundant relative to the gadoids. The model has about 3-7 t/km2 of herring against 0.7-1.9 t/km2 of
        gadoids, a ratio of 2-10. The SMS North Sea ratio is about 1.7-2.4 (herring 3.2-4.7 against gadoids 1.94).
     2. The attack rate a (0.04) is the same on every pair. So every gadoid can feed on herring like a saithe.
   - Real haddock and whiting are limited by habitat, gape and search mode (benthic feeders, and whiting on
     sandeel/Norway pout/crustaceans), which the pool does not represent.
   - The literature cannot set a parameter here. It gives two checks for the model: herring at about 5 % of the pooled
     ration, and gadoid M2 on herring at about 0.15 /yr.
4. **Saithe is the only gadoid whose herring diet resembles the model's**, at 25 % of ration and higher at large sizes.
   - Saithe is 9-19 % of the pool's survey biomass, is declining, and is semi-pelagic and migratory. In the survey it
     comes in single big hauls (`ices_gadoid_f.md`).
   - A pool parameterised on saithe-like piscivory but haddock-like abundance is the combination that generates the
     overshoot.
5. **Consequence for the fix ranking (`recommendation.md`).** Adding K stops the gadoids compounding. It does not by
   itself bring gadoid herring consumption down to SMS levels. With gadoids held at the start level (resume 184.12) the
   model's herring rose +23 %/yr and then levelled off, and its predation loss was 0.07-0.13 %/d (0.26-0.47 /yr), still
   about 2x SMS.
   - So K is necessary, and the diet share is a secondary calibration target.
