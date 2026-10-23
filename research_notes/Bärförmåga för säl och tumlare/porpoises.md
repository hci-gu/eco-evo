# Harbour porpoise (Phocoena phocoena): does the model group need a regional carrying capacity K?

Written 2026-10-23 for the Mareld / Poseidon Nord model. Nothing in the code, the library, the tests or
`mareld_resume.txt` was edited. This note builds on these earlier notes and does not repeat them:
- `research_notes/Täthetsberoendets styrka per grupp/mammals_and_seabirds.md` (section 3)
- `research_notes/Litteraturbaserad growth rate per FG/mammals.md`
- `research_notes/Vintermetabolism sill och tumlare/porpoise_winter.md`

## 0. Conventions

**Source grades.**
- **[F]**: I read the full text (PDF or HTML), or at least the cited table or section of it.
- **[A]**: I read the abstract only (OpenAlex or the publisher page).
- **[S]**: I saw only a search snippet or search summary. These numbers are not verified.
- **[2]**: secondary. The claim is cited inside a source I read, and the original was not opened.
- **[N]**: not found, or could not be accessed.

**Labels.**
- "(Own calculation)" marks arithmetic I did. Every input is shown.
- "(Own judgement)" marks inference.

**Model facts used.** These come from the assignment, `fgconfig/fg_library.yaml`, `mareld2.yaml` and `coordinates.txt`.
- **Box.** 60 x 60 km = 3,600 km2, from 57.81 to 58.31 N and 10.07 to 11.15 E, on the southern slope of the Norwegian Trench.
  - Depth: minimum 43 m, maximum 533 m, mean 168 m.
  - Cells shallower than 100 m: 832. Shallower than 200 m: 2,600 of 3,600.
  - Mean depth falls from about 270 m in the northern sixth of the box to 86 m in the southern sixth, which is the strip just north of Skagen (Own calculation, from the depth array).
- **Porpoise biomass.**
  - Start biomass 12 t: `inference_initial_biomass: 12`, training range 6-18 t. That is 3.33 kg/km2, or 0.0033 t/km2.
- **Porpoise rates.**
  - r_max 0.10/yr and M1 0.10/yr.
  - Gillnet bycatch 0.017/yr at mean effort, carried on the bottom-trawl map (resume section 165).
  - No predator.
- **Porpoise energetics** (library comments).
  - Indivisible unit: one animal = 50 kg.
  - Eat cost = field metabolic rate (FMR) = 0.36 MJ/kg/day.
  - Break-even ration on clupeids: 5.3 % of body mass per day.
- **K mechanism.** The group loses r_max/1460 x B_mean/K per tick, so it grows at r_max(1 - B/K) - F. The loss r_max x B/K acts at **every** B, not only near K.

---

## 1. B_today: is 12 t the porpoise biomass of the box?

### 1.1 Which SCANS block covers the box

**SCANS-IV (2022).**
- Two aerial blocks cover the eastern Skagerrak and northern Kattegat. Their combined area of 23,114 km2 is about the area of SCANS-III block P1 (Own calculation).
  - NS-L: 15,135 km2, primary effort 913.7 km.
  - NS-M: 7,979 km2, primary effort 905.4 km.
  - Source [F]: [Gilles, A. et al. 2023. Estimates of cetacean abundance in European Atlantic waters in summer 2022 from the SCANS-IV aerial and shipboard surveys. Final report, Univ. St Andrews / ITAW, Table 1](https://tethys.pnnl.gov/sites/default/files/publications/Gilles_etal_2023.pdf)
- The block map in Gilles et al. (2023, Fig. 1) is a low-resolution raster [F, figure viewed].
- A Danish team report has a clearer map of the same blocks (Fig. 2, marked "TEMPORARY DRAFT"; used here for block geometry only) [F, figure viewed]. [DCE Notat N2022/74, 2022: SCANS-IV Danish team field report (cruise lead S. Sveegaard)](https://dce.au.dk/fileadmin/dce.au.dk/Udgivelser/Notater_2022/N2022_74.pdf)
  - It shows NS-L as the eastern Skagerrak north of a line from the tip of Skagen east-north-east to the Swedish coast, reaching up to the Oslofjord.
  - It shows NS-M as the northern Kattegat south of that line.
- **The box (south edge 57.81 N, about 8 km north of Skagen at 57.74 N) lies in NS-L** (Own judgement from the maps). At most a sliver of its south-east corner falls in NS-M.
- The same report notes that porpoise sightings cluster along the NS-L/NS-M boundary just north of Skagen. On "just three transects in NS-M, 99 porpoises were observed" [F].

**SCANS-III (2016).**
- Ship block 1, "Skagerrak/Kattegat" (23,451 km2), had too little ship effort. It "was also surveyed by air (block P1)". P1 is the same area [F]. [Hammond, P.S. et al. 2021. Estimates of cetacean abundance in European Atlantic waters in summer 2016 from the SCANS-III aerial and shipboard surveys (revised), Tables 1-2](https://scans3.wp.st-andrews.ac.uk/files/2021/06/SCANS-III_design-based_estimates_final_report_revised_June_2021.pdf)

**SCANS-II (2005) and SCANS (1994).**
- SCANS-II "covered the [Belt Sea] AU and adjacent waters in one large block" (Gilles et al. 2023 [F]). No Skagerrak-only density exists for 2005.
- I did not extract the SCANS-II or 1994 block densities [N].

### 1.2 Densities (summer, aerial line transect)

| Survey, block | Animals/km2 | CV | Abundance (95 % CI) | Grade |
|---|---|---|---|---|
| SCANS-IV 2022, NS-L (box block) | 0.519 | 0.354 | 7,853 (3,431-14,608) | [F] Gilles et al. 2023, Table 6 |
| SCANS-IV 2022, NS-M (N Kattegat) | 0.924 | 0.355 | 7,375 (3,136-13,944) | [F] same |
| SCANS-III 2016, P1 (Skagerrak/Kattegat) | 1.077 | 0.302 | 25,367 (10,114-41,642) | [F] Hammond et al. 2021 |
| SCANS-IV 2022, North Sea AU mean | 0.55 | 0.17 | 338,918 | [F] Gilles et al. 2023 (abundance and CV in the trend text); 0.55 [2] via `mammals_and_seabirds.md` |

Reading of the table:
- NS-L and NS-M pooled give 0.659/km2 in 2022, against 1.077 in 2016 (Own calculation). With CVs of 0.30-0.35 both values sit inside each other's intervals, so this is not evidence of a decline (Own judgement).
- Gilles et al. (2023) fit no trend for this sub-area. SCANS-III fitted +1.24 %/yr (95 % CI -39 to 67 %, p = 0.81) for the wider "Skagerrak/Kattegat/Belt Seas" area [F].

**Static acoustic monitoring.**
- SAMBAH covered the Baltic Proper only. It is irrelevant to the box (`porpoise_winter.md`, section 6).
- I found no offshore C-POD series for the Swedish Skagerrak [N].
- The nearest are:
  - six coastal C-PODs on the Norwegian Skagerrak coast (2023-24, a master's thesis; more detections in spring and at night) [S]: [UiA thesis](https://uia.brage.unit.no/uia-xmlui/handle/11250/3142561)
  - C-PODs at Kullaberg/Skälderviken in the southern Kattegat, Belt Sea population [S]: [Mattsson 2017, SLU](https://stud.epsilon.slu.se/10223/1/Mattsson_E_170608.pdf)
- Neither gives a density for the box.

### 1.3 Density gradient inside and around the box

**Satellite tracking (Sveegaard et al. 2011) [F, paper read in the thesis reprint].**
- 64 porpoises were tagged from 1997 to 2007. The 24 tagged at Skagen "never moved south of Anholt".
- The all-year high-density areas for the Skagerrak group are "a large area at the border between northern Kattegat and the southeastern Skagerrak (hereafter called the tip of Jylland) as well as three smaller areas along the Norwegian trench".
- In spring and summer the Skagerrak animals "stay close to the tip of Jylland". In autumn and winter they move "further west into the northeastern North Sea (although high porpoise density still remains in Skagerrak)". The autumn-winter high-density areas include "the tip of Jylland, an area along the slope to the deep Norwegian Trench".
- Source: [Sveegaard, S., Teilmann, J., Tougaard, J. et al. 2011. High-density areas for harbor porpoises identified by satellite tracking. Mar. Mamm. Sci. 27: 230-246. doi:10.1111/j.1748-7692.2010.00379.x](https://doi.org/10.1111/j.1748-7692.2010.00379.x), full text read in [Sveegaard 2011, PhD thesis, Aarhus Univ.](https://www2.dmu.dk/pub/phd_sign.pdf)

**Porpoises and herring (Sveegaard et al. 2012).**
- In Kattegat/Skagerrak, "harbour porpoise distribution was best explained solely by the distribution of herring" [A]. [Sveegaard, S., Nabe-Nielsen, J., Stæhr, K.-J. et al. 2012. Mar. Ecol. Prog. Ser. doi:10.3354/meps09959](https://doi.org/10.3354/meps09959)
- The thesis summary adds that porpoise and herring densities were highest "in frontal zones, namely along the Norwegian Trench and the turbulent transition between Kattegat and Skagerrak". "no significant correlation of depth to porpoise distribution was found in this study" [F, thesis].

**IMMA designation.**
- The "Eastern North Sea and Southern Slopes of the Norwegian Trench Complex" IMMA (42,648 km2) includes, from tracking, high-density areas "extending from the northern Kattegat, southern Skagerrak along the southern slope of the Norwegian Trench". Porpoises use the area at higher density than surrounding waters "throughout the year" [F, web page]. [IUCN-MMPATF IMMA factsheet](https://marinemammalhabitat.org/?p=9476)

**Inference** (Own judgement).
- The box sits on both habitat features that the tracking data single out: the tip-of-Jutland front along its southern edge and the trench slope.
- So the box density is more likely at or above the NS-L block mean than below it.
- In the central and southern North Sea, seasonal habitat models select depth, distance to shore and distance to sandeel grounds, SST, fronts and day length as density predictors [A]. That model area does not include the Skagerrak. [Gilles, A., Viquerat, S., Becker, E.A. et al. 2016. Seasonal habitat-based density models for a marine top predator, the harbor porpoise, in a dynamic environment. Ecosphere 7: e01367. doi:10.1002/ecs2.1367](https://doi.org/10.1002/ecs2.1367)
- I found no published within-block gradient with depth or distance offshore for the Skagerrak. The box is 25-30 km off the Swedish coast, so a coastal-versus-offshore gradient is not the issue here.
- **Seasonality.** SCANS densities are July values. In winter part of the Skagerrak group moves west, but the tip of Jutland and the trench slope stay high-density areas.
- A monthly box-level density would need the Waggitt et al. (2020) 10 km monthly maps, which I did not open [N]. [Waggitt, J.J. et al. 2020. J. Appl. Ecol. 57. doi:10.1111/1365-2664.13525](https://doi.org/10.1111/1365-2664.13525)

### 1.4 Population mean body mass

**Literature.**
- Adult females 55-65 kg, adult males 46-51 kg (Lockyer 2003, via `mammals.md` [S there]).
- Wild animals in the FMR study were 26-76 kg; Rojano-Donate et al. (2018) imply about 62 kg for adults (`porpoise_winter.md`).
- Captive adults: about 41-56 kg seasonally (Kastelein, `porpoise_winter.md`).

**Stable age distribution** (Own calculation).
- Inputs: the mass-at-age schedule assembled in `mammals.md` (5, 28, 38, 45, 50, 54, 57, then 58 kg) and survival 0.80 / 0.88 / 0.93 for age 0 / 1 / adults.
- Mean mass in a stationary population is 47.5 kg. With a growing population (lambda 1.04) it is 44.3 kg.
- With Lockyer- or Kesselring-like survival (0.75-0.80 / 0.85-0.87) it is 42-43 kg.
- Counting calves at a July weight of about 15 kg instead of 5 kg adds about 1-1.5 kg.
- The 58 kg plateau is female-biased. With a sex-averaged plateau of about 53 kg, every value drops by about 3 kg.
- **Central 45 kg, range 40-55 kg.** The library's 50 kg unit sits inside this range.

### 1.5 Conversion to the 3,600 km2 box (Own calculation; animals = density x 3,600)

| Density source | Animals/km2 | Animals in box | t at 40 kg | t at 45 kg | t at 55 kg |
|---|---|---|---|---|---|
| SCANS-IV NS-L 2022 (point estimate) | 0.519 | 1,868 | 74.7 | **84.1** | 102.7 |
| NS-L lower 95 % limit (3,431/15,135) | 0.227 | 816 | 32.6 | 36.7 | 44.9 |
| NS-L upper 95 % limit (14,608/15,135) | 0.965 | 3,475 | 139.0 | 156.4 | 191.1 |
| SCANS-IV NS-L + NS-M pooled | 0.659 | 2,372 | 94.9 | 106.7 | 130.4 |
| SCANS-IV NS-M 2022 | 0.924 | 3,327 | 133.1 | 149.7 | 183.0 |
| SCANS-III P1 2016 | 1.077 | 3,877 | 155.1 | 174.5 | 213.2 |
| North Sea AU mean 2022 | 0.55 | 1,980 | 79.2 | 89.1 | 108.9 |
| **Model start, 12 t** | 0.061-0.083 | 218-300 | 12 | 12 | 12 |

**Does 12 t hold? No.**
- The model holds about 0.07 animals/km2. The block it sits in held 0.52 in 2022 and the same area held 1.08 in 2016.
- The survey point estimate is **about 7x** the model start: 84 t vs 12 t at 45 kg.
- Even the pessimistic corner, the lower 95 % limit at 40 kg (33 t), is **2.7x** the model start.
- On the 2016 value the ratio is about 13-18x.
- These densities are July values. A winter density in the box is probably lower but not 7x lower (section 1.3), and I found no measurement of it [N] (Own judgement).
- **Model energetics check** (Own calculation).
  - At FMR 0.36 MJ/kg/day and 6.75 MJ/kg prey, porpoises eat about 19.5 t of prey per t of porpoise per year.
  - So 84 t of porpoises would eat about 1,640 t/yr, against a pelagic pool of about 11,500 t (3.2 t/km2).
  - 12 t of porpoises eat about 234 t/yr.
  - The real porpoises in the box are therefore a measurable but not dominant consumer of the pelagic pool.

## 2. Where the 12 t came from

- The resume has **no SCANS hit at all**. `grep -n SCANS mareld_resume.txt` returns nothing.
- The start value appears in `mareld2.yaml` from the commit that introduced range-based initial biomass (`d17b4dd`, "Add Inference tab, --resume, range-based initial biomass ...").
- No literature derivation of 12 t was found.
- I also checked the Ecopath route, since the porpoise diet was fitted to the North Sea 1991 Ecopath model (resume section 151). That model's toothed-whale group (harbour porpoise plus white-beaked and white-sided dolphins) has B = 0.017 t/km2 (Mackinson & Daskalov 2007, via `mammals.md` [2]). That is 61 t in the box (Own calculation), five times the 12 t start. So 12 t is not the Ecopath value either. The Ecopath figure is itself of the same order as the survey-based 75-103 t.
- The earlier note `mammals_and_seabirds.md` already flagged that B_ref is "about 0.06-0.11 of the survey-based biomass" and "deserves a separate check of how 12 t was chosen".
- **This is a finding in its own right:** the porpoise start biomass is not anchored to any survey or to the Ecopath biomass, and the survey says it is about 7x too low (range 2.7-18x).

## 3. Status of the North Sea assessment unit (the box's population)

**Management unit.**
- The box (58.1 N, Skagerrak) belongs to the **North Sea AU**, which runs "from the northern Kattegat through the Skagerrak to the whole North Sea". The Belt Sea AU starts south of 56.95 N in the Kattegat.
- Source: Owen et al. 2024 [2, read in full for `mammals_and_seabirds.md`, not re-read here] and SCANS-IV Fig. 2 [F].
- **Belt Sea numbers (Owen et al. 2024, K = 50,000, mPBR 24/yr) do not apply to the box.**

**Abundance and trend.**
- 289,200 (1994), 355,400 (2005), 345,400 (2016) [2, OSPAR QSR 2023 via `mammals_and_seabirds.md`] and 338,918 (2022; CV 0.17) [2, same note].
- SCANS-IV trend +0.51 %/yr (95 % CI -1.14 to 2.20, p = 0.32).
- The four surveys have 80 % power to detect a 0.88 %/yr decline (Gilles et al. 2023, Table 30, Fig. 8b) [F].
- The authors warn that "these results should not be interpreted as providing evidence that there have been no declines" [F].

**Distribution shift.**
- From the north-west North Sea in 1994 to the south in 2005 and 2016, maintained in 2022. The highest densities in 2022 were in the central and south-western North Sea (Gilles et al. 2023 [F]; search summary of the same).
- So within a flat total, local densities have moved by factors of several over two decades (Own judgement).

**Bycatch (OSPAR QSR 2023) [F].**
- N_best 345,000 (CV 0.18).
- RLA threshold 1,622/yr, under the objective "recover to or be maintained at 80% of carrying capacity" with probability 0.8 over 100 years.
- 2020 bycatch estimates:
  - 5,974 (95 % CI 3,176-10,739)
  - 1,627 (922-3,325) without one country's skewed data
  - "both estimates result in the threshold being exceeded"
- As fractions of N_best: 1.73 %/yr and 0.47 %/yr (Own calculation). The model's 0.017/yr is the upper figure.
- On K, OSPAR does not report the depletion estimate. It states that "Current depletion levels of marine mammal populations are largely unknown".
- Source: [OSPAR QSR 2023, Marine mammal bycatch](https://oap.ospar.org/en/ospar-assessments/quality-status-reports/qsr-2023/indicator-assessments/marine-mammal-bycatch/)

**ASCOBANS.**
- Interim objective: "to restore and/or maintain stocks/populations to 80% or more of the carrying capacity".
- Total anthropogenic removals above 1.7 % of the best population estimate are "unacceptable".
- Precautionary aim: bycatch below 1 % [S; quoted in Genu et al. 2021, which I read: "a total anthropogenic removal above 1.7% of the best available ..." [F]].
- The ASCOBANS North Sea Conservation Plan itself was not opened [N].

**Is the population at, below or near K?**
- **No source states it.** K appears only as a management reference point (80 % of K), never as an estimate for this AU.
- DEPONS calibration treats the North Sea population as being at equilibrium (section 4) [F].
- The earlier note's equilibrium analysis (x = N/K of about 0.83-0.98 at r = 0.10, 0.57-0.95 at r = 0.04) is an inference from a flat trend under bycatch, not a measurement.

## 4. Published K or density-dependence parameters in porpoise population models

| Model | How density dependence enters | K value | Grade |
|---|---|---|---|
| RLA (OSPAR, North Sea AU; Genu et al. 2021) | Pella-Tomlinson, N_t = N_{t-1} - C_{t-1} + r N_{t-1}(1 - D_{t-1}^2). That is theta = 2: the PDF text renders the exponent as "D_{t-1} x D_{t-1}", and I did not confirm it in the Stan code, with D = N/K estimated from survey and bycatch series. Priors: uniform on r (upper bound 0.10, "given recent evidence on the maximum growth rate", Forney et al. 2021) and uniform on D_T in [0,1]. D_0 = 1: the population is assumed at K before removals. Operating model: density-dependent birth rate, MNPL about N(0.6, 0.05), initial depletion 0.3-0.9 | K = 500,000 is a **simulation constant** for tuning, not an estimate | [F] |
| PBR (Wade 1998; Owen et al. 2024 for the Belt Sea) | Generalised logistic operating model, R_max 0.04 default, MNPL about 0.5-0.6 of K | Belt Sea: K = 50,000, "rough" = the 1994 estimate. Nothing for the North Sea | [F] (Genu); [2] (Wade) |
| SCANS-II management procedures (Winship & Hammond) | Tuneable procedures that set bycatch limits, of Pella-Tomlinson type (inferred from the SCANS-II framework) | Example limits per block were tabulated; no K value found | [S] |
| DEPONS (Nabe-Nielsen et al. 2014, 2018; Cordier et al. 2026, DEPONS 4.0) | **No explicit K.** Agents with energy budgets forage on food patches that recover after depletion. Survival probability depends on energy storage (parameter beta). The population "stabilise[s] at a carrying capacity" that emerges from food | Tuned, not estimated: the food-to-energy factor IR2EA was adjusted so the North Sea landscape settles at about 16,000 agents (Kattegat about 320), for computing time. This "reflects the equilibrium population dynamics observed for harbour porpoises in the North Sea (Gilles et al., 2023)" | [F] SR705; [A] 2014, 2018 |
| iPCoD (Harwood, King, Sinclair) | Stochastic Leslie matrix. Applicants list "lack of density dependence" as a known limitation | none | [S] |

**Sources for the table.**
- [Genu, M., Gilles, A., Hammond, P.S. et al. 2021. Evaluating strategies for managing anthropogenic mortality on marine mammals: an R implementation with the package RLA. Front. Mar. Sci. 8: 795953. doi:10.3389/fmars.2021.795953](https://doi.org/10.3389/fmars.2021.795953) [F]
- [Cordier, A., Gallagher, C.A., Weller, F.G. & Nabe-Nielsen, J. 2026. Calibration of DEPONS 4.0. DCE Scientific Report 705](https://dce.au.dk/fileadmin/dce.au.dk/Udgivelser/Videnskabelige_rapporter_700-800/SR705.pdf) [F, sections 2-3]
- [Nabe-Nielsen, J., Sibly, R.M., Tougaard, J. et al. 2014. Effects of noise and by-catch on a Danish harbour porpoise population. Ecol. Model. (2014; online 2013). doi:10.1016/j.ecolmodel.2013.09.025](https://doi.org/10.1016/j.ecolmodel.2013.09.025) [A]. The population is "sensitive ... to the speed at which food recovers after being depleted".
- [Nabe-Nielsen, J., van Beest, F.M., Grimm, V. et al. 2018. Predicting the impacts of anthropogenic disturbances on marine populations. Conserv. Lett. 11: e12563. doi:10.1111/conl.12563](https://doi.org/10.1111/conl.12563) [A]

**How these models obtain density dependence.**
- Either it is **assumed** in a generalised-logistic form with K as a free scale, with depletion estimated or drawn from a prior (RLA, PBR).
- Or it **emerges from food** (DEPONS: patch depletion and recovery, and energy-dependent survival).
- None estimates a porpoise K from data. None measures density dependence in porpoise vital rates; the earlier note's statement "No study found tests density dependence in porpoise vital rates" stands.
- The empirical pathway is food: pregnancy follows prey energy density, and starvation follows prey failure (`mammals_and_seabirds.md`, section 3).

## 5. Is the box a sample of a population regulated beyond it?

**Share of the population** (Own calculation).
- At the NS-L density the box holds about 1,870 animals. That is 0.55 % of the North Sea AU (338,918) and about 24 % of the NS-L block (7,853).

**Range and movement** [F, Sveegaard et al. 2011].
- The Skagerrak group ranged over the Skagerrak and north-eastern North Sea, with a mean swim rate of 0.85 km/h.
- 0.85 km/h is about 20 km/day, so an animal could cross the 60 km box in about 3 days (Own calculation).
- Home ranges of immatures are "up to four times larger" than those of matures in the Skagerrak group. Kernel areas are given only in a figure, which I did not digitise [N for the km2 values].
- Seasonal net movement is westward into the north-eastern North Sea in autumn and winter, "not a coordinated migration but ... a gradual net movement" [F, thesis].

**Genetics and management** (Owen et al. 2024 and SCANS-IV [F]).
- The North Sea population is continuous from the northern Kattegat to the whole North Sea.
- The box is not a closed unit.

**Inference** (Own judgement).
- **Yes, the box is a small, open sample of a population of about 340,000.** Whatever regulates that population acts at the scale of the North Sea:
  - prey fields (herring, sprat, sandeel)
  - bycatch over the whole AU
  - redistribution, like the 1994-2005 shift
- This is the same argument used for the pelagic pool (stock-scale compensation plus net exchange).
- **But the conclusion differs from the pelagic case.** The porpoise's local regulator is food, the porpoise follows food (herring) in and out of the box, and the model already moves and starves porpoises on food.
- What a "regional K" would represent for the pelagic pool, namely compensation that the box cannot represent, is for porpoises represented: by food tracking inside the box, and by the off-grid migration pool (`--migration`) when it is on.

## 6. Calculation: candidate K values and what they would do

### 6.1 The assignment's candidate, K = 12/(1 - 0.017/0.10) = 14.5 t (Own calculation, checked)

- The arithmetic is right: 1 - 0.17 = 0.83, and 12/0.83 = 14.46 t = 0.0040 t/km2.
- **The premise fails.** It needs 12 t = B_today. The survey says B_today in the box is about 37-190 t, central 84 t (section 1.5).
- **Mechanics in the model.**
  - At 12 t the K loss would be 0.10 x 12/14.5 = **0.083/yr**. This competes with the net r_max of 0.10 (M1 is already inside the derived g), leaving 0.017 for bycatch: equilibrium at 12 t by construction, before any starvation.
  - At 16 t, the level reached in the 15-year run, it would be 0.110/yr, above r_max.
  - It binds from tick 1. It would cap the group at about 12 t and stop the rise to 16 t that followed the pelagic pool.
  - It would double-count the food regulation that already causes 39-48 % of the losses.
- **Verdict:** reject this candidate.

### 6.2 A survey-anchored K, from K = B_today/(1 - h/r) with theta = 1 (the model's form) (Own calculation)

Equilibrium x = N/K under removal rate h:

| r | h | x (theta = 1) | x (theta = 2, RLA form as rendered in Genu et al.) | K/B_today (theta = 1) |
|---|---|---|---|---|
| 0.10 | 0.0047 | 0.953 | 0.976 | 1.05 |
| 0.10 | 0.017 | 0.830 | 0.911 | 1.20 |
| 0.04 | 0.0047 | 0.883 | 0.939 | 1.13 |
| 0.04 | 0.017 | 0.575 | 0.758 | 1.74 |

With B_today = 84.1 t (NS-L, 45 kg):

| r | h | K (t) | K (t/km2) |
|---|---|---|---|
| 0.10 | 0.0047 | 88 | 0.0245 |
| 0.10 | 0.017 | 101 | 0.0281 |
| 0.04 | 0.0047 | 95 | 0.0265 |
| 0.04 | 0.017 | 146 | 0.0406 |

The full range over masses (40-55 kg) and the NS-L confidence interval is K of about 34-330 t.

**What a K of 88-146 t would do in the model as it stands (12-16 t; the engine uses r_max 0.10):**
- Loss = 0.10 x B/K:
  - 0.0136/yr at 12 t and 0.018/yr at 16 t for K = 88 t
  - 0.0118 and 0.0158/yr for K = 101 t
  - 0.0082 and 0.011/yr for K = 146 t
- This is not zero. The K term in this engine is a linear density-dependent loss that acts at any B: about 1-2 %/yr, the same size as the bycatch.
- Over 3 years it would remove about 3-5 % of the group (Own calculation).
- It would not **set** the level: food sets it at 12-16 t, far below K. It would only shift the food-limited equilibrium slightly down.
- In the sense of "the population reaches K", it **would not bind within 3 years** or at any level seen in the runs.

**Consistency caveat** (Own judgement).
- If r = 0.04 is used to compute K, the engine still applies r_max = 0.10 in r_max(1 - B/K).
- The equilibrium the engine produces (x = 1 - h/r_max) is then not the one K was derived for.
- A K derived this way is only internally consistent if the r in the formula equals the library r_max.

### 6.3 r_max: is 0.10/yr too high?

- `mammals.md`: porpoise-specific evidence points to about 0.09-0.10/yr:
  - Lockyer 9.4 %
  - Caswell et al. median 10 %
  - Forney et al. 2021: 9.6 %/yr observed after bycatch ended
- Wade's 0.04 is a deliberately conservative management default for cetaceans in general.
- OSPAR's RLA sets the upper bound of the r prior to 0.10 citing Forney et al. 2021 (Genu et al. 2021 [F]). The management side now treats 0.10 as the plausible maximum, not as an overestimate.
- **The North Sea data cannot discriminate.** A flat trend (+0.5 %/yr, not significant) under 0.5-1.7 %/yr bycatch fits r = 0.10 near K (x 0.83-0.98) and r = 0.04 further below K (x 0.58-0.94) equally well (Own judgement).
- **Inside the model, r_max matters less than it seems** (Own judgement).
  - Growth is reserve-funded and the group is food-limited. Realised growth is set by intake, not by r_max.
  - r_max only sets the derived growth coefficient g. That is the ceiling paid at s*, not the realised rate.
  - The run evidence (-4 to -8 %/yr when the pool was low, rising when it rose) shows the realised rate is food-driven.
- **r and K interact only if a K is added.** For a 14.5 t-type K, which binds, the outcome is r-sensitive. For a survey-anchored ceiling, which does not bind, it is barely r-sensitive.

## 7. Inference (Own judgement)

1. **The literature gives no K for this population.**
   - Published porpoise models either assume a logistic K as a free scale (RLA, PBR) or let food produce the ceiling (DEPONS).
   - No study measures density dependence in porpoise vital rates.
   - The documented pathway is food → condition → pregnancy and starvation, and the model represents it explicitly.

2. **The model porpoise is already food-regulated, not runaway.**
   - Gadoids needed K because they grew at r_max - F without a brake. That is not the porpoise case.
   - The porpoise case, from the runs: starvation 39-48 % of losses, decline when the pool is low, rise when it rises.

3. **The box is an open 0.5 % sample of a population of about 340,000**, regulated at North Sea scale. A box-scale K would be a local stand-in for that regulation. But the regulation it would stand in for is food and movement, which the model has.

4. **The real discrepancy is B, not K.**
   - The model holds about 7x fewer porpoises than the survey. The pessimistic bound is 2.7x, the 2016 survey 13-18x.
   - The model's own food-limited equilibrium sits at 12-16 t.
   - So either the model under-supplies porpoise food (intake parameters, encounter rates, diet), or it over-charges porpoise energy, or both.
   - A K cannot fix this; it would only push further down.
   - This needs a separate calibration check before any start-biomass change. Raising the start to about 84 t without it would most likely starve the excess away.

5. **Seasonality caveat.** Survey densities are July. The model's start date is random (`start_day_of_year: random`). A winter box density below the summer value is plausible but unquantified.

## Source status and fetch log

| Source | Grade | Route |
|---|---|---|
| Gilles et al. 2023 (SCANS-IV) | [F] | tethys PDF, pdftotext. Tables 1, 6 and 30, trend text; Fig. 1 rendered and viewed |
| Hammond et al. 2021 (SCANS-III revised) | [F] | St Andrews PDF, pdftotext. Tables 1-2, porpoise block table, trend text |
| DCE Notat N2022/74 (SCANS-IV Danish team) | [F] | DCE PDF. Text plus Fig. 2 map (marked temporary draft) |
| Sveegaard et al. 2011 MMS | [F] | Reprint in Sveegaard 2011 PhD thesis (dmu.dk PDF) |
| Sveegaard 2011 thesis summary | [F] | same PDF |
| Sveegaard et al. 2012 MEPS | [A] | OpenAlex |
| IMMA factsheet, Eastern North Sea / Norwegian Trench | [F] | web page |
| OSPAR QSR 2023 bycatch | [F] | oap.ospar.org HTML (first 100k of 102k characters) |
| OSPAR QSR 2023 abundance; Owen et al. 2024; Mackinson & Daskalov 2007 | [2] | via `mammals_and_seabirds.md` and `mammals.md` (read there), not re-read |
| Genu et al. 2021 (RLA) | [F] | Frontiers PDF |
| Cordier et al. 2026, DEPONS 4.0 (DCE SR705) | [F] | DCE PDF, sections 2-3 |
| Nabe-Nielsen et al. 2014, 2018 | [A] | OpenAlex |
| Gilles et al. 2016 Ecosphere; Waggitt et al. 2020 | [A] | OpenAlex. Maps and data not opened |
| ASCOBANS 80 % of K / 1.7 % / 1 % objectives | [S] | search summary; 1.7 % also quoted in Genu et al. 2021 [F] |
| iPCoD "lack of density dependence" | [S] | search snippet of wind farm applicant documents |
| SCANS-II management procedures (Winship & Hammond) | [S] | search summary |
| Norwegian Skagerrak C-POD thesis; Kullaberg C-POD study | [S] | search summary |

**Not obtained [N]:**
- SCANS-II and SCANS-1994 block densities for the Skagerrak
- Sveegaard et al. 2011 kernel home-range areas in km2 (figure only)
- Waggitt et al. 2020 monthly density grid for the box
- ASCOBANS North Sea Conservation Plan text
- Swedish offshore static acoustic data for the Skagerrak
- Scheidat et al. (WUR edepot 272660): the download returned HTML, not the PDF

---

## Recommendation

**(a) Do not give porpoises a regional carrying capacity.** Reasons:
1. No published K or measured density dependence exists for this population. Management K values are reference points or simulation constants (RLA K = 500,000; Belt Sea K = 50,000 "rough"). DEPONS lets K emerge from food.
2. The model already regulates porpoises by the documented pathway: food, through starvation and reserve-funded growth. The runs show food-tracking, not runaway growth.
3. The box is an open sample of a North Sea-scale population, and the model already moves porpoises with their prey.
4. The candidate K = 14.5 t rests on 12 t = B_today, which the surveys contradict. It would bind from the first tick (loss 0.083/yr at 12 t) and double-count food regulation.

**If a K is wanted anyway** (for symmetry with the other DMs, or as a safeguard against runaway in future food-rich scenarios), use the survey-anchored route:
- **K = B_today/(1 - h/r_max) with r_max = the library value.**
- With B_today = 84 t (SCANS-IV NS-L 0.519/km2, 45 kg), h = 0.017 and r_max = 0.10:
  - **K is about 100 t = 0.028 t/km2.**
  - Range 34-330 t over the survey CI and 40-55 kg.
- At today's model biomass (12-16 t) it would add a loss of about 0.012-0.016/yr, about the size of the bycatch. It would never be approached within 3 years, or at any level seen in the runs.
- Define it as a formula of B_today, not as a fixed number, so it survives a later correction of the start biomass.

**What should be done instead** (outside this note's remit; suggestions only):
- Treat the 12 t start as unanchored. The survey-based box biomass is about 84 t (range 37-190 t; 2016: 155-213 t).
- Find out why the model's food-limited equilibrium sits at about 12-16 t:
  - porpoise intake and attack rates
  - diet access to herring at the trench slope
  - cost terms
- Do this before raising the start biomass. A K should be reconsidered only if porpoises, once fed realistically, show unbounded growth.
