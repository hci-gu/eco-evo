# Literature trail: field consumption vs measured prey density for water-column predators

Goal: a field (or semi-field) pairing of a water-column predator's consumption
(stomach content with an evacuation model, or daily ration) with a MEASURED prey
density, from which an attack rate (initial slope) can be derived. Predators in
priority order: saithe, hake (European, Pacific), spurdog, ling, pollack,
anglerfish.

Tags: [F] full text read here (pdftotext of the saved PDF), [A] abstract or
search-engine summary only, [S] secondary (cited by another source).
PDFs are in `data/lit/` (named by DOI with / -> _). They are untrusted; parsed
with pdftotext only.

Already known and not redone (task brief): Rindorf & Gislason 2005 (whiting),
Fall & Fiksen 2020, Kaartvedt et al. 2005, Dolgov et al. 1992, Livingston 1983,
Tanasichuk euphausiid share (abstract), SMS key-run saithe ration.

---

## 1. Smith & Smith 2020 - multispecies functional responses, NW Atlantic [F]

Smith BE, Smith LA (2020) Multispecies functional responses reveal reduced
predation at high prey densities and varied responses among and within trophic
groups. Fish and Fisheries 21: 891-905. DOI 10.1111/faf.12468.
File: `data/lit/10.1111_faf.12468.pdf` (NOAA Institutional Repository copy,
repository.library.noaa.gov/view/noaa/29560). Licence: "Published 2020. This
article is a U.S. Government work and is in the public domain in the USA"
(p. 891); Unpaywall lists is_oa False.

The ONLY source found that fits a functional response with explicit attack
rates for saithe (= "pollock", Pollachius virens), spiny dogfish, goosefish
(Lophius americanus) and silver hake (Merluccius bilinearis) on field data.

- Area/years: northeast US shelf (MAB, SNE, Georges Bank, Gulf of Maine,
  Scotian Shelf, > 290 000 km2), NEFSC bottom-trawl survey, spring (Mar-May)
  and fall (Sep-Nov), 1973-2016 (p. 893).
- Consumption: "Annual means of per capita consumption (g) and prey density
  (number of fish per nm2, number of zooplankton per 100 m3 and number of
  benthos per L) for each spring and fall season were the base for all
  functional response models" (p. 893). Eggers / Elliott-Persson:
  C_daily = 24 E D, E = alpha exp(0.115 T), T the seasonal bottom temperature;
  alpha prey-specific from Temming & Herrmann 2003 or 0.004; zooplankton 0.01
  (Eq. 1-2, p. 895-896). Per capita = g per predator individual (NOT g/g).
- Prey density: fish and squid (Atlantic herring, Atlantic mackerel,
  butterfish, Loligo, sand lance, silver hake) "estimated by weighting the
  number of individuals caught in the swept area of the tow (0.01 nm2) by the
  area (nm2) of the stratum" (p. 895) - bottom-trawl numbers per nm2, no
  catchability correction ("As a scalar, accounting for catchability could
  improve absolute estimates of prey density, but this would have minimal
  effect on the functional response", p. 901). Zooplankton (Copepoda,
  Euphausiidae, Hyperiidae, Mysida) from 61 cm bongo, 333 um, per 100 m3.
- Model: multispecies C_i = a_i X_i^n / (1 + sum_j a_j h_j X_j^m), types II
  (n = m = 1), III (n = m = 2), IV (n = 1, m = 2), Bayesian (JAGS) (Eq. 4,
  p. 896).
- Size classes (Table 1, p. 894): pollock S <= 20 (benthivore), M 21-50 and
  L 51-80 cm (planktivores), XL > 80 cm (piscivore); spiny dogfish S <= 36 and
  M 36-80 (planktivores), L > 80 (piscivore); goosefish S-XL all piscivores;
  silver hake S planktivore, M, L piscivores.
- Result: "For piscivores, nearly all responses were Holling type III with
  the exception of one being Holling type II (hyperbolic; satiation)"
  (abstract); "Holling type III response was the best-fit model for all
  predators (solid blue) except XL Pollock (Holling type II ...)" (Fig. 4
  caption, p. 898). "Pollock exhibited a Holling type II response for its
  largest size category when switching to a diet with high fish consumption"
  (p. 901). Planktivore classes (pollock M-L, dogfish S-M) are type III or IV
  on zooplankton (Fig. 5).
- Numbers: "Parameter values of the best-fit functional response models for
  all predator-prey combinations are in Table S3" (p. 899). Table S3 is in the
  Wiley supporting information only; the Wiley page returned HTTP 403 and it
  is NOT in the NOAA IR copy. Not obtained (the NOAA IR record lists only
  the main PDF, no supplement); the fitted a and h for pollock, dogfish,
  goosefish and silver hake sit in the Wiley supporting information.
- Y unit: the methods model the seasonal value (C_season = C_daily x 91.25,
  Eq. 3), and the plotted points (up to ~1600 g for M pollock 21-50 cm,
  ~5000 g for XL pollock) cannot be daily, so the axis is g per predator per
  season (inferred, not stated in the captions). %BW/d = g / 91.25 / W.
- Fig. 5 read by eye (lower half of p. 898): M (21-50 cm) and L (51-80 cm)
  pollock on Euphausiidae (bongo, per 100 m3, ~200-2000) are best fitted by
  TYPE IV (dashed green): the curve peaks at ~300-400 g/season at the lowest
  sampled densities (~200-400 per 100 m3) and declines thereafter; points
  scatter 0-1600 g with no rising trend over the sampled range. For a ~0.6 kg
  M pollock, 300 g/season is ~0.55 %BW/d. Section 3.3: "most best-fit models
  included decreased predation at high prey densities" (p. 900). The same
  form for pollock on Copepoda, Hyperiidae, Mysida. This is a plateau / no
  positive slope for saithe on krill over the sampled densities (bongo nets
  undersample adult euphausiids, so the true densities are higher).
- Fig. 4 read by eye (p. 898; page rendered at 400 dpi, not digitised):
  XL pollock (> 80 cm) columns are fitted to roughly 6-12 season-year
  points per prey, almost all at or near zero consumption; the type II
  curves (herring axis 0-3 x1000 per nm2, curve ~1300 g at 3) lie above
  nearly every observed point, so the fit is unconstrained by data and no
  usable slope comes from it. L spiny dogfish (> 80 cm) on herring: the
  type III curve saturates at ~150 g by ~2-3 x1000 herring per nm2 (the
  lowest densities sampled) with points scattered 0-1200 g with no visible
  trend over 0-17 x1000 per nm2 - a plateau over the observed range.
  The y axis is "Per capita consumption (g)"; whether a daily or a seasonal
  (x 91.25) value is plotted is not stated in the caption.
- Caveats for an attack rate: per-individual grams (needs the class's mean
  weight), prey in numbers (needs prey mean weight), shelf-wide annual means
  (the slope is a large-scale, overlap-inclusive "population" response, as in
  Kempf et al. 2008), bottom-trawl density of pelagic prey (herring, sand lance,
  mackerel) with unknown q.

## 2. Lopez-Lopez, Bartolino & Preciado 2015 - European hake, Cantabrian Sea [F]

Lopez-Lopez L, Bartolino V, Preciado I (2015) Role of prey abundance and
geographical variables in a demersal top predator's feeding habits (Merluccius
merluccius). Mar Ecol Prog Ser 541: 165-177. DOI 10.3354/meps11490.
File: `data/lit/10.3354_meps11490.pdf` (int-res.com). Licence: Inter-Research
copyright, free to read on the publisher site; Unpaywall is_oa False; not an
open licence.

- Area/years: Cantabrian Sea shelf (ICES VIIIc), DEMERSALES bottom-trawl
  survey, "every autumn (mid-September to the end of October) between 1990 and
  2013", 70-500 m plus extra tows (p. 166). Hake > 18 cm, n = 16 878 stomachs.
- Consumption measure: "the standardised stomach fullness (the volume of the
  stomach as a percentage of the predator's weight, V%BW ...)" (p. 166) -
  a stomach content, not a ration (no evacuation model). "Among the full
  individuals, stomach fullness reached maximum values of 80%, with median and
  mean values of 7.6 and 10.3%" (p. 169); 45 % empty.
- Prey density: "net catch abundance data for predator and prey (ind. km-2)
  were interpolated by means of generalized additive models (GAMs)" per haul
  and year, log-transformed (p. 168-169). Bottom trawl; blue whiting q unknown.
- Result: "abundances of blue whiting over 1000 ind. km-2 had a positive
  effect on the full/empty ratio (Fig. 2d)" (p. 169); "this positive
  relationship was triggered by a feeding threshold: a minimum blue whiting
  abundance of 100 to 1000 ind. km-2, suggesting a Holling type III response
  of the predator to its main prey" (p. 173). Deviance explained 2.29 %
  (binomial) and 5.69 % (fullness) (p. 169).
- Use: shape/threshold only (Type III onset), not an absolute attack rate.
  Fullness is not ration, explained deviance is tiny, and the effect sizes are
  only in figures (GAM smooths on a cube-root scale).

## 3. DeVries, Gartland & Latour 2025 - spiny dogfish, NW Atlantic [F]

DeVries C, Gartland J, Latour RJ (2025) Patterns in spiny dogfish consumption
by sex and maturity stage relate to prey availability and environmental
forcing in the Northwest Atlantic. Front Mar Sci 12. DOI
10.3389/fmars.2025.1621343. File: `data/lit/10.3389_fmars.2025.1621343.pdf`.
Licence: CC BY ("open-access article distributed under the terms of the
Creative Commons Attribution License (CC BY)", p. 1).

- Consumption: "minimum daily biomass consumed by spiny dogfish ... was
  estimated by coupling spiny dogfish catch and stomach content data with a
  gastric evacuation model (Eggers, 1977; Elliott and Persson, 1978)";
  "PCC is per capita prey consumption (kg prey/kg predator)" (Eq. 1, p. 3-4).
  NEFSC and NEAMAP surveys, spring and fall, 1980-2019; prey: Atlantic herring,
  Atlantic menhaden, longfin squid; prey biomass density in kg/km2 from VAST
  models.
- Result: correlations only (Kendall tau, Siegel regression): "Of the sixty
  potential relationships between an availability metric of a prey species
  (harvest, abundance, or overlap) and its consumption, twenty were
  significant ... eight out of sixteen for Atlantic herring" and "Except for
  annual harvest, all significant relationships between prey availability and
  consumption of that prey species were positive" (Sec. 3.6). No functional
  response fitted: "While not explicitly modeled, prey switching is a key part
  of the theoretical underpinning for Holling's Type III functional response
  relationship, which has been reported to best describe predation by most
  piscivores in the Northwest Atlantic, including spiny dogfish (Holling,
  1959; Smith and Smith, 2020)" (Sec. 4).
- Data: "The raw data supporting the conclusions of this article will be made
  available by the authors" (Data availability). Per-year values only as
  maps/tiles in figures. No slope extractable from the text. The
  supplementary material (Tables S1-S8, model equations) was NOT opened.

## 4. Trenkel et al. 2005 - Celtic Sea hake, cod, megrim, whiting [F]

Trenkel VM, Pinnegar JK, Dawson WA, du Buit MH, Tidd AN (2005) Spatial and
temporal structure of predator-prey relationships in the Celtic Sea fish
community. Mar Ecol Prog Ser 299: 257-268. DOI 10.3354/meps299257.
File: `data/lit/10.3354_meps299257.pdf` (archimer.ifremer.fr copy).
Licence: Inter-Research copyright ("Resale or republication not permitted
without written consent of the publisher"); Unpaywall lists it free to read.

- Presence/absence of prey in stomachs vs bottom-trawl survey density,
  1982-1995. No ration. Cites Pinnegar et al. 2003 [S]: "The average
  occurrence of blue whiting in the stomachs of hake and megrim was positively
  related to blue whiting stock abundance, as determined from ... ICES stock
  assessments" (p. 258). Conclusion: "opportunistic predators and general
  independence between predator and prey distributions" (abstract).
- Qualitative only.

## 5. Homrum 2012 PhD thesis - Faroe saithe [F, partly]

i Homrum E (2012) Faroe saithe ... (PhD thesis, hav.fo). File:
`data/lit/homrum_2012_phd_thesis_farsai.pdf` (Faroe Marine Research Institute
web; licence not stated). Much of the text uses an obfuscated font
(pdftotext returns shifted characters); read in the legible parts only.

- Stomach content as a partial fullness index (Paper I), no ration, no
  paired prey density. The quotes below are partly HAND-DECODED from the
  shifted font (character shift -3), not verbatim copy: "The interannual variability in stomach fullness in
  Faroese waters in summer is mainly driven by the amounts of blue whiting in
  the stomach contents (Figure 14). This is in accordance with the recruitment
  to the blue whiting stock (ICES, 2011b), which decreased drastically in
  2005" (p. 13-14). "In a preliminary study, there were no clear conclusions
  regarding the relationship between stomach contents in summer and the
  weight gain in the subsequent months (Paper V)" (p. 16).
- Saithe recruitment at age 3 vs Norway pout / sandeel biomass index (Fig. 15)
  - recruitment, not consumption. Indirect only.

## 6. Cormon et al. 2016 - North Sea saithe growth vs Norway pout [A]

Cormon X, Ernande B, Kempf A, Vermard Y, Marchal P (2016) North Sea saithe
Pollachius virens growth in relation to food availability, density dependence
and temperature. Mar Ecol Prog Ser 542: 141-151. DOI 10.3354/meps11559.
Growth coefficient vs Norway pout biomass (food) and saithe abundance; no
consumption. Indirect only.

## 7. Kempf, Floeter & Temming 2008 - North Sea cod and whiting [A]

Kempf A, Floeter J, Temming A (2008) Predator-prey overlap induced Holling
type III functional response in the North Sea fish assemblage. Mar Ecol Prog
Ser 367: 295-308. Cod and whiting only (not in this brief's predator list).
Abstract: "A rapid increase in relative stomach contents was observed if the
prey populations passed the abundance threshold of the prey refuge ... At even
higher abundances a saturation effect in relative stomach contents was
detected." Mechanism = changing spatial overlap. Relevant as the North Sea
precedent for a large-scale Type III with a plateau.

## 8. Lauerburg et al. 2018 - North Sea whiting [F, out of scope]

Mar Ecol Prog Ser 594: 213-230, DOI 10.3354/meps12533. Whiting: "In the
period of low forage fish availability, mean stomach content mass was on
average 60% less than in the reference periods" (abstract). Whiting only.

## 9. Not obtained / no paired data found

- Tanasichuk 1999, Fish Oceanogr 8: 150-156, DOI
  10.1046/j.1365-2419.1999.00100.x [A, Crossref abstract]: summers 1989,
  1990, 1995-97; "Hake persisted in selecting krill longer than 17 mm and
  consistently preferred T. spinifera, even though there was a substantial
  reduction in euphausiid abundance ... The biomass of euphausiids available
  to hake after 1993 averaged 27% of that for 1991, the pre-ENSO year, and
  was 12% of the 1991 value in 1996." With the known ~89 % euphausiid diet
  share this is a candidate plateau (diet unchanged at 0.12-0.27 of the
  prey biomass), but the yearly rations are only in the full text (Wiley,
  not OA, 403). Not obtained.
- Tanasichuk, Ware, Shaw & McFarlane 1991, CJFAS 48: 2118-2128, DOI
  10.1139/f91-251 [A, Crossref abstract]: August 1983, 1985-89 and
  June-October 1988; "Summer ration estimates were higher for hake feeding
  on the continental shelf than for fish feeding along the shelf-break.
  Daily ration declined over summer because the euphausiid ration dropped,
  even though the herring ration increased." "The dogfish ration in August
  was 1.6 times that of hake." No prey density in the abstract. Not OA.
- Mackas et al. 1997, CJFAS (euphausiid and Pacific hake aggregation, echo
  integration of both) [A] - co-location only, no consumption in abstract.
- Slotte A, Runde A, Skaret G, Axelsen BE, Ferno A, Johannessen A,
  Nottestad L, Ovredal JT (2005) Spatial and temporal overlap and trophic
  interactions between Atlantic saithe and Norwegian spring spawning herring
  on herring spawning grounds. ICES CM 2005/L:03 [A, NVA record
  hdl 11250/100773]: "it was observed that stomachs of large saithe (>50 cm)
  on average contained 2-3 herring, while the stomachs of smaller saithe
  were filled with herring eggs." Acoustic overlap measured, but no density
  or ration in the abstract. The PDF (L-2005.pdf, Norwegian copyright act
  licence) returned Forbidden from the NVA download API.
- IMR working document "Management Strategies for Northeast Arctic Saithe:
  should predation on herring be taken into account?" (hdl 11250/100699) [A]:
  saithe consumption of herring from stomach samples; no density pairing in
  the abstract. Not obtained.
- NOAA FEAT (Pacific hake survey): states the aim "determining how much krill
  are being eaten by hake, and if this amount is related to the abundance of
  krill detected acoustically" - no published result found.
- Ling, pollack (P. pollachius), anglerfish (L. piscatorius / budegassa):
  NOT CHECKED (no dedicated search within the time cap). Lophius americanus
  is in Smith & Smith 2020 (type III, all sizes).
- NOT CHECKED / not opened within the time cap: Bendall / Ellis (NE Atlantic
  spurdog), Bergstad 1991 (Norwegian Deep gadoids), Du Buit 1991 (saithe),
  Mehl / Olsen (Barents / Norwegian coast saithe on herring), Buckley &
  Livingston 1997 and Rexstad & Pikitch 1986 (Pacific hake), Velasco & Olaso,
  Mahe et al. 2007, Cabral & Murta (European hake ration), Icelandic saithe
  (one search, nothing read). "Not found" above means not found in what was
  searched, not that no such source exists.

---

## Bottom line (2026-11-22, ~40 min search)

- No source among those searched gives a saithe ration paired with a
  measured prey density in t/km2 from which an absolute attack rate can be
  computed; ling, pollack and European anglerfish were not searched (see 9).
- Closest structure: Smith & Smith 2020 (pollock = saithe, spiny dogfish,
  goosefish, silver hake; fitted multispecies type II/III with attack rates),
  but the parameters are in Wiley Table S3 (not obtained), consumption is
  g per individual, prey are bottom-trawl numbers per nm2 with no q, and the
  XL pollock fit is unconstrained by its few, mostly zero points.
- Shape evidence only: Lopez-Lopez et al. 2015 (European hake on blue
  whiting, type III threshold 100-1000 ind/km2, fullness not ration),
  Kempf et al. 2008 (North Sea cod and whiting, type III with saturation,
  overlap-driven).
- Plateau / weak-slope evidence: Smith & Smith 2020 Fig. 5 M and L pollock
  (saithe 21-80 cm) on euphausiids: type IV, peak at the lowest sampled
  bongo density, no positive slope; Fig. 4 L spiny dogfish
  on herring (flat over the sampled range); Tanasichuk 1999 Pacific hake
  kept its euphausiid selection at 0.12-0.27 of the 1991 euphausiid biomass
  (rations not seen); Homrum 2012 (Faroe saithe summer fullness follows
  blue whiting recruitment - a slope, qualitative).
