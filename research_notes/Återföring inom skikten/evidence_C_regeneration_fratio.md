# C. NH4 regeneration, f-ratio, NH4 turnover, NH4 vs NO3 kinetics

Context: Skagerrak/Kattegat model, one DIN pool per layer (NO3 + NH4, K = 0.5 umol/l),
model f = 0.55, target annual f 0.2-0.4, annual PP ~150 g C m-2 yr-1.
Already known and not redone: Bendtsen & Richardson 2018; Bjornsen 1993; Rosenberg et al. 1990;
Richardson 2003.

Grades: [F] full text read, [A] abstract read (publisher/Crossref/OpenAlex/IMIS),
[S] search-engine snippet or secondary citation only, [N] not found / no access.
"(Own calculation)" marks my arithmetic.

---

## 0. Short answers

- **Who regenerates NH4:** organisms <200 um (protists plus bacteria) do most of it in coastal
  and shelf waters. Mesozooplankton supply roughly <10 % to about 1/3. Coastal estimates are at
  the low end: A Coruna <10 % [A]; Oslofjord summer regeneration came mostly from the 45-200 um
  fraction [A]. In the well-mixed western Channel in summer, picoplankton/bacteria (<1 um)
  supplied 20-45 % and microplankton about 50 % [S]. A defensible model split for summer is
  ~50-80 % micro/nano protists, ~10-45 % bacteria/picoplankton (the upper bound is Maguer's
  summer <1 um share), ~10-20 % mesozooplankton. That split is my synthesis, not a measured
  partition.
- **Seasonal f, regional:**
  - SE Kattegat: about 80 % nitrate in the March bloom. In summer NH4 is ~50 % and urea ~30 % of
    uptake. The bloom is only 16 % of annual N uptake (2.26 mol N m-2 yr-1) [S].
  - Western English Channel: annual f 0.32. Summer NO3 is 27 % of uptake, NH4 55 %, urea 13 % [F].
  - Oslofjord: urea is 19 % of uptake (Apr-Oct mean) [A].
  - The annual f of a eutrophic temperate coastal sea is **~0.25 (SE Kattegat, unconfirmed [S]) to
    0.32 (W Channel [F])**. The Channel value is the only solid anchor.
- **NH4 turnover in summer:** about 1 h in the N-depleted summer Baltic [A]. "A few hours" in the
  summer Oslofjord [A]. Summer surface DIN in the Skagerrak and Kattegat is <=0.1 umol/l, below
  the reporting limit (SMHI, July 2022) [F].
- **Kinetics:**
  - Fasham et al. 1990 is confirmed [F]: K1 (NO3) = 0.5, K2 (NH4) = 0.5 mmol m-3,
    psi = 1.5 (mmol N m-3)-1. K2 = K1 was a placeholder chosen "in the absence of good
    experimental data".
  - Oceanic K is ~20-30 nM, and K_NH4 is about the same as K_NO3 [A].
  - Ks scales with cell size: <=0.5 uM for small oceanic species, >5 uM for Gonyaulax [A].
  - Field inhibition data: in the Channel, NO3 uptake halves at NH4 = 0.5 umol/l [F]. Harrison
    et al. 1996 found Ki ~40-50 nM in the ocean, and NH4 never fully stopped NO3 uptake [A].
  - Combining the two pools is acceptable when the model's f is defined by the source of the
    supply rather than by the uptake kinetics (see section 4).

---

## 1. Partition of NH4 regeneration by size fraction / organism

| Source | Site, season | Finding (exact numbers) | Grade |
|---|---|---|---|
| Paasche & Kristiansen 1982, Mar Biol 69:55-63, doi:10.1007/BF00396961 | Oslofjord, June-Sept, 0-2 m, 15N isotope dilution | "Daytime regeneration rates at 0-2 m depth were 0-28% of the calculated assimilation rates" at ambient NH4. Regeneration was faster in an August dinoflagellate bloom than in the June and Sept diatom-dinoflagellate blooms. Most NH4 "appeared to be produced by juvenile copepods, rotifers, tintinnids, and heterotrophic dinoflagellates in the size fraction 45-200 um". Summer DIN above the pycnocline is very low, and the inorganic N pools turn over "within a few hours". | [A] (IMIS abstract; the 45-200 um sentence via Springer snippet [S]) |
| Glibert 1982, Mar Biol 70:209-222, doi:10.1007/BF00397687 | 2 oceanic, 1 coastal (Vineyard Sound), 1 estuarine (Chesapeake) region | The smallest size fraction (<10 um) was usually the most important in remineralizing NH4. The apparent bacterial fraction (<1 um) may increase in importance after blooms. NH4 uptake and remineralization fluxes were mainly in balance. No percentages were retrieved. | [S] (Springer page blocked; no abstract in Crossref/OpenAlex) |
| Glibert, Lipschultz, McCarthy & Altabet 1982, L&O 27:639-650 | Methods (Sargasso, Chesapeake, cultures) | Routine 15N uptake underestimates NH4 uptake "by a factor of =2". Regeneration "over relatively brief periods can supply the daily nitrogen requirements of the phytoplankton". Detection limit ~0.03 ug-at/l. No size partition. | [F] (PDF: hahana.soest.hawaii.edu .../Glibert_et_al._1982_L&O.pdf) |
| Maguer, L'Helguen, Madec & Le Corre 1999, Cont Shelf Res 19:1755-1770, doi:10.1016/S0278-4343(99)00037-0 | Western English Channel, well-mixed coastal station, annual cycle, <200 um water | Microplankton (200-15 um) supplied ~50 % of regeneration from early spring to late summer. Nanoplankton supplied 2-13 % in summer. Picoplankton (<1 um, "mainly associated with bacteria") supplied 20-45 % in summer. Nanoplankton ("mainly associated with ciliates") supplied 83-88 % in winter (Frontiers 2022 citation). The size-class labels differ between the two secondary sources. | [S] (ScienceDirect snippet plus Lee et al. 2022 Front Mar Sci 9:1017434, which cites it) |
| Bode, Barquero, Gonzalez, Alvarez-Ossorio et al. 2004, J Plankton Res 26:11-28, doi:10.1093/plankt/fbh003 | A Coruna coastal upwelling, monthly 1995-1997 | NH4 was "33 and 43% of total N uptake in high and low production situations". The NH4 demand "was closely matched by regeneration rates of microplankton (0.14-0.25 mmol N m-2 h-1), whereas zooplankton contributed on average <10% to N regeneration". | [A] |
| Hernandez-Leon et al. (year not confirmed; ULPGC repository 10553/49884) | Global, open ocean | Mesozooplankton regeneration was 12-23 % of the phytoplankton plus bacterial N requirement, highest in tropical and subtropical waters. | [S] |
| Fernandez-Urruzola et al. 2012 (ISMS abstract, ULPGC 10553/7239) | Oceanic | Mesozooplankton are "responsible for 12% to 33% of ammonium recycling". Recycling efficiency runs from 50 % in coastal waters to ~95 % in oligotrophic tropical waters. | [S] |
| Hoch & Kirchman 1995, L&O 40:886-897, doi:10.4319/lo.1995.40.5.0886 | Delaware estuary and coast, 1988 and 1990 | Heterotrophic bacteria took 10-25 % of total NH4 uptake at the bay mouth in summer and 15-35 % at coastal and offshore stations (1990). The share was <5 % at salinity <20. About 50 % of bacterial N demand was met by NH4 in summer. Bacterial NH4 uptake is relatively high in oligotrophic water and low in eutrophic water. | [A] (OpenAlex) |
| Klawonn et al. 2019, ISME J, doi:10.1038/s41396-019-0386-z | Baltic Proper, N-depleted surface water, summer | NH4 production was >=91 % regenerated and <=9 % from N2 fixation. It supported 78-97 % and 2-16 % of primary production respectively. NH4 was "rapidly recycled within 1 h". Heterotrophic bacteria took ~20 % of NH4 consumption and picocyanobacteria ~20-40 %. The consumption turnover time of 1.2 +- 1.0 h comes from a snippet. | [A] (the 1.2 h figure is [S]) |
| Lee et al. 2022, Front Mar Sci 9:1017434 | Yeongsan estuary (turbid, eutrophic) | Bacteria <0.7 um, rather than zooplankton, did most of the regeneration. Regeneration was 0.1-1.2 and potential uptake 0.2-1.5 umol L-1 h-1, highest in summer. | [A] (abstract text in PDF) |
| Dickson & Wheeler 1995, MEPS 121:239-248, doi:10.3354/meps121239 | Oregon upwelling | 22-51 % of the NH4 removed from the dissolved pool was not recovered in particles, so uptake was underestimated. Uptake was 11-17 mmol N m-2 d-1 in the upwelling season and 3 in the non-upwelling season. Regeneration did not differ between seasons. No size partition in the abstract. | [A] |
| Bronk & Steinberg 2008, "Nitrogen Regeneration", in Nitrogen in the Marine Environment 2nd ed., pp. 385-467, doi:10.1016/B978-0-12-372522-6.00008-6 | Review | Cited elsewhere for: NH4 excretion is the largest source of recycled N in the euphotic zone, and microzooplankton are the major NH4 regenerators (with Bronk et al. 1994). No numbers retrieved. | [S] |
| Kirchman 1994, Microb Ecol 28:255-271, doi:10.1007/BF00166816 | Review | Citation confirmed; no abstract or numbers retrieved. | [N] |
| Harrison 1978 / 1992; Koike & Hattori; LaRoche; Riegman & Noordeloos 1998; Sanders et al.; Weston et al. 2005; Kattegat/Danish 15N isotope-dilution studies | -- | Not retrieved within the call budget. No Kattegat or Skagerrak isotope-dilution regeneration study surfaced in the searches. | [N] |

**Synthesis for item 1 (my interpretation, not a measured partition):**
- Coastal and shelf summer regeneration is dominated by organisms <200 um.
- The two direct coastal mesozooplankton estimates are low: Bode 2004 gives <10 %. In Paasche &
  Kristiansen the 45-200 um fraction includes copepod nauplii and juveniles, so "microzooplankton"
  there is partly juvenile metazoans.
- Bacteria are net NH4 consumers more often than net regenerators. They take 10-35 % of NH4
  uptake (Hoch & Kirchman; Klawonn ~20 %), so a "bacteria regenerate X %" share is
  method-dependent.
- A summer split of roughly 50-80 % protists (<200 um), 10-45 % bacteria/picoplankton (10-35 %
  as NH4 uptake share, Hoch & Kirchman; 20-45 % as regeneration share, Maguer [S]) and 10-20 %
  mesozooplankton is consistent with all of the above.
- Caveat: in the Oslofjord, daytime surface (0-2 m) regeneration was only 0-28 % of
  assimilation (Paasche & Kristiansen [A]). That is a daytime, surface-only measurement in a
  fjord with external N loading, but it means the measured surface regeneration did not always
  cover summer uptake. The rest must come from night-time and deeper regeneration, from urea,
  or from external supply.

---

## 2. Seasonal f-ratio, NH4 vs NO3 vs urea, regional

| Source | Site | Numbers | Grade |
|---|---|---|---|
| Sahlsten, Sorensson & Pettersson 1988, J Exp Mar Biol Ecol 121:227-246, doi:10.1016/0022-0981(88)90090-1 | SE Kattegat (Laholm Bay area), 0-6 m, data 1982-1986 | Spring bloom: NO3 accumulated over winter was "about 80% of total daily uptake" (frequent sampling, early March 1984). Summer and early autumn: "ammonium and urea" ~50 % and ~30 % of uptake. Annual N uptake 2.26 mol N m-2 yr-1, of which the spring bloom was 16 %. Annual new production as a share of total uptake ~25 %; this figure appeared in only one search summary and is unconfirmed. | [S] (ScienceDirect 403; numbers from two independent search summaries of the abstract) |
| (Own calculation) | -- | 2.26 mol N m-2 yr-1 x 6.625 (Redfield C:N) x 12 g/mol = **~180 g C m-2 yr-1**, close to the 150 g C target. A summer of 50 % NH4 + 30 % urea leaves <=20 % for NO3 + NO2, so summer f <= 0.2. | -- |
| L'Helguen, Madec & Le Corre 1993, Oceanol Acta 16:653-660 (archimer.ifremer.fr/doc/00100/21129/18749.pdf) | Western English Channel, permanently well-mixed, 48 45'N 3 58'W, 45 m, 1988, 50 % light depth | Seasonal mean uptake (umol N l-1 h-1, % of total) is in the table below. Annual f-ratio = 0.32. "Regenerated production accounts for about 70% of the annual production". | [F] |
| Kristiansen 1983, Mar Biol 74:17-24, doi:10.1007/BF00394270 | Oslofjord, Apr-Oct | Urea was "0 to 53% (mean 19%) of summed NH4+ + NO3- + urea uptake". Urea uptake was higher in June-August and inhibited at NH4 > 1-2 ug-at N/l. | [A] (IMIS/Springer abstract) |
| Paasche & Kristiansen 1982, Estuar Coast Shelf Sci 14:237-249, doi:10.1016/S0302-3524(82)80014-5, "Nitrogen nutrition of the phytoplankton in the Oslofjord" | Oslofjord | Citation confirmed; abstract not retrieved. L'Helguen 1993 cites it for NO3 uptake stopping completely at NH4 of 1-2 umol/l. | [N] (secondary [F]) |
| Moneta, Veuger, van Rijswijk, Meysman et al. 2014, Estuar Coast Shelf Sci 147:78-86, doi:10.1016/j.ecss.2014.05.022 | Coastal North Sea, Marsdiep inlet, seasonal | Algal and bacterial DOM were preferred in autumn and winter, "whereas NH4+ was preferentially taken up in spring and summer". Urea was used mainly as an N source in spring and summer. No percentages in the abstract. | [A] |
| Harrison, Platt & Lewis 1987, J Plankton Res 9:235, doi:10.1093/plankt/9.1.235 | 8 coastal data sets | f rises asymptotically with NO3, but f_max and the initial slope vary between regions by more than 10x. Ambient NH4 explains the regional differences. f against NO3/(NO3+NH4) is "approximately linear" and regionally consistent (slopes within 2x). Including urea and correcting for isotope dilution "systematically reduce f-ratio estimates". | [A] |
| Rees et al. 2002, DSR II 49:2905-2927, doi:10.1016/S0967-0645(02)00063-2 | Northern North Sea, coccolithophore bloom, June 1999, size-fractionated | Citation confirmed; no abstract retrievable (no abstract in Crossref/OpenAlex, ScienceDirect blocked). | [N] |
| Rees et al. 1999 North Sea Project; Owens; Sanders & Jickells; Le Corre; Kristiansen North Sea / Norwegian coastal water f-ratios | -- | Not found. Rees, Joint & Donald 1999 (DSR I 46:483) covers the Celtic Sea shelf edge in early spring, not the North Sea. | [N] |

L'Helguen et al. 1993, Table 2 [F], uptake in umol N l-1 h-1 (% of total uptake):

| Season | NO3 | NH4 | Urea | NO2 |
|---|---|---|---|---|
| Winter | 0.0039 (38) | 0.0033 (33) | 0.0017 (17) | 0.0012 (12) |
| Spring | 0.0168 (39) | 0.0184 (42) | 0.0046 (11) | 0.0036 (8) |
| Summer | 0.0154 (27) | 0.0312 (55) | 0.0073 (13) | 0.0029 (5) |
| Autumn | 0.0026 (25) | 0.0043 (42) | 0.0030 (29) | 0.0004 (4) |
| Annual | 0.0103 (32) | 0.0152 (48) | 0.0043 (13) | 0.0022 (7) |

(Own calculation: counting NO2 as new N, the Channel annual f would be 0.39.) This is a
permanently mixed, never-N-depleted system: summer NO3 stayed >0.8 umol/l and NH4 ~0.6 umol/l.
In a stratified Skagerrak or Kattegat, summer f should therefore be lower than the Channel's 27 %
NO3 share.

**Urea:** in temperate coastal seas urea takes 13 % of annual uptake (Channel) and 19 % in summer
(Oslofjord, mean Apr-Oct). The SE Kattegat summer value, ~30 %, is the high end. A model that
lumps urea into "NH4" (regenerated) is consistent with how these f-ratios were calculated: in all
of them urea counts as regenerated.

**Regional synthesis:** the annual f of a eutrophic temperate coastal or shelf sea is about 0.25
(SE Kattegat [S]) to 0.32 (W Channel [F]). Spring blooms reach about 0.8 (Kattegat, March). Summer
is <=0.2 in the Kattegat (own calculation from the shares) and 0.27 in the mixed Channel. This
supports the 0.2-0.4 target. The model's 0.55 is too high.

---

## 3. NH4 turnover and summer concentrations

| Source | Numbers | Grade |
|---|---|---|
| Klawonn et al. 2019 (Baltic, summer, N-depleted) | NH4 "rapidly recycled within 1 h"; consumption turnover 1.2 +- 1.0 h; no clear diel pattern | [A] / [S] |
| Paasche & Kristiansen 1982 Mar Biol (Oslofjord, summer) | Inorganic N pools above the pycnocline turn over "within a few hours" | [A] (paraphrase via IMIS) |
| SMHI R/V Svea cruise report, 12-18 July 2022 (smhi.se exp_2822eng.pdf) | Skagerrak surface: "dissolved inorganic nitrogen (DIN) was below detection limit (<=0.1 umol/l)"; phosphate 0.04-0.07, silicate 0.4-1.6 umol/l. Kattegat and the Sound: "The concentration of DIN was <0.1 umol/l". DIN 0.10-0.38 umol/l only at the next (Baltic) station group. Surface-layer boundary 5-20 m (Skagerrak), 10-20 m (Kattegat). | [F] |
| L'Helguen et al. 1993 (mixed W Channel) | Summer NH4 ~0.6 umol N/l, NO3 always >0.8 umol/l, urea 0.04-0.21 umol N/l | [F] |
| Glibert et al. 1982 L&O | NH4 detection limit ~0.03 ug-at/l; uptake underestimated ~2x with routine 15N | [F] |
| Kirchman, Keil & Wheeler 1990, L&O 35:1258 (subarctic Pacific, for comparison) | NH4 from undetectable (<=0.05 uM) to 0.4 uM | [A] |

Implication: in summer, the Skagerrak and Kattegat surface DIN (NH4 included) is <=0.1 umol/l
and the pool turns over in roughly 1 h to a few hours. (Own calculation: with the model's
K = 0.5 umol/l, DIN = 0.1 gives N/(N+K) = 0.17, so uptake runs at 17 % of its maximum.)
At turnover times this short, summer production is set by the regeneration flux, not by the
standing pool.

---

## 4. Half-saturation constants and NH4 inhibition; is one DIN pool acceptable?

| Source | Numbers | Grade |
|---|---|---|
| Fasham, Ducklow & McKelvie 1990, J Mar Res 48:591-639 (PDF via web.archive.org copy of elischolar.library.yale.edu viewcontent article=2980) | Table 1: half-saturation for phytoplankton nutrient uptake K1 = 0.5 mMol m-3; NH4 inhibition parameter psi = 1.5 (mMol N m-3)-1; Vp 2.9 d-1; bacterial K4 0.5; bacterial NH4/DON uptake ratio 0.6. Text: K1 = 0.5 "representative of experimental observations (Goldman and Glibert, 1983), although values as low as 0.1 have been suggested". "In the absence of good experimental data the half-saturation constant for ammonium uptake, K2, was given the same value as K1". psi = 1.5 is "Wroblewski's (1977) estimate". Sensitivity ranges tested: K1 0.25-0.75; psi 0.75-3. Model annual f = 0.71 against an observed Bermuda estimate of 0.31 (Platt & Harrison 1985). Seasonal f runs from winter 0.8-0.9 to summer "ca 0.1-0.2". **The user's recollection (k1 0.5, k2 0.5, psi 1.5) is confirmed.** The equation image did not extract; the text says only that the Wroblewski (1977) parameterization is used. The usual form, Q1 = NO3 exp(-psi NH4)/(K1+NO3), Q2 = NH4/(K2+NH4), is written from recall and was not verified here. | [F] (parameter values); equation form [N] |
| Eppley, Rogers & McCarthy 1969, L&O 14:912-920, doi:10.4319/lo.1969.14.6.0912 | 16 culture species. Ks "varied approximately in proportion to cell size and inversely with specific growth rate". Small-celled oceanic species such as Coccolithus huxleyi had the lowest Ks (<=0.5 uM) and Gonyaulax polyedra the highest (>5 uM). | [A] (Crossref) |
| Harrison, Harris & Irwin 1996, L&O 41:16-32, doi:10.4319/lo.1996.41.1.0016 | North Atlantic surface. K_N ranged over 2-3 orders of magnitude, covarying with ambient NO3; oceanic K_N averaged ~20-30 nM. K_A and K_N "were of similar magnitude in oceanic waters". pm(A) almost always exceeded pm(N), by an order of magnitude or more in oceanic waters. "NH4 is preferentially utilized over NO3 over the full spectrum of nitrogen concentrations, nanomolar to micromolar." Inhibition Ki covaried with K_N, but oceanic Ki (~40-50 nM) always exceeded K_N. Maximum inhibition Im was "rarely complete (i.e. Imax < 1), even at 2,000 nM ammonium". Conclusion: N parameters in open-ocean models "should be re-examined". | [A] (Crossref) |
| L'Helguen et al. 1993 (W Channel, field) | NO3 uptake starts to fall when NH4 > 0.2 umol/l and is cut by 50 % at 0.5 umol/l, which they note is identical to the Ki estimated by Harrison et al. (1987) for coastal regions. Above that, relative NO3 uptake levels off and stays >20 % of total uptake. Complete stop of NO3 uptake was reported at NH4 1-2 umol/l (McCarthy et al. 1977; Paasche & Kristiansen 1982). | [F] |
| Litchman, Klausmeier, Schofield & Falkowski 2007, Ecol Lett 10:1170-1181, doi:10.1111/j.1461-0248.2007.01117.x | Abstract confirms trait compilations and trade-offs; K scales allometrically with cell size (smaller cells have lower K). The coefficient and exponent were NOT retrieved and are not quoted here. | [A] (exponent [N]) |
| MacIsaac & Dugdale 1969/1972 | A copy exists (waterboards.ca.gov ... macisaacanddugdale1972.pdf) but was not read. | [N] |

**Assessment: is one combined DIN pool with K = 0.5 acceptable?**

1. *The K value.* Fasham's 0.5 for both forms was a placeholder, and the field values span about
   20 nM (oceanic) to about 0.5-1 uM (coastal). A single K = 0.5 umol/l is within the coastal
   range for a mixed community. It is high for picoplankton and small flagellates, the summer
   regenerated-production community (Eppley 1969: small cells <=0.5 uM). Treat it as the
   community value of a diatom-dominated spring bloom. Harrison 1996 found K_A and K_N of the
   same order, so one K for the pooled DIN is not contradicted by the kinetics.
2. *Preference and inhibition.* These matter only for which form is taken up when both are
   present. In stratified Skagerrak/Kattegat summer water, DIN is <=0.1 umol/l (SMHI), so there
   is almost no NO3 left to be inhibited. Inhibition shapes the f-ratio mainly in mixed or
   transitional conditions: spring, autumn entrainment, and the Channel-type permanently mixed
   case. In the Channel, NO3 uptake halves at NH4 = 0.5 and never falls below ~20 % of total
   uptake [F].
3. *What the combined pool cannot do.* It cannot produce a kinetics-driven f. In a one-pool model
   the f-ratio is purely a supply budget: f = new N supply / total uptake. At steady state over
   a year, that equals entrainment plus diffusion from below, divided by (that same flux plus the
   regenerated return: excretion, grazing share, remineralisation). A model f of 0.55 therefore
   means the regenerated return is too small relative to the supply from below. It is not a
   kinetic error.
   (Own calculation: for f = 0.3, regenerated supply must be 0.7/0.3 = 2.3x the new supply. At
   f = 0.55 it is 0.45/0.55 = 0.8x, so regeneration must rise ~2.9x relative to new supply, or
   the supply from below must fall.)
4. *Measurement bias, which cuts both ways.* Including urea and correcting for isotope dilution
   lower measured f (Harrison et al. 1987 [A]). Glibert et al. 1982 [F] and Dickson & Wheeler 1995
   [A] report that NH4 uptake is underestimated by ~2x, or by 22-51 % unrecovered NH4, which also
   biases f upward. Measured f-ratios are therefore if anything upper bounds, which strengthens
   the case for a target around 0.25-0.35.
5. *Bacterial competition.* Heterotrophic bacteria take 10-35 % of NH4 uptake (Hoch & Kirchman
   1995 [A]; Klawonn 2019 ~20 % [A]). In a one-pool model without bacteria, the "regenerated"
   flux to phytoplankton should be net of bacterial uptake. Fasham 1990 gives bacteria an
   NH4/DON uptake ratio of 0.6 [F].

Conclusion: a combined DIN pool is acceptable for this purpose, given the Skagerrak/Kattegat
evidence that summer DIN is near zero and turns over in hours. The f-ratio target must be met
through the regeneration budget, by raising the share of grazing returned as NH4 and the
remineralisation, not through the half-saturation constant. Splitting the pool would add
inhibition and preference effects that matter mostly in spring and autumn. Fasham's
two-pool model with psi = 1.5 still produced an annual f of 0.71, against 0.31 observed, so
splitting the pools is no guarantee of a low f.

---

## 5. Not found / gaps (for follow-up)

- Rees et al. 2002 (North Sea, June 1999): size-fractionated f values. Not retrieved.
- Rees et al. 1999 North Sea Project seasonal f; Owens; Sanders & Jickells; Weston et al. 2005;
  Riegman & Noordeloos 1998 (MEPS, int-res, open; not reached in the call budget).
- Norwegian coastal water seasonal f (Kristiansen and others). Not found.
- Any Kattegat or Skagerrak 15N isotope-dilution regeneration study. None surfaced.
- Glibert 1982 Mar Biol size-fraction percentages; Maguer et al. 1999 primary numbers (only
  secondary versions seen); Kirchman 1994 and Bronk & Steinberg 2008 numbers.
- Litchman 2007 K-volume allometry coefficient and exponent (in the Ecol Lett paper; Table S1
  of Litchman et al. 2009 PNAS).
- Sahlsten et al. 1988: the "~25 % annual new production" figure needs the full text.
