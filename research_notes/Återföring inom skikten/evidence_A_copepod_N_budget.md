# A. Nitrogen budget of feeding copepods (mesozooplankton) - literature notes

Target: Skagerrak model, 58 N, mixed layer 6-17 m, 5-14 C; copepods Calanus
finmarchicus, Temora longicornis, Acartia tonsa/clausi, Pseudocalanus, Centropages.

Grades: [F] full text read, [A] abstract only, [S] search snippet only,
[N] not found / no access. "(Own calculation)" = my arithmetic on the cited numbers.
Searched 2026-10-24.

---------------------------------------------------------------------------

## 0. Bottom line (for model parameterisation)

| Quantity (share of INGESTED N unless stated) | Best value | Range in sources | Main source(s) |
|---|---|---|---|
| Dissolved N released by excretion (NH4 + urea), saturating food | ~0.20-0.35 | 0.21 (Kiorboe, saturated) - 0.34 (Saba, NH4 excretion only) | Kiorboe et al. 1985 [F]; Saba 2010/2011 [F] |
| Excretion at LOW ration (near maintenance) | ~0.5 or more | 0.50 (Kiorboe, 19 ug N/l); "40 to >100 %" (Eucalanus) | Kiorboe et al. 1985 [F]; Gardner & Paffenhofer 1982 [A] |
| Dissolved N from sloppy feeding (NH4 + urea), diatom prey | ~0.15-0.18 for Acartia; possibly much larger for Calanus on bloom diatoms (see note B) | 8 % NH4 + 10 % urea of PON ingested (Acartia); C-based sloppy loss 5 % (Acartia) to 49 % of C removed (Calanus) | Saba 2010 [F]; Moller et al. 2003 [F] |
| Urea leached from fresh faecal pellets | ~0.06 | 6 % of N ingested | Saba 2010 [F] |
| Total dissolved N (all routes) | ~0.5-0.6 | 0.49 of N removed (figure) / 0.58 of N ingested (abstract numbers) | Saba 2010 [F] (Own calculation of sums) |
| Egestion (particulate faecal N) | ~0.2-0.4 | 0.05 (low food) - 0.40 (high food); 0.17-0.23 (Saba); 1 - 0.62 = 0.38 (Calanus, Corner) | Kiorboe 1985 [F]; Saba 2010 [F]; Corner et al. 1967 via Ikeda & Motoda 1978 [F, secondary] |
| N assimilation efficiency | 0.6-0.95 (food-dependent; falls with ration) | Acartia 0.95/0.64/0.60; Calanus 0.54-0.68 (mean 0.62) | Kiorboe 1985 [F]; Corner et al. 1967 [F secondary] |
| N gross growth efficiency (growth/egg N / ingested N) | ~0.3-0.4 | Acartia 0.40-0.49; Calanus 0.14-0.38; Saba 0.34 of N removed | Kiorboe 1985 [F]; Corner/Butler via Ikeda & Motoda [F secondary]; Saba 2010 [F] |
| Share of excreted N as NH4 | 0.4-0.9 (method/diet-dependent) | >0.8-0.9 (older lit.); urea 13-59 % of TDN (Saba); organic N 62-89 % (Miller & Glibert, mesocosms) | Kiorboe 1985 [F]; Saba et al. 2009 [F] |
| Mesozooplankton excretion as share of phytoplankton N demand | ~10-30 % typical | 4 % (herbivorous, upwelling) - 44 % (Kuroshio), up to 77-90 % (Long Island Sound) | see section 3 |

Note A - which number to use for the excreted share. Kiorboe 1985 and Saba 2010
close their budgets differently, so their excretion shares are not averageable:
Kiorboe measured ingestion, egg production, respiration and NH4 excretion on
copepods acclimated 2 d to each food level (1-2 h NH4 incubations, blank-corrected)
and got AE by difference (DOM loss hidden inside "assimilated"); Saba measured
release over 3 h, derived egested N from pellet C with an ASSUMED pellet C:N of 4.3,
and got growth as the residual. Recommendation: use Kiorboe's ration curve
(0.50 -> 0.25 -> 0.21 of ingested N excreted, low -> intermediate -> saturating
food) for the excreted-share-of-ingestion parameter, and use Saba / Miller & Glibert
only for the NH4:urea split and for the sloppy-feeding and pellet-leaching routes.

Note B - sloppy feeding depends on copepod:prey size. N was not measured for
Calanus. In C, sloppy loss ranges from 5 % of C ingested (Acartia on
T. weissflogii, Saba 2010) to 49 % of C removed (Calanus spp. on spring-bloom
diatoms, Disko Bay, Moller et al. 2003). Sloppy feeding occurs when the
copepod:prey ESD ratio is below ~55 (Moller 2005, as cited by Saba et al. 2009).
For Calanus grazing large bloom diatoms (the Skagerrak March bloom case),
sloppy-feeding release may be comparable to or larger than excretion; on small
summer flagellates it should be small. Moller et al. 2003 themselves note "When
the available prey is composed of smaller cells, e.g. during summer, DOC
production by sloppy feeding will probably be lower".

Key mechanistic points:
- The excreted SHARE of ingested N is highest at low ration and falls as ration
  rises (basal/maintenance excretion is a fixed cost; feeding raises the absolute
  rate ~4-5x but less than ingestion). Kiorboe 1985 [F].
- Temperature: excretion rate Q10 ~1.8-2.0 (Ikeda et al. 2001) [S]. Because
  ingestion also rises with temperature, the excreted SHARE of ingestion is
  roughly temperature-independent at a given relative ration (my inference; no
  source measured the share across temperature).
- Lab data are mostly 18-20 C (Kiorboe 18 C, Saba 20 C); Calanus data by Corner
  (Plymouth) are near ambient temperature (not verified, original not read).

---------------------------------------------------------------------------

## 1. NH4 excretion and urea/DON release as a share of ingested N (and C)

### 1.1 Kiorboe, Mohlenberg & Hamburger 1985, MEPS 26:85-97 [F]
"Bioenergetics of the planktonic copepod Acartia tonsa: relation between feeding,
egg production and respiration, and composition of specific dynamic action."
DOI 10.3354/meps026085. PDF: https://www.int-res.com/articles/meps/26/m026p085.pdf
Acartia tonsa fed Rhodomonas baltica, 18 C, 27 psu (methods).

Table 3 (read from the page image, p. 90), per ug copepod dry weight per day:

| Food | I (ingestion) | G (egg prod.) | R / E (resp. or excretion) | AE = (G+R)/I | NGE = G/(G+R) | GGE = G/I |
|---|---|---|---|---|---|---|
| ug C/l, ng C/ugdw/d | | | | | | |
| 0 | 0 | - | 16 | - | - | - |
| 92 | 128 | 63 | 41 (44) | 0.81 (0.84) | 0.60 (0.59) | 0.49 |
| 367 | 464 | 188 | 66 | 0.55 | 0.74 | 0.41 |
| 916 | 600 | 234 | 73 (79) | 0.51 (0.52) | 0.76 (0.75) | 0.39 |
| ug N/l, ng N/ugdw/d | | | | | | |
| 0 | 0 | - | 5 | - | - | - |
| 19 | 26 | 13 | 13 | 0.95 | 0.50 | 0.49 |
| 75 | 94 | 37 | 23 | 0.64 | 0.62 | 0.40 |
| 187 | 121 | 48 | 25 | 0.60 | 0.65 | 0.40 |

(Own calculation) NH4-N excreted / N ingested = 13/26 = 0.50 (low food),
23/94 = 0.245 (intermediate), 25/121 = 0.21 (saturating).
(Own calculation) Egested N / ingested N = 1 - AE = 0.05, 0.36, 0.40.
(Own calculation) Respired C / ingested C = 41/128 = 0.32, 66/464 = 0.14, 73/600 = 0.12.
(Own calculation) Starved vs saturated: excretion 5 vs 25 ng N/ugdw/d (5x);
respiration 16 vs 73 ng C/ugdw/d (4.6x).
=> The "0.25 excreted, 0.15 respired" quoted by Kuijper et al. 2004 corresponds to
the intermediate food level (367 ug C/l = 75 ug N/l); at full saturation the
numbers are 0.21 and 0.12; near the threshold they are 0.50 and 0.32.

Quotes:
- "Rates of ammonia excretion and respiration increased with algal concentration in
  a decelerating manner. Respiration and excretion rates of copepods fed at
  saturation food concentration were more than 4 times higher than those for
  starved individuals." (abstract)
- "The conversion efficiencies expressed in terms of nitrogen showed similar trends,
  except that nitrogen assimilation was more (60 to 95 %) and net growth less (50 to
  65 %) efficient than the corresponding carbon conversions." (p. 91)
- "...ammonia constitutes more than 80 to 90 % of excreted nitrogen in copepods
  (Corner & Newell 1967, Mayzaud 1973, ...)" (methods) and "Urea constitutes less than
  10 % of total N-excretion" (discussion).
- Atomic O:N was "nearly constant and independent of food concentration"
  (Table 2: 8.17, 8.89, 7.80, 7.88).
- Caveat by the authors: budgets assume "that no dissolved organics are excreted or
  otherwise lost by the copepods" - so AE (and the "assimilated" pool) include any
  DOM loss; sloppy feeding is not separated.

### 1.2 Saba 2010 (PhD dissertation, VIMS) chapter 4 = Saba, Steinberg & Bronk 2011, JEMBE 404:47-56 [F, dissertation version]
"The relative importance of sloppy feeding, excretion, and fecal pellet leaching in
the release of dissolved carbon and nitrogen by Acartia tonsa copepods."
JEMBE DOI 10.1016/j.jembe.2011.04.013 (journal version not read).
Dissertation full text: https://scholarworks.wm.edu/server/api/core/bitstreams/01c41079-2d70-422c-8070-fa18d97d6dcd/content
(item: "The role of copepods and heterotrophic dinoflagellates in the production of
dissolved organic matter and inorganic nutrients", W&M ScholarWorks).
Acartia tonsa on the diatom Thalassiosira weissflogii, 20 C.

- Chapter abstract: "Excretion and sloppy feeding were the dominant modes of DOC
  production (11 and 5% of particulate organic C ingested, respectively) and NH4+
  release (34 and 8% of particulate organic nitrogen, PON, ingested, respectively).
  Urea, however, was predominately produced via sloppy feeding and fecal pellet
  leaching (10% and 6% of PON ingested, respectively). Urea release via sloppy
  feeding accounted for 54% of total measured nitrogen (TMN; NH4+ + urea) release.
  TMN release was > 100% of copepod body N d-1".
- Urea excretion proper came out slightly negative and was set to 0.
- Figure 5 (N flow, ng N ind-1 h-1; % of N removed from suspension):
  removed 53 (100 %); sloppy feeding (NH4+, urea) 8 (15 %); excretion (NH4+, urea)
  15 (28 %); egestion (PON) 12 (23 %), of which leaching (urea) 3 (6 %) and pellets
  (PON) 9 (17 %); growth/egg production (PON) 18 (34 %).
  Egestion N was derived from pellet C with pellet C:N 4.3 (Butler & Dam 1994).
- Figure 4 (C flow, % of C removed): sloppy feeding DOC 5 %, excretion DOC 10 %,
  egestion POC 18 %, respiration + growth 67 % (by difference).
- (Own calculation) Dissolved N, all routes: 15 + 28 + 6 = 49 % of N removed
  (figure basis), or 34 + 8 + 10 + 6 = 58 % of PON ingested (abstract basis).
  The two bases differ because "removed" includes the sloppy-feeding loss that was
  never ingested.
- "Nonetheless, more N was released as NH4+ and urea (via sloppy feeding,
  excretion, and fecal pellet leaching) or egested as PON than accumulated into
  growth/egg production (PON)."
- "Release via excretion and fecal pellet leaching was consistently lower in
  non-feeding copepods compared to feeding copepods... Thus, zooplankton excretion
  rates measured in the absence of food will likely underestimate excretion."
- The authors themselves suspect the 8 % NH4 + 10 % urea "sloppy feeding" N may
  partly be bacterial transformation in the first 20 min or basal excretion -
  i.e. the sloppy-feeding vs excretion split is uncertain; the total is firmer.

### 1.3 Saba, Steinberg & Bronk 2009, MEPS 386:147-161 [F]
"Effects of diet on release of dissolved organic and inorganic nutrients by the
copepod Acartia tonsa." DOI 10.3354/meps08070.
PDF: https://www.int-res.com/articles/meps_oa/m386p147.pdf
Acartia tonsa, 20 C (cultures/incubations), diets: heterotrophic dinoflagellates
(uZ), diatom T. weissflogii, mixed; ingestion 66-72 % body C d-1 (Expt B).
- NH4+ release 1.4-17 ng N ind-1 h-1; urea 0-4.1 ng N ind-1 h-1.
- Bulk DON could not be measured (high NOx background); DPA below detection, "Thus,
  the released organic N we report is urea."
- "the portion of total N release as urea (0.6 to 6.6% in uZ, 13 to 16% in DIATOM,
  and 32 to 59% in MIX; Fig. 3c) is similar to that measured by Miller & Glibert
  (1998; 30 to 54%)".
- Citing Miller & Glibert 1998: "organic N excretion (urea and dissolved primary
  amines, DPA) by Acartia tonsa copepods was between 62 and 89% of total N excreted
  in mesocosm experiments".
- DOC release as % of food C ingested: 67-100 % on O. marina, 5.8-20 % on
  T. weissflogii (includes sloppy feeding and pellet leaching).
- NH4+ and DOC release highest on the carnivorous diet, lowest on the mixed diet
  (attributed to higher C and N GGE on mixed diets); release not proportional to
  ingestion ("uncoupling of ingestion and nutrient release rates", abstract).
- N release as a share of ingested N is not tabulated in this paper.

### 1.4 Other measurements of ration dependence
- Gardner & Paffenhofer 1982, J Plankton Res 4:725 [A] (Eucalanus pileatus on
  Thalassiosira, three food levels): "animals released ammonium at rates ranging
  from 40 to > 100% of their N ingestion rates"; animals starved 18-22 h released
  ~60 % of the well-fed rate (paraphrased from abstract: ~30 vs 49 nmol NH4 mg
  AFDW-1 h-1). https://academic.oup.com/plankt/article-abstract/4/3/725/1486439
- Miller & Landry 1984, Mar Biol (DOI 10.1007/BF00393012) [S]: Calanus pacificus
  excretion was "a constant 6.6 nM NH4+ copepod-1 h-1 or about 10% of body N d-1",
  i.e. ingestion-independent. Conflicts with Kiorboe 1985 (4-5x increase with
  food) and Saba 2010 (fed > non-fed). Kiorboe's acclimation for 2 d to each food
  level is the more relevant protocol for a model with sustained rations.
- Hasegawa, Koike & Mukai 2001, MEPS 210:167-174 [A] (15N, natural coastal
  plankton, Japan): "The percentage of accumulation/removal varied from 9 to 75%.
  In 3 out of 5 experiments, the copepods released dissolved nitrogen from their
  food rather than accumulating it." DOI 10.3354/meps210167. Moller et al. 2003
  [F] cite these studies as showing "a total release of up to 91% of dissolved
  organic or inorganic nitrogen removed from suspension and 69% of carbon removed".
- Saba 2010 dissertation abstract [F]: "A large proportion of ingested PON was lost
  as dissolved NH4+ and urea from copepods via all release mechanisms."

### 1.5 Temperature
- Ikeda, Kanno, Ozaki & Shinada 2001, Mar Biol 139:587-596 (DOI
  10.1007/s002270100608) [S]: global copepod data, -1.7 to 29.0 C; for ammonia
  excretion 38 species / 58 data sets; body mass and temperature explained 74-80 %
  of the variance; Q10 1.8-2.0 for ammonia excretion; N was the best body-mass unit.
  Coefficients of the regression NOT retrieved (Springer paywall; abstract elided
  in Semantic Scholar) - do not use a guessed equation.
- Hernandez-Leon et al. 2008, J Plankton Res 30:577-585 [A]: specific
  ammonium excretion "highest in equatorial waters and decreased rapidly pole-ward";
  O/N indicated "a protein-based metabolism with the exception of the temperate and
  subpolar regions, which showed a more lipid-based metabolism."
- Ikeda & Motoda 1978 (Fish Bull) [F] used temperature- and size-dependent lab
  excretion functions; useful only as precedent.
- Inference (no source): with Q10 ~2 both for ingestion and excretion, the
  share-of-ingestion parameters above can be used across 5-14 C; the absolute
  basal (starved) excretion should be Q10-scaled.

---------------------------------------------------------------------------

## 2. Assimilation efficiency N vs C; egestion; N growth efficiency

- Kiorboe 1985 [F] (Acartia, Rhodomonas): AE_N 0.95 / 0.64 / 0.60 vs AE_C 0.81 /
  0.55 / 0.51 at low / intermediate / saturating food; GGE_N 0.49 / 0.40 / 0.40;
  GGE_C 0.49 / 0.41 / 0.39; NGE_N 0.50-0.65 vs NGE_C 0.60-0.76. N is assimilated
  more efficiently but used less efficiently for growth than C. AE here is
  "apparent" (includes DOM loss).
- Ikeda & Motoda 1978, Fish Bull 76 (Table 3, compilation) [F, secondary for the
  originals]: https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/fish-bull/ikeda.pdf
  - Calanus finmarchicus + C. helgolandicus: N digestion efficiency 54-68 % (mean
    62), N gross growth efficiency 14-34 % (Corner et al. 1967).
  - Calanus finmarchicus + helgolandicus: N digestion 62 %, K1(N) 27 %; P digestion
    77 %, K1(P) 17 % (Butler et al. 1970). K1(N) 21-38 % (Butler et al. 1969).
  - K1(N) 36 % (Corner et al. 1965).
  - Carbon K1: Pseudocalanus elongatus 14-18 %, Temora longicornis 17-27 % (Harris &
    Paffenhofer 1976); C. helgolandicus 3.7-35 % (Paffenhofer 1976).
  - Digestion (assimilation) efficiency, any element: Temora longicornis 52-98 %
    (32P, Berner 1962); C. finmarchicus 53-78 % (14C, Marshall & Orr 1955b).
- Saba 2010 [F]: egestion 23 % of N removed (17 % as pellet PON + 6 % leached as
  urea); growth/eggs 34 % of N removed; egestion 18 % of C removed.
- Moller, Thor & Nielsen 2003, MEPS 262:185-191 [F] (Calanus finmarchicus,
  C. glacialis, C. hyperboreus, Disko Bay spring bloom, 0-2 C, diatoms; 14C):
  "Half of the carbon cleared by Calanus spp. was released as DOC through sloppy
  feeding. Freshly expelled fecal pellets lost more than 20% of their carbon
  content within the first hour, corresponding to 6% of the carbon cleared."
  Budget (Fig. 3, assumes AE 60 %): "49% of the carbon removed from suspension ...
  through sloppy feeding, 31% was assimilated and 20% was egested." Authors note
  sloppy feeding is smaller when prey are small (summer), and that high food may
  speed gut passage and raise pellet leakage. DOI 10.3354/meps262185.
- Kuijper, Anderson & Kooijman 2004, J Plankton Res 26:213-226 [F]: Acartia uses
  N for egg production "with a low and constant efficiency of ~0.4 (Checkley, 1980;
  Kiorboe, 1989) and an even lower efficiency of 0.081 throughout the whole life
  cycle"; their DEB model gives N to pellets "constant at 0.312" of intake and
  compares with "fractions of 0.15 respired and 0.25 excreted observed by Kiorboe et
  al. (1985) for A. tonsa feeding on Rhodomonas baltica with a C:N of 5.75 (at high
  food concentration)". https://research.vu.nl/ws/files/2007525/172389.pdf
- Besiktepe & Dam 2002, MEPS 229:151-164 [A]: Acartia tonsa, 5 diets, 10-1150 ug C/l:
  "a decrease in estimated assimilation efficiency with increasing food
  concentration was apparent only for" the diatom and autotrophic dinoflagellate
  diets; diatoms give "relatively low assimilation efficiency". (Carbon only.)
  DOI 10.3354/meps229151.
- Straile 1997 (GGE_C ~0.26) - not redone.

---------------------------------------------------------------------------

## 3. Copepod / mesozooplankton excretion vs phytoplankton N demand

No North Sea, Skagerrak, Kattegat, Celtic Sea or Georges Bank figure was found [N]
in ~6 targeted searches. Kiorboe et al. 1990 (MEPS 59:19, Skagerrak transect,
May 1987) [F, skimmed] measured 15N uptake and copepod fecundity but gives no
copepod-excretion share. Available estimates:

| System | Share of phytoplankton N demand supplied by (meso)zooplankton excretion | Grade / source |
|---|---|---|
| Global open ocean (upper 200 m) | 12-23 % of phytoplankton + bacterial requirements | [A] Hernandez-Leon et al. 2008, JPR 30:577 (DOI 10.1093/plankt/fbn021) |
| Kuroshio and adjacent seas | 11-44 % (herbivorous zooplankton, NH4 4-24 mg N m-2 d-1) | [F] Ikeda & Motoda 1978 |
| N Pacific subtropical gyre | 40-50 % | Eppley et al. 1973, cited in Ikeda & Motoda 1978 [F secondary] |
| Long Island Sound; Washington/Oregon coast (summer) | up to 77-90 % | Harris 1959; Jawed 1973, cited in Ikeda & Motoda 1978 [F secondary] |
| NW Iberian upwelling/shelf | 30.3 % (N) and 21.7 % (P) in the multivorous food web; ~4 % in the herbivorous web | [S] Isla, Ceballos & Anadon 2004, Estuar Coast Shelf Sci 61 |
| Catalan Sea (NW Mediterranean) | 43 % on average | [S] Springer DOI 10.1007/BF00350108 (authors not verified) |
| Nantucket Shoals (near Georges Bank) | 20-40 % (from GDH activity) | [S] Springer DOI 10.1007/BF00393256 (authors not verified) |
| Concepcion shelf, Chile (dominant copepods) | up to 26.7 % of phytoplankton NH4 demand; DON (urea + non-urea) the largest excreted component | [S] Perez-Aragon et al. (academia.edu listing) |
| OMEX II, station P200 (NE Atlantic margin) | ~8 % of total phytoplankton N uptake | [S] BODC OMEX II report 19a |

Reading: 10-30 % is the typical shelf/temperate range; low values (4-8 %) occur in
diatom-dominated "herbivorous" blooms (high new production, large phytoplankton N
demand), high values in stratified/regenerating summer systems. Note most of these
count NH4 only; adding urea/DON from excretion and sloppy feeding (Saba: urea up to
~50 % of released N) would raise them.

---------------------------------------------------------------------------

## 4. Time scale / lability of the released N

- NH4+ and urea: directly usable by phytoplankton (and bacteria); no lag
  (standard; Saba 2009 [F] stresses that released DOP/urea "can be readily
  available to phytoplankton and bacteria").
- Release itself is fast: Saba 2010 [F] - sloppy-feeding release rates "were highest
  in the first 20-minutes"; "Products of sloppy feeding and excretion (DOC, NH4+,
  urea) will be rapidly released during feeding activity in the euphotic zone ...,
  and sinking fecal pellets can potentially transfer POC, PON, and urea below the
  euphotic zone." Moller et al. 2003 [F]: pellets lose >20 % of C within the first
  hour, then release declines ("most of the solutes have leaked out by this time").
- Lability of sloppy-feeding DOM: Titelman et al. 2008, Aquat Biol 2:131-141 [S]:
  sloppy feeding releases high-quality DOM readily used by microbes; copepod feeding
  raised bacterial production, phosphatase activity and DNA uptake. Vargas et al.
  2007, JMBA 87:667-674 [A/S]: bacterial growth high in Acartia-grazed water.
- Bacterial turnover of the organic fraction: dissolved free amino acids turn over
  in 11-36 h off southern California (faster by day); reviews give "minutes to
  days" for DFAA in euphotic waters; glycine >3 d [S, PMC/Springer snippets, exact
  papers not verified]. No direct urea turnover time found [N].
- Model implication (inference): treating copepod NH4 + urea + sloppy-feeding DON as
  one immediately available NH4 flux is defensible at a 1-6 h tick; a separate
  labile DON pool would turn over in ~0.5-2 days.

---------------------------------------------------------------------------

## 5. Not found / not accessed [N]

- Steinberg & Saba 2008 chapter (Nitrogen in the Marine Environment, 2nd ed.) - not
  accessible; the Saba 2010 figures are "Modified from Moller et al. (2003) and
  Steinberg & Saba (2008)".
- Moller 2005 (JPR 27:27, prey-size-dependent sloppy feeding) and Moller 2007
  (L&O 52:79) - not read; Saba 2009 [F] cites the Moller (2005) threshold: sloppy
  feeding occurs when copepod:prey ESD ratio < 55.
- Miller & Glibert 1998; Checkley 1980; Checkley et al. 1992; Corner & Davies 1971;
  Bamstedt; Mayzaud; Gaard; Daro; Hernroth; Gentsch 2009 - originals not read.
- Ikeda 1985 / Ikeda et al. 2001 regression coefficients - not retrieved.
- Saba, Steinberg & Bronk 2011 journal version - only the dissertation chapter read
  (numbers may differ slightly in the published version).
