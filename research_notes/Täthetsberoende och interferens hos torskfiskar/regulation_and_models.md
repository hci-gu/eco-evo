# What regulates gadoid populations, and how ecosystem models implement it: cannibalism, juvenile density dependence, foraging-arena vulnerability and density-dependent mortality

Scope note: about 58 tool calls (web searches, page fetches, Crossref/Europe PMC look-ups, local text extraction). Wiley, Springer, cdnsciencepub (CJFAS), Taylor & Francis and PMC returned 403 or a browser check, so several papers rest on abstracts or snippets. Each finding says which.

- **Read in full:** Mackinson & Daskalov 2007 (Cefas Tech. Rep. 142; PDF text extracted locally, including the fitted vulnerability tables), the CSIRO Atlantis User Guide Part I (May 2025, extracted locally), and Ribeiro et al. (Icelandic EwE, arXiv).
- **Data computed for this note:** the ICES WGSAM North Sea SMS 2023 key-run outputs (`partial_m2.out`, `summary.out`), from which I computed new cannibalism-versus-biomass numbers.
- **Abstracts:** Neuenfeldt & Koster 2000, Anderson & Gregory 2000, Fromentin et al. 2001, Lorenzen & Enberg 2002, Hazlerigg et al. 2012, Kuparinen & Hutchings 2014.
- **Snippets only:** Yaragina et al. 2009, Myers & Cadigan 1993, Lorenzen 2008, Rose & Kulka 1999, Swain & Sinclair 1994, Ahrens et al. 2012, Fulton et al. 2003, OSMOSE, mizer.
- **Sibling note:** a few SMS partial-M2 numbers come from `research_notes/Restdödlighet M0 hos torskfiskar/ices_sms_assessments.md` and are marked as such.
- **Bibliography:** author lists, volumes and pages of the DOI-linked papers were confirmed through Crossref.

Model context:
- **Predation:** `a_eff = a / (1 + a*h*B_prey_visible + w*B_pred)`, with gadoid `interference` w = 1.0 per (t/km2). The value was chosen to remove 2-cell bands.
- **M1:** constant, `natural_mortality: 0.000284` per tick = 0.41 /yr.
- **Cannibalism:** the `gadoids_preys_on_gadoids` pair (library M2 0.2168 /yr, Ecopath North Sea 1991) is unchecked.
- **Spawn:** 1250-5000 t over 3600 km2, a grid mean of 0.35-1.4 t/km2 that is higher in colonies.
- **Ticks:** 1460 per year (6 h).

---

## Q1. Cannibalism as a regulator in gadoids: size, ages, density dependence, functional form

### Takeaway
Cannibalism is a real and at times dominant regulator of gadoid juveniles. It acts almost only on ages 0-2, extending to age 4 in Barents Sea cod. Measured per unit of stock biomass it is modest. In the North Sea SMS key run, cod-on-cod predation removes on average 0.07 /yr of cod biomass (range 0.03-0.16). All predation of the four SMS gadoids on each other, which is what a pooled "gadoids eat gadoids" link stands for, removes 0.23 /yr (range 0.10-0.45; 0.229 in 1991). That closely matches the library's Ecopath 1991 value of 0.217 /yr. In SMS this loss is nearly proportional to predator biomass (log-log exponent 0.89-0.98), but that is largely structural. In the Barents Sea and the Baltic, cannibalism also tracks juvenile abundance and falls when alternative prey is plentiful, so it behaves like a predation link with prey switching, not a fixed mortality.

### Cited Findings
- **NEA (Barents Sea) cod, 1947-2006.** "Cannibalism is probably the most important and also the most variable cause of natural mortality for age 1-4 Northeast Arctic cod." Survey indices of age 1-3 cod correlate with young cod in cod stomachs. "An inverse relationship between mortality induced by cannibalism and capelin abundance was found." Cannibalism was high in 1947-1965 and again in the mid-1990s. (Snippet; the functional form used is UNVERIFIED.) - [Yaragina, Bogstad & Kovalev 2009, Mar. Biol. Res. 5:75-85](https://doi.org/10.1080/17451000802512739)
- A 1994 ICES Marine Science Symposia paper, "Cannibalism and year-class strength in Atlantic cod ... (Barents Sea, Iceland, and eastern Newfoundland)", exists and is cited for cannibalism as a population regulator. (Existence confirmed; content not read.) - [IMR repository record](https://imr.brage.unit.no/imr-xmlui/handle/11250/107973)
- **Northern cod (NAFO 2J3KL), ages 0-3.** "Mortality was a density-dependent function of the adult population, increasing non-linearly with abundance." Two-year-olds in abundant year classes "experienced a significant increase in cannibalism". (Fetched abstract; no rates given.) - [Anderson & Gregory 2000, ICES J. Mar. Sci. 57:349-359](https://academic.oup.com/icesjms/article/57/2/349/620448)
- **Baltic cod.** In the eastern Baltic "an average of 25-38% of the 0-group and 11-17% of the 1-group were removed by predation by adults". A year class may lose "about 31% and 44% of the initial number as a result of cannibalism" by age 2. In the western Baltic, 19 % of the 0-group and 9 % of the 1-group are eaten per year. "Predation was most intense in 1978-1984, a period with high juvenile abundance and large adult stock sizes in both areas." (Fetched abstract.) - [Neuenfeldt & Koster 2000, ICES J. Mar. Sci. 57:300-309](https://doi.org/10.1006/jmsc.2000.0647)
- **North Sea 0-group.** Whiting "suffered substantial cannibalism and interspecific predation". The 1991 stomach programme (~39,000 whiting stomachs) found predation on young age groups much higher than previously assumed. (Snippets.) - [VLIZ 310740](https://www.vliz.be/imisdocs/publications/ocrd/310740.pdf), [VLIZ 275252](https://www.vliz.be/imisdocs/publications/ocrd/275252.pdf)
- **SMS 2023 key run, mean 2018-2022, partial M2** (computed in the sibling note):
  - Cod on cod: age 0 0.094 /yr (5 % of age-0 M2), age 1 0.123 (12 %), age 2 0.058, age 3 0.016.
  - Whiting on whiting: age 0 0.329 (21 %), age 1 0.254 (30 %, the largest single predator), age 2 0.086.
  - Most M2 on cod ages 1-3 now comes from harbour porpoise and grey seal.
  - [partial_m2.out](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/partial_m2.out)
- **New computation, SMS 2023 key run, 1974-2022.**
  - Method: biomass eaten = sum over quarters and predator/prey ages of Part.M2 x N.bar x west, divided by the prey's mean quarterly BIO. Density uses a North Sea area of 570,000 km2. 2023 (one quarter only) is excluded.
  - Cod eaten by cod per unit cod biomass: mean 0.073 /yr, range 0.029-0.159, at 0.14-0.83 t/km2 of cod. Linear fit: rate = 0.008 + 0.169 x B_cod, r = 0.96, log-log exponent 0.89.
  - Cod-on-cod M2 by prey age: age 0 0.08-0.50 /yr, age 1 0.09-0.53, age 2 0.02-0.21, with the peak in 1977-81 (cod 430-470 kt).
  - All predation among cod, whiting, haddock and saithe per unit of their pooled biomass: mean 0.233 /yr, range 0.100-0.445, at 1.45-5.36 t/km2. Linear fit: rate = 0.016 + 0.0875 x B_gadoid, r = 0.86, exponent 0.98.
  - In 1991 the pooled density was 1.92 t/km2 (1096 kt) and the rate 0.229 /yr.
  - Whiting on whiting is 0.07-0.19 /yr with no trend. Haddock never eats haddock in SMS.
  - (Computed from fetched files: [summary.out](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/summary.out), [partial_m2.out](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/partial_m2.out), [species_names.in](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/species_names.in))

### Inferences
- **The SMS fits are not empirical proof that cannibalism scales as k*B.** SMS computes M2 from predator abundance x ration x size/species suitability, with diets fitted to stomachs. This is MSVPA-type logic and the Lewy & Vinther 2004 formulation, which I did not re-read here. Prey mortality therefore rises with predator abundance by construction, and the shared downward trend of cod and its cannibalism inflates r. What the fit does give is the size of the term the multispecies model uses at North Sea densities: about 0.17 /yr per (t/km2) for cod on cod, and about 0.09-0.12 for all intra-gadoid predation. The 0.12 is the purely proportional value, 0.229/1.92 in 1991.
- **The library number is the right size.** The SMS intra-gadoid loss (0.233 /yr mean, 0.229 in 1991) agrees with the library's 0.2168 for a pooled gadoid group. Most of it is interspecific (cod and saithe eating juvenile whiting and haddock), not true cannibalism.
- **The data point to a predation link, not a fixed M(B).** In the Barents Sea and Baltic, cannibalism rises with both adult and juvenile abundance and falls with alternative prey, which is multi-prey saturation. The pair cannot be learned because the reward is group biomass, not because the biology is wrong.

---

## Q2. Where density dependence acts in the gadoid life cycle

### Takeaway
The classical view puts regulation in juvenile survival, and for cod it is well supported: density-dependent mortality was found in 8 of 11 Skagerrak coastal cod populations. Density-dependent growth in the recruited phase is also widespread: 9 of 16 populations in Lorenzen & Enberg 2002, and 14 of 16 in Lorenzen 2008. A spatial channel works through the area occupied, which expands and contracts with abundance (Gulf of St. Lawrence cod; northern cod hyperaggregation). Compensatory adult mortality is the least supported channel. In collapsed Northwest Atlantic cod, adult M rose at LOW abundance, i.e. an Allee effect. I found no study that splits regulation quantitatively among stages for a gadoid, so no percentages are given.

### Cited Findings
- **Skagerrak coastal cod (the model's own region).** The authors "analyzed 136 time series (covering from 44 to 73 yr) of juvenile cod" from "11 populations from the Norwegian Skagerrak coast". "Moderate to strong levels of DDM (direct or delayed) were estimated in eight of the 11 populations." Regulation is an interaction of density-dependent and stochastic processes, possibly linked to vegetation cover. (Fetched abstract.) - [Fromentin et al. 2001, Ecology 82:567-579](https://doi.org/10.1890/0012-9658(2001)082[0567:EODDAS]2.0.CO;2)
- **Demersal fish in general.** Myers & Cadigan estimated density-dependent juvenile mortality under measurement error, assuming "mortality proportional to log abundance". They are cited for cod year-class strength being set in the larval stage and "attenuated by density-dependent juvenile mortality". (Snippets.) - [Myers & Cadigan 1993, CJFAS 50:1576-1590](https://doi.org/10.1139/f93-179)
- **Density-dependent growth.** "It is generally assumed that fish populations are regulated primarily in the juvenile (pre-recruit) phase ... We analyse 16 fish populations ... and detect significant density-dependent growth in nine." "Regulation by density-dependent growth alone is sufficient to generate the observed relationship." (Fetched abstract.) - [Lorenzen & Enberg 2002, Proc. R. Soc. B 269:49-54](https://doi.org/10.1098/rspb.2001.1853)
- Lorenzen 2008, "Fish population regulation beyond 'stock and recruitment'": "of 16 study populations, 14 show significant density dependence" (in growth, per the snippet context). It matters most near carrying capacity. The venue is Bull. Mar. Sci. 83(1), not Rev. Fish. Sci. (Snippet.) - [Ingenta record](https://www.ingentaconnect.com/contentone/umrsmas/bullmar/2008/00000083/00000001/art00010)
- **Ontogeny (laboratory zebrafish, cited as a general pattern).** Density dependence in mortality "was strong during the early juvenile phase but declined thereafter and virtually ceased prior to maturation". Growth was "strongly density-dependent in the late juvenile and adult phases". "Density-dependent growth having the greater impact on population biomass while mortality had the greatest impact on numbers." (Fetched abstract.) - [Hazlerigg, Lorenzen, Thorbek, Wheeler & Tyler 2012, PLoS ONE 7:e37550](https://doi.org/10.1371/journal.pone.0037550)
- **Spatial: area occupied.** In southern Gulf of St. Lawrence cod, the area holding 90-95 % of the population "was density dependent, expanding as population size increased". (Snippet. Attribution to Swain & Sinclair 1994, CJFAS 51, is likely but UNVERIFIED.) - [Swain profile page](https://independent.academia.edu/SwainDouglas)
- **Spatial: hyperaggregation.** Northern cod CPUE rose as the stock fell, because the fishery tracked a contracting distribution. The basin-model (MacCall) framing is that a declining population contracts to the best habitat. (Snippets.) - [Rose & Kulka 1999, CJFAS 56(S1):118-127](https://doi.org/10.1139/f99-207)
- **Depensation.** "Increased adult mortality at low abundance can reverse compensatory population dynamics into its opposite - a demographic Allee effect." The rise in M in NW Atlantic cod is "ostensibly attributable in some populations to increased predation by seals". (Fetched abstract.) - [Kuparinen & Hutchings 2014, R. Soc. Open Sci. 1:140075](https://doi.org/10.1098/rsos.140075)
- **Stock-recruitment assumptions in the North Sea EwE.** Stock-recruitment curves "tend to 'flat' over a wide range of spawning stock sizes", which implies "strong compensatory increases in juvenile survival rate as spawning stock declines". From MSVPA data, "the compensatory response of cod and herring would be expected to be lower than that of haddock, whiting and saithe". (Fetched full text, section 4.1.) - [Mackinson & Daskalov 2007](https://www.cefas.co.uk/publications/techrep/tech142.pdf)

### Inferences
A biomass pool has two places to put these mechanisms:
1. Per-biomass production falls with density. This covers recruitment compensation, density-dependent growth and food limitation. In the simulator it is what intake interference does through the energy reserve.
2. Per-biomass loss rises with density, as density-dependent M does.

How the mechanisms map:
- **Juvenile density-dependent mortality** fewer recruits per unit of adult biomass. That is lost production, so it belongs in channel 1, not in a mortality on standing, adult-dominated biomass.
- **Density-dependent growth** is plainly channel 1.
- **Cannibalism** removes existing biomass (channel 2), but only the small juvenile part, in proportion to adult density.
- **Spatial contraction** should emerge from learned movement, provided local crowding has a local cost.

---

## Q3. How ecosystem models implement density dependence for adult gadoids

### Takeaway
None of the major ecosystem models gives adult gadoids an explicit density-dependent natural mortality as the default.
- **Ecosim:** regulation comes from foraging-arena predator dependence (the vulnerability v), from feeding-time adjustments that also scale "other mortality", and from emergent Beverton-Holt recruitment in split pools.
- **Atlantis:** Beverton-Holt recruitment plus an optional quadratic mortality mQ*B. The user guide says mQ should be low except for top predators.
- **OSMOSE:** regulation emerges from size-based opportunistic predation (including cannibalism) and starvation.
- **SMS:** M1 is a fixed 0.2 /yr, and M2 is driven by predator abundance.

The foraging-arena response is Beddington-DeAngelis without handling time. The North Sea Ecosim fits put adult gadoid predators at K = 3-6 (1991) and 2.7-20 (1973) on the multiplier scale. That is weaker predator dependence than w = 1.0 gives at local densities of 1 t/km2 or more.

### Cited Findings
**Ecopath with Ecosim**
- The Cefas Appendix 1, eq. 3, gives `C_ij = v_ij a_ij B_i B_j / (v_ij + v'_ij + a_ij B_j)` "with default setting v_ij = v'_ij". v is the exchange rate between prey states "vulnerable to predation and ... invulnerable to predation".
- At low predator biomass or high v the flow approaches Lotka-Volterra, `Cij = aBiBj`. At high predator biomass or low v it approaches donor control, `Cij = vijBi`, "so vij can be interpreted as the maximum possible instantaneous mortality rate that a predator can cause on a prey". "The default value of 2 indicates 'mixed control'." Direct estimates "are not available from the literature", and v is fitted to time series. (Fetched full text.) - [Mackinson & Daskalov 2007, Appendix 1](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- The full EwE6 form is `Q_ij = (v a B_i B_j T_i T_j S_ij M_ij / D_j) / (v + v T_i M_ij + a M_ij B_j S_j T_j / D_j)`, where T is feeding time and "Dj is the impact of handling time as a limit to consumption". The scale runs "1 to +inf, where v = 2 represents mixed-trophic control".
- The Icelandic by-predator fit gives cod 4+ v = 1.186, saithe 4+ > 100, haddock 3+ > 100, other codfish > 100. (Fetched full text, Table 7.) - [Ribeiro et al. 2019, arXiv:1810.00613](https://arxiv.org/abs/1810.00613)
- **North Sea EwE, fitted by predator** (one v per predator column):
  - 1991 model (Table 4.5): adult cod 6, adult haddock 6, adult saithe 6, adult whiting 3, hake 3; juvenile cod 4, juvenile saithe 2, juvenile whiting 1.1, juvenile haddock 1. The cannibalism link adult cod -> juvenile cod is therefore 6.
  - 1973 model (Table 4.6): adult haddock 20, adult cod 6, adult whiting 5, adult saithe 2.7; juvenile whiting 2.6, juvenile cod 2.2, juvenile haddock 1.01, juvenile saithe 1.
  - (Fetched full text. The pages are rotated, so I mapped values from text coordinates and checked them against the juvenile-cod prey row: 4, 6, 1.1, 3 for predators 13-16. The 1973 alignment is slightly less certain.) - [Mackinson & Daskalov 2007, Tables 4.5-4.6](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Same report: "at vulnerabilities above 3 the dynamics became unstable" until feeding-time factors were set. "The fraction of unexplained (other) mortality sensitive to changes in feeding time ... was assumed to be 1 for all except planktonic and sessile groups". The adult-juvenile parameters were tuned to give "'emergent' Beverton-Holt type stock-recruitment relationships". (Fetched full text, sections 4.1-4.2.)
- Origin of foraging arena theory. (Bibliographic/snippet.) - [Walters, Christensen & Pauly 1997, Rev. Fish Biol. Fish. 7:139-172](https://doi.org/10.1023/A:1018479526149); [Ahrens, Walters & Christensen 2012, Fish Fish. 13:41-59](https://doi.org/10.1111/j.1467-2979.2011.00432.x)

**Atlantis**
- Non-predation mortality is `M = (mL + mQ*x + mSt + ...) * x * mortsc`. The terms cover "disease, behaviour-based density dependence (e.g. male rivalry), predators not explicitly represented".
- Guidance: "linear mortality terms should be as low as possible ... while quadratic mortality should be low (or zero) for lower trophic levels and of moderate size for the largest of the top predators". "If high linear and quadratic mortality values are used, the model will be insensitive to species interactions."
- Separate mL and mQ are set for juveniles and adults.
- Recruitment: "The most common function applied for fish species is the Beverton-Holt (BH) function", `Rc = Sp*BHa/(Biom + BHb)`.
- Feeding: most models use a modified Holling II response. Ratio-dependent responses are available.
- (Fetched full text, sections 10.2.11, 10.6, 10.9.2.) - [CSIRO Atlantis User Guide Part I, 2025](https://research.csiro.au/atlantis/wp-content/uploads/sites/52/2025/05/AtlantisUserGuide_PartI.pdf)
- "A quadratic mortality term applied to intermediate predators (such as piscivores) is sufficient to achieve plausible model behaviour." Behaviour is far more sensitive to the grazing term than to the closure term. (Snippet.) - [Fulton, Smith & Johnson 2003, Ecol. Model. 169:157-178](https://doi.org/10.1016/S0304-3800(03)00268-0)

**OSMOSE, size spectrum, SMS**
- **OSMOSE:** "size-based opportunistic predation based on the spatial co-occurrence of a predator and its prey". Starvation applies when predation efficiency falls below a critical threshold. (Snippets; the threshold and formula are UNVERIFIED.) - [OSMOSE predation docs](https://documentation.osmose-model.org/odd_des/submodel/mort/pred.html)
- **mizer:** default stock-recruitment is "Beverton-Holt style", with calibrated R_max. (Snippet.) - [mizer vignette](https://mirror.linux.duke.edu/cran/web/packages/mizer/vignettes/mizer_vignette.pdf)
- **SMS:** M1 is a fixed 0.2 /yr for gadoids in every key run. (Sibling note.) - [natmor1.in](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/natmor1.in)
- In about 6 targeted searches I found no model that uses `M = M0 + k*B` for adult gadoids specifically. The closest is Atlantis's generic mQ.

### Inferences: foraging arena equals Beddington-DeAngelis without handling time
Set v' = v in the Cefas equation. Per-predator intake is then

    f = v a B_i / (2v + a B_j) = (a/2) B_i / (1 + (a/(2v)) B_j)

This is BD with attack rate a/2, h = 0 and **w_FA = a/(2v)**. The EwE6 D_j term puts handling back. Interference in the simulator and vulnerability in Ecosim are therefore the same mathematical object, read in opposite directions: a large v means weak interference.

On the multiplier scale, K is the factor by which a prey's predation mortality can rise above baseline as predator biomass goes to infinity. For the simulator's BD the same ratio is `1 + (1 + a h B_prey)/(w B_pred,0)`. Matching the two gives

    w B_pred,0 = (1 + a h B_prey)/(K - 1)

which is about 1/(K - 1) when handling is not limiting. So K = 2 corresponds to w*B = 1, K = 3 to 0.5, K = 6 to 0.2, and K = 1.1 to 10. With w = 1.0 at local gadoid densities of 1-5 t/km2, the simulator sits at K ≈ 2-1.2. That is more bottom-up than any fitted North Sea adult-gadoid link (K = 3-6 in 1991, 2.7-20 in 1973). This is my own derivation from the cited equations.

---

## Q4. Interference or density-dependent mortality for a pooled, age-less gadoid group

### Takeaway
The two options do different jobs.
- **Intake interference (BD / foraging arena)** represents food-mediated regulation: density-dependent growth, recruitment compensation through juvenile food, and predator dependence. Ecosim relies on it. The North Sea fits, however, suggest gadoids should carry weaker interference than w = 1.0, and raising w would not remove the biomass that cannibalism removed.
- **Density-dependent mortality**, `M1(B) = M0 + k*B`, represents losses that bypass the gadoids' own food: inter-gadoid predation, cannibalism, disease and unmodelled predators. It is Atlantis's mQ and the shape of SMS's pooled intra-gadoid predation.

The more defensible split for a biomass pool is to keep interference no stronger than now and restore the removed cannibalism M2 as a per-cell density-dependent M. Calibrate it to give about 0.22 /yr at the North Sea reference density of about 2 t/km2.

### Cited Findings
- Density-dependent growth mostly shapes biomass and density-dependent mortality mostly shapes numbers (Hazlerigg et al. 2012, quoted in Q2). Recruited-phase density-dependent growth "alone is sufficient" to produce the among-population pattern (Lorenzen & Enberg 2002, Q2).
- Atlantis: quadratic mortality stands in for "behaviour-based density dependence" and missing predators, and should be low except for top predators, or the model becomes "insensitive to species interactions". (Fetched full text; Q3.)
- Ecosim fits needed predator dependence of v ≈ 2-6 plus feeding-time effects, with "other mortality" tied to feeding time. (Fetched full text; Q3.)
- Cannibalism scales with adult and juvenile abundance and falls with alternative prey (Neuenfeldt & Koster 2000; Yaragina et al. 2009; Q1).
- Adult M can be depensatory at low abundance, which a k*B term cannot represent (Kuparinen & Hutchings 2014; Q2).

### Inferences: pros and cons
- **(a) Raise w**
  - Pro: already in the code, with a local learnable gradient that rewards spreading out.
  - Pro: literature-compatible as a form, since BD equals foraging arena, and it covers density-dependent growth and recruitment compensation.
  - Con: w = 1.0 already exceeds the predator dependence the North Sea fits support.
  - Con: it brakes only through food, so it is weak exactly when prey is abundant and the gadoids run away.
  - Con: it does not restore the 0.22 /yr sink, and it releases herring, zooplankton and benthos from predation as a side effect.
- **(b) Per-cell density-dependent M**
  - Pro: restores the missing sink with a literature-anchored size, independent of food, as cannibalism is.
  - Pro: learnable, because crowding kills locally, which matches cannibalism concentrating where adults and juveniles overlap.
  - Pro: the standard Atlantis closure device.
  - Con: the k*B shape is partly an SMS construct, and it ignores alternative prey.
  - Con: it hits adult-dominated pool biomass, while the real loss falls on ages 0-2. A pool cannot avoid this.
  - Con: a large k decouples gadoids from species interactions.
  - Con: it has no depensation, but that is acceptable for cannibalism.
- **(c) Other options**
  - Make M0 rise with hunger, as Ecosim's feeding-time-sensitive "other mortality" does. This is food-linked and overlaps with (a).
  - A fixed, non-decision cannibalism pair. This is mathematically close to (b).

---

## Implications for the model

1. **Do not use higher w as the cannibalism replacement.**
   - On Ecosim's multiplier scale, w*B_pred,0 ≈ 1/(K - 1). The fitted North Sea adult gadoid predators have K = 3-6 (1991), i.e. w*B ≈ 0.2-0.5.
   - w = 1.0 at 1 t/km2 already corresponds to K ≈ 2, and at 5 t/km2 to K ≈ 1.2.
   - The literature points to w at or below its current level, justified as density-dependent growth, not cannibalism.
   - Caveat: w = 1.0 was chosen to remove the 2-cell bands, so lowering it has a known artifact cost. A per-cell k*B term supplies a similar dispersal incentive, so the two may trade off. That needs an A/B run, not a literature verdict.
2. **Restore the sink as per-cell density-dependent M1:** `M1(c) = M0 + k*B_gadoid(c)`.
   - The defensible range is k ≈ 0.09-0.12 /yr per (t/km2), i.e. about 6.0e-5 to 8.2e-5 per tick per (t/km2) at 1460 ticks/yr.
   - 0.0875 is the SMS 1974-2022 slope (with intercept 0.016). 0.12 is purely proportional, from 0.229 /yr at the 1991 pooled density of 1.92 t/km2.
   - Either value reproduces the library's 0.2168 /yr at about 1.8-2.3 t/km2, not at 1 t/km2. Calibrating k = 0.2168 at 1 t/km2 would be roughly twice too strong.
   - Applied per cell, the term gives a local signal the policy can learn. Applied to total group biomass, it gives none.
3. **Check the brake strength before adopting it.**
   - With k ≈ 0.09-0.12 and a net surplus of 0.35-0.40 /yr, local equilibrium is r/k ≈ 3-4.6 t/km2. That is inside the observed North Sea pooled range (1.5-5.4 t/km2), but well above the grid-mean spawn density (0.35-1.4).
   - At mean density the term adds only about 0.03-0.17 /yr. Unless colonies sit at several t/km2, it will not stop 35-40 %/yr growth quickly.
   - The real North Sea brake is mostly 0-group and age-1 M2 of 1-2 /yr from gurnards, porpoises, seals and birds (sibling note). The porpoise and seal part is explicit in the model. The rest has no home in an age-less pool.
   - Measure where local gadoid density settles in a run before fixing k.
4. **Growth budget.** If the term becomes part of M1, `growth_budget.py` should count its value at the reference state (≈ 0.217 /yr at about 2 t/km2, or at the model's own reference density) in g, as it counted the checked pair before section 160. Otherwise r_max is no longer met at the reference state.
5. **No depensatory term.** The literature on adult M rising at low abundance concerns seal predation on collapsed stocks, and that is already explicit through the seal FG.
6. **Open question.** No source gives cannibalism per unit adult biomass for the Skagerrak itself. Fromentin et al. show strong juvenile density-dependent mortality there, but in numbers, which cannot be converted to a biomass rate without age-specific weights.

---

## Source status

| Source | Read as | Used for |
|---|---|---|
| [Mackinson & Daskalov 2007, Cefas TR 142](https://www.cefas.co.uk/publications/techrep/tech142.pdf) | Full text (extracted locally) | Ecosim equation, v meaning, fitted v tables 4.5/4.6, feeding-time M0, emergent B-H |
| [Ribeiro et al. 2019 (arXiv)](https://arxiv.org/abs/1810.00613) | Full text | EwE6 equation, v scale, by-predator v for cod/saithe/haddock |
| [Atlantis User Guide Part I, 2025](https://research.csiro.au/atlantis/wp-content/uploads/sites/52/2025/05/AtlantisUserGuide_PartI.pdf) | Full text (extracted locally) | mL/mQ mortality and guidance, B-H recruitment, feeding responses |
| [SMS 2023 key-run files](https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2023) | Data files; computed | Cannibalism and intra-gadoid loss vs density, by year and age |
| Sibling note `Restdödlighet M0 hos torskfiskar/ices_sms_assessments.md` | Local prior research | Partial M2 by predator 2018-22, M1 convention |
| [Neuenfeldt & Koster 2000](https://doi.org/10.1006/jmsc.2000.0647), [Anderson & Gregory 2000](https://academic.oup.com/icesjms/article/57/2/349/620448) | Abstracts | Baltic and northern cod cannibalism |
| [Fromentin et al. 2001](https://doi.org/10.1890/0012-9658(2001)082[0567:EODDAS]2.0.CO;2) | Abstract | Skagerrak juvenile DDM, 8/11 populations |
| [Lorenzen & Enberg 2002](https://doi.org/10.1098/rspb.2001.1853), [Hazlerigg et al. 2012](https://doi.org/10.1371/journal.pone.0037550), [Kuparinen & Hutchings 2014](https://doi.org/10.1098/rsos.140075) | Abstracts (Europe PMC) | DD growth; ontogeny of DD; depensatory M |
| [Yaragina et al. 2009](https://doi.org/10.1080/17451000802512739), [Myers & Cadigan 1993](https://doi.org/10.1139/f93-179), [Lorenzen 2008](https://www.ingentaconnect.com/contentone/umrsmas/bullmar/2008/00000083/00000001/art00010) | Snippets | NEA cannibalism; juvenile DDM form; DD growth 14/16 |
| [Rose & Kulka 1999](https://doi.org/10.1139/f99-207), Swain & Sinclair 1994 ([profile](https://independent.academia.edu/SwainDouglas)) | Snippets (attribution of the latter unverified) | Spatial density dependence |
| [Walters et al. 1997](https://doi.org/10.1023/A:1018479526149), [Ahrens et al. 2012](https://doi.org/10.1111/j.1467-2979.2011.00432.x), [Fulton et al. 2003](https://doi.org/10.1016/S0304-3800(03)00268-0) | Bibliographic / snippet | Foraging arena origin; quadratic closure |
| [OSMOSE docs](https://documentation.osmose-model.org/odd_des/submodel/mort/pred.html), [mizer vignette](https://mirror.linux.duke.edu/cran/web/packages/mizer/vignettes/mizer_vignette.pdf), [VLIZ 310740](https://www.vliz.be/imisdocs/publications/ocrd/310740.pdf), [VLIZ 275252](https://www.vliz.be/imisdocs/publications/ocrd/275252.pdf), [IMR record (Bogstad et al. 1994)](https://imr.brage.unit.no/imr-xmlui/handle/11250/107973) | Snippets / record only | Model mechanics; North Sea 0-group cannibalism; existence of 1994 paper |
