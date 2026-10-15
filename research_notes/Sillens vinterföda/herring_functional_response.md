# Herring functional response on zooplankton: Type II or Type III, and at what half-saturation?

Literature check for the Mareld model (Skagerrak, ~58.1 N 10.8 E, ~100 m). Written 2026-10-15.
No code or config was changed.

Companion note: `winter_food_and_feeding.md` (same folder): winter zooplankton biomass, winter ration,
krill. Its numbers are not re-derived here.

Conventions: **[F]** fulltext read (pdftotext or full HTML), **[A]** abstract read, **[S]** search-engine
snippet only (NOT verified; no number from an [S] source is used as evidence), **[2nd]** number read in a
document that cites the original. Each question separates *Literature*, *Calculation* and *Inference*.

---

## 0. Model state checked (read from the code, 2026-10-15)

- `lib/environments/ecosystem_env/state.py` lines 141-158: a predator with exactly one active prey gets
  Type III automatically; a pair-level `functional_response: 2|3` overrides it (section 141). Herring
  (`pelagic_fish`) has one prey (`zooplankton`), and `pelagic_fish_preys_on_zooplankton` in
  `fgconfig/fg_library.yaml` has no `functional_response` key, so it is Type III.
- `lib/environments/ecosystem_env/interactions.py::holling_a_eff`:
  Type II `f = aB/(1 + ahB + I)`, Type III `f = aB^2/(1 + ahB^2 + I)`, `I = w * B_pred(cell)`,
  B = *visible* prey biomass. The same `a` is used in both forms (so a's units differ implicitly).
- Library: `handling_time` 25 on the herring-zooplankton pair, `interference` 0.7 for `pelagic_fish`,
  `visibility_floor` 0.2, `dark_ratio` 0.1. Zooplankton is a decision maker and can hide (Rest).
- Herring is not the only zooplankton predator: `gadoids_preys_on_zooplankton` is checked
  (`predation_mortality` 0.089). Gadoids are generalists, so that pair is Type II.

Half-saturation of the bare (I = 0) model curves:

| a | ah (km2/t) | Type III Z_1/2 = 1/sqrt(ah) | Type II Z_1/2 = 1/(ah) |
|---|---|---|---|
| 0.04 (annual mean) | 1.0 | 1.00 t/km2 | 1.00 t/km2 |
| 0.0264 (December, x0.66) | 0.66 | 1.23 t/km2 | 1.52 t/km2 |

With Beddington-DeAngelis interference the half-saturations become
Type II `(1 + I)/(ah)` and Type III `sqrt((1 + I)/(ah))`. The value "1 t/km2" is the I = 0 case only.

---

## Sources and how they were read

| Source | Read as | Link |
|---|---|---|
| Arrhenius & Hansson 1999, MEPS 191:295-299, YOY Baltic herring consumption vs zooplankton | [F] PDF; Fig. 2 read from the rendered page image | https://doi.org/10.3354/meps191295 (PDF https://www.int-res.com/articles/meps/191/m191p295.pdf) |
| Bernreuther 2007, PhD thesis Univ. Hamburg, manuscript 4 (gastric evacuation and snatch rate vs copepod density, juvenile herring) | [F] pdftotext | https://ediss.sub.uni-hamburg.de/bitstream/ediss/1765/1/DissertationMatthiasBernreuther.pdf |
| Rose, Werner, Megrey et al. 2007, Ecol. Model. 202:184-195, herring growth with 3D-NEMURO (NEMURO.FISH consumption equation and calibrated K, Table 1) | [F] PDF | https://doi.org/10.1016/j.ecolmodel.2006.06.020 (PDF https://www.pmel.noaa.gov/foci/publications/2007/rose0518.pdf) |
| Gibson & Ezzi 1992, J. Fish Biol. 40:577-590, profitability of particulate vs filter feeding | [A] via OpenAlex | https://doi.org/10.1111/j.1095-8649.1992.tb02607.x |
| Gibson & Ezzi 1985, Mar. Biol. 88:109-116, particle concentration and feeding mode | [S] (citation confirmed via OpenAlex; abstract only in a search snippet) | https://doi.org/10.1007/BF00397157 |
| Gibson & Ezzi 1990 (prey size and filter-feeding threshold) | [2nd] via Bernreuther 2007 | (cited there) |
| Batty, Blaxter & Libby 1986, Mar. Biol. 91:371, filter feeding in the dark | [2nd] via Bernreuther 2007; citation via OpenAlex | https://doi.org/10.1007/BF00428631 |
| Batty, Blaxter & Richard 1990, Mar. Biol. 107:383, light and feeding | [S] | https://doi.org/10.1007/BF01313419 |
| Kiørboe & Munk 1986, Environ. Biol. Fishes 17:133, larval herring feeding vs nauplii density | [S] (citation via OpenAlex) | https://doi.org/10.1007/BF00001743 |
| Varpe & Fiksen 2010, Ecology 91:311, light, prey phenology and herring foraging | [A] via OpenAlex | https://doi.org/10.1890/08-1817.1 |
| Kalinkat, Rall, Uiterwaal & Uszko 2023, Front. Ecol. Evol. 11:1033818, "Empirical evidence of type III functional responses and why it remains rare" | [F] (first 100k of 105k characters, via WebFetch summary with quotes) | https://doi.org/10.3389/fevo.2023.1033818 |
| Kiørboe, Saiz, Tiselius & Andersen 2018, L&O 63:308, adaptive feeding and functional responses in copepods | [A] via OpenAlex | https://doi.org/10.1002/lno.10632 |
| Morozov & Arashkevich 2008, Math. Model. Nat. Phenom. 3:131, sigmoid response from vertical heterogeneity | [A] via OpenAlex | https://doi.org/10.1051/mmnp:2008061 |
| Morozov 2010, J. Theor. Biol., emergence of Type III zooplankton functional response | title only (no abstract in OpenAlex) | https://doi.org/10.1016/j.jtbi.2010.04.016 |
| Ribeiro et al. 2019 (arXiv 1810.00613), Icelandic EwE: Ecosim consumption equation (eq. 3) | [F] pdftotext | https://arxiv.org/abs/1810.00613 |
| mizer model description (size-spectrum model): feeding level | [F] web page via WebFetch | https://sizespectrum.org/mizer/articles/model_description.html |
| OSMOSE documentation: energy uptake "Holling's type 1" | [S] | https://documentation.osmose-model.org/bioen_odd_des/submodel/energy_uptake.html |
| Atlantis marked "Type II" in an FAO model-comparison table | [S] | https://www.fao.org/docrep/pdf/010/a1149e/a1149e04.pdf |
| Heikinheimo 2011, Ecol. Model. 222:1731, Baltic cod-herring-sprat (cod's response Type II vs III) | [S] | https://ideas.repec.org/a/eee/ecomod/v222y2011i10p1731-1742.html |
| Casini, Cardinale & Arrhenius 2004, ICES JMS 61:1267, herring/sprat selectivity vs prey relative abundance | [S] | https://doi.org/10.1016/j.icesjms.2003.12.011 |

**Failed fetches (recorded, not used for numbers):**
- Megrey et al. 2007, Ecol. Model. 202:144-164 (NEMURO.FISH herring description; the original one-box
  WCVI K values): digitalcommons.unl.edu PDF returned a Cloudflare challenge page twice (also with a
  Referer header); the corpora.tika.apache.org copy did not download; ScienceDirect not tried (paywall).
  Rose et al. 2007 (fulltext) was used instead: same consumption equation, recalibrated K.
- Varpe & Fiksen 2010 fulltext (Wiley pdfdirect): returned HTML, not the PDF. Only the abstract was read,
  so the form of their ingestion model (encounter-based, light-dependent) is not verified here.
- Springer abstract pages (Gibson & Ezzi 1985; Kiørboe & Munk 1986; Batty et al. 1986, 1990; Munk et al.
  1989): "Client Challenge" bot page via curl and an idp.springer.com login redirect via WebFetch. Semantic
  Scholar: abstract elided by the publisher, then HTTP 429.
- OSMOSE documentation pages: TLS error (curl exit 35) and "socket closed" (WebFetch) twice each.
- Atlantis (Audzijonyte et al. 2019, MEE 10:1814): NOAA repository PDF returned "Access Denied", the UTAS
  eprints page had no PDF link and the Wiley pdfdirect link was not tried after the Varpe failure. Atlantis's
  functional-response options are therefore **not verified**.
- Utne, Hjøllo, Huse & Skogen 2012 (Mar. Biol. Res. 8:527, Norwegian Sea herring consumption of
  *Calanus*): the Zenodo record was deleted after a take-down request (HTTP 410); no other copy found.
- Kulatska / Gårdmark Baltic models: no paper with a herring-on-zooplankton functional response was found.
- Not consulted: Blaxter & Holliday 1963 (review of clupeid behaviour and physiology; no online copy
  searched), Möllmann et al. 2004 original (see Q1, [2nd] only).

---

## Q1. Which functional response do herring and other clupeids show on zooplankton?

### Literature

1. **Juvenile herring, lab, copepods (Bernreuther 2007, manuscript 4) [F].** Groups of 20 juvenile
   herring, 9.9-13 cm TL, 13 C, frozen copepods ("mainly c4-6 *Cyclops* spp.", particle weight
   0.011 mg DW). There were 16 experiments at 10-1600 copepods/L, with snatch rate counted over 30 min.
   "The mean snatching rate increased rapidly with increasing food density to a rate of approx. 0.8
   snatching movements per second at 600 copepods per liter." Fitted Michaelis-Menten (Fig. 7):
   `SR = 0.849 * conc / (77.6 + conc)` (snatches/s, conc in copepods/L, R2 = 0.87). Their words: "similar to
   a Holling type II functional response". Filter feeding appeared at >= 900 copepods/L. Field densities
   where Bornholm Basin herring fed in June 2001 were "14 to 127 prey items * m-3" (sic, per m3), and the
   author concludes that filter-feeding densities "are probably only reached in plankton patches".
2. **YOY herring, field, Baltic (Arrhenius & Hansson 1999) [F].** In situ consumption (Elliott-Persson
   stomach method, explosive sampling, 4128 fish of 25-94 mm, July-November, northern Baltic proper)
   against zooplankton density in the layer from the surface to fish depth (3 to ~30 m, WP-2 90 um).
   Only zooplankton biomass was significant (temperature was not). Linear fit
   `WSC = 1.9e-3 + 4.4e-2 * zoopl` (r2 = 0.55). Type II fit (r2 = 0.63, Fig. 2):
   **`WSC = 0.21 * zoopl / (1 + 26 * zoopl)`**, WSC in g DW food per g DW fish per daylight hour, zoopl in
   g DW m-3 (zooplankton DW assumed 13 % of WW). Observed zooplankton range ~0.01-0.11 g DW m-3. The
   linear fit has a *positive* intercept. Nothing in the data points to an accelerating (sigmoid) lower limb.
3. **Feeding-mode switch (Gibson & Ezzi 1992) [A].** "At low concentrations the fish feed by selective
   capture of individual particles, but the capture rate achievable by this method is constrained by the
   maximum rate at which they can bite. Filter-feeding is not subject to this constraint because capture
   rate is directly proportional to prey concentration and above a critical prey concentration its
   capture rate exceeds that of biting." Filtering costs an estimated 1.4-4.6x more energy than biting.
   So the switch gives a saturating (Type II-like) biting curve, then a second linear rise at **high**
   density. It is an upturn above the biting plateau, not a low-density refuge.
4. Gibson & Ezzi 1985 [S]: filter feeding on *Artemia* nauplii only above ~50 nauplii/L, biting below.
   Gibson & Ezzi 1990 [2nd, via Bernreuther]: 50 % of fish filter at 18 *C. finmarchicus*/L and at 114
   medium *Artemia*/L. Batty et al. 1986 [2nd]: filter feeding in darkness at > 70 *Artemia* nauplii/L.
   These are threshold densities for filter feeding, not functional-response fits.
5. Larval herring (Kiørboe & Munk 1986) [S]: feeding rate saturated above ~30 *Acartia* nauplii/L
   (~5 ug DW/L). This is a saturating curve, but the source is snippet-only and covers larvae. Not used.
6. Light (Batty et al. 1990) [S]: biting needs > 0.001 lx on natural zooplankton. This acts on the attack
   rate, not on the shape. The model already applies it via `dark_ratio` / `light_threshold_deg`.
7. Varpe & Fiksen 2010 [A]: a "mechanistic model of ingestion rate, including both solar irradiance and
   prey abundance" explains the seasonal condition of Norwegian spring-spawning herring better than prey
   peaks do. The response form was not verified (fulltext failed).
8. Selectivity (Casini et al. 2004) [S]: selectivity "related to prey relative abundance". That is
   switching *among* zooplankton taxa, not evidence of a sigmoid on the total.

**No herring or other clupeid study found here reports a Type III (sigmoid) response on zooplankton.**
Every fit to herring data (2 studies [F], 1 [A]) is Type II or linear. Sprat, alewife and anchovy/sardine:
no functional-response fit was found or read (searched, nothing usable). Sprat is "considered" an obligate particulate
feeder, but Bernreuther 2007 [F] notes no laboratory study has tested this.

Möllmann et al. 2004 (J. Fish Biol. 65:1563, central Baltic herring and sprat feeding) [2nd, via
Bernreuther 2007]: average summer herring stomach content 0.18 % BW. These are diet and fullness data,
with no ration-vs-density fit. The original was not read.

### Calculation: lab and field half-saturations in model units

The model's Z is visible zooplankton over the whole column in t/km2 (= g WW m-2). A literature
concentration K (g WW m-3) becomes a column-equivalent `K_col = K * d`, d = depth of the layer holding
the zooplankton that herring can reach. d = 50 m and 100 m bracket it here; the YOY layer was 3-30 m.

| Source | K as published | K (g WW m-3) | K_col, d = 50 m | K_col, d = 100 m |
|---|---|---|---|---|
| Arrhenius & Hansson 1999, field YOY | 1/26 = 0.0385 g DW m-3 | 0.0385/0.13 = **0.296** | **14.8** | 29.6 |
| Bernreuther 2007, lab juveniles | 77.6 copepods/L x 0.011 mg DW = 0.854 g DW m-3 | 0.854/0.13 = 6.6 (4.3 at 20 % DW) | 213-328 | 427-657 |
| Model, I = 0 | - | - | 1.0 (Dec 1.23 / 1.52) | 1.0 |

The tank value is far above the field value. Tank arenas and frozen prey are not field conditions
(Kalinkat et al. 2023 list arena realism as a main pitfall), so only the field value is carried forward.

Lowest prey density covered by data:
- Arrhenius & Hansson: ~0.01 g DW m-3 = 0.077 g WW m-3 -> K_col-equivalent **3.8 t/km2 (50 m) to
  7.7 t/km2 (100 m)**. If one uses their own layer depth instead, 0.077 x 30 m = 2.3 t/km2.
- Bernreuther: 10/L x 0.011 mg = 0.11 g DW m-3 = 0.85 g WW m-3.
- Model winter visible Z = 0.45 t/km2 is **5-17x below the lowest field datum** (2.3-7.7 t/km2).

### Inference

Where data exist (column-equivalent >~2-8 t/km2), herring intake rises with prey density in a
saturating, Type II-like way, and no study reports an accelerating lower limb. None of the data reach
the model's winter densities, so **the literature cannot discriminate II from III at Z = 0.45-1.5**. It
only says that no sigmoid has been seen where it was measured. The fits are weak tests even within their
range: Arrhenius & Hansson gain only r2 0.55 -> 0.63 from linear to Type II, and Kalinkat et al. 2023
note that "Always incorporate low resource densities; otherwise, detecting an H3 is impossible!".
Both empirical K values are for juveniles. Adult Cmax is lower and K may differ.

---

## Q2. What do established bioenergetics and ecosystem models use?

### Literature

| Model | Response in prey density | Source |
|---|---|---|
| NEMURO.FISH, Pacific herring | Multispecies **Type II**: `C_ij = CMAX * (PD_j v_ij / K_ij) / (1 + sum_k PD_k v_ik / K_ik)`, PD in g m-3, K "half-saturation coefficient (g m-3)" | Rose et al. 2007 eq. 2 [F] |
| mizer (size spectrum; used for Baltic/North Sea) | `f = E/(E + h)`, "a standard Holling functional response type II to represent satiation" | mizer docs [F] |
| Ecosim (EwE) | Foraging arena, eq. 3: `Q_ij = v_ij a_ij B_i B_j T_i T_j S_ij M_ij / D_j / (v_ij + v_ij T_i M_ij + a_ij M_ij B_j S_ij T_j / D_j)`. **Linear in prey biomass B_i**; the denominator holds the *predator* B_j (saturation in predator density, vulnerability v); handling time enters through D_j | Ribeiro et al. 2019 [F] |
| OSMOSE | "Holling's type 1" (linear up to a maximum ingestion) | docs [S] |
| Atlantis | Type II in an FAO comparison table; Type III option not verified | [S] |
| Baltic cod-herring-sprat (Heikinheimo 2011) | Type III for **cod on clupeids** gave "more realistic stock dynamics" than II | [S]; different trophic link, not used for herring |

NEMURO.FISH K values (Rose et al. 2007 Table 1 [F], g m-3; K calibrated with PEST to steady-state
weight-at-age, "All K values were started at 0.5 g m-3"). Units: the herring model's g m-3 is wet weight
(the uM N conversion assumed "dry weight was 20% of wet weight and nitrogen was 7% of dry weight").
Zooplankton is averaged "over the top 50 m". Rose et al. say recalibration was needed because 3D-NEMURO
zooplankton densities "were generally lower than those used in our previous one-box application".

| K (g WW m-3) | WCVI | PWS | BS |
|---|---|---|---|
| Age-2+ on large zooplankton (K32) | 0.757 | 0.604 | 0.457 |
| Age-2+ on predatory zooplankton (K33) | 0.541 | 0.490 | 0.530 |
| Age-1 on small / large / predatory | 0.469 / 0.425 / 0.435 | 0.546 / 0.197 / 0.437 | 0.065 / 0.336 / 0.352 |

### Calculation

Column-equivalent K for adults (K32/K33 = 0.46-0.76 g WW m-3), with d = 50 m as in NEMURO:
0.46 x 50 = **23 t/km2** to 0.76 x 50 = **38 t/km2** (46-76 t/km2 if spread over 100 m).
Together with Arrhenius & Hansson (15-30 t/km2), **literature half-saturations are ~15-40 g WW m-2
column-equivalent**, i.e. 15-40x the model's bare 1 t/km2.

### Inference

Verified [F]: NEMURO.FISH and mizer use Type II, and Ecosim is linear in prey. OSMOSE (Type I) and
Atlantis (Type II) are consistent with that but read only as [S]. None found uses Type III for herring on
zooplankton. NEMURO K values are calibration
parameters: they absorb the model's own zooplankton level, so they agree with the field K only in
order of magnitude (both ~0.3-0.8 g WW m-3).

---

## Q3. Is Type III justified for a single-prey planktivore on one aggregated zooplankton pool?

### Literature

- Kalinkat et al. 2023 [F]: H3 mechanisms are learning, prey switching ("used as the overarching
  hypothesis to explain H3 responses"), lower effort at low density ("energetically inefficient to
  continue foraging with the same effort at low resource densities"), spatial refuges, induced prey
  defences, prey clumping, body size, and "suboptimal resources or circumstances". "The classical view
  that H3 is the 'vertebrate functional response' seems outdated". Earlier reviews put the H3 share
  "generally below ~15%". Filter feeders can show H3 when they "reduce their clearance rate if resources
  become scarce".
- Kiørboe et al. 2018 [A]: in copepods, active foragers that can switch feeding mode change "from type II
  to III", but "obligate active feeders ... follow a type II response inconsistent with the theoretical
  prediction".
- Morozov & Arashkevich 2008 [A]: column-integrated grazing by copepods can turn sigmoid "as a result of
  zooplankton accumulation and feeding in layers with high phytoplankton density", even when each layer is
  non-sigmoid. This is a model result for copepods on phytoplankton, not fish.
- Gibson & Ezzi 1992 [A] (Q1): herring's own behavioural switch adds intake at **high** density.

### Inference: each mechanism against the model

| Type III mechanism | Applies to herring? | Already in Mareld? |
|---|---|---|
| Switching to other prey as zooplankton gets scarce | Yes in nature (winter herring take larger, benthic-nektonic prey; companion note Q2; Casini 2004 [S]) | **No.** Herring has one prey. In nature switching cuts intake *of zooplankton* but keeps *total* intake. With one prey, Type III cuts total intake with nothing to replace it, which biases winter intake low. Switching among sizes or taxa *inside* the pool sums Type II curves; it does not make the pool total sigmoid unless herring leave the pool |
| Prey refuge | Possible (deep or diapausing *Calanus*, darkness) | **Yes**: zooplankton Rest/hide with `visibility_floor` 0.2, and the daylight attack modulation. Type III on top counts the refuge twice |
| Less foraging effort at low density | Yes (overwintering herring cut activity) | **Yes**: the policy's eat/rest choice and the hunger gate. Type III on top counts it twice |
| Learning | Not documented for herring on zooplankton | - |
| Prey clumping / vertical heterogeneity | Possible (Morozov-type sigmoid on a column-integrated pool) | Partly: `a` = 0.04 is already an aggregate calibration. This is the one pro-III argument, but it is theoretical, made for copepods, and its sign depends on how patchiness scales with total biomass |

The code's own rationale (state.py comment) is a model-side one: "giving sparse prey a Type III refuge"
for specialists. That is a stabilisation choice, not something measured for herring. The Baltic cod
example (Heikinheimo 2011 [S]) shows Type III being chosen for realistic dynamics in a *different* link.

**Conclusion Q3:** the literature does not support Type III for herring at the low-density end at any
stated density. The mechanisms that would produce it are either absent from the model's single-prey
structure (switching, where Type III is then a bias) or are already represented elsewhere (refuge,
effort). Type III cannot be ruled out empirically below ~2-8 t/km2 column-equivalent, because no data
exist there.

---

## Q4. Literature curves vs the model at winter (0.45-1.5) and summer (10-13 t/km2)

### Calculation A: the model's bare curve, %BW/day (f x 4 ticks x 100), h = 25

`f_III = aZ^2/(1 + ahZ^2 + I)`, `f_II = aZ/(1 + ahZ + I)`. The ratio is III/II. I = 0 is the caller's
"no interference" case. I = 5-10 is shown because it reproduces the caller's realised winter ration.

| Z (t/km2) | a | I = 0: III / II (ratio) | I = 5: III / II (ratio) | I = 10: III / II (ratio) |
|---|---|---|---|---|
| 0.45 | 0.0264 (Dec) | 1.89 / 3.66 (0.51) | 0.35 / 0.76 (0.46) | **0.19 / 0.42** (0.46) |
| 0.45 | 0.04 | 2.69 / 4.97 (0.54) | 0.52 / 1.12 (0.47) | 0.29 / 0.63 (0.46) |
| 1.0 | 0.04 | 8.00 / 8.00 (1.00) | 2.29 / 2.29 (1.00) | 1.33 / 1.33 (1.00) |
| 1.5 | 0.0264 | 9.56 / 7.96 (1.20) | 3.17 / 2.27 (1.40) | 1.90 / 1.32 (1.44) |
| 1.5 | 0.04 | 11.1 / 9.60 (1.15) | 4.36 / 3.20 (1.36) | 2.72 / 1.92 (1.42) |
| 10 | 0.04 | 15.8 / 14.5 (1.09) | 15.1 / 10.0 (1.51) | 14.4 / 7.62 (1.89) |
| 13 | 0.04 | 15.9 / 14.9 (1.07) | 15.5 / 10.9 (1.41) | 15.0 / 8.67 (1.73) |

These are curve values before the hunger gate, eat fraction and visibility.

Consistency check: with w = 0.7, I = 5-10 means herring cell densities of ~7-15 t/km2. **Please confirm
this against the run.** The realised winter 0.18-0.3 %/d matches Type III at I ~5-10, and every number
below depends on it.

Effective half-saturation with interference: Type II `(1+I)/(ah)` = 6-11 t/km2 at I = 5-10 (a = 0.04),
Type III `sqrt((1+I)/(ah))` = 2.4-3.3 t/km2.

### Calculation B: saturation fraction f/f_max, literature Type II vs model (I = 0)

| Z (t/km2) | Model III | Model II | A&H 1999, d = 50 m (K_col 14.8) | A&H, d = 100 m (29.6) | NEMURO K 0.6, d = 50 m (30.2) |
|---|---|---|---|---|---|
| 0.45 | 0.17 | 0.31 | 0.030 | 0.015 | 0.015 |
| 1.0 | 0.50 | 0.50 | 0.063 | 0.033 | 0.032 |
| 1.5 | 0.69 | 0.60 | 0.092 | 0.048 | 0.047 |
| 10 | 0.99 | 0.91 | 0.40 | 0.25 | 0.25 |
| 13 | 0.99 | 0.93 | 0.47 | 0.31 | 0.30 |

Initial slope (%BW/d per g WW m-2): model Type II a x 400 = 16 (Dec 10.6). Model Type III effective
f/Z at Z = 0.45 = 6.0 (Dec 4.2). Literature: about Cmax/K_col, e.g. YOY winter Cmax ~9.6 %/d (below) /
15-30 = 0.3-0.65. **At I = 0 the model's low-density intake per unit prey is ~16-50x the literature
value with Type II, and ~6-20x with Type III.** It is the interference term that brings realised intake down.

### Calculation C: Arrhenius & Hansson's YOY curve on Mareld densities (illustrative)

Convert WSC to wet/wet: fish DW = 0.22 WW (their regression), food DW = 0.13 WW, so x 0.22/0.13 = 1.69.
Max = 0.21/26 x 1.69 = 0.0137 per daylight hour. That gives 9.6 %/d at ~7 h (December daylight at 58 N)
and 24.6 %/d at 18 h. These are small YOY fish at summer temperatures, so the absolute values overstate adults.

| Z (t/km2) | d = 50 m, winter 7 h | d = 100 m, winter 7 h | d = 50 m, summer 18 h | d = 100 m, summer 18 h |
|---|---|---|---|---|
| 0.45 | 0.28 %/d | 0.14 %/d | 0.73 | 0.37 |
| 1.5 | 0.88 | 0.46 | 2.27 | 1.19 |
| 12 | (4.3) | (2.8) | 11.0 | 7.1 |

### Calculation D: what-if, model ceiling 16 %/d with Type II and literature K_col (not a proposal)

`16 * Z/(Z + K_col)`, I = 0:

| Z | K_col 15 | 23 | 30 | 38 |
|---|---|---|---|---|
| 0.45 | 0.47 | 0.31 | 0.24 | 0.19 %/d |
| 1.5 | 1.45 | 0.98 | 0.76 | 0.61 |
| 10 | 6.4 | 4.9 | 4.0 | 3.3 |
| 13 | 7.4 | 5.8 | 4.8 | 4.1 |

### Inference

1. **Winter.** At Z = 0.45, Type III gives ~0.46-0.54x the Type II intake at any I. Switching the pair to
   Type II roughly doubles winter intake: from 0.19-0.29 %/d (Type III, I = 5-10) to 0.42-0.63 %/d. That
   lands inside the literature winter ration of 0.4-1 %BW/d (companion note).
2. **Summer: the effect reverses.** At Z = 10-13 with I >= 5, Type III gives **1.4-1.9x more** intake
   than Type II (annual-mean a; more with the higher summer daylight a), because `ahZ^2` overwhelms `1 + I` while `ahZ` does not. A switch to Type II would also
   cut summer curve intake (e.g. from 15 to 7.6-10 %/d at Z = 10). Whether realised summer intake falls
   depends on whether the hunger gate or the curve binds. The caller's summer ration of 2.3-2.9 %/d is
   well below either curve value, which suggests the gate binds.
3. **Scale.** The literature half-saturation (15-40 t/km2 column-equivalent; field YOY and calibrated
   NEMURO only) is 15-40x the model's bare value. This only matters at the model's realised I. With
   I = 5-10, the model's effective Type II half-saturation (6-11 t/km2) is within ~2-5x of the
   literature range. That interference check was left to a separate review and decides whether `a`/`h`
   need any change.
4. Calculations C and D only show that a Type II curve with a literature-sized K yields winter
   rations of ~0.2-0.9 %/d at Mareld densities. They do not calibrate anything: juvenile data,
   layer-depth assumptions, and K values that are partly calibration artefacts.

---

## Verdict

- **Shape: Type II is supported, weakly. There is no evidence for Type III in herring.** Every fit to
  herring data (lab juveniles [F], field YOY [F]) uses Type II or a linear response. Verified models [F]:
  NEMURO.FISH and mizer use Type II, and Ecosim is linear in prey. OSMOSE (Type I) and Atlantis (Type II)
  are consistent with that but read only as [S]. No sigmoid has been reported
  for any clupeid. Herring's only documented feeding switch (biting -> filtering) adds intake at *high*
  density (tens of prey/L). The data do not reach the model's winter densities (5-17x lower), so the
  low end is empirically **uncertain**. The Type III mechanisms are either missing in a single-prey
  model (switching; Type III is then a pure intake loss) or already modelled (refuge, effort).
- **Half-saturation:** literature ~15-40 g WW m-2 column-equivalent (0.3-0.8 g WW m-3 local, juveniles
  and calibrated NEMURO). Model bare: 1 t/km2 (Dec 1.2-1.5). Effective `(1+I)/(ah)`, which is 6-11
  t/km2 at I = 5-10. Any rescaling must wait for the interference check.

## Open gaps

- No adult-herring functional response measured in the field. No data below ~0.08 g WW m-3.
- Megrey et al. 2007 one-box K values and Utne et al. 2012 (Norwegian Sea herring IBM) were not read.
- Herring cell density in the Mareld run (sets I) still needs confirming. The I = 5-10 used here is
  inferred from the realised winter ration, not read from the model.
