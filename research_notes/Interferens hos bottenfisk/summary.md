# Interference in the benthivore group: is w = 0.45 km2/t, applied per pair, supported?

Written by the research subagent, saved by the main agent (2026-10-28, after resume 223.28). Research
only: no library, manifest, code, resume or AGENTS.md edit. Builds on `reports/Täthetsberoende och
interferens hos torskfiskar.md` and `research_notes/Täthetsberoende och interferens hos torskfiskar/`
(meta-analyses, EwE vulnerabilities, density-dependent growth), which are not redone here.

Tags: [A] measured/experimental, [F] full text read, abs = abstract only, snip = search snippet only,
[D] data analysed here, J = judgement.

## Headline

1. **The unit conversion (Q1) does not work.** No source gives a Beddington-DeAngelis (BD) w in km2/t
   that transfers to a 1 km2 cell. w = m(1+x)/((1-m)P_ref) needs a reference density P_ref and the
   predator's saturation x, and no study of these fish reports either; plausible inputs give w from
   0.02 to 22 km2/t. The only fish measurements at cell scale are haul-scale elasticities of stomach
   content against own density; they include local depletion, so they bound interference from above (J).
2. **Fish evidence (Q2).** No study measures behavioural interference in adult cod, haddock or flatfish.
   Haul-scale field tests: cod on capelin, Barents Sea: no effect [F]; small cod on benthos, Baltic: no
   effect [D]; large cod (total prey) and flounder (benthos), Baltic: elasticity m ~0.19-0.21, depletion
   included [D]; haddock condition is *higher* where haddock are more abundant (abs). Flatfish site-scale
   signals (Kattegat, Irish Sea) are prey-per-consumer ratios = exploitation, which the model has per
   cell. Juvenile/nursery work attributes density effects to food; the only documented behavioural
   interference in gadoids is territoriality in newly settled cod.
3. **Structure (Q3).** The per-pair form is Beddington's mechanism for a predator that splits its time
   into segregated foraging modes and loses time to conspecifics only while searching. It does not
   double-count, but makes interference strongly prey-specific: at I = 9.5 the point values cut whiting
   88 %, herring 60 %, epifauna 20 %, infauna 14 % (cell-averaged in the brief: 60 / 40 / 14 / 12 %). The
   multi-prey shared-denominator form cuts every pair by 8 %; Crowley-Martin (CM) is prey-neutral too.
   Even on its own terms the engine over-charges: under segregated modes only conspecifics in the same
   mode interfere (I_j = w eat_j P), but the engine charges the full P in every pair. No fish evidence
   says interference is stronger on fish prey; other taxa point to it being *weakest* on buried,
   sedentary prey and growing with handling time and kleptoparasitism risk.
4. **Scale (Q4).** Skagerrak NS-IBTS hauls 2021-25 [D]: benthivore-weighted own density is **3.1x** the
   survey mean (2.1-3.8; haddock 4.4x, cod 2.5x); Baltic large cod 3.6x, flounder 2.6x. The trained
   policy's **13x** (~30x at the start) is 3-4x more clumped than real fish. With w = 0.45 and field-like
   crowding I ~1.4-2.3, not 9.5.
5. **Recommendation (Q5).** w = 0.45 has no biological basis; the defensible default is **w = 0**. If
   interference is kept, make it prey-neutral and bounded by the field data; the like-for-like
   prey-neutral copy of today's seabed effect is CM with w' ~0.01 km2/t (= 0.45/(1 + x_bed)).
   Confidence: high that the per-pair penalty on fish prey is unsupported; moderate that w = 0 is the
   right default; low on any non-zero number.

## Model facts (read from the repo)

- `interactions.py::holling_a_eff`, `predation.py`: per pair `a_eff = a B/(1 + a h B + I)` (Type III
  B^2), `I = w_X * B_X(c)`, shape (N_dm, 1, H, W); demand `B_pred * eat_j * a_eff * hunger`. Nothing
  sums a h B across prey: each pair saturates and is interfered with on its own; the eat share splits
  time linearly.
- `interference`: benthivores 0.45 (judgement carried over from the section-86 band fix, "No fish
  measurement exists"), planktivores 0.05, water-column predators 0.1, whiting 0.1, seals/porpoises 0.

## Q1. Transfer of a BD w in km2/t to 1 km2 cells

No, not directly (J). The meta-analyses report the dimensionless Hassell-Varley exponent m from arenas,
mostly invertebrates (DeLong & Vasseur 2011, median 0.7). In BD form the local exponent is
m = I/(1+x+I), so w = m(1+x)/((1-m)P_ref). The haul-scale slopes (swept area ~0.07-0.15 km2) are the
only cell-compatible measurements; they include depletion, which the model resolves explicitly.

w implied by a field slope (`tables/w_implied_by_field_elasticity.csv`) [D]:

| m | P_ref (t/km2) | x = 0 | x = 1 | x = 5.3 | x = 38 |
|---|---|---|---|---|---|
| 0.185 | 0.4 (Baltic large-cod crowding) | 0.57 | 1.1 | 3.6 | 22 |
| 0.185 | 5.1 (Skagerrak benthivore crowding) | 0.045 | 0.09 | 0.28 | 1.7 |
| 0.10 | 5.1 | 0.022 | 0.044 | 0.14 | 0.85 |

Intake per individual at k x the regional mean own density, relative to 1x (evidence rows R = k^-m;
model rows P = k x 1.0 t/km2, the run's grid mean; `tables/intake_vs_crowding.csv`):

| Form | 1x | 5x | 13x | 30x |
|---|---|---|---|---|
| HV m 0.7: tank meta-analysis median, extrapolated (implausible for fish) | 1 | 0.32 | 0.17 | 0.09 |
| HV m 0.07: population-scale density-dependent growth (earlier note) | 1 | 0.89 | 0.84 | 0.79 |
| **HV m 0.185: Baltic large cod, total prey, haul scale, depletion incl. [D]** | 1 | 0.74 | 0.62 | 0.53 |
| **HV m 0.21: Baltic flounder, benthic prey, depletion incl. [D]** | 1 | 0.71 | 0.58 | 0.49 |
| m 0: Baltic small cod on benthos [D]; Barents cod on capelin [F] | 1 | 1 | 1 | 1 |
| Model per-pair BD w 0.45, infauna (x 59) | 1 | 0.97 | 0.92 | 0.82 |
| Model per-pair BD w 0.45, epifauna (x 38) | 1 | 0.96 | 0.88 | 0.75 |
| Model per-pair BD w 0.45, herring (x 5.3) | 1 | 0.79 | 0.56 | 0.34 |
| Model per-pair BD w 0.45, whiting (x 0.25) | 1 | 0.49 | 0.24 | 0.12 |
| Shared-denominator BD w 0.45 (S = 102.6), all pairs | 1 | 0.98 | 0.95 | 0.89 |
| CM w' 0.01, all pairs (prey-neutral copy of today's seabed effect; by hand) | 1 | 0.96 | 0.89 | 0.78 |
| CM w' 0.02, all pairs | 1 | 0.93 | 0.81 | 0.64 |
| CM w' 0.05, all pairs | 1 | 0.84 | 0.64 | 0.42 |

Reading (J): the seabed pairs lose 8-12 % at 13x, inside the field envelope (0-42 %, depletion
included); the fish-prey pairs lose 44-76 %, outside it and outside the only fish-prey test (no effect).
The Baltic slopes cover hauls up to ~6x the survey mean, so 13x and 30x are extrapolations.

## Q2. Fish-specific evidence

### Field tests at cell scale (stomach content vs own density, per haul)

- **Barents Sea cod on capelin.** Fall, Johannesen, Englund, Johansen & Fiksen 2021, Ecography
  44:802-815, https://doi.org/10.1111/ecog.05473 [F, A]. 1944 stomachs, 455 stations, 2004-15; cod >= 30
  cm per km2 from each 1.4 km haul, capelin from 1.9 km acoustic segments. The final GAM "retained all
  covariates except cod density"; prediction II (consumption falls with cod density) answered "no".
  Caveat: consumption digestion-limited once capelin is available. Verdict: null for interference and
  exploitation, on fish prey.
- **Baltic cod and flounder.** Lindmark et al., bioRxiv https://doi.org/10.1101/2024.04.22.590538
  (2025-10-02, CC-BY; journal version not checked) [F]; data/code https://github.com/maxlindmark/cod-interactions
  [D]. BITS 2015-22, 5982 cod and 3882 flounder stomachs, haul biomass (kg/km2); Tweedie sdmTMB with
  spatial and spatiotemporal fields (range 15-19 km). Effects per SD of ln(density): large cod on its
  total prey -0.37 (SE 0.11); flounder on its benthic prey -0.28 (SE 0.08); small cod on its benthic prey
  0.00 (SE 0.09); small-cod density on flounder benthic prey +0.26 (SE 0.07, co-aggregation). Divided by
  SD ln(density) (2.00, 1.32): **m = 0.185 (0.08-0.29) large cod, 0.21 (0.09-0.33) flounder**, 12-14 %
  less stomach content per doubling. A GLM without spatial fields gives -0.045 and -0.02 (CIs span 0):
  without the prey field, co-location with rich prey masks crowding. Interference and exploitation not
  separated (J: depletion in the hypoxia-squeezed Baltic more likely).
- **North Sea haddock.** Hiddink, Jennings & Kaiser 2005, J. Fish Biol. 67:111-124,
  https://doi.org/10.1111/j.0022-1112.2005.00912.x (abs): at 24 sites hepatosomatic condition higher
  where haddock more abundant, at a given temperature; not ideal-free. J: crowding cost small vs site
  quality.
- **Irish Sea plaice and dab.** Johnson et al. 2015, Proc. R. Soc. B 282:20142336,
  https://doi.org/10.1098/rspb.2014.2336 [F, A]: 15 stations (1 x 2 km) along a trawling gradient; fish
  and prey declined together, "the ratio of fish to prey remained unchanged"; stomach fullness and gut
  energy constant (414 plaice, 575 dab). Exploitation framing, no interference test.
- **Kattegat flatfish and Nephrops.** Hiddink et al. 2016, J. Appl. Ecol. 53:1500-1510,
  https://doi.org/10.1111/1365-2664.12697 (abs; Dryad needs login, manuscript 403): plaice and long rough
  dab condition and stomach contents follow prey per consumer; dab no effect; Nephrops condition improved
  as Nephrops fell. Exploitation.
- **North Sea cod and whiting switching.** Rindorf, Gislason & Lewy 2006, MEPS 325:243-253 [F]: local
  predator abundance tested only on the switching slope b ("none of the reference fishes were
  significantly affected"); intake not tested. **Correction:** the `predatory_fish` library comment
  "Rindorf et al. 2006: predator density did not affect intake" should read "did not affect prey
  switching (diet composition); intake not tested". The 1998 ICES CM paper (CC:5) only proposes such a
  test [F].

### Population and aggregation scale

- **Cod and capelin, Newfoundland.** Rose & Leggett 1990, Ecology 71:33-43,
  https://doi.org/10.2307/1940245 (abs): densities positively correlated above 4-10 km, negatively
  below the 3.5 km aggregation size; "when actually feeding, cod achieved coherent, in-phase
  distributions with capelin at scales as small as 3.5 m". J: feeding aggregations sit inside a cell
  with their prey.
- **Baltic cod on herring and sprat.** Essington & Hansson 2004, CJFAS 61:2215-2226,
  https://doi.org/10.1139/f04-146 (abs): sprat mortality highly sensitive to cod biomass, herring only
  moderately. Stock-scale predator dependence incl. overlap, not behavioural interference.

### Juvenile and nursery fish (helper notes, `data/lit_juvenile_and_foraging_mode.md`)

- Flatfish nurseries: density effects exist, mechanism always food limitation; no flatfish study of
  behavioural interference or kleptoparasitism. Le Pape & Bonhommeau 2015, doi:10.1111/faf.12063 [F];
  Modin & Pihl 1994 (abs, A: 1.4 vs 10 fish/m2, lower length-at-age in the dense year); Pihl & van der
  Veer 1992, van der Veer & Witte 1993, van der Veer et al. 2010 (doi:10.3354/meps08786): growth follows
  food; Poiesz et al. 2019 [F]: digging plaice slow in summer, epibenthic flounder not (a change in a,
  not w); Ciotti et al. 2013, doi:10.3354/meps10087: density-dependent growth at ~25 km.
- Juvenile gadoids: Tupper & Boutilier 1995, doi:10.3354/meps118295 (abs): territoriality in newly
  settled cod at shelters, no intake numbers. Rogers et al. 2011, PNAS, doi:10.1073/pnas.1010314108,
  Skagerrak [F]: no density effect on age-0 cod length up to ~50 per beach-seine station, then strong
  (~5 % of hauls above; mechanism unresolved). No kleptoparasitism found in gadoids or flatfish.

## Q3. Prey-type or foraging-mode specificity

Algebra (J): Beddington's mechanism (time lost to conspecifics during search) with time split into
modes eat_j, each searching and handling prey j only: T_j = Ts_j(1 + a_j h_j B_j + c t_w P) ->
f_j = eat_j a_j B_j/(1 + x_j + wP), exactly the engine's form. Simultaneous search shares the term:
f_j = a_j B_j/(1 + sum x_k + wP).

- The per-pair form does not double-count: each pair's I acts on its share of time; the total cut is
  the intake-weighted mean of the pair cuts.
- It assumes interference only during search, diluted by the pair's own saturation, so low-x pairs
  (fish prey) are hit hardest.
- Under segregated modes only same-mode conspecifics interfere: the consistent term is w eat_j P; the
  engine uses w P in every pair, over-charging even on its own mechanism.

Cuts at I = 9.5:

| Form | Infauna | Epifauna | Herring | Whiting |
|---|---|---|---|---|
| Per-pair BD | 14 % | 20 % | 60 % | 88 % |
| Shared denominator | 8.4 % | 8.4 % | 8.4 % | 8.4 % |
| Crowley-Martin, w 0.45 | 90 % | 90 % | 90 % | 90 % |

For the same local exponent a CM coefficient must be (1+x) times smaller than a BD w (39-60x on the
seabed pairs): w'_CM ~0.45/(1 + x_bed) ~0.008-0.012 km2/t.

Foraging-mode evidence (helper notes; mostly birds and crabs, abs/snip unless marked): Stillman et al.
2002: interference set by kleptoparasitic attack distance, scaling with handling time. Yates et al.
2000 [A]: redshank on mobile Corophium interfere above ~100 birds/ha without aggression (prey
depression); godwits digging lugworm none below 300/ha. Triplet et al. 1999: oystercatcher threshold
50-100/ha, stronger when prey scarce (w not constant). Smallegange et al. 2006/2007: shore crabs at 4-16
per m2 spend 14-18 % of time in agonistic encounters and avoid conspecifics during handling too
(CM-like). Vahl et al. 2005: aggression a poor proxy for interference. Stouffer & Novak (77 datasets;
bioRxiv preprint [F]): BD only 15, CM only 11, both 23, neither 28; only 43 % consistent with
independent feeding and interference. Salmonids (snip): drift feeders defend stations, benthic feeders
non-aggressive.

J: buried infauna and burrowed Nephrops sit at the low-interference end; mobile epifauna (prey
depression) and fish prey have no fish data either way. The per-pair pattern that taxes fish prey 4-8x
more than the seabed is an untested assumption.

## Q4. Scale: is a per-cell I of 5-20 plausible?

Tank-to-cell density gap 10^3-10^4 (earlier note). Feeding aggregations are metres to a few km and
co-located with prey (Rose & Leggett).

Skagerrak NS-IBTS crowding [D] (`scripts/ibts_patchiness.py` on the extracts in
`research_notes/Torskfiskarnas artsammansättning`: 265 valid GOV hauls, Q1 and Q3 2021-25, wing-spread
swept area, q = 1; Lloyd = biomass-weighted own density / survey mean):

| Taxon | Mean t/km2 | Lloyd / mean, median (range, 10 surveys) | p90 / mean | Max / mean | Max t/km2 |
|---|---|---|---|---|---|
| Benthivore guild | 1.67 | **3.06** (2.11-3.80) | 2.1 | 6.5 | 29.5 |
| Haddock | 1.09 | 4.36 (2.64-9.53) | 2.2 | 8.2 | 27.4 |
| Dab | 0.09 | 3.09 | 2.5 | 5.3 | 16.9 |
| Plaice | 0.06 | 3.36 | 2.5 | 7.2 | 9.2 |
| Cod | 0.16 | 2.47 | 2.7 | 4.4 | 1.5 |
| Long rough dab | 0.14 | 2.42 | 2.5 | 5.1 | 1.3 |

Baltic BITS crowding [D] (Lindmark data): large cod 3.55 (2.1-5.4), flounder 2.6, small cod 5.0.

Model vs field: model 13x (~30x at start) vs field 2.5-4.4x. At w = 0.45 field crowding (1.67 x 3.06 ~
5.1 t/km2) gives I ~2.3; the model mean 1.0 t/km2 with field crowding I ~1.4. Single hauls of 17-29
t/km2 (I ~8-13) occur only in the extreme tail. I = 5-20 as the biomass-weighted average is not
plausible; as a local extreme it is.

Caveats: a haul (0.07-0.15 km2) is smaller than a cell, so 1 km2 crowding is if anything lower; IBTS
stations are on trawlable ground; the IBTS region is larger and more heterogeneous than the grid (field
crowding overstated, conservative here); the Lindmark crowding uses only hauls with stomachs. Part of
the model's clumping may be habitat (depth-based K, section 207), not behaviour; not separated. Why the
policy clumps 3-4x harder than real fish is a question about payoffs, not w: candidates refuge patches,
the depleted-patch training design, colony spawn, the 0.5 t indivisible unit (J, not investigated).

## Q5. Recommendation

1. **Biological basis for 0.45: none.** It is the section-86 band fix rescaled by w x B; no measurement
   for adult cod, haddock or flatfish.
2. **Default under the project rule: w = 0 for the benthivores** (J): exploitation is explicit per cell
   (refuge, accessible fractions, prey per cell); the two haul-scale tests that found nothing are cod on
   fish prey and small cod on benthos; the non-zero slopes (m ~0.2, Baltic) are not separated from
   depletion; haddock condition rises with local abundance; juvenile work attributes density effects to
   food.
3. **Why this differs from the earlier report (w 1-2).** That support came from m across taxa and
   population-scale density-dependent growth, both including depletion and neither at cell scale. The
   only haul-scale fish-prey test found no effect; exploitation is explicit per cell; the model's
   crowding is 3-4x beyond the field, so the "w x B held constant" rescaling was anchored to a density
   real fish do not live at.
4. **If a non-zero value is kept** (stability, bands): one prey-neutral factor per predator cell, CM
   1/(1 + w'P) on every pair or a shared denominator with S = sum a_k h_k B_k. Like-for-like replacement
   of today's seabed effect: **w'_CM ~0.01** (0.008-0.012), 11 % at 13x on every pair instead of 8-12 %
   on the seabed and 44-76 % on fish prey. Field upper bound (depletion included) m <= 0.2 at field
   crowding -> w' <= 0.05, which would cut the seabed 36 % at 13x, more than today: a bound, not a
   target. In a shared-denominator BD with S ~100, today's 0.45 gives m ~0.05 at 13x. The per-pair form
   is a structural error for a predator whose foraging modes are not shown to be segregated; even if they
   were, it should charge w eat_j P, not w P.
5. **Confidence.** High that the per-pair penalty on herring and whiting is unsupported; moderate that
   w = 0 is the right default; low on any non-zero number.
6. **Check to run later** (not run, GPU busy): w = 0 (and w'_CM = 0.01) with the same policies in the
   inference world; look for the section-86 two-cell bands; log the benthivore-weighted crowding against
   the field ~3x. If the bands return, that is a numerical/spatial problem to solve without a biological
   parameter.

## Files

- `scripts/interference_algebra.py` -> `tables/interference_algebra.csv`
- `scripts/intake_vs_crowding.py` -> `tables/intake_vs_crowding.csv`, `tables/w_implied_by_field_elasticity.csv`
- `scripts/lindmark_density_feeding.py` (Baltic SDs, naive GLM, patchiness):
  `python3 -I <script> data/lindmark_cod_interactions/aggregated_stomach_data.csv tables`
- `scripts/ibts_patchiness.py`:
  `python3 -I <script> "<research_notes>/Torskfiskarnas artsammansättning" "<research_notes>/Fiskgrupper i rutan" tables`
- `data/lindmark_cod_interactions/` (CC-BY, from GitHub; untrusted input, only parsed)
- `data/lit_juvenile_and_foraging_mode.md` (helper notes)

Access failures: Hiddink 2016 manuscript and Dryad (403 / login); Temming et al. 2007 Ecosystems (not
OA); int-res.com 429 to the helper (Rindorf 2006 PDF retrieved directly).
