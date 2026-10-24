# Benthos as gadoid food in the Mareld box: K, r, start biomass and diet share

Purpose: the model's `benthic_community` (logistic, r 0.82 /yr, K 10 t WW/km2, start 4.44 t WW/km2,
trawl loss ~0.2 /yr) cannot carry the gadoids' benthos demand (~1.5 t WW/km2/yr). Its maximum surplus
production is rK/4 = 2.05, or (r-F)^2 K/(4r) = 1.17 with trawling, so benthos halves in 5-8 years and the
gadoids then starve. This note asks which parameter the literature says is wrong.

Box: 57.81-58.31 N, 10.07-11.15 E, 43-533 m (mean 168 m), mud. 1 t/km2 = 1 g/m2.

Tags: **[F]** read in full text (incl. values read off a figure), **[A]** abstract or snippet only,
**[S]** secondary citation, **[D]** primary monitoring data downloaded and computed (SHARK). Untagged
arithmetic is mine. The evidence files in this folder hold the quotes and URLs:
- `evidence_A_standing_stock.md`: standing stock, composition, conversions (Q1, Q2).
- `evidence_B_production_and_food_limitation.md`: P/B, r, food limitation (Q3, Q6).
- `evidence_C_diets_and_consumption.md`: diets and consumption budgets (Q4, Q5).

## Bottom line

| Parameter | Now | Recommended | Central | Confidence |
|---|---|---|---|---|
| K, gadoid-prey fraction (t WW/km2) | 10 | 50-120 | 80 | order of magnitude: high (in-grid data); exact value: low |
| r (/yr) | 0.82 | 0.8-1.5 | 1.0 | low-medium |
| Start biomass, today (t WW/km2) | 4.44 | 40-100 | 60 (= measured edible stock, = B* at K 80, r 1.0) | medium |
| Gadoid benthos diet share (by mass) | ~0.40 | 0.25-0.35 | 0.30 | medium (North Sea data, no Skagerrak) |

**The lever the evidence supports most is a higher K**, by a factor of 5-12. The edible standing stock
measured inside the grid is 5-40x the model's K, and the model's start (4.44) is 10-40x below it.
- A higher r has some support for the short-lived prey fraction, but it cannot close the gap alone:
  r 1.5 at K 10 gives 2.8 t/km2/yr against a measured edible production of tens of t/km2/yr.
- A lower diet share (0.30 instead of 0.40) is supported, but it cuts demand only ~25 %. That leaves
  the pool at the knife edge (1.1 vs 1.17 max).
- "None" is not supported: K = 10 t/km2 contradicts every in-grid measurement.

## 1. Standing stock in and near the box (Q1)

Three long-term monitoring stations of Josefson (1990) lie inside the grid: Stn 10 = SHARK LYSE 8
(300 m, inside the Mareld wind-farm polygon), Stn 12 = SHARK SK3 (100 m, SE grid) and Stn 8 (100 m).

| Site | Depth | g WW/m2 | Basis | Tag |
|---|---|---|---|---|
| Stn 10 / LYSE 8, 1973-88 | 300 m | 50-85 (trend line; single grabs 25-140) | without large species (echinoids, Arctica, Nephrops, sea pens) | [F] fig., Josefson 1990 |
| same, incl. heart urchins | 300 m | ~65-110 | + 2.7 Brissopsis/m2 x 5-9 g | [D] + my estimate |
| Stn 12 / SK3, 1973-88 | 100 m | ~100-200 rising to ~500 | without large species | [F] fig. |
| SK3, 1996-2011 (62 grabs) | 95-100 m | 635 total; 387 without echinoids | SHARK, kg->g unit fix (see caveat) | [D] |
| Stn 8, 1983-88 | 100 m | ~105-190 | without large species | [F] fig. |
| Kattegat, Skold et al. 2018 | 23-65 m | 50-132 | without heart urchins and Arctica | [F] fig. 6c |
| North Sea, Heip et al. 1992 | mostly < 100 m | 7 g AFDW/m2 = ~45-115 WW | 16 % (polychaete) to 6 % (bivalve) AFDW/WW | [F] |

- Biomass rose by a median factor of 1.8 at 14 Skagerrak-Kattegat stations over 1972-88. The
  polychaete increase grew with depth (Josefson 1990) [F].
- **Caveat:** the SHARK 1996-2010 wet weights appear to be stored in kg although labelled g. The fix is
  my inference. It is backed by per-animal weights and by agreement with Josefson's 1988 values.
  SMHI should confirm it before the numbers are cited formally.
- The 71-146 g WW/m2 quoted earlier for the Kattegat is not in Skold et al. 2018. Its Fig. 6c gives
  50-132.
- Comparison: the deep Skagerrak is not poorer than the Kattegat or the shallower North Sea.
  - 300 m: about equal to the Kattegat and the North Sea mean.
  - 100 m: 3-5x higher.
  - This fits the Norwegian Trench being the main depocentre for organic matter (section 5).

**Epifauna (Q1, separately):**
- Nephrops, FU 3-4: UWTV 0.21 burrows/m2 in 2024, 2975 million animals, and 5063 million in 2025 [F].
  - With mean weights of 23-53 g, that is ~5-11 g WW/m2 (2024) and ~8-19 (2025) on the Nephrops
    grounds. This assumes one animal per burrow.
- Pandalus, Skagerrak strata at 100-500 m: 0.28 t/km2 in 2021 (0.48 at 200-300 m) and 0.83 in 1997,
  the series maximum [F]. These are swept-area values with no catchability correction, i.e. minima.
- Brittle stars are caught by the grabs, so they are already in the macrofauna figures:
  - SK3: Amphiura spp. 253 g WW/m2 [D];
  - 300 m: the small Amphilepis norvegica, 148 ind/m2 [D].
- Crabs and other shrimps: no area-based stock was found.
- AFDW/WW conversions (Ricciardi & Bourget 1998) [F]: polychaetes 16.0 %, bivalves with shell 5.8 %,
  ophiuroids 7.4 %, echinoids 3.5 %, decapods 16.5 %, amphipods 16.0 %.
  - Rumohr et al. 1987 [F]: Amphiura filiformis 13.7-17.0 %, A. chiajei 12.2-13.5 %.

## 2. Edible (prey-sized) fraction (Q2)

No study gives a quantitative "fish-food benthos" fraction. Cramer & Daan 1986 [F] say fish "feed only on
specific components, which are rarely singled out in benthos studies". The split below is from the
in-grid composition data [D] and is mine.

- **300 m (LYSE 8), share by number:**
  - small polychaetes 44 % (Aphelochaeta, Heteromastus);
  - small bivalves 44 % (Parathyasira, Abra nitida, Nucula);
  - crustaceans 5 %, ophiuroids 4 %, Brissopsis 0.1 %.
  - Nearly all of the "without large species" mass is prey-sized. Brissopsis may still be 15-30 % of
    the total mass.
  - Edible fraction: ~70-100 % of the without-large-species mass (my judgement; the 0.7 lower factor
    allows for sipunculans, priapulids and the largest worms), i.e. ~35-85 g WW/m2 on the 1973-88
    trend line, or ~50-75 % of the total including Brissopsis.
- **100 m (SK3), share by biomass:** heart urchins 39 % (inedible), Amphiura 40 %, polychaetes 10 %,
  bivalves 7 % (mostly the small Abra nitida), crustaceans < 1 %.
  - Fully edible: polychaetes, bivalves, crustaceans and other taxa, ~130 g WW/m2.
  - Amphiura is mostly eaten as arm tips. Regenerated biomass is ~22 % of the adult biomass (Skold
    et al. 1994) [F], so ~55 g/m2 is accessible. Haddock also eat whole discs, which makes this a
    lower bound.
  - Edible: ~185-380 g WW/m2, i.e. ~30-60 % of the total 635.
- **Box (mostly 100-300 m, deeper than 100 m on most of the area):** the edible stock is about
  **50-150 g WW/m2**, plus Nephrops 5-19 on its grounds and Pandalus 0.3-0.8.
  - This is today's standing stock under all predators and trawling, i.e. a B*, not K.

## 3. Production and an effective r for the prey fraction (Q3)

| Taxon / class | Area | P/B or r (/yr) | Method | Tag |
|---|---|---|---|---|
| Amphiura filiformis, population / adults | Skagerrak 40 m | P/B 0.46 / 0.37; lifespan >= 20 yr | cohort growth | [F] Skold et al. 1994 |
| Amphiura arm regeneration | same | 0.34 g AFDW/m2/yr = 13.3 % of P, ~0.06 /yr of B | measured | [F] (flux derived) |
| Polychaetes, small crustaceans | 6-7 C, 200 m | P/B 0.8-1.8 | Brey 1999 model, run by agent B | computed |
| Polychaeta / Crustacea / community | North Norway 69 N | 0.75 / 0.45 / 0.29 | Brey model | [F] Nilsen et al. 2006 |
| North Sea small infauna / infauna / small mobile epifauna | North Sea | 0.9 / 1.0-1.3 / 1.4-1.9 | Ecopath | [F] Mackinson & Daskalov 2007 |
| Nephrops (FU 3-4) | Skagerrak-Kattegat | M 0.2-0.3, so P/B >= 0.2-0.3 | assessment | [F] ICES 1993 |
| Pandalus (3a4a) | Skagerrak / Norw. Deep | M 0.75-1.2; 1.71 at age 0.5 | assessment | [F] ICES, WKPRAWN 2022 |
| r by longevity 1-3 / 3-10 / > 10 yr | global trawl recovery | 2.41 / 1.24 / 0.34 (r = 5.31/longevity) | meta-analysis | [F] Hiddink et al. 2019 |
| Community recovery (the current r) | global / mud | 0.82 (0.42-1.53); 0.68 on mud | meta-analysis | [S] library comment, Hiddink 2017, Pitcher 2022 |

- Measured r/(P/B) is roughly 0.7-2 (agent B, section A.4). So r is of the order of P/B, or somewhat
  above it, for the same fauna.
- Prey-fraction mix in the box:
  - At 300 m the mass is small polychaetes and bivalves. I assume the 1-3 yr class (r ~1.2-2.4); no
    source gives their lifespan, and thyasirids and nuculids can be slow-growing. That is one reason
    the central r stays at 1.0, not 1.5.
  - At 100 m Amphiura dominates (> 10 yr class, r ~0.3-0.5), and its fish-available arm flux is slow
    (~0.06 /yr of B).
  - Nephrops is slow (~0.3); Pandalus is fast (~1).
- **Effective r:** a biomass-weighted guess is **0.8-1.5 /yr (central 1.0)**. The current 0.82, which
  implies a mean lifespan of ~6.5 yr, sits at the low end.
- **Production:** with an edible stock of 50-150 and P/B ~1, production of the edible fraction is ~50-150 t WW/km2/yr, at
  least 25x the model's present maximum surplus of 2.05.

## 4. Gadoid diets: benthos share by mass (Q4)

All data are North Sea. No Skagerrak or Kattegat diet by mass was found. The Norwegian Deep abstracts
say only that haddock eat mainly polychaetes, crustaceans and ophiuroids, and that Norway pout eat
mainly euphausiids (Albert 1994) [A]. In every source euphausiids count as zooplankton, and shrimps
(Crangon, Pandalus) count as benthos.

| Species | ICES CRR 219, 1991, by age [F] | Mackinson & Daskalov juv / adult [F, EcoBase file] | Greenstreet 1997 [S via Heath 2005, F] |
|---|---|---|---|
| Cod | 0.47 (age 1), 0.29, 0.21, 0.17, 0.12 (5+) | 0.40 / 0.26 | 0.34 |
| Haddock | 0.38-0.45 (ages 1-6+) | 0.67 / 0.54 | 0.42 |
| Whiting | 0.08-0.18 | 0.29 / 0.12 | 0.24 |
| Saithe | 0.00-0.014 | 0.03 / 0.01 | 0.002 |
| Norway pout | - | 0.24 | 0.06 |

- Cramer & Daan 1986 [F]: cod and haddock got "approximately 45 %" of their food from benthos (1981).
- **The printed Table 3.4 of Mackinson & Daskalov 2007 is misaligned for predators 13-24.**
  - Its columns sum to 0.39-1.54, and it gives Norway pout 36 % crabs.
  - Use the EcoBase model file (model 457), whose columns sum to 1.
  - The local parse in `Litteraturbaserad growth rate per FG/ecopath_ns1991/diet.json` reproduces the
    broken print. Its gadoid diets should not be used. The "~28 % benthos" renormalisation should be
    rechecked if it came from there.
- **Community share** for today's Skagerrak IBTS mix (cod 0.058, haddock 0.543, whiting 0.267,
  saithe 0.121, hake 0.011), consumption-weighted (agent C):
  - without Norway pout: **0.28-0.42** (CRR 0.28, Greenstreet 0.31, M&D 0.35-0.42);
  - with Norway pout at 0.5-1.0x the biomass of the other species: **0.17-0.36**.
- The model's ~0.40 is at the top of the range. **0.30 (range 0.25-0.35)** is the better target.
  Haddock, which dominates today's mix, sets the share. The model's flatfish and wrasse would push it up
  a little, and Norway pout would pull it down.
- Demand: 0.69 t/km2 x Q/B 4.3-5.3 x share 0.30 = **0.9-1.1 t WW/km2/yr** of benthos. At share 0.40
  it is 1.2-1.5.

## 5. Budgets and limitation (Q5, Q6)

**Consumption vs production (Q5):**
- North Sea 1991 Ecopath, recomputed from the EcoBase file [F]:
  - All fish eat 30 g WW/m2/yr of benthos, 5.6 % of benthic production (7.7 % excluding sessile
    epifauna and meiofauna).
  - **Gadoids eat 3.7 g WW/m2/yr, ~0.7 %.**
  - Fish take most of the large mobile prey: crabs 82 %, Nephrops 71 %, shrimp 30 % of production.
- Cramer & Daan 1986 [F]: cod and haddock ate 0.5-0.8 g C/m2/yr, i.e. ~5-8 g WW at 0.1 g C per g WW
  (my conversion). That equals "10-25 % of the total benthos production".
- Heath 2005 [F]: fish ate 72 / 53 / 33 % of the production of benthic carnivores (crabs, shrimps,
  Nephrops) in the 1970s / 80s / 90s.
- No Skagerrak or Kattegat budget was found. The Kattegat EwE benthos parameters were not retrieved.
- **Reading:** real gadoids take a few g WW/m2/yr from a benthos producing tens to hundreds. In the model
  they take 1.5 from a pool that can yield at most 1.2-2. Only the large mobile epifauna (Nephrops,
  crabs, shrimp) is heavily used by fish, and it is a few g/m2 at most.

**Food vs predation limitation (Q6):**
- **Food limitation:** Josefson 1990 [F] tracked 14 stations, including the in-grid 300 m station.
  - Biomass rose x1.8, most at the deep offshore Skagerrak stations, lagging land run-off by about
    2 years.
  - He rejects predation release on three stated grounds and concludes the results are "in
    accordance with hypotheses that soft-sediment infauna in the area is food limited".
  - Josefson & Jensen 1992 and Josefson et al. 1993 agree [A].
- **Supply:** the Norwegian Trough buries 19.4 g C/m2/yr of organic carbon (Diesing et al. 2021) [F];
  sediment oxidation is 9.6-79 g C/m2/yr (Stahl et al. 2004) [A]. The food ceiling inferred from this
  (agent B, several assumed steps) is ~20-500 g WW/m2/yr of macrofauna production.
- **Predation (counter-evidence):** in the shallow Kattegat no-take zone, brittle stars fell 45-48 %
  as flatfish increased (Skold et al. 2018, 2025) [A].
- **Synthesis:** food sets the ceiling (K, and how fast a depleted pool regrows), and predators decide
  how much of it stands. In this depocentre the food ceiling is high. Food limitation does not justify
  a low K; it supports K as a bottom-up ceiling well above the model's 10.

## 6. Closing the budget: candidate (K, r)

Constant demand H, trawl F 0.2 /yr. Maximum sustainable H = (r-F)^2 K/(4r). B* is the upper equilibrium
of rB(1-B/K) - FB = H. All in t WW/km2 (/yr).

| K | r | max H | B* at H 1.5 | B* at H 1.1 | B* with no gadoids |
|---|---|---|---|---|---|
| 10 | 0.82 | 1.17 | none (collapse) | 4.7 | 7.6 |
| 10 | 1.5 | 2.82 | 7.3 | 7.7 | 8.7 |
| 30 | 0.82 | 3.52 | 19.9 | 20.7 | 22.7 |
| 50 | 1.0 | 8.0 | 38.0 | 38.6 | 40.0 |
| 60 | 0.82 | 7.03 | 42.8 | 43.5 | 45.4 |
| **80** | **1.0** | **12.8** | **62.1** | **62.6** | **64.0** |
| 120 | 1.0 | 19.2 | 94.1 | 94.6 | 96.0 |

- Today's start of 4.44 is the knife-edge equilibrium of the current parameters at share ~0.3
  (B* 4.7). It is not a measured state: section 136 of the resume calls the 16000 t a hand-set value
  (spawn 12000-20000 t).
- The start should be today's measured edible stock, ~60 t/km2 (range 40-100). The box is mostly
  deeper than 100 m, so it sits near the 300 m station. K then follows from B* = 60: K ~80 at r 1.0
  (K ~80-85 at r 0.82).
- The 300 m data are 1973-88, and the fauna rose after that, so 60 is a conservative start.
- With K >= ~30 the pool carries the gadoids with a margin of 2x or more; at K 80 the margin is ~10x.
- The model's K is "K net of the unmodelled predators" (invertebrates, flatfish). The literature K
  could therefore be higher than the observed standing stock. That is why the range goes to 120.

## 7. Model checks before adopting (outside this review)

- **Raising K changes what the gadoids see.** The gadoids -> benthos pair is over-saturated already at
  today's density: a h is 3.48 per t/km2, so half-saturation is ~0.3 t/km2 (`Täthetsberoende och
  interferens hos torskfiskar/model_context.md`). Resume section 136 saw a ~95 % benthos diet at K 10.
  - A 6-12x denser pool with the same attack rate only widens that.
  - Scale the pair's `max_intake_rate` by ~10/K_new as a starting point, then tune it toward a share
    of 0.25-0.35.
- **The trawl depletion was calibrated as 0.2 /yr of a whole community.** The prey fraction (small,
  short-lived, infaunal) is less sensitive per pass (Hiddink 2017/2019 longevity effect). Keeping
  F 0.2 is conservative.
- Changing K or the start does not change the observation layout. In practice retraining is still
  needed: a ~10x shift in the benthos observation channel lies outside the stored obs-norm statistics,
  and the trained behaviour was learned on a scarce benthos.

## Gaps

- No benthos biomass at 300 m after 1988.
- No Norwegian-side (NIVA, MAREANO) data for the trench slope.
- The SHARK unit fix is unconfirmed.
- No measured deep-Skagerrak P/B for polychaetes, and no A. chiajei production.
- No Skagerrak gadoid diet by mass.
- No Skagerrak or Kattegat benthos budget.
- Not obtained: Josefson 1981 (the two deep Skagerrak communities, 100 and 300 m), Rosenberg et al.
  1987/1996, Bergstad 1991, DAPSTOM, the Kattegat EwE report.
