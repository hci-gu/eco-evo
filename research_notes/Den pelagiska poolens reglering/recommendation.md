# Recommendation: what should hold the pelagic pool near today's level?

This note ranks the candidate closures (a)-(f) from the brief and gives concrete values with the arithmetic. The
evidence is in the five companion notes in this folder:
- `stock_composition.md` (Q1): which species are at the site, their weights, densities and historical levels
- `productivity.md` (Q2): present-day pool r, and today's realised surplus production
- `regulation.md` (Q3): predation M2 by predator (SMS 2023), bottom-up evidence, density dependence
- `migration_scale.md` (Q4): presence calendar, exchange rates, the grid-versus-stock scale problem
- `zooplankton_consumption.md` (Q5): Q/B, consumption share, top-down control of zooplankton

It also builds on earlier notes that were not redone:
- `../Täthetsberoendets styrka per grupp/pelagic_fish.md`: steepness and compensation
- `../Fiskeridödlighet i Skagerrak idag/ices_pelagic_f.md`: pooled F
- `../Torskfiskarnas reglering/`: the gadoid precedent and its method

Everything in this note is my inference built on those notes, unless a companion note is cited for a number. I
changed no code or library value.

Model run used throughout: resume 185.17, a 3-year CPU inference with frozen fix68/fix69 policies.

## 1. What the evidence establishes (short)

**Composition: the library pool is the wrong mix for the site** (`stock_composition.md`)
- The grid is at 57.8-58.3 N, 10.1-11.2 E (from `coordinates.txt`): ICES rectangles 44G0/45G0, SD 27.3.a.20, mostly
  100-200 m deep over mud.
- None of the 217 mapped sandeel grounds lies inside the grid, and IBTS caught essentially no sandeel there.
- Site weights, annual mean by biomass [D/C]:

  | Species | Weight |
  |---|---|
  | Herring (NSAS juveniles + WBSS + local) | 0.73 |
  | Mackerel (about 5 months a year) | 0.19 |
  | Sprat | 0.06 |
  | Horse mackerel | 0.02 |
  | Sandeel | 0 |

- The library's r_max 0.648 is the North Sea 1991 Ecopath mix, which is 36 % sandeel at r 1.02.

**Density: 3.33 t/km2 is right as an annual mean, wrong as a year-round constant** (`stock_composition.md`,
`migration_scale.md`)
- The annual mean is about 3.2 t/km2 (range 1.5-5) [C].
- Summer (Jul-Sep) is about 5.4 t/km2, herring 65 % and mackerel 28 %. Winter (Jan-Mar) is about 1.3 t/km2, 94 %
  herring, mostly 1-2-ringers.
- The IBTS winter/August ratio for all pelagics is 0.34 [D].
- Today's level is a depleted state:
  - NSAS SSB is 35-45 % of the 1950s level.
  - WBSS is about 0.4x its 1995 level.
  - Skagerrak herring catches were 109-168 kt/yr in 1991-95, against under 1 kt/yr now.

**Productivity: today's r is about half the library value** (`productivity.md`)
- The same FishBase CMSY priors with the site weights give r_max **0.44**, not 0.648.
- Recruits per spawner have fallen in NSAS (1.7x), WBSS (2.25x), mackerel (3.6x) and horse mackerel (1.6x). Sprat
  shows no decline.
- Euler-Lotka r_now = r_max (1 - ln d / 3), as in the gadoid note.
- The real stocks are not growing at r. Their realised surplus production at today's density is **0.12-0.14 /yr**,
  slightly less than stock-level catch (about 0.16 /yr). The pool is flat to slowly declining (-2 to -4 %/yr).
- NSAS herring, the dominant member, has had surplus production of 0.00-0.06 /yr since 2002 (SMS 2023,
  `regulation.md` 4.1).

**Regulation: neither predation nor food explains the stability of the real stocks** (`regulation.md`)
- **Predation is not the missing regulator.**
  - SMS 2023 M2 at herring-dominated site weights is 0.26-0.32 /yr, and M1 + M2 is 0.38-0.44.
  - The model already applies M1 0.58 + M2 0.31 = 0.89.
  - Model gadoid M2 on the pool (0.30 budgeted, 0.25-0.4 realised) is at or above the SMS gadoid M2 on herring
    (0.15-0.19).
  - Per-capita predation is weakly depensatory (elasticity -0.10 in SMS), so more of it does not regulate.
- **Food does not regulate adult biomass in the literature.**
  - Bottom-up control acts on recruitment: larval overwintering (Payne et al. 2009), sandeel via *Calanus* (van Deurs
    et al. 2009), sprat via *C. helgolandicus* (Lynam et al. 2017).
  - Herring weight-at-age elasticity to stock size is about -0.03 (`regulation.md` section 2).
- **Herring do not control North Sea zooplankton.**
  - Lynam et al. 2017 find the predation "not strong enough" to change zooplankton trajectories. Heath 2005 points to
    bottom-up control.
  - Top-down control is documented only for Baltic sprat at about 15 t/km2 of clupeids (Casini et al. 2009)
    (`zooplankton_consumption.md`).
- **Where density dependence is detectable, it acts at stock scale.**
  - Herring 1974-2022: Schaefer r0 0.60, K 3.6 Mt. None detectable since 2002.
  - Sprat: r0 1.1-1.2, K 0.8 Mt.
  - Sandeel: cohort-level.
  - Source: SMS output, `regulation.md` 4.1.

**Scale: local biomass is a window on migratory stocks** (`migration_scale.md`)
- The box is 0.6 % of the North Sea and 8-12 % of the Skagerrak. It holds about 0.4 % of NSAS biomass.
- Recruits are imported:
  - NSAS larvae drift 730-1030 km;
  - WBSS spawn at Rügen, about 490 km away;
  - mackerel and horse mackerel spawn elsewhere.
- Departure is scheduled by age (NSAS leave at age 1-2) and season (adults and mackerel leave Aug-Oct), not by local
  density.
- Mass is grown locally, but production of new individuals is not.

## 2. What the model pool does, and why

**The run (resume 185.17)**
- Annual means were 4.07 / 5.54 / 6.77 t/km2 from a start of 3.33.
- Realised net rates [C]:
  - year 1: about ln(4.07/3.33)/0.5 ≈ 0.40 /yr, reading the annual mean as the mid-year value;
  - year 1 -> 2: ln(5.54/4.07) = 0.31;
  - year 2 -> 3: ln(6.77/5.54) = 0.20.
- Ideal-food net is r_max - F = 0.648 - 0.10 = 0.548 /yr. The realised net falls short because the pool depletes its
  zooplankton (annual mean 5.88 -> 3.59 -> 2.94 t/km2).

**The food-only closure (c) settles far too high** [C]
- A straight line through the realised rate against the mean biomass of each interval (0.31 at 4.81 and 0.20 at 6.16)
  hits zero at about **B_food ≈ 8.6 t/km2**.
- That is 2.6x today's level, and it is reached only with zooplankton grazed to about a third of the observed site
  level.
- Model consumption at the doubled pool is 10-14x the zooplankton standing stock per year, 37-64 % of the site
  production. The North Sea Ecopath pool takes 1.9x its standing stock and 20-23 % of production
  (`zooplankton_consumption.md` section 5).

**Why the budget overshoots**
- The derived g pays r_max + M1 + M2 = 0.648 + 0.58 + 0.31 = **1.54 /yr** at s*.
- Literature P/B of a herring-dominated pool is about 0.9-1.1:
  - Ecopath NS 1991: juvenile herring 1.31, adult herring 0.80, sprat 2.28, mackerel 0.60;
  - weighted at the site mix, about 1.0 [C].
- Model Q/B is 6.2-7.2 against 4.5-5.3 in the literature (`zooplankton_consumption.md` section 1).

## 3. The arithmetic that decides the ranking

**Present-day r at the site weights** [C, from the per-species r_now in `productivity.md` section 3]

| Species | Weight | r_max | r_now (range) |
|---|---|---|---|
| Herring (Skagerrak NSAS/WBSS mix) | 0.73 | 0.44 | 0.35 (0.27-0.40) |
| Mackerel | 0.19 | 0.43 | 0.25 (0.20-0.35) |
| Sprat | 0.06 | 0.48 | 0.52 (0.48-0.56) |
| Horse mackerel | 0.02 | 0.39 | 0.33 (0.26-0.39) |
| **Pool** | 1.00 | **0.44** | **0.34** (0.27-0.40) |

- r_max: 0.73 x 0.44 + 0.19 x 0.43 + 0.06 x 0.48 + 0.02 x 0.39 = 0.321 + 0.082 + 0.029 + 0.008 = 0.440.
- r_now: 0.73 x 0.35 + 0.19 x 0.25 + 0.06 x 0.52 + 0.02 x 0.33 = 0.256 + 0.048 + 0.031 + 0.007 = 0.341.
- The low and high ends combine all the per-species low or high values.

**F at the site weights** [C]
- Stock-level u from `ices_pelagic_f.md`: herring HERAS mix 0.101, mackerel 0.215, sprat 0.448, horse mackerel
  0.038.
- 0.73 x 0.101 + 0.19 x 0.215 + 0.06 x 0.448 + 0.02 x 0.038 = 0.074 + 0.041 + 0.027 + 0.001 = **0.14 /yr**.
- The library F 0.10 is the central value of `ices_pelagic_f.md` (range 0.005-0.19). The site weights move it up
  because of mackerel and sprat.
- The values below use F 0.10, the current library value, and show F 0.14 as a sensitivity.

**What each candidate removes at B_ref = 3.33 t/km2 with ideal food**

| Fix | Ideal-food net at B_ref (/yr) | Removes (/yr) |
|---|---|---|
| none (today) | 0.648 - 0.10 = 0.548 | 0 |
| (b) r_max 0.648 -> 0.44 (site mix, composition only) | 0.34 | 0.21 |
| (b) r_max -> 0.34 (site mix, present productivity) | 0.24 | 0.31 |
| (b) r_max 0.34 and F 0.14 | 0.20 | 0.35 |
| (a) or (d) regional closure with K or I from stationarity | 0 | the remainder, by construction |
| (e) +0.06 M2 (mackerel/horse mackerel self-pair at site weights), outside g | 0.49 | 0.06 |
| (e) the same added to M1 (counted in g) | 0.548 | 0: it is paid back by a higher g |

- Only a regional closure closes the gap by itself.
- Lowering r_max is the largest literature-backed partial fix, and it changes what the closure has to do. It must
  come first, and K (or I) must be re-derived after it, as in the gadoid note.

## 4. Ranking by literature support

### 1. (b) Lower r_max to the site's present-day value: strongest support, do it first, partial fix

**Why**
- Two independent corrections both point down.
  - Composition: no sandeel at a 100-200 m muddy site, so r_max goes from 0.648 to 0.44 with the same FishBase priors.
  - Present productivity: recruits per spawner have fallen 1.6-3.6x in herring, mackerel and horse mackerel, so r
    goes from 0.44 to 0.34.
- This is the same correction and method the user accepted for the gadoids (r 0.51 -> 0.37).
- **Value: r_max = 0.34 /yr (range 0.27-0.40).**
  - If the user wants only the composition correction and not the productivity decline, use 0.44.

**Side effect, a check rather than a parameter**
- g is derived from r_max + M1 + M2, so the gross budget falls from 1.54 to 0.34 + 0.58 + 0.31 = 1.23 /yr (-20 %).
- Scaling the model's Q/B by the same factor gives 6.2-7.2 x 0.80 = **5.0-5.8**, close to the literature 4.5-5.3.
  This is an independent consistency check that the budget was too high.

**What it does not do**
- The ideal-food net at B_ref is still +0.24 /yr (0.34 - 0.10), and nothing caps it at higher density except food.
  Food would then settle well above today's level, at a lower B_food than 8.6.

**Confidence:** high that 0.648 is too high for the site; moderate on 0.34. The generation-time rule T = 3/r is the
largest structural uncertainty, as in the gadoid note.

### 2. (a) Regional carrying capacity K, as for the gadoids: strong support for a stock-scale closure, zero new code

**Why**
- The engine already exists (`regional_carrying_capacity`, section 185). It works for any DM with r_max and stays out
  of the derived g.
- Literature support for some stock-scale compensation is strong:
  - Clupeiformes B_MSY/B0 is 0.26 (Thorson et al. 2012).
  - Herring steepness is about 0.67 and sprat about 0.5.
  - SMS sprat shows K-type compensation, and herring 1974-2022 does too.
  - Source: `../Täthetsberoendets styrka per grupp/pelagic_fish.md` and `regulation.md` 4.
- Real stocks are held near today's level by recruitment at stock scale, which a regional (grid-mean) term
  represents. Local crowding does not hold them there.

**How to read the term at this scale**
- The literature does not support local crowding as the cap for migratory pelagics in a 60 km box
  (`migration_scale.md` section 7; pelagic_fish.md Q6).
- Read r B_region / K instead as the net effect of stock-level recruitment compensation plus net emigration, which
  rises with the regional density.
- The term is the same either way. Only its justification changes. Say this in the library comment.

**Concrete form (unchanged engine)**
- Loss per tick on every cell: (r_max / 1460) x (B_region / K), on biomass and reserve.
- It is outside g, so the ideal-food net becomes r_max (1 - B/K) - F.

**K values** [C]. There are three anchors, all at B_ref = 3.33 t/km2. The per-tick coefficient is c = r_max / (1460 K).

| Anchor | Formula | r_max | F | K (t/km2) | c per tick per t/km2 |
|---|---|---|---|---|---|
| A1, stationary at ideal food (gadoid rule) | B_ref / (1 - F/r) | 0.34 | 0.10 | **4.72** | 4.9e-5 |
| A1 with F 0.14 | | 0.34 | 0.14 | 5.66 | 4.1e-5 |
| A2, stock-anchored: real stocks at B/K = 1 - SP/r_now, SP 0.115 at the site weights | B_ref / (1 - 0.115/0.34) | 0.34 | - | **5.03** | 4.6e-5 |
| A3, realised-anchored: cancels the realised net at B_ref | r B_ref / S_real | 0.34 | 0.10 | 6.3-12.6 (S_real 0.18-0.09, estimated) | 3.7e-5 - 1.9e-5 |
| A1 with the composition-only r | B_ref / (1 - F/r) | 0.44 | 0.10 | 4.31 | 7.0e-5 |
| A1 with the library r (shown only for comparison) | | 0.648 | 0.10 | 3.94 | 1.13e-4 |

Notes on the anchors:
- **A2** is new relative to the gadoid note. It transfers the real stocks' position on their own surplus-production
  curve to the box, using SP/B at today's density (`productivity.md` section 5: 0.12-0.14 at its provisional
  weights, 0.115 at the final site weights below).
  - SP at the site weights, from the per-stock 2016-2024 rates in `productivity.md` section 5 (herring mix
    0.677 x 0.12 + 0.323 x 0.07 = 0.104, mackerel 0.00, sprat 0.63, horse mackerel 0.07): 0.73 x 0.104 + 0.19 x 0
    + 0.06 x 0.63 + 0.02 x 0.07 = **0.115 /yr**. That is below the stock-level F at the same weights (0.14), which is
    the consistency check: the real pool is slowly declining today.
  - At K 5.03 the ideal-food net at B_ref is 0.34 (1 - 0.662) - 0.10 = +0.015 /yr: roughly stationary, slightly below
    the observed slow decline of the stocks.
- **A3** needs the realised surplus at B_ref under r 0.34, which only a run gives.
  - The 0.09-0.18 is estimated from the 3-year run. Its realised net in year 1 was about 0.73x the ideal net,
    i.e. a shortfall of about 0.15 /yr.
  - Applied to r 0.34 that gives 0.24 - 0.15 = 0.09 if the shortfall is absolute, or 0.24 x 0.73 = 0.18 if it is
    proportional.

**Recommended K**
- **K = 5.0 t/km2** with r_max 0.34 and F 0.10.
  - It is the stock-anchored value (A2), between the strict ideal-food value 4.7 (A1) and the realised anchor 6.3-12.6 (A3).
  - The gadoid K was set the same way: between the ideal-food and realised anchors, then tuned against a retrained run.
- Tune within **4.7-6.5** against a retrained run.
- Expected behaviour (inference, assuming the food shortfall persists):
  - the pool settles slightly below B_ref, about 2.8-3.3 t/km2 annual mean;
  - zooplankton stays near its observed range.
- Use 4.7 only if a probe shows the realised surplus at B_ref is close to the ideal surplus.

**Advisory: do not put K on the library r**
- K 3.94 with r 0.648 imposes a loss of 0.648 x 4.07 / 3.94 = 0.67 /yr at the year-1 mean of 4.07.
- That is about double the realised surplus. It would push the pool down hard, then let it recover on a 1.54 /yr
  budget that still over-consumes zooplankton.
- Lower r first.

**Caveats**
- K is only as good as the stationarity assumption.
  - The real stocks are flat to slowly falling, so A2 is if anything slightly high.
  - The pool is a depleted state (stock_composition.md section 9). Like the gadoid K, K is today's effective
    capacity, not the historical ceiling.
- A regional K gives the policy no gradient. That is correct here: no adult behaviour inside the box changes stock
  recruitment.

**Confidence:** high in the mechanism at stock scale; moderate on K 5.0.

### 3. (d) Immigration/emigration toward a regional stock: best mechanistic fit, but new engine code and a purpose decision

**Why**
- This is what the population actually is: recruits imported from distant spawning grounds, departure scheduled by
  age and season.
- Ecosim's open-population terms (I - eB; Christensen & Walters 2004) and Atlantis' scheduled migrations exist for
  exactly this case (`migration_scale.md` section 5).
- No model found closes a domain smaller than the stock with a local logistic.

**D1, demographic exchange (Ecosim form) on the regional pool:** dB/dt |mig = I - e B_region, outside g
(`migration_scale.md` section 6)
- e = **0.8 /yr** (0.5-1.5), from 1-2-year juvenile residence and the cross-sectional age-structure losses at the site.
- I from stationarity, I = B_ref (e - r_loc), where r_loc is the local net rate without migration.
- With option (b) applied, r_loc = 0.34 - F.
- **F belongs inside the box only at the in-area rate.**
  - With explicit emigration, stock-level F double counts the fishing that happens after the fish leave.
  - The in-area rate is 0.005 (Skagerrak herring catch over HERAS biomass), up to about 0.05 for the mackerel and
    sprat fished as they pass.
- Values at B_ref = 3.33 t/km2 [C]:

  | r_max | F | r_loc | e | I (t/km2/yr) | I per tick |
  |---|---|---|---|---|---|
  | 0.34 | 0.02 | 0.32 | 0.8 | 3.33 x 0.48 = **1.60** | 1.10e-3 |
  | 0.34 | 0.10 | 0.24 | 0.8 | 1.86 | 1.28e-3 |
  | 0.34 | 0.02 | 0.32 | 0.5 | 0.60 | 4.1e-4 |
  | 0.648 | 0.10 | 0.548 | 0.8 | 0.84 | 5.8e-4 |

- e must exceed r_loc, or there is no equilibrium.
- Sensitivity to a local change Δ in the net rate: d ln B*/dΔ = 1/(e - r_loc) = 2.1 per unit.
  A wind-farm effect of ±0.1 /yr therefore moves B* by about ±21 %.
- Seasonal version: inflow in Apr-Jul, and e raised by 2-4 /yr in Aug-Oct. This reproduces the IBTS winter/August
  ratio of 0.34 and the summer mackerel pulse. It is the only option that fixes the year-round 3.33 problem.
- **Mass-balance note.** Immigrants bring mass grown elsewhere, so I is a true source outside the zooplankton budget.
  The model's grazing then reflects local feeding by fish produced elsewhere, which is what the site is.

**D2, physical relaxation (B_ext(t) - B)/tau with tau 4-18 d**
- This pins the box to an exogenous B_ext(t) = 3.33 (1 + 0.49 cos(2π(d - 213)/365)).
- A wind-farm impact can then only redistribute biomass, not change the total, by more than a few per cent.
- Use it only if the user decides the box is a window on a fixed regional stock.

**Purpose decision for the user**
- With D1 or K, local processes keep leverage on local pelagic biomass.
- With D2, they do not.
- The literature supports D2 physically (box exchange time of weeks), but D2 removes the effect the model exists to
  show. That choice is not a literature question.

**Confidence:** high in the mechanism; moderate in e; low in the seasonal shape beyond the IBTS ratio. It needs new
engine code.

### 4. (c) Food only (current state): weak support, rejected

- It settles at about 8.6 t/km2 with zooplankton at a third of the observed level and grazing at 10-14x standing
  stock per year.
- The literature regulator of adult small pelagics is not prey depletion (section 1 of this note).
- The zooplankton depletion is partly an artefact of a closed prey box. Skagerrak *Calanus* is largely imported
  through the Norwegian Trench; Skagerrak water turns over in about 100 days (`zooplankton_consumption.md` section 4).
- Keep food as a modulator, not as the closure.

**Confidence:** high that it is not sufficient.

### 5. (e) More predation M2: not supported as the regulator

- The model's M1 + M2 (0.89) is already 2x the SMS natural mortality of a herring-dominated pool (0.38-0.44).
- The missing predators (mackerel and horse mackerel on juvenile herring and sprat) are worth about 0.06 /yr at site
  weights, not 0.142. Muted seabirds would add 0.03-0.04.
- Any loss added to M1 enters the derived g and is paid back at ideal food. Only a loss kept outside g regulates.
- Per-capita predation is depensatory in SMS (-0.10) and in the model (gadoid Type III saturated, -0.29 /yr; see
  pelagic_fish.md Q4). More of it adds no compensation.
- **Flag, not a closure:** M1 0.58 (Ecopath M0 0.39 + non-FG predation 0.19) is high against the SMS residual M1 of
  0.10-0.20.
  - Lowering it would lower g and Q/B, but not the ideal-food net.
  - It is a turnover and consumption calibration, and a separate decision.

**Confidence:** high.

### (f) Other points the evidence raised

1. **Seasonal start and presence.**
   - The pool starts in January at 3.33 t/km2, while the observed winter stock is about 1.3 t/km2. The model also
     keeps the summer visitors (mackerel, WBSS adults) on the bank all winter.
   - Only a seasonal D1 fixes this. With K or (b) alone, consider reading 3.33 as the annual mean and checking the
     winter stock against about 1.3.
2. **F at site weights is 0.14**, not 0.10, if mackerel and sprat are weighted by their summer presence. This is a
   small lever (0.04 /yr) and is in the K table.
3. **The gadoid coupling.**
   - The gadoid K (1.02) was calibrated in a world where the pelagic pool was growing.
   - A pool held near 3.3 t/km2 lowers gadoid intake. Gadoid herring share was 62-76 % against an Ecopath 26-39 %
     (resume 185.7).
   - Recheck gadoid stationarity after the pelagic change.
4. **Other zooplanktivores at the site.** In Q3 the wider box also holds Norway pout (about 0.34 t/km2) and blue
   whiting (about 0.67 t/km2) (`stock_composition.md`). Both eat zooplankton and are in neither `pelagic_fish` (by the
   library's definition) nor `gadoids` (excluded there because they eat zooplankton). They are consumers of the site's
   zooplankton the model lacks. Adding them to the pool would raise its summer biomass by about a third and lower
   the pooled r; leaving them out means the model's zooplankton has one fewer predator than the real one.
5. **Zooplankton import.** The site's *Calanus* is largely advected. A closed zooplankton box will stay vulnerable to
   any planktivore surplus. That is a separate issue from the pool's closure, flagged in
   `zooplankton_consumption.md` section 4.

## 5. Suggested combination (for an A/B, not a decision)

1. **r_max 0.648 -> 0.34**
   - The site mix without sandeel is 0.44. Present productivity (Euler-Lotka on the R/SSB decline) brings it to 0.34.
   - Library comment: the site weights herring 0.73 / mackerel 0.19 / sprat 0.06 / horse mackerel 0.02 / sandeel 0,
     and research_notes/Den pelagiska poolens reglering/productivity.md.
2. **`regional_carrying_capacity` K = 5.0 t/km2** for `pelagic_fish`.
   - Derivation: A2 above, B_ref / (1 - SP/r) = 3.33 / (1 - 0.115/0.34) = 5.03.
   - The loader derives c = 0.34 / (1460 x 5.0) = 4.7e-5 per tick per t/km2.
   - Tune within 4.7-6.5 after a retrain.
3. **Keep F 0.10** (or 0.14 at site weights). Re-derive K if F changes: K = 3.33 / (1 - F/0.34) at ideal food.
4. **Leave predation as is.** Do not add M2 or M1 to regulate. Treat the M1 0.58 vs SMS question separately.
5. **Later, if the user wants the seasonal pool**, replace K with D1:
   - e 0.8 /yr, I = 1.6 t/km2/yr at in-area F 0.02;
   - seasonal I in Apr-Jul and extra e in Aug-Oct;
   - decide first between D1 (local leverage) and D2 (window on a fixed stock).
6. **Checks after retraining**, over 3-4 years:
   - pool annual mean within ±10 %/yr of 3.3 t/km2, and no starvation crash;
   - zooplankton annual mean not drifting down, staying near the observed 7-10 t/km2 range or at least the model's
     year-1 level;
   - pool Q/B 4.5-6;
   - consumption under about 30 % of zooplankton production;
   - realised M2 0.25-0.35 /yr;
   - gadoids still stationary under their K.

Implementation notes for the caller (not acted on):
- `growth_budget.py` derives g from r_max, so the r change moves g and the stored growth_rate.
- The K loss is outside g by construction (section 185.2).
- Policies trained without K on the pool will meet a different world, so a retrain is needed.
- A frozen-policy probe can measure the realised surplus at B_ref under r 0.34 before choosing between K 4.7 and 6.5.

## 6. What is literature and what is inference

**Literature (cited in the companion notes)**
- the species presence calendar;
- sandeel habitat;
- ICES stock series;
- SMS M2 and residual M1;
- Ecopath P/B and Q/B;
- the zooplankton consumption shares;
- the absence of top-down control of North Sea zooplankton by herring;
- the Ecosim and Atlantis treatment of open populations.

**Computed by the companion notes or here**
- the site weights;
- the densities;
- d and r_now;
- surplus-production rates;
- the pool-weighted r, F and M2;
- the K and I values;
- B_food 8.6.

**Inference (mine)**
- reading the K term as stock-level compensation plus density-related net emigration;
- the choice of anchor A2 for K;
- the expected settling level;
- the ranking itself.

The largest uncertainties:
- T = 3/r in the Euler-Lotka step;
- mackerel's weight (0.19, about 5 months a year);
- the realised food shortfall under r 0.34, which only a run measures.
