# Recommendation: what should hold the gadoids near today's level?

This note ranks the candidate fixes (a)-(e) from the task by literature support, adds one candidate the evidence
raised (f), and gives concrete values. The evidence is in the four companion notes:
- `stock_trends_and_mortality.md` (Q1, Q2)
- `r_max_and_recruitment.md` (Q3)
- `density_dependence.md` (Q4)
- `predator_prey_cycles.md` (Q5)

Everything in this note is inference built on those notes. I changed no code or library value.

## The arithmetic that decides the ranking

- The pool grows at about **r_max - F = 0.51 - 0.12 = 0.39 /yr** whenever food is ideal, at any density.
  - Seals and porpoises under-take their budgeted M2 by about 0.06, so it is about 0.45 /yr with that added.
  - Resume 184.13 measured +0.35, +0.25 and then -0.23 (the starvation crash).
- A fix must remove about 0.4 /yr *at today's density* and still let the stock recover from low density.
- What each candidate removes at B_ref = 0.69 t/km2:

| Fix | Removes at B_ref (/yr) | Share of 0.39 |
|---|---|---|
| (a) F 0.12 -> 0.17-0.20 | 0.05-0.08 | 13-21 % |
| (b) r_max 0.51 -> 0.35-0.45 | 0.06-0.16 | 15-41 % |
| (c) Schaefer/B-H closure with K from stationarity | the whole residual, by construction | 100 % |
| (d) cannibalism, 0.016 + 0.0875 B | 0.076 | 19 % |
| (e) do nothing | 0 | 0 % |

- Only (c) closes the gap by itself. (a), (b) and (d) are calibration choices that change how much (c) has to do.
  K must be re-derived after them.

## Ranking by literature support

### 1. (c) Density dependence as a carrying capacity: strongest support

**Why it ranks first**
- The library's r_max is the FishBase/CMSY prior, which is the r of a Schaefer model with K. The model kept r and dropped
  K (`density_dependence.md`).
- Compensatory stock-recruitment is the norm in NE Atlantic gadoids.
  - Density-dependent recruitment appears in 68-78 % of 80 stocks (Rindorf et al. 2022).
  - Beverton-Holt steepness is 0.66-0.79 (FishLife).
  - In the Skagerrak itself, juvenile density-dependent mortality has a = -0.25 to -0.50 in 8 of 11 cod populations
    (Fromentin et al. 2001).
- Ecosystem models close fish groups the same way: Atlantis with B-H recruitment, Ecosim with emergent B-H, CMSY and
  SPiCT with logistic production.
- The data agree. The pool was stationary in 2010-2019 at F 0.23-0.30. A Schaefer pool with r 0.51 then sits at
  B/K = 0.43-0.55, and predicts a modest, decelerating rise of +0.11-0.17 /yr after an F cut. It does not predict
  +35 %/yr compounding (`density_dependence.md`).

**Concrete form**
- Add a loss on every gadoid cell of
  `L = (r_max / 1460) * (B_region / K)` per tick,
  where B_region is the grid-mean gadoid density in t/km2.
- Remove biomass and reserve together, as M1 does.
- Leave L out of the derived g in `growth_budget.py`, so the budget keeps paying r_max + M1 + M2. Net growth at ideal
  food then becomes r_max(1 - B/K) - F.
- It keys on regional biomass because recruitment compensation acts at the stock and nursery scale, outside an offshore
  60 x 60 km box.
  - It is not a per-cell crowding cost, so it gives the policy no gradient. That is correct here, because no adult
    behaviour on the grid avoids a recruitment limit.

**K values, from stationarity at B_ref = 0.69 t/km2 (K = B_ref / (1 - F/r_max))**

| r_max | F | B_ref/K | K (t/km2) | Coefficient per tick per (t/km2) = r_max/(1460 K) |
|---|---|---|---|---|
| 0.511 | 0.12 | 0.765 | **0.90** | 3.9e-4 |
| 0.511 | 0.20 | 0.61 | 1.14 | 3.1e-4 |
| 0.45 | 0.17 | 0.62 | 1.11 | 2.8e-4 |
| 0.40 | 0.12 | 0.70 | 0.99 | 2.8e-4 |

- **Calibration basis.** The table cancels the ideal-food surplus of 0.39.
  - Adding the seal/porpoise M2 shortfall (0.45) gives K ≈ 0.78.
  - Cancelling the probe's realised net of +0.35 gives K ≈ 1.0.
  - So use K 0.8-1.0 at F 0.12, start at 0.90, and adjust it once against a retrained run (`density_dependence.md`).
- **B-H alternative** (option C2 in `density_dependence.md`): `L = (1/T) ln(1 + (alpha_hat - 1) B/K0)`, with h 0.73
  (alpha_hat 10.8), T 4.7 yr and K0 ≈ 1.3 t/km2. It matches the logistic at B_ref and compensates more strongly at low B.
  Prefer it if no-fishing scenarios will be run.

**Caveats**
- K is only as good as the stationarity assumption. That assumption holds for 2005-2020. Since 2021 the pool has been
  2x higher after the haddock pulse and is now drifting down by about 3 %/yr.
- K scales with the model's absolute biomass. The IBTS wing-spread density, about 2.7-3.3 t/km2, would make K 4x larger.
  This 4x model-versus-survey gap is still unresolved (`ices_gadoid_f.md`).
- The term is also a recruitment-limitation term, so it removes the pool's ability to grow faster than r(1 - B/K). That
  is intended.

**Confidence:** high in the mechanism, low-moderate in the value of K.

### 2. (b) Lower r_max: moderate support, partial fix

- 0.51 is a faithful low-density maximum. FishBase gives 0.48-0.52 (95 % CL about 0.33-0.77); stock CMSY fits give
  0.37-0.74; Myers 1997 gives cod 0.53-0.82.
- Productivity is lower today, though:
  - North Sea cod R/SSB is about 2.5x lower than before 1990;
  - haddock and hake R/SSB is 3-5x lower than before 2000-2010;
  - warming has reduced North Sea MSY by 35 % (Free et al. 2019; cod θ -0.44 /°C);
  - North Sea cod shows hysteresis (Sguotti et al. 2019).
- A **present-productivity r_max of 0.40 (range 0.35-0.45)** is defensible, but it is my estimate, not a published value.
- Depensation is not a reason to lower it.
- On its own this leaves +0.23 to +0.33 /yr at ideal food.
- `growth_budget.py` derives g from r_max, so this change lowers g at every density. Re-derive K afterwards.

**Confidence:** moderate that today's r is below 0.51, low on the exact number.

### 3. (a) Higher total F: weak-moderate support, partial fix

- As an all-gear, all-area rate on *today's* haddock-dominated mix, 0.12 is right.
  - Recreational fishing adds about 2-10 % of cod removals (ICES 2.2-6.0 %; Radford et al. 2018: 10 %).
  - Discards are already included in the catches.
  - Fishing outside the grid is already inside a stock F, so adding it would double count.
- The pooled F depends on the species mix:
  - 0.11 for today's survey mix;
  - about 0.17 for the Ecopath-1991 mix, which the library's diet and budget come from;
  - 0.20 for the 2005-2019 survey mix;
  - 0.23-0.30 under 2010-2019 fishing.
- **Recommended: F 0.17-0.20** if the pool is meant to be the long-term mix that eats herring, keeping 0.12 only for a
  "today's haddock pulse" scenario.
- Zero growth through F alone would need F ≈ 0.5, which no source supports for the offshore pool.
- The coastal Z of 1.0-1.75 is inshore cod (IMR 2025-42) and should not be used.

**Confidence:** high that 0.12 is not too low for today's mix; moderate on the 0.17-0.20 for a long-term mix.

### 4. (d) Restore cannibalism: real but small, with a learnability problem

- Intra-gadoid predation is about 0.23 /yr at North Sea densities of about 2 t/km2. That gives 0.016 + 0.0875 B, or
  0.076 /yr at B_ref (SMS).
- It falls on ages 0-2 and drops when alternative prey is plentiful.
- As a checked pair it cannot be learned by a group-biomass reward (resume 160). As a fixed per-cell loss it is fine.
- It closes about 20 % of the gap, and only if it is left out of the derived g. Counted in g, as a checked pair is, it
  is paid back by higher growth at B_ref.
- Use it as a complement to (c), not instead of it. If both are used and it is left out of g, re-derive K with
  r_eff = r_max - 0.076, which gives K ≈ 0.95 at F 0.12.

**Confidence:** high that the loss exists, moderate on its size; low that it suffices.

### 5. (e) The stock really is recovering and the model is right: least support

- The pool doubled in 2020-2023, but through two haddock year classes (8.7 and 7.8 billion age-0 against a median of
  about 1.5) and strong whiting year classes, spawned at low SSB.
  - That is recruitment, not compounding growth of standing biomass.
  - Pooled SSB has fallen about 3 %/yr since 2023 (SAG 2026). One sub-search reading with a 2025 haddock value gives
    +3 %/yr; either way it is near zero.
- The members that eat herring are declining:
  - cod is depleted (F 2x FMSY, SSB below Btrigger in all substocks);
  - saithe is below Blim;
  - hake is down 60 % since 2015.
  The members that are growing (haddock, whiting) eat little herring.
- No gadoid-herring overshoot cycle is documented for the North Sea or Skagerrak.
  - Herring is regulated by recruitment and fishing (Dickey-Collas et al. 2010; Lynam et al. 2017; HAWG 2026).
  - Gadoid M2 on herring has been 0.11-0.22 /yr for 49 years without a trend (SMS 2023).

**Confidence:** high that this is not supported.

### (f) Herring share of the gadoid diet: a calibration target, not on the original list

- Model gadoids take 40-52 % of their diet as herring, i.e. 1.8-2.45 BW/yr.
- SMS gadoids with the Skagerrak mix take about 5 %, 0.15-0.22 BW/yr. That is 8-16x less.
- Gadoid M2 on herring in the model is 0.5-1.2 /yr against 0.15 in SMS.
- Even with the gadoids held fixed (resume 184.12), the herring predation loss is about 2x SMS.
- (c) stops the compounding. (f) is what keeps herring from carrying the gadoid budget.
- Targets the model can be checked against:
  - herring at about 5 % of pooled gadoid ration (W1 weights; 6.5 % with box weights);
  - gadoid M2 on herring at about 0.15 /yr (all fish and mammal predators about 0.2);
  - library budget 0.30 /yr, from Ecopath 1991.

## Suggested combination (for an A/B, not a decision)

1. Keep r_max **0.51**, or 0.45 if the user wants present productivity.
2. Set F to **0.17** (long-term mix), or keep 0.12 and accept a haddock-pulse world.
3. Add the regional logistic closure with **K = B_ref/(1 - F/r_max)**: 0.90 t/km2 for r 0.51 with F 0.12, 1.11 for r 0.45
   with F 0.17. Leave it out of the derived g.
4. Optionally add per-cell cannibalism 0.016 + 0.0875 B as a real loss. Note the budget choice:
   - Counted in g, as `gadoids.md` item 4 proposes and as a checked pair would be, it is paid back by higher growth. It
     then closes nothing at B_ref and only adds a density slope.
   - Left out of g, it removes 0.076 /yr, and K must be re-derived with r_eff = r_max - 0.076 (K ≈ 0.95 at r 0.51,
     F 0.12).
5. Check after retraining:
   - pooled gadoid biomass stays within ±10 %/yr of 0.69 over 4 years;
   - herring at about 5-10 % of gadoid intake;
   - gadoid M2 on herring at 0.15-0.3 /yr;
   - no winter starvation crash.

Implementation note for the caller, which I did not act on:
- `lib/world/growth_budget.py` derives g from r_max + M1 + checked M2 at s*. Any r_max change moves g, and any
  closure term must deliberately stay out of that sum.
- A per-cell term (cannibalism) and a regional term (K) behave differently under learned movement. Only the per-cell
  term gives a gradient the policy can learn from.

## Correction (main session, after review): (f) compares the wrong prey set

- `pelagic_fish` is a pool, not herring alone. The library's r_max comment lists herring, sprat, mackerel, horse
  mackerel, sandeel and other planktivores. `gadoids` is likewise a pool of all gadoid species, so diet shares and
  rates are pool averages.
- The 4.7-6.5 % "herring share" from SMS above is the herring-only share. It is not a target for the pool.
- Computed from the Ecopath NS 1991 tables (`research_notes/Litteraturbaserad growth rate per FG/ecopath_ns1991`).
  Predators are the gadoid groups 13-21, 24 and 25: gadoid B 1.96 t/km2, ration 5.07 BW/yr.

| Prey set | Share of gadoid ration | Eaten / B_gad (BW/yr) | M2 on the prey set (/yr) |
|---|---|---|---|
| herring | 4.1 % | 0.21 | 0.16 |
| + sprat | 9.6 % | 0.49 | 0.30 |
| + mackerel, horse mackerel | 10.0 % | 0.51 | 0.18 |
| + sandeel | 25.8 % | 1.31 | 0.30 |
| + Norway pout, blue whiting | 39.4 % | 2.00 | 0.39 |

- In the model the pelagic share is 40-52 % without a carrying capacity and 73 % with K 0.9 (1.8-3.2 BW/yr).
  - The model is about 1-2.5x the Ecopath pool value, not 8-16x.
  - The library budget M2 0.2997 matches Ecopath's pool M2 of 0.30.
