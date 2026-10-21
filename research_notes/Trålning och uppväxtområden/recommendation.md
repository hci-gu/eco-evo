# Recommendation: should a trawl-free park raise the gadoid K? (Q5)

Proposed coupling (not built): **K_eff = K_today · (1 + share · ΔH)**, where
- H is a 0-1 benthic habitat index, averaged over the grid;
- share is the area's share of the stock's nursery habitat;
- K_today = 1.02 t/km2 (section 185).

Evidence is in the four companion notes:
- `mechanism_evidence.md` (Q1)
- `quantitative_links.md` (Q2)
- `nursery_areas.md` (Q3)
- `benthic_recovery_and_wind_farms.md` (Q4)

Everything in this note is **[Own] inference or computation** built on those notes. I changed no code or library
value.

## Verdict

- **The form is defensible. The mechanism at this site is not. The size is negligible.**
- **Expected effect for the real park:**
  - Park: 521 km2, 14.5 % of the grid, carrying 27 % of the grid's trawl effort.
  - If effort is removed: **+0.03 to +0.1 % of K, or +0.0003 to +0.001 t/km2.** The upper bound is about +0.3 %
    (+0.003 t/km2).
  - If the effort is displaced inside the grid: **about 0**.
  - This is two orders of magnitude below the uncertainty in K itself:
    - K is 0.8-1.4 t/km2 from the stationarity fits;
    - the model-versus-IBTS gap is still unresolved at 4x.
- **Recommendation:**
  1. Leave the coupling out of the baseline model.
  2. In the noll/projekt report, state that the habitat-mediated K effect of the park was assessed and is below 0.3 %
     of K, citing this note.
  3. If a visible sensitivity case is wanted, implement the depth-weighted variant below with the upper-bound
     parameters. Label it a sensitivity, not a prediction.
- **Do not use share = 1.**
  - Treating the grid as the whole nursery gives ΔK/K = ΔH_grid ≈ 7.4 % (4.5-11.7 %), i.e. +0.075 t/km2.
  - That overstates the effect 100-300x.
- **Confidence:**
  - High that the effect is below 1 % of K.
  - Low on its exact value, and even on its sign for the food pathway.

## Why each factor is small

Decompose the coupling into four factors:

ΔK/K = Σ_species w_s · s_s · ΔH_s · e_s

| Factor | Meaning | Supported value | Basis |
|---|---|---|---|
| w_s | species weight in the pool | haddock 0.54, whiting 0.27, saithe 0.12, cod 0.06, hake 0.01 | today's survey mix (task brief) |
| s_s | share of the stock's juvenile habitat inside the grid | haddock 0.5-1 % (0.2-2 %); whiting 0.1-0.5 %; cod < 0.1 %; saithe ≈ 0 | `nursery_areas.md` (inference; no published km2 exists) |
| ΔH_s | change in H over that species' juvenile habitat in the grid when the park is untrawled | haddock band 150-300 m: 0.17 (0.10-0.26); grid mean 0.074 (0.045-0.117) | `quantitative_links.md` table C |
| e_s | elasticity of recruitment/K to H | shelter pathway on mud: ≈ 0 (no evidence); food pathway: unknown sign, \|e\| ≤ 1; Atlantis-style linear scalar: 1 | `mechanism_evidence.md` |

**Haddock dominates.**
- ΔK/K ≈ 0.54 × 0.0075 × 0.17 × e ≈ **0.07 % × e**.
- Upper bound: 0.54 × 0.02 × 0.26 × 1 = **0.28 %**.
- Whiting adds at most about 0.02 %. Cod and saithe add about 0.

**The user's grid-mean form gives the same order.**
- s_pool 0.3-0.6 % (upper 1.5 %) × ΔH_grid 0.074 (0.045-0.117) = **0.02-0.04 %**, with an upper bound of 0.18 %.

In absolute terms, at K = 1.02 t/km2:

| Case | ΔK/K | ΔK (t/km2) | Equilibrium B at F 0.12 (Schaefer) |
|---|---|---|---|
| Central, depth-weighted, e = 1 | 0.07 % | 0.0007 | 0.6890 to 0.6895 |
| Upper bound | 0.3 % | 0.003 | 0.689 to 0.691 |
| Effort displaced in the grid | < 0.01 % | < 0.0001 | unchanged |
| share = 1 (not supported) | 7.4 % | 0.075 | 0.689 to 0.740 |

The "Equilibrium B" column assumes B* = K(1 - F/r) with r 0.37 and F 0.12, so B* moves by the same fraction as K.

**Parks of 10-20 % of the grid.** The effect scales with the park's share of *trawl effort* in the juvenile depth band,
not with its area.
- Mareld is 10 % of the grid area but carries 22 % of the effort, because it sits on the 200-300 m Pandalus ground.
- Poseidon Nord is 4 % of the area and carries 5 % of the effort, mostly at 100-200 m.
- A 10-20 % park on the same heavily trawled ground gives about 0.02-0.4 % of K, taking the central-to-upper range
  scaled by effort share.
- The same area on the lightly trawled shelf (< 150 m, 1.4-1.6 MW-h/km2/yr) gives less: H there is already about 0.85
  at d 0.10, r 0.82.

## Why the mechanism does not transfer to the site

1. **Shelter.**
   - The only demonstrated habitat-to-juvenile-survival link is for 0-group cod on shallow cobble, sponge, eelgrass
     and maerl (Lindholm 1999; Tupper & Boutilier 1995; Gotceitas et al. 1997; Pihl et al. 2006).
   - The grid has no coast, 25 km2 shallower than 50 m, and mud below 150 m.
   - Cod, whiting and saithe settle elsewhere. Juvenile haddock on mud show no sediment or structure preference.
2. **Food.**
   - For benthivores the trawling response is hump-shaped and can be positive at moderate SAR (Hiddink et al. 2008,
     2016; van Denderen et al. 2013).
   - The gadoid evidence is null (whiting, Irish Sea mud) or confined to haddock diet on gravel (Smith et al. 2013).
   - The food pathway is also **already explicit in the model**: `gadoids_preys_on_benthic_community` is checked, and
     `benthic_community` is depleted per cell by the same trawl map. A K(H) on top would count it twice.
3. **Production.**
   - No wind farm or trawl closure has been shown to raise gadoid recruitment.
   - The closest analogue to Mareld, the Kattegat no-take zone (426 km2, Nephrops mud), doubled Nephrops but did not
     raise cod (Sköld et al. 2022).
   - Floating foundations add almost no reef. The one floating-farm study saw no fish effect (Hestetun et al. 2024).
4. **Scale.**
   - These stocks recruit over at least 10^5 km2 and are driven by larval drift, temperature and spawning stock.
   - The "lower historical K" behind section 185 (decades of trawling plus warming) is a stock-wide property.
   - A 521 km2 park cannot restore it. Only a stock-wide trawl scenario would make the coupling matter, and even then
     the trawling share of the K loss is unquantified (Collie et al. 2017: "Current research attempts to scale up").

## If it is implemented anyway (sensitivity case)

Use a per-cell, depth-weighted form normalised to noll, not a grid-mean H:

    K_eff(t) = K_today · [1 + e · s_pool · ( mean_{i in N} H_i(t) - mean_{i in N} H_i,noll )]

| Item | Value | Basis |
|---|---|---|
| N | cells with 150 ≤ depth < 300 m (1,378 km2; 398 km2 of park) | juvenile haddock band (Albert 1991) |
| H_i(t) | `benthic_community` B/K in cell i, smoothed over about 1 yr | the model's own RBS_tot; reading it avoids a second trawl bookkeeping |
| H_i,noll | the same quantity in the noll run at equilibrium | makes K_eff = K_today in noll by construction |
| s_pool | 0.004 (central: 0.54 x 0.0075) | sensitivity 0.011 (0.54 x 0.02) |
| e | 1 for the sensitivity, 0 for the baseline | no evidence for an intermediate value |

Notes:
- **Substrate weighting.** The grid is mud throughout, so per-substrate weighting reduces to the depth weighting above.
  If a sediment map later shows gravel or Bratten rock, those cells would be the only ones where the shelter pathway has
  literature support. They are mostly untrawled already, since Bratten effort is about 0.
- **Index choice.**
  - If the coupling is meant to represent *structure* (shelter), RBS_sen is the conceptually right index: long-lived
    fauna, r 0.2-0.5 /yr, recovery over 10-20+ yr.
  - The single-pool `benthic_community` cannot provide it. It would need a separate slow pool, and that is not worth
    building for a < 0.3 % effect.
- **Lag.** H recovers in about 4 yr (r 0.82) for biomass and 15+ yr for sea pens. K should follow H, not the trawl
  map, so the effect ramps over the park's lifetime.
- **Expected output.** Pooled gadoid biomass in projekt rises by at most about 0.003 t/km2 from this term. That is
  invisible against interannual noise.

## The larger noll/projekt lever is fishing mortality, not habitat

This note did not research this channel beyond the arithmetic below. I record it because it dominates.
- In the model, bottom-trawl F on gadoids is per cell and proportional to the effort map. The park holds 27 % of the
  grid's effort.
- If projekt removes that effort, grid-mean gadoid F falls from 0.12 to about 0.088 /yr, assuming gadoids are spread
  in proportion to area.
- The Schaefer equilibrium then moves from 0.689 to 0.778 t/km2. That is **+13 %, about 50-200x the habitat term**.
- The same scale question arises as for the habitat share:
  - the model treats the grid as the region that the K closure regulates;
  - the real stock-wide F barely changes for a 521 km2 closure, and effort is likely displaced (Murawski et al. 2005;
    Sköld et al. 2022: the zone was "too small to protect cod").
- The noll/projekt comparison should state whether the park's effort is removed from or displaced within the grid. It
  should also report the gadoid response as an upper bound for a mobile, stock-scale population.

## Key sources (full citations in the companion notes)

- Collie J, Hiddink JG, van Kooten T, Rijnsdorp AD, Kaiser MJ, Jennings S, Hilborn R. 2017. Indirect effects of
  bottom fishing on the productivity of marine fish. *Fish Fish.* 18:619-637.
  doi [10.1111/faf.12193](https://doi.org/10.1111/faf.12193).
- Lindholm JB, Auster PJ, Kaufman LS. 1999. *Mar. Ecol. Prog. Ser.* 180:247-255.
  doi [10.3354/meps180247](https://doi.org/10.3354/meps180247).
- Lindholm JB, Auster PJ, Ruth M, Kaufman L. 2001. *Conserv. Biol.* 15:424-437.
  doi [10.1046/j.1523-1739.2001.015002424.x](https://doi.org/10.1046/j.1523-1739.2001.015002424.x).
- Pitcher CR, et al. 2022. *PNAS* 119:e2109449119. doi [10.1073/pnas.2109449119](https://doi.org/10.1073/pnas.2109449119).
- Hiddink JG, et al. 2017. *PNAS* 114:8301-8306. doi [10.1073/pnas.1618858114](https://doi.org/10.1073/pnas.1618858114).
- Foley NS, et al. 2012. *Int. J. Ecol.* 2012:861635. doi [10.1155/2012/861635](https://doi.org/10.1155/2012/861635).
- Harvey CJ. 2014. *Can. J. Fish. Aquat. Sci.* 71:1020-1029. doi [10.1139/cjfas-2013-0594](https://doi.org/10.1139/cjfas-2013-0594).
- Audzijonyte A, et al. 2025. *Atlantis User's Guide Part I*. CSIRO (sections 10.2.4 and 10.9.3).
- Albert OT. 1991. ICES CM 1991/G:45; Albert OT. 1994. *ICES J. Mar. Sci.* 51:31-44.
  doi [10.1006/jmsc.1994.1004](https://doi.org/10.1006/jmsc.1994.1004).
- Sköld M, et al. 2022. *ICES J. Mar. Sci.* 79:2231-2246. doi [10.1093/icesjms/fsac152](https://doi.org/10.1093/icesjms/fsac152).
- Hestetun JT, Mayers K, Mugu S, Dahlgren TG. 2024. NORCE report 4-2024.
- Mareld EIA (MKB), Bilaga C, 17 April 2023 (`MKB Mareld.pdf`).
