# Biomass-weighted present-day r for the gadoid pool

Computed in the main session from the numbers in `r_max_and_recruitment.md`. Nothing here is a published value.

## Method
- **r_max per species:** the FishBase CMSY prior (cod 0.51, haddock 0.50, whiting 0.49, saithe 0.52, hake 0.48).
- **Productivity decline d per species:** median recruits per unit SSB, early period over recent period, from ICES SAG:

  | Species | Early | Recent | d (range) |
  |---|---|---|---|
  | cod | 6.87 (1963-89) | 2.58-2.68 | 2.6 |
  | haddock | 30-50 | 6-11 | 4.7 (2.7-8.3) |
  | hake | 11-17 | 3.4-3.8 | 3.9 (2.9-5.0) |
  | saithe | 1.0 | 0.63 | 1.6 |
  | whiting | | | 1 (no decline) |

- **Euler-Lotka:** r ≈ ln(a)/T. Dividing the low-density reproductive rate a by d gives r_now = r_max - ln(d)/T.
  - T is taken from the CMSY rule r ≈ 3/t_gen (Froese et al. 2017), so r_now = r_max (1 - ln(d)/3).
  - With Myers et al. 1997 for cod (T ≈ 3.9 yr instead of 5.9), cod r_now is 0.26 instead of 0.35. T is the largest
    single uncertainty and lowers every value by up to ~0.05-0.08.
- **No separate warming factor.** Free et al. 2019 act mainly through recruitment, so applying both would count it twice.

## Per species

| Species | r_max | d | r_now |
|---|---|---|---|
| cod | 0.51 | 2.6 | 0.35 (0.26 with T 3.9) |
| haddock | 0.50 | 4.7 (2.7-8.3) | 0.24 (0.15-0.33) |
| whiting | 0.49 | 1.0 | 0.49 |
| saithe | 0.52 | 1.6 | 0.44 |
| hake | 0.48 | 3.9 (2.9-5.0) | 0.26 (0.22-0.31) |

## Pool, by weighting

| Weights | cod / haddock / whiting / saithe / hake | r_max | r_now (range from d) |
|---|---|---|---|
| W1, survey today | 0.058 / 0.543 / 0.267 / 0.121 / 0.011 | 0.50 | **0.34** (0.29-0.39) |
| W3, box rectangles | 0.051 / 0.356 / 0.387 / 0.192 / 0.012 | 0.50 | **0.38** (0.35-0.41) |
| Ecopath NS 1991 mix | 0.140 / 0.226 / 0.334 / 0.292 / 0.008 | 0.50 | **0.40** (0.38-0.42) |

## Reading
- The pool's present-day r is about **0.34-0.40**, and about 0.27-0.35 with Myers' shorter generation time.
- The 0.20 used in resume 184.17 (0.51/2.5) scaled r linearly with the R/SSB decline. Under Euler-Lotka, a
  recruitment decline lowers r by ln(d)/T, not by a factor d, so 0.20 was too low.
- At r 0.34-0.40 and F 0.12, the pool still has about +0.2/yr at ideal food. The lower r explains half of the
  0.39 surplus. The rest must come from density dependence at today's effective capacity.
  - That capacity can be lowered by habitat loss (trawled nurseries) and warming, while the historical ceiling stays
    much higher.
