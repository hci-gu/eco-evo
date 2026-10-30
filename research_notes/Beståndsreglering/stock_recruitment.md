# WP1 - Stock-recruitment of the member stocks and compensation at today's SSB

Work package 1 of `plan.md`. Question: how strongly does recruitment per adult tonne fall with adult (spawning)
biomass at stock scale, at today's state, for the members of the four fish groups, so that a density-dependent
recruitment weight (e.g. rho / (1 + A_bar/K_R), Beverton-Holt in the adult biomass) can replace the regional
logistic loss r_max B/K? Nothing here is fitted to model output, and no engine or library value was changed.

Grades on numbers: **D** computed from data (ICES SAG series, my scripts), **F** stated in a source, **A** analogy,
**J** judgement. "e" throughout is the elasticity d ln(R/S) / d ln S at today's SSB: 0 = no compensation
(recruits proportional to spawners), -1 = full compensation (recruitment independent of SSB), below -1 =
over-compensation (Ricker-type). For a Beverton-Holt term rho0/(1 + S/b), e = -x/(1+x) with x = S/b, so
x = -e/(1+e) and rho(S_today)/rho0 = 1 + e.

## Takeaway

1. **Inside today's regime (year classes 2000-2024; NSAS herring 2002-2024) recruitment of most member stocks does
   not follow SSB over the SSB range they have seen** [D]. Constant recruitment beats recruitment proportional to SSB
   by 3-15 AICc units for haddock (13), the Viking and Northwestern cod substocks and their sum (11-12), witch (9),
   Norway pout (14), sprat (9), blue whiting (4.6), NSAS herring (3.2), western horse mackerel (6.3), hake (29) and
   whiting and plaice (hockey stick above its breakpoint, 14 and 9). The model-averaged elasticity at today's SSB is
   e = -0.8 to -1.1 for these stocks, i.e. near-full compensation. But today's SSB sits inside a narrow range
   (S_min/S_today 0.15-0.8, S_max/S_today 1.0-2.4), so this says only that the S-R curve is *flat over the observed
   range*; the slope at the origin (BH a) is not identified for any of them (BH b drops below the lowest observed
   SSB; a runs to implausible values).
2. **Exceptions, where the data show no compensation or cannot tell** [D]: WBSS herring (SSB at 0.5 Blim, R
   proportional to S, e = 0.0, prop beats const by 33 AICc - the stock is on the ascending limb), saithe (today's SSB
   is the lowest since 2000; R fell with S, prop best, e -0.06 stationary but -0.73 if a productivity trend is
   allowed), mackerel (stationary e -0.88, but a 0.41x-per-decade fall of R/S explains the data equally well, e
   -0.16), anglerfish (prop best, e -0.34 to -0.46), spurdog (recruitment is a structural pup-production output,
   not evidence; the assessment's own pup-survival term gives e -0.25 to -0.87, A).
3. **Productivity shifts dominate the long series** [D]. Recruits per spawner fell after ~2000 by a factor 0.24
   (haddock), 0.27 (old North Sea cod), 0.30-0.46 (cod substocks), 0.26 (western horse mackerel), 0.57 (saithe),
   0.63-0.66 (both herring stocks); a model with a step in log a at 2000 (2002 for NSAS herring) beats every
   stationary model by 13-29 AICc for the gadoids. Fitted stationary over the full series, North Sea cod reads the
   decline as "no compensation" (proportional model best, e = 0.0) - a confounding artefact; within regimes the same
   data give e = -0.9 to -1.0. Plaice moved the other way (x 1.6). Literature agrees (Payne et al. 2009; Brunel &
   Boucher 2007; Szuwalski et al. 2015: 85 % of stocks without an SSB signal show shifts in mean recruitment).
4. **Group compensation at today's state** (Table 3; post-2000, all models averaged, members clipped to [-1, 0]):
   planktivorous_fish e = **-0.74** [90 %: -0.96, -0.51]; benthivorous_fish **-0.97** [-0.99, -0.84];
   predatory_fish **-0.65** [-0.83, -0.22] (saithe-driven, least certain); whiting **-0.98** [-1.00, -0.07].
   Read as a Beverton-Holt term, x = A_today/K_R = -e/(1+e) = 2.8 (planktivores), ~40 (benthivores, whiting), 1.9
   (predators); this transform explodes near e = -1 and the planktivore value is an average of WBSS (x ~ 0) and
   members with x 5 to inf.
5. **The data-anchored x and the life-history x disagree; this is unresolved and is for the synthesis** [D vs A/J].
   - Data floor: a BH term can make (R/S at S_today)/(R/S at the lowest SSB since 2000) no smaller than
     r = S_min/S_today; solving (1 + r x)/(1 + x) = observed ratio gives x_data (Table 2): haddock **inf** [17.8, inf]
     (ratio 0.15 at r 0.15 - recruitment flat over a 7x SSB rise), sprat, witch, cod substock sum and Viking cod inf
     (lower 90 % bounds 1.3-2.1), Northwestern cod 32, blue whiting 14, western horse mackerel 14, Norway pout 7,
     NSAS herring 4.6 [0.4, inf], hake 4.0, whiting 2.9 [0.1, inf], plaice 1.2 [0.2, 3.7]; WBSS herring and mackerel
     0.2; saithe not computable (today below all post-2000 SSB).
   - Life history: at a fished equilibrium 1 + x = alpha_hat SPR_F/SPR_0; with alpha_hat 6-15 (Myers et al. 1997;
     FishLife steepness 0.6-0.8) and SPR_F/SPR_0 0.3-0.5, x = 1-6.5 (e -0.5 to -0.87). It assumes stationarity and
     equilibrium, which Table 1 shows are violated, so it cannot override the data floor.
   - Consequence: with x = 4 the BH term would give haddock ~2x more recruitment at today's SSB, relative to its 2000s
     low-SSB years, than observed. For benthivores and whiting the data support "flat above a breakpoint" (hockey
     stick, or BH with K_R at or below the lowest recent SSB, x >~ 5-20); the life-history x 1-6 is a lower-bound
     alternative. For planktivores x is member-specific (WBSS ~0, herring NSAS ~5, pout/blue whiting/sprat 7-inf);
     for predators it is set by saithe and is not identified (J).
6. **Density dependence acts mainly at recruitment** [F]: density-dependent recruitment in 68 % of 80 NE Atlantic
   stocks and recruitment effect > growth effect in 50 of 60 (Rindorf et al. 2022); 97 % of populations with both
   significant (Zimmermann et al. 2018). Late (adult) growth density dependence is also common (54 %), clear in North
   Sea plaice, whiting, haddock, herring, sprat and mackerel; early growth is mostly density-independent;
   density-dependent maturation is untested for these stocks. North Sea cod is the odd one: no significant density
   dependence anywhere in Rindorf's analysis.
7. **Confidence.** High that compensation at today's SSB is strong (e < -0.6) for haddock, whiting, cod (within
   regimes; Rindorf et al. 2022 find no density dependence for North Sea cod over the whole series, as does my
   stationary full-series fit - the disagreement is the productivity-shift confound), NSAS herring, sprat, Norway pout,
   hake, plaice. Moderate for the group values of planktivores and benthivores. Low for the
   predatory_fish group (saithe and mackerel are ambiguous) and for any absolute a or K_R. Recruitment compensation
   acts on stock SSB (North Sea or wider), not on the box's biomass; mapping K_R to the box's A_bar is WP3's share
   problem and is not solved here.

## 1. Per-stock results

Data: ICES SAG, latest published advice assessments (assessment years 2024-2026; the old cod series 2022), downloaded 2026-10-29 by
`scripts/sr_fetch.py`. R at age a in year y + a is paired with SSB in year y. Recruitment years from the terminal
year on are dropped. R units: thousands at the recruitment age (Kattegat cod: relative index). "S_today" = mean SSB of
the last three years up to the terminal year (2023-2025 for most; 2019-2021 for the old cod assessment).
Columns: dAICc prop / const = AICc of the proportional (no compensation) / constant (full compensation) model minus
the best stationary model; e = AICc-weighted elasticity at S_today with a moving-block bootstrap 90 % interval
(1000 replicates); "stationary" = prop, const, BH, Ricker, hockey stick; "all" = those plus linear-trend models
(both periods) and step-shift models (full series). The bootstrap interval can exclude the point value when the
weights jump between models.

Table 1. Full series.

| Stock | Group | R age | Year classes (n) | Best stationary | Best overall | dAICc prop | dAICc const | Shift: a_post/a_pre (dAICc) | e stationary [90 %] | e all [90 %] | sigma, AR1 (BH) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| her.27.3a47d | plankt. | 0 | 1947-2024 (78) | hs | ricker_shift | 83.1 | 67.4 | 0.66 (2.4) | -1.00 [-1.05, -0.97] | -0.89 [-1.12, -0.73] | 0.46, 0.52 |
| her.27.20-24 | plankt. | 0 | 1991-2024 (34) | hs | const_trend | 0.3 | 34.1 | 0.63 (4.2) | -0.01 [-0.05, 0.22] | -0.90 [-1.00, -0.06] | 0.38, 0.43 |
| spr.27.3a4 | plankt. | 0 | 1980-2024 (45) | ricker | bh_trend | 4.8 | 18.8 | 1.44 (2.5) | -0.53 [-0.92, -0.07] | -0.70 [-1.00, -0.53] | 0.50, 0.39 |
| nop.27.3a4 | plankt. | 0 | 1984-2024 (41) | const | const | 21.3 | 0.0 | 0.79 (-1.1) | -0.93 [-0.99, -0.39] | -0.93 [-0.99, -0.57] | 0.64, 0.07 |
| whb.27.1-91214 | plankt. | 1 | 1981-2023 (43) | hs | const_trend | 3.9 | 4.3 | 1.44 (-1.5) | -0.85 [-1.31, -0.40] | -0.94 [-1.97, -0.39] | 0.74, 0.64 |
| had.27.46a20 | benth. | 0 | 1972-2024 (53) | const | const_shift | 13.4 | 0.0 | 0.24 (29.3) | -1.27 [-3.22, -0.57] | -1.03 [-1.52, -1.00] | 1.08, 0.58 |
| cod.27.47d20 | benth. (old whole-stock SAM, 2022) | 1 | 1963-2019 (57) | prop | const_shift | 0.0 | 23.6 | 0.27 (25.0) | -0.00 [-0.06, 0.08] | -0.92 [-1.00, -0.41] | 0.70, 0.37 |
| cod.27.46a7d20_sum | benth. (sum of 3 substocks) | 1 | 1983-2023 (41) | prop | const_trend | 0.0 | 1.2 | 0.40 (18.8) | -0.29 [-1.04, 0.02] | -1.01 [-1.75, -0.74] | 0.69, 0.21 |
| cod.27.46a7d20_viking | benth. (used) | 1 | 1983-2023 (41) | const | const_trend | 6.9 | 0.0 | 0.46 (14.5) | -1.03 [-1.97, -0.49] | -1.11 [-1.83, -0.97] | 0.67, 0.14 |
| cod.27.46a7d20_southern | - | 1 | 1983-2023 (41) | prop | const_trend | 0.0 | 16.4 | 0.30 (15.6) | 0.01 [-0.29, 0.09] | -1.00 [-1.00, -0.89] | 0.79, 0.47 |
| cod.27.46a7d20_northwestern | - | 1 | 1983-2023 (41) | hs | const_shift | 4.3 | 0.5 | 0.42 (13.0) | -0.91 [-1.18, -0.08] | -0.92 [-1.12, -0.36] | 0.74, 0.14 |
| cod.27.21 | - (Kattegat, relative SSB) | 1 | 1997-2023 (27) | const | const_trend | 13.4 | 0.0 | 0.33 (3.0) | -0.57 [-0.81, -0.16] | -0.99 [-1.00, -0.64] | 0.87, 0.31 |
| ple.27.420 | benth. | 1 | 1957-2023 (67) | bh | bh_trend | 8.1 | 2.2 | 1.62 (14.8) | -0.80 [-1.04, -0.44] | -0.87 [-1.05, -0.62] | 0.47, 0.49 |
| ple.27.21-32 | - (Kattegat-Baltic) | 1 | 2002-2023 (22) | ricker | ricker | 13.0 | 36.8 | - | 2.42 [-1.23, 3.30] | 1.07 [-1.00, 2.29] | 0.68, 0.81 |
| wit.27.3a47d | benth. | 1 | 1950-2023 (74) | bh | bh_trend | 6.6 | 14.2 | 0.96 (-2.1) | -0.51 [-0.94, 0.00] | -0.56 [-0.97, -0.04] | 0.42, 0.84 |
| pok.27.3a46 | pred. | 3 | 1967-2021 (55) | bh | const_trend | 7.3 | 2.1 | 0.57 (19.9) | -0.37 [-0.79, -0.03] | -1.00 [-1.00, -0.75] | 0.48, 0.58 |
| mac.27.nea | pred. | 2 | 1998-2022 (25) | const | prop_trend | 6.1 | 0.0 | - | -0.59 [-0.98, -0.08] | -0.27 [-0.99, -0.02] | 0.69, 0.59 |
| hom.27.2a3a4a5b6a7a-ce-k8 | pred. (used, SS3) | 0 | 1982-2024 (36) | prop | const_trend | 0.0 | 10.8 | 0.26 (4.9) | -0.01 [-0.94, 0.01] | -0.99 [-1.00, -0.87] | 0.93, 0.37 |
| hom.27.4bc7d | - | 1 | 2002-2021 (20) | prop | const_trend | 0.0 | 10.1 | - | 0.01 [-0.69, 0.05] | -0.97 [-0.98, -0.50] | 0.50, 0.30 |
| hke.27.3a46-8abd | pred. (SS3) | 0 | 1978-2024 (47) | hs | hs | 55.6 | 2.4 | 1.10 (-2.9) | -0.98 [-1.11, -0.83] | -0.97 [-1.19, -0.73] | 0.42, 0.33 |
| anf.27.3a46 | pred. (SS3) | 0 | 1981-2024 (44) | ricker | ricker_shift | 73.5 | 5.3 | 1.38 (0.4) | -0.60 [-0.98, -0.53] | -0.63 [-0.96, -0.44] | 0.41, 0.30 |
| dgs.27.nea | pred. (structural) | 0 | 1975-2024 (48) | prop | prop_shift | 0.0 | 50.8 | 1.50 (41.2) | -0.19 [-0.88, 0.16] | 0.06 [-0.24, 0.46] | 0.26, 0.79 |
| whg.27.47d | whiting | 0 | 1978-2024 (47) | hs | bh_shift | 19.9 | 13.9 | 0.77 (2.1) | -0.93 [-1.09, -0.68] | -0.92 [-1.39, -0.84] | 0.30, 0.57 |

"Shift" = a BH with separate log a before and after year class 2000 (2002 for NSAS herring) and a common b;
a_post/a_pre < 1 is a productivity loss; the dAICc in brackets is best stationary minus best shift model (positive =
the shift is supported). Kattegat-Baltic plaice (ple.27.21-32) has a Ricker with negative 1/b (R/S rising with S
during a strong increase; today's SSB above all fitted SSB): not usable, and not a member stock (Skagerrak plaice is
in ple.27.420). The 2024 ple.27.21-23 assessment gives the same picture (prop best) and was superseded.

Table 2. Year classes 2000 on (NSAS herring 2002 on), and compensation at today's SSB.

| Stock | S_today (kt) | S_today/Blim | S_min/S_t - S_max/S_t | Best stationary | Best overall | dAICc prop | dAICc const | e stationary [90 %] | e all [90 %] | Sbp/S_today (HS) | (R/S)_today/(R/S)_Smin, all [90 %] | x_data [90 %] | Grade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| her.27.3a47d | 1710 | 2.06 | 0.82-1.44 | const | const | 3.2 | 0.0 | -0.81 [-1.03, -0.38] | -0.83 [-1.02, -0.42] | 0.82 (min bound) | 0.85 [0.82, 0.95] | 4.6 [0.4, inf] | D |
| her.27.20-24 | 99 | 0.50 | 0.92-3.66 | prop | prop | 0.0 | 33.0 | 0.01 [-0.03, 0.12] | -0.16 [-0.98, 0.01] | 3.66 (max bound) | 0.99 [0.94, 1.52] | 0.2 [0, 4.3] | D |
| spr.27.3a4 | 147 | 1.91 | 0.51-2.24 | const | const_trend | 9.2 | 0.0 | -0.93 [-1.00, -0.41] | -0.99 [-1.00, -0.90] | 0.51 (min bound) | 0.51 [0.51, 0.70] | inf [1.6, inf] | D |
| nop.27.3a4 | 33 | 1.15 | 0.79-5.13 | const | const | 13.6 | 0.0 | -0.95 [-1.44, -0.28] | -0.96 [-1.44, -0.52] | 0.90 | 0.81 [0.77, 0.89] | 7.1 [1.1, inf] | D |
| whb.27.1-91214 | 5612 | 3.74 | 0.47-1.22 | const | const | 4.6 | 0.0 | -1.13 [-3.04, -0.85] | -1.10 [-2.65, -0.80] | 0.67 | 0.51 [0.32, 0.67] | 13.8 [1.7, inf] | D |
| had.27.46a20 | 771 | 5.57 | 0.15-1.03 | const | const | 13.1 | 0.0 | -1.07 [-10.25, -1.03] | -1.05 [-9.44, -0.99] | 0.15 (min bound) | 0.15 [0.07, 0.20] | inf [17.8, inf] | D |
| cod.27.47d20 | 49 | 0.70 | 0.65-1.72 | const | const | 15.2 | 0.0 | -1.05 [-1.24, -0.83] | -1.03 [-1.13, -0.43] | 0.65 (min bound) | 0.65 [0.64, 1.12] | inf [0, inf] | D |
| cod.27.46a7d20_sum | 60 | 1.16 | 0.64-1.55 | const | const | 11.9 | 0.0 | -1.07 [-1.58, -0.96] | -1.04 [-1.48, -0.80] | 0.64 (min bound) | 0.63 [0.57, 0.75] | inf [2.1, inf] | D |
| cod.27.46a7d20_viking | 19 | 1.35 | 0.62-1.75 | ricker | const_trend | 11.1 | 0.8 | -1.30 [-1.83, -0.83] | -1.18 [-1.80, -0.67] | 0.64 | 0.60 [0.51, 0.79] | inf [1.3, inf] | D |
| cod.27.46a7d20_southern | 9 | 0.64 | 0.42-2.91 | hs | const_trend | 3.3 | 2.8 | -0.30 [-0.99, -0.07] | -1.00 [-1.00, -0.87] | 1.62 | 0.42 [0.42, 1.65] | inf [0, inf] | D |
| cod.27.46a7d20_northwestern | 32 | 1.36 | 0.30-1.56 | const | const | 12.2 | 0.0 | -1.01 [-1.19, -0.76] | -1.00 [-1.15, -0.82] | 0.30 (min bound) | 0.32 [0.30, 0.64] | 31.7 [1.1, inf] | D |
| cod.27.21 | 0.127 (rel.) | - | 0.73-16.21 | const | const_trend | 16.8 | 0.0 | -0.70 [-0.92, -0.29] | -0.95 [-1.00, -0.49] | 1.48 | 0.74 [0.73, 1.55] | 20.6 [0, 73.2] | D |
| ple.27.420 | 1003 | 2.56 | 0.44-1.31 | hs | hs | 9.1 | 7.8 | -0.93 [-1.26, -0.68] | -0.94 [-1.19, -0.30] | 0.82 | 0.69 [0.56, 0.90] | 1.2 [0.2, 3.7] | D |
| ple.27.21-32 | 82 | 7.41 | 0.10-0.90 | ricker | ricker | 13.0 | 36.8 | 2.42 [-1.24, 3.20] | 1.07 [-1.00, 1.73] | 0.90 (max bound) | 4.57 [0.10, 8.70] | 0 [0, inf] | D |
| wit.27.3a47d | 4 | 1.24 | 0.49-1.02 | const | const | 9.1 | 0.0 | -1.08 [-2.28, -0.80] | -1.07 [-2.08, -0.70] | 0.51 | 0.49 [0.34, 0.71] | inf [1.3, inf] | D |
| pok.27.3a46 | 118 | 0.86 | 1.28-2.38 | prop | const_trend | 0.0 | 4.3 | -0.06 [-0.76, 0.10] | -0.73 [-0.94, -0.03] | 2.20 | 1.21 [1.01, 1.41] | n/a [n/a, n/a] | D |
| mac.27.nea | 3642 | 1.19 | 0.63-3.68 | const | prop_trend | 8.0 | 0.0 | -0.88 [-0.98, -0.12] | -0.16 [-0.98, -0.02] | 0.99 | 0.95 [0.72, 1.50] | 0.2 [0, 3.2] | D |
| hom.27.2a3a4a5b6a7a-ce-k8 | 810 | 1.43 | 0.83-3.57 | const | const_trend | 6.3 | 0.0 | -0.89 [-1.32, -0.45] | -0.94 [-1.18, -0.72] | 0.83 (min bound) | 0.84 [0.81, 0.98] | 14.0 [0.1, inf] | D (SS3 caveat) |
| hom.27.4bc7d | 73 | 0.40 | 1.34-3.99 | prop | const_trend | 0.0 | 10.1 | 0.01 [-0.69, 0.04] | -0.97 [-0.98, -0.48] | 3.23 | 1.33 [1.24, 1.61] | n/a [n/a, n/a] | D |
| hke.27.3a46-8abd | 124 | 2.02 | 0.31-2.36 | const | bh_trend | 29.1 | 0.0 | -1.00 [-1.06, -0.00] | -0.81 [-1.00, -0.15] | 0.36 | 0.45 [0.36, 0.99] | 4.0 [0.0, 14.5] | D (SS3 caveat) |
| anf.27.3a46 | 44 | 1.72 | 0.53-1.01 | prop | prop | 0.0 | 2.9 | -0.34 [-1.07, 0.67] | -0.46 [-0.99, 0.05] | 0.94 | 0.83 [0.56, 1.06] | 0.6 [0, 14.9] | D (SS3 caveat) |
| dgs.27.nea | 621 | 2.54 | 0.31-1.00 | prop | const_trend | 0.0 | 40.9 | -0.29 [-0.48, 2.61] | -0.80 [-0.96, 0.53] | 0.89 | 0.47 [0.34, 2.25] | 3.3 [0, 21.4] | J (structural R) |
| whg.27.47d | 370 | 2.57 | 0.31-1.07 | hs | hs | 13.8 | 5.2 | -0.99 [-1.16, -0.39] | -0.98 [-1.06, -0.07] | 0.53 | 0.48 [0.31, 0.95] | 2.9 [0.1, inf] | D |

Notes to Table 2:
- x_data = the Beverton-Holt S_today/b that reproduces the previous column (ratio q) given r = S_min/S_today:
  x = (1 - q)/(q - r); "inf" when q <= r (recruitment flat or falling over the range); interval from the ratio's
  interval; "n/a" when today's SSB is below the period's lowest.
- "(min/max bound)": the hockey-stick breakpoint sits at the lowest (highest) SSB of the period, i.e. the
  hockey stick degenerates to constant (proportional) recruitment.
- The last column is the data-anchored compensation: recruits per spawner at today's SSB relative to the lowest SSB
  of 2000-2024. A value > 1 (saithe, North Sea horse mackerel) means today's SSB is *below* every SSB in the period,
  so the column measures nothing there.
- With today's SSB inside a 2-3x range, a BH b below the lowest SSB and a "const" winner, the fitted BH a (recruits
  per tonne at the origin) is not identified: e.g. haddock, cod and witch post-2000 BH a = 6 000-64 000 thousand per
  tonne against an observed R/S of 3-30. Only lower bounds follow: a >= max observed R/S (`tables/sr_fits.csv`,
  columns a and b).

Origin slopes where the full series does identify a curve [D] (BH a, thousand recruits per tonne SSB; b = SSB at
half of Rmax): NSAS herring a 63, b 0.33 S_today (563 kt, against Blim 829 kt); sprat a 1 100, b 1.3 S_today;
whiting a 126, b 0.48 S_today; North Sea plaice a 11.3, b 0.43 S_today; witch a 20.5, b 0.90 S_today; saithe a 1.75
(age 3), b 1.44 S_today; hake a 67, b 0.10 S_today; blue whiting a 10.2 (age 1), b 0.61 S_today. These mix
productivity regimes (Table 1) and are for orientation only.

## 2. Group compensation

Table 3. Group elasticity e at today's SSB, weighted by member biomass share; member values clipped to [-1, 0], the
range a Beverton-Holt recruitment term can represent; 90 % intervals combine the member bootstrap draws (judgement
members uniform over their range). x = -e/(1+e) is the BH A_today/K_R with the same elasticity, 1+e = rho(A_today)/rho0.

| Group | e post-2000 stationary [90 %] | e post-2000 all [90 %] | e full stationary [90 %] | e full all [90 %] | x (post-2000 all) | 1+e (post-2000 all) |
|---|---|---|---|---|---|---|
| planktivorous_fish | -0.70 [-0.81, -0.42] | -0.74 [-0.96, -0.51] | -0.78 [-0.81, -0.71] | -0.89 [-0.98, -0.71] | 2.84 | 0.26 |
| benthivorous_fish | -0.97 [-0.99, -0.89] | -0.97 [-0.99, -0.84] | -0.90 [-0.96, -0.64] | -0.92 [-0.97, -0.86] | 39.4 | 0.03 |
| predatory_fish | -0.30 [-0.75, -0.18] | -0.65 [-0.83, -0.22] | -0.42 [-0.75, -0.20] | -0.86 [-0.94, -0.70] | 1.87 | 0.35 |
| whiting | -0.99 [-1.00, -0.39] | -0.98 [-1.00, -0.07] | -0.93 [-1.00, -0.68] | -0.92 [-1.00, -0.84] | 42.5 | 0.02 |

Member mapping (`scripts/sr_groups.py`, member rows in `tables/sr_groups.csv`):
- **planktivorous_fish**: herring 0.79 split NSAS 77 % / WBSS 23 % (WBSS 14-32 % of herring biomass in the HERAS
  Skagerrak strata 2024-2025, `../Den pelagiska poolens reglering/stock_composition.md`, F; midpoint J); blue whiting
  whb.27.1-91214; Norway pout nop.27.3a4; sprat spr.27.3a4. All assessed. WBSS at e ~ 0 pulls the group from ~-0.9
  to -0.70 (stationary) / -0.74 (all).
- **benthivorous_fish**: haddock had.27.46a20; cod = Viking substock of cod.27.46a7d20 (the northern North Sea
  / Skagerrak component, J; the substock sum gives the same post-2000 e, -1.04/-1.07); plaice ple.27.420 (includes
  Subdivision 20, Skagerrak); witch wit.27.3a47d. **No assessment**: long rough dab (0.126) and dab (0.045) take the
  mean of plaice and witch (A; dab.27.3a4 has no analytical assessment); starry ray (0.015) J range -0.6 to 0
  (elasmobranch, few large eggs - weak compensation expected); hagfish (0.013) J range -1 to 0 (unknown); "others"
  (0.046) take the mean of the assessed members (J).
- **predatory_fish**: saithe pok.27.3a46 (0.651 - it sets the group); mackerel mac.27.nea; horse mackerel = western
  stock hom.27.2a3a4a5b6a7a-ce-k8 (it includes 3.a and 4.a; the North Sea stock hom.27.4bc7d is the southern North
  Sea); hake hke.27.3a46-8abd (SS3); anglerfish anf.27.3a46 (SS3). **Not used from data**: spurdog (0.080), whose
  SAG recruitment is pup production from the assessment's structural model; instead the assessment's density-dependent
  pup survival Q = 1 + (Qfec - 1)(1 - N_pup/R0), Qfec 2.976 (stock annex 2025, WGEF 2026, F), gives
  e = -(Qfec - 1) d / Q = -0.25, -0.50, -0.87 at depletion d = N_pup/R0 = 0.3, 0.5, 0.7 (d is J: SSB is 2.5 Blim with
  Blim = 0.2 B0, so d ~ 0.5); used as a uniform range -0.87 to -0.25 (A); ling + velvet belly
  + pollack (0.024) take the mean of the assessed members (J; lin.27.346-91214 has no analytical assessment).
- **whiting**: whg.27.47d only (Skagerrak whiting whg.27.3a is a data-limited rfb stock, no S-R).

The elasticity of a group's total recruitment to a proportional change in all members' SSB is the
recruitment-weighted, not the biomass-weighted, mean of the member elasticities; biomass shares are a proxy (J).

## 3. Life-history cross-check of the origin slope

At a fished equilibrium a BH stock satisfies a SPR_F/(1 + x) = 1 and a SPR_0 = alpha_hat, so
1 + x = alpha_hat (SPR_F/SPR_0) and e = -(1 - 1/(alpha_hat SPR_F/SPR_0)) [identity].
- alpha_hat (maximum lifetime spawners per spawner): North Sea cod 9.0, Skagerrak cod 11.2 (Myers et al. 1997, F);
  from FishLife steepness h via alpha_hat = 4h/(1-h): haddock 0.66 -> 7.8, whiting 0.71 -> 9.8, saithe 0.79 -> 15,
  cod 0.79 -> 15 (Thorson 2020, F via `../Täthetsberoendets styrka per grupp/gadoids.md`); herring h 0.61-0.74 ->
  6-11, sprat 0.49-0.57 -> 4-5, mackerel 0.81-0.89 -> 17-32 (`../Täthetsberoendets styrka per grupp/pelagic_fish.md`, A).
- SPR_F/SPR_0 at today's F: 0.3-0.5 (J; F near or below FMSY for most of these stocks).
- Result: x = 0.8-6.5 for the gadoids and herring (e -0.44 to -0.87; haddock 1.3-2.9, cod 1.7-6.5), 0.2-1.5 for
  sprat (e -0.17 to -0.6), 4-15 for mackerel (e -0.8 to -0.94) [A/J].
- These are weaker than the post-2000 fits for the gadoids (e ~ -1, x_data inf for haddock and cod). The fits are
  biased toward -1 by time-series bias (Walters 1985) and errors in SSB (Walters & Ludwig 1981) and are confined to a
  narrow SSB range. The life-history route assumes a stationary equilibrium (violated: Table 1) and uses alpha_hat
  from the pre-1995 regime; today's lower productivity lowers alpha_hat and so x - it does not reconcile the two.
- The binding constraint is haddock's data floor: no BH with x below ~18 (lower 90 % bound) reproduces flat
  recruitment over its 0.15-1.0 S_today range. For benthivores and whiting the S-R data favour a flat term above a
  breakpoint (hockey stick, or K_R <= S_min); x = 1-6 from life history is the alternative to test, not the central
  value (J). Not resolved here; rho0 in either case follows from rho(A_today) = rho0/(1 + x) and today's stationarity,
  not from these fits.

## 4. ICES reference points and the S-R conventions

SAG reference points (D, `data/sag/refpts_*.json`), t:

| Stock | Blim | Bpa | MSY Btrigger | FMSY | S_today/Blim |
|---|---|---|---|---|---|
| her.27.3a47d | 828 874 | 1 049 521 | 1 100 000 | 0.23 | 2.06 |
| her.27.20-24 | 197 344 | 274 225 | 274 225 | 0.05 | 0.50 |
| spr.27.3a4 | 77 157 | 99 842 | - | - | 1.91 |
| nop.27.3a4 | 28 865 | 47 281 | - | - | 1.15 |
| whb.27.1-91214 | 1 500 000 | 2 250 000 | 2 250 000 | 0.32 | 3.74 |
| had.27.46a20 | 138 250 | 192 109 | 192 109 | 0.167 | 5.57 |
| cod Viking / Southern / NW | 14 027 / 14 433 / 23 422 | 19 491 / 21 549 / 32 547 | = Bpa | 0.198 / 0.243 / 0.189 | 1.35 / 0.64 / 1.36 |
| cod.27.47d20 (2022) | 69 841 | 97 777 | 97 777 | 0.28 | 0.70 |
| ple.27.420 | 391 695 | 544 290 | 544 290 | 0.123 | 2.56 |
| wit.27.3a47d | 3 293 | 4 576 | 4 576 | 0.163 | 1.24 |
| pok.27.3a46 | 137 653 | 191 280 | 191 280 | 0.279 | 0.86 |
| mac.27.nea | 3 067 017 | 4 119 337 | 4 119 337 | 0.191 | 1.19 |
| hom.27.2a3a4a5b6a7a-ce-k8 | 566 678 | 787 443 | 787 443 | 0.08 | 1.43 |
| hom.27.4bc7d | 179 580 | 269 555 | - | - | 0.40 |
| hke.27.3a46-8abd | 61 563 | 78 405 | 78 405 | 0.24 | 2.02 |
| dgs.27.nea | 244 829 | 342 760 | 690 186 | - | 2.54 |
| whg.27.47d | 144 175 | 201 845 | 201 845 | 0.35 | 2.57 |
| anf.27.3a46 | 25 686 | 35 692 | 38 604 | 0.137 | 1.72 |

Blim basis and the S-R models ICES uses: section 4.1.

Comparison with my hockey-stick breakpoints (post-2000, Table 2 and `tables/sr_compensation.csv`) [D]: the breakpoint
lies at 0.5-2 Blim for most stocks, often at the lowest SSB of the period (haddock, cod, sprat, western horse
mackerel), i.e. no breakpoint inside the post-2000 data; this is why ICES mostly sets an empirical Blim (type 5,
lowest SSB with good recruitment, or Bloss) or takes the breakpoint from the full series (4.1). Today's SSB is above Blim for all used members except WBSS herring (0.5), saithe (0.86) and the old cod
series (0.70; the substock Viking is at 1.35).

### 4.1 Blim basis and the S-R models in ICES's MSY work

From the 2026 advice sheets (all read in full text) and the benchmark / working group reports and annexes named (full
text unless stated), via a sub-search; Blim bases quoted from the advice sheets. F throughout.

| Stock | Blim basis | S-R in EqSim / MSY simulations, and period | Source |
|---|---|---|---|
| her.27.3a47d | "Breakpoint in the segmented regression of the stock-recruitment time-series (1947-2023, excluding the recovery period 1979-1990)" | segmented regression with breakpoint at Blim, recruitment 2002 on ("low productivity regime since 2002"); MSE operating models 50/50 segreg and BH | advice; IBPNSHerring 2021; HAWG 2024; WKMSEHerring 2025 |
| her.27.20-24 | "Empirical Blim, mean of the three lowest spawning-stock biomass (SSB) values (2010, 2013, 1996) with above median recruitment" | hockey stick estimated inside SAM, 1991-2024; its breakpoint was rejected as Blim | WKBHERSPRAT 2026 |
| spr.27.3a4 | "Empirical Blim, mean of the three lowest ... (1991, 2008 and 2013) with above median recruitment" | escapement strategy; MSE hockey stick with resampled plateau; inside SMS a hockey stick fixed at 77 231 t (earlier down-weighted, "tiny influence") | WKBHERSPRAT 2026 |
| nop.27.3a4 | "SSB in 4th quarter of 2012 ... the year with the lowest SSB with recruitment well above median" (re-estimated each assessment) | escapement; forecast resamples recruitment 1984-2026; no S-R inside SESAM found | WKBWNG 2026 |
| whb.27.1-91214 | "Approximately Bloss" | segmented regression at Bloss, 1981-2014 | WKBWMS 2016 |
| had.27.46a20 | "Lowest estimated spawning-stock biomass (SSB) that resulted in high recruitment (1979)" (type 1, spasmodic) | segmented regression at Blim, recruitment 2000-2024 ("indications of a drop in productivity ... since 2000") | WGNSSK 2025 |
| cod Viking, NW | "Estimated breakpoint of segmented regression, using the full time-series" (= Bloss, SSB 2006) | segmented regression at Bloss, 1998-2025 ("evidence suggests another [regime shift] from around 1998") | WGNSSK 2026 |
| cod Southern | breakpoint of a segmented regression on the truncated series 1998-2025 (type 2) | the same, 1998-2025 | WGNSSK 2026 |
| cod.27.21 | no reference points (category 3; SAM "indicative of trends only") | - | advice |
| ple.27.420 | "Lowest spawning-stock biomass (SSB) level that in the past has resulted in above median recruitment" (type 5, mean of 1996, 1997, 2002) | model average of segreg + BH (Ricker excluded), 1957-2024 | WGNSSK 2025 annex 8 |
| wit.27.3a47d | "Breakpoint in a segmented regression, based on recruitment 1950-2020" | segmented regression 1950-2020 | WGNSSK 2023 |
| pok.27.3a46 | "Mean spawning-stock biomass (SSB) in 1990, 1992, and 2022, which are the three lowest SSB levels resulting in above median recruitment" (type 5) | segmented regression at Blim, recruitment 2004-2025 (change point ~2002 in R/SSB, linked to "warming of the North Sea") | WGNSSK 2026; WKBGAD 2024 |
| mac.27.nea | "Lowest spawning-stock biomass (SSB) level that in the past has resulted in good recruitment (2002, 2004, 2005)" (type 5) | EqSim mix ~50 % segreg, 27 % BH, 22 % Ricker, all 25 pairs; forecast uses recruitment 2017-2026 for "low stock productivity" | WKBMACNSSH 2025 |
| hom.27.2a3a4a5b6a7a-ce-k8 | "Bloss in 2015, from the benchmark assessment" (type 5) | segmented regression at Blim, pairs 1998-2022 (pre-1998 dropped: the 1982 year class was up to 70 % of SSB) | WKBHMB 2024 |
| hom.27.4bc7d | "Bespoke method" (inflection of a loess smoother through the S-R pairs; segreg breakpoint unstable) | EqSim segreg + Ricker + BH, 2002-2021; FMSY "could not be reliably estimated" | WKBHMB 2024 |
| hke.27.3a46-8abd | "The breakpoint of the segmented regression stock-recruitment relationship; female-only", pairs 1978-2019 | EqSim BH with the SS3 steepness and R0 imposed | WKANGHAKE; stock annex 2024 |
| anf.27.3a46 | "SSB2004; lowest observed SSB historically" | EqSim BH, 1993-2019 | advice only |
| dgs.27.nea | "0.2 x B0, where B0 is the virgin total biomass" (B0 1 224 140 t) | none; harvest rate from the assessment model | WGEF 2026 |
| whg.27.47d | "Mean spawning-stock biomass (SSB) in 2016 and 2019 which are the two lowest SSB levels resulting in above median recruitment" (type 5) | segmented regression at Blim, 1983-2023 | WKBWNG 2026 |

S-R inside the assessments (F): hake SS3 BH on female SSB with steepness *estimated* (~0.9 at the benchmark; 0.99 in a
base case judged "not biologically realistic"), sigmaR 0.4 fixed, deviations from 1978; anglerfish SS3 BH with h
**0.85 fixed**, sigmaR 0.8, main deviations 1993-2025 (early ones from 1981); western horse mackerel SS3 BH with h
**0.75 fixed** (FishLife), sigmaR 1.1, three recruitment eras with separate deviation vectors (era years not found);
spurdog pup survival as in section 2 (Qfec 2.976, sigma 0.2, deviations from 1975); WBSS herring SAM with an internal
hockey stick; cod, saithe, witch, whiting and blue whiting SAM with random-walk recruitment (no S-R inside).
Consequence (J): for anglerfish and western horse mackerel part of the SAG S-R signal is the fixed-h BH.

Reading (J): ICES treats nearly all of these stocks as flat above a breakpoint (segmented regression at Blim or
Bloss) and restricts the S-R period to the post-shift regime for NSAS herring (2002), haddock (2000), cod (1998),
saithe (2004) and western horse mackerel (1998) - the same picture as Tables 1-2. The Blim of most stocks is an
"empirical" low SSB with good recruitment, i.e. the data show no SSB at which recruitment demonstrably fell.

## 5. Literature: where density dependence acts, and regime shifts

Status: FULL = full text read (by a literature sub-search for this note), FETCH = full text through a summariser,
ABS = abstract only, SNIP = snippet or secondary source.

**Recruitment vs growth vs maturation**
- Rindorf A, van Deurs M, Howell D, et al. 2022. Strength and consistency of density dependence in marine fish
  productivity. Fish and Fisheries 23:812-828, doi:10.1111/faf.12650. FULL incl. supplement Tables S1-S5.
  - "density-dependent recruitment in 68% of the stocks" (78 % excluding pelagic stocks with SSB trends); "Density
    dependence in late growth occurred in 54% of the stocks, whereas early growth was generally density-independent";
    "the effect of density on recruitment exceeded the effect of density on growth in 50 out of 60 stocks". F.
  - Pelagic stocks: 48 % without S-R density dependence; demersal 31 %; benthic 11 %. F.
  - Per stock (dAIC proportional minus flexible S-R, > 0 = density-dependent; steepness h = R(0.2 max SSB)/max R;
    late-growth correlation with density): NSAS herring +59.8, h 0.87, late growth r -0.49; WBSS herring -2.3, h
    0.23; North Sea sprat +4.7, h 0.38, growth -0.54; Norway pout +11.8, h 0.52, over-compensation 0.64; blue
    whiting -3.5, h 0.45; mackerel +7.0, h 0.66, growth -0.43; western horse mackerel +3.9, h 0.51; haddock +4.5, h
    0.98, growth -0.47; North Sea cod -1.1, h 0.20, growth -0.05 (nothing significant); plaice +14.7, h 0.76, growth
    -0.58; saithe +13.6, h 0.79, early growth -0.51; whiting +15.9, h 0.76, growth -0.50; hake +60.3, h 0.98. F.
  - Caveat in the paper: the flexible model "may ... have a tendency to conclude Ricker-type over-compensation when
    data are noisy". Maturation is not analysed. F.
  - Agreement with my fits (J): same verdicts for NSAS herring, sprat, Norway pout, haddock, plaice, whiting, hake
    (compensation), WBSS herring (none) and saithe (compensation over the full series). Disagreement: cod (they find
    none; my stationary full-series fit also finds none, the shift and post-2000 fits find full compensation) and
    blue whiting (they find none; I find weak support for const over prop, 4.6 AICc, post-2000).
- Zimmermann F, Ricard D, Heino M 2018. Density regulation in Northeast Atlantic fish populations: density dependence
  is stronger in recruitment than in somatic growth. J. Anim. Ecol. 87:672-681, doi:10.1111/1365-2656.12800. FULL.
  Density dependence in recruitment in 70 % of populations, in growth in 26-69 % by method; "97% of populations with
  statistically significant parameters for growth and recruitment showed a larger impact of density-dependent
  recruitment on population biomass". North Sea whiting is one of the few with growth density dependence comparable
  to recruitment. F.
- Lorenzen K, Enberg K 2002. Proc. R. Soc. B 269:49-54, doi:10.1098/rspb.2001.1853. ABS. Density-dependent growth in 9
  of 16 populations; argued to be a key regulating process. F.
- Andersen KH, Jacobsen NS, Jansen T, Beyer JE 2017. When in life does density dependence occur in fish populations?
  Fish Fish. 18:656-667, doi:10.1111/faf.12195. ABS. Theory: "in small habitats ... regulation occurs late in life,
  while it can occur early in large habitats". F.
- Hunter A, Speirs DC, Heath MR 2019. PLoS ONE 14:e0212176, doi:10.1371/journal.pone.0212176. FETCH. North Sea
  herring and sprat growth falls with abundance; herring matures earlier and smaller at high abundance (the opposite
  of a compensating delay); sprat maturation not affected. F.
- Olafsdottir AH et al. 2016. ICES J. Mar. Sci. 73:1255-1265, doi:10.1093/icesjms/fsv142. ABS. NEA mackerel growth
  negatively affected by mackerel and Norwegian spring-spawning herring stock size; "Growth of the latest cohort was
  34% of the maximum cohort growth recorded". F.
- Haddock growth: Jones 1983 (J. Cons. 41:50-62, doi:10.1093/icesjms/41.1.50; SNIP via Taylor & Stefansson 1999, J.
  Northw. Atl. Fish. Sci. 25:101-114, FULL): the 1962 and 1967 year classes grew slowly at first and caught up later;
  the effect is "by no means universal". No source found on the 1999/2005/2019-20 year classes. F.

**Recruitment regime shifts**
- Szuwalski CS, Vert-Pre KA, Punt AE, Branch TA, Hilborn R 2015. Fish and Fisheries 16:633-648,
  doi:10.1111/faf.12083. ABS. "recruitment and spawning biomass are not positively related over the observed range of
  stock sizes for 61% of 224 stocks"; "85% of stocks for which spawning biomass may not drive recruitment ... exhibit
  shifts in average recruitment". F.
- Sellinger E, Szuwalski C, Punt AE 2024. Fish. Res. 269:106862, doi:10.1016/j.fishres.2023.106862. FULL (thesis
  version). 432 stocks: 14 % with an SSB influence, 57 % without; 44 % of environment-influenced stocks with at least
  one shift; detection methods disagree. F.
- Vert-pre KA, Amoroso RO, Jensen OP, Hilborn R 2013. PNAS 110:1779-1784, doi:10.1073/pnas.1214879110. FETCH. Surplus
  production of 230 stocks: abundance hypothesis 18.3 %, regimes 38.6 %, mixed 30.5 %, random 12.6 %. F.
- Brunel T, Boucher J 2007. Fish. Oceanogr. 16:336-349, doi:10.1111/j.1365-2419.2007.00435.x. ABS. NE Atlantic
  recruitment: "a long-term decline, with a stepwise change occurring in 1987", followed by most North Sea
  populations "especially the gadoids". F.
- Beaugrand G et al. 2003. Nature 426:661-664, doi:10.1038/nature02164. ABS. Warming since the mid-1980s changed the
  plankton "in a way that reduces the survival of young cod". F.
- Olsen EM et al. 2011. Proc. R. Soc. B 278:504-510, doi:10.1098/rspb.2010.1465. ABS. Food and climate model explains
  45 % of North Sea cod recruitment variance; Ricker and Beverton-Holt "only explained about 10 per cent". F.
- Payne MR et al. 2009. ICES J. Mar. Sci. 66:272-277, doi:10.1093/icesjms/fsn211. ABS. NSAS herring: "an
  unprecedented 6 sequential years of poor juvenile production" from 2002 at a large SSB, set in the larval
  overwintering phase, coinciding with a regime shift around 2000. F.
- Nash RDM, Dickey-Collas M 2005. Fish. Oceanogr. 14:279-291, doi:10.1111/j.1365-2419.2005.00336.x. ABS. 1976-2000:
  "In 80% of the years, year class strength reflected SSB". F.
- Dickey-Collas M et al. 2010. ICES J. Mar. Sci. 67:1875-1886, doi:10.1093/icesjms/fsq033. ABS. "Stock-recruitment
  relationships must be viewed as being fluid and dependent on ecosystem change." F.
- Britten GL, Dowd M, Worm B 2016. PNAS 113:134-139, doi:10.1073/pnas.1504709112. ABS. Recruitment capacity fell by
  about 3 % of the historical maximum per decade, "highly negative trends in the North Atlantic". F.

**Statistical pitfalls**
- Walters CJ 1985. CJFAS 42:147-149, doi:10.1139/f85-018. ABS. Time-series bias: "the stock will appear too
  productive when it is low, and too unproductive when it is large" - compensation looks stronger than it is. F.
- Walters CJ, Ludwig D 1981. CJFAS 38:704-710, doi:10.1139/f81-093. ABS. Errors in SSB: "Large errors make
  recruitments appear to be independent of spawning stocks" - also toward e = -1. F.
- Barrowman NJ, Myers RA 2000. CJFAS 57:665-676, doi:10.1139/f99-282. ABS. BH usually estimates a larger Rmax and
  maximum reproductive rate than the hockey stick. F.
- Gilbert DJ 1997. CJFAS 54:969-977, doi:10.1139/f96-272. ABS. Recruitment "states" fit; Ricker fits "may have been
  spurious" because SSB depends on lagged recruitment. F.
- Dickey-Collas M et al. 2015. ICES J. Mar. Sci. 72:111-116, doi:10.1093/icesjms/fsu022. ABS. Patterns in recruitment
  series "may be determined by the model used to generate them". F.

**Reading (J).** The literature supports putting the stock-scale regulation in recruitment, and it supports a
recruitment term with time-varying productivity (regimes) rather than one fixed curve. It also says that a share of
the regulation (growth of recruited fish) is late-life and already partly in the engine through intake. For the
planktivores, density-dependent growth is as well supported as S-R compensation.

## 6. Methods

- **Fetch** (`scripts/sr_fetch.py`): SAG `SummaryTable`, `FishStockReferencePoints` and `StockDownload` for 24
  assessment keys (keys and labels in `data/sag/keys.json`). North Sea cod since 2023 is assessed as three substocks
  (Viking, Southern, Northwestern; keys 26001-26003, 2026); the old whole-stock SAM (key 17652, 2022) gives the long
  series.
- **Periods**: full series and "post2000" = year classes from max(2000, shift year), i.e. 2002 on for NSAS herring
  (its documented low-productivity regime), 2000 on for every other stock.
- **Pairs** (`scripts/sr_build.py` -> `tables/sr_pairs.csv`): R(y + a) with SSB(y), a = SAG recruitment age (0 for
  herring, sprat, Norway pout, haddock, hake, whiting, anglerfish, spurdog, western horse mackerel; 1 for blue
  whiting, cod, plaice, witch, North Sea horse mackerel; 2 mackerel; 3 saithe). Dropped: recruitment years from the
  terminal year on; years where the SAG 95 % interval gives a log-CV < 0.05 (recruitment fixed by the assessment:
  anglerfish before 1981, spurdog before 1975) or > 0.5 (horse mackerel recent years, spurdog 2015-16).
- **Fits** (`scripts/sr_fit.py` -> `tables/sr_fits.csv`, `tables/sr_compensation.csv`): least squares on ln R
  (lognormal errors, Gaussian ML). Models: prop (ln R = ln a + ln S, no compensation), const (full compensation), BH
  (b profiled on a grid then refined), Ricker (linear in 1/b, unconstrained), hockey stick (breakpoint profiled over
  observed SSB and midpoints, constrained to [S_min, S_max] as in ICES segreg), each stationary model with a linear
  time trend in ln a (prop, const, BH; both periods), and with a step in ln a at year class 2000 (NSAS herring 2002)
  and common b (prop, const, BH, Ricker; full series). AICc with k counting sigma. Periods as above (n >= 12). Today's SSB = mean of the last three SSB years up to the terminal year.
- **Derived**: e at S_today per model (BH -x/(1+x), Ricker -S/b, HS -1 above and 0 below the breakpoint, const -1,
  prop 0), AICc-weighted over the stationary set ("stationary") and over all fitted models ("all"); ratio of R/S at
  S_today to R/S at the origin and at the period's lowest SSB, and x_data = (1 - q)/(q - r) solving (1 + r x)/(1 + x) = q for that ratio q with
  r = S_min/S_today (`scripts/sr_tables_md.py`; inf when q <= r). A power-law slope (Cushing, ln R = c + beta ln S, e =
  beta - 1) with an AR(1)-inflated SE is in the CSV as a descriptive check.
- **Uncertainty**: moving-block bootstrap of the (S, R) pairs (block 5 years, 1000 replicates, seed fixed), every
  model refitted, weights recomputed, S_min recomputed; S_today fixed. 5 and 95 % quantiles.
- **Groups** (`scripts/sr_groups.py` -> `tables/sr_groups.csv`): four variants (post-2000 / full x stationary /
  all), member e clipped to [-1, 0]; draws combined independently by draw index.
- Reproduce (from the repo root):
  `python3 -I research_notes/Beståndsreglering/scripts/sr_fetch.py research_notes/Beståndsreglering/data/sag`, then
  `sr_build.py data/sag tables/sr_pairs.csv tables/sr_ssb_series.csv`, `sr_fit.py tables/sr_pairs.csv
  tables/sr_ssb_series.csv data/sag tables 1000`, `sr_groups.py tables/sr_compensation.csv tables/sr_boot_draws.csv
  tables/sr_groups.csv`, `sr_tables_md.py tables/sr_compensation.csv tables/sr_groups.csv` (paths relative to this
  folder; usage lines in each script).

## 7. Caveats

1. **SSB and R come from the same assessment** (errors-in-variables, Walters & Ludwig 1981; time-series bias, Walters
   1985). Both push the fitted elasticity toward -1. The AR(1) of the residuals is 0.3-0.8 for most full series, so
   the AICc differences overstate the evidence; the block bootstrap partly corrects the intervals, not the AICc.
2. **Compensation and productivity change are confounded.** Wherever SSB trended, a trend or shift in productivity
   and the S-R slope trade off: WBSS herring, saithe, mackerel and the cod series give opposite answers with and
   without a trend model. "e all" includes the trend/shift models and is my preferred column; "e stationary" is the
   bound if productivity were constant.
3. **Narrow SSB contrast after 2000**: the S-R curve is identified only over 0.3-2.4x today's SSB. Nothing here says
   what recruitment would do at 0.1x today; the hockey-stick breakpoints at the lower bound show this.
4. **Assessment-model structure**: hake, anglerfish and western horse mackerel (SS3) estimate recruitment deviations
   around an internal BH curve (hake h estimated ~0.9-0.99; anglerfish h 0.85 and horse mackerel h 0.75 fixed), so
   part of the "data" S-R is the assumption (4.1); spurdog recruitment is pup production from a structural model (its
   own pup-survival term used instead, A); WBSS herring's SAM has an internal hockey stick and sprat's SMS a hockey
   stick fixed at Blim, both weakly weighted.
5. **Stock scale, not box scale.** Every number is about stock SSB (North Sea or the whole NE Atlantic for blue
   whiting, mackerel, western horse mackerel, spurdog). The box holds a small share of each stock (WP3); a stock-scale
   BH in the box's own adult biomass is only right if the box's recruitment scales with stock SSB.
6. **Over-compensation**: haddock, blue whiting, witch and Viking cod post-2000 have bootstrap draws with e < -1
   (Ricker). A BH term cannot represent that; I clipped at -1 for the groups.
7. **Biomass shares as weights** (section 2), and the NSAS/WBSS split, are J.
8. Grades: Tables 1-3 D (computed), except the judgement members in Table 3 and the spurdog row (J). Section 3 is
   A/J. Literature entries F with the fetch status given.

## Files

- `scripts/sr_fetch.py`, `scripts/sr_build.py`, `scripts/sr_fit.py`, `scripts/sr_groups.py`, `scripts/sr_tables_md.py`
- `tables/sr_pairs.csv` (lagged pairs with use flags), `tables/sr_ssb_series.csv`, `tables/sr_fits.csv` (every model x
  stock x period: a, a_post, trend, b, b/S_today, sigma, AR1, AICc, weights, e, ratios), `tables/sr_compensation.csv`
  (per stock x period summary with bootstrap intervals), `tables/sr_boot_draws.csv`, `tables/sr_groups.csv`
- `data/sag/` raw SAG JSON (untracked)
