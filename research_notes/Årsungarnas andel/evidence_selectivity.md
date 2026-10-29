# Fishing selectivity on the juvenile compartment J (age 0 + age 1), per impact and group (2026-10-29)

Question: s_J = F on the J tonnes / F on the group's whole biomass, per fishing impact and group, so
that J loses s_J F and the library F (catch / total biomass, stock level) stays the group's mean rate.
Grades: **D** computed here from data, **F** stated in a source, **A** analogy, **J** judgement.

Scripts (`scripts/`, run with `python3 -I`): `sms_juvenile_f.py` (SMS quarterly F at age, stock
biomass weights), `assessment_juvenile_f.py` (2026 single-stock assessments, 2015-25),
`box_weighted_sj.py` (SMS quarterly F at age 0/1 weighted by the BOX age-0/age-1 shares of
`tables/juvenile_share_by_species.csv`, then the group sum). Output in `tables/*juvenile_f*_stdout.txt`,
`tables/box_weighted_sj_stdout.txt`. Data provenance: `data/selectivity/README.md`.

## (a) Result

For J = age 0 + age 1 all year (variant A); the B / C readings of "until the second summer" are in
"J definition" below (planktivores 0.29 / 0.59, benthivores 0.07 / 0.12).

| impact x group | s_J central | range | grade | basis |
|---|---|---|---|---|
| `pelagic_trawling` x planktivores (98 % of their F) | **0.7** | 0.35-0.85 | D | box-weighted stock F at age; 0.73 (2001-25 shares), 0.61 (1991-96 shape). Low end = sprat J unfished in the box (herring + pout alone 0.36) |
| `bottom_trawling` x planktivores (2 % of their F) | **1.5** | 1.0-2.2 (cap) | J (A) | the catch is herring and pout discards in grid Nephrops and shrimp trawls (~90 t herring discards in 3.a.20 Nephrops trawls, FDI); fish small enough to pass a 35 mm / 19 mm grid are retained in the codend while larger ones are sorted out (Frandsen et al. 2009, section c), so the catch is J-biased |
| `bottom_trawling` x benthivores (~100 % of their F) | **0.15** | 0.08-0.35 | D (range high end J) | box-weighted stock F at age; 0.15 (2001-25), 0.13 (1991-96); 0.08 with the 2021-25 SAM F of haddock and cod. High end: local shrimp/90 mm trawls retain 10-20 cm gadoids (section c) |
| `pelagic_trawling` x benthivores (F 0.0004) | **4** | 2-5.3 (cap) | J | industrial small-mesh (16-32 mm) bycatch of haddock is small fish; at 0.2 % of the F it does not matter |

Implied adult multiplier if the group mean is kept: s_A = (1 - s_J p_J)/(1 - p_J); hard cap s_J <= 1/p_J
(all F on J). With the 2001-25 annual-mean J shares p_J = 0.46 (planktivores) and 0.19 (benthivores):

| group | p_J | s_J | s_A | cap |
|---|---|---|---|---|
| planktivores | 0.46 | 0.73 | 1.23 | 2.2 |
| benthivores | 0.19 | 0.15 | 1.20 | 5.3 |

### How it was computed [D]

1. Per stock, per quarter and age (SMS North Sea key run 2023, 2013-22): F (per quarter), biomass
   N.bar x west. Box J biomass per quarter = species weight x box age-0 and age-1 shares (q-corrected
   GOV, `summary.md`; 2001-25 has Q1 and Q3, Q2 := Q1, Q4 := Q3). Then
   F_J,box = 4 sum_q (F_q0 b_q0 + F_q1 b_q1) / sum_q (b_q0 + b_q1) and r_s = F_J,box / F_all,stock
   (F_all = all ages, stock biomass weights; the same convention as the library's catch / total biomass).
   Per-tick constant loss on J removes the same tonnes as the seasonal F at age.
2. Species F_J = r_s x the library's species F (herring: 0.92 x NSAS 0.145 + 0.08 x WBSS 0.0087,
   since the box J is ~92 % NSAS by weight - HAWG 2026 Table 3.2.4, WBSS 5 % of 0-wr and 12 % of 1-wr
   in 3.a.20 Q3 - while the adults are ~0.37 WBSS; sprat 0.45, pout 0.03, blue whiting 0.18; cod 0.39, haddock 0.06, plaice 0.04, ...).
3. Group: F_J,group = sum_s W_s j_s F_J,s / sum_s W_s j_s (W = model member weight, j = annual-mean
   box J share); s_J = F_J,group / F_group, F_group = sum_s W_s F_s / sum_s W_s (0.114 and 0.179, i.e.
   the library's 0.10 and 0.17 before rounding). Species without an age key (blue whiting, long rough
   dab, dab, witch, rays, gurnard, lemon sole) have j = 0; giving long rough dab j 0.3 at plaice-like
   r 0.3 leaves the benthivore s_J at 0.15.

Per species (variant A; r_J box-weighted, SMS 2013-22, 2001-25 shares / 1991-96 shares; r_A = stock F
on the non-J cells / stock F_all; absolute F = r x library species F, /yr):

| species | r_J | box J share j | r_A | F_J | F_A |
|---|---|---|---|---|---|
| herring (NSAS) | 0.30 / 0.34 | 0.49 / 0.83 | 1.11 | 0.040 | 0.105 |
| sprat | 2.39* / 2.34 | 0.67 / 0.61 | 3.40 | 1.08 | 1.53 |
| Norway pout | 0.96 / 0.72 | 0.70 / 0.58 | 3.45 | 0.029 | 0.10 |
| haddock | 0.19 / 0.15 | 0.33 / 0.79 | 1.43 | 0.011 | 0.086 |
| cod | 0.16 / 0.14 | 0.17 / 0.42 | 1.21 | 0.062 | 0.47 |
| plaice | 0.32 / 0.32 | 0.02 / 0.05 | 1.14 | 0.013 | 0.046 |

\* r_J is against the STOCK F_all (stock p_J 0.88), with a box-weighted numerator; for sprat it exceeds
its own box cap 1/j = 1.5, so the species r cannot be applied per species. At group level it is
consistent: F_all,box = sum_s W_s (j_s F_J,s + (1 - j_s) F_A,s) / sum_s W_s = 0.117 (planktivores) and
0.183 (benthivores) against F_group 0.114 and 0.179, so s_J and the conservation s_A hold.

Sprat dominates the planktivore result: 3 % of the group biomass and 4 % of its J tonnes, but 56 % of
the J catch, because the box sprat J is mostly calendar age 1 in Q3, exactly what the sprat fishery
takes (age 1 S1 F 0.3-0.9 per half-year). Herring alone gives s_J 0.40 against the herring mix F.

Season (r by quarter, box shares 2001-25): herring Q1 0.07, Q2 0.15, Q3 0.46, Q4 0.36; sprat 0.27,
0.00, 4.1, 3.6; pout 0.03, 0.29, 1.0, 2.3; haddock 0.28, 0.28, 0.14, 0.14; cod ~0.16 all year. A
constant s_J puts some of the J catch in H1 where the stocks lose it in H2.

### J definition (main sensitivity) [D]

J here = age 0 + age 1 in every quarter (variant A, as the shares in `summary.md`). "Until their
second summer" also reads as B (age 1 leaves J at the start of Q3; plan_B.md's gamma) or C (age 1
stays through Q3, leaves in Q4). Same computation, 2001-25 shares:

| J variant | planktivores p_J | s_J | s_A | benthivores p_J | s_J | s_A |
|---|---|---|---|---|---|---|
| A: age 0 (Q3-Q4) + age 1 all year | 0.46 | 0.73 | 1.23 | 0.19 | 0.15 | 1.20 |
| C: age 1 leaves in Q4 | 0.37 | 0.59 | 1.24 | 0.15 | 0.12 | 1.15 |
| B: age 1 leaves in Q3 | 0.28 | 0.29 | 1.27 | 0.11 | 0.07 | 1.11 |

The planktivore swing is sprat and pout: their fishery takes calendar age 1 in its second autumn
(r_J sprat 2.39 -> 1.86 -> 0.13, pout 0.96 -> 0.43 -> 0.13), while herring stays 0.27-0.30. The
single-stock sprat assessment confirms it is not an SMS artefact: S1 (Jul-Dec) F at age 0 is
0.01-0.04 per half-year, age 0 in S2 (calendar age 1, Jan-Jun) 0.007-0.02, age 1 in S1 0.29-0.91
(HAWG 2026 Table 11.1.8). Use the s_J of the variant that matches the model's J (graduation timing);
with a graduation rate gamma rather than a date, A or C is closest.

## (b) Stock-level F at age used

Annual F at ages 0 / 1 / 2 / 3 / 4, biomass-weighted F on ages 0-1 (F_J), on all ages (F_all), r =
F_J / F_all with stock biomass weights, p_J = J share of stock biomass. [D on the cited tables]

| stock, source, years | F0 | F1 | F2 | F3 | F4 | F_J | F_all | r | p_J |
|---|---|---|---|---|---|---|---|---|---|
| NSAS herring, HAWG 2026 SAM (Tables 2.6.2.3-4, 2.6.1.3-4), 2015-25 | 0.065 | 0.027 | 0.091 | 0.157 | 0.211 | 0.033 | 0.188 | 0.18 | 0.14 |
| same, 2021-25 | 0.066 | 0.028 | 0.117 | 0.178 | 0.221 | 0.033 | 0.187 | 0.18 | 0.14 |
| WBSS herring, HAWG 2026 (3.6.12-13, 3.6.3-4), 2021-25 | 0.000 | 0.003 | 0.020 | 0.024 | 0.040 | 0.002 | 0.029 | 0.08 | 0.23 |
| sprat spr.27.3a4, HAWG 2026 SMS (11.1.8-9), model years 2015-24 | half-year F, see (a) | | | | | 0.233 | 0.317 | 0.74 | 0.90 |
| haddock had.27.46a20, WGNSSK 2026 SAM run, 2015-25 | 0.002 | 0.025 | 0.110 | 0.192 | 0.219 | 0.015 | 0.086 | 0.18 | 0.19 |
| same, 2021-25 | 0.001 | 0.012 | 0.049 | 0.087 | 0.112 | 0.009 | 0.069 | 0.13 | 0.11 |
| cod Viking substock, WGNSSK 2026 multistock SAM (4.8-4.13), 2021-25 (no age 0) | - | 0.036 | 0.179 | 0.417 | 0.518 | 0.036 | 0.442 | 0.08 | 0.06 |
| SMS key run 2023, 2013-22 (quarterly; stock weights): herring | 0.055 | 0.043 | 0.103 | 0.166 | 0.225 | 0.065 | 0.196 | 0.33 | 0.14 |
| sprat (SMS age 0 F = 0) | 0.000 | 0.388 | 0.852 | 0.832 | - | 0.127 | 0.193 | 0.66 | 0.88 |
| Norway pout | 0.005 | 0.272 | 1.251 | 1.323 | - | 0.133 | 0.195 | 0.68 | 0.88 |
| haddock | 0.003 | 0.054 | 0.227 | 0.424 | 0.520 | 0.039 | 0.194 | 0.20 | 0.35 |
| cod (SMS age 0 F = 0) | 0.000 | 0.052 | 0.236 | 0.437 | 0.453 | 0.043 | 0.304 | 0.14 | 0.19 |
| plaice | 0.000 | 0.049 | 0.154 | 0.195 | 0.229 | 0.038 | 0.152 | 0.25 | 0.16 |
| whiting (reference) | 0.001 | 0.026 | 0.069 | 0.120 | 0.146 | 0.017 | 0.072 | 0.24 | 0.36 |
| SMS 1991-96: herring / haddock / cod / sprat | 0.23 / 0.02 / 0 / 0 | 0.31 / 0.18 / 0.26 / 0.57 | | | | | | 0.86 / 0.29 / 0.32 / 0.75 | |

- HAWG 2026: https://ices-library.figshare.com/articles/report/Report_of_the_Herring_Assessment_Working_Group_for_the_Area_South_of_62_N_HAWG_/31424315
- WGNSSK 2026 cod section and haddock SAM run (NShaddock_WGNSSK2026_Run1, stockassessment.org): local
  copies in `../Torskfiskarnas bytesandelar/data/stock/`.
- SMS: https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2023 (`summary.out`).
- [F] ICES Advice 2026 her.27.3a47d Table 1: interim-year "Fages 0-1 (2026) 0.052" vs "Fages 2-6 0.168"
  (ratio 0.31 to Fbar, not to F_all); "Fishing pressure on juveniles in 2025 is estimated to be the highest
  in the past 10 years" (`../Fiskeridödlighet i Skagerrak idag/ices_pelagic_f.md`).
- Annual (SAM) vs quarterly (SMS): the annual models spread the age-0 catch over a full year of age-0
  biomass, which halves r for herring (0.18 vs 0.33). The model's age-0 tonnes exist only from Jul-Aug,
  so the quarterly weighting is the right one; the SAM rows give the 2021-25 level (haddock and cod r
  0.08-0.13, lower than SMS 2013-22: the strong 2019-20 haddock year classes and the low recent F).

## (c) Caveats

- **Age 0 vs age 1.** Gadoid age 0 is practically unfished (F0 0.001-0.003 haddock, 0 cod in both
  assessments); all of the gadoid J F is on age 1 (15-25 cm, near the 90 mm diamond-mesh L50).
  Herring is the reverse: F on 0-ringers (0.055-0.085) exceeds F on 1-ringers (0.02-0.04), the
  0-group being taken as bycatch in the industrial sprat fisheries (NSAS fleets B in the North Sea and D
  in 3.a). Sprat and pout J F is on calendar age 1 in Q3-Q4.
- **Seasonality.** Age 0 is absent from catches and from the box in H1 (Q1-Q2 r 0.07-0.15 for herring
  vs 0.36-0.46 in H2). A constant s_J is a time-mean; it moves J catch into H1.
- **The sprat fishery and the C/D fleets.** Juvenile-targeting fisheries exist: the North Sea/3.a
  small-mesh sprat trawl (16-32 mm, "Industrial trawl 100 %") takes 0-group herring (fleets B/D) and the
  sprat it targets are mostly age 1. In 3.a the NSAS catch (Table 2.2.9) is juvenile-heavy: 0-1 wr are
  0.12-0.69 of the catch weight in 2015-21 and 0.8-0.9 in 2024-25 [D, catch numbers x assumed 0-wr
  8-12 g, 1-wr 25-40 g]; against the box herring J share 0.49 (species level) that is a LOCAL s_J ~0.7 (2015-21) and ~1.7
  (2024-25). The 3.a herring fishery is ~0.5-0.7 kt/yr since 2022, so this changes nothing at the stock
  level that the library's F is built on. In 1991-96, with a large 3.a C/D fishery, SMS herring r was
  0.86: the regime matters.
- **Local gears vs stock level.** The library F is stock-level (catch / stock biomass), so the stock
  s_J is the consistent centre. Skagerrak gear evidence points the other way for the dominant local
  bottom fleets: Danish 90 mm diamond codend L50 cod 15-17 cm, haddock 15 cm, whiting 18 cm (Swedish
  grid / 70 mm square mesh 26-27, 26, 33 cm; Frandsen, Madsen & Krag 2010, ICES JMS 67: 1721-1731,
  Kattegat-Skagerrak [A, summary]); a rigid sorting grid in a Nephrops trawl "increased the retention
  of small cod and haddock" (cod 10-19 cm, haddock 11-15 cm; Frandsen, Holst & Madsen 2009, Fish. Res.
  97: 243-252 [A, summary]); the shrimp trawl's 19 mm grid passes fish below ~20 cm into a 35 mm
  codend (Larsen et al. 2021, PLoS ONE, Barents Sea [A]). So the local shrimp trawl, the largest share
  of grid bottom-trawl hours, has a gear-level s_J above 1 for gadoids and herring; its absolute gadoid
  catch is small and is already inside the stock assessments' discard estimates. No Skagerrak
  catch-at-age by metier was found; the high end 0.35 for benthivores is J.
- **Discard survival.** Assessments count discards as dead; J catch is mostly discards, so if some
  survive s_J is a little lower.
- **Benchmarks.** The 2026 cod benchmark rescaled F; SMS (2023, pre-benchmark) is used for the shape only.
- **Unkeyed flatfish** (~27 % of the benthivores) have no F at age; plaice r 0.25-0.32 is the analogy.
  Dab F 0.18 is mostly discards of fish above ~15 cm, which also argues for r well below 1.
- Shares of 1991-96 are 2-4x those of 2001-25; s_J is fairly stable across them (0.61-0.73, 0.13-0.15)
  but s_A is not (1.2 vs 1.7-2.0), because s_A depends on p_J.
