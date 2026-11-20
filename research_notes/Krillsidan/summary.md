# The krill side: the krill pairs' factor by season and against the booked flow (2026-11-20)

Main-agent fork (saved by the main agent). Nothing in the library, code, tests, manifest, AGENTS.md or the resume
was changed; nothing was committed. No value was set from model output: the frozen fix105 world only sized
candidates on library COPIES.
- **Trail:** `evidence.md` (K0-K6).
- **Scripts:** `scripts/booked_level.py` (policy-free; the factor the booked flows imply, pair by pair),
  `scripts/season_profile.py` (seasonal profile of the factor), `scripts/make_copy_c.py`. Output in `tables/`.
- **Frozen sizing:** `tables/fill_decomp_copy_c_s*.txt`, `tables/budget_gate_copy_c.txt`.

Tags: [F] full text, [S] held note, [D] derived, [J] judgement. PROXY = not a model species or not the box.

## Why it matters

Krill carries ~0.84 of whiting's uncapped potential and most of the planktivores' outside spring. The winter
anchors of "Mötesfaktorn per grupp" give ~0.01. Applied with zooplankton x0.36 and fish x0.16 (copy a), that took
the planktivores' krill to 0.03 of the potential and the implied M2 to 0.10 against the booked 2.07, so the diet
gate went OFF. Three readings were possible:
- the winter factor does not hold all year;
- the krill density D at the sites is far below 5 t/km2;
- the booked krill share is wrong.

## Per predator and season

| predator, season | field krill ration (or share) | D | factor (field a / chain a) | tags | basis |
|---|---|---|---|---|---|
| whiting Q1 (Skagerrak 1981; NE+NW NS 1981; Oslofjord Mar) | 0.14 / 0.45 / 0.94 %BW/d | D_lib 5 (in-box IKMT winter 7-9, floor 3, other years) | 0.0024 / 0.0088 / 0.028 at the 100-150 m class | D on F, J | stomach; ICES DB rows shared with the booked ration |
| whiting Q2 (Skagerrak Apr 1991; Oslofjord Jun) | 0.70 / 0.29 | D_lib 3 | 0.030 / 0.0095 | D on F, J | stomach; Apr 1991 IS the booked share's source |
| whiting Q3 | ~0 (open NS ~0.01 of ceiling; Skagerrak 2022 thin) | - | ~0.1 of the annual profile | F, J | stomach |
| whiting Q4 (Oslofjord Nov) | 0.41 | D_lib 6.5 | 0.0057 | D on F | stomach, independent |
| whiting, booked annual | 0.575 (diet 0.50 x Q/B 4.2) | 5 x pool shape, equal cells | **0.022** | D | the booked flow itself |
| planktivores Q1 (group Nov-Mar; pout NS 1980; pout Oslofjord Mar) | 0.24 / 0.20 / 0.75 | D_lib 5 | 0.011 / 0.009 / 0.037 | D on F, J | stomach; shared |
| pout Q1 vs Q3 (Albert 1994) | krill share 0.79-0.81 vs 0.45-0.59; content higher in winter | - | profile 1.5 / 1.3 / 0.45 / 0.7 (Q1..Q4) | F, D | stomach; the booked share's source |
| planktivores, booked annual | 0.60 (share 0.36 x Q/B 6.2) | 5 | **0.029** (zooplankton 0.024) | D | booked |
| water column (saithe) | 1991 Q2 pulse 24-49 %W, Q4 ~0; 1981 flat | - | booked annual **0.029**; profile Q3 0.2-0.6 | F, D | stomach / booked |
| mackerel | 1981 Q1 82, Q2 19, Q3 20, Q4 43 %W | - | booked annual **0.032**; profile Q3 0.25 | F, D | stomach / booked |

No co-located stomach ration and krill density exists for any model species in the notes or in two narrow searches.
- The Oslofjord papers held in full (Onsrud 2004, 2005) give layers and stomachs, but no krill density.
- The PROXY with a density (Wilson 2009, juvenile pollock) uses a fixed ration, so it gives no a.
- The level of D is therefore unmeasured at every site. The nearest pair is the in-box winter IKMT 1987-91 (7-9
  t/km2) against the Q1 stomachs of 1980-81, from other years [J]. At that D the winter factors are x0.55-0.7 of
  their values at D_lib 5.

## Reconciliation with the booked M2 [D]

The booked krill flows imply a factor of 0.022-0.032 on a (equal cells, foraging the whole tick, shared handling;
`booked_level.py`). Each predator's other pairs sit beside it:

| predator | krill | other pairs | common |
|---|---|---|---|
| planktivores | 0.029 | zooplankton 0.024 | 0.026 |
| whiting | 0.022 | planktivores 0.030, epifauna 0.059 | 0.028 |
| water column | 0.029 | fish 0.015-0.034 | 0.031 |
| mackerel | 0.032 | planktivores 0.036; zooplankton (Macy) 0.095 | 0.049 |

- The benthivores sit at 0.007.
- **The winter stomach anchors agree with the booked krill flow.** The x2-2.5 between them is conventions. The
  anchors put the fish on 100-150 m beds and use the krill-only handling term. For the planktivores they also take
  the winter ration, 0.24 against the annual 0.60.
- **So the krill pair is not the outlier.** Measured the same way as the other pairs, its factor equals theirs.
- **Copy a broke the gate by mixing methods.** Its other factors were clearance or slope anchors:
  - zooplankton V1 0.36 and Kulke 0.6-1.2;
  - Rindorf's fish slope 0.16-0.30;
  - mackerel's lab clearance, 3x its krill.

  These sit 5-15x above every booked-flow or stomach-at-density factor. The split in "Mötesfaktorn per grupp"
  (krill ~0.01, fish ~0.16, zooplankton ~0.36) is a split by anchor METHOD, not by prey type. This revises that
  note's "not shared" reading.
- **The pout krill share may be high [J].** The booked pout krill ration (~0.8 %BW/d annual at Q/B 4.5) is at or
  above the top of the measured pout winter rations (0.20-0.75 %BW/d). Albert has pout eating more in winter, so
  the planktivores' booked krill share may be high by up to x1.5-4. The whiting share (0.50, from Skagerrak April
  1991) sits inside its measured spread.

**The factor is seasonal.** Every quarterly series puts field krill intake in Q1-Q2 and a minimum in Q3. The chain's
krill encounter peaks in Q3-Q4, carried by:
- the ingestion Q10, 1.30 in summer for planktivores and mackerel;
- the krill season, Q2 0.6 and Q4 1.3.

A constant factor therefore misplaces krill by x3-5 between the quarters, and more on the 1991 shares. A
season-keyed factor would be a season switch. The candidate physical drivers [J], none carried by the model as such:
- **Vertical availability:** krill overwinter in a dense, deep day layer where the gadoids sit, and spread and
  migrate further in summer. The pool's depth shape is fixed over the year.
- **Krill size and energy:** adults are largest in winter and spring; the cohort is small in autumn (4.3 vs 6
  kJ/g).
- **The library season itself:** the Q2 trough rests on spring trawl catches, against the Q2 stomach pulses.
- **The fish's summer switch to other prey,** carried only as far as the model's other prey rise in summer.
- **The Q10 on the krill a,** which runs against the field's Q3 minimum.

## Frozen sizing [D]

Copy c applies one booked-consistent factor per predator to every pair (planktivores 0.026, whiting 0.028, water
column 0.031, mackerel 0.049). It keeps the diet shares; krill prey side 0.96 of booked.

| group | uncapped, live -> copy c | fill | need |
|---|---|---|---|
| planktivores | 0.54 -> 0.09 | 0.22-0.24 | 0.26 |
| whiting | 0.75 -> 0.26-0.33 | 0.42-0.63 | 0.39 |
| water column | 0.76 -> 0.22 | 0.31-0.34 | 0.40 |
| mackerel | 0.55 -> 0.065 | 0.37-0.39 | 0.28 |

- Every group falls far below need. The frozen fish meet much less than the equal-cell, full-time encounter
  because of light, visibility, the zooplankton's season and the placement trained while sated. So the booked-flow
  factor cannot be carried into the engine as it is.
- The level that feeds the fish lies between copy c and copy a. It is not chosen here: that would be fitting to
  model output.

## Verdict

1. **The krill factor is not anomalous.** On its booked flow it is 0.022-0.032, the same as the other pairs of the
   same predator. The winter stomach anchors (~0.01) agree within the conventions. The planktivores' booked krill
   share may still be up to x1.5-4 high on the pout stomachs [J].
2. **What the anchors disagree on is method.**
   - Factors from rations divided by a mean density (all the krill rows, every booked flow) lie at 0.01-0.05.
   - Factors from a clearance or a density-gradient slope (V1, Kulke, Rindorf, Macy) lie at 0.16-1.
   - The x5-30 between them is what the chain does not represent: time actually spent in the prey's layer, the
     density the fish actually meet, and fish not eating prey that is present. This is the search-effort /
     encounter-realisation step of 294.4, now placed between two kinds of field number.
3. **The krill factor is seasonal (Q1-Q2 high, Q3 low) on every quarterly series.** The drivers would be krill
   vertical availability and size, the library's Q2 trough and summer Q10, none of them in the model as a physical
   term.

**Settling datum:** a co-located stomach ration and krill density, from echosounder or nets, in the krill layer the
fish occupy. Ideally Skagerrak or Oslofjord gadoids and pout, in Q1 and Q3, giving a field a per season with D
measured.

Not obtained: Dalpadado & Bogstad 2004 and Orlova et al. 2005 (Barents cod vs krill availability, PROXY).

## Main agent's reading (2026-11-20)

- The booked-flow factor (0.02-0.03) is computed at equal cells with foraging the whole tick; in the frozen world
  it starves every group (copy c). The equal-cell algebra and the frozen engine disagree by the same heterogeneity
  and placement that broke the model-side "needed" factors (294.8), so neither the booked flow nor the model sets
  the level.
- A booked-flow factor is in any case a calibration to the field ration (booked M2 = diet share x Q/B). The only
  anchors that measure a without the ration are the clearance and slope ones (0.16-0.36, single studies).
