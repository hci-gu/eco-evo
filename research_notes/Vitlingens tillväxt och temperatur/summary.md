# Whiting growth and temperature (2026-11-10)

Question: do warm shallow beds pay small whiting through faster growth at ample food? If so, by how much, and in what form would a temperature term on growth go into the model?

Research only. Nothing in fgconfig/, lib/, tests/, mareld2.yaml or mareld_resume.txt was changed, and no model was run. Written by the research subagent, saved by the main agent. Sources are numbered E1-E23 in `evidence.md`. Derived numbers come from `scripts/gadoid_proxy_growth.py` -> `tables/gadoid_proxy_growth_stdout.txt`. Downloads are listed in `data/README.md`.

Grades: [F] field, [L] lab, [J] judgement, [A] abstract or secondary only.

Sizes use whiting W = 0.0042 L^3.12 (g, cm; via E8). Under 20 cm is under 48 g; 20-30 cm is 48-171 g; 30 cm and over is 171 g and more.

## Verdict

1. **Partly supported, and only through gadoid proxies. There is no whiting growth-temperature series.**
   - The only whiting lab growth data are at one temperature (13-14 C; E1).
   - The one whiting experiment with a temperature factor found no temperature effect on net conversion efficiency at a given consumption rate. Temperature enters only indirectly, through rations expressed relative to satiation (E3 [L][A]).
   - Cod fed to satiation (E5 [L]) grow faster in warmer water when small and slower when large. Relative to the model's bottom-layer mean of 7.4 C:

     | Cod size | 12 C | 14 C | 16 C |
     |---|---|---|---|
     | 12-30 g | x1.42-1.50 | x1.43-1.59 | x1.34-1.61 |
     | 109 g | x1.07 | x0.85 | x0.47 |
     | 447 g | x0.94 | x0.81 | x0.62 |

   - So for whiting under ~20 cm, beds at 10-14 C in Aug-Oct are near or below a small gadoid's growth optimum (13-17 C). For whiting of 30 cm and more they are at or above it (8-10 C) [J, proxy].
2. **The gain comes from intake, not from a separate growth term.**
   - At a fixed ration, temperature does not raise growth: whiting conversion efficiency is independent of temperature (E3). The haddock fit is G = -0.3 + 35.1 C - 0.07 T, with C as printed (a daily fraction), and its temperature term is slightly negative (E7 [L]).
   - Satiation growth rises because Cmax rises faster than maintenance, up to an optimum that falls with size.
   - The model already has this: ingestion_q10 2.25 and metabolism_q10 2.3.
   - 2.25 gives x1.45 from 7.4 to 12 C, the same as small cod's satiation growth (x1.42-1.50). It overshoots above ~13 C: x1.71 at 14 C and x2.01 at 16 C, against cod x1.34-1.61 [D].
   - **A growth Q10 on top of these is not supported.** It would double-count the same physiology [J].
3. **Field growth of whiting does not track temperature.**
   - Food and fishing explain length-at-age; temperature does not:
     - North Sea: temperature R2 0.02-0.08 against forage fish R2 0.58 (E8 [F]);
     - Clyde and west of Scotland: temperature coefficients small and not significant (E10 [F]);
     - Irish Sea: length-at-age is temperature-independent (Gerritsen 2003, via E8 [A]).
   - Warming is associated with smaller asymptotic size, not larger: whiting L-infinity fell 13-29 % with +1-2 C bottom temperature (E11 [F]), and fell in the Adriatic (E12 [A]). That is adult size over decades, not juvenile growth across a 5-8 C spatial contrast.
4. **Habitat use fits a warm-water juvenile, but no study separates temperature from food or predators.**
   - Age-0 fish west of Scotland are densest at 12-13 C bottom water in Oct-Nov and less dense with depth (E16 [F]).
   - Black Sea recruits sit shallower and warmer (10-15 C) than adults (E20 [A]).
   - 0-group whiting live in 1-2 m Skagerrak eelgrass at ~19 C in Jul-Sep, but 58 % died within 5 days at 23 C (E17 [F][L]).
   - Bristol Channel 0-group movements follow their shrimp prey (E18 [A]).
5. **Does it pay to be there? Not answerable from the literature.**
   - No study weighs growth against risk or food for any juvenile gadoid's depth choice.
   - Order of magnitude [J]:
     - ~x1.4 satiation growth for 3 warm months on 10-50 g fish is ~x1.1 on annual somatic growth;
     - none for adults.
     - Food differences in the field are larger than that (E8: stomach contents -60 % between periods).

**Form, if one is wanted [J].**
- No new parameter is needed. Evaluate the existing ingestion and metabolism Q10s at each cell's bed temperature instead of the calendar layer mean. That is a code mechanism, not a library value.
- Bound it by size:
  - Q10 ~2.0-2.3 valid from 7 to ~13 C for fish under ~50 g (to ~15 C under ~15 g);
  - flat (~1) at ~100 g;
  - declining above ~9-10 C for fish over 200 g.
- Model-side check, not evidence: the growth branch's ceiling B g (s - u) p carries no temperature factor. If that ceiling binds at ample food (whiting fill ~0.98, `Vitlingens ranson`), the ingestion Q10 cannot show up as faster growth in warm cells: they would only cost more.
  - The biologically consistent fix is to give that ceiling the same temperature factor as Cmax, with annual mean 1. That is a growth_q10 equal to ingestion_q10, not one on top of it.
  - The loader refuses growth_q10 together with kappa (184.7). Whether that refusal also covers material_growth_share is not checked here.
  - Whether the ceiling binds should be checked in a probe before anything is changed.

## Q1. Growth at ample food vs temperature (lab)

- **Whiting.**
  - Jones & Hislop 1972 (E1 [L][A]), at 13-14 C: weight gain is linear in ration; length growth of 26-32 cm fish saturates at 6.5 mm per 10 days.
  - Andersen & Riis-Vestergaard 2003 (E3 [L][A]): kappa = 0.426 C^0.11, no significant temperature or size effect. The tested temperatures are not obtained.
  - Not found:
    - satiation growth per temperature;
    - an optimum or upper decline;
    - a growth Q10;
    - a size-specific temperature response.
- **Cod proxy** (E5 [L], satiation, 1-16 C):
  - Topt falls from 17.0 C (1.5 g) to 15.5-16.3 (12 g), 13.2 (29 g), 10.1 (109 g) and 8.1-8.8 (447-2213 g). Gmax falls from 5.6 to 0.30 %/d.
  - Q10 over 7-13 C: 2.15-2.32 at 12 g, 2.04 at 29 g, 1.05 at 109 g, 0.62-0.82 at 447-2213 g.
  - A 2 C departure from Topt costs only 1.4-5.2 % of growth.
  - The 2002 food-unlimited model (E6 [L][A]) gives the same size dependence (Topt 14.3 C at 50 g, 5.9 C at 5000 g).
- **Age-0 cod and haddock** (E7 [L]):
  - Growth and scope for growth peak at 12 C and fall at 15 C (cod GGEmax 3.6-6.2 %).
  - Gross efficiency is highest at 5-8 C (26-32 %).
  - Haddock (3.8 g) satiation growth is the same at 8 and 12 C (observed 3.0-3.1 %/d).
- **Saithe** (E7b [F][L][A]): SGR at age 0-1 is highest in the warmer south. Growth optima move to lower temperatures as fish grow.
- **Growth efficiency vs temperature:**
  - whiting: no significant effect (E3);
  - cod: the optimum for feed conversion falls from 15 C at 2 g to 7 C at 2000 g, below Topt for growth (E5);
  - age-0 cod: efficiency peaks at 5-8 C (E7).

## Q2. Field growth vs temperature

- **North Sea** (E8 [F]): adult length-at-age follows forage fish, not temperature. Temperature was even 0.6 C higher in the poor-growth period.
- **North Sea, 8 species** (E11 [F]):
  - L-infinity is down 13 % (N) and 29 % (S) with +1-2 C.
  - The correlation with bottom temperature at ages 0-2 is r -0.43 to -0.54.
  - The authors stress it is not proof: fishing and density are not excluded.
- **West of Scotland** (E10 [F]): whiting L-infinity and k have non-significant negative temperature coefficients; size-selective Nephrops trawling is favoured.
- **Irish Sea:** length-at-age is temperature-independent (via E8 [A]).
- **Baltic** (E4 [A]): western Baltic females grow slower than North Sea females; temperature is not separated.
- **Adriatic** (E12 [A]): L-infinity fell from 29.5 to 22.8 cm in 30 years, with warming as the authors' hypothesis.
- **0-group:**
  - Severn/Bristol: 58 to 115 mm SL from Aug to Nov (E9 [F]);
  - SW Netherlands: ~170 mm SL by the end of the season (E14 [A]);
  - Brittany: ~190 mm TL by Nov-Dec (via E9);
  - west Scotland sea lochs are larger than Bristol.
  - These are latitudinal contrasts confounded with food and season length; none analyses temperature.
- **Skagerrak/Kattegat:** no whiting growth-temperature study found.
- **Otolith chronologies:** none found for whiting.

## Q3. Thermal habitat of juvenile whiting

- **Age-0 west of Scotland** (Oct-Nov; E16 [F]): densest at 12-13 C bottom water over fine sediment, less dense with depth, and strongly aggregated (80-96 % of fish in 20 % of the space).
- **Age-1** (Feb-Mar): 8-8.5 C optimum; shallower (50 m) when colder.
- **Mature fish:** denser up to 9.7 C offshore only.
- **Skagerrak coast** (E17 [F][L]):
  - 0-group whiting (12 cm, 13 g) are in eelgrass at 18.9 C in Jul-Sep.
  - CTmax ~29-30 C, the lowest acclimation of the species tested.
  - 23 C killed 58 % in 5 days.
  - The "preferred 12-18 C" in that paper is an AquaMaps envelope [A], not a measurement.
- **Black Sea** (E20 [A]): recruits at 40-75 m and 10-15 C, adults deeper and colder.
- **Adriatic** (E19 [A]): adults at the range edge prefer ~15.4 C and ~45 m.
- **Seasonal movement:** Bristol 0-group leave offshore after ~10 months, following Crangon prey (E18 [A]).
- **Cod proxy** (E22 [A]): cod leave Skagerrak shallows above ~15-16 C.
- **Not found:**
  - a lab preferred temperature for whiting;
  - telemetry of juvenile whiting;
  - any study separating temperature from food or predators in nursery choice.
- **The consistent picture [J]:**
  - Juveniles sit warmer and shallower than adults, as a falling Topt with size predicts.
  - The upper bound is ~16-18 C for 0-group, with 23 C lethal over days.
  - The box's 10-16 C at 10-30 m in Aug-Oct is inside that window.

## Q4. What a model value would look like

- **Form:** a dome with a size-dependent optimum (cod: Topt = 17 C at 2 g to ~8 C at 1-2 kg), not a single Q10 [L proxy].
- **As a group-level approximation [J]:**
  - the existing intake and metabolism Q10s (2.25 / 2.3), referenced to the layer's annual mean 7.4 C, evaluated per cell;
  - valid up to ~13 C for whiting under ~20 cm;
  - capped or declining above ~13-15 C for juveniles and above ~9-10 C for fish of 30 cm and more.
- **Measured vs judged:**
  - Measured: cod and haddock lab responses (E5, E7), and whiting's temperature-independent net efficiency (E3, abstract).
  - Judged: whiting = cod at equal weight. Whiting may tolerate slightly more heat (E17: whiting survived 23 C better than cod; its range reaches the Adriatic and Black Sea).
- **Growth-specific term:** not supported beyond intake and metabolism (point 2 of the Verdict). The one exception is the model-side cap noted under Form.

## Key values

| Source | Species | Size | T range (C) | Value | Grade |
|---|---|---|---|---|---|
| E1 Jones & Hislop 1972 | whiting | 26-32 cm | 13-14 only | max length growth 6.5 mm/10 d; weight gain linear in ration | L, A |
| E3 Andersen & Riis-Vestergaard 2003 | whiting | not obtained | not obtained | kappa = 0.426 C^0.11; no significant T effect | L, A |
| E5 Bjornsson et al. 2001 | cod | 1.5 g | 1-16 | Topt 17.0, Gmax 5.63 %/d | L |
| E5 | cod | 11.6-29 g | 1-16 | Topt 13.2-16.3; Q10(7-13) 2.0-2.3; G12/G7.4 1.42-1.50 | L (D) |
| E5 | cod | 109 g | 1-16 | Topt 10.1; G12/G7.4 1.07; G16/G7.4 0.47 | L (D) |
| E5 | cod | 447-2213 g | 1-16 | Topt 8.1-8.8; Q10(7-13) 0.62-0.82 | L (D) |
| E6 Bjornsson & Steinarsson 2002 | cod | 50 / 5000 g | 2-16 | Topt 14.3 / 5.9 | L, A |
| E7 Peck et al. 2003 | cod age-0 | 0.9 / 7.3 g | 5-15 | growth peaks at 12 C; 15 C GGEmax 6.2 / 3.6 %; GGE best at 5-8 C | L |
| E7 | haddock age-0 | 3.8 g | 8, 12 | Gmax 3.0-3.1 %/d at both; T term -0.07 at fixed ration | L |
| E8 Lauerburg et al. 2018 | whiting | adults | North Sea Q1/Q3 | length-at-age R2: T 0.02-0.08, food 0.58 | F |
| E10 Hunter et al. 2016 | whiting | all ages | summer SST | L-infinity -3.19 (SE 3.69) cm/C; k -0.15 (SE 0.095) /yr/C; n.s. | F |
| E11 Baudron et al. 2014 | whiting N/S | L-infinity | +1-2 C bottom | L-infinity -13 % / -29 %; r with bottom T at age 0-2 -0.54 | F |
| E16 Burns et al. 2019 | whiting age-0 | Oct-Nov | bottom T | peak density 12-13 C, falls with depth | F |
| E17 Tamarit-Castro et al. 2026 | whiting 0-group | 12 cm, 13 g | 19 / 23 | in eelgrass at 18.9 C; 58 % died in 5 d at 23 C; CTmax ~29-30 | F, L |
| E20 Dagtekin et al. 2026 | whiting recruits | - | 10-15 | recruits at 40-75 m, 10-15 C; adults deeper | F, A |
| model (library) | whiting | group | bottom layer | ingestion_q10 2.25, metabolism_q10 2.3; x1.45 at 7.4 -> 12 C | - |

## Gaps

1. No whiting growth-temperature series in the lab. Hislop's tank temperatures are unknown. Andersen & Riis-Vestergaard 2003 (temperatures, sizes, satiation growth) was not obtained in full text.
2. No whiting otolith chronology. No Skagerrak/Kattegat growth-temperature analysis.
3. No 0-group nursery growth vs temperature with food separated. Hamerlynck & Hostens 1993 was abstract only.
4. No lab thermal preference for whiting, and no telemetry of juvenile depth vs temperature.
5. No study of the growth vs risk or food trade-off behind shallow nursery use, for any gadoid.
6. The proxy assumption (whiting = cod at equal weight) is untested. Whiting is somewhat more heat-tolerant and more southern.
7. Ross et al. 2018, Gerritsen et al. 2003, Kerby et al. 2013, Tobin et al. 2010 and Kutsyn 2024 were abstract or secondary only.
8. Model-side (not literature): whether the unscaled growth ceiling binds for whiting at ample food.

## Main-agent code check (2026-11-10)

- The refusal covers both fed-only laws. `config_loader.py:900-911` raises for `income_growth_kappa` and for `material_growth_share` when growth_q10 is set, and `state.py:433-435` does the same. Whiting (`material_growth_share` 0.48) cannot carry a growth_q10 today.
- `movement.income_growth` (`movement.py:365`) pays G = min(B g (s - u) p, kappa_m I p / ec, max(I - C, 0), R' - u ME B'):
  - the ceiling B g (s - u) p has no temperature factor;
  - the material cap kappa_m I p / ec follows the intake and so its ingestion_q10.
- Which of the four binds for whiting per cell is not measured.
- Whiting's library values are metabolism_q10 2.3 (t_ref annual_mean) and ingestion_q10 2.25. Both are per tick on the calendar, not per cell (`interactions.py:567-653`).
