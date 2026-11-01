# Zooplankton summer rest: is seasonal dormancy real for the modelled group, and what does it buy? (2026-11-01)

Research only; library, manifest, code and tests not edited. Tags: [F] value read in the source (ABS = abstract only); [S] search snippet or secondary citation, not checked in the source; [D] derived here (scripts in `scripts/`, output in `tables/`); [J] judgement. Quotes with page numbers are in `evidence.md` (sections E1-E9). Written by the research subagent, saved by the main agent.

## Verdict

1. **The resident 0-25 m community has no summer dormancy at this site.** This is the model's group since 183.6. It feeds and grows through the summer.
   - At A17, resident carbon (total minus Calanus and Metridia) rises from 20,500 ug C/m3 in May to a peak of 24,900 in June, and is still 12,100 in July [D, E1].
   - Its dormancies bridge winter, not summer, and none of them is something an adult "rest" action can express [F, E7]:
     - Acartia clausi stops reproducing in Dec-Jan.
     - Centropages hamatus, Temora and Acartia survive the winter as resting eggs in shallow sediment. At A17, C. hamatus has a median of 0 in every month except June [D].
     - Pseudocalanus acuspes switches to resting populations in early autumn in North Norwegian fjords.
   - Krill migrate up and down every day (DVM) and are not dormant [F].
   - The only summer signals in the residents are Jul->Aug drops of Pseudocalanus (median 1,280 -> 128 ind/m3) and Temora (738 -> 72) [D]. For Pseudocalanus, a stay below the thermocline is documented in the Baltic [S]. Dormancy is not shown, and which Pseudocalanus species occurs at A17 is unresolved.
   - So "the modelled group should not rest in summer" is the evidence-based reading, in the sense of seasonal dormancy [J].
   - **Resting would not lower the summer stock anyway.** The field decline from June to August (total 11.4 -> 4.6 t/km2, resident 7.4 -> 1.6) is animals leaving and dying, not animals becoming inactive. A learned summer rest keeps tonnes in the pool. It would raise the model's August stock, which is already 9.0 against 4.6 (total) or 1.6 (resident) [D/J].

2. **Summer dormancy is real for Calanus finmarchicus CV, and it shows in the A17 target.**
   - In 0-25 m, Calanus falls from 66,300 to 13,600 ug C/m3 between May and June. That is 19.7 -> 4.0 t WW/km2, about 16 t/km2 lost in one month [D, E1].
   - Gullmar fjord (120 m deep, on the Swedish coast 10-30 km east of the box edge), Hirche 1983, p. 283 and 287 [F]:
     - The overwintering stock "commences at the end of May by accumulation of CV".
     - From August to December, "99 %" of the population were CV, "almost exclusively below 50 m", torpid, with empty guts.
   - At Arendal, C. finmarchicus drops abruptly in July, once the surface exceeds 12 C (Falkenhaug et al. 2022) [F].
   - The animals leaving in June are the new generation of the Dec-Mar Calanus that 183.6 excluded as import. Both ends of the Calanus cycle are outside the resident group, yet the Apr-Nov target (the A17 total) still contains them [D/J]. That is where the expectation of a "summer rest" comes from.
   - Note: A17 itself (58.275 N, 10.513 E) lies inside the box bounds.

3. **What dormancy buys, and where it works.**
   - **Metabolism.** At the same temperature, dormant CV respire 0.15-0.25 of active CV [F]:
     - Hirche 1983, Table 1 (Korsfjord, deep vs surface CV on the same dates, both at 6 C): 0.11/0.72 and 0.18/0.79.
     - Maps et al. 2014 (meta-analysis of 15 species): "about one-fourth" of actively growing copepods.
     - Ingvarsdottir et al., via Grigor et al. 2022: 10-20 %, with swimming activity down 88.5 %.
     - Colder water adds to this: the box's 150-300 m layer is about 7 C, against 14-17 C at 0-10 m in Jul-Aug [D, E2].
   - **Mortality.** In deep dormancy habitats (380 m and deeper) it is 0.004-0.027/d, against 0.05-0.10/d for active late stages in summer: 2-20x lower [F/S, E6].
     - Oceanic basins: 0.004/d [S].
     - Deep fjord without mesopelagic fish: 0.008-0.009/d (Bagoien et al. 2001).
     - Clear deep fjord with mesopelagic fish: 0.024-0.027/d (Bagoien et al. 2001).
   - **This depends on depth, and the box is shallow.**
     - In the 120 m Gullmar fjord, the chaetognath Sagitta elegans "consumed 5 to 18% of the deeper dwelling Calanus community daily" (Tonnesson & Tiselius 2005) [F ABS].
     - Hirche 1983 p. 287: "the CV from Gullmarfjord could survive for only 120 d", and "In places like the Gullmarfjord the species considered here cannot overwinter" [F].
     - Melle et al. 2014, on the North Sea: older stages "cannot migrate vertically far enough to avoid predation, because of the relatively shallow bathymetry" [F].
   - **Deep cells inside the box.** The 200-300+ m cells (28 % of the grid) are 6.6-8.6 C and well oxygenated (5.2-6.3 ml/l), so they offer no hypoxic refuge [D, E2].
     - They are dark by day: at Kd 0.15 the water is below the herring biting threshold under about 112-123 m even in June (zooplankton_refuge.md).
     - The limits there are how long the reserves last and tactile or mesopelagic predators, not light.
     - At 7 C a CV can stay dormant about 130-136 d [D, from Saumweber & Durbin 2006, S], against a median of 250 d in the NE Atlantic (Melle et al. 2014) [F]. A June descent inside the box would therefore run out around Oct-Nov [D].
     - Falkenhaug et al. 2022 say diapausers enter via the deep Atlantic inflow in the Norwegian Trench, and that overwintering aggregations exist in the Trench [F]. That the January Calanus at A17 is mainly inflow rather than local diapausers is a reading [J].

4. **The model's Rest cannot carry a dormancy, and the trained behaviour is already plausible.**
   - **Starvation comes within days.** The reserve buffer from full down to the maintenance line is 338 MJ/t, 7.5 % of body energy. At rest that lasts 2.7 d (Sep) to 5.0 d (Mar), and 2.8-3.6 d in Jun-Aug. After that, starvation removes up to 34 %/d [D, `tables/rest_payoffs_stdout.txt`].
   - **Even a diapause-level cost buys only weeks.** At 0.15-0.25 of the active cost, at 7 C, the same buffer lasts 17-30 d. A real diapause lasts 4-8 months on lipid stores many times larger. A policy that rested through the summer would starve, so not learning it is correct for this group [D/J].
   - **Rest share vs field hiding.** The trained May-Sep rest share (0.14-0.26) falls inside the range implied by field hiding:
     - The daytime hiding share is 0.13-0.44 at floor 0.2 (zooplankton_refuge.md).
     - May-Sep days at 58 N are 15-18 h, so as a 24 h mean that is about 0.6-0.75 of the daytime value, i.e. about 0.08-0.33 [D].
     - Whether the trained rest follows the day-night cycle was not checked; only monthly shares were given.
   - **Correction to the brief.** "80 % hidden" holds only for whiting and the water-column predators. The two predators that book about 94 % of zooplankton M2 (planktivores 1.6, mackerel 0.13 /yr) have pair `visibility_floor` 0.5, so a resting zooplankter is 50 % visible to them.
     - With summer rest at 0.2, the model's zooplankton is about 1 - 0.2 x 0.5 = 0.90 visible to herring by day. That is at the top of the literature's 0.65-0.90 daytime range [D].
   - **Non-feeding share.** In Jun-Aug, 0.72-0.80 of the biomass does not feed (rest + move). Field copepods that do not ambush prey feed 27-99 % of the time, median about 50 % (Tiselius & Jonsson 1990) [F]. So the model does not under-rest in energy terms; Rest vs Move is a choice inside the model that biology does not settle.

## Parameter table (zooplankton)

| Key / quantity | Current | Proposed | Basis |
|---|---|---|---|
| Seasonal dormancy | none | **none (leave)** | The resident community has no summer dormancy; real dormancy belongs to Calanus CV, outside the group since 183.6 [J] |
| `resting_cost` | 1.0 (default) | **1.0, leave** | For the residents, Rest is a daytime hide with no metabolic cut beyond temperature (winter_stock.md Q3: none found). Rest already costs 1/4 of eating, matching fed vs starved respiration (>4x, Kiorboe et al. 1985). A diapause factor of 0.15-0.25 would overpay every daytime rester about 4-6x [J] |
| Rest at a deeper temperature (e.g. the `at_50m` layer) | n/a | **leave off** (optional at most) | From the manifest's own tables, cost at 50 m / cost at mean_0_50m: Jun 0.83, Jul 0.75, Aug 0.71, Sep 0.78, but Jan-Mar 1.10-1.13 (50 m is warmer in winter) [D]. Only the minority that migrates daily reaches 50 m by day; most residents do not migrate (Daro 1988; Irigoien 2004) [F]. Small effect [J] |
| `natural_mortality` | 0.015812/tick = 0.063/d = 23.1/yr | unchanged | Outside this question (split-M1 work is in 183.2) |
| `m1_visual_share` | 0 (off) | **off** (if ever set: 0.05-0.10) | About 2 of about 20 /yr is visual predation by unmodelled fish (139.1). Norway pout is now inside `planktivorous_fish` (209), so the remainder is smaller [D/J] |
| `m1_tactile_share` | 0 (off) | **off** (if ever set: 0.5-0.75) | 10-15 of 19.7 /yr is tactile (139.1) [S/D] |
| `depth_risk_ratio` | 1 (inert) | **1** | Evidence points both ways. Higher risk at depth: S. elegans takes 5-18 %/d of the deep Calanus at 120 m, and Pseudocalanus migrates in reverse to avoid invertebrate predators. Lower risk: invertebrate-only predation at true diapause depth gives 0.008-0.009/d. No Skagerrak day/night profile was found [F/J] |
| `hide_reference` | default 0.5 | if ever switched on: **0.10-0.30, from field biology** (never the trained share) | Field daytime hiding is 0.13-0.44 at floor 0.2, the floor `natural_mortality.py` uses; it is 0.2-0.7 at floor 0.5. As a 24 h mean that is 0.6-0.75 of the daytime value [D]. Sign check from `build`: with reference 0.5 and a population hiding 0.2, v_ref = 0.6 and V = 0.84. The visual M1 then runs at 1.4x its budgeted share, above `natural_mortality`. The normalisation holds only at the realised mean hiding [D] |
| `m1_visual_dark_ratio` / `m1_visual_threshold_deg` | defaults 0.1 / -6 | defaults (inert) | No zooplankton-specific source found |
| Floor in the visual-M1 term | the group's own 0.2 (code) | **0.5, if the term is used** | The unmodelled visual predators are herring-like, and the herring and mackerel pairs use 0.5. `natural_mortality.build` takes the group's 0.2. This is a code note, not a library value [D/J] |

Even with the "if set" values, the shares would move at most about 10 % of M1 at rho = 1. That is why "leave off" stands, consistent with 139.3 [J].

## Open points for the user

1. **The Apr-Nov target still contains the Calanus excluded in Dec-Mar.** The new Calanus generation's CV that leave 0-25 m in June are worth about 16 t/km2 at A17. Three options, not chosen between:
   - (a) **Resident-only target all year.** At C/WW 0.084, Apr-Nov becomes 2.12, 6.10, 7.40, 3.59, 1.56, 0.79, 2.03, 2.66 t/km2 [D].
     - The model (3.05, 7.59, 7.95, 6.86, 9.00, 3.77, 2.45, 1.25) would be close in Apr-Jun, 2-6x high in Jul-Sep, and 2x low in Nov.
     - Consistent with 183.6; the group then has no dormant member of any weight.
   - (b) **Keep the total target and add a June export** (the new Calanus generation leaving the layer and the box). A17, Hirche 1983 and Falkenhaug 2022 support it, but it is the counterpart of the import 183.6 declined.
   - (c) **A separate dormant Calanus compartment.** It would have its own lipid reserve, metabolism 0.15-0.25x at the deep temperature, and mortality 0.004-0.027/d. This is a structural change, and the Gullmar evidence says such a stock is short-lived in a box this shallow.
2. **Does the trained rest follow the day-night cycle?** If it is a daytime hide with night feeding, the May-Sep rest share is plausible. If it is flat over 24 h, it double-counts the herring and mackerel `dark_ratio`. One frozen run, binned by light level, would settle it.
3. **Pseudocalanus in summer.** The Jul-Aug drop at A17 could be a move below 25 m rather than death. The species is unresolved, and Pseudocalanus is about 20 % of resident carbon in July [D].
4. **Library documentation.** `fg_library.yaml` still lists `member_species: "copepods (Calanus, Metridia, small species)"`. That conflicts with the resident-only decision in 183.6 and may itself be a source of the surprise.
5. **Not obtained:**
   - Summer vertical profiles of Calanus in the Skagerrak or Norwegian Trench.
   - Density of mesopelagic fish (Maurolicus, Benthosema) in the deep cells.
   - Hirche 1984 (Crustaceana Suppl 7), the Gullmar seasonal distribution.
   - Gislason et al. 2007: the 0.004/d comes from a search summary [S].
   - Bagoien et al. 2001 and Tonnesson & Tiselius 2005 were read as abstracts only.

## Files

- `evidence.md`: quotes with pages; A17 tables (E1-E2); code facts (E9).
- `scripts/a17_stage_season.py` -> `tables/a17_stage_season_stdout.txt`: A17 taxa and Calanus stages by month, means and medians.
  - SHARK's `sample_id` is empty for 2007-18, so visits are keyed on date + id + depth. A naive key pools those years into one false visit.
- `scripts/a17_depth_temperature.py` -> `tables/a17_depth_temperature_stdout.txt`: A16/A17 temperature and oxygen by depth and month, 1991-2020.
- `scripts/rest_payoffs.py` -> `tables/rest_payoffs_stdout.txt`: model rest and eat costs by month, days to starvation, dormancy costs, mortality comparison.
- `scripts/openalex_abstracts.py` -> `tables/openalex_batch1.txt`, `tables/openalex_batch2.txt`: abstracts.
- `data/README.md`: downloads (Hirche 1983, Melle 2014, Wilson 2016, Grigor 2022; one failed PNAS download, deleted) and the SHARK files reused read-only from the notes "Attackrater ur motesteori" and "Exponering av bottenbyten".
