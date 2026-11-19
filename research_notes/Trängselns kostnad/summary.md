# The cost of crowding: what holds field demersal fish near Lloyd 3?

Research note (2026-11-19). Research only: no code, library or test changed. Builds on
research_notes/"Klumpningens spakar" (section 278.6), "Interferens hos bottenfisk",
"Täthetsberoende och interferens hos torskfiskar", "Födobegränsning" and resume 278-278.6. Those
results are cited, not redone.

Grades: [F] full text read; [A] abstract only; [D] data analysed here; [S] secondary (cited
elsewhere, a search snippet, or a sibling note). "(computed)" marks my arithmetic on published
numbers. Numbers from Auestad 2023, Thorson 2016, Pednekar 2024, Engelhard 2008 and Kaiser &
Spencer 1994 / Kaiser & Ramsay 1997 were checked against the PDFs by the main author; rows marked
"(helper)" rest on the literature helpers' reading.

Two yardsticks are used throughout:
- An intake cost is an elasticity of intake per tonne on local own density. The rough break-even
  (278.6) is about -0.05 at clump densities.
- A mortality cost is in /yr, against the clumping saving of 0.16-0.25 /yr.

The "range" column gives the density span a result was measured over, as a multiple of the mean.
The model's clumps sit at 30-90x, so nearly every field number is an extrapolation.

## Headline

- **Field hauls in the Skagerrak never approach the model's clumps** [D].
  - Data: 1866 valid Q1 and Q3 GOV hauls, 1991-2025, in the 10 Skagerrak and box rectangles.
  - Scale-free first: the densest haul is at most 9.5x (Q1) and 13x (Q3) that survey's mean for the model's six species, against clumps at 30-90x.
  - Hauls above 20 t/km2 hold 0 % (Q1) and 11 % (Q3) of the group's survey biomass, and none exceeds 31 t/km2. The model's clumps hold most of the biomass at 50-150 t/km2.
  - The absolute comparison is only indicative: a 3.7 km tow spans about four cells and can dilute a one-cell clump up to ~4x, and GOV catchability adds about +-2x.
  - The densest hauls recur at the same stations, around 80 m, year after year: haul-scale aggregations are anchored to sites.
  - Field crowding in 2021-25 (group 2.7 in Q1, 3.5 in Q3) is the highest of the 35 years (earlier decades 1.5-2.6). The model's gap from the long-run field level is wider still.
  - Elsewhere, 100 t/km2 or more is seen only for hours in cod converging on a spawning-capelin pulse, or on migration and spawning routes.
- **Direct field evidence on intake (haul-scale stomach content against own density).**
  - Values: haddock -0.07 (Barents Sea, south-east area); over three areas the precision-weighted value is -0.06 (unweighted -0.02 to -0.05). Baltic large cod -0.19 and flounder -0.21. Small cod and Barents cod 0.
  - Weighted over the box group (haddock 0.68, cod 0.10, flatfish 0.22): about -0.04 to -0.11, central about -0.07 (computed). With unweighted haddock, the low end is -0.01.
  - That range straddles the -0.05 break-even.
  - Every value is measured over at most ~5-25x the mean, with depletion and interference never separated. The haddock SEs are optimistic (pseudo-replicated stomachs).
- **New condition test: no cost detected** [D].
  - 50 551 weighed fish, Skagerrak and Kattegat, 1991-2025, with additive year and rectangle effects. A second variant uses a station-site effect: the same site in different years.
  - Haddock weight-at-length rises 0.5-0.8 % per doubling of own density (CI above 0 in both variants). Cod is flat (CI about -0.4 to +0.3 % per doubling); plaice is flat to positive.
  - An ideal free distribution (IFD) equalises payoffs, so a real crowding cost would also produce this.
  - Weight-at-length understates a growth loss. Condition cannot test fish that co-occur for only days, as the model's clumps do.
- **New basin test: the haddock and group signals differ** [D].
  - Haddock crowding falls as box abundance rises: slope -0.17 to -0.20 over a 59-135x range of abundance. The bootstrap CI excludes 0, and the sampling null is about 0. Thorson et al. 2016 find the same sign across stocks (effective area: gadoids +0.106).
  - Haddock of 20 cm and over in Q3 show only -0.05 (n.s.), so part of the haddock signal is small fish.
  - Weighted over the group, the sign reverses: group crowding rises with group abundance in Q3 (+0.43, CI +0.31..+0.55). That is driven by flatfish (+0.62) and plaice (+0.25), whose rich years come as local summer patches. This is a composition and patch effect, not tested further.
  - So the basin pattern shows a density-dependent payoff for haddock at 10-50 km, not for the group. Its size cannot be recovered from distribution data.
- **No field study shows intake per fish falling with fish density at cell scale over days, on benthic prey.**
  - What exists: fish converge on trawl-made food within 2-3 h (3-12x the fish marks), eat it out, and leave within about 1-3 days. That is the model's own clump time scale.
  - The croppable, renewable prey (brittle-star arm tips, siphons) regrows over weeks.
  - An illustrative calculation, resting on an abstract-only stock estimate, puts one 4-day hold at a small share of the tips.
- **The field evidence on swamping is mixed.**
  - For: predator numbers rise with prey at only 0.23 (whiting) and 0.34 (weever) on sandeel. Whiting goes negative against night sandeel and between grids, and shows none against clupeids or juvenile gadoids. Per-capita intake plateaus at 6-17 % of maximum ration. Capelin in the densest regions survived best.
  - Against (abstract only): at site scale in reef and kelp fish, per-capita mortality rises with prey density, through an aggregative response (Anderson 2001; White & Warner 2007).
  - The North Sea station-scale mortality index falls with density but is not significant (Rindorf & Gislason 2005).
  - The model's own aggregation, about 0.03-0.36 (computed), is a differently weighted statistic. It is in the same order as the field slopes, not shown to be inside their range.
  - Nothing found shows real predators converging strongly enough to remove the payoff, but the evidence for demersal prey is thin.
- **Other costs have no field support.**
  - Oxygen: bottom water at 100-300 m in the box sits at 72-100 % saturation, against cod growth limits at 45-56 %.
  - No measurements were found linking parasites, aggression or kleptoparasitism to adult gadoid or flatfish crowding.

## Q1. Density-dependent habitat selection and local payoff against density

| Study | Species, area | Scale | Currency | Magnitude | Range (x mean) | Separates depletion? | Grade |
|---|---|---|---|---|---|---|---|
| Auestad 2023, MSc thesis, Univ. Bergen, hdl:11250/3072358, Table 6 | haddock, Barents Sea, Aug-Oct 2009 | haul | ln stomach fullness (non-empty) on ln own density | SE area: -0.070 (SE 0.029) by length group, -0.068 (0.031) station total; NW -0.067 (0.091), -0.029 (0.083); SW +0.079 (0.099), -0.044 (0.140). R2 < 0.01. Individual stomachs from 49 stations are pseudo-replicates, so the SEs are optimistic | top haul ~20-30x the median, ~17 t/km2 (computed, 0.35 kg) | no | [F] |
| Lindmark et al., bioRxiv 10.1101/2024.04.22.590538 (prior note) | Baltic cod, flounder | haul, spatial GLMM | stomach content | large cod -0.185, flounder -0.21, small cod 0 | up to ~6x | no | [D] (prior) |
| Fall et al. 2021, 10.1111/ecog.05473 (prior note) | Barents cod on capelin | haul | consumption | own density not retained | - | - | [F] (prior) |
| Lindmark et al. 2023, ICES JMS 80:1539, 10.1093/icesjms/fsad084 | SE Baltic cod | haul, ICES rectangle | Le Cren condition | "did not detect a negative effect of cod density on cod condition" (haul +, rectangle -, both CIs span 0) | - | no | [F] preprint |
| Hiddink, Jennings & Kaiser 2005, 10.1111/j.0022-1112.2005.00912.x | North Sea haddock, 24 sites | site | liver index | positive: better where haddock were most abundant | - | no | [A] |
| Jónsdóttir et al. 2024, ICES JMS 81:587, 10.1093/icesjms/fsad108 | Icelandic haddock | area means over 30-47 000 km2 | weight, condition, liver | negative over years; "no clear decline" in stomach fullness at the highest density | 0.8-8 t/km2 | no | [S] (publisher HTML read by helper) |
| **This note: condition vs haul density** (`tables/condition_vs_density.csv`) | Skagerrak + Kattegat, Q3 (feeding season) | haul; additive year + rectangle effects, or year + station-site effects (same site, different years) | ln weight-at-length | per doubling of own density, rectangle / site variant: haddock +0.53 % (CI +0.27..+0.79) / +0.56 % (+0.19..+0.92), adults +0.84 %; cod -0.01 % (-0.22..+0.21) / -0.04 % (-0.42..+0.33); plaice +0.19 % / +0.58 % (+0.19..+0.97), adults -0.32 % (-0.74..+0.11); witch +1.2 % (n 89 hauls). Q1 similar. | SD of ln density within rectangle or site ~1.1-1.6, i.e. ~10x | no | [D] |
| Casini et al. 2016, 10.1098/rsos.160416 | Baltic cod | whole stock, annual | condition | abundance explained 36 % of variance (1976-93); not a local test | stock | - | [F] |
| Thorson et al. 2016, Proc R Soc B 283:20161853, 10.1098/rspb.2016.1853 | 92 demersal species incl. NS-IBTS | survey domain | effective area on abundance | mean 0.061 (SE 0.014); Gadiformes 0.106 (0.026); Pleuronectiformes 0.076 (0.025) | stock time series | - | [F] (manuscript) |
| **This note: basin test** (`tables/basin_test.csv`) | Skagerrak, NS-IBTS 1991-2025 | 10 rectangles | slope of ln(crowding) on ln(survey mean) | haddock Q1 -0.20 (bootstrap -0.31..-0.06), Q3 -0.17 (-0.31..-0.06), resampling null +0.02; cod +0.15 (n.s.); long rough dab Q3 -0.27; plaice +0.25, flatfish Q3 +0.62, group Q3 +0.43 | abundance 59-135x (haddock) | - | [D] |
| Swain & Wade 1993, 10.1139/f93-083; Swain & Sinclair 1994, 10.1139/f94-104 | cod, s. Gulf of St Lawrence | strata | stock area | "abundance explained 63-94 % of ... stock area"; D90-95 expands, D50 core does not | 1971-90 | - | [A] |
| Marshall & Frank 1995, 10.1139/f95-099; Blanchard et al. 2005, 10.1139/f05-109 | juvenile haddock (Scotian Shelf), juvenile cod (North Sea) | strata | distribution; length-at-age | habitat selection is density dependent; no growth signal (haddock); contracts to optimal temperature at low stock (cod) | - | - | [A] |

Reading:
1. **The only haddock test of intake against density is weak.**
   - The basis is one year, one of three areas, and an unpublished thesis.
   - Its -0.07 is 6x the model's -0.012. The precision-weighted 3-area value is -0.06; unweighted it is -0.02 to -0.05.
   - The North Sea haddock evidence (Hiddink; this note's condition test) points the other way: dense where good.
2. **Under an IFD, payoff does not vary with density** (Shepherd & Litvak 2004).
   - So the condition nulls are what an IFD with a real crowding cost would also produce.
   - Payoff-against-density regressions can show a cost only where fish are not at an IFD, or once habitat quality is controlled finely enough. This is the Lindmark lesson: the naive GLM gave -0.02 to -0.05, the spatial-field model -0.19 to -0.21.
3. **What the condition test does bound.**
   - A sustained intake elasticity e changes weight-at-length by about 1.34 e tau over a residence time tau in years. The 1.34 is the benthivore budget Q/B 4.46 times the growth-ration slope 0.3 (computed).
   - For e = -0.05 that is -0.0056 per unit ln density at tau = 1 month, and -0.0026 at 2 weeks.
   - Weight-at-length responds less than total weight, because part of lost growth is lost length. So 1.34 e tau is an upper bound on the expected signal.
   - Haddock's CI lies above 0 in both the rectangle and the site variant. With 1.34 e tau as an upper bound, that only weakly argues against e = -0.05.
   - Cod's site-variant CI (-0.006..+0.005) cannot exclude e = -0.05 at tau of 1 month or less.
   - Fish that co-occur only for days, as the model's clumps do, cannot be tested by condition at all.
   - So the test does not detect a cost; it does not rule one out.
4. **Basin evidence.**
   - Box haddock spread out as they become abundant, more strongly than the cross-stock gadoid mean. The Skagerrak is at the edge of the North Sea haddock range, so part of this is stock-scale spillover.
   - This shows that some density-dependent payoff steers haddock at 10-50 km. It does not give the size of the payoff: a power-law payoff (q_i d^-m) gives a constant distribution shape for any m, so the shape change identifies only the non-power-law part.
   - Cod and the flatfish show no basin signal, and plaice, dab and flatfish crowding rises with abundance. That is consistent with year-class or summer-feeding patches, but untested.

## Q2. Aggregation size and the upper tail of local density

Skagerrak, NS-IBTS GOV 1991-2025 [D] (`tables/haul_tail.csv`, `crowding_tail_basin_stdout.txt`, `dense_haul_sites.csv`).
- Densities use q = 1 and a nominal 0.1416 km2/h swept area.
- A haul averages a ~3.7 km tow, 0.07 km2. At a single 1 km2 cell the density could be higher, and GOV q differs from 1 (haddock possibly > 1 on the door basis, cod and flatfish < 1). Treat the absolute values as +-2x.
- Anchor: 2021-25 haddock crowding 4.61 here against 4.36 in the earlier note (differences: haul set 272 vs 265, nominal swept area); cod 2.48 against 2.47.

| Taxon | Q | Hauls | Mean t/km2 | p99 | Max | Hauls > 20 / > 50 | Biomass share in hauls > 20 | Max / survey mean |
|---|---|---|---|---|---|---|---|---|
| model group (6 spp.) | 1 | 929 | 0.84 | 5.1 | 9.9 | 0 / 0 | 0 | 9.5 |
| model group | 3 | 937 | 1.88 | 17.9 | 30.7 | 8 / 0 | 0.115 | 13.3 |
| haddock | 3 | 937 | 0.75 | 9.5 | 28.6 | 3 / 0 | 0.113 | 22.7 |
| cod | 3 | 937 | 0.37 | 3.4 | 14.7 | 0 / 0 | 0 | 18.2 |
| flatfish | 3 | 937 | 0.76 | 4.7 | 29.5 | 2 / 0 | 0.075 | 15.4 |

- Crowding over 35 years (median per decade, discreteness-corrected; `crowding_tail_basin_stdout.txt`):
  - model group 1.5-2.7 in Q1 and 2.1-3.5 in Q3;
  - haddock 2.9-9.3;
  - cod 1.6-3.7;
  - flatfish 1.4-3.4.
  - 2021-25 is the most crowded period for the group (2.7 in Q1, 3.5 in Q3); the 1991-2020 decades sit at 1.5-2.6. The field target of ~3 is therefore at the top of the long-run range, and the model's gap is if anything wider.
  - The discreteness correction (Lloyd 1967, Poisson sampling of individuals) changes nothing material.
- Persistence: each Q3 survey's densest haul lies a median 0.6 km from a top-decile haul of another year, against 4.5-9 km for a random haul (haddock 0.57 against 4.6 km). Dense hauls are recurring, site-anchored places (44F8 and 44G0, median depth 82 m). The model's clumps, by contrast, wander: day-to-day overlap is 0.6 and cells are held for less than 4 days (278.3).

Literature (Q1/Q2 helper, `lit/q1_q2_ifd_aggregations.md`):

| Source | Context | Density | Size and persistence | Grade |
|---|---|---|---|---|
| Auestad 2023, Fig. 21 | haddock, Barents Sea, feeding season | top haul ~47 000 fish/km2, ~17 t/km2 (computed, 0.35 kg); bulk 0.1-2 t/km2 | haul | [F] |
| Pednekar et al. 2024, Commun Biol 7:1386, 10.1038/s42003-024-06952-6 | cod (61 cm) converging on spawning capelin, Finnmark, February | shoal edge "0.06 cod/m2", about 130 t/km2 (computed, 2.2 kg); ~2.5 million cod | formed within ~1 h of dawn; observed ~4 h | [F] |
| Rose & Kulka 1999, 10.1139/f99-207 | northern cod on a migration corridor during the collapse | 450 000 t in 7000 km2 = 64 t/km2 (computed); cores > 0.1 fish/m2 | 1990-92 | [A] |
| Makris et al. 2019, 10.1111/faf.12331 | cod spawning, Lofoten | not read | "tens of kilometres" | [A] |
| Kristensen et al. 2021, Ecol Evol, 10.1002/ece3.7939 | Skagerrak fjord cod, telemetry | - | 95 % home range 1-25 ha (mean 7.1) | [F] |

Reading:
- Feeding-season densities above ~30 t/km2 are not seen in Skagerrak or Barents Sea haddock hauls.
- Densities of 50-150 t/km2 occur only as short-lived shoals around a prey pulse (hours), on migration routes, or at spawning. Spawning is kept apart; it holds the densest cod aggregations.
- No source shows persistent packing at km scale while feeding. I found no shoal dimensions or nearest-neighbour distances for haddock or flatfish.

## Q3. Prey depletion at aggregation scale

Helper notes: `lit/q3_q5_depletion_other.md`. The channel sought was intake rate falling under saturation; total prey biomass falling (which the model already does) does not count.

| Study | System | Measured | Magnitude | Time scale | Grade |
|---|---|---|---|---|---|
| Kaiser & Spencer 1994, MEPS 112:41, 10.3354/meps112041 | gurnards, whiting, dogfish over beam-trawl tracks, Irish Sea | side-scan fish marks over fished against unfished ground (Table 4); stomachs | 0.58 vs 0.05 marks per 2 min after tow 3, 20.4 vs 6.41 after tow 6; gurnard and whiting ate more prey after trawling | hours | [F] |
| Kaiser & Ramsay 1997, MEPS 152:307, 10.3354/meps152307 | dab, Irish Sea, 40 m | density on a trawled line; stomach index | dab density lower on the trawled line for 48 h, not aggregating; stomach index up on day 1 only; dab normally "graze the exposed arms" of Amphiura | 1-3 d | [F] |
| Kaiser & Hiddink 2007, MEPS 350:267, 10.3354/meps07194 (review) | scavengers on trawl damage and carrion | arrival, residence | arrive 30-50 min; abundance raised "up to 72 h ... thereafter declines" | 0.5 h-3 d | [F] review |
| Sköld, Loo & Rosenberg 1994, MEPS 103:81, 10.3354/meps103081 | Amphiura filiformis, Gullmarsfjord, 40 m | arm regeneration | regenerated tissue 12-30 % (mean 22 %) of biomass; regrowth 9.7 mm per 41 d; 13 % of production | weeks | [F] |
| Duineveld & van Noort 1986, Neth J Sea Res 20:85 | dab and Amphiura, Oyster Ground | arm cropping | 4 % of animals with intact arms; dab crop ~6 % of the arm stock per year | year | [A] |
| Hill et al. 2000, 2002, J Fish Biol | juvenile plaice, laboratory | search | area-restricted search on high local prey density; "win-shift" when dispersed | seconds to minutes | [A] |

Reading:
- No source measures intake per fish against local fish density at cell scale over days on benthic prey. The elasticity cannot be computed from this literature.
- What the field does show is the model's own behaviour: fish converge on a fresh food patch within hours, deplete it, and leave within 1-3 days. A short hold is the field's giving-up time on pulses, not evidence against costs.
- A croppable, renewable fraction is real (arm tips, siphons, lugworm tails; regrowth over weeks).
  - Illustrative estimate (computed by the helper from the Duineveld & van Noort abstract): croppable Amphiura tips are at most ~14 t WW/km2.
  - A 50-150 t/km2 clump eats 2.4-7.3 t/km2 in 4 days at Q/B 4.46.
  - It would remove 2-10 % of the tips at a 20 % diet share, and 17-52 % only if arm tips were the whole diet.
  - So a cost from depletion would build up over revisits within the regrowth time, not within one hold.
  - The tips came from a 40 m site and the southern North Sea. The box's 100-300 m mud has other ophiuroids at lower density.
- Paywalled and not read: Hall et al. 1990 (dab cages), Peterson & Quammen 1982, de Vlas 1985, Rosenberg & Selander 2000, Groenewold & Fonds 2000 (abstract only).

## Q4. Does field predation on demersal fish swamp?

Helper notes: `lit/q4_predator_swamping.md`.

(a) Aggregative response: how many more predators converge.

| Study | Predator / prey | Scale | Magnitude | Grade |
|---|---|---|---|---|
| Engelhard et al. 2008, MEPS 369:213, 10.3354/meps07575, Table 3 | whiting, lesser weever / sandeel | stations ~12.5 km apart | ln predator numbers on ln sandeel: whiting +0.23 (day sandeel) but -0.42 (night sandeel) and -2.60 between grids (fewer whiting in the sandeel-rich grid); weever +0.34 (night). Per-capita sandeel eaten +0.08 / +0.14 | [F] |
| Temming et al. 2004, J Fish Biol 64:1351, 10.1111/j.0022-1112.2004.00400.x | whiting, haddock / sandeel | 15 x 20 nm, 5 d | whiting density rises with sandeel in its stomachs (r2 0.48); haddock not significant | [F] (helper) |
| Rindorf & Gislason 2005, JEMBE 324:1, 10.1016/j.jembe.2005.04.013 | whiting / clupeids, juvenile gadoids | 5 stations x 48 h | "no aggregative response"; plateau at 10-17 % (clupeids) and 6 % (gadoids) of maximum ration; clupeid mortality index "decreased to virtually zero at high densities", but the correlations are not significant ("not conclusive") | [F] (helper) |
| Anderson 2001, Ecology 82:245, 10.1890/0012-9658(2001)082[0245:PRPRAD]2.0.CO;2 | kelp bass / kelp perch | lab tanks vs field plots | lab: per-capita mortality "inversely density dependent to density independent"; field: "strongly density dependent, with evidence for an aggregative response" | [A] |
| White & Warner 2007, Ecology 88:3044 | reef piscivores / juvenile wrasse | cm microsites vs ~2400 m2 sites | per-capita mortality falls with group size at microsites, but "increased with settler density" at the site scale | [A] |
| Pednekar et al. 2024 (above) | cod / capelin | shoal | capelin in the densest regions "have the highest survival rates" | [F] |
| Temming et al. 2007, Ecosystems 10:865 | whiting / 0-group cod | 18 km2, 5 d | > 50 million juvenile cod removed from a hot spot | [S] |
| McConnell et al. 1999, J Appl Ecol 36:573; Sharples et al. 2012, PLoS ONE 7:e37216 | grey and harbour seals | trips | repeated trips to discrete areas (patch fidelity); no numerical slope against prey density | [A] / [F] (helper) |
| Sveegaard et al. 2012, MEPS 468:245, 10.3354/meps09959 | porpoise / herring | ~870 km2 cells | partial Mantel statistic 0.254; no slope | [F] (helper) |

(b) Patch-scale saturation. Fish intake plateaus at prey patches, mostly below gut capacity (time windows): whiting reaches 6-17 % of maximum ration, and Fall et al. 2021 find cod on capelin digestion-limited. A plateau fixes intake per predator just as a full gut does. Porpoises feed almost continuously and follow encounters (Wisniewska et al. 2016 [F, helper]).

Reading:
- The evidence is mixed.
  - Dilution and swamping act at the group and microsite scale (capelin; White & Warner microsites; the non-significant whiting mortality index).
  - At site scale, the two reef and kelp studies (abstract only) find per-capita mortality rising with density through predator aggregation. Temming 2007 (second-hand) is a demersal case of a hot spot being wiped out.
- For North Sea predators on demersal or sediment-bound prey, the measured aggregative slopes are ~0.2-0.35 or negative (log(x+1) station regressions with many zeros, ~10 km apart).
- The model's predators concentrate 1.1-3.4x on the clumps at 30-90x, an implied elasticity of roughly 0.03-0.36 (computed). That is a differently weighted statistic, so it is of the same order, not shown to be inside the field range.
- Reading for 278.6: no field evidence was found that real predators remove the payoff, but none rules it out for demersal prey at km scale. The cost side remains the more promising place to look, and confidence that the predator side holds no lever is low-moderate.
- Gaps: no seal or porpoise numerical slope at 1-10 km, no saithe aggregation study, and no per-capita mortality test for North Sea demersal fish.

## Q5. Other crowding costs (brief)

- **Oxygen: irrelevant** [D, helper].
  - SHARK A16/A17 bottles at 100-300 m, 1990-2024: median 6.0 ml/l, minimum 4.56 ml/l; saturation median 89 %, minimum 72 %.
  - Cod growth is cut at 45-56 % saturation (Chabot & Dutil 1999 [A]).
  - A clump would draw down a 10 m bottom layer by ~0.3 % per day (computed).
- **Parasites.** Lernaeocera passes through a flatfish intermediate host, and seal-borne worms scale with predator density, not crowding. No measured link was found between local host density and directly transmitted parasites or ulcers in cod.
- **Aggression and kleptoparasitism.** Captive cod are aggressive in autumn and before spawning, but "When food was given all aggressive behaviour was lost" (Brawn 1961 [A]). There are no field data; the earlier note found none either.
- **Group-keeping cost.** None measured. Gadoids are facultative shoalers, and schooling energetics, where measured in other fish, is a benefit.

## What could enter the model

These are recommendations only. No value is chosen to reach the -0.05 break-even or to match a crowding number. Each mechanism must change payoffs and stand on its own field data.

| Candidate | Field magnitude it implies | Clears ~-0.05 at clump densities? | Confidence |
|---|---|---|---|
| A. **Prey-neutral local crowding term on intake** (Crowley-Martin, or a shared-denominator term on own local density, applied once per cell across all prey; the per-pair form remains unsupported, per "Interferens hos bottenfisk"). Size it from the field haul-scale stomach elasticities in their measured range, group-weighted. | -0.04 to -0.11, central ~-0.07, using the precision-weighted haddock -0.06 (-0.01 at the low end if haddock is unweighted, -0.02); cod 0 to -0.19; flatfish 0 to -0.21 via the flounder proxy. Depletion is included, so the mechanism is unresolved. | Straddles it. At 30-90x the answer is set by the functional form, not by data: everything is measured at at most ~25x. | low-moderate |
| B. **Depletion of a croppable, renewable bed fraction** (arm tips, siphons, emergent prey) with weeks-long renewal, so that intake depends on the exposed fraction near the clump. The model already has the structure (infauna refuge exchange 2.5 /yr; epifauna depletes under clumps to ~0.2 K). It costs nothing because the half-saturations sit 40-400x below the accessible density. | Illustratively (abstract-only stock estimate), a 4-day hold removes 2-10 % of the croppable tips: small. A cost needs revisits within the regrowth time and a half-saturation set on the croppable fraction, which "Födobegränsning" does not support (bed pairs saturated in the field). | No, on present evidence | low |
| C. **Stronger predator aggregative response** | North Sea station slopes 0.2-0.35 or negative; the model is roughly 0.03-0.36 (a different statistic). Site-scale density-dependent mortality is found in reef and kelp fish (abstract only). | Not shown | low-moderate (that it is not the lever) |
| D. Oxygen, parasites, aggression, group-keeping | no field magnitude | No | moderate |

Notes for the user's decision:
1. **An unexplained field signal.** The field shows negative haul-scale stomach elasticities (-0.07 haddock in one area, -0.2 Baltic) even though the bed pairs are saturated in the field too. Something lowers intake per fish where fish are dense that saturated depletion alone does not explain. Candidates: interference (no fish measurement), depletion of a croppable fraction, or habitat confounding. That gap is what to resolve before choosing A or B.
2. **The cleanest field constraint is the distribution itself.** Over 35 years, field haul densities stay below ~31 t/km2, and dense sites recur in place. A diagnostic, not a target: log the model's share of benthivore biomass above 20 and 50 t/km2 against the field's 0-11 % and 0 %. Also log the model's haul-scale elasticity of intake on own density at field-like crowding, with a spatial control, against the field -0.02 to -0.2 (model -0.012).
3. **The site anchoring of field aggregations points at persistent habitat quality.** The ~80 m stations recur year after year, which suggests field patchiness is set by where food is produced, not by moving swamping clumps. This is an observation, not a lever. Any habitat term must come from habitat data, not from a density cap.
4. **What would most reduce the uncertainty.**
   - A spatial GLMM of North Sea or Skagerrak haddock stomach content against own haul density, Lindmark-style. The ICES stomach database holds almost no haddock records with haul data (see "Födonivå och mättnad i fält"). The Norwegian-Russian joint database (Barents haddock) is the realistic source.
   - The full text of Hall et al. 1990 (dab cages) and Groenewold & Fonds 2000.

## Files

- `scripts/build_haul_density.py`: haul densities (t/km2) per species from the "Lek i rutan" DATRAS extracts.
  - Run: `python3 -I scripts/build_haul_density.py "../Lek i rutan/data" "../Fiskgrupper i rutan/tables/length_weight_all.csv" tables [all]`
  - Output: `tables/haul_density.csv`, or `tables/haul_density_all.csv` with `all` (Kattegat included).
- `scripts/crowding_tail_basin.py` (`python3 -I ... tables`): `tables/survey_crowding.csv`, `haul_tail.csv`, `basin_test.csv`, `crowding_tail_basin_stdout.txt`.
- `scripts/condition_vs_local_density.py` (`python3 -I ... "../Lek i rutan/data" tables`): `tables/condition_vs_density.csv`, `condition_vs_density_stdout.txt`.
- `scripts/dense_haul_sites.py`: `tables/dense_haul_sites.csv`.
- `lit/q1_q2_ifd_aggregations.md`, `lit/q3_q5_depletion_other.md`, `lit/q4_predator_swamping.md`: helper notes with full citations, quotes and access failures.
- `data/README.md`: inputs (no new raw data; DATRAS public, ICES attribution).
- `papers/`: PDFs, text extractions and Unpaywall metadata, ~28 MB, mixed licences (`papers/LICENCES.md`). Not to be committed, except the CC BY and public-domain files listed there. The derived tables and scripts may be committed.

Not accessed: cdnsciencepub and Wiley full texts (Swain & Wade, Swain & Sinclair, Marshall & Frank, Blanchard 2005, Rose & Kulka, Hiddink 2005), Armstrong & Schindler 2011, Temming 2007, Hall et al. 1990, Groenewold & Fonds 2000 (full text), Tamdrari 2010, Myers & Stokes 1989.

## Model against the two field diagnostics (main agent, section 278.7)

Frozen fix104 u1267 on the pre-279 library, 30 x 30, seeds 0 and 1, one probe year after the warm-up (`model/scripts/cf_field_diag.py`, `model/tables/`). A cell's tonnes are t/km2 at the reference grid's density scale; the benthivore mean is 1.57-1.58 t/km2.

| Quantity | Model (year; Q1 / Q3) | Field (GOV hauls, box) |
|---|---|---|
| Biomass share above 20 t/km2 | 0.80-0.81; 0.79 / 0.82 | 0 (Q1) / 0.11 (Q3) |
| Biomass share above 50 t/km2 | 0.63-0.64; 0.58-0.60 / 0.69-0.70 | 0 / 0 |
| Densest cell / mean | 141-143; 112-121 / 158-202 | 9.5 / 13.3 (densest haul / survey mean) |
| Densest cell, t/km2 | 223-229 | 10 / 31 |
| Lloyd | 63-66 | 2.7 / 3.5 (2021-25) |

- **The upper tail is the cleanest mismatch.** It holds even after a 4x tow dilution and a 2x catchability error on the field side.

Intake elasticity on own density, biomass-weighted, with ln(accessible bed prey) as the spatial control:

| State | Eaten | Eaten, with control | Pre-cap | Pre-cap, with control |
|---|---|---|---|---|
| Spread to field-like crowding (k ~3) | -0.007 | -0.007 / -0.008 | +0.02 to +0.03 | +0.02 to +0.03 |
| Clumped (k ~30-40) | -0.014 / -0.031 | -0.013 / -0.015 | ~0 to +0.02 | +0.02 / +0.03 |

- **Field comparison:** -0.02 to -0.07 for haddock, -0.2 for large cod and flounder.
- **Model reading:** the model has no crowding cost at field crowding. What it eats falls only through the reserve cap, and its pre-cap demand rises slightly with density, as fish sit where more prey is.

**Label check (`model/scripts/check_eaten.py`, seed 0, 40 ticks).**
- The realised gain after the harvest caps is 1.007 times the post-room-cap demand ("eaten"); the small excess is the prey-reserve correction.
- The elasticity is -0.015 on the realised gain and -0.013 on "eaten".
- So "eaten" is what the fish get.

## What closing the fill gap would do (main agent, section 278.8)

The reserves are set to the month's field `start_fill` in each cell: water column 0.71, whiting and benthivores their monthly values.
- Setup: frozen fix104 u1267, pre-279 library, the cf_lever snapshots, one-day windows (`model/scripts/cf_fill.py`).
- This is a payoff readout only; the policies never saw these fills.
- The table gives means over 8 snapshots. "All" is all-predator M2 on the benthivores per year; the saving of clumping is spread minus clumped. "Room" is the share of pre-cap demand that fits under the reserve-room cap; 1 means not capped.

| Fill set to field | Clumped, all | Saving at Lloyd 3 | Saving at field placement | Water column room in benthivore cells | Benthivore room | Benthivore elasticity, clumped / Lloyd 3 |
|---|---|---|---|---|---|---|
| none (as trained) | 0.148 | 0.249 | 0.431 | 0.45-0.57 | 0.70-0.80 | -0.022 / -0.009 |
| predators | 0.276 | 0.424 | 0.736 | 1.00 | 0.70-0.81 | -0.023 / -0.009 |
| benthivores | 0.157 | 0.255 | 0.444 | 0.49-0.60 | 1.00 | +0.012 / +0.024 |
| both | 0.283 | 0.412 | 0.726 | 1.00 | 1.00 | +0.013 / +0.023 |

- **Hungry predators make clumping pay more, not less.** With the water column and whiting at field fill, the saving at Lloyd 3 rises from 0.25 to 0.42 per year.
- **Hungry predators also overshoot the booking.** All-predator M2 on benthivores at field placement reaches ~1.0 per year, against the booked 0.25.
- **Hungry benthivores show no crowding cost.** Their elasticity turns positive (+0.01 to +0.04): fish in the clumps eat more per tonne, because the clumps sit where bed prey is richer and the bed pairs are saturated.
- **Reading.** Closing the fill gap is a mismatch in its own right, but it is not a lever on clumping; it enlarges the payoff.
- **Consequence for candidate A.** A crowding term on intake at the field size (central -0.07) gives, from k ~3 to k ~40, (40/3)^-0.07 = 0.83, about a 17 % intake loss in the clumps. It would face a saving of ~0.4 per year once the predators are hungry, larger than the ~0.25 that set the -0.05 break-even. So even the central field value would probably not suffice.
