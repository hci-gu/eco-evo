# Satiation in the demersal piscivores: why field saithe, whiting and hake are not full (2026-11-20)

Research subagent; saved by the main agent. Nothing in the library, manifest, code, tests, AGENTS.md or the resume
was changed. No value is taken from model output; the fix104 feeding levels (resume 278.4) are the yardstick only.

Files:
- `evidence.md`: the trail, M0-M7.
- `lit_foraging_limits.md`, `lit_costs_sinks.md`, `lit_fill_series.md`: three literature subagents.
- `tables/mechanisms.csv`: the sized summary. Raw tables: `tables/foraging_limits_raw.csv`,
  `costs_sinks_raw.csv`, `fill_series_raw.csv`.
- `scripts/model_side.py`, `scripts/mechanism_sizes.py`: the arithmetic, with `tables/*_stdout.txt`.
- `data/`: downloads, untrusted, do not commit.
- `model/`: the main agent's model part (resume 289: fill_decomp.py on frozen fix105, band_plateaus.py). It was
  not used as evidence in this note.

Tags: [F] full text, [A] abstract, [S] secondary, [D] derived, [J] judgement.

## 1. The question in one currency [D]

1. **The model already eats the field ration.**
   - Whiting realised 0.38 of the model ceiling (125 kJ/kg/d), which is 48 kJ/kg/d. The field SMS ration at
     120-250 g is 41-52, i.e. 0.33-0.41 of the ceiling.
   - The field "0.25 of Cmax" is against the lab Wmax, which is 1.1-1.34 model ceilings.
   - The realised ration is exactly cost + field P at fill 1: whiting 0.388, water column 0.435.
2. **The difference is the potential.** Uncapped, the model fish would eat 0.66 (whiting) and 0.73 (water
   column) of the ceiling, ~1.7x the need.
3. **Under the model's growth law the fill is a knife edge in the potential.** Fill 0.1 -> 1.0 needs a feeding
   level of 0.275 -> 0.388 (whiting) and 0.350 -> 0.435 (water column).
4. **Refill speed.** The model's surplus (~6 MJ/t/tick net) refills whiting's 0.43 swing in ~36 days; the field
   needs 3-6 months. So the field potential in the refill season is at most ~0.44-0.55 (whiting) and ~0.47-0.62
   (water column), while somatic growth also peaks.

## 2. Per mechanism

1. **Foraging time, risk, other activities: weak, not sizable for adults.**
   - Diel rest and vertical migration exist: saithe are less active at night and in winter [A]; hake migrate
     vertically [A]. There is no adult foraging-time share.
   - With gut buffering, a daily window alone gives >= 0.6-0.9 of Cmax.
   - Proxies: a predator cue cut juvenile cod intake to 0.22-0.44 [S]. North Sea Ecosim vulnerabilities
     (whiting 3, saithe 6, hake 3) cannot be converted to Cmax [F].
   - Model: a fish's rest has no payoff (fish floors 1.0); a cap on T would be `predator_share`.
2. **Per-prey-group plateaus: supported as a pattern, sized in one study.**
   - Rindorf & Gislason 2005 [F]: fitted, density-independent plateaus per fish-prey group: clupeids 0.12-0.20,
     gadoids 0.07-0.14 on the model scale.
   - "Other food" (0.10-0.26) is an observed range, not a fitted plateau, so a krill-alone bound (<= ~0.26) is [J].
   - With three groups saturated and additive: 0.29-0.61.
   - Consistent: negative prey switching (Rindorf et al. 2006 [A]), saturating diet shares (Kempf 2008 [F]). No
     field evidence that krill fills a gadoid.
   - Model lacks it: shared handling lets one abundant prey approach the ceiling.
3. **Digestion, meals, lipostasis.**
   - Meal dynamics set the empty shares (hake 0.47-0.54 [A], whiting Q1 0.45), not the mean.
   - Appetite capped by the reserve is contradicted: farm-associated saithe have stomach content x2.35 and HSI
     x1.4-1.8; the wild fish eat 0.43-0.65 of the farm-fed intake (Dempster 2011 [F]). Farmed cod livers reach
     >12 % against 2-6 % wild [S].
   - The model has the digestion-limited form already.
4. **Patchiness: weak.** Cod and capelin anti-correlate below ~3.5 km [A]. At most 0.05 of intake while pairs
   saturate [D]. Inert in the model while saturated.
5. **Seasonal sinks: small as sinks, large as a seasonal ration.**
   - Saithe spawning migration: 0-2 % (at most 3 %) of the ration, outside the box [D]. Warming is already in the
     Q10. The 279 whiting cut is at most 0.028 of fill.
   - The SMS seasonal ration, taken as supply-limited, gives a whiting reserve swing of 0.31-0.60 ME (minimum
     ~1 Jul, maximum Oct-Jan) against the field 0.43 (May-Jul to Sep-Nov).
   - The saithe (adult stock) Q1 dip is ~11 % of the annual ration.
   - Model: the ration follows the reserve room, not the food, so the cycle is absent.
6. **Costs: no.** Field metabolic rate / model cost: saithe 0.93 (0.6-1.5), whiting 0.78 (0.60-1.55), from 5
   otolith fish each [D]. The model's cost plus SDA is at or above field values.
7. **Food acquisition (Dempster natural experiment): the strongest single datum.** Field reserves are limited by
   getting food. It does not separate 1, 2 and 4.

## 3. Overall answer [J]

**Why field fish are not full.** Their potential intake is close to their need. They are limited by acquiring
food (capture and encounter while foraging, and the season), not by appetite, cost or sinks. Two independent
supports:
- the farm experiment;
- whiting's measured seasonal ration produces its measured reserve cycle.

**What the model lacks.** A low intake rate while foraging that one abundant prey cannot push to the shared
ceiling.
- Per-prey plateaus are the only field-sized form.
- The field totals (0.17-0.42) do not discriminate, because any field total equals the need. The support is the
  plateaus' independence of prey density and of the other prey.
- It is not a guarantee: all groups saturated sum to 0.61.

**The knife-edge caveat.** The fill is a knife edge in the potential, so the plateau values decide between
starvation and a full liver within a few hundredths. The data are 15 points from one study (early September,
five stations, whiting 20-35 cm).

**Seasonal windows.** A seasonal feeding window (Q1 depression) is supported, but it needs a cause, not a season
switch.

## 4. Field fill series

Library convention: fill = HSI x lipid share x 39.5 MJ/kg / 2000 MJ/t.

**Saithe** (no monthly wild HSI anywhere):

| Source | Fill |
|---|---|
| NIFES Norwegian Sea, Q1 | 0.63 |
| NIFES Norwegian Sea, Apr-Jun | 0.67 |
| NIFES Norwegian Sea, Sep | 1.02 |
| Box size class (43 cm NIFES Oct-Dec; 41-43 cm Mathers Aug) | 0.47-0.71 |
| Coastal wild controls | 0.62-0.81 |
| Farm-fed | 1.09 |

- Mathers 1992 (northern North Sea, box sizes): condition lowest in April, highest in August (ratio 0.77), from
  winter fasting.
- Shape [J]: minimum Mar-Apr, maximum Aug-Sep.
- Immatures: unresolved.
- The library's flat 0.71 is about the mean of the coastal and box-size values.

**Whiting** (Black Sea monthly HSI; lipid share 0.5-0.6 [J]):

| Months | Fill |
|---|---|
| May-Aug | 0.27-0.32 |
| Sep-Oct | 0.40-0.49 |
| Nov-Apr | 0.59-0.71 |

- The minimum tracks local spawning: Iceland in August, W Scotland in Apr-Jun.
- The haddock proxy's 0.30 -> 0.73 is within range.
- No North Sea or Skagerrak monthly series exists; Lauerburg 2016 confounds year with season.

**Hake:** Portugal, female HSI 3.2 in March, 5.8 in July; immatures follow the same pattern (Costa 2013 [F]).

## 5. Load-bearing sources

1. Rindorf & Gislason 2005, doi:10.1016/j.jembe.2005.04.013 [F].
2. Dempster et al. 2011, doi:10.1371/journal.pone.0015646 [F].
3. ICES WGSAM 2024, Table 3-8 [F].
4. Erdoğan Sağlam & Sağlam 2012, doi:10.4194/1303-2712-v12_4_11 [F]; Mathers et al. 1992,
   doi:10.3354/meps086031 [F].
5. Alewijnse 2022 otolith field-metabolism dataset, doi:10.6084/m9.figshare.22110335 (weak).

## 6. Settling datum

The intake of whiting or saithe on krill alone at high krill density, as a fraction of Wmax.
- A plateau of ~0.1-0.25 makes per-prey plateaus the mechanism.
- If krill fills them to Cmax, the limit is time or risk, which has no adult data.

Second: a length-standardised monthly HSI for immature saithe and for whiting in the Skagerrak.

## 7. Gaps

Full texts not obtained:
- Andersen & Riis-Vestergaard 2004;
- Steinhausen 2005;
- Bergstad 1991;
- Bodiguel 2009;
- Fernandez-Jover 2011 (open-access copy located, not read).

Missing data:
- adult predation mortality on these species;
- adult food-for-safety experiments;
- hake field metabolic rate;
- Skagerrak Ecosim vulnerabilities.
