# Krill escape and energy budget: summary (2026-11-10)

Research only, by a research subagent. Nothing in the library, code, tests, manifest, resume or AGENTS.md
was changed. No model output set or selected any value. The trail with quotes is in `evidence.md` (E1-E39,
search log G1-G8); conversions in `scripts/krill_rates.py` -> `tables/krill_rates_stdout.txt`.

Tags on claims: **[M]** measured (a value or observation reported by the source from its own data),
**[S]** secondary (a review, a model, a figure quoted second-hand, or an abstract-only reading of a
measurement), **[I]** my inference or arithmetic.

## 0. How the krill pairs' a was derived (what is already inside it)

All five fish -> krill pairs come from the section 220 encounter form, carried unchanged from
`Krill i Skagerrak/pairs/scripts/krill_pairs.py` through `wc_depth/` to the section 262 re-derivation
(`Krill över grunda bottnar/rederive/`) (E1):

    a_k(c, t) = scan pi p_cap T / 1e6 * sum_z p_k(z) phi(z) * sum_fish n v_rel R(z, t)^2

| term | in a? | value | basis |
|---|---|---|---|
| capture probability per encounter, `p_cap` | **yes** | 0.5 (0.3-0.7), every predator | "krill capture (220 p_krill; tail-flip escape) [J]" (E1 line 284; E2; E3: "J, Langbehn's piscivore pc 0.7") |
| searched share of the visual disc, `scan` | yes | 0.5 (0.25-1) | Varpe & Fiksen 2010 (E1) |
| DVM / vertical overlap | **yes** | day layer `zc`, `Hk` 20-40 m, `ztop_min` 60 m; night migrating share `m_mig` 0.45 (0.35-0.60, Tarling 1999); separate day and night profiles | E1 lines 300-306 |
| light-dependent detection, dark floor | yes | R(z,t) from the light at depth; `R_nv` 0.1 m (0.05-0.2) [J] | E1, E3 |
| krill swimming speed in v_rel | yes | 0.04 m/s | E1 |
| swarming / dilution / confusion | **no** | the dispersed-layer form | "Skagerrak krill occur as layers, not swarms, so the dispersed-layer encounter applies [J]; no school form was run" (E2) |
| avoidance of passing fish schools (Christiansen et al. 2024) | **no** | - | only noted, abstract level, in `Födobegränsning/fish_krill` (E5) |

So a capture-success factor and DVM are both already in a. A literature capture value can only
**replace** p_cap = 0.5 (multiply a by p_new / 0.5), never multiply on top of it. A field-p variant
(krill 0.6 (0.4-0.8), x1.2) was already run as a sensitivity in "Vattenpelarens par i uppmätt ljus" A2
and not applied (E4).

The pairs are saturated at the box krill density, except mackerel's (E6): sat = a h B / (1 + a h B) at
5 t/km2 is 0.97 whiting, 0.95 water column, 0.92 benthivores, 0.84 planktivores, 0.48 mackerel; "Even a
5x lower a keeps sat > 0.8" for whiting and the water column.

## 1. Capture success and escape of euphausiids against fish

**No measured capture probability of any fish on any euphausiid was found** (G1), in light or in
darkness (G4). The library's 0.5 has no euphausiid measurement behind it; what follows brackets it.

- **Closest measurement, a mysid proxy** [M, abstract]: juvenile Australian salmon (*Arripis trutta*), a
  lunging predator, on swarms of the tail-flipping mysid *Paramesopodopsis rufa*: successful attacks were
  39 % of advances when alone and 55 % in groups of six (E14). If the advances not counted as successes or
  approaches are failed attacks, success per attack is 0.62 and 0.73 [I]; the abstract does not say.
  Either reading lies inside the library's 0.3-0.7.
- **Review figure** [S]: "about half of piscivore attacks on average are successful compared with 70-80%
  for planktivorous fishes" (Juanes et al. 2002, E20). Not euphausiid-specific.
- **Escape performance**:
  - *E. superba* tail flip: 57 cm/s (17.3 BL/s) peak, 273 m/s2 [M, one animal, E13]. Kils (1982):
    35-100 cm/s in one flip, ~50 cm/s over at least 50 cm [S, E13].
  - Coherent diving of *M. norvegica* away from herring schools by day: 10-27 cm/s, mean 19 cm/s, the
    layer top dropping 20 m [M, E9].
  - *M. norvegica*'s own tail-flip speed was not found (G3); nor any reaction distance (G2).
- **Swimming of the fish that eat the krill** [M, abstract, E11]: deep-living whiting and Norway pout at
  14-16 cm/s by day and 10-11 cm/s at night; fish in the upper 10-30 m at 16-24 cm/s. Krill cruise at
  ~3.5 cm/s, rising to ~10 cm/s at night after oxygenation [M, abstract, E17].
- **Swarms**:
  - Swarming is "generally presumed" to protect "mainly derived from evasion and dilution factors once an
    attack is launched", but "swarms may also attract and make krill vulnerable" [S, E8].
  - Norwegian Sea swarms: ~200 ind/m3, ~11 t each [M, E8]. The large saithe stacked under them had fish
    and squid in their stomachs, not krill [M, E8].
  - The confusion effect was demonstrated only on mysids: grouped fish attack more successfully (E14), and
    cohesive escape works better against lunging than against ambush predators (E15, E16) [M,
    abstract].
  - No quantified dilution or confusion effect of krill swarms on fish predation was found [I].
- **Light dependence**:
  - Daytime avoidance is strong. Large herring schools (> 10 m high) emptied the krill layer by > 55 %
    (median 68 %) in > 75 % of daytime encounters; schools 5-10 m by a median 12 %; schools < 5 m caused
    no change. At night the density changes "centred around 0" [M, E9]. The voids lasted 5-15 min [M].
  - Stomachs in the Oslofjord [M, E10]:
    - Whiting had krill equally often and in equal numbers day and night.
    - Norway pout had equal krill per stomach, but krill in fewer stomachs at night (44-48 % against
      59-75 %).
    - Herring had krill in clearly fewer stomachs at night (17-20 % against 31-50 %).
    - Stomach fullness and digestion state showed no clear diel pattern.
  - That fits the library's dark ratios (whiting none, planktivores 0.57) and is no reason to change them
    [I].
  - Gadoids swim slower at night, read by the authors as a switch to non-visual feeding [M, interpretation
    S; E11].

## 2. Bioenergetics of *M. norvegica* (and Thysanoessa)

Rates converted to the model's units in `tables/krill_rates_stdout.txt` [I]: DW/WW 0.22 (0.20-0.27), 0.44
J per umol O2, 5000 MJ/t.

| quantity | value | tag | source |
|---|---|---|---|
| energy density | 5.0 kJ/g WW annual mean; 4.3 in spring-summer, ~6 in autumn-winter; Kattegat 20.5 (July) - 26.0 (November) kJ/g DW | M (sibling note) | E28 |
| lipid | Kattegat up to 48 % DW towards winter, drawn down over winter "with loss of lipid, and decrease in weight but not in length"; Balsfjord 30-47 % DW; Kattegat 1.6-9.6 % WW | M / S | E39, E33 |
| respiration (routine, measured in swimming krill) at ambient T | 30-35 umol O2/g DW/h in three populations at their own temperatures = ~75 MJ/t/d (69-93), 19 MJ/t/tick, 1.5 %/d of body energy | M; units I | E21, E33 |
| respiration, box-local fjords | Kosterfjorden Dec-Sep, 5-6 C: 21.9-40.6 umol/g DW/h = 51-94 MJ/t/d. Gullmarsfjord Sep, 6.5 C: 46.9 = 109 MJ/t/d | M (via review table); units I | E33 Table 4.4 |
| Q10 of respiration, acute | 1.55-2.46 per 4 C step; 1.87-2.12 over 4-16 C. Another study: 1.5-2.4 | M | E21, E36 |
| seasonal temperature compensation | no seasonal difference in Clyde or Kattegat krill; equal rates at each population's ambient T; acclimatisation over "days to weeks" | M / S | E27, E21, E33 |
| hypoxia | regulates O2 uptake down to ~30 % saturation (Pc 6-7 kPa at 10 C); survives at most 1 h of anoxia | M (via review) | E33 |
| copepod predation | lab 1.2 Calanus/ind/h (SD 0.86) at 1.2-2 prey/l; ration 12.7 % DW/d (0-33 %); field 0-10 copepods/ind/h and 0-125 % per individual, 1-20 % per station mean; no diel rhythm; digestion 2-3.4 h | M | E22 |
| copepod clearance | 0.6-1.0 l/ind/h = 1.6-4.0 x 10^4 m3/t/tick | I from M | E22, tables/ |
| maximum ingestion, cultured food (type not stated in the record), 10 C | 5.6 % body C/d; maximum clearance 59 ml/ind/h = ~2.4 x 10^3 m3/t/tick | M (one row); units I | E34 |
| clearance contrast | copepods (particulate, 0.6-1.0 l/h) against cultures (59 ml/h): x10-17 | I | - |
| assimilated energy, model of Clyde summer | 25 J/d (males), 35 J/d (females) = 109-152 MJ/t/d at 0.23 g, about 2-3 %/d of body energy | S (model built on measured intake) | E23 |
| diet | omnivore. Copepods: C. finmarchicus 85-95 % of copepod prey mass; Temora and Pseudocalanus in the upper layers. Sedimenting algae cropped at depth in spring, live phytoplankton at night, detritus. Carnivory "40 % or less" on average, lower in the Skagerrak than in N Norway, higher where the copepod:phytoplankton ratio is higher. Daytime visual feeding on overwintering Calanus at a 120 m site | M | E22, E25, E26, E37 |
| Thysanoessa | T. raschii "almost never took copepod prey" (6 % carnivorous); T. inermis in the Skagerrak "seldom fed on copepods"; both are cold-stenotherm, with respiratory capacity topping out at 12 C | M | E22, E36 |
| starvation | **M. norvegica: not found** (G5); starving lowers its O2 uptake (direction only) | S | E33 |
| starvation, T. inermis, 4 C, 28 d | 80 % survival; DW 30.0 -> 20.5 mg (-32 %, 1.35 %/d); lipid 42 -> 28 % DW; no metabolic down-regulation (~6 umol O2/h/g FW = 63 MJ/t/d); lipid lasts ~63 d | M | E35 |
| starvation, E. superba | lipid 21 -> 9 % DW in 19 d; respiration down by up to 70 % in winter or without food | S | E33, E35 |
| growth | modal adults 22 -> 31 mm from early March to early May; little length growth in summer-autumn; ~2 mm shrinkage or size-selective loss over winter | M | E23 |
| P/B | 2 /yr (1.2-3.5) | S (sibling note) | E28 |
| adult mortality (total Z) | 0.003 /d = 1.1 /yr (females 1.46, males 0.73), Clyde, from a cohort ratio in one June | M (a single-ratio estimate) | E23 |
| lifespan | 2-2.5 yr | S | E28 |
| assimilation efficiency | **M. norvegica: not found** (G6). E. pacifica 0.64-0.83 on marine snow; E. superba 0.63-0.86 for algal lipid | S | G6 |
| temperature dependence of ingestion | **not found** (G7). Ingestion as a % of body calories fell with body size and temperature (Artemia food, direction only) | S | E24 |

## 3. Behaviour a learned hide would stand for

- **Day depth and amplitude** [M]:
  - Oslofjord: below ~70 m by day in a 20-40 m layer, below the fish schools (E10, E31).
  - Kattegat: 80-100 m by day (E32).
  - The box geometry is in the sibling notes: day layer 70-150 m, deeper over deep beds.
  - At dusk the krill follow isolumes. In the Oslofjord the ascent is "mainly arrested at ~10-30 m" and
    the krill then scatter through the column (E31).
  - Clyde: the layer reaches the upper layers 30 min after sunset and leaves 30 min before sunrise (E23).
- **Who does not migrate** [M, E32]:
  - At night in the Kattegat, 65 % of non-moulting females and 38 % of non-moulting males were above
    65 m. Almost 90 % of moulting krill stayed below 65 m.
  - So 35-62 % of the non-moulting adults, plus nearly all moulting ones, stay deep. Migration did not
    depend on size.
- **DVM is persistent** [M, E31]: "irrespective of fluctuations in distribution and abundance of food".
- **DVM is light-cued** [M, abstract, E30]: krill at 70-90 m in the Gullmarsfjord responded to a 77 %
  partial eclipse.
- **Swarming** [M]:
  - Oslofjord: coherent group reactions above ~0.2 ind/m3 (E9).
  - Norwegian Sea: swarms of ~200 ind/m3 (E8).
  - The sibling notes judge the Skagerrak krill a layer, not swarms [P, E2].
- **Does DVM or swarming reduce fish predation, quantified?** No field measurement of a reduced krill
  predation rate was found.
  - Tarling (2003) explains the drift of the Clyde sex ratio from 1:1 to ~3:1 males by October with the
    females' shallower, riskier DVM. The sex ratio is [M]; the attribution to visual predation is a model
    whose predator density was fitted [S] (E23).
  - De Robertis (2002) is a model of size-dependent DVM timing, consistent with acoustics [S] (E19).
  - Kaartvedt et al. (1996) is an "antipredation window" interpretation [S] (E18).
  - Christiansen et al. (2024) is the only quantified avoidance: a daytime emptying under large schools
    [M] (E9).
- **Against the hide reading**:
  - Fish "invaded the krill layer at night" (E31) [M].
  - Whiting take krill equally day and night; pout take krill at night too, equal per stomach though in
    fewer stomachs (E10) [M].
  - So DVM shelters the krill from the visual daytime predators (herring schools) but not from the
    gadoids that follow the layer [I].

## 4. Retention and residence (the 18-day exchange)

- **Nothing found moves the sibling note's ~18 d** (e 20 /yr, range 5-50; E38) [I].
- **Fjord stocks are exchange-dominated** [M, abstract]: in a Norwegian fjord, "exchange with coastal
  waters was more important for the standing stock than local growth within the fjords" (E29).
- **Retention by migration is local** [M, abstract, E29]: T. raschii "possibly maintained its horizontal
  position by means of vertical migrations" over a shoaling bottom at ~130 m.
- **The only krill-vs-water ratio is from a sill fjord**: Masfjorden 0.43-0.60. The box's inshore half
  flows north in both layers, so migration there does not retain [S/I, E38].
- **Deep water below the 270 m sill** renews every ~1-3 years (Aure & Dahl 1994 via E38) [S]. That is a
  density renewal, not horizontal flushing, and the krill layer is above it.
- **Not obtained**:
  - Liljebladh & Thomasson 2001 (Gullmarsfjord ADCP) and Kaartvedt 2010 (DVM review) (G8).
  - Any tagging or population-genetic residence estimate for Skagerrak krill.

## Implications

### (a) A capture-success (escape) factor on fish -> krill

**Not supported as a new factor: capture success is already inside a as p_cap = 0.5 (0.3-0.7), and the
gadoid and planktivore pairs are saturated, so a factor on a barely moves their krill intake.**

- **What the literature gives** [M proxy / S]:
  - Lunging fish on tail-flipping crustacean swarms: 0.39-0.55 per advance, 0.62-0.73 per attack (E14).
  - Piscivores ~0.5, planktivores 0.7-0.8 (E20).
  - The bracket is 0.4-0.75, around the library's 0.5. No euphausiid-specific number exists.
- **Per predator type** (all unmeasured on krill):

  | predator type | p range from the literature | replacement factor on a (p / 0.5) | basis |
  |---|---|---|---|
  | planktivores (herring, pout, blue whiting, sprat; particulate on single krill) | 0.4-0.75 | x0.8-1.5 | E14, E20 [S/I] |
  | gadoids (whiting, saithe, cod, haddock) | ~0.4-0.6 | x0.8-1.2 | E20 [S/I] |
  | mackerel | no basis | - | G1 |

- **Light dependence** [M]: the diel stomach pattern (E10) already sits in the library's dark ratios
  (whiting none, planktivores 0.57). There is no capture-in-darkness measurement to add.
- **A mechanism genuinely missing from a** [I]: the daytime avoidance of large schools (E9).
  - It acts on encounter (local density), by day only, and only under schools taller than 5-10 m. That
    means herring and other planktivore schools, and possibly mackerel.
  - It has no effect at night and none on dispersed gadoids.
  - It cannot be made a number without the share of planktivore biomass in large schools by day, which
    nothing in hand gives. It belongs in the planktivores' daytime term, not in p.
- **Effect if applied anyway** [I]:
  - Under shared handling with fixed attack, the krill share of a saturated diet is the encounter share
    s, and a factor x on a_krill turns it into x s / (x s + 1 - s). With s 0.6, x 0.8 gives 0.55 and x
    0.5 gives 0.43; the total ration is unchanged while sat > 0.8.
  - Every replacement factor in the literature range moves a by x0.8-1.5. A 1.8-8x excess cannot come
    from p.
  - Only mackerel (sat 0.48) would respond nearly in proportion, but it has no p data at all.

### (b) Krill as a decision maker: draft parameter table

**First constraint: DVM is already inside every fish -> krill a (the overlap integral and m_mig, E1).**
- A learned hide that stands for DVM would count it twice unless the fish -> krill a's are first
  re-derived without the vertical-overlap term, in a same-layer form. This is the open issue of
  convention 3 in "Attackrater ur mötesteori" for the copepods: "the copepods' daytime refuge is already
  in the overlap, so the pair's visibility floor 0.5 may count it twice".
- A hide standing for swarming or school avoidance (E9) would not double count, but it acts by day and
  on schooling planktivores only.

Keys as the loader reads them; units MJ/t, per 6-h tick unless stated.

| library key | draft value (range) | tag | source | gap |
|---|---|---|---|---|
| is_decision_maker | true | - | - | - |
| energy_content | 5000 (4300 spring-summer, ~6000 autumn-winter) | M | E28 | seasonal E is not an engine feature |
| max_energy_reserve | lipid 1.6-9.6 % WW x 39.5 kJ/g lipid = 630-3800 MJ/t [I]; Kattegat up to 48 % DW in winter | I from M | E33, E39 | the reserve/lean split and the u of starvation onset |
| resting_metabolism | routine 13-27 MJ/t/tick (51-109 MJ/t/d at 5-6.5 C, Kosterfjorden and Gullmarsfjord); central ~19 (30-35 umol/g DW/h acclimatised) | M; units I | E21, E33 | measured on **swimming** krill = routine, not resting; the resting share is unknown |
| feeding_cost, movement_cost | no M. norvegica value; night swimming 3x day speed (E17); starvation lowers O2 uptake (E33) | M direction only | E17, E33 | the fed/fasted ratio |
| metabolism_q10 | acute 1.9-2.1 (1.55-2.46) | M | E21, E36 | seasonal acclimatisation (E27, E33) means the seasonal Q10 is below the acute one; unquantified |
| metabolism_t_ref | the krill layer's temperature (deep layer by day) | I | - | the engine has one layer per group; DVM crosses the thermocline (acute Q10 applies there) |
| ingestion_q10 | none found | - | G7 (E24 direction only) | yes |
| growth_q10 | none found | - | - | yes |
| maintenance_level u, starve_rate | M. norvegica none. T. inermis at 4 C: 80 % alive after 28 d, -1.35 % DW/d, lipid store ~63 d | M (Thysanoessa) | E35 | M. norvegica time to death (G5) |
| appetite | none found | - | - | gate vs reserve_room: no data |
| attack_choice | learned (diet shifts with the copepod:phytoplankton ratio) | M direction | E37, E22 | - |
| movement_speed | 0 (drifts like the zooplankton; the exchange carries advection) | I | E38 | - |
| r_max / derived g | P/B 2 /yr (1.2-3.5); spring growth 22 -> 31 mm in ~2 months | S / M | E28, E23 | r_max at low density not measured |
| natural_mortality (M1) | none separable; adult Z 1.1 /yr (0.73-1.46) includes predation | M (single cohort ratio) | E23 | M1 |
| outside_density, outside_exchange_rate, outside_season | as today: 5 t/km2, 20 /yr, Q1-Q4 1.0 / 0.6 / 1.1 / 1.3 | P | E38, sibling note | - |
| krill_preys_on_zooplankton | clearance 1.6-4.0 x 10^4 m3/t/tick (0.6-1.0 l/ind/h at 1.2-2 Calanus/l); ration 12.7 % DW/d mean, 33 % max (lab) | I from M | E22 | functional-response shape and half-saturation; the km2 form needs the band convention |
| krill_preys_on_phytoplankton, _subsurface_phytoplankton | maximum clearance ~2.4 x 10^3 m3/t/tick, 5.6 % body C/d at 10 C (cultures) | M; units I | E34 | food type of the record; deep cropping of sedimenting algae (E26) has no model counterpart |
| krill_preys_on_microzooplankton | none found | - | - | yes |
| assimilation_factor (per pair) | none for M. norvegica; congeners 0.64-0.86 | S | G6 | yes |
| detritus | eaten (E37) | M | - | no detritus group in the model |

**Feasibility check [I]:**
- The box's 5 t/km2 at a routine 50-110 MJ/t/d, plus production at P/B 2 (~27 MJ/t/d), assimilated at
  0.65-0.85, needs about 100-210 MJ/t/d of food. That is 0.5-1.0 GJ/km2/d, or ~40-85 t/km2/yr at the
  zooplankton's 4500 MJ/t.
- At "40 % or less" carnivory (E22), up to ~16-34 t/km2/yr would come from copepods. That is up to
  1.7-3.5 /yr of new M2 on the 9.72 t/km2 A17-calibrated zooplankton. It is unbooked, and of the order of
  the planktivores' own booked M2 on it.
- Making krill a decision maker therefore also re-opens the zooplankton calibration. As an open pool,
  its local feeding is implicit in the outside density.

### (c) What each option would and would not explain

**Arithmetic of the excess [I].**
- Booked M2 2.11 /yr; realised 4.6-5.1.
- By predator (booked x ratio): planktivores 1.67 x 1.8 ~ 3.0, mackerel 0.10 x 7-8 ~ 0.75, water column
  0.119 x 4 ~ 0.48, whiting 0.187 x 1.4-2.3 ~ 0.26-0.43, benthivores 0.035 x (rising) ~ 0.1. Sum ~4.6-4.8.
- The planktivores carry ~2/3 of the realised flow and ~1.3 of the ~2.6 /yr excess. Mackerel, the
  largest ratio, carries ~0.65 of the excess.

**Caveat on the brief's ratios.** I could not tell whether "realised/booked" is per predator tonne or
total flow. a_revisit measured ~2x per tonne in fix90 while whiting and the water column also grew to
1.6-9.9x start.
- If the ratios are total flow: much of the excess is predator biomass. The lever is then the predators'
  own mortality and growth (a mortality question, as a_revisit concluded).
- If they are per tonne: the lever is ration and diet share. `tools/probes/probe_diets.py` prints both
  (realised ration against field Q/B, realised against booked shares).

**(A) Capture factor.**
- Would explain: at most a 10-25 % trim of the krill share in saturated diets (x0.8 from the literature
  floor), and a near-proportional cut for mackerel. For mackerel no capture data exist, its knee
  position makes a matter, and its booked krill M2 rests on no in-area diet (pairs note, "Not obtained").
- Would not explain: a 1.8x planktivore excess or a 4x water-column excess. Those pairs are saturated,
  and the literature does not put p below the library's range.
- Applying a p below ~0.3 would be a number with no source behind it, chosen to fit the model's output.

**(B) Krill as a decision maker.**
- Would explain: a krill that can lower its visible fraction behaves like the fish prey. Under shared
  handling with fixed attack, the fish would then meet relatively fewer krill whenever other prey are
  visible. If every prey hides, their ration falls (food limitation), which the open pool now masks by
  being the one prey that cannot refuse.
- This addresses the brief's asymmetry: "the fish prey that CAN hide are under-eaten". The krill's
  over-predation and the fish prey's under-predation are two sides of one diet-share budget [I].
- Would not explain or justify:
  - The field evidence (E10, E31) says the gadoids that eat most krill per tonne (whiting, pout) take it
    day and night, and fish invade the layer at night.
  - So a strong learned refuge against them has no field support beyond what the overlap term already
    gives. With the overlap kept in a, the hide counts DVM twice (b).
  - It also needs food pairs and a bioenergetic set with four open gaps: assimilation, starvation,
    ingestion Q10, resting vs routine. And it adds an unbooked M2 of up to ~1.7-3.5 /yr on the
    A17-calibrated zooplankton.
- What it would leave unexplained: the import. With k = 20 /yr the exchange supplies several times the
  stock per year (a_revisit: ~7x, at 0.6-0.7 of the outside density). Whatever the krill does locally,
  the box's krill flow is set mostly by the outside density and k. The evidence in section 4 does not
  lower k [I].

**Neither option addresses two candidates the evidence points to [I].**
1. The predators' ration and diet bookings: whether realised rations exceed field Q/B under
   `reserve_room`.
2. The fish prey's hide being stronger than their field refuge. A field side for it exists only for
   herring schools (schooled by day, dispersed at night; E9, E31).

Both are checkable with probe_diets without any new parameter.

## Gaps (summary)

1. Fish -> euphausiid capture probability, per encounter or per attack, in any light (G1, G4).
2. M. norvegica reaction distance and tail-flip speed (G2, G3).
3. M. norvegica starvation survival and the resting (vs routine) metabolism (G5).
4. Assimilation efficiency and the ingestion Q10 (G6, G7).
5. A functional response of krill on copepods or phytoplankton with a half-saturation.
6. The share of planktivore biomass in large daytime schools (needed for E9).
7. The four Advances in Marine Biology 57 chapters (feeding, DVM, population dynamics, predation) and
   Jager & Ravagnan's DEB parameters were not available open (G8).

## Main-agent check: the excess per predator tonne (fix96, probe 40 at update 390, 3 probe years)

`python3 tools/probes/probe_diets.py results/pzbpgp248nsrpfix96`. The ratios in the brief were the
prey-side M2 per pair (tonnes eaten / mean krill biomass), i.e. total flow. Split per predator tonne:

| predator | realised ration / field Q/B | krill diet share realised / booked | krill M2 realised / booked (flow) |
|---|---|---|---|
| planktivores | 6.75-7.43 / 5 = 1.35-1.49 | 0.47-0.51 / 0.40 = 1.19-1.27 | 1.73-1.86 |
| mackerel | 6.22-6.88 / 2.2 = 2.8-3.1 | 0.58-0.67 / 0.39 = 1.48-1.73 | 7.4-8.4 |
| water column | 3.82-3.88 / 2.83 = 1.35-1.37 | 0.81-0.87 / 0.36 = 2.24-2.42 | 3.65-4.09 |
| whiting | 3.41-3.87 / 4.2 = 0.81-0.92 | 0.59-0.77 / 0.56 = 1.06-1.39 | 1.44-2.30 |
| benthivores | 5.78-6.53 / 4.46 = 1.30-1.46 | 0.018-0.111 / 0.027 = 0.67-4.18 | 1.66-13.2 |

So the excess is mostly per tonne: rations above field Q/B (every fish group but whiting, mackerel
x3) and, for the water column and mackerel, a krill-heavy diet; predator biomass adds to it for whiting
(1.2-1.6 x start in the probe). Caveat: mackerel's field Q/B 2.2 is an annual stock value, while the box
holds it mainly in its feeding season, so part of its x3 is the comparison, not the model [I]. The water
column's krill share comes with its fish prey under-eaten (planktivores 0.17-0.27, benthivores 0.22-0.31
of booked share), which is the asymmetry of hiding fish prey against a krill pool that cannot hide.

## Main-agent check 2: do the fish prey hide more than the field supports? (test 1)

`diagnostics/scripts/hide_pattern.py`, table `diagnostics/tables/hide_pattern_pzbpgp248nsrpfix96.json`:
frozen fix96 policies, the probe world (30 x 30, seed 20260530, unfished, the run's currents), two
years. Biomass-weighted rest (= hide) share per DM and month, split by tick of day (dark = the two
6-h ticks around midnight, bright = the two around noon; a light threshold cannot be used at 6 h,
since no tick in Mar-Sep has a mean light below 0.01).

Year 2, range over the months (dark / bright):

| DM | rest share |
|---|---|
| planktivores | 0.05-0.16 / 0.08-0.20 (highest May-Oct; 0.01-0.05 more by day) |
| whiting | 0.02-0.09 / 0.02-0.08 |
| benthivores | 0.01-0.04 / 0.02-0.04 |
| mackerel | 0.00-0.02 / 0.00-0.02 |
| water column | 0.00-0.01 / 0.00-0.01 |
| zooplankton | 0.01-0.80 / 0.02-0.81 (seasonal, May-Sep 0.7-0.8) |

**Verdict: the fish prey hardly hide**, so their hide action does not explain the water
column's under-eating of fish prey. The policy-free diet gate (`tools/probes/budget_gate.py`,
run 2026-11-10) already gives the water column's realised diet at the start densities with every
prey fully visible except for stage access: krill 0.87 of the encounter vs booked 0.36
(field spread 0.81), planktivores 0.105 vs 0.498, benthivores 0.014 vs 0.077. The probe realised
0.81-0.87 krill. The skew is in the pairs (the krill a against the fish a's and stage access),
the known open item of resume 252.3-252.5 ("Vattenpelarens krill mot fisk"), where the box-light
measurement of 253 went against the dark-light reading that would have cut the krill a.
A learned krill hide would act on this, but the evidence above gives no refuge against the
gadoids beyond the DVM overlap already in a.

## Main-agent check 3: why the fish rations exceed the field Q/B (test 2)

`diagnostics/scripts/ration_budget.py`, table `diagnostics/tables/ration_budget_pzbpgp248nsrpfix96.json`:
frozen fix96 policies, the probe world as in check 2, two years; year 2 per tonne-year. I =
assimilated intake net of SDA (what `movement.income_growth` receives), C = action costs, G =
energy paid into new biomass, rest = I - C - G (mainly the reserve that leaves with lost,
moved or emigrating biomass and has to be refilled), cut = share of the demand's energy removed
by the `reserve_room` cap.

| DM | ration | fill | cut | C split rest/eat/move | realised P/B (library P) | I | C | G | rest |
|---|---|---|---|---|---|---|---|---|---|
| planktivores | 6.59 | 0.99 | 0.51 | 0.08/0.81/0.11 | 0.62 (0.66) | 21585 | 15490 | 2468 | 3626 |
| benthivores | 6.36 | 0.99 | 0.31 | 0.02/0.93/0.05 | 0.73 (0.74) | 12512 | 5802 | 5325 | 1386 |
| whiting | 3.64 | 0.99 | 0.55 | 0.04/0.94/0.02 | 0.60 (0.61) | 12166 | 8365 | 2358 | 1443 |
| water column | 3.43 | 0.99 | 0.47 | 0.00/0.95/0.05 | 0.41 (0.42) | 14130 | 10805 | 1710 | 1616 |
| mackerel | 6.15 | 0.90 | 0.48 | 0.01/0.93/0.07 | 0.35 (-) | 20902 | 7862 | 1626 | 11414 |

Each term as a share of the field-implied intake, field Q/B x (I / ration) (the realised energy
per tonne eaten) [D]:

| DM | field Q/B | C | G | rest | sum = ration / field Q/B |
|---|---|---|---|---|---|
| planktivores | 5.00 | 0.95 | 0.15 | 0.22 | 1.32 |
| benthivores | 4.46 | 0.66 | 0.61 | 0.16 | 1.43 |
| whiting | 4.20 | 0.60 | 0.17 | 0.10 | 0.87 |
| water column | 2.83 | 0.93 | 0.15 | 0.14 | 1.21 |
| mackerel | 2.20 (annual stock value) | 1.05 | 0.22 | 1.53 | 2.8 |

Findings [D, I]:
- Every resident fish group sits at fill 0.97-1.00 all year and the reserve_room cap removes
  30-56 % of the energy it would otherwise take. Food never limits them, so the ration is set by
  what they spend: C + G + the refill of the reserve that leaves with lost biomass at fill 1.
  The field fills cycle (planktivore start_fill / outside_fill 0.20-0.96 over the year).
- Growth is not the excess: realised somatic P/B equals the library's `somatic_production`.
- Costs are: the fish eat in 81-95 % of their biomass-ticks, so C is almost all eat cost
  (RM x feeding_cost x Q10). For the planktivores and the water column C alone uses 0.93-0.95 of
  the field-implied intake, leaving nothing for growth and refill. For the benthivores growth
  takes 0.61 (growth_cost c 0.69 prices a tonne at ec + c (ec + s ME) = 7326 MJ at fill 1).
- Mackerel: 55 % of its intake is the "rest" term - immigrants arrive at the month's fill and are
  topped up (fill 0.58 in May -> 1.00 from August) and emigrants leave full. Its field Q/B is an
  annual stock value against a box that holds it mainly while it feeds, so its x2.8 is not a
  like-for-like comparison.
- So the library's per-tonne energy demand at a full reserve (costs + growth + refill) is
  larger than the field Q/B supplies for planktivores, water column and benthivores. The
  appetite rule only exposes it: with surplus food the fish eat to their budget. Two readings,
  not separated here: (1) the food supply is above the field's (the field fish are food-limited
  part of the year, the model's never; the krill pool and the summer zooplankton above A17 are
  candidates), (2) the cost and growth parameters do not close with the field Q/B even in
  principle (a policy-free closure check per group would decide it).

## Main-agent check 4: food surplus or parameters? (tests of the two readings)

### 4a. Policy-free closure (`diagnostics/scripts/closure.py`, `tables/closure.json`)

Library and manifest only. Field-implied intake I = field Q/B x q (q = assimilated energy per
tonne eaten over the BOOKED diet, net of SDA); costs = RM (annual-mean Q10) x (e feeding_cost +
(1 - e) resting_cost); G = somatic P x tonne price; refill = (M1 + booked M2 on the group + adult
emigration) x s ME. e* = the eating share of biomass-time at which the budget closes.

| DM | field Q/B | I field | RM | feeding_cost | G (s field / 1) | refill (s field / 1) | e* (s field / s 1) | e* at the realised diet's q | realised eating share |
|---|---|---|---|---|---|---|---|---|---|
| benthivores | 4.46 | 9800 | 4862 | 1.18 | 5068 / 5421 | 448 / 728 | **-0.66 / -1.38** | -2.24 | ~0.95 |
| planktivores | 5.00 | 16211 | 9737 | 1.65 | 2640 | 1143 / 1817 | **0.43 / 0.32** | 0.35 | ~0.8 |
| water column | 2.83 | 11047 | 4803 | 2.14 | 1756 | 8 / 11 | 0.82 / 0.82 | 0.60 | ~0.95 |
| whiting | 4.20 | 14153 | 5650 | 1.50 | 2379 | 748 / 1342 | 1.90 / 1.69 | 1.76 | ~0.95 |
| mackerel | 2.20 | 7226 | 6701 | 1.02 | 0 | 98 / 147 | 3.18 / 2.81 | 3.21 | ~0.95 |

- Benthivores: the budget cannot close at the field Q/B even resting all the time: resting
  metabolism (4862) + growth at the c 0.69 tonne price (5068-5421) already exceeds the field
  intake (9800). The 223.9 Q/B (4.46) and the 241 growth cost do not fit together [D].
- Planktivores: closes only if they eat in at most 32-43 % of their biomass-time; the trained
  fish eat ~80 %. Water column: at most 60-82 % (the krill-heavy realised diet lowers q 3903 ->
  3484 and so needs 12 % more tonnes for the same energy); trained ~95 %. Whiting and mackerel
  close with time to spare.

### 4b. Food arms (`diagnostics/scripts/ration_budget.py` ARM=nokrill|halfkrill; `tables/ration_budget_*`)

Frozen fix96 policies, the probe world, two years; the krill pool set to 0, or halved, before
every tick (diagnostic arms, no field basis for either factor). Year 2:

| DM | arm | fill (annual; monthly range) | cap cut | ration | P/B (field P) | mean biomass vs base |
|---|---|---|---|---|---|---|
| planktivores | base | 0.99 (0.98-1.00) | 0.51 | 6.59 | 0.62 (0.66) | 1 |
| | half krill | 0.68 (0.16-0.99) | 0.12 | 5.74 | 0.23 | 0.63 |
| | no krill | 0.25 (0.09-0.39) | 0.02 | 3.05 | 0.04 | 0.53 |
| water column | base | 0.99 | 0.47 | 3.43 | 0.41 (0.42) | 1 |
| | half krill | 0.49 (0.47-0.53) | 0.13 | 2.02 | 0.11 | 0.91 |
| | no krill | 0.44 (0.37-0.61) | 0.11 | 1.56 | 0.10 | 0.88 |
| whiting | base | 0.99 | 0.55 | 3.64 | 0.60 (0.61) | 1 |
| | half krill | 0.89 (0.79-0.99) | 0.22 | 3.26 | 0.41 | 0.77 |
| | no krill | 0.50 (0.11-0.78) | 0.01 | 2.21 | 0.14 | 0.54 |
| benthivores | base / half / none | 0.99 in every arm | 0.23-0.31 | 6.36-6.98 | 0.72-0.73 | 0.65-0.69 (predators' diet shift) |
| mackerel | base / half / none | 0.90 / 0.68 / 0.65 | 0.48 / 0.06 / 0.02 | 6.15 / 4.51 / 4.31 | 0.35 / 0.16 / 0.13 | ~1 (exchange) |

Verdict [D, I]:
- The pelagic fish's surplus is the krill. Halving the pool turns the planktivores' fill into a
  seasonal cycle like the field's (0.16-0.99 against start_fill 0.20-0.96) and brings the ration
  to 5.74 against the field 5.0; the water column falls below its field ration. The krill pool
  is a near-inexhaustible supply: it relaxes towards 5 t/km2 at k 20 /yr, so it can deliver
  ~k x T ~ 100 t/km2/yr, while the booked flow is its field P (M2 2.11 x 5 ~ 10).
- But at field-like food the frozen policies cannot pay for field growth: planktivore P/B
  0.23 against 0.66, water column 0.11 against 0.42, and the planktivores lose a third of their
  biomass in two years. That is the closure in 4a: they forage far more of the time than the
  budget allows. In the surplus world foraging time is free (at a full reserve the costs are
  refunded by food the cap lets through), so ARS has had no reason to economise; whether a
  policy trained with less food would forage <= 0.3-0.4 of the time is not answered by frozen
  policies.
- Benthivores: food is not the issue (the beds saturate them with or without krill); the
  parameters cannot close at their field Q/B (4a).
- Whiting: consistent in both tests; mackerel: the excess is the exchange (immigrants topped up).

## Correction 2026-11-11: the M1 unit in closure.py

`closure.py` read `natural_mortality` (per tick in the library) as a per-year rate, so M1 (0.16-0.29 /yr)
was missing from the refill (found by `Planktivorernas energibudget`). Fixed and re-run
(`tables/closure.json`). Corrected e* (field fill / fill 1): planktivores 0.29 / 0.10 (was 0.43 / 0.32),
water column 0.75 / 0.72, whiting 1.80 / 1.52. For the benthivores e* is no longer informative since
feeding_cost is 1.02 (section 267: eat and rest cost almost the same); use the Q/B needed in
`Bottenfiskens budget`. Mackerel's row compares with the annual stock Q/B and is not meaningful
(`Makrillens ranson`). The planktivores' failure to close is the comparison Q/B 5.0, not their costs
(`Planktivorernas energibudget`).
