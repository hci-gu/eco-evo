# Benthos accessibility for the gadoids: accessible fraction, benthivore share, attack rate

Purpose: the `gadoids -> benthic_community` pair is saturated today (half-saturation 1/(a h) = 0.29
t/km2 against a stock of ~90 t WW/km2), and the pooled gadoids become ~80 % benthos eaters. This note
asks which measurable biology should set the pair: how much of the prey-sized stock fish can reach, how
much of the pooled gadoid biomass can feed on it, and how fast. It follows `../Bentos och torskfiskar/
benthos_for_gadoids.md` (K, r, start biomass), which it does not repeat.

Principle: parameters come from mechanisms; diet is an emergent outcome and is used only as a check.
Tags: **[F]** full text read (incl. values read off a figure), **[A]** abstract/snippet, **[S]**
secondary. Untagged arithmetic and judgement are mine. Quotes and URLs are in the evidence files:
- `evidence_A_accessibility.md`: vertical distribution, burial depths, exposed parts (Q1).
- `evidence_B_gadoid_foraging.md`: morphology, rations, Cmax, swept-area attack rate (Q2).
- `evidence_C_recent_diets.md`: in-area diets by mass for validation (Q3).

## Bottom line

| Parameter | Now | Recommended (min / central / max) | Basis | Confidence |
|---|---|---|---|---|
| Accessible fraction f of the prey-sized stock (new per-pair parameter) | 1 (implicit) | **0.08 / 0.20 / 0.45** | burial depth + exposed arm tips/siphons | medium at 300 m (one measured profile), low at 100 m (judgement) |
| Benthivore share phi of pooled gadoid biomass (scales the pair's response) | 1 (implicit) | **0.54-0.57 / 0.60 / 0.68** (0.30-0.40 if Norway pout is ~1/3-1/2 of the group) | mouth position, jaws, barbel, habitat; no diet | medium-high for the ranking, medium for the cod and whiting scores |
| Attack rate a, per tonne of benthivore (km2 t^-1 per 6 h) | 0.04 (pooled, on total B) | **0.007 / 0.21 / 5.8** | swept area: speed x width x time x capture | low: two of four inputs unmeasured; a range, not a number |
| Validation band, benthos share of pooled gadoid diet by mass | model ~0.8 | **0.35-0.50** | in-area data 1984-91 (no post-2000 by-mass data exist) | medium |

**The structural finding matters more than any of the numbers.** Intake on the pair is
a f B / (1 + a h f B). The attack rate and the accessible fraction enter only as the product a f.
Under every defensible combination the pair stays saturated (section 1). So **neither f nor a is
the lever for the benthos share**. The levers are the ceiling of the pair (phi, section 3) and how
profitable benthos is relative to herring and krill. f is still a real biological parameter. It
does not move today's diet, but it governs local depletion and the interplay of trawling and
predation, and it matters in any low-stock scenario.

## 1. Saturation arithmetic (why a and f do not move the diet)

x = a h f B, the fraction of the ceiling reached is x/(1+x). B = 90 t/km2.

| Case | a | h (ticks) | f | x | Fraction of ceiling |
|---|---|---|---|---|---|
| Today's model | 0.04 | 87 | 1 | 313 | 0.997 |
| Model a, central f | 0.04 | 87 | 0.20 | 63 | 0.984 |
| Mechanistic central | 0.21 | 87 | 0.20 | 329 | 0.997 |
| Worst corner (a min, Cmax-derived h, f min) | 0.007 | 154 | 0.08 | 7.8 | 0.886 |

- The half-saturation in total benthos, 1/(a h f), is 1.4 t/km2 at a 0.04 and f 0.2. It is
  ~12 t/km2 at the worst corner. Both are well below the 90 t/km2 stock.
- The phi scaling in section 3 (a x phi, h / phi) leaves x unchanged, so it does not change this
  table.
- Harvest cap: the group's benthos demand at the ceiling is 0.69 x 1/87 = 0.008 t/km2 per tick.
  The accessible stock is f B = 18 t/km2, so the cap does not bind either.
- Benthos would leave saturation only below f ~0.003 at the model's a, or ~0.0006 at the central a.
  No burial or exposure datum is that low.

## 2. Accessible fraction f (Q1)

**Measured (evidence A, Tables 1-3):**
- **Vertical biomass profile, Skagerrak SK, 280 m mud** (Dauwe et al. 1998, digitized from Fig. 2)
  [F, +-5 pp]: **27 % of AFDW in 0-2 cm, 51 % in 0-5 cm**, 30 % in 5-10 cm, 19 % in 10-20 cm.
  - Brackets: German Bight mud 34 / 65 % and Frisian Front 13 / 33 % (0-2 / 0-5 cm) [F].
  - No 0-1 cm biomass slice exists in these muds.
- **Taxa at SK** [F]:
  - Abra nitida and Thyasira flexuosa are confined to 0-2 cm.
  - Nucula peaks at 5-10 cm.
  - Polychaetes make up 70-80 % of the biomass at 10-20 cm.
- **Other burial depths:** thyasirids burrow 12-64 mm deep [F]. The Amphiura disc sits at 4-6 cm,
  up to 8 cm [S]. Heteromastus lives head-down to ~15 cm [S].
- **Exposed parts:**
  - Amphiura filiformis raises ~2 of 5 arms, 10-20 mm above the sediment, and few in still water
    (Loo et al. 1996) [F].
  - Abra nitida holds its siphon > 0.5 cm above the surface. It is active 25 % of the time at 10 C
    and 55 % at 15 C (Gremare et al. 2004) [F]; the bottom here is ~6 C, so expect less.
  - Cirratulid palps roam the surface while the body stays in the burrow [S].
- **Cropping:**
  - Regenerating tissue is 12-30 %, mean 22 %, of adult A. filiformis biomass, and 13.3 % of
    production goes to arm regeneration, ~0.06 x B per year (Skold et al. 1994, Skagerrak) [F].
  - Dab eat ~6 % of the arm population per year, and only 4 % of Amphiura have intact arms
    (Duineveld & Van Noort 1986) [A].
- **The only direct "harvestable fraction" found is a wader analogue:** 89 % of the prey biomass was
  not harvestable by knot, being too large, too small or too deep (Zwarts & Wanink 1993) [F].
  - No subtidal fish study separates total from fish-available benthos.
  - Hiddink et al. 2016 define availability as prey per fish, with no depth refuge [A].

**Range** (evidence A, "Reasoned range"). The reach depth is judgement: 0-1 cm min, 0-2 cm central,
0-5 cm max (haddock digging). The taxon weights are from the in-grid composition.

| Depth | min | central | max | Driver |
|---|---|---|---|---|
| 100 m (Amphiura ~50 % of edible mass) | 0.06 | 0.18 | 0.38 | Amphiura only 0.02 / 0.06 / 0.15 (arm tips) |
| 300 m (small polychaetes, bivalves) | 0.10 | 0.26 | 0.48 | matches SK's 27 % in 0-2 cm |
| **Box-wide** | **0.08** | **0.20** | **0.45** | weighted slightly to the 100-200 m Amphiura ground |

**Model form.**
- The visibility floor cannot carry f. For an NDM prey the hidden fraction is 0
  (`interactions.visible_prey_biomass` hides only biomass a DM has put in Rest), so benthos is
  always fully visible.
- f needs a new per-pair parameter, e.g. `accessible_fraction`, that multiplies the prey biomass in
  both the Holling term and the harvest cap.
- Two design choices lie outside this review:
  - Does the policy observe f B or B? Either way the checkpoints become invalid.
  - The form assumes instant mixing: the exposed or shallow fraction refills from the pool within a
    tick. The alternative is a two-compartment refuge.
- Do not add regrowth of arm tips and siphons on top of the logistic. It is already part of the
  measured production behind r_max.

## 3. Benthivore share phi (Q2, morphology only)

Classification uses mouth position, jaw relation, barbel and taste-bud equipment and demersal
habitat (FishBase morphology pages, FAO 1990 gadiform catalogue [F]; Nagabhushanam 1965 and Harvey
& Batty 2002 [A]). Trait databases (Beukhof et al. 2019 `feeding.mode`) and guilds are built from
FishBase diet tables [F], so they are circular here and serve only as a cross-check. They give the
same ranking.

| Species (IBTS share) | Morphology / habitat | Score |
|---|---|---|
| Haddock (0.54) | "sub-terminal/inferior" mouth, lower jaw shorter; demersal, "more common from 80 to 200 m" | 1.0 |
| Cod (0.06) | terminal mouth but "chin barbel well developed", taste-bud barbel of infauna feeders; 150-200 m | 0.5-1.0 |
| Whiting (0.27) | terminal mouth, barbel "absent or small"; 30-100 m, deep edge at 200 m | 0-0.3 |
| Saithe (0.12), hake (0.01) | lower jaw longer / projecting; pelagic or piscivore build | 0 |
| Norway pout (lumped) | "superior" mouth, "a pelagic feeder" | 0-0.2 |

- **phi = 0.57 / 0.60 / 0.68** (min / central / max); 0.54 if cod is scored 0.
- Lumping in Norway pout at 0.5-1x the biomass of the other species gives 0.27-0.52, central
  0.30-0.40. Flatfish would raise phi.
- Below ~200 m the mix shifts toward saithe and Norway pout (Bergstad 1990) [A], so phi is lower on
  the deep part of the box.

**Model form (mean field).**
- A pooled tonne that is phi benthivores and 1-phi non-benthivores eats benthos at phi x the
  haddock response: f_pair(B) = phi a f B / (1 + a h f B).
- In parameters: **a_pair = phi a and h_pair = h / phi**. At saturation this is the "ceiling scaled
  by phi" asked for, and a stays per tonne of benthivore, comparable with section 4.
- At phi 0.6, h_pair = 145 ticks.
- Caveat: phi caps benthos *intake*, not diet *share*. A policy can still put its whole eat effort
  on benthos and eat 100 % benthos at a reduced intake. The hard-bound alternative is an eat-share
  cap e_benthos <= phi, which is cruder and has a worse gradient.

## 4. Mechanistic attack rate (Q2)

a = v w T p_f p_c / W, the area cleared per tonne of fish per 6-h tick (evidence B, section 3a).
- Inputs: 35 cm haddock, 0.39 kg (FishBase L-W [F]). Search speed 0.10-0.35 m/s, after cod in situ
  "below 1.0 bl/s ... most common" [A]. Search width 0.05-0.30 m (assumption). Time searching
  0.25-1 (assumption). Capture probability 0.1-1 (assumption).
- **Result: a = 0.007 / 0.21 / 5.8 km2 t^-1 tick^-1.** Per pooled tonne (x phi 0.6) that is
  0.004 / 0.12 / 3.5.
- No reaction distance, capture probability, clearance rate or per-item handling time has been
  measured for any gadoid (or flatfish) on infauna.
- The model's 0.04 (on total B) is close to 0.21 x f 0.2 = 0.042. That is a coincidence, not a
  validation.
- **Verdict:** a is supportable only as a three-orders-of-magnitude range. Because a f is not
  identifiable from intake or diet, keep 0.04 and treat it as fit-only. By section 1 it hardly
  matters while the stock is ~90.
- **In the library:** keep `max_intake_rate` 0.04 and h 87 as the per-benthivore response, and add
  f (0.2) and phi (0.6) as separate multipliers: B x f, a x phi, h / phi. The pair's ceiling is then
  0.6/87 per tick and its half-saturation on the total stock 1/(a h f) = 1.4 t/km2. Do not fold f
  into a or apply phi twice.

## 5. Recent diet numbers, for checking only (Q3)

**No benthos share by mass from after 2000 was found for any gadoid in the Skagerrak, Kattegat or
Norwegian Deep.** The recent sources do not fill this gap:
- CINEA 2024 (2022-23 stomachs) reports by number only, with ICES IIIa n = 12 cod and 20 whiting.
- Werner 2015 (hake, 2014) reports frequency of occurrence only.
- SMS (the North Sea multispecies model) lumps non-fish prey into "other prey".
- No gadoid diet paper by "Kappa et al." exists; the only hit was unrelated seal work.

In-area data by mass (evidence C) [F, my calculation from the papers' tables]:

| Species | Area, years | Benthos share by mass | Note |
|---|---|---|---|
| Haddock 20-49 cm | Norwegian Deep incl. Skagerrak, 1984-87 (Albert 1994) | winter 0.86, summer 0.63, year **0.75** (0.68-0.78); by 10-cm class 0.80-0.98 winter, 0.54-0.96 summer | dry weight inflates shells; low end halves ophiuroids + bivalves |
| Cod | Skagerrak, Feb 1991 (Dorman et al. 1992) | <25 cm 0.88-1.00; 25-50 cm 0.56-0.74; 50-70 cm 0.49-0.52; >=70 cm 0.25-0.37; pooled **0.46** (0.54 with Pasiphaea) | Kattegat pooled 0.23 (krill 25 %) |
| Norway pout | Norwegian Deep 1985-87 (Albert 1994b) | winter ~0.01, summer 0.08-0.14 | 0.60 on a SW Norway bank (Albert 1995) |
| Whiting | total North Sea 1981 (Hislop et al. 1991) | 10-15 cm 0.36-0.45; 25-40 cm 0.05-0.06; >50 cm 0 | no in-area data |
| Saithe, hake | CRR 219; Werner 2015 | ~0.01; hake not by mass | |

Community share for the IBTS mix (biomass-weighted, evidence C):
- **0.47 (0.41-0.50)** from the in-area values.
- 0.33 with Norway pout at half the biomass of the other species.
- 0.27 with the whole-North-Sea haddock value (0.42, CRR 219).

**Validation band: 0.35-0.50.** The low end allows for a further dry-to-wet correction and some
Norway pout.

Consistency: phi 0.60 with haddock's own 0.75 and cod's 0.50 gives 0.54 x 0.75 + 0.06 x 0.50 = 0.44,
inside the band. This is a check, not a fit.

**Two corrections to the brief and the previous review:**
- "No gadoid species is ~80 % benthos" does not hold in the area. Norwegian Deep haddock is 0.80-0.98
  in winter, and Skagerrak cod under 25 cm is 0.88-1.00. The pooled group at 0.8 is still too high,
  by ~1.6-2x.
- The earlier target of 0.30 (0.25-0.35) came from whole-North-Sea 1981/1991 data. In-area data put
  the pooled share higher. Use 0.35-0.50 instead.

## 6. Model checks (not changes made here)

- **The pairs do not add.** The eat shares are one slice of a single action distribution
  (`decisions.actions_from_probabilities`). Total intake is therefore at most max_j 1/h_j, not the
  sum over pairs (`predation.py`). This corrects the brief's "each pair saturates independently".
- **The ceiling is high for all pairs.** The energy-limited 1/h is 138 kJ/kg/d. That is ~1.6x
  (0.9-1.8x) the cod Cmax at 7 C, which is 2.1 % BW/d at 500 g and 1.9 % at 1 kg (Hansson et al.
  1996 via FB4 [F, S]).
  - It is 2-3x the field rations of 0.9-1.7 % BW/d [F].
  - The h implied by Cmax is 135-154 ticks. The model's 87 matches only the top of the
    satiation data (Braaten & Gokstad 1980 [F]).
  - The correction is group-wide and applies to every pair alike. Keep it separate from phi, or it
    is counted twice.
- **The benthos share is set by relative profitability.** All gadoid ceilings are energy-equalised
  (resume 148). Benthos is always saturated; herring and krill often are not. So the policy favours
  benthos. With phi < 1 a saturated benthos return falls to phi x 35 MJ/t/tick, and the diet share
  becomes emergent. Judge it against section 5 after training, not by tuning phi.
- **Any of these changes needs retraining:** a new pair parameter, a shifted benthos
  intake, possibly a changed observation.

## 7. Gaps and what can only be fitted

Fit-only (no mechanistic measurement exists):
- Search width and capture probability, hence a (and only in the product a f).
- How deep gadoids reach into mud (0-1 / 0-2 / 0-5 cm), including haddock digging depth.
- The arm-to-disc mass ratio of Amphiura. Everything about Amphilepis norvegica (300 m).
- A haddock Cmax function (only cod exists); whiting and saithe rations.

Data gaps:
- No benthos share by mass from after 2000, in the area, for any species. The ICES stomach database
  holds the 2013 and 2022-23 data, but no public summaries by mass were found.
- The vertical biomass profile rests on one 280 m station, sampled once. There is none at 100 m and
  no 0-1 cm slice.
- Not obtained: Bergstad 1991, Mattson 1990/1992, Pihl 1994, Schuckel 2010, de Vlas 1979,
  Duineveld & Van Noort 1986 (full text), Rosenberg's Norwegian Trench paper, Huwer et al. 2014,
  and the 1978 Gothenburg thesis on six Skagerrak/Kattegat gadoids.
