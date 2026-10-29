# Saithe diet on juvenile gadoids and flatfish - summary

Question (resume 223.31): the water-column group (`predatory_fish`; saithe 0.651 of its biomass, 0.77 of
its attack rate on benthivores) books 6.1 % of its food on `benthivorous_fish` (M2 0.083 /yr), but the
library's encounter-theory a (0.053 per whole-group tonne, accessible 0.24) gives ~21 % of the diet at
today's densities (shared-handling test, 223.27) and the trained fix80 policies realise 15-23 % (M2
0.33-0.39). Which side does the field support?

Sources: `evidence_saithe.md`, `evidence_spurdog_anglerfish.md` (literature agent, marks [S]/[J]), and
the main agent's SMS 2023 key-run calculation below (files in
`../Fiskgrupper i rutan/benthivores/data/sms2023`).

## 1. Answer

The field supports the booked share (6.1 %; field ~4 %, range 2-9), not 15-23 %.

| Member (share of group biomass) | Benthivores in diet, by mass | Basis |
|---|---|---|
| saithe (0.651) | **5 % (2-9)**, almost all 0-1 group haddock | CRR 164 (1980-82) 5.5 %, CRR 219 (1991) 5.0-5.2 %, SMS observed stomachs 4.1 % (40-59 cm 3.6 %), Mehl 2005 (Norway 1998-2003) 5.8 %; Ecopath 0.4 %, Iceland < 1.5 % |
| spurdog (0.080) | ~6 % (2-15), mostly plaice | Ecopath 1991 6.7 % |
| anglerfish (0.022) | ~15-20 % (8-25) | Ecopath 19 %, Iceland ~21 %, Shetland 8 % |
| mackerel, horse mackerel, hake, ... | ~0 | (not reviewed; pelagic or off the bed) |

Weighted by biomass (or by consumption with Q/B saithe 3.0, spurdog 1.5 J, anglerfish 1.8, the rest
0 benthivores), the group lands at **~4 %** (range ~2-9 %; the literature agent's consumption
weighting 4.4 %, 2.4-7.2), against the library's 6.1 %. No multi-season, multi-size
average reaches 15 %; such shares appear only in narrow cells (saithe 30-45 cm in Q3 of good haddock
years, very large saithe in summer, single coastal years) [J, evidence_saithe.md].

## 2. SMS 2023 key run, consumption side (main agent)

`summary.out` ration x N.bar = saithe consumption; `partial_m2.out` x N.bar x west = haddock eaten by
saithe. Saithe Q/B 2.63 / 2.74 / 2.78 /yr in 1974-85 / 1986-2000 / 2001-22 (the model's group 2.74).
Haddock share of saithe consumption 16.4 / 7.5 / **4.0 %**; whiting 5.0 / 3.5 / 3.0 %; cod 0 (cod is not a
saithe prey in SMS). Share vs haddock annual-mean biomass density (North Sea 575 000 km2, J): r 0.90,
log-log slope 1.11; at the model's haddock density (0.386 x 1.53 = 0.59 t/km2) the fit gives 0.17 t
haddock per t saithe per yr = **6.3 %** of a Q/B of 2.74. SMS's proportional response is its own
suitability x availability / (sum + other food) form, so the density slope is NOT independent evidence;
the level is stomach-informed (SMS modelled 1980-91 9.1 % vs observed stomachs 4.1 % in the same years,
evidence_saithe.md entry 4: SMS overshoots). Prey side: saithe partial M2 on haddock 2001-22 age 0 0.39,
age 1 0.52, age 2 0.13, age 3+ <= 0.04 /yr; on haddock 1+ 0.15 /yr.

## 3. What this says about the model

- **The booked M2 0.083 stands; the pair's a is what disagrees.** (Revised in section 6: the ~4x comes from
  an encounter test that does not apply to multi-layer predators; what the field fixes is f, and the rest is
  the learned time at the bed.) The encounter a with f 0.24 implies
  ~4x the field diet share (21.5 vs ~5.5 %); the realised 15-23 % comes from that a plus the saturated per-pair
  response and the learned eat share (223.31).
- **Prey identity.** Saithe eat 0-group haddock (7-11 cm, Q3-Q4) and 1-group (12-25 cm, Q1-Q2); cod is a
  trace (cod : haddock ~1 : 10 by number in 1981, absent in 1991 tables, none south of 67 N in Norway)
  and flatfish <= 2 % (only saithe > 70 cm). Haddock 0-1 group is 0.30 of haddock biomass (SMS 2001-22),
  i.e. 0.30 x 0.386 = **0.12 of the benthivore group** - half the library's accessible 0.24, which counts
  juvenile cod and small flatfish within the gape that saithe in practice do not take [J].
- **Season.** The haddock share is ~0 in Q1 at every saithe age (1991) and peaks in Q3 when the 0-group
  settles; the engine has no such window for the pair.
- **Density dependence.** Observed stomachs: Q3 haddock share 1.1 % in 1987 (poor year class) vs 6.9-8.5
  % in 1981, 1986, 1991; Norway between-year 1-23 %. Directionally a response to haddock abundance, n
  small; no functional-response study of saithe on haddock exists.
- Not resolved here (per the rule, no a fitted): which factor of the encounter a (0.053 = saithe's
  layer-scored reactive field, speed, capture, vertical overlap; params_split_223.md 3.3, range
  0.018-0.17) overstates the field rate. The prey-identity point halves a f (share 21.5 -> 12 %); on a f the
  field needs x4.7-6.0 overall (diet odds a f B / sum), so x2.4-3.0 remains, from overlap/season or
  capture, to be derived, not fitted.

## 4. Weak points

- Every by-weight North Sea saithe diet is pre-2000 (1980-82, 1986-87, 1991), mostly the north-west
  North Sea; no Skagerrak or Norwegian Trench saithe diet with numbers was reached (Bergstad 1991 Sarsia
  75 and Du Buit 1991 not accessed).
- Spurdog and anglerfish rest on Ecopath 1991 (a balanced model) plus one or two field sources each.
- The SMS density slope is structural (section 2).

## 5. The settlement window, derived (2026-10-29)

Sources: `evidence_settlement.md`, `evidence_saithe_season.md` (literature agent, DATRAS scripts
`datras_haddock_*.py`, `datras_saithe_haddock_overlap.py`), and the main agent's box survey table
`tables/box_gape_by_quarter_stdout.txt` (`scripts/box_gape_by_quarter.py`: NS-IBTS GOV 2001-25, the
Mareld box, depth-stratified, Q1 = Jan-Feb, Q3 = Jul-Sep with median August).

**One cohort through a 19 cm saithe gape** (40-60 cm saithe):
- Spawning Feb-May (peak Mar-Apr; no cue named in any source). Pelagic juveniles through June (4 cm on
  18 June, at the surface by night, few near the bed): in the water column but not saithe prey.
- **Settlement is size-triggered, not dated:** gradual from ~3 to ~7.8 cm, with the switch to bottom
  prey. The date follows hatching plus temperature-driven growth (0.6-0.8 mm/day): mid-July to
  mid-August offshore [J]. Settles mainly at 50-150 m; at 150-200 m 1/15 and at 200-300 m 1/40 of the
  50-100 m density (IBTS Q3).
- In the box (GOV): August 0-group 8-17 cm, mode 12-13 cm; January-February the same cohort is 14-24
  cm, mode 19-20 cm (box fish are larger than the northern-shelf 17-18 cm). The window **opens in July
  (settlement) and closes in March-April of age 1 as the cohort outgrows 19 cm**; Apr-Jun no haddock is
  inside the gape on the bed (1-group above, new 0-group still pelagic).
- Haddock < 19 cm as a share of haddock biomass, catchability-corrected (Walker et al. 2017 Q3
  q-at-length): Aug 0.234, Jan-Feb 0.095 (raw 0.074 / 0.066; q at <= 11 cm is 0.0116, so the
  correction is large and uncertain).

**Time mean.** Monthly profile Jul 0.12, Aug-Sep 0.23, Oct-Dec 0.20 -> 0.15, Jan-Feb 0.095, Mar 0.03,
Apr-Jun 0 [J interpolation]: mean 0.111 of haddock biomass = **0.043 of the benthivore group** (x 0.386
haddock share; raw survey 0.018). The window is open ~8 of 12 months; its peak (0.09 of the group)
is twice the mean. Against the pair's accessible 0.24 that is x0.18 (raw x0.075).

**Which basis.** The engine's benthivore biomass is GOV wing swept area at q = 1 for every length
(`../Fiskgrupper i rutan/gadoid_level.md`), a convention with ~2x uncertainty, not a claim that the
uncaught 0-group is absent. 0.043 treats the group biomass as the true stock (relative q-at-length
applied, the 0-group adds ~20-25 % in Q3); 0.018 treats it as literally what the GOV retains.

**Check against the stomachs - WITHDRAWN (2026-10-29, see section 6).** An earlier version said f 0.043
gives saithe 4.7 % and closes saithe. That used the PAIR's a ratios (benthivores 0.053 vs planktivores
0.0117), but the pair's planktivore a is mostly mackerel (0.49) and horse mackerel (0.12), which take no
benthivores; saithe's own ratio is ~10.6, not 4.5. With saithe's own a the encounter diet at f 0.043 is
~16 % (section 6). And section 6 shows that this encounter test does not apply to a predator whose prey
live in different layers. What stands from section 5 is the prey-side value itself: settled haddock
inside a 19 cm gape, time mean 0.043 of the group (raw 0.018).

**The Q1 low is prey and predator together.** Prey: the 1-group is 17-20 cm in Jan-Feb (box mode
19-20), at or past the 19 cm gape (24-34 % already above it on the northern shelf, 72 % by March), and
in-gape biomass in the box is 0.4 of August (q-corrected). Predator: 26-50 % empty stomachs in Q1 1991 at every age (not only spawners),
saithe deeper than the 1-group (141-154 vs 77-98 m) and half the Q3 overlap. Q1 is not always zero
(1981: 6-11 % in 40-70 cm saithe with a 2x-median 1-group).

**Phase.** The window (Jul-Feb) covers the months in which the trained water column falls back on the
benthivores (rig months ~Oct-Feb, 223.31); its closed months (Mar-Jun) are when herring pay most. So
the window's main effect on the model is its LEVEL, not its phase.

**Implementation options (not applied; user's call under the behaviour rule).**
- A. A constant accessible share at the time mean: the member-weighted f of section 6 (~0.05-0.08).
  No calendar; library-only; checkpoints valid. Recommended.
- B. A juvenile stage in the benthivores (settlement flux size-triggered from temperature-driven
  growth, growth out of the gape on the bottom temperature). Mechanistic, but the spawning date has no
  known cue and would be a fixed calendar anchor - against the rule unless a physical cue is found -
  and it is a structural change on both engines.

Weak: q at <= 11 cm dominates the correction; Dec and Apr-Jun interpolated, March few hauls; model's
haddock share 0.386 assumed on the same basis as the survey biomass; spurdog/anglerfish not derived;
Bergstad 1991 (Trench season-by-depth) not accessed.


## 6. Member f on the benthivores, and what the encounter test can and cannot say (2026-10-29)

Member shares of the pair's layer a (`../Rovfiskens uppdelning/tables/split_encounter_223_stdout.txt`):
saithe 0.77, spurdog 0.10, hake 0.07, velvet belly 0.04, anglerfish 0.01 (not "spurdog + anglerfish
0.23" as written before). Guild shares below each gape in the box (mean Q1/Q3, raw survey,
`tables/box_gape_by_quarter_stdout.txt`): < 12 cm 0.008, < 19 cm 0.124, < 27 cm 0.304 (haddock+cod
0.114, flatfish/rays/gurnard 0.190), < 28 cm 0.324, < 45 cm 0.735.

**Prey-side f per member [J from gape, field diet identity and the box table]:**

| Member (share of a) | Gape | What it takes (field) | f |
|---|---|---|---|
| saithe (0.77) | 19 cm | settled haddock, cod a trace, flatfish <= 2 % | 0.043 (raw 0.018; section 5) |
| spurdog (0.10) | ~27 cm | benthivores ~6 % of diet, mostly plaice (Ecopath 4.8 %), haddock 0.6 %, cod 0 | ~0.2 (0.15-0.3): flatfish/rays < 27 cm 0.19 + a little haddock. That a shark takes buried flatfish fits electroreception of buried prey (Kalmijn 1971 on dogfish and rays; not re-checked here) [J] |
| hake (0.07) | ~28 cm | gadoids 16-48 % of the Skagerrak/Kattegat 2024 hake diet, whiting the main one; benthivore part unknown | 0.1-0.3 (unresolved) |
| velvet belly (0.04) | ~10-12 cm | small shark of shrimps and small fish | ~0.01 (< 12 cm 0.008) |
| anglerfish (0.01) | > 40 cm | haddock, cod, long rough dab, dab (Ecopath 19 %, Iceland ~21 %) | ~0.7 (0.6-0.75; < 45 cm 0.735) |

**Pair f = sum c_m f_m** = 0.77 x 0.043 + 0.10 x 0.2 + 0.07 x (0.1-0.3) + 0.04 x 0.01 + 0.01 x 0.7 =
**0.067-0.081** (raw saithe basis 0.018: 0.048-0.062), against the library's 0.24. x = a h f B at the
start: 2.68 -> 0.75-0.90 (raw 0.54-0.69): the pair goes from saturated to half-saturated and its take
becomes density-dependent.

**The encounter test does not apply to multi-layer predators (revises 223.27).** Each pair's a is
defined INSIDE the prey's layer, with the time in that layer outside a (the eat share; section 220
method): bed fish are a strip 2R x v (no /H), pelagic prey a disc / H (H 70 m for herring). Summing
these a's in one shared denominator (the 223.27 shared-handling test, and the 21.5 % "implied by a")
assumes the predator searches every layer at once. For saithe, spurdog and hake that is not the
engine's diet and not a prediction about a f; their diet is set by the time they spend at the bed,
i.e. the learned eat share. `scripts/member_encounter_diet.py` (`tables/member_encounter_diet_stdout.txt`)
shows the consequence: with each member's own a the benthivore share barely moves with f (saithe <60
25 % at f 0.24, 16 % at 0.043; spurdog 30 -> 20 %), partly because the bed term dominates where the
predator sits (per-cell shares near 0 or 1 at the clumped start). The library check of the same
script gives 19.0 % for the pair (223.27: 21.5 % effort-scaled at the budget ration). For anglerfish,
a bed ambusher meeting all its prey at the bed, the test is fair: 54 % at f 0.7 vs field 17-20 %
(~30 % renormalised over the modelled prey); the gap sits on its planktivore a (0.00038 per tonne; it
meets near-bed Norway pout), not on f - negligible at 0.01 of the pair.

**What it leaves for chain 1.** With f ~0.07-0.08 the pair is half-saturated and density-dependent.
The rest of the gap between the realised 15-23 % and the field ~4-6 % is the learned eat share (time
at the bed). The field number to compare it with is an OUTCOME (the share of saithe foraging spent on
bed prey), not an input; a stays untouched per the rule.
