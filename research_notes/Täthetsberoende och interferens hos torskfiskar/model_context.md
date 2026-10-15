# Model context: Beddington-DeAngelis interference and density dependence for the Mareld "gadoids" FG

Scope: model / repo facts and frozen-policy measurements only, taken from the local
repository /home/simon/eco-evo on 2026-10-14 (branch coev-repro, HEAD 147e4c5, plus
the untracked working tree). Each "source" is a file:line reference in that
repository, not a web URL. Tags: **[M]** model fact read in the code or config;
**[R]** stated in the repo's own documentation (resume, comments); **[C]** computed
for this note (probe runs or arithmetic on repo numbers). Every measurement below
uses the **frozen** fix55 policies (copy of results/pzbpgp248nsrpfix55 taken
2026-10-14 18:18; md5 gadoids 6b56d8d6, herring 6e3ce33f, porpoises f5593c83,
zooplankton 4f5412af). A frozen policy does not re-optimise when w changes, so
sections 3-4 are a static, first-order answer.

Probe (untracked, copies the loading pattern of tools/probes/intake_ceiling.py:
`train._ProbeEnvBuilder(project_path='mareld2.yaml', grid_size=(60, 60),
apply_natural_mortality=True, migration=True, currents=CurrentConfig(0.1, 20, 0, 12.0))`
+ `inference.load_policies_and_stats`, seed 20260530; daylight and temperature from
mareld2.yaml; gadoids start at 2500 t, herring 12000 t):

```bash
S=/tmp/claude-1001/-home-simon-eco-evo/ccaf6a1d-4766-43b1-96b8-776d76934db7/scratchpad
cd $S && python3 gadoid_interference_probe.py $S/fix55_frozen --years 2 --w 1 --out runs/w1.json
#   --w W            gadoid interference in memory (env.dm_interference[gadoids])
#   --init-scale K   multiply the initial gadoid biomass and reserve (same layout)
python3 summarise.py w1 w3      # annual tick-weighted means from runs/*.json
```

One 2-year run takes ~46 s on the CPU engine. The probe calendar starts in mid-May,
so "year 1" is May-May. The first month bucket (seq 0, ~2 weeks) is the spawn colony
geometry (25 colonies, sigma 1 cell, fg_library.yaml:240-243), not behaviour;
it is shown in the monthly table but left out of the year-1 means.

## 1. Code path and units of w (Q1)

### Takeaway
w is a per-FG constant in 1/(t per cell) = km2/t (cells are 1 km2). It enters the
Holling denominator as `I = w * B_gadoid(c)`, the gadoids' **total** biomass in the
cell (resting/hidden fish included), and is broadcast over **every** prey pair of the
predator. The hunger gate h multiplies the response afterwards, so the two are
separable in one tick, but h rises when interference lowers the reserve. For the
type III herring pair the term sits next to `a h B^2`. The growth budget and s* never
read w, and w is not tick-rescaled.

### Cited Findings
- [M] Response: `a_eff_ii = a*Bp/(1 + a*h*Bp + interference)`,
  `a_eff_iii = a*Bp^2/(1 + a*h*Bp^2 + interference)`, mixed by the type III mask
  `m3` — lib/environments/ecosystem_env/interactions.py:284-310 (formula :307-310).
  Docstring: "I = w_X * B_X(c), i.e. the predator's own biomass in the cell"
  (:295-299); interference "only lowers the curve, it never changes that shape" (:301-303).
- [M] CPU call site: `predator_biomass = np.stack([env.fgs[fid].biomass ...])`
  (predation.py:10-11) — the full cell biomass, not a visible fraction (visibility is
  applied only to prey, :29-30, :50-54). `interference = env.dm_interference *
  predator_biomass[:, None, :, :]`, shape (N_dm, 1, H, W), "broadcasts over the prey
  axis", intraspecific only — lib/environments/ecosystem_env/predation.py:33-40.
  Passed to `holling_a_eff` at :57-59; the no-handling branch divides the bare rate by
  `1 + I` (:60-63).
- [M] Demand: `predator_biomass * actions.eat * effective_intake_rate * hunger`
  (predation.py:66-71); the harvest cap `MAX_HARVEST_FRAC = 0.999` of visible prey
  (:85-95, constants.py:16) is the only later limit. [R] Section 160.4: for gadoids the
  cap never binds, so demand is intake.
- [M] Hunger `h = clip((1 - s)/(1 - u), 0, 1)` — lib/world/functional_group.py:117-129.
  Gadoid u = 0.5 (fg_library.yaml:201).
- [M] GPU mirror: `self.interference = env.dm_interference` reshaped [1, D, 1, 1]
  (lib/gpu/ecosystem.py:69-74); `inter = self.interference * b_dm[:, :, None]` with
  `b_dm = biomass[:, self.dm_index]` (total biomass, :355, :379); same type II/III
  denominators (:388-392) and the `a / (1 + inter)` branch (:393-394). Parity test
  tests/test_gpu_ecosystem.py:195.
- [M] Parameter: `self.interference = float(params.get('interference', 0.0))`, clipped
  at 0 (lib/world/functional_group.py:44-51); cached as `env.dm_interference`
  (N_dm,1,1,1) and `env._has_interference` (lib/environments/ecosystem_env/state.py:167-176).
  Editing `env.dm_interference[i]` in memory is enough to change a run (this probe).
- [M] Library values: zooplankton 0.0 (fg_library.yaml:116), pelagic_fish 0.7 (:196),
  **gadoids 1.0 (:250)**, porpoises 0.0 with the comment that they never reach
  interfering densities (:323-326). FG editor field range 0-10 (fgconfig/fgconfig.py:732).
- [M] Not tick-rescaled: "interference (1/ton) and min_split_biomass (kg)" are listed as
  tick-independent — lib/world/tick_time.py:150-157. (a h and w B are both dimensionless.)
- [M] Gadoid pairs it acts on (all checked pairs, a = 0.04 per tick, fg_library.yaml:239):
  herring h 217, **functional_response 3**, dark_ratio 0.6 (:439-475; h :455, type :461,
  light :474); benthos h 87 (:547-552); zooplankton h 130 (:587-592). Cannibalism
  `gadoids_preys_on_gadoids` is `preys_on: false` since section 160 (:560-565).
  [C] a h = 8.68 (herring, per (t/km2)^2), 5.2 (zooplankton), 3.48 (benthos) per t/km2:
  half-saturation 0.34, 0.19, 0.29 t/km2 of visible prey.
- [M] Growth budget ignores w: `ideal_intake` uses `intake_ceiling(a, h) * assimilation
  * energy` of the best checked pair (lib/world/growth_budget.py:107-138);
  `ideal_reserve` / `_ideal_gate` solve s* from that intake (:141-151, :300-323). The
  probe docstring states "no interference" (tools/probes/ideal_growth.py:9).
  [C] growth_budget('gadoids'): gross 0.9814 /yr (r_max 0.511 + M1 0.415 + porpoise M2
  0.0557), best prey zooplankton, **s* = 0.7218**, g = 0.003031 /tick.

### Inferences
- Because h multiplies f after interference, a lower f is partly paid back by a lower
  s and a higher h. The realised intake loss is therefore smaller than (1 - R), and the
  cost shows up as a lower reserve, i.e. lower production `g (s - u)`.
- For type III herring, `f ~ a B^2 / (1 + I)` at low herring density: interference
  deepens the herring refuge where gadoids are dense. At high herring density
  (`a h B^2` in the hundreds) the term is negligible.

## 2. Frozen-policy measurement at the library w = 1 (Q2)

### Takeaway
The fix55 gadoid policy spreads the stock over almost the whole grid: ~2000-3600 of
3600 cells hold > 0.01 t, and the biomass-weighted cell density is narrow (year-1
p10/p50/p90 ~0.77/1.37/1.86 t/km2). At that density `wB` is ~1.2-1.5, against an
eat-weighted saturation of ~3-28 (zooplankton, seasonal), ~18-25 (benthos) and
~50-370 (herring, type III). Interference removes **~6 %** of the gadoid intake
(intake-weighted R = 0.941; demand ratio D/D0 = 0.937). The gadoids grow 2500 ->
3574 t in year 1 (1.43x) and -> 5273 t in year 2. Gross production is 0.834 /yr
against losses 0.478 /yr, and herring falls 12000 -> 6276 t over two years.

### Cited Findings
- [C] Monthly, gadoids, w = 1 (sat = eat-weighted `a h B_vis` per pair, with `a h B_vis^2`
  for herring and the light multiplier included; wB, s and h biomass-weighted over
  gadoid cells; R = (1+sat)/(1+sat+wB) intake-weighted over cells and pairs):

| seq month | B end (t) | cells > 0.01 t | dens p10 / p50 / p90 (t/km2) | sat zoo | sat benthos | sat herring | wB | R | s | h |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 May (spawn) | 2512 | 1261 | 1.40 / 5.43 / 18.41 | 38.5 | 15.1 | 3948 | 7.84 | 0.893 | 0.682 | 0.64 |
| 1 Jun | 2593 | 1990 | 0.84 / 2.11 / 3.67 | 26.2 | 19.4 | 366 | 2.26 | 0.937 | 0.690 | 0.62 |
| 2 Jul | 2713 | 2438 | 0.80 / 1.64 / 2.32 | 23.4 | 19.8 | 252 | 1.60 | 0.954 | 0.725 | 0.55 |
| 3 Aug | 2838 | 2679 | 0.75 / 1.50 / 1.97 | 17.8 | 19.6 | 205 | 1.44 | 0.950 | 0.725 | 0.55 |
| 4 Sep | 2941 | 2837 | 0.73 / 1.43 / 1.80 | 12.8 | 19.1 | 168 | 1.35 | 0.939 | 0.703 | 0.59 |
| 5 Oct | 3006 | 2929 | 0.75 / 1.33 / 1.68 | 8.6 | 18.9 | 139 | 1.30 | 0.930 | 0.668 | 0.66 |
| 6 Nov | 3059 | 2993 | 0.75 / 1.27 / 1.60 | 4.7 | 18.3 | 98 | 1.26 | 0.932 | 0.656 | 0.69 |
| 7 Dec | 3102 | 3046 | 0.77 / 1.24 / 1.57 | 2.5 | 18.4 | 77 | 1.24 | 0.931 | 0.647 | 0.71 |
| 8 Jan | 3139 | 3097 | 0.77 / 1.22 / 1.57 | 3.1 | 18.8 | 71 | 1.21 | 0.930 | 0.641 | 0.72 |
| 9 Feb | 3191 | 3167 | 0.77 / 1.19 / 1.53 | 7.5 | 19.8 | 73 | 1.19 | 0.932 | 0.653 | 0.69 |
| 10 Mar | 3295 | 3269 | 0.75 / 1.16 / 1.50 | 14.7 | 21.0 | 100 | 1.18 | 0.941 | 0.694 | 0.61 |
| 11 Apr | 3440 | 3342 | 0.77 / 1.16 / 1.53 | 22.9 | 22.2 | 155 | 1.19 | 0.953 | 0.722 | 0.56 |
| 12 May | 3600 | 3405 | 0.77 / 1.19 / 1.64 | 27.6 | 23.6 | 221 | 1.23 | 0.959 | 0.727 | 0.55 |
| year 2 (Jun-May) | 5273 | 3460-3590 | ~0.95 / 1.35 / 2.07 | 2.5-27.6 | 22.7-24.9 | 54-254 | 1.28-1.56 | 0.945 | 0.691 | 0.62 |

- [C] Year-1 means (seq 1-12): biomass-weighted sat zoo 13.4, benthos 18.1, herring 92
  (the herring mean is dominated by dense herring cells); visible prey at gadoid cells
  zoo 2.58, benthos 5.21, herring 2.43 t/km2; intake shares herring 0.40, zoo 0.37,
  benthos 0.23.
- [C] Pair reduction factors at w = 1 (year 1): **zooplankton 0.878, benthos 0.938,
  herring 0.964**. Interference tilts the realised diet toward the most saturated pair
  (herring); at w = 20 the herring intake share is 0.47 against 0.40 at w = 0.
- [C] Year-1 balance (rates on mean biomass 3017 t): gross production 0.834, M1 0.415,
  M2 (porpoises) 0.063, starvation 0.000, net +0.356 /yr. Year 2 (mean 4445 t):
  gross 0.847, net +0.382: **no density dependence visible between 3000 and 4400 t**.
  Herring taken by gadoids 5972 t (0.50 /yr of mean herring) in year 1, 8369 t
  (0.92 /yr) in year 2. These match the task's fix55 reading (1.45x per episode, losses
  ~0.48 /yr).
- [C] Reserve vs s*: mean s 0.688 (year 1), 0.691 (year 2) against s* 0.722.
  (s - u)/(s* - u) = 0.847 and gross/budget = 0.834/0.981 = 0.850; at w = 0 0.901 vs
  0.904. Production is set by where the reserve sits; w = 1 accounts for ~5 of the ~15
  points of shortfall, the rest is not interference (seasonal food, winter s 0.64-0.65).
- [R] Section 86.3 calibration (lrgrid16, 16x16, banded run): gadoid density p50 0.017,
  p90 1.95 t/cell, sat p50 6.3, sat/p90 3.2; library = one third ->
  `interference: 1.0` (mareld_resume.txt:7590-7612). The gadoid h was 25 before
  section 147 (fg_library.yaml comment at :443-454), so `a h` was 1.0 per t/km2 then.
  [C] Re-applying the 86.3 recipe to today's year-1 numbers: sat/p90 = 13.4/1.86 = 7.2
  (zoo), 18.1/1.86 = 9.7 (benthos), 17 for the two type II pairs together; one third of
  that is **2.4, 3.2 and 5.7**. Relative to prey saturation, today's w = 1 is ~2-6x
  weaker than the section-86 rule intended, because the energy-limited handling times
  (sections 147-148) raised `a h` 3.5-8.7x. (Approximate: 86.3 used the median cell,
  this note biomass-weighted means.)

## 3. Sensitivity to w, frozen policies (Q3)

### Takeaway
Raising w lowers gadoid growth roughly linearly in year 1 (net +0.41 at w = 0, +0.36
at 1, +0.28 at 3, +0.20 at 5, +0.04 at 10). The cost is paid through the reserve:
mean s falls 0.700 -> 0.633 and h rises 0.60 -> 0.72. From w ~ 5 cells start to drop
below u and starve. Herring benefit: taken by gadoids 6222 t at w = 0 vs 5021 t at
w = 10. In **year 2 net growth recovers to +0.26-0.27 /yr at every w from 10 to 20**:
herring rebuilds (10000-12400 t), gadoid density thins (p50 0.7-1.0) and R stalls
near 0.77. This is static. A trained policy re-optimises against w: section 86.4
shows trained gadoids spreading out and dropping aniso from 19 to 0.3, while replaying
the old policy under the new w only moved it 24 -> 19 (mareld_resume.txt:7614-7652).
Trained fish would dodge part of the cost measured here.

### Cited Findings
- [C] Sweep, gadoids start 2500 t, same world and policies. Year-1 means exclude the
  spawn bucket. Rates /yr on mean biomass. Net = gross - M1 - M2 - starvation.
  Herring taken = tonnes eaten by gadoids (rate on mean herring biomass).

| w | B end y1 | mean s y1 | R y1 | gross y1 | starv. y1 | net y1 | herring taken y1 | herring end y1 | B end y2 | R y2 | net y2 | herring end y2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3771 (1.51x) | 0.700 | 1.000 | 0.887 | 0.000 | +0.410 | 6222 t (0.52) | 9960 | 5791 | 1.000 | +0.423 | 5328 |
| 0.5 | 3664 (1.47x) | 0.693 | 0.967 | 0.858 | 0.000 | +0.381 | 6082 t (0.51) | 10121 | 5505 | 0.969 | +0.401 | 5855 |
| **1** | **3574 (1.43x)** | 0.688 | 0.941 | 0.834 | 0.000 | +0.356 | 5972 t (0.50) | 10248 | 5273 | 0.945 | +0.382 | 6276 |
| 2 | 3425 (1.37x) | 0.678 | 0.898 | 0.792 | 0.000 | +0.313 | 5798 t (0.48) | 10450 | 4911 | 0.908 | +0.354 | 6952 |
| 3 | 3299 (1.32x) | 0.670 | 0.865 | 0.757 | 0.002 | +0.276 | 5659 t (0.47) | 10613 | 4626 | 0.880 | +0.331 | 7497 |
| 5 | 3067 (1.23x) | 0.657 | 0.816 | 0.704 | 0.021 | +0.204 | 5439 t (0.44) | 10879 | 4176 | 0.840 | +0.303 | 8380 |
| 10 | 2607 (1.04x) | 0.633 | 0.743 | 0.620 | 0.097 | +0.042 | 5021 t (0.40) | 11396 | 3421 | 0.790 | +0.267 | 9970 |
| 12 | 2453 (0.98x) | 0.626 | 0.725 | 0.596 | 0.134 | -0.019 | 4871 t (0.39) | 11582 | 3203 | 0.779 | +0.263 | 10487 |
| 15 | 2231 (0.89x) | 0.616 | 0.704 | 0.568 | 0.203 | -0.115 | 4637 t (0.37) | 11862 | 2914 | 0.770 | +0.263 | 11232 |
| 20 | 1912 (0.76x) | 0.601 | 0.680 | 0.531 | 0.330 | -0.274 | 4259 t (0.33) | 12297 | 2521 | 0.765 | +0.274 | 12374 |

- [C] wB (year 1, biomass-weighted): 0.69 (w 0.5), 1.37 (1), 2.69 (2), 3.97 (3),
  6.39 (5), 11.9 (10), 14.0 (12), 16.8 (15), 21.2 (20); density p50 drifts 1.40 ->
  1.03 t/km2 from w 0 to 20, so the frozen field itself thins a little.
- [C] Pair R at w = 3 / 10 / 20 (year 1): zooplankton 0.738 / 0.536 / 0.432, benthos
  0.847 / 0.678 / 0.574, herring 0.926 / 0.863 / 0.824.
- [C] At w = 10 the stock falls Oct-Feb (2706 -> 2429 t) with s 0.57-0.61 and
  rises again Mar-Sep. The starvation in year 1 is a winter loss, when zooplankton
  saturation is 2.5-7.
- [C] Density instead of w (w = 1, initial gadoid biomass and reserve scaled, same
  layout). This is the density dependence the model has today, with prey depletion
  included:

| initial t | dens p50 / p90 y1 | wB y1 | R y1 | mean s y1 | gross y1 | starv. y1 | net y1 | herring taken rate y1 | herring end y1 | B end y2 | net y2 | herring end y2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1250 | 1.03 / 1.51 | 1.01 | 0.957 | 0.696 | 0.872 | 0.000 | +0.380 | 0.24 | 13075 | 2933 | +0.468 | 12634 |
| 2500 | 1.37 / 1.86 | 1.37 | 0.941 | 0.688 | 0.834 | 0.000 | +0.356 | 0.50 | 10248 | 5273 | +0.382 | 6276 |
| 5000 | 1.90 / 2.90 | 2.06 | 0.915 | 0.667 | 0.741 | 0.000 | +0.278 | 1.14 | 5439 | 7181 | +0.081 | 2042 |
| 7500 | 2.43 / 4.19 | 2.79 | 0.883 | 0.630 | 0.582 | 0.019 | +0.110 | 1.67 | 3118 | 6542 | -0.243 | 1514 |
| 10000 | 2.69 / 5.69 | 3.44 | 0.864 | 0.593 | 0.462 | 0.197 | -0.184 | 1.92 | 2542 | 6362 | -0.261 | 1384 |
| 5000, w = 3 | 1.82 / 2.78 | 5.88 | 0.818 | 0.644 | 0.640 | 0.000 | +0.176 | 1.02 | 6288 | 6040 | +0.012 | 3187 |

- [C] Quadrupling the density moves R only 0.94 -> 0.86, while the herring take rate
  goes 0.50 -> 1.92 /yr and herring collapses to ~1400-2500 t. Gadoid regulation at
  w = 1 comes from **herring depletion**, not interference. Once herring is gone
  the gadoid diet shifts to zooplankton (intake share 0.37 -> 0.63) and benthos.

## 4. Which w gives zero net growth at 2500 t? (Q4, computed + inference)

### Takeaway
There are two readings, and they give different answers:
- **Net growth = 0 over year 1** (gross = M1 + M2 + starvation): linear interpolation
  between w = 10 (+0.042) and w = 12 (-0.019) gives **w ~ 11.4** [C]. That zero is
  reached with gross ~0.60 and **starvation already ~0.12 /yr**. The required term
  is wB ~ 13 (biomass-weighted), i.e. ~10x today's 1.37.
- **Gross production = realised losses 0.478 /yr** with no starvation: gross falls only
  0.0074 /yr per unit w between w = 15 and 20. Extrapolated, gross = 0.48 needs
  **w ~ 27-30** [C, extrapolation], deep in the starving regime (> 0.33 /yr at w = 20).
  The identity gross ~ g (s - u) * 1460 would put the mean reserve at s ~ 0.61
  (u + 0.478/4.43) [C]. The identity only holds while no cell is below u. Once cells
  starve, gross counts only the positive part and the mean s understates it (w = 20:
  s 0.601, gross 0.531).
Neither holds in year 2: at every w from 10 to 20 net growth is back at
+0.26-0.27 /yr, because herring rebuilds and the frozen field thins [C]. Under frozen
behaviour, w alone delays the overshoot by about a year. It does not pin gadoids at
2500 t [inference].

### Cited Findings
- [C] Density equivalent at w = 1: because only the product wB enters (Section 1), w ~
  11 at 2500 t is the interference a ~10x denser cell field would get at w = 1
  (biomass-weighted ~13 t/km2, ~25-30 kt with today's spread). That is a static
  equivalence and ignores prey depletion. The init-scale runs show that at w = 1
  herring depletion halts the gadoids first: year-1 net crosses zero between 7500 t
  (+0.11) and 10000 t (-0.18), interpolated **~8400 t**. Starting from 5000 t, year-2
  net is +0.08 with herring at 2042 t.
- [C] Static stability check: realised losses are density-independent here (M1 0.415 /yr
  as a fixed per-tick rate, M2 by porpoises 0.03-0.08 /yr). The only density-dependent terms in the
  gadoid balance are food (prey depletion, reserve, h) and w. Cannibalism, the usual
  density-dependent loss, is unchecked (section 160.6, mareld_resume.txt:14594-14608).
- [R] Section 160.6 already names "a density-dependent non-behavioural loss" as the
  candidate if the overshoot stays after retraining (mareld_resume.txt:14605-14607).
  Section 86.7 notes that only {0, w/3, w} were ever swept
  (mareld_resume.txt:7710-7712).

### Gaps
- All of this is frozen behaviour. Section 86.4 shows the policy re-optimises against w
  (spreads out, eats where it stands). A retrained policy at high w would probably
  spread further and lower its own wB, so the static numbers are an upper bound on
  how much regulation a given w buys.
- Two simulated years, one seed, one world (the probe world). The winter starvation at
  high w depends on the initial reserve and the May start.
- The "~0.48 /yr realised losses" target ignores the herring feedback. A w that holds
  gadoids at 2500 t also keeps herring near 12000 t, which feeds them back up.

## What this means for choosing w

- **At today's densities w = 1 is a weak term.** The fix55 policy spreads gadoids
  almost uniformly (p50 ~1.4 t/km2, p90 ~1.9), so wB ~1.4 against prey saturation
  ~13-20 (zooplankton/benthos) and ~100+ (herring, type III). It costs ~6 % of intake,
  ~0.05 /yr of production (0.887 -> 0.834) and ~250 t of herring per year. It still
  does its section-86 job (anti-stripe: finite optimum for aggregation), which this
  probe cannot judge.
- **The section-86 calibration is stale.** Handling times went up 3.5-8.7x in
  sections 147-148, so `a h B_prey` is now 2-3x the 86.3 value relative to gadoid
  density. Re-running the same "sat/p90, then one third" recipe on fix55 gives w ~2.4
  (zooplankton) to ~3.2 (benthos), ~5.7 for the summed type II pairs. A w of 2-3 is
  the value that keeps the original design intent. It is not a fit to any data.
- **w cannot be the gadoid density regulator by itself.** To stop the year-1 overshoot
  under frozen behaviour w must be ~11 (net) or ~27-30 (gross = losses), both with
  heavy winter starvation, and growth resumes in year 2 at +0.26 /yr whatever w is.
  Interference is a per-cell, intake-side term. A spread-out stock escapes it, and
  BD discounts the herring pair least (type III, high saturation), so a large w also
  pushes the diet toward herring.
- **What actually regulates gadoids today is herring depletion** (w = 1: zero net
  between 7500 and 10000 t, with herring at 1400-3100 t). If the target is gadoids
  near 2500 t with herring intact, the missing piece is a density-dependent **loss**
  (what cannibalism provided, section 160.6) or a lower budget (s* without
  interference overstates attainable production: gross/budget = 0.85 at w = 1). A
  stronger w is not that piece.
- **If w is raised** (2-3 per the 86 logic), the static cost at 2500 t is net growth
  +0.31 to +0.28 instead of +0.36, and gadoids eat about 170-310 t less herring per year than at w = 1 (5798 / 5659 vs 5972 t). That
  must be confirmed with a retrained policy, since section 86.4 shows trained fish
  re-optimise against the term. Including w in s* (an `ideal_intake` that sees the
  expected wB) would be a separate, budget-side change.
