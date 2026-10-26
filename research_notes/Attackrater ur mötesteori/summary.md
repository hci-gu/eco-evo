# Attack rates from encounter theory: summary of the three group notes

Written by the main agent from `planktivores_encounter.md`, `benthivores_encounter.md` and
`predators_encounter.md` (2026-10-25, after resume section 219). Biology-first: no value here was
set from model output. Nothing in the library, code or manifest was changed. `data/` (~350 MB of
downloads) is not for commit.

Engine meaning of `a` (lib/environments/ecosystem_env/interactions.holling_a_eff): area cleared per
tonne of FORAGING predator (inside predator_share) per 6 h tick, km2/t/tick, against visible (and
accessible) prey in t/km2; annual mean over the daylight multiplier. Library value today: 0.04 on
every fish pair (inherited, no source).

## Derived values (central, 5-95 % Monte Carlo)

| Pair | h | a central | range | 1/(a h) t/km2 | vs library 0.04 | night/day (library dark_ratio) |
|---|---|---|---|---|---|---|
| planktivores -> zooplankton | 40 | 0.007 | 0.0018-0.016 | 3.6 | 1/6, not saturated | 0.06 (0.25, -8 deg; half-point at -0.6 to -2.3 deg) |
| benthivores -> benthos | 110.7 | 0.26 | 0.12-0.63 | 0.035 (accessible) | x6, saturated either way | 0.84 (none) |
| benthivores -> planktivores | 262.9 | 0.016 | 0.0013-0.069 | 0.24 | 0.4x, near the knee | ~0.2, J (0.6) |
| benthivores -> predators | 193.8 | 0.022 | 0.0018-0.089 | 0.24 | 0.55x, near the knee | 0.15-0.4 (none) |
| predators -> planktivores | 279.4 | 0.0056 (dispersed variant 0.012) | 0.0019-0.018 | 0.64 | 1/7, near the knee | 1.25 (0.13-2.4); dispersed 0.35 (0.6) |
| predators -> zooplankton | 176.5 | 0.0015 | 0.0009-0.0054 | 3.8 | 1/27, not saturated | 0.51 (none) |
| predators -> benthos | 117.6 | 0.030 | 0.0095-0.14 | saturated | ~1x | 0.99 (none) |
| predators -> benthivores | 176.5 | 0.18 | 0.058-0.65 | 0.032 | x4.5, saturated either way | 0.62 (none) |

Pattern: the benthic pairs and predators -> benthivores are saturated at any a in range (a does
not matter for intake there); every pair onto pelagic prey (zooplankton, planktivores) comes out
2-27x below 0.04 and moves from saturated to density-dependent.

## Conventions to harmonise before any value is used

1. Vertical overlap is folded into `a` on every fish pair (no accessible_fraction there). The
   benthivore note offers a same-layer a (0.12 / 0.10) with an accessible fraction (0.13 / 0.21)
   instead. May double-count with predator_share's "access" stage scores.
2. Schooling is inside `a` for predators -> planktivores, not in the benthivore note. If the prey's
   hide action and visibility floor already stand for schooling, use the dispersed variant (0.012).
3. Prey hiding: the planktivore note finds the copepods' daytime refuge is already in the overlap,
   so the pair's visibility floor 0.5 may count it twice.
4. Pool content: the planktivore note followed the A17 calibration (no krill in the zooplankton
   pool); the predator note puts krill at ~12 % and gives predator_share on zooplankton ~0.35
   instead of 0.48 (the product phi x a unchanged).
5. Benthos accessible_fraction 0.2 is the benthivores' infauna access; the exposed share predators
   can eat is ~0.01 (predator note). Pair saturated either way.
6. Light: the -5 / -8 deg thresholds suit fish at 30-50 m, not this grid's bottom (median 142 m);
   night/day > 1 (predators on planktivores) and dusk/dawn peaks cannot be set in the engine.
7. Seasonality: the planktivore a swings ~20x over the year (prey depth, Calanus overwintering);
   the engine's m(t) spans 0.74-1.33. Candidate library values 0.007 (annual time mean), 0.0125
   (annual consumption at low density), 0.0137 (summer daytime with today's dark_ratio).

## Dominant uncertainties

Reactive / detection distance in dim water (no measurement for adult herring or any predatory
member; ~1 decade each), capture success on fish prey (no gadoid measurement), the light profile at
depth, the vertical position of foraging fish relative to prey. Validation data (kept separate):
Arrhenius & Hansson YOY herring clearance (favours R proportional to length), Persson 2012 and
Holmes 1979 (benthos pair), Barents Sea MULTSPEC half-saturation 1.57 t/km2 (6-7x above the
benthivore fish-pair value), Engelhard 2008 (0.012-0.035, matches the dispersed predator variant).

## Gaps and restrictions

Not obtained: Holmes & Gibson 1983, de Groot 1971, Rindorf & Gislason 2005 (HTTP 403), any gadoid
capture-success figure; search budgets ran out partway. Holmes 1979 (thesis) restricts publication
of derived information; Braaten & Gokstad 1980 is cite-with-consent.

## Addenda: limited gap search (2026-10-25)

Two capped searches (15 searches / 25 fetches each): `addendum_vision_light.md`,
`addendum_validation_behaviour.md`. Most publisher full texts stayed closed; none of the dominant
gaps (herring reactive distance, gadoid capture success on fish, a measured PAR profile below 50 m)
is closed.

- Herring: Thetmeyer & Kils 1995 [F] support v 1.0 BL/s and half the visual field; a Coregonus
  proxy Ke ~10 lx gives a 0.007 -> ~0.006.
- Capture: Ellis & Gibson 1997 [F] cod p ~1 up to prey:predator length 0.33 (lab upper bound, central
  p kept); Staniland 1995 whiting on shrimp 0.8 / 0.4 -> predators -> zooplankton a at most 0.0024.
- Light: Arst 1993 [F] Kd ~0.20 in the top 20 m offshore Skagerrak vs the 0.30 used (Aas 2014
  Secchi regression). At 0.20-0.25 the benthivore fish pairs rise x1.2-1.5 (0.016 -> 0.020-0.025,
  0.022 -> 0.027-0.033). Not yet confirmed by re-running scripts/benth_encounter.py.
- Verified: the Beauchamp et al. 1999 RD equation; Macy et al. 1998 mackerel threshold 1e-7
  uE m-2 s-1.
- Validation, Rindorf & Gislason 2005 [F]: half-saturation converted to the pool gives a 0.0063 vs
  the derived 0.0056 (passes, but three judgement factors leave 0.0005-0.10, so it cannot separate
  the variants or exclude 0.04); Type II shape confirmed; fish are eaten "almost exclusively during
  dawn and dusk" - argues for the dispersed predator variant (night/day 0.35) over 1.25; gadoid prey
  half-saturate ~20x the clupeids, implying predators -> benthivores a ~0.03 vs the derived 0.18
  (weak, mixed prey); whiting intake plateaus at 10-17 % of the maximum ration (a check of the eat
  share after training, not of a).
- Speeds: Onsrud et al. 2005 [S] whiting 0.6 BL/s, night x0.7 -> predators -> planktivores
  0.0041-0.0054; pout 1.3 BL/s changes the planktivore pair +2 %. Winter sprat at ~50 m by day: -1 %.
- Still missing: herring reactive distance (Varpe & Fiksen 2010 table), Meager 2005/2010, a Skagerrak
  PAR profile below 50 m, Rindorf 2002 gear / swept area (would narrow the validation ~1 decade),
  Onsrud 2005, Albert 1994, Utne 2012 full texts, Hiddink 2016 station data (Dryad token).

## Re-run: benthivore fish pairs at a clearer top 20 m (2026-10-25)

`scripts/benth_encounter.py` got an optional 4th argument `kd_top` (Kd of the top 20 m in the central
light profile; default 0.30 reproduces `tables/benth_encounter_stdout.txt` byte for byte). Deeper
layers kept (0.12 at 20-50 m, 0.06 below); the lowK / highK profiles unchanged. Outputs:
`tables/benth_encounter_kd{0.30,0.25,0.20}_stdout.txt`.

| Kd top 20 m | median daylight bottom lux | -> planktivores a (MC 5-95 %) | 1/(a h) | night/day | -> predators a (MC 5-95 %) | 1/(a h) | night/day |
|---|---|---|---|---|---|---|---|
| 0.30 | 0.0034 | 0.0160 (0.0013-0.069) | 0.24 | 0.08 | 0.0219 (0.0018-0.089) | 0.24 | 0.12 |
| 0.25 | 0.0093 | 0.0191 (0.0013-0.071) | 0.20 | 0.06 | 0.0259 (0.0018-0.092) | 0.20 | 0.10 |
| 0.20 | 0.025 | 0.0219 (0.0013-0.076) | 0.17 | 0.06 | 0.0297 (0.0018-0.096) | 0.17 | 0.09 |

Central a rises x1.19 (0.25) and x1.37 (0.20) on both pairs, slightly below the addendum's hand
estimate (x1.2-1.5); the MC range barely moves (the profile is one of three drawn equally, and the
other factors dominate). The benthos pair is light-independent and unchanged (0.255). Night/day falls
a little as the day gets brighter. The fish pairs stay at 0.4-0.75 of the library 0.04 and near the knee.

## Applied to the library (resume section 220, 2026-10-26)

| Pair | derived (variant used) | applied a | dark_ratio / threshold |
|---|---|---|---|
| planktivores -> zoo | 0.017, foraging-fish overlap (`planktivores_a.py --foraging-overlap`) | 0.017 | 0.02 / 0 deg |
| benthivores -> benthos | 0.26 | 0.26 | none |
| benthivores -> planktivores | 0.019 at Kd 0.25 | 0.019 | 0.06 / 0 deg |
| benthivores -> predators | 0.026 at Kd 0.25 | 0.026 | 0.10 / 0 deg |
| predators -> planktivores | 0.012 dispersed (schooling left to the prey's hide action) | 0.012 | 0.35 / -5 deg |
| predators -> zoo | 0.0015-0.0024 (A17 pool, no krill) | 0.002 | 0.5 / -5 deg |
| predators -> benthos | 0.030 | 0.030 | none |
| predators -> benthivores | 0.18 (conflict with Rindorf & Gislason ~0.03 kept open) | 0.18 | 0.6 / -5 deg |
