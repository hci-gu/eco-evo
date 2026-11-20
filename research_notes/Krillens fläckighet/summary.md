# Krill patchiness at the model grain, and whether gadoids track it (2026-11-20)

A research subagent wrote this with two literature subagents; saved by the main agent. It draws on literature and
data only: no model run, and no change to the library, manifest, code, tests, AGENTS.md or resume. No value was
chosen from model output.

**Files**
- `evidence.md`: the stomach and net arithmetic, the conversion and the corrections.
- `q1_evidence.md`, `q2_evidence.md`: the literature trails.
- `tables/evidence.csv`: all rows.
- `scripts/` and `tables/*_stdout.txt`: the scripts and their output.
- `data/`: downloads; untrusted, not to be committed.

**Tags:** [F] full text; [A] abstract; [S] secondary, including texts read only through a fetch summary; [D] derived
here; [J] judgement.

## Question

The model's whiting and water column could eat about 0.75 of their ceiling while foraging, against about 0.4 in the
field. Krill carries 0.82 of whiting's spare capacity, and the model's krill is a smooth open pool. Two questions:
- Is krill patchy enough, at 1 km x 6 h, to explain "eats a lot when it meets krill, meets it only sometimes"?
- Do gadoids track krill patches?

Only the residual variability within a bed-depth class and season counts here. The depth table and the quarterly
season are already in the model.

## Answers

| # | Question | Answer | Region / proxy | Tag |
|---|---|---|---|---|
| Q1a | Patch size | Every regional description is a continuous diffuse day layer: Oslofjord, Skagen, Gullmarsfjord, Scotian basins. Swarms of 50-120 m occur (Norwegian Sea) but are sub-cell. Residual structure is reported at 2-5 km (St Lawrence radii ~2 x 5 km) and ~10 km (residual Moran's I < 0.4 at lag 1 on a 10 km grid, weaker at 20 km) | Region: descriptive only. Scales: proxies | F, A, F |
| Q1b | Variability at ~1-2 km | Between-transect CV of adult *M. norvegica*: 0.54 by day (0.34-0.73), 0.35 at night | Gullmarsfjord, 8-10 Sept 2003, one survey series; transect length judged at ~1-2 km (Everson 2007, Table 5) | D on F |
| Q1c | Persistence at a point | Predator and trawl voids last minutes to hours. The 1-2 km pattern held at r 0.62-0.85 over 6-12 h, reorganised once in 3 d, then held at r 0.78-0.99 over 32-49 h. Basin means swing x2-4 over 1-3 d. Krill left whiting stomachs at fixed stations after 24-42 h. Advection at 5-15 cm/s over 5-10 km gives 7-31 h | Oslofjord; Gullmarsfjord; Scotian Shelf (proxy); North Sea | F, D, D, F, D |
| Q1d | Occupancy and in-patch ratio | Layer at 1-2 km: occupancy near 1, Lloyd's index 1.1-1.5. Swarms and slope piling reach 10-120x the mean but are sub-cell or gradient-type. Net hauls at stations, weeks apart (gear and DVM inflate patchiness): mean/max median 0.46; Skagerrak stations with krill 0.18-0.54. Two low between-station p_eff values (0.20-0.28) each rest on one haul; without it, 0.61-0.70 | Gullmarsfjord; St Lawrence and Scotian (proxy); in-box IKMT 1987-91; Skagerrak April 1991; IMR trench 2023/25 | D, F, D, F |
| Q1e | Day/night, season, bed depth | Day is patchier: CV x~1.5, day variance higher at 120 kHz (F 2.44, p 0.016). Season: no regional data. Patchier over beds < 150 m (in-box nets; a gradient, already modelled) | Gullmarsfjord; in-box IKMT | F, D |
| Q2a | Fish-krill co-occurrence at km scale | No whiting or saithe coefficient at any scale. Proxies are weak, adjacent or negative: hake vs euphausiid hotspots co-occur on 2-39 % of hotspot area (3 km grid); pollock vs Thysanoessa weak, slightly negative (37 km); cod vs capelin negative below 3.5 km, positive above 4-10 km. Saithe stacked under krill swarms had eaten fish | BC hake, Bering pollock, Newfoundland cod (proxies); Norwegian Sea (Kaartvedt 2005, not the Oslofjord) | S, F, A, F |
| Q2b | Tracking speed | Not measured. Bounds only: pout and whiting swim 10-16 cm/s in the krill layer, so 1 km takes 1.7-2.8 h | Oslofjord | F, D |
| Q2c | Stomach structure | Krill is a between-haul prey. Hauls with krill: whiting >= 20 cm 0.35 (0.53-0.62 in krill strata), saithe 0.52; the top 10 % of hauls hold 0.84-0.91 of the krill. Presence clusters by ICES rectangle (~55 km, same year and quarter): within-rectangle variance share saithe 0.17 vs a null of 0.59, whiting 0.41 vs 0.63. Same rectangle within 3 d: saithe 0.85 vs 0.59 marginal; whiting 0.34 vs 0.39 (29 pairs; disagrees, thin). Only 0.55 of saithe beside a krill-eating haul-mate held krill (n 20). Inside a continuous Oslofjord layer whiting FO was 0.35-0.80, with half-full stomachs | ICES stomach DB, NE/NW North Sea and Skagerrak, 1981-91; Oslofjord | D, F |
| Q3 | Stomach vs acoustic occupancy | Not consistent at the cell grain. Stomach-implied p per cell-tick is 0.06-0.42 (central ~0.2); the acoustic p_eff is 0.76-0.89. The stomach statistic is occupancy x P(eats \| in krill) and carries rectangle-scale (regional, bed-class) structure, so it is a lower bound on occupancy. The two also differ in region, season and decade. Reading [J]: the gap is access or behaviour inside the layer plus 10-50 km structure. This rests on the one fjord measurement | derived | D, J |

## Derived model-grain values

Two columns, deliberately not blended. They differ in region (fjord vs open North Sea), season (September vs
Q1-Q3), decade (2003 vs 1981-91) and in what they measure (krill vs krill eaten).

| quantity | from acoustics (low / central / high) | from stomachs (low / central / high) | tags |
|---|---|---|---|
| occupancy p at 1 km x 6 h, day | 0.65 / 0.77 / 0.90 | 0.06 / ~0.2 / 0.42 (lower bound) | D on F; D on J inputs (rho_krill(6 h) 0.8, L 5 km) |
| occupancy at night | 0.85 / 0.89 / 0.95 | - | D on F |
| in-patch density multiplier (1/p) | 1.1 / 1.3 / 1.5 | 2.4 / 5 / 16 (includes access) | D |
| spatial correlation length | 1 / 5 / 10-15 cells | - | J on A/F proxies; the low is the Antarctic far proxy (flat beyond ~1 km) |
| decorrelation time at a cell | 1 / 3 / 8 ticks (6 h / 18 h / 2 d) | detection window 2-6 ticks | D/J. Open-Skagerrak currents (5-15 cm/s) give <= 5 ticks; 8 needs fjord-like flow or the Gullmarsfjord 32-49 h correlations |
| predator tracking rate | no evidence. Bounds: 0 (no aggregative response found) to <= 1 tick (swim speed). No central value | - | J |

## What it would do (yardstick only)

Effect on a saturating predator that does not track krill, with the other prey left smooth. X is taken from
"Intag under födosök" section 0 (resume 289). Nothing was chosen from this.

- **Whiting.**
  - At the acoustic p (0.77-0.9): x0.93-0.97 on the potential, which is negligible.
  - At the stomach p (0.1-0.3): x0.64-0.74, a potential of 0.47-0.55. That lies inside the field refill-season
    bound of 0.44-0.55 in "Mättnad hos rovfisk".
  - The x0.66 target (potential 0.49) needs p <= 0.13. A potential of 0.39 is unreachable at any p: the floor at
    p -> 0 is 0.43.
  - Any tracking moves every number back toward 1.
- **Water column.** x0.86-0.99 at any p, so krill patchiness is irrelevant to its gap.

## Verdict [J]

1. **The one cell-scale measurement shows patchiness too small to matter.** That measurement is a sill fjord over
   two September days, with a judged transect length.
   - The layer there is near-continuous at 1 km: p_eff ~0.77 by day and ~0.89 at night.
   - Swarm variance is sub-cell.
   - The cost to whiting is 3-7 % of the potential.
   - Transferring this to the open box is a judgement [J], not a measurement.
2. **The stomach pattern ("meets it sometimes") is real, but it is mostly not 1 km patchiness.**
   - It clusters at the scale of an ICES rectangle: strongly for saithe, less for whiting. That is regional or
     bed-class structure at weeks scale.
   - Everywhere it mixes occupancy with not eating krill that is present (Onsrud 2004).
3. **Whether patchiness matters is conditional.**
   - At the acoustic p: nothing.
   - At the stomach p, read as untracked cell-scale patchiness: most of whiting's gap, into the refill-season band.
   - Water column: nothing at any p.
   - Which reading holds for whiting is what the settling datum decides.
4. **Gadoids: no evidence that they track krill at km scale.** Every proxy shows weak, adjacent or negative overlap.
   The rival reading, a limit inside the layer (access, vertical position, appetite), is supported at least as well
   as patch finding.

## Settling datum still missing

Echosounder data from the box or the open Skagerrak:
- 38/120/200 kHz, by day and night, over beds of 100-250 m;
- 1 km EDSUs along a line of at least 20 km, repeated over 2-3 days, giving the residual variogram and a
  decorrelation time inside one depth class;
- ideally with whiting and saithe trawl stomachs on the same line.

Cheaper partial routes:
1. The Oslofjord WBAT raw data (PANGAEA 973718, ~1.8 TB, CC-BY) for a fixed-point autocorrelation function.
2. IMR or DTU survey raw files at 120/200 kHz from the Skagerrak (HERAS, IBTS).

## Load-bearing sources

1. Everson, Tarling & Bergström 2007, ICES JMS 64:39, doi:10.1093/icesjms/fsl010 [F; Table 5 transcribed and
   spot-checked].
2. Onsrud et al. 2004, ICES JMS 61:1278, doi:10.1016/j.icesjms.2004.09.005 [F]; Onsrud & Kaartvedt 1998, MEPS
   171:209, doi:10.3354/meps171209 [F]; Onsrud, Kaartvedt & Breien 2005, CJFAS 62:1822, doi:10.1139/f05-090 [F].
3. Rindorf 2003, MEPS 249:265, doi:10.3354/meps249265 [F, quote re-verified]; the ICES stomach DB as processed in
   "Krill över grunda bottnar" [D].
4. Kaartvedt et al. 2005, MEPS 299:1, doi:10.3354/meps299001 [F]; Christiansen et al. 2024, ICES JMS 81:1764,
   doi:10.1093/icesjms/fsae123 [S, fetch summary].
5. Proxies:
   - Simard & Lavoie 1999, CJFAS 56:1182, doi:10.1139/f99-063 [A];
   - Plourde et al. 2017, DFO CSAS Res Doc 2016/111 [F; Moran's I sentence verified];
   - Cochrane et al. 1994, MEPS 107 [F];
   - Evans et al. 2026, Ecol Appl, doi:10.1002/eap.70141 [S];
   - Ressler et al. 2014, MEPS 503:111 [F];
   - Rose & Leggett 1990, Ecology 71:33 [A];
   - Bogstad, Pennington & Vølstad 1991, ICES CM 1991/O:15 [F];
   - Klevjer et al. 2010, MEPS 409 (Antarctic, far proxy) [F].

## Corrections

- Kaartvedt et al. 2005 is the Norwegian Sea (~70 N, 4 E), not the Oslofjord as the coordinator's hint said. The
  saithe there had eaten fish.
- Rindorf 2003: krill "disappeared completely after 24, 31 and 42 h at Locations 1, 3 and 4", three of five sampled
  locations. The Q2 agent's "3 of 4 modelled sites" infers that Location 5 was not modelled for krill; the sentence
  does not say so.
- Rindorf & Gislason 2005's lack of an aggregative response concerns fish prey, not krill.
- The brief's "0.32-0.75 vs 0.09-0.17" mixes bands. Central against central, the ratio is 0.15-0.27, consistent with
  the stomach-DB 0.22-0.43.

## Gaps

- No krill acoustics, variogram or mooring series from the open Skagerrak or the trench.
- No seasonal patchiness data.
- Nothing on *Thysanoessa* in the region.
- No telemetry of any gadoid against krill.
- No per-haul krill density paired with stomachs.
- The saithe ration ratios are lower bounds: 139 krill-positive hauls had unquantified krill.
- Not obtained:
  - full texts of Simard & Lavoie 1999, McQuinn 2015, Liljebladh & Thomasson 2001, Thomasson 2003, Tarling 2001
    (Ligurian), Mackas 1997, Pennington & Vølstad 1994, Bogstad et al. 1995, Ressler 2012, Swartzman 1999,
    Rindorf 2002;
  - the Tanasichuk hake ration tables.
