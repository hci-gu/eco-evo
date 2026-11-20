# Krill patchiness at the model grain: evidence trail (2026-11-20)

Research subagent with two literature subagents. Literature and data only: no model run, and nothing in the library,
manifest, code, tests, AGENTS.md or the resume was changed. No value is chosen from model output; section E of
`tables/model_grain_stdout.txt` uses the resume-289 decomposition only as a yardstick for effect size.

Tags: [F] full text read, [A] abstract only, [S] secondary (including full texts read only through a summarising
fetch, written "[S, fetch summary]" in `q1_evidence.md`), [D] derived here, [J] judgement.

## Files
- `q1_evidence.md`: literature trail for Q1 (patch size, persistence, occupancy, day/night, against). Rows in
  `tables/q1_raw.csv`.
- `q2_evidence.md`: literature trail for Q2 (do gadoids track krill), with its own ICES haul statistics. Rows in
  `tables/q2_raw.csv`.
- This file: the stomach and net-haul arithmetic done here (S1-S3), the reconciliation, and the model-grain
  conversion (G). Rows in `tables/main_rows.csv`.
- `tables/evidence.csv`: all rows merged (`scripts/merge_rows.py`).
- Scripts (run with `python3 -I`, paths as arguments): `stomach_occupancy.py`, `haul_spread.py`, `model_grain.py`,
  `merge_rows.py`; q1_*.py and q2_*.py from the literature agents. Stdout in `tables/*_stdout.txt`.
- `data/`: downloads, untrusted, not to be committed. Licences not checked.

## S1. ICES stomach DB: between-haul vs within-haul krill occurrence [D]

`scripts/stomach_occupancy.py` on `Krill över grunda bottnar/stomachs/tables/records_linked.csv.gz` (NE and NW North
Sea, Skagerrak, Kattegat). The haul is the unit. A haul counts only when the species has at least one record with
prey rows (hauls with no prey rows at all are dropped: saithe 132, whiting 28, cod 80). It is krill-positive when any
record holds krill. Whiting >= 20 cm. The 2022 whiting uploads are reported separately (mostly records without prey
rows). Hauls with krill present but unquantified (saithe 139) are left out of the ration ratios, on both sides.
Ration index per record sqrt(S/S_max), S_max 2.5 %BW, krill 0.19 g per counted krill (the central band of
"Krillintag och jaktsätt"); the band factor cancels in every ratio below.

| species, selection | hauls | P_h (share of hauls with krill) | per-fish FO | within-rectangle share of presence variance | ration all / positive hauls | ration all / krill-dominated |
|---|---|---|---|---|---|---|
| whiting >= 20 cm, 1981-91 | 112 | 0.35; strata (>= 5 hauls) 0.10-0.62, median 0.18 | single fish 0.19 (n 43) | 0.41 vs null 0.63 (22 rect-year-quarter groups, 8 mixed) | 0.45 | 0.22 |
| whiting, krill strata (NE NS Q1 1981, NW NS Q1 1981, Skagerrak Q2 1991) | 54 | 0.53-0.62 | - | - | 0.57-0.71 | 0.28-0.43 |
| saithe, 1981-87 and 2026 | 322 | 0.52; strata median 0.50 (19 of 20 between 0.1 and 0.9) | single fish 0.34 (n 122); beside a krill-eating haul-mate (leave-one-out) 0.55 (n 20) | 0.17 vs null 0.59 (76 groups, 11 mixed) | >= 0.42 (183 hauls; lower bound) | >= 0.23 (lower bound) |
| Norway pout, 1980 | 48 | 0.75 | - | - | 0.86 | 0.52 |
| cod (proxy), 1981-2025 | 260 | 0.07 (0.43 in NE NS Q1 1981) | leave-one-out 0.18 (n 28) | 0.25 vs null 0.76 | 0.16 | 0.05 |
| haddock (proxy), 2025 | 45 | 0.04 | - | - | 0.04 | - |
| whiting 2022 uploads | 18 | 0.06 | - | - | - | - |

ICC of krill presence among single saithe 0.39 (19 hauls; thin). The null of the within-rectangle share is
(N - G)/(N - 1), the share expected if presence ignored the rectangle, since the groups hold only 2-3 hauls. The 139
saithe hauls left out of the ration ratios are all krill-positive, so the saithe ration ratios are lower bounds.

The Q2 agent's independent script on the same records, restricted to cruises with any krill and to saithe 35-70 cm
(`tables/q2_haul_krill_stats2_stdout.txt`), gives:
- share of hauls with krill: whiting 0.31, saithe 0.57;
- length-class clustering inside hauls (whiting 0.67 vs 0.33 expected; ICC 0.51 whiting, 0.43 saithe);
- top 10 % of hauls hold 0.84-0.91 of the summed krill;
- same rectangle within 3 days, P(second haul has krill | first has): saithe 0.85 vs 0.59 marginal; whiting 0.34 vs
  0.39 (29 ordered pairs).

The two analyses agree.

Reading [D/J]:
- **Saithe.** Krill presence is a between-haul property, and mostly a BETWEEN-RECTANGLE one.
  - The within-rectangle share of presence variance is 0.17, against 0.59 expected without clustering. A rectangle is
    about 55 x 55 km, and the comparison is within year and quarter.
  - A second haul in the same rectangle within 3 days is krill-positive 0.85 of the time, against 0.59 marginal.
  - So the saithe signal is mainly regional or bed-depth structure at >= tens of km and weeks. That is gradient-type
    structure, which the model's depth table and season carry in part.
- **Whiting.** Presence is also clustered by rectangle, though less than in saithe: within share 0.41 against a null
  of 0.63 (58 hauls in 22 groups). The same-rectangle pair test disagrees on thin data (0.34 vs 0.39 marginal, 29
  ordered pairs). Some sub-rectangle variation is possible for whiting, but it is not shown.
- **Within hauls (saithe).** Only about half the single saithe whose haul-mate ate krill held krill themselves
  (leave-one-out 0.55, n 20). Fish caught together had not all fed in the same krill over the past day. Too thin to
  size a patch from.
- **Ration ratios.**
  - Population ration / ration in krill hauls: 0.42-0.45, rising to 0.57-0.71 in whiting's krill strata.
  - Population ration / krill-dominated records: 0.22-0.23, and 0.28-0.43 in the krill strata.
  - Central with central, this matches the brief's whiting 0.09-0.17 against 0.75 (0.15-0.27).
- **Caveats.**
  - Pooled 1980s records (Number > 1) hide zeros inside a pool.
  - The square root of a pool mean overstates the ration slightly (Jensen).
  - The stomach statistic is occupancy x P(eats | in krill). Inside a continuous fjord layer the per-fish FO of
    whiting was 0.35-0.80 (Onsrud 2004 [F], `q2_evidence.md`).
  - Most 1980s hauls have no bed depth, so the bed gradient cannot be removed.

## S2. Repeated net hauls and station spread in and near the box [D on F]

`scripts/haul_spread.py`, stdout `tables/haul_spread_stdout.txt`.

**Bøhle & Moksness 1991 Table 1, fixed stations** (transcribed in "Krill i Skagerrak").
- Gear: IKMT, night, 5-65 m, ~2.3 km of tow; repeated across the cruises of one winter.
- 14 of 22 station-winters had at least one zero haul.
- Two-state occupancy p2 = mean/max. Its floor is 1/n, so only stations with n >= 5 are used: median 0.46 (0.19-0.67),
  13 stations.
- By bed depth: median 0.49 over beds >= 150 m, 0.35 over beds < 150 m.
- What it measures: time at a point at a weeks lag, in the upper 65 m at night. A zero can be a layer that stayed deep,
  so this is an upper bound on horizontal patchiness.

**Bøhle in-box station-winter means, between stations.**
- CV 1.62 (p_eff 0.28).
- Without the one 30 l haul (st. 51, 1989-90, flagged by the authors): CV 0.81 (p_eff 0.61).
- The low value rests on one haul.

**IMR NSEC macroplankton trawl, April-May 2023/25, stations tens of km apart.**
- West/north trench: CV 0.46 (p_eff 0.83).
- East/Skagerrak: CV 2.02 (p_eff 0.20), driven by st. 418 (440).
- All stations without st. 418: CV 0.66 (p_eff 0.70).
- Again the low value rests on one haul.

## S3. Detection window of krill in a gadoid stomach [D]

WGSAM 2024 GER (whiting parameters, which the library also uses for saithe), at 28 cm, 8.4 C and krill 4.3-6 kJ/g:
- one 0.2 g krill is 90 % evacuated in 7-9 h;
- a 1 %BW krill meal in 22-29 h;
- a 3 %BW krill meal in 37-49 h.

Hard parts may stay identifiable longer [J]. Rindorf 2003 [F]: krill "disappeared completely" from stomachs at fixed
stations after 24, 31 and 42 h.

Detection window W: 12 / 24 / 36 h, which is 2 / 4 / 6 ticks.

## G. Conversion to the model grain [D/J]

`scripts/model_grain.py`, stdout `tables/model_grain_stdout.txt`.

1. **Two-state equivalence.** For a predator saturated inside patches, a field with CV c acts like occupancy
   p_eff = 1/(1 + c^2) at density m/p_eff. Here 1 + c^2 is Lloyd's patchiness.
2. **Change of support.** From the ~1.5 km Gullmarsfjord transect to the 1 km cell, with g(x) the exponential
   line-average factor:
   - CV 0.54 by day becomes 0.54-0.56 at 1 km (p_eff 0.76-0.77) for L of 2-10 km;
   - CV 0.35 at night becomes 0.35-0.36 (p_eff 0.88-0.89).
   - The swarm nugget (50-120 m swarms) is sub-cell and averages out.
3. **Decorrelation at a fixed cell.** T_E = 1/(1/T_bio + U/L). With L 5-10 km, U 5-15 cm/s and T_bio 1-3 d:
   - T_E is 7-31 h, i.e. 1.1-5.2 ticks, and rho(6 h) is 0.41-0.83.
   - The high end of 8 ticks needs fjord-like currents (2 cm/s, L 10 km, T_bio 3 d: 47 h). It also matches the
     Gullmarsfjord r of 0.78-0.99 over 32-49 h after the one reorganisation.
   - Field anchors: Gullmarsfjord r 0.62-0.85 at 6-12 h, with one reorganisation inside 3 d; Scotian basin means
     swing x2-4 over 1-3 d; stomach krill vanished after 24-42 h.
4. **Stomach-side p per cell-tick.**
   - Inputs: P 0.20 / 0.35 / 0.50, W 12-36 h, rho_tick 0.44-0.70 [J inputs].
   - Result: p 0.06-0.42, central ~0.2.
   - This p is a lower bound on spatial occupancy, because P is occupancy x P(eats | in krill).
5. **Effect size, a yardstick only.** A two-state krill term that the predator does not track, with the other prey
   smooth, and X and the krill share from "Intag under födosök" section 0:

   | p | 0.9 | 0.77 | 0.6 | 0.45 | 0.3 | 0.2 | 0.1 |
   |---|---|---|---|---|---|---|---|
   | whiting potential (relative to smooth) | x0.97 | x0.93 | x0.87 | x0.81 | x0.74 | x0.70 | x0.64 |
   | water column potential | x0.99 | x0.98 | x0.97 | x0.95 | x0.93 | x0.91 | x0.88 |

   - Whiting: the foraging-time factor's x0.66 (potential 0.49) is reached only at p <= 0.13. The x0.52 (potential
     0.39) is not reachable at any p; the floor at p -> 0 is x0.586, a potential of 0.43.
   - Water column: x0.86 even at p 0.05.
   - Reading: at the acoustic p (0.77-0.9) whiting moves by x0.93-0.97. At the stomach p (0.1-0.3) it moves by
     x0.64-0.74, a potential of 0.47-0.55. That range lies inside the refill-season bound of 0.44-0.55 in "Mättnad
     hos rovfisk". It holds only if the stomach p were all cell-scale patchiness and the fish did not track krill.

## Corrections to the brief and to earlier notes
- Kaartvedt et al. 2005 (MEPS 299:1) is in the Norwegian Sea (~70 N, 4 E), not the Oslofjord. The saithe under the
  swarms had eaten fish, not krill [F].
- Rindorf 2003: re-read here in the copy held in "Attackrater ur mötesteori"/data [F]. Krill "disappeared completely
  after 24, 31 and 42 h at Locations 1, 3 and 4". That is three of the five sampled locations. The Q2 agent's "3 of 4
  modelled sites" infers that krill was not modelled at Location 5; the sentence does not say so [J].
- Rindorf & Gislason 2005's lack of an aggregative response concerns FISH prey across 5 sites, not krill; E6 of
  "Intag under födosök" should not be read as krill evidence.
- The brief's pair "0.32-0.75 vs 0.09-0.17" mixes bands. 0.32/0.75 is the low/central band of the krill-dominated
  whiting mean; 0.09-0.17 spans selections at the central band (model scale). Central against central, the ratio is
  0.15-0.27, consistent with S1's 0.22-0.43.
