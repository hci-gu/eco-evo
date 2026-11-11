# Norway pout schooling - summary (2026-11-11)

Research subagent; the main agent saved this text. Nothing changed in the library,
code, tests, manifest, AGENTS.md, resume or research_notes/"Stimskydd i motes-a";
no training; nothing set from model output. Per-source evidence: `evidence.md`
(E1-E18, model facts M1-M3). Grades: [F] field, [L] lab, [A] abstract or search
summary only, [S] secondary, [D] derived here, [J] judgement.

## Files

- `evidence.md`: E1-E18, M1-M3.
- `scripts/pair_terms.json`: copied unchanged from research_notes/"Stimskydd i
  motes-a"/scripts/ (term shares of today's a by prey species and period, each
  term's rule-A radius).
- `scripts/school_factor_note_original.py` -> `_stdout.txt` (+ .json): an
  unmodified copy of that note's school_factor.py; stdout equals the note's.
- `scripts/school_factor_pout.py` -> `school_factor_pout_stdout.txt`,
  `school_factor_pout.json`: the same algebra with this note's pout and blue
  whiting parameters; also prints the school-size derivation, the note's readings
  and a sweep of pout sigma_day x (N, R_s).
- `papers/`: Unpaywall records only (Albert 1994, Onsrud 2004, Fassler 2007); no
  open PDF found, none placed there. Aglen 1996 stays [A] (its DOI resolved to
  another paper; the record was removed).
- Texts read from copies already in research_notes/ (Albert 1994, Albert 1995,
  Raitt 1968, Rindorf 2003, Rindorf & Gislason 2005) and from open web PDFs via
  WebFetch (ICES CM papers Pedersen et al. 2004, Adlerstein & Ehrich 2000,
  Petrakis et al. 2000; ICES FishMap factsheet; HERAS 2022; Fassler 2010 thesis),
  text extracted in the session scratchpad.

```
python3 -I school_factor_pout.py pair_terms.json school_factor_pout.json
python3 -I school_factor_note_original.py pair_terms.json school_factor_note_original.json
```

## Question

Norway pout is 0.35 of the planktivore pool and carries 0.52-0.54 of the WC and
whiting planktivore pair a and 0.85 of the benthivore pair. Does it school; how big
and dense are its schools; what share of its biomass is in schools by day and by
night; what does that do to the schooling factor F on the fish -> planktivore
pairs? The schooling note left pout at sigma day 0.5 / night 0.25, N 5000, R_s
8 m, all [J], giving F between its central (0.57 / 0.83 / 0.69 for WC / whiting /
benthivores) and clupeids-only (0.77 / 0.88 / 0.91) readings.

## Findings

### 1. Does pout school? Two states, only one observed in the box

- Night [F]: a dispersed near-bed layer, not schools. Norwegian Deep incl. the
  Skagerrak, echo recordings checked against catches over 16 cruises 1984-87
  (Albert 1994, E1): "At night the recordings were typically more or less
  homogeneous in the lowest 5-10 m of the water column ... the traces were often
  present continuously at relatively high density over long distances". Off
  Lofoten "clean Norway pout (lower 25 meters)" were recorded "largely as resolved
  single targets" (E4). In the Oslofjord pout are part of the dusk-rising
  scattering layer; the schooling fish there are herring and sprat (E6).
- Day in the box: unobserved. Albert's traces vanished within 10 min of sunrise and
  returned at sunset while catches stayed equal or rose (E1): "the fish were
  distributed below the detection height of the echo sounder of 1.2-1.5 m above the
  bottom". Schools or a dense carpet there is not known.
- Day elsewhere [F]: discrete, herring-sized school echotraces near the bed
  (Fassler 2010 / Fassler et al. 2007, E5): on the summer herring survey of the NW
  North Sea "Both species occur close to the seabed in densely packed schools,
  which are similar in shape and size"; 62 pout echotraces of mostly 5-15 cm fish,
  about the herring schools' mean Sv. Selected by trawling on major schools: they
  prove the state exists, not the biomass share in it.
- Against clupeid-like schooling: Raitt 1968 (E2): "There is no evidence to
  suggest that Norway pout form 'organized' schools in the way in which herring
  do"; Rindorf & Gislason 2005 (E12) class whiting's gadoid prey, pout included, as
  "non-schooling, more demersal prey". The fishery's "dense concentrations" (E2,
  E3) and Albert's summer drop in occurrence (96 -> 82 %) are km-scale patchiness,
  which the grid already carries, not the unit a predator meets as one.

### 2. Diel cycle: sigma day and night

- Vertical position [F]: by day within 1.5 m of the bed; at dusk a 5-10 m layer,
  seldom > 20 m above the bed (E1). Dickson 1960 (via E2): "just clear of the
  bottom but are even further off the bottom by night". Off Norway near the bottom
  by day, diel migration varying between sites (E7, [A]).
- Trawl catchability [F]: GOV night/day = exp(-0.52) = 0.59 (northern North Sea,
  Nov 1997, E9); Fladen Aug 1961 "twice as big a catch by day" (E2); no time-of-day
  effect in Albert's GLM (E1) or at Albert 1995's 140 m station (E10). Fits a fish
  on the bed by day and off it at night (escaping over the headline); says nothing
  about schooling.
- Feeding: more intense at night on krill in the Deep (E1); no day/night difference
  in the Oslofjord (E6); small pout feed by day on copepods (E10).
- sigma night = 0 (0-0.1) [F: E1, E4, E6]. The note's 0.25 is not supported; it
  did not change the note's central readings (night pout terms had S >= 1, capped),
  only the high-protection reading.
- sigma day = 0.4 (0.15-0.7) [J]. For: schools exist by day (E5); the fishery
  finds "dense schools ... within a few metres of the seabed" (E3, [S]). Against:
  no school echoes off the bed by day in the box (E1), "no organized schools" (E2),
  whiting take pout around 15 h (E12).

### 3. Predation on pout

- Timing [F, small n]: whiting ingested pout in "only one peak around 15.00
  hours", unlike the dawn and dusk peaks on herring, sprat, haddock and whiting
  prey (Rindorf & Gislason 2005, E12; one site). Demersal whiting at 90 m with 85 %
  fish prey "fed primarily during daylight hours" (Pedersen 2000, via E13). If pout
  were protected by day as clupeid schools are, daytime ingestion should be rare
  (Andersen 2017: day risk <= 0.04 of night for sprat schools, Stimskydd E17).
  Points to weak day protection, on one site.
- Predators in the layer [F]: Albert's night recordings show "distinct single-fish
  echoes of larger gadoids" inside the pout layer (E1).
- Main predators: whiting, cod and saithe eat most 1+ pout ([S] E2, E3).
- Not found: saithe or cod stomach timing on pout; capture success per attack on
  pout; per-capita risk vs pout group size. Rindorf & Gislason's gadoid
  half-saturation (q >= 514 vs 20-159 for clupeids) cannot be read as protection
  (catchability unknown, E12).

### 4. Blue whiting (brief)

Johnsen & Godo 2007 [A]: over < 350 m more dispersed and higher at night. Torgersen
et al. 1997 [A]: the deep layer rises at night, dense shelf-break concentrations
stay near the bottom day and night. No school size found. sigma day kept 0.2
(0-0.5) [J]; sigma night 0.1 -> 0 (0-0.1) [A]. Effect on F at most 0.01.

### 5. School size and packing [D]

Chain printed in `school_factor_pout.py` section 1.
- Packing: ICES 38 kHz TS relations (gadoids 20 log L - 67.5, herring 20 log L -
  71.2). At the shared echotrace Sv ~ -45 dB (E5), 10 cm pout pack at 1.8 fish/m3
  (0.6-11 over the 5-95 % Sv) vs 0.57 for herring: ratio 3.1, or 1.0-1.6 if herring
  TS at depth is 3-5 dB lower (E16).
- N: (a) a Fernandes-sized school (24 x 24 x 4.3 m ellipsoid, 1300 m3) at 1.8
  fish/m3 gives ~2300 fish (730-15 000); (b) a Misund herring school of 25 000 fish
  x the density ratio gives 25 000-78 000. Used: N 10 000 (2500-50 000).
- R_s: half-length of herring-like schools (10.5-13.5 m, E5 with E14) + 0-5 m
  detection near the bed by day: 13 m (10-18) [J]. The note's 8 m was smaller than
  the school's own half-length.
- k = 1 (0.3-3), kept. A 12 cm pout weighs ~10-15 g, so one meal for a 0.7 kg
  saithe may be more than one pout; a higher k only raises S (less protection).

### 6. Recomputed F

a x F, S capped at 1; all but pout and blue whiting at the note's central values.

| pair | note central | note clupeids only | pout low | pout central | pout high | sweep: sigma_day 0.25-0.5, N 5000-50 000, R_s 8-18 |
|---|---|---|---|---|---|---|
| WC -> planktivores | 0.571 | 0.774 | 0.766 | 0.630 | 0.399 | 0.51-0.73 |
| whiting -> planktivores | 0.833 | 0.881 | 0.881 | 0.881 | 0.550 | 0.68-0.88 |
| benthivores -> planktivores | 0.687 | 0.914 | 0.914 | 0.770 | 0.417 | 0.58-0.91 |
| mackerel -> planktivores | 0.714 | 0.714 | 0.714 | 0.714 | 0.714 | 0.714 (no pout term, M1) |

WC -> mackerel stays 0.244; the pairs on juvenile benthivores and whiting stay 1.00.
Readings: pout low sigma 0.15 / 0, N 2500, R_s 18 (S > 1 capped, so equal to
clupeids-only apart from the blue whiting day term); central sigma 0.4 / 0, N
10 000, R_s 13; high sigma 0.7 / 0.1, N 50 000, R_s 10; all k = 1, herring at the
note's central values.

Night/day at pout central (note central in brackets): WC 0.059 -> 0.098 (0.109);
whiting 0.212 -> 0.247 (0.265); benthivores 0.141 -> 0.191 (0.219); mackerel keeps
its stomach ratio.

- Against the note's central, F rises on every pout pair (WC +0.06, whiting +0.05,
  benthivores +0.08): larger derived R_s (13 vs 8 m) and sigma_day 0.5 -> 0.4.
- Against clupeids-only, pout central is 0.14 lower on WC and benthivores, equal on
  whiting.
- Whiting barely moves: its rule-A day radius on pout is small (0.42 m), so a
  school with R_s >= 12 m and N <= 10 000 gives S >= 1 (no protection); only R_s
  10 m or N >= 50 000 takes that pair below 0.85.
- sigma_day x N sets the spread, not the night: at sigma_day 0.4 the sweep runs WC
  0.57-0.77 and benthivores 0.65-0.91 over N 2500-50 000 and R_s 8-18.

## Pout and blue whiting parameters

| parameter | note value | this note: central (range) | grade | basis |
|---|---|---|---|---|
| pout sigma night | 0.25 (0-0.9) | 0 (0-0.1) | [F] | homogeneous near-bed layer, resolved single targets (E1, E4, E6) |
| pout sigma day | 0.5 (0-0.9) | 0.4 (0.15-0.7) | [J] | schools exist (E5, E3) but not seen off the bed in the box (E1); "no organized schools" (E2); day predation peak (E12) |
| pout N per school | 5000 (500-50 000) | 10 000 (2500-50 000) | [D] on [F] | Sv packing 1.8 (0.6-11) /m3 x herring-like volumes (E5, E14, E15) |
| pout R_s | 8 m (5-15) | 13 m (10-18) | [D/J] | half-length of herring-sized schools + 0-5 m detection |
| pout day height | - | < 1.5 m above bed; night 5-10 m layer, < 20 m | [F] | E1 (already in the band profiles, M2) |
| pout trawl night/day | - | 0.59 (GOV); day 2x (Fladen); n.s. (Albert) | [F] | E9, E2, E1, E10 |
| k (all prey) | 1 (0.3-3) | kept | [J] | note |
| blue whiting sigma day | 0.2 (0-0.5) | 0.2 (0-0.5) | [J] | no school data |
| blue whiting sigma night | 0.1 (0-0.5) | 0 (0-0.1) | [A] | dispersed and higher at night over < 350 m (E17, E18) |

## Gaps

- Pout's day state in the box is unobserved: the bottom 1.5 m at 100-300 m is the
  vertical echosounder's dead zone. A bottom-mounted or lander echosounder, or
  trawl-camera work, would settle sigma_day.
- No pout school dimensions or counts, only Sv-derived packing; the Fassler schools
  are summer juveniles on the NW shelf, not the box's 1-group fish on the Trench
  slope.
- Predator-side data thin: no capture success per attack on pout, no saithe or cod
  stomach timing on pout, no per-capita risk vs pout group size; the 15 h whiting
  peak is one site.
- Confusion in a dense near-bed carpet (rather than discrete schools) is not
  represented by the school algebra; no data either way.
- Blue whiting in the Skagerrak: no aggregation data.
- budget_gate not re-run with these F values.

## Verdict

Night is settled: Norway pout is a dispersed near-bed layer at night, so its night
terms get no schooling factor [F: Albert 1994; Lofoten; Oslofjord]. The note's
sigma_night 0.25 should be 0; clupeids-only is right for the night.

Day is not settled for the box: pout can form herring-sized, densely packed school
echotraces near the bed by day [F, Fassler], but in the Norwegian Deep and
Skagerrak the daytime fish sit inside the bottom 1.5 m dead zone; the older
literature denies clupeid-like schooling [S/F]; the one predator time series puts
pout ingestion mid-afternoon, not in the twilight windows [F, one site].

Pair factors [D] on a [J] sigma_day: WC -> planktivores 0.63 (0.51-0.73), whiting
-> planktivores 0.88 (0.68-0.88), benthivores -> planktivores 0.77 (0.58-0.91),
mackerel -> planktivores 0.71 (no pout term). Every pair falls between the note's
central and clupeids-only readings; whiting equals clupeids-only, WC and
benthivores roughly midway. Clupeids-only (0.77 / 0.88 / 0.91) is defensible until
the day state is observed, but not established. Nothing supports the note's
high-protection pout corner (sigma >= 0.7 with N >= 50 000 at all hours).

Grade: night [F]; day sigma [J]; N and R_s [D] on [F] packing; pair factors [D],
spread set by the [J] sigma_day.

## User decisions

- 2026-11-11: the pout values enter a combined derivation with the new capture
  probability (research_notes/"Fangstsannolikhet for fiskbyten"); nothing applied.
