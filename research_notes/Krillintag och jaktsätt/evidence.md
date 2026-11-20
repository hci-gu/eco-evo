# Krill intake and foraging modes: evidence trail (2026-11-20)

Research subagent. Research only: nothing in the library, manifest, code, tests, AGENTS.md or the resume was changed;
no model output set or selected any value. Tags: [F] full text, [A] abstract only, [S] secondary, [D] derived here
(script and stdout named), [J] judgement.

Files:
- `scripts/` (run from `scripts/`, stdout in `tables/`):
  - `ices_krill_content.py`, `ices_krill_counts.py`: krill in single and pooled ICES stomach records.
  - `krill_pool_rations.py` -> `tables/krill_pool_rations_stdout.txt`: krill ration / GER Cmax in krill-dominated
    records.
  - `krill_ration_population.py` -> `tables/krill_ration_population_{all,no2022,prey}_stdout.txt`: the same over all
    stomachs, in three record selections.
  - `twilight_share.py` -> `tables/twilight_share_stdout.txt`.
  - `q1_*`, `q2_*`: the two literature subagents' helpers.
- `tables/main_rows.csv` (this file's numbers), `tables/q1_raw.csv`, `tables/q2_raw.csv` (the literature subagents'),
  `tables/evidence.csv` (all rows merged).
- `q1_evidence.md`, `q2_evidence.md`: the literature subagents' trails (quotes).
- `data/`: downloads, one folder per source, untrusted, not to be committed.

Input data reused, not re-downloaded: the processed ICES stomach records of research_notes/"Krill över grunda
bottnar"/stomachs/tables/records_linked.csv.gz (built there from the public ICES stomach database; whole North Sea,
Skagerrak, Kattegat, 1980-2026).

## Part A. In-repo evidence for Q1 (krill intake at high krill density)

### A1. The conversion: stomach content to a fraction of Cmax [D]

- The ICES gadoid gastric evacuation model (WGSAM 2024 Table 3-7 [F], read in research_notes/"Fiskgrupper i
  rutan"/predators/data/diet/sms_consumption/WGSAM_archimer_107912.txt l. 2832-2875):
  R (g/h) = rho M b e^(delta T) L^lambda E^-0.85 S^0.5 (alpha 0.5); saithe = whiting = 0.00171, 1.35, 0.081.
- research_notes/"Fiskgrupper i rutan"/predators/scripts/calc_ger_cmax.py defines the group Cmax with this model at a
  sustained fill S_max = 2.5 %BW (calibrated so cod reproduces the FB4 cod Cmax; 5 %BW the upper bound). The model's
  ceilings are 1.2 (whiting, 125 vs 104 kJ/kg/d) and 1.23 (water column, 119 vs 97) x that GER Cmax.
- At the same L and T the ration on a krill stomach over Cmax (energy) is sqrt(S / S_max) x (E_krill/4.184)^0.15 / M.
  T, rho and lambda cancel; the energy term is 0.99-1.03 for E 4-5 kJ/g. M, the krill armament relative to fish, is
  not in the WGSAM table: 1-1.5 [J] (Tanasichuk et al. 1991 [A]: Pacific hake "digested euphausiids most rapidly").
- Bands: central S_max 2.5 %, M 1; low S_max 5 %, M 1.5 (x0.47).
- Caveats:
  - A stomach sample is a snapshot. A pool of 10-25 fish from one haul with a high mean content says the local
    fish fed at about that rate over the last ~day (the gut integrates 1-2 days); it is not a 24-h round-the-clock
    ration.
  - Counted krill are converted at 0.15-0.23 g fresh; the weighed records give 0.09-0.30 g, median ~0.18 g
    (Skagerrak April 1991 whiting; northern North Sea 1982 saithe) [D]. Partly digested krill weigh less, so the
    counted rows lean high.
  - Pooled 1980-82 records give the lower bound of a length class; midpoints assumed (saithe +5 cm, others +2.5)
    [J]. sqrt of a pool mean slightly overstates the mean of sqrt.
  - Krill-dominated = krill >= 0.8 of the counted prey items. This selects fish that met krill; it is the "where
    krill is dense" subset the question asks for, not a random sample. No krill density was measured at any haul.

### A2. Krill-dominated stomachs: krill alone brings whiting and saithe to about Cmax [D]

`tables/krill_pool_rations_stdout.txt`:

| species | records / stomachs | krill ration / GER Cmax, stomach-weighted mean (low band) | top records | share of records >= 1 |
|---|---|---|---|---|
| whiting >= 20 cm | 44 / 244 | 0.90 (0.38) | 2.29 (0.96): 10 stomachs, 32.5 cm, NW North Sea Jan 1981, S 10-16 %BW | 0.34 |
| saithe 35-70 cm | 48 / 553 | 0.58 (0.24) | 1.81 (0.76): 10 stomachs, 45 cm, NW North Sea Oct 1982, 425 krill per stomach, S 6.5-10 %BW; 1.26 (0.53): 25 stomachs, NE North Sea Nov 1981 | 0.12 |
| Norway pout | 35 / 465 | 0.33 (0.15) | 1.00 (0.47) | 0.03 |
| cod 15-45 cm (analogue) | 14 / 88 | 0.40 (0.19) | 0.89 | 0 |

- **On the model-ceiling scale** (divide by 1.2 for whiting, 1.23 for saithe), the stomach-weighted mean is:
  - whiting >= 20 cm: 0.75 central, 0.32 low band;
  - saithe: 0.47 central, 0.20 low band.

  This is the scale on which the brief's 0.1-0.25 and the clupeid plateau 0.12-0.20 are stated. Against lab Wmax
  (1.1-1.34 model ceilings) divide again.
- **Share of krill-dominated stomachs above a threshold, model scale** (low band / central). This is the statistic
  that discriminates:

  | species | > 0.25 | > 0.5 |
  |---|---|---|
  | whiting >= 20 cm | 0.48 / 0.89 | 0.24 / 0.62 |
  | saithe | 0.36 / 0.65 | 0.02 / 0.46 |

- **Weighed against counted krill.**
  - Whiting with weighed krill only (Skagerrak and Kattegat, April 1991; 20 records, 54 stomachs): 0.88 / 0.41 of
    GER Cmax. Counted only: 0.99 / 0.42. The headline holds on weighed records.
  - All 48 krill-dominated saithe records are counted.
  - The three weighed saithe pools (northern North Sea, Jul/Oct 1982; 9.1-13.9 g krill per stomach) have no length.
    Bracketed at 35-55 cm they give 0.17-0.42 (low band) and 0.36-0.90 (central) of the model ceiling, below the
    counted top pools (1.5).
- Single-fish maxima agree: one 21-cm whiting with 40 krill in the Oslofjord (Onsrud et al. 2004 p. 1284 [F]),
  6-12 g = 8-16 %BW [D]; one 20-cm whiting in the Skagerrak 1991 with 14.4 %BW of weighed krill [D].

### A3. The same fish over all stomachs: krill averages ~0.1 of Cmax [D]

`tables/krill_ration_population_*_stdout.txt`, all stomachs in the NE/NW North Sea, Skagerrak and Kattegat,
krill-free stomachs counted as zero. Three record selections:

| selection | whiting >= 20 cm | saithe |
|---|---|---|
| all records | 0.110 of GER Cmax (2483 stomachs) | 0.089 (4049) |
| without the 2022 IBTS uploads | 0.151 (1800) | 0.089 |
| records with prey rows only (an upper bound: true empties dropped) | 0.199 (1370) | 0.132 (2712) |

- The 2022 uploads are mostly records without prey rows. research_notes/"Krillparen mot vintermagar" already flagged
  them as unusable; in the Skagerrak and Kattegat they are 1230 of 1300 whiting records.
- On the model scale: whiting 0.09-0.17, saithe 0.07-0.11.
- By area and quarter (all records): whiting 0 to 0.37 (Skagerrak Q2 0.37; NE and NW North Sea Q1 0.28; Q3 ~0);
  saithe 0.04-0.13 in Q1-Q3 and 0.24-0.55 in Q4.
- Reconciliation with research_notes/"Krillparen mot vintermagar": its winter whiting krill ration, <= 0.15-0.3 of
  Cmax over all stomachs on the shelf, is this population-level number. The 0.75-0.9 above is the krill-dominated
  subset.
- The share of stomachs in krill-bearing records is 0-0.87 by cell: krill is met in some hauls and quarters and not
  in others.

### A4. Reading of A2-A3 [J]

- The population average (0.1-0.2 of GER Cmax) is the size of Rindorf & Gislason's "other food" range (0.08-0.22).
  But it averages fish with much krill and fish with none.
- **Whiting.** A krill plateau at <= 0.25 of the model ceiling is contradicted at the central estimate (0.89 of
  krill-dominated stomachs above it). It is not excluded at the low band, where half the stomachs (0.48) lie above
  it.
- **Saithe.** Not excluded at the low band: the mean is 0.20 and 0.36 of stomachs lie above 0.25. Contradicted only
  at the central band (0.47 mean; 0.65 above 0.25).
- So the evidence leans against a low krill plateau without excluding it. The band width comes from S_max (2.5 vs
  5 %BW) and the krill armament (1-1.5), not from the stomach data.
- It does not show a functional response: no krill density exists for any haul. It shows that krill alone can fill
  these fish to about Cmax.
- Rindorf 2003 [F] adds the time and space side: krill "disappeared completely after 24, 31 and 42 h at Locations 1,
  3 and 4" (p. 267) and "Krill appeared to be a patchily distributed prey as some periods were categorised by
  complete lack of krill in the stomach contents" (p. 273).
- Kaartvedt et al. 2005 [F] is the counter-case by size: four saithe of 60-110 cm under Norwegian Sea
  Meganyctiphanes swarms of ~200 ind/m3 (~380 000 m3, ~11 t per swarm) "had fish and the cephalopod Gonatus sp. in
  their stomachs. Apparently, the saithe preyed on planktivores attracted by the krill, rather than on the krill
  itself" (p. 3-4). The box's saithe are ~40 cm; the 1981-82 pools of 35-55 cm saithe ate krill to ~Cmax.

### A5. Twilight windows [D] (`tables/twilight_share_stdout.txt`)

- Early September at Rindorf's sites (54.5-58 N): civil twilight 0.060-0.066 of 24 h (1.4-1.6 h), nautical
  0.12-0.14 (3.0-3.3 h); daylight 0.55-0.56.
- The box (58.1 N): civil twilight 0.063-0.118 of the day by month, nautical 0.13-0.28.
- Against the clupeid plateau 0.12-0.20 of the model ceiling: a twilight-only capture window at the full rate would
  be of that size (nautical) or half of it (civil). This is a size check, not evidence for the mechanism; a full
  stomach buffers 1-2 days, so the window caps intake only if capture (not digestion) is the limit inside it [J].

## Part B. Literature (subagents)

See `q1_evidence.md` and `q2_evidence.md` for the quotes. The synthesis (summary.md) was returned to the main
agent as text for saving, since the harness blocked the subagent from writing it.
