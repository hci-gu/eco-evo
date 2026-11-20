# Krill intake at high density, and foraging modes, in whiting, saithe and hake (2026-11-20)

Research subagent with two literature subagents; saved by the main agent. Nothing in the library, manifest, code,
tests, AGENTS.md or the resume was changed, and nothing was committed. No value was chosen from model output. Model
ceilings appear only to put field numbers on the model's scale.

**Files**
- `evidence.md`: the in-repo trail (Part A).
- `q1_evidence.md`, `q2_evidence.md`: the literature trails, with quotes.
- `tables/evidence.csv`: all rows (question, species, quantity, low, central, high, unit, source, tag). It merges
  `main_rows.csv`, `q1_raw.csv` and `q2_raw.csv`.
- `scripts/`: run from `scripts/`, stdout in `tables/*_stdout.txt`.
- `data/`: downloads, untrusted, not to be committed.

Tags: [F] full text, [A] abstract, [S] secondary, [D] derived here, [J] judgement.

**Scales**
- "GER Cmax" is the ICES gadoid gastric-evacuation Cmax (WGSAM 2024 Table 3-7, sustained fill 2.5 %BW) that
  research_notes/"Fiskgrupper i rutan" used for the group ceilings.
- The model ceiling is 1.2 (whiting) or 1.23 (water column) x GER Cmax.
- The brief's 0.1-0.25 and the clupeid plateau 0.12-0.20 are on the model-ceiling scale. All Q1 numbers below are on
  that scale unless marked.

## Q1. Intake on krill alone at high krill density

### Verdict

- **No functional response exists.** No study measures a gadoid's or hake's intake against a measured krill density,
  in the field or the lab [F/A].
- **Indirect evidence leans against a krill plateau at 0.1-0.25, without excluding it.**
  - Where krill dominates the stomach, whiting and saithe usually eat krill above 0.25 of the model ceiling at the
    central estimate.
  - At the conservative band, about half (whiting) or a third (saithe) of those stomachs do.
- **The low field mean is not a plateau.** It is about 0.1-0.2 of Cmax over all stomachs, an average over fish with
  much krill and fish with none.

### Numbers

1. **Krill-dominated stomach records** (krill >= 0.8 of counted prey; ICES stomach DB, North Sea, Skagerrak and
   Kattegat, 1980-82 and 1991) [D].
   - Method: ratio = sqrt(S/S_max)/M, using the same GER and S_max as the group Cmax.
   - Bands: central S_max 2.5 %BW, M 1; low S_max 5 %, M 1.5. M is the krill armament [J]; Andersen 1999 [A]
     suggests about 1.

   | species | records / stomachs | mean, low / central | stomachs above 0.25, low / central | stomachs above 0.5, low / central |
   |---|---|---|---|---|
   | whiting >= 20 cm | 44 / 244 | 0.32 / 0.75 | 0.48 / 0.89 | 0.24 / 0.62 |
   | saithe 35-70 cm | 48 / 553 | 0.20 / 0.47 | 0.36 / 0.65 | 0.02 / 0.46 |

   - Whiting with weighed krill only (Skagerrak and Kattegat, April 1991): 0.41 / 0.88 of GER Cmax. The headline holds
     on weighed records.
   - Saithe:
     - All 48 krill-dominated records are counts x 0.15-0.23 g; weighed krill run 0.09-0.30 g, median 0.18.
     - The three weighed saithe pools (northern North Sea 1982, 9-14 g krill per stomach, no length; bracketed 35-55
       cm) give 0.17-0.42 at the low band and 0.36-0.90 central.
   - Norway pout: mean 0.13 / 0.27. Cod (analogue): 0.16 / 0.33.
   - Maxima:
     - top pools reach 1.5-1.9 central (0.6-0.8 low);
     - one 21-cm whiting in the Oslofjord held 40 krill, 8-16 %BW (Onsrud 2004 [F]/[D]).
   - Caveat: a pool from one haul is a snapshot of the last ~day, not a 24-h ration. The selection is "fish that met
     krill". No krill density exists for any haul.
2. **All stomachs, krill-free ones counted as zero** [D], in GER Cmax units:

   | selection | whiting >= 20 cm | saithe |
   |---|---|---|
   | all records | 0.11 | 0.09 |
   | without the 2022 IBTS uploads | 0.15 | 0.09 |
   | records with prey rows only | 0.20 | 0.13 |

   - The 2022 uploads are mostly records without prey rows, as flagged in "Krillparen mot vintermagar".
   - On the model scale: whiting 0.09-0.17, saithe 0.07-0.11.
   - By area and quarter: whiting 0-0.37, saithe 0.04-0.55. The share of stomachs in krill-bearing records runs 0-0.87.
   - Rindorf 2003 [F]: krill "disappeared completely after 24, 31 and 42 h" at three of five sites, and is "a patchily
     distributed prey".
   - This population level is the "<= 0.15-0.3 of Cmax" winter whiting krill ration in "Krillparen mot vintermagar".
     The krill-dominated subset is a different statistic.
3. **Literature** (`q1_evidence.md`). The cod and hake fractions are on cod Cmax scales, not the model scale.
   - **Barents Sea cod on 84 % krill** (Rybachy Bank, June 1988, 1.5 C; Dolgov et al. 1992 [F], quote verified):
     0.64 %BW/d by the dos Santos model, which is 0.48-0.56 of the Hansson and 0.34-0.39 of the Jobling cod Cmax [D].
     - The Russian Bajkov and Fortunatova methods give 2.1-2.2 %BW/d. They are read as inflated [J].
     - The one low station is Murman 1989: krill aggregations seen acoustically, the krill part 0.08-0.22 [D]; mixed
       diet, no density value.
   - **Pacific hake, 500 mm, 72-90 % euphausiids, 8.2 C** (Livingston 1983 [F]): 2.5 %BW/d, which the author calls
     "probably high". That is 0.56-1.1 of a cod-proxy Cmax [D/J].
   - **SW Vancouver Island hake** (Tanasichuk 1999, 2002 [A/S]): krill availability fell to 0.12-0.27 of its 1991
     level, while the euphausiid share of the ration stayed ~89 %. The ration tables were not obtained.
   - **Bering Sea pollock** (Buckley et al. 2016 [F]): euphausiid content 0.14-0.50 %BW. Only pollock >= 60 cm tracked
     acoustic krill biomass.
   - **Whiting evacuate krill like fish of equal energy** (Andersen 1999 [A]).
   - **A size limit:** four saithe of 60-110 cm under krill swarms of ~200 ind/m3 had eaten fish and squid, not krill
     (Kaartvedt et al. 2005 [F]). The box's saithe are ~40 cm, and 35-55 cm saithe in 1981-82 ate krill heavily.

### What it implies [J]

**Krill plateau**
- Whiting: a krill plateau at <= 0.25 is contradicted centrally but not excluded at the low band.
- Saithe: not excluded at the low band.
- A plateau sized to the population mean (~0.1) would reproduce the average for the wrong reason.
- The field pattern is krill met in some hauls and periods (km scale, 1-2 days) and eaten heavily there.
- The band width comes from S_max and the armament, not from the stomach data. A krill-specific GER armament, or
  Cmax, would narrow it.

**Fish prey**
- Rindorf & Gislason's plateaus stand: each fish prey 0.06-0.17, at most 0.20, of the maximum daily ration [F].
- Their mechanism is a twilight vulnerability window: fish prey were "ingested almost exclusively during dawn and
  dusk". That is a time limit per prey type, not handling.
- The twilight share at their sites is 0.06 (civil) to 0.13 (nautical) of 24 h [D]. This is a size check only: a gut
  buffering 1-2 days makes the window bind only if capture limits intake inside it.

**Mechanism**
- Supported: prey-specific time windows (fish) plus patchy krill.
- A per-prey handling plateau on krill is not supported.
- Demand-limited eating (the Onsrud winter verdict C) is not tested by any source here.

## Q2. Foraging modes and their time shares

### Verdict

- **No time budgets.** No source measures the time whiting, saithe or hake spend in each foraging mode, or the intake
  per mode.
- **What the data are:** prey-specific clock windows from stomachs by time of day, saithe activity categories from
  tags, and positions.
- **Modes are real.** The windows are prey-specific, so the field structure is modes with their own timing. The
  shares below are windows, not budgets [F/D].

### Whiting
- **Bed prey:** eaten at night at four of five North Sea sites (44-73 m beds, September).
  - The night window is 0.37-0.39 of 24 h [F/D].
  - At the 137 m site there was no diel benthic pattern and almost no crabs or flatfish. The author writes that
    whiting there "perhaps did not forage on the bottom" (Rindorf 2003 [F]). The box's median bed is 143 m. This is one
    site and one month.
- **Krill:**
  - North Sea: at dawn and from early afternoon to after dusk, little at midday [F].
  - Oslofjord, 120 m bed: krill are 82 % of the diet day and night; the fish sit in the krill layer, about h 10-50 m
    by day and in the upper 50 m at night (Onsrud 2004 [F/D]).
- **Fish prey:** at dawn and dusk in September; the window is bounded at 0.05-0.33 of 24 h, limited by 4-h sampling
  [D]. Sandeel at night in May-June [F].
- **Position:** whiting >= 20 cm are in the bed layer day and night by GOV catches, with no diel catch difference
  (Adlerstein & Ehrich 2000 [F]). "Vitlingens fiskbyten" also places whiting in a 5-20 m near-bed layer.

### Saithe
- **Tags off Iceland** (51 DSTs, 42-83 cm; Armannsson & Jonsson 2012 [F], verified): high activity by day, much lower
  at night, especially in winter. Rapid vertical runs take 0.15 of the time (0.002-0.33), with no day/night split.
- **Farm saithe** (Skilbrei & Ottera 2016 [F]), shares of fish-months: no DVM 0.38, deeper by day 0.17, deeper by
  night 0.24, away by day 0.16. They aggregate at night and disperse by day.
- **Height above the bed:** no tag measures it. The held day profile exp(-h/15 m) comes from HERAS and Neilson [S].

### Hake
- 29-40 cm fish migrate vertically across the thermocline "to forage" (de Pontual 2012/2013 [A/F]). That the ascent is
  at night is assumed [J].
- Juveniles (Mediterranean) eat euphausiids pelagically at night (a window of 0.32 of 24 h) and mysids near the bed
  by day (Modica 2011 [F]).
- Adults over 40 cm: no data.

### One brake or separate saturation?
- **Separate per-mode limits:**
  - Fish and other-food intakes are not negatively correlated across five sites (r -0.30 to 0.73 [F]). Far below the
    maximum, though, this does not discriminate.
  - The only shared-handling fit, among fish prey, "provided very poor fits"; the authors blame patchiness [F].
- **A shared soft brake:** Rindorf 2002 [A]. Partly digested food lowers the probability of a new meal "even at levels
  where the fish was physiologically able to ingest an average meal". Meal size is unchanged, so the brake acts on
  search. About 15 % of fish were severely depressed. The full text was not obtained.
- **Lean [J]:** per-mode limits set by window x encounter, plus one shared soft fullness brake on search. Each leg
  rests on one study.

### Search profiles: candidate modes [J]

The modes below are candidates. The time shares are windows, not budgets.

**Tension for whiting**
- Position data put whiting in the near-bed layer (GOV, the library's band).
- The diet data say they search the bed only at night over shallow beds. They take krill where and when the krill
  layer passes, or follow the layer in a fjord, and fish prey at twilight.
- Over the deeper half of the box the krill layer sits 50-150 m off the bed by day.

| mode | when | where |
|---|---|---|
| bed | night | h 0-2 m; possibly absent over beds deeper than ~140 m |
| krill | day and night | in the krill layer (fjord: h 10-50 m by day, upper 50 m at night), or at the transitions near the bed |
| fish | twilight | no position data |

**Saithe:** active by day and resting, aggregated, at night. The 0.15 share of rapid vertical runs is the only
measured excursion share, and it is not linked to prey.

**Effect size:** not computed. The knife-edge caveat of "Mättnad hos rovfisk" applies: the fill moves within a few
hundredths of the feeding level.

## Load-bearing sources
1. **ICES stomach DB**, as processed in "Krill över grunda bottnar" (`records_linked.csv.gz`), with the WGSAM 2024 GER
   (Table 3-7) [F]. Derived here [D].
2. **Rindorf**: Rindorf & Gislason 2005, doi:10.1016/j.jembe.2005.04.013 [F]; Rindorf 2003, doi:10.3354/meps249265
   [F]; Rindorf 2002, doi:10.1111/j.1095-8649.2002.tb00897.x [A].
3. **Rations on krill**: Dolgov, Yaragina, Ajiad, Mehl & Bogstad 1992, 5th PINRO-IMR Symposium, hdl:11250/107818 [F];
   Livingston 1983, Fish Bull 81:629 [F].
4. **Saithe tags**: Armannsson & Jonsson 2012, doi:10.1093/icesjms/fss121 [F]; Skilbrei & Ottera 2016,
   doi:10.1093/icesjms/fsv261 [F].
5. **Krill layer and swarms**: Onsrud et al. 2004, ICES JMS 61:1278 [F]; Kaartvedt et al. 2005,
   doi:10.3354/meps299001 [F].

## Open
1. **No krill density paired with a gadoid ration.** The Tanasichuk 1991/2002 and Robinson 2000 hake ration tables
   (paywalled) would show whether the hake ration was flat in krill biomass.
2. **No krill-specific armament or Cmax for whiting and saithe.** It sets the width of the Q1 band.
3. **No measured mode time budget for any species.** There is no whiting tag study, no adult hake data north of
   Biscay, and no height above the bed from any tag.
4. **Rindorf 2002 full text.** It holds the size of the fullness brake; DTU Orbit returned 403.
5. **Whiting's day and night position over beds deeper than 140 m** is unmeasured, and that is where the box's
   whiting live.
