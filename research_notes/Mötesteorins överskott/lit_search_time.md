# Search time and activity: does the search side carry the encounter-theory excess? (2026-11-20)

Literature subagent. Research only. Rows: `tables/search_time_raw.csv` (written by `scripts/st_make_csv.py`).
Tags: [F] full text read, [A] abstract (verbatim phrase), [S] secondary (another paper's citation, a held
sibling note, or a search/fetch-tool summary), [D] derived here (arithmetic shown), [J] judgement.
PROXY = not a model species. "held" = full text or note already in research_notes (not refetched).

**Important finding for the caller:** Onsrud, Kaartvedt & Breien 2005 (CJFAS 62:1822, doi:10.1139/f05-090),
listed in the brief as unobtained, is already held in full at
`research_notes/Krillens fläckighet/data/onsrud2005/paper.txt`. The chain notes already use it
(`Attackrater ur mötesteori/addendum_validation_behaviour.md`, `Observabilitetsparen/fish.md` 7.3). It was
re-read here for pause and cycle content. The NVA repository copy cannot be downloaded without authentication.

Sibling note used heavily (not redone): `research_notes/Vilans avkastning/lit_activity_budgets.md`. It covers
DST/telemetry activity budgets, satiation-dependent activity and diel feeding windows.

## Two quantities kept apart

1. **Pauses inside a search bout.** If detection continues during pauses, the pauses are search time. Then
   the error is a **factor on a** through the speed term. It is not a time share.
2. **Whole-day idle time.** This is time with neither search nor handling: rest on the bed, digesting,
   sheltering. It is a **time share** (in the model, the policy's eat share). It is not a factor on a.

Tags measure neither cleanly. DSTs measure vertical activity, acoustic positioning measures displacement
between fixes, and split-beam tracking measures speed over a 5-10 s track.

## Q1. Intermittent (saltatory, pause-travel) search

- **Kramer & McLaughlin 2001**, Am Zool 41:137, doi:10.1093/icb/41.2.137 [A, PROXY: all taxa]:
  - Animals with intermittent locomotion pause for "nearly 50% of their locomotion time (range 6-94%)".
  - "pauses increase the capacity of the sensory systems to detect relevant stimuli". Named mechanisms:
    velocity blur, relative-motion detection, foveation, attention, sensory interference.
  - Full text closed. No fish-specific table was obtained.
- **O'Brien, Evans & Browman 1989**, Oecologia 80:100, doi:10.1007/BF00789938; **O'Brien, Browman & Evans
  1990**, Am Sci 78:152 [S, search summaries only; PROXY]:
  - White crappie search only while stationary.
  - Search time rises at low temperature and falls on large prey.
  - Run/pursuit speeds 2 cm/s at 4 C and 6 cm/s at 20 C.
  - Pause/run durations were not obtained. The Springer page is behind a bot wall. O'Brien & Evans 1991
    (Verh Int Ver Limnol 24:2371, "saltatory search behavior in five species of planktivorous fish") gave
    only a title.
- **Anderson et al. 1997**, Behav Ecol 8:307, doi:10.1093/beheco/8.3.307 [S, TLDR only]:
  - Theory treating search as "time-distance" functions, with cruise search as one limit. No numbers.
- **Hill, Burrows & Hughes 2000**, J Fish Biol 56:1497, doi:10.1111/j.1095-8649.2000.tb02160.x; **2002**,
  J Fish Biol 61:1255; **2003**, J Fish Biol 63:117 [S, search summary]:
  - Juvenile plaice "search more efficiently for prey when stationary than while moving".
  - Intensive search means shorter moves, more turning per unit distance and shorter pauses.
  - No move or pause durations were obtained (Wiley 403; Bangor portal has no abstract).
- **Holmes 1979** (held; via `Attackrater ur mötesteori/benthivores_encounter.md`) [S]:
  - Plaice in the lab were active 59 % of the time on worms and 26 % on Corophium. The denominator (whole
    trial or feeding bout) was not checked.
  - The shuffle gait has an upper limit of ~10 cm/s.
  - Pleuronectids "pause frequently to scan".
- **Sogard & Olla 1996**, MEPS 133:43 (held [F] in sibling note) [S, PROXY juvenile pollock]: "Across all
  rations, fish typically swam continuously but at different rates; periods of passive sinking were not
  observed". A gadoid is a continuous-swimming searcher, not a pause-travel one.
- **Onsrud et al. 2005** [F]:
  - Track-mean speeds 2-61 cm/s. "Very few tracks had swimming speeds above 50 cm s-1. Likewise, few
    tracks had swimming speeds below 5 cm s-1."
  - So whiting, pout and herring in the krill layer were moving in nearly all tracks.
  - Caveat [J]: tracks are 5-10 s, and moving fish enter beams more often than stationary ones.
- **Up-down search cycle** (Onsrud et al. 2005 [F]; species unidentified, the authors lean to herring):
  - The one tracked cycle had a slow spiralling ascent (vavg ~11 cm/s, "lasting for more than 1 min") and
    a straight rapid descent (~39 cm/s).
  - Janssen 1981 [S via Onsrud, PROXY blueback herring]: "the fish only fed during the ascending phase of
    the cycle".
  - [D], on the [J] assumption that ascent and descent cover equal path length and that only the ascent
    searches:
    - t_desc/t_asc = 11/39 = 0.28, so search time share = 1/1.28 = **0.78**.
    - Cycle mean speed = (11 + 39 x 0.28)/1.28 = 17.2 cm/s. Effective search speed = 11 x 0.78 = 8.6 cm/s.
    - Factor vs continuous search at the cycle mean: **0.50**. Factor vs the 14-16 cm/s modal speeds:
      **0.54-0.61**.
  - This rests on one cycle of an unidentified fish. It is a possible mechanism, not a herring value.

**Algebra for pause-travel [D].** A fish moves at v_m for t_m and pauses for t_p, and detects in pauses
with radius R_p. Suppose the step d = v_m t_m is not longer than ~2 R_p, so successive scanned volumes join
into a cylinder. Then:
- New volume per cycle ~ pi R_p^2 d.
- Effective clearance per total time E = pi R_p^2 v_m t_m/(t_m + t_p) = pi R_p^2 v_bar, where v_bar is the
  mean speed over moving plus paused time.
- Relative to "continuous at v_m with moving radius R_m", the factor is (1 - f_pause) (R_p/R_m)^2.

So:
- If the chain speed is a **time-averaged** speed (telemetry displacement, or a move-plus-pause mean), the
  pauses are already inside v and no factor applies.
- If the chain speed is a **moving** speed, the factor is (1 - f_pause), ~0.5 (0.06-0.94) on Kramer &
  McLaughlin's all-taxa figure. A larger paused radius partly offsets it. That offset is the reason
  saltatory search is chosen at all [J].
- If steps are much longer than 2 R_p, E falls further, to about one scanned hemisphere per cycle.

**Answer to Q1 [J].**
- No full text gave a fish effective search rate per total time to set against continuous cruising.
- The direction is clear. Saltatory search gives E = scan area x MEAN speed, not x moving speed. Where
  the chain uses a moving speed of a pause-travel species, a is high by about 1/(1 - f_pause), ~x2 (range
  ~x1.1-x16, all-taxa range).
- The model's pelagic and krill-layer fish appear to swim continuously (Onsrud: few tracks < 5 cm/s;
  mackerel lacks a swim bladder). Juvenile pollock in the lab also swam continuously (Sogard & Olla, PROXY).
- For adult cod, pauses are unresolved. Winger's speeds (modes 0.1-0.4, means 0.7-0.9, instantaneous from
  0.0 BL/s) fit intermittent movement equally well. Either way the pauses sit inside a displacement speed
  over >= 17 s.
- The pause-travel members (flatfish) already carry a move-plus-pause mean (0.12 BL/s).
- So **within-bout pauses are not shown to be a large error source** for cod, flatfish, whiting and pout.
  The exception is a cyclic search like the herring up-down pattern (factor ~0.5-0.8, [D] on [J]).
  Saithe, hake and mackerel are not resolved either way.

## Q2. In situ activity and speed of the model species

**Provenance of the chain speeds** (checked in `Attackrater ur mötesteori/` and the held Winger text):

**Cod 0.4 BL/s.** Winger 2004 ch. 4 [F, held]:
- Ramfjord, 4 cod, Aug-Sep, 5-6 C.
- Each position is a 10-s average, and fixes are at least 17 s apart. Speed is the displacement between
  fixes.
- Modes are 0.1-0.4 BL/s, hourly means 0.7-0.9 BL/s, instantaneous speeds 0.0-5.1 BL/s.
- "Although each fish remained active throughout its diel cycle, time of day had a significant effect on
  the swimming speeds".
- [J] This is a displacement speed: pauses shorter than ~17 s are inside it, and tortuosity makes it
  lower. Positioning noise inflates the speed of a resting fish. The mode (0.4) lies below the mean, so the
  chain value is already conservative against these data.

**Cod lab.** Björnsson 1993 [S, held]: 0.6 BL/s at about half the maximum intake; 0.2 BL/s when food is
unlimited or absent. Satiated or starved fish search at one third of the speed.

**Cod, Løkkeborg 1998** [S, abstract paraphrase]: speed slightly lower during a 5-h night period. In an odour
plume, cod made a rheotactic search at a moderately higher speed. No numbers.

**Whiting 0.8 BL/s, Norway pout 0.33 BL/s.** Onsrud et al. 2005 [F]:
- Speeds in the 80-110 m layer: day/ascent modes 15-16 cm/s, night 10 cm/s; "about 1 body length s-1 by
  day and 0.7 body length s-1 by night".
- Upper 10-30 m at night: 16-24 cm/s (herring 18.7-24.2 cm; whiting mostly 15-19 cm), about 1 BL/s.
- The authors say larger whiting "somewhat lower swimming speeds were indicated".
- The whiting here are juveniles. Adult whiting: no in situ data.
- Against the chain: `Attackrater ur mötesteori/addendum_validation_behaviour.md` reads 15 cm/s at the
  model whiting length L_num 25.2 cm as **0.60 BL/s**, a ratio of 0.75 to the chain's 0.8. Onsrud's ~1 BL/s
  holds for the 14-19 cm fish actually tracked. The model length is the right one for the group, so
  **0.75 is the central ratio**.
- Pout at 0.33 BL/s is below these modes. The same addendum gives 15 cm/s at L_num 11.8 cm = 1.27 BL/s, a
  ratio of 3.9.
  Onsrud reports night pout paths "as low as 0.3 body length s-1" (via the chain note).

**Herring 1.0 BL/s.** Chain [S, held]:
- Kiel herring cruised at 1.0 BL/s in a tank (attacks 3.5 BL/s).
- Gibson & Ezzi 1985/1992: 1-1.5 BL/s when biting [S via Onsrud].
- Field feeding schools 0.73-0.94 m/s gross (2.5-3 BL/s); deep schools 1.1 BL/s.
- Overwintering (not feeding) 0.25-0.42 m/s (Huse & Ona 1996).
- In situ krill-layer fish ~1 BL/s [F].
- Herring cruise continuously in schools. The search-phase restriction above is the only search-side
  factor found.

**Saithe 1.0 BL/s.** Pedersen 2001, J Fish Biol 58:1073 [S, fetch-tool summary of the abstract; DOI not
verified]:
- North Sea split-beam tracking: mean 1.05 m/s, median 0.93 m/s, no size relation.
- 25-29 cm fish: night 1.08 m/s, day 0.72 m/s. [D] 2.7-4.0 BL/s.
- A stationary target gave an apparent 0.15 m/s.
- [J] Not clean evidence for a ratio >= 1. Ping-to-ping 3-D speeds, possible current or vessel effects, and
  a sample of moving fish.
- Armannsson & Jónsson 2012 (DST, held via sibling) [S]: less vertical activity at night and in winter.
  No share.

**Hake 0.5 BL/s (J):**
- No in situ speed or activity telemetry exists in what was found. DST studies give diel vertical
  migration only (de Pontual et al. 2012, held in sibling notes).
- Juvenile hake have a rest phase (Módica et al. 2011, held). **Gap.**

**Haddock:** no in situ speed or activity telemetry. **Gap.**

**Mackerel 1.8 BL/s.**
- Macy et al. 1998 (lab, held) [S]: feeding 1.30-1.80 BL/s, resting 0.39-0.49 BL/s.
- Godø et al. 2004, ICES JMS 61:1093 [S, fetch-tool summary]: school "migration speed varied from 0 to 6 m
  s-1" (mean westward component 1.7 m/s; fish 36.1 cm). The authors call the maximum "unrealistic unless a
  strong vessel effect is involved". **Unusable** for feeding speed.
- Nøttestad (held via chain) [S]: schools ~3.8 BL/s, over ground or through water unclear.
- Mackerel swims continuously (no swim bladder). No in situ feeding-vs-cruising speed was found.

**Whole-day activity shares** (time-share term, not a; sibling note [S, held]):
- Cod DST vertical activity: 0.13 (Jun-Aug feeding grounds), 0.23 (late Aug-Oct, nocturnal), 0.44-0.51
  (migrations). Source: Righton et al. 2000, grey literature, pointer only.
- ~0.70 of cod fish-days are "seabed" days with no vertical movement (Hobson et al. 2007). Hobson et al.
  2009 show such days can include horizontal travel.
- The Ramfjord coastal cod were active throughout the diel cycle [F].
- Plaice active 26-59 % of feeding sessions (lab, Holmes) [S].
- Hall 1987: field haddock eat 15-22 % of the maximum ration [S].
- These measure vertical activity or lab activity, not foraging time. They are of the same order as the
  brief's x0.2-0.6, but that does not by itself show that idle time explains the excess.
- **No direct "time spent foraging" estimate for any wild adult model species was found.**

## Q3. Field search rate measured together with intake

**Not found.** Searches:
- planktivore field consumption vs visual encounter model, search-time fraction in situ;
- O'Brien/Evans saltatory search;
- plaice pause-travel;
- cod telemetry with bait (Løkkeborg).

Closest items:
- Model-vs-stomach consistency checks: van Deurs et al. 2015, MEPS 520:235 (sandeel, PROXY); Aarflot et
  al. 2019 (capelin-like model). Neither measures a search rate in the field. [S, search summary; not
  read]
- Løkkeborg 1998: bait-location probability inside vs outside the odour plume. A detection result, not a
  clearance rate [S].

No study measured a wild fish's volume or area searched per hour alongside its intake. **Gap.**

## Closing estimates per model group

(i) = in situ search speed / chain speed. (ii) = share of a foraging bout actually spent searching. "Pauses"
says whether within-bout pauses are search time (already inside v, or a factor on a) or idle time. All
values are [J] unless tagged.

| Group (chain v) | (i) low / central / high | Evidence | (ii) low / central / high | Pauses |
|---|---|---|---|---|
| Benthivores: cod/haddock 0.4 BL/s | 0.5 / 1.0 / 2.0 | cod modes 0.1-0.4, hourly means 0.7-0.9 BL/s [F, displacement over >= 17 s]; lab 0.2 (satiated/unfed) to 0.6 BL/s [S]; haddock none | 0.7 / 0.9 / 1.0 | cod pauses unresolved, but inside the displacement speed either way (no extra factor on a); idle days (DST 0.13-0.5 active; 0.7 seabed days) are a time share, not a |
| Benthivores: flatfish 0.12 BL/s | 0.5 / 1.0 / 2.0 | v is already a move+pause mean (shuffle <= 10 cm/s; active 26-59 % of lab feeding sessions [S]) | 1.0 / 1.0 / 1.0 | pauses are search time (scan while stationary [S]), inside v; no further factor |
| Whiting 0.8 BL/s | 0.5 / 0.75 / 1.25 | in situ 15 cm/s by day = 0.60 BL/s at L_num 25.2 cm (chain addendum), ~1.0 BL/s at the tracked 14-19 cm; night x0.7 [F]; adults none | 0.7 / 0.9 / 1.0 | few tracks < 5 cm/s [F]: continuous; if whiting cycle up-down, factor ~0.5-0.8 [D on J] |
| Water column: saithe 1.0, hake 0.5 BL/s | 0.5 / 1.0 / 1.5 | saithe in situ >= 0.9 m/s (2.7-4 BL/s at 25-29 cm) [S, artefact-prone]; hake none | saithe 0.8 / 1.0 / 1.0; hake 0.3 / 0.7 / 1.0 | saithe continuous cruisers; hake may ambush (juvenile rest phase [S]); if so, a continuous-search a is the wrong form, not a pause factor |
| Planktivores: herring 1.0, pout 0.33 BL/s | herring 0.7 / 1.0 / 1.5; pout 1 / 3 / 4 | herring in situ ~1 BL/s [F], lab 1-1.5 [S]; pout 15 cm/s = 1.27 BL/s at L_num 11.8 cm (x3.9, chain addendum), ~0.3 BL/s at night [S] | herring 0.5 / 0.75 / 1.0; pout 0.8 / 1.0 / 1.0 | herring: if searching only on the ascent of an up-down cycle, effective speed-time 0.5-0.6 of continuous [D, one cycle, J species]; that is a factor on a |
| Mackerel 1.8 BL/s | 0.6 / 1.0 / 1.2 | lab feeding 1.3-1.8 BL/s [S]; field school speeds unusable or ambiguous | 0.7 / 1.0 / 1.0 | obligate continuous swimmer; no pauses; feeding is diel (daytime peak [S]): a time share |

**Bottom line [J].**
- **Within-bout search (speed and pauses) is not the source where the evidence reaches.**
  - Cod: the chain uses a modal displacement speed below the hourly means.
  - Flatfish: pauses are already inside v.
  - Pout: 0.33 BL/s is below the measured ~1.3 BL/s.
  - Whiting: ratio ~0.75 (x0.75 on its a; with the night x0.7 the chain addendum gets pair a x0.74-0.86).
  - Herring: a search-phase cycle could give x0.5-0.8 [D on J, one cycle].
  - None of these reaches x0.05-0.3 on a.
- **Saithe, hake, mackerel, haddock and adult whiting are not shown either way:** no usable in situ
  feeding speed or pause data. Hake may ambush, in which case a continuous-search a is the wrong form.
- **Whole-day idle time is the one candidate of the right order** in time-equivalent terms (x0.2-0.6).
  - Evidence: cod DST vertical activity 0.13-0.5, ~0.7 seabed days; satiated cod in the lab swim at one
    third of the half-ration speed.
  - Limits: it covers cod only, it is a vertical-activity proxy, and the share numbers are pointer-grade
    (Righton 2000 grey literature). The Ramfjord coastal cod were active all day.
  - It is neither confirmed nor excluded. It is shared by all of a predator's prey and belongs to the time
    share (the policies' 89-97 % foraging), not to a.
- The other shared candidates, reactive distance/scan and capture, were outside this note.

## Sought, not found / not used

- Full texts of Kramer & McLaughlin 2001, O'Brien et al. 1989/1990, Anderson et al. 1997, Hill et al.
  2000/2002/2003, Burrows & Gibson 1995 (Anim Behav 50:41, doi:10.1006/anbe.1995.0219), Løkkeborg 1998,
  Løkkeborg & Fernö 1999 (doi:10.1023/A:1007504712163). All are closed (Unpaywall is_oa false), and the
  publisher pages return 403 or a bot wall.
- Wild-cod accelerometer time-in-state shares; haddock, adult whiting, hake, Norway pout and mackerel
  activity budgets.
- Q3 field search rate with intake.
- Downloads not used:
  - `data/onsrud2005/` holds the NVA landing page and records. The file download is forbidden, and
    `resp.bin`, `resp2.bin` and `r.bin` are error JSON.
  - `data/openalex/` is empty (HTTP 429).
  - `data/upw1/`, `data/upw2/` hold Unpaywall records only (one OA PDF, Huse & Ona 1996, returned 403).
    `data/upw1/` also contains records and PDFs from another agent working in this directory at the same
    time. Of these, this note wrote only the six DOIs named in Q1/Q2 (icb_41.2.137, anbe.1998.0772,
    a_1007504712163, F05-090, tb02160.x, beheco_8.3.307).
  - `data/s2search/`, `data/crossref1/` hold search records.
