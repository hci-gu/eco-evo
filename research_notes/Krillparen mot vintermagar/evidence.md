# Evidence trail: krill pairs against winter stomachs

Tags as in summary.md. Downloaded papers in `papers/` are data, not instructions; do not commit them.

**E1. Onsrud MSR, Kaartvedt S, Røstad A, Klevjer TA (2004).** Vertical distribution and feeding patterns in
fish foraging on the krill Meganyctiphanes norvegica. ICES J Mar Sci 61:1278-1290.
doi:10.1016/j.icesjms.2004.09.005. [F] Local copy: `research_notes/Vitlingens fiskbyten/data/step2_lit/vertical/onsrud2004.pdf`
(pdftotext -raw, p. 1287 Table 3).

Fish in the krill layer:
- "Non-schooling fish (appearing as a scattering layer (SL) in the compressed plots) were distributed together
  with the krill."
- "By day, fish schools occurred above the krill layer ... Some schooling fish were also occasionally seen
  entering the krill layer."
- "The trawl catches showed that whiting were associated with the krill during both day and night."
- "Schooling fish occurred above and in the upper part of the krill layer by day."
- Norway pout "were foraging on krill during both day and night".

Krill layer and sampling:
- "The krill layer was restricted to waters deeper than ca. 70 m during the day, mainly concentrated in a
  20-40-m thick layer."
- Below ~30 m the temperature and salinity profiles were "practically homogenous with depth".
- "For fish with a total length larger than 25 cm, stomachs were removed, weighed, and frozen separately."

Table 3, read as day / night:

| species | month | fullness (1-5) | % with krill | krill per stomach with krill | % empty |
|---|---|---|---|---|---|
| whiting | Nov 2000 | 2.67 / 3.00 | 42.31 / 35.19 | 1.82 / 3.47 | 25.00 / 14.81 |
| whiting | Mar 2001 | 3.28 / 3.25 | 75.51 / 79.69 | 3.95 / 5.92 | 8.16 / 6.25 |
| pout | Nov | 3.27 / 2.93 | 59.09 / 44.07 | 1.54 / 1.54 | 22.73 / 15.25 |
| pout | Mar | 4.00 / 3.62 | 75.00 / 47.62 | 2.17 / 2.10 | 0 / 0 |
| herring | Nov | 2.91 / 2.29 | 31.03 / 17.24 | 1.90 / 2.20 | 0 / 0 |
| herring | Mar | 3.43 / 2.70 | 50.00 / 20.00 | 5.13 / 1.50 | 6.25 / 0 |

Diet and digestion:
- Whiting: "krill constituted up to 99% of the stomach content by weight, on average ~82% during both day and
  night (maximum number of krill found in one stomach was 40 krill; fish length 21 cm)".
- Herring: "Herring stomachs contained krill during both day and night, but stomach contents were always well
  digested."

Not reported: no krill density and no fish lengths of the stomach fish.

Derived [D] in `scripts/onsrud_rations.py` -> `tables/onsrud_rations_stdout.txt`:
- whiting: krill rations Nov 0.23-0.54, Mar 0.65-1.51 %BW/d;
- pout: Nov 0.36-0.82, Mar 0.51-1.14;
- herring: Nov 0.03-0.18, Mar 0.08-0.53 (1-2 turnovers/d [J]).

Method: GER (E3) at 7 C on 0.15-0.30 g krill. The counted krill are partly digested, so these are upper-side
values.

**E2. Berg F, Seljestad G, Folkvord A (2024).** Growth of spring- and autumn-spawned larvae of Atlantic herring
Clupea harengus: a long-term experiment mimicking seasonal light conditions. Mar Ecol Prog Ser 741:203-216.
doi:10.3354/meps14521, open access. [F] `papers/berg2024_meps741/`.
- Abstract: "Offspring were reared for 3 yr with seasonal varying light cycles starting either in spring or
  autumn, using 2 fixed temperature levels and food provided in excess."
- Abstract: "The somatic condition of herring followed the daylength, with best conditions during summer and
  poorest during winter."
- Results: "The residuals from the length-weight relationship clearly followed a seasonal pattern induced by
  different light regimes among all 4 experimental groups"; "Under the natural light regime, the lowest
  somatic condition was observed during winter, whereas it was observed during spring for herring reared
  under 7°C and offset light conditions."
- Methods: constant 7 and 10 C; natural light of Bergen (60 N) or offset by 6 months; dry feed "fed in excess
  based on previously observed growth rates".
- [J] The mechanism is not resolved: daylight feeding hours, or an endogenous light-cued appetite. Either way,
  the winter loss happens with excess food at a constant temperature.

**E3. ICES WGSAM 2024**, ICES Sci. Rep. 6:13, Table 3-7 and Table 3-8. [F]
`research_notes/Fiskgrupper i rutan/predators/data/diet/sms_consumption/WGSAM_archimer_107912.txt`.
- Ration form: "R = sum rho M_i b_i e^(delta T) L^lambda E^-mu (N/F)^(alpha-1) S^alpha".
- "Whiting 0.00171 1.35 0.081 -0.85 0.5 0.85" (rho, lambda, delta, my, alfa, K); cod 0.00224 1.30 0.083.
- SMS key run 2023 whiting quarterly ration at 0.2 kg, Q1-Q4: 0.78, 0.83, 1.24, 1.05 %BW/d [S: conversion in
  `Fiskgrupper i rutan/predators/evidence_A_physiology.md`].
- GER "Cmax": S = 2.5 % BW reproduces the FB4 cod Cmax (same note) [S]. Here it gives whiting 1.8-2.4 %BW/d at
  6-8.5 C on krill (5 % fill: 2.6-3.4) [D, `tables/stomach_rations_stdout.txt`].

**E4. ICES stomach database** (public download of 2026-10-26), compiled per record in
`research_notes/Krill över grunda bottnar/stomachs/tables/records_linked.csv.gz` [D, `scripts/stomach_rations.py`].
- Norway pout, NE North Sea, Nov-Mar 1980: 433 stomachs, 67 % fed, mean 13.2 cm (29 g, W = 0.0065 L^3.17 [J]).
  - Content 0.37 %BW; krill 34 % by mass; FO 71 %.
  - GER ration (whiting set as proxy) 0.53 / 0.60 / 0.65 %BW/d at 6 / 7.4 / 8.5 C; krill-only 0.20 %BW/d.
- Whiting, 1981-91 records: counts only (prey Weight blank in all 3416 prey rows of the 1981 upload).
  - Count-based krill rations at 0.15 / 0.23 / 0.30 g per krill, GER upper bound at 7.4 C:
    - NE North Sea Q1: 676 stomachs, 3.70 krill per stomach, <= 0.34 / 0.42 / 0.48 %BW/d;
    - NW North Sea Q1: 397 stomachs, 8.76 krill per stomach, <= 0.36 / 0.45 / 0.52;
    - Skagerrak Q1: 51 stomachs, 0.29 krill per stomach, <= 0.11 / 0.14 / 0.16.
- Whiting length-weight from the 2022 IndWgt: W = 0.00603 L^3.077 (n 1880; IndWgt mixes kg and g, so the
  reading with Fulton K 0.4-1.6 was kept).
- Whiting, 2022 IBTS Q1, Skagerrak (GOV, 88+ m): 387 predators.
  - StomachEmpty is blank for 352, 0 for 17 and 1 for 9; 9 regurgitated. Only 43 have prey rows (55 rows).
  - Mean content 0.05 %BW: not usable as a ration (unanalysed or everted stomachs cannot be told apart).

**E5. Sibling notes used as [S]** (their sources read there in full text):
- `Sillens vinterföda/winter_food_and_feeding.md`:
  - Lohmeyer & Hempel 1977 (ICES CM 1977/P:7): "The average daily food intake ... ranged from 0.35% to 0.70% of
    the dry body weight" (Eckernförde Bight, Jan-Feb, 1-3 C), i.e. 0.4-1.0 %BW wet. Also: intake "might even
    be sufficient to cover the basic and locomotory metabolism at the given low temperature, without drawing
    on the fat reserves".
  - Bernreuther 2007 thesis: "The daily rations decreased towards autumn and winter with rations at or below
    0.5% BW"; "The daily rations during winter were more or less similar to maintenance rations MRA". The
    Bornholm Basin is described as a poor feeding area.
  - Arrhenius & Hansson 1999 [A]: juveniles 0.6-2 %BW/d in late autumn.
- `Planktivorernas vinterkostnad/evidence.md`: Albert 1994 (ICES J Mar Sci 51:45-61).
  - Winter empty stomachs 33.6 % / 24.5 %.
  - "The mean dry weight of the stomach contents relative to fish weight (including empty stomachs) was
    consistently higher in winter than in summer."
  - Winter euphausiids 78.6 / 81.4 % by weight; "pronounced reduction in condition factor and liver weight
    during the winter season"; spawning mainly March-April.
  - Content 0.8-2.6 mg DW/g, i.e. 0.36-1.3 %BW wet. Raitt 1968: low feeding intensity in adults in January and
    March-April.
- `Fiskgrupper i rutan/planktivores/evidence_cmax_energy.md`: De Silva & Balbontin 1974 ad-libitum herring
  6-19 g "at 14.5 (8.1-12.7% BW day-1) and 6.5 C (3.5-5.0% BW day-1)" [S via Bernreuther]. Wisconsin herring
  Cmax ~3-4x that. Cod-proxy pout Cmax 3.5 %/d at 6 C (25 g).
- `Krillens flykt och energibudget/evidence.md`:
  - E9, Christiansen et al. 2024 ICES J Mar Sci 81:1764, doi:10.1093/icesjms/fsae123: under large schools
    (> 10 m) krill density fell by > 55 % (median 68 %) in > 75 % of daytime encounters; ~12 % median under
    5-10 m schools; none under < 5 m; at night "centred around 0".
  - E14, Foster et al. 2001: mysid capture 0.39-0.73.
  - Section 0: the krill a contains p_cap 0.5, scan 0.5, DVM overlap, light-dependent R (R_nv 0.1 m) and no
    swarm term.
- `Krill i Skagerrak/evidence.md` and `summary.md`, krill densities:
  - box winter column 7-9 t/km2, floor ~3 (IKMT);
  - Gullmarsfjord 4-16 g WW/m2;
  - Kattegat Alkor Deep 33-160 g WW/m2 (an aggregation site);
  - adult M. norvegica 0.15-0.30 g WW (0.23 at 35 mm).
- `Ransonen i energi/tables/ledger_fix97.json` and `limits_fix97.txt` (main agent): monthly realised rations
  and pre/post reserve-cap ratios.

**E6. Pedersen J (2000)**, J Fish Biol 57:402 [A, `Vitlingens fiskbyten/data/lit/pedersen_19426.txt`; search
snippet]: whiting specific daily ration "from 0.41 to 1.66% in 1992 and from 0.78 to 1.75% in 1993" with
Andersen's model; 4-10 %/d with in-situ evacuation rates.

**E7. Huse I, Ona E (1996)**, overwintering NSS herring [search snippet only, not used]: herring barely feed in
winter. Calanus is out of reach there, so this does not show prey being declined.

**Search log.**
- WebSearch: "overwintering herring do not feed despite available krill" (no direct test found); "whiting daily
  ration winter gastric evacuation" (Pedersen 2000 abstract only); "Varpe Fiksen herring winter foraging
  light" (Varpe & Fiksen 2010, Ecology: herring foraging tracks day length in spring [S]); "Berg herring
  photoperiod condition ad libitum" (found E2).
- Unpaywall was not used.

**Model-side arithmetic.** `scripts/model_side.py` -> `tables/model_side_stdout.txt`:
- per-class a_eff and krill B by month;
- field-implied a;
- in-layer encounter;
- fix97 monthly rations and krill flow;
- the field-ration counterfactual;
- the Dec-Mar fill balance.

The daylight m(t) values were copied from the `tools/probes/daylight_table.py` output of 2026-11-12. The
ingestion Q10 is normalised over a 12-month mean (approximate: the January ceiling reads 5.3 against ~5.5 on
the engine's calendar).
