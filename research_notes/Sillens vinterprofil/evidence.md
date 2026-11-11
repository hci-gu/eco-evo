# Herring and sprat winter profile - evidence (2026-11-11)

Grades: [F] field data or full text read, [L] lab, [A] abstract only, [S] secondary (another paper's citation,
a search summary, or another note's reading), [D] derived here, [J] judgement. "Read there" means another note in
this repository read the source in full; the number is quoted from that note.

## Model facts (repository, verified 2026-11-11)

- M1. The herring and sprat band profile phi_j comes from `krill_pairs_wc.herring_prof`, used by
  "Overlapp i fiskparen"/scripts/band_pairs.py. By day it is f_b(bed) uniform in the bottom 10 m plus
  (1 - f_b) uniform 10-100 m. f_b is the HERAS July bottom-10-m share: 0.66 at 70 m, 0.10 at 95 m, 0.04 at 120 m,
  0.06 at 140 m, 0.05 at 160 m and deeper. By night the profile is uniform 0-50 m. It is not seasonal. Its
  equal-cell box mean of f_b is 0.127 (seasonal_pairs_stdout section 1).
- M2. The engine has one annual a per pair. The only time variation is a(t) = a m(t), with m from the sun elevation
  against `light_threshold_deg` and normalised to an annual mean of 1 (`lib/world/daylight.attack_multiplier`), plus
  ingestion Q10.
- M3. Benthivores -> planktivores takes a different path (benth_encounter.py). The herring band share by day is a
  constant 0.15, and the schooling note splits the pair into pout 0.85 and herring 0.15
  (school_factor_pout.py rows).
- M4. The combined central (research_notes/"Fangst och stim i fiskparen") gives a = a_lib x (p_new / p_lib) x
  F(p_new). The resulting a are WC 0.02797, whiting 0.06357, mackerel 0.04579 and benthivores 0.02905. At this
  central the gate's planktivore prey total is 0.89 and WC -> krill is 0.97 on the diet gate.
- M5. Monthly presence as a fraction of August, from research_notes/"Bestandsreglering" migration.md section 4:
  herring 0.60 in Nov-Mar, predatory_fish 0.09-0.13 in Jan-Mar, mackerel 0.01-0.02, whiting 1, benthivores
  0.71-0.73. **The herring winter presence comes from GOV Q1/Q3 biomass ratios.** It therefore includes any seasonal
  change in near-bed availability (see D2).

## Field data and literature

- E1. NS-IBTS GOV, valid hauls, Q1 (Jan-Feb) and Q3 (Aug-Sep) 2000-25, plus Q2 and Q4 1991-97, from DATRAS as
  parsed by research_notes/"Sejens djuplage" [F data, D here; scripts/gov_herring_season.py]. The GOV headline is
  ~4.6 m, so the trawl indexes the near-bed layer.
  - Box herring kg/h Q1/Q3 by bed class: < 100 m 0.18, 100-150 m 0.51, 150-250 m 1.04.
  - The 150-250 m class by length: < 20 cm 6.1 vs 0.2 kg/h, 20-25 cm 30.6 vs 10.8 (x2.8), >= 25 cm 15.8 vs 39.2
    (x0.40; the WBSS adults leave).
  - Skagerrak 200-250 m: 34.7 vs 3.2 kg/h. Skagerrak 250-300 m: 10.1 vs 0.4.
  - Length mix over 150-250 m beds: in Q1, 0.10 / 0.58 / 0.33 (< 20 / 20-25 / >= 25 cm); in Q3, 0.01 / 0.18 / 0.81.
  - In winter, NSAS 1-2 ringers of 20-25 cm sit near the bed over the deep beds, where in summer the near-bed
    herring are mostly adults.
  - Sprat is absent over beds >= 100 m in both quarters; its box Q1/Q3 ratio is 0.03 over < 100 m beds.
- E2. Time of day in the same GOV data, eastern North Sea and Skagerrak, 3-12 E, >= 56 N [F data, D here].
  - Q1, beds >= 100 m, year-demeaned log(1 + kg/h): sun > 10 deg +0.45, 0-10 deg -0.25, twilight -0.38, dark +0.10
    (n 19). Midday against low sun is therefore about x2.0.
  - Q3, beds >= 100 m: +0.10 vs -0.10, about x1.2.
  - Beds < 100 m, both quarters: dark -0.9 (Q1) and -2.3 (Q3) against day ~0. The fish leave the bed at night in
    both seasons.
  - In Jan-Feb the noon sun at 58 N reaches only 10.6-18.6 deg.
- E3. HAWG 2023 (ICES Sci Rep 5:23, doi 10.17895/ices.pub.22182034), Table 2.6.2.2: NSAS SAM catchabilities
  [F, read in research_notes/"Vitlingens fiskbyten"/data/step2_lit/gov].
  - IBTS-Q1 age 1: 0.1059 (CV 0.067). IBTS-Q3 age 1 (the same 1-ringers in August): 0.04618 (CV 0.122).
  - NSAS stock annex B.3.2 (2022) [F]: both indices are delta-GAM standardised from DATRAS in the same way.
  - The ratio is 2.29 (90 % range 1.82-2.88). Correcting for length with Walker et al. 2017 GOV efficiency gives
    1.63-2.29 (D1).
- E4. Simmonds & Rivoirard 2000, ICES CM 2000/K:32, IBTS Q1 1983-97 [F, read there: "Planktivorernas golv" E5]:
  there is a time-of-day effect for 1 and 2 ring herring, with "predictable lower catches in the dark period of the
  early morning".
- E5. ICES CM 1998/D:12, IBTS report [F by a literature sub-agent; not verified here]: the HAWG 1-ringer index of NSAS uses
  "daytime catches in all statistical rectangles ... both in the North Sea and in the Skagerrak/Kattegat", and "only
  day-light hauls are used for herring". The report gives no reason for this rule.
- E6. Woodhead 1963, ICES CM 1963 Symposium No. 18 [S: read by a sub-agent, not verified here; secondary on older work]: diurnal changes
  were "largely due to the migration of 'small' herring from the bottom at night" (Fulton 1922 / Lucas 1936);
  "small immature herring" catches fell at night (Richardson 1960). Season not stated.
- E7. Onsrud, Kaartvedt, Rostad & Klevjer 2004, ICES JMS 61:1278, doi 10.1016/j.icesjms.2004.09.005 [F, text in
  research_notes/"Vitlingens fiskbyten"/data/step2_lit/vertical]. Inner Oslofjord, 120 m site, Nov 2000 and Mar 2001
  (June pooled).
  - Herring "belonged to the deepest schools" by day. Day catches rose with depth (Fig. 5 bins 0-25 ... 75-100 m).
  - Herring fed by day "in the upper and middle part of the krill assemblage". Polychaetes and isopods in the
    stomachs show "feeding at depth during the day".
  - At night herring were most abundant in the upper 50 m.
- E8. Vestheim et al. 2014, J Plankton Res 36:503, doi 10.1093/plankt/fbt112 [S: tool extraction read by a sub-agent; PDF not
  obtained here, quotes unverified]. Bunnefjorden, 150 m, winter 2005-06.
  - Krill by day in early winter "from 90 m and all the way to the bottom".
  - Overwintering Calanus mostly at 85-145 m.
  - Sprat mixed with the deepest krill by day.
- E9. Kaartvedt, Rostad & Klevjer 2009, MEPS 390:237, doi 10.3354/meps08196 [F, papers/]: overwintering sprat in a
  150 m fjord sat deep by day, fed on Calanus in the daytime or not at all, and schooled by day.
- E10. Andersen et al. 2017, MEPS 583:195, doi 10.3354/meps12319 [F, papers/]. Bornholm Basin, BITS, March: "In the
  daylight hours ... the fishes were situated close to the bottom and thus within reach of the trawl". Citing
  Cardinale et al. 2003 and Nilsson et al. 2003, it describes clupeids "schooling close to the bottom during daytime,
  ascending and dispersing at dusk". The fish caught were herring of 15-22 cm and sprat.
- E11. Cardinale et al. 2003, Aquat Living Resour 16:283, Baltic, October [A]: larger herring sit deeper; the fish
  aggregate at the bottom by day and disperse at the surface at night.
- E12. Petrakis, MacLennan & Newton 2001, ICES JMS 58:50, doi 10.1006/jmsc.2000.0989 [A]: Scottish IYFS
  (February) shows day-night differences for juvenile herring; the abstract gives no size.
- E13. Tarling et al. 1998, Hydrobiologia 375/376:331 [A]: in the Kattegat Alkor Deep (131 m), Meganyctiphanes
  stays deep by day in both March and July. The night dispersal lasts 17:00-05:00 in March and 23:00-02:00 in July.
- E14. Falkenhaug et al. 2022, Front Mar Sci 9:779335 [F, papers/]: C. finmarchicus enters the Skagerrak in winter
  as diapausing individuals via the deep Norwegian Trench inflow, and overwinters in the Trench. No depth is given in
  metres.
- E15. Solberg et al. 2012 MEPS 464:245 and 2015 Prog Oceanogr [A]: overwintering sprat swim near the bottom at
  ~150 m in moderate hypoxia and school in midwater when the deep water is anoxic; they sit shallower at night.
- E16. Skov et al. 2000, J Avian Biol 31:135 [A/S]: winter aggregations of immature herring in the Skagerrak and
  Kattegat are "relatively stable and predictable". No depths are given.

## Derived here

- D1. Stock-wide winter/summer near-bed availability of NSAS 1-ringers: 2.29 raw (gov_herring_season section 4).
  Correcting for size gives 1.63 (Walker Absolute, 15 vs 20 cm class) to 2.29 (Walker Efficiency, flat).
- D2. Box over 150-250 m beds:
  - Q1/Q3 kg/h is 1.04 while the column presence falls to ~0.6 (M5), which implies about x1.7 near-bed availability.
  - The deep beds gain 20-25 cm NSAS fish in winter (x2.8 near the bed) and lose WBSS adults.
  - The 100-150 m class shows no rise (0.51 / 0.6 = 0.85).
  - The GOV cannot separate a horizontal shift from a vertical one inside a bed class.
- D3. Midday concentration at the bed (E2): x2.0 in Q1 against x1.2 in Q3 over beds >= 100 m.
- D4. Choice of the multiplier k on f_b: central 2 (range 1.5-3).
  - D1-D3 cluster at 1.6-2.3.
  - The upper end 3 allows for the deep-bed juvenile evidence (D2).
  - k 1 is the null.
  - [D on F; J for a single multiplier over all beds]
- D5. Winter day midwater layer: central 50-120 m. The low bracket is the July 10-100 m; the high bracket is
  70 m to bed - 10.
  - Grounds: the herring "deepest schools" by day in winter, feeding in the upper/middle krill layer > 70 m (E7);
    winter krill from 90 m down to the bed and Calanus at 85-145 m (E8); the box's krill day-layer centre at
    95-145 m (262).
  - [J on F analogues: fjords, not the box]
- D6. Months: Nov-Mar central. Brackets Dec-Feb and Oct-Apr. [J]
- D7. Night profile: 0-50 m unchanged (E7, E2 dark deficit).
- D8. Schooling sigma for herring and sprat: 0.9 by day and 0 at night, unchanged. The dense non-feeding wintering
  layers are NSS and WBSS adults outside the box ("Planktivorernas golv" E1, E2). The schooling factor F is
  recomputed with the seasonal term shares and their R_eff at the new p.
- D9. The mackerel rows of seasonal_pairs section 4b and of flow_engine cannot be interpreted. The derived
  mackerel series takes its day/night structure from rule A (night/day ~0.83), while the library keeps the stomach
  0.16, so the series comes out winter-high by construction.
- D10. Pairs, monthly swing and gate: scripts/seasonal_pairs.py -> tables/seasonal_pairs_stdout.txt,
  tables/seasonal_pairs.json, tables/gates/; and scripts/flow_engine.py -> tables/flow_engine_stdout.txt.

## Not obtained

- Any acoustic or depth-resolved winter profile of 1-2 ringer herring in the Skagerrak, Kattegat or eastern North
  Sea: a height above the bed, or a bottom-5-10-m share. There is no Q1 acoustic survey of 3.a in the ICES
  acoustic database. The list has Q1 entries only for IBWSS, 6aSPAWN and BIAS.
- A winter Meganyctiphanes or Calanus day depth for the open Skagerrak.
- PDFs of Vestheim 2014 and Petrakis 2001 (Unpaywall found no file) and of Cardinale 2003 (not open). Unpaywall
  JSON records are in papers/.
- Nielsen, Lundgren, Jensen & Staehr 1999 (DFU, vertical distribution of herring in the Sound, 1993-98): identified
  but not read.
