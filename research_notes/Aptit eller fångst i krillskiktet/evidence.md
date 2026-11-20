# Evidence trail: appetite or capture inside the krill layer (whiting first)

Tags: [F] full text read, [A] abstract only, [S] secondary (another note or a review), [D] derived here
(arithmetic), [J] judgement. Proxies (any species other than whiting) are flagged "PROXY".

Files:
- this trail (main agent, sections E0-E6) and three literature trails by subagents: `lit_appetite.md`
  (appetite side: easy food by season, lab seasonal intake, condition vs ration), `lit_inlayer.md` (acquisition
  side: field functional responses, chronology, predator-krill interactions), `lit_seasonal.md` (Q2 seasonal
  rations);
- `tables/evidence.csv` (all sourced numbers, merged), `tables/*_raw.csv` (per trail);
- `scripts/` with stdout in `tables/*_stdout.txt`;
- `data/` (downloads, untrusted, not to be committed).

No model run. No model output chooses a value: the yardstick (T 0.905, X 5.45 = krill 4.47 + other 0.98, need
0.388, maintenance 0.272) is the resume 289 decomposition quoted in "Intag under födosök" and "Mättnad hos
rovfisk", used only to size effects.

## E0. The conflict, restated for whiting alone [D]

`scripts/model_scale.py` -> `tables/model_scale_stdout.txt`, section B. Model ceiling 31.34 MJ/t/tick = 125.4
kJ/kg/d = 2.51 %BW/d of krill at 5.0 kJ/g (2.09 at the ~6 kJ/g of autumn-winter krill).

| datum (whiting only) | %BW/d | model scale (fraction of ceiling) | tag |
|---|---|---|---|
| Oslofjord Nov 2000, krill only | 0.23-0.54 | 0.09-0.22 | D from F (Onsrud 2004; `onsrud_three_months.py`) |
| Oslofjord Mar 2001, krill only | 0.65-1.51 | 0.26-0.60 | D from F |
| Oslofjord Jun 2001, krill only | 0.20-0.46 | 0.08-0.18 | D from F (thin; E1) |
| Oslofjord Nov, total (krill 0.82 by mass) | 0.28-0.66 | 0.11-0.26 | D |
| Oslofjord Mar, total | 0.79-1.84 | 0.32-0.73 | D |
| North Sea Q1 1981, population krill ration (all stomachs) | - | 0.23-0.26 | D ("Krillintag och jaktsätt") |
| krill-dominated stomachs >= 20 cm, low / central band | - | 0.32 / 0.75 | D (same) |
| SMS key run Q1 total, 0.2 kg | 0.78 | 0.25-0.34 | S (WGSAM 2024) |
| model maintenance / need | - | 0.272 / 0.388 | library |

Energy convention: krill at 5 kJ/g throughout (the GER rations already assume E 5 kJ/g); the SMS totals at a diet
E of 4.0-5.5 kJ/g [J].

Reading [J]:
- "They take about maintenance" holds for the Oslofjord November whiting (0.11-0.26 total against maintenance
  0.272) and the population means (0.23-0.34). It does NOT hold for the March whiting in the same layer
  (0.32-0.73 total, central ~0.5), nor for the fish that met krill in the North Sea in Q1 1981 (0.32-0.75 on
  krill alone).
- The verdict "eats about maintenance with krill present" in "Krillparen mot vintermagar" was carried mainly by
  herring sources (PROXY), none of which involved krill: Berg 2024 (lab, dry feed in excess), Bernreuther 2007 and
  Lohmeyer & Hempel 1977 (field, no krill); plus the Onsrud stomachs. For
  whiting the winter data show a wide spread inside one layer and one season, not a level pinned at maintenance.

## E1. Onsrud et al. 2004: November, March and June at one site [F]

Onsrud MSR, Kaartvedt S, Røstad A, Klevjer TA (2004) ICES J Mar Sci 61:1278-1290,
doi:10.1016/j.icesjms.2004.09.005. Local copy `research_notes/Vitlingens fiskbyten/data/step2_lit/vertical/onsrud2004.pdf`
(pdftotext -raw, read in full for this question).

What differed between November 2000 and March 2001:
- Light: "The 1% light depth was particularly shallow in November (4 m), and occurred between 12e15 m for the other
  study periods." Discussion: "The high light extinction was associated with record high run-off from land, caused by
  extensive rainfall prior to the sampling period." [F]
- Oxygen: "low oxygen content at midwater depth in November (min. 1.5 ml l-1, oxygen saturation 22.8% at 62 m)";
  "In the period with lowest oxygen content of the basin water (November 2000; oxygen saturation of 22.8%), both fish
  and krill were abundant." [F]
- Temperature: "Below ~30 m for all surveys, the vertical temperature and salinity profiles were practically
  homogenous with depth" [F]. No Nov vs Mar deep-temperature difference is reported in the text [F]; the figure
  values were not digitised.
- Fish position: benthopelagic fish lived "pelagically below about 90 m in November" and were "associated with the
  bottom in March" [F]; the authors ascribe the November pelagic habit to the high light extinction [F].
- Krill density: not reported for either month [F]. Fish aggregated under the ship ("an about eight times higher
  biomass directly beneath the vessel compared to 80 m away", March) and "backscattering ascribed to krill became more
  prominent at low fish abundance" [F].
- Whiting stomachs (Table 3, re-read here; day / night): fullness Nov 2.67 / 3.00, Mar 3.28 / 3.25, Jun 3.08 / 2.23;
  with krill Nov 42 / 35 %, Mar 76 / 80 %, Jun 43 / 35 %; krill per stomach WITH krill Nov 1.82 / 3.47, Mar 3.95 /
  5.92, Jun 2.17 / 1.44; empty Nov 25 / 15 %, Mar 8 / 6 %, Jun 7 / 27 % [F]. Night >= day in krill per stomach in
  Nov and Mar, not in Jun [F]. The June row was not used in "Krillparen mot vintermagar".
- Krill rations by the same GER method (`scripts/onsrud_three_months.py` -> `tables/onsrud_three_months_stdout.txt`):
  Nov 0.23-0.54, Mar 0.65-1.51, Jun 0.20-0.46 %BW/d = 0.09-0.22 / 0.26-0.60 / 0.08-0.18 of the model ceiling [D].
- June limits [F/J]: no n per cell; no krill density; no fish length; "catches in June were totally dominated by
  jellyfish (Cyanea capillata and Aurelia aurita)"; surface "maximum of 20(C in June 2001" (the deep water stays
  homogeneous below ~30 m). Stomachs with fish structures 0.04 / 0.08 in June as printed (column labelled %; Nov 0.13 /
  0.04), with copepods 0 / 7.69 % [F]: the June fish did not fill on fish or copepods; what made up the day fullness
  3.08 is not stated (the whiting diet also held "a few decapods, amphipods, and polychaetes") [F/J].
- Caveats on the premise "the fish stood in the layer" (from `lit_inlayer.md` [F]): the stomach hauls are not tied
  to the krill layer in the paper (the trawl-catch figure pools hauls), and fish were ~8x denser under the anchored
  vessel. Over all stomachs (krill-free included) the whiting mean is 0.7-4.7 krill per fish, overdispersed among
  krill-bearing fish (variance/mean 3.5-22) [D in `tables/inl_onsrud_counts_stdout.txt`].
- Digestion: "the state of digestion for whiting captured at night suggested that food had been eaten recently" [F].
- Krill antipredator behaviour: "The 'voids' in the krill recordings in the proximity of deep fish assemblages (cf.
  Figure 3) suggest that the krill performed instantaneous escape reactions during day. It appeared that M. norvegica
  would seek out water with reduced fish abundance at night" [F].

Reading [D/J]:
- The whiting krill ration rose x2.8 (0.23-0.54 -> 0.65-1.51 %BW/d) from November to March at the same site and in
  homogeneous deep water (deep temperature assumed equal: not digitised). Fish position differed: benthopelagic fish
  pelagic below ~90 m in November, bottom-associated in March.
- The March sample (27-28 March 2001, Figs 2D and 4) lies inside whiting's Feb-Jun spawning window ("Lek i rutan"),
  and it is the high month. Maturity and lengths of the stomach fish are unknown, so this is one whiting sample
  against a spawning-bound depression, set against one year for it (Robb et al. 1994, Q1 1991) [J].
- Light cannot be the cause: night >= day in both months, so detection by vision does not limit whiting here; the
  dark November is consistent but not needed [J].
- A lipostatic appetite predicts the Nov -> Mar direction (field whiting fill high in Nov-Feb, falling in Mar-Apr;
  library start_fill 0.73 Nov, 0.56 Mar, 0.30 Jun) [J]. It predicts the HIGHEST intake in June, at the fill minimum,
  and June is as low as November. So the three months do not fit a lipostat [J]. This rests on the fill cycle, which
  comes from Black Sea whiting HSI and a haddock proxy ("Mättnad hos rovfisk", "Lek i rutan"), not from Oslofjord fish.
- Nor do they fit a winter-keyed appetite: the low months are November AND June, the high month March [J].
- Krill density, krill size, fish length and maturity are unreported in all three months, so the series does not
  discriminate appetite from encounter on its own; it only removes the winter-only and the lipostatic forms [J].
- The 40-krill whiting (21 cm, ~60 g, 8-16 %BW) and "food eaten recently" at night show that whiting in the layer can
  and do take krill fast in winter; a capture probability near zero is excluded [J].

## E2. Addition or substitution in the ICES stomach database [D]

`scripts/ices_addition.py` -> `tables/ices_addition_stdout.txt`; inventory `scripts/ices_inventory.py` ->
`tables/ices_inventory_stdout.txt`. Input: the records already linked in "Krill över grunda bottnar"
(`stomachs/tables/records_linked.csv.gz`).

Prediction [J]: an acquisition-limited fish that meets krill adds it to its other food (krill-bearing records hold
more in total); a regulated fish substitutes (totals equal, non-krill food lower).

Data limits: whiting weights exist only for April 1991 (Skagerrak, Kattegat) and 2020-2024 (almost no krill); the 1981
whiting year is counts only. Pools of one length class per haul dilute both contrasts.

| stratum | krill-bearing / krill-free total %BW | total K/0 | non-krill K/0 | within-haul total K/0 (hauls) |
|---|---|---|---|---|
| whiting >= 20 cm, Skagerrak Apr 1991 (17 / 19 records) | 2.39 / 0.78 | 3.05 | 0.48 | 1.63 (2 hauls, both up) |
| whiting >= 20 cm, Kattegat Apr 1991 (3 / 31) | 0.84 / 0.78 | 1.07 | 0.19 | 1 haul, down |
| PROXY Norway pout, Q1 1980, three areas | 0.12-0.49 / 0.11-1.21 | 0.40-1.41 | 0.20-0.77 | 0.87-5.3 (14 hauls, 9 up) |
| PROXY Norway pout, Q3 1980, two areas | 0.25-0.84 / 0.06-0.34 | 2.45-4.19 | 0.09-2.77 | - |
| PROXY saithe, Q1 2026, NE / NW North Sea | 0.31 / 0.51; 2.51 / 1.05 | 0.61; 2.39 | 0.42; 2.21 | 7.96 (7 hauls, 6 up); 0.10 (2 hauls) |
| PROXY cod, Q1 1981, C+S / NE North Sea | 0.44 / 1.16; 0.86 / 1.40 | 0.38; 0.62 | 0.33; 0.45 | 0.17-0.53 (17 hauls) |

Counted whiting, Q1 1981 (no weights): krill-bearing records held fish prey less often than krill-free ones
(NE 0.45 vs 0.77, NW 0.42 vs 0.84) [D].

Fed records only (empties dropped from both groups, since holding krill selects recently fed fish): whiting
Skagerrak Apr 1991 unchanged (3.05; no empty records there); saithe Q1 2026 NE flips to 0.39, NW 1.62; the rest as
above [D].

Reading [J]: whiting in the Skagerrak in April add krill to a partly reduced other diet (x3 in total, x0.5 on the
other food): the addition pattern of an acquisition-limited fish, on 36 records from one month. The proxies are mixed
and the cod contrast runs the other way (krill-eating cod have less food in total, i.e. krill is taken where other
food is scarce). The Q1 count data say krill and fish prey are met in different hauls, which is placement, not
appetite. The test is weak: it cannot separate "met a krill patch" from "was hungrier". It is one supporting line,
not a discriminator.

## E3. Local secondary sources bearing on winter appetite (not re-read) [S]

- PROXY cod, Balsfjord 69 N, one year (Klemetsen 1982, ICES J Mar Sci 40:101, doi:10.1093/icesjms/40.2.101 [A via
  "Torskfiskars ljusberoende vid jakt/gadoid_diel_feeding.md"]): "The fullness of the stomachs showed no seasonal
  trend"; neither spawning nor polar darkness appeared to affect feeding intensity; krill "probably the only prey
  taken in midwater".
- PROXY cod and haddock, Svalbard fjords, January polar night (Larsen et al. 2023, Polar Biol,
  doi:10.1007/s00300-023-03167-z [S via the same note, snippet]): > 70 % feeding, few stomachs well filled, heavy
  overlap on Thysanoessa in krill-rich years.
- Whiting, North Sea (Mergardt & Temming 1997, ICES J Mar Sci 54:226, doi:10.1006/jmsc.1996.0190 [A via the same note]): a single feeding peak at
  22-24 h, minimum 08-10 h.
- Whiting (Rindorf 2002, J Fish Biol 61:579, doi:10.1111/j.1095-8649.2002.tb00897.x [A]): fresh food less likely when
  partly digested food is present; meal size unchanged; lab whiting resume feeding at ~40 % fullness, ~21 h apart.
  A satiation brake on search, i.e. the digestion-limited Holling form, not a seasonal appetite.
- Whiting and haddock on sandeel, northern North Sea, July 1996 (Temming, Götz, Mergardt & Ehrich 2004, J Fish Biol
  64:1351, doi:10.1111/j.0022-1112.2004.00400.x [A, OpenAlex abstract; closed access]): "a pronounced aggregation of
  whiting in the south-west part of the area, where sandeel catches were also highest. This pattern was thought to be
  the result of an aggregative behaviour of whiting"; "The stomach contents in both predators increased rapidly during
  the night". Reading [J]: in summer whiting track a dense benthic prey patch and fill on it - a hungry,
  acquisition-limited fish. There is no comparable aggregative response of gadoids to krill ("Krillens fläckighet").

## E4. Model-scale sizing per mechanism [D]

`scripts/model_scale.py` -> `tables/model_scale_stdout.txt`, section C (the per-mechanism table is in summary.md).
Field krill intake is re-anchored on in-area cells (C1): Skagerrak Q1 1981 0.05 -> potential 0.47; booked field flow
(M2 0.1873 x 5.0 / 0.446 = 2.10 t/t/yr, booked in section 245 from the in-area diet share 0.50 x Q/B 4.2) 0.229 ->
f 0.15, potential 0.56; Skagerrak Apr 1991 0.30 -> 0.60; Oslofjord three-month mean 0.14-0.33 -> 0.52-0.62. The North
Sea quarterly mean (0.06-0.09 -> 0.48-0.49) is not in-area and probably at lower krill density [J].

## E5-E6. Literature trails

See `lit_appetite.md`, `lit_inlayer.md`, `lit_seasonal.md` (subagent trails, quotes there). The main agent's
reading of them is in summary.md (the harness refused the main agent's write of summary.md; its full text is in the
hand-back for the caller to save); spot checks are logged at the end of this file. Merge log:
`scripts/merge_csv.py` -> `tables/merge_csv_stdout.txt`.

## Spot checks of the subagent trails (main agent, against the downloaded texts)

| claim in a trail | checked in | result |
|---|---|---|
| Kock et al. 1994: median feeding intensity 1.07 % (1985), 0.27 % (1991), 2.19 % (1992) | `data/inl_kock1994_meps108/paper.txt` l. 682-683 | confirmed |
| Nedreaas 1985: saithe on 1 March "showed clear signs of starvation"; krill and decapod larvae "very scarce in the plankton samples" | `data/nedreaas1985/paper.txt` l. 54, 760-763, 921-923 | confirmed |
| Robb et al. 1994: "a marked seasonal difference in the proportions of the whiting stomachs" classed empty (Q1) | `data/vliz275252/t.txt` l. 326 | confirmed |
| Hislop 1983: "no obvious between-quarter differences in the weights of food" (1981) | `data/ices_cm1983_g59/r.txt` l. 100 | confirmed |
| Onsrud 2004 Table 3, June 2001 whiting row (used here, not in the trail) | pdftotext -raw of the local copy | confirmed: 3.08(1.21) 2.23(1.03); 42.86 34.62; 2.17(2.33) 1.44(1.01); empty 7.14 26.92 |
| Uglem 2020 spring jig row: 166 +- 106 g pellets, 60 % with pellets, 3.09 kg | `../Mättnad hos rovfisk/data/uglem2020/a.txt` l. 188 | confirmed |
| Wilson et al. 2009: stomach krill "increased with krill abundance in the plankton and/or predator length (Fig. 2)" | `data/inl_wilson_pmel687/paper.txt` l. 387-388 | sentence confirmed; the panel R2 0.53 / 0.38 / 0.23 sit in the figure and were not re-checked |

Email audit: the folder (scripts, tables, md, data) holds no contact address; fetch scripts send a generic
User-Agent; the Unpaywall helper reads the address from the environment only.
