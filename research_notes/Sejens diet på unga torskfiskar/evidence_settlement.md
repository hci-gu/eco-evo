# Haddock early life, settlement, growth and nursery depth (items 1, 2, 4, 5)

Literature agent, 2026-10-29. Question: can the seasonal window of 0-1 group haddock in saithe
stomachs (Q1 ~0, Q2 small, Q3 peak at 75-110 mm, Q4 90-135 mm) be derived from the prey's life
history (spawning -> pelagic phase -> settlement -> growth out of the predator's preferred size /
gape) and physical drivers, rather than a calendar switch?

Marks: [S] = stated in the source (quote or table reference); [J] = my derivation or reading of a
figure. Downloaded PDFs and text extracts are in `data/settlement/` and `data/settlement/excerpts/`;
the DATRAS survey extraction is in `data/settlement/datras/` (scripts in `scripts/datras_*.py`).

Gape assumption used throughout: 19 cm maximum prey length for 40-60 cm saithe (given by the caller,
not re-derived here). Preferred prey size from Floeter & Temming 2005 is given in entry 12.

---

## A. Timeline (one cohort, northern North Sea / Norwegian Trench edge)

Lengths are total length. "Gape" = inside a 19 cm maximum for 40-60 cm saithe. Rows marked [J] are
interpolated or read from figures; the others are anchored to the entry given. Era: "IBTS" values
are 2005-2023 (entry 13); Albert values are 1984-1987 (entry 8) and run 2-3 cm smaller at every
stage (his Feb-Apr 1-group 14.4-16.8 cm west vs IBTS 17-18 cm), consistent with density-dependent
growth in the larger 1980s year classes [J]. Do not splice the two eras within one row sequence.

| Month (cohort age) | Stage / position | Mean length (range) | Inside 19 cm gape? | Anchor |
|---|---|---|---|---|
| Feb-May (y0) | spawning Feb-May, peak Mar-Apr; pelagic eggs, 1-3 weeks to hatch | eggs 1.2-1.7 mm | n/a | 1, 2, 3 |
| Apr (y0) | pelagic larvae, east of Shetland in late April 1996 | larvae (few mm) [J] | n/a | 5 |
| May (y0) | pelagic larvae -> pelagic juveniles (transform at 2-3 cm on Georges Bank) | 1-3 cm [J] | yes (pelagic, not on the bed) | 9 |
| Jun (y0) | pelagic 0-group, diel vertical migration, near surface in the evening, "few fish caught near the bottom" (60 N 0 E, 147 m water) | 4.1 cm (1.7-7.3), 17-20 June 1991 | yes (pelagic) | 6 |
| Jul (y0) | settlement transition offshore: demersal by day, up at night; 0-group present at 80-300 m in the Norwegian Deep from late July | 6-8 cm settling [J]; IBTS July 10.6 cm (gear-selected, biased up) | yes | 7, 8, 9, 13 |
| Aug (y0) | demersal 0-group on the plateau and Trench slope | IBTS 12.4 cm (p10-p90 10-15) | yes | 8, 13 |
| Sep (y0) | demersal | IBTS 13.7 cm (11-16) | yes | 13 |
| Oct-Nov (y0) | demersal, 80-200 m; 0-group = 65-98 % of haddock numbers in the Norwegian Deep | Albert 1984-87: 10.4-11.5 cm (west), 10.7-14.0 cm (Skagerrak); 2005-2023 era ~14.5-16 cm [J, interpolated] | yes | 8, 13 [J] |
| Dec (y0) | demersal, moving deeper for winter | ~16-17 cm [J, interpolated Sep 13.7 -> Jan 17.7, 2005-2023 era] | yes | 13 [J] |
| Jan-Feb (y1) | 1-group, demersal, deeper in winter (Norwegian Deep) | IBTS 17.2-17.9 cm; 24-34 % >= 19 cm; Albert 14.4-16.8 (west), 19.3-19.4 (Skagerrak) | mostly (66-76 %), but above the preferred size | 8, 13 |
| Mar (y1) | 1-group | IBTS 19.2 cm, 72 % >= 19 cm (few hauls) | mostly not | 13 |
| Apr-Jun (y1) | 1-group | ~20-23 cm [J, interpolated] | no (only very large saithe) | 13 [J] |
| Jul-Sep (y1) | 1-group | IBTS 23.1-25.7 cm, 92-98 % >= 19 cm | no | 13 |

Reading [J]: the bed-resident 0-group is within the gape from settlement (July) until about March
of age 1; the gape alone therefore does not close the window in Q1. What closes it earlier is (a)
the preferred, not maximum, prey size of saithe (entry 12: 11-22 cm for 40-60 cm saithe, realised
75-135 mm), which the 1-group passes during winter, and (b) predator-side effects (see
`evidence_saithe_season.md`). The window opens at settlement (July), not at hatching: the June
0-group is pelagic and absent from saithe stomachs.

---

## B. Spawning and the pelagic phase (item 1)

### 1. Wright, Tobin, Gibb & Gallego 2011 - haddock reproductive investment, North Sea

- Citation: Wright, P.J., et al. 2011. Changes in reproductive investment of North Sea haddock...
  Mar. Ecol. Prog. Ser. 432: 149-160. doi:10.3354/meps09168. PDF: `data/settlement/wright2011_haddock_reproductive_investment.pdf`.
- Area / years: North Sea, IBTS Q1 SMALK 1970s-2000s; two putative sub-populations, W (coastal
  west) and E (offshore east, the northern North Sea towards the Norwegian Trench).
- [S, p. 150] "The bottom trawl surveys were undertaken between January and March, overlapping the
  February to May spawning period of haddock in the North Sea (Hislop 1984, Wright & Gibb 2005)."
- [S, p. 150] "major spawning areas in the northern North Sea and off the Scottish east coast";
  "tagged coastal haddock rarely move into the deeper north east waters".
- Weight: the spawning window Feb-May is a secondary statement citing Hislop 1984; no peak date.

### 2. Coull, Johnstone & Rogers 1998 - Fisheries sensitivity maps in British waters

- Citation: Coull, K.A., Johnstone, R., Rogers, S.I. 1998. Fisheries Sensitivity Maps in British
  Waters. UKOOA. PDF: `data/settlement/coull1998_sensitivity_maps.pdf`, text `excerpts/coull1998.txt`.
- [S, "Spawning Periods For Main Commercial Species", p. 17] Peak-spawning asterisks: haddock
  Feb, Mar, Apr; cod Feb, Mar; saithe Jan, Feb; Norway pout (shelf) Feb, Mar. [J] Months assigned
  from the column positions of the asterisks in the text extraction; the shaded full periods are
  graphical and were lost in extraction.

### 3. ICES FishMap haddock factsheet

- Citation: ICES FishMap species factsheet - haddock (EU-RFP repository, ~2006). PDF:
  `data/settlement/ices_fishmap_haddock.pdf`.
- [S] "Spawning takes place from March to May, at depths of 100-150 m." "The pelagic eggs are
  1.2-1.7 mm in diameter and take one to three weeks to hatch." "After metamorphosis, the 0-group
  remain pelagic until they reach a size of approximately 7 cm, after which they settle to a
  demersal life style." Pelagic post-larvae (3-10 cm) eat euphausiids, appendicularians, decapod
  larvae, copepods and small fish (citing Robb & Hislop 1980).
- [S] Depth: 40-300 m, preference 75-125 m; in the Norwegian Deep "generally not found in waters
  more than 300 m deep".
- Weight: secondary compilation, no years; the ~7 cm settlement size is the standard textbook value.

### 4. Gonzalez-Irusta & Wright 2016 - spawning grounds of haddock (metadata only)

- Citation: Gonzalez-Irusta, J.M., Wright, P.J. 2016. Spawning grounds of haddock
  (Melanogrammus aeglefinus) in the North Sea and West of Scotland. Fish. Res. 183: 180-191.
  doi:10.1016/j.fishres.2016.05.028. Not downloaded (paywalled); abstract via marine.gov.scot.
- [S, abstract] IBTS haddock in spawning condition 2009-2015, GAMs; "An optimum temperature for
  spawning of 7 C was evident"; spawners prefer high-salinity water in the northern North Sea;
  avoid mud-rich sediments; spawning grounds shifted relative to historic reports.
- Use [J]: this is a spatial temperature optimum, not a timing cue. No source found that names
  photoperiod or a temperature threshold as the cue for the onset of haddock spawning in the North
  Sea; the timing cue item is unresolved.

### 5. Gallego, Heath, Basford & MacKenzie 1999; Heath & Gallego 1998 (abstracts only)

- Gallego, A., et al. 1999. Variability in growth rates of larval haddock in the northern North
  Sea. Fish. Oceanogr. 8: 77-92. doi:10.1046/j.1365-2419.1999.00101.x.
  [S, abstract] larvae mapped "in late April 1996", concentration "off the east coast of the
  Shetland Isles"; survey "coincided with the onset of the spring plankton production bloom";
  "larvae hatched later in the spring displaying higher growth at age than those hatched earlier";
  particle tracking suggests "differences in temperature history between cohorts" explain it.
- Heath, M.R., Gallego, A. 1998. Bio-physical modelling of the early life stages of haddock in the
  North Sea. Fish. Oceanogr. 7: 110-125. doi:10.1046/j.1365-2419.1998.00061.x.
  [S, abstract] survivors "represent the fastest-growing individuals"; growth reflects "food,
  turbulence and temperature" along the drift path; survivors come from "a restricted part of the
  spawning distribution".
- Wright, P.J., Gibb, F.M. 2005. Selection for birth date in North Sea haddock and its relation to
  maternal age. J. Anim. Ecol. 74: 303-312. doi:10.1111/j.1365-2656.2005.00924.x (OA at Wiley, but
  blocked to download). [S, abstract] birth dates of survivors to the demersal phase in 1994, 1996,
  1999; first-time spawners (age 2) spawn "much later"; "Survivorship was consistently poor from
  the late spawning period", possibly due to "the short length of the growing season prior to
  settlement".
- Weight: abstracts only; no numbers on the temperature-growth relation were reached.

### 6. Bromley, Watson & Hislop 1997 - pelagic 0-group gadoids, northern North Sea, June 1991

- Citation: ICES J. Mar. Sci. 54: 846-853. doi:10.1006/jmsc.1996.0211. PDF:
  `data/settlement/bromley1997_pelagic0group_feeding.pdf`.
- Area / method: 60 00'N 00 00'E (60 miles SE of Lerwick), water depth 147 m, 17-20 June 1991;
  IYGPT midwater trawl, 18 x 30 min tows at 10 m, 35 m and near the bottom over 24 h.
- [S, Table 1] haddock: mean 699 per hour, 450 stomachs, mean length 4.14 cm (SD 1.16, min 1.7,
  max 7.3). Saithe 0-group 4.56 cm, cod 3.91, whiting 3.21, Norway pout 4.78.
- [S, Results] "fish underwent a diel vertical migration, congregating at the surface in the
  evening and dispersing downwards in the day ... Few fish were caught near the bottom."
- [S, Methods] "Surface temperature was 10.3 C, declining to 8.3 C at a depth of 45 m."
- [S, Discussion] the pelagic food web "is transitory and disappears when the 0-groups become
  demersal early in the summer"; "Fish from early spawnings enter a simple food web and are not
  subjected to predation by larger 0-groups."
- Use [J]: the offshore northern North Sea 0-group is still pelagic in mid-June at ~4 cm, so
  settlement there is July (see entry 13 for the derivation).

### 7. Bromley & Kell 1999 - vertical migration of pelagic 0-group gadoids (abstract)

- Citation: Bromley, P., Kell, L.T. 1999. Vertical migration and spatial distribution of pelagic
  0-group gadoids prior to and during settlement in the North Sea. Acta Adriatica 40(2): 7-17.
  Abstract at Cefas (publication id 12585).
- [S, abstract] pelagic 0-groups of ~3 cm show diel vertical migration, timing variable (1991:
  near-surface peak in the evening; 1992: midday, also for haddock west of Denmark); "During
  settlement ... in July 1994 in the central North Sea, 0-group cod of 4-6 cm in length were
  demersal by day and pelagic at night. Cod of over 8 cm in length were wholly demersal."
- Use [J]: the transition is gradual over ~4-8 cm (shown for cod; haddock numbers not in the
  abstract), with a day-bed / night-water phase.

### 8. Albert 1994 - Ecology of haddock in the Norwegian Deep (key source for the Trench edge)

- Citation: Albert, O.T. 1994. Ecology of haddock (Melanogrammus aeglefinus L.) in the Norwegian
  Deep. ICES J. Mar. Sci. 51: 31-44. doi:10.1006/jmsc.1994.1004. PDF (scanned, read as images):
  `data/settlement/albert1994_haddock_norwegian_deep.pdf`.
- Area / years / method: Norwegian Deep slopes, subareas 1-3 (west/north, 59-62 N) and 4
  (Skagerrak), 1984-1987, 13 bottom-trawl surveys, 638 stations; winter (31 Jan-13 Apr), summer
  (26 Jul-14 Sep), autumn (5 Oct-9 Nov) [S, p. 31, Table 1].
- Depth [S, p. 33, Fig. 2]: "the catch rate of haddock fell sharply with depth down to 300 m and the
  species was not regularly caught beyond 250-m depth. All catches of more than 500 specimen h-1
  were taken in the depth interval 80 to 200 m."
- 0-group [S, p. 33]: "0-group haddock was present in the area in July-August, but the relative
  abundance of this group was probably underestimated due to mesh selection in the trawl. In
  autumn, 0-group contributed from 65% (1987) to 98% (1984) to the total number of haddock".
- Length at age [S, Table 2]:

  | | 0-group autumn 1984 | 1985 | 1986 | 1987 | 1-group winter 1984 | 1985 | 1986 | 1987 |
  |---|---|---|---|---|---|---|---|---|
  | West (subareas 1-3), cm | 10.54 | 10.41 | 11.53 | 10.92 | 14.72 | 16.82 | 14.41 | 16.15 |
  | Skagerrak, cm | 10.72 | 11.81 | 14.03 | 13.27 | - | 19.40 | 19.29 | 19.37 |

- Summer length structure [J, Fig. 4]: in late July-mid September the 0-group mode is ~7-9 cm at
  80-149, 150-199 and 200-300 m; autumn mode ~10-12 cm; winter 1-group mode ~13-15 cm.
- Depth by age and season [S, Table 3, % frequency of occurrence in 80-149 / 150-199 / 200-250 m;
  mean log catch rate when present]:
  0-group summer 40.0 / 38.9 / 21.4 % (log N 3.64 / 3.44 / 3.71); 0-group autumn 89.3 / 57.1 /
  15.9 % (4.19 / 4.17 / 2.67); 1-group winter 81.3 / 75.0 / 46.7 % (4.54 / 3.78 / 3.58).
  [J, Fig. 8] 0-group summer depth distribution interquartile ~140-200 m, autumn ~120-190 m;
  1+ winter ~110-210 m.
- [S, p. 36] "all age-groups of haddock were distributed deeper in winter than in summer and
  autumn"; "haddock undertook seasonal migrations up and down the slopes of the Norwegian Deep".
- [S, p. 37] summer 0-9 cm haddock stomachs contained ophiuroids and polychaetes (benthic prey),
  i.e. the 0-group caught in summer is already feeding on the bed.
- Weight: high for the Trench edge (the only source on the slope itself); 1984-87 only; trawl mesh
  under-samples the smallest 0-group.

### 9. Lough et al. 1989 - juvenile cod and haddock on Georges Bank (mechanism of settlement)

- Citation: Lough, R.G., et al. 1989. Ecology and distribution of juvenile cod and haddock in
  relation to sediment type and bottom currents on eastern Georges Bank. Mar. Ecol. Prog. Ser.
  56: 1-12. doi:10.3354/meps056001. PDF: `data/settlement/lough1989_georgesbank_juvenile_haddock.pdf`.
- Area / years: NE Georges Bank, 70-100 m, submersible dives and bottom trawls, 4-13 Aug 1986 and
  late July 1987.
- [S] "Transformation to the pelagic juvenile stage occurs when the larvae are 2 to 3 cm";
  "transition from pelagic to demersal life occurs when juveniles are 4 to 6 cm in length";
  "Pelagic juvenile gadids are widespread on the bank in late spring, but by late July, they have
  become demersal"; "During the day, demersal juveniles remain within a few centimeters of the
  bottom ... At night, regardless of the current, they rise off the bottom to feed on
  invertebrates in the near-bottom water column."
- [S] early Aug 1986 haddock 5-13 cm (mode 7 cm); late July 1987 haddock 4-9 cm (mode 6 cm).
- Weight: different stock and warmer bank; used only for the mechanism (size threshold, day-bed /
  night-water behaviour, gravel refuge hypothesis).

### 10. Lapolla & Buckley 2005 - hatch dates of YOY haddock, Georges Bank / Gulf of Maine

- Citation: Mar. Ecol. Prog. Ser. 290: 239-249. doi:10.3354/meps290239. PDF:
  `data/settlement/lapolla_buckley2005_haddock_hatchdates.pdf`.
- [S] survivors' hatch dates peaked "between February and mid March", earlier than the egg-based
  peak (April); YOY "> 200 d old" at autumn capture; mean growth 0.67 mm d-1 (range 0.41-1.14) on
  Georges Bank, 0.66 in the Gulf of Maine; egg stage "within 10 to 20 d at the temperatures
  typically experienced by eggs on GB (Page & Frank 1989)".
- [S, Discussion] early hatched fish grow "through the size of maximum vulnerability" before
  migratory fish predators (herring, mackerel) arrive; "Early settlement ... may also confer a
  survival advantage".
- Weight: mechanism and growth-rate scale only (NW Atlantic).

### 11. Bastrikin, Gallego, Millar, Priede & Jones 2014 - settlement at a Scottish coastal nursery

- Citation: ICES J. Mar. Sci. 71: 2101-2113. doi:10.1093/icesjms/fsu029. Read from the OUP article
  page via the web tool (not downloadable here); numbers below are as returned from the page text.
- Area / years: Stonehaven Bay (56 58'N 02 11'W), three reefs at 22, 32 and 42 m, 0.8-2.4 km
  offshore; intensive sampling 2004, limited 2005-2006; 1,099 haddock of 10-155 mm.
- [S, abstract] settled size "78 (+-4) mm for haddock" (cod 49, whiting 85); haddock "settled in one
  pulse, lasting ~2 weeks (second half of May), favouring deeper, farther offshore locations".
- [S, Results] "haddock settlement was also a gradual process, which took place in the 29 (+-6)-78
  (+-4) mm length interval" (pelagic prey < 50 % of diet weight from 29 mm; benthic prey > 50 % at
  78 mm); 88 (+-3) mm if juveniles eating Ammodytes are excluded; first pelagic catch 26 April 2004,
  peak 17 May, absent from pelagic samples by ~1-7 June; demersal numbers rise from 2 June, peak
  early July. No temperature data reported.
- Weight: high for size at settlement and its gradualness; the DATE is for a shallow coastal
  site and does not transfer offshore (entry 6 shows 41 mm pelagic fish at 60 N in mid-June) [J].

### 12. Floeter & Temming 2005 - prey size preference of North Sea saithe (gape vs preference)

- Citation: Analysis of prey size preference of North Sea whiting, saithe, and grey gurnard.
  ICES J. Mar. Sci. 62: 897-907. doi:10.1016/j.icesjms.2005.03.004. PDF:
  `data/settlement/floeter_temming2005_prey_size_saithe.pdf`.
- Data [S, Table 2]: 4,208 saithe stomachs with fish prey, 1980-1982 and 1991 (Q1 861, Q2 1,031,
  Q3 1,153, Q4 1,163); predator classes 250 mm to 1500 mm.
- [S] "saithe has by far the highest PWRs [preferred predator/prey weight ratios]"; "Only saithe
  larger than 80 cm switched to a preference for relatively small prey"; PWR decreased with
  predator size except in quarter 1.
- [J, Fig. 5b read] preferred PWR for 35-80 cm saithe ~18-50 (Q1 ~25 at 35 cm, ~18 at 40 cm, ~48
  at 80 cm; Q3 ~80 at 35 cm, ~38 at 60 cm, ~27 at 80 cm). With isometric length-weight, preferred
  prey length / predator length = PWR^(-1/3) = 0.27-0.38, i.e. 11-22 cm for 40-60 cm saithe; the
  realised haddock prey in saithe stomachs (75-135 mm, SMS file) sit at the lower end.
- Use [J]: the 1-group in Jan-Feb (17-18 cm) is above the size most 40-60 cm saithe prefer even
  though it is inside the 19 cm maximum. The window narrows by preference before it closes by gape.

### 13. DATRAS NS-IBTS 2005-2023 - juvenile haddock length by month and depth (own extraction)

- Source: ICES DATRAS web service, NS-IBTS, HH/CA/HL exchange data, Q1 (Jan-Mar) and Q3
  (Jul-Sep), 2005-2023; plus Q1 1981 and 1991. Scripts: `scripts/datras_haddock_juveniles.py`,
  `scripts/datras_haddock_alk.py`; outputs `data/settlement/datras/summary.txt`,
  `haddock_length_alk_summary.txt`, `summary_1981.txt`, `summary_1991.txt`, CSVs.
- Method [J]: population length = HL length frequencies (numbers per hour) split by an age-length
  key (year x quarter x region x 1 cm) from the CA records (the raw CA mean is length-stratified
  and biased). Regions: NE = lat >= 57 N and lon >= 3 E (Norwegian Trench edge, Skagerrak and the
  plateau east of 3 E); NW = lat >= 57 N, lon < 3 E. GOV bottom trawl: the smallest 0-group is
  under-sampled, so Q3 means are biased upward.
- ALK-weighted length [J, `haddock_length_alk_summary.txt`]:

  | | NE mean (p10-p90), mm | NW mean (p10-p90), mm | share >= 19 cm (NE / NW) |
  |---|---|---|---|
  | Q3 age 0, July | 110 (90-130) | 103 (80-130) | 0 / 0 |
  | Q3 age 0, August | 125 (100-150) | 113 (90-140) | 0 / 0 |
  | Q3 age 0, September | 140 (120-170) | 127 (100-150) | 0.004 / 0 |
  | Q1 age 1, January | 180 (160-210) | 170 (150-190) | 0.33 / 0.23 |
  | Q1 age 1, February | 178 (150-210) | 172 (150-200) | 0.35 / 0.24 |
  | Q1 age 1, March (few hauls) | 193 (170-210) | 191 (170-210) | 0.72 / 0.73 |
  | Q3 age 1, July-September | 238-257 | 232-241 | 0.89-0.97 / 0.98 |

- 1981 vs 1991 Q1 age 1 (raw CA means, length-stratified, comparable between years) [J,
  `summary_1981.txt`, `summary_1991.txt`]: 1981 January NW 129 mm (p10-p90 110-140), NE 158 mm;
  1991 January NW 182 mm, NE 186 mm, February NW 193, NE 188 mm. The 1990 year class entered its
  first winter ~5 cm longer than the 1980 year class. 1981 Q1 saithe stomach haddock were 125 mm
  (SMS observed stomachs, `../sms_keyrun2023/saithe_observed_stomachs.txt`), i.e. the 1980 year
  class at its survey size. See `evidence_saithe_season.md` entry 3.
- Q3 growth [J]: 0-group mean rises ~105 -> 124 -> 137 mm from July to September (all regions),
  ~0.5 mm d-1; Sep -> Jan ~0.3 mm d-1 (winter).

---

## C. Settlement depth at the Trench edge (item 5)

### 14. DATRAS NS-IBTS Q3 2005-2023 - 0-group by depth band (own extraction)

- Source / method as entry 13; 0-group = haddock < 150 mm in Q3 (the age-0 p90 is 130-170 mm, so
  the cut misses some large 0-group and admits few age 1) [J]. Mean numbers per hour over all valid
  Q3 hauls in the band, zeros included [`data/settlement/datras/summary.txt`]:

  | Depth band | NE hauls | NE 0-group n/h | NE share of hauls with 0-group | NW hauls | NW n/h |
  |---|---|---|---|---|---|
  | 0-50 m | 349 | 287 | 0.55 | - | - |
  | 50-100 m | 441 | 761 | 0.77 | 583 | 537 |
  | 100-150 m | 202 | 261 | 0.85 | 1172 | 192 |
  | 150-200 m | 188 | 52 | 0.64 | 262 | 12 |
  | 200-300 m | 37 | 20 | 0.24 | 33 | 0.8 |

- [J] The 0-group settles mainly at 50-150 m; on the Trench slope it is present (frequent at
  150-200 m) but at ~1/15 (150-200 m) to ~1/40 (200-300 m) of the 50-100 m density. Albert 1994
  (entry 8), sampling only the slope >= 80 m, finds the summer 0-group at 80-250 m with similar
  catch rates when present and an interquartile depth ~140-200 m; the two agree that settlement
  extends down the slope to ~250 m but thins below ~150 m.
- Seasonal shift [S, Albert 1994]: "all age-groups of haddock were distributed deeper in winter
  than in summer and autumn". The IBTS Q1 vs Q3 comparison by depth band is mixed (NE 200-300 m:
  Q1 1-group 47 n/h vs Q3 0-group 20 n/h; 150-200 m: 38 vs 52 n/h) and does not by itself show a
  winter descent [J].

---

## D. Predation and settlement (item 4)

No study was reached that measures predation on 0-group haddock by saithe as a function of
settlement timing. What exists:

- Q3 saithe stomachs hold 0-group haddock of 75-110 mm, Q4 90-135 mm [S, SMS observed stomachs,
  existing file]. [J] These are just-settled to recently settled sizes (settlement 70-90 mm,
  entries 3, 11) and lie at the lower half of the Q3 0-group distribution (p10-p50 80-130 mm,
  entry 13), consistent with saithe taking haddock once they are on or near the bed and with
  saithe's preferred prey size (entry 12).
- Bromley et al. 1997 (entry 6) show that predation in the pelagic phase is by larger 0-group
  gadoids (whiting cannibalism, cod and haddock eating whiting) and that "Fish from early spawnings
  ... are not subjected to predation by larger 0-groups"; it says nothing about saithe after
  settlement.
- Lough et al. 1989 (entry 9) hypothesise that the pebble-gravel bed favours "survival through
  predator avoidance"; Lapolla & Buckley 2005 (entry 10) hypothesise predation risk rises for
  later-hatched cohorts. Neither measures predation.
- Temming, Floeter & Ehrich 2007 (Ecosystems 10: 865-876, doi:10.1007/s10021-007-9066-3): not
  accessed. From secondary snippets it describes whiting aggregating on a dense patch of 0-group
  cod (a "predation hot spot"); not shown to involve saithe or haddock.
- Hay, Hislop & Shanks 1990 (Neth. J. Sea Res. 25: 113-130) and Lynam & Brierley 2007 (Mar. Biol.
  150: 1397-1401): not accessed. From abstracts/snippets: 0-group whiting and haddock shelter
  under scyphomedusae in June-July (0-group surveys 1971-1986); jellyfish abundance correlated
  with 0-group survival. [J] This is a pelagic-phase refuge, ending at settlement.

---

## E. Physical drivers of settlement timing [J, from the entries above]

1. Settlement is size-triggered, not date-triggered: haddock settle over ~3-8 cm, completing at
   ~7-9 cm (78 +- 4 mm Stonehaven; 4-6 cm Georges Bank; ~7 cm FishMap), coinciding with the switch
   to benthic prey (entries 3, 9, 11). Date of settlement = hatch date + time to grow to ~7-8 cm.
2. Hatch date follows spawning (Feb-May, peak Mar-Apr) plus a temperature-dependent egg stage of
   1-3 weeks (10-20 d) (entries 1-3, 10).
3. Growth to settlement size is temperature- and food-limited: later-hatched larvae grew faster
   at age through a warmer temperature history (entry 5); juvenile haddock growth peaks at 12 C
   (Peck et al. 2003, MEPS 251: 233-243, `data/settlement/peck2003_juvenile_haddock_growth_temperature.pdf`:
   6.0-9.6 cm SL haddock at 8 and 12 C, Gmax 3.0-3.1 % dry weight d-1, i.e. ~1 % length d-1 or
   ~0.8 mm d-1 at 8 cm [J, isometric]); field growth 0.66-0.67 mm d-1 hatch-to-autumn (entry 10).
4. Worked offshore estimate: mean 41 mm on 18 June at 60 N (entry 6) + 30-40 mm at 0.6-0.8 mm d-1
   = 40-65 d -> settlement completes mid-July to mid-August at the northern shelf / Trench edge,
   consistent with Albert's demersal 0-group mode of 7-9 cm from late July (entry 8) and with the
   first 0-group haddock in saithe stomachs in Q3 at 75-110 mm. The coastal Stonehaven date (end
   May) is ~6-8 weeks earlier; with water ~10 C at the surface in June offshore (entry 6), a 2 C
   warmer or earlier-hatched coastal cohort plausibly explains it, but no temperature-specific
   settlement study for the North Sea was reached.
5. Light: pelagic 0-group migrate to the surface in the evening (entry 6); newly settled juveniles
   hold the bed by day and rise at night (entries 7, 9), so the transition is a gradual day/night
   shift, not a step.
6. Exit from the saithe window: growth after settlement (~0.5 mm d-1 in Q3, ~0.3 mm d-1 in winter)
   carries the cohort through saithe's preferred prey size (11-22 cm for 40-60 cm saithe) in
   autumn-winter and past the 19 cm gape in March-April of age 1 (entries 12, 13). Haddock growth
   is density-dependent in the North Sea (the 1990 year class was ~5 cm longer in its first winter
   than the 1980 year class, entry 13), so a weak year class leaves the window sooner.

## F. Weak or inaccessible

- Not accessed (paywalled or blocked): Demain et al. 2011 (J. Fish Biol. 79: 89-111), Robb &
  Hislop 1980, Russell 1976, Koeller et al., Hislop 1984, Heath & Gallego 1998 and Gallego et al.
  1999 full texts, Wright & Gibb 2005 full text (OA but download blocked), Hay et al. 1990, Lynam &
  Brierley 2007, Temming et al. 2007, Holden 1981 (0-group gadoid surveys, ICES CRR 99).
- No North Sea haddock source gives a spawning cue (temperature or photoperiod) or a measured
  temperature dependence of settlement size or pelagic duration; item E.4 is a derivation.
- The December length and Apr-Jun of age 1 are interpolations; March of age 1 rests on few hauls.
- IBTS Q3 lengths are upward-biased (GOV selection); Albert's summer 0-group is also under-sampled.
- Bastrikin et al. 2014 numbers were read through a web summariser from the OUP page (no local copy).
