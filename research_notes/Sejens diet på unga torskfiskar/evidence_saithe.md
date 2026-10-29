# Saithe (Pollachius virens) diet on juvenile gadoids and flatfish - evidence

Question: what share of saithe food (by mass) is the model's `benthivorous_fish` group (haddock 38.6 %,
cod 31.4 %, long rough dab 12.6 %, other flatfish, starry ray by biomass), and does it change with prey
density? The model books 0.05 (0.02-0.15) for saithe; encounter theory gives ~21 % for the
water-column group and trained policies realise 15-23 %.

Marks: [S] = number or quote taken from the source (table/page given). [J] = my derivation or
judgement. "Benthivore share" below = haddock + cod + flatfish (+ rays if any), % of stomach content
wet weight, unless stated otherwise. Whiting, Norway pout, poor cod, blue whiting, silvery pout are NOT
benthivores in the model and are kept out.

Scripts (all `python3 -I`, read-only on the downloads): `scripts/yos_benthivore_share.py` (CRR 164 /
CRR 219 tables, Mehl 2005 consumption tables), `scripts/sms_saithe_stomachs.py` (SMS observed
stomachs), `scripts/sms_saithe_haddock_share.py` (SMS modelled consumption), 
`scripts/ecopath1991_benthivore_prey.py` (Ecopath 1991 diet vectors). Outputs are in `data/`.

---

## 1. Daan (ed.) 1989 - ICES Stomach Sampling Project 1981 (CRR 164)

- Citation: Daan, N. (ed.) 1989. Data base report of the stomach sampling project 1981. ICES Cooperative
  Research Report 164. doi:10.17895/ices.pub.4533. File `data/crr164_daan1989_stomach1981.pdf`;
  page image `data/excerpts/crr164_p72_table8E_saithe.png`.
- Area: northern North Sea only. "Because the samples were restricted to the northern part of the North
  Sea, the data have not been worked up by Roundfish Area" [S, sect. 3.6, p. 12].
- Years / season: 1980, 1981, 1982 pooled; grouped into a winter (Q1+Q4) and a summer (Q2+Q3) half-year
  [S, sect. 3.6]. 2190 stomachs (113 in 1980, 1338 in 1981, 739 in 1982) [S, Table 6-E, p. 49].
- Predator length classes (lower bound, mm): 250, 300, 400, 500, 700, 1000; n per class winter 2/55/98/340/
  408/203, summer 1/97/208/403/354/21 [S, Table 8-E].
- Method: % of stomach content wet weight, samples combined by length group [S, sect. 3.6, 5.3].
- Table 8-E, % weight [S] (haddock / cod / flatfish = *Limanda limanda*):

| Size class | Winter haddock | Winter cod | Winter dab | Summer haddock | Summer cod | Summer dab |
|---|---|---|---|---|---|---|
| 300 (30-40 cm) | 9.92 | - | - | 35.16 | 0.38 | - |
| 400 (40-50 cm) | 2.84 | - | - | 3.89 | - | - |
| 500 (50-70 cm) | 5.92 | - | - | 5.36 | 0.78 | - |
| 700 (70-100 cm) | 3.67 | - | 0.34 | 12.75 | 0.39 | - |
| 1000 (>100 cm) | 4.93 | - | 1.18 | 31.82 | - | - |

  Norway pout at 500-1000 mm: 67.4 / 75.5 / 74.2 (winter), 21.7 / 29.3 / 14.4 (summer); euphausiids 28-76 %
  below 50 cm and 21 % at 50-70 cm in winter [S, Table 8-E].
- All gadoids (Norway pout + haddock + cod + whiting) [J from Table 8-E]: 40-50 cm 14 % winter / 44 %
  summer; 50-70 cm 73 / 28 %; 70-100 cm 80 / 42 %. Of this, Norway pout is 11-75 %, haddock 3-13 %, cod
  0-0.8 %, whiting 0-1 %.
- Benthivore share [J, `yos_benthivore_share.py`]: 30-40 cm 22.7 %, 40-50 cm 3.4 %, 50-70 cm 6.0 %,
  70-100 cm 8.6 %, >100 cm 19 % (half-year mean; weighting half-years by stomach content gives the same to
  +-0.2). Weighted by the box length mix (40-59 cm 71 %, >= 60 cm 29 %): **5.5 %** (40-59 cm 4.7 %,
  >= 60 cm 7.3 %).
- Caveats [S, sect. 5.6]: "it proved impossible to obtain a sufficient number of stomachs in 1981 and
  samples from different years had to be combined"; "the stomach content data for saithe and mackerel can
  not be expected to contain a reliable signal of relative year class strength". Haddock year classes in
  1980-82 were large (SMS 0-group Q3 numbers 15.7, 22.0, 9.9 x 10^9 [S, SMS summary.out]) - so this is
  a high-haddock period, not a low one [J].
- Weight: high. Primary, by weight, northern North Sea, but PRE-2000 and pooled.

## 2. ICES 1997 - Stomach Sampling Project 1991 (CRR 219)

- Citation: ICES 1997. Database report of the Stomach Sampling Project 1991. ICES Cooperative Research
  Report 219, 426 pp. (coordinator J.R.G. Hislop). doi:10.17895/ices.pub.4626. File
  `data/crr219_ices1997_stomach1991.pdf`; text excerpts `data/excerpts/crr219_table8E_saithe_1991.txt`,
  `crr219_table6E_saithe_stomachs_by_area.txt`.
- Area: "Total North Sea" tables, but 3012 of 3210 saithe stomachs came from roundfish area 1 (north-west
  North Sea); area 2: 53, area 3: 12, area 7: 127 [S, Table 6-E]. Area 1 = north-western North Sea
  (Shetland-Orkney-Fladen) [J]. "Very few saithe stomachs were obtained from these inshore grounds and
  only saithe aged 3+ and older are included in the MSVPA" [S, p. 6].
- Year / season: 1991, all four quarters. Predators: age 3 to 9+, mean length 39-97 cm [S, Table 8-E-1..4].
- Method: % by weight. Unidentified Gadidae are listed separately in Table 8 (not redistributed there)
  [S, Table 8-E; p. 7-8 on redistribution for Tables 11-13 only].
- Table 8-E, haddock % weight by age (mean length) [S]:

| Quarter | age 3 | 4 | 5 | 6 | 7 | 8 | 9+ | n stomachs (ages 3..9+) |
|---|---|---|---|---|---|---|---|---|
| Q1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 371/304/46/32/18/16/10 |
| Q2 | 0.01 | 0.06 | 0.13 | 0.03 | 3.65 | 8.64 | 9.06 | 451/429/105/57/40/66/35 |
| Q3 | 23.34 (42 cm) | 10.73 | 5.14 | 3.36 | 1.21 | 1.13 | 1.06 | 242/72/28/10/13/14/12 |
| Q4 | 8.01 | 5.69 | 3.09 | 4.85 | 3.53 | 1.30 | 1.53 | 628/165/27/8/5/3/13 |

  Cod (*Gadus morhua*): not listed in any of the 28 age x quarter cells, i.e. absent or below the table's
  precision [S/J]. Flatfish: unidentified Pleuronectidae 0.03/0.01 (Q2, ages 3-4), non-commercial
  Pleuronectidae 1.04-1.20 (Q3, ages 7-9+), *Limanda limanda* 1.94 (Q4, 9+), *Microstomus kitt* 0.00 [S].
  Unidentified Gadidae 0-8.4 % (largest at ages 3-4) [S]. Norway pout 14-80 % and herring up to 91 % (large
  saithe) dominate [S].
- Benthivore share [J, `yos_benthivore_share.py`]: by age 7.8 / 4.6 / 1.9 / 2.0 / 2.9 / 4.1 / 4.0 %
  (quarters weighted by mean content x (1 - empty)); equal-quarter weights give 7.9 / 4.1 / 2.1 / 2.1 /
  2.4 / 3.0 / 3.7. Box length mix: **5.0-5.2 %** (ages 3-4 = 40-55 cm: 6.0-6.2 %; ages 5+: 2.6-3.0 %).
  Upper bound if ALL unidentified Gadidae were haddock/cod: 8.9 % - an overestimate, since Norway pout
  and blue whiting dominate the identified gadoids [J].
- All gadoids (Norway pout + haddock + whiting + unidentified + non-commercial Gadidae) [J, quarters
  weighted as above]: by age 58 / 63 / 50 / 40 / 25 / 23 / 18 %, of which Norway pout 44 / 51 / 42 / 29 /
  15 / 11 / 7 %. Box length mix: 52 % gadoids (40-59 cm 61 %, >= 60 cm 31 %), haddock ~5 %, cod 0.
  Non-commercial Gadidae reach 11-15 % in Q2 at ages 6-8 (probably blue whiting) [J].
- Weight: high. The largest by-weight sample (3210 stomachs), but north-west North Sea, 1991.

## 3. ICES WGSAM SMS key run 2023 - observed saithe stomachs (1981, 1986, 1987, 1991)

- Citation: ICES WGSAM North Sea SMS key run 2023, file `summary_stom.out`,
  https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2023 (copy in `data/sms_keyrun2023/`).
  Report: ICES 2024, WGSAM, ICES Sci. Rep. 6:13, doi:10.17895/ices.pub.25020968.
- Content: observed relative stomach content weight (`stomcon`) by year, quarter, predator length class
  and SMS prey species, as used to fit SMS; prey not modelled in SMS (krill, flatfish, blue whiting,
  pearlside, poor cod, ...) are in "other food" [S, file header; J for interpretation]. Saithe prey species
  in SMS: whiting, haddock, herring, N. sandeel, Norway pout; **cod is not a saithe prey in SMS** [S, rows
  present]. Samples: 1981 Q1-Q4, 1986 Q3, 1987 Q3, 1991 Q1-Q4; 1246 haul-samples in total (1986-87 rows
  for 300/350 and 500/600 mm are duplicated in the file; removed where stated) [S/J].
- Haddock share of stomach content, haul-weighted [J, `data/sms_keyrun2023/saithe_observed_stomachs.txt`]:
  all 4.1 %; by year 1981 5.6, 1986 6.9 (7.2 with duplicates), 1987 0.9-1.1, 1991 2.3 %; by predator length
  < 40 cm 7.6, 40-59 cm 3.6, >= 60 cm 2.9 %. Other food 43 %, Norway pout 44 %, herring 7 %, whiting 0.7 %.
- Haddock prey lengths (weight-weighted mean of prey length classes) [S, file]: Q3 75-110 mm (0-group),
  Q1 125 mm (1-group), Q4 90-135 mm, Q2 175-225 mm (saithe >= 80 cm; a 0.4 % trace at 175 mm in the 500 mm class).
- Large saithe in Q2 1991 is the least certain cell: CRR 219 (age-based) gives haddock 3.7-9.1 % at ages
  7-9+, while the SMS length-based file gives 0-2.7 % at 700-1000 mm. Same raw data, different
  aggregation [J].
- SMS fit vs observation (haul-weighted haddock share) [J]: 1981 obs 5.6 / fit 9.3; 1986 6.9 / 8.1;
  1987 1.1 / 1.0; 1991 2.3 / 1.4.
- Weight: high for the size/season pattern; it is the same 1981 and 1991 raw data as entries 1-2 plus
  1986-87 Q3, aggregated differently.

## 4. SMS key run 2023 - modelled consumption of haddock by saithe, 1974-2022

- Source: same repository, `who_eats_whom_level2.csv` (eaten biomass by predator, prey, year, quarter)
  and `summary.out` (saithe ration x N.bar = total consumption). Output
  `data/sms_keyrun2023/saithe_haddock_share_by_year.txt`.
- Haddock as % of saithe total consumption [J, from model output]: mean 1974-2022 7.7 %; range
  0.7 % (2012) to 28.9 % (1975); 1980-91 9.1 %; 2013-22 4.8 %; 2018-22 7.3 % (2018 2.3, 2019 4.6, 2020
  10.8, 2021 11.0, 2022 7.6). Whiting 2-8 %, Norway pout 5-38 %, herring 1-42 %, other food 34-53 %.
- Note: the modelled 1980-91 mean (9.1 %) is about twice the observed haul-weighted stomach share of the
  same years (4.1 %, entry 3). Part is consumption weighting (large saithe, Q3), part SMS overshooting 1981
  [J]. Use the observed stomach shares as the field number; SMS is a model.
- Prey side, for context: SMS 2018-22 haddock 1+ M2 0.40 /yr, of which saithe 0.19 [S, earlier file
  `../Fiskgrupper i rutan/predators/evidence_B_population.md`, sect. 3.1].
- Partial M2 of saithe on haddock [J, `scripts/sms_saithe_partial_m2.py`, input
  `../Fiskgrupper i rutan/predators/data/population/sms2023_partial_m2.out`, output
  `data/sms_keyrun2023/saithe_partial_m2_on_haddock.txt`]: summed over quarters and saithe ages, age 0
  0.28 /yr and age 1 0.47 /yr in 2018-22 (0.37 and 0.52 over 1974-2022). Across 1974-2022 the per-capita
  rate does not change with haddock abundance (log-log slope of M2 on N: -0.04 for age 0, +0.05 for
  age 1). In SMS, therefore, saithe's consumption of haddock rises in proportion to haddock numbers and the
  diet share rises with them (slope 0.75): a near-linear response, because haddock is a small part of the
  food available to saithe. Model-internal, not an observation.
- Weight: medium (model output, not observation).

## 5. ICES public stomach download 1981 / 1991 - saithe by number (earlier file)

- Source: `../Fiskgrupper i rutan/predators/tables/saithe_whiting_1981_1991_by_number.txt` (script
  `saithe_1981_counts.py` there). Northern North Sea saithe >= 40 cm, 53 pooled records, months 1, 2, 6, 7,
  11, 19 554 items, no weights [S].
- Fish items only: haddock 412, cod 42 (cod : haddock = 1 : 10 by number), Norway pout 419, pearlside 2664,
  silvery pout 150, sandeel 88, unidentified Gadidae 618 [S].
- The earlier rough mass split "fish part ~P 50 %, Bf 50 %" from assumed prey weights is superseded by the
  by-weight tables of entries 1-2 (haddock 3-6 % of total food at 40-70 cm, Norway pout 67-75 % in winter)
  [J].
- Weight: low for mass shares; useful for the cod : haddock ratio.

## 6. Ecopath North Sea 1991 (Mackinson & Daskalov 2007) - diet matrix

- Citation: Mackinson, S., Daskalov, G. 2007. An ecosystem model of the North Sea to support an ecosystem
  approach to fisheries management: description and parameterisation. Cefas Sci. Ser. Tech. Rep. 142.
  Diet read from EcoBase model 457 (`../Fiskgrupper i rutan/benthivores/data/ecobase/model_457.xml`),
  output `data/ecopath1991_benthivore_prey_stdout.txt`. Commercial-species diets compiled "using data from
  the 1991 year of stomachs (Hislop 1997)" [S, earlier file].
- Saithe adult (> 40 cm): haddock adult 0.0021 + juvenile 0.0007, dab 0.0006, long rough dab 0.0004,
  gurnards 0.0003 -> benthivores **0.4 %** [S items; J sum]. "Other gadoids (small)" (poor cod, bib,
  rocklings, silvery pout) 3.9 % [S]. Juvenile saithe (< 40 cm): benthivores 0.01 % [S].
- Correction to the earlier file [J]: its saithe "Bf 0.04-0.05" (and spurdog "Bf 0.24") came mostly from
  "Other gadoids (small)", which is not the model's benthivore group. On the haddock/cod/flatfish
  definition Ecopath gives 0.4 % for adult saithe - lower than the stomach tables (entries 1-2) it was
  built from, presumably after balancing.
- Weight: low (balanced model, not an observation).

## 7. Greenstreet 1996 (via FishBase) - North Sea saithe by quarter and age

- Citation: Greenstreet, S.P.R. 1996. Estimation of the daily consumption of food by fish in the North Sea
  in each quarter of the year. Scottish Fisheries Research Report 55. Not obtained; FishBase v25.04 diet
  records 6107-6130 (`data/fishbase/fishbase_v25.04_diets_saithe_spurdog_anglerfish.txt`).
- Only major groups: e.g. 40-50 cm fish 55-78 %, euphausiids 21-45 %; 50-100 cm fish 57-89 % [S, FishBase
  rows]. No gadoid / flatfish split. Weight: none for this question.

## 8. Mehl 2005 - Northeast Arctic saithe, Norwegian coast 62-71 N

- Citation: Mehl, S. 2005. Stomach analyses of Northeast Arctic saithe sampled during the saithe survey
  Varanger-More 1998-2003. Preliminary results. ICES AFWG 2005 Working Document 7. hdl:11250/100697.
  File `../Fiskgrupper i rutan/predators/data/diet/lit_gadoids_elasmo/mehl2005_neasaithe_stomach.pdf`.
- Area / season / years: coastal banks north of 62 N, October-November 1998-2003; 5865 stomachs; % wet
  weight; same methods as the North Sea projects [S, Methods, Table 1].
- Haddock in 40-60 cm saithe: "Other common fish prey was haddock (10-24 cm), Norway pout (5-19 cm) and
  blue whiting (10-19 cm). Small cod (5-14 cm) was only found sporadically in two years" (area A) [S, p. 5];
  areas B-D: "No cod prey was found in this area" [S, pp. 9-13].
- Q4 consumption tables (t) [S, Tables 13-14]; haddock + cod as % of total [J]:

| Year | North of 67 N | South of 67 N | All |
|---|---|---|---|
| 1998 | 14.5 | 12.2 | 13.9 |
| 1999 | 7.1 | 0.0 | 4.7 |
| 2000 | 1.4 | 0.1 | 1.1 |
| 2001 | 1.0 | 1.0 | 1.0 |
| 2002 | 22.7 | 0.5 | 11.4 |
| 2003 | 3.9 | 0.0 | 1.7 |
| pooled | 7.6 | 2.3 | **5.8** |

  (Haddock 1694-41333 t, cod 0-1805 t per year in the north [S]). Flatfish not reported [S].
- Weight: medium-high: post-2000, by weight, large n, consumption-weighted; but Norwegian coast with
  coastal haddock/cod nurseries, autumn only.

## 9. Icelandic MFRI 2024 - diets of 36 groundfish species, 1996-2023

- Citation: Hafrannsoknastofnun (Marine and Freshwater Research Institute) 2024. Faeda 36 tegunda botnfiska
  a Islandsmidum arin 1996-2023 / Diet of 36 groundfish species in Icelandic waters 1996-2023. Haf- og
  vatnarannsoknir HV 2024-01. https://www.hafogvatn.is/static/research/files/hv2024_01.pdf. Excerpt
  `data/excerpts/hv2024_01_iceland_pp63-65_138-139.pdf` (+ PNGs).
- Saithe SW area: 23 501 stomachs examined, 13 751 with food, all months, 1997-2023 [S, p. 63]. Table 22
  (% weight): capelin 33.6, euphausiids 14.3, unidentified fish 12.4, blue whiting 7.7, herring 7.1, Norway
  pout 6.8, sandeel 4.8, hyperiids 1.9, myctophids 1.0; the ten groups sum to 90.1 % [S, Table 22, p. 64].
  NE area: 6233 examined, 4369 with food [S, p. 66].
- Haddock and cod as prey (all predators): saithe is not among the predators shown in Fig. 159c (haddock;
  smallest bar shown ~1.5 %) or Fig. 160c (cod) [S, pp. 138-139]. So haddock and cod are each < ~1.5 % of
  Icelandic saithe diet by weight [J, read from the bar charts].
- Weight: medium as a contrast (huge n, post-2000), low for transfer (capelin system, different prey field).

## 10. Jaworski & Ragnarsson 2006 - Iceland 1992

- Citation: Jaworski, A., Ragnarsson, S.A. 2006. Feeding habits of demersal fish in Icelandic waters: a
  multivariate approach. ICES J. Mar. Sci. 63:1682-1694. doi:10.1016/j.icesjms.2006.07.003 (read online).
- Saithe: 677 stomach samples (up to 10 fish each), 1992, March / July-August / Nov-Dec, 17-145 cm [S,
  Table 1]. Diet "almost entirely fish (mainly ammodytids and capelin) and euphausiids" [S, Results];
  sandeel 73 % in summer, capelin 44 % in March, euphausiids 79 % at > 400 m [S]. Haddock and cod not
  mentioned as saithe prey [S].
- Weight: low (no gadoid numbers).

## 11. Faroes

- Hojgaard, D.P. 1999. Food and parasitic nematodes of saithe, Pollachius virens (L.), from the Faroe
  Islands. Sarsia 84:473-478. File `data/hojgaard1999_faroe_saithe.pdf`. Coastal 1-2 year olds (80 + 150):
  no cod or haddock; unidentified flatfish 0.2 % W in yearlings [S, Table 1]. 24 adults from the Faroe
  Shelf: "The fish species identified were Trisopterus esmarki, Micromesistius poutassou ... and
  Melanogrammus aglefinus" [S, p. 476] - occurrence only.
- Homrum, E.i, Hansen, B., Hatun, H., Steingrund, P. 2012. Growth, maturation, diet and distribution of
  saithe in Faroese waters. Mar. Biol. Res. 8:246-254. doi:10.1080/17451000.2011.627921. Not accessed
  (paywall); abstract only: adult diet dominated by young blue whiting in August; Norway pout, sandeel and
  blue whiting about equal in March [S, abstract via pure.fo]. No gadoid shares available.
- Weight: low.

## 12. NW Atlantic pollock (same species)

- Tyrrell, M.C., Link, J.S., Moustahfid, H., Smith, B.E. 2007. The dynamic role of pollock (Pollachius
  virens) as a predator in the Northeast US continental shelf ecosystem: a multi-decadal perspective.
  J. Northw. Atl. Fish. Sci. 38:53-65. doi:10.2960/J.v38.m605. File `data/tyrrell2007_pollock_nwatlantic.pdf`.
  5569 stomachs 1973-2005, <= 50 and > 50 cm [S, Table 1]. Prey categories: euphausiids, sandlance, squid,
  herrings, mackerel, hakes, decapod shrimp, other crustaceans, "other fish" ("includes: flatfish,
  sculpins, eels, and rockfish") [S, p. 58]. No cod or haddock category: the gadoid prey are hakes
  (*Merluccius*, *Urophycis*) [S/J]. "Pollock diet composition has substantially shifted through time and
  appears closely tied to the relative availability of prey" [S, abstract].
- Bowman et al. 2000 (NOAA Tech. Memo. NMFS-NE-155) via FishBase records 3516-3518, 1977-80: Georges Bank
  n 46 (sandlance 56 %, krill 30 %), Gulf of Maine n 28 (squid 47 %, krill 39 %, silver hake 4.9 %),
  Scotian Shelf n 7 [S, FishBase rows]. No cod/haddock/flatfish. The full memo tables were not obtained
  (only the 11-page preliminary PDF).
- Carruthers et al. 2005, J. Fish Biol. 66:327-347 (Scotian Shelf): not accessed; Tyrrell 2007 cites it for
  "recent sharp declines of euphausiid prey in pollock diets to the diminution in abundance of these pelagic
  prey" [S, Tyrrell p. 62].
- Weight: low for the share (different prey field), medium as evidence that the diet tracks prey
  availability.

## 13. Northern Norway fjord - 0-group gadoids

- Pedersen, T., Strand, H.K., Christie, H.C., Moy, F. 2020. Predation mortality from ambush and cruising
  predators on newly-settled 0-group gadoids. J. Exp. Mar. Biol. Ecol. 529:151396.
  doi:10.1016/j.jembe.2020.151396. Abstract via AGRIS only.
- Gill nets 3-10 m, September 2013-2015: frequency of occurrence of 0-group gadoids "lower for saithe
  (FO = 0.8%)" vs cod 9 %, shorthorn sculpin 10 % [S, abstract]. Saithe mainly ate sandeel [S].
- Weight: low (shallow fjord, occurrence only), but an independent post-2000 low.

## 14. Not accessed (paywall or no copy found)

- Du Buit, M.H. 1991. Food and feeding of saithe (Pollachius virens L.) off Scotland. Fish. Res. 12:307-323.
  doi:10.1016/0165-7836(91)90015-8. Secondary: "pollock in the waters surrounding Scotland consumed
  significant quantities of Norway pout, blue whiting and haddock" [S, Tyrrell et al. 2007, p. 62]. No
  numbers obtained.
- Bergstad, O.A. 1991. Distribution and trophic ecology of some gadoid fish of the Norwegian Deep. 1.
  Accounts of individual species. Sarsia 75:269-313 (and part 2, 75:315-325). Only the abstract quote in
  the earlier file: "Particularly important trophic relationships exist ... between small saithe and blue
  whiting as predators and the euphausids as prey, and between saithe and Norway pout" [S]. This is the
  only Trench-specific saithe diet; no haddock numbers.
- Greenstreet 1996 SFRR 55 (only FishBase major-group rows), Carruthers et al. 2005, Homrum et al. 2012
  (full texts), Bowman et al. 2000 main tables.
- UiB/IMR master projects on North Sea saithe stomachs from IBTS Q1/Q3 and a coastal survey, 2009-2012
  (supervisors J. Devine, I. Huse; https://www.uib.no/en/rg/evofish/70899): any thesis or data not located.
  If it exists it would be the only post-2000 North Sea saithe diet.
- No Skagerrak saithe diet found. The ICES 1991 data hold 12 saithe stomachs from roundfish area 3 [S,
  CRR 219 Table 6-E]; the earlier file found one saithe stomach in the northern North Sea in 2024.

---

## Size dependence [J from entries 1-3, 8]

- Small saithe (30-45 cm) take the most haddock, but only in Q3-Q4 when 0-group haddock (7-11 cm) are on
  the bed: 35 % (30-40 cm, 1980-82 summer), 23 % (age 3, 42 cm, Q3 1991), 37 % (300 mm, Q3 1981, SMS
  file). In Q1 the share is ~0 at every age in 1991.
- 40-60 cm (the box's main class): 3-6 % across all sources with weights (CRR 164 3.4-6.0, CRR 219 4.1-7.8,
  SMS observed 3.6, Mehl area A haddock 10-24 cm).
- > 60 cm: 2-4 % (1991) to 7-9 % (1980-82); very large saithe (> 80-100 cm) eat 1-group haddock (17-22 cm)
  in Q2 (8.6-9.1 %, 1991) and up to 32 % in summer 1980-82 (21 stomachs).
- Prey: 0-group haddock (75-110 mm) in Q3-Q4, 1-group (12-25 cm) in Q1-Q2. Cod is a trace (cod : haddock
  ~1 : 10 by number in 1981; 0 in 1991 tables; none south of 67 N in Norway). Flatfish <= 2 %, only in
  saithe > 70 cm (dab, unidentified Pleuronectidae).

## Density dependence

- Observed, North Sea (entry 3) [S data / J reading]: Q3 haddock share in saithe stomachs vs SMS 0-group
  haddock numbers in Q3 of the same year: 1981 7.6 % at 22.0 x 10^9; 1986 6.9 % at 12.3 x 10^9; 1991 8.5 %
  at 7.2 x 10^9; 1987 1.1 % at 1.8 x 10^9. Directionally consistent with a saturating response (low only
  in the poorest year class), but n = 4, different sampling designs per year, and CRR 164 itself warns that
  its saithe data carry no reliable year-class signal.
- Norway (entry 8): haddock + cod 1-23 % of Q4 consumption between years in the north (14.5 % 1998,
  22.7 % 2002, 1.0-1.4 % 2000-2001). Not tested here against coastal haddock year-class strength [J].
- SMS (entry 4) implies a strong response: haddock share vs haddock 0-1 biomass in Q3, r = 0.94, log-log
  slope 0.75 over 1974-2022 [J]. This is a property of SMS's fitted suitability x availability /
  (sum + other food) form, NOT an independent observation.
  The per-capita M2 of saithe on 0- and 1-group haddock is flat against haddock numbers (slopes -0.04,
  +0.05): in SMS the response is near-linear over the observed range, with no saturation or switching.
- NW Atlantic (entry 12): diet "closely tied to the relative availability of prey" [S], qualitative.
- No functional-response or prey-switching study of saithe on haddock was found. (Rindorf et al. 2006, MEPS
  325:243, covers cod and whiting only.)

## Best estimate for the box (deep northern North Sea / Skagerrak edge, saithe 40-60+ cm) [J]

- **Central 5 % of saithe food by mass is benthivores (almost all juvenile haddock); range 2-9 %.**
  Support: CRR 164 5.5 %, CRR 219 5.0-5.2 %, SMS observed haddock 4.1 % (40-59 cm 3.6 %), Mehl 2005 5.8 %.
  Low tail: Ecopath 0.4 %, Iceland < 1.5 %, NW Atlantic ~0. High tail: CRR 219 upper bound 8.9 %, SMS
  model 2020-21 ~11 % with the strong 2019-2020 haddock year classes.
- Today's 0-group haddock (SMS Q3 2019-22 mean ~5.6 x 10^9 [S, summary.out]) is near 1991 (7.2 x 10^9),
  below 1981-86 (12-22 x 10^9), so the stomach evidence points to the middle of the range, not above it.
- The Trench edge at 100-300 m is deeper than the haddock nursery grounds sampled in area 1, so the
  in-box share should sit at or below the North Sea value [J].
- A share of 15-23 % appears in the field only in narrow cells: saithe 30-45 cm in Q3 of years with
  0-group haddock, very large saithe in summer, single coastal years (Norway 2002), or SMS years with
  extreme haddock year classes (1974-75). None of the multi-season, multi-size averages comes near it.
- Model mapping: the field share is haddock 0- and 1-group. Cod (31 % of the group) is a trace and
  flatfish (~30 %) <= 2 %, only for saithe > 70 cm. Treating the whole benthivore biomass as saithe prey
  overstates what saithe can reach [J].

## Weak numbers

- Every by-weight North Sea saithe diet is PRE-2000 (1980-82, 1986-87, 1991). No post-2000 North Sea and
  no Skagerrak saithe diet exists in what I could reach.
- CRR 219 is 94 % roundfish area 1 (north-west North Sea); CRR 164 pools three years.
- Density dependence rests on 4 observed year aggregates plus model-internal SMS behaviour.
- Mehl 2005 is coastal Norway, autumn only, preliminary.
- Iceland < 1.5 % is read from bar charts in which saithe is absent.
