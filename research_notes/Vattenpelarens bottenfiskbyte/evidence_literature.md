# Water-column predators and the bed layer - literature evidence

Question: the `predatory_fish` -> juvenile benthivore attack rate (settled 0-1 group haddock/cod,
7-25 cm, within ~1-5 m of the bed) was computed as if saithe searched INSIDE a 2-m near-bed layer
all of its foraging time. What share of saithe time/biomass is within a few metres of the bed, what
is the thickness H_saithe of the layer saithe actually occupy, and what do saithe in the Trench /
Skagerrak / northern North Sea eat? Plus (task 5) the thickness H_juv of the layer that holds the
settled juveniles.

Tags: [F] stated in the source (page/figure given), [D] computed here from source data, [J] judgement.
Downloads: `data/lit/` (see `data/lit/README.md`). Sources already in hand in
`../Sejens diet på unga torskfiskar/` are cited by path, not re-downloaded.
Date of search: 2026-10-31.

---

## A. Saithe vertical distribution relative to the bed

### A1. Neilson et al. 2003 - pre-spawning pollock aggregations, Scotian Shelf (acoustics, bottom-referenced)
- Neilson, J.D., Perley, P., Stevens, C., Clark, D., Melvin, G. 2003. Diel vertical distribution and
  characteristics of pre-spawning aggregations of pollock (Pollachius virens) as inferred from
  hydroacoustic observations: the implications for survey design. ICES J. Mar. Sci. 60: 860-871.
  doi:10.1016/S1054-3139(03)00068-7. PDF: `../Sejens diet på unga torskfiskar/data/settlement/neilson2003_pollock_prespawning_acoustics.pdf`.
- [F p.862] Sa integrated in six bottom-referenced layers "1-5, 6-10, 11-15, 16-20, 21-25 and 26-30 m
  off bottom. The first metre off bottom was excluded". Only 0-30 m above bed is integrated.
- [F abstract] "while pollock can occur up to 30 m off bottom, the greatest proportion remains within
  1-5 m off bottom during both day and night"; aggregations at 230-270 m (Shelf Edge), < 175 m (Banana);
  Sept-Oct 1999; most fish at the Banana < 45 cm (DFO 2003/075 p.10).
- [D, read off Fig. 7 p.868, medians of per-transect proportion of 0-30 m signal, +-5 pp]:

  | Site, period | 1-5 m | 6-10 m | 11-30 m (each 5-m layer) |
  |---|---|---|---|
  | Banana, day | ~27 % | ~14 % | ~7-10 % |
  | Banana, night | ~37 % | ~23 % | ~5-10 % |
  | Shelf Edge, day | ~68 % | ~3 % | ~2-8 % |
  | Shelf Edge, night | ~80 % | ~6 % | ~1-3 % |

  [D] Assuming uniform density inside the 1-5 m layer, a 2-m slice holds ~0.4-0.5 of it: 11-18 % of
  the 0-30 m signal at the Banana, 27-40 % at the Shelf Edge.
- [F p.868] the prediction that the 1-5 m share is higher by day "is not supported by our study";
  "the evidence supporting the hypothesis of changes in the day-night occurrence of pollock within the
  ADZ is equivocal". No feeding data.
- Use [J]: an UPPER bound on near-bed share - pre-spawning aggregations on topographic features,
  selected where bottom-trawl catches were > 95 % pollock, and nothing above 30 m was counted.

### A2. Neilson et al. 2003 (DFO CSAS Res. Doc. 2003/075) - year-round commercial echo logger
- Neilson, J.D., Perley, P., Stevens, C., Clark, D., Melvin, G. 2003. Can hydroacoustic survey
  approaches yield abundance indices of pollock on the Scotian Shelf? DFO CSAS Res. Doc. 2003/075.
  PDF: `data/lit/dfo282231/dfo282231_pollock.pdf`.
- [F p.8] "Peak biomasses of pollock were observed during the December and January period. During that
  time, aggregations of pollock ranged up to 30 m off bottom, but 75% of the biomass occurred within
  10 m of the bottom." Mean depth ~150 m. Data in 1-m bins 0-30 m above bed, only sets with >= 95 %
  pollock (Fig. 7 p.22).
- [D, Fig. 7, the "75 % of biomass" line by bimonth]: Dec-Jan 10 m; Feb-Mar 5 m; Apr-May 10 m;
  Jun-Jul ~13 m; Oct-Nov ~3 m. Share of the 0-30 m biomass in the lowest 2 m (bar sums, +-5 pp):
  Dec-Jan ~14 %, Feb-Mar ~34 %, Jun-Jul ~20 %, Oct-Nov ~45 %.
- [F p.12] "The best correlations of net catch and the acoustic signal occurred when the signal was
  integrated up to 30 m off bottom", adding "Norwegian workers (Aglen et al. (1999) have described a
  similar result with saithe". The Aglen et al. 1999 abstract (E2) is Barents Sea haddock/redfish and
  does not mention saithe, so the citation may be loose; the verifiable saithe result is Aglen 1996 (A3).
- Use [J]: conditional on a fishing set on a near-bed aggregation (selected, 0-30 m only). Within such
  aggregations the 2-m share is 15-45 %; the effective thickness is H ~ 2 m / share ~ 4-13 m.

### A3. Aglen 1996 - bottom trawl vs echo integration, Norwegian surveys (incl. saithe)
- Aglen, A. 1996. Impact of fish distribution and species composition on the relationship between
  acoustic and swept-area estimates of fish density. ICES J. Mar. Sci. 53: 501-505.
  doi:10.1006/jmsc.1996.0072. Abstract only (OUP PDF not retrievable; abstract via OpenAlex).
- [F abstract] "For large haddock ..., saithe (Pollachius virens (L.)), cod ..., redfish ... the best
  correlation between catch and acoustic data was obtained when applying the cumulative acoustic
  value (from the bottom upward) at 30 to 100 m above the seabed, while for Norway pout ... the best
  correlation was obtained at about the height of the headline of the trawl (4 m). Attempts to
  compensate the acoustic values for losses in the acoustic dead zone did not improve the
  correlations."
- Use [J]: the saithe that a bottom trawl samples come from a layer 30-100 m thick; Norway pout, a
  truly near-bed gadoid, matches the 4-m headline. Saithe are not a 2-m-layer fish.

### A4. Nedreaas 1997 - IMR NE Arctic saithe acoustic survey (coastal banks 62-71 N)
- Nedreaas, K. 1997. Evaluation of the North-East Arctic saithe (Pollachius virens) acoustic survey.
  ICES CM 1997/Y:20. doi:10.17895/ices.pub.27150897. PDF: `data/lit/nedreaas1997/`.
- [F p.3] "Mid-water tows are less common. Saithe are difficult to catch in mid-water, and seem to dive
  in response to ship noise, so even those well above bottom may be sampled with a bottom trawl";
  "in areas where ... the fish signals are entirely pelagic, a mid-water trawl may be appropriate".
- [F p.4, p.11] "The bottom channel (from bottom to 10 meters above bottom) and the pelagic zone are
  treated separately"; saithe "mostly are registrated shallower than 200-300 meters" (p.10).
- Use [J]: saithe in bottom-trawl catches is not evidence of near-bed residence; no layer shares given.

### A5. Skilbrei & Ottera 2016 - acoustic depth tags, saithe at salmon farms, SW Norway
- Skilbrei, O.T., Ottera, H. 2016. Vertical distribution of saithe (Pollachius virens) aggregating
  around fish farms. ICES J. Mar. Sci. 73: 1186-1195. doi:10.1093/icesjms/fsv261 (read via the OUP
  HTML page; companion: Ottera & Skilbrei 2014, ICES J. Mar. Sci. 71: 2484-2493, doi:10.1093/icesjms/fsu096).
- [F abstract] "Typical residence depth was from 25 to 50 m in summer and 60 to 90 m during winter";
  "The fish were usually distributed through ~100 m of the water column"; no DVM in 38 % of
  observations. [F] "The mean depth below the farms was ~100 m" (50-200 m). Fish 38-59 cm (Table 1).
- [F intro] saithe "may stay close to the bottom at 150-250 m depth" (citing Rosen & Holst 2013).
- [D] Height above the local bed under the farms: ~50-75 m in summer, ~10-40 m in winter.
- Use [J]: farm-attracted coastal fish, but the right size (38-59 cm) - mid-water, not bed-bound.

### A6. Kaartvedt et al. 2005 - large saithe patrolling krill swarms, Norwegian Sea
- Kaartvedt, S., Rostad, A., Fiksen, O., Melle, W., Torgersen, T., Breien, M.T., Klevjer, T.A. 2005.
  Piscivorous fish patrol krill swarms. Mar. Ecol. Prog. Ser. 299: 1-5. doi:10.3354/meps299001.
  PDF: `data/lit/kaartvedt2005/`.
- [F p.3-4] krill swarms at 100-200 m (~70 N 4 E, May 2004); "Underneath most swarms was a 'stack' of
  fish that could extend almost 100 m below the krill"; trawls at 200-300 m caught "four 6 to 10 kg
  (60 to 110 cm) large saithe"; saithe stomachs held "fish and the cephalopod Gonatus sp."; "the
  saithe preyed on planktivores attracted by the krill, rather than on the krill itself".
- Use [J]: oceanic site (bottom far below; not stated in the paper) - saithe forage fully pelagically
  when prey are pelagic. Larger than box saithe.

### A7. Armannsson & Jonsson 2012 - DSTs, Iceland (already in hand)
- ICES J. Mar. Sci. 69: 1372-1381, doi:10.1093/icesjms/fss121;
  `../Sejens diet på unga torskfiskar/data/settlement/armannsson2012_saithe_dst_iceland.pdf`.
- [F p.1373] "Compared with other gadoid species, saithe spend more time in the pelagic habitat and
  less time at the bottom (Scott and Scott, 1988)"; [F p.1378] saithe "are visual feeders that forage
  in the daylight hours"; reduced night activity. Depth only, no bottom reference.

### A8. ICES FishMap saithe sheet (already in hand)
- `../Sejens diet på unga torskfiskar/data/settlement/ices_fishmap_saithe.pdf`. [F] "Adults sometimes
  display extensive diel vertical migrations and may be found hundreds of metres above the bottom
  during night time, as well as close to the bottom, down at depths of more than 300 m during day time."

### A9. Bergstad 1991 (abstracts only; full text paywalled)
- Bergstad, O.A. 1991. Distribution and trophic ecology of some gadoid fish of the Norwegian Deep.
  1. Accounts of individual species. Sarsia 75: 269-313, doi:10.1080/00364827.1991.10413455;
  2. Food-web linkages and comparisons of diets and distributions. Sarsia 75: 315-325,
  doi:10.1080/00364827.1991.10413456. Abstracts via OpenAlex.
- [F abstract 1] "All the species are benthopelagic and epibenthic predators on fish and crustaceans";
  saithe and blue whiting "are the two more abundant gadoids"; western and southern slopes are feeding
  areas. [F abstract 2] "primarily benthopelagic and pelagic predators"; strongest links "small saithe
  and blue whiting as predators and the euphausids as prey, and between saithe and Norway pout".

### A10. Synthesis - share near the bed and H_saithe [J unless tagged]

| Source | Region, season, size | Metric | Near-bed (0-2 / 0-5 m) | Layer H |
|---|---|---|---|---|
| A1 Neilson 2003 | Scotian Shelf, Sep-Oct, pre-spawning, < 45 cm to adults | % of 0-30 m signal | 11-40 % / 27-80 % [D] | ~5-15 m |
| A2 DFO 2003/075 | Scotian Shelf, year round, fishing sets | % of 0-30 m biomass | 14-45 % / ~40-75 % [D] | 3-13 m (75 % line) [D] |
| A3 Aglen 1996 | Norwegian surveys | trawl-acoustic match | - | 30-100 m [F] |
| A5 Skilbrei 2016 | SW Norway coast, 38-59 cm | depth tags | ~0 / ~0 in summer [D] | ~100 m [F] |
| A6 Kaartvedt 2005 | Norwegian Sea, 60-110 cm | echograms | 0 | ~100 m stack [F] |
| A8 FishMap | NE Atlantic | narrative | "close to the bottom" by day | up to hundreds of m at night [F] |

- The only bottom-referenced numbers (A1, A2) come from aggregations that were sampled BECAUSE they sat
  on the bed (spawning/pre-spawning, bottom-trawl confirmed), and they exclude everything above 30 m.
  They give the ceiling: 15-45 % of biomass in the lowest 2 m, H ~ 4-13 m.
- Every source on dispersed, feeding saithe puts them in a layer 30-100 m thick (A3, A5, A6), with
  night ascents of tens to hundreds of metres (A8). Visual daytime feeding (A7) and the A8 statement
  suggest daytime is when they are nearest the bed, but no offshore source separates daytime
  foraging position from night-time position (A1's own day/night test was equivocal).
- Box (100-300 m depth, saithe 40-70 cm, feeding area on the Trench slope; A9): H_saithe central
  ~50 m, range 10 m (winter/spawning aggregations, Q1) to 100 m (summer feeding, dispersed).
- Share of saithe TIME/BIOMASS within ~2 m of the bed: central ~0.05-0.1, range 0.02-0.35.
  [D/J] Year-weighted: ~1/4 of the year in near-bed aggregations at 2-m share ~0.3 plus ~3/4 dispersed
  in a ~50-m layer at ~0.03 (2/50 uniform, a little more with a bottom-weighted profile) gives
  0.25 x 0.3 + 0.75 x 0.03 ~ 0.10. Within ~5 m: central ~0.2 (range 0.05-0.7).
- Share of FORAGING time within ~2 m: lower, central ~0.05 (0.02-0.15) [J]: the near-bed season is
  Q1 spawning, when saithe stomachs hold ~0 haddock at all ages (CRR 219, 1991; see
  `../Sejens diet på unga torskfiskar/summary.md` section 3), i.e. near-bed time is not haddock-feeding
  time; the haddock-eating season (Q3-Q4) is the dispersed feeding season.
- Consistency check [D/J], not a correction factor: the previous folder (`summary.md` section 3) finds
  the encounter a (saithe inside the prey layer all the time) gives ~4x the field diet share (21.5 vs
  ~5.5 %). The within-5-m share here (central ~0.2 = 1/5) reproduces that ~4x gap; the within-2-m
  share (0.05-0.1, i.e. 10-20x) overshoots it. Either the relevant occupancy is the ~5-m one (prey
  reach a few metres up at night, E; a visual predator detects prey some metres below it), or the
  encounter calculation has other offsets not visible from here. Which share to use is for the model
  side to decide.

## B. Other members of the water-column group

### B1. Spurdog - Klocker et al. 2024
- Klocker, C.A., Albert, O.T., Ferter, K., Bjelland, O., Lennox, R.J., Albretsen, J., Pohl, L., Dahlmo,
  L.S., Queiroz, N., Junge, C. 2024. Seasonal habitat use and diel vertical migration in female
  spurdog in Nordic waters. Movement Ecology 12: 62. doi:10.1186/s40462-024-00498-2 (read via PMC11380420).
- [F Results] "median depths at day are significantly deeper (198.5 m, IQR 162-231 m) than at night
  (60.0 m, IQR 37.5-88 m)"; nDVM dominant Nov-May; Jun-Sep "almost exclusively the first 25 m".
- [F Intro/Discussion] tagged in Hardangerfjord and Osterfjord, "deep coastal fjords commonly extending
  to 650 m"; "tagged individuals likely spent a considerable amount of time off the bottom"; the lack
  of horizontal positions "does not permit to link the shark's depth to the local bathymetry".
- [D/J] In 600+ m fjords, 198.5 m by day is ~400 m above the bed: a light/temperature-seeking depth,
  not a bed-seeking one. 19 pregnant females only. In a 100-300 m box the same day depth would put
  some spurdog near the bed by default; by night (60 m) they are far above it. Near-bed share [J]
  ~0.1-0.3 by day, ~0 by night.

### B2. Hake - de Pontual et al. 2012 (abstract)
- de Pontual, H., et al. 2012. Diel vertical migration of European hake Merluccius merluccius and
  associated temperature histories: insights from a pilot data-storage tagging (DST) experiment.
  J. Fish Biol. 81 (doi:10.1111/j.1095-8649.2012.03345.x). [F abstract] "evidence of a diel vertical
  migration pattern"; "Frequent vertical movements across the thermocline"; "rapid temperature changes
  >7 C" (Bay of Biscay). [J] near the bed by day, tens of metres up at night (above a seasonal
  thermocline). Bozzano et al. 2005 (juveniles, NW Med) not accessed.

### B3. Anglerfish - Ofstad et al. 2022
- Ofstad, L.H., Hatun, H., Pedersen, T., Steingrund, P., Mikkelsen, B. 2022. Horizontal and vertical
  migration of anglerfish Lophius piscatorius in relation to hydrography in Faroese waters. Front.
  Mar. Sci. 9: 823066. doi:10.3389/fmars.2022.823066 (10 DSTs, 2009-2012).
- [F] "only around 5% of the total data storage tag recordings showed more than 5 m vertical movement";
  "anglerfish were inactive 95% of the time, so it is evidently a demersal species"; activity
  "higher during night time than day time"; occasional "rapid changes in depth from >150 to 0-80 m".
- [J] the one member for which a 2-m bed layer is correct (~0.95 of time on the bed; ambush feeder).

### B4. Velvet belly - Bergstad, Wik & Hildre 2003 (Skagerrak, 300-700 m)
- J. Northw. Atl. Fish. Sci. 31: 165-180, doi:10.2960/J.v31.a12; full table in the 2001 NAFO SCR
  01/132 version. PDFs: `data/lit/bergstad2003/`.
- [F abstract] "Etmopterus spinax fed mostly on micronektonic crustaceans such as euphausids";
  "predominantly pelagic and benthopelagic predators" (SCR p.7).
- [F Table 4, SCR p.17, %W] < 20 cm: Meganyctiphanes 88.5 %; >= 20 cm: blue whiting 29.5 %,
  Teleostei 16.4 %, Pandalus borealis 12.4 %, Pasiphaea 15.0 % [D: 4.64 + 10.39], M. norvegica 7.1 %, Maurolicus 6.5 %,
  Gadidae 2.4 % (n 128, 56 empty). No juvenile haddock/cod identified.

## C. Saithe diets - new sources (only items not in the previous folder)

| Source | Area, period, size | (a) juv. haddock/cod/flatfish | (b) pout/herring/sprat/blue whiting | (c) krill/zooplankton | (d) whiting | (e) mackerel | Access |
|---|---|---|---|---|---|---|---|
| Bergstad 1991 (A9) | Norwegian Deep 1984-87 | n.g. | Norway pout a main link [F abstract] | euphausiids main prey of SMALL saithe [F abstract] | n.g. | n.g. | abstract only |
| Homrum et al. 2012 | Faroes, adults | n.g. | Aug: "dominated by young blue whiting"; Mar: "Norway pout, sandeel and blue whiting are equally important" [F abstract] | n.g. | n.g. | n.g. | abstract only; doi:10.1080/17451000.2011.627921 |
| Kaartvedt 2005 (A6) | Norwegian Sea, 60-110 cm, n = 4 | 0 | "fish" [F] | not krill itself [F] | - | - | full |
| Nedreaas 1987 | W Norway coast, 0-2 group | 0 | - | "almost only on pelagic plankton"; Calanus most common [F abstract via search] | - | - | not retrieved |

- No new numbers by mass for (a), (d), (e) were found. The quantitative North Sea series remain those
  already in hand (CRR 164, CRR 219, SMS, Mehl 2005: juvenile haddock ~5 % of saithe food).
- Bottom association of prey [J]: the main saithe prey named across these sources (euphausiids, Norway
  pout, blue whiting, herring, sandeel, Maurolicus, Gonatus) are pelagic or benthopelagic; Norway pout
  is the only one with a near-bed (4 m) distribution (A3).

## D. Where saithe catch juvenile haddock (task 4)
- No direct observation of the height at which saithe take haddock was found.
- Indirect [J]: (i) juvenile haddock appear in saithe stomachs mainly in Q3-Q4, after settlement, and
  are ~0 in Q1 (previous folder, `summary.md` section 3); (ii) settled 0-group gadoids leave the bed at
  night to feed in the near-bottom water column (E1) - when saithe, visual feeders, are least active
  (A7, Skilbrei 2016); (iii) saithe "dive in response to ship noise" (A4) and pursue prey vertically
  over tens of metres (A6), so near-bed attacks are likely brief excursions from a layer above, by day.

## E. Task 5 - layer thickness of settled 0-1 group haddock and cod (7-25 cm)

- E1. Lough, R.G., et al. 1989. Mar. Ecol. Prog. Ser. 56: 1-12, doi:10.3354/meps056001;
  `../Sejens diet på unga torskfiskar/data/settlement/lough1989_georgesbank_juvenile_haddock.pdf`.
  [F abstract] "During the day, demersal juveniles remain within a few centimeters of the bottom";
  "At night, regardless of the current, they rise off the bottom to feed on invertebrates in the
  near-bottom water column". [F p.4] "After sunset, part of the population rose off the bottom ca 3 to
  5 m"; "At night, most juveniles were on the bottom, but some fish were several meters up";
  [F p.5] "The larger fish appear to be more demersally oriented; they stay closer to the bottom both
  day and night". Fish 3-9 cm (submersible counts, Georges Bank, 1986-87).
- E2. Aglen, A., et al. 1999. How vertical fish distribution may affect survey results. ICES J. Mar.
  Sci. 56: 345-360, doi:10.1006/jmsc.1999.0449 (abstract only). [F] "A large proportion of fish were
  found in the acoustic bottom dead zone"; "Small haddock and redfish dominated the acoustic recordings
  at night"; the largest day-night catch drops were "in small haddock and redfish ... consistent with
  the observation that these species were pelagic at night" (Barents Sea).
- E3. Korsbrekke, K., Nakken, O. 1999. ICES J. Mar. Sci. 56: 284-291, doi:10.1006/jmsc.1999.0440
  (abstract). [F] day/night catch ratio "For haddock ... highest at the minimum size group, 12-15cm, and
  decreased with increasing size"; for cod it "peaked at a length interval 23-31cm". Engas & Soldal
  1992 (ICES J. Mar. Sci. 49: 89-95, doi:10.1093/icesjms/49.1.89) [F abstract]: "Catch rates of cod and
  haddock were higher by day". [J] small haddock leave the 0-4 m trawl layer at night.
- E4. Michalsen 1996 (ICES J. Mar. Sci. 53: 389-395, doi:10.1006/jmsc.1996.0054) [F abstract] "tidal
  currents affect the vertical distribution of cod and haddock and, hence, the availability to the
  bottom trawl". Aglen 1996 (A3): Norway pout (same size range, 10-20 cm) matches the 4-m headline;
  "large haddock" match 30-100 m [F].
- E5. Bromley & Kell 1999 (previous folder, `evidence_settlement.md` entry 7) - "cod > 8 cm fully
  demersal" (the model's basis); pelagic 0-group (< ~7 cm) migrate vertically through the column.
- Synthesis [J]: by day settled 0-1 group haddock/cod sit within ~0-1 m of the bed (E1, E2 dead zone);
  at night a fraction (small haddock most, larger cod least) rises to ~3-5 m (E1) and, in the Barents
  Sea, small haddock become "pelagic" (E2, E3), i.e. above the 4-5 m headline. Day-night average
  H_juv central ~2 m (day ~1 m, night ~3-5 m), range 1-5 m; haddock thicker than cod. The model's 2 m
  is supported as a central value; a 12-25 cm haddock night layer of ~5-10 m cannot be excluded
  (E2 gives no height). The prey is highest off the bed at night, when saithe are least active.

## F. Not accessed / gaps
- Bergstad 1991 Sarsia 75 parts 1-2 (Taylor & Francis 403; Unpaywall no OA): the one source with saithe
  diet and depth by season on the Trench slope. Abstracts only.
- Du Buit 1991 Fish. Res. 12: 307-323 (doi:10.1016/0165-7836(91)90015-8): paywalled, no abstract online.
- Homrum et al. 2012 (Faroes): abstract only. Nedreaas 1987 (young saithe, W Norway): search snippet only.
- Aglen 1996, Aglen et al. 1999, Korsbrekke & Nakken 1999, Michalsen 1996, Engas & Soldal 1992:
  OUP PDFs not retrievable here; abstracts only, so no height-by-length numbers.
- Lough & Potter 1993 (Fish. Bull. 91: 281-303, juvenile haddock/cod vertical distribution): PDF link
  redirects; not read. Rosen & Holst 2013 (DeepVision, saithe near bed at 150-250 m): not read.
- Sarno et al. 1994 / Glass et al. 1992 (saithe telemetry at a reef, Scottish sea loch): not read.
- Norwegian/German saithe acoustic-trawl survey (SATS / NORACU, Norwegian Trench shelf edge, spawning
  season; Thunen WH cruise 414, 2018): report not retrievable; the Thunen project states "an unknown
  proportion of saithe is distributed outside of the sampled layers" (search snippet). This survey
  is the best future source for bottom-referenced saithe profiles in the box itself.
- Mattson 1990/1992 (Sarsia soft-bottom fish diets): parts found are cod and haddock, not saithe.
- No source gives bottom-referenced saithe profiles from the Norwegian Trench/Skagerrak, and none
  separates daytime foraging height from night-time resting height for offshore saithe.
