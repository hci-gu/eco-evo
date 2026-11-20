# Seasonal energy-reserve series for saithe, whiting and hake (literature, 2026-11-20)

Purpose: field monthly or quarterly series of liver or lipid reserves (HSI, liver lipid,
condition) to replace the model's proxy fills: whiting uses a haddock proxy (0.30 in
May-Jul, 0.73 in Sep-Nov) and the water column (mostly saithe) a flat 0.71.

Tags: [F] full text read; [A] abstract only; [S] secondary; [D] derived here (arithmetic
shown); [J] judgement. Values marked "fig" were read off a figure; the reading precision is
given each time. HSI denominators differ between sources (whole weight, gutted weight,
or weight minus stomach contents). Each table states which one it uses.

Already known and not redone (cite only): NIFES 2013 saithe liver by quarter (Norwegian /
Barents Sea); Dominguez-Petit et al. 2010 hake Galicia HSI; Cooper 1983 whiting W Scotland;
Hislop 1991; Pedersen & Hislop 2001; Bull 1928; NIFES haddock liver lipid; NEA cod / haddock
HSI; Skjaeraasen et al. 2017 (captive saithe).

---------------------------------------------------------------------------------------------

## 1. Saithe (Pollachius virens)

**Bottom line [J]:** no new monthly HSI series for wild saithe exists in the open literature
searched (OpenAlex, Crossref, Unpaywall, web; English, Norwegian, Icelandic terms). What there is:
(a) one North Sea series of condition factor with four sampling dates, from fish of the box's size;
(b) coastal-Norway HSI in three seasons for large, mostly immature fish, which is flat;
(c) a summer HSI level for wild control fish, with the farm-fed maximum beside it.

### 1.1 Mathers, Houlihan & Cunningham 1992, North Sea (Beryl oilfield)
Mathers EM, Houlihan DF, Cunningham MJ (1992) Estimation of saithe *Pollachius virens* growth
rates around the Beryl oil platforms in the North Sea: a comparison of methods. Mar Ecol Prog
Ser 86:31-40. doi:10.3354/meps086031 [F]

- Area and dates: Beryl oilfield, northern North Sea. Four trips: Aug 1988,
  Oct 1988, Apr 1989, Aug 1989. Ages 2+ to 5+. In Aug 1989, mean total length was 41.2 +- 1.0 cm
  (n 18) at Beryl Alpha and 43.5 +- 3.1 cm (n 5) at Bravo. This is the best size match to the box
  (35-60 cm). The paper does not report maturity [F].
- Indices: Fulton K, the caecal-somatic index (CSI), and GLI = gall-bladder weight x 100 / liver
  weight. GLI measures feeding (bile is stored when the fish do not feed), not the reserve. The
  liver-somatic index was measured, but the paper gives it only for Aug 1989 (Table 2). It does
  not report seasonal LSI values.
- Size correction: the paper corrects a parameter to a 500 g standard fish only where the
  allometry was significant. Table 3 lists RNA, the enzymes and CSI (CSI is not significant),
  but not K. So Fig. 3A K is taken to be UNCORRECTED raw K, and the catch's size composition
  differs between dates [F/J].

| Date | K raw (fig 3A, +-0.02) | CSI % (fig 3B, +-0.05) | GLI (fig 3C, +-0.5) | LSI % (Table 2) |
|---|---|---|---|---|
| Aug 1988 | 1.13 | 1.8-1.9 | ~0.3 | - |
| Oct 1988 | 1.09 | 1.48 | ~0.3 | - |
| Apr 1989 | 0.87 | 1.35 | 16.5 | - |
| Aug 1989 | 1.12 | 1.9 | 3.2-3.5 | 5.40 +- 0.49 (Alpha, n 18); 5.95 +- 0.76 (Bravo, n 5) |

- Mean weight by age class, as printed (Table 1; not cohort-tracked across the year boundary):
  age 5+ was 1388 g (Aug 88, n 5), 1255 g (Oct 88, n 16), 805 g (Apr 89, n 21) and 1137 g
  (Aug 89, n 7). Age 4+ was 935, 931, 657 and 704 g; the last is a single fish (n 1). The paper reports a significant fall from
  October to April and a rise from April to August [F].
- Text [F]: "highest values in August and the lowest in April in most cases"; GLI "increased in
  winter because bile accumulated as feeding was reduced". The growth estimates from RNA were
  negative (weight loss) in April 1989.
- [D] Amplitude of condition: K min / max = 0.87 / 1.13 = 0.77. Minimum in April (after the
  Feb-Mar spawning, end of winter), maximum in August.

### 1.2 Uglem et al. 2020, coastal mid-Norway (Hitra), unassociated controls
Uglem I, Toledo-Guedes K, Sanchez-Jerez P, Ulvan EM, Evensen TH, Saether BS (2020) Does waste
feed from salmon farming affect the quality of saithe (*Pollachius virens* L.) attracted to fish
farms? Aquaculture Research 51:1720-1730. doi:10.1111/are.14519 [F] (accepted manuscript, RUA
repository, hdl 10045/103852)

- Dates: "six different occasions during summer, autumn and spring in 2013-2015". The paper does
  not give months [F]. HSI = liver weight / whole weight x 100 (whole weight measured before
  gutting).
- Maturity: the fish are mostly immature. Only 15 of 181 spring fish (8.2 %) had GSI > 2 [F].
- The authors excluded the Ryfylke "UA" fish, because 7-19 % of them had pellets in their
  stomachs. Only the Hitra UA fish are used here as controls.

| Season (Hitra UA) | Gear | n | TL cm | K | HSI % | GSI % |
|---|---|---|---|---|---|---|
| Autumn | jig | 30 | 68 +- 4 | 0.95 +- 0.13 | 6.9 +- 3.1 | 1.82 |
| Autumn | net | 30 | 68 +- 3 | 0.94 +- 0.10 | 6.8 +- 1.4 | 1.79 |
| Spring | net | 31 | 66 +- 6 | 0.84 +- 0.07 | 6.2 +- 4.4 | 0.63 |
| Summer | jig | 20 | 62 +- 6 | 0.86 +- 0.07 | 6.0 +- 2.2 | 0.25 |
| Summer | net | 20 | 69 +- 4 | 0.90 +- 0.08 | 6.8 +- 2.7 | 0.43 |

Farm-associated (FA) Hitra fish had HSI 5.6-10.7 (means). Ryfylke FA fish had up to 11.6, and
the excluded Ryfylke "UA" spring fish 12.0 +- 5.1.
- [D] HSI min / max for UA = 6.0 / 6.9 = 0.87; K = 0.84 / 0.95 = 0.88. Condition is lowest in
  spring and highest in autumn.
- [J] This is a WEAK test of whether immatures cycle. These fish (62-69 cm) are larger than the
  box fish. "Immature" is the authors' inference from mean GSI. Autumn GSI (1.8) against spring
  (0.6) could equally mean maturing fish in autumn and spent fish in spring, i.e. a mixed
  population. The SDs (1.4-4.4) are large, and there are only three seasons with no months. The
  immature HSI cycle is therefore UNRESOLVED, not shown to be flat.

### 1.3 Dempster et al. 2011, coastal Norway, summer, controls versus farm fish
Dempster T, Sanchez-Jerez P, Fernandez-Jover D, Bayle-Sempere J, Nilsen R, Bjorn P-A, Uglem I
(2011) Proxy measures of fitness suggest coastal fish farms can act as population sources and
not ecological traps for wild gadoid fish. PLoS ONE 6:e15646. doi:10.1371/journal.pone.0015646 [F]

- June to September 2007, at Ryfylke (59 N), Hitra (63.6 N) and Oksfjord (70 N). HSI = liver
  weight / (wet weight - stomach contents) x 100.
- UA saithe fork length (Table 1): Ryfylke 54.2 cm (n 30), Hitra 34.3 cm (n 88), Oksfjord
  43.4 cm (n 97). Mostly immature (the authors: "less than 2 kg ... likely to be immature").

| Area | UA HSI (fig 3b, +-0.2) | FA HSI (fig 3b, +-0.2) | UA FCI (fig 3a, +-0.01) | FA FCI |
|---|---|---|---|---|
| Ryfylke | 8.5 | 9.3 | 1.08 | 1.17 |
| Hitra | 6.0 | 8.7 | 1.15 | 1.28 |
| Oksfjord | 5.2 | 9.3 | 1.10 | 1.17 |

- The text gives FA/UA = 1.4-1.8 for HSI and 1.06-1.12 for FCI. Ryfylke shows no difference [F],
  and its UA fish may be farm-influenced (see 1.2) [J].
- [J] Summer (Jun-Sep) HSI of unsubsidised immature saithe is 5-6 %, at the North Sea Aug 1989
  level (5.4-6.0). Farm feeding raises it to about 9-10 % (means) and above 10 % in individuals;
  "HSI > 10" is the authors' own class for long farm residence. So the liver can reach about 1.5
  times the natural summer level when food is unlimited. The natural level is food-limited, and
  ample food lifts it well above the field level.

### 1.4 Searched but not usable
- LOCATED BUT NOT READ: Fernandez-Jover D et al. (2011) Waste feed from coastal fish farms: a
  trophic subsidy with compositional side-effects for wild gadoids. Estuar Coast Shelf Sci
  91:559-568. doi:10.1016/j.ecss.2010.12.009. There is an OA submitted version on figshare
  (record 25000757). It may give liver lipid and HSI of control saithe; the season is unknown.
- LOCATED BUT NOT READ: a Skog et al. study of saithe near Norwegian salmon farms (higher
  liver-somatic index in farm fish), surfaced in a search (core.ac.uk/download/498695265.pdf).
  Citation and season not checked.
- Hojgaard DP 1997, Frodskaparrit 45:57-68 (young Faroese saithe, sampled every two months in
  1996). The abstract covers parasites only and reports no condition [A].
- Sab1996a.pdf (AWI ePIC, epic2-clone.awi.de/3752): not retrieved (connection refused). Its species
  and content are not verified, so it is NOT cited as a saithe source.
- Lovern et al. 1933, Biochem J 27:1461: liver oils of halibut, not saithe.
- Uglem et al. 2017 J Appl Ichthyol 33:374 (taste) and Arechavala-Lopez et al. 2015, Mar Coast Fish
  7:59, doi:10.1080/19425120.2014.1001540: liver lipid FA versus UA, no seasonal series [A].
- Pollock in the Gulf of Maine and Scotian Shelf: no liver or condition series found (web search).

---------------------------------------------------------------------------------------------

## 2. Whiting (Merlangius merlangus)

### 2.1 Erdogan Saglam & Saglam 2012, SE Black Sea: monthly HSI by sex (best series)
Erdogan Saglam N, Saglam C (2012) Population parameters of whiting (*Merlangius merlangus euxinus*
L., 1758) in the South-Eastern Black Sea. Turkish J Fish Aquat Sci 12:831-839.
doi:10.4194/1303-2712-v12_4_11 [F]

- Sampling: monthly, with gillnets (32 mm mesh, the same net every month) at 25-70 m, Sinop to
  Giresun. n = 1884 in all; monthly n not given. TL 10.1-23.1 cm, mean 14.9 cm. This is the small
  Black Sea form: L50 is about 13-14 cm, so most fish are mature [S, from a western Black Sea
  study seen in a search snippet; J].
- Date inconsistency in the paper: the abstract says 2009-2010; the Methods say Nov 2010 - Jan
  2012; the figure axis runs Nov-10 to Oct-11.
- HSI = liver weight / total weight x 100.
- Spawning (the paper's GSI): mainly March-July, with some autumn spawning. GSI drops to its
  minimum in July.
- Text [F]: "HSI was higher from November to April and lower from May to October with an
  increasing trend." The authors attribute the March-July decrease to reproduction.

HSI % (fig 6, read +-0.1):

| Month | Nov | Dec | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Female | 6.0 | 6.25 | 6.3 | 5.6 | 6.15 | 5.55 | 2.5 | 3.0 | 2.35 | 3.0 | 4.0 | 4.2 |
| Male | 4.4 | 5.5 | 4.7 | 4.3 | 3.9 | 4.4 | 2.0 | 2.5 | 2.3 | 1.85 | 3.1 | 3.9 |
| Both | 5.2 | 5.85 | 5.5 | 5.0 | 5.05 | 5.0 | 2.2 | 2.7 | 2.3 | 2.4 | 3.55 | 4.0 |

- [D] Female min / max = 2.35 (Jul) / 6.3 (Jan) = 0.37. Male = 1.85 (Aug) / 5.5 (Dec) = 0.34.
  Both sexes = 2.2 (May) / 5.85 (Dec) = 0.38.
- [J] The fall from April to May is a single step (5.0 -> 2.2). The gear is the same, but the
  depth range (25-70 m) and the port varied, so part of the step may be a sampling artefact. The
  recovery from August to December is gradual and looks biological.

### 2.2 Lauerburg 2016, North Sea: HSI by sex and length, Jul-Aug 2007 versus Apr 2012
Lauerburg RAM (2016) Governing factors for the size-at-age: the influence of sex, food
availability, temperature and fishery on the growth performance of North Sea whiting *Merlangius
merlangus* and cod *Gadus morhua*. PhD thesis, Univ. Hamburg. https://ediss.sub.uni-hamburg.de/handle/ediss/6770 [F]

- 2007: German small-scale bottom-trawl survey boxes, July-August. 2012: VECTORS cruise in April
  (chapter II calls it "first quarter"). GOV trawl. The denominator is inconsistent in the thesis:
  chapter II's Methods say liver / somatic mass x 100, while the Table II-VI caption says "total
  fish body mass". Chapter I uses total mass and notes that gutted mass gives values 9-17 %
  higher.
- Median HSI (Table II-VI):

| Sex | TL cm | Jul-Aug 2007 | Apr 2012 | n 2007 / 2012 |
|---|---|---|---|---|
| F | 25-29.9 | 3.76 | 3.13 | 99 / 147 |
| F | 30-34.9 | 2.88 | 3.72 | 65 / 219 |
| F | 35-39.9 | 2.58 | 4.36 | 18 / 120 |
| F | 40-49.9 | 2.80 | 6.10 | 3 / 42 |
| M | 25-29.9 | 2.41 | 2.44 | 91 / 161 |
| M | 30-34.9 | 2.10 | 2.48 | 45 / 135 |
| M | 35-39.9 | 2.04 | 3.67 | 8 / 15 |

- In 2012, mean relative liver mass rose with length, from 1.45 % (F) and 1.21 % (M) below 15 cm
  to 4.6 % (F) and 3.8 % (M) at 35-39.9 cm. At equal length, females are higher than males by
  0.25-1.27 % [F].
- Year is confounded with season, and the thesis gives both readings: chapter I explains the lower
  2007 values by spawning losses (and notes that female livers in April carry vitellogenin);
  chapter II by low food in 2007 (stomach contents about half). Not usable as a seasonal amplitude
  [J]. It does show that summer HSI after spawning in large whiting is about 2-3 %, and that it
  rises steeply with length.

### 2.3 Jonsson 2011, Iceland (Matis report 36-11): condition and fillet fat in four seasons
Jonsson A (2011) Breytileiki a eiginleikum lysu (*Merlangius merlangus*) eftir arstima / Seasonal
variations in quality and processing properties of whiting. Matis report 36-11, Reykjavik.
https://matis.is/wp-content/uploads/skyrslur/36-11-Lysa-eiginleikar-eftir-arstima.pdf [F] (in Icelandic)

- Area and dates: SE of Vestmannaeyjar; Oct 2010, Mar 2011, Jun 2011, Aug 2011. Gutted fish;
  10 fish per sample for K. TL was 47 +- 7 cm (Oct), 46 +- 4 (Mar), 40 +- 2 (Jun, smaller fish)
  and 48 +- 3 (Aug).
- Gutted-weight Fulton K (Mynd 4, bars read +-0.02; error bars are SD of 10 fish): Oct 0.87,
  Mar 0.91, Jun 0.77, Aug 0.72. The text gives a range of 0.72-0.91 and says March is
  significantly higher than August (P = 0.02). The August minimum follows the May to mid-July
  spawning [F]. No liver was measured.
- Fillet fat (Mynd 10, read +-0.005 %; pooled sample of 10 fish, measured three times): Oct 0.150,
  Mar 0.077, Jun 0.136, Aug 0.100 %. The text says October is highest and March significantly
  lower than October and June [F].
- [D] K min / max = 0.72 / 0.91 = 0.79. Maximum in March (before spawning), minimum in August
  (after spawning). The June fish were smaller (40 cm), and K is size-confounded.

### 2.4 Searched but not usable
- Vallisneri et al. 2006, Acta Adriat 47:159-165 (Adriatic reproduction): GSI only, no HSI [F].
- A Fish Physiol Biochem study of Black Sea whiting (Fulton K, Kr and the liver condition index
  LKr, used to predict maturity): known only from a repository listing. Monthly values unknown.
- Aksu 2020 (JNRS 9:39) and Bal 2021 (MSTB 10:99, Marmara): no liver data [F, by grep].

---------------------------------------------------------------------------------------------

## 3. European hake (Merluccius merluccius)

### 3.1 Costa 2013, Portuguese coast, monthly HSI 2005-2010
Costa AM (2013) Somatic condition, growth and reproduction of hake, *Merluccius merluccius* L., in
the Portuguese coast. Open J Mar Sci 3:12-30. doi:10.4236/ojms.2013.31002 [F]

- Data: market samples from 12 ports plus on-board samples, Jan 2005 - Dec 2010, pooled by month.
  8212 fish screened; monthly n not given. HSIt = liver / total weight x 100. The groups are split
  by length (females < / > 38 cm, males < / > 26 cm), not by staged maturity.
- Text [F]: the HSI pattern is "similar for mature and immature fish". The female maximum comes at
  the start of Q4 (October). Males above 26 cm show "a sharp increase in June". GSI rises through
  the first half-year while HSI tends to fall. Hake spawn almost all year, with several peaks.

HSIt % (fig 7, read +-0.3):

| Month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F < 38 cm | 3.8 | 2.9 | 3.1 | 3.4 | 3.1 | 3.3 | 5.7 | 4.1 | 5.7 | 8.5 | 5.5 | 3.6 |
| F > 38 cm | 5.2 | 4.3 | 3.2 | 3.4 | 3.3 | 3.6 | 5.8 | 4.1 | 4.2 | 5.7 | 5.0 | 3.6 |
| M > 26 cm | 4.6 | 2.7 | 2.4 | 2.6 | 2.4 | 8.0 | 2.5 | 3.8 | 4.0 | 4.4 | 2.9 | 1.7 |

- [D] Females > 38 cm: min / max = 3.2 (Mar) / 5.8 (Jul) = 0.55. Females < 38 cm: 2.9 (Feb) /
  8.5 (Oct) = 0.34.
- [J] The October value for small females (8.5) and the June value for males (8.0) are
  single-month spikes in pooled landings data. They should not be taken as the cycle maximum
  without that caveat. The robust feature is a low plateau from February to June (about 3) and
  higher values from July to November (4-6).

### 3.2 Searched but not retrieved
NW Mediterranean ("Parasitism, condition and reproduction of the European hake in the
northwestern Mediterranean", ICES JMS 71:1088, 2014, doi:10.1093/icesjms/fst217; authors not
checked): OUP returned 403. The northern and central Adriatic (JMSE 9:389, 2021,
doi:10.3390/jmse9040389) and Oran Bay (2022, doi:10.15666/aeer/2002_947965) reproduction papers
were seen by title only and may contain monthly HSI; they were not read.

---------------------------------------------------------------------------------------------

## 4. What sets the cycle: phase against spawning

| Species / area | Spawning | Minimum | Maximum | Amplitude min/max | Source |
|---|---|---|---|---|---|
| Saithe, N North Sea (2+-5+, ~42 cm) | Feb-Mar (stock) | K (raw): Apr (only winter-spring date) | K: Aug | 0.77 (K) [D] | Mathers 1992 |
| Saithe, mid-Norway coast (62-69 cm, mostly immature) | few spawners | K: spring; HSI flat | K: autumn | HSI 0.87, K 0.88 [D] | Uglem 2020 |
| Whiting, SE Black Sea (~15 cm, mature) | Mar-Jul (+ autumn) | HSI: May-Aug | HSI: Dec-Jan | 0.34-0.38 [D] | Saglam & Saglam 2012 |
| Whiting, Iceland (~46 cm) | May - mid-Jul | K (gutted): Aug | K: Mar | 0.79 (K) [D] | Jonsson 2011 |
| Whiting, W Scotland | (spring) | K: Apr-Jun | K: Aug-Sep | - | Cooper 1983 (known) |
| Hake, Portugal | almost all year, peaks | HSI: Feb-Jun | HSI: Jul-Nov | 0.34-0.55 [D] | Costa 2013 |

[J] For WHITING, the condition or HSI minimum falls at or just after the local spawning season,
not in a fixed calendar month: August in Iceland (spawning May-mid Jul), May-August in the Black
Sea (spawning Mar-Jul), and Apr-Jun in W Scotland (the known case). For whiting, the phase is
better transferred relative to spawning time. Lauerburg 2016 [F] gives North Sea spawning as
"January until July", from south to north. A minimum around May-Jul for Skagerrak whiting is
therefore plausible [J], close to the haddock-proxy timing now in the model. This does not hold
for the other two species. Hake spawns almost all year, so its Feb-Jun low cannot be tied to
spawning. In saithe, Mathers' April date is the only winter-spring sample, so it cannot separate
post-spawning loss from winter fasting (GLI points to fasting).

Drivers stated in the sources:
- **Feeding season / food:** Mathers 1992 has winter fasting (GLI x50 in April), weight loss from
  October to April and negative RNA growth estimates in April. Lauerburg 2016 reports HSI 15-54 %
  lower in a low-food year at equal length. Dempster 2011 and Uglem 2020 find that farm food lifts
  HSI 1.4-1.8x, so the liver tracks food directly.
- **Reproduction:** Saglam & Saglam 2012 attribute the drop in HSI from March to July to spawning.
  Lauerburg 2016 notes that vitellogenin inflates female liver weight before and during spawning.
  Costa 2013 finds GSI rising while HSI falls in H1.
- **Immatures:** the only "immature" saithe series (Uglem 2020, coastal, 3 seasons, 62-69 cm,
  maturity inferred from GSI) shows K varying by about 12 % but no detectable HSI cycle. This is a
  weak test; the question is unresolved. The hake immature-size group (F < 38 cm) does cycle,
  and Costa says it matches the mature fish. No source here tests "soma before liver" directly.
- **Temperature:** no source in this set separates temperature from food.

## 5. Gaps
- Saithe: no monthly HSI or liver lipid for wild fish at 35-60 cm in the North Sea or Skagerrak.
  The best available is K at 4 dates (Mathers) and HSI in 3 seasons for coastal immature fish
  (Uglem). The minimum is likely April and the maximum August-October, but the HSI amplitude
  for immatures is unknown and may be small.
- Whiting: no North Sea or Skagerrak monthly HSI. The only monthly HSI is for the small Black Sea
  form, whose spawning timing differs.
- Hake: only Portugal was read; nothing from the North Sea.
- No source gives HSI at a fixed length across seasons. The series are mean HSI of the catch,
  which carries a size-composition bias (Saglam, Costa).
