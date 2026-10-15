# Herring winter food and winter feeding: Skagerrak / Kattegat (Mareld site, ~58.1 N 10.8 E, ~100 m)

Model state checked (supplied by the caller, not re-derived here):
- `zooplankton` (mesozooplankton, ~2/3 small neritic copepods, ~1/3 Calanus; also the gadoids' krill proxy).
  At the cells where herring are: Jan 1.5, Feb 3.2, Mar 6.7, Apr 10.5, May 12.5, Jun 11.9, Jul 10.6,
  Aug 8.8, Sep 6.5, Oct 4.7, Nov 2.6, Dec 1.5 g WW m-2 (whole column). Grid annual mean ~7-13 (B_ref 9.7).
- Herring ration (zooplankton eaten, % body weight per day, wet): Dec 0.18, Jan 0.30, Feb 1.4, Mar 3.5,
  Apr-Jul 2.3-2.9, Sep 1.8, Oct 1.1, Nov 0.5 (Aug not supplied).
- Herring energy: lean 4000 MJ/t, reserve up to 7000 MJ/t (~18 % fat full; 396 MJ/t per fat-%).
  Reserve fill Jan 0.26-0.29 (~5 % fat), Sep 0.71-0.75, May-Jul 0.79-0.86.
  Winter cost 23-27 kJ/kg/d (Dec-Feb, ~60 % resting), summer 44-57 kJ/kg/d.
- Library values used in my calculations (read from `fgconfig/fg_library.yaml`): zooplankton
  `energy_content` 4500 MJ/t (4.5 kJ/g WW); `pelagic_fish_preys_on_zooplankton` `assimilation_factor` 0.8.
- Night attack rate 10 % of day attack (daylight calendar).

Builds on: `reports/Vintermetabolism sill och tumlare.md` and
`research_notes/Vintermetabolism sill och tumlare/herring_fat_cycle.md` (fat cycle, Q10, Kenyon 2022,
Rojbek 2014, JNCC 633, Onsrud 2004 krill frequencies). Those are not redone here; they are cited as
"prior note".

Conventions: **[F]** fulltext read, **[A]** abstract read, **[S]** search snippet only (NOT verified),
**[2nd]** secondary (number read in another document that cites the original), **[D]** computed by me
from raw data. Every section separates *Literature*, *Calculation* and *Inference*.

---

## Sources and how they were read

| Source | Read as | Link |
|---|---|---|
| SMHI national monitoring, SHARK zooplankton data, stations Å17 and Anholt E | [D] raw data via API (2026-10-15) | https://shark.smhi.se/api/docs |
| Falkenhaug, Broms, Bagøien & Nikolioudakis 2022, Calanus at Arendal St. 2, Skagerrak | [F] (first ~100k chars) | https://doi.org/10.3389/fmars.2022.779335 |
| Onsrud, Kaartvedt, Røstad & Klevjer 2004, ICES JMS 61:1278, Oslofjord fish on krill | [F] (first 100k of 110k chars) | https://academic.oup.com/icesjms/article/61/8/1278/629825 |
| Lohmeyer & Hempel 1977, ICES CM 1977/P:7, winter food of western Baltic herring | [F] (OCR text layer of scanned PDF) | https://www.ices.dk/sites/pub/CM%20Doccuments/1977/P/1977_P7.pdf |
| Bernreuther 2007, PhD thesis Univ. Hamburg, herring/sprat feeding, Bornholm Basin (manuscripts 1 and 7) | [F] (pdftotext) | https://ediss.sub.uni-hamburg.de/bitstream/ediss/1765/1/DissertationMatthiasBernreuther.pdf |
| Segers, Dickey-Collas & Rijnsdorp 2007, ICES JMS 64:60, North Sea herring diet in February | [F] (OCR text, partially garbled) | https://www.vliz.be/imisdocs/publications/ocrd/260977.pdf (doi:10.1093/icesjms/fsl002) |
| Last 1989, J Fish Biol 34:489, food of North Sea herring 1983-86 | [A] via OpenAlex; publisher page 403 | https://doi.org/10.1111/j.1095-8649.1989.tb03330.x |
| Mackinson & Daskalov 2007, Cefas Tech. Rep. 142, North Sea Ecopath (sections 3, 10.2, 13.26) | [F] (pdftotext) | https://www.cefas.co.uk/publications/techrep/tech142.pdf |
| Båmstedt & Karlson 1998, MEPS 172:149, euphausiid predation on copepods, Skagerrak | [F] (abstract and methods skimmed) | https://www.int-res.com/articles/meps/172/m172p149.pdf |
| Kiørboe & Nielsen 1994, L&O 39:493, copepods southern Kattegat | [A] via OpenAlex | https://doi.org/10.4319/lo.1994.39.3.0493 |
| Arrhenius & Hansson 1994, MEPS 110:145, YOY herring consumption | [A] via OpenAlex | https://doi.org/10.3354/meps110145 |
| Arrhenius & Hansson 1999, MEPS 191:295, YOY herring growth resource-limited | [A] via OpenAlex | https://doi.org/10.3354/meps191295 |
| Bachiller et al. 2018, PLoS ONE 13:e0190345, NE Atlantic pelagic consumption (NSS herring) | [F] (first 100k chars) | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190345 |
| Casties 2025, SMHI/GU presentation, summer zooplankton WW | [2nd] via sibling note `Zooplankton energibudget/copepod_energetics.md` | https://havsmiljo.se/pdf/presentationer/vms25/zooplankton-casties-vattenmiljoseminariet-jan-2025-final.pdf |
| Kenyon et al. 2022; Rojbek et al. 2014; JNCC 633 (Pedersen & Hislop 2001); Slotte 1999; Polte et al. 2021 | [2nd] via prior note (read there as fulltext/abstract) | see prior note |

**Failed fetches (recorded, not used):**
- int-res.com abstract pages for MEPS 228:193 (Kaartvedt et al. 2002, krill carnivory, Oslofjord) and
  MEPS 172:149: HTTP 429 (rate limit, retry-after 3600 s). MEPS 172:149 PDF later obtained directly;
  MEPS 228:193 PDF URL returned 404.
- Wiley (Last 1989 fulltext): HTTP 403. Abstract obtained through OpenAlex instead.
- Springer (Boysen & Buchholz 1984, Mar Biol 79:195, Kattegat krill, doi:10.1007/BF00951828): redirect
  to login (303); OpenAlex has no abstract. epic.awi.de records for Buchholz's Kattegat krill papers:
  "Access Denied" (Anubis bot protection). **No Kattegat/Skagerrak krill density was obtained.**
- Nielsen et al. 2001 (Fish. Res. 50:235, WBSS in the Sound): OpenAlex has no abstract; not read.
- Clausen et al. 2015 (J Fish Biol, herring condition by area): only a search snippet; not used for numbers.
- Dana vol. 10 pp. 29-59 (Parmanne et al. 1994): fetched, but it is a Baltic stock review with no
  seasonal growth numbers; not used.

---

## Q1. Seasonal cycle of mesozooplankton biomass; winter vs summer

### Literature
- Kiørboe & Nielsen 1994 (southern Kattegat): "The seasonal distribution of copepod biomass ... was
  unimodal with peak concentrations in June-July"; "the continuous decline in copepod biomass between
  early August and late October overlapped the most significant production event" and "must ... have
  been caused by elevated mortality" [A]. No winter values in the abstract.
- Falkenhaug et al. 2022, Arendal St. 2 (northern Skagerrak, 75 m bottom, WP2 180 um, 50-0 m, 1994-2019)
  [F]: C. finmarchicus CV "present in low numbers in winter (< 90 ind. m-2 during early September - mid
  April)", annual maximum ~2700 ind m-2 (median) in early May; adult females practically absent until
  early January, peak ~600 ind m-2 in late March; Calanus CI-IV medians < 80 ind m-2 from late December
  to late February. Overwintering: the station lies "on the northern slope of the Norwegian Trench";
  diapausing individuals enter via the deep Norwegian Trench Atlantic inflow; C. finmarchicus "is
  believed not to overwinter in the North Sea - Skagerrak in substantial numbers"; "no information on
  vertical distributions are available from this site". No biomass values in the paper.
- Casties 2025 (summer Jun-Aug total zooplankton WW; [2nd] via sibling note, approximate figure
  readings): Anholt E and Släggö ~10-55 g WW m-2 in 1998-2017, all stations ~3-17 g WW m-2 in 2018-2023.
- Mackinson & Daskalov 2007 [F]: North Sea Ecopath herbivorous/omnivorous zooplankton 16 t WW km-2.

### Calculation (SHARK raw data) [D]
Request (reproducible):
```
POST https://shark.smhi.se/api/sample/download   (Content-Type: application/json)
{"params":{"tableView":"sharkdata_zooplankton","delimiters":"point-tab","lineEnd":"unix",
           "encoding":"utf_8","headerLang":"internal_key"},
 "query":{"fromYear":1990,"toYear":2025,"dataTypes":["Zooplankton"],
          "stationName":["Å13","Å14","Å15","Å16","Å17","A17","A16","SLÄGGÖ","ANHOLT E",
                         "N14 FALKENBERG","FLADEN","BROA E"]}}
```
Returned 102,981 rows; only Å17 (58.275 N 10.513 E, water depth ~345 m, the station nearest Mareld),
ANHOLT E (56.67 N 12.12 E, Kattegat, ~60 m) and one FLADEN visit. **SLÄGGÖ has no zooplankton data in
SHARK** (a sea-basin query for Skagerrak + Kattegat confirmed only Å17, Anholt E, Fladen). No
wet-weight parameter exists for these stations; "Carbon content" in ug m-3 exists only for 2019-2023
(Å17 37 visits, Anholt E 59 visits). All carbon hauls are WP2, 90 um mesh, **0-25 m** (one 0-30 m).
Carbon quality flags: none set (4174 rows, all blank). Carbon covers crustacean mesozooplankton,
appendicularians and chaetognaths (Calanoida 75 %, unidentified Copepoda 9 %, Cyclopoida 6 %,
cladocerans 5 %, appendicularians 3 %, Sagitta 2 % of all carbon) - it matches the model's
`zooplankton` definition well; meroplankton, medusae and krill are not in it.

Processing: sum of all taxon/stage rows of "Carbon content" (ug m-3) per sample; mean per visit; mean of
visits per calendar month. 0-25 m stock: mg C m-2 = ug m-3 x 25 / 1000. WW conversion stated
explicitly: C/WW = (DW/WW) x (C/DW). With DW/WW 0.2 and C/DW 0.42 -> C/WW 0.084 (sibling-note value);
with DW/WW 0.15 and C/DW 0.4 -> C/WW 0.06. So g WW m-2 = g C m-2 / 0.084 (low) to / 0.06 (high);
the high value is 1.4x the low.

**Å17 (eastern Skagerrak), 0-25 m, 2019-2023, n = 2-4 visits per month**

| Month | C ug m-3 (mean) | range | Calanus share of C | mg C m-2 (0-25 m) | g WW m-2 (C/WW 0.084-0.06) | Model, herring cells (whole column) |
|---|---|---|---|---|---|---|
| Jan | 3,923 | 836-7,011 | 0.70 | 98 | 1.2-1.6 | 1.5 |
| Feb | 2,477 | 1,358-3,662 | 0.71 | 62 | 0.7-1.0 | 3.2 |
| Mar | 14,490 | 6,752-22,227 | 0.64 | 362 | 4.3-6.0 | 6.7 |
| Apr | 64,867 | 37,296-83,863 | 0.88 | 1,622 | 19.3-27.0 | 10.5 |
| May | 87,326 | 34,984-184,187 | 0.76 | 2,183 | 26.0-36.4 | 12.5 |
| Jun | 38,426 | 12,560-80,475 | 0.35 | 961 | 11.4-16.0 | 11.9 |
| Jul | 23,059 | 8,243-37,354 | 0.47 | 576 | 6.9-9.6 | 10.6 |
| Aug | 15,405 | 11,635-23,306 | 0.65 | 385 | 4.6-6.4 | 8.8 |
| Sep | 12,287 | 4,881-19,694 | 0.78 | 307 | 3.7-5.1 | 6.5 |
| Oct | 9,733 | 6,416-13,930 | 0.29 | 243 | 2.9-4.1 | 4.7 |
| Nov | 11,642 | 5,440-21,368 | 0.23 | 291 | 3.5-4.9 | 2.6 |
| Dec | 4,008 | 1,850-6,165 | 0.12 | 100 | 1.2-1.7 | 1.5 |
| Annual mean | 23,970 | | | 599 | 7.1-10.0 | 6.75 |

**Anholt E (Kattegat), 0-25 m, 2019-2023, n = 2-5 visits per month** (g WW m-2 at C/WW 0.084-0.06):
Jan 2.8-4.0, Feb 5.9-8.3 (median-based 3.4-4.8; one visit at 47,297 ug m-3), Mar 8.3-11.6,
Apr 18.2-25.4, May 15.3-21.4, Jun 12.3-17.2, Jul 8.7-12.2, Aug 5.1-7.1, Sep 3.6-5.1, Oct 5.5-7.7,
Nov 3.4-4.8, Dec 1.7-2.4; annual mean 7.6-10.6. Calanus share of carbon 4-21 % except Sep (59 %).

Ratios (carbon, C/WW cancels):

| | Dec-Feb mean / annual mean | Dec-Feb mean / monthly peak | Dec / May |
|---|---|---|---|
| Å17 carbon | 0.145 | 0.040 | 0.046 |
| Anholt E carbon | 0.46 (median-based 0.39) | 0.19 (0.16) | 0.11 |
| Å17 copepod abundance, 2007-2023 (n = 5-14 per month) | 0.31 | 0.16 | - |
| Anholt E copepod abundance, 1996-2023 (n = 17-23 per month) | 0.40 | 0.25 | - |
| **Model, herring cells** | **0.31** | **0.165** | **0.12** |

Copepod abundance (ind m-3, 0-25 m, monthly means): Å17 Jan 2,487, Feb 7,372, Nov 4,703, Dec 5,058 vs
May-Jul 25,155-31,068. The abundance ratio is less steep than the carbon ratio because winter
individuals at Å17 are fewer but the summer peak carbon is dominated by large Calanus CV/adults
(Calanus share 0.76-0.88 of C in Apr-May).

Depth check (only depth-split haul in the data): Anholt E 1996-11-05, abundance 0-20 m 16,264 ind m-3
(x 20 m = 325,280 ind m-2) and 20-50 m 7,108 ind m-3 (x 30 m = 213,248 ind m-2). The 20-50 m layer adds
66 %, i.e. 0-50 m = 1.66 x 0-20 m in November. n = 1; carbon for this haul is reported only per
analysed fraction and cannot be converted.

Sanity check against the sibling note: Anholt E Jun-Aug mean of my carbon-derived values is
(12.3 + 8.7 + 5.1)/3 = 8.7 g WW m-2 at C/WW 0.084, inside the Casties 2018-2023 summer-total range
(~3-17 g WW m-2), consistent given that Casties' "total" likely includes non-copepod taxa.

Caveats: (1) 2019-2023 only (the carbon parameter is absent earlier), and the Casties trend puts these
in the low-biomass end of the 1998-2023 record; (2) WP2 90 um at 0-25 m misses krill and the deeper
part of the column; (3) Å17 is at the 345 m Norwegian Trench slope, Mareld is ~100 m.

### Inference
- **Whole-column scaling.** The model's zooplankton is a whole-column stock; SHARK is 0-25 m. In summer
  most copepods sit above the thermocline (~10-35 m), so whole column / 0-25 m is plausibly ~1.2-1.6.
  In winter the column is mixed or inverted (prior note: Feb surface 3.4 C, 50 m 7.2 C) and overwintering
  Pseudocalanus/Calanus stages sit deeper; the one in-data depth split gives 1.66 for 0-50 vs 0-20 m in a
  60 m Kattegat column. A factor of ~1.5-3 for a 100 m column in winter is my estimate (not measured).
  Applied to Å17: winter (Dec-Feb) whole column ~1.5-4.4 g WW m-2 (1.03-1.45 x 1.5-3); summer peak
  ~30-60 g WW m-2.
- **Diapause accessibility.** At a ~100 m site there is no diapause habitat for C. finmarchicus; the
  overwintering stock is in the Norwegian Trench (Falkenhaug: not in substantial numbers in the
  North Sea-Skagerrak itself). The Calanus carbon at Å17 in Jan-Feb (~70 % of a very small total) is
  mostly adult G0 returning from depth (Falkenhaug's interpretation of Jan-Feb adults). The winter food
  available to herring over the Mareld bank is therefore the active neritic fraction plus whatever krill
  are present (Q2).
- **Verdict Q1: realistic** in winter level. Model Dec-Jan 1.5 g WW m-2 sits between the 0-25 m Å17
  values (1.2-1.7) and my whole-column estimate (1.5-4.4); Feb 3.2 is inside the whole-column range.
  Model annual mean (6.75 at herring cells, 7-13 grid) matches the 0-25 m annual means (7.1-10.6). The
  model's winter/peak ratio (0.165) is *less* steep than Å17 carbon (0.04) and equal to Anholt E carbon
  and the abundance ratios (0.16-0.25). The mismatch is in summer: the observed Apr-May peak (19-36
  g WW m-2 at 0-25 m, Calanus-driven) is ~2-3x the model's (10.5-12.5), and the model peak is broad
  (May-Jul) where observations are sharp (Apr-May). If anything the model's winter food is slightly
  *generous* relative to the Å17 0-25 m data, not too low.

---

## Q2. Krill in winter: biomass, depth, role in herring winter diet

### Literature
- Onsrud et al. 2004, inner Oslofjord (59 48 N, 120 m, sill 19 m) [F]: "The krill layer was restricted
  to waters deeper than ca. 70 m during the day", 20-40 m thick, ascending at dusk; herring "foraging
  in the upper and middle part of the krill assemblage" by day, shallowest (0-50 m) at night. Table 3
  (herring > 25 cm): Nov 2000 day/night - fullness 2.91/2.29 (scale 1 = empty, 5 = full/distended),
  % with krill 31.0/17.2, krill per stomach (with krill) 1.90/2.20, % with copepods 51.7/25.0,
  % empty 0/0. Mar 2001 day/night - fullness 3.43/2.70, % with krill 50.0/20.0, krill per stomach
  5.13/1.50, % with copepods 12.5/20.0, % empty 6.25/0. Also Centropages and cladocerans in November.
  "Herring stomachs contained krill prey during both day and night." No krill density/biomass given
  ("Bottom-trawl krill catches ... insignificant").
- Last 1989, North Sea, 5762 stomachs, February 1983/84/86, May and August 1986 [A]: "The principal
  prey organisms were the copepods Calanus finmarchicus and Temora longicaudata [sic; probably
  T. longicornis, as transcribed by OpenAlex], but Euphausiacea and
  the post-larval stages of Ammodytes spp. and clupeoids contributed a large percentage of the weight."
  (Season-by-season split not in the abstract.)
- Segers et al. 2007, southern North Sea, February 2004, herring 82-288 mm [F, OCR]: "copepods
  dominated the diet numerically (59%) and were also found in the most stomachs (40%)"; "copepods,
  amphipods, and euphausiids appear to be taken most"; fish eggs in 19 % of stomachs, eaten mainly when
  crustaceans were scarce ("the fullest stomachs tended to contain fewer fish eggs").
- Lohmeyer & Hempel 1977, western Baltic (Eckernförde Bight, Jan-Feb 1976, 1-3 C, 571 fish; Oderbank
  mid-Nov, 273 fish) [F]: juveniles 12-16 cm ate copepods ("almost exclusively Temora longicornis") and
  plaice eggs; herring > 18 cm: gobiids (Pomatoschistus minutus) "more than three quarters of the
  total food mass"; Oderbank 91 % gobiids by weight. No krill (western Baltic).
- Mackinson & Daskalov 2007, North Sea Ecopath, section 10.2 [F]: euphausiid mean biomass 0.78 g WW m-2
  (Thysanoessa, from CPR at 10 m, Lindley 1980, DW/WW 0.225), + 50 % for M. norvegica = 1.17 g WW m-2;
  maximum estimate 2.4 g WW m-2; the model needs 3.4 g WW m-2 carnivorous zooplankton (P/B 4, Q/B 12.5)
  "to meet the consumption by higher trophic levels"; euphausiids "are often undersampled because they
  escape plankton nets"; on Fladen in March-April euphausiids "represented more than 90% of the standing
  stock of zooplankton biomass" (Williams & Lindley 1980a, [2nd]).
- Båmstedt & Karlson 1998 [F]: in the Skagerrak T. inermis "seldom fed on copepods"; M. norvegica eats
  mainly C. finmarchicus (85-95 % of copepod prey mass); small M. norvegica < 50 mg DW.
- SHARK [D]: euphausiids appear in only 0-60 % of WP2 0-25 m hauls at Å17, mostly larvae, 0.1-34
  ind m-3 - the net is not quantitative for krill.

### Calculation
- North Sea euphausiid / copepod biomass ratio (Ecopath): 0.78-2.4 / 16 = 0.05-0.15 (annual);
  required carnivorous zooplankton 3.4 / 16 = 0.21.
- Ration from Onsrud stomachs (order of magnitude). Herring > 25 cm ~ 150-250 g (my assumption).
  Adult M. norvegica 30-80 mg DW (Båmstedt: "small" < 50 mg) / 0.225 DW/WW = 0.13-0.36 g WW per krill.
  Krill-bearing stomachs: 1.9-5.1 krill x 0.13-0.36 g = 0.25-1.8 g = 0.1-0.9 % of a 200 g herring.
  Times the fraction with krill (0.17-0.50) -> population-mean krill content ~0.02-0.45 % BW. With one
  to two stomach turnovers per day at 7-8 C (assumed; herring evacuate most contents within 24 h even
  in summer per the IMR snippet [S]) -> krill ration ~0.02-0.9 % BW d-1, plus copepods.

### Inference
- Krill are 1-2-year animals with no 10-20-fold winter crash, so in winter (when mesozooplankton is at
  4-15 % of its peak at Å17) a krill stock of ~1-2 g WW m-2 could equal or exceed the mesozooplankton
  stock. This is plausible but **not measured** for the Skagerrak/Kattegat: no winter krill biomass for
  the region was found (Boysen & Buchholz 1984 and Buchholz & Boysen-Ennen 1988 could not be read).
- At Mareld (~100 m) the daytime krill layer (> 70 m in the Oslofjord) is only marginally present;
  krill would be available mainly at night when they ascend, which runs against the model's 10 % night
  attack (Q3).
- Herring winter diet in the region is **mixed**, not "mostly krill": Oslofjord November has more
  stomachs with copepods (52 % by day) than with krill (31 %); March is krill-dominated by number per
  stomach; North Sea February is copepods by number with euphausiids/sandeel post-larvae important by
  weight; western Baltic winter is gobiids and Temora. Since the model's `zooplankton` already stands
  in for krill for gadoids, it is defensible as the herring winter food too; the omission is a possible
  extra winter stock of order 1 g WW m-2 that the model's seasonal zooplankton cycle does not carry.
- **Verdict Q2: uncertain / probably slightly low** - the model has no separate low-seasonality krill
  pool, so its winter food may be understated by up to ~1-2 g WW m-2 (Ecopath North Sea range), i.e. up
  to ~1x the model's Dec-Jan zooplankton. The data needed to close this (Kattegat krill density,
  Buchholz) were not accessible.

---

## Q3. Winter feeding: ration (% BW d-1) and day vs night

### Literature
- Lohmeyer & Hempel 1977 [F], western Baltic: Eckernförde Bight Jan-Feb (1-3 C) 42 % of herring had
  food; Oderbank mid-Nov ~63-75 % (< 28 cm). "The average daily food intake - the calculation includes
  those with empty stomachs - ranged from 0.35% to 0.70% of the dry body weight" (Eckernförde); Oderbank
  "0.72% to 1.31% of the dry body weight"; fish dry weight "22% of fresh weight". These are below the
  "annual averages ... around 1.5% of the body weight for the size group 10 - 25 cm" (Zalachowski et al.
  1976, [2nd]). Conclusion: "Baltic herring is more actively feeding in late autumn and in winter than
  previously expected. The food consumption might even be sufficient to cover the basic and locomotory
  metabolism at the given low temperature, without drawing on the fat reserves, which might mainly be
  used for gonad development." Feeding peaks "around noon and sun set" (daylight sampling only).
  Introduction: "In winter, feeding activity of herring is very limited" (Hardy 1924 et al., [2nd]);
  Blaxter & Holliday 1958 no feeding below 4 C in captivity, others observed feeding at <= 1 C ([2nd]).
- Bernreuther 2007, ms 7, Bornholm Basin (central Baltic), gastric evacuation model, Apr 2002 - Nov 2003
  [F]: herring daily ration peaks June (2.9 % BW, 20/21 cm; < 1.4 % for 24/25 cm); "The daily rations
  decreased towards autumn and winter with rations at or below 0.5% BW. In July and November 2003 higher
  daily rations of 1.0 to 2.2% BW were observed." Maintenance ration incl. activity (MRA) "highest levels
  of 1.0 to 1.4% BW in September 2002, decreasing afterwards to lowest levels of ~ 0.2% BW in March and
  April 2003"; "The daily rations during winter were more or less similar to maintenance rations MRA";
  rations exceeded maintenance from April to July/August. Ms 1: herring stomach fullness 0.7 % BW in
  July 2002, 0.1-0.2 % BW in March 2003, 0.4-0.6 % BW in November 2003. Sprat: ~0.1 % BW d-1 in
  January 2003. Assumed prey 20 kJ g-1 DW, DW/WW 0.27 (5.4 kJ g-1 WW).
- Onsrud et al. 2004 [F]: herring stomach fullness in November and March 2.3-3.4 on a 1-5 scale with
  0-6 % empty; "neither stomach fullness nor state of digestion revealed clear diel patterns in food
  uptake"; krill eaten day and night.
- Mackinson & Daskalov 2007 [F]: North Sea adult herring Q/B 4.34 yr-1, juveniles 5.63 yr-1
  (Greenstreet 1996, [2nd]).
- Bachiller et al. 2018 [F] (Norwegian spring spawners, a different, fasting stock): feeding Apr 1 -
  Sep 15; "Very little feeding was done in August and September"; "during winter the feeding nearly
  stops".

### Calculation
- Lohmeyer, dry -> energy: 0.35-0.70 % of fish DW = 0.0035-0.0070 x 220 g DW per kg fish = 0.77-1.54
  g prey DW kg-1 d-1; x 20 kJ g-1 DW = 15-31 kJ kg-1 d-1 gross; x 0.8 (model AE) = 12-25 kJ kg-1 d-1
  assimilated. Oderbank (Nov): 32-58 gross, 25-46 assimilated.
- Lohmeyer, dry -> wet ration: wet % = dry % x (fish DW/WW 0.22) / (prey DW/WW). With prey DW/WW 0.20
  (fish, crustaceans) -> 0.39-0.77 % BW wet; with 0.15 -> 0.51-1.03 %. Jan-Feb wet ration ~0.4-1.0 % BW.
  (The two DW/WW ratios nearly cancel; the range is set by the prey DW/WW assumption.)
- Bernreuther in energy: MRA 0.2 % BW x 5.4 kJ g-1 = 10.8 kJ kg-1 d-1 gross (Mar-Apr); winter ration
  <= 0.5 % -> <= 27 kJ kg-1 d-1 gross; Sep MRA 1.0-1.4 % -> 54-76.
- Model, assimilated energy = ration x 4.5 kJ g-1 x 0.8: Nov 0.5 % -> 18.0; Dec 0.18 % -> 6.5;
  Jan 0.30 % -> 10.8; Feb 1.4 % -> 50.4; Mar 3.5 % -> 126 kJ kg-1 d-1. Break-even ration against the
  model's winter cost 23-27 kJ kg-1 d-1: 23/3.6 to 27/3.6 = 0.64-0.75 % BW d-1.
- Annual mean: North Sea Ecopath adult Q/B 4.34 yr-1 = 1.19 % BW d-1 (juveniles 1.54). Model annual
  mean ration ~1.8 % BW d-1 (Q/B ~6.5 yr-1), assuming Aug = 2.2 % (not supplied).
- Day vs night (Onsrud; ordinal scale, 1 = empty): night/day fullness 2.29/2.91 = 0.79 (Nov) and
  2.70/3.43 = 0.79 (Mar); above-empty part (2.29-1)/(2.91-1) = 0.68 and (2.70-1)/(3.43-1) = 0.70.
- Effect of the model's night factor: the prior report gives the realised daylight factor on herring's
  attack rate on zooplankton as 1.4 (summer) -> 0.66 (February), after the library's annual-mean
  normalisation and -8 deg threshold (model probe, prior report). I do not recompute it here.
- Model-internal check: model March ration 3.5 % BW d-1 at 6.7 g WW m-2 zooplankton exceeds the
  Apr-Jul ration of 2.3-2.9 % at 10.5-12.5 g WW m-2 - intake rises while food is ~half.

### Inference
- **By month:** model Dec 0.18 % and Jan 0.30 % are at or below the low end of the literature winter
  rations (Lohmeyer ~0.4-1.0 % wet Jan-Feb; Bernreuther <= 0.5 %, about maintenance) - roughly 1.5-3x
  too low in Dec-Jan. Model Nov 0.5 % is at the low end of Bernreuther/Lohmeyer November (0.5-2.2 %
  wet). Model Feb 1.4 % and especially Mar 3.5 % are **above** every winter/early-spring value found
  (Bernreuther March fullness 0.1-0.2 % BW; maintenance ~0.2 %), and Mar 3.5 % exceeds the June peak in
  the Bornholm Basin (2.9 %). Caveat: the Bornholm Basin is described by its author as a poor feeding
  area, and WBSS spawn in Mar-Apr, when feeding is reduced. The stronger argument is model-internal:
  the March ration is higher than the model's own summer ration at twice the food, so it is driven by
  appetite (lean fish after the January minimum, hunger gate open), not by food availability.
- Lohmeyer's Jan-Feb fish were western Baltic, 1-3 C, 20 m bight, mostly juvenile or maturing
  (stage III-IV), bottom-trawled in daylight; they are the closest winter ration found, not a
  Skagerrak adult measurement.
- **Energetics:** the literature says winter intake roughly equals winter maintenance (Lohmeyer,
  Bernreuther). The model's Dec-Jan assimilated intake (6.5-10.8 kJ kg-1 d-1) is about half of
  Lohmeyer's (12-25) while its cost (23-27) is at the upper end of the literature range (Bernreuther
  gross maintenance ~11 at Mar-Apr temperatures; Rudstam total at 5 C ~32, prior note). So the winter
  deficit comes from both sides: intake a bit low, cost on the high side.
- **Day/night:** the only regional data (Onsrud) show night stomach contents ~70-80 % of day contents
  with the same digestion state and herring inside the krill layer at night; herring can also
  filter-feed in darkness (Batty et al. 1986, cited by Onsrud and Segers, [2nd]). The model's night
  attack of 10 % of day is a much stronger asymmetry than these data suggest. Caveat: stomach fullness
  integrates several hours, so night fullness is not the night feeding rate; Lohmeyer (daylight-only
  sampling) found feeding peaks at noon and sunset. A night ratio of ~0.3-0.5 is easier to defend for
  winter than 0.1, but this is my inference, not a measured value.
- **Verdict Q3: too low in Dec-Jan (by ~1.5-3x), too high in Feb-Mar**; the 10 % night attack is
  stronger than the data support and is a plausible contributor to the Dec-Jan shortfall (long December nights; day length ~6.5 h at 58 N, more usable light with the -8 deg threshold).

---

## Q4. Winter condition / weight loss

### Literature (from the prior note; not redone)
- Kenyon et al. 2022 (NSAS, muscle fat): 9.1 % in September falling to 5.3 % at end of November during
  spawning; no Jan-Apr data; spawning decline 0.11 %/day vs 0.03 %/day for overwintering NSS herring
  [2nd, prior note].
- JNCC 633 / Pedersen & Hislop 2001: adult North Sea herring 6.5 kJ g-1 in Q1 (20-25 cm), 8.5 kJ g-1 for
  25-40 cm (Wanless et al. 2005), 10.1-11.9 kJ g-1 in Q3 [2nd, prior note].
- Rojbek et al. 2014 (central Baltic SD 25, whole body): ~10 % in November, 3-5 % in maturing spring
  spawners in March; low point after spawning [2nd, prior note].
- Lohmeyer & Hempel 1977 [F]: winter intake "might even be sufficient to cover the basic and locomotory
  metabolism ... without drawing on the fat reserves, which might mainly be used for gonad development".
- Bernreuther 2007 [F]: winter rations about equal to maintenance in the Bornholm Basin.

### Calculation
- Model fat-% = fill x 7000 / 396 = fill x 17.7: Jan 0.26-0.29 -> 4.6-5.1 %; Sep 0.71-0.75 ->
  12.6-13.3 %; May-Jul 0.79-0.86 -> 14.0-15.2 %.
- Model Nov (~9 %) -> Jan (~5 %): 4 fat-points x 396 MJ/t = 1,584 MJ/t over ~61 d = 26 kJ kg-1 d-1 net
  loss = 0.066 fat-%/day. This equals the model's winter cost (23-27 kJ kg-1 d-1) with intake near zero,
  so the loss is cost-driven.
- Literature rates: NSS overwintering 0.03 %/d; Baltic spring spawners Nov -> Mar (10 % -> ~4 %, over
  ~120 d, including gonad build-up) 0.05 %/d; NSAS during spawning 0.11 %/d. Tissue mismatch: Kenyon's
  0.03 and 0.11 %/d are muscle-fat points, Rojbek's is whole-body lipid, and the model's 0.066 %/d is a
  whole-body reserve equivalent - same order of magnitude, not the same quantity.
- Q1 target: 6.5 kJ g-1 ~ (6.5 - 4.0)/0.396 = ~6 % fat (prior note's conversion), fill ~0.36; the prior
  report's Jan-Feb validation band is fill 0.28-0.45 (5-8 % fat).

### Inference
- The model's January level (4.6-5.1 % fat, fill 0.26-0.29) sits at, or just below, the lower edge of
  the literature Q1 band (5-8 %, fill 0.28-0.45). The rate of loss (0.066 %/d) is ~1.3x the Baltic
  spring-spawner Nov-Mar rate (which includes gonad growth) and ~2x the fasting NSS rate - on the high
  side for a stock that feeds in winter. The timing is off: the model's minimum is in January, the
  literature minimum is after spawning (Mar-May), and the model refills from February (Feb ration 1.4 %,
  Mar 3.5 %) when WBSS are still maturing and spawning.
- **Verdict Q4: slightly too much loss, too early** (level borderline-realistic, rate ~1.3-2x, minimum
  ~2-3 months early). Consistent with Q3: a winter intake nearer 0.4-0.7 % BW d-1 in Dec-Jan would
  roughly balance the 23-27 kJ kg-1 d-1 cost and hold fat near 6-9 %.

---

## Q5. Timing of annual production

### Literature
- Bernreuther 2007 [F] (Bornholm Basin): ration > maintenance from April to July/August 2002; from
  August until the onset of winter "herring fed at rates that were less than or equal to maintenance
  levels".
- Kenyon et al. 2022 (NSAS) [2nd, prior note]: muscle fat rises from 4.5 % to 16.1 % during May-June,
  plateaus, and falls from September with spawning.
- Bachiller et al. 2018 [F] (NSS herring, other stock): feeding window April 1 - September 15, little
  feeding in August-September, near-stop in winter.
- Arrhenius & Hansson 1994 [A] (Baltic YOY, 25-84 mm): consumption "ranging from 17 % of body weight
  d-1 in summer to 2 % d-1 at the end of October". Arrhenius & Hansson 1999 [A]: "from about 8 to 15% of
  body weight for small fish in the summer to 0.6 to 2% for larger juveniles in late autumn"; growth of
  YOY herring is food-limited.
- Lohmeyer & Hempel 1977 [F]: winter intake possibly enough to cover maintenance; fat goes to gonads.
- No source retrieved gives a month-by-month fraction of annual somatic + gonad production for WBSS or
  NSAS adults.

### Calculation
- None possible beyond the above; the literature gives a window, not monthly fractions.

### Inference
- For adults, near-zero net somatic growth from October to February is supported: every source puts net
  energy gain in April-August (Bernreuther, Kenyon, Bachiller) and winter intake at about maintenance
  (Lohmeyer, Bernreuther). For spring spawners, gonad growth from autumn to spawning (Jan-Mar) is a
  transfer of reserve energy, not a loss of production, and the model has no spawning. The model's
  production in April-August and near-zero in October-February is therefore **realistic** as a pattern;
  what is less realistic is the strong February-March refill (Q3/Q4).
- Juveniles keep feeding later into autumn (2 % BW d-1 at end of October, 0.6-2 % in late autumn) and
  WBSS juveniles in the Kattegat are not spawning, so their growth season is plausibly longer (into
  October) than the adults'. The pooled `pelagic_fish` FG cannot represent this; with ~2/3 small
  clupeids/juveniles in some pools it would argue for slightly more autumn production than adults alone.
- **Verdict Q5: realistic** for adults (Oct-Feb near-zero), somewhat too abrupt for juveniles.

---

## Summary table

| Item | Literature / data | Model | Verdict |
|---|---|---|---|
| Winter (Dec-Feb) mesozooplankton | Å17 0-25 m 0.7-1.7 g WW m-2; whole column est. 1.5-4.4; Anholt E 0-25 m 1.7-8.3 | 1.5 / 1.5 / 3.2 | Realistic |
| Winter / peak ratio | Å17 C 0.04; Anholt E C 0.16-0.19; abundance 0.16-0.25 | 0.165 | Realistic (Skagerrak steeper) |
| Summer peak | Å17 Apr-May 19-36 (0-25 m) | 10.5-12.5 | Model ~2-3x low, peak too broad |
| Annual mean | 7.1-10.6 (0-25 m) | 6.75 (herring cells), 7-13 grid | Realistic |
| Krill winter stock | NS Ecopath 0.78-2.4 g WW m-2 (annual); region not measured | not separate | Possibly missing ~1-2 g WW m-2 |
| Winter ration Dec-Jan | 0.4-1.0 % BW wet (Lohmeyer: western Baltic, 1-3 C, mainly < 18 cm), <= 0.5 % ~ maintenance (Bernreuther, Bornholm Basin) | 0.18-0.30 % | Too low (1.5-3x) |
| Ration Mar vs summer | - | 3.5 % at 6.7 g WW m-2 > 2.3-2.9 % at 10.5-12.5 | Appetite-driven artefact |
| Ration Feb-Mar | ~0.2-0.5 % (Bernreuther, March fullness 0.1-0.2 % BW) | 1.4 / 3.5 % | Too high |
| Night feeding | Night fullness ~70-80 % of day; krill eaten day and night (Onsrud) | 10 % of day | Asymmetry too strong |
| Fat loss Nov-Jan | 0.03-0.05 %/d (fasting NSS; feeding Baltic spring spawners incl. gonads) | 0.066 %/d | Somewhat high, minimum too early |
| Jan fat | Q1 ~5-8 % (6.5 kJ g-1 ~ 6 %) | 4.6-5.1 % | Borderline low |
| Production timing | Net gain Apr-Aug; winter ~maintenance | Apr-Aug production, ~0 Oct-Feb | Realistic for adults |

## Open gaps
- No winter krill density/biomass for Skagerrak/Kattegat (Boysen & Buchholz 1984 is the obvious source).
- No winter zooplankton data below 25 m at Å17; whole-column scaling is estimated from one Anholt E haul.
- No WBSS-specific winter stomach data from the Sound/Kattegat (Nielsen et al. 2001 not read).
- No monthly production fractions for WBSS/NSAS adults; no quarterly HAWG weight-at-age tables pulled.
- Winter maintenance cost for adult Atlantic herring at 4-7 C remains unmeasured (prior note); the
  literature range used here (gross ~11-32 kJ kg-1 d-1) spans a factor of 3.
