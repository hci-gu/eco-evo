# Appetite or acquisition in the krill layer: the appetite-side literature trail (2026-11-20)

Research subagent; literature and data only. No model run, no value chosen from model output, nothing outside this
folder changed.

**Question served.** Whiting and Norway pout stood in the Oslofjord krill layer in Nov and Mar with half-full
stomachs (Onsrud et al. 2004). (A) APPETITE: the fish are not hungry (seasonal, lipostatic or light-cued
depression). (B) ACQUISITION: they are hungry but get little per unit time. This file collects the evidence that
separates them on the appetite side, especially outside summer.

**Rule applied to every entry.** A datum bears on (A) only where food was shown to be in excess (or at least
present). For each datum I state what it cannot exclude:
- temperature (the model already has an ingestion Q10, so a winter drop at ambient temperature is not evidence
  for A);
- the fish's own spawning (a known, short, maturity-bound fast);
- a residual seasonal or lipostatic depression.

**Tags.** [F] full text read (PDF -> pdftotext -> `scripts/app_read.py`); [F-w] full text read through the
WebFetch tool on the publisher's HTML (the quotes are the tool's extraction, short fragments, not checked against
a PDF); [A] abstract only; [S] secondary (a review, a citing paper or an earlier note in research_notes/);
[D] my arithmetic (`scripts/app_derive.py` -> `tables/app_derive_stdout.txt`); [J] judgement.
[F-w] is not in the caller's tag set: weigh it as [S] (the numbers are a tool's extraction); in the CSV those rows
carry tag S with "full text via WebFetch" in `method`.
**Proxy flag.** Whiting is the target species; Norway pout second. Saithe, cod, haddock and herring entries are
PROXIES and flagged as such.

Earlier notes reused (not redone): "Mättnad hos rovfisk" (MR), "Krillparen mot vintermagar" (KV), "Intag under
födosök" (IF), "Ljusstyrd aptit" (LA), "Lek i rutan/lit_cod_haddock.md" (LR), "Tillväxt mot ranson" (TR),
"Höstens tillväxt och sommarens aptit" (HT).

---------------------------------------------------------------------------------------------------------------

## 1. Easy food outside summer

### 1.1 Uglem et al. 2020 - saithe at Hitra and Ryfylke salmon farms, summer / autumn / spring [F] PROXY (saithe)

Uglem I, Toledo-Guedes K, Sanchez-Jerez P, Ulvan EM, Evensen TH, Saether BS (2020) Does waste feed from salmon
farming affect the quality of saithe (*Pollachius virens* L.) attracted to fish farms? Aquaculture Research
51:1720-1730. doi:10.1111/are.14519. Read: the accepted manuscript already downloaded by MR
(`../Mättnad hos rovfisk/data/uglem2020/a.txt`), Methods 2.1-2.3 and Table 1 in full. MR read only the Hitra UA
rows; this entry adds the FA rows, the pellet columns and both areas.

**Design, quoted.**
- "at six different occasions during summer, autumn and spring in 2013-2015 (Figure 1, Table 1)". "FA fish were
  captured less than 500 m from farms, containing salmon above 1 kg ... sampling of saithe was for practical
  reasons carried out at different farms ... that contained larger salmon. This is the part of the production
  cycle when the feeding rate, and therefore the amount of waste feed available to the wild fish, is at its
  highest. UA fish were sampled from locations 5-10 km distant from the nearest farm".
- Gear: "bottom gillnets which were soaked between 15 and 17 hr, and automatic jigging machines that hauled the
  fish on board in 2-3 min".
- HSI: "HIS = (LW/W) x 100" with W the whole weight before gutting. "Stomach contents from the foregut were
  examined and the prey identified into five broad categories (salmon feed pellets, fish, crustaceans, bivalves
  and other organic matter) and weighed".
- Months: NOT GIVEN. The only dated samples: "fish collected at Hitra during spring 2015 and autumn 2014"
  (texture subset). A timeboxed search of the companion paper on the same Hitra gear comparison (Toledo-Guedes,
  Ulvan & Uglem 2016, Aquat Living Resour 29:203, doi:10.1051/alr/2016013) did not reach its text (publisher 403,
  repository host unreachable). **Month unknown; there is no Dec-Feb occasion.**
- Maturity: "Saithe should normally spawn during late winter or early spring ... Out of 181 fish caught during
  spring, only 15 (8.2%) had clearly mature or maturing gonads with a GSI above 2".
- Ryfylke UA: "between 7% and 19% of pellets, prevalence in their diet was found, while UA fish from Hitra had no
  pellets in their stomachs ... Thus, UA fish from Ryfylke were excluded".
- Summary sentence: "The average prevalence and mass of pellets in stomachs from FA saithe from Hitra was 44% and
  105 g (+-104 g)".

**Table 1, Hitra (means +- SD, quoted).** Columns: n, TL cm, W kg, K, HSI, GSI, pellet prevalence %, pellet
weight g, pellets in diet %.

| Season | Gear | UA | FA |
|---|---|---|---|
| Autumn | jig | 30, 68, 3.05, 0.95, 6.9 +- 3.1, 1.82, 0, -, - | 31, 60, 2.14, 0.96, 5.6 +- 2, 0.93, 29, 56 +- 77, 100.0 |
| Autumn | net | 30, 68, 2.96, 0.94, 6.8 +- 1.4, 1.79, 0, -, - | 30, 66, 3.1, 1.05, 8.2 +- 2.4, 1.27, 33, 141 +- 163, 88.8 |
| Spring | jig | - | 30, 69, 3.09, 0.93, 6.1 +- 3.5, 0.63, 60, 166 +- 106, 93.6 |
| Spring | net | 31, 66, 2.44, 0.84, 6.2 +- 4.4, 0.63, 0, -, - | 30, 68, 3.19, 1.04, 10 +- 4.8, 1.12, 50, 89 +- 63, 97.1 |
| Summer | jig | 20, 62, 2.09, 0.86, 6 +- 2.2, 0.25, 0, -, - | 25, 68, 3.18, 0.98, 10 +- 3.2, 0.36, 88, 78 +- 87, 92.0 |
| Summer | net | 20, 69, 2.99, 0.9, 6.8 +- 2.7, 0.43, 0, -, - | 48, 66, 2.92, 1.01, 10.7 +- 3.8, 0.24, 23, 96 +- 112, 95.3 |

**Table 1, Ryfylke (FA; UA farm-influenced).** Autumn jig FA 25, 45 cm, 0.9 kg, HSI 11.6 +- 2, prevalence 72,
5 +- 3 g (excluded by the authors as below commercial size). Autumn net UA 36, 59 cm, HSI 8.9 +- 2.8, GSI 3.11,
prevalence 19, 94 +- 127 g; FA 25, 57 cm, HSI 8 +- 3.8, GSI 3.44, prevalence 40, 38 +- 75 g. Spring jig FA 30,
57 cm, HSI 7.2 +- 3.4, prevalence 70, 54 +- 41 g. Spring net UA 30, 62 cm, HSI 12 +- 5.1, GSI 3.52, prevalence
7, 79 +- 67 g; FA 30, 57 cm, HSI 9.1 +- 3.4, prevalence 70, 156 +- 73 g. Summer jig FA 30, 58 cm, HSI 9.5 +- 2.5,
prevalence 90, 32 +- 38 g. Summer net FA 23, 65 cm, HSI 7.8 +- 3.6, prevalence 26, 168 +- 208 g.

**What the "pellet weight" column means [D].** The mean of the six Hitra FA values (56, 141, 166, 89, 78, 96) is
104 g, the text's "105 g". For a mean over all fish (zeros included) at prevalence p, the SD cannot be below
sqrt(p m_pos^2 - (p m_pos)^2) with m_pos = mean / p. Autumn jig (p 0.29, 56 g) needs SD >= 88, autumn net
(0.33, 141 g) >= 201, spring jig (0.60, 166 g) >= 136, spring net (0.50, 89 g) >= 89, summer net (0.23,
96 g) >= 176; the reported SDs are 77, 163, 106, 63, 112. So the column is the mean
pellet mass **in fish that had pellets**, not over all fish (`app_derive.py`).

**Derived per season, Hitra FA [D] (jig only for meal size; nets soaked 15-17 h, so net stomachs are partly
digested and not comparable).**
- Pellet mass in a pellet-eating jigged fish as % of mean FA body weight: autumn 2.6 %, spring 5.4 %, summer 2.5 %.
- Pellet prevalence (jig): autumn 29 %, spring 60 %, summer 88 %; (net): 33 / 50 / 23 %.
- HSI FA/UA within gear: autumn jig 0.81 (FA fish 8 cm shorter), autumn net 1.21, spring net 1.61, summer jig
  1.67, summer net 1.57.

**Reading.**
- [J] Saithe at Hitra take pellets in autumn and spring as well as summer, and the largest jig-caught pellet meals
  are in spring (5.4 % BW of pellets in the foregut at capture). A fish with an appetite depression does not
  hold a 5 % BW meal; spring saithe offered easy food eat it. For saithe this is direct evidence against an
  appetite shut-off in spring. Pellets are calorie-dense, so such a meal is large in energy too, but stomach
  pellets are hydrated and no energy value is given in the paper, so no ration equivalent is derived.
- [J] The HSI surplus (FA/UA 1.2 in autumn net, ~1.6 in spring and summer) shows the extra food is retained as
  liver in spring as in summer. The autumn surplus is smaller (0.8-1.2); prevalence is lower (29-33 %) and the
  FA jig fish were smaller. Waste-feed supply at a given farm depends on its production stage, which the authors
  held at "larger salmon", so the autumn dip cannot be read as appetite. HSI integrates the preceding weeks to
  months; a spring HSI of 10 % in FA fish implies they kept eating pellets through the preceding winter months
  or refilled fast in early spring - it cannot separate those [J].
- Cannot exclude: temperature (ambient; not given), month (unknown; "spring" may be April-May, after the
  saithe spawning window), farm-to-farm supply. It does exclude the fish's own spawning as a driver for the
  immature majority.
- **Bears on (A) vs (B):** favours (B) for saithe in spring, weakly in autumn: with easy food, intake and liver
  are above the unsubsidised fish's level, so the unsubsidised level is not an appetite set point. The autumn
  month is unknown, so November specifically is not shown; winter proper (Dec-Feb) is not covered. PROXY for
  whiting (saithe 57-69 cm).

### 1.2 Otterå & Skilbrei 2014 - saithe resident at farms for up to two years, closer in winter [F-w] PROXY

Otterå H, Skilbrei OT (2014) Possible influence of salmon farming on long-term resident behaviour of wild saithe
(*Pollachius virens* L.). ICES Journal of Marine Science 71:2484-2493. doi:10.1093/icesjms/fsu096. Read: the
publisher's HTML full text through WebFetch (first 100 000 of 115 000 characters); the PDF was refused (403).

- Fish and dates (Table 1): experiment 1 caught 21 April, released 7 May 2010, "42.5 cm" mean (38-49 cm);
  experiment 2 caught 10 November, released 26 November 2010, "51.0 cm" mean (42-59 cm). "followed their
  movements for up to 2 years in an area in Southwestern Norway".
- Season: "there was a suggestion of a weak seasonal trend ... with closer connections to the farms during
  winter (November 2010-January 2011 and November 2011-January 2012)". "During winter, in average, 82% of the
  saithe observed in the area had some degree of connection to the farms"; "The corresponding number during the
  summer was 64%".
- Food as the attractor: "saithe were less often recorded after the salmon at that site had been slaughtered and
  feeding had ceased" (site F13).
- Natural easy food in late winter: tag-return reports said "stomachs of the saithe were full of herring roe".
- Temperature: "Surface water temperature usually ranges from 3°C in winter to 17°C in summer".
- No stomach, HSI or ration data of their own; the "stomach contents were twice as large in saithe from farm
  locations" sentence cites Dempster et al. 2011 (summer) [S].

**Reading [J].** Box-sized saithe (38-59 cm) associate with farms more in Nov-Jan (0.82) than in summer (0.64), and
leave a farm when its feeding stops. A fish without winter appetite has no reason to hold to a feeding station
in winter. This is behavioural, indirect evidence of winter feeding motivation; it measures association, not
intake. Cannot exclude shelter as a co-attractor (the authors and MR list it). PROXY (saithe).

### 1.3 Skilbrei & Otterå 2016 - vertical distribution of farm saithe by season [F-w] PROXY

Skilbrei OT, Otterå H (2016) Vertical distribution of saithe (*Pollachius virens*) aggregating around fish farms.
ICES Journal of Marine Science 73:1186-1195. doi:10.1093/icesjms/fsv261. Read: HTML full text through WebFetch
(first 100 000 of 122 000 characters). MR had the abstract only.

- "Typical residence depth was from 25 to 50 m in summer and 60 to 90 m during winter"; "fish moved 10-20 m
  closer to the surface during mid-winter" (December), which "may be associated with the use of artificial
  light".
- "saithe ascend towards the net-pens to feed on waste feed during the day"; "Cultured salmon are fed during the
  day"; the cages are "30-40 m deep".
- Same fish as 1.2 (two 2010 releases), analysis to 31 January 2012.

**Reading [J].** In winter the fish sit 20-60 m below the cage bottoms, which lowers their access to falling
pellets; a lower winter pellet intake at farms (if found) would therefore not be appetite evidence. The midwinter
rise towards the lit cages is compatible with feeding, but the paper gives no stomach data. PROXY.

### 1.4 Dempster et al. 2009 and Uglem et al. 2009, 2014 - what the farm literature does not cover [F]

- Dempster T et al. (2009) Coastal salmon farms attract large and persistent aggregations of wild fish: an
  ecosystem effect. MEPS 385:1-14. doi:10.3354/meps08050 [F]: "Our estimates are applicable only for summer;
  substantial seasonal variability in aggregations has been documented around Mediterranean fish farms ... and
  similar seasonal variability in assemblages may be expected around cold temperate salmon farms."
- Uglem I, Dempster T, Bjorn P-A, Sanchez-Jerez P (2009; authors as cited in Uglem 2020) High connectivity of salmon farms revealed by
  aggregation, residence and repeated movements of wild fish among farms. MEPS 384:251-260.
  doi:10.3354/meps08001 [F]: saithe tagged in July ("tagged on July 10 ... released on July 13"), "observed daily
  at one or more of the 3 farms in the fjord over a 3 mo period"; "When resident at a farm, saithe spent 8 to
  10 h d-1 close to the sea-cages". Summer-autumn only.
- Uglem I, Karlsen O, Sanchez-Jerez P, Saether B-S (2014) Impacts of wild fishes attracted to open-cage
  salmonid farms in Norway. Aquacult Environ Interact 6:91-103. doi:10.3354/aei00112 [F]: no season-resolved
  feeding data; it lists the need for data on waste feed eaten "in relation to factors like season". It notes
  captive saithe "fed fish pellets in excess for 8 mo" (Otterå et al. 2009, below, not obtained) and "no major
  change in the hepatosomatic index between groups (Bjorn et al. 2007, Saether et al. 2012)" in two Norwegian
  reports, i.e. not every farm sample shows a liver surplus [S].
- Fernandez-Jover D et al. (2011) Waste feed from coastal fish farms: a trophic subsidy with compositional
  side-effects for wild gadoids. Estuar Coast Shelf Sci 91:559-568. doi:10.1016/j.ecss.2010.12.009: NOT
  OBTAINED (publisher 403; the figshare record 25000757 is metadata only; CORE refused). Search snippet only [S]:
  "6-96% of their diet was composed of food pellets"; higher body and liver condition in FA cod, higher body
  condition in FA saithe at one location. Season not seen.
- Otterå H, Karlsen O, Slinde E, Olsen RE (2009) Quality of wild-captured saithe fed formulated diets for 8 months.
  Aquaculture Research 40:1310-1319. doi:10.1111/j.1365-2109.2009.02230.x: NOT OBTAINED (Wiley 403). Could hold a
  captive saithe intake series across seasons; flagged as the best unread saithe source for item 2.
- Skog TE et al. (2003) Aquaculture Research 34:999-1007, doi:10.1046/j.1365-2109.2003.00901.x: not OA
  (Unpaywall false); not read.

### 1.5 Høines, Bergstad & Albert 1992 - cod and haddock gorge on herring eggs and herring in March-April [F] PROXY (haddock, cod)

Høines Å, Bergstad OA, Albert OT (1992) Seasonal variation of the diet of cod (*Gadus morhua* L.) and haddock
(*Melanogrammus aeglefinus* L.) at a herring spawning ground. ICES C.M. 1992/G:9 (Demersal Fish Committee), 23 pp.
No doi (ICES CM paper). Read: full text, VLIZ copy (`data/vliz_274776/`, pdftotext), and Figures 7-8 viewed as
rendered page images (values read off the plots, +-0.5 g for Fig. 7, +-2 g for Fig. 8).

- Site and timing: "a historically important spawning area of the Norwegian Spring Spawning Herring off the
  island of Karmøy in southwestern Norway"; "The herring spawns over a period of 5 - 6 weeks in March and April
  at this location". "In 1991 and 1992 the first major herring concentrations appeared in the last week of
  February ... by the third week of March most had retracted from the main grounds. Eggs were still found in
  haddock stomachs two weeks into April."
- Sampling (Table 1): Danish seine 22-25 Oct 1990, 13-18 Dec 1990, 11-14 Feb 1991, 22-25 Apr 1991, 11-13 Jun
  1991, 13-17 Oct 1991; research trawl 4-26 Mar 1991 and 6-11 Jul 1991. 1068 haddock and 821 cod stomachs.
- Content weight: "Mean weight per stomach was calculated based on all stomachs, empty ones included". "Above 30 cm
  (TL), all length groups in March and April had significantly more stomach contents than corresponding length
  groups in other seasons (Kruskal-Wallis test, p < 0.05)." "the length groups of cod and haddock which preyed
  upon herring eggs had more contents in March than at other times when herring eggs were unavailable."
- Diet: haddock > 30 cm, herring eggs "contributed from 35 to 100 per-cent in terms of weight"; "In March, both
  cod and haddock apparently responded immediately and strongly to the presence of a new food resource ... in
  some predator length groups virtually complete." Cod > 30 cm in the herring season: *Clupea harengus* 81.53 %W.
- Table 2 (haddock), "Mean weight of contents (g)": herring season < 30 cm 0.76 (n 62, 7 empty), > 30 cm 13.83
  (n 496, 41 empty); outside the herring season < 30 cm 0.48 (n 38, 0 empty), > 30 cm 2.22 (n 470, 60 empty).
- Table 3 (cod): herring season < 30 cm 1.98 (n 51), > 30 cm 79.04 (n 126); outside < 30 cm 1.10; > 30 cm not
  given (Danish-seine cod "were processed fully onboard ... but not weighed").
- Fig. 7, haddock mean content (g, read off): Feb 40-49 cm ~0.7, > 50 cm ~0.6; Mar 30-39 ~5.8, 40-49 ~14.7,
  > 50 ~14.9; Apr 30-39 ~13.1, 40-49 ~21.2, > 50 ~23.9; Jun 30-39 ~1.7, 40-49 ~2.1; Jul 30-39 ~1.5, 40-49 ~3.4,
  > 50 ~5.5; Oct 40-49 ~0.4, > 50 ~1.2; Dec 30-39 ~2.8, 40-49 ~7.3, > 50 ~7.6.
- Fig. 8, cod mean content (g, read off): Mar 30-49 cm ~23, 50-69 cm ~78; Jul 30-49 ~10, 50-69 ~43.
- The authors' caveat: "This may reflect a higher feeding rate during the herring season. However, as yet no
  direct estimates of consumption were made."
- Other gadoids: "Saithe (*Pollachius virens*) and pollack (*Pollachius pollachius*) were the more abundant
  species feeding on herring eggs." The full study is Høines & Bergstad (1999) J Fish Biol 55:1233-1257,
  doi:10.1111/j.1095-8649.1999.tb02073.x [A]: "strong interspecific diet overlap occurred ... when all the
  gadoids fed on herring eggs during the herring spawning season in the spring".

**Derived [D]** (`app_derive.py`). The load-bearing number is the tabulated one: haddock > 30 cm, herring season
(Mar-Apr) / outside (Oct, Dec, Feb, Jun, Jul pooled) = 13.83 / 2.22 g = **6.2**. Cod 30-49 cm March / July ~2.3;
50-69 cm ~1.8 (read off). Empty share, haddock > 30 cm: 8.3 % in the herring season, 12.8 % outside.
The February-based ratios (haddock 40-49 cm ~0.7 -> ~21 g in April, > 50 cm ~0.6 -> ~24 g, same gear) are NOT
used: the February values are inside the +-0.5 g read-off error, so the ratio is unbounded above, and February is
the annual minimum of Fig. 7 (December reads ~7.3-7.6 g for > 40 cm, July ~3.4-5.5 g).

**Reading.**
- [J] Haddock and cod at a herring spawning ground had ~6x (haddock, tabulated) and ~2x (cod, March vs July) the
  stomach content during the herring season than outside it, and the herring season (March-April) falls in
  their own spawning season. Late February and March are the coldest months on this coast (general knowledge,
  not from the paper), so the rise is not a temperature effect.
- [J] The paper does not give maturity. February's near-empty haddock stomachs could themselves be a
  pre-spawning fast of the kind Homans & Vladykov (1954) describe for NW Atlantic haddock (2.1 below); if so,
  the fish broke it when herring eggs arrived, or the egg feeders were the non-spawning part of the population.
  Either way, at the population level, easy dense food in March-April was eaten in quantity in the season of
  both the field low (February) and spawning.
- Cannot exclude: stomach content is not a ration (herring eggs may be evacuated at another rate than
  polychaetes and ophiuroids; no evacuation correction); research trawl (March, July) vs Danish seine (other
  months); which individuals were mature.
- **Bears on (A) vs (B):** for (B) in late winter - a late-winter appetite depression, if it exists in these
  fish, does not stop them eating a dense easy food. Haddock and cod; saithe and pollack join qualitatively.
  PROXY (whiting not reported here).

### 1.6 Wind-turbine reefs (Belgian North Sea) - one search; no season-resolved gadoid winter data [F/A] PROXY

- Vandendriessche S, Reubens J, Derweduwen J, Degraer S, Vincx M (2013) Offshore wind farms as productive sites for
  fishes? Chapter 14 in Degraer S et al. (eds) Environmental impacts of offshore wind farms in the Belgian part of
  the North Sea. RBINS. VLIZ copy (`data/vliz_252757/`) [F]. Pouting at the turbines "generally had fuller
  stomachs compared to the Gootebank (mean Fullness Index of 1.5 +- 1.4 and 0.6 +- 0.8 respectively)" - season
  not separated in the chapter. Cod: "During summer and autumn fish were present for an extended period of time.
  By the end of December however, most fish were no longer detected and throughout the winter months
  (December-March) few detections were encountered."
- Reubens JT, Vandendriessche S, Zenner AN, Degraer S, Vincx M (2013) Offshore wind farms as productive sites or
  ecological traps for gadoid fishes? Mar Environ Res 90:66-74, doi:10.1016/j.marenvres.2013.05.013 [A]: pouting
  "No significant differences in condition index ... At the WARs, they were slightly larger and stomach fullness
  was enhanced compared to the surrounding sandy areas."
- [J] The reef food raises pouting stomach fullness (~2.5x) but not condition. The sampling season of the
  fullness data is not given in the chapter (which states that cod and pouting were "clearly attracted towards
  the wind farm in summer and autumn"), and cod leave the reefs in winter, so this source cannot speak to winter
  appetite. Stopped here (one search).
  Note: pouting (*Trisopterus luscus*) is a congener of Norway pout; PROXY.

### 1.7 Trawl discards and damaged benthos - not searched in depth [search log]

One search (search log S9) returned no winter-resolved gadoid stomach or condition data. Kaiser & Spencer (1994)
MEPS 112:41 (whiting and gurnards eat more over fresh beam-trawl tracks) was done in spring and summer [S: search
summary]. Not pursued.

---------------------------------------------------------------------------------------------------------------

## 2. Lab and aquaculture intake across seasons (food in excess)

Most of this item is already in earlier notes; it is summarised here as [S] and only new sources are added.
No source was found that measures **voluntary intake by month at constant temperature in whiting, Norway pout or
saithe**. That is the main gap of this item.

### 2.1 Already established in earlier notes [S]

- **Cod, spawning anorexia** (LR, `Lek i rutan/lit_cod_haddock.md`, Q5-Q6). Fordham & Trippel (1999) J Appl
  Ichthyol 15:1-9, doi:10.1046/j.1439-0426.1999.00098.x, through citing papers: "Atlantic cod ... cease feeding for
  as much as 70 days during prespawning and much of the spawning period" (Dutil et al. 2006 [F there]). The full
  text was not obtained here either (Unpaywall lists a Wiley OA copy; the host returned 403). PROXY.
- **Cod, appetite not depressed during vitellogenesis** (LR): Yoneda & Wright 2005 [A there]: maturing females grew
  like immatures through Nov-Feb in captivity. Dutil et al. 2000 [F there]: wild Gulf of St Lawrence cod "fed but
  did not meet their energy requirements in winter prior to spawning", then fasted in April-May (spawning). PROXY.
- **Cod, light** (LA): Taranger et al. 2006 [A there]: continuous light raised growth "mainly as a result of the
  weight loss of the natural-light groups during spawning"; maturation "results in loss of appetite and weight
  during spawning". Juvenile cod on demand feeders (Glasgow MSc 2009 [A there]): intake was lowest in early
  summer, "the opposite result to that expected". PROXY.
- **Cod, hyperphagia after deprivation and intake at high liver** (TR section 3; HT Q3): compensatory hyperphagia
  after 5-12 weeks of deprivation (Jobling et al. 1994 [S]); on formulated diets intake stays at ~1.0-1.4x Cmax
  up to HSI 10-17 % with no shut-off as the liver passes the wild maximum [S]. So cod appetite responds to a
  depleted reserve (a lower lipostatic bound exists) but does not cap intake at wild liver levels. PROXY.
- **Haddock** (LR, TR): Hislop, Robb & Gauld 1978, J Fish Biol 13:85, doi:10.1111/j.1095-8649.1978.tb03416.x [A,
  re-read here via Crossref]: "Individual haddock ... were maintained at different feeding levels in an aquarium
  from November until the completion of spawning"; "The relation between daily growth in wet weight, g, and daily
  surplus energy intake, kcal, was: G = (0.295 E) - 0.328". The abstract says nothing about refused food in winter
  or around spawning. Homans & Vladykov 1954 [A there]: wild NW Atlantic haddock "practically cease to take food"
  approaching spawning, then "feed voraciously for a few weeks". PROXY.
- **Salmonids** (LA, HT): Metcalfe & Thorpe 1992 [A]: parr appetite fell 96 % Sep-Dec "even in the presence of
  excess food"; Bull 1996 [F]: anorexia with 24-h feeders. These show that an endogenous, light-entrained winter
  anorexia exists in some fishes. NOT gadoids.
- **Herring** (KV, LA): Berg et al. 2024 MEPS 741:203 (not redone): condition followed day length at constant 7-10 C
  with food in excess; the form (appetite, hours, allocation) is unresolved. PROXY for the herring share only.

### 2.2 New: Skjæraasen et al. 2004 - photoperiod and appetite in wild cod [S] PROXY

Skjæraasen JE, Salvanes AGV, Karlsen Ø, Dahle R, Nilsen T, Norberg B (2004) The effect of photoperiod on sexual
maturation, appetite and growth in wild Atlantic cod (*Gadus morhua* L.). Fish Physiol Biochem 30:163-174.
doi:10.1007/s10695-005-4319-6. Not obtained (Springer login wall; no abstract in Crossref, Semantic Scholar or
Europe PMC). Content from a search-engine summary only [S], no number used: wild coastal cod on a constant short
day vs a simulated natural photoperiod from June 2000; the short-day fish matured ~3 months earlier and spawned
from early November; appetite fell in both sexes two to three months before spawning, more in males, and length
growth fell with it.

**Reading [J].** The cod appetite fall is tied to the fish's own maturation clock (it moved with the advanced
spawning), not to the calendar winter. Whether immatures in the same tanks kept their appetite is not in the
summary. PROXY.

### 2.3 New: Otterå et al. 2009 - wild saithe fed formulated diets in captivity for 8 months [A] PROXY

Otterå H, Karlsen Ø, Slinde E, Olsen RE (2009) Quality of wild-captured saithe (*Pollachius virens* L.) fed
formulated diets for 8 months. Aquaculture Research 40:1310-1319. doi:10.1111/j.1365-2109.2009.02230.x. Abstract
(Europe PMC): "Young wild saithe were collected and fed either a lipid-rich salmon diet or a lean cod diet for 8
months ... Diet clearly influenced the growth rate of the fish, and many fish reached a very high hepatosomatic
index when fed on a salmon diet. However, many fish had a low feed intake and thus a low rate of growth."
Months, temperature and intake values are not in the abstract; full text not obtained (Wiley 403).

**Reading [J].** Saithe fed pellets in excess can drive the liver far above wild values (no lipostatic cap at wild
levels), but individual intake varied widely in captivity. Whether the low eaters were seasonal, sick or stressed
cannot be told from the abstract. This is the best unread source for a captive saithe season series.

### 2.4 New: Mazlan 2001 - whiting on demand feeders through autumn-winter [A]

Mazlan AG (2001) Food consumption patterns and dietary digestibility of whiting (*Merlangius merlangus* L.) fed in
laboratory conditions. PhD thesis, University of Wales, Bangor. No doi. Read: the repository abstract through
WebFetch (short quotes only): "whiting fed well during the autumn-winter (15°C) seasons" on artificial feed from a
demand feeder; "Peak feeding activities occurred between 'dawn' and noon' despite continuous 24h lighting";
"Overall growth performance was poor".

A one-time attempt to download the thesis PDF (repository file 57909405) returned 403.

**Reading [J].** The only captive whiting source with a season in it. At 15 °C (as given; whether the
temperature was held constant is not stated in the abstract) and continuous light the whiting kept feeding in
autumn-winter. Continuous light removes the photoperiod cue, so this cannot test a
light-entrained depression; it shows no free-running winter anorexia over one season. WHITING (target species),
but weak: one abstract, poor growth, no numbers.

### 2.5 Item 2 synthesis [J]

- Gadoid appetite falls around the fish's OWN spawning (cod, haddock; 1-2 months of near-fast plus a late
  pre-spawning decline), moves with the maturation clock under photoperiod manipulation, and is followed by
  hyperphagia. That is maturity-bound, not winter-bound.
- No gadoid source shows a winter depression of voluntary intake in IMMATURE fish at constant temperature with
  food in excess. The proxy evidence points the other way: continuous light gains in cod come mainly from removing
  the spawning loss (Taranger 2006 [S]); juvenile cod on demand feeders ate least in early summer [S]; whiting at
  15 C under continuous light fed well in autumn-winter [A].
- A lipostatic element exists at the low end (cod hyperphagia after deprivation [S]) but not as a cap at wild
  liver levels (cod and saithe on pellets reach HSI far above wild values [S, A]).

---------------------------------------------------------------------------------------------------------------

## 3. Condition and liver against ration in the same wild fish in winter

### 3.1 Nedreaas 1985 - young saithe on krill in March: stomachs at summer levels, starving [F] PROXY (saithe)

Nedreaas K (1985) Food and feeding habits of young saithe, *Pollachius virens* (L.), on the coast of western
Norway. ICES C.M. 1985/G:22. No doi. Read: full text, ICES CM archive scan (`data/nedreaas1985/`, pdftotext), and
Figure 5 viewed as a rendered page image (values read off, +-0.2 g).

- Fish and sampling: "taken with purse seine (1 June only) or handline from the beginning of June to the end of
  September 1982. To look at the feeding of the saithe in the winter a sample was taken 1 March 1983." Area A
  (Tælavåg): "almost all the saithe were two years old, and mainly 30-34 cm ... In March 1983 the same yearclass
  dominated (now three years old)". 328 stomachs; "Only 0.6 percent of the stomachs were empty."
- Growth: "In area A the mean length of the 1980-yearclass, (two-year-old), increased by 3.1 cm from 1 June 1982
  until 1 March 1983."
- Abstract: "In the winter when the krill seemed to be the most important single food organism, the saithe showed
  clear signs of starvation."
- Discussion: "In March the saithe gave clear impressions of starvation. By starvation the cod liver becomes small
  and reddish, and loss of fat causes the red colour (Love 1980). The gall bladder increases in volume and the
  colour of the gall becomes darker. All these starvation symptoms were present in the saithe. Larvae of the
  gastric parasite *Anisakis simplex* were found in great amounts in the tiny liver ... krill was the main food
  for the saithe at this time." "In the beginning of March, the saithe examined seem mostly to have been preyed on
  krill, and the observed transport of krill away from the coast may have led the saithe to follow." The krill
  species in this area: "*Thysanoessa inermis* was the most important krill species".
- Fig. 5 (area A, mean total stomach content, g, read off): 30-34 cm: 1 Jun 1.35, 25 Jun 3.8, 29 Jul 1.9, 31 Aug
  0.6, 30 Sep 3.5, **1 Mar 2.1**; 35-39 cm: 25 Jun 1.4, 29 Jul 2.0, 31 Aug 1.7, 30 Sep 4.1, **1 Mar 2.0**.
- [D] March content 2.0-2.1 g, inside the Jun-Sep range of occasion means (0.6-4.1 g); at the paper's W = 0.0085
  L^3 (349 g at 34.5 cm) it is ~0.6 % BW wet.

- Caveats in the paper itself: the Fig. 5 weight is "the total wet weight of the stomach content (incl. not
  identified fragments, gastric juices, and small parasites)", and the March fish carried a heavy *Anisakis*
  load. Food presence in March is NOT shown: "Krill and decapoda larvae were very scarce in the plankton samples"
  (Juday net, 0.18 mm, which samples krill poorly [J]), and the author links the March krill diet to "the
  observed transport of krill away from the coast".

**Reading.**
- [J] Immature (2-3-year-old) saithe of box size, with krill as the main stomach item in late winter, stomach
  contents at the summer level and almost none empty, and yet small red livers, enlarged dark gall bladders and
  almost no growth. A fish at a satisfied lipostatic set point would not be in that state, and a depleted fish
  with a working lipostat would be hyperphagic if food were easy (cod hyperphagia [S]). So the datum is
  **against lipostatic satiety**.
- [J] It is **neutral between a seasonal (A) and (B)**: eating little (equal content at a lower winter
  evacuation rate means a lower ration than in summer) while losing liver is also what a seasonal or light-cued
  anorexia produces (salmon parr, 2.1). And food presence is not shown - the krill may have been leaving the
  coast - so the low ration may simply be low supply rather than low rate inside a layer.
- Cannot exclude also: one March sample in one year; starvation signs are qualitative (no HSI); the content
  weight includes gastric juice and parasites; the gall-bladder sign usually means hours to days without a meal,
  which sits oddly with 0.6 % empty stomachs [J].
- PROXY (saithe); the closest match found to the Oslofjord setting (gadoid, box size, krill, winter), but not a
  test of (A) vs (B) by itself.

### 3.2 Albert 1994 - Norway pout in the Norwegian Deep: more food in winter stomachs, liver lost [S] TARGET (pout)

Albert OT (1994) Biology and ecology of Norway pout (*Trisopterus esmarki*) in the Norwegian Deep. ICES J Mar Sci
51:45-61, doi:10.1006/jmsc.1994.1005 (title as cited in `Fiskgrupper i rutan`). [S]:
read in full (local scan) by `Planktivorernas vinterkostnad/evidence.md`; not re-read here. Winter cruises 31 Jan
- 13 Apr 1985-87. Winter empty stomachs 33.6 % (< 14 cm) and 24.5 % (>= 14 cm) vs summer 27.8 % and 18.6 %; mean
dry content of non-empty stomachs 16.9 / 57.8 mg in winter vs 23.5 / 31.2 mg in summer; "The mean dry weight of
the stomach contents relative to fish weight (including empty stomachs) was consistently higher in winter than in
summer"; winter diet euphausiids 78.6 / 81.4 % by weight; "a pronounced reduction in condition factor and liver
weight during the winter season"; "Close to 100% matured for the first time as II-group", spawning from about
15 February, mainly March-April.

**Reading [J].** Pout eat krill through winter, with more food in the stomach relative to body weight than in
summer, and still lose liver and condition. Against lipostatic satiety (they eat while depleting). It is the
stronger of the two winter datasets against a SEASONAL (A) too: bottom temperature in the Deep varies little
with season ("6-9 °C", Raitt [S via the pout note]), so higher winter content is unlikely to come only from
slower evacuation, and a fish with a seasonal anorexia would not carry more food in winter than in summer. The
liver loss is what a fish paying for its first (and usually only) spawning on a krill ration shows; spawning is
the confound. The earlier note (LA) judged that a winter appetite suppression would contradict pout's data, and
nothing here changes that. TARGET SPECIES (pout).

### 3.3 Mathers et al. 1992 - northern North Sea saithe, winter fasting [S] PROXY

From MR (`lit_fill_series.md` 1.1) [S]; authors and title checked on the local text: Mathers EM, Houlihan DF,
Cunningham MJ (1992) Estimation of saithe *Pollachius virens* growth rates around the Beryl oil platforms in the
North Sea: a comparison of methods. MEPS 86:31-40, doi:10.3354/meps086031: condition "highest values in August and the lowest in April in most cases"; the
gall-bladder-liver index "increased in winter because bile accumulated as feeding was reduced"; RNA-based growth
negative in April 1989; weight lost Oct-Apr. Ages 2+-5+, the older ones mature (spawning Feb-Mar). No stomach
weights by season and no food-field measure, so it cannot tell low appetite from low food [J]. PROXY.

### 3.4 Lauerburg 2016 thesis - North Sea whiting stomach content and liver track forage fish supply [F] TARGET

Lauerburg RAM (2016) Governing factors for the size-at-age: the influence of sex, food availability, temperature
and fishery on the growth performance of North Sea whiting *Merlangius merlangus* and cod *Gadus morhua*. PhD
thesis, University of Hamburg (ediss 6770) [F, local text from MR
`../Mättnad hos rovfisk/data/uhh_ediss6770/a.txt`]. Q1 2012 vs Q3 2007 research-cruise whiting: "fish from the
third quarter in 2007 had lower liver masses expressed as percentage of the total body mass compared with first
quarter samples from 2012"; "the mean stomach content mass of individuals from 2012 was almost twice as high as
compared with 2007" (females > 25 cm). Chapter 2: "the average stomach content masses were substantially lower
during the period of observed low forage fish abundance" (2000-2007), together with lower growth. From the ICES
stomach data: "In 1981, as well as in 1991, whiting showed higher percentages of empty stomachs in the third
quarter compared with the first quarter. Moreover, the stomach content mass in both years is higher in third
quarter samples than in first quarters".

**Reading [J].** Across years, whiting intake and liver go with prey-fish supply: more prey, fuller stomachs,
bigger livers, faster growth. That is the acquisition signature (B). It is interannual, and the 2007 vs 2012
contrast is confounded with quarter; the 1981/1991 Q1 < Q3 stomach mass is season plus temperature plus prey
field, so it cannot be read as winter appetite. TARGET SPECIES (whiting), indirect.

### 3.5 Item 3 synthesis [J]

- The two winter datasets with stomachs and condition in the same fish on krill (young saithe, Nedreaas; pout,
  Albert) both show fish EATING (stomachs at or above summer levels, few empty) and LOSING liver. That pattern
  is against lipostatic satiety (a regulated low demand keeps the liver). Between a seasonal (A) and (B) the
  saithe datum is neutral (low winter ration while depleting fits both, and food presence is not shown); the pout
  datum leans against a seasonal (A) (more content in winter at near-constant temperature) but carries the
  spawning confound. The (A) vs (B) weight outside summer therefore rests on the easy-food data of item 1.
- No winter dataset was found in which wild whiting, pout or saithe keep their liver while eating little with
  food shown present. That is the pattern (A) predicts; it was not found, which is absence of evidence, not
  evidence against (A).

---------------------------------------------------------------------------------------------------------------

## 4. Search log (what was searched; [x] = found and used, [-] = not found or not obtained)

- S1 [x] Uglem 2020 months: companion Toledo-Guedes, Ulvan & Uglem 2016 (doi:10.1051/alr/2016013): publisher PDF
  and HTML 403, NINA repository host unreachable. Months remain unknown.
- S2 [x] Otterå & Skilbrei 2014, Skilbrei & Otterå 2016: OUP PDFs 403; HTML read through WebFetch.
- S3 [-] Fernandez-Jover et al. 2011 ECSS: publisher 403; figshare 25000757 is metadata only; RUA page empty to
  the fetch tool; CORE 403. Search snippet only.
- S4 [-] Skog et al. 2003 Aquac Res: not OA (Unpaywall). Not read.
- S5 [x] Dempster 2009 MEPS, Uglem 2009 MEPS, Uglem 2014 AEI: OA via Unpaywall, read; no winter feeding data.
- S6 [-] Otterå et al. 2009 Aquac Res (captive saithe 8 months): Wiley 403; abstract via Europe PMC.
- S7 [-] Fordham & Trippel 1999, Pedersen & Jobling 1989 (Aquaculture 81:161), Skjæraasen et al. 2004, Hislop 1975
  (ICES JMS 36:119), Jones & Hislop 1978 (ICES JMS 38:244): no full text; abstracts absent from Crossref, Semantic
  Scholar and Europe PMC except Hislop et al. 1978 (Crossref). OpenAlex rate-limited (429).
- S8 [x] Wind farms (Reubens 2013 MER; Vandendriessche et al. 2013 chapter): one search; season-pooled.
- S9 [-] Trawl discards (whiting): one search; Kaiser & Spencer 1994 spring-summer only.
- S10 [x] Herring spawning grounds (Høines et al. 1992 ICES CM; Høines & Bergstad 1999 JFB [A]).
- S11 [x] Young saithe winter krill (Nedreaas 1985 ICES CM 1985/G:22), found through a saithe captivity search.
- S12 [-] Saithe captive intake by season: one search, nothing beyond Otterå 2009.
- S13 [x/-] Whiting captive intake by season: one search; Mazlan 2001 thesis abstract only; Seyhan & Grove 1998
  (Fish Res, doi:10.1016/s0165-7836(97)00080-5) is meal dynamics, not season (already in IF).
- S14 [-] Carss 1990 Aquaculture 90:29 (Scottish farms, saithe): abstract only via search; no season.
- S15 [-] Mazlan 2001 thesis PDF (Bangor repository file 57909405): 403 to a generic client; abstract only.
- S16 [-] Nedreaas 1985 IMR report version (brage handle 11250/104030): host unreachable; the ICES CM version
  was read instead.
- Citations named in the task, status:
  - Bjordal & Skar 1992 (saithe tagging at a farm, ICES CM): not searched (time).
  - Uglem et al. 2008 (cod at farms): not searched; cited in Uglem 2014 only for movements.
  - Dempster et al. 2010 (Species-specific patterns of aggregation of wild fish around fish farms, ECSS,
    doi:10.1016/j.ecss.2009.11.007): located through Crossref, not read.
  - Mavraki et al. 2020-2021 (wind-farm fish diets): not searched beyond S8 (time).
  - Groenewold & Fonds 2000, Bergmann et al. 2002 (discards): not searched beyond S9.
  - Cod gorging on capelin or herring in winter-spring: only Høines 1992 (herring, S10); capelin not searched
    (IF has Pednekar 2024 [S]).
- Not searched (time): Norway pout lab intake (none expected); haddock broodstock intake (LR found none).

---------------------------------------------------------------------------------------------------------------

## 5. What the appetite-side evidence says about (A) vs (B) [J]

By season, strongest first. Every gadoid datum except Mazlan (whiting) and Albert / Lauerburg (pout, whiting) is a
PROXY.

| Season | Datum | Points to | Tag |
|---|---|---|---|
| Late winter (Mar-Apr) | Haddock > 30 cm stomach content x6.2 (herring season vs rest of year, tabulated); cod Mar 1.8-2.3x Jul; inside their spawning season (Høines 1992) | B | F, proxy |
| Late winter (1 Mar) | Immature saithe on krill: stomachs at summer level, tiny red livers, starvation signs; krill presence not shown (Nedreaas 1985) | against lipostatic satiety; neutral seasonal A vs B | F, proxy |
| Winter (Feb-Apr) | Pout on krill: more content per body weight than summer at near-constant deep temperature, liver lost; spawning confound (Albert 1994) | against lipostatic satiety; leans against seasonal A | S, target |
| Nov-Jan | Saithe hold closer to farms in winter (0.82 vs 0.64), leave when feeding stops; association, not intake (Otterå & Skilbrei 2014) | weak B (motivation present) | F-w, proxy |
| Spring | Saithe pellet meals 5.4 % BW (jig), HSI x1.6 at farms (Uglem 2020) | B | F, proxy |
| Autumn | Pellets in 29-33 %, HSI x0.8-1.2 (Uglem 2020); smaller surplus, supply not controlled | weak B | F, proxy |
| Summer | Dempster 2011: stomach x2.35, HSI x1.4-1.8 (MR) | B | S, proxy |
| Around own spawning | Cod and haddock near-fast 1-2 months; appetite falls 2-3 months before spawning and moves with the maturation clock (LR, Skjæraasen 2004) | A, maturity-bound | S, proxy |
| Autumn-winter, lab | Whiting fed well at 15 C, continuous light (Mazlan 2001) | no A found | A, target |

- **No gadoid evidence of a winter appetite depression in immature fish** was found; the one appetite depression
  that is documented is tied to the fish's own spawning. North Sea whiting spawn "January until July" (south to
  north; Lauerburg 2016 via MR [S]) and pout from mid-February, mainly March-April (Albert 1994 [S]), so that
  window can overlap the March Oslofjord sample for mature fish, but not November [J].
- **A lipostat exists only as a lower bound** (cod hyperphagia after deprivation [S]); fish on easy food push the
  liver well above wild values in summer and spring (saithe HSI 10-12 %, cod > 12 % [S]), so the wild level is not
  a defended set point.
- **What is missing**: an intake datum with easy food in NOVEMBER (one of the two Oslofjord months) or Dec-Feb
  for any gadoid. November is covered only by farm association (Otterå & Skilbrei 2014, Nov-Jan) and by Uglem's
  undated "autumn" occasion with a weak HSI surplus. Also missing: any monthly intake at constant temperature for
  whiting, pout or saithe. Winter proper is therefore a gap; late winter (March-April) is covered by proxies.
