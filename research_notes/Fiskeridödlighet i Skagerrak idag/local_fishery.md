# The local fishery in the Mareld / Poseidon Nord grid (57.81-58.31 N, 10.07-11.15 E): rules, effort, landings and a crude local F

Scope note: about 30 web tool calls plus local work. Local files read: the Mareld EIA text (`mkb.txt` in the session scratchpad, extracted from `/home/simon/eco-evo/MKB Mareld.pdf`; line numbers below refer to `mkb.txt`), PDF pages 179-180 of the EIA rendered as images to read the labels of Figure 9-21 and Figure 9-22, `/home/simon/eco-evo/make_pelagisk.py` and `/home/simon/eco-evo/coordinates.txt`. I also downloaded the EMODnet Human Activities WFS layers `fishingbottomottertrawls`, `fishingpelagic`, `fishingbottomseines` and `fishingstaticgears` for 56.9-59.6 N, 6.9-12.0 E and summed them over the model grid. I used the SWEREF 99 TM transform copied from `make_pelagisk.py`, and a c-square counts as in the grid when its centre falls inside the grid square. These computations are marked (Local). Full texts read: EU Regulation 2017/118 (EUR-Lex), EU Regulation 2019/1241 (EUR-Lex, mesh tables), the 1984 Sweden-Denmark boundary treaty, the ICES Greater North Sea Fisheries Overview 2020 (PDF), two Swedish government press releases, an SU explainer and a Danish Fisheries Agency page. Several sources rest on search-engine snippets only and are marked (Snippet). Rectangle-level landings data for 2021-2024 (SLU/HaV/DK/NO) were NOT found in open form, so the catch-based numbers come from the EIA's 2010-2020 data.

Geometry used throughout (Local, computed): ICES rectangle 45G0 (58.0-58.5 N, 10-11 E) is 3,257 km2, all sea, and 1,961 km2 of it (60%) lies in the grid. 44G0 (57.5-58.0 N) is 3,303 km2, with 1,211 km2 (37%) in the grid. 45G1 and 44G1 contribute 239 and 171 km2. **The grid is therefore split between 45G0 (about 54% of grid area) and 44G0 (about 34%), not "inside 45G0".** The EIA's statement that the area is in 45G0 refers to the Mareld wind farm only. The Mareld wind farm itself covers 369 km2 (`mkb.txt:1961`).

## Q1. Regulations in and around the grid

### Takeaway
The grid is open to all three Skagerrak nations (Sweden, Denmark, Norway). It is fished mainly with bottom otter trawls (mostly northern shrimp, *Pandalus*, with 35 mm mesh and a 19 mm grid, and Nephrops/fish trawls) and with pelagic trawls. EU Skagerrak mesh rules apply everywhere: a 120 mm baseline, or 90 mm, with derogations that require sorting grids. The only permanent spatial closure inside the grid is the southern edge of the **Bratten** reef zones (EU 2017/118, all fishing banned), in the NW corner and along the top rows. The Swedish-Danish zone boundary runs diagonally through the grid, putting about half of it in Danish waters. No decision moves the Skagerrak trawl limit out to 12 nm: the 2025-2026 Swedish trawl-limit moves concern the Baltic. The Swedish ban on bottom trawling in marine protected areas (MPAs) inside the trawl limit (from mid-2026) touches at most the easternmost coastal edge of the grid. Koster is outside the grid, at about 58.8-59 N.

### Cited Findings
- **Bratten, EU Delegated Regulation 2017/118.** "It shall be prohibited to carry out any fishing activity in Areas 2". The Bratten zones are among the Areas 2 (Annex II), and "All fishing vessels present in Bratten shall be fitted with and maintain in operation an automatic identification system". Zone BRATTEN 14 spans 58.26667-58.53608 N, 10.02858-10.26041 E. Zones 9A/9B (58.28-58.33 N, 10.46-10.51 E) and zones 7D, 7E and 12 (58.30-58.33 N, 10.39-10.45 E) also reach below the grid's top edge at 58.31 N. Entry into force was 20 days after publication in OJ L 19 of 25 Jan 2017. (Fetched full text) - [EUR-Lex 32017R0118](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32017R0118)
- The EIA's version of the same measure: "Genom beslut av EU-kommissionen i januari 2017, har yrkesfisket reglerats genom att det har inrättats fiskefria zoner för samtliga länders fiskare i området. Dessutom beslutades om obligatorisk användning av AIS i hela området." (Local, `mkb.txt:1629-1631`)
- **Bratten zones mapped to model cells (Local, computed).** Pixel row 0 is the north edge and column 0 the west edge, using bounding boxes, so these are upper bounds. Zone 14 covers rows 0-5 and columns 0-11, at most about 72 cells. 9A/9B cover rows 0-5, cols 22-26. 7D, 7E and 12 cover rows 0-2, cols 19-22. The EMODnet bottom-otter-trawl layer has 0-2 MWh per c-square at 58.275-58.325 N, 10.03-10.25 E in 2015-2018, against 100-200 in the cells next to it. That matches the zone 14 closure.
- **Sweden-Denmark boundary (1984 agreement).** Article 3 lists "Point A 58° 15' 41,2" N, 10° 01' 48,1" E; Point B 58° 08' 00,1" N, 10° 32' 32,8" E; Point C 57° 49' 00,6" N, 11° 02' 55,6" E". (Fetched full text) - [SWE-DNK 1984 treaty (Marine Regions)](https://marineregions.org/documents/SWE-DNK1984CS.PDF)
  - (Local, computed) In grid pixels: A ~ (col -2.5, row 5.4), B ~ (col 26.6, row 21.8), C ~ (col 54.3, row 58.8). The line runs from the NW corner to the SE corner, and 104 of the 216 c-squares in the grid (48%) lie on the Danish side.
  - Mareld corner 2 (E 237816, N 6452763; `coordinates.txt`) sits about 0.6 km from point B, so Mareld's western edge follows the boundary.
  - Point A is the western end of the line and the grid's NW corner is just east of it, so any Norwegian zone inside the grid is at most a sliver. Bratten zone 14, a Swedish site, has its western edge at 10.0286 E from 58.267 to 58.536 N, so the Swedish zone runs north from point A west of the grid's edge (10.065 E). That last point is an inference.
- **Reciprocal access.** The 1966 Skagerrak Agreement gave Danish, Norwegian and Swedish vessels access up to 4 nm from each other's baselines. It expired on 7 Aug 2012. Its successor, signed 15 Jan 2015, "maintains exclusive access beyond 4 nautical miles from the baselines and ensures continued reciprocal access". (Snippet) - [EP Legislative Observatory summary](https://oeil.europarl.europa.eu/oeil/en/document-summary?id=1443198)
  - The EIA confirms that Danish, Norwegian, Swedish and British vessels fish in 45G0 (Local, `mkb.txt:7239-7241`).
- **EU Skagerrak mesh rules, Reg. (EU) 2019/1241, Annex V Part B.**
  - "vessels shall use a mesh size of at least 120 mm or at least 90 mm in Skagerrak and Kattegat".
  - Derogations: "At least 70 mm (square mesh) or 90 mm (diamond mesh) Skagerrak and Kattegat Directed fishing for Norway lobster ... A sorting grid with a maximum bar spacing of 35mm ... shall be fitted". "At least 35 mm Skagerrak and Kattegat Directed fishing for Northern prawn ... A sorting grid with a maximum bar spacing of 19mm". "At least 16 mm Whole area Directed fishing for small pelagic species ... Norway pout" (35 mm grid for pout).
  - Smaller meshes are conditional on "by-catches of cod, haddock and saithe do not exceed 20 %".
  - (Fetched full text, original 2019 text. A 2022 amendment, 2022/200, limits the 90 mm derogation to otter trawls according to a search snippet.) - [EUR-Lex 32019R1241](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32019R1241)
- **2020 cod measures.** "additional measures for demersal fishing in the Skagerrak and the North Sea to protect cod ... enter into force on 15 August 2020". Towed gears "from 90 mm throughout the Skagerrak (the EU part)" are subject to supplementary measures. (Snippet; the page returned 403 when fetched) - [The Fishing Daily](https://thefishingdaily.com/?p=11574)
  - "a new national Danish cod plan was implemented in August 2020", which introduced among other gear a SELTRA 300 square-mesh panel. (Snippet) - [DTU Aqua report 380-2020](https://www.aqua.dtu.dk:443/-/media/institutter/aqua/publikationer/rapporter-352-400/380-2020_testing-a-large-opening-as-an-alternative-to-seltra-300-panel.pdf)
- **Real-time closures for juvenile gadoids, North Sea/Skagerrak.** Triggered when juveniles are 10% by weight of cod, haddock, whiting and saithe in a sample of at least 200 kg. "The area will be closed for 21 days." Pelagic trawls, purse seines and pots stay allowed. (Fetched full text) - [Danish Fisheries Agency, RTC page](https://lfst.dk/english/fishery-control-and-enforcement/real-time-closures)
- **Swedish trawl limit.** It "ligger 4 respektive 3 nautiska mil från baslinjen i Skagerrak". "Dock finns också många så kallade inflyttningsområden från trålgränsen där man ändå får tråla". HaV has a 2025 assignment "att ta bort eller drastiskt minska inflyttningsområdena". (Fetched full text) - [SU Östersjöcentrum, Q&A on trawling, 2025-10-21](https://www.su.se/enheter/stockholms-universitets-ostersjocentrum/nyheter/nyhetsartiklar/2025-10-21-fragor-och-svar-om-tralning)
  - Shrimp trawling inside the limit: "får bedrivas på minst 60 meters djup", with a species-sorting grid and at least 35 mm mesh, under HVMFS 2015:7. Older text, current version not verified. (Snippet) - [HVMFS 2015:7](https://www.havochvatten.se/download/18.39e6d68414ca353051ff1074/1430823225736/HVMFS%202015-7-ev.pdf)
- **Swedish MPA bottom-trawl ban.** A general ban on bottom trawling in marine national parks, nature reserves, biotope protection areas and Natura 2000 sites. It applies only "innanför trålgränsen", where only Swedish vessels may fish, and it enters into force on 1 July 2026. (Fetched full text) - [Regeringen press release, June 2025](https://www.regeringen.se/pressmeddelanden/2025/06/generellt-forbud-mot-bottentralning-infors-i-marina-skyddade-omraden/)
  - The SU page gives 1 June 2026. I treat the government's 1 July 2026 as authoritative.
  - HaV has proposed exceptions for shrimp trawling in Koster-Väderöfjorden and a phase-out in Gullmarsfjorden by 1 July 2027. (Snippet) - [HaV konsekvensutredning](https://www.havochvatten.se/download/18.53868cbd19cc4272f336e1e4/1773396052451/Konsekvensutredning%20-%20Bottentr%C3%A5lning%20MPA%20Koster-V%C3%A4der%C3%B6%20och%20Gullmarsfjorden%20slutlig.pdf)
- **Moving the trawl limit to 12 nm** covers the Baltic only: vessels over 24 m with active gear "från Skanör till Haparanda, inklusive Öland och Gotland". The release says nothing about the Skagerrak. (Fetched full text) - [Regeringen, 1 Sept 2026](https://regeringen.se/pressmeddelanden/2026/09/permanent-utflyttning-av-tralgransen-for-storskaligt-pelagiskt-fiske/)
- **Danish side.** As of early 2023, H1 Skagens Gren og Skagerrak was designated for reef protection but still had no ban on bottom-towed gear on the reefs. (Snippet) - [Danish ministerial answer, eu.dk](https://eu.dk/samling/20231/beslutningsforslag/B102/spm/1/svar/2044099/2858624.pdf). I did not find a later Skagerrak trawl ban.
- **National interest for commercial fishing.** "Den planerade vindkraftparken ligger till stor del inom ett utpekat riksintresse för yrkesfiske, Norra Skagerrak utsjöområde ... utpekat fångstområde för räka. Sydöst ... Södra Skagerrak utsjöområde, som är utpekat för havskräfta, torsk och 'flatfisk m.fl. bottenfiskar'". "Norra Skagerraks trålgränsområde" lies about 13 km east of Mareld and is designated for shrimp and Nephrops. (Local, `mkb.txt:1702-1708`)

### Inferences
- In the model grid, **towed bottom gear (otter trawl) and pelagic trawl are allowed almost everywhere**. The exceptions are the Bratten cells in rows 0-5 (no fishing of any kind) and, after mid-2026, any Swedish MPA inside the trawl limit at the far eastern edge. I could not place the trawl-limit line inside the grid. The "trålgränsområde" 13 km east of Mareld suggests the limit lies at or just beyond the grid's eastern columns.
- Gear selectivity matters more for gadoid fishing mortality than closures do. The dominant local bottom fishery is shrimp trawling with a 19 mm grid, which is designed to exclude fish. Its gadoid bycatch is limited by gear rules and by the 20% bycatch cap, not by area.

## Q2. Local effort and landings

### Takeaway
2010-2020 landings in 45G0 by the three Nordic fleets averaged about 9,900 t/yr, or about 3.0 t/km2/yr of all species. By tonnage this is mostly herring, northern shrimp, blue whiting and Norway pout. The gadoids pooled in the model (cod, saithe, haddock, hake) are only about 5.9% of tonnage, and pelagic fish proper (herring, mackerel, horse mackerel, sprat) about 37%.

The grid's bottom-otter-trawl effort density is consistently about 1.5-1.8x the nine-rectangle neighbourhood and about 1.75-2.1x the Skagerrak average. Pelagic effort density is about the same as the neighbourhood (0.75-1.4x). Static-gear effort by VMS vessels is essentially zero. Inside Mareld the Swedish fleet fishes about 1.7x the rectangle-average effort per km2, almost entirely for northern shrimp.

### Cited Findings
- **Fleet composition.** "Inom just ICES-rektangel 45G0 står det pelagiska trålfisket och bottentrålfisket för den största fångsten (57,3 respektive 36,7 %). Av den resterande fångsten står fiskeredskapen snurrevad (5,1 %), nät/garn (0,6 %), burar/fällor (0,2 %), långrev (0,1 %) och övriga redskap (0,2 %)." (Local, `mkb.txt:7248-7251`)
- **Landings by nation.** "Mellan åren 2010-2020 låg den årliga genomsnittliga totala fångsten för norska fiskeflottan på 4 334,5 ton, för den svenska fiskeflottan 3 024,4 ton och den danska fiskeflottan 2 532,8 ton". UK catches are not included (`mkb.txt:7240-7241`). 45G0 had the third-highest total after 44F9 and 44G0. (Local, `mkb.txt:7305-7309`)
  - Sum: 9,891.7 t/yr. Divided by 3,257 km2 that is 3.04 t/km2/yr (computed).
- **Internal inconsistency in the EIA.** Table 9-10 gives the Swedish fleet "25 132" t in 45G0 over 2010-2020, which is 2,285 t/yr over 11 years. The text above says 3,024.4 t/yr. (Local, `mkb.txt:7442-7456`). I have not resolved which is right. The difference is 30%.
- **Species shares, 45G0, three fleets, 2010-2020.** The text says "sill (32,5 %), nordhavsräka (21,2 %), kolmule (blåvittling) (17,8 %) och vitlinglyra (10,5 %)" (Local, `mkb.txt:7331-7333`). The remaining bars were read from the rendered Figure 9-22 (Local, EIA PDF p. 180):
  - makrill 3.7, gråsej 3.1, rödtunga 2, torsk 1.8, havskräfta 1.8, tobis 1.1
  - kolja 0.9, taggmakrill 0.8, marulk 0.5, rödspätta 0.3, kummel 0.1, skarpsill 0.1, övrigt 1.9
  - Whiting (vitling) has no bar of its own.
- **Gear-by-rectangle landings, Figure 9-21** (Local, EIA PDF p. 179, read visually, approximate):
  - 45G0 totals range from about 4,000 to about 14,000 t/yr, mostly midwater trawl and bottom trawl.
  - 44G0 is mostly midwater trawl, about 7,000-20,000 t/yr, with a peak near 30,000 t in 2018.
- **Swedish fleet in Mareld vs the rest of 45G0, 2010-2020** (Local, `mkb.txt:7442-7456`):
  - 45G0: 25,132 t, EUR 73,458,000, 226,610 h.
  - Inside Mareld: 2,788 t, EUR 15,034,000, 44,816 h.
  - "Fiskeansträngningen inom projektområdet under 2010-2020 har uppmätts till 44 816 h vilket motsvarar ungefär 20 % utav den totala fiskeansträngningen i ICES-rektangel 45G0."
  - Computed per km2 per year over 11 years: inside Mareld 0.69 t, 11.0 h and EUR 3,700. In 45G0 as a whole 0.70 t, 6.3 h and EUR 2,050. So effort density is 1.74x and value density 1.8x the rectangle average, while tonnage density is the same.
- **Catch composition, Swedish logbooks.** "nordhavsräkan utgör majoriteten av yrkesfiskets fångster (viktförhållande); 86,7 % inom 45G0 och 98,47 % inom området för den planerade vindkraftparken i jämförelse med havskräftan som utgör 12,61 % av totala fångsten inom 45G0 respektive 0,84 %". In context these are shares of the shellfish catch. (Local, `mkb.txt:3925-3928`)
  - Shrimp "generellt är förekomsten jämnt fördelad inom djupintervallet 100-500 meter". (Local, `mkb.txt:3932-3933`)
  - Depth at Mareld is about 180-450 m. (Local, `mkb.txt:1058`)
- **VMS maps in the EIA.**
  - Swedish effort in the project area is "25,1 - 500 h/c-ruta där den högsta fiskeansträgningen sker i östra och norra delen av området", against 25.1-250 just outside. (Local, `mkb.txt:7376-7379`)
  - The Danish fleet "är som mest aktiv söder och sydväst om området ... över 500 VMS pingar/c-ruta jämförelsevis med 25,1-200 VMS pingar/c-ruta inom projektområdet". (Local, `mkb.txt:7394-7397`)
  - There are "stora områden norr och sydöst om projektområdet där fiskeansträgningen är likvärdig". (Local, `mkb.txt:7615-7616`)
- **Stock-scale context from the EIA.** Swedish herring landings for the whole Skagerrak were 11,400 t in 2018 and 8,800 t in 2019. Norway took 63% of the northern shrimp catch in 2019. (Local, `mkb.txt:7164-7171`)
- **EMODnet effort, grid vs neighbourhood** (Local, computed). These are ICES Fisheries Overview VMS layers. Values are MWh per year per 0.05 x 0.05 deg c-square, divided by area.

  | Gear, period | grid MWh/km2/yr | 9-rect mean | ratio | Skagerrak polygon mean | ratio |
  |---|---|---|---|---|---|
  | Bottom otter trawl 2015-2018 | 4.92 | 3.17 | 1.55 | 2.63 | 1.87 |
  | Bottom otter trawl 2018-2021 | 6.03 | 4.02 | 1.50 | 3.43 | 1.75 |
  | Bottom otter trawl 2021-2024 | 3.02 | 1.67 | 1.81 | 1.46 | 2.07 |
  | Pelagic trawl/seine 2015-2018 | 0.131 | 0.096 | 1.36 | 0.079 | 1.66 |
  | Pelagic trawl/seine 2021-2024 | 0.023 | 0.031 | 0.75 | 0.032 | 0.73 |
  | Bottom seine 2021-2024 | 0.077 | 0.252 | 0.31 | 0.251 | 0.31 |
  | Static gears 2021-2024 | 0.000 | 0.040 | 0.00 | 0.060 | 0.00 |

  - The 9-rectangle set is 44F9-46G1, the EIA's Fig. 9-19 neighbourhood: 1,223 c-squares with effort, 19,964 km2. The Skagerrak polygon is hand-drawn (Hanstholm-Lindesnes, Skagen-Tistlarna): 1,797 c-squares, 29,412 km2.
  - The grid holds 216 c-squares (3,536 km2): 120 in 45G0, 70 in 44G0, 14 in 45G1 and 12 in 44G1.
  - Absolute Skagerrak totals fall by about 60% from the 2018-2021 window to the 2019-2022 window for bottom otter trawl (101,000 to 32,500 MWh). The grid ratios stay at 1.5-2.1. I read the drop as a likely change in data submission, not as a real trend.
  - Bottom otter trawl on the Danish side of the boundary inside the grid: 51% of grid effort in 2015-2018, 34% in 2021-2024. In 2021-2024 the SW part of the grid (57.8-58.1 N, 10.0-10.5 E) fell from 100-150 to 1-10 MWh per c-square. I found no regulation that explains this (see Q1, Danish side).
- **Layer metadata.** "Fishing effort data are only shown for vessels >12 m having vessel monitoring systems (VMS)". Effort is in "mW fishing hours", by ecoregion and gear. (Fetched full text, WFS GetCapabilities abstract) - [EMODnet Human Activities WFS](https://ows.emodnet-humanactivities.eu/wfs?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetCapabilities)
  - The ICES overview adds: "Fishing effort by vessels < 12 m may be significant, especially in inshore areas ... they are therefore not included". Its national effort figure is "by EU nation", and in the mixed-fisheries section "Norway has not provided effort information since 2016". (Fetched full text) - [ICES Greater North Sea Fisheries Overview 2020](https://www.vliz.be/imisdocs/publications/360597.pdf)
  - I could not confirm whether Norwegian VMS is in the effort maps.

### Inferences
- **Coverage gap.** Norway lands the largest share in 45G0, about 44% of the three-fleet tonnage. If Norwegian vessels are missing from the EMODnet layers, the maps may leave out a third to a half of local effort. They would then understate shrimp-trawl and pelagic effort on the Swedish/Norwegian side more than Danish effort. Treat the absolute MWh values as a lower bound and use only ratios and spatial patterns.
- **Shrimp fishery.** Swedish tonnage density inside Mareld equals the rectangle mean, but effort and value are about 1.75x. That fits a shrimp fishery: high value, low tonnage per hour. The deep-water parts of the grid are a shrimp-trawl ground, not a gadoid or herring hotspot.

## Q3. Crude local F (INFERENCE - all numbers in this section are my calculations)

### Takeaway
Using 45G0 landings per km2 against the model's standing biomass:
- **gadoids C/B = 0.26 per year** (range about 0.2-0.4 after the corrections below);
- **pelagic fish C/B = 0.34** (herring alone 0.30);
- 0.60 if blue whiting and Norway pout are counted as "pelagic fish".

Scaling stock-wide F by relative effort gives local bottom-trawl pressure about **1.5-2x the Skagerrak average**. Pelagic pressure is **about the same as the neighbourhood (0.75-1.4x)**.

Read together: local F on gadoids is probably similar to or somewhat higher than the stock-wide F, not lower. The effort-based factor exaggerates it, because most local bottom-trawl hours are shrimp trawling with fish-excluding grids. Local F on pelagic fish is about the same as stock F. The "restrictive fishery" framing fits management at stock level (TACs, mesh, grids), not a locally light fishery. Uncertainty is at least a factor of 2 either way.

### Cited Findings
- **Cod discards, stock-wide (cod.27.47d20, 2022 advice).** "catches in 2023 should be no more than 26 008 tonnes". The advice projected 2022 discards of "2691" t, "Assuming 2021 discards ratio at age". Discards are therefore about 10% of catch. (Snippet. I confirmed that the ASD page is cod.27.47d20 and contains "26 008"; the 2691 figure comes from the search snippet.) - [ICES ASD advice 3056](https://asd.ices.dk/viewAdvice/3056)
- All other inputs come from the Q2 Cited Findings: 45G0 landings by nation and species, and the EMODnet ratios.

### Inferences: calculation
- Three-fleet landings in 45G0 are 9,891.7 t/yr, or 3.04 t/km2/yr.
- Model gadoids (cod 1.8 + saithe 3.1 + haddock 0.9 + hake 0.1 = 5.9%): 584 t/yr, or **0.18 t/km2/yr**. Against 0.7 t/km2 that is C/B = **0.26**.
- Pelagic narrow (herring 32.5 + mackerel 3.7 + horse mackerel 0.8 + sprat 0.1 = 37.1%): 3,670 t/yr, or **1.13 t/km2/yr**. Against 3.3 t/km2 that is **0.34**.
  - Herring alone: 0.99 t/km2/yr, or 0.30.
- Pelagic wide (adding blue whiting 17.8 and Norway pout 10.5, total 65.4%): 1.99 t/km2/yr, or **0.60**.
  - Blue whiting and Norway pout are gadids by taxonomy but small planktivores by function. They are not in the model's gadoid list. If they were assigned to gadoids, gadoids would be 34.2% of tonnage, or 1.04 t/km2/yr, and C/B would be about 1.5, which does not fit a 0.7 t/km2 standing stock.
- If the model biomass is an annual mean, Baranov gives C = F x B_mean, so C/B is directly an F. If it is a start-of-year biomass, then F = u x Z / (1 - e^-Z) and F is somewhat larger than u.
- Effort route: grid bottom-otter-trawl density / neighbourhood density = 1.5-1.8, or 1.75-2.1 against the Skagerrak polygon. Pelagic = 0.75-1.4.

### Inferences: corrections and why the range is wide
1. **Upward:**
   - Discards are not in the landings. Stock-wide they are about 10% of catch (see Cited Findings). I found no Skagerrak shrimp/Nephrops-specific figure for cod.
   - UK catches are excluded.
   - Whiting may sit in "övrigt" (up to 1.9%). If all of it were whiting, gadoids would be 7.8%, giving C/B 0.34.

2. **Downward / scope:**
   - If Table 9-10's Swedish 2,285 t/yr is correct instead of the text's 3,024.4 t/yr, the three-fleet total falls to 9,152 t/yr. Every C/B above then drops by about 7.5%: gadoids 0.24, pelagic 0.32.
   - The model's 0.7 t/km2 is total standing stock including juveniles. Landings come from fishable sizes, so F on the exploitable part is higher than C/B, while F averaged over all sizes is lower.
   - The 2010-2020 window predates the August 2020 cod measures and the low recent cod TACs, so "today" may be lower for cod.
   - The EIA has no 2021-2024 data. In EMODnet, grid bottom-trawl effort halves from the 2018-2021 to the 2021-2024 window (6.0 to 3.0 MWh/km2/yr), which may be a data artefact.
3. **44G0** (34% of the grid) lands roughly as much as or more than 45G0, mainly with midwater trawls. Including it would probably raise pelagic C/B for the grid.
4. **Migratory pelagics.** Herring and mackerel catches come from seasonal migrants: North Sea autumn spawners and western Baltic spring spawners feeding in the Skagerrak. Local C/B is then a ratio between a local catch and a local mean density, not a population F.

## Q4. Spatial pattern and the choice of impact map

### Takeaway
Bottom-otter-trawl effort covers almost every cell of the grid:
- 2015-2018: no zero c-squares; 63 of 216 c-squares hold 50% of the effort.
- 2021-2024: 35 of 216 hold 50%.
- It thins to zero at Bratten (top rows, NW) and is lower in the SE.

The pattern has shifted. In 2015-2018 effort was broad over the western and southern half. In 2021-2024 it is concentrated in a NNE-SSW band at about 10.6-10.95 E (the Mareld/Poseidon area and the deep shrimp grounds) plus a row at about 58.15-58.2 N.

Pelagic effort is patchy:
- 2015-2018: 16 c-squares hold 50%, 87 cells are zero, concentrated at 58.0-58.15 N, 10.0-10.6 E.
- 2021-2024: 8 c-squares hold 50%, 159 are zero.

Bottom seines are confined to the southern edge (Danish side), and static gears are absent.

For a gadoid impact field, a normalised bottom-otter-trawl map (`emodnet:fishingbottomottertrawls`) is better than a uniform field mainly because of the Bratten zeros and the 2-3x E-W contrast. Even so, a uniform field with Bratten masked is a defensible simplification. For pelagic fish, the pelagic map is patchy and unstable between 4-year windows, and the model's pelagic fish are mobile, so a uniform field (or a multi-window average) is more robust than one window's map.

### Cited Findings
- **Available EMODnet layers.** GetCapabilities lists `emodnet:fishingbottomottertrawls`, `fishingbottomseines`, `fishingbeamtrawls`, `fishingdredges`, `fishingpelagic` and `fishingstaticgears`, plus `fishingsurface` / `fishingsubsurface` (swept-area disturbance by bottom-contacting gear). Titles take the form "Average MW Fishing hours: Bottom otter trawls". (Fetched full text) - [EMODnet WFS capabilities](https://ows.emodnet-humanactivities.eu/wfs?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetCapabilities)
- **Resolution and windows** (Local, from the downloaded features):
  - 0.05 x 0.05 deg c-squares, about 2.9 x 5.6 km or about 16.4 km2 at 58 N.
  - Five averaging windows: 2015-2018, 2017-2020, 2018-2021, 2019-2022 and 2021-2024, with `fo_year` from 2020 to 2025.
  - Attribute `mw_fshn`. Each feature also carries `stat_rc`, the ICES rectangle.
- **Check against existing files** (Local): the largest pelagic 2015-2018 value in a grid c-square is 22.49 MWh/yr, which matches the "22.5" in `coordinates.txt`.
  - `make_pelagisk.py` writes each c-square's value into the overlapping pixels, area-weighted. Pixel values are therefore MWh/yr per c-square, not per km2 as its docstring says. Divide by about 16.4 to get per km2. Only scaling is affected.
- **Bottom otter trawl 2021-2024, MWh per c-square** (Local): along 58.075 N, 6-35 at 10.0-10.5 E, 114-208 at 10.6-10.95 E, 3-46 east of 11.0 E; along 58.175 N, 106-218 almost all the way across.
- **The ICES overview** notes that the "greatest physical disturbance of the seabed in the North Sea occurs ... in the central Skagerrak". (Fetched full text) - [ICES GNS Fisheries Overview 2020](https://www.vliz.be/imisdocs/publications/360597.pdf)

### Inferences
- Bottom-trawl hours mostly track the shrimp grounds (100-500 m), so they are a better map of **benthic disturbance** and of shrimp/Nephrops removal than of gadoid F. Gadoid bycatch per hour in grid-equipped shrimp trawls is low. Weighting the gadoid impact by this layer is a reasonable proxy, but its contrast should be damped, for example with weights^0.5, or blended 50/50 with uniform.
- Any map should zero the Bratten cells: zone 14 at rows 0-5, cols 0-11, plus the small zones at rows 0-5, cols 19-26. Clip with the actual polygons, not bounding boxes.

## Implications for the model's fishing impact
1. **Magnitude.**
   - Gadoids: harvest about 0.2-0.4 per year of the model's gadoid biomass (central about 0.26-0.3), or the stock-wide F from the companion notes times 1-1.5.
   - Pelagic fish: about 0.3 per year of the 3.3 t/km2 (herring, mackerel, sprat, horse mackerel). Go up to 0.6 only if the model's pelagic group is meant to include blue whiting and Norway pout.
   - None of this supports a local F well below the stock F.
2. **Spatial field.**
   - Gadoids: a normalised `fishingbottomottertrawls` field (mean of the 2018-2021 and 2021-2024 windows, damped), or uniform, with Bratten cells set to 0.
   - Pelagic fish: uniform, or the mean of all five `fishingpelagic` windows. Do not use a single window.
3. **Scenarios.** The wind farms will exclude trawling from about 369 km2 (Mareld) plus Poseidon Nord. The EIA expects effort to move to the surroundings (`mkb.txt:8806-8809`). For a "projekt" scenario, set the impact inside the farm polygons to about 0 for towed gear and raise the rest of the grid in proportion. Inside Mareld, Swedish effort is about 20% of the 45G0 total.
4. **Caveats to carry.** VMS covers only vessels over 12 m. Norwegian coverage is unverified. 2010-2020 landings predate the 2020 cod measures. Absolute EMODnet totals jump between windows, so use ratios.

## Source status

| Source | What it gave | Status |
|---|---|---|
| Mareld EIA (`mkb.txt` / `MKB Mareld.pdf`) sections 5.8.1, 5.9.1.1, 9.11, 11.1, 15.2 | Landings by nation, gear and species for 45G0; Swedish effort in Mareld; VMS descriptions; Bratten; national interest areas | Local (full) |
| EIA Fig. 9-21 and 9-22 (PDF pp. 179-180, rendered) | Species labels and gear-by-rectangle bars | Local (image, read visually) |
| EMODnet WFS layers (bottom otter, pelagic, bottom seine, static) | Effort ratios and spatial pattern | Local (data downloaded and computed) |
| EU Reg. 2017/118 | Bratten closures and AIS | Fetched full text |
| EU Reg. 2019/1241, Annex V B | Skagerrak mesh and grid rules | Fetched full text (2019 version) |
| SWE-DNK 1984 treaty | Zone boundary points A-C | Fetched full text |
| ICES Greater North Sea Fisheries Overview 2020 | VMS caveat; Norway effort data missing (mixed fisheries) | Fetched full text |
| Regeringen June 2025 and Sept 2026 | MPA bottom-trawl ban from 1 July 2026; Baltic-only move of the trawl limit | Fetched full text |
| SU Q&A 2025 | Trawl limit 4/3 nm in the Skagerrak; inflyttningsområden | Fetched full text |
| Danish Fisheries Agency RTC page | Real-time closure rules | Fetched full text |
| EP Observatory (Skagerrak Agreement) | Reciprocal access beyond 4 nm | Snippet |
| The Fishing Daily; DTU Aqua 380-2020 | 2020 cod measures; Danish cod plan and SELTRA 300 | Snippet |
| HVMFS 2015:7; HaV konsekvensutredning 2026 | Shrimp trawl rules; Koster exceptions | Snippet |
| eu.dk ministerial answer 2023 | Skagens Gren og Skagerrak without trawl ban | Snippet |
| ICES ASD cod advice 2022 | Discards about 10% of catch, stock-wide | Snippet |
| Rectangle landings 2021-2024; Skagerrak cod discards; Norwegian VMS in ICES maps; EIA appendix C11 | Not found / not accessible | Gap |
