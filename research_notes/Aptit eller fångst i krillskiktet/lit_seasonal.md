# Seasonal ration and krill intake of whiting, Norway pout and saithe: literature trail (2026-11-20)

This note is literature and data only. No model was run, and no value was chosen from model output.
Nothing outside this folder was changed.

**Questions.** Does the low krill intake of field gadoids hold outside winter? Does the field ration
show a seasonal appetite cycle that is independent of food?

**Tags.**
- [F] full text read.
- [A] abstract only.
- [S] secondary: a review or another note in `research_notes/`.
- [D] derived here (arithmetic).
- [J] judgement.

Proxies are marked **PROXY**.

**Files.**
- `scripts/sea_quarterly_rations.py`: all arithmetic, with stdout in `tables/sea_quarterly_rations_stdout.txt`.
  It also writes `tables/seasonal_raw.csv`, one row per sourced or derived number.
- `scripts/sea_openalex_abstract.py`: prints OpenAlex abstracts.
- Downloads are in `data/<source>/`. They are untrusted and not to be committed. They were read only with
  pdftotext and `python3 -I`.

## 0. Conventions (reused, not redone)

**GER ration.** The model is ICES WGSAM 2024 Table 3-7, as transcribed in
`Krillparen mot vintermagar/evidence.md` E3 [S]. Whiting and saithe share the parameters (rho 0.00171,
lambda 1.35, delta 0.081, E^-0.85, alpha 0.5):

    R = rho e^(dT) L^1.35 E^-0.85 (N/F)^-0.5 S^0.5   (g/h)

- S is the mean content over all stomachs, empties included.
- F/N is the share of non-empty stomachs.

**GER Cmax.** This is the same expression with every fish fed and S_max = 2.5 %BW (central) or 5 %BW
(low band). That is the convention of `Fiskgrupper i rutan/predators/evidence_A_physiology.md` 1a and
of `Krillintag och jaktsätt` [S].

**Ration / Cmax = sqrt((F/N)(S/W)/S_max)** [D]. Temperature, prey energy density and the GER constants
cancel when ration and Cmax are taken at the same temperature and diet. So this ratio is the
season's ration as a fraction of Cmax **at the season's temperature**.

**Absolute rations** (%BW/d) use:
- the North Sea temperatures 6 / 7 / 10 / 8 C for Q1-Q4, which Greenstreet et al. 1997 [F] took from
  Daan 1989;
- E = 5.0 kJ/g [J], with 4.2 as a sensitivity (+16 % ration).

**Pooled means are upper-side** [J]: sqrt(mean S) >= mean(sqrt S).

## 1. Sources

### W1. Robb AP, Bell MA, MacMillan J, Hislop JRG (1994). Analysis of the whiting stomachs collected in the North Sea during the 1991 ICES stomach sampling project. ICES CM 1994/G:42. [F]

- URL: https://www.vliz.be/imisdocs/publications/ocrd/275252.pdf (scanned with an OCR layer).
- Files: `data/vliz275252/doc.pdf`; text in `r.txt` (raw) and `t.txt` (layout). Table columns were
  assigned from `t.txt` positions.
- OCR letter errors in the quotes below were corrected and nothing else.

**Sample.** About 39,000 stomachs, whole North Sea, all four quarters of 1991.

**Method.** Mean content "WS = W/F x (F+R)/(F+R+E)":
- regurgitated fish are given the valid fishes' mean;
- "Fish with empty stomachs have been included in the calculation" (Table 5).

**Quotes:**
- "There was a marked seasonal difference in the proportions of the whiting stomachs classified as empty in
  1991. Empty stomachs were encountered far more frequently in Quarter 1 than at other times of year. This
  was also the case for the cod, haddock, saithe and mackerel sampled in 1991 (Anon, 1993)."
- "All the research vessels involved in the sampling recorded a large percentage of empty stomachs in
  Quarter 1. At other times of year, although there was no consistent seasonal pattern, the percentages of
  stomachs assigned to the 'empty' and 'regurgitated' categories appeared to vary according to which
  vessel had collected the samples."
- "This could indicate that the low values of mean stomach content weight estimated for Quarter 1 were not
  a consequence of the subjective judgement of the stomach sampler but were a real biological phenomenon."
- Prey by season:
  - "Sandeels and Norway pout were significant components of the diet at all times but the former were
    more important in Quarter 2, whereas Norway pout were the dominant fish prey in Quarter 4."
  - "Haddock were only prominent in Quarters 3 and 4 when the majority of the fish eaten were 0-group
    stages."
  - herring "were most noticeable in Quarter 3".
  - "(most noticeably in Quarter 2) annelids".
- "a large proportion of the crustaceans eaten in the southern North Sea were Crangonids, whereas
  Euphausiacea were predominant in the northern areas."
- Q1 1991 empties by area:
  - Table 2 (all sizes): Q1 45 % empty, against 12 / 12 / 8 % in Q2-Q4.
  - Table 3, whiting 200-350 mm in Q1:
    - Roundfish Areas 1-3 (northern): 52-73 % empty;
    - Areas 5-7 (southern): 14-53 % empty.
  - Area totals Q1: 54.2, 42.3, 46.5, 52.6, 37.5, 43.1, 29.5 % for Areas 1-7.
  - "Northern" and "southern" follow Hislop et al. 1983 (W2): "southern North Sea (RRAs 5, 6 and 7)" and
    "northern North Sea (RRAs 1, 2 and 3)".
- **The one in-area whiting row.** Figure 1 (page image checked) puts Roundfish Area 8 in the Skagerrak
  and Area 9 in the Kattegat.
  - Area 8 was sampled in Q4 1991 only: 1045 stomachs, 18.3 % empty, against 7.9 % for the North Sea in
    Q4 (Tables 1-3) [F].
  - The 200-300 mm classes there were 12.4-17.6 % empty.
  - Table 5 gives no area-level stomach weights, so no Skagerrak ration follows.

**Transcribed** [F]: Table 5 (L, W, S by size class and quarter, 100-400 mm) and the Table 3 Total rows
(% empty by size).
- The 200 mm Q1 S is printed "-0.83"; it is read as 0.83 [J].
- Rows 50-80 mm are incomplete after OCR and were not used.

**Derived** [D] (`tables/sea_quarterly_rations_stdout.txt`, section 1). Whiting 200-400 mm, ration / GER
Cmax at S_max 2.5 % (5 % in parentheses):

| | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| ratio, adults 200-400 mm | 0.42 (0.30) | 0.83 (0.59) | 0.83 (0.58) | 0.78 (0.55) |
| ratio, juveniles 100-150 mm | 0.51 | 0.76 | 0.92 | 0.75 |
| ration %BW/d, adults (6/7/10/8 C, E 5) | 0.71 | 1.58 | 1.97 | 1.54 |
| S/W incl. empties, adults | 0.89 % | 1.94 % | 2.02 % | 1.61 % |
| empty, adults | 50 % | 11 % | 14 % | 6 % |

Q1 relative to the mean of Q2-Q4, by how the stomachs are classified [D]:

| convention | adults | juveniles |
|---|---|---|
| A, Robb's | 0.52 | 0.63 |
| B, regurgitated counted as empty | 0.60 | - |
| C, the empty/regurgitated split as an artefact | 0.92 | - |

- Under C, every quarter has the same fed share and only the content per fed fish differs (Q1 S_fed/W
  1.76 % against 1.71-2.37 %).
- Empty plus regurgitated, all sizes, is 61 / 46 / 58 / 50 % in Q1-Q4. So most of the Q1 dip sits in
  the empty versus regurgitated call.
- The authors judge the Q1 empties real.

### W2. Hislop JRG, Robb AP, Brown MA, Armstrong D (1983). A preliminary report on the analysis of the whiting stomachs collected during the 1981 North Sea stomach sampling project. ICES CM 1983/G:59. [F]

- URL: https://www.ices.dk/sites/pub/CM%20Doccuments/1983/G/1983_G59.pdf.
- Files: `data/ices_cm1983_g59/` (`t.txt` layout, `r.txt` raw).

**Quotes:**
- "There were no obvious between-quarter differences in the weights of food in the stomachs."
- "The two main components of the crustacean prey were euphausiids and crangonids. Crangonids were
  relatively more important than euphausiids in the youngest whiting age groups but less important in the
  older fish."
- "appreciable quantities of squid were found in some of the quarter 1 samples."
- Rations by Daan 1973 (rectilinear), "a digestion time of 2.5 days was applied". Jones 1974 "gave values
  which were approximately 1.5 times greater".

- "The diet of whiting in the southern North Sea (RRAs 5, 6 and 7) included a greater proportion of
  annelids, whilst molluscs (mainly cephalopods) were relatively important in the northern North Sea (RRAs
  1, 2 and 3) in the first quarter of the year."

**Table 1** (per quarter and Roundfish Area).
- The page image was checked. The headers print "Total Full Reg Empty", and Area 1 Q1 "Full" is 528
  (the OCR gave "5213").
- Each area's TOTAL row sums.
- **The printed headers conflict with two statements** [D/J]. Over the year, the column printed "Empty"
  sums to 9817 and "Full" + "Reg" to 9400 (49 %).
  - Hislop et al. 1991 (W3): "Almost 50% were either empty or ... regurgitation. The contents of the
    remaining stomachs (approximately 9800) were analysed". That fits the printed "Empty" column being the
    analysed (fed) stomachs.
  - Robb et al. 1994 (W1): Q1 1991 empties (45 %) were "considerably higher than the equivalent figure for
    1981", and Q2-Q4 "similar to or lower".
- **Reading B** swaps the printed "Full" and "Empty" headers. It fits both statements; the printed
  reading fits neither [J].

Summed over Areas 1-7 [D]:

| quarter | N | empty, reading B (col. 2) | col. printed "Empty" | 1991 empty (Robb Table 2) |
|---|---|---|---|---|
| Q1 | 7832 | 25.8 % | 47.4 % | 45 % |
| Q2 | 4211 | 21.6 % | 45.2 % | 12 % |
| Q3 | 3727 | 26.4 % | 58.0 % | 12 % |
| Q4 | 3447 | 21.1 % | 59.3 % | 8 % |

- **Under either reading, 1981 has no Q1 excess.** Under reading B, 1981 empties are 21-26 % in every
  quarter.
- The 1991 Q1 value is then about 1.7x the 1981 Q1 value, as Robb says.

**Table 2** (euphausiids by quarter and age group): the OCR is garbled in both extraction modes. No
numbers were taken. The whiting euphausiid content by quarter is in the ICES stomach DB, which the main
agent processes.

### W3. Hislop JRG, Robb AP, Bell MA, Armstrong DW (1991). The diet and food consumption of whiting (*Merlangius merlangus*) in the North Sea. ICES J Mar Sci 48:139-156. doi:10.1093/icesjms/48.2.139. [A]

- Paywalled: Unpaywall reports it not OA, and a direct OUP download was refused by the permission
  system. The abstract comes from OpenAlex (`data/openalex_hislop1991/record.json`, printed by `sea_openalex_abstract.py`).
- Quotes:
  - "The stomachs of more than 19000 North Sea whiting were examined in 1981. Almost 50% were either empty
    or were judged to have lost all or part of their contents through regurgitation."
  - "S=0.009 x W1.057".
  - "Some geographical and seasonal differences in stomach contents were detected; annelids represented a
    significant proportion of the food of whiting in the southeastern North Sea, as did cephalopod
    molluscs in the north, during the first part of the year."
  - "attempts have been made to estimate the annual food consumption".
- The quarterly rations in the paper were not seen. Greenstreet 1997 cites the paper for diets.
- Derived [D]: the 1981 S/W from this fit is 1.17-1.24 % at 95-278 g (all quarters). Robb's 1991 annual
  fit gives 1.85-2.19 %.

### W4. ICES WGSAM 2024, ICES Sci. Rep. 6:13, Table 3-8, and the SMS key-run input `NS2023_consum_ab.in`. doi:10.17895/ices.pub.25020968. [F data, D]

- Local copy: `Fiskgrupper i rutan/predators/data/diet/sms_consumption/`.
- Quotes:
  - "The estimated rations by individual strata (year, quarter, predator and predator size class used in
    sampling) are combined into one equation for ration from mean weight (ration=a*W^b) where 'a' and 'b'
    are dependent on quarter (Table 3-8)."
  - "Food rations were estimated as part of the 2020 key run".
- The SMS temperatures are not given in the text read [J]. The GER Cmax here uses the Daan temperatures.

Derived [D], section 2 of stdout:

| species | statistic | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| whiting 0.2 kg (L 29.5 cm) | ration %BW/d | 0.78 | 0.83 | 1.24 | 1.05 |
| | ration / GER Cmax | 0.43 | 0.42 | 0.49 | 0.49 |
| saithe 1.066 kg (L 50 cm) | ration %BW/d | 0.43 | 1.25 | 0.66 | 0.78 |
| | ration / GER Cmax | 0.27 | 0.71 | 0.30 | 0.41 |

- Whiting:
  - S_max 5 % gives 0.30-0.35.
  - Normalised to 6 C by the GER temperature term: 0.78 / 0.77 / 0.90 / 0.89. **The whiting ration
    follows temperature**: Q1 and Q2 are 0.87 of Q3 and Q4.
- Saithe, normalised: 0.43 / 1.15 / 0.48 / 0.66. **Q2 is about 2.5x Q1 and Q3**, which the temperature
  does not explain.
- The brief's saithe values (0.44 / 1.27 / 0.67 / 0.79) differ by about 2 % from these (91.25 d per
  quarter here).
- Caveats [J]:
  - The fit pools sampling years and sizes. It cannot separate 40-cm immature saithe from spawners.
  - The SMS saithe stomachs are few (547-1241 per quarter in 1981/1991; ICES CM 1992/G:12 Table 4.1
    [F]).

### W5. Greenstreet SPR, Bryant AD, Broekhuizen N, Hall SJ, Heath MR (1997). Seasonal variation in the consumption of food by fish in the North Sea and implications for food web dynamics. ICES J Mar Sci 54:243-266. [F] **PROXY (guild level)**

- Local copy: `Fiskgrupper i rutan/planktivores/data/greenstreet1997/` (`t.txt` layout, Tables 1 and 3).
- The DOI was not verified (OpenAlex rate-limited). URL:
  academic.oup.com/icesjms/article-pdf/54/2/243/1715685/54-2-243.pdf.

**Quotes:**
- "Sea temperature was assumed to be 6°C in quarter 1, 7°C in quarter 2, 10°C in quarter 3, and 8°C in
  quarter 4 (Daan, 1989)."
- "Daily rations were determined by applying Jones' (1974) digestion model to stomach weight data given
  in Daan (1989)" (demersal piscivores: "cod, haddock, whiting and saithe, making up between 75% and 80%
  of the total biomass").
- Pout: "Jones' (1984) digestion model was applied to the Norway pout mean stomach content wet weight
  values".
- "Linear change in biomass between quarters 1 and 4 was assumed to estimate the biomass in quarters 2
  and 3."
- "Carnivore biomass was estimated from the CPR counts of 'total euphausiids' and 'total hyperiids'";
  "our zooplankton carnivore group also includes hyperiid amphipods as well as euphausiids."
- "food consumption is highest in the warmer summer months when metabolic rates are raised ..., more food
  is available, and fish populations are augmented by the presence of young fish".
- Appendix II puts *Meganyctiphanes*, *Thysanoessa*, but also Pasiphaea, *Euchaeta*, cephalopods,
  chaetognaths, cnidarians and ctenophores in "meso-zooplankton carnivore".

**Derived** [D] (Table 3 consumption / Table 1 biomass):

| guild | statistic | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| demersal piscivores | ration %BW/d | 1.11 | 1.33 | 1.80 | 1.37 |
| | normalised to 6 C | 1.11 | 1.22 | 1.30 | 1.16 |
| | meso-zoo carnivore share | 0.166 | 0.219 | 0.162 | 0.116 |
| | meso-zoo carnivore intake, %BW/d | 0.19 | 0.29 | 0.29 | 0.16 |
| pelagic planktivores (pout, herring, sprat, sandeel) | ration %BW/d | 1.09 | 1.66 | 1.67 | 1.18 |
| | meso-zoo carnivore share | 0.43 | 0.36 | 0.35 | 0.34 |

- **CPR carnivore biomass** [F], North Sea: 5.44 / 9.48 / 3.31 / 4.75 g WW/m2. Relative to the annual
  mean: 0.95 / 1.65 / 0.58 / 0.83.
  - The CPR samples near-surface water [J]. It does not measure the deep adult *M. norvegica* layer.
- **The krill season** [F]:
  - "Euphausiids reproduce in the late winter-early spring following which most of the mature adults die
    (Lindley, 1978, 1982a; Mauchline, 1980, 1984). The high production rates in quarter 2 can therefore be
    attributed to large numbers of young in the population, while the population decline is associated
    with the loss of the adults that make up the bulk of the population in late winter."
  - Growth: "estimates of daily specific growth rates as high as 0.40 are obtained in quarter 3 for
    euphausiids in their earliest stages of development. Daily specific growth in the older generation
    euphausiids is much slower at around 0.02 to 0.04 in quarters 2, 3, and 4" (after Lindley 1982a).
  - [J] Large adult krill, the size a whiting or saithe takes, are most abundant in late winter (Q1). In
    Q2-Q3 the stock is young and small. *Thysanoessa* and *Meganyctiphanes* are not separated in these
    sources; the Thysanoessa/Meganyctiphanes item is not resolved here.

### W6. Greenstreet SPR (1996). Estimation of the daily consumption of food by fish in the North Sea in each quarter of the year. Scottish Fisheries Research Report 55. **Not obtained.**

- The report exists: FishBase refs 12223 and 42340; marineinfo gives 1995, ISBN 0-7480-5141-4, "16
  pages plus 70 tables".
- No online full text was found. It holds the species-level quarterly rations behind W5.

### W7. Pedersen J (2000). Food consumption and daily feeding periodicity: comparison between pelagic and demersal whiting in the North Sea. J Fish Biol 57:402-416. doi:10.1111/j.1095-8649.2000.tb02180.x. [A]

- Unpaywall reports it not OA. The abstract is the marineinfo copy in
  `Vitlingens fiskbyten/data/lit/pedersen_19426.txt` [S].
- Quotes:
  - "Using Anderson's evacuation model the specific daily ration ranged from 0.41 to 1.66% in 1992 and
    from 0.78 to 1.75% in 1993."
  - "Pelagic North Sea whiting Merlangius merlangus fed at night, while demersal whiting fed by day."
- The months are not in the abstract. The range cannot be assigned to seasons.
- Pedersen 1999, J Fish Biol 55:1096, doi:10.1111/j.1095-8649.1999.tb00744.x [A, same folder]:
  "Pelagic North Sea whiting ... preyed upon (pelagic) sprat and sandeel, while demersal whiting preyed
  upon (demersal) Norway pout and herring". No season was given.

### W8. Kikkert AH (1993). Analysis of the cod samples collected in the North Sea during the 1991 international stomach sampling project. ICES CM 1993/G:13. [F] **PROXY (cod)**

- URL: https://www.vliz.be/imisdocs/publications/ocrd/275167.pdf. Files: `data/vliz275167/`.
- Quotes:
  - "Only in the first quarter is a higher percentage (circa 15%) found for larger cod as well."
  - "The lowest feeding coefficient was found in the first quarter. This might be due to a reduced food
    intake during the spawning period (Arntz, 1974; Rae, 1967); the results presented in Figure 4 show
    that the mean percentage of empty stomachs in the first quarter in the range of mature cod exceeds the
    values for the other quarters."
- Table 7, phi (S = phi L^3, b fixed 3), x 1e-5: 15.41 / 22.69 / 19.02 / 16.42 [F]. Q1 / mean(Q2-Q4) =
  0.80 [D].

### W9. Dorman J, Johnsson B, Larsson P-O (1992). Food and feeding of cod in the Skagerrak and the Kattegat in 1991. ICES CM 1992/G:70. [F] **PROXY (cod), in-area Q1**

- URL: https://www.vliz.be/imisdocs/publications/ocrd/274761.pdf. Files: `data/vliz274761/`.
- Quotes:
  - "In total 336 cod were sampled in the Skagerrak, 653 in the Kattegat and 166 in The North Sea. Of
    those 87,5, 89,7 and 91 % respectively had stomach content." (Q1 1991)
  - "in the Kattegat euphasians (krill) dominate (63% of the crustaceans ...)".
  - "in February 1981 cod had eaten insignificant amounts of krill in the same area, (only about 2% of
    total weight of crustaceans compared to our 63%)".
- [J] In-area Q1 1991 cod were mostly feeding. Krill in the Kattegat in Q1 varies strongly between years.

### W10. ICES (1992). Progress report on the ICES 1991 North Sea stomach sampling project. ICES CM 1992/G:12. [F]

- URL: https://www.vliz.be/imisdocs/publications/ocrd/278195.pdf. Files: `data/vliz278195/`.
- Saithe quote: "In 1991, the principal fish prey were Norway pout and herring. The stomachs also
  contained large amounts of Mysids."
- Table 4.1: saithe stomachs per quarter, 1981 / 1991: 547 / 1115, 185 / 1241, 899 / 708, 559 / 812.
- There is no seasonal ration for saithe. No 1991 saithe analysis paper (an analogue of W1) was found.

### P1. Albert OT (1995). Diel changes in food and feeding of small gadoids on a coastal bank. ICES J Mar Sci 52:873-885. [F]

- Local copy: `Fiskgrupper i rutan/planktivores/data/albert1995/`.
- Sampling: "Haddock and Norway pout were sampled at 140 m in July"; 59°21'N 5°05'E, SW Norway.
- Quotes:
  - "For Norway pout, daily wet weight ration was estimated as 1.1% of body weight. For comparison,
    Jones' method (Jones, 1974) gave daily wet weight ration of 3.3% and 1.1% of body weight for haddock
    and smaller Norway pout, respectively."
  - "Temperature used was 7°C".
  - "The small Norway pout were feeding on copepods, mostly in the middle of the day".
- Table 4 (July), Norway pout: Euphausiacea 3.0 %W (3.3 %F). Calanoida 6.6 %W plus *C. finmarchicus*
  10.1 %W. *Pandalus borealis* 21.9 %W.
- Discussion: "Albert (1994) found in the Norwegian Deep ... that Norway pout were feeding mostly at
  night. The diet was then dominated by euphausiids".
- [D] **PROXY** (whiting GER, L 13 cm, W 22 g, E 4.0, 7 C): 1.1 %BW/d is 0.46 of GER Cmax (0.33 at S_max
  5 %).

### P2. Albert OT (1994). Biology and ecology of Norway pout in the Norwegian Deep. ICES J Mar Sci 51:45-61. doi:10.1006/jmsc.1994.1005. [S]

- Read in full in `Planktivorernas vinterkostnad/evidence.md`. The local scan has no text layer.
- Winter cruises 31 Jan - 13 Apr 1985-87.
- Empties (< 14 / >= 14 cm): 33.6 / 24.5 % in winter, 27.8 / 18.6 % in summer.
- "The mean dry weight of the stomach contents relative to fish weight (including empty stomachs) was
  consistently higher in winter than in summer (Fig. 10)".
- Winter euphausiids 78.6 / 81.4 %W.
- "a pronounced reduction in condition factor and liver weight during the winter season".
- Spawning mainly in March-April.
- Content 0.36-1.3 %BW wet (sibling [D]). This is the range of Fig. 10, which compares winter with
  summer, so the range is **not winter-only**: the season cannot be separated.
- [D] **PROXY**: 0.31-0.65 of whiting-GER Cmax (empties 18.6-33.6 %, both seasons).
- Do not set this against July 0.46 (P1) as a seasonal contrast.

### P3. Raitt DFS (1968). Synopsis of biological data on the Norway pout. FAO Fish. Synop. 33 Rev. 1. [F]

- Local copy: `Fiskgrupper i rutan/planktivores/data/fao_raitt1968_pout/r.txt` l. 733-745, 899-904.
- Quotes:
  - "Data on seasonal variations in feeding, although inconclusive, suggested that in the adult fish
    there may be three periods of low feeding intensity, in January, March-April (the main spawning
    period) and August. Apart from a slight drop in March, juvenile fish sampled from September to April
    showed no marked variations in feeding intensity."
  - Ursin 1963: "Fish were shown to be losing in condition during the winter but seemed to gain rapidly
    in condition in the spring."

### K1. Onsrud MSR et al. (2004). ICES J Mar Sci 61:1278. doi:10.1016/j.icesjms.2004.09.005. [S]

From `Krillparen mot vintermagar/evidence.md` E1 (Oslofjord, 120 m; whiting and pout in the krill layer
day and night):

| | whiting | pout |
|---|---|---|
| empties, day / night, Nov | 25.0 / 14.8 % | - |
| empties, day / night, Mar | 8.2 / 6.3 % | - |
| krill ration, Nov (%BW/d) | 0.23-0.54 | 0.36-0.82 |
| krill ration, Mar (%BW/d) | 0.65-1.51 | 0.51-1.14 |
| krill share | ~82 %W | - |

- The sibling's GER at 7 C on 0.15-0.30 g krill gives whiting 1.8-2.4 %BW/d at 6-8.5 C.
- So the whiting krill ration is ~0.1-0.3 of Cmax in November and ~0.3-0.8 in March [D, rough].

### K2. ICES stomach DB krill rations (the main agent's, from the brief and `Krillintag och jaktsätt` [S])

Whiting krill ration / GER Cmax (pooled 1980-91 records):

| area | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| NE North Sea | 0.31 | 0.00 | - | 0.007 |
| NW North Sea | 0.28 | 0 | 0.012 | 0.08 |
| Skagerrak | 0.06 | 0.37 | - | - |
| Kattegat | - | 0.05 | - | - |

- The Skagerrak Q2 cell is April 1991 (106 fish). Euphausiids are 0.537 of that diet
  (`Vitlingens ranson`).
- Saithe krill ration by area and quarter: 0.04-0.55 of Cmax. NE North Sea Q3 krill mass share: 0.31-0.58
  (`Krill över grunda bottnar`).
- [J] These cells pool 1981 and 1991. **Do not combine them with the 1991-only totals of W1** as if they
  came from one sample.

### K3. Krill presence by season (in-repo [S])

- `Krill i Skagerrak/summary.md`: quarterly stock factors 1.0 / 0.6 / 1.1 / 1.3, tagged there as [J] on
  [F] inputs.
  - The Q2 low is the post-spawning loss before recruits grow.
  - The Kattegat individual dry weight rises 2.5-3x from March to November.
- `Krill över grunda bottnar/summary.md`: daytime near-bed tows by RV Dana HERAS, June-July 2021-25.
  - 3 of 4 on 100-170 m beds took krill (733, 183, 54, 0 kg).
  - 0 of 14 on beds < 100 m took krill.
- W5 CPR (near-surface): Q2 peak, Q3 low (above).

### Q. Saithe, secondary

- **Mathers, Houlihan & Cunningham 1992**, MEPS 86:31, doi:10.3354/meps086031 [S, via
  `Mättnad hos rovfisk/lit_fill_series.md` 1.1]. Beryl oilfield, northern North Sea, ages 2+-5+ (~42 cm).
  - The gall-bladder index (GLI), with K and CSI:

    | | Aug 1988 | Oct 1988 | Apr 1989 | Aug 1989 |
    |---|---|---|---|---|
    | GLI | ~0.3 | ~0.3 | 16.5 | 3.2-3.5 |
    | K | 1.13 | 1.09 | 0.87 | 1.12 |

  - "GLI increased in winter because bile accumulated as feeding was reduced".
  - RNA growth was negative in April.
  - Maturity is not reported [S]. At ~42 cm they are mostly immature [J].
- **Spawning** in February-March on the shelf edge (`Fiskgrupper i rutan/predators/evidence_B_population.md`
  section 4 [S]).
- **Bergstad 1991** (Sarsia 75:269-325, Norwegian Deep diets by season):
  - paywalled; the abstracts are in `Vattenpelarens bottenfiskbyte/evidence_literature.md` A9 [A];
  - the strongest links are "small saithe and blue whiting as predators and the euphausids as prey".
  - Not re-chased.
- **Du Buit 1991** (Fish Res 12:307) and **Homrum et al. 2012** are logged closed there. Not re-chased.

## 2. Answers

Every answer below is [J] built on the tagged numbers above.

### Q1-Q2: season by species

How to read the whiting rows [J]:
- The **level** is SMS. It is a composite of sampling years and conditional on the Daan temperatures.
- The **seasonal shape in one year** is Robb 1991. Its level is upper-side (sqrt of a pooled mean) and
  1.5-2x SMS in Q2-Q4.

| species | season | ration %BW/d | ration / GER Cmax | krill | krill presence |
|---|---|---|---|---|---|
| whiting | Q1 (6 C) | 0.78 | SMS 0.43; 1991 shape 0.5-0.6 of Q2-Q4 (Robb, adults); 1981 with no Q1 excess | 0.28-0.31 of Cmax NE/NW North Sea; 0.06 Skagerrak; Oslofjord Mar 0.65-1.5 %BW/d | adults at their seasonal maximum (W5); Skagerrak factor 1.0 |
| | Q2 (7 C) | 0.83 | SMS 0.42 | 0 NE/NW North Sea; 0.37 Skagerrak (Apr 1991); 0.05 Kattegat | young of the year (W5); factor 0.6 |
| | Q3 (10 C) | 1.24 | SMS 0.49 | 0.012 | near-bed on 100-170 m beds in Jun-Jul; factor 1.1 |
| | Q4 (8 C) | 1.05 | SMS 0.49 | 0.007-0.08; Oslofjord Nov 0.23-0.54 %BW/d | factor 1.3 |
| saithe 1.07 kg (SMS) | Q1 | 0.43 | 0.27 | DB range 0.04-0.55 of Cmax by area and quarter | as whiting |
| | Q2 | 1.25 | 0.71 | | |
| | Q3 | 0.66 | 0.30 | NE North Sea mass share 0.31-0.58 | |
| | Q4 | 0.78 | 0.41 | | |
| Norway pout | Deep, winter and summer together | ~0.4-1.3 (one turnover per day, sibling J) | 0.31-0.65 (proxy) | winter ~80 %W | |
| | coastal bank, July | 1.1 | 0.46 (proxy) | 3 %W | |
| | Oslofjord, Nov / Mar | krill alone 0.36-0.82 / 0.51-1.14 | - | in the krill layer | |

Krill stock factors are from `Krill i Skagerrak` [S/J].

What whiting eat in place of krill in Q2-Q3 (W1 [F]):
- sandeels (Q2);
- annelids (Q2, south);
- herring (Q3);
- 0-group haddock (Q3-Q4);
- Norway pout (Q4).

**Was krill there?** Krill is not absent in Q2-Q3: it was near the bed on 100-170 m beds in June-July
(K3). But adult krill peak in late winter, and the summer stock is young and small (W5). No krill density
sits with any of these stomach series.

**Verdict on low krill intake [J].** For North Sea whiting it is not winter-only. Outside Q1 it is lower
still (0-0.08 of Cmax), while fish prey dominate.
- The exception is the Skagerrak in April 1991 (0.37), the one in-area spring cell.
- At guild level (W5, 1981 diets) carnivorous zooplankton intake is highest in Q2-Q3. That likely reflects
  saithe and haddock, not whiting [J].
- For pout, krill dominates in the Deep in winter and is minor in July on a coastal bank. Those are
  different sites.

### Q3: ration / Cmax by season

- **Statistic** [D]: the GER ratio is temperature-free.
- **SMS whiting** sits flat at 0.42-0.49.
- **SMS saithe:** 0.27 (Q1) to 0.71 (Q2).
- The Q2-Q4 1991 ratios of 0.78-0.83 (Robb) are upper-side. Use them for shape only.

### Q4: an appetite cycle independent of food?

**Not shown** [J]. No source pairs a low-ration quarter with prey that was demonstrably present and a
temperature that was not lower.

**Candidates:**
1. **Whiting Q1 1991** (adults, northern North Sea).
   - The temperature-free ratio is 0.52-0.60 of Q2-Q4 under Robb's convention (0.92 if the empty calls
     were an artefact).
   - It is not a recurring cycle:
     - in 1981 the Q1 empties were no higher than in other quarters under either reading of Hislop 1983
       Table 1 (reading B: 21-26 % every quarter);
     - the 1981 report found "no obvious between-quarter differences" in content (W2);
     - the SMS composite ration follows temperature (normalised Q1 0.78 against Q3 0.90).
   - Within 1991:
     - the Q1 empties concentrate in the 200-350 mm classes (53-57 %, against 23-36 % at 100-150 mm and
       32 % at 400 mm);
     - they were higher in the northern areas (1-3) than the southern ones (5-7);
     - [J] the southern North Sea is not warmer in Q1, so the north-south pattern does not point to
       temperature.
   - Q1 is when the large adult krill peak (W5), but krill in Q1 1991 is not shown.
   - **Test for main:** split the DB krill cells into 1981 Q1 and 1991 Q1.
2. **Saithe Q1** (SMS; Mathers April GLI).
   - This is the spawning season. In the ~42 cm Beryl fish, mostly immature [J], it could be a winter
     cycle.
   - No prey was measured at Beryl, and SMS has thin data.
   - SMS Q3 (0.30) is also low although warmest. Unexplained, data-thin [J].
3. **Pout** (Raitt: January, March-April spawning, August).
   - The author calls it "inconclusive".
   - Albert 1994 found the winter content higher than the summer content: no winter appetite drop.

**Pattern** [J]: where a seasonal low appears, it lines up with spawning or maturation (cod W8 by the
author's reading; saithe; pout), or with a year-specific Q1 (whiting 1991). Otherwise the ration follows
temperature (SMS whiting; the W5 guild is flat within x1.2 after normalising) and prey type.

**For main:**
- the 1981 against 1991 Q1 split above;
- Hislop 1983 Table 2 (euphausiids by quarter and age) is unreadable OCR;
- the quarterly rations in Hislop 1991 and Greenstreet 1996 were not obtained.

## 3. Search log

- WebSearch, Greenstreet 1996 SFRR 55: the reference exists (FishBase, marineinfo); no full text.
- WebSearch, Hislop 1991:
  - FishBase has no abstract;
  - OUP is paywalled, and the download was refused by the permission system (not retried);
  - the abstract came via OpenAlex (`data/openalex_hislop1991/`).
  - The search found the VLIZ copies of ICES CM 1983/G:59 and 1994/G:42 (W2, W1).
- Unpaywall (`data/upw1`, `data/upw2`):
  - Hislop 1991, Pedersen 1999, Pedersen 2000, Patterson 1985 (Irish Sea whiting,
    doi:10.1093/icesjms/42.2.152) and Gordon 1977 (W Scotland whiting, doi:10.1111/j.1095-8649.1977.tb05709.x)
    are not OA.
  - Albert 1994 and Sparholt 2002 are OA at OUP, but the PDF returned 403. Albert 1994 is already read via
    the sibling note.
- WebSearch, "Pedersen ... 1992 1993 cruise ... month": no full text or months found.
- WebSearch, Norway pout daily ration by quarter: nothing beyond Albert 1995. The core.ac.uk hits returned
  403 or were unidentified.
- WebSearch, saithe seasonal feeding intensity: Fujii 2016 (platform saithe, multi-season stomachs) was
  located but not read. No North Sea saithe ration by season was found besides SMS.
- WebSearch, the 1991 saithe stomach analysis paper: not found.
  - Found and read instead: ICES CM 1992/G:12 (W10) and the cod papers W8 and W9.
  - Also downloaded and not used, then deleted: ICES CM 1989/G:41 (planning group) and 1993/G:15 (rays).
  - ICES CRR 219 (the 1991 database report) was not pursued, because its content is the ICES stomach DB
    that the main agent processes.
- WebSearch, Kattegat/Skagerrak whiting seasonal diet: nothing in-area beyond the ICES DB and Robb's
  Area 8 row.
- WebSearch, sandeel overwintering: snippets only (doi:10.1016/j.fishres.2004.09.002 returned 403). The
  claim that sandeels are buried and unavailable in winter is left as [J], not verified.
- OpenAlex:
  - the title searches for the Greenstreet 1997 and Albert 1995 DOIs were rate-limited, so their DOIs are
    not given (URLs instead);
  - the error bodies were read once with an inline `python3 -I -c` and then deleted;
  - every other download was read only with pdftotext or `scripts/sea_*`.
- Thysanoessa against Meganyctiphanes by season: not found in the sources read. W5 gives the euphausiid
  life cycle only.
