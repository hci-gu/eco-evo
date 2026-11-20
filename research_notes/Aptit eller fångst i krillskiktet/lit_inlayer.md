# In-layer acquisition and the field functional response on krill: literature trail (2026-11-20)

Research subagent, literature and data only. Nothing outside this folder was changed, nothing was committed, and no
value was chosen from or fitted to model output.

**Question served.** Whiting and Norway pout stood in the Oslofjord krill layer day and night with half-full
stomachs and 1.5-6 krill per stomach. Two readings compete:
- (A) appetite: the fish are not hungry;
- (B) acquisition inside the layer: hungry, but krill are hard to detect or catch, or are rejected, or are met only
  in short bouts.

This file collects the evidence on the acquisition side and on the field functional response.

**Tags.**
- [F] full text read (for scanned papers: page images read).
- [A] abstract only.
- [S] secondary: another note in this repository, a review, or a text read only through the fetch tool's quote
  extraction (fragments, not checked against the page).
- [D] derived here: arithmetic, a script, or a value read off a figure.
- [J] judgement.

**Proxy flag.** Every species other than whiting is a proxy. The distance is named at each entry.

**Files.**
- `tables/inlayer_raw.csv`: one row per number (211 rows), written by `scripts/inl_make_csv.py`.
- `scripts/inl_onsrud_counts.py` -> `tables/inl_onsrud_counts_stdout.txt`.
- `scripts/inl_wilson_table2.py` -> `tables/inl_wilson_table2_stdout.txt`.
- `scripts/inl_oa.py`: OpenAlex helper, copied from "Krillintag och jaktsätt".
- `data/inl_*`: downloads (untrusted, not to be committed).

---------------------------------------------------------------------------------------------------------------

## 0. The two discriminators, in brief

**(1) Did stomach krill rise with a measured krill density?**
- Yes, at station scale, for juvenile walleye pollock and capelin in the Gulf of Alaska (L5). This is a proxy, and it
  is the krill component of the stomach only. Total stomach content was not modelled on krill density, so a
  demand-limited fish switching to krill would give the same pattern.
- Yes between years for the krill-specialist icefish (L6, L7). This is a far proxy, and its krill availability is
  partly inferred from the diet itself.
- Not consistently for Barents Sea cod (L8). In some years krill were abundant but little eaten, because capelin or
  redfish were available.
- Unknown for hake: the krill share was held while the ration was not obtained (sibling note).
- **Never measured inside a krill layer, and never for whiting.**

**(2) Where krill intake was low, were the fish lean (acquisition) or in good condition (appetite)?**
- Icefish in krill-poor years: low intake, low condition, gonads sacrificed (L6, L7). This points to acquisition,
  in a far proxy and in years of low krill.
- Barents cod: krill-heavy diets go with low fat and low stomach fullness (L8 [S]). Cod eat krill mainly when
  capelin is scarce, so the low fat may reflect the capelin shortage rather than krill being hard to get. The
  confound is not separable from the fragments.
- Oslofjord whiting and pout: no condition data (L1). The discriminator cannot be applied there.
- The appetite side (Berg et al. 2024 herring; Bernreuther 2007) is in "Krillparen mot vintermagar" and was not
  redone here.

---------------------------------------------------------------------------------------------------------------

## 1. Item 1: field intake against measured euphausiid density

### L5. Wilson, Jump & Buchheister 2009 [F]

**Citation.** "Ecology of small neritic fishes in the western Gulf of Alaska. II. Consumption of krill in relation
to krill standing stock and the physical environment." Mar Ecol Prog Ser 392:239-251. doi:10.3354/meps08237.
- PDF from the NOAA PMEL publication server: `data/inl_wilson_pmel687/`.
- Proxy: juvenile walleye pollock (39-250 mm SL, a gadoid), capelin and eulachon, mostly on *Thysanoessa*.
  Station scale; krill density integrated over the water column. Not a layer study.

**Design**
- "Samples were collected during September 2000, 2001, and 2003 over a 48-site grid".
- "Most sites were sampled once during the day and again at night, usually within 24 h."
- "Krill standing stock (ind. km-2) at each site was computed from depth-integrated population density ... using
  samples collected at night".
- Stomach krill was regressed on krill abundance (Eq. 3): "Wky is krill weight (g^0.25 fish-1), Lky is mean length
  (mm) of predators examined, and Aky is krill abundance (ind. m-2)^0.25 at collection site k during year y". Only
  night data were used.

**Result (Fig. 2, p. 244)** [F]
- "Depending on predator group, krill weight in predator stomachs increased with krill abundance in the plankton
  and/or predator length (Fig. 2)."
- Panel statistics:

  | group | predictors in the fit | R2 | p |
  |---|---|---|---|
  | Age-0 pollock | length and krill abundance | 0.53 | < 0.001 |
  | capelin | krill abundance | 0.23 | < 0.001 |
  | Age-1+ pollock | krill abundance and year | 0.38 | 0.012 |
  | eulachon | length only; no krill-abundance term | 0.10 | 0.043 |

- Both axes are fourth-root transformed and a straight line was fitted. **No saturation shape can be read from
  this** [J]: a plateau at the top could be hidden by the transform, and none was tested.
- Read off the figure [D]: the Age-1+ pollock sites span (ind m-2)^0.25 of ~2.5-7, i.e. ~39-2400 ind m-2. Capelin
  sites span up to ~9, i.e. ~6600 ind m-2.
- **Total stomach content was modelled without krill** (Eq. 4): "Wlyd is stomach content weight (g^0.25 fish-1) and
  Llyd is mean length (mm) of predators". Fig. 3 shows it against predator length, with year and diel levels. So
  whether TOTAL intake rose with krill density is not shown.

**Rations, empties and depletion**
- Daily rations [F]: Age-0 pollock 1.60 %BW, Age-1+ pollock 1.67 %BW ("the model did not account for a significant
  amount of variation in %BW (p = 0.096)"), capelin 0.68, eulachon 0.50.
- Empty stomachs [F]: Age-0 3.2 %, Age-1+ 2.3 %, capelin 19 %, eulachon 31 %.
- Depletion [F]: "Daily consumption ranged from 0 to 20 krill d-1 and never exceeded 12% of the standing stock". The
  unit is printed as krill d-1; per m2 is implied [J].
- Medians [F]: "0.015% (N = 43), 0.026% (N = 39), and 0.018% (N = 26)".

**Table 2 (p. 248, transcribed in `scripts/inl_wilson_table2.py`)** [F/D]
- Area-mean krill density is 1.2-13.7 g m-2 over 14 area-years [D]. The box's 5 g m-2 lies inside that range.
- Per capita krill numbers rank-correlate with area krill density at rho 0.33-0.51 (n 12) [D].
- **This "response" is built in.** Per capita consumption is "PW x DR / 100 x KP / KW" (Eq. 2), with one daily ration
  per group. It varies only through the krill diet share, predator weight and the size of krill in the stomach.
- The 2001 increase coincides with small krill: "Krill mean body weight was low in 2001 and high in 2003". The
  authors' "compensatory functional response" is therefore in krill numbers and diet share, not in a measured ration
  [J]. This is the same problem as Robinson 2000 (see "Krillintag och jaktsätt" S3).

**Other quotes** [F]
- Fish without krill where krill was present: "no krill were recovered from predator stomachs where Wilson et al.
  (2006a) estimated that consumption was 30% of the standing stock".
- Fish with krill where nets caught none: "non-null consumption estimates were paired with null standing stock
  estimates (i.e. fish contained krill where we collected none)".
- Caveat from the authors: "Individuals differ in motivation to feed depending on hunger ... Fine-scale spatial and
  temporal effects on predator and prey probably were important, but are not well addressed by water column-integrated
  sampling at widely spaced sites."

**The authors' own mechanism** [F]: "We suggest that the increase in krill abundance during 2001 triggered
prey-switching behavior by predators, resulting in a compensatory functional response. The relatively small size of
krill during 2001 might have reinforced this response if, despite possible reductions in foraging efficiency,
predators maintained ration sizes by consuming more krill."

**Reading** [J]
- This is the only station-paired regression of a gadoid's stomach krill on a measured local krill density.
  Buckley et al. 2016 is a between-year series of six points.
- Stomach krill rose with density over 1-2 orders of magnitude, at column densities of the box's order (1-14 g m-2
  area means).
- **It does not separate (A) from (B).** A rising krill component with total content unshown is what an
  encounter-limited fish would show, and also what a demand-limited fish would show if it switched to krill as krill
  became commoner (the authors' reading).
- What it does not show: rations (fixed), total intake against krill, behaviour inside a layer, adult gadoids,
  *M. norvegica*, or winter.

### L6. Kock, Wilhelms, Everson & Gröger 1994 [F]

**Citation.** "Variations in the diet composition and feeding intensity of mackerel icefish *Champsocephalus gunnari*
at South Georgia (Antarctic)." Mar Ecol Prog Ser 108:43-57. doi:10.3354/meps108043.
- From NERC NORA: `data/inl_kock1994_meps108/`.
- Far proxy: a krill specialist (Antarctic krill), summer, ca 8700 stomachs.

**Numbers** [F]
- "Average feeding intensity (median, and 25 % and 75 % percentiles) was 1.07 % (0.31 to 2.40%) in 1985, 0.27 % (0.0
  to 1.21 %) in 1991 and 2.19 % (0.63 to 4.43 %) in 1992."
- In 1991 "the proportion of empty stomachs often exceeded 40% per station".
- 1991 was a krill-shortage year: "In years of krill shortage, such as in 1991, krill was replaced by T. gaudichaudii."
- 1992 was a high-krill year ("years of high krill abundance around the island, such as 1992").
- On condition and reproduction: "an unusually high proportion of sexually mature fish showed no signs of the gonad
  development necessary for spawning in that season ... the shortage of suitable food may have forced the fish to
  sacrifice gonad maturation in order to maintain body size."

**Circularity** [J]
- The 1991 krill scarcity is supported by predator breeding failures and by comparison with 1977/78 net and echo
  surveys.
- The authors also propose the diet itself as the index: "extensive food studies in mackerel icefish could serve as an
  indicator of krill availability". No krill density was measured at the stations in this paper.

**Reading** [J]
- Between years, a krill specialist's stomach content tracked krill availability over x8 (median 0.27 -> 2.19 %BW).
- In the scarce year it lost reproductive condition. That is acquisition limitation at low krill, not appetite. It is
  a far proxy and inter-annual, not in-layer.

### L7. Main, Collins, Mitchell & Belchier 2009 [A]

**Citation.** Polar Biol 32:569-581. doi:10.1007/s00300-008-0552-7. Paywalled; abstract read from the NORA record
through the fetch tool.
- "Stomach contents from 2239 C. gunnari (130-560 mm total length) were examined."
- "In a season of poor krill availability (summer of 2003-2004) the proportion of krill in the diet, stomach fullness
  and fish condition (indicated by length-weight relationships) were significantly lower".
- "A large reduction (> 80%) in the estimated annual (2005) biomass of the C. gunnari stock directly followed the
  season of poor krill availability."

**Reading** [J]: the same direction as L6. Low krill gives low fullness and low condition, in a far proxy.

### L8. Orlova, Dolgov, Rudneva & Nesterova 2005 [S: fetch-tool fragments only]

**Citation.** "The effect of abiotic and biotic factors on the importance of macroplankton in the diet of Northeast
Arctic cod in recent years." ICES J Mar Sci 62:1463-1474. doi:10.1016/j.icesjms.2005.08.003.
- Unpaywall: not OA. The OUP page served the full text to the fetch tool, but I saw only extracted fragments. **None
  of the fragments below is verified against the page.**
- Proxy: adult cod, Barents Sea, mainly *Thysanoessa*.

**Density measure.** It is an index from "annual surveys of macroplankton ... during autumn-winter by the near-trawl
net in the bottom layer (6-10 m from the bottom)", in ind 1000 m-3. The fetch output says the article does not state
whether it was matched to the stomach stations or times. **It is not a paired measurement.**

**Fragments**
- High krill but low use:
  - 1986-87: "the higher abundance of euphausiids compared with the long-term mean notwithstanding, their importance in
    the cod diet was very low".
  - 1987: "Low cod predation on euphausiids was associated with prolonged feeding by cod on redfish during
    over-wintering".
  - 1995: "cod did not consume euphausiids at all" in April, while eating shrimp.
- Where krill was concentrated and eaten:
  - "where concentrations of euphausiids were high, were high levels of cod consumption observed (with 60-90%
    frequency of occurrence)";
  - 1997 on the Murmansk Bank slopes: "very high (1-5 individuals m-3)".
- Krill feeding and low intake:
  - 1988, high krill and low capelin: "high rates of euphausiid consumption were probably one of the reasons for the
    low intensity of cod feeding";
  - 1995: "the index of cod stomach fullness did not exceed 1.5-2".
- Condition:
  - "The low fat content of cod is sometimes associated with feeding on postspawning euphausiids";
  - "a significant negative relationship was revealed between energy potential of cod and weight percentage of
    hyperiids and, to a lesser extent, of euphausiids".
- Summary statement: consumption is "not always dependent on the abundance of winter concentrations of euphausiids".

**Reading** [J]
- Cod use of krill follows the availability of preferred fish prey as much as krill density.
- When cod do live on krill, their fullness and fat are low. This fits krill as a low-yield prey that does not fill a
  gadoid even where dense (acquisition or profitability, reading B), not a satiated fish.
- **Confound:** cod eat krill mainly when capelin is scarce, so low fat and fullness in krill years may be the capelin
  shortage itself, not krill being hard to get or unprofitable.
- Fragment-level evidence only. Buckley et al. 2016 for pollock points the same way ("euphausiids became more
  important in the annual average diet when feeding success ... was the lowest"; "Krillintag och jaktsätt" S1).

### L8b. Dalpadado & Bogstad 2004 [S: search-summary paraphrase only]

**Citation.** "Diet of juvenile cod (age 0-2) in the Barents Sea in relation to food availability and cod growth."
Polar Biol 27:140-154. doi:10.1007/s00300-003-0561-5.
- Unpaywall: not OA. OpenAlex has no abstract. No repository copy found.
- The search tool's paraphrase of the abstract: age-0 and age-1 cod fed mainly on crustaceans, with krill and
  amphipods up to 70 % of the diet. That consumption was mainly in stomachs from the central Barents Sea and near the
  Polar Front, where those prey are abundant in summer. Age-2 cod shift toward crustaceans when capelin is low.
- No number from it is used. Whether it pairs intake with a krill density could not be checked.

### L9. Johannesen et al. 2012 [F] (a negative result)

**Citation.** "Feeding in a heterogeneous environment: spatial dynamics in summer foraging Barents Sea cod." Mar Ecol
Prog Ser 458:181-197. doi:10.3354/meps09818. OA PDF in `data/inl_johannesen2012_meps458/`.
- The largest spatial cod-prey study dropped krill: "Due to this strong diurnal vertical migration of krill, only
  night hauls could be used ... which reduced the data set by 2/3. We therefore chose not to include krill in our
  analysis".
- Krill was "~10% of the individual consumption".
- Small cod: "a high proportion of stomachs of small cod were empty or contained mostly krill and unidentified
  crustaceans".
- Reading [J]: the absence of a cod-krill functional response is a sampling problem in that literature, not a null
  result.

### L10. Shaw, Tanasichuk, Ware & McFarlane 1989 [F] (Pacific hake; no pairing)

**Citation.** "Biological and species interaction survey of Pacific hake, sablefish, spiny dogfish and Pacific
herring off the southwest coast of Vancouver Island, August 1987." Can MS Rep Fish Aquat Sci 2027.
- From publications.gc.ca: `data/inl_dfo_msr2027/`.
- 40 trawl hauls; 16 vertical bongo hauls at standard stations, not at the hake hauls. **No pairing.**
- Trevor Channel: "a shallow layer appearing to be a dense plankton layer ranging in depth from 66-73 m and a deeper
  layer of hake ranging in depth from 77-91 m. The plankton layer contained high concentrations of glass shrimp,
  herring, and euphausiids."
- Diet: "hake smaller than 40 cm fed only on euphausiids".
- Stomach volume per stomach, everted excluded: 7.8 cc inshore (hake 23-44 cm) against 16.8 and 26.9 cc offshore
  (larger fish; size confounded).
- Herring: "Herring fed on euphausiids exclusively; 39% of the fish examined had empty stomachs."
- Reading [J]: hake sat just below a dense euphausiid layer, not in it, at the time of the echogram (time of day not
  stated in the passage). This matches Mackas et al. 1997 (L12).

### L11. Tanasichuk, Ware, Shaw & McFarlane 1991 [A]

**Citation.** CJFAS 48:2118-2128. doi:10.1139/f91-251. Unpaywall: not OA.
- "Summer ration estimates were higher for hake feeding on the continental shelf than for fish feeding along the
  shelf-break. Daily ration declined over summer because the euphausiid ration dropped, even though the herring ration
  increased."
- "Both predators fed sporadically between dawn and dusk. They evacuated prey linearly over time and digested
  euphausiids most rapidly."
- The ration tables were not obtained. Two web snippets give conflicting euphausiid shares of the ration (88 % vs 84 %
  for hake); neither is used.

### L12. Mackas et al. 1997 [A]

**Citation.** "Aggregation of euphausiids and Pacific hake (Merluccius productus) along the outer continental shelf
off Vancouver Island." CJFAS 54:2080. doi:10.1139/f97-113. Unpaywall: not OA.
- "Both euphausiids and hake share similar horizontal distributions, although separated vertically by tens of meters
  during daylight hours."

### Carried from sibling notes, not re-read [S]

- **Robinson 2000** (J Plankton Res 22:1649): no relation between euphausiids consumed per tonne of fish and
  euphausiid biomass. The rations may have been fixed. See "Krillintag och jaktsätt" S3.
- **Tanasichuk 1999 / 2002** [A there]: the krill share of the hake ration held at ~0.89 while krill available to hake
  fell to 0.12-0.27 of 1991.
- **Buckley et al. 2016**: only pollock >= 60 cm tracked acoustic krill biomass, over 6 survey years.
- **Yamamura et al. 2013** (MEPS 491:221, doi:10.3354/meps10445) [F, read here]: Doto-area pollock diets over 1989-2011
  with no euphausiid density series. It cannot be used for item 1.

---------------------------------------------------------------------------------------------------------------

## 2. Item 2: feeding chronology and digestion state inside a krill layer

### L1. Onsrud, Kaartvedt, Røstad & Klevjer 2004 [F]

**Citation.** "Vertical distribution and feeding patterns in fish foraging on the krill *Meganyctiphanes norvegica*."
ICES J Mar Sci 61:1278-1290. doi:10.1016/j.icesjms.2004.09.005.
- Local copy: research_notes/Vitlingens fiskbyten/data/step2_lit/vertical/onsrud2004.pdf. Table 3 was read from the
  rendered page.
- Not a proxy for whiting or pout.
- **No quantitative krill density.** The only krill measures are "The volume of krill captured in each tow was noted"
  (litres per haul; the bottom-trawl Table 2 has a "Krill (l)" column) and the 120 kHz echograms. Neither is paired
  with the stomach fish.
- **Which hauls the stomach fish came from is not stated.** Fish came from 66 pelagic hauls targeted at acoustic
  layers at several depths, and from 24 daytime bottom hauls in which whiting and pout also occurred. Table 3 pools
  them by day and night.

**Methods** [F]
- "For fish with a total length larger than 25 cm, stomachs were removed, weighed, and frozen separately." Smaller
  fish, among them the 21-cm whiting with 40 krill, were frozen whole.
- "Stomach fullness was categorized in groups on a scale from 1 to 5 (1 denotes empty, 5 full/distended)".
- "State of digestion was categorized to groups on a scale from 1 to 5 (1 denotes undigested, 5 fully
  digested/unrecognizable)".
- The Table 3 caption says "November 2002"; the cruises were November 2000. The "fish structures (%)" column holds
  values such as 0.13 and 0.02, which look like proportions.
- **Table 3 gives no n per cell.** No test between cells is possible from it.

**Table 3, whiting** [F] (mean, SD in brackets)

| cell | fullness | % with krill | krill per stomach | digestion | % empty |
|---|---|---|---|---|---|
| Nov day | 2.67 (1.37) | 42.31 | 1.82 (0.98) | 2.89 (0.91) | 25.00 |
| Nov night | 3.00 (1.20) | 35.19 | 3.47 (3.47) | 1.88 (0.91) | 14.81 |
| Mar day | 3.28 (1.03) | 75.51 | 3.95 (4.28) | 3.03 (1.17) | 8.16 |
| Mar night | 3.25 (0.97) | 79.69 | 5.92 (11.39) | 2.77 (1.09) | 6.25 |
| Jun day | 3.08 (1.21) | 42.86 | 2.17 (2.33) | 3.32 (1.25) | 7.14 |
| Jun night | 2.23 (1.03) | 34.62 | 1.44 (1.01) | 2.92 (1.75) | 26.92 |

Norway pout and herring rows are in the CSV.

**Text** [F]
- "For whiting, krill constituted up to 99% of the stomach content by weight, on average ~82% during both day and
  night (maximum number of krill found in one stomach was 40 krill; fish length 21 cm)."
- Results, whiting: "No systematic trends were apparent comparing proportion of stomachs containing krill or state of
  digestion during the day and night (Table 3)."
- Discussion: "The proportion of stomachs with krill structures were similar, or higher by day than by night, while
  the state of digestion for whiting captured at night suggested that food had been eaten recently."
- Pout: "No significant differences in stomach fullness, or normalized stomach wet weight could be found between
  day/night (chi-square and Kruskal-Wallis tests, p > 0.05)."
- Herring: "stomach contents were always well digested".

**Derived** [D] (`scripts/inl_onsrud_counts.py`)
1. **"Krill per stomach" is conditional on krill being present.** Any distribution with a share p0 = 1 - FO of zeros
   has SD >= m sqrt(p0/(1 - p0)).
   - In 11 of 14 cells the reported SD is below that bound. Example, pout Nov day: SD 0.78 against a minimum of 1.28.
     The other 3 cells cannot decide; none contradicts.
   - So 1.5-6 krill are counts among the fish that had krill. Unconditionally, whiting held 0.5-4.7 krill per stomach
     on average.
2. **Overdispersion among whiting with krill.** Variance/mean:
   - Nov day 0.53, Nov night 3.47, Mar day 4.64, Mar night 21.9, Jun day 2.50, Jun night 0.71.
   - In Mar night, Jun day and Jun night the SD exceeds the log-series limit (11.39 vs 8.51, 2.33 vs 2.00, 1.01 vs
     0.89), the largest SD any zero-truncated negative binomial can have at that mean.
   - **That excess is not robust.** By the inferred n, June has only ~6-12 (day) and ~9 (night) krill-bearing
     stomachs, so an SD 13-17 % over the limit is within sampling noise. March night (~51 krill-bearing stomachs)
     rests on a few large counts, including the 40-krill fish.
   - What holds is the plain overdispersion in Nov night and in March (variance/mean 3.5-22): most whiting with krill
     held 1-3, and a few held many.
3. **Excess krill-free whiting.** In Nov day the positives fit a near-Poisson.
   - A homogeneous Poisson intake would leave 0.26 of stomachs krill-free; the observed share is 0.58.
   - **Caveat:** the stomach hauls are not identified (above). Some krill-free fish may have been caught outside the
     krill layer (daytime bottom trawls, other pelagic depths), so the excess need not be fish that sat in the layer
     and ate none.
   - Pout fit a homogeneous Poisson by day: 0.39 implied vs 0.41 observed (Nov), 0.245 vs 0.250 (Mar). At night pout
     had more krill-free stomachs (0.56 and 0.52); copepod stomachs were 32-48 % at night.
4. **Digestion.** Night minus day for whiting krill was -1.01 (Nov), -0.26 (Mar) and -0.40 (Jun) on the 1-5 scale.
   - Only November (about one within-cell SD) stands. March and June are inside SDs of 1.1-1.75, consistent with the
     authors' "No systematic trends".
   - Within-cell SDs of 0.9-1.75 show a spread of stages.
   - Pout krill were well digested day and night (3.4-3.7).
5. Inferred n, from the percentages: whiting Nov 52/54, Mar 49/64, Jun 14-28/26. These are guesses only (the
   smallest n consistent with two printed percentages).

**Reading** [J]
- Whiting: fresh krill at night in November, krill in 35-80 % of stomachs, and overdispersed counts among those with
  krill in November nights and in March. This is not a uniform trickle in which every fish takes the same few krill.
- Among whiting with krill, a few had recent large meals and most had one to three krill. That is the signature of
  bouts or local patches (B: short bouts), or of differing hunger states (A); the table cannot separate them. The
  krill-free fraction may include fish from outside the layer.
- Pout by day looks like a homogeneous slow trickle (Poisson, 1.5-2 krill per positive stomach, well digested). That
  fits either a low steady encounter rate or a low steady demand.

### L4. Albert 1994 [F, page images of a scanned PDF]

**Citation.** "Biology and ecology of Norway pout (Trisopterus esmarki Nilsson, 1855) in the Norwegian Deep." ICES J
Mar Sci 51:45-61. doi:10.1006/jmsc.1994.1005.
- Local copy: research_notes/Fiskgrupper i rutan/planktivores/data/albert1994/albert1994.pdf, rendered with pdftoppm.
  Quotes are transcribed from the images.
- Not a proxy for pout. The region includes the Skagerrak slope; winter 1986-87.

**Table 8** [F]
- Winter, < 14 cm: 277 stomachs, 33.6 % empty, *M. norvegica* 42.8 %W, %F in non-empty stomachs 29.3.
- Winter, >= 14 cm: 188 stomachs, 24.5 % empty, *M. norvegica* 59.1 %W, %F 47.2.
- Summer: empty 27.8 % (< 16 cm) and 18.6 % (>= 16 cm).

**Meal structure** [F]
- "Each stomach contained only a small number of different taxa and size-groups. Among the non-empty stomachs, 43%
  contained only one prey category, and among the four predator groups in Table 8 from 78 to 92% had no more than
  three categories."
- "There was further significant individual variation in species composition within the same station and length
  group, and this was not explained by differences in digestion state."

**Diel series** [F]
- "The stomach content data from seven hauls from the same locality made over a period of 22 h ... dominated by M.
  norvegica and calanoid copepods (70-99% and 0-27% by weight, respectively)".
- "From midnight until the early afternoon the proportion of empty stomachs steadily increased together with the
  proportion of the contents which was heavily digested (Fig. 11). At the same time the mean weight of the stomach
  contents (excluding empty stomachs) relative to fish weight was halved. At the last haul at 2020 h the stomach
  contents were only slightly or moderately digested, the relative mean weight had increased and the proportion of
  empty stomachs decreased."
- "the foraging activity of Norway pout preying on euphausiids and copepods was low in the middle of the day and ...
  the main feeding period was at night."
- The digestion scale is in Table 2: 1 fresh ... 5 digestion almost complete; "Regurgitated stomachs were rejected."

**Fig. 11, read off at 200 dpi** [D]
- n per haul: 16, 18, 19, 26, 18, 13, 29.
- Empty: 0, 22, 21, 45, 72, 30, 30 % at ~0.5, 5, 9, 11.5, 14, 17 and 20 h.
- Relative weight: 2.2, 1.9, 1.65, 1.35, 1.0, 1.0, 2.0.
- Stage-5 share of content weight: < 5 % at 0-5 h, 40-50 % at 14-17 h, ~3 % at 20 h.
- The season of this series is not stated on the pages read. The caption says a 20-h period; the text says 22 h.

**Reading** [J]
- In the region's own pout on *M. norvegica*, intake comes as a nightly feeding period followed by a daytime pause:
  stomachs empty out and digest by mid-afternoon, and fresh food is back by 20:20.
- Stomachs hold one or a few prey types, with unexplained individual differences at a station. That fits meals taken
  from one prey type at a time (patch or bout structure) rather than a continuous mixed trickle.
- One series, one site. It does not say whether the daytime pause is no hunger (A) or no catchable krill in reach by
  day (B). The pout sit near the bed by day, below the krill layer (L1, L2).

### L2. Onsrud, Kaartvedt & Breien 2005 [F]

**Citation.** "In situ swimming speed and swimming behaviour of fish feeding on the krill Meganyctiphanes norvegica."
CJFAS 62:1822-1832. doi:10.1139/f05-090.
- Local copy: research_notes/Krillens fläckighet/data/onsrud2005/paper.txt. Same Oslofjord site, November 2000.

**Speeds** [F]
- "Swimming speeds of deep-living fish, mainly Norway pout ... and whiting ..., were highest during the day (speeds
  centred around 14-16 cm s-1) and decreased somewhat by night (modes around 10-11 cm s-1). Fish in the upper 10-30 m
  swam somewhat faster (speeds ranging from 16 to 24 cm s-1)."
- That is "about 1 body length s-1 by day and 0.7 body length s-1 by night".

**Track structure** [F]
- "The majority of tracks were short (10-20 pings), and the NGDRxy largely revealed straight (portions of) swimming
  paths with little vertical excursions."
- During dusk: "245 tracks were descending, while 109 tracks were ascending".
- One search cycle: "Only one cycle was successfully tracked, displaying slow, spiralling, upward swimming (vavg ~11
  cm s-1) ... and straight, rapid, downward swimming (vavg ~39 cm s-1)."
- The authors read it as visual search from below. "We were unable to establish which fish species adopted the
  up-and-down swimming strategy"; they lean to herring.
- Pout at night: "Swimming trajectories, generally curved paths with no velocity fluctuations, with swimming speed as
  low as 0.3 body length s-1". The authors hypothesise lateral-line detection: "this behaviour suggests that prey
  movement may contribute in mediating encounters."
- Krill speed: "krill swimming speed with mode ~4 cm s-1 (Klevjer and Kaartvedt 2003), which corresponds to 30%-40% of
  the measured modes for deep-living fish at night."

**Not in the paper** [F]: no attack, strike, pursuit or capture events were resolved, and no speed or turning rate was
related to krill density.

**Reading** [J]: the gadoids in the layer swam at ~1 BL s-1 on mostly straight short tracks. The authors note that
"swimming speed could vary greatly within long tracks", and the one tracked cycle had 39 cm s-1 descents, but bursts
were not analysed as feeding events. The tracks say nothing either way about attacks on krill.

### Hake, pollock and whiting chronology elsewhere

- **Pacific hake** (L11 [A]): "fed sporadically between dawn and dusk".
  - Livingston 1983 [S, via "Krillintag och jaktsätt" S5]: "Stomach content weight per fish weight is highest at 1800
    h, with a value of 2.5%".
- **Juvenile walleye pollock** (L5 [F]):
  - Age-0: "%BW increased from noon to late night" and "krill comprised >= 60% of stomach content mass in any given 3
    h time interval".
  - Age-1+: "an increase in the percentage of fish remains from stomachs of individuals that corresponded with peaks
    in %BW at sunrise and sunset".
  - Capelin: "%BW was highest just after sunset when krill comprised 89% of stomach contents".
- **Whiting, North Sea** [S, via "Torskfiskars ljusberoende vid jakt/gadoid_diel_feeding.md"]:
  - Rindorf 2002 (J Fish Biol): the probability of fresh food fell when partly digested food was present, while meal
    size did not change. This is an appetite brake on search (A-side mechanism). Abstract-level only in that note.
  - Mergardt & Temming 1997: one nocturnal peak at 22-24 h.

---------------------------------------------------------------------------------------------------------------

## 3. Item 3: predator-krill interaction observations

### L1 (Onsrud et al. 2004), acoustic observations [F]

The paper reports two separate kinds of "void". Keep them apart.

**(a) Fish voids when ships passed** (fish, not krill):
- "Repeatedly, virtually all fish targets disappeared from the acoustic beam, resulting in 'voids' in the recordings
  (Figure 2). This coincided with ships passing at a distance of 200-400 m."
- "all fish targets disappeared from the ensonified volume, and then reappeared 10-20 min later."

**(b) Krill voids beside deep schooling fish** (krill):
- "The 'voids' in the krill recordings in the proximity of deep fish assemblages (cf. Figure 3) suggest that the krill
  performed instantaneous escape reactions during day."
- Figure 3 shows schooling fish entering the krill layer: "Some schooling fish were also occasionally seen entering
  the krill layer (e.g. Figure 3)". The text places most herring in "the deepest schools".
- At night: "It appeared that M. norvegica would seek out water with reduced fish abundance at night (cf. Figure 2)".
- **Dispersed gadoids:** "Non-schooling fish (appearing as a scattering layer (SL) in the compressed plots) were
  distributed together with the krill." No krill void around them is reported, and no absence is reported either:
  **not reported**. Reading [J]: the measured escape response is to schools, as in Christiansen et al. 2024 (sibling
  note, herring schools).

**Vessel caveat** [F]
- "Sv from the layer below 75 m differed by ca. 9 dB, indicating an about eight times higher biomass directly beneath
  the vessel compared to 80 m away from the vessel."
- "Fish accumulated beneath the research vessel when the ship was anchored at a fixed location during acoustic
  studies, apparently resulting in artificially high local fish abundances. Since we suggest that krill respond to the
  presence of fish, such high fish abundance may bias studies of interactions between the fish predators and their
  krill prey."
- Reading [J]: the stomach fish came from trawls, not from under the ship. But any in-layer encounter or avoidance
  number from this site's acoustics carries the ~8x fish excess.

### L2 (Onsrud et al. 2005)
No attack or escape events were tracked (above).

### L13. Klevjer & Kaartvedt 2006 [S: fetch-tool fragments]

**Citation.** "In situ target strength and behaviour of northern krill (Meganyctiphanes norvegica)." ICES J Mar Sci
63:1726-1735. doi:10.1016/j.icesjms.2006.06.013. The OUP PDF returned 403; the HTML was read through the fetch tool.
- Daylight only.
- Krill tracks: "larger krill deeper in the water column were prone to more upward swimming"; "In one case, the
  organism was swimming in tight loops".
- **No reaction to fish and no predator-krill events reported.**

### Carried [S]
- **Klevjer & Kaartvedt 2011** (L&O 56:765; "Krillens flykt och energibudget" E17): krill swim ~3.5 cm s-1 by day;
  after oxygenation the "nocturnal average swimming speeds increased to ~10 cm s-1". The authors attribute this to
  "increased feeding activity under lessened danger of predation in darkness".
- **Christiansen et al. 2024**: daytime krill avoidance of large herring schools, none at night (sibling notes).

### Not found (see the search log)
- Video or tracking of gadoids attacking euphausiids, with attack rates or success.
- Krill vigilance or local avoidance of single demersal fish.
- Rejection of krill after capture.
- A krill profitability (energy per item against pursuit time) for any fish.
- A lab functional response of any fish on euphausiids.

---------------------------------------------------------------------------------------------------------------

## 4. What the evidence says about (A) appetite against (B) acquisition inside the layer [J]

1. **Encounter (B-encounter).**
   - Where density varied at station or year scale and was measured, stomach krill rose with it in juvenile pollock
     and capelin (L5). Total stomach content rose with krill availability in icefish between years (L6, L7).
   - In L5 only the krill component was regressed, at column densities of the box's order (area means 1-14 g m-2).
     That fits encounter limitation and also demand-limited switching to krill (the authors' reading). It does not
     separate (A) from (B).
   - The icefish (a far proxy, krill specialist) show that total intake and condition fall when krill is scarce. That
     is encounter limitation at LOW krill, not evidence about a fish already inside a dense layer.
   - All of it is column-integrated or inter-annual. It says nothing about the rate inside a dense layer once a fish is
     in it.
2. **Capture and escape (B-capture).**
   - The only measured krill escape is from fish schools (L1 voids, Christiansen 2024). Nothing measured shows krill
     escaping dispersed gadoids, and nothing shows that they do not.
   - No attack-success data exist (sibling G1). The in-layer tracks were not analysed for attacks (L2).
   - **No support either way.**
3. **Rejection or low profitability (B-rejection).**
   - Cod: krill feeding comes with low fullness and low fat (L8 [S]). Pollock: krill is a fallback when feeding success
     is lowest (Buckley 2016).
   - This points to krill as a low-yield prey, but it is a between-prey choice at regional scale, not in-layer rejection.
4. **Short bouts (B-bouts).**
   - Whiting counts at the Oslofjord site are overdispersed in November nights and March, and there are more
     krill-free fish than a homogeneous Poisson gives (haul origin unknown). Fresh krill at night in November show
     night feeding (L1 [D]).
   - Region-own pout feed in a nightly bout and pause by day (L4).
   - Single-prey stomachs dominate (L4).
   - These patterns fit bout-structured intake. Variable hunger (A) would produce the same dispersion.
5. **Appetite (A).**
   - The in-layer data neither exclude nor show it.
   - Half-full stomachs (fullness ~3 of 5) with fresh and digested krill together mean the fish kept feeding. They were
     not starving, but neither were they full.
   - The A-side evidence remains Berg 2024 for herring (in excess food, condition follows day length) and Rindorf
     2002's fullness brake on search in whiting [S].
6. **The missing measurement.** None of these settles whiting in the Skagerrak.
   - The datum that would is stomach contents and condition of whiting trawled inside the krill layer, at a measured
     layer density (acoustics or MOCNESS), at several densities.
   - The Oslofjord site has no krill density, and the GoA set is column-integrated juvenile pollock.

---------------------------------------------------------------------------------------------------------------

## 5. Corrections to sibling notes

1. **"Krillintag och jaktsätt", Q1 verdict:** "No study measures a gadoid's or hake's intake against a measured krill
   density, in the field or the lab". This is amended.
   - Wilson, Jump & Buchheister 2009 (L5) regress night stomach krill weight of juvenile walleye pollock (a gadoid) on
     krill abundance measured at the same site, both at night. The relation is significant (Age-0 R2 0.53, Age-1+
     0.38).
   - It measures the krill component of stomach content, not ration and not total intake (total content was modelled
     on length only). It is column-integrated, not in-layer, and covers juveniles on *Thysanoessa*.
   - Its Table 2 "per capita consumption" rests on a fixed ration and is not a ration response.
2. **"Krillens fläckighet", Q2a/verdict 4:** "no evidence that they track krill at km scale". Wilson 2009 Part II
   (L5) cites Part I (MEPS 392:223, not read) for juvenile pollock: "krill was identified ... as being well associated
   with fish geographic distributions", and "Walleye pollock >120 mm and eulachon aggregated with krill over the shelf
   in relatively high-flow areas associated with sea valleys" [S, Part I not read]. That is mesoscale co-location of a
   juvenile gadoid with krill, not tracking at 1 km.
3. **Onsrud et al. 2004**, as cited in the brief ("1.5-6 krill per stomach"): the counts are means **among
   stomachs with krill** [D, section 2]. Over all stomachs the means are 0.5-4.7 for whiting.
   - "Krillparen mot vintermagar" `scripts/onsrud_rations.py` already reads them this way ("krill per stomach WITH
     krill", multiplied by the share with krill). The derivation here confirms that reading; no number there changes.

---------------------------------------------------------------------------------------------------------------

## 6. Search log

The Unpaywall helper was used for DOIs; results are listed per source.

**Searched and found (used above)**
- WebSearch, pollock: "age-0 walleye pollock stomach fullness euphausiid prey density acoustic backscatter Gulf of
  Alaska". Led to Wilson et al. 2009 (L5), via PMEL.
- WebSearch, icefish: "mackerel icefish ... krill availability condition South Georgia". Led to Kock 1994 (L6) and
  Main 2009 (L7).
- OpenAlex (before the rate limit), Barents cod: Johannesen 2012 (L9). The Orlova 2005 DOI came via WebSearch.
- WebSearch, DFO grey literature for the hake ration tables (one query). Led to Can MS Rep 2027 (L10), which has no
  ration table.

**Searched and not found**
- A haul-level pairing of hake stomachs with acoustic krill backscatter. The NOAA hake survey states it is examining
  diets; no paper was found.
- In-situ video or tracking of fish attacking euphausiids. Only tank studies of Antarctic krill tail-flips (not fish)
  turned up.
- A laboratory functional response of any fish on euphausiids. None was found; the hits were krill as predators and
  goldfish or dogfish on other prey.

**Not run**
- Three OpenAlex searches died on HTTP 429:
  - lab functional response;
  - icefish;
  - intake vs krill density.
- The second and third were re-run through WebSearch (above). The first was re-run through WebSearch, with no hit.
- Not searched for lack of time:
  - Dwyer, Bailey & Livingston 1987 (pollock ration, Bering);
  - Adams et al. 2007 (Gulf of Alaska pollock prey selection);
  - Ressler et al. 2012 (Bering krill acoustics; Ressler 2014 is held by "Krillens fläckighet");
  - Wilson 2009 Part I (MEPS 392:223), which is the obvious next read for co-location;
  - Bergstad 1991 (Sarsia 75), already known as paywalled from sibling notes.

**Access failures**
- Unpaywall, not OA:
  - Tanasichuk et al. 1991, Tanasichuk 1999 and 2002;
  - Robinson 2000;
  - Mackas et al. 1997;
  - Dalpadado & Bogstad 2004 (doi:10.1007/s00300-003-0561-5);
  - Main et al. 2009;
  - Orlova et al. 2005.
- Unpaywall OA but the download failed:
  - Klevjer & Kaartvedt 2011 (Wiley 403);
  - Klevjer & Kaartvedt 2006 (OUP PDF 403; HTML read through the fetch tool).
- CCAMLR Science PDFs: HTML on direct download and 403 to the fetch tool.
  - "08everson-kock" (an icefish condition index said to respond to krill density) was not read.
  - So was "02-Kock-et-al".
- CORE (core.ac.uk/download/30844108) and IMR Brage ("Year-to-year dynamics"): Cloudflare challenge pages. Deleted;
  not read.
- publications.gc.ca returned an archive notice first, then the PDF on a second request with the same session.

**Checked and of no use for this question**
- Yamamura et al. 2013 (no krill density).
- Wilson 2005 Fish Bull 103:207 (Kodiak age-0 pollock; mixed zooplankton prey; `data/inl_wilson_pmel0417/`, read only
  far enough to see no krill-density pairing).
- Onsrud & Kaartvedt 1998: krill DVM. Only "fish invaded the krill layer at night", already in sibling notes.
