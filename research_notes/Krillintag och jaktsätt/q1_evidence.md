# Q1 evidence: gadoid and hake intake on krill at high krill density, as a share of Cmax (2026-11-20)

Literature subagent for Q1. Research only; nothing outside this folder was changed. No value was fitted to or
chosen from model output. Tags: [F] full text read, [A] abstract only, [S] secondary, [D] derived here (script and
stdout named), [J] judgement. Page numbers are the article's own (pdftotext form-feed count from the first page);
for the 1992 report and the Buckley manuscript they are document pages.

Files written by this agent:
- `scripts/q1_oa.py` (OpenAlex abstracts), `scripts/q1_nva.py` (IMR Brage / NVA open files),
  `scripts/q1_pages.py` (page of a quote), `scripts/q1_cod_cmax.py` -> `tables/q1_cod_cmax_stdout.txt`,
  `scripts/q1_buckley_pf.py` -> `tables/q1_buckley_pf_stdout.txt`.
- `tables/q1_raw.csv` (one row per number).
- Downloads in `data/q1_<source>/` (untrusted, not to be committed).

Already held by the caller and not redone: Onsrud et al. 2004, Kaartvedt et al. 2005, Rindorf & Gislason 2005,
Rindorf 2003, Dempster et al. 2011, and the ICES stomach-record analysis in `evidence.md` Part A.

## Cmax used for conversions

- Cod, two functions, both on fish prey (`scripts/q1_cod_cmax.py`):
  - Hansson et al. 1996 (ICES JMS 53:107) Table 2 as in FB4 `Parameters_official.csv` ("North Sea cod", CEQ 2):
    CA 0.099, CB -0.195, CQ 2.41, CTO 13.7, CTM 21; g wet prey / g wet fish / d (FB4 prey ED 4184 J/g). Same function
    as research_notes/"Fiskgrupper i rutan"/planktivores/scripts/cod_cmax_proxy.py.
  - Jobling 1988 as used by Neuenfeldt et al. 2020 eq. 2: 0.223 e^(0.104T - 0.000112T^3) W^0.802 kJ/d, read as kJ/d
    and converted at 4.184 kJ/g as in research_notes/"Exponering av bottenbyten"/scripts/C_cmax_bottom.py [S]. Jobling's
    function was "established ... from data on satiation rations of capelin" (Andersen et al. 2017, via
    research_notes/"Magtömning och energitak"/summary.md [S]).
  - At 1.3-2.5 C and 1-2 kg, Jobling is 1.41-1.43 x Hansson (Hansson 0.95-1.23, Jobling 1.34-1.75 %BW/d); at 8.2 C
    and 0.7-1 kg, 1.47 x. The Hansson temperature term (F(T) 0.43-0.48 at 1.3-2.5 C) is an extrapolation below most
    of its data [J]. Fractions below are given for both.
- All fractions are WET-MASS ratios (krill or amphipod wet mass over a fish-prey wet-mass Cmax). The energy ratio is
  the mass ratio x E_prey / 4.184. With krill at ~3.5-4.5 kJ/g wet (the caller's `evidence.md` A1 uses 4-5 kJ/g)
  the factor is ~0.84-1.08 [J]; it does not change any conclusion below.
- Pacific hake: no Cmax function or ad-lib lab ration was found. Francis 1983 (CJFAS 40:1925) was not read; Rexstad &
  Pikitch 1986 used its consumption equations as model rations (S4), i.e. it is not a measured Cmax [S]. Hake
  conversions use the cod functions as a labelled proxy [J].
- Walleye pollock: Buckley et al. 2016 report stomach contents, not rations, so no Cmax conversion was made.
- Whiting / saithe: GER-based Cmax as defined in `evidence.md` A1 (caller's file); not repeated here.

## Sources

### S1. Buckley, Ortiz, Kotwicki & Aydin 2016, Deep-Sea Res II 134:302-311, doi:10.1016/j.dsr2.2015.10.009 [F]
Walleye pollock, eastern Bering Sea, summer bottom-trawl survey 1987-2011, daytime stomachs, six length classes.
Accepted manuscript from the NOAA repository (`data/q1_buckley2016_dsr2/`).
- Euphausiid density: AFSC acoustic euphausiid biomass, 2004 and 2006-2010 (Ressler et al. 2012), "may also include
  some hyperiid amphipods" (Methods 2.3). Units not given (relative series in Fig. 8).
- Manuscript p. 11: "Correlation between the increase in estimated EBS euphausiid biomass and the PF values of
  euphausiids was significant (r = 0.89, P = 0.02) for the largest length category of pollock in this short time
  series (Fig. 8)." The other five length classes were not significant.
- Manuscript p. 11: "The significant negative relationship (r = -0.46, P < 0.05) between %bw and euphausiid %W in the
  diet of 40-49 cm pollock indicates that interannually lower stomach fullness was associated with a higher relative
  consumption of euphausiids".
- Manuscript p. 15: "euphausiids became more important in the annual average diet when feeding success (based on
  stomach fullness) was the lowest. These interannual patterns appear similar to the role of euphausiids as a more
  consistent, baseline food source in seasons when other prey becomes less available".
- Table 2 (stock-weighted annual means): total fullness 0.37-1.19 %BW, mean 0.785; euphausiid %W 22.9-65.0.
  Euphausiid partial content = %W x fullness: 0.14-0.50 %BW, mean 0.31 [D] (`tables/q1_buckley_pf_stdout.txt`).
- Weight: weak. Six survey years, one of six size classes, euphausiid index includes hyperiids, daytime stomach
  contents (snapshot), not rations. No Cmax comparison possible from the paper.

### S2. Tanasichuk, Ware, Shaw & McFarlane 1991, CJFAS 48:2118-2128, doi:10.1139/f91-251 [A]
Pacific hake and spiny dogfish, SW Vancouver Island, August 1983 and 1985-89, June-October 1988.
- "Summer ration estimates were higher for hake feeding on the continental shelf than for fish feeding along the
  shelf-break. Daily ration declined over summer because the euphausiid ration dropped, even though the herring
  ration increased."
- "They evacuated prey linearly over time and digested euphausiids most rapidly."
- Euphausiids are 93 % of the hake daily ration (Tanasichuk 1998, MEPS 173:181, p. 182 [F], citing this paper).
- The ration values (%BW/d) were not obtained (paywalled; two routes tried).
- Reading [J]: within a season the hake's total ration fell together with its euphausiid component, and herring did
  not make up the difference. The abstract gives no krill measurement, so the cause (krill availability, krill size,
  hake condition, season) is not established.

### S3. SW Vancouver Island krill availability and hake consumption across years
- Tanasichuk 1999, Fish Oceanogr 8:150-156, doi:10.1046/j.1365-2419.1999.00100.x [A]: "Hake persisted in selecting
  krill longer than 17 mm and consistently preferred T. spinifera, even though there was a substantial reduction in
  euphausiid abundance ... The biomass of euphausiids available to hake after 1993 averaged 27% of that for 1991,
  the pre-ENSO year, and was 12% of the 1991 value in 1996."
- Tanasichuk 1998, MEPS 173:181-195, doi:10.3354/meps173181 [F] (`data/q1_tanasichuk1998_meps/`), p. 194: "I find
  that prey availability for hake has declined since 1991. In 1996, it was 0.13 of the 1991 value." Basis:
  length-specific euphausiid biomass from net samples in Barkley Sound.
- Tanasichuk 2002, Fish Oceanogr 11:18-30, doi:10.1046/j.1365-2419.2002.00185.x [A]: "I found significant
  interannual variations in daily ration of the dominant planktivorous fish species, but euphausiids remained the
  most important prey. Pacific hake ... fed on larger (>17 mm) T. spinifera, even though the biomass of this part of
  the euphausiid biomass decreased by 75% between 1991 and 1997".
- Ressler, Holmes, Fleischer, Thomas & Cooke 2007, Mar Fish Rev 69(1-4):1-24 [F] (`data/q1_ressler2007_mfr/`), p. 13,
  reporting Tanasichuk 2002 [S]: "Tanasichuk (2002) reported that Pacific hake daily ration varied significantly among
  years during 1985-98 in August off Vancouver Island, but the proportion of the daily ration composed of
  euphausiids (mostly Thysanoessa spinifera >17 mm in length) averaged 89% even as euphausiid prey became less
  abundant." Same page: "The daily ration for juvenile and adult Pacific hake has been estimated variously between
  0.4 and 3.5% of body weight per day in the literature (Francis, 1983; Livingston, 1983; Rexstad and Pikitch, 1986;
  Tanasichuk et al., 1991)."
- Robinson 2000, J Plankton Res 22:1649-1662, doi:10.1093/plankt/22.9.1649 [A]: August consumption by the seven most
  abundant pelagic fish 1985-1997 from "Empirical data on fish feeding rates, diets and biomass"; "The total
  estimated consumption of euphausiids per tonne of fish averaged 0.93, and there was no significant trend in the
  time series. In addition, there was no significant relationship between the total fish community consumption or
  euphausiids consumed per tonne of fish and euphausiid biomass, and there was no change in the fraction of
  euphausiids in the diet of small or large Pacific hake", while adult euphausiid biomass showed "an apparent
  fivefold decline ... in the 1990s".
  - Not verified: whether the per-year hake ration was measured each year or carried as a constant. If constant, the
    flat result is built in. The figure is for the whole fish community (hake 60 %, herring 30 %). The unit and
    period of "0.93" are not stated in the abstract; not converted.
- What these show together: the krill SHARE of the hake ration stayed ~0.89 while the krill available to hake fell
  to 0.12-0.27 of 1991. The hake ration itself varied significantly among years (Tanasichuk 2002), and whether it
  tracked krill biomass is not known from what was obtained. Robinson's flat per-tonne consumption is abstract-only,
  community-wide and possibly built from fixed rations. No ration level relative to a Cmax can be computed.

### S4. Rexstad & Pikitch 1986, Fish Bull 84:947-956 [F] (`data/q1_rexstad1986_fb/`)
Pacific hake, Washington-Oregon, August 1983 (El Nino).
- Consumption was not measured. p. 949: "the percent of total body weight consumed daily was calculated based on the
  equations of Francis (1983)". Table 1 %BWD by size class 35-39 to 55+ cm: 1.10, 0.98, 0.84, 0.65, 0.40. These are
  model rations, not observations; they are not used for any Cmax fraction.
- On satiation in a low-krill year, p. 954: "Miller et al. (1984) noted a decline in the relative abundance of
  T. spinifera off the Oregon coast during 1983 in comparison with other years. Thus, feeding to satiation during
  evening hours may have been impossible; consequently, feeding occurred whenever euphausiids were encountered.
  Additional circumstantial evidence of aberrant feeding behavior of Pacific hake in 1983 is their severely
  depressed growth (Francis and Hollowed 1985)." The authors' interpretation, not a measurement.

### S5. Livingston 1983, Fish Bull 81:629-636 [F] (`data/q1_livingston1983_fb/`)
Pacific hake ("Pacific whiting"), Washington-Oregon, summer 1967. Diet by weight, summer 1967 (Table 3):
euphausiids 72.2 % (Oregon, 491 mm) and 90.2 % (Washington, 503 mm; 35.3 + 0.3 + 54.6). No krill density measured.
- Diel series, 14-15 July 1967, 251 stomachs with food, p. 634: "Stomach content weight per fish weight is highest
  at 1800 h, with a value of 2.5%".
- Daily ration from the pooled 1967 set (1,749 stomachs, Oregon and Washington; Elliott & Persson, exponential
  evacuation for small crustaceans, 8.2 C), p. 635: "The calculated daily ration of Pacific whiting, using Equations
  (1) and (2) on the 1967 data, is equal to 2.5% body weight/d for an average Pacific whiting size of 500 mm.
  Although this estimate is probably high, due to the use of the evacuation rate only for small crustaceans, it is
  comparable to other estimates."
- Cod functions as proxy (no hake Cmax) at 0.7-1.0 kg (weight of a 500 mm hake is a judgement; the paper's
  length-weight equation is not legible in the scan) and 8.2 C [D, J]: Hansson 2.04-2.19, Jobling 2.99-3.21 %BW/d.
  - Total ration: 1.14-1.23 (Hansson), 0.78-0.83 (Jobling) of Cmax.
  - Krill part at 0.72-0.90 of the diet: 0.82-1.10 (Hansson), 0.56-0.75 (Jobling).
  - Read as "of the order of the cod-scale Cmax", not as a precise fraction.

### S6. Dolgov, Yaragina, Ajiad, Mehl & Bogstad 1992, "Daily rations of cod from the Barents Sea", 5th PINRO-IMR
Symposium (IMR, Bergen; report, 15 p.), hdl:11250/107818 [F] (`data/q1_brage_cod1992/03.pdf`; authors and venue from
the NVA record; the scanned byline is garbled)
North-East Arctic cod, 24-h stations, summer 1988-1989, bottom trawl every ~4 h, 30 cod per catch. Table 1:
- Rybachy Bank, 27-28 June 1988, 1.4-1.6 C, 192 stomachs.
- Demidov Bank, 5-8 July 1988, 1.2-1.4 C, 390 stomachs.
- Goose Bank southern slope, 29 July - 2 August 1989, 1.2-2.3 C, 634 stomachs.
- Murman shallow, 5-7 August 1989, 2.3-2.6 C, 420 stomachs.
- Kanin-Kolguev shallow, 13-16 August 1989, 2.17-2.38 C, 569 stomachs.
Table 2 gives mean weights of 1042-2067 g, but the columns are not clearly assigned to stations in the scan;
conversions use 1.0-2.1 kg.
- Abstract, p. 1: "The daily rations of cod which fed on Euphausiacea, Themisto, fish and shrimps varied from 2.10 to
  3.7 % and from 0.28 to 0.99 % of the body weight in 1988 and 1989, respectively."
- Krill-dominated, Rybachy Bank 1988, p. 4: "Euphausiids formed the bulk (84.1% in terms of weight) of the cod diet
  ... The daily rations as calculated using the 3 methods constituted 2.20, 2.10 and 0.64% of the body weight,
  respectively." The three methods are Fortunatova, Bajkov and the dos Santos model.
- Acoustically recorded krill aggregations, Murman shallow 1989, p. 3: "interaction between two different water
  masses ... which induced formation of euphausiid concentrations to be recorded by the acoustic devices in the
  feeding ground. Cod were registered as large individual schools and their behaviour was determined by migrations
  of euphausiids." The diet was euphausiids 48.4 %W, fish 31.3 %, Decapoda 16 %. p. 4: "The daily ration as
  calculated by the 3 methods constituted 0.33, 0.28 and 0.48% of the body weight, respectively". No krill density
  value is given.
- Themisto (large crustacean zooplankton) stations:
  - Demidov Bank 1988, Themisto 74.1 %W, p. 3: "The daily rations calculated using the 3 methods constituted 3.76,
    3.61 and 0.74% of the body weight, respectively."
  - Goose Bank 1989, Themisto 64.5 %W, p. 3: "0.91, 0.85 and 0.99% of the body weight".
- Authors' reading, p. 5: "The high values of cod daily rations in the summer of 1988 can be caused by inadequate
  feeding on capelin due to a reduction in the capelin stock, which resulted in cod feeding on other preys."
- Conversions [D] (`tables/q1_cod_cmax_stdout.txt`). The prey part assumes the stomach weight share equals the
  ration share. Values are Hansson / Jobling fractions over 1.0-2.1 kg.
  - Method check: at Demidov the Fortunatova and Bajkov rations are 3.3-4.0 (Hansson) and 2.3-2.8 (Jobling) times
    Cmax for the total ration. Several times Cmax at 1.3 C is not credible as a 24-h intake. Those two methods are
    read as inflated [J], and the dos Santos model is the lead value.
  - Rybachy 1988, krill part, dos Santos: 0.54 %BW/d = 0.48-0.56 (H), 0.34-0.39 (J). The Fortunatova/Bajkov
    values, 1.58-1.91 (H) and 1.11-1.35 (J), are probably inflated.
  - Murman 1989, krill part: 0.14-0.23 %BW/d = 0.11-0.22 (H), 0.08-0.15 (J). The total ration is 0.23-0.45 (H),
    0.16-0.32 (J); all three methods agree within 1.7x.
  - Demidov 1988, Themisto part, dos Santos: 0.55 %BW/d = 0.50-0.58 (H), 0.35-0.41 (J).
  - Goose Bank 1989, Themisto part: 0.55-0.64 %BW/d = 0.48-0.64 (H), 0.34-0.45 (J).

### S7. Yaragina & Marshall 2000, ICES JMS 57:42-55, doi:10.1006/jmsc.1999.0493 [F] (`data/q1_yaragina2000_icesjms/`)
North-East Arctic cod liver condition index (LCI) vs capelin. Krill was not measured or named in the analysis.
- p. 50: "The annual mean LCI decreases rapidly when the biomass of Barents Sea capelin stock falls below
  approximately one million tonnes or a capelin:cod biomass ratio of 1:1."
- p. 52: "Both low and high annual mean LCI were observed for cod in the 41-60 cm size range when the frequency of
  occurrence of capelin in cod stomachs was low ... High LCI when capelin are scarce suggests that these cod can
  utilize alternative prey to achieve values comparable to those attained when capelin are abundant. For these
  length classes there was an upper boundary in annual mean LCI between 6 and 7%". Larger cod "are, however, more
  dependent than small cod on capelin to achieve these high LCI values."
- Reading [J]: alternative prey did not, on average, replace the energy lost with capelin, though for 41-60 cm cod it
  sometimes did. Krill was among the alternatives (S6, S8). No krill-specific ration.

### S8. Dalpadado & Bogstad 2004, Polar Biol 27:140-154, doi:10.1007/s00300-003-0561-5 [S]
Full text and abstract not retrieved (Springer paywall; OpenAlex has no abstract). Paraphrase of abstract listings
returned by a web search (not a verbatim quote):
- age 0-1 cod ate mostly krill and amphipods (up to 70 % of the diet);
- for 2-year-olds, krill and amphipods became more important when capelin was scarce;
- the total fullness index was positively related to capelin in the diet and to growth.
Reading [J]: for juvenile cod, krill is the fallback when capelin is scarce. Not a krill functional response.

### S9. Andersen 1999, J Fish Biol 54:287-301, doi:10.1111/j.1095-8649.1999.tb00830.x [A]
Whiting gastric evacuation, lab: "Krill Meganyctiphanes norvegica was evacuated at a rate similar to fish prey of
equal energy density, while the evacuation of brown shrimp Crangon crangon took almost twice as long." Relevance
[J]: the digestive ceiling of a whiting on krill equals that on fish of equal energy density, so gut processing does
not cap a krill-only ration below the fish-prey Cmax. The abstract has no ad-lib krill feeding data.

### S10. Other numbers met
- Juvenile European hake (9-12 cm TL), NW Mediterranean, 24-h survey (Modica et al. 2011, MEPS 440:177-189,
  doi:10.3354/meps09341) [A]:
  - euphausiids eaten at night, mysids by day;
  - daily ration 1.5 %BWW (exponential model) or 4.09-5.09 %BWW (square-root model);
  - no prey density, no Cmax.
- Blue whiting, Norwegian Sea Apr-Sep, mostly euphausiids:
  - realised ration 0.02 g/g/d, back-calculated from growth with a bioenergetics model (Bachiller et al. 2018, PLoS
    ONE 13:e0190345, p. 13) [S], via research_notes/"Fiskgrupper i rutan"/planktivores/evidence_cmax_energy.md 2.5;
  - against the cod-proxy Cmax at 100 g there (2.7-3.2 %/d) that is ~0.6-0.75 [S, D there];
  - a growth-based ration measures need, not intake vs krill density.
- Barents Sea 2015, 27 000 stomachs of 70 species (Skjoldal, Eriksen, Ono et al. 2025, J Fish Biol 106:1734-1751,
  doi:10.1111/jfb.16058) [A]:
  - "The long tail in the high end reflects high variation with no clear sign of a plateau, as could be expected
    from the concept of a 'full stomach'";
  - mean and median content of non-empty stomachs 2.0 and 1.1 %BW;
  - "this suggests relatively low feeding rates of the various fish predators but sufficient to meet their energy
    demands";
  - not krill-specific.
- Holt et al. 2019 (ICES JMS, Barents cod diet; author manuscript `data/q1_holt2019_icesjms/`) [F, skimmed]: diet
  composition only, euphausiids important to cod < 30 cm; no ration vs krill density.
- Aydin 2014 PICES presentation (`data/q1_aydin2014_pices/`) [F]: BSIERP FEAST model with "emergent" functional
  responses; no empirical pollock-krill functional response. Not used.

## Verdict

(a) Field or lab functional response of a gadoid or hake on euphausiids.
- None found. No study fits intake against a measured krill density for any gadoid or hake, in the field or the lab.
  The closest lab study (Andersen 1999) measures evacuation, not intake vs density.
- Closest field contrasts, all indirect:
  - Vancouver Island Pacific hake (S2-S3) [A, S]: the krill share of the ration stayed ~0.89 over a 4-8x fall in the
    krill available to hake. The ration itself varied significantly among years, with no link to krill shown. The
    flat community consumption per tonne (Robinson 2000) is abstract-only and may rest on fixed rations. This shows
    hake kept eating krill as krill declined; it does not show a ration flat in krill density.
  - Within a season the hake ration fell with its krill component (S2) [A]; the cause is not established.
  - Bering Sea pollock (S1) [F]: euphausiid stomach content 0.14-0.50 %BW over 25 years [D]. Only fish >= 60 cm
    tracked acoustic euphausiid biomass (r 0.89, 6 years). This hints at intake rising with density in the largest
    fish, but rests on one of six length classes. In 40-49 cm pollock euphausiids behave as a fallback food.

(b) Maximum observed krill intake as a fraction of Cmax.
- High side:
  - Barents cod, 84 % krill (Rybachy 1988, 1.5 C): 0.48-0.56 of Hansson Cmax and 0.34-0.39 of Jobling Cmax, by the
    dos Santos model [D]. The two Russian methods give >1 but are inflated (S6 method check).
  - Themisto at the same lab and method: 0.35-0.64 [D].
  - Pacific hake, 72-90 % krill, 8.2 C: 0.56-1.10 of a cod-proxy Cmax [D, J]. Livingston calls his 2.5 %BW/d
    "probably high".
  - Caller's ICES pools (`evidence.md` A2): krill-dominated whiting 0.90, saithe 0.58 of GER Cmax, top pools 1.8-2.3.
  - Held Onsrud 2004 range: 0.1-0.7.
- Low side, the one observation that fits a low plateau: Murman shallow 1989, where krill concentrations were
  recorded acoustically and cod schools followed them. The krill ration there was 0.08-0.22 of Cmax (total
  0.16-0.45) [D]. It does not decide the question:
  - the diet was mixed (48 % krill, 31 % fish, 16 % shrimp);
  - no density value is reported, so "dense" is the authors' description, not a number;
  - it is one 2-day station.
- Summary: where a gadoid feeds mainly on krill, field rations reach ~0.35-0.6 of a cod Cmax by the most
  conservative method, and ~1 in the hake and ICES-pool estimates. One acoustically documented krill station gave
  0.1-0.2.

(c) Per-prey plateau, time/risk limit, or demand-limited intake [J].
- The data lean against a krill-specific ceiling of 0.1-0.25 of Cmax as a general rule:
  - single-prey krill or Themisto feeding repeatedly reaches 0.35-0.6 of Cmax by conservative methods (S6), and more
    by others (S5, caller's A2);
  - whiting evacuate krill as fast as fish of equal energy (S9).
- They do not exclude a per-prey plateau near 0.3-0.5 for cod on krill at 1-2 C. The Murman station shows that a
  low krill ration also occurs beside acoustically visible krill.
- Low population means (0.1-0.3) are explained at least partly by encounter and patchiness:
  - krill is met in some hauls, seasons and depth windows and not others (caller's A3);
  - krill serves as a fallback when preferred fish prey is scarce (S1 40-49 cm pollock, S6 1988 reading, S7, S8).
- Demand-limited intake is not tested directly by any source. Skjoldal et al. 2025 read typical Barents fish
  feeding as low but sufficient for energy demand [A], which is compatible with it.
- Two points suggest availability limits intake at low krill: Rexstad's 1983 interpretation (S4) and the
  within-season decline (S2). Both are weak.
- Diel window: hake and cod feed in bouts tied to krill vertical migration (S5, S6). A daily time window exists, but
  it did not keep the krill stations below ~0.35-0.6 of Cmax.

(d) Gaps.
- No paired measurement of a gadoid ration and the local krill density (ind/m3 or g/m2) at haul or patch scale
  was found. The Murman 1989 station had acoustics but reports no density.
- No Cmax or ad-lib lab ration on krill exists for Pacific or European hake, saithe, whiting or Norway pout. Hake
  conversions rest on a cod proxy [J]. The two cod functions differ 1.4x at 1-2.5 C.
- Not obtained (paywalled):
  - Tanasichuk et al. 1991 ration values;
  - Tanasichuk 2002 yearly ration table;
  - Robinson 2000 inputs (were rations measured per year?);
  - Dalpadado & Bogstad 2004.
- The first three would settle whether the Vancouver Island ration (not just its krill share) was flat in krill
  biomass.
- Russian 24-h ration methods disagree 3-5x at some stations (S6). Only the dos Santos model values are used as
  leads.
