# Field energy costs and seasonal sinks: saithe, whiting, hake (literature)

Literature agent, 2026-11-20. Research only: no library, code or resume edit. Arithmetic is in
`scripts/costs.py` (`python3 -I`), with output in `tables/costs_stdout.txt`. Raw rows are in
`tables/costs_sinks_raw.csv`. Downloads are under `data/<id>/` and are untrusted; do not commit them.

**Tags**
- [F] full text read with pdftotext.
- [A] abstract only (OpenAlex or Crossref).
- [S] secondary: a search-engine snippet, a summariser, or an earlier repo note.
- [D] derived here, arithmetic in `costs.py`.
- [J] judgement.

**Units.** 1 mg O2/kg/h = 0.0816 MJ/t/tick (13.6 kJ/g O2, 6 h).

**The model's charges** (without SDA 0.12 of intake and without growth):
- whiting: 5.8 MJ/t/tick at 7.4 C (RM 3.87 x 1.5);
- water column: 7.04 MJ/t/tick at 8.4 C (RM 3.29 x 2.14).

**Already known, cited only:**
- Alewijnse 2022 otolith dataset: E2 in `Fältmetabolism kolja och plattfisk`.
- Chung 2019 cod calibration.
- Andersen & Riis-Vestergaard 2004 abstract.
- SMS rations.
- `Lek i rutan`: gonads, timing, Hislop 1975, Skjaeraasen 2017.

## Bottom line

1. **No new field metabolic rate (otolith, telemetry-based or heart-rate) exists for saithe, whiting or
   hake** beyond the five Channel whiting and five Channel saithe in the Alewijnse 2022 dataset. I searched
   OpenAlex and the web: the otolith-FMR literature for North Sea gadoids other than cod covers plaice,
   haddock, long rough dab, witch, dab and lemon sole only.
2. **Saithe / water column: the model is near the field value, not far below it.**
   - Otolith FMR at 8.4 C: 6.6 MJ/t/tick, 5-fish range 4.2-8.3.
   - Model on the same footing: 7.04 + SDA (about 1.3 at the SMS ration) = 8.3.
   - Ratio field/model: about 0.9 (range 0.6-1.5) on the non-SDA charge. The 0.60-1.18 spread of five
     fish does not bound the top, because the cod calibration may read low; the only field-derived upper
     constraint is the kappa_wild 3.27 x SMR of Andersen & Riis-Vestergaard (10.8 MJ/t/tick, ratio 1.53).
   - Including SDA: 0.79 (0.51-1.0).
   - Ration-route closure, R = A x SMS ration energy - P x body energy [D/J]: 6.2-7.3 MJ/t/tick (A
     0.7-0.8, P 0.42 /yr, E 5 080 MJ/t, gonads ignored). It lies between the otolith 6.6 and the model's
     8.35 with SDA. SMS rations read low for haddock (`Fältmetabolism`), so treat it as a lower bound.
   - The central A&RV multiplier (2.0-2.5) gives the same ratio, but the model's 2.14 was built from it,
     so that is a consistency check, not a test.
   - Verdict: near the model. "Far above" is unsupported but not excluded.
3. **Whiting: undetermined.**
   - The only FMR comes from 94 g Channel fish at 17 C, extrapolated 10 C down. It gives 3.9-4.5
     MJ/t/tick, which is 1.0-1.16 x the model's RM.
   - That is a floor signal, as the cod calibration also gave for plaice. No conclusion about "below the
     model" follows from it.
4. **Seasonal sinks, new sizes:**
   - Saithe spawning-migration locomotion: paid outside the box, at most about 3 % of the annual ration
     per tonne of the group.
   - The saithe Q1 feeding dip in SMS is larger, about 11 % of the annual ration as missing intake.
   - Summer warming is already charged by the engine's Q10.
   - Hake spawns on income (current intake), not on reserves.

---

## Part 1. Energy costs in the field

### Saithe (water column 0.797)

**Standard metabolic rate (SMR)**
- Steinhausen, Steffensen & Andersen 2005, Mar Biol 148:197-204, doi:10.1007/s00227-005-0055-9 [S]: "the
  extrapolated SMR were similar for saithe and whiting, whereas AMR was twice as high for saithe". Whiting
  SMR is about 59.2 mg O2/kg/h, i.e. 4.83 MJ/t/tick ([S], earlier repo note `Vintermetabolism sill och
  tumlare/q10_metabolism.md`). Temperature and size are still not recovered; four routes failed (paywall,
  no OA copy, abstract elided in Crossref and Semantic Scholar). **Gap.**
- Tytler 1978 (in McLusky & Berry, Physiology and behaviour of marine organisms, 82-93) has gadoid
  swimming-metabolism data and is cited by Steinhausen 2005 [S, Crossref reference list]. Not obtained.
- Earlier repo value (`Fiskgrupper i rutan/params_predatory_fish.md`): saithe SMR 3.5 MJ/t/tick at 8.4 C
  [S].

**Activity and conversion efficiency**
- Andersen & Riis-Vestergaard 2004, CJFAS 61, doi:10.1139/f04-153 [A, cited]. Activity multiplier
  2.0-2.5 x SMR at optimum cruising speed:
  - range 1.44-3.27 with kappa_wild;
  - 1.6-6.7 with kappa_lab (the abstract's own reading of the lab figure is that lab saithe stored lipid
    cheaply);
  - kappa_wild 0.43-0.50.
  - Full text: paywalled, no OA copy (Unpaywall). **Not obtained.**
- Andersen & Riis-Vestergaard 2003, J Fish Biol, doi:10.1046/j.1095-8649.2003.00030.x [A]. Net food
  conversion efficiency from growth experiments on natural prey: kappa = 0.272 C^0.18 (saithe) and
  0.426 C^0.11 (whiting). No significant effect of temperature or body size. The units of C are not in the
  abstract, so this cannot be used numerically.

**Cost of transport (COT) at the optimum speed Uopt** [S, attribution unconfirmed]
- Snippet values: saithe Uopt 1.4 BL/s, COT 0.14 J/N/m; whiting Uopt 1.0 BL/s, COT 0.15 J/N/m. The snippet
  did not confirm that these come from Steinhausen 2005.
- Continuous cruising at Uopt would cost [D, assumed lab lengths]:
  - 30 cm saithe: 153 mg O2/kg/h = 12.5 MJ/t/tick;
  - 50 cm saithe: 254 mg O2/kg/h = 20.8 MJ/t/tick.
- That is 1.8 x and 3.0 x the model's 7.04 (whiting 25 cm, 7.9: 1.4 x its 5.8) [D], so it is an
  **upper bound only**. Wild saithe rest at
  night and are less active in winter (next item).

**Activity pattern, data storage tags (DST)**
- Armannsson & Jonsson 2012, ICES JMS 69:1372, doi:10.1093/icesjms/fss121 [F, qualitative; Figure 3 not
  read]. 51 DSTs, 42-83 cm (mean 54 cm), Iceland, hourly depth: "The average hourly depth change was
  highest during daylight hours, and nocturnal activities were much reduced in winter"; "The activity is
  lowest in July ... the month of the year with the shallowest median depth". Median depth was 140 m in
  March-April and < 50 m in summer.
- Skilbrei & Otterå 2016, ICES JMS 73:1186, doi:10.1093/icesjms/fsv261 [A]. Acoustic tags at 15 fish farms
  over about 2 years: "reduced activity level at night and seasonality in swimming depth"; 25-50 m in
  summer, 60-90 m in winter.
- Neither study gives a speed or an energy figure. **No accelerometer or tail-beat telemetry exists for
  wild saithe** [J, after searching].

**Field metabolic rate (otolith)** [S, cited dataset; D conversion]
- Alewijnse 2022 dataset, saithe: n = 5, Channel, 11.1 C, 1.9-3.9 kg, C_resp 0.143 (0.106-0.164).
  With the Chung 2019 cod calibration this is VO2 101 (65-128) mg O2/kg/h.
- At 8.4 C:
  - Q10 2.3: 80.6 mg O2/kg/h = **6.58 (4.24-8.32) MJ/t/tick**;
  - Q10 2.7: 6.30 (4.06-7.97) MJ/t/tick.
- FMR/SMR [D]: 6.58 / 3.5 (repo saithe SMR) = **1.9**, inside Chung's cod range of 1.9-3.5.
- The ranges are the minimum and maximum of 5 fish, not confidence intervals. The cod calibration read
  plaice FMR at about SMR (`Fältmetabolism`), so this FMR may be biased low [J].

**Bioenergetic parameter sets**
- No Wisconsin/FB4 set for saithe was found. The AmP (DEB) entry for *Pollachius virens* uses generic
  pseudo-data (p_M 18 J/d/cm3 is the generic prior, not a measurement) [S, summariser]. Not usable.
- Cormon et al. 2016, ICES JMS 73:1370, doi:10.1093/icesjms/fsw050 [F]. Saithe consumption is SMS-style,
  r = c_q w^d_q with quarterly coefficients fitted in the hindcast. No new physiology.

**Verdict, saithe.** At 8.4 C the field cost is **near the model's**.
- Otolith FMR / model non-SDA charge: 0.93 (five fish 0.60-1.18). Allowing for a low-reading calibration,
  about 0.9 (0.6-1.5); the top is the kappa_wild 3.27 x SMR bound (10.8 MJ/t/tick).
- Ration route (A 0.7-0.8, P 0.42, E 5 080): 6.2-7.3 MJ/t/tick [D/J], a lower bound.
- FMR / (model + SDA at the SMS mean ration): 0.79 (0.51-1.0). The SDA is 0.12 x 10.9 MJ/t/tick of intake,
  with prey E 5.5 MJ/kg [J].
- The otolith FMR already contains SDA and the synthesis cost of growth. Read strictly, the model's total
  is therefore at or slightly above the central field value [D/J].

### Whiting

- **SMR** [S]: 59.2 mg O2/kg/h, temperature and size unknown (see saithe). If measured at 10-12 C, it
  corresponds to 40-47 mg O2/kg/h at 7.4 C with Q10 2.3 [D]. The model's RM of 3.87 equals 47.4 mg
  O2/kg/h, which is consistent.
- **Activity in tanks.** Hammer 1997 (Arch Fish Mar Res 45:1-16) and Hammer & Schwarz 1996 (44:13-32) on
  the spontaneous swimming of juvenile whiting and the cost of prolonged swimming are cited in Steinhausen
  2005 [S, Crossref reference list]. Not obtained.
- **Otolith FMR** (Alewijnse) [S, D]: n = 5, Channel, 17.2 C, 65-199 g, C_resp 0.162 (0.139-0.216), VO2
  125 (96-250) mg O2/kg/h.
  - At 7.4 C: 4.50 (3.48-9.01) MJ/t/tick with Q10 2.3; 3.85 (2.97-7.70) with Q10 2.7.
  - FMR / model RM: 1.16 and 0.99.
  - The top fish sits near the calibration asymptote (0.243).
- **No telemetry** (speed, DST) was found for whiting in the North Sea or Skagerrak.
- **Verdict, whiting: undetermined.**
  - Taken at face value, the field FMR is 0.78 (0.60-1.55) of the model's 5.8.
  - However, FMR equal to SMR is not a credible field value for an active gadoid. Together with the
    extrapolation across 10 C from small Channel fish, this makes it a floor, not an estimate [J].
  - The model's 1.5 x RM stays a judged value. Nothing here supports raising or lowering it.

### Hake (water column 0.048)

- **Respirometry**: none found for juvenile or adult *M. merluccius*. The only temperature physiology found
  covers yolk-sac larvae, at 10-22 C [S, search]. **Gap.**
- **Bioenergetics**: Bodiguel et al. 2009, J Sea Res, doi:10.1016/j.seares.2009.02.006 [title and
  Semantic Scholar tldr only]. A dynamic, mechanistic PCB bioaccumulation model of European hake that
  relates contamination to diet, metabolism, reserve and reproduction, so it should hold a metabolic
  parameter set [J]. Paywalled, not obtained. **Gap.**
- **Activity**: de Pontual et al. 2012, J Fish Biol, doi:10.1111/j.1095-8649.2012.03345.x [A]. DSTs
  show diel vertical migration, "frequent vertical movements across the thermocline", "rapid temperature
  changes > 7 C", and homing to feeding grounds. No energy figure.
- **Verdict, hake**: no field cost can be computed. Its weight in the group is 0.048.

---

## Part 2. Seasonal energy sinks in or near the box

### Saithe

**Spawning migration**
- Homrum et al. 2013, ICES JMS 70:782, doi:10.1093/icesjms/fst048 [F, text; Figure 2 not read]:
  - "The migrated distance from tagging site increased with increasing recapture length and age of
    saithe".
  - "For adult saithe tagged in Norwegian waters (length at tagging > 60 cm) there was a seasonal pattern
    in the distance from tagging site, with the farthest distances observed in February-March and the
    shortest in the late summer months ... saithe come back to the same feeding grounds".
  - 6.6 % of adults tagged in Norway were recaptured outside Continental waters.
  - Fish < 50 cm moved the shortest distances.
  - The February-March signal is from adults tagged mainly in northern Norway, whose spawning grounds lie
    farther south. Immature saithe tagged just north of 62 N were often recaptured south of it (Jakobsen
    1981, cited there), and North Sea tagging is "scarce" (Newton 1984; Fontaine 1985) [F]. **No tagging
    source gives the box-to-shelf-edge distance**; the distance below is geometry only.
- **Distance** [D, endpoints J]. Great-circle distance from the box (58.05 N, 10.6 E) to the spawning
  shelf edge:
  - 397 km to 59 N, 4 E;
  - 486 km to 60 N, 3 E;
  - 585 km to 61 N, 2 E (Tampen).
  - Round trip: **800-1 170 km**.
- **Extra locomotion per migrating fish** [D, adult COT J]: gross COT x g x distance, minus the model's
  routine charge (28.2 kJ/kg/d) over the travel days.
  - COT 0.06-0.10 for 60-80 cm adults [J; the 0.14 snippet value is for smaller fish]: **about 0 to 650
    MJ/t of migrants**.
  - Slow travel at 0.5 m/s gives about 0, because the routine charge already covers it. At 0.84 m/s it is
    160-650.
  - Upper bound at COT 0.14 and 0.84 m/s: 1 080 MJ/t.
- **Per tonne of the group** [D]: multiply by the mature share of saithe biomass, 0.25-0.6 (`Lek i rutan`),
  and the saithe share, 0.797.
  - About 0-310 MJ/t/yr, upper bound 520.
  - Relative to the SMS annual ration (0.79 %BW/d x 5.5 MJ/kg = 15 900 MJ/t/yr): **0-2 %, at most about
    3 %**.
- **This cost is paid outside the box.** Adults are an emigrating and immigrating window in the model, so
  it belongs to the exchange, not to the in-box cost [J].
- Gonads: females 400-920 MJ/t ([cited], `Lek i rutan`).

**Winter feeding depression**
- SMS rations, Q1-Q4: 0.44 / 1.27 / 0.67 / 0.79 %BW/d ([cited]).
- The Q1 shortfall against the annual mean is (0.79 - 0.44) % x 90 d x 5.5 MJ/kg = **about 1 730 MJ/t, about
  11 % of the annual ration** [D].
- This is a population mean that includes immature fish. It is missing intake, not an extra cost.
- In the box in Q1, 26-50 % of saithe stomachs were empty at all ages, immature fish included ([cited],
  `Lek i rutan`).
- Activity: reduced at night and in winter (Armannsson 2012 [F]; Skilbrei 2016 [A]), so winter costs fall
  with intake.
- No study separates the winter depression of immature saithe from that of spawning adults. **Gap.**

**Diet in the Norwegian Trench** (Bergstad 1991, Sarsia, doi:10.1080/00364827.1991.10413455 and
.10413456 [A])
- Surveys in winter, summer and autumn, 1984-87.
- "small saithe and blue whiting as predators and the euphausids as prey, and between saithe and Norway
  pout" are the strongest links.
- The seasonal diet percentages are in the full text, which was not obtained. **Gap.**
- Du Buit 1991: not found in OpenAlex before rate-limiting. **Gap.**

**Temperature**
- The water column sits on the `at_50m` layer, 6.9-11.7 C. With Q10 2.3, cost is x0.88 in winter and
  x1.32 in late summer relative to 8.4 C [D].
- **The engine already charges this** through `metabolism_q10`. It is not a missing sink.

### Whiting

- Gonads, timing, the Hislop 1975 weight loss and the condition cycle: all [cited], `Lek i rutan`. No new
  sources found.
- **Migrations**: Tobin, Wright, Gibb & Gibb 2010, Mar Biol, doi:10.1007/s00227-009-1387-7 ("The
  importance of life stage to population connectivity in whiting ... northern European shelf") and Hislop &
  MacKenzie 1976, ICES JMS, doi:10.1093/icesjms/37.1.98 ("Population studies of the whiting ... northern
  North Sea") are titles only, not read. No Skagerrak-specific whiting tagging was found. **Gap.**
- **Temperature**: the model puts whiting on the `bottom` layer (7-8 C all year), so there is no seasonal
  cost swing. Whiting that spend time higher in the water column in summer would pay more. No measurement
  was found.

### Hake

- **Reproductive energy is income-financed.** Domínguez-Petit & Saborido-Rey 2010, Fish Res 104:83,
  doi:10.1016/j.fishres.2009.09.002 [S, abstract via summariser]:
  - gonad energy correlates with liver and muscle energy;
  - batch fecundity correlates with liver energy density;
  - egg production "depends more on energy intake during spawning season and growth capacity than on
    energy reserves accumulated several months before spawning season begins".
- García-Fernández et al. 2020, MEPS 650:125, doi:10.3354/meps13420 [A]: in NW Spain hake spawn in
  winter-spring, summer and autumn; more and likely better-quality oocytes are produced in winter-spring.
- Box hake: a summer spawning minority vs "exclusively immature" fish in the shrimp survey ([cited], `Lek i
  rutan`).
- **Implication [J]**: a reserve drain is the wrong form for hake. Its gonad cost is paid out of intake
  while it spawns.
- Daily temperature swings above 7 C during diel vertical migration (de Pontual 2012 [A]) mean its
  experienced temperature is not one layer's temperature. At a group weight of 0.048 this is minor.

### Annual energy budgets (ration = respiration + growth + gonads + reserve change)

- The only one found is Andersen & Riis-Vestergaard 2004 for North Sea saithe (abstract; it is annual, not
  seasonal).
- The ration-route closure for the water column above (6.2-7.3 MJ/t/tick) is the annual budget built
  here from SMS ration and the box P [D/J].
- None was found for whiting or hake, and none with a seasonal split. **Gap.**

---

## Gaps (what would move the verdict)

1. Andersen & Riis-Vestergaard 2004 full text: the saithe field ration, growth and activity by
   season/size.
2. Steinhausen 2005 full text: temperature and size of the SMR, and the speed-to-VO2 regressions.
3. Any otolith FMR for saithe or whiting at North Sea or Skagerrak temperatures. None exists for hake.
4. Hake respirometry and the Bodiguel 2009 DEB parameters.
5. Bergstad 1991 seasonal diet table.
6. Whiting Skagerrak migration and winter feeding.

## Downloads (`data/`)

- `armannsson2012/` (OUP, read).
- `homrum2013/` (OUP, read).
- `cormon2016/` (OUP, read).
