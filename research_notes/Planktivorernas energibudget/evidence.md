# The planktivores' energy budget: evidence (2026-11-11)

Research subagent. Literature, published data already held in sibling notes, and arithmetic. Nothing in the
library, code, tests, manifest, resume or AGENTS.md was changed. No model was run and no value here was
chosen to match model output.

Tags: [F] full text read (local copy, path given); [A] abstract only; [S] second-hand (a sibling note or a
citing document; path given); [D] arithmetic here (`scripts/`, output in `tables/`); [J] judgement.
Units: MJ per tonne; tick = 6 h; 1 MJ/t/tick = 4 kJ/kg/d; 1 mg O2/g/h = 325.4 kJ/kg/d (13.56 J/mg O2);
1 mg O2/kg/h = 119.1 MJ/t/yr (13.6 J/mg, as in the sibling otolith note).

Sibling folders referred to (Swedish letters spelled out here; real names under research_notes/):
- KF = "Krillens flykt och energibudget"
- FM = "Faltmetabolism kolja och plattfisk" (a-umlaut)
- BB = "Bottenfiskens budget"
- BE = "Bioenergetik for alla fiskgrupper" (o-umlaut)
- PV = "Planktivorernas vinterkostnad"
- SV = "Sillens vinterbudget"
- SF = "Sillens vinterfoda" (o-umlaut)
- KB = "Kolmulens biomassa"
- FR = "Fiskgrupper i rutan" (planktivores/ holds the local PDFs used below)
- OP = "Oppen population" (O-umlaut)

## Search log

- G1. OpenAlex (api.openalex.org): HTTP 429 "Too Many Requests" on four requests on 2026-11-11 (three searches,
  one retry after a pause). No OpenAlex result is used; everything below is from local full texts and notes.
- G2. Unpaywall helper: not used. No DOI needed for a new download could be read from a local reference list
  except ones already held locally. No file was downloaded; `data/README.md` lists the local inputs.
- G3. FishBase `popqb.parquet` (FR/planktivores/data/fishbase_hf/): not readable here (no pyarrow in the
  environment); not used. The member Q/B values come from E22.
- G4. Not obtained: Johnstone, Wardle & Almatar 1993 itself (value via E5), Rudstam 1988 (parameters via E3),
  Slotte 1999, any Trisopterus respirometry, any herring otolith-carbon study (none in E16).

## Library and model terms

E1. `fgconfig/fg_library.yaml`, `planktivorous_fish` (lines 554-798) and its pairs, read 2026-11-11 [F]:
member shares herring 0.554, blue whiting 0.075, Norway pout 0.350, sprat 0.021; `maintenance_level` 0.079;
`start_fill` 0.50 0.36 0.26 0.20 0.32 0.81 0.94 0.96 0.88 0.88 0.76 0.68 (mean 0.629); `max_energy_reserve`
4990; `resting_metabolism` 8.0; `metabolism_q10` 1.5, `metabolism_t_ref` 14.0; comment: "t_ref 14 C because 8
MJ/t/tick (32 kJ/kg/d at rest) is the Rudstam/Rose et al. 2008 herring resting term at ~14 C, i.e. already a
summer value"; "an EFFECTIVE Q10 1.5 (1.4-1.6) for the one-layer group: 35 % of it is pout and blue whiting in
~7 C water all year (25 g pout costs 6.5-8.9 MJ/t/tick, gadoid SMR)"; `sda_fraction` 0.11; `feeding_cost` 1.65
("herring 1.57-2.12 (Rudstam activity at 1 BL/s for 15-25 cm), gadoid members 1.5 [J], cost-weighted 1.65
(1.40-2.19); was 2.0 with the SDA in it"); `movement_cost` 1.75; no `resting_cost` key (engine default 1.0);
`energy_content` 4000; `somatic_production` 0.66; `appetite: reserve_room`; `natural_mortality` 0.000186986
per tick (= 0.273 /yr). Pairs: `planktivorous_fish_preys_on_zooplankton` M2 1.3, assimilation 0.8;
`..._preys_on_krill` M2 1.67, assimilation 0.8. Predators' M2 on the group: benthivores 0.1023, water column
0.174, mackerel 0.0186, whiting 0.0545, porpoises 0.00946, seals 0.00501 (seabirds 0.0167 muted) = 0.364 /yr.
Prey energy: zooplankton 4500, krill 5000 MJ/t. `mareld2.yaml` mean_0_50m (Jan-Dec) 6.3 5.4 5.2 6.1 7.8 9.9
12.1 13.4 13.8 12.8 10.5 8.4 C (annual mean 9.33 C); at_50m 7.7-11.7; bottom 6.58-8.27 (annual 7.40).
`lib/diagnostics/field_diet.py` FIELD_QB: `'planktivorous_fish': (5.0, "members' Q/B, params_planktivorous_fish.md")`.

E2. KF summary.md, main-agent checks 3 and 4 [F]. Trained run (fix96 frozen policies, year 2): planktivore
ration 6.59 t/t/yr, fill 0.99, cap cut 0.51, cost split rest/eat/move 0.08/0.81/0.11, realised P/B 0.62, I
21 585, C 15 490, G 2 468, rest 3 626 MJ/t/yr. Closure table: field Q/B 5.00, I field 16 211, RM 9 737,
feeding_cost 1.65, G 2 640, refill 1 143 / 1 817, e* 0.43 / 0.32. Probe 40: ration 6.75-7.43, krill diet
share 0.47-0.51 against booked 0.40, krill M2 realised/booked 1.73-1.86.
- `diagnostics/scripts/closure.py` line 69: `m1 = float(p.get('natural_mortality', 0.0) or 0.0)` and
  `loss = m1 + m2_on + float(emig[1])`, used as a per-year rate in `refill = loss * s * ME`. The library value
  is per tick (E1: 0.000186986), so M1 (0.273 /yr) is missing from every group's refill in `closure.json`
  [D: `tables/budget_terms_stdout.txt` section 2].

## Metabolism

E3. FB4 parameter file (`FR/predators/data/physiology/FB4_Parameters_official.csv`) [F]:
- "Herring (adult),Clupea harengus,...,Rudstam 1988,3,0.642,-0.256,1,13,15,23,0.1,0.01,1,0.0033,-0.227,
  0.0548,0.03,0,9,15,0.13,0,3.9,0.149,0.175,1,0.16,0,0,1,0.1,..." i.e. REQ 1, RA 0.0033 g O2/g/d, RB -0.227,
  RQ 0.0548 /C (Q10 1.73), RTO 0.03 s/cm, RTL 9 C, RK1 15 cm/s, RK4 0.13, ACT 3.9, BACT 0.149, SDA 0.175 (of
  assimilated), FA 0.16, UA 0.10. Juvenile row: same respiration. Swimming speed VEL = RK1 W^RK4 at T <= RTL,
  ACT W^RK4 e^(BACT T) above; activity multiplier e^(RTO VEL).
- "Baltic herring (YOY) ... Arrhenius 1998; Rudstam 1988; Blaxter 1960 ... Respiration model parameters were
  taken from Rudstam 1989."
- "Atlantic cod (juvenile & adult), Hansson et al. 1996; Mateo 2007, ... 2,0.008,-0.172,1.88,21,24,...,1.25,
  NA,0.17" (REQ 2 Kitchell, RA 0.008, RB -0.172, RQ 1.88, RTO 21, RTM 24, ACT 1.25, SDA 0.17).
- [D] Rose resting at 129 g and 14 C = 32.0 kJ/kg/d = 7.99 MJ/t/tick, the library's 8.0.

E4. KB `fields/tables/season_cost_kappa_stdout.txt` [F]: "herring library 8.0 at 14 C = Rose et al. 2008
resting term at 129 g"; annual-mean member resting costs (MJ/t/tick): herring 5.69, sprat 10 g 10.16, blue
whiting 100 g Hansson 7.42, pout Hansson 25 g bottom 8.92, "pout measured-SMR reading 25 g, bottom 6.45"; best
effective Q10 on mean_0_50m 1.40-1.60 depending on the pout reading.

E5. Bernreuther M 2007, PhD thesis, Univ. Hamburg, manuscript 3 (Bernreuther, Herrmann & Temming,
"Conversion efficiency and temperature dependency of metabolic rate in juvenile herring"),
https://ediss.sub.uni-hamburg.de/bitstream/ediss/1765/1/DissertationMatthiasBernreuther.pdf
(`FR/planktivores/data/ediss1765/b.txt`) [F]:
- l. 4514-4527: "Starvation weight loss trials were performed using groups of herring maintained at 9.7, 14.2
  and 17.9 C"; "M = a * e b*T with b = 0.071 (Q10 = 2.0, J * g DW-1 *day-1) and b = 0.080 (Q10 = 2.2, J *
  fish-1 * day-1)". Fish 9-10 cm TL (1.0-1.5 g DW).
- l. 5113-5126: "The estimated oxygen consumption rates in our experiments ranged from 0.076 mg O2 g WW-1 h-1
  at 9.7 C to a maximum of 0.322 mg O2 g WW-1 h-1 at 14.2 C." "Rates of routine respiration of adult Atlantic
  herring (25.5 to 31.0 cm) were 0.093 mg O2 g WW-1 h-1 at 9.3 C (Johnstone et al. 1993). This rate was
  similar to the estimates for our smaller (10.5 to 10.8 cm, TL), juvenile herring at 9.7 C (0.076 and 0.133
  mg O2 g WW-1 h-1), which is surprising since specific rates usually decrease with fish size".
- l. 5156-5161: "Since the experimental fish in our study were either constantly swimming or exhibited
  spontaneous swimming activity, we did not measure standard metabolism. However, the probability of the
  previously mentioned metabolic depression was high, which would result in the measurement of a 'fasting
  metabolic rate' ... The fasting metabolic rate is intermediate between standard and routine metabolism."
- l. 4533-4539: herring bioenergetics models "suffer from 'parameter stealing' and 'species-borrowing'" and
  were "parameterized using data collected on ... blueback herring ..., alewife ... and Atlantic menhaden".
- [D] Johnstone's routine / Rose resting at 160 g and 9.3 C = 1.29; Bernreuther fasting / Rose resting at 6.6
  g and 9.7 C = 0.69 (0.50-0.87).

E6. Johnstone ADF, Wardle CS, Almatar SM 1993. "Routine respiration rates of Atlantic mackerel, Scomber
scombrus L., and herring, Clupea harengus L., at low activity levels." J Fish Biol 42:149-151 [S, via E5
reference list l. 750-752; the paper itself not obtained, G4]. Value: 0.093 mg O2/g WW/h at 9.3 C, adults
25.5-31.0 cm. 9.3 C is the box's annual-mean 0-50 m temperature (E1). [D] 30.3 kJ/kg/d = 7.6 MJ/t/tick.

E7. Meskendahl L 2013, PhD thesis, Univ. Hamburg (sprat metabolism; Meskendahl et al. 2010 Mar Biol
157:1917), https://ediss.sub.uni-hamburg.de/bitstream/ediss/4862/1/Dissertation.pdf [S, via PV evidence.md
1c, which read `FR/planktivores/data/ediss4862/d.txt` l. 1340-1690]: fish 3.11-9.71 g at 9-21 C; routine
"RR=0.074 WW 1.077 e 0.080 T", standard "RS=0.069 WW 1.073 e 0.078 T" (mg O2 ind-1 h-1); "the concept of standard
metabolism may not be meaningful in schooling planktivorous fish". [D] 8 g sprat on the 0-50 m table
(extrapolated below 9 C): routine 61, standard 56 kJ/kg/d annual mean (22 430 / 20 327 MJ/t/yr).

E8. Schurmann H, Steffensen JF 1997, J Fish Biol 50:1166 [A, via PV evidence.md 1b]: juvenile cod SMR "35.5,
57.0 and 78.2 mg O2 kg-1 h-1 ... at 5, 10 and 15 C"; test-fish mass not in the abstract (PV assumed 100-400 g);
cod SMR mass exponent 0.87 (Tirsgaard et al. 2015, [S] in PV). [D] scaled with -0.13 to pout 12 / 25 g on the
bottom table: 21.6 / 19.6 kJ/kg/d (7 866 / 7 150 MJ/t/yr); the Hansson et al. 1996 cod term (E3) at the same
sizes is 1.82-1.88 x that (40.4 / 35.6 kJ/kg/d), as PV found ("the Hansson 'resting' term sits at the measured
routine level, or above it").

E9. Alewijnse SR 2022 thesis dataset, `alewijnse_master_data.csv`, doi:10.6084/m9.figshare.22110335 (local
copy in FM/data/alewijnse2023/; licence conflict noted there, not copied here) [F, parsed]:
- 117 species. Planktivore members present: blue whiting (Micromesistius poutassou, 28 rows), sprat (Sprattus
  sprattus, 5 rows); also anchovy (Engraulis encrasicolus, 5). Herring and Norway pout are absent.
- Blue whiting at the Barents site 72.797 N 23.288 E (SUMIE, 2018, outer otolith edge), the site of the cod
  and haddock in FM: 18 fish, 102-304 g, 4.3-7.6 C, C_resp mean 0.163 (cod at the site 0.125, haddock 0.115).
  Other sites: 57.10 N -12.07 E (2006, 2009; 7 fish, 6.8-9.3 C) and 54.08 N -13.92 E (2006; 3 fish, 9.9-11.9 C).
- Sprat: C_resp 0.336-0.415 (whole otoliths, IFREMER, 4-25 g, 7.3-12.8 C); anchovy 0.295-0.368. Both above
  the cod calibration's asymptote 0.243 (E10), so no VO2 can be computed.
- [D] `tables/otolith_bw_stdout.txt`: blue whiting / cod FMR at equal mass and temperature 1.21 (95 % CI
  0.91-1.61; cod + blue whiting >= 100 g, T fitted, Q10 3.3), 1.26 (1.01-1.57; with haddock), 1.51 (Q10 2
  imposed), 1.26 (Q10 3). Mass slope b 0.03-0.06. Cod-anchored value at 100 g and 7.5 C 12 270-14 705 MJ/t/yr
  (CI 9 230-17 740). Face value per site: 15 342 (Barents, 5.5 C), 8 069 (57 N, 7.7 C), 10 692 (54 N, 10.9 C).

E10. Chung MT, Trueman CN, Godiksen JA, Holmstrup ME, Gronkjaer P 2019, Communications Biology 2:24,
doi:10.1038/s42003-018-0266-5 [S, via FM]: calibration C_resp = 0.243 (1 - exp(-0.0088 VO2)); cod field
metabolism 1.9-3.5 x SMR. FM's cod anchor: 11 338 MJ/t/yr at 1.2 kg and 7.5 C (2.2 x SMR) [S].

E11. Onsrud MSR, Kaartvedt S, Breien MT 2005. Can J Fish Aquat Sci 62:1822-1832, doi:10.1139/f05-090 [A, via
KF E11]: "Swimming speeds of deep-living fish, mainly Norway pout (Trisopterus esmarkii) and whiting
(Merlangius merlangus), were highest during the day (speeds centred around 14-16 cm s-1) and decreased
somewhat by night (modes around 10-11 cm s-1). Fish in the upper 10-30 m swam somewhat faster (speeds ranging
from 16 to 24 cm s-1)." [D] ~1 body length/s by day for 14-16 cm pout, night/day 0.7.

E12. Huse I, Ona E 1996. ICES J Mar Sci 53:863 [A, via SV evidence table row 12 and l. 274-275]:
"Overwintering NSS herring swim constantly at 0.25-0.42 m/s; not quiescent", "to offset negative buoyancy at
depth".

E13. Slotte A 1996 / 2000 (NSS herring fasting loss) [S, via PV 1a and SV]: ~2 fat-points lost Sep-Dec in
non-feeding NSS herring; mass-scaled to the site mix 10.8-14.4 kJ/kg/d; "the weekly relative loss is 3-4 times
higher during the spawning migration than during the wintering" [S, search summary in PV]. PV's herring
Jan-Feb routine: central 17.2 kJ/kg/d (11-28) from E5's fasting rate.

## Consumption

E14. Greenstreet SPR, Bryant AD, Broekhuizen N, Hall SJ, Heath MR 1997. Seasonal variation in the consumption
of food by fish in the North Sea and implications for food web dynamics. ICES J Mar Sci 54:243-266
(local copy FM/data/q3/greenstreet1997/g.txt; FR/planktivores/data/greenstreet1997/) [F]:
- Table 1, Pelagic Planktivore biomass density Q1-Q4: 6.40, 7.52, 8.67, 9.84 gWW m-2 (l. 210).
- Table 3, Pelagic Planktivores total consumption Q1-Q4: 69.56, 124.80, 145.11, 115.60 mgWW m-2 d-1 (l. 477).
- l. 547-549: "Four key species, Norway pout, herring, sprat and sandeel, made up between 97% and 100% of the
  biomass of the Pelagic Planktivore guild".
- l. 508-516: "Jones' (1984) digestion model was applied to the Norway pout mean stomach content wet weight
  values"; "Herring stomach contents weight data were obtained from Koster et al. (1990), and Daan's (1973)
  digestion model was used to estimate daily consumption rates. A gastric emptying time of 12 h was assumed
  (Koster et al., 1990), but times as fast as 6 h have been observed (Daan et al., 1985). Sprats were
  considered to have the same daily consumption rate ... as that part of the herring population up to 15 cm".
- l. 545-553: Jones' (1982) conversions "1.36 for mackerel and herring, 1.10 for sprats, sandeels, and Norway
  pout" (kcal per g of fish); "Assuming ... an average grams biomass to kilo-calories energy conversion factor of
  1.0 for the food consumed by fish (e.g. Steele, 1974)".
- [D] daily ration 1.09 / 1.66 / 1.67 / 1.17 % BW/d; annual Q/B 5.11; Q1 / annual 0.78.

E15. Albert OT 1995. Diel changes in food and feeding of small gadoids on a coastal bank. ICES J Mar Sci
52:873-885 (`FR/planktivores/data/albert1995/t.txt`) [F]: Norway pout 11-19 cm at 140 m in July; "Temperature
used was 7 C"; "For Norway pout, daily wet weight ration was estimated as 1.1% of body weight. For comparison,
Jones' method (Jones, 1974) gave daily wet weight ration of 3.3% and 1.1% of body weight for haddock and
smaller Norway pout, respectively." (l. 803-806). Smaller pout: "Mean total dry weight by day was twice that at
night (p<0.01)" (l. 367); "The larger Norway pout showed no regular diel" pattern (l. 380-381).

E16. Arrhenius F, Hansson S 1994. In situ food consumption by young-of-the-year Baltic Sea herring Clupea
harengus: a test of predictions from a bioenergetics model. Mar Ecol Prog Ser 110:145-149,
https://www.int-res.com/articles/meps/110/m110p145.pdf (`FR/planktivores/data/arrhenius_hansson1994/t.txt`) [F]:
"The specific food consumption of these fish increased with decreasing body length (25 to 84 mm) and increased
with increasing temperature, ranging from 17 % of body weight d-1 in summer to 2 % d-1 at the end of October.
The YOY food consumption rates do not differ significantly from consumption rates estimated from growth and a
bioenergetics model (Rudstam 1988 ...) based on physiological data (respiration rates, egestion and excretion)
from adult fish." Stomach contents "lowest at nighttime, and indicating that herring did not feed in
darkness". Fish 0.04-3 g, 9-19 C.

E17. Arrhenius F, Hansson S 1993. Food consumption of larval, young and adult herring and sprat in the Baltic
Sea. Mar Ecol Prog Ser 96:125-137 (`FR/planktivores/data/arrhenius_hansson1993/a.txt`) [F]: Rudstam model;
zooplankton "constant energy density of 2850 J g-1 wet wt" (l. 90), mysids/amphipods 3720-3980; Table 5
individual conversion efficiency (growth / consumption, 1 Jan - 31 Dec) herring age 1 5.2 %, age 2 4.0 %, age 3
3.0 %; Table 2 mean weights at 1 Jan, e.g. SD 22-24 age 1 12.7 g, age 2 23.9 g, age 3 48.3 g. [D] rough Q/B
= (dW / W_mean) / efficiency: age 1 11.8-17.7, age 2 16.9 /yr wet prey (34-66 GJ/t/yr ingested). Rough: the
efficiency and the weights are from different tables and populations.

E18. Bachiller E, Utne KR, Jansen T, Huse G 2018. Bioenergetics modeling of the annual consumption of
zooplankton by pelagic fish feeding in the Northeast Atlantic. PLoS ONE 13:e0190345,
doi:10.1371/journal.pone.0190345 (`FR/planktivores/data/bachiller2018/b.txt`) [F]:
- l. 286-290: "consumption was modelled according to a standard bioenergetics model with species specific
  parameter values" (consumption is a model output; stomachs give only the diet, l. 245-249).
- l. 279-281: feeding period "from April 1st ... to September 15th" for NSS herring, "to September 30th" for
  blue whiting.
- l. 352-353: "The swimming speed of NEA mackerel, NSS herring and blue whiting was defined as in Varpe et al.
  (2005), being one body length per second."
- Table (l. 752-779): herring RA 0.0033, beta -0.227, rho 0.0548, "Topt 0.03 as slope for swimming speed (SW)
  dependence", SW 1 (body length s-1); egestion 0.16, excretion 0.10, SDA 0.175. Blue whiting RA 0.008, beta
  -0.172, Winberg act 1.25.
- l. 384-386: "The energy density of copepods was set to 3600 J g-1 ... For euphausiids and amphipods an energy
  density of 4000 J g-1 was used".
- l. 645-648: "NSS herring consumed 0.05 g prey g fish-1 day-1, whereas blue whiting showed the lowest value,
  0.02"; l. 826-828: herring "consuming between 9.5 and 12 times their biomass in zooplankton"; blue whiting "a
  C/B ratio between 4.7 and 6".
- Reading [J]: with SW in body lengths/s the activity term e^(0.03 x 1) is 1.03; with SW converted to cm/s (32
  cm herring) it is 2.6. The paper's table does not settle the unit. [D] at 0.05-0.06 g/g/d x 3.6 kJ/g x 0.609 =
  110-132 J/g/d for respiration plus growth, against Rose resting ~17 J/g/d at 300 g and 6 C: only the cm/s
  reading (~44 + fattening ~49 J/g/d) uses that budget.

E19. Bernreuther 2007 thesis, manuscript 7 (Bornholm Basin) [F, b.txt l. 9112-9123, 10191-10205]: "In
herring, mean daily rations were estimated with a gastric evacuation model combined with field stomach
estimates and compared with maintenance rations"; "both sprat and herring were to some extent food limited in
the Bornholm Basin, since estimates of mean daily rations per fish were partly lower than maintenance
rations"; "The discrepancy between results of bioenergetic and gastric evacuation models in sprat indicated
that the observed energy intake required to fuel somatic growth of field fish was accomplished by intensive
feeding in areas outside of the basin". Herring: "highest daily rations in June (2.9% BW)"; "rations at or
below 0.5% BW" in autumn and winter; maintenance incl. activity "1.0 to 1.4% BW in September 2002 ... ~0.2% BW
in March and April 2003".

E20. Bernreuther 2007 thesis, manuscript 1 (diel feeding, Baltic) [F, b.txt l. 4192-4205, 3122-3127]: "We
found freshly ingested prey at every fishing station and every time of day and night, indicating that both
species feed at night, but to a lower extent than during the day"; "the feeding activity of both herring in
sprat was related to daylight"; "peaks of mean stomach content during the day ... lowest values around 02:00
hours".

E21. Onsrud MSR, Kaartvedt S, Rostad A, Klevjer TA 2004. ICES J Mar Sci 61:1278-1290,
doi:10.1016/j.icesjms.2004.09.005 [S, via KF E10]: herring, whiting and Norway pout "foraging on krill ...
during both day and night"; % stomachs with krill day/night: herring Nov 31.0/17.2, Mar 50.0/20.0; pout Nov
59.1/44.1, Mar 75.0/47.6.

E22. FR `params_planktivorous_fish.md` section 3 table [F]: member Q/B "Herring 4.7 (juvenile 5.63, adult 4.34;
Heath 2005 ration 4.4)", "Sprat 6.0" (Ecopath), "Norway pout 4.5 (4-5.05)" (Albert 1995 / Ecopath), "Blue
whiting 7.5 (6-9)" (Bachiller 2018 ~6; Ecopath 9.06); "Group 5.0" at the section 209 weights (herring 0.79,
blue whiting 0.105, pout 0.075, sprat 0.03). Flag 5: "EcoBase 457 also differs from the report's Table 3.3 for
juvenile herring (B 0.085, Q/B 12.5 ... against 0.63, 5.63". Section 2: "Field rations of adult NSS herring
(Bachiller et al. 2018, ~5-6 %/d at 2-6 C) sit at the Wisconsin value" - contradicted by E18 (Bachiller's
rations are the Wisconsin model's output, not a field measurement).
- Heath MR 2005, ICES J Mar Sci 62:847 (`FR/planktivores/data/heath2005/h.txt`) [F]: rations built on
  Greenstreet et al. 1997 ("daily ration data presented by Greenstreet et al., 1997", l. 268).

E23. BE `evidence.md` F.2-F.3 [F]: planktivore feeding_cost 1.65 = "Rudstam 1988 activity term e^(RTO VEL),
RTO 0.03 s/cm [F FB4 row], at the 1 BL/s field speed (Varpe et al. 2005 via Bachiller [F secondary]) for
15-25 cm box herring: 1.57-2.12, central 20 cm 1.82 [D]. Rudstam's own VEL equation gives 2.15 at <= 9 C and
2.4-5.0 at 10-14 C for a 60 g fish [D] - an upper reading. Gadoid members 1.5"; SDA moved out of feeding_cost
into `sda_fraction` (no SDA left in feeding_cost).

E24. PV summary.md [F]: Jan-Feb group routine 22 kJ/kg/d (4.5-8.2 MJ/t/tick) = herring 17 (11-28), pout 29
(23-41), sprat 44; "Herring is charged ~1.35x its routine and pout ~0.8x"; "Not obtained: ... Johnstone et
al. 1993 (herring routine respiration)".

## Growth and sizes

E25. OP `tables/closure.csv` [F]: planktivores J/B 0.417, g_J 1.1498, g_A 0.3072, P 0.6586 (SMS 2023 key run
2013-22 weight at age). [D] 0.417 x 1.150 + 0.583 x 0.307 = 0.659.

E26. KB `fields/tables/season_cost_kappa_stdout.txt` section 1 [F]: SMS quarterly per-tonne weight gain
ln(w[q+1]/w[q]): herring adult 2+ "Q1 +0.393 Q2 +0.045 Q3 -0.161 Q4 -0.086 annual 0.19"; herring J "annual
1.91"; Norway pout J 1.53, adult 0.46, all 1.38; sprat J 0.34, adult 0.28. The annual figure is net of the
negative (fat drawdown, spawning) quarters.

E27. SV `winter_budget.md` section 4 [F]: herring length and weight at age in the site rectangles, Q3 (Aug)
age 0 11.9 cm 10.9 g, age 1 18.6 cm 45.3 g, age 2 23.8 cm 107.7 g, 3+ 27.0 cm 161.9 g (biomass shares 0.28,
0.36, 0.25, 0.12); Q1 1-wr 14.0 cm 16.6 g (0.42), 2-wr 20.9 cm 56.6 g (0.47), 3+ 25.2 cm ~110 g (0.11). PV:
pout SMS Q1 age 1 9 g (0.75 of biomass), age 2 25 g; sprat age 1 4 g, age 2 8 g.

E28. BB summary.md and FM summary.md [F]: method followed here (term-by-term source check; field-closure Q/B =
(field metabolism + P x tissue energy) / energy per tonne eaten; levers table); FM: "FMR barely falls with
mass (b about 0.03)" in the otolith data.
