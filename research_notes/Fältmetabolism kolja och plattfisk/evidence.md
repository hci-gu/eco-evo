# Evidence - field metabolism of haddock and the flatfish

Research subagent, 2026-11-10. Literature and published data only; no code, library, resume or
AGENTS.md edit. Tags on each entry: [F] full text or data file read, [A] abstract only, [S] secondary
(via a prior project note or another paper). Arithmetic is in `scripts/` (run with `python3 -I`),
printed output in `tables/`. Units: 1 mg O2/kg/h = 13.6 J/mg x 24 h = 0.326 kJ/kg/d = 119.1 MJ/t/yr =
0.0816 MJ/t per 6-h tick (`tables/otolith_fmr_stdout.txt`, first line).

Note on the otolith method used throughout (E1-E5). The otolith carbon proxy gives C_resp, the share
of respiratory (diet) carbon in the blood DIC that the otolith records, from a two-source mixing model
with assumed diet and seawater-DIC d13C. C_resp is converted to O2 consumption with a calibration that
exists for ONE species, Atlantic cod (E1). Every FMR value below for haddock or a flatfish is therefore
[measured isotope + inferred conversion]. Within one site and one calibration, ratios between
species do not depend on the calibration's absolute level, only on the assumption that one
calibration curve holds for all of them (E3 discusses it).

---

## E1. Chung et al. 2019 - cod calibration and wild cod FMR [F]

Chung M-T, Trueman CN, Godiksen JA, Holmstrup ME, Gronkjaer P (2019) Field metabolic rates of teleost
fishes are recorded in otolith carbonate. Communications Biology 2:24.
https://doi.org/10.1038/s42003-018-0266-5 (CC BY 4.0). Full text: prior download
`../Bentivorernas energibudget/data/chung2019/chung2019.xml`.

- Calibration (reared cod, 80 fish at 4 temperatures): "M oto = C (1 - e^(-k (oxygen consumption)))
  ... gave a decay constant and an upper bound of 8.80x10^-3 +- 1.38x10^-3 ... and 0.243 +- 0.035".
  The calibration O2 is RMR_fed: "RMR fed ... to be 1.55(+-0.0832) x SMR".
- Wild cod (Newfoundland): "The FMR oto factorial scope (defined as the ratio FMR oto /SMR) decreased
  from 3.5 to 1.9 with increasing oxygen consumption"; activity "37-59% of FMR oto ... This
  corresponds to 0.73-2.0 times SMR"; "FMR oto was always higher than the sum of SMR and SDA, even
  when the maximum reported value, 0.4 times total energy intake, was used for the SDA estimation".
- Used here: the calibration constants (C 0.243, k 0.0088 per mg O2/kg/h) to convert every C_resp in
  E2; cod's FMR multiplier 2.2 (1.9-3.5) is the prior notes' anchor (`Tillvaxt mot ranson` W.1).

## E2. Alewijnse 2022 thesis dataset - otolith C_resp for 120 marine teleosts [F, data]

Alewijnse SR, Crimmen OA, Cooper N, Trueman CN (2023) Otolith-derived field metabolic rates (C_resp
values), experience temperatures and stable isotopes (d13C and d18O). Figshare dataset,
https://doi.org/10.6084/m9.figshare.22110335 and https://doi.org/10.5258/SOTON/D2292; supports
Alewijnse SR (2022) Macroecological study of otolith-derived field metabolic rates of marine fishes,
PhD thesis, University of Southampton. File `data/alewijnse2023/alewijnse_master_data.csv` (1 351 rows).
Licence: figshare record says CC BY 4.0; the README inside says "University of Southampton License ...
cannot be reproduced or quoted extensively from without first obtaining permission" - see
data/README.md.

README definitions used: "c_resp_mean = the mean C_resp value estimated from d13C of the otolith, using
Bayesian methods"; "temp_mean = the mean experienced temperature (degrees Celcius) estimated from
d18O"; "body mass = the body mass of the fish (wet weight) in grams"; "sample_type = whether the sample
was taken from the otolith's outer surface (outer) or the whole otolith (whole)".

Species summaries (`tables/alewijnse_species_stdout.txt`; VO2 from E1's calibration at the mean C_resp):

| species | n | C_resp mean (SD) | T C mean (range) | mass g median (range) | VO2 mg O2/kg/h | where |
|---|---|---|---|---|---|---|
| cod | 97 | 0.126 (0.040) | 3.8 (0.4-11.0) | 1549 (4-22902) | 83 | Barents 72.8N, Iceland, NW Atl., North Sea (whole) |
| haddock | 62 | 0.112 (0.027) | 5.5 (2.0-14.8) | 517 (27-4300) | 70 | Barents 72.8N (SUMIE, outer, n 52), Georges Bank / Gulf of Maine (Begg 2001, whole, pooled, n 10) |
| long rough dab | 22 | 0.130 (0.029) | 3.4 (0.7-7.2) | 28 (4-490) | 87 | Barents 72.8N, outer |
| witch | 5 | 0.065 (0.010) | 8.4 (7.8-8.7) | 484 (215-1027) | 35 (fish 28-41) | northern North Sea 55.8-60.3N, Cefas 2010, outer |
| dab | 4 | 0.190 (0.017) | 17.8 (17.1-18.3) | 152 (100-198) | 174 | eastern English Channel 50.6N, 2011, outer |
| lemon sole | 5 | 0.154 (0.028) | 14.6 (13.4-15.9) | 192 (97-234) | 114 | 54.75N, 0.5W, 2017, outer |
| whiting | 5 | 0.162 | 17.2 | 94 | 124 | Channel |
| saithe | 5 | 0.143 | 11.1 | 2923 | 101 | Channel |

The Barents site (72.797N 23.288E, 2018-2019, outer-edge samples) has cod (n 64), haddock (52) and
long rough dab (22): one place, one seawater-DIC baseline for the site (1.21-1.22 permil), one diet
baseline per species (cod -22.15, haddock -21.40, long rough dab -20.27 permil), one calibration
(`tables/otolith_fmr_stdout.txt`). Recomputing C_resp by two-source mixing on each species' own diet
value and on each of the three as a common value gives haddock/cod 0.72-0.76 and long rough dab/cod
0.84-0.95 at equal mass and temperature (adults >= 100 g, Q10 2), so the ratios do not hinge on the
diet baseline. The adult haddock at the site are 406-4 300 g (the model's effective 275 g is below
this range); the long rough dab >= 100 g are only 8 fish (111-490 g).

## E3. Jones et al. 2023 - North Sea plaice otolith FMR [F]

Jones J, Hunter E, Hambach B, Wilding MR, Trueman CN (2023) Individual variation in field metabolic
rates of wild living fish have phenotypic and ontogenetic underpinnings: insights from stable isotope
compositions of otoliths. Frontiers in Ecology and Evolution 11:1161105.
https://doi.org/10.3389/fevo.2023.1161105 (CC BY). `data/jones2023/`.

- Sample: CEFAS archive, North Sea ICES IVb and IVc, survey-caught, "targeted individuals assessed to
  have been spawned 4 years prior to capture"; "Individual body mass varied between approximately 200
  and 800 g, and experienced temperatures varied over 12 degrees"; 281 adults.
- Conversion: the linear form of the cod calibration, "Mass - specific oxygen consumption rate (mgO2
  Kg-1 hr-1) = (Cresp - 0.041 +- 0.00102)/9.71*10^-4 +- 9.57*10^-5".
- Table 3 (adults at capture): temperature mean 8 C (SD 2.8, range 0.2-17); O2 consumption mean 54,
  SD 36, max 203, min -76 mg O2/kg/h; 95 % CI on an individual value 58. First year of life: 79
  mg O2/kg/h at 14 C. INCONSISTENCY: the text says "Adult Cresp values range between 0.03 and 0.22,
  with an average of 0.12", Table 3 prints the mean C_resp as "0.94". On their equation 54 mg/kg/h
  corresponds to C_resp 0.094, so Table 3's O2 column (54) is used here and 0.12 is read as a typo
  (0.12 would give 81).
- Authors' caveats, verbatim: "Estimated oxygen consumption rates inferred within this study are
  similar to standard metabolic rates based on laboratory respirometry, and are therefore lower than
  expected. However, we are unsure of the appropriate parameters used to convert Cresp values to
  oxygen consumption for plaice without prior experimentation, therefore we are drawing on calibration
  parameters derived for Atlantic cod." "These Cresp values are consistent with data from other
  pleuronectiformes (Alewijnse, 2022), suggesting relatively low field metabolic rates for this order
  compared to other sampled teleosts when body mass and temperature are controlled." "Plaice are known
  to make use of environmental transport (selective tidal stream transport) which may reduce energy
  costs associated with directed movement (Metcalfe et al., 1990, 2006)." "Species-specific variations
  in optimum blood carbonate concentrations could potentially contribute to relatively low estimates of
  apparent oxygen consumption rates in plaice. However, we feel it is unlikely that the slope of the
  calibration term will vary systematically across species".
- Mass effect: "the associated coefficients are small compared to those expected based on allometric
  scaling or metabolic theory".

## E4. Jones, Hunter & Trueman 2025 - seasonal FMR of North Sea plaice [A]

Jones J, Hunter E, Trueman CN (2025) Seasonal and sex-based variations in energy use in wild plaice:
Differences in energetic costs of spawning and feeding behaviours outweigh effects of temperature on
field metabolism. Fisheries Research 289:107476. https://doi.org/10.1016/j.fishres.2025.107476 (CC BY;
publisher page refused automated download, HTTP 403; abstract via OpenAlex only).

"Here we describe how total energy use varies seasonally within a sample of 272 free-roaming European
plaice from the North Sea. Estimated energy (field metabolic rate) use peaked in early summer months
prior to spawning, co-incident with formation of otolith opaque zones and increases in body condition,
but decoupled from seasonal variations in experienced temperature. Maximum observed FMR estimates in
years characatreised [sic] by high metabolc [sic] rates, approach laboratory inferred maximum metabolic
rates, implying maximum feeding rates are limited by respiratory capacity rather than prey
availability." "Males showed reduced energy use through summer months compared to females, but higher
apparent energy use outside of the summer feeding season." No monthly numbers obtained.

## E5. Other otolith-proxy studies (comparators) [A]

- Smolinski S et al. (2021) Differences in metabolic rate between two Atlantic cod populations
  estimated with carbon isotopic composition in otoliths. PLoS ONE 16:e0248711.
  https://doi.org/10.1371/journal.pone.0248711. "Cresp was on average 0.275 and 0.295 in Icelandic and
  NEA stock, respectively." (Age 3 and 8 increments; their own baseline method, higher than the
  Alewijnse values for the same stocks - C_resp levels are not comparable across baseline choices.)
- Chung M-T, Jorgensen K-EM, Trueman CN, Knutsen H, Jorde PE, Gronkjaer P (2021) First measurements of
  field metabolic rate in wild juvenile fishes show strong thermal sensitivity but variations between
  sympatric ecotypes. Oikos 130:287. https://doi.org/10.1111/oik.07647. Juvenile cod, Skagerrak coast
  of southern Norway: "The combined ecotype median of field metabolic rate increased with increasing
  temperature until around 16 C"; "The cold-adapted fjord ecotype maintained higher field metabolic
  rates at cooler temperatures than the warm-adapted North Sea ecotype".
- Sherwood GD, Rose GA (2003) Influence of swimming form on otolith d13C in marine fish. MEPS
  258:283. https://doi.org/10.3354/meps258283. "Among-family variance in otolith d13C was uniquely
  explained by a morphometric index for aerobic swimming activity (caudal aspect ratio, r2 = 0.61)".
- Begg GA, Weidman CR (2001) Stable d13C and d18O isotopes in otoliths of haddock Melanogrammus
  aeglefinus from the northwest Atlantic Ocean. MEPS 216:223. https://doi.org/10.3354/meps216223.
  Whole-otolith pooled samples; these are the "Begg_2001" rows of E2 (12-15 C), not used for the
  multipliers.

## E6. Laboratory SMR used as denominators [S unless stated]

All via `../Fiskgrupper i rutan/benthivores/evidence_benthivores_A_bioenergetics.md` and
`tables/bioenergetics_A_smr.csv` unless marked.
- Cod: Schurmann H, Steffensen JF (1997) J Fish Biol 50:1166, https://doi.org/10.1111/j.1095-8649.1997.tb01645.x
  [A]: "The SMR was 35.5, 57.0 and 78.2 mg O2 kg-1 h-1 ... at 5, 10 and 15 C"; "Previously reported
  SMR for Atlantic cod from arctic waters at 4 C was twice that measured at 5 C in the present study."
  Juveniles; mass not in the abstract. Prior group table: 3.53 MJ/t/tick at ~7 C = 43.3 mg/kg/h,
  applied to 1.2 kg.
- Haddock: Norin T et al. (2019) PeerJ 7:e7784 [S, prior note F]: 12 C, 63 g, SMR 84.3, RMR 119.4,
  MMR 284.9 mg O2/kg/h; cod 68 g SMR 156.7 -> haddock/cod 0.54. Perez-Casanova JC (2009) thesis [S,
  prior note F]: routine MO2 of 40-50 g juveniles, haddock/cod 0.64-0.75 at 2 and 11 C. Prior group
  table haddock 2.50 MJ/t/tick at 7 C (275 g) = 30.6 mg/kg/h, a judgement between routes.
  Jones 1978 (E8) appendix, Tytler 1969 haddock respirometry refitted at zero swimming speed: 0.16
  W^0.8 exp(0.081(T-10)) mg O2/h -> 275 g at 8 C: 0.16 x 275^0.8 x exp(-0.162) / 0.275 = 44 mg/kg/h
  = 3.6 MJ/t/tick (arithmetic here). A measured haddock SMR at the model's mass, 1.44 x the judged 2.50.
- Plaice: Fonds M, Cronie R, Vethaak AD, van der Puyl P (1985) ICES CM 1985/G:54 [S, prior note F]:
  DOCmin fit 44.6 (150 g) and 35.9 (400 g) mg O2/kg/h at 10 C, Q10 2.01 -> 38.7 and 31.2 at 8 C.
- Witch: Moser ML, Ross SW, Sulak KJ (1996) MEPS 144:57 [S, prior note F]: 5.33 g fish at 6.0 C, "0.067
  mg O2 g-0.7 h-1" -> 14.9 (150 g) and 11.1 (400 g) mg/kg/h at 6 C when scaled with exponent 0.7.
- Long rough dab: the prior group table's 1.50 MJ/t/tick (18.4 mg/kg/h) is NOT a lab SMR: it is
  MacKinnon 1972's field winter (fasting) metabolism of Nova Scotia American plaice at ~3 C, Q10-scaled
  (`Tillvaxt mot ranson` evidence W.1). No lab SMR for the species was obtained here: MacIsaac PF,
  Goff GP, Speare DJ (1997) Comparison of routine oxygen consumption rates of three species of
  pleuronectids at three temperatures. J Appl Ichthyol 13:171,
  https://doi.org/10.1111/j.1439-0426.1997.tb00117.x [A] measured routine O2 of American plaice at
  2-14 C ("at each temperature, ROC rates were significantly different among the three species"), but
  the full text was refused (HTTP 403).

## E7. Flatfish maximum metabolic scope [S]

Fonds et al. 1985 (via prior note): "Both species can increase their standard metabolic rate by a
factor 3." Duthie GG (1982) J Exp Biol 97:359 [S, prior note A]: flatfish standard oxygen consumption
"significantly lower than that of roundfish", maximum consumption and scope "low in comparison to
salmonids". Used only as a plausibility ceiling for FMR/SMR.

---

Entries E8-E19 were read by three literature sub-searches of this note (field rations, SDA and
growth cost, seasonality); the tag is the reading level they reported. Downloads in data/q3, q4, q5.

## E8. Jones 1978 - haddock consumption from stomach contents and growth [F]

Jones R (1978) Estimates of the food consumption of haddock and cod. J Cons int Explor Mer 38:18-27.
https://doi.org/10.1093/icesjms/38.1.18. "the rate at which food enters a stomach must be equal to the
rate at which it is eliminated"; haddock intake "= 840 (col A)(col B) kcal/yr" at 8 C (Q 0.16,
continuous feeding); "1 g wet weight of haddock food ingested providing about 0.6 kcals" (Hertling
1940). Table 6 North Sea annual intake (kcal/yr): 250 g 1218, 450 g 1572, 600 g 2352. Table 2 growth
(g/yr) at mean weight: 217 g -> 132, 362 g -> 157, 521 g -> 162. Eggs "about (0.9)(0.25) W". Caveat:
"for haddock, the estimates of energy intake on the basis of mean stomach contents were mostly lower
than the estimates based on the energy requirements". SMR in the appendix (Tytler 1969 refit at zero
speed): 0.16 W^0.8 exp(0.081(T-10)) mg O2/h.
Sub-search arithmetic (A 0.8, P somatic + gonad, 8 C): R = 36.4 / 25.8 / 30.7 kJ/kg/d at 250 / 450 /
600 g = 13 300 / 9 400 / 11 200 MJ/t/yr = 2.47 / 1.97 / 2.49 x the paper's SMR. With single-meal
evacuation (0.61 x Q): 19.0 / 13.3 / 16.7 kJ/kg/d (1.0-1.35 x SMR), a lower bound. The SMS ration
(Q/B 2.84 at 2.5 kJ/g) gives R 0.53 x SMR, impossible for a growing fish: its C or prey energy is low.

## E9. Jones 1974 [A]; Hall 1987 [A]

Jones R (1974) J Cons 35:225, https://doi.org/10.1093/icesjms/35.3.225: "the single meal technique
giving lower rates of digestion than the regular feeding technique". Hall SJ (1987) J Fish Biol
31:479, https://doi.org/10.1111/j.1095-8649.1987.tb05253.x: "wild haddock feed at between 15 and 22%
of their maximum capacity"; dab "daily consumption rates that were approximately 40% lower than
those shown by haddock".

## E10. Greenstreet et al. 1997 - quarterly guild rations [F]

Greenstreet SPR et al. (1997) ICES J Mar Sci 54:243-266, https://doi.org/10.1006/jmsc.1996.0183.
Demersal benthivores ("Plaice ..., common dab ..., and lemon sole ... accounting for between 78% and
86% of the guild's biomass") consumption / biomass by quarter: 0.94 / 1.74 / 1.54 / 1.67 %BW/d, mean
1.47 (5.4 /yr); demersal piscivores (cod, haddock, whiting, saithe; "Jones' (1974) digestion model" on
Daan 1989 stomachs; "Sea temperature was assumed to be 6 C in quarter 1, 7 C in quarter 2, 10 C in
quarter 3, and 8 C in quarter 4") 1.11 / 1.33 / 1.80 / 1.37, mean 1.40. Q1/mean 0.64 and 0.79
(temperature included). Benthivore production "assumed to be the same as for the Demersal Piscivore
guild".

## E11. Rijnsdorp & Ibelings 1989 - plaice spawning-fast metabolism and growth [F, via Rijnsdorp 1992 thesis]

Rijnsdorp AD, Ibelings B (1989) J Fish Biol 35:401-415, https://doi.org/10.1111/j.1095-8649.1989.tb02992.x.
"Metabolic rate during the spawning season thus amounts up to 9.1 kJf.day-1"; with egg correction
"449 kJ (382 kJf) or 6.4 kJf day"; at 6.8 C "the resting metabolic rate (R) of a standard mature
plaice can be estimated at 3 kJ.day-1 and maximum metabolic rates at 15 kJ.day-1. Fonds et al. (1985)
estimated the resting metabolic rate at 4 kJ.day-1 and the feeding metabolic rate at 8 kJ.day-1."
Ripe female 498 g (Dec), spent 379 g. Sub-search arithmetic: winter R 14.6-20.8 kJ/kg/d = 5 330-7 590
MJ/t/yr at 6.8 C = 1.6-2.3 x Fonds resting. Somatic growth 543 kJ/yr + eggs 594-790 kJ: P 7.1-8.3
kJ/kg/d. Annual R with the Greenstreet guild ration (E10) at 2.5 kJ/g prey and A 0.8: ~21-22 kJ/kg/d
= 7 700-8 100 MJ/t/yr (weak: guild ration, borrowed prey energy).
Abstract: gross body energy fell "44% ... in mature females, 27% ... in mature males, and 9% ... in
immatures" over spawning.

## E12. Rijnsdorp, van Lent & Groeneveld 1983 [F]

ICES CM 1983/G:31. "From December till March, the time of spawning, it does not feed. (TODD, 1911)"
but "has only a small emperical base"; "only in the last week of March some of the fish had remainings
of food in their intestines". 487 g female: 2 832 -> 1 602 kJ, "a 43 % decrease". "log winter
metabolism = 0.503 + 0.74 log weight" -> 311 kJ per season at 487 g = 5.3-7.1 kJ/kg/d (90-120 d);
temperature not given.

## E13. MacKinnon 1973 - American plaice population energy flow [A]

MacKinnon JC (1973) J Fish Res Board Can 30:1717-1728, https://doi.org/10.1139/f73-272. "The
ecological efficiency is 17% ... Metabolic expenditures constitute the largest fraction (62%) of
population energy intake and about 80% of this amount is consumed during summer." Model-based,
population level including 0+ fish; with MacKinnon 1972 (P 92 kcal/yr for a 35 cm female) R ~336
kcal/yr, ~5 000 MJ/t/yr only under an assumed ~280 g (not in the source).

## E14. Collie 1987 - yellowtail flounder stomach-content consumption [F]

Collie JS (1987) MEPS 36:205-213, https://doi.org/10.3354/meps036205. "0.824, 0.605, and 0.477 kg wet
wt fish-1 yr-1" at 348 / 298 / 333 g; "the stomach-content consumption estimates seem too low". R
from these is below any plausible SMR: stomach-content C of flatfish on soft prey with gadoid
evacuation rates is not reliable.

## E15. SDA at realistic meals [F/A]

- Soofiani NM, Hawkins AD (1980) ICES CM 1980/G:36 [F, poor OCR; cite the 1982 J Fish Biol paper]:
  20-83 g cod, satiation meals 1.9-3.6 %BW, repeated: "For fish fed to satiation, the energy
  expenditure was 12.9, 11.92 [French abstract 14.92], 16.14 and 21.47% of the ingested energy at 7,
  10, 15 and 18 C"; "with consecutive feeding the effect is cumulative"; their apparent SDA includes
  "Excited feeding behaviour, with increased locomotor and other activities".
- Johnston IA, Battram JC (1993) Mar Biol 115:7, https://doi.org/10.1007/BF00349380 [A]: single
  satiating meal, SDA "15 to 23% of the energy ingested" (sculpin, notothenioid, hawkfish).
- Jordan AD, Steffensen JF (2007) Physiol Biochem Zool 80:178, https://doi.org/10.1086/510565 [A]:
  2.5 and 5 %BM meals, SDA coefficient "equivalent between the different feeding rations".
- Auchterlonie N (1998) PhD thesis, Univ. Stirling, https://storre.stir.ac.uk/bitstream/1893/1917/ [F]:
  halibut, single meals 0.13-0.85 %BW, mean coefficient 9.8 % over temperatures (5.8 % at 8 C);
  "S.D.A. coefficient decreased with increasing meal size" (significant only at 14 C).
- Secor SM (2009) J Comp Physiol B 179:1, https://doi.org/10.1007/s00360-008-0283-7 [A]: "Meals that
  are large, intact or possess a tough exoskeleton ... generate a larger SDA".
- Peck MA, Buckley LJ, Bengtson DA (2005) J Fish Biol 66:911 [S] and (2003) CJFAS 60:929 [A]: juvenile
  haddock feeding cost ~3-4 % of consumed energy (small meals).
- Chabot D, Koenker R, Farrell AP (2016) J Fish Biol 88:81, https://doi.org/10.1111/jfb.12836 [A]:
  "spontaneous locomotor activity is a frequent problem that leads to overestimation of SDA".
- No SDA coefficient for haddock or any flatfish at >= 3 %BM or satiation was found.

## E16. Protein synthesis inside SDA [A/F]

- Lyndon AR, Houlihan DF, Hall SJ (1992) J Comp Physiol B 162:209, https://doi.org/10.1007/BF00357525
  [A]: cod fed "a meal of whole sandeel at 6% body weight"; "the integrated increment in whole animal
  protein synthesis contributed between 23% and 44% of the post-prandial increase" (corrects the prior
  note's 23-51 % snippet).
- Houlihan DF, Hall SJ, Gray C, Noble BS (1988) CJFAS 45:951, https://doi.org/10.1139/f88-117 [A]: "a
  300-g cod growing at 1.0%.d-1 synthesised 1.25 g of protein with 0.4 g of this protein remaining as
  growth"; "In starving fish there was a constant level of protein synthesis".
- Goodrich HR, Wood CM, Wilson RW, Clark TD (2024) J Exp Biol 227:jeb246722,
  https://doi.org/10.1242/jeb.246722 [F]: "synthesising 1 g of protein is estimated to cost ~50 mmol
  of ATP equivalents or 3.5 kJ"; "It is therefore more probable that the SDA represents an increase in
  rates of protein synthesis and turnover".
- Jobling M (1983) J Fish Biol 23:549 [A]: "SDA may represent a short-term increase in rates of
  protein synthesis and turnover following feeding".
- Sub-search arithmetic from Houlihan 1988 (upper bound, all turnover charged to growth): 0.46-0.80 J
  per J of protein deposited at 1 %/d; 0.37-0.64 at 2 %/d.

## E17. Swimming and foraging activity of flatfish [A/F]

- Priede IG, Holliday FGT (1980) J Exp Biol 85:295, https://doi.org/10.1242/jeb.85.1.295 [A]: plaice
  "log10 R = 0.3318V + log10(2.4T + 26.52)" (mg O2/kg/h, V in BL/s); "The cost of swimming in plaice
  is very similar to that of typical round fish such as haddock but the resting metabolic rate is
  lower than for haddock". Sub-search: ~0.35-0.70 kJ/kg per hour swum at 0.6-1 BL/s, 7.5 C.
- Auchterlonie 1998 ch. 5 [F]: tank halibut "remain at rest for periods of between 76% and 94% of any
  24 hour period"; motor activity 9-20 % of resting energy at normal densities.
- Duthie GG (1982) J Exp Biol 97:359 [A]: flatfish standard O2 consumption lower than in roundfish.
- Metcalfe JD, Arnold GP, Webb PW (1990) JMBA UK 70:149 [A]: tidal-stream transport "can
  theoretically save energy" (no numbers obtained).

## E18. Seasonal activity of plaice from data storage tags [F]

- Hunter E, Metcalfe JD, O'Brien CM, Arnold GP, Reynolds JD (2004) MEPS 279:261,
  https://doi.org/10.3354/meps279261: "The plaice spent 1/4 to 1/3 of their time swimming in midwater
  following release in December and January ... the number of hours swimming had declined to
  virtually zero hours by May"; "following completion of the post-spawning migration, plaice rarely
  left the seabed".
- Hunter E, Cotton RJ, Metcalfe JD, Reynolds JD (2009) MEPS 392:167, https://doi.org/10.3354/meps08213
  (139 mature females): "Most swimming occurred during expected times of migration and spawning
  (October to March) ... rarely left the sea-bed during summer"; "Average swimming times in excess of
  5 h per day were observed only for DST1 West fish during the months of December and January".
- Rijnsdorp AD (1989) ICES J Mar Sci 46:35 [A]: females in spawning condition ~5 weeks (late Jan -
  early Mar), males >= 11 weeks; "During the spawning season mature plaice hardly fed."

## E19. Seasonal feeding of haddock, dab, long rough dab [F/A]

- ICES CM 1980/H:21 (PINRO; Barents Sea haddock, 255 136 stomachs) [F, poor OCR]: "Seasonal maximum
  of stomach fullness for haddock falls to March ... The second peak ... in July ... the third one - in
  September"; empty stomachs "highest in November-January and in April".
- Podrazhanskaya SG, Shestov VP, NAFO Sci Coun Studies 1:51 [F]: Grand Bank haddock feeding index
  spring 39.3, summer 78.2, autumn 106.6, winter 63.5 (winter n 9).
- Hislop 1984 / Wright & Gibb 2005 via MEPS 432:149 [F]: "the February to May spawning period of
  haddock in the North Sea".
- Michalsen K, Johannesen E, Bogstad B (2008) ICES J Mar Sci 65:571 [F] (cod comparator): "cod do
  feed, even when they are in a spawning state".
- Knust R (1987) Inf Fischwirtsch 34:161 [F]: dab on the Dogger Bank, food intake through the year
  "konstant"; German Bight a pronounced annual cycle.
- Saborowski R, Buchholz F (1996) J Fish Biol, https://doi.org/10.1111/j.1095-8649.1996.tb00015.x [F]:
  dab, "reduced but not suspended feeding activity" in winter (males).
- Ntiba MJ, Harding D (1993) Neth J Sea Res 31:189 [A]: "The North Sea long rough dab feeds throughout
  the year with a definite maximum from April till August when over 85% of the fish sampled were
  feeding"; Dec-Jan 31 % feeding [S, via Gonzalez et al. 2003 NAFO SCR 03/23].
- No measured winter metabolic depression for any of the six species was found.
