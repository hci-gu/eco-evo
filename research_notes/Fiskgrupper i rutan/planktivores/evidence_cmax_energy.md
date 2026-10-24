# Planktivorous fish: Cmax, energy density and Q10

Group `planktivorous_fish` in the eastern Skagerrak box. Members by annual-mean biomass:
herring 0.79, blue whiting 0.105, Norway pout 0.075, sprat 0.03. Model prey (zooplankton)
4.5 kJ/g WW; tick 6 h (4 ticks/day). Current pair `planktivorous_fish_preys_on_zooplankton`:
h 25, ceiling 0.04 t/t/tick = 16 %BW/day = 0.72 kJ/g/day.
Tags: [F] full text read / data downloaded, [A] abstract only, [S] secondary, [D] computed here.
Script: `scripts/cmax_eval.py` (run `python3 -I cmax_eval.py`); full output in `scripts/cmax_eval_output.txt`.
Companion scripts from the pout / blue whiting search: `scripts/cod_cmax_proxy.py`.

## 1. Bottom line

### 1a. Cmax per species (Wisconsin-type Cmax, mean of the monthly values) [D]

Energy reading: Cmax is a WET-MASS ration. Energy = Cmax x the prey energy density the
function stands for (central reading: herring/sprat 2.85 kJ/g, Baltic zooplankton as used with
the Rudstam model; pout/blue whiting 4.18 kJ/g, the "lean fish" value of the cod model). See 2.3.

| Species (W, layer) | Cmax g/g/d, year / Apr-mid-Sep | kJ/g/d at annual-mean T | kJ/g/d at feeding-season T | kJ/g/d mean of months, year / season | %BW/d on 4.5 kJ/g, year / season | h, year / season |
|---|---|---|---|---|---|---|
| Herring (60 g, 0-50 m; juvenile set) | 0.165 / 0.180 | 0.52 (T 9.3 C) | 0.55 (T 10.2 C) | 0.47 / 0.51 | 10.5 / 11.4 | 38 / 35 |
| Herring, adult set (sensitivity) | 0.179 / 0.193 | 0.57 | 0.59 | 0.51 / 0.55 | 11.4 / 12.2 | 35 / 33 |
| Blue whiting (100 g, 50 m; cod proxy) | 0.032 / 0.031 | 0.135 (T 8.4 C) | 0.129 (T 7.7 C) | 0.135 / 0.128 | 3.0 / 2.9 | 134 / 140 |
| Norway pout (25 g, 50 m; cod proxy) | 0.042 / 0.040 | 0.177 | 0.168 | 0.176 / 0.168 | 3.9 / 3.7 | 102 / 107 |
| Sprat (10 g, 0-50 m; herring function) | 0.261 / 0.285 | 0.82 | 0.87 | 0.75 / 0.81 | 16.6 / 18.1 | 24 / 22 |
| **Biomass-weighted (energy space)** | - | **0.46** | **0.49** | **0.42 / 0.46** | **9.4 / 10.1** | **43 / 39** |

h = 4 / (fraction of body mass per day on 4.5 kJ/g prey). Weighting is done in kJ/g/day, then
converted. Feeding season = 1 Apr - 15 Sep (Bachiller et al. 2018), September at half weight.
"Mean of months" (mean of the 12 monthly Cmax values) is lower than "Cmax at the mean T" because
f(T) is concave; it is the right annual value for a constant h.

### 1b. Sensitivity of the weighted ceiling to the energy reading [D]

| Reading (herring+sprat ED / pout+BW ED, kJ/g) | weighted kJ/g/d, year / season | %BW/d on 4.5 kJ/g | h, year / season | current h 25 above Cmax by |
|---|---|---|---|---|
| Low: NEMURO.FISH zooplankton 2.58 / 4.18 | 0.38 / 0.42 | 8.5 / 9.2 | 47 / 43 | x 1.87 / 1.73 |
| **Central: Baltic zooplankton 2.85 / 4.18** | **0.42 / 0.46** | **9.4 / 10.1** | **43 / 39** | **x 1.71 / 1.58** |
| Copepod 3.6 (Bachiller) / 4.18 | 0.53 / 0.57 | 11.7 / 12.7 | 34 / 32 | x 1.37 / 1.26 |
| Copepod 3.6 / fat fish 5.86 | 0.54 / 0.58 | 11.9 / 12.9 | 34 / 31 | x 1.34 / 1.24 |
| Mass limit read on the model prey (4.5 everywhere) | 0.65 / 0.71 | 14.5 / 15.7 | 28 / 25 | x 1.10 / 1.02 |
| Lab ad-libitum herring rations instead of Wisconsin (herring, sprat at 3.7 kJ/g food; see 2.2) | 0.15 / 0.16 | 3.3 / 3.6 | 120 / 112 | x 4.8 / 4.5 |

Bottom line:
- With the Wisconsin herring Cmax read as energy at the zooplankton energy densities it is used
  with (2.6-3.6 kJ/g), the biomass-weighted ceiling is 0.38-0.57 kJ/g/day, i.e. h 32-47 on the
  model's 4.5 kJ/g zooplankton. The current h 25 (0.72 kJ/g/day) is 1.3-1.9x above that
  (central 1.6-1.7x).
- h 25 equals Cmax only if the Wisconsin wet-mass ration is applied unchanged to 4.5 kJ/g prey
  (a pure mass limit). That contradicts the project's energy-limit convention (resume 148).
- The annual-mean vs feeding-season difference is small: h 43 vs 39 (central). The model has one h
  and the temperature does not scale it; the mean-of-months annual value (h ~43, central) is the
  consistent choice. The feeding-season value (h ~39) is the choice if the ceiling should bind
  in the season when herring actually eat near it (Apr-Sep).
- Herring sets the answer (79 % of the weight). The deep members barely move it: dropping pout and
  blue whiting changes the weighted h by < 15 %.
- Biggest uncertainty: the herring Cmax itself. The Wisconsin function (Rudstam 1988) gives
  3-4x the ad-libitum rations measured in the lab experiment it was reportedly built on (De Silva &
  Balbontin 1974, section 2.2). Field data on adult NSS herring in cold water (Bachiller 2018) sit
  at the Wisconsin value, not at the lab value. I treat the Wisconsin value as the Cmax and the lab
  value as a lower bound. Judgement: a group h of ~35-45 is defensible; h 25 is not supported by any
  measured Cmax unless prey are counted by mass.

### 1c. Energy density and fat per species (details and quotes in section 3)

Model form: ED = lean ED + 0.396 kJ/g per fat %; max reserve = max fat % x 396 MJ/t.

| Species | Lean ED (protein part), kJ/g | Max fat % (season) | Min fat % (season) | Implied max reserve, MJ/t | Seasonal mean ED, kJ/g | Status |
|---|---|---|---|---|---|---|
| Herring, 15-25 cm in the box | 4.0 (15-19 % protein; library) | ~15-16 (Q3, juveniles 15-20 cm: 10.1 kJ/g); adults 16-22, mean 18.6 (Jun-Jul) | ~1 (Q1, juveniles 15-20 cm: 4.4 kJ/g); adults 4.5 (late May, after spawning) | juveniles ~6100; adults ~7000-7400 (library 7000) | ~7.3 (15-20 cm, Q1/Q3 mean); ~8.8 (20-25 cm) | [F]/[S] measured (P&H 2001 via JNCC 633) |
| Sprat, 10-13 cm | ~4.0 by protein (17.2 %); measured ED implies ~3.3 (Spitz 6.5 kJ/g minus 8.2 % x 0.396); 3 % weight, immaterial for the group | ~15-20 (Oct-Dec; North Sea 10-12 cm 17-22 % in Dec) | ~4-6 (Apr-Jun, after spawning) | ~6000-7900 (central 17 % -> ~6700) | ~6.5-7.6 measured (Spitz autumn 6.5; Wanless all-year 7.6) | [F] fat series; ED [F]/[S] |
| Norway pout, 10-20 cm | ~4.0 (16-19 % protein) | ~5 (season unknown; Torry 4.2-5.1 %) | ~2 (judgement; Trisopterus proxies 1.8-2.8 % in autumn) | ~2000 | ~4.8-5.1 (proxy; T. minutus 5.1, Torry 4.4-4.8 calc.) | no measured seasonal data; proxies |
| Blue whiting, 20-30 cm | ~3.8 (Spitz 4.4 kJ/g at 1.5 % fat; 14-17 % protein) | ~9 (Aug; Faroese landings, mostly liver fat) | ~2 (Apr-May, after spawning) | ~3600 | ~6.0 year (mean fat 5.7 %); ~6.7 in Q3 when it is in the box (fat 7.4 %) | [F] fat series (adults); ED [D] |

[D] Biomass-weighted max reserve: ~6250 MJ/t with the herring at 7000 (adult value), ~5550 MJ/t
with juvenile herring at 6100. The library's 7000 MJ/t is an adult-herring value; the box's
juvenile herring and the lean gadoid members pull the group down by 10-20 %. Lean ED ~4.0 kJ/g
fits all four members within +-5 %.

### 1d. Q10

| Species | Q10 standard/routine metabolism | Basis | Reference layer |
|---|---|---|---|
| Herring | 2.2 | sprat respirometry 9-21 C (Meskendahl 2013), clupeids 1.8-2.2 [F, existing note] | 0-50 m (current) |
| Sprat | 2.2 (RS pooled 2.18; RR 2.12-2.23) | Meskendahl 2013 [F] | 0-50 m |
| Norway pout | no measurement; use 2.2 (cod SMR 5-15 C, Schurmann & Steffensen 1997) or 2.4 (median within-species teleost acclimation, Clarke & Johnston 1999) | proxy [S] | near-bottom 6-8 C, not 0-50 m |
| Blue whiting | no measurement; same proxy 2.2-2.4 | proxy [S] | 6-9 C below 100 m (Q3) |

## 2. Cmax: sources and details

### 2.1 Herring - Wisconsin function (Rudstam 1988 lineage)
- Equation: Cmax = CA W^CB f(T), CA 0.642, CB -0.256, units "g prey (g fish)^-1 day^-1", W in g
  wet weight. f(T) = Thornton & Lessem (FB4 CEQ 3) with CK1 0.10 at te1, 0.98 at te2 and te3,
  CK4 0.01 at te4. te1/te2/te3/te4 = 1/15/17/25 C ("juvenile", age-1) and 1/13/15/23 C ("adult",
  age-2+). [F] FB4 `Parameters_official.csv` rows "Herring (juvenile)" and "Herring (adult)",
  source "Rudstam 1988" (local copy `../benthivores/data/fb4/`); [F] Rose et al. 2008, Table 1
  ("CMAX at (te2 + te3)/2 0.642", "for CMAX 0.256", xk1 1, xk2 15,15,13, xk3 17,17,15, xk4 23,25,23).
  The NEMURO.FISH herring (Megrey et al. 2007; Rose et al. 2007, 2008) uses the same constants
  (verified via Rose et al. 2007, 2008 and PICES Sci. Rep. 20; Megrey 2007 itself not obtained).
- Origin of CA and CB: "the slope for Atlantic herring (wet weight 8 - 15 g) was -0.256 (De Silva
  and Balbontin 1974). Rudstam (1988) used the slope and intercept derived by De Silva and Balbontin
  (1974) in the bioenergetics model for adult Atlantic herring" [F] PICES Sci. Rep. 20 (2002),
  p. 88. "Maximum daily consumption rate (g of prey per g body mass of herring per day) is
  estimated using an allometric function of mass from ad libitum feeding experiments conducted at
  the optimum temperature." [F] same, consumption section.
- Units and prey energy density: the function carries no prey energy density. In NEMURO.FISH the
  consumption is "g prey (g fish)^-1 day^-1" and is converted to fish mass by CALz/CALf with
  "CALz = 2580 joules / gram zoop" and "CALf = 5533 joules / gram fish" [F] PICES Rep. 20;
  Rose et al. 2007 eq. 1 [F]. Arrhenius & Hansson 1993 ran the Rudstam model with "zooplankton prey
  ... constant energy density of 2850 J g-1 wet wt", mysids 2976-3720, amphipods 3980 J/g [F] MEPS
  96, p. 126. Bachiller et al. 2018 used copepods 3600 J/g, euphausiids/amphipods 4000 J/g [F].
  So the wet-mass Cmax has been applied to prey of 2.6-4.0 kJ/g; the central reading here uses
  2.85, the range 2.58-3.6.
- Temperature factor [D] (juvenile / adult set): 5 C 0.39/0.46; 6 C 0.49/0.58; 7 C 0.60/0.70;
  8 C 0.70/0.80; 10 C 0.85/0.91; 12 C 0.93/0.97; 14 C 0.97/0.98. Adult values match the existing
  table in `Vintermetabolism sill och tumlare/q10_metabolism.md` to 3 decimals.
- Monthly Cmax, 60 g, 0-50 m, juvenile set [D], g/g/d: Jan 0.119, Feb 0.097, Mar 0.092, Apr 0.114,
  May 0.153, Jun 0.190, Jul 0.210, Aug 0.216, Sep 0.217, Oct 0.214, Nov 0.197, Dec 0.165
  (optimum 0.225). Adult set: Jan 0.140 ... Mar 0.109 ... Aug-Sep 0.221 ... Dec 0.186.
  In kJ/g/d at 2.85 kJ/g: 0.26 (Mar) to 0.62 (Sep); at 3.6 kJ/g: 0.33 to 0.78.

### 2.2 Herring - measured maximum and field rations (checks)
- Lab ad libitum (De Silva & Balbontin 1974, J. Fish Biol. 6: 645): O-group herring "fed to
  satiation twice daily" at 14.5 and 6.5 C over 19 weeks; "Food intake per fish at 14.5 C was about
  three times that at 6.5 C"; "weight exponent ... at 14.5 C was 0.744" [A] (abstract via OpenAlex).
  Rations: "juvenile herring (6.1 to 18.8 g wet weight) ... mysids, minced mussels and squid ...
  fed ad libitum rations at 14.5 (8.1 - 12.7% BW day-1) and 6.5 C (3.5 - 5.0% BW day-1)" [S]
  Bernreuther 2007 thesis, p. 89-90. "De Silva & Balbontin (1974) reported values between 12 and
  4 % depending on temperature and length of the fish ... (6 to 18 g WWT)" [S] Arrhenius & Hansson
  1994, MEPS 110, p. 148.
  - [D] The Wisconsin Cmax at the same sizes and temperatures is 30-40 %/d at 14.5 C and 17-26 %/d
    at 6.5 C, i.e. 3-4x the reported ad-libitum intake. The Wisconsin temperature factor gives
    only 1.5-1.8x between 6.5 and 14.5 C, against ~2.5-3x measured. I could not resolve how 0.642
    was derived from these data (paper paywalled; not stated whether rations were wet or dry
    weight of food). The field check below (adult NSS herring) is what discriminates between the two
    readings.
  - [D] A curve through the lab midpoints (10.4 %/d at 14.5 C, 4.25 %/d at 6.5 C, 10.7 g; Q10 3.1;
    mass exponent -0.256) gives a 60 g herring 2.4 %/d (Mar) to 6.2 %/d (Sep), annual mean of months
    4.0 %/d. At ~3.7 kJ/g food (judgement for mysids/mussel/squid) that is 0.147 kJ/g/d (year),
    0.161 (season) = 3.3-3.6 %/d on 4.5 kJ/g, h 112-123.
- Field, adult NSS herring (Norwegian Sea, 28-38 cm, Apr-mid-Sep, ambient 2-6 C): "NSS herring
  consumed 0.05 g prey g fish-1 day-1 ... equivalent to ... 6% for NSS herring" [F] Bachiller 2018,
  p. 13. This is a realised ration back-calculated from growth (C = R+F+E+S+G), not a Cmax. Bachiller
  2018 contains NO Cmax function (Table 6 lists respiration, egestion, excretion, SDA and growth
  only) [F]. [D] The Wisconsin adult Cmax at 300 g is 5.0 %/d at 4 C and 8.7 %/d at 6 C, so the NSS
  herring eat at P ~0.6-1 of it; the lab curve extrapolated to 300 g and 4 C gives only 1.4 %/d,
  below the realised field ration (by ~3.6x). This is why the Wisconsin value is kept as the Cmax.
  Caveat: "2-6 C" is Bachiller's descriptor of the herring's habitat, and the ration is
  back-calculated from growth and fat deposition.
- Field, YOY Baltic herring (0.04-3 g, 8.7-17.4 C): daily ration 2.1-17 % BW wet; the
  Rudstam-model P-values (fraction of Cmax) were 0.11-0.18 [F] Arrhenius & Hansson 1994, Table 1.
  Even well-fed small fish ate < 20 % of the Wisconsin Cmax.
- Field, North Sea annual mean: herring 0.0120 g/g/d, sprat 0.0145 [F] Heath 2005, Table 3 (from
  Greenstreet 1997). Realised annual means, 5-10x below Cmax; they are not ceilings.
- Max single meal (herring cecum storage): mean 5.6 %, max 9.8 % BW wet [F] Bernreuther 2007,
  thesis manuscript on stomach/cecum storage (lab, stained copepods; page not noted). Meal size, not a daily ration.

### 2.3 Sprat
- No sprat-specific Cmax function was obtained. Baltic models used the herring function for sprat:
  "In the bioenergetics model, sprat were assumed to have the same characteristics as equally sized
  herring (Rudstam et al. 1992)" [F] Arrhenius & Hansson 1993, p. 126. Frisk 2012 (DTU thesis) and
  Frisk et al. 2015 (Ecol. Model. 299: 130) parameterised a sprat Wisconsin model; full text not
  obtained (DTU Orbit 403; paywalled).
- Herring function at 10 g [D]: optimum 0.356 g/g/d; mean of months 0.261 (year), 0.285 (season).
- Checks:
  - YOY Baltic sprat (35-84 mm): "Daily food intake ... ranging from 11% of body wet weight d-1 in
    August to 2% in the beginning of November"; "The actual food intake rates differ significantly
    from the rates estimated from a bioenergetics model based on parameters from a mixture of young
    and adult clupeids" [A] Arrhenius 1998, ICES JMS 55: 319.
  - Sprat 11-13 cm, Baltic, August 2015: realised daily rations 3.4-4.8 %BW (new method) and
    2.4-3.6 %BW (established method), Table 6.1 [F] Kulke 2018 thesis.
  - Sprat, Bornholm Basin 2002: bioenergetics-model daily ration 2.2-3.3 %BW (age 1, Jun-Nov);
    gastric-evacuation model <= 1.35 %BW [F] Bernreuther 2007, Manuscript 7.
  - Post-larval sprat (lab): maintenance and maximum consumption 5.5 and 42 % of somatic energy
    content per day [S] (search snippet of Peck et al. 2015, Env. Biol. Fish. 98: 1117). Post-larvae
    only; not transferable to 10 g fish.
  - [D] Lab herring curve at 10 g: 3.7-9.8 %/d (Mar-Sep), mean 6.3 %/d.
- Sprat weight is 3 %, so this choice moves the group value by < 5 %.

### 2.4 Norway pout
- No Cmax function or lab maximum ration exists (search; FB4 has no row; SMS has pout as prey only)
  [F/S, pout agent].
- Proxy: Hansson et al. 1996 cod Cmax (ICES JMS 53: 107), Table 2: a 0.099, b -0.195, Topt 13.7,
  Tmax 21, Q 2.41 (Kitchell form), "Cmax is maximum consumption in g g-1 d-1, W is wet weight in
  g"; fitted to "experiments on cod fed at or near maximum rations (temperature range 3-16 C, size
  range 35-750 g)" (Jobling 1982, 1988; footnote a). Energy densities in the same paper: cod
  1000 cal/g; prey "fat fish: 1400 cal g-1, lean fish: 1000 cal g-1, and others: 900 cal g-1"
  (p. 110) [F]. The ration is wet mass of mostly fish food (4.2-5.9 kJ/g).
- [D] 25 g pout (below the 35 g fitted range): 3.5 %/d at 6 C, 3.8 at 7 C, 4.1 at 8 C; on the 50 m
  table, mean of months 4.2 %/d (3.7-5.1). At 4.18 kJ/g: 0.16-0.18 kJ/g/d = 3.7-3.9 %/d on 4.5 kJ/g,
  h 102-107. Monthly, 50 m table [D], g/g/d: Jan 0.040, Feb 0.039, Mar 0.037, Apr 0.037,
  May 0.038, Jun 0.039, Jul 0.041, Aug 0.043, Sep 0.048, Oct 0.051, Nov 0.049, Dec 0.045.
- Realised rations: "For Norway pout, daily wet weight ration was estimated as 1.1% of body weight"
  (11-15 cm, July, 140 m off SW Norway, copepods) [F] Albert 1995 ICES JMS 52: 873, p. 881. North
  Sea annual mean 0.0122 g/g/d [F] Heath 2005 Table 3. Ecopath Q/B 5.05/yr = 1.38 %/d [F]
  Mackinson & Daskalov 2007. The cod-proxy Cmax is ~3x these.

### 2.5 Blue whiting
- No Cmax function or lab ration exists. Bachiller 2018 used Hansson cod RESPIRATION parameters
  (RA 0.008, beta -0.172, RQ 1.88, Tmax 24, Topt 21, ACT 1.25) for blue whiting, not a Cmax [F].
- Proxy: cod Cmax at 100 g [D]: 2.7 %/d at 6 C, 2.9 at 7 C, 3.2 at 8 C; 50 m table mean of months
  3.2 %/d. At 4.18 kJ/g: 0.13-0.135 kJ/g/d = 2.9-3.0 %/d on 4.5 kJ/g, h 134-140.
  Monthly, 50 m table [D], g/g/d: Jan 0.031, Feb 0.030, Mar 0.028, Apr 0.028, May 0.029, Jun 0.030,
  Jul 0.032, Aug 0.033, Sep 0.036, Oct 0.039, Nov 0.037, Dec 0.034. Blue whiting is in the box in
  Q3, where the value (0.032-0.036) is slightly above the Apr-mid-Sep mean (0.031); this does not
  move the weighted group value.
- Realised: "blue whiting showed the lowest value, 0.02 g prey g fish-1 day-1" (Norwegian Sea,
  Apr-Sep, mostly euphausiids at 4.0 kJ/g) [F] Bachiller 2018 p. 13. Stomach contents 1.5 %BW
  (Barents Sea, 27 cm) [F] Dolgov 2010; FishBase empirical Q/B 9.06/yr (2.5 %/d) [S]. The proxy
  Cmax is ~1.5x the realised ration.

### 2.6 Temperature layer for the deep members
- The 50 m table has its maximum in October (11.7 C); near-bottom water at 100-200 m is ~6-8 C
  year-round. Norway pout: highest catches over 100-200 m "associated with bottom temperatures of
  6-8C" (spring), "7-9C" (autumn) [F] Raitt 1968 FAO Synopsis 33 rev. 1, p. 2:3. Blue whiting:
  "occurring in waters from 4 to 7C" (Norwegian Sea feeding) [A] Bachiller et al. 2016 PLoS ONE
  e0149238. At a flat 7 C the cod-proxy Cmax is ~9 % below the 50 m mean-of-months value
  (pout 0.038 vs 0.042 g/g/d; blue whiting 0.029 vs 0.032). Effect on the group value < 2 %.

## 3. Energy density and fat

Material from the energy agent (sources in `data/`, see README). Pedersen & Hislop 2001 (J. Fish
Biol. 59: 380, doi 10.1111/j.1095-8649.2001.tb00137.x) tables were NOT obtained (closed; Wiley,
ResearchGate, DTU Orbit blocked). Its herring rows are known via JNCC Report 633; no open source
reprints its Norway pout or sprat rows.

Method warning: Torry Advisory Note 38 (Murray & Burt) computes energy from composition, "Cal/lb =
4.5 (4.1 x % protein + 9.3 x % fat)" [F]; converted at 1 Cal/lb = 0.009224 kJ/g [D]. These run
~10-20 % below bomb calorimetry; do not mix them with Spitz 2010 or Pedersen & Hislop values.

General relations:
- Pedersen & Hislop 2001: "A general relationship was established for gadoids and sandeel
  ED=-3.1492+0.3459 DS and herring ED=-4.6395+0.4170 DS" (DS = % dry mass) [A].
- Herring "% water = 80.1 - 0.94 x % fat" [F] (Torry). Combined with the herring ED relation [D]:
  ED = 3.66 + 0.392 x fat % (2 % -> 4.4, 10 % -> 7.6, 16 % -> 9.9 kJ/g). The slope matches the
  model's 0.396 kJ/g per fat %; the intercept is ~0.3 kJ/g below the model's lean 4.0.
- The gadoid relation reproduces Spitz 2010 measured ED within 2 % for blue whiting, T. luscus and
  whiting, but is 16 % high for T. minutus; the herring relation is 15 % high for sprat [D].

### 3.1 Herring
- North Sea ED by quarter and length (Pedersen & Hislop 2001 via JNCC 633 Table 2, p. 5-6) [S]:
  Q1 15.0-19.9 cm 4.4, 20.0-24.9 cm 6.5, 25-40 cm 8.5 (Wanless); Q3 15.0-19.9 cm 10.1, 20.0-24.9 cm
  11.0, 25.0-39.9 cm 11.9 kJ/g. [D] With lean 4.0 and 0.396 kJ/g per %: Q1 15-20 cm ~1 % fat,
  Q3 ~15 %; 20-25 cm Q1 ~6 %, Q3 ~18 %.
- NSAS adult muscle fat (Kenyon et al. 2022): 4.5 % late May -> 16.1 % June/July -> 9.1 % in
  September; annual maximum 16-22 %, mean 18.6 % [F, existing note `Vintermetabolism sill och
  tumlare/herring_fat_cycle.md`]. Muscle, not whole body.
- Spitz 2010, one 20 cm sample: lipid 10.7 %, 10.2 kJ/g [F].
- Library: lean 4000 MJ/t, max reserve 7000 MJ/t (~18 % fat) - an adult value. The box's 1-2
  ringers peak near 15-16 % (~6100 MJ/t) and are leaner in winter (Q1 ~1 % fat at 15-20 cm) than
  the adults' ~6 %.

### 3.2 Sprat
- Torry Table 1 [F], whole fish (energy calculated [D]): 7-9 cm water 80-81 %, fat 3.8-4.1 %,
  4.06 kJ/g; 9-11 cm 75-76 %, 7.5-9.5 %, 5.63 kJ/g; 12-15 cm 70-73 %, 11.1-14.1 %, 7.66 kJ/g.
- Spitz 2010 Table 1 [F] (Bay of Biscay, autumn, 7-13 cm, n 246): water 70.9 %, protein 17.2 %,
  lipid 8.2 % [3.4-11.2], ash 2.5 %, ED 6.5 [4.8-7.3] kJ/g (bomb calorimetry).
- Wallace & Hulme 1977 (MAFF Fish. Res. Tech. Rep. 35) [F, scanned]: "F = -1.059W + 86.556"
  (fat vs water %, r -0.981, n 182; winter only). North Shields: "the fat content of large fish is
  higher than that of small fish and ... decreases between December and March" (p. 10). Fig. 8
  readings [D, +-1.5]: 10-12 cm 17-22 % fat in Dec 1969, 11-15 % in Mar 1970; 7-8 cm 9-12 % -> 3-7 %.
- Baltic annual cycle (Rojbek et al. 2014, Fig. 4b, whole fish, 85-125 mm) [F; readings D]:
  Nov 9.5-15.1 %, Feb 7.4-13.2, Mar 4.3-9.4, May 3.9-6.0, Aug 12.6-12.7; text "the lipid level
  varied between 3.6 and 17.5% of ww".
- Arrhenius 1998: YOY sprat ED rose "from 3700 to 5300 Joules g-1 wet weight" Aug -> Dec [A].
  JNCC 633: sprat 7.6 kJ/g, all sizes and quarters (Wanless et al. 2005) [S]. Wadden Sea Apr-Jul
  6-16 cm 5.67 +- 1.70 kJ/g (Maathuis et al. 2025 Table 2) [F].
- [D] Proxy ED = 2.98 + 0.394 x fat % (Wallace & Hulme fat-water + herring relation, +-15 %):
  8.2 % fat -> 6.2 (Spitz measured 6.5). The model form (4.0 + 0.396 x fat) gives 7.2 at 8.2 %,
  ~10 % high; sprat lean tissue is wetter than herring's.

### 3.3 Norway pout
- Torry Table 1 [F], whole ungutted: water 73-77 %, fat 4.2-5.1 %, protein 16.0 %; calculated
  4.43-4.80 kJ/g [D].
- Proxies, Spitz 2010 Table 1 [F], Bay of Biscay, autumn: poor cod (T. minutus) 14-18 cm water
  73.8 %, protein 18.8 %, lipid 2.8 %, ED 5.1 [5.0-5.2]; pouting (T. luscus) 22-26 cm lipid 1.8 %,
  ED 4.7 kJ/g.
- Season: "Fish were shown to be losing in condition during the winter but seemed to gain rapidly
  in condition in the spring" [F] Raitt 1968, FAO synopsis. [D] Lowest ED in Q1-Q2 (spawning
  Jan-Apr), highest in Q3-Q4. No fat by month found.
- Gadoid: fat sits mainly in the liver, so whole-body fat stays low (~2-5 %).

### 3.4 Blue whiting
- Faroese landings 2001, whole fish, Soxhlet (Jacobsen et al., ICES CM 2002/O:15) [F]: "The fat
  content in blue whiting is at a minimum in April/May (just above 2%) after spawning ... maximum fat
  content in August (nearly 9%), which remains high in the remainder of the year"; "low and constant
  fat content in the fish tissue, at most about 1 %. However, the fat content, mainly in the liver,
  varies from 2 to 9% during the year." Fig. 1 by month [D, +-0.2]: Jan 6.0, Feb 5.3, Mar 3.9,
  Apr 2.5, May 2.1, Jun 5.7, Jul 6.6, Aug 8.6, Sep 7.1, Oct 6.8, Nov 6.9, Dec 6.4 % (mean 5.7;
  Jul-Sep 7.4). Model fish 27 cm; landings may be larger adults than the box's fish.
- Spitz 2010 [F], 14-20 cm, autumn: water 77.9 %, protein 17.4 %, lipid 1.5 % [1.1-1.7], ED 4.4
  [4.0-4.7] kJ/g (juveniles).
- Torry [F]: water 79-80 %, fat 1.9-3.0 %, protein 13.8-15.9 %, 3.1-3.9 kJ/g (calculated).
- Bachiller 2018: muscle 4500 J/g assumed from cod, plus a liver term - an assumption, not data [F].
- [D] Q3 ED for 20-30 cm, fat replacing water from the Spitz composition: 7 % fat -> 6.4-6.6,
  9 % -> 7.1-7.4 kJ/g; May (2 %) ~4.6. Treat 6.3-7.3 in Jul-Sep as an upper bound.

## 4. Q10 and reference temperature

- Herring and sprat: Q10 2.2 (Meskendahl 2013 sprat respirometry, 9-21 C: pooled RS c 0.078 /C =
  Q10 2.18) [F, existing note `Vintermetabolism sill och tumlare/q10_metabolism.md`]. Not redone.
- Norway pout, blue whiting, poor cod, bib: no respirometry found (web search; FishBase oxygen
  table has no rows for these species) [F/S, pout agent]. Proxies: cod SMR Q10 2.2 over 5-15 C
  (2.6 over 5-10 C; Schurmann & Steffensen 1997) [A, existing note]; Hansson cod respiration curve
  apparent Q10 1.7 over 6-10 C [D]; within-species teleost acclimation median 2.40, across species
  1.83 (Clarke & Johnston 1999) [A]. Recommendation: Q10 2.2 for both (same as the herring value),
  so the group keeps one Q10.
- Reference layer: pout and blue whiting live in 6-9 C bottom/deep water and never see the 0-50 m
  summer peak (13.8 C) or the 50 m October peak (11.7 C). [D] With Q10 2.2 a flat 7 C costs 0.83x
  the 0-50 m annual mean; with t_ref 14 C the monthly multiplier averages 0.71 on 0-50 m, 0.65 on
  the 50 m table and 0.58 at a flat 7 C. Pout + blue whiting are 18 % of the group, so using their
  own layer would lower the group's mean resting cost by ~3 % (0.18 x (1 - 0.83)). The group has one
  temperature layer; keeping 0-50 m (herring 79 %) is fine. If anything, the deep share argues for a
  slightly colder effective layer, not for the 50 m table, whose autumn peak the deep members do not
  experience.

## 5. Sources

| Source | What was used | Status |
|---|---|---|
| FB4 `Parameters_official.csv` (Deslauriers et al. 2017; local `../benthivores/data/fb4/`) | herring juvenile/adult and cod Cmax parameters | [F] |
| Rose et al. 2008, Trans. Am. Fish. Soc. 137: 278 (`data/rose2008/`) | herring Cmax Table 1 | [F] |
| Rose et al. 2007, Ecol. Model. 202: 184 (`data/megrey2007/rose2007.pdf`) | units g prey/g fish/d, CALz/CALf conversion | [F] |
| PICES Sci. Rep. 20 (2002), MODEL/REX (`data/pices_rep20/`) | Rudstam took CA, CB from De Silva & Balbontin; CALz 2580, CALf 5533 J/g | [F] |
| Megrey et al. 2007, Ecol. Model. 202: 144 | NEMURO.FISH herring | not obtained (constants via Rose 2007/2008, PICES 20) |
| Rudstam 1988, Kieler Meeresforsch. Sonderh. 6: 312 | original herring model | not obtained |
| De Silva & Balbontin 1974, J. Fish Biol. 6: 645 | lab ad-lib rations, exponent 0.744, 3x from 6.5 to 14.5 C | [A]; rations [S] via Bernreuther 2007, Arrhenius & Hansson 1994 |
| Arrhenius & Hansson 1993, MEPS 96: 125 (`data/arrhenius_hansson1993/`) | Rudstam model with zooplankton 2850 J/g; sprat = herring | [F] |
| Arrhenius & Hansson 1994, MEPS 110: 145 (`data/arrhenius_hansson1994/`) | YOY rations, P 0.11-0.18 | [F] |
| Arrhenius & Hansson 1993, ICES CM 1993/L:23 (`data/vliz275185/`) | YOY 20 -> 4 %/d | [F] |
| Arrhenius 1998, ICES JMS 55: 319 | YOY sprat 11 -> 2 %/d; ED 3700 -> 5300 J/g | [A] |
| Bachiller et al. 2018, PLoS ONE 13: e0190345 (`data/bachiller2018/`) | no Cmax; realised rations; prey ED; BW respiration from cod | [F] |
| Bernreuther 2007, PhD thesis Hamburg (`data/ediss1765/`) | De Silva rations and foods; sprat rations; meal size | [F] |
| Kulke 2018, PhD thesis Hamburg (`data/ediss7589/`) | sprat 11-13 cm daily rations | [F] |
| Meskendahl 2013, PhD thesis Hamburg (`data/ediss4862/`) | sprat Q10 2.2; sprat Cmax Q10 "lower" (unpublished) | [F] |
| Frisk 2012 thesis / Frisk et al. 2015 Ecol. Model. 299: 130 | sprat Wisconsin model | not obtained |
| Peck et al. 2015, Env. Biol. Fish. 98: 1117 | post-larval sprat max 42 % of energy/d | [S] snippet |
| Hansson et al. 1996, ICES JMS 53: 107 (`data/hansson1996/`) | cod Cmax, its data basis and prey ED | [F] (pout agent) |
| Heath 2005, ICES JMS 62: 847 (`data/heath2005/`) | annual rations herring, sprat, pout | [F] |
| Greenstreet et al. 1997 (`data/greenstreet1997/`) | how the pout ration was derived | [F] |
| Albert 1995, ICES JMS 52: 873 (`data/albert1995/`); Albert 1994 (`data/albert1994/`, scan) | pout ration 1.1 %/d; depth | [F] |
| Mackinson & Daskalov 2007, Cefas TR 142 | pout Q/B 5.05, BW Q/B 9.06 (FishBase formula) | [F] |
| Dolgov 2010, IIASA IR-10-035 (`data/dolgov2010/`) | BW stomach content 1.5 %BW | [F] |
| Raitt 1968, FAO Fish. Synopsis 33 rev. 1 (`data/fao_raitt1968_pout/`) | pout bottom temperatures 6-9 C | [F] |
| Bachiller et al. 2016, PLoS ONE e0149238 | BW 4-7 C | [A] |
| Schurmann & Steffensen 1997; Clarke & Johnston 1999 | Q10 proxies | [A] via existing note |
| Pedersen & Hislop 2001, J. Fish Biol. 59: 380 | ED vs dry mass relations; quarterly ED (herring via JNCC 633) | [A]; tables not obtained |
| JNCC Report 633 (Ransijn et al. 2019; local `../benthivores/data/jncc633/`) | herring ED by quarter/length; sprat 7.6 | [F] (values [S] from P&H 2001, Wanless 2005) |
| Spitz et al. 2010 (`data/spitz2010/`) | composition and ED: sprat, blue whiting, Trisopterus, herring | [F] |
| Torry Advisory Note 38, Murray & Burt (`data/torry_composition/`) | water, fat, protein: pout, sprat, blue whiting; herring fat-water formula | [F] (energy calculated) |
| Wallace & Hulme 1977, MAFF Tech. Rep. 35 (`data/wallace_hulme1977/`) | sprat fat by size and winter month | [F] (scans) |
| Rojbek et al. 2014, ICES JMS 71: 56 (`data/rojbek2014/`) | Baltic sprat lipid cycle | [F] (figure readings) |
| Jacobsen et al., ICES CM 2002/O:15 (`data/ices_cm2002_O15/`) | blue whiting fat by month | [F] (figure readings) |
| Maathuis et al. 2025 (`data/terns2025/`) | Wadden Sea sprat ED | [F] |
| Dubreuil & Petitgas 2009 (`data/archimer6143/`); Andreasen et al. 2017 (`data/noaa52475/`) | quote P&H relations; no tables | [F] |
| Kenyon et al. 2022, ICES JMS 79: 88 | NSAS fat cycle | [F] via existing note |
| Hislop et al. 1991; Wanless et al. 2005; Arrhenius & Hansson 1996 | sprat/herring ED | not obtained |
| Greenstreet 1996 SFRR 55; Timokhina 1974; Prokopchuk & Sentyabov 2006; Bergstad 1991 | pout/BW rations | not obtained |
