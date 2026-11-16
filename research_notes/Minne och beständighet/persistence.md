# Persistence of the observed targets in a fixed ~1 km cell (2026-11-14)

Research subagent. Nothing in code or library changed.

The question: an observer remembers what it met in a neighbour cell. Is that knowledge still true at a lag of
6 h, 1 d, 3 d, 1 week or 1 month? This note covers the target side only, from field data: how long the
target's density in a FIXED cell of about 1 km persists. The memory and ranging side is in
`memory_fish.md` and `memory_mammals.md`, written by other agents.

Grades:
- [F]: measured, read in the full text.
- [F/A]: measured, read only in an abstract or snippet.
- [S]: secondary, cited by another paper.
- [D]: derived by me; the arithmetic is shown.
- [J]: judgement.

A `*` after a grade means a helper agent read the source and I have not re-read it. Every other [F] I read in
the PDF myself; the PDFs are in `papers/`.

## 0. What is measured, and how it is converted

**Quantity.** The model needs an Eulerian quantity: the target's density in a fixed cell, at t and at t + lag.
Most "school", "patch" and "aggregation" studies are Lagrangian instead: they follow the group. To get from
one to the other [D]:

    P_cell(lag) ~ P_group_persists(lag) x P(group still covers the cell | displacement D(lag), patch size L)

- If the group is habitat-bound (a bank, a trench, a spawning site, a bed), the second factor stays near 1.
- If the group drifts or swims a net D that is larger than L, the second factor goes to 0. This holds however
  long the group itself lives.
- design.md 1.2 gives the net displacement per 6 h: herring 4 km, mackerel 8, pout and whiting 0.5-0.7,
  cod 0.3, porpoise 5.5.

**Two components.** Where the data allow, I write the raw density correlation as

    rho(lag) = rho_H + (1 - rho_H) exp(-lag / T)        (within one season)

- rho_H is the share of the cell-to-cell variance that is a stationary pattern over the horizon: habitat, a
  hotspot that stays in place, the depth class.
- T is the turnover time of the remaining transient part.

For the model the raw density is what matters, because the observer remembers a density, not an anomaly.
Most of the field evidence gives instead P_hot(lag): the probability that a cell seen as high or occupied is
still high or occupied. I report that where no correlation exists.

**Discrete targets at low density [D].** When a 1 km cell holds about one detectable unit or fewer (one school,
one porpoise), the instantaneous cell density is mostly shot noise from whether the unit is there. Take the
count as Poisson(lambda_c), where lambda_c has mean m and variance V across cells. Then
var(N) = m + V, and only V / (m + V) can persist after the unit has left. Two examples:

| case | m | CV of lambda_c | persistent share |
|---|---|---|---|
| porpoise (84 t at 45 kg, 3600 km2) | ~0.5 per cell | 1 | 0.25 / 0.75 = 0.33 |
| porpoise, flatter field | ~0.5 per cell | 0.5 | 0.06 / 0.56 = 0.11 |
| planktivores by day (4.72 t/km2 at a 5 t school) | ~0.94 schools per cell | 1 | ~0.5 |

**Two traps** [J, supported by the sources in sections 1 and 3]:
- **The decay is not monotone over the day.** Pelagic fish form schools by day and a diffuse layer at night,
  and krill and Calanus migrate vertically every day. rho(6 h) can therefore be LOWER than rho(24 h). A 6 h
  lag compares day with dusk or night; a 24 h lag compares the same phase.
  - Herring school counts per nmi are high around mid-day and low at dawn and dusk (Beare, Reid & Petitgas
    2002, ICES JMS 59:469, 10.1006/jmsc.2002.1201) [F].
  - For a column-integrated density (one model cell is the whole water column) vertical migration matters
    less than the school/layer switch.
- **The 1-month lag mixes in the season.** Mackerel are nearly absent in winter, and porpoise waiting times
  differ about tenfold between February and August (section 6). I state rho(1 mo) "within season" and say
  where the seasonal term dominates.

## 1. Pelagic planktivores: herring, sprat, Norway pout, blue whiting

Three levels have different lifetimes, and the field data are almost all Lagrangian.

**(a) The single school: minutes to hours.**
- Pitcher, Misund, Fernö, Totland & Melle (1996), ICES JMS 53:449, 10.1006/jmsc.1996.0063 [F/A]. They followed
  14 Norwegian Sea herring schools for about 1 h each with high-resolution sonar. Over 230 behavioural events
  occurred at roughly 5 min intervals: approaching, joining and splitting.
- OAWRS imaging at a fixed pixel of 100 m to 1 km, inside very large shelf shoals (US east coast, other sea):
  "the temporal autocorrelation scale of population change within a very large shoal is extremely short, on the
  order of minutes".
  - Original: Makris et al. 2006, Science 311:660, 10.1126/science.1121756.
  - I read it as quoted in Jagannathan et al. 2009, MEPS 395:137, 10.3354/meps08266, so [S].
- The same group documents shoals that form and dissolve within hours:
  - Gulf of Maine herring form pre-spawning shoals at dusk.
  - Barents Sea capelin, a planktivore analogue, reached critical density about 10 min after dawn twilight and
    formed lineated shoals over 10 km long. Cod ate about 10.6 million of them within about 4 h. Outside the
    shoals the capelin density "remained low and stable" (Pednekar, Jain, Godø & Makris 2024, Commun Biol
    7:1386, 10.1038/s42003-024-06952-6) [F].
- Result: the school-level transient has T well under 6 h at a 1 km cell. Herring also move a net 4 km per
  6 h (design.md), so at 6 h nothing of the transient term is left [D].

**(b) The aggregation or hotspot (km scale): days to about 2 weeks, where the prey is anchored.** The only
quantitative Eulerian revisit study I found is on capelin, a planktivorous schooler (Newfoundland, July 2000,
spawning season; other sea). Davoren, Montevecchi & Anderson (2003), Ecol Monogr 73:463, 10.1890/02-0208 [F]:
- Survey design: 38 kHz acoustics in 2.25 km blocks. Hot spots were defined as blocks with above-average
  capelin and were revisited at intervals of 3-14 d.
- Funk Island:
  - 7 of 14 revisited hot spots held capelin at every visit over two weeks.
  - Only 3 of 14 stayed above the mean abundance at every visit.
  - In one area "capelin schools ... were always ephemeral" and "highly mobile" inside a persistent hot spot.
    In two other areas schools sat "in the exact same position on different days".
- Great Island: 4 of 5 hot spots held capelin at every visit over one week, and 2 of 5 stayed above the mean.
- [D] Selected hot spots at a 2.25 km scale:
  - P_hot(presence, 1-2 wk) = 0.5-0.8.
  - P_hot(still above the mean, 1-2 wk) = 0.2-0.4.
  - These are upper values for the model's case, because spawning capelin are tied to beaches and banks.

**(c) Habitat-bound site use: weeks to years.**
- Langård et al. (2014), MEPS 501:251, 10.3354/meps10718 [F]. In Lindåspollene, a semi-enclosed Norwegian
  system, pre-spawning herring formed one school confined to a small area of about 7000 m2 at the same trench
  in all study years. They stayed from February to April and moved north-west towards the spawning grounds
  late in the period.
- Georges Bank pre-spawning herring sit within about 40 nmi of the spawning grounds every year (Jech & Stroman
  2012, Aquat Living Resour 25:1, 10.1051/alr/2012003) [F/A]. This is too coarse for a 1 km cell.
- Juvenile herring and sprat in a North Sea tidal inlet, from a fixed bottom-mounted echosounder (Marsdiep, one
  profile every 90 min for a year; Maathuis et al. 2024, MEPS 741:169, 10.3354/meps14368) [F]:
  - Season, tide, time of day and wind together explained only 36.1 % of the deviance in log NASC.
  - The residual is printed as N(0, 1.262); it is unclear whether 1.262 is the SD or the variance.
  - At a fixed beam, most of the variability at 90 min resolution is neither cyclic nor predictable. The
    authors report no autocorrelation.

**For the Skagerrak box [J].** The box's planktivores are feeding and overwintering migrants and juveniles, not
fish waiting to spawn at a fixed site. Level (c) applies only where a depth or trench feature anchors them. The
model already carries much of that as the depth tables. I take:
- P_hot = about 0.5 at 6 h, falling to 0.3-0.4 at 3 d and 0.2-0.4 at 1 wk.
- About 0.1-0.2 at 1 month within a season.
- For raw density rho, scale by the persistent share of section 0 (about 0.5 by day): rho(6 h) about 0.2-0.4,
  rho(1 d) about 0.3-0.4 at the same diel phase, and lower beyond.

Pout and blue whiting are slower (net 0.5 km per 6 h) and more deep-water bound. Their transient term survives
longer, about 1 d [D from D and a 1 km cell], but no persistence data exist for them.

## 2. Mackerel schools

- No Eulerian data. Schools tracked by fishing-vessel sonar in the Norwegian Sea reached up to about 6 m/s, with
  dominant east and west headings (Godø et al. 2004, ICES JMS 61:1093, 10.1016/j.icesjms.2004.06.009) [F/A].
  This is Lagrangian and gives speeds only.
- The net displacement of 8 km per 6 h (design.md) and the seasonal presence (the model's x0.1 to x2.2 season)
  mean the cell-level transient is gone within a tick [D].
- No study shows mackerel returning to fixed km-scale feeding sites within a season. P_hot is therefore close to
  the base rate from 1 d on [J]. At 1 month the seasonal term dominates and is not a persistence of the cell.

## 3. Zooplankton (Calanus) and the krill layer

**Calanus and mesozooplankton.**
- I found no field Eulerian decorrelation time at a 1 km cell. Moored ADCP and echosounder studies report the
  diel and lunar cycles. Pinot & Jansá (2001), DSR I 48:1651, 10.1016/S0967-0637(00)00095-9 [F/A], found the
  night-time depth following the 29.5 d moon cycle. None reports an ACF of day-to-day anomalies that I could
  read.
- Urmy, Horne & Barbee (2012), ICES JMS 69:184, 10.1093/icesjms/fsr205 [F/A]: 18 months of a bottom-mounted
  echosounder in Monterey Bay (other sea, 900 m slope).
  - Backscatter variance rises toward long periods as a power law, with exponents -1.05 to -1.585 and peaks at
    12 h and 24 h.
  - A red spectrum like this has no single decorrelation time: day-to-day persistence is real, and the variance
    keeps growing at weeks to months.
- At a fixed point the Eulerian time scale is set mainly by advection of patches past the cell. Folt & Burns
  (1999, TREE 14:300, 10.1016/S0169-5347(99)01616-X), as cited by Klais et al. 2016 [S]: above about 1 km,
  advection, eddies and currents dominate the patchiness.
- Skagerrak currents [F/A]: the Norwegian Coastal Current had daily mean speeds of 10-40 cm/s in SKAGEX 1990,
  with near-surface speeds up to 1 m/s once a baroclinic component developed (Danielssen et al. 1997, ICES JMS
  54:753, 10.1006/jmsc.1996.0210).
- [D] T_E ~ L / U:
  - Patch scale L 5-10 km [J; plankton patches km to tens of km, Mackas, Denman & Abbott 1985 (S)] and an
    offshore surface drift U of 0.1-0.2 m/s give 0.36-0.72 km/h, so T_E is about 7-28 h.
  - With exp(-lag / T_E):
    - 6 h: 0.4-0.8
    - 1 d: 0.03-0.4
    - 3 d: ~0
  - rho_H is the part tied to fronts, retention zones or the bathymetry. It is unknown for the box; I put it at
    0.1-0.3 [J].
- The SCOPEX right-whale work shows Calanus in dense patches along tidal and plume fronts, while the
  recirculation features carry drifter time scales of 48-57 d (US GLOBEC newsletter, secondary) [S]. Patches are
  persistent where a front is anchored and ephemeral elsewhere.
- Model mismatch [J]: the model's zooplankton has movement_speed 0 and no advection. In the model its cell
  density changes only through growth and grazing, so the model's own rho will be far higher than the field's.
  The field value is still what an observer's prior should encode.

**Krill layer (Meganyctiphanes).**
- Christiansen et al. (2024), ICES JMS 81:1764, 10.1093/icesjms/fsae123 [F, full-text HTML; Skagerrak-adjacent
  inner Oslofjord]:
  - Setting: a bottom-moored, upward-facing 200 kHz WBAT at 100 m and 150 m sites; 14 deployments of 2-5 d
    between February 2020 and January 2022; currents "usually below 2 cm/s".
  - Voids in the krill layer, by day:
    - After the largest fish schools passed, voids "typically lasted 5-15 min".
    - After a trawl, voids lasted more than 60 min and "lasted for hours".
    - At night the voids "disappeared quickly".
  - The authors conclude that krill antipredator behaviour makes the layer patchy by day.
- The layer itself is a basin feature: day depth below about 70 m and night ascent to 10-30 m (Onsrud &
  Kaartvedt 1998, MEPS 171:209, 10.3354/meps171209) [F/A].
- [D/J] The voids are a minutes-to-hours transient on top of a near-stationary, habitat-bound layer: rho_H is
  high and T is short.
- Other seas show the same split:
  - St Lawrence estuary: krill aggregations at the head of the Laurentian Channel recur every season, but change
    pattern and abundance rapidly with tides and advection (Cotté & Simard 2005, MEPS 288:199) [F/A].
  - South Georgia: a mooring time series showed "step-changes" that vessel surveys did not see (Brierley et al.
    2006, Limnol Oceanogr Methods 4:18, 10.4319/lom.2006.4.18) [F/A].
- For the box's deep krill (beds of 100 m and deeper, slow deep currents) I take rho at about 0.8 at 6 h, 0.7 at
  1 d, 0.6 at 3 d, 0.5 at 1 wk and 0.3-0.5 at 1 month [J]. The model's own krill (an open pool with an 18 d
  exchange residence and a static depth shape) will be at least this persistent. That bounds the model, not
  the field.

## 4. Demersal fish: cod, haddock, whiting, saithe

**Eulerian, assemblage level.** Hammond & O'Brien (2000), ICES CM 2000/K:11 [F, poster text]:
- Setting: Yorkshire coast in January. 19 acoustic grids over 15 days in a 220 km2 box, integrating 0.5-5 m
  above the seabed. Cod were about 10 % of the catch by weight, most of the rest whiting.
- The GAM's time term says abundance "varied by about 50% in a period of two to three days".
- Spatially, "fish were found most reliably on the inshore edge" (a wreck area), with "brief appearances of
  large fish concentrations" offshore.
- [D] So there are both a persistent hotspot (habitat-bound, rho_H high there) and transient offshore
  concentrations lasting less than a few days. The 50 % is the swing of the box total, not rho.

**Lagrangian residency, coastal cod.**
- Villegas-Ríos et al. (2018), J Anim Ecol 87:1309, 10.1111/1365-2656.12872 [F, PMC full text]: Tvedestrand
  fjord on the Skagerrak coast; 64 cod tracked for about 244 d.
  - Weekly 95 % kernel home range 0.06 ± 0.06 km2.
  - [D] A circle of that area has radius sqrt(0.06/pi) = 0.14 km. A resident fjord cod stays inside one 1 km
    cell for weeks.
- Espeland et al. (2007), ICES JMS, 10.1093/icesjms/fsm028 [F/A*]: home range about 27 ha over a spawning
  season on the Skagerrak coast.
- Kristensen et al. (2021), Ecol Evol [F/A*]: 42 % of North Sea-type cod left a Skagerrak fjord versus 7 % of
  fjord-type cod. Ecotype decides residency.
- Gulf of Maine (other sea):
  - Lindholm et al. (2007), MEPS 342:239 [F/A*]: 35 % of cod showed high site fidelity and 51 % left soon after
    release.
  - Dean et al. (2014), ICES JMS 71:1474, 10.1093/icesjms/fsu040 [F/A*]: a spawning site barely moved within or
    between years.

**Structure-bound saithe.** Uglem et al. (2009), MEPS 384:251, 10.3354/meps08001 [F/A*], salmon farms in
northern Norway:
- 63 % of tagged saithe were detected daily at the farms, spending 8-10 h/d at the cages.
- They moved 2-21 times between farms 1.6-4.7 km apart.
- This shows high rho_H at an attractor even for a mobile fish.

**Short lags.** Day:night differences in near-bed availability and semidiurnal current cycles move the catchable
or visible density in a cell within hours (Aglen et al. 1999, ICES JMS, 10.1006/jmsc.1999.0449) [F/A*].

**Gaps.** I found no telemetry or repeat-haul persistence data for haddock or whiting, and none for cod on the
open shelf at 100-300 m.

**My reading [J]:**
- Benthivores (cod, flatfish, haddock; net 0.1-0.7 km per 6 h): P_hot about 0.8 at 6 h, 0.7 at 1 d, 0.6 at 3 d,
  0.5 at 1 wk, 0.3 at 1 month.
- Whiting (net 0.7 km per 6 h, mostly whiting in the Hammond box): about 0.7 / 0.6 / 0.4 / 0.3 / 0.2.
- Water column (saithe-dominated, net 3 km per 6 h): about 0.4 / 0.3 / 0.2 / 0.2 / 0.1 away from attractors.
  The hake, ling and anglerfish members are sub-cell residents and much higher.

## 5. Benthic infauna, mobile epifauna, Nephrops

- **Nephrops.**
  - Power et al. (2019), Sci Data 6:7, 10.1038/s41597-019-0013-x [F*]: mean distance from release about 231 m
    after 340-726 d at liberty. Recaptures were by pots, which may bound the measured distances.
  - Burrows are persistent structures.
  - The accessible part cycles. About 1 visible animal per 10 burrows at the peak windows (Aguzzi et al. 2021,
    Sci Rep 11:5797, 10.1038/s41598-021-85240-3) [F*]. Emergence is tidal in the Bay of Biscay (Bahamon et al.
    2025, ICES JMS) [F/A*]. The Skagerrak is near-microtidal, so diel emergence probably dominates there [J].
- **Infauna.** Recovery after disturbance takes months for biota of naturally disturbed habitats and years for
  muddy-sand and long-lived biota (Kaiser et al. 2006, MEPS 311:1, 10.3354/meps311001) [F/A*]. So a depletion
  persists over months as well [D].
- **Crabs.** Swedish and Norwegian Skagerrak tagging finds stationary males and migrating females (Ungfors et
  al. 2007, Fish Res; Sarsia 1996) [S*]. No numbers.
- [D] The stock density in a 1 km cell persists to rho of about 1 at all lags up to 1 month. Only the
  accessible or exposed share flickers, at diel or tidal periods; in the model that is the refuge exchange.

## 6. Mammals as targets

**Harbour porpoise, fixed-site acoustics (western Baltic, Nysted).**
- Carstensen, Henriksen & Teilmann (2006), MEPS 321:295, 10.3354/meps321295 [F]:
  - Encounters are click series separated by more than 10 min of silence.
  - The mean waiting time between encounters was about 6 h in the baseline period.
  - Baseline waiting times "never exceeded 2 d".
  - The residuals of the log waiting times were modelled with an ARMA(1,1) structure, because a short waiting
    time tends to follow a short one. No coefficients are given.
- Teilmann & Carstensen (2012), Environ Res Lett 7:045101, 10.1088/1748-9326/7/4/045101 [F]:
  - T-POD maximum detection range 350 m.
  - Encounter duration 2.6 min (February) to 4.2 min (April).
  - Waiting times "from 59 h in February to 5.6 in August".
  - About three encounters per T-POD per week in January-March, versus several per day in summer.
- Horns Rev, North Sea: mean interval 5.9 h during construction and 7.5 h after piling (Tougaard et al. 2009,
  JASA 126:11, 10.1121/1.3132523) [F/A*].

**Wadden Sea.** Zein et al. (2019), PLoS ONE 14:e0213348, 10.1371/journal.pone.0213348 [F]: after removing the
diel, tidal and seasonal cycles, the ACF plots "showed significant temporal autocorrelation of the residuals".
GEE blocks were set from those plots at 1000 min (about 17 h) for T-PODs and 3000 min (about 50 h) for
C-PODs. This is a visual correlation length, not a fitted T.

**Scale.**
- Effective detection radius about 0.1-0.19 km, i.e. 3-11 % of a 1 km2 cell [F/A* Kyhn et al. 2012; Nuuttila
  et al. 2018; D].
- GPS-tagged porpoises in the Skagerrak and Wadden Sea average 0.65 m/s (van Beest et al. 2018, Mar Biol
  165:95, 10.1007/s00227-018-3346-7) [F*]. [D] 0.65 m/s = 2.3 km/h, so one animal crosses a 1 km cell in about
  26 min.
- Bay of Fundy (other sea): one porpoise had at least one position within 5 km of a tidal eddy on 71 % of 45
  days, and within 10 km on 93 % (Johnston, Westgate & Read 2005, MEPS 295:279, 10.3354/meps295279) [F]. This is
  hotspot persistence at a 5-10 km scale, not at 1 km.

**[D] Porpoise reading.**
- A given animal's presence in the cell decorrelates in under 1 h.
- What persists is the site's arrival rate. Its anomaly carries a residual correlation over about 17-50 h, and
  its mean shifts with the season and with the habitat (stable site differences: waiting time 8.8 h in the
  reference area versus 22.3 h in the impact area, Teilmann & Carstensen 2012 [F]).
- With about 0.5 animals per cell, the persistent share of the cell's variance is 0.1-0.35 (section 0).
- Hence rho(6 h) is about 0.1-0.35, falling to the rho_H floor of about 0.1-0.3 [J] from 2-3 d on.

**Harbour seal, foraging-site reuse.**
- Iorio-Merlo et al. (2022), Proc R Soc B 289:20212261, 10.1098/rspb.2021.2261, Moray Firth, 31 seals with GPS
  tags:
  - Trips last 38.65 ± 34.79 h [F]. About 10 trips per month [F*].
  - At sea, 57 % of the time is area-restricted search [F*].
  - Spatial memory of earlier ARS on a 1 km x 1 km grid predicted where ARS started [F]; ΔAIC 258 [F*].
- Dietz et al. (2013), ICES JMS 70:186, 10.1093/icesjms/fss144 [F*], Kattegat: mean distance to the haul-out
  9.7 km in summer and 36.9 km in winter. The 90 % home range grows from 1722 to 10 608 km2.
- Russell et al. (2014), Curr Biol 24:R638 [F/A*]: two seals followed the same North Sea pipeline on several
  trips, for up to 10 d at a time.
- [J] A seal's use of a foraging cell repeats trip to trip, about every 1.5 d, for weeks within a season.
  - P(a seal in the cell at t + lag | a seal there at t) is high within one ARS bout (hours).
  - It drops at 6 h to 1 d (the trip ends and the seal hauls out).
  - It recovers on the next trips.
  - So rho is non-monotone: low at 6 h to 1 d, with a stable foraging-patch floor through 1-4 weeks.
  - Seal density is lower than porpoise density, so the shot-noise share is larger and rho at the cell level is
    lower still.

## 7. Table

P_hot = P(a cell seen high or occupied is still so). rho = raw density autocorrelation. "~" = central value,
range in parentheses. Every row is within one season.

| target | 6 h | 1 d | 3 d | 1 wk | 1 mo | scale | basis | grade |
|---|---|---|---|---|---|---|---|---|
| planktivore school (single unit) | ~0 | ~0 | 0 | 0 | 0 | school, 10-100 m | school events every ~5 min (Pitcher 1996), pixel autocorrelation of minutes (Makris 2006), net drift 4 km per 6 h | F/A, S, D |
| planktivore aggregation / hotspot, P_hot | ~0.5 (0.3-0.8) | ~0.5 (0.3-0.7) | ~0.4 | ~0.3 (0.2-0.5) | ~0.15 (0.1-0.2) | 1-2.25 km | Davoren 2003: presence 0.5-0.8 and above-mean 0.2-0.4 over 1-2 wk at selected hotspots (capelin, spawning season) | F (other sea) + J |
| planktivores, raw rho in a 1 km cell | 0.2-0.4 (diel flip) | 0.3-0.4 | 0.2-0.3 | 0.1-0.3 | ~0.1 | 1 km | P_hot x persistent share ~0.5 (school shot noise, section 0) | D, J |
| habitat-bound herring site (pre-spawning, trench) | ~1 | ~1 | ~1 | 0.9 | 0.7 (moves towards spawning) | 7000 m2 site | Langård 2014 | F (special case) |
| mackerel | 0.1-0.3 | ~0.1 | ~base | ~base | season, not persistence | 1 km | 8 km per 6 h, sonar-tracked schools (Godø 2004) | F/A + J |
| zooplankton (Calanus) | 0.4-0.8 | 0.1-0.4 | 0.1-0.3 (rho_H floor) | 0.1-0.3 | seasonal | 1 km | T_E = L/U = 7-28 h (Skagerrak 0.1-0.4 m/s; patches 5-10 km); red spectrum (Urmy 2012) | D, J, thin |
| krill layer (deep beds, basins) | ~0.8 | ~0.7 | ~0.6 | ~0.5 | 0.3-0.5 | 1 km, beds of 100 m and deeper | Oslofjord WBAT: voids 5-15 min to hours, layer persists over 2-5 d deployments (Christiansen 2024) | F + J |
| benthivores (cod, haddock, flatfish) | ~0.8 | ~0.7 | ~0.6 | ~0.5 | ~0.3 | 1 km | weekly home range 0.06 km2 (Villegas-Ríos 2018); about 50 % box swing in 2-3 d (Hammond 2000) | F + J |
| whiting | ~0.7 | ~0.6 | ~0.4 | ~0.3 | ~0.2 | 1 km | Hammond 2000 (assemblage mostly whiting), net 0.7 km per 6 h | F + J, thin |
| water column (saithe-dominated) | ~0.4 | ~0.3 | ~0.2 | ~0.2 | ~0.1 (attractors 0.6+) | 1 km | net 3 km per 6 h; daily farm fidelity 63 % (Uglem 2009) | F/A* + J |
| infauna, epifauna, Nephrops stock | ~1 | ~1 | ~1 | ~1 | ~0.95-1 | 1 km | Nephrops moves ~231 m in 1-2 yr; recovery takes months to years | F*, F/A* |
| Nephrops accessible share | diel/tidal flicker | ~1 (same phase) | ~1 | ~1 | ~1 | burrow | ~0.1 visible per burrow; tidal or diel emergence | F*, J |
| porpoise, cell presence | 0.1-0.35 | 0.1-0.3 | ~rho_H 0.1-0.3 | ~rho_H | season shifts the mean about 10x | 1 km (PODs sample 3-11 %) | encounters 2.6-4.2 min; waiting 5.6-59 h; residual ACF ~17-50 h (Zein 2019) | F + D |
| seal, cell use | low (bout ends) | 0.1-0.3 | 0.2-0.4 (trip repeats) | 0.2-0.4 | 0.1-0.3 (seasonal range x6) | 1 km | trips 1.6 d, memory on a 1 km grid predicts use (Iorio-Merlo 2022) | F + J, thin |

Notes on reading the table:
- **Which row to use.** The model's neighbour channel is the target's cell biomass, a group average, not an
  instantaneous count of schools or animals. The rule:
  - Biomass channel: use the P_hot or hotspot row.
  - What a real animal meets at a point (a discrete-unit encounter): use the raw rho or shot-noise row.
- Applied per target:
  - Planktivores: use "planktivore aggregation / hotspot, P_hot" (~0.5 at 6 h and 1 d, ~0.3 at 1 wk). The
    raw-rho row is what a fish at a real point would see.
  - Mackerel: the row already is the hotspot reading, and it is close to the base rate from 1 d on.
  - Porpoises and seals: the model's cell biomass of a few animals per cell is itself discrete. Use the shot-noise
    values (porpoise 0.1-0.35 at 6 h) when the model resolves animals. Use the rho_H floor (0.1-0.3) as the part
    of a remembered density that stays valid beyond 1-2 d.
- The benthic rows are stock values.
- The fixed-attractor rows (Langård, farms, wrecks) show that rho_H, not individual residency, carries the
  persistence. Most of that rho_H is already in the model as depth tables and static habitat.

## 8. Gaps

1. **No field rho(lag) for any pelagic fish at a fixed 1 km cell in the North Sea or Skagerrak.** The only
   Eulerian time series (Marsdiep inlet) report deviance shares, not an ACF. A reanalysis of a moored or
   observatory echosounder record (Marsdiep, Oslofjord WBAT) would close this.
2. **The planktivore P_hot rests on one capelin study in the spawning season, in another sea.** That study
   selected hotspots for being high, and spawning prey are more anchored than the box's feeding herring.
   Treat it as an upper bound.
3. **No measured Eulerian decorrelation for Calanus.** Only advection scaling [D] and a red spectrum from another
   sea. Model mismatch: the model's zooplankton does not drift, so its own rho is far above the field's.
4. **Krill.** The Oslofjord basin is quieter (currents below 2 cm/s) than the open Skagerrak. Persistence over
   beds of 100-150 m with stronger flow is not measured.
5. **No persistence data for haddock, whiting or offshore cod at 100-300 m.** All cod residency is coastal or
   structure-bound, and the Hammond box is in January and coastal.
6. **Porpoise.** No fitted AR coefficient or P(detection | detection) at a lag. The 17-50 h correlation length is
   a block size chosen visually. A POD samples only 3-11 % of the cell.
7. **Seals.** No measured share of trips that revisit the same 1 km cell. Only the memory model (Iorio-Merlo) and
   pipeline anecdotes.
8. **The diel non-monotonicity** (school versus layer, day versus night) is shown qualitatively (Beare 2002,
   Christiansen 2024) but not quantified as rho(6 h) versus rho(24 h).
9. **Mackerel.** No evidence of any km-scale site fidelity within a season.
