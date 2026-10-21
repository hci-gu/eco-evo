# Herring growth allocation: should somatic growth be funded from the reserve?

Question: the model's decision makers grow whenever their reserve fill s is above u, whether or not
they are eating. The herring (`pelagic_fish`) therefore keeps turning fat into biomass in Oct-Dec, enters
winter lean, and the stock loses 25-30 %/yr. A calendar switch (growth only May-Sep) removed most of it.
Is strongly seasonal somatic growth established for herring and temperate fish, what drives it, how do
established fish models fund growth, and which biologically grounded growth law could replace the
calendar?

Model state checked (read from the repo, 2026-10-20; nothing edited):
- Growth law, `lib/environments/ecosystem_env/population_change.py`: `energy_surplus = s - u`;
  dB = B g (s - u) if s >= u, else dB = B starve_rate (s - u). With mass balance (default) the gain is
  capped by and debited from the reserve above u: `E_avail = max(0, R - u ME B)`,
  `dB = min(B g q, E_avail / energy_content)`, `R -= dB energy_content`. New biomass carries no reserve,
  so growth also dilutes s. Nothing in the growth branch reads this tick's intake.
- `lib/world/growth_budget.py`: DM g = (r_max + M1 + sum M2) / ((s* - u) 1460). s* solves
  h(s*) C_max = cost + P (energy_content + s* ME) with h(s) = (1 - s)/(1 - u), C_max the assimilated
  Holling ceiling on the best prey and cost = resting_metabolism x feeding_cost.
- Herring numbers (library, and growth_budget run read-only for `mareld2.yaml`'s active set, seabirds
  muted) [D]: C_max = 144 MJ/t/tick (zooplankton, 1/h 0.04 x 4500 MJ/t x assimilation 0.8);
  cost while eating 15.2 MJ/t/tick (8 x 1.9); u 0.2; ME 7000 MJ/t; energy_content 4000 MJ/t; budget
  1.541 /yr (r_max 0.648 + M1 0.58 + gadoids 0.300 + porpoises 0.012 + seals 0.002); s* = 0.857;
  g = 0.001606 per tick.

Builds on (not redone, cited as [2nd]):
- `research_notes/Sillens vinterbudget/winter_budget.md` (sibling): the Oct-Feb reserve budget of the
  trained run (growth payment + dilution = ~38 % of the Oct-Feb fat drawdown; biomass growth 0.27 %/d
  Oct-Nov, 0.17 %/d Nov-Dec on an implied ration of 0.40 and ~0 % BW/d); IBTS condition and cohort
  weights at the site (ages 2+ lose 10-27 % of their weight Aug -> Jan-Feb); NSAS/NSS fat cycles
  (Kenyon 2022, Slotte 1996, Slotte & Fiksen 2000, Rojbek 2014); winter cost and starvation threshold.
- `research_notes/Vintermetabolism sill och tumlare/` and `research_notes/Sillens vinterföda/`
  (winter ration ~ maintenance; Varpe & Fiksen 2010).

Source tags: **[F]** full text read (pdftotext or full article page); **[A]** abstract only;
**[2nd]** second-hand, via a sibling note or a paper that cites the original; **[D]** my own data
analysis (raw data or model arithmetic); **[L]** laboratory study; **[S]** synthesis, i.e. my own
combination of numbers from different sources; **[Sn]** search-engine snippet or database summary
only, not verified. Everything under "Inference" is mine.

---

## Evidence table

| # | Source | Tag | What it gives |
|---|---|---|---|
| 1 | Growth law and budget code + library, herring | [D] | What the current law funds growth from; s*, kappa needed for the alternatives |
| 2 | ICES DATRAS NS-IBTS Q1/Q3 and BITS Q1/Q4 CA records 2016-2024 | [D] | Length at age by quarter: share of the annual length increment laid down Aug -> Jan-Feb (Kattegat, E Skagerrak) and Nov -> Feb-Mar (central Baltic) |
| 3 | Bachiller et al. 2018, PLoS ONE 13:e0190345 | [F] | NSS herring growth (feeding) period from weight-at-length change: 1 Apr - 15 Sep; length growth "drops down significantly" in winter; standard bioenergetics balance |
| 4 | Iles 1964, J Cons int Explor Mer 29:166 | [Sn] | North Sea herring: somatic growth peaks May-June, before gonad growth July-September |
| 5 | Iles 1984, chapter in Potts & Wootton (eds), Fish Reproduction | [Sn] | Exists (pp. 331-347/348); content not retrieved |
| 6 | Foy & Paul 1999, TAFS 128:1193 | [A] | Age-0 Pacific herring: energy density up Oct -> Nov (4.4 -> 5.0 kJ/g) then down to 3.9 in March; length still increasing Oct-Mar (growth or size-selective loss); photoperiod suggested over temperature |
| 7 | Sewall et al. 2019, MEPS 623:195 | [A] | Age-0 Pacific herring: shift from protein growth to lipid storage at ~76 mm FL in November |
| 8 | Arrhenius & Hansson 1994, MEPS 110:145 | [F] | Rudstam 1988 herring model = Hewett & Johnson (Wisconsin) framework, consumption from growth; YOY ration 17 % BW/d in summer -> 2 % at end of October |
| 9 | Bernreuther et al. 2013, J Appl Ichthyol 29:331 (doi 10.1111/jai.12045) | [A] [L] | Juvenile Atlantic herring at 16 C: maintenance ration 2.0 % DM/d, net growth efficiency 42 %, gross 25 % |
| 10 | dos Santos Schmidt et al. 2022, JEMBE 552:151737 | [F] [L] | Herring reared 3.5 yr at ~9 C, ad libitum: photoperiod (6-month offset) shifts vitellogenesis by 6 months; no group difference in length, weight or condition; "relatively high body reserves throughout the year, even during the winter" |
| 11 | Kurita, Meier & Kjesbu 2003, J Sea Res 49:203 | [Sn] | NSS herring females (capital breeder): ovary weight correlates with muscle condition late in maturation |
| 12 | Metcalfe & Thorpe 1992, J Anim Ecol 61:175 | [A] [L] | Salmon parr: appetite -96 % Sep -> Dec "even in the presence of excess food"; growth virtually stops; appetite defends an energy level |
| 13 | Metcalfe, Bull & Mangel 2002, Evol Ecol Res 4:871 | [A] [L] | Same deficit imposed in winter vs summer: winter refeeding restores lipid with negligible length growth; summer refeeding restores lipid and length |
| 14 | Bull, Metcalfe & Mangel 1996, Proc R Soc B 263:13 | [Sn] | Salmon winter anorexia modelled as appetite set by long-term energy insurance (abstract elided; database summary only) |
| 15 | Sæther, Johnsen & Jobling 1996, J Fish Biol 48:1113 (doi 10.1006/jfbi.1996.0114) | [A] [L] | Arctic charr at 4 C, excess food: feeding and growth cycle (high mid-summer, low autumn) persists under constant 12:12 LD |
| 16 | Jones et al. 2002, J Anim Ecol 71:916; Bacon et al. 2005, J Anim Ecol 74:1 | [A] | Salmon parr reserve/structure model fitted to field growth; late-season growth stops because summer maintenance balances assimilation; most growth in a 10-week spring window, seasonal effects beyond temperature |
| 17 | Hurst & Conover 2003, Ecology 84:3360; Hurst, Schultz & Conover 2000, TAFS 129:145 | [A] | Age-0 striped bass: lipid rises faster than length in autumn and winter; lipid stored before winter |
| 18 | Post & Parkinson 2001, Ecology 82:1040 | [A] | Age-0 rainbow trout: growth vs storage strategies; storage pays for fast-growing cohorts |
| 19 | Hurst 2007, J Fish Biol 71:315 | [A] | Review of winter mortality in fishes (causes, size selection, management); no allocation rule |
| 20 | Boeuf & Le Bail 1999, Aquaculture 177:129 | [A] | Review: long day length stimulates growth in many fish (salmonids, marine); light acts on melatonin, GH, thyroid hormones |
| 21 | Imsland et al. 2007, J Fish Biol 71:1371 (doi 10.1111/j.1095-8649.2007.01600.x) | [A] [L] | Cod: 3 months of continuous light as juveniles -> 1-9 % larger at harvest |
| 22 | Martin, Heintz, Danner & Nisbet 2017, J Anim Ecol 86:812 (doi 10.1111/1365-2656.12667) | [A] | DEBlipid: kappa of assimilation pays maintenance then growth; 1 - kappa to storage lipid; energy density rises fastest in autumn across fish species |
| 23 | Broekhuizen et al. 1994, Funct Ecol 8:770 | [A] | Reserve/structure compensatory-growth model; allocation controlled by the reserve:structure ratio |
| 24 | Gurney, Jones, Veitch & Nisbet 2003, Ecology 84:2777 | [F] | Same model class, equations: structure grows only from the net assimilation stream, dS/dt = k p A; k = 0 below a reserve-target control range |
| 25 | Ortiz Durán, van der Meer & van de Wolfshaar 2026, Ecol Model 522:111828 | [F] | Standard DEB and its failure under seasonal food; DEB-Storage: structure growth = 0 when kappa x mobilisation < maintenance, deficit paid from a storage pool |
| 26 | Deslauriers et al. 2017, Fisheries 42:586 (Fish Bioenergetics 4.0) | [A] | Wisconsin model software and parameter library (105 models, 73 species) |
| 27 | Megrey et al. 2007, Ecol Model 202:144; Rose et al. 2008, TAFS 137:278 | [Sn] | NEMURO.FISH Pacific herring bioenergetics (DOIs confirmed, abstracts elided) |
| 28 | Persson et al. 1998, Theor Pop Biol 54:270 | [Sn] | Size-structured consumer model with reversible/irreversible mass (DOI confirmed, abstract elided) |
| 29 | Varpe, Jørgensen, Tarling & Fiksen 2009, Oikos 118:363 (doi 10.1111/j.1600-0706.2008.17036.x) | [A] | Capital vs income breeding in a seasonal state-dependent model (zooplankton) |
| 30 | Morgan, Rideout & Colbourne 2010, MEPS 404:185 | [A] | Cod: temperature did not change growth in length or weight; it changed condition and reproductive investment |

---

## 1. Is somatic growth strongly seasonal in herring and temperate fish?

### Literature
- **NSS herring.** Bachiller et al. 2018 [F] defined the growth period from the change in weight at
  length of the 32 cm group (2005-2010): **1 April (day 91) to 15 September (day 258)**. They state that
  the three pelagic stocks "grow during the feeding season, whereas during winter the feeding nearly
  stops and the length growth drops down significantly". Their consumption model runs only for that
  window.
- **North Sea herring.** "The period of most rapid somatic growth (May and June) precedes that of rapid
  gonad growth (July, August and September)" (Iles 1964, via search snippet [Sn]; full text not read).
- **Fat cycle (sibling, [2nd]).** NSAS muscle fat minimum 4.5 % in late May, 16.1 % by late June
  (Kenyon 2022). This is weight at length, not structure.
- **Juvenile Pacific herring.** Standard length kept increasing from October to March, "indicating either
  growth or mortality of smaller fish". Energy density rose from 4.4 to 5.0 kJ/g between October and
  November and then fell to 3.9 kJ/g in March (Foy & Paul 1999) [A].
- **Other temperate fish.** Salmon parr in the Girnock Burn did most of their growth "in a 10-week period
  in spring" (Bacon et al. 2005) [A]. Arctic charr fed in excess had growth "highest in mid-summer and
  lowest in autumn" (Sæther et al. 1996) [A] [L]. Striped bass and rainbow trout juveniles store
  lipid before winter and deplete it over winter (Hurst & Conover 2003; Hurst et al. 2000; Post &
  Parkinson 2001) [A].
- **Annuli.** Herring are aged by winter rings (the IBTS data used here and in the sibling note give age
  in winter rings). I did not retrieve a herring otolith or scale study that dates annulus formation or
  the end of the opaque zone by month (Not found).

### Data: length growth by season, IBTS/BITS [D]
Source: ICES DATRAS CA records (getCAdata), herring (Aphia 126417), 2016-2024, downloaded 2026-10-20.
- NS-IBTS: Q1 = Jan-Feb, Q3 = August. Kattegat = rectangles 41-43 x G0-G2. Eastern Skagerrak = the
  sibling's box (44F9, 44G0, 44G1, 45F9, 45G0, 45G1, 46F9, 46G0).
- BITS: all aged herring in BITS come from Polish and Russian hauls in the central Baltic (rectangles
  37-41 x G5-H0). No aged BITS herring exist for the Kattegat or the western Baltic in these years.
  BITS Q1 is normally Feb-Mar and Q4 November (survey design, not checked per haul).
- Method: a cohort is followed as age a in Q1 of year y, age a in Q3 (or Q4) of year y, and age a + 1 in
  Q1 of year y + 1. Mean length is weighted by `CANoAtLngt`, and a cell needs >= 15 fish. "In-season" =
  Q1 -> Q3/Q4 of the same year. "Off-season" = Q3/Q4 -> next Q1. Rows with fewer than 2 cohorts are
  dropped.

| Region, age in Q1 | in-season (Jan-Feb -> Aug), mm | off-season (Aug -> Jan-Feb), mm | off-season share of the year's length increment | cohorts (in / off) |
|---|---|---|---|---|
| Kattegat, 1-group | +43.4 | +13.9 (sd 5.6) | 0.24 | 3 / 5 |
| Kattegat, 2-group | +31.8 | +2.4 (sd 15.0) | 0.07 | 3 / 4 |
| E Skagerrak, 1-group | +45.9 | +22.0 (sd 8.4) | 0.32 | 3 / 5 |
| E Skagerrak, 2-group | +24.8 | +5.5 (sd 3.5) | 0.18 | 3 / 5 |
| E Skagerrak, 3-group | +12.0 | +6.5 (sd 6.2) | 0.35 | 3 / 5 |
| Kattegat / E Skagerrak, 0-group (Aug -> Jan-Feb) | - | +14.4 / +11.8 | - | 0 / 5 |

| Central Baltic, age in Q4 | off-season (Nov -> Feb-Mar), mm | cohorts |
|---|---|---|
| 0-group | +3.1 | 2 |
| 1-group | +4.6 | 2 |
| 2-group | +3.1 | 2 |

For scale, the one central Baltic cohort followed through a whole year (2023, 1- and 2-group) grew
24 and 16 mm from Q1 to Q4.

Caveats:
- "Off-season" from August still contains the late August-October part of the feeding season. These
  shares are **upper bounds on the length growth after the feeding season**, not winter growth rates.
- Who is sampled changes between quarters: WBSS adults leave the area in autumn (sibling, Payne 2009),
  smaller fish die faster, and the GOV and TV3 trawls are size selective. Foy & Paul 1999 make the same
  caveat for their own length series. Some off-season increments (e.g. the 0-groups) could be
  selection, not growth.
- 3-5 cohorts per row, and ageing error is not handled.

### Inference
- For herring, the evidence for a length-growth season is direct for NSS herring (Bachiller: Apr-mid
  Sep; length growth drops in winter) and North Sea herring (Iles 1964 [Sn]: peak May-June). It is
  consistent with the eastern Skagerrak/Kattegat data: for 2- and 3-ringers, 65-93 % of the yearly length
  increment is laid down between Jan-Feb and August. In the central Baltic only ~3-5 mm accrue between
  November and Feb-Mar.
- Juveniles (0- and 1-group) keep adding length into autumn. Foy & Paul and Sewall find the same in
  Pacific herring, and Sewall puts the switch to storage at ~76 mm in November. So the growth season is
  size dependent: it ends earlier the larger the fish.
- The model's 0.27 %/d (Oct-Nov) and 0.17 %/d (Nov-Dec) biomass growth on 0-0.4 % BW/d rations has no
  counterpart in adults. For the juvenile-dominated winter population at the site it is too high but
  not absurd for the 0-1 group in October.

---

## 2. What drives the seasonality?

### (a) Food availability alone
- For: NSS herring hardly feed in winter (Bachiller [F]; Slotte, sibling [2nd]). Pacific juveniles
  stopped meeting metabolic needs from feeding after November (Foy & Paul [A]). Baltic YOY ration fell
  from 17 % BW/d in summer to 2 % at the end of October (Arrhenius & Hansson 1994 [F]). The site's
  winter ration is about maintenance (sibling [2nd]).
- In the salmon parr fits, late-season growth stopped because high summer maintenance balanced still-high
  assimilation (Jones et al. 2002 [A]). That is a budget outcome, not an allocation decision.
- Against "food alone": the cases where fish with excess food still stop growing (b, c below).

### (b) A state-dependent allocation rule that stores lipid before winter
- Salmon parr with excess food: appetite fell 96 % from early September to late December, growth
  "virtually" stopped "under environmental conditions that would otherwise allow growth". Appetite rose
  again when reserves were depleted, "the salmon effectively having a 'defended energy level'"
  (Metcalfe & Thorpe 1992) [A] [L].
- The same nutritional deficit, imposed at two seasons: "In winter, fish restored lipid reserves, but
  their growth in length over the recovery period was negligible. In summer, fish allocated resources to
  growth in length as well as the restoration of lipid reserves". The authors add that "current
  physiological and energetic models of animal growth cannot account for such seasonal variation"
  (Metcalfe et al. 2002) [A] [L]. This is the clearest case of allocation, not food, deciding.
- Striped bass: lipid scaled isometrically with length in summer "but increased at a greater rate than
  did body length in autumn and winter" (Hurst & Conover 2003) [A].
- Herring: Pacific YOY shift "from protein-based growth to lipid storage" at ~76 mm FL in November
  (Sewall 2019) [A]. Juveniles stored energy Oct -> Nov while still feeding (Foy & Paul 1999) [A]. The
  PWS Science Center describes the same pattern: in autumn "a higher proportion of their food energy is
  stored as fat" (web page, no numbers) [Sn].
- Cross-species synthesis: energy density "increases fastest in periods of transition between
  favourable and unfavourable conditions for growth (i.e. fall)" (Martin et al. 2017) [A]. They
  reproduce this with a fixed kappa split, i.e. without a seasonal decision (section 4).
- Not found: any herring experiment with ample autumn food that measures structure and lipid separately.

### (c) Photoperiod and endogenous rhythms (GH/IGF-1, melatonin)
- Arctic charr at constant 4 C with excess food: the feeding and growth cycle persisted under constant
  12:12 LD. The authors read this as "circannual rhythms of food consumption and growth" (Sæther et al.
  1996) [A] [L].
- Review: long day length stimulates growth in many marine fish and salmonids. Light acts through
  photoreceptors (eye, pineal) on melatonin, somatotropin (GH) and thyroid hormones. Whether light
  improves feed conversion or only intake was open (Boeuf & Le Bail 1999) [A]. Cod given continuous light
  as juveniles were 1-9 % larger at harvest (Imsland et al. 2007) [A] [L].
- Herring: photoperiod is the zeitgeber for vitellogenesis. A 6-month offset light regime shifted oocyte
  development by 6 months, and feeding opportunity did not assist it (Schmidt et al. 2022) [F] [L]. The
  same experiment found no difference in length, weight or condition between the light regimes. But the
  fish were fed ad libitum at ~9 C, had reached tank maximum size early, and carried "relatively high
  body reserves throughout the year, even during the winter period". So it is not a test of seasonal
  somatic growth. NSS spring condition tracks day length more than the Calanus peaks (Varpe & Fiksen
  2010, sibling [A]). Foy & Paul 1999 [A] suggest the length of overwinter starvation is "related more
  to photoperiod than to mean water temperature".
- Not found: any GH, IGF-1 or melatonin seasonality measured in herring.

### (d) Temperature
- Salmon parr grew fastest "at temperatures below those that previous published models report as
  necessary". The temperature effect "significantly and markedly" differed by season, and seasonal
  effects explained another 18 % of the variation (Bacon et al. 2005) [A].
- Cod on the Grand Bank: temperature did not influence growth in length or weight. It changed condition
  and reproductive investment (Morgan et al. 2010) [A].
- Juvenile Atlantic herring metabolism rises with temperature (9.7-17.9 C). The maintenance ration at
  16 C is 2.0 % DM/d (Bernreuther et al. 2013) [A] [L]. Cold slows the winter fat loss: Pacific YOY
  lipid was highest in the coldest year (Sewall 2019) [A].
- Inference: temperature sets the cost side (already in the model via Q10). No source found makes it the
  switch that stops herring growth in autumn, when the water at the site is still at its warmest.

### (e) Reproduction
- North Sea (autumn-spawning) herring: somatic growth peaks May-June, gonad growth July-September (Iles
  1964 [Sn]). Gonads are thus built from summer income, after the length growth.
- Spring spawners mature through autumn and winter on stored energy. NSS herring are called a "capital
  breeder" with ovary weight tied to condition (Kurita et al. 2003 [Sn]). Vitellogenesis is
  photoperiod-entrained (Schmidt 2022 [F]). Spring spawners spend proportionally more stored energy on
  non-reproductive metabolism during the long maturation (Bradford, sibling [Sn]).
- Iles 1984 ("Allocation of resources to gonad and soma in Atlantic herring") was not retrieved
  (section 3).
- Inference: in herring, reproduction competes with growth for the reserve in autumn (spring spawners) or
  in late summer (autumn spawners). The model has no gonad pool, so this demand is folded into r_max as
  population production. Any rule that lets the reserve pay for autumn growth also lets it pay for what
  is really gonad or winter fuel.

### Inference: weighing (a)-(e)
- Food (a) plus cost (d) explain why growth stops in herring once the ration falls to maintenance. They
  do not by themselves explain why fish keep fat instead of converting it. That needs (b), (c) or a
  model in which structure is funded from income (section 4).
- The direct "ample food, growth stops, lipid stored" evidence is salmonid and charr laboratory work.
  It shows allocation (b) and an endogenous or photoperiod rhythm (c) can each stop somatic growth with
  food present. For herring the evidence is indirect: Sewall's size-dependent switch and photoperiod
  control of maturation.

---

## 3. Citation check

| Citation as given | Exists? | What it actually is and shows |
|---|---|---|
| Hurst 2007, J Fish Biol, winter mortality in fishes | **Yes.** Hurst TP 2007, "Causes and consequences of winter mortality in fishes", J Fish Biol 71:315-345, doi 10.1111/j.1095-8649.2007.01596.x [A] | A review of documented winter mortality (many freshwater, fewer marine populations). Thermal stress and starvation are the best-studied causes; predation and pathogens are understudied. Mortality is often size selective against the smallest fish. Marine fish can migrate away from thermal stress. Management responses are proposed. **It does not give an allocation or growth rule.** Use it for "winter mortality is size selective and partly starvation", not for growth seasonality. |
| Post & Parkinson 2001, Ecology, energy allocation in young fish | **Yes.** Post JR, Parkinson EA 2001, "Energy allocation strategy in young fish: allometry and survival", Ecology 82:1040-1051, doi 10.1890/0012-9658(2001)082[1040:EASIYF]2.0.CO;2 [A] | Eight experimental cohorts of **age-0 rainbow trout**. Storage (lipid) and structure scale biphasically with length (inflexion ~10 cm). Lipid concentration depends on mass and growth rate. Compared with storage maximization, a growth-maximization strategy gives a 5 % net survival advantage for slow, small cohorts. For the fastest cohorts, storage maximization gives a 7 % advantage. So the optimal growth/storage split depends on state. It is not a herring or seasonal-switch study. |
| Iles 1984, allocation to gonad and soma in Atlantic herring | **Exists, not read.** Iles TD 1984, "Allocation of resources to gonad and soma in Atlantic herring Clupea harengus L." In: Potts GW, Wootton RJ (eds), Fish Reproduction: Strategies and Tactics, Academic Press, London, pp. 331-347 (some reference lists say 331-348) [Sn] | Book chapter, no DOI, no abstract found. Other papers cite it for the inverse relation of fat and water in herring somatic tissue [Sn]. The somatic-before-gonad timing (May-June vs Jul-Sep) comes from a **different** paper, Iles 1964, "The duration of maturation stages in herring", J Cons 29:166-188, doi 10.1093/icesjms/29.2.166 [Sn]. One search summary dated that paper 1974, but Crossref gives 1964. |
| Boeuf & Le Bail 1999, Aquaculture, light and fish growth | **Yes.** Boeuf G, Le Bail P-Y 1999, "Does light have an influence on fish growth?", Aquaculture 177:129-152, doi 10.1016/S0044-8486(99)00074-5 [A] | Review. Larvae need a minimum light intensity; long days help larval rearing when food is available. In older fish intensity effects are inconsistent, and **day length matters more: long photoperiod stimulates growth in many species** (marine fish, salmonids; Atlantic salmon best studied). The review covers melatonin, somatotropin and thyroid hormones. It leaves open whether light improves conversion efficiency or only intake. **No herring.** |

---

## 4. How do established fish models fund growth?

| Model | What funds structural growth | Growth when intake falls but lipid is high? | Tag |
|---|---|---|---|
| Wisconsin / Fish Bioenergetics (Hewett & Johnson software; FB4, Deslauriers 2017) | G = C - (R + F + E + S), S = SDA (Bachiller's Eq. 1). One body pool; G < 0 is weight loss | **No.** Growth is the residual of this step's consumption. Negative balance means weight loss, never growth | [F] (Bachiller Eq. 1), [A] (FB4) |
| Herring versions: Rudstam 1988 (Baltic); Bachiller 2018 (NSS); NEMURO.FISH (Megrey 2007; Rose 2008) | Same balance with herring constants. Rudstam's model was built in the Hewett & Johnson software and consumption is fitted from observed growth. Bachiller runs it only over the Apr-Sep growth window and adds energy density change as a separate store | **No** (same residual logic). Bachiller treats fat storage as a flux separate from somatic growth | [F] (Arrhenius & Hansson 1994; Bachiller 2018); [Sn] NEMURO.FISH |
| Standard DEB (Kooijman; via Ortiz Durán 2026) | dV/dt = (kappa v[E] V^(2/3) - [p_M] V)/(kappa [E] + [E_G]). Structure is paid from the **mobilised reserve**, whose density [E] relaxes towards f [E_m] | **Yes.** After food drops, [E] decays at a finite rate and structure keeps growing until kappa x mobilisation < maintenance. Under seasonal food limitation the standard formulation then lets structure shrink or the reproduction buffer pay, which "does not align with common empirical observations" | [F] |
| DEB-Storage (Ortiz Durán et al. 2026) | As standard DEB, plus a storage pool fed by (1 - kappa) p_C. When kappa x mobilisation < maintenance: **dV/dt = 0**, the deficit comes from storage, and structure never shrinks. Fitted to North Sea plaice | **Partly.** Growth from the reserve continues while [E] is high, but structure stops as soon as the mobilised flux cannot pay maintenance, and the lipid-like storage is protected for winter | [F] |
| DEBlipid (Martin et al. 2017) | kappa of **assimilated** energy pays maintenance first, the surplus goes to growth; 1 - kappa goes to storage lipid (juveniles) or reproduction (adults) | **No.** Growth = max(0, kappa A - maintenance) uses this step's assimilation. Autumn lipid gain emerges because the growth share hits zero before the lipid share does | [A] |
| Broekhuizen et al. 1994; Jones et al. 2002; Gurney et al. 2003 (reserve R + structure S) | dS/dt = k p A: structure is built only from the net assimilate stream A (feeding-rate costs already netted), k = share of its "protein" part. Under k-control k = 0 when mS/A > 1 - X_d, i.e. when income cannot hold the reserve fraction at its target X_d. "Under starvation conditions ... structural carbon remains constant" | **No.** With A = 0 structure is constant whatever the reserve. With low A, all income goes to reserves | [F] (Gurney); [A] (Broekhuizen, Jones) |
| Persson et al. 1998 / de Roos & Persson (reversible/irreversible mass) | Net energy after maintenance is split between reversible and irreversible mass by a condition-dependent rule | Not checked; abstract and text not retrieved | [Sn] |
| **Mareld today** | dB = B g (s - u), paid from R above u | **Yes**, at the full rate g (s - u), independent of intake | [D] |

### Inference
- The Mareld law is the standard-DEB pattern in its simplest form: growth is paid from the reserve at a
  rate set by the reserve level, with no reference to this tick's income. Its failure mode (growing on
  fat after food has gone) is the one Ortiz Durán et al. 2026 wrote DEB-Storage to fix, and the one
  Metcalfe et al. 2002 say "current physiological and energetic models" cannot capture.
- Every seasonal fish growth model in the table either funds structure from this step's income
  (Wisconsin by construction, the Jones et al. salmon fit, DEBlipid's cross-species synthesis) or stops
  it when income cannot pay maintenance (the plaice DEB-Storage fit). Standard DEB and Mareld do not.

---

## 5. Candidate replacement rules

Notation, per tonne of biomass in a cell and per tick:
- I = this tick's assimilated intake (MJ/t), already computed in predation. C = this tick's metabolic
  cost (MJ/t), already debited.
- N = I - C, the net surplus. EC = `energy_content`, ME = `max_energy_reserve`, s = R/(B ME), u =
  `maintenance_level`.
- The starvation branch (s < u) is left as it is in all three rules; the sibling note covers its
  threshold.

### Rule A: kappa-split income growth (DEBlipid form)
```
G   = max(0, kappa * I - C)            # structural growth energy, this tick's income only
dB  = B * G / EC
dR  = B * (I - C - G)                  # = (1 - kappa) I when kappa I >= C, else I - C (reserve pays deficit)
```
- Basis: Martin et al. 2017 [A]; Gurney et al. 2003 [F] (structure from the assimilate stream);
  Wisconsin residual logic [F]. A herring-specific anchor for the growth conversion is Bernreuther 2013
  [A] [L]: net growth efficiency 42 % at 16 C.
- Behaviour in autumn: growth stops as soon as I < C/kappa. In the band C < I < C/kappa the fish still
  stores fat without growing, which is the "fastest energy density increase in fall" pattern. Below C,
  the reserve pays only maintenance. No calendar is needed; the switch follows the ration.
- Calibration [D]: kappa replaces g as the derived parameter. With the same ideal conditions and hunger
  gate that `growth_budget` uses (h(s) C_max intake, cost 15.2, production P = 1.541/1460 per tick,
  dilution s ME P), the steady state is s* = 0.857, the same s* as today, and
  kappa = (cost + P EC)/(h(s*) C_max) = 19.4/25.8 = **0.75** for herring. The algebra reduces to
  `ideal_reserve` unchanged, so the s* machinery carries over. Only g -> kappa changes.

### Rule B: income growth with a reserve target (Broekhuizen/Gurney k-control)
```
k   = k_max * clip((s - s_T) / (1 - s_T), 0, 1)     # share of the surplus put into structure
dB  = B * k * max(0, N) / EC
dR  = B * (N - k * max(0, N))
```
- Basis: Broekhuizen 1994 [A], Gurney 2003 [F] (k = 0 below the control range, reserve held at a
  target); Metcalfe & Thorpe 1992 (defended energy level) [A] [L].
- Behaviour: as in A, no growth without income. In addition, all surplus goes to fat until s reaches
  s_T, so spring refill comes before growth.
- Optional seasonal target: s_T rising when day length shortens, e.g. s_T = s_T0 + delta x
  max(0, -dD/dt)/max|dD/dt| from the daylight calendar. Basis: Metcalfe et al. 2002 and Sæther 1996
  (salmonids and charr only) [A] [L].
- Two parameters more than A, and the photoperiod version is a calendar in biological clothing; no
  herring data support it.
- Against it for herring: NSAS refill fat (May -> June) at the same time as the May-June length peak
  (Kenyon [2nd], Iles 1964 [Sn]). That is parallel allocation (A), not reserve first (B).

### Rule C: keep reserve-funded growth, add a DEB-Storage brake
```
dB  = B * g * (s - u)        only if I >= C (this tick's income pays maintenance)
dB  = 0                      otherwise; the reserve pays C - I
```
- Basis: Ortiz Durán et al. 2026 [F]. It is the smallest change to the current law and keeps g and its
  derivation.
- Behaviour: growth stops whenever this tick's income cannot pay maintenance. But at intakes just above
  that line the reserve still pays the full g (s - u). With s ~ 0.6-0.7 in October that is
  (0.4-0.5) x 0.0064 = 0.26-0.32 %/d, financed from fat. It removes the Nov-Feb growth (ration ~0) but not all of
  October.

### Rejected: standard DEB mobilisation
This is what the model already does in effect: reserve-density-driven growth that continues after food
stops. The evidence above (section 4) is against it for seasonal fish.

### Expected effect on the herring (inference, from the sibling's budget [2nd] and section 0 numbers)
- Oct-Nov: the implied assimilated intake is 14.4 kJ/kg/d against a resting cost of 26.7 (sibling
  table). So I < C, and Rules A and B give **zero growth from October**; Rule C does the same here; it differs from A only in ticks where I >= C, which the Oct-Nov mean
  does not show. The
  sibling attributes 10.7 + 11.5 kJ/kg/d (growth payment + dilution) in Oct-Nov and ~1,270 MJ/t over
  Oct-Feb to growth. That reserve now stays in the fish.
- With costs and intake as in the run, s in February would be roughly 0.21 + 1,270/7,000 ~ **0.39**
  (~7 % fat) instead of 0.21 [S]. That is inside the 6-11 % NSAS adult Q1 range of the sibling note.
  This is a first-order estimate; behaviour (hunger gate, resting) would also change.
- Autumn biomass is no longer held flat by growth. Expect a decline of ~0.26-0.41 %/d from October
  (M1 0.16 %/d + predation 0.10-0.25 %/d, sibling) instead of from January. The winter loss is spread
  over Oct-Apr instead of Jan-Apr, and fewer fish die of starvation in Feb-Mar.
- **Risk.** All annual production must now come from the months with I > C/kappa. If the summer ration
  is too small, the stock decline gets worse, not better: the reserve can no longer convert stored fat
  into biomass. The decisive probe is the annual balance (production Apr-Sep vs Z ~1/yr), not the
  winter curve. The sibling note reached the same conclusion: the summer surplus is too small.

### All decision makers or per group?
- Inference: one code path for all DMs, with kappa derived per group by `growth_budget`, as g is today.
  The hunger gate already supplies the state dependence, and the income rule is the general fish
  bioenergetics default (Wisconsin, DEBlipid, Gurney).
- Zooplankton: growth from income with a lipid store fits copepod capital/income breeding (Varpe et al.
  2009 [A]). The rule would remove growth on stores in diapause-like winter states. That may affect the
  zooplankton spring timing tuned in section 181 and must be re-probed.
- Gadoids (liver store) and marine mammals: the same logic applies to production from income. No
  group-specific evidence was gathered here.
- A per-group switch (`growth_funding: reserve | income`) would let the change be A/B tested on herring
  first.

### What the model needs
- I and C per cell for this tick, available to the growth stage. Intake is computed in predation and
  the cost in movement. Today the growth branch reads only R, so the per-tick assimilated gain must be
  kept (or the reserve change since the start of the tick stored) until `_apply_growth`. The GPU engine
  needs the same buffer.
- `growth_budget.py`: derive kappa instead of g (Rule A). The s* equation is unchanged (shown above).
  The FG editor would show kappa.
- Rule B's seasonal target only: day length or dD/dt from `lib/world/daylight.py` (already per tick, on
  the daylight calendar). Rules A and C need no calendar.

---

## Recommendation

Ranked by strength of evidence:

1. **Rule A, kappa-split income-funded growth. Favoured; moderate evidence.**
   - The seasonal fish growth models checked fund structure from this step's income (Wisconsin [F],
     Gurney/Jones [F]/[A], DEBlipid [A]) or stop it when income cannot pay maintenance (DEB-Storage
     [F]). Mareld's reserve-funded law is the standard-DEB pattern that DEB-Storage was written to
     replace [F].
   - Herring field data: the growth season is the period of rising weight at length, i.e. a positive
     net energy balance (NSS 1 Apr - 15 Sep, Bachiller [F]). Most adult length growth is done by August at the site [D]. Fat and length build up
     together in May-June (Kenyon [2nd], Iles 1964 [Sn]).
   - It gives autumn fat storage without growth as an emergent property (Martin 2017 [A]), with no
     calendar. It needs one derived parameter (kappa ~0.75 for herring [D]) and keeps s*.
   - Apply it to all DMs through one code path, but A/B test it on herring first, then re-probe
     zooplankton.
2. **Rule C, the DEB-Storage brake. Acceptable as a minimal first step; moderate evidence, partial fix.**
   - It keeps g and its derivation, and stops growth whenever income cannot pay maintenance [F].
   - It leaves part of the October growth on fat.
3. **Rule B, the reserve target, with or without a day-length target. Weakest for herring.**
   - Supported by salmonid and charr lab work [A] [L] and by photoperiod control of herring maturation
     [F] [L], but no herring growth data.
   - It contradicts the parallel fat-and-length gain of herring in May-June, and adds parameters.
   - Keep it as the fallback if Rule A leaves autumn growth in the juveniles that the data do not show.

Not recommended: the calendar switch (no mechanism; it also blocks growth in a good October), and
standard DEB mobilisation (it is the current behaviour).

Caveat on all three: the trained policies were learned under the current law. Changing what the
reserve buys changes incentives, so judge the result with retrained policies.

---

## References

- Arrhenius F, Hansson S 1994. In situ food consumption by young-of-the-year Baltic Sea herring
  Clupea harengus: a test of predictions from a bioenergetics model. MEPS 110:145-149.
  https://doi.org/10.3354/meps110145 [F]
- Bachiller E, Utne KR, Jansen T, Huse G 2018. Bioenergetics modeling of the annual consumption of
  zooplankton by pelagic fish feeding in the Northeast Atlantic. PLoS ONE 13:e0190345.
  https://doi.org/10.1371/journal.pone.0190345 [F]
- Bacon PJ, Gurney WSC, Jones W, McLaren IS, Youngson AF 2005. Seasonal growth patterns of wild juvenile
  fish: partitioning variation among explanatory variables, based on individual growth trajectories of
  Atlantic salmon (Salmo salar) parr. J Anim Ecol 74:1-11 (online 2004).
  https://doi.org/10.1111/j.1365-2656.2004.00875.x [A]
- Bernreuther M, Herrmann JP, Peck MA, Temming A 2013. Growth energetics of juvenile herring, Clupea
  harengus L.: food conversion efficiency and temperature dependency of metabolic rate. J Appl Ichthyol
  29:331-340 (online 2012). https://doi.org/10.1111/jai.12045 [A]
- Boeuf G, Le Bail P-Y 1999. Does light have an influence on fish growth? Aquaculture 177:129-152.
  https://doi.org/10.1016/S0044-8486(99)00074-5 [A, via AGRIS record]
- Broekhuizen N, Gurney WSC, Jones A, Bryant AD 1994. Modelling compensatory growth. Funct Ecol
  8:770 (end page not checked). https://doi.org/10.2307/2390237 [A]
- Bull CD, Metcalfe NB, Mangel M 1996. Seasonal matching of foraging to anticipated energy requirements
  in anorexic juvenile salmon. Proc R Soc B 263:13-18. https://doi.org/10.1098/rspb.1996.0003 [Sn]
- Deslauriers D, Chipps SR, Breck JE, Rice JA, Madenjian CP 2017. Fish Bioenergetics 4.0: an R-based
  modeling application. Fisheries 42:586-596. https://doi.org/10.1080/03632415.2017.1377558 [A]
- Foy RJ, Paul AJ 1999. Winter feeding and changes in somatic energy content of age-0 Pacific herring
  in Prince William Sound, Alaska. TAFS 128:1193-1200.
  https://doi.org/10.1577/1548-8659(1999)128<1193:WFACIS>2.0.CO;2 [A]
- Gurney WSC, Jones W, Veitch AR, Nisbet RM 2003. Resource allocation, hyperphagia, and compensatory
  growth in juveniles. Ecology 84:2777-2787. https://doi.org/10.1890/02-0536 [F; PDF from
  https://strathprints.strath.ac.uk/4580/]
- Hurst TP 2007. Causes and consequences of winter mortality in fishes. J Fish Biol 71:315-345.
  https://doi.org/10.1111/j.1095-8649.2007.01596.x [A]
- Hurst TP, Conover DO 2003. Seasonal and interannual variation in the allometry of energy allocation in
  juvenile striped bass. Ecology 84:3360-3369. https://doi.org/10.1890/02-0562 [A]
- Hurst TP, Schultz ET, Conover DO 2000. Seasonal energy dynamics of young-of-the-year Hudson River
  striped bass. TAFS 129:145-157. https://doi.org/10.1577/1548-8659(2000)129<0145:SEDOYO>2.0.CO;2 [A]
- ICES DATRAS, NS-IBTS Q1/Q3 and BITS Q1/Q4 CA records (getCAdata) 2016-2024, downloaded 2026-10-20.
  https://datras.ices.dk/WebServices/DATRASWebService.asmx [D]
- Iles TD 1964. The duration of maturation stages in herring. J Cons int Explor Mer 29:166-188.
  https://doi.org/10.1093/icesjms/29.2.166 [Sn]
- Iles TD 1984. Allocation of resources to gonad and soma in Atlantic herring Clupea harengus L. In:
  Potts GW, Wootton RJ (eds) Fish Reproduction: Strategies and Tactics. Academic Press, pp. 331-347.
  No DOI. [Sn; existence and pages only]
- Imsland AK, Foss A, Koedijk R, Folkvord A, Stefansson SO, Jonassen TM 2007. Persistent growth effects
  of temperature and photoperiod in Atlantic cod Gadus morhua. J Fish Biol 71:1371-1382.
  https://doi.org/10.1111/j.1095-8649.2007.01600.x [A]
- Jones W, Gurney WSC, Speirs DC, Bacon PJ, Youngson AF 2002. Seasonal patterns of growth, expenditure
  and assimilation in juvenile Atlantic salmon. J Anim Ecol 71:916-924.
  https://doi.org/10.1046/j.1365-2656.2002.00679.x [A]
- Kurita Y, Meier S, Kjesbu OS 2003. Oocyte growth and fecundity regulation by atresia of Atlantic
  herring (Clupea harengus) in relation to body condition throughout the maturation cycle. J Sea Res
  49:203-219. https://doi.org/10.1016/S1385-1101(03)00004-2 [Sn]
- Martin BT, Heintz R, Danner EM, Nisbet RM 2017. Integrating lipid storage into general
  representations of fish energetics. J Anim Ecol 86:812-825. https://doi.org/10.1111/1365-2656.12667
  [A]
- Megrey BA, Rose KA, Klumb RA, Hay DE, Werner FE, Eslinger DL, Smith SL 2007. A bioenergetics-based
  population dynamics model of Pacific herring coupled to a lower trophic level NPZ model. Ecol Model
  202:144-164. https://doi.org/10.1016/j.ecolmodel.2006.08.020 [Sn]
- Metcalfe NB, Bull CD, Mangel M 2002. Seasonal variation in catch-up growth reveals state-dependent
  somatic allocations in salmon. Evol Ecol Res 4:871 (end page not checked). No DOI found. [A, via OpenAlex]
- Metcalfe NB, Thorpe JE 1992. Anorexia and defended energy levels in over-wintering juvenile salmon.
  J Anim Ecol 61:175 (end page not checked). https://doi.org/10.2307/5520 [A]
- Morgan MJ, Rideout RM, Colbourne EB 2010. Impact of environmental temperature on Atlantic cod Gadus
  morhua energy allocation to growth, condition and reproduction. MEPS 404:185-195.
  https://doi.org/10.3354/meps08502 [A]
- Ortiz Durán EA, van der Meer J, van de Wolfshaar KE 2026. DEB-Storage: a Dynamic Energy Budget model
  for organisms in seasonally variable environments. Ecol Model 522:111828.
  https://doi.org/10.1016/j.ecolmodel.2026.111828 [F; PDF from https://edepot.wur.nl/725048]
- Persson L, Leonardsson K, de Roos AM, Gyllenberg M, Christensen B 1998. Ontogenetic scaling of foraging
  rates and the dynamics of a size-structured consumer-resource model. Theor Pop Biol 54:270-293.
  https://doi.org/10.1006/tpbi.1998.1380 [Sn]
- Post JR, Parkinson EA 2001. Energy allocation strategy in young fish: allometry and survival. Ecology
  82:1040-1051. https://doi.org/10.1890/0012-9658(2001)082[1040:EASIYF]2.0.CO;2 [A]
- Rose KA, Megrey BA, Hay D, Werner F, Schweigert J 2008. Climate regime effects on Pacific herring
  growth using coupled nutrient-phytoplankton-zooplankton and bioenergetics models. TAFS 137:278-297.
  https://doi.org/10.1577/T05-152.1 [Sn]
- Rudstam LG 1988. Exploring the dynamics of herring consumption in the Baltic: applications of an
  energetic model of fish growth. Kieler Meeresforsch Sonderh 6:312-322. No DOI. [2nd, via Arrhenius &
  Hansson 1994]
- Sæther B-S, Johnsen HK, Jobling M 1996. Seasonal changes in food consumption and growth of Arctic charr
  exposed to either simulated natural or a 12:12 LD photoperiod at constant water temperature. J Fish
  Biol 48:1113-1122. https://doi.org/10.1006/jfbi.1996.0114 [A]
- dos Santos Schmidt TC, Berg F, Folkvord A, Pires AMA, Komyakova V, et al. 2022. Is it possible to
  photoperiod manipulate spawning time in planktivorous fish? A long-term experiment on Atlantic
  herring. JEMBE 552:151737. https://doi.org/10.1016/j.jembe.2022.151737 [F; PDF from figshare
  23006210]
- Sewall F, Norcross B, Vollenweider J, Heintz R 2019. Growth, energy storage, and feeding patterns
  reveal winter mortality risks for juvenile Pacific herring in Prince William Sound, Alaska, USA. MEPS
  623:195-208. https://doi.org/10.3354/meps13013 [A here; the sibling note read it in full]
- Varpe Ø, Jørgensen C, Tarling GA, Fiksen Ø 2009. The adaptive value of energy storage and capital
  breeding in seasonal environments. Oikos 118:363-370. https://doi.org/10.1111/j.1600-0706.2008.17036.x
  [A]
- Sibling notes listed under "Builds on" (Kenyon et al. 2022; Slotte 1996; Rojbek et al. 2014; Payne et
  al. 2009; Varpe & Fiksen 2010; Bradford 1991) [2nd].

## Failed fetches and not found

Failed fetches:
- Boeuf & Le Bail 1999 full text: ScienceDirect HTTP 403; HAL record hal-02696574 blocked by bot
  protection. Abstract read via the AGRIS record.
- Martin et al. 2017 full text (Wiley pdfdirect and full-text page): HTTP 403. The equations of
  DEBlipid are therefore described from the abstract only; the kappa value it uses was not seen.
- Arrhenius 1996 (ICES JMS 53:792) and Iles 1964 (ICES JMS 29:166) PDFs on academic.oup.com: HTML
  challenge page instead of PDF.
- Sewall et al. 2019 PDF on int-res.com: "File not found".
- Fish Bioenergetics 4.0 on Deep Blue (hdl 2027.42/141352): Cloudflare challenge.
- Schultz & Conover 1997 (Oecologia 109:516, latitudinal differences in energy storage, Menidia): no
  abstract in OpenAlex, Semantic Scholar HTTP 429.
- Abstracts elided by the publisher in Semantic Scholar and absent in OpenAlex: Megrey et al. 2007,
  Persson et al. 1998, Kurita et al. 2003, Bull et al. 1996.
- Schmidt et al. 2022 on NVA (hdl 11250/3016888): page is a JavaScript app; the figshare copy was used.

Not found:
- Iles 1984 chapter text or abstract.
- A month-by-month herring length series (otolith marginal increment, or scale or otolith annulus
  timing) for the North Sea, Skagerrak-Kattegat or western Baltic.
- Aged BITS herring from the Kattegat or western Baltic (SD 21-24) in DATRAS 2016-2024; and any autumn
  (Sep-Dec) survey with aged herring at the site.
- Any herring experiment with ample autumn food that measures structural growth and lipid separately.
- GH, IGF-1 or melatonin seasonality measured in herring.
- Rudstam 1988 itself (Kieler Meeresforschungen Sonderheft; no DOI, no online copy found).
- The kappa value used in DEBlipid (Martin 2017), and an Add-my-Pet DEB parameter set for Atlantic
  herring (not searched beyond OpenAlex).
