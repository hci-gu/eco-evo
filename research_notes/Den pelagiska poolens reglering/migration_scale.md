# Migration, residence and the grid-versus-stock scale problem for the `pelagic_fish` pool

Task Q4 of the pelagic-pool review. Question: how much of the small-pelagic biomass in a 60 x 60 km box in the
Skagerrak (Mareld, about 58.3-58.8 N, 10.5-11.0 E, about 100 m deep) is set by seasonal immigration and emigration
rather than by local production, and what does that mean for closing the pool with a logistic r_max?

Tags:
- **[literature]**: quoted or paraphrased from a source. The source table at the end says how each was read.
- **[computed]**: my arithmetic on cited numbers or on survey data I downloaded.
- **[inference]**: my interpretation, including all model suggestions.

Not redone here (see the sibling notes):
- `Sillens vinterbudget/winter_budget.md` section 4: IBTS herring at the site in Q1 vs Q3, WBSS adults to the Sound,
  NSAS juveniles leaving at age 1-2.
- `Nettoförflyttning per funktionsgrupp/herring.md`: herring swimming and migration speeds.
- `Fiskeridödlighet i Skagerrak idag/ices_pelagic_f.md`: HERAS 2025 Skagerrak strata (98 kt herring, 32 % WBSS) and
  the WBSS share by age and quarter (HAWG 2026 Table 3.2.4).

---

## 1. Takeaway

- **Recruits are imported.** Almost none of the pool's recruits are produced in the box. [literature + inference]
  - Herring larvae drift 3-4 months from North Sea spawning grounds 730-1030 km away [computed] into the Skagerrak and
    Kattegat nursery. WBSS spawn at Rügen, about 490 km away. The local fjord spring spawners are small.
  - Mackerel and horse mackerel spawn elsewhere and visit in summer and autumn. Sandeel is essentially absent at the
    site's depth.
- **Mass is produced locally.** Juvenile herring grow about tenfold in weight while resident, and summer visitors
  feed on local zooplankton. Somatic production is therefore a local process. [inference]
- **Departure is scheduled, not density-driven.** It is set by age (NSAS leave at age 1-2) and by season (adults and
  mackerel leave in Aug-Oct). Local density does not trigger it. [literature]
- **Season: the site in winter holds about one third of its August biomass.** [computed, IBTS GOV 2015-2024, all
  pelagic species]
  - Q1 (Jan-Feb) is 244 kg/h and Q3 (Aug) is 727 kg/h, a ratio of **0.34**.
  - Q1/Q3 by species: mackerel 0.009, horse mackerel 0.012, sprat 0.03, herring 0.45.
  - In August, herring is 73 % of the index, sprat 15 % (two patchy years), mackerel 9 % and horse mackerel 2 %.
    In winter, herring is 98 %, mostly 1- and 2-ringers.
- **Scale: the box is tiny relative to the stocks.** [computed, inference]
  - It is 0.6 % of the North Sea area and 8-12 % of the Skagerrak.
  - At 3.33 t/km² it holds 12 kt, which is 0.4 % of NSAS total biomass.
  - A herring school can cross it in 1-2 days. With feeding-mode dispersal, the box's biomass exchanges with its
    surroundings in about 4-18 days.
- **Consequence for the model.** A closed-box logistic with the life-cycle r_max has no demographic basis at this
  scale. [inference]
  - Local biomass does not produce local recruits.
  - The losses that hold the real local stock down include emigration. Emigration is missing from the model, and in
    the real stocks it is paid for by a recruit inflow that is independent of local biomass.
  - This is the same structure as Ecosim's immigration and emigration terms (Christensen & Walters 2004), and it gives
    a concrete closure (section 6).

---

## 2. Presence calendar by component

Months are approximate. "Fraction of year" is the share of the year that the component's biomass is at or near the
site.

| Component | Months at the site | Residence | Fraction of year | Origin of recruits | Evidence |
|---|---|---|---|---|---|
| NSAS juveniles (0-2 wr) | year-round; arrive as late larvae or 0-group in spring-summer, leave at age 1-2 | about 1-2 yr per cohort | 1.0 (cohorts replace each other) | North Sea spawning grounds 730-1030 km away | [literature] stock annex; Clausen et al. via winter_budget.md; [computed] IBTS age structure |
| NSAS adults (3+ wr) | mainly summer feeding (May-Aug) in the eastern North Sea and outer Skagerrak; spawn Aug-Jan in the North Sea | months | about 0.3-0.5 | North Sea | [literature] stock annex; [computed] IBTS Q1/Q3 at age 3+ = 0.21 (winter_budget.md) |
| WBSS adults | May/Jun-Aug feeding in the Skagerrak and eastern North Sea; through the Kattegat in Jul-Aug; overwinter in the Sound (Aug/Sep-Mar); spawn at Rügen (Mar-May) | about 3-4 months | about 0.25-0.35 | Rügen and the western Baltic | [literature] HAWG 2026 s. 3.13.1; Nielsen 1996/2001 via herring.md |
| WBSS juveniles | 1-ringers occur in the Skagerrak; most are in the Kattegat | about 1 yr | ? | western Baltic | [literature] Clausen et al. via winter_budget.md |
| Local Skagerrak and fjord spring spawners | coastal, partly overwinter coastally | year-round, coastal | small offshore share | local | [literature] Eggers et al. 2015; Berg et al. 2017 (via winter_budget.md); WGIPS 2026 genetics |
| Sprat (Skagerrak-Kattegat) | patchy offshore in summer; inshore and in fjords in winter; HERAS 2025 found all 3.a sprat in the Kattegat in July | ? | offshore about 0.2-0.4 | partly local, partly North Sea; unresolved | [literature] HAWG 2026 s. 11.11.1; ICES FishMap; [computed] IBTS Q1/Q3 = 0.03 |
| Mackerel | Jun-Oct; leaves the northern North Sea by Oct-Nov; overwinters at the shelf edge and in the Norwegian Trench; some overwinter in the Skagerrak, Kattegat and Sound | about 4-5 months | about 0.35-0.4 | NEA spawning grounds, North Sea component in the central North Sea | [literature] Jansen & Gislason 2011; Jansen et al. 2012; ICES FishMap; [computed] IBTS Q1/Q3 = 0.009 |
| Horse mackerel | summer-autumn; "In autumn ... leave the North Sea to return in the spring" | about 3-5 months | about 0.3 | North Sea stock (southern North Sea) and western stock | [literature] ICES FishMap; [computed] IBTS Q1/Q3 = 0.012 |
| Sandeel | resident on sandbanks; feeds in the water column Apr-Jul and is buried from late summer to April | on banks only | at the site about 0 | bank-local | [literature] Henriksen et al. 2021; [computed] IBTS occurrence 1-4 % of hauls at the site, about 0 kg/h |

Notes on the calendar:
- **WBSS share by quarter.** In Q1 the WBSS share of 3-6-ringers in SD 20 is 0.35-0.63, against 0.32-0.46 in Q3 (HAWG
  2026 Table 3.2.4, quoted in `ices_pelagic_f.md`).
  - These are proportions of the herring present, not abundances. Adult abundance at the site in Q1 is about 1/5 of
    August (winter_budget.md).
  - So the adults that stay in winter are a mix of both stocks.
- **NSAS juveniles in the Skagerrak and Kattegat track year-class strength.** "The proportions of juveniles of North
  Sea origin found in these areas varies with the strength of the year class, with higher proportions in the
  Skagerrak and Kattegat when the year class is good" (NSAS stock annex). This is a density-dependent spill-over at
  the stock scale. Local density does not cause it.
- **Where the 1-ringers are.** In 2026, "the majority of the 1-ringers of the 2024 year-class were found in the German
  Bight, along the Danish west coast and in the Kattegat/Skagerrak area" (HAWG 2026 s. 2.3.3.2).

---

## 3. Quantified exchange

### 3.1 IBTS index of all pelagics at the site, Q1 vs Q3 [computed]

**Data.**
- Source: ICES DATRAS NS-IBTS `getCPUELength`, 2015-2024.
- Box: 57.6-58.6 N, 9.8-11.5 E, depth 40-200 m, the same as winter_budget.md.
- Hauls: Q1 148 (116 in Jan, 32 in Feb); Q3 94 (90 in Aug, 4 in Sep).

**Conversion to biomass.**
- Biomass = CPUE at length x W(L) with generic length-weight laws (cm, g). The laws are my assumptions, not fitted to
  the site:
  - herring 0.0066 L^3.09; sprat 0.0055 L^3.10; mackerel 0.0053 L^3.17;
  - horse mackerel 0.0090 L^2.98; sandeel 0.0018 L^3.25.

| kg/h (mean over hauls) | Q1 | Q3 | Q1/Q3 | Q3 median | Q3 occurrence |
|---|---|---|---|---|---|
| herring | 239.8 | 530.8 | 0.45 | 53 | 0.95 |
| sprat | 3.2 | 111.2 | 0.03 | 0 | 0.31 |
| mackerel | 0.6 | 68.2 | 0.009 | 9.6 | 0.84 |
| horse mackerel | 0.2 | 16.3 | 0.012 | 1.2 | 0.71 |
| sandeel (all Ammodytidae) | 0.0 | 0.0 | - | 0 | 0.01 |
| anchovy and sardine | 0.1 | 0.1 | - | 0 | ≤0.11 |
| **total** | **243.9** | **726.6** | **0.34** | | |

**Year-to-year variation.** By year the Q1 herring index ranges 16-652 kg/h and the Q3 index 147-1935 kg/h. The
sprat Q3 mean comes mostly from 2015 (476) and 2023 (265).

**Caveats.**
- The GOV is a demersal trawl fished by day. It under-samples pelagic schools, mackerel in particular. These are
  relative indices within one gear, not abundances.
- The herring Q1/Q3 here (0.45) includes the age-0 fish in Q3. winter_budget.md's 0.33 excludes them.

### 3.2 Herring turnover from the age structure at the site [computed]

From the mean IBTS numbers per hour in winter_budget.md:
- Q1: age 1 = 3743, age 2 = 1227, age 3+ = 152.
- Q3: age 0 = 15 516.

Read as a cross-section under steady recruitment, the total loss rate (Z + emigration E) is:
- **Age 1 to 2:** ln(3743/1227) = **1.1 /yr**.
- **Age 2 to 3+:** ≥ ln(1227/152) = **2.1 /yr**. The 3+ group is a plus group, so the per-year rate is higher still.
- **Age 0 (Aug) to age 1 (Feb):** ln(15516/3743)/0.45 yr = 3.2 /yr. This step also includes a change in catchability
  with size.

Splitting off emigration:
- Fishing in 3.a is about 0 today (u ≈ 0.005, ices_pelagic_f.md). Juvenile herring M is about 0.4-1.1 (Ecopath NS
  1991 juveniles 1.07; adults 0.36).
- So the emigration rate is roughly **E ≈ 0.3-0.7 /yr from age 1 to 2** and **≈ 1-1.7 /yr from age 2 onwards**.
- The mean residence of a juvenile cohort after its first winter is therefore about 0.6-1.5 yr. Including the first
  months (arrival in spring-summer), total residence is about 1-2 yr.

This is order-of-magnitude only, because year-class strength varies about 10x.

### 3.3 Share of NSAS larvae in Division 3.a in February [computed from literature]

- Source: the NSAS stock annex, Table B.3.4.1 (IBTS-0, year classes 1976-2010). It gives density by area in February,
  with area weights of 83/34/86/102/37/93/31/31 x 10^9 m²; Div. 3.a is 31.
- Share of the total larval abundance in Div. 3.a:
  - mean **3.8 %**, median 3.0 %, range 0-16 % (1982);
  - Div. 3.a is 6.2 % of the survey area.
- In February most larvae are still in transit, so this is a lower bound on the share of the year class that ends up
  in the Skagerrak-Kattegat nursery. By Q1 of the following year, the "Kattegat/Skagerrak area" is one of the three
  main 1-ringer concentrations (HAWG 2026).
- Gap: I found no published fraction of each NSAS year class that grows up in 3.a. The stock annex says it rises with
  year-class strength.

### 3.4 Distances and drift [computed]

Great-circle distances from the site (58.55 N, 10.75 E):

| From | Distance (km) |
|---|---|
| Buchan spawning ground | 730 |
| Orkney/Shetland | 760 |
| Banks (Flamborough) | 820 |
| Downs | 1030 |
| Rügen | 490 |
| The Sound | 320 |

- The planktonic phase lasts "around three to four months" (stock annex), which implies a mean larval drift of about
  6-8 km/d to reach the Skagerrak.

### 3.5 Exchange time of the 60 x 60 km box [computed / inference]

- **Directed migration.** At 0.5 m/s (about 43 km/d) a school crosses the box in about 1.4 days (herring.md).
- **Feeding-mode dispersal, treated as diffusion.** I use D = (daily net displacement)²/4:
  - The Misund population floor of 9-13 km/d gives D ≈ 20-40 km²/d. Slower feeding gives 10 km²/d.
  - The slowest-decaying mode of a 60 km square has τ = L²/(2π²D). That is 18, 7 and 4 days for D = 10, 25 and
    50 km²/d.
- So, physically, the box's herring biomass relaxes toward the surrounding density on a time scale of **days to a few
  weeks**. That is two orders of magnitude faster than population growth (r about 0.3-0.6 /yr).

---

## 4. The scale argument [computed / inference]

| Area | km² | Box share | Source |
|---|---|---|---|
| North Sea | 570 000 | 0.6 % | Wikipedia (intro and infobox; the Geography section gives 750 000) |
| Skagerrak | 30 000-47 000 (240 km x 80-140 km; infobox 47 000) | 8-12 % | Wikipedia |
| Mareld box | 3 600 | - | model |

- **Stock biomass and densities.**
  - NSAS total biomass is 2.96 Mt (2025), about 5.2 t/km² over the North Sea.
  - HERAS July 2025 Skagerrak herring (98 kt) is 2.1-3.3 t/km² over the Skagerrak.
  - The model's 3.33 t/km² (12 kt) is 0.4 % of NSAS total biomass.
- **Where r_max lives.** r_max (FishBase CMSY prior, a Schaefer r) is a whole-life-cycle property: spawning, larval
  survival and nursery density dependence.
  - For the box's stock, all of these happen 300-1000 km away (herring, mackerel, horse mackerel) or at bank scale
    (sandeel).
  - The density dependence that the logistic K stands for (recruitment compensation; `Täthetsberoendets styrka per
    grupp/pelagic_fish.md` Q1, Q6) acts at stock scale.
- **What the box's local biomass depends on:**
  1. the recruit inflow, which tracks stock year-class strength, not local B;
  2. local somatic growth on local zooplankton;
  3. local mortality;
  4. scheduled emigration (age 2 juveniles, adults in Aug-Oct).
- **Implication.**
  - A local logistic r_max(1 - B/K) misattributes (1) to local B. When local food is good, local B compounds at
    r_max - F, because the model has (2) and (3) but neither the departure (4) nor a bound on (1).
  - That is what the 3-year run shows: the pool roughly doubles, with no starvation and predation near budget.
  - A logistic K keyed on regional B (the gadoid precedent) is a phenomenological substitute. It holds B near today's
    level but attributes the regulation to the wrong process.
  - An emigration term is the mechanism the literature points to for small pelagics in a sub-stock box. Its pairing
    with an inflow independent of B is what makes local B regulated.
- **Production and residence.** The pool's gross P/B of about 1.5 /yr (library budget r_max + M1 + M2 = 1.54) is a
  fair description of what residents produce: juvenile herring grow about tenfold in weight in 1-2 yr, ln 10 / 1.5 ≈
  1.5 /yr. What the budget lacks is that the biomass then leaves.

---

## 5. How other regional and ecosystem models treat migratory pelagics

- **Ecopath and Ecosim (Christensen & Walters 2004).** [literature]
  - The Ecopath master equation carries a net migration term: P = Y + M2·B + E + BA + M0·B, where E = net migration
    (emigration minus immigration).
  - Ecosim integrates dB_i/dt = g_i Σ Q_ji - Σ Q_ij + I_i - (M0_i + F_i + e_i) B_i. Here e_i is the emigration rate,
    and I_i is the immigration rate, "assumed constant over time, and hence independent of events in the ecosystem
    modeled".
  - Ecospace adds cell-to-cell dispersal driven by habitat preference and predation risk.
  - This is exactly the "I - eB" form proposed below. Its steady state is B = I/(e - r_local).
- **Atlantis (Audzijonyte et al., Atlantis User Guide Part I, 2024-2025 edition, s. 11.4).** [literature] A cohort or
  biomass pool can "leave the model domain for a set period of time", with the following parameters:
  - StartTofY and EndTofY (calendar days), and Leave_Period and Return_Period (days over which they leave and return);
  - MigPropSizeInc (growth while away; "Growth outside the model domain adds nitrogen into the system");
  - MigPropSurvive (the share that returns);
  - optionally an external population model (`isExternal`).
  - The documented use is for "large mammals such as whales ... but it can be applied to any group".
  - Most models use it: "most models include some form of migration into/out of the model".
- **OSMOSE (documentation, "Migration mortality").** [literature, search summary only]
  - Schools that are outside the simulated area according to their seasonal or age distribution maps are subject to
    `mortality.out.rate.sp#`, "the annual mortality rate for species that leave the simulated area".
  - While out of the domain, predation, growth, fishing, natural mortality and starvation do not apply.
- **Small-domain offshore-wind food-web models (Bay of Seine, Courseulles-sur-Mer).**
  - Raoux et al. 2017 (Ecopath, 37 compartments) and Halouani et al. 2020 (Ecospace, a fishing closure in the wind
    farm, catches up to +7 % "near the wind farm" but "limited in space") were found. [literature, abstracts]
  - I could not retrieve how they treat migratory small pelagics: the full texts were blocked. Not verified.
- **Whole-basin end-to-end models.** ECOSMO E2E-type models (Daewel et al.; a 2019 GMD figure shows North Sea and
  Baltic fish biomass) put the domain around the whole stock rather than adding migration terms. A 2026 v2.0 preprint
  runs a 1D central North Sea column without migration terms. [literature, snippet/abstract]

Pattern [inference]:
- Models whose domain is smaller than the stock add an explicit exchange: Ecosim I and e, Atlantis scheduled
  migrations, OSMOSE out-of-domain periods.
- Models whose domain contains the stock close it through recruitment.
- None of the documented approaches closes a sub-stock domain with a local logistic on the life-cycle r_max.

---

## 6. Proposed immigration/emigration term [inference, with computed calibration]

Two variants. Which one fits depends on what the model is for (section 7).

### Variant D1: demographic exchange, Ecosim form (recommended if local impacts should be able to change local biomass)

Per tick, on the grid-total (regional) pool:

```
dB_region/dt |_mig = I(t) - e * B_region
```

**Emigration.**
- e·B is removed proportionally from every cell, biomass and reserve together (emigrants take their fat with them),
  as M1 is.
- It is kept OUT of the derived growth rate, like the gadoid K term.

**Immigration.**
- I(t) is added at the start biomass distribution (or the grid boundary) with reserve at the spring reference fill.
- Recruits and visitors arrive lean in spring and summer [inference from winter_budget.md].

**e (emigration rate).**
- Juvenile herring residence of 1-2 yr gives e ≈ 0.5-1.0 /yr. The cross-sectional estimate in 3.2 gives
  0.3-1.7 /yr.
- Summer visitors leave entirely each autumn, which a constant e cannot represent (see the seasonal version below).
- Central e = **0.8 /yr** (5.5e-4 per 6-h tick); range 0.5-1.5.

**I (immigration), from stationarity at B_ref = 3.33 t/km².**
- I_mean = B_ref (e - r_loc), where r_loc is the pool's local net rate without migration.
- Two anchors, as in the gadoid note:
  - **Ideal-food anchor**, r_loc = r_max - F = 0.648 - 0.10 = 0.548: I = 3.33 x (0.8 - 0.548) = **0.84 t/km²/yr**
    (5.7e-4 t/km² per tick).
  - **Realised anchor** (the 3-year run, year 2): r_loc ≈ ln(5.54/4.07) = 0.31. Then I = 3.33 x (0.8 - 0.31) =
    **1.63 t/km²/yr**.
- Re-derive I after any change to r_max, F or M1, and tune it against a retrained run, as for the gadoid K.
- e must exceed r_loc, or there is no equilibrium. With e = 0.5 and r_loc = 0.548 the pool still grows.
  - So the low end of e works only together with a lower r_max (fish.md and the productivity note: about 0.44
    without sandeel).

**Sensitivity to local impacts.**
- B* = I/(e - r_loc), so d ln B*/d r_loc = 1/(e - r_loc).
- A wind-farm-scale change of r_loc by -0.1 /yr changes B* by -0.1/0.49 = **-20 %** (realised anchor, e 0.8). With
  e = 1.5 it changes B* by -8 %.
- Local processes keep real leverage.

**Seasonal version (better).**
- I(t) is concentrated in Apr-Jul: 0-group recruits and arriving WBSS adults, mackerel and horse mackerel.
- e(t) is raised in Aug-Oct, when adults, mackerel and horse mackerel leave.
- A simple form: e(t) = e_0 + e_s·1[Aug-Oct], with e_0 ≈ 0.5 /yr (juvenile emigration) and e_s chosen so that the Q1
  biomass is about 0.34-0.5 of the August biomass.
  - Leaving about 40-60 % of August biomass over 3 months needs e_s ≈ ln(1/0.5)/0.25 ≈ 2-4 /yr during those months.
  - The matching extra inflow in Apr-Jul is set so that the annual mean is stationary.

### Variant D2: physical relaxation to an exogenous regional density (recommended only if the box is a window on a fixed regional stock)

```
dB_region/dt |_mig = (B_ext(t) - B_region) / tau
```

**Seasonal shape of B_ext.**
- B_ext(t) = B_mean · s(d), with s(d) = 1 + A cos(2π (d - d_peak)/365).
- A = (1 - 0.34)/(1 + 0.34) = **0.49**. This reproduces the IBTS Q1/Q3 = 0.34 with an annual mean of 1.
- d_peak ≈ day 213 (1 Aug), from IBTS Q3 in August and HERAS in July. The trough is about 1 Feb.
- If 3.33 t/km² is read as an annual mean, B_ext runs from about 1.7 t/km² (Feb) to about 5.0 t/km² (Aug).
- A component-built alternative:
  - residents (juvenile herring, local spring spawners, Skagerrak sprat) flat at about the Q1 level;
  - visitors (adult herring, mackerel, horse mackerel, part of the sprat) as a pulse from Jun to Sep/Oct, at about
    40-66 % of August biomass ([computed] (727 - 244)/727 = 0.66; lower if the age-0 recruits present in Q3 are counted
    as residents).

**tau.**
- **Physical exchange of the box: 4-18 days** (3.5).
- Using this tau makes B_region track B_ext almost exactly. The equilibrium is B* = B_ext/(1 - tau·r_loc), which is
  1.02 B_ext for tau = 0.05 yr and r_loc = 0.31.
- A "demographic" tau of 1-2 yr (residence) is D1 under a different name. In that case B_ext is not the regional
  density but I·tau, and it must be calibrated, not observed: B_ext = B_ref (1 - tau r_loc).

**Per tick.**
- (B_ext - B)/(tau·1460) per 6-h tick, with tau in years.
- For tau = 14 d (0.038 yr) that is 1.8 % of the gap per tick.

### Common implementation notes [inference]

- **Act on the regional total, distribute by cell.** Use the regional (grid-total) biomass, as for the gadoid K. A
  per-cell relaxation would erase the learned spatial distribution and give the policy a gradient toward spreading.
  - Distribute emigration in proportion to cell biomass.
  - Distribute immigration either in proportion to current biomass (keeps the learned pattern) or along the grid edge
    (lets the policy learn to move inward). The edge option is more mechanistic but changes the observation statistics.
- **Composition is fixed in the pool.** A single pool cannot carry the seasonal composition change (mackerel 9 % in
  August, 0 % in winter). Diet and predator-share targets must be compared on a seasonal, biomass-weighted basis.
- **Fishing.** The model's F 0.10 is the stock-level rate (`ices_pelagic_f.md`). With an explicit emigration term,
  part of that F is incurred after the fish have left. Keeping F 0.10 inside the box together with e double counts
  part of it. The in-area rate is about 0.005.
  - [inference] With D1, keep F at the stock rate only if the box is meant to stand for the stock's average fish.
    Otherwise use about 0.01-0.05 and let e carry the departure.
- **Consistency with the derived growth.** The library budget (growth_budget.py) pays r_max + M1 + M2. As with the
  gadoid K, the migration terms must stay outside it, so they act as net regulation rather than being paid back by
  higher g.

---

## 7. The trade-off: a purpose decision for the user [inference]

- **D2 with a short tau (weeks) pins the pool.**
  - Total pelagic biomass in the box is then set by B_ext(t), i.e. by the regional stock.
  - A wind-farm impact (fishing closure, reef effect, noise, local zooplankton change) can only redistribute biomass
    inside the box and shift it by about tau·Δr (a few per cent).
  - That is realistic for a 60 km window on a 0.5-3 Mt migratory stock: no local measure in 3600 km² changes NSAS or
    mackerel stock size. It does mean the model cannot show local production effects on pelagic biomass, only on
    distribution, condition and diet.
- **D1 (e about 0.8 /yr) gives local processes leverage.**
  - B* moves by about 1/(e - r_loc) per unit change in the local net rate, i.e. about 20 % per 0.1 /yr.
  - That keeps the closed-box behaviour the training reward relies on, while bounding growth.
  - It is still a statement that the recruit supply is external and independent of local B.
- **A regional logistic K on local B (the gadoid precedent) is closer to D1 than to D2** in that local impacts move
  B*. It attributes the cap to local crowding, however, which the literature does not support for migratory small
  pelagics at this scale (`Täthetsberoendets styrka per grupp/pelagic_fish.md` Q6).
- Choosing between "the box as a window on a fixed regional stock" (D2) and "the box as a semi-open population whose
  local production matters" (D1, or K) decides what kind of wind-farm effect the model can show at all. The
  literature cannot make that choice.

---

## 8. Gaps

- No published fraction of each NSAS year class that grows up in 3.a, and no residence time measured by tagging. The
  1.0-2.1 /yr cross-sectional loss rates in 3.2 are confounded by year-class variation and gear catchability.
- No Skagerrak-specific mackerel or horse-mackerel biomass, and no arrival and departure dates. Jansen & Gislason 2011
  report sequential arrival and that large fish arrive earlier and leave later than small, but the full text with the
  month table was not retrieved.
- Sprat stock structure in the Skagerrak (local fjord populations against the North Sea stock) is unresolved per HAWG
  2026. The offshore sprat calendar here rests on two patchy IBTS years.
- How Raoux et al. 2017, Pezy et al. 2018 and Halouani et al. 2020 handle migratory pelagics in the small Bay of Seine
  domain was not retrieved.
- Mareld site depth and sediment were not re-checked against sandeel habitat. The IBTS shows essentially no sandeel at
  the site. The GOV is a poor sandeel sampler, but sandeel banks in 3a are on the Danish side and in shallower water.

---

## Source status

| Source | Used for | How read |
|---|---|---|
| ICES Stock Annex her.27.3a47d (NSAS), [figshare 18622589](https://ices-library.figshare.com/articles/report/Stock_Annex_Herring_Clupea_harengus_in_Subarea_4_and_divisions_3_a_and_7_d_autumn_spawners_North_Sea_Skagerrak_and_Kattegat_eastern_English_Channel_/18622589) | spawning components and nursery areas; larval phase 3-4 months; year-class strength and 3.a share; IBTS-0 table by area (computed 3.a share) | full text (PDF to text) |
| ICES 2026, HAWG report, ICES Sci. Rep. 8:16, [figshare 31424315](https://ices-library.figshare.com/articles/report/Report_of_the_Herring_Assessment_Working_Group_for_the_Area_South_of_62_N_HAWG_/31424315) | WBSS migration (s. 3.13.1); 1-ringer distribution 2026 (s. 2.3.3.2); sprat stock unit and local Skagerrak populations (s. 11.11.1); sprat migration uncertainty | full text (text search; file downloaded by a sibling fork) |
| ICES DATRAS NS-IBTS `getCPUELength` 2015-2024, [web service](https://datras.ices.dk/WebServices/DATRASWebService.asmx) | site Q1/Q3 index by species (section 3.1) | data downloaded and processed |
| `Sillens vinterbudget/winter_budget.md` s. 4 (repo) | IBTS herring age structure at the site; WBSS and NSAS movements; Eggers 2015; Berg 2017; Clausen 2014 | read locally (not redone) |
| `Nettoförflyttning per funktionsgrupp/herring.md` (repo) | herring speeds; Nielsen 1996 (the Sound); WBSS timing | read locally |
| `Fiskeridödlighet i Skagerrak idag/ices_pelagic_f.md` (repo) | HERAS 2025 Skagerrak strata; WBSS share by age and quarter; in-area F | read locally |
| Christensen & Walters 2004, Ecol. Model. 172:109-139, [doi:10.1016/j.ecolmodel.2003.09.003](https://doi.org/10.1016/j.ecolmodel.2003.09.003) ([VLIZ PDF](https://vliz.be/imisdocs/publications/ocrd/55191.pdf)) | Ecopath net migration term; Ecosim I_i (constant) and e_i | full text (PDF to text; the Ecosim equation is garbled in the scan and was written out from the term definitions) |
| Audzijonyte et al., Atlantis User Guide Part I (CSIRO, 2025 upload), [PDF](https://research.csiro.au/atlantis/wp-content/uploads/sites/52/2025/05/AtlantisUserGuide_PartI.pdf) | s. 11.4 migrations into and out of the domain; parameters | full text (PDF to text) |
| OSMOSE ODD documentation, ["Migration mortality"](https://documentation.osmose-model.org/odd_des/submodel/mort/migrat.html) | out-of-domain mortality `mortality.out.rate.sp#`; processes suspended outside | search summary only (page fetch failed) |
| Jansen & Gislason 2011, Cont. Shelf Res. 31:64-72, [doi:10.1016/j.csr.2010.11.003](https://doi.org/10.1016/j.csr.2010.11.003) | mackerel sequential arrival; large fish earlier and later; some overwinter in Skagerrak, Kattegat and the Sound | abstract (repository page) |
| Jansen et al. 2012, PLoS ONE 7:e51541, [doi:10.1371/journal.pone.0051541](https://doi.org/10.1371/journal.pone.0051541) | mackerel return in late summer; left the northern North Sea by October (2009); few caught in the Kattegat/Skagerrak in Q4-Q1 | full text via page-summarising fetch |
| ICES FishMap species factsheets (mackerel, horse mackerel, sprat), [ICES](https://www.ices.dk/about-ICES/projects/EU-RFP/EU%20Repository/ICES%20FIshMap/) | mackerel overwintering and summer densities; horse mackerel leaves the North Sea in autumn; sprat inshore in winter | full text (PDF to text) |
| Henriksen et al. 2021, Ecol. Evol. 11:16786, [doi:10.1002/ece3.8310](https://doi.org/10.1002/ece3.8310) | sandeel emergence (CPUE peak day 111-133), buried late summer to winter, site fidelity | full text via page-summarising fetch (first 100k characters) |
| Halouani et al. 2020, J. Mar. Syst. 212:103434, [doi:10.1016/j.jmarsys.2020.103434](https://doi.org/10.1016/j.jmarsys.2020.103434); Raoux et al. 2017, Ecol. Indic. 72:33, [doi:10.1016/j.ecolind.2016.07.037](https://doi.org/10.1016/j.ecolind.2016.07.037) | small-domain wind-farm EwE/Ecospace; spillover limited in space | abstract or search summary; treatment of migrants not retrieved |
| Vijayakumaran et al. 2026, ECOSMO E2E v2.0, [EGUsphere preprint](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-3078/) | 1D central North Sea fish module, no migration terms mentioned | abstract page |
| Corten & Kuiter 1981, ICES CM 1981/H:17, [VLIZ PDF](https://www.vliz.be/imisdocs/publications/ocrd/274271.pdf) | IYFS 1981: Skagerrak 1-ringers mostly autumn spawners, Kattegat mostly spring spawners | full text (scan, partial) |
| Wikipedia, [Skagerrak](https://en.wikipedia.org/wiki/Skagerrak) and [North Sea](https://en.wikipedia.org/wiki/North_Sea) | areas for the scale argument | page summary |
| Corten 2002 (conservatism in herring migrations); Clausen et al. 2007; Bekkevold et al. 2023; Nøttestad et al. 2016 (mackerel IESSNS) | - | not reached (only cited via the stock annex or HAWG, or not found) |
