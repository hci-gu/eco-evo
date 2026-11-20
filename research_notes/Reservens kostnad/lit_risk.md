# Does a larger energy reserve raise predation risk in fish? (literature trail)

Question: the "mass-dependent predation risk" leg of optimal-reserve theory (Lima 1986; McNamara &
Houston 1990; Houston, McNamara & Hutchinson 1993; Witter & Cuthill 1993; Gosler, Greenwood &
Perrins 1995). In birds the cost is that body mass impairs take-off and manoeuvring. Fish are close
to neutrally buoyant, so that mechanism is **PROXY-mechanism** for fish until shown otherwise.

Tags: [F] full text read, [A] abstract only, [S] secondary (claim read in another source or in a
search-indexed abstract snippet), [D] my arithmetic, [J] judgement. One row per number in
`tables/risk_raw.csv`; derived ratios in `scripts/risk_tucker_ratios.py`.

Mass type on every row: **meal**, **gonad**, **reserve** (lipid / liver / condition), or **none**
(growth rate, injury, size). Only "reserve" rows can size a reserve cost, and a "reserve + structure"
row (long starvation that also strips muscle) cannot size it within the normal fill range.

Fill = reserve / maximum reserve. Scales used: gadoid HSI 2 % -> 8 % ~ fill 0.1 -> 1; herring
whole-body fat 2 % -> 20 % ~ fill 0 -> 1 (given in the task).

Sources looked at: ~30 (cod and two congener studies; the rest PROXY). Full text read: 3 (Sewall 2021, Tucker 2016, Biro 2005).

---

## 1. Swimming and escape performance against condition / lipid

### 1.1 Atlantic cod (target species)

- **Lapointe, Guderley & Dutil 2006**, Physiol Biochem Zool 79:109-119, doi 10.1086/498290 [A]
  (PubMed abstract). 24 wild Gulf of St Lawrence cod measured at K 0.676 +- 0.076 and again after
  9 wk feeding at K 0.923 +- 0.096; "These levels of condition are representative of wild cod in
  the Gulf of St. Lawrence during the spring and during the fall-early winter, respectively."
  Result: "Ucrit increased and cost of transport at 1.3 and 1.5 body lengths s-1 decreased with
  improved condition, but ... sprint performance, and time to exhaustion did not change." Mass-
  specific AMR, aerobic scope and post-exhaustion MR decreased, SMR unchanged.
  - This is the only contrast found in a target species that lies inside the natural reserve range.
    Higher condition gave equal sprint and better Ucrit: no cost, a small benefit.
  - The mass-specific drop in AMR and scope is what extra non-respiring mass (liver oil) per kg would
    give even if the scope per fish did not change [J]. It is not shown to be a performance loss.
  - Full text not reachable (no OA copy, JEB/UChicago walls), so no magnitudes.
- **Martinez et al. 2003**, J Exp Biol 206:503-511, doi 10.1242/jeb.00098 [A]. 16 wk fed vs
  starved: K 0.81 vs 0.54; "starved fish swimming only 30% of the time (and distance) of fed fish";
  the number of burst-coast movements was linked with condition factor.
- **Martinez et al. 2004**, J Exp Biol 207:2979-2990, doi 10.1242/jeb.01142 [A]. 12 wk fed vs
  starved, K 1.0 vs 0.5. Ucrit and sprint "differed by much the same extent between starved and fed
  cod" (fed better); fed cod used fast fibres harder (more lactate per burst-coast).
- **Martinez et al. 2002**, Physiol Biochem Zool 75:90-100, doi 10.1086/339213 [A]. Starve-feed-
  starve cycle: "sprint performance improved after feeding but did not fall with the second
  starvation"; individual sprint hierarchies kept.
- Where the Martinez contrasts sit [J]: cod die in lab starvation at K 0.47-0.54 with HSI (som)
  0.6-1.0 % (Dutil et al. 1995), and K <= 0.65 is the lab starvation zone (Dutil & Lambert 2000,
  doi 10.1139/f00-023) - both [S] via `research_notes/Svältgräns och startreserv/fish.md` 2.1. The
  starved groups (K 0.5-0.54) are therefore **at or below the starvation line**, with muscle
  catabolised. They show that a fish with no reserve and wasted muscle swims worse. They say
  nothing about whether going from half to full reserve costs anything.
- Fill mapping of Lapointe [J]: wild May cod at K 0.69-0.74 have HSI (som) 2.2-3.0 % (Lambert &
  Dutil 1997, doi 10.1139/f96-149 [S], same note), i.e. fill ~0.1-0.2 on the HSI 2 -> 8 % scale.
  Autumn HSI in the same northern Gulf stock is 4.3-5.7 % [S, search-indexed snippet], fill
  ~0.4-0.6. But Lapointe's fish were lab-fed for 9 wk, and lab-fed cod reach HSI ~8.6 % (same
  note, fish.md: start group of a lab fasting study at HSI 8.6 %) - so their end fill is **unknown, somewhere in 0.5-1**. Lapointe spans
  **fill ~0.15 -> 0.5-1 [J]**, and over that span the sign is "more reserve = equal or better
  swimming". No study found shows a contrast lying *within* the upper half of fill (0.5 -> 1).

### 1.2 Herring (congener PROXY)

- **Sewall, Norcross & Heintz 2021**, CJFAS 78:881-893, doi 10.1139/cjfas-2020-0293 [F]
  (`data/risk_pacherring_noaa31517/`). YOY Pacific herring (Clupea pallasii), 20 wk winter rations.
  - Table 5 (p.886), April: lipid wet 8.72 +- 1.80 % (full), 7.73 +- 2.49 (reduced), 6.92 +- 1.21
    (fasted). p.889: "Critical swimming speeds (BL s-1) did not significantly differ among
    treatments in April following the winter ration period (ANOVA; F[2,27] = 0.14, R2 = 1.03%,
    p = 0.870) or in May following spring re-feeding (ANOVA; F[4,43] = 1.11, R2 = 9.43%,
    p = 0.360), although the fasted herring were slowest among treatments within April and May,
    while the re-fed herring were fastest among treatments within May".
  - p.887: "Lipid levels among full-ration herring were more than twice those of fasted herring".
  - p.891: "continuously fed herring had high fat stores and submaximal RNA/DNA and swimming speed,
    suggesting they were not allocating all available energy consumed towards growth". This is the
    only sentence in the trail that can be read as "fatter = slower"; it is a non-significant
    nominal ordering in May, set against fasted fish being slowest in both months.
  - p.881 abstract: "Metabolically processing stored fat rather than foraging and incurring greater
    predation risk may thus be an advantageous strategy" - the risk they name is the risk of
    foraging, not of carrying fat.
  - [D] On the 2 -> 20 % scale the April fills are 0.27 / 0.32 / 0.37 (adult scale applied to YOY,
    [J]). Over that span, no measurable swimming effect of reserve.
- Atlantic herring, North Sea or Norwegian: no study of fat content vs swimming or escape found.
  Search-indexed results only gave buoyancy work (Baltic herring with less fat have larger swim
  bladders; fat is lighter than seawater and adds lift) [S, not in CSV].

### 1.3 Mackerel, whiting, haddock, saithe, Norway pout, hake, sprat

Nothing found on reserve vs swimming, escape or predation. Walleye pollock (cod congener): **Sogard &
Olla 2002**, MEPS 243:165-177, doi 10.3354/meps243165 [S] (via Sewall 2021 p.891 and a search-indexed
abstract; int-res page rate-limited): pollock caught up in growth after 2-3 wk deprivation "without
impacting critical swimming speed"; the response to a simulated predator did not differ between
previously deprived and continuously fed fish of either species.

Mechanism note [J]: lipid (density ~0.93) is lighter than seawater. In a swimbladder-less fish
(mackerel) a fatter body sinks less, so less lift and less swimming is needed to hold depth; in a
physoclist gadoid, liver oil gives part of the lift the swimbladder otherwise must hold. The bird
mechanism has two halves. The **weight-support** half (mass to lift at take-off) does not transfer:
the water carries the fat. The **inertia** half could: buoyancy cancels weight, not mass, so in a
fast-start the extra non-muscle mass must still be accelerated (plus a larger cross-section and drag).
That route is shown in fish only for embryo and gonad mass (1.5, Ghalambor; Plaut) and is
**unmeasured for fat**.

### 1.4 Other fish (PROXY): lean = slower

- **Penghan, Pang & Fu 2016**, Fish Physiol Biochem 42:909-918, doi 10.1007/s10695-015-0184-0 [A].
  Rose bitterling fasted 0-6 wk: Vmax = -0.062X + 1.568 m/s (15 C) and -0.091X + 1.755 (25 C), X in
  weeks; [D] -4.0 and -5.2 % per week. No lipid measure.
- **Walker et al. 2005**, Funct Ecol 19:808-815, doi 10.1111/j.1365-2435.2005.01033.x [A]: "a one
  standard deviation increase in fast-start performance increases the odds of surviving a predation
  strike 2.3-fold" (guppy vs pike cichlid). This is the converter from escape performance to risk.
  It has no reserve data.

### 1.5 Meal and gonad mass: the masses that do cost (kept apart)

| Source | doi | Mass | Effect | Tag |
|---|---|---|---|---|
| Alsop & Wood 1997, trout | 10.1242/jeb.200.17.2337 | meal (satiation ration, SDA) | Ucrit -15 % vs fasted; MO2max unchanged | [A] |
| Thorarensen & Farrell 2006, chinook | 10.1086/505512 | meal (2 % bm) | Ucrit -9 % at 15 h; gone after 3 d | [A] |
| Billerbeck et al. 2001, Menidia | 10.1111/j.0014-3820.2001.tb00835.x | meal | fed fish: lower burst speed and endurance | [A] |
| Lankford et al. 2001, Menidia | 10.1111/j.0014-3820.2001.tb00836.x | meal | "Silversides with large meals in their stomachs suffered significantly higher predation mortality than unfed silversides" | [A] |
| Plaut 2002, mosquitofish | 10.1046/j.1365-2435.2002.00638.x | gonad (embryos) | mass 0.47 -> 0.72 g, Ucrit 25.0 -> 20.1 cm/s, back to 24.7 after birth; [D] -0.37 % Ucrit per % mass | [A] |
| Ghalambor et al. 2004, guppy | 10.1086/421412 | gonad (embryos) | fast-start velocity and distance fall over pregnancy | [A] |

The Menidia fast-growth contrasts (Billerbeck, Lankford: ~1.0 vs 0.5 mm/d; Munch & Conover 2003,
doi 10.1111/j.0014-3820.2003.tb00389.x, mortality higher "despite being 40% larger") are about
growth rate (mass type none). Billerbeck notes northern fish also "store more energy", but storage
is confounded with growth and meal size there and was not tested as a cause [J].

Plaut's own reading is an aerobic constraint, not hydrodynamics ("Swimming kinematics was not
affected by pregnancy"). So even gonad mass costs mainly through physiology.

---

## 2. Condition-selective predation: do predators take lean or fat prey?

- **Tucker, Hipfner & Trudel 2016**, Ecology 97:461-471, doi 10.1890/15-0564.1 [F]
  (`data/risk_tucker2016/`). Rhinoceros auklet bill loads vs coastal trawl availability, juvenile
  sockeye, pink and chum salmon (PROXY). Condition = residual of ln mass on ln length (mass-at-length;
  lipid, muscle and gut together). p.466: "85.5% of predated salmon were smaller than the mean ...
  and 76.2% were below average in terms of mass (i.e., poor condition); 64% were both small and in
  poor condition in contrast to 26.5% for the trawl survey salmon at large." "Within a size category,
  fish in poor condition consistently had the higher electivity index." "Large positive sockeye
  salmon and chum salmon were very strongly avoided in both years (~-1)".
  - [D] Relative risk poor/good from Table 3 (cells with trawl share >= 0.03): 2012 small fish
    2.96 (sockeye 3.8, pink 3.0, chum 1.7); 2013 small fish 1.84; large chum 2013 5.7; in 2012 the
    predated share of large positive-residual fish was 0.00 (rounded) in all three species. So the
    risk of a smolt with a negative mass residual was **~2-4 x** that of one with a positive
    residual. The split is the sign of the residual, not a fill unit; only a [J] guess: lower vs
    upper half of the fill range.
- **Booth & Beretta 2004**, Oecologia 140:289-294, doi 10.1007/s00442-004-1608-1 [A]. Damselfish
  settlers fed 5 d to high or low lipid condition: "All of the low condition fish but only 33% of high
  condition fish in mixed groups were consumed by fish predators", "73% of feeding strikes by
  predators were directed at low condition fish"; field persistence 100 % vs 25 %. [D] RR 3.0,
  strike odds 2.7.
- **Hoey & McCormick 2004**, Oecologia 139:23-29, doi 10.1007/s00442-004-1489-3 [A]. Field
  experiment on settlers: predation selective for fish that "were low in total lipids, and had a high
  standardized weight (Fulton's K)". Lipid sign: lean taken. K sign: heavier-at-length taken. The
  only "heavier is riskier" datum in the trail, and the abstract separates it from lipid; what the
  extra mass was (gut, water) is not stated.
- **Hostetter et al. 2012**, TAFS 141:1586-1599, doi 10.1080/00028487.2012.716011 [A]. 25 909
  steelhead smolts; tern and cormorant predation "increased significantly with decreases in steelhead
  external condition" - injuries, descaling, disease, parasites (mass type none).
- **Mesa et al. 1994**, J Fish Biol 45(Suppl A):81-96, doi 10.1111/j.1095-8649.1994.tb01085.x [A].
  Review: "most of this research indicates that such prey are significantly more vulnerable to
  predation"; mechanisms include "poor fast-start performance, inability to shoal effectively".
- Predators choosing **fat** individuals within a prey species: none found (two targeted searches).
  Seabird and seal work selects among prey *species* by energy density, or takes gravid (gonad-
  laden) capelin (Davoren & Montevecchi 2003, MEPS 258:253 [S], disputed by Carscadden 2005) - that
  is a gonad case and a predator preference, not reserve-driven vulnerability.

---

## 3. State-dependent risk and stored less under risk (fish)

- Direction, lean fish take more risk: **Killen, Marras & McKenzie 2011**, J Anim Ecol 80:1024-1033,
  doi 10.1111/j.1365-2656.2011.01844.x [A] (sea bass; risk-taking rose after 7 d deprivation in
  proportion to individual mass loss); **Damsgard & Dill 1998**, Behav Ecol 9:26-32,
  doi 10.1093/beheco/9.1.26 [S] (coho regaining weight accept more risk to feed).
- Reserve stored late under predation: **Biro, Post & Abrahams 2005**, Proc R Soc B 272:1443-1448,
  doi 10.1098/rspb.2005.3096 [F] (`data/risk_biro2005/`). Whole-lake age-0 trout: lipid on length
  slope 0.118 in July, 0.242 in August; storage/structure slope 0.189 -> 0.648; "a switch from a
  strategy favouring somatic growth to one favouring lipid accumulation at approximately 50 mm FL".
  p.1446: "the size-dependent risk of predation selects for increased risk-taking and energy
  allocation to soma (at the expense of lipids) to maximize somatic growth rates and therefore
  minimize time spent in a highly vulnerable size range." Low-food lakes 25 % survival vs 42 % in
  high-food lakes (Biro et al. 2003a, [S]).
  - Reading [J]: fish here do keep their reserve low under predation, but the cost of lipid is
    **forgone somatic growth under size-dependent risk**, not a predation cost of carrying fat. It
    is not support for the bird mechanism.
- **Post & Parkinson 2001**, Ecology 82:1040-1051 [S]; **Schultz & Conover 1997**, Oecologia
  109:516-529, doi 10.1007/s004420050112 [A]; **Biro et al. 2004**, CJFAS 61:1513-1519,
  doi 10.1139/f04-083 [A] (winter mortality ~60 % in the largest to >90 % in the smallest age-0
  trout; no survivor below the lab minimum lipid) - the benefit side: small, lean fish die in winter.
- Optimal reserve below maximum in fish: **Metcalfe & Thorpe 1992**, J Anim Ecol 61:175-181,
  doi 10.2307/5520 [S]: juvenile salmon turn anorexic in winter with food present, "defended energy
  levels" below maximum; the proposed cost is the predation exposure of foraging at low temperature
  (low escape scope), again a cost of acquiring, not carrying (Bull, Metcalfe & Mangel 1996 [S]).
- No fish experiment found in which fish stored **less** because carrying more raised risk, or more
  when risk was low because of a carrying cost.
- Bird theory: **Gosler, Greenwood & Perrins 1995**, Nature 377:621-623, doi 10.1038/377621a0 [S]:
  great-tit mass rose when sparrowhawks declined and fell as they recovered (PROXY-mechanism).

---

## 4. Conversion to "relative risk per unit fill"

| Contrast | Fill span | Effect | Per unit fill | Tag |
|---|---|---|---|---|
| Cod, Lapointe 2006 | ~0.15 -> unknown 0.5-1 | sprint, time to exhaustion unchanged; Ucrit up | escape cost ~0 (sign: none or benefit) | [A] + [J] fill |
| Cod, Martinez 2003-04 | below 0 -> ~0.5-1 | endurance 0.30x, Ucrit and sprint lower when starved | not a fill-range contrast | [A] + [J] |
| Pacific herring, Sewall 2021 | 0.27 -> 0.37 (April) | Ucrit no difference (p 0.87) | ~0 per 0.1 fill (no power to see small effects) | [F] + [D] |
| Salmon smolts vs auklet, Tucker 2016 | lower vs upper half of condition (~fill [J]) | RR poor/good 1.8-3.8 | lean riskier by ~2-4x (the low end of the range; does not test the mass-dependent leg) | [F] + [D] |
| Damselfish, Booth & Beretta 2004 | low vs high lipid | RR 3.0 | lean riskier | [A] + [D] |
| Mosquitofish, Plaut 2002 | gonad +53 % mass | Ucrit -20 % | -0.37 % per % mass (gonad, not reserve) | [A] + [D] |

[J] **Illustration only - not a bound, not usable as a parameter.** If fat mass acted like embryo
mass (Plaut, -0.37 % Ucrit per % body mass), a cod going from HSI 2 % to 8 % (+6 % of body mass,
mostly liver oil) would lose ~2 % Ucrit. Chaining that (an aerobic measure, mosquitofish) to Walker's
fast-start odds (2.3x per SD, guppies) with an unsourced fast-start SD of ~10-20 % of the mean gives
~+10-20 % odds of being caught in a strike. Each link is a different species, measure or guess;
Walker's SD could not be checked (Wiley 403), and Lapointe saw Ucrit rise over this span.

---

## 5. Verdict

**Measured.** Across cod, a herring congener, salmonids, sea bass and damselfish, the fish data run
one way: **lean fish are at higher risk**. They swim and escape worse once the reserve is gone and
muscle is used (cod endurance 0.30 x; bitterling -4 to -5 % Vmax per week fasted). They take more
risk when food-deprived. Predators pick them preferentially: ~2-4 x relative risk for negative-mass-residual
smolts in the field (Tucker 2016 [F, D]), 3 x in damselfish (Booth & Beretta [A, D]).

**Absent.** No study found in any of the target species, or in any fish, shows that a larger
**lipid or liver reserve within the natural range** lowers escape performance or raises predation.
The only in-range gadoid contrast (Lapointe 2006, fill ~0.15 -> unknown 0.5-1 [J]) gives unchanged
sprint and higher Ucrit. The herring congener (fill 0.27 -> 0.37) gives no Ucrit difference. No
contrast lying within the upper half of the fill range (0.5 -> 1) was found in any target species. The one "heavier at length is
riskier" datum (Hoey & McCormick: high Fulton K selected) comes with the opposite sign for lipid in
the same fish.

**Only meal or gonad.** The swimming and predation costs of carried mass that do exist in fish are
**meal** (Ucrit -9 to -15 %, transient; higher predation of Menidia with full stomachs) and
**gonad/embryo** (Ucrit -20 % for +53 % mass in mosquitofish). They act through the aerobic scope
that SDA or embryos take, more than through drag. Neither is a reserve effect.

**Where fish do keep reserves below the maximum** (age-0 trout storing late, winter-anorexic salmon,
herring burning fat rather than feeding), the cost named is **the risk of foraging to acquire the
reserve** (exposure, risky habitat, forgone somatic growth under size-dependent risk), not the risk
of carrying it.

So a reserve-dependent predation risk in these fish is: **direction** - negative (risk falls as the
reserve rises) at the low end, and **not shown** at the high end; **size** of any carrying cost - not
measured, consistent with ~0 from fill ~0.15 to 0.5-1 in cod (sprint unchanged). The lean-is-riskier
evidence is the other end of the range and does not test the mass-dependent leg. Mechanism [J]: the
weight-support half of the bird mechanism does not transfer (water carries the fat, lipid adds
buoyancy); the inertia half (accelerating extra mass in a fast-start) could, and is unmeasured for fat.

## 6. Gaps

- No reserve vs escape/predation study in whiting, haddock, saithe, Norway pout, hake, sprat,
  mackerel or Atlantic herring.
- Lapointe 2006 and Martinez 2003/2004 full texts were not reachable: no Ucrit/sprint magnitudes,
  no HSI per group (fill mapping of Lapointe is [J] from K via Lambert & Dutil [S]).
- No contrast within fill 0.5 -> 1 in any target species; no within-species "predator picks fatter fish"
  study found.
- Robinson & Pitcher 1989 (herring hunger and schooling, J Fish Biol 34:631) and Skajaa, Ferno &
  Folkvord 2003 (cod larvae hunger vs predation risk) were identified but not retrieved.
- Downloads that failed (HTML error pages) were deleted: Martinez 2003 (scispace, CiteSeerX), Hostetter
  2012 (birdresearchnw). Theory papers (Lima 1986, McNamara & Houston 1990, Houston et al. 1993, Witter
  & Cuthill 1993) were not opened; their dois are not verified.
