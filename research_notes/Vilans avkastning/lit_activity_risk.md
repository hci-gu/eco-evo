# Activity-dependent predation risk in fish: literature trail (Q1)

Question: does field/lab evidence support a predation risk that depends on
whether a fish is foraging (active) or resting, at equal light, depth and
school state, and by which physical mechanism? Tags: [F] full text read,
[A] abstract only, [S] secondary, [D] derived here, [J] judgement.
PROXY = not one of the model's species, or not marine.

Each number is classed by what it measures:
- MOT = detection by prey motion (visual or lateral line)
- SPD = encounter rate through prey swimming speed
- POS = habitat/position shift (depth, shelter, school) - mostly already in the model
- VIG = vigilance/escape (overlaps the model's q = 0.8)
- BUN = bundled (several of the above, cannot be separated)

Rows of numbers: `tables/activity_risk_raw.csv`.

## Sources read

### Biro, Post & Parkinson 2003, Ecology 84:2419-2431, doi 10.1890/02-0416 [F] (class BUN)
Age-0 rainbow trout (PROXY, freshwater) in nine small lakes, cannibalistic
adult trout the only predator; food raised by fertilisation.
- Abstract (p. 2419): "In high-food lakes, age-0 trout spent less time moving,
  fewer individuals swam continuously, and those swimming continuously swam at
  slower speeds ... Age-0 trout also used deep, risky habitats less when food
  was abundant. This lower activity, combined with avoidance of risky
  habitats, coincided with 68% higher survival in high-food lakes."
- p. 2426: "the predicted mean mortality of age-0 trout in high-food lakes was
  only 58% in comparison to 75% mortality in low-food lakes".
- p. 2422: continuously active fish "swam at speeds up to 40 cm/s".
- Fig. 3 (proportion of time moving, proportion continuously active, speed)
  has no values in the text; not digitised.
- [D] Seasonal instantaneous mortality ratio ln(0.25)/ln(0.42) = 1.60 (low
  food / high food). This bundles activity with deep-habitat use; the
  authors also note the low-food lakes were clearer (p. 2428), so it is an
  upper bound for activity alone, and not a rest-vs-forage ratio.

### Biro, Abrahams, Post & Parkinson 2004, Proc R Soc B 271:2233, doi 10.1098/rspb.2004.2861 [F] (class POS)
Domestic (fast-growth) vs wild rainbow trout (PROXY) in whole lakes with or
without common loons (bird predator, PROXY).
- Behaviour measured is daytime use of the food-rich pelagic habitat
  (floating gillnet catches), not activity at a fixed position.
- "predator presence reduced trout survival by 62% (i.e. less than half as
  many survivors" - this is predator presence vs absence for all trout, NOT
  the strain contrast. The strain contrast (Fig. 2a: domestic better than wild
  without loons, worse at the highest risk) has no value in the text.
- Use: position/habitat risk-taking, not a rest payoff at equal position.

### McCormick & Meekan 2010, MEPS 407:173-185, doi 10.3354/meps08583 [F] (class POS; null for activity)
Newly settled damselfish Pomacentrus wardi (PROXY, coral reef) on patch
reefs; logistic model on 186 fish (81 survived, 105 died by Census 3, ~24 h).
Separately the paper states (p. 177): "Loss over the first 24 h for the
experimental fish was 59%".
- Predictors in the logistic model (p. 176): bite rate, total distance moved,
  mean and max distance ventured, boldness, temperature. Best model kept only
  boldness (Wald 9.49, p = 0.002) and MaxDV (Wald 3.98, p = 0.046).
- Table 1 (ANCOVA means): survived boldness 1.66, MaxDV 1.89 cm; died 1.38,
  2.39 cm. Boldness HELPED survival (opposite sign to the activity-risk
  hypothesis); distance from shelter hurt (a shelter case, not open mud).
- Activity per se (bite rate, distance moved) did not enter the best model:
  a null for activity at this scale.

### Martel & Dill 1995, Ethology 99:139-149, doi 10.1111/j.1439-0310.1995.tb01095.x [F] (class MOT)
Juvenile coho parr (PROXY, freshwater; 9.5 cm TL) in a 0.5 x 0.5 m enclosure
inside a channel holding three common mergansers (bird, PROXY predator).
- Methods (p. 141): "We used a white background to eliminate crypsis: the
  goal of the experiment was to isolate the effects of prey movement on
  predator detection time."
- Results (p. 142, Table 1): "Fish moving when birds dipped their heads were
  attacked on 15.2 % of these occasions, whereas stationary fish were attacked
  on only 1.4 % of these occasions" (22/145 vs 11/768; G = 7.62, p < 0.01).
  These are head-dips pooled over 33 fish (one trial each) and three birds.
  The reported G looks small for these pooled counts; the unit the test used
  is not stated - treat the ratio as indicative.
- p. 143-144: head-dip DURATION did not differ when fish were moving or
  stationary (Table 2), and bird dip rate was not correlated with fish
  movement rate (per bird r_s = 0.1, -0.06, 0.71).
- Lag time to detection fell inversely-exponentially with fish movement rate.
- [D] Attack per look moving / stationary = 10.6. A non-cryptic background,
  a bird and a small arena: an upper bound for a visual predator that sees the
  prey anyway (contrast Ioannou & Krause, where motion mattered only for
  cryptic prey - the two disagree on this; different predators and scales).

### Pohlmann, Grasso & Breithaupt 2001, PNAS 98:7371, doi 10.1073/pnas.121026298 [F] (class BUN = MOT + SPD, non-visual)
European catfish (PROXY) hunting guppies (PROXY) in infrared darkness.
- "Of the 94 observed attacks, 59 resulted in successful captures of guppies.
  Seventy-five (80%) of all attacks occurred on moving prey, which is
  significantly higher than expected from the proportion of time guppies spent
  swimming (43% of total time ...; P < 0.0001)."
- 46% of attacks from behind; "Of all rear attacks, 95% were directed toward
  moving prey."
- Wakes "up to 10.3 s old" followed, over "up to 121 cm (~55 prey-body
  lengths)" in a 120 cm tank; authors expect longer in quiet natural water
  (Hanke et al. 2000: a wake measurable for 3 min in still water) [S].
- [D] attack rate per unit time moving / stationary = (0.80/0.43)/(0.20/0.57)
  = 5.3. In a 1.2 m tank a moving guppy also meets the catfish more often by
  its own motion, so this bundles encounter-by-speed with wake detection.
- Relevance: a swimming fish leaves a hydrodynamic/chemical trail that widens
  its effective encounter cross-section to a lateral-line/olfactory hunter; a
  still fish leaves none. [J] Among the mechanisms found here it is the only
  one that does not need light (night, deep beds).

### Ioannou & Krause 2009, Biol Lett 5:191-193, doi 10.1098/rsbl.2008.0758 [F] (class MOT)
Three-spined sticklebacks (40 mm, from an estuary; PROXY) attacking chironomid
larvae (PROXY prey) on cryptic or conspicuous backgrounds, moving or still.
- "The prey needed to be both cryptic and motionless to avoid being attacked
  by the fish, i.e. there was a significant interaction between movement and
  crypsis in the probability of attack (GLM, background x movement: LRT 1,68 =
  5.92, p < 0.05). When prey were both cryptic and still, they were almost
  never detected".
- Experiment 2: from cryptic groups the fish took more active individuals
  (p < 0.001); "the fish selected prey randomly from conspicuous groups
  regardless of prey activity (p = 0.803)".
- Proportions only in Fig. 2 (not digitised).
- [J] Implication: motion raises detection where the prey would otherwise
  blend in. A fish seen in silhouette from below or against open water is
  conspicuous; motion would then add little. A countershaded fish on mud may
  be the cryptic case.

### Wright & O'Brien 1984, Ecol Monogr 54:65, doi 10.2307/1942456 [S via Wetterer & Bishop 1985, Ecology 66:457, read F]
- Wetterer & Bishop (p. 458): "Wright and O'Brien (1984) found that fish could
  locate moving diaptomid copepods at three times the distance at which they
  could locate equivalent-sized nonmoving individuals. Most diaptomids are
  motionless much of the time, moving only intermittently. This behavior helps
  make these zooplankton much less vulnerable to visual detection".
- White crappie (PROXY) on copepods (PROXY prey) near the acuity limit.
- [D] area ratio 3^2 = 9. A detection ratio for small prey near the acuity
  limit, not an activity-risk ratio for prey fish. [J] Not transferable as is
  to fish prey several cm long, whose detection is likely contrast- rather
  than motion-limited.
- Abstract of the 1984 paper [A] confirms a step model (location, pursuit,
  attack, retention), field tested.

### Utne-Palm 1999, J Fish Biol 54:1244, doi 10.1111/j.1095-8649.1999.tb02052.x [A] (class MOT; marine predator)
Two-spotted goby (marine, PROXY) on Calanus: "Both prey contrast ... and prey
mobility caused reaction distance of G. flavescens to increase. The effect of
prey mobility or prey transparency on reaction distance was independent of
turbidity level." Values not in the abstract; full text not obtained.

### Ware 1973, J Fish Res Board Can 30:787, doi 10.1139/f73-134 [A]
Visual risk model for rainbow trout on benthic invertebrates: "prey activity,
prey exposure, prey density, and prey size are four of the major determinants
of prey risk in benthic food chains". No ratio in the abstract.

### Menidia trio (Atlantic silverside, marine forage fish, PROXY) [A]
- Lankford, Billerbeck & Conover 2001, Evolution 55:1873, doi
  10.1111/j.0014-3820.2001.tb00836.x: "Silversides with large meals in their
  stomachs suffered significantly higher predation mortality than unfed
  silversides. Differences in predation vulnerability were likely due to
  swimming performance, not attractiveness to predators." Fast growth (~1.0
  vs ~0.5 mm/d) also raised vulnerability. Predators: "natural, common piscine
  predators" (species not named in the abstract). Numbers not obtained
  (paywalled; the BioOne page shows only the abstract).
- Billerbeck, Lankford & Conover 2001, Evolution 55:1863, doi
  10.1111/j.0014-3820.2001.tb00835.x: "Fed fish had lower burst speeds and
  less endurance than fasted fish from the same population."
- Munch & Conover 2003, Evolution 57:2119, doi
  10.1111/j.0014-3820.2003.tb00389.x: in mesocosms up to 30 d "the
  fast-growing population suffered significantly higher mortality throughout
  the trials despite being 40% larger".
- [J] This is a cost of HAVING EATEN (a full stomach slows the escape), not of
  moving. It persists into a rest tick after a feeding tick, so in a 6 h
  model it does not make rest safer; it makes recent intake risky.

### Krause & Godin 1996, Behav Ecol 7:264, doi 10.1093/beheco/7.3.264 [A] (class VIG)
Guppies (PROXY) vs blue acara cichlid. "Nonforaging guppies reacted sooner to
and initiated flight further away from the approaching model than did
foraging fish collectively"; free-swimming cichlids: risk nose-down foraging
> horizontal foraging > nonforaging. Values not in the abstract (prior note
[A]). Godin & Smith 1988 Nature 333:69 (doi 10.1038/333069a0): OpenAlex has
no abstract; remains [S] from prior notes.

### Yokogawa et al. 2007, PLoS Biol 5:e277, doi 10.1371/journal.pbio.0050277 [F] (class VIG, opposite sign)
Adult zebrafish (PROXY): "Inactivity was thus associated with an increased
arousal threshold to electrical stimulation, with the greatest differential
response observed at 0.38 V"; "at high voltage, all animals demonstrated a
response regardless of prior activity state (~100% at 2 V)". Sleep defined
as >= 6 s of immobility. Response percentages only in Fig. 1E.

### Norman, Munson, Cortese, Koeck & Killen 2024, J Exp Biol 227, doi 10.1242/jeb.247138 [F] (context; review of fish sleep)
- Increased arousal threshold shown in zebrafish (mechanical and electrical
  stimuli), Mozambique tilapia (electrical and food stimuli); divers "could
  lift individual fish from the bottom of the tank when they appeared to be
  asleep (Russell, 1971)" [S].
- No gadoid or clupeid sleep/arousal-threshold measurement in the review's
  text (searched for cod, herring, Gadus, Clupea; Atlantic cod appears only
  in a seismic-survey reference).
- [J] The resting state of a fish is not a vigilant state: rest may LOWER
  escape probability per detected encounter. This cuts against a rest payoff
  and must be weighed with the motion-detection gain.

### Meta-analyses of behaviour and survival [A]
- Moiron, Laskowski & Niemela 2020, Ecol Lett 23:399, doi 10.1111/ele.13438:
  "there was no directional relationship between riskier behaviour and greater
  mortality ... behaviour explained a significant, but small, portion (5.8%)
  of the variance in survival. We also found that risky (vs. 'shy')
  behavioural types live significantly longer in the wild, but not in the
  laboratory."
- Smith & Blumstein 2008, Behav Ecol 19:448, doi 10.1093/beheco/arm144:
  "bolder individuals had increased reproductive success ... but incurred a
  survival cost"; exploration had a positive effect on survival.
- [J] Between-individual "activity/boldness" is a poor proxy for the within-
  individual forage/rest switch; these say only that the generic sign is not
  robust across studies.

### Theory and amphibian proxies [A]
- Werner & Anholt 1993, Am Nat 142:242, doi 10.1086/285537: models with
  growth and mortality "linear with activity levels" (foraging speed or time
  active), plus activity-independent mortality. An assumed form, not a
  measurement.
- Anholt & Werner 1995, Ecology 76:2230, doi 10.2307/1941696: bullfrog
  tadpoles at low food "moved more often and more quickly" and "suffered
  higher predation mortality" from dragonfly larvae. No numbers in abstract.
- Skelly 1994, Anim Behav 47:465, doi 10.1006/anbe.1994.1063: no abstract
  indexed (Crossref/OpenAlex); not read. Dropped.
- Walters & Juanes 1993, CJFAS 50:2058, doi 10.1139/f93-229 [A]: theory;
  juveniles forage in "limited but risky areas near refuges" (refuge case).

### Encounter by prey speed (class SPD)
Gerritsen & Strickler 1977 (doi 10.1139/f77-008, not read) as written in a
2026 PLoS One chaetognath paper (doi 10.1371/journal.pone.0340065, PMC12758766,
read) [S]. The page text had its math markup stripped (operators lost); the
layout below is reconstructed and checked against the known limit C -> pi R^2
N v when the prey is still:
C = pi R^2 N [(x + y)^3 - |x - y|^3] / (6 x y), x prey speed, y predator speed.
[D] For prey speed u <= predator speed v this is pi R^2 N (u^2 + 3 v^2)/(3 v),
so swimming / still (u = 0) = 1 + (u/v)^2 / 3: at most 1.33 (u = v), 1.08 at
u/v = 0.5, 1.01 at u/v = 0.2. For u > v (slow or ambush predator) the ratio
is (u^2 + 3 v^2)/(3 u v), unbounded as v -> 0. Resting at u = 0 is an upper
bound: pelagic schoolers and fish without a swim bladder never stop.
No measurement of encounter rate vs prey speed in fish was found.

### Scharf, Buckel, McGinn & Juanes 2003, JEMBE 294:41, doi 10.1016/S0022-0981(03)00260-0 [F, partial pdf] (context, marine)
Bluefish and other piscivores on bay anchovy, menhaden, silverside, age-0
striped bass. Interspecific; the authors infer (p. 56) that "Higher predator
attack rates on striped bass prey may be related to higher activity levels
of striped bass in the presence of predators compared to Atlantic silverside,
which demonstrate reduced activity and utilization of surface waters as a
refuge". Activity was not contrasted within a species; no ratio.

### Other items read, not used
- Stamps 2007, Ecol Lett 10:355, doi 10.1111/j.1461-0248.2007.01034.x [A]:
  hypothesis paper; predicts that faster-growing individuals "will take more
  risks in foraging contexts"; "tentative empirical support". No number.
- Jones & Godin 2010, Proc R Soc B 277:625, doi 10.1098/rspb.2009.1607 [F]:
  juvenile convict cichlids (PROXY); more exploratory fish were slower to react
  to a robotic predator while foraging (r = 0.52, p = 0.010). Every fish was
  tested while foraging - no forage vs non-forage contrast; not used.

## Not found / not read
- Any marine adult or sub-adult gadoid or clupeid number for active vs resting
  risk: NOT FOUND in two standard and one extended web search (cod, haddock,
  whiting, herring, sprat) plus the Menidia/forage-fish line. Not an
  exhaustive database search. Gotceitas,
  Fraser & Brown 1995 (prior notes) concerns PREDATOR activity, not prey.
- Any field measurement at equal light, depth and school state: NOT FOUND.
  Every whole-system number (Biro 2003, 2004) bundles activity with habitat.
- Lankford et al. 2001 fed vs unfed numbers: paywalled, not obtained.
- Biro et al. 2006 J Anim Ecol 75:1165: Unpaywall OA link returned 403; not
  read. A UTS repository file thought to be it was Biro, Post & Abrahams
  (energy allocation, Proc R Soc B), not used.
- Lateral-line detection range of moving vs still fish prey: only search-
  snippet level ("one to two body lengths"; toadfish ~0.4 body length), no
  primary source read; not used.
- Howick & O'Brien 1983, Wright & O'Brien 1982: no abstracts; not used.
- Milinski & Heller 1978, Pitcher, Lang & Turner 1988 (doi
  10.1007/BF00300573), Hoey & McCormick 2004: no abstracts indexed; not used.
- Godin & Crossman 1994, Biro, Post & Parkinson 2003 Ecology 84:691: not
  attempted.
- (a) A MEASURED encounter rate vs prey swimming speed in fish: not searched
  specifically; only the Gerritsen-Strickler form is given.
- (c) Exposure by position when foraging on open mud / open water (leaving
  the bed or the school, burial of flatfish or sandeels): not attempted beyond
  the bundled whole-lake studies and the shelter studies above.
- Krause & Godin 1996 full text: the OUP page shows only the abstract.

## Synthesis [J]
1. Measured active/rest risk ratios exist only for PROXY systems: 10.6 per look
   (merganser, non-cryptic background, pooled head-dips), 5.3 per unit time
   (catfish in the dark, wake tracking + own-motion encounter), and a bundled
   seasonal mortality ratio 1.6 (age-0 trout, activity + habitat). Krause &
   Godin give the sign for foraging posture (vigilance) without numbers in the
   abstract. Separately, for mechanism (b) only: detection distance x3 / area
   x9 for moving copepods (crappie) - a detection ratio, not an r.
2. The prey-speed encounter term alone adds at most a factor 1 + (u/v)^2/3
   (<= 1.33 when the predator is at least as fast as the prey).
3. Motion detection is conditional: it matters for cryptic prey (Ioannou &
   Krause) and for non-visual hunters (Pohlmann), and less for prey already
   conspicuous to a visual predator. The merganser result (10.6 on a white
   background) disagrees; it is a bird at < 1 m.
4. Opposite-sign evidence: resting fish have raised arousal thresholds
   (zebrafish, tilapia), and a fed fish swims and escapes worse (Menidia), so
   rest after a meal is not a safe state. Meta-analyses find no consistent
   sign between "risky" behavioural types and survival.
5. No number found for the model's species; any rest payoff for whiting,
   saithe, cod, haddock, hake, Norway pout, herring, sprat or mackerel would
   be carried over from these proxies by mechanism, not measured.
