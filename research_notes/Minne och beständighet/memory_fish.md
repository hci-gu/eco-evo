# Memory of the fish observers: where food was, where a predator was (2026-11-14)

Research subagent. Nothing in code or library changed.

Purpose: the MEMORY side of the new neighbour reading. Under it, a fish knows a neighbour if it visited that neighbour
within its memory span AND the target's distribution has persisted since. The persistence side is a separate note.
This note asks only how long a fish's place knowledge lasts, for two kinds of information: (a) where food was,
(b) where a predator was. Builds on research_notes/Observabilitetsparen/fish.md (sections 2, 7: movement, senses),
design.md section 1 (reading B, the memoryless w), and "Nettoforflyttning per funktionsgrupp"/herring.md and gadoids.md.

Grades: [F] measured, full text read here; [F/A] measured, abstract or search snippet only; [S] secondary (a measured
value seen only as cited by another paper); [D] derived here by arithmetic; [J] judgement. Downloads in `papers/`
(untrusted data, read as text): `bjornsson2010/` (ICES CM 2010/I:09), `dean2012/` (NAJFM 32:124), `brydges2008/`
(Anim Behav 75:935), `utne1997/` (Sarsia 82:129). Also in `papers/`: the Unpaywall `.json` records for the DOIs tried
here (no open copy for most), the Jutfelt & Hedgarde 2013 PDF, and files from sibling notes.

## 0. Three axes, kept apart

1. **Kind of memory.** (i) Place memory of a food site. (ii) Place memory of danger. (iii) Stimulus memory: recognising
   a predator odour, a hook, or a feeding sound. Only (i) and (ii) are what a neighbour channel would carry. (iii) is a
   proxy. A minnow that recognises pike odour for two months says nothing about how long it avoids a PLACE.
2. **Time scale.** Minutes to hours (a school re-forms after an attack); days to weeks (patch and site learning, returns
   in telemetry); months to years (migration conservatism, spawning-site fidelity). Only hours to ~3 weeks can move w at a
   6 h tick. The year-scale results are reported but are not used in the span table.
3. **What telemetry shows.** Site fidelity alone is not memory: a cod that never leaves its 5 ha proves nothing about
   recall. The usable signal is a **return after absence**: how long the fish was away before it came back. That interval
   is pulled out below wherever a study gives it.

Unit: 1 tick = 6 h, so 1 d = 4 ticks and 1 week = 28 ticks.

## 1. Spatial memory of food locations

### 1.1 Lab evidence on place memory of food patches (non-model species: a proxy for every group)

| species | result | span | grade, source |
|---|---|---|---|
| three-spined stickleback | fish fed in one of two patches still preferred "their" patch when tested; long-term memory of the patch where it had fed "for at least 8 d" (abstracts disagree on whether 1-4 or 1-14 d intervals were tested) | >= 8 d | [F/A] Milinski 1994, Ecology 75:1150 (DOI not verified) |
| three-spined stickleback, 8 populations | spatial food-patch task, retention tested at 7 and 21 d. River fish (unstable habitat) returned to the last rewarded patch after 7 d but not after 21 d. Pond fish remembered from day to day but not after 7 d. High-predation fish remembered at 7 d, neither group at 21 d | river 7-21 d; pond 1-7 d | [F] Brydges, Heathcote & Braithwaite 2008, Anim Behav 75:935, doi 10.1016/j.anbehav.2007.08.005 (abstract and results read) |
| fifteen-spined stickleback, three-spined stickleback | "memory window" for learned PREY-HANDLING skill (not place): 8 d (resident marine), 10 d (anadromous), > 25 d (freshwater) | 8 to > 25 d | [F/A] Mackney & Hughes 1995, Behaviour 132:1241 |
| fifteen-spined stickleback, corkwing wrasse | radial maze: fish learned visual cues for patch status and tracked renewal (win-stay / win-shift) on patches whose status changes over a tidal cycle | hours (tidal) | [F/A] Hughes & Blight 2000, Anim Behav 59:613, doi 10.1006/anbe.1999.1351 |
| stickleback | "Fading short-term memory for patch quality in sticklebacks", Anim Behav 33:678 (1985), doi 10.1016/S0003-3472(85)80097-X | not read | title only; a lead |

The lab spans for food places are **1 day to 3 weeks**. Two consistent findings: the span is matched to how stable the
habitat is, and it differs between populations of one species (Brydges 2008; Mackney & Hughes 1995) [F, F/A]. So memory
span and the persistence of food are not independent quantities: a fish's forgetting is an adaptation to how long its
food stays put [J]. For the model, a group's memory span should be read alongside the persistence of its prey field.

### 1.2 Gadoids: cod (benthivores 0.31), saithe (water column 0.80)

- **Learning a feeding site, cod in a sea cage** (Bjornsson, Karlsson & Gudbjornsson 2010, ICES CM 2010/I:09, full,
  `papers/bjornsson2010/`) [F]. Twenty naive adult cod were trained in an 11 m cage with two feeding platforms, a
  250 Hz sound and six sessions a day. They ate on day 3, first responded to the sound on day 6 and responded
  consistently from day 8. In that 10-day run there was "no evidence" that they used the sound's direction; excluding
  days 1-4 they took 4-6 min to find food. With trained "teachers" present (Experiment 2), learning took under a day,
  and from days 2-6 the fish used the sound's direction and reached the correct platform in ~1 min (see 3.1). This is the time to LEARN a site at 1-10 m scale, not a
  retention time. Published version: Bjornsson, Karlsson & Macrander 2018, J Fish Biol, doi 10.1111/jfb.13783 [F/A].
- **Released conditioned cod (ranching).** 1600 pond-reared one-year cod, conditioned to a 160 Hz signal at a feeding
  location, were released into a Norwegian fjord and came to the feeding site. About 25 % of the cod attracted were of
  wild origin (Midling, Kristiansen, Ona & Oiestad 1987, ICES CM working paper [F/A]; the 25 % via Bjornsson 2010 [S]).
  Icelandic "herd cod" fed at four stations stayed apart from unfed wild cod, shallower and warmer, through a season
  (Bjornsson 2019, CJFAS, doi 10.1139/cjfas-2018-0305 [F/A]). Retention after feeding stopped is not reported in what I
  could read. Red sea bream (proxy): "three months later a large fraction of the conditioned fish responded to the sound
  signal" (Fujiya et al. 1980, via Bjornsson 2010 [S]).
- **Stimulus memory in cod (proxy only).** A light-food trace conditioning was retained "after at least 3 months"
  (Nilsson, Kristiansen, Fosseidengen, Ferno & van den Bos 2008, Anim Cogn 11:215, DOI not verified) [F/A]. This is
  cue and time memory, not place memory.
- **Daily returns to feeding and resting sites, juvenile cod** (Clark & Green 1990, Can J Zool 68:1434, doi
  10.1139/z90-214) [F/A]. In summer, 28-33 cm cod ranged > 3 km/day and migrated daily between deep inactive water
  (30 m) and shallow feeding water (< 15 m). In autumn they were active by day in home ranges of 545-2582 m2 and
  inactive at night at **consistent resting sites** over rock. This is a return after ~12 h away, repeated daily. It
  sets a floor of >= 1 d for place memory [J], and the cues (depth, substrate) may be enough to re-find the sites.
- **Return after 8 days away** (Dean, Hoffman & Armstrong 2012, NAJFM 32:124, doi 10.1080/02755947.2012.663457, full,
  `papers/dean2012/`) [F]. One tagged cod (F08) left a Massachusetts Bay spawning aggregation right after tagging on
  24 June and returned eight days later, on 3 July. This is a spawning site, not a food site, and n = 1.
- **Saithe at salmon farms** (artificial food sources):
  - **Daily trips.** Over 3 months, 63 % of tagged saithe were detected daily at one or more of three farms. Residents
    spent 8-10 h/day by the cages, and 15 of 24 fish made 2-21 trips between farms 1.6-4.7 km apart (Uglem et al. 2009,
    MEPS 384:251, doi 10.3354/meps08001) [F/A]. These are repeated returns to km-distant food sites at intervals of
    about a day.
  - **Returns after weeks to months.** 62 saithe were tracked for up to 2 years in Ryfylke. Many were at farms for
    > 75 % of the time, and **some had gaps of weeks or months without detection before they reappeared**. Detections
    at one farm fell after the salmon were harvested and feeding stopped, possibly also because nets were removed and
    lights switched off (Ottera & Skilbrei 2014, ICES JMS 71:2484-2493, DOI not verified) [F/A+: full text summarised
    by a fetch helper and paraphrased, not read line by line here].
  - Caveat [J]: a farm is a permanent, loud, odorous and lit landmark. A return after months may be re-detection, not
    recall. The daily returns are the cleaner evidence that saithe keep km-scale food-site knowledge for >= 1 d.
- **Year-scale homing (out of scope for the span).** In a century of tagging of 174 cod groups, 41 % were sedentary,
  18 % accurate homers, 20 % inaccurate homers and 20 % dispersers (Robichaud & Rose 2004, Fish Fish 5:185, doi
  10.1111/j.1467-2679.2004.00141.x) [F/A]. Transplanted cod homed better from a familiar migration route than from an
  unfamiliar area, which points to landmark learning (Windle & Rose 2005, Fish Res, doi 10.1016/j.fishres.2005.05.006)
  [F/A]. Natal homing in Skagerrak-Kattegat cod (Svedang et al. 2007, via gadoids.md) [S].
- **Haddock (0.39 of the benthivores), whiting, pollack, hake, ling, spurdog**: no study of food-place memory or return
  after absence was found.

### 1.3 Clupeids and mackerel (planktivores, mackerel)

- **Herring, hours to weeks: nothing found.** No study tracks a herring school's return to a food patch. Schools
  change status every 5.5 min and split or join every ~35 min (Pitcher et al. 1996, via fish.md) [F/A]. So a school is
  not a persistent carrier of memory beyond about an hour. Place memory over days would have to be held by individual
  fish that keep reshuffling between schools [J].
- **Herring, years (out of scope).** Migration habits form early in adult life and pass from old to young fish
  (Corten 2002, Rev Fish Biol Fish 11:339) [F/A]. (A model of the same mechanism, Huse, Railsback & Ferno 2002,
  J Fish Biol 60:571, is cited from memory and was not retrieved.) Removing older fish was followed by an ~800 km shift of the NSS herring
  spawning in 2021-2024, read as "fishery-induced collective memory loss" (Slotte et al. 2025, Nature 642, doi
  10.1038/s41586-025-08983-3) [S, press coverage only]. Herring therefore can keep place information for years, but
  only as a route or seasonal ground, not a patch.
- **Norway pout, blue whiting, sprat, mackerel**: nothing found at any scale except mackerel's seasonal migration
  fidelity, which was not reviewed here.

### 1.4 Flatfish (part of the benthivores)

- **Juvenile plaice, site fidelity and homing at 100 m** (Burrows, Gibson, Robb & Maclean 2004, J Fish Biol 65:620)
  [F/A]. More than 90 % were recaptured within 100 m of release after 6 weeks, with a daily stay probability of 0.78 in a
  100 m zone. Fish moved 100 m away along the shore drifted back toward their capture sites over 6 weeks; controls did
  not. This is a home-site preference held for >= weeks, at a sub-cell scale.
- No adult-plaice or dab memory of feeding sites was found.

## 2. Memory of a predator at a place

### 2.1 Place avoidance after a predator cue (direct measurements; none on model species as the prey)

| prey (predator) | setting | avoidance of the PLACE lasts | grade, source |
|---|---|---|---|
| two-spotted goby (cod) | two feeders in a tank; cod seen, smelled or both at one feeder, then 2 days of feeding with no cod | avoided the site significantly the next day; not significant on day 2. Odour gave longer persistence than sight. "After 1-2 days the gobies returned to the earlier affected feeding patch" | [F] Utne & Bacchi 1997, Sarsia 82:129 (full, `papers/utne1997/`) |
| brook stickleback (minnow alarm cue) | field; area marked, cue source removed | strongest in the first 2 h; area use back to premarking level by 4 h | [F/A] Wisenden, Chivers & Smith 1994, J Chem Ecol 20:2975, doi 10.1007/BF02098403 |
| fathead minnow, dace (alarm cue) | field | avoided labelled areas >= 2 h; field avoidance 2-4 h generally; alarm cue itself active 3-6 h | [F/A] Wisenden et al. 1995, 2009 (via search summary of the minnow literature) |
| Atlantic salmon parr (predator seen) | lab | feeding motivation reduced up to ~1 h | [S] Metcalfe 1987, via Utne & Bacchi 1997 |
| bluegill (predator seen) | lab | foraging rate recovered within ~15 min | [S] Gotceitas 1990, via Utne & Bacchi 1997 |
| reef herbivores (reef sharks) | field video, 5 min before vs after a shark passed | "negligible" effect on grazing | [F/A] Tebbett et al. (search summary only; year and DOI not seen) |
| three-spined stickleback (simulated bird attack at a favoured patch) | lab | 15 of 16 fish took up to 15 days to LEARN to avoid the patch; once attacks stopped, both populations resumed the patch alike (rate not seen) | [F/A] Huntingford & Wright 1989, Behav Processes 19:181 |

Reading [J]. A predator encounter makes a fish avoid the place for **minutes (open-water, visual) to about one day
(a resident prey with a stationary predator, odour cue)**. The longest measured value in a marine fish with a model
predator (cod) is 1 day, and it fades on day 2. Utne & Bacchi argue that the 1-2 d persistence suits a predator that
sits still. The freshwater field values (2-4 h) cannot be told apart from the decay of the cue itself (3-6 h).

### 2.2 Gadoids leaving a place after an aversive event that is not a predator (upper bounds, confounded)

- **Seismic shooting** (Engas, Lokkeborg, Ona & Soldal 1996, CJFAS 53:2238, doi 10.1139/f96-177) [F/A]. Trawl catches
  of cod and haddock fell by ~50 % up to 18 nmi away, and "did not return to preshooting levels during the 5-day
  period after seismic shooting ended". Confounds: displacement over tens of km. Whether fish stayed away because they
  remembered, or because they had moved far and drift back slowly, cannot be told apart [J].
- **Gill-net fishery opening on a spawning aggregation** (Dean et al. 2012, full) [F]. Half the tagged cod were caught
  within 9 h. The other three left within 18 h and none returned in the 9 days the receivers stayed out. The one fish
  that had come back after 8 days (F08, 1.2) left again 13 h after the fishery opened. A single trawl pass on spawning
  cod left an avoidance "hole" five times the trawl width for over an hour (Morgan et al. 1997, via Dean 2012) [S].
- So danger can make gadoids stay away from a km-scale area for **>= 5-9 d**. In both cases the danger was massive and
  persisted (shooting for 5 days, a fishery staying open), and spawning motives interfere [J]. Neither is a single
  predator encounter.

### 2.3 Herring and sprat with predators

- **School structure, minutes.**
  - Under killer whale attack, herring showed an anti-predator event on average every 3.9 min. Attacked schools were
    denser and more circular, and schools above 460 m2 in cross-section were not attacked (Nottestad & Axelsen 1999,
    Can J Zool 77:1540, doi 10.1139/z99-124) [F/A].
  - Under fin whales, intra-school events came every 15.3 min and attacks every 170 min (Nottestad et al. 2002, via
    herring.md and search) [F/A].
  - Gas-bubble release by overwintering herring under killer whale and saithe attack (Nottestad 1998, ICES JMS 55:1133)
    [F/A].
- **No study** was found of herring, sprat or any clupeid avoiding a place for hours or days after a predator
  (mammal, gadoid, seabird). The school's response is local and immediate, and the school itself reshuffles every
  ~35 min (1.3) [F/A, J]. For porpoises and seals, which the herring meet only when found by them (design.md 1.3),
  there is no evidence of a lasting place memory at all.

### 2.4 Recognition of a predator stimulus (proxy, NOT place memory)

- Two-spotted goby: after cod was present in 3 consecutive feeding sessions, the gobies avoided cod odour, and the
  response was "still significant 6 mo after the experience". Naive gobies did not respond to odour alone (Utne-Palm
  2001, MEPS 218:267, DOI presumably 10.3354/meps218267, not verified) [F/A].
- Fathead minnow: >= 2 months (Chivers & Smith 1994). Salmonids after one learning event: 10-21 d (Brown & Smith 1998;
  Mirza & Chivers 2000). Juvenile rainbow trout: shy fish still responded after 8 d, bold fish did not. All via Ferrari
  et al. 2010 (Proc R Soc B 277:2205, PMC2880143; tadpoles) and press coverage [S].
- Ferrari et al. 2010: a learned predator response decays faster when risk is uncertain or rare: "adaptive forgetting"
  [F/A, tadpoles].
- Juvenile cod strongly avoid predator odour in a flume, so the sense and response exist; retention was not tested
  (Jutfelt & Hedgarde 2013, Front Zool 10:81, doi 10.1186/1742-9994-10-81) [F, abstract and methods read].
- Reading [J]: recognising what a predator is lasts weeks to months, but avoiding where it was lasts hours to a day.
  The neighbour channel is about where, so the short span is the relevant one.

## 3. Social transmission within schools and shoals

### 3.1 Measured time constants

- **Cod, site and cue knowledge** (Bjornsson et al. 2010, full) [F].
  - Training took 6-7 days for 20 naive cod alone, less than a day for 10 naive cod with 9 trained cod, and less than a
    day for 19 naive cod with 1 trained cod.
  - In the 1-teacher run, the first student followed the teacher to a platform 8 min later and a group of students
    2 min after that.
  - After 2 days the teacher was removed, and "already in the first acoustic session on their own the students reacted
    correctly".
  - One leader at a time, lasting 1-4 days; leaders ate more and grew faster.
  - The 2018 JFB version: < 2 d with one teacher vs ~1 week alone [F/A].
  - So one informed cod passes site knowledge to a group of ~20 within hours to a day, and the group keeps it after the
    informant leaves.
- **French grunts (reef, proxy).** Fish moved to new schooling sites learned the residents' twilight migration routes
  and kept using them after the residents were removed. Controls did not (Helfman & Schultz 1984, Anim Behav 32:379,
  doi 10.1016/S0003-3472(84)80272-9) [F/A]. Days of following are needed, but the exact number was not seen.
- **Guppies (proxy).**
  - Naive fish shoaling with trained founders took the founders' route to food, and kept the preference when tested
    alone after 5 days of trials (Laland & Williams 1997, Anim Behav 53:1161, doi 10.1006/anbe.1996.0318) [S, via review].
  - Traditions, including a maladaptive longer route, persisted 3 days after all founders had been removed (Laland &
    Williams 1998, Behav Ecol 9:493) [F/A].
- **Golden shiners (proxy).**
  - A trained minority leads naive shoal-mates to the right place at the right time of day (Reebs 2000, Anim Behav
    59:403) [S].
  - Groups track a moving light gradient that single fish and small groups cannot; the sensing is emergent from local
    interactions, not from pooled individual estimates (Berdahl et al. 2013, Science 339:574, doi
    10.1126/science.1225883) [F/A].
- **Within-school alarm propagation, seconds.**
  - Waves of agitation crossed Peruvian anchovy schools under sea-lion attack at 7.45 m/s, against a school speed of
    0.3 m/s (Gerlotto et al. 2006, ICES JMS 63:1405, doi 10.1016/j.icesjms.2006.04.023) [F/A].
  - Capelin and cod formed density waves over > 10 km within minutes (Pednekar et al. 2024, via fish.md 7.2) [F].
- **Herring, between schools.** Split/join every ~35 min (Pitcher 1996) [F/A]. Year-scale transmission of migration
  routes old to young (Corten 2002; Slotte et al. 2025) [F/A, S].

### 3.2 Does a school "know" more than one fish? [J on F]

Yes, in two senses:
1. **Pooling.** One informed fish sets the movement of ~20 within hours (cod) and a minority leads a shoal (shiners).
2. **Emergent sensing.** A group responds to gradients no single member resolves (Berdahl 2013).

Time constants for the model:
- Inside a school, seconds (alarm waves).
- Between schools of a cell, tens of minutes (herring split/join).
- For a learned food site spreading through a group, under 1 day (cod).

All three are at or below one tick. So pooling within the cell's biomass can be treated as complete within a tick for
schooling fish (design.md 1.3, the pooling bullet). This bears on the coverage of the neighbours (more tracks pooled),
not on how long anything is remembered. For solitary or loosely aggregated bed fish (flatfish, anglerfish, ling) no
transmission was found [J].

## 4. A memory span at a 6 h tick, per group and information type

### 4.1 How the spans were built [J]

- **Food place.**
  - The direct gadoid evidence is returns after ~0.5-1 d (cod daily sites, saithe daily farm trips), one return after
    8 d (cod, spawning site) and returns after weeks to months (saithe at farms, landmark-confounded).
  - The lab spans for food places are 1-21 d (sticklebacks), matched to habitat stability.
  - Central 7 d is the stickleback river value and the cod 8-d return. The range 1-21 d spans the lab and gadoid
    evidence. The year-scale homing is excluded.
- **Predator place.**
  - Central 6 h (1 tick), range 15 min to 2 d. This spans open-water and freshwater recoveries (minutes to 4 h) and
    the one marine prey-of-cod value (1 d, gone by 2 d).
  - For demersal residents the central is set at 1 d, since the goby case (resident prey, stationary predator) is the
    closest analogue.
  - The gadoid seismic and fishery bounds (>= 5-9 d) are disturbance upper bounds, not single-encounter memory, and are
    left out of the range.
- **Biomass weighting was not possible.** At most one member per group has data (cod in the benthivores, saithe in the
  water column, plaice for the flatfish), so each row is that member's value or a proxy, not a weighted mean. Haddock,
  whiting, Norway pout, blue whiting, sprat and mackerel have no data, so every value for them is a proxy.

### 4.2 The table

| group (members) | food-place span, central (range) | predator-place span, central (range) | basis | grade |
|---|---|---|---|---|
| planktivores (herring 0.55, pout 0.35, blue whiting, sprat) | no measured value; proxy 2 d = 8 ticks (0.5-7 d = 2-28 ticks) | 1 h, < 1 tick (minutes to 6 h) | herring: no patch memory found; schools reshuffle every 35 min; prey field (plankton) less stable than a bed, so the short end of the lab range (pond fish 1-7 d) [J]; predator: school responses last minutes (Nottestad), no place avoidance found | food [J, proxy]; predator [F/A + J] |
| mackerel (0.94) | no measured value; proxy 2 d = 8 ticks (0.5-7 d) | 1 h, < 1 tick (minutes to 6 h) | nothing on mackerel; same reasoning as herring (pelagic, ephemeral prey) | [J, proxy] |
| water column (saithe 0.80, spurdog, hake, anglerfish, ling, pollack) | 7 d = 28 ticks (1-21 d = 4-84 ticks); saithe returns to food sites seen after weeks-months (landmark-confounded) | 6 h = 1 tick (15 min to 1 d) | saithe daily farm trips 1.6-4.7 km and long gaps then return (Uglem 2009; Ottera & Skilbrei 2014); the rest no data | food [F/A + J]; predator [J, proxy] |
| whiting | 7 d = 28 ticks (1-21 d), gadoid proxy | 6 h = 1 tick (15 min to 1 d) | no data; cod/saithe proxy | [J, proxy] |
| benthivores (haddock 0.39, cod 0.31, flatfish ~0.24) | 7 d = 28 ticks (1-21 d); plaice home-site preference >= 6 weeks at 100 m scale | 1 d = 4 ticks (6 h to 2 d) | cod daily site returns (Clark & Green 1990), 8 d return (Dean 2012), learning a site 6-8 d alone / < 1 d socially (Bjornsson 2010); plaice homing (Burrows 2004); predator: goby avoids cod's site 1 d (Utne & Bacchi 1997); haddock no data | food [F + J]; predator [F proxy + J] |

### 4.3 How the span meets the revisit interval (for the persistence side; w is not computed here) [D/J]

> Correction (main session, summary.md section 3): the independent-tick form below overstates coverage for
> ranging groups. Under a walk calibrated to the measured 6 h and 24 h displacement the ticks are not
> independent; see `tables/grid_memory_persistence.txt` (e.g. porpoises -> diffuse 0.45 at 4 ticks, not 0.85).

From design.md 1.3, the memoryless w is the chance per tick that a group's track enters a given neighbour (diffuse
targets). Assuming independent ticks, the chance it was visited within the last M ticks is 1 - (1 - P)^M:

| group | P per tick | M = 1 tick | M = 4 ticks (1 d) | M = 28 ticks (7 d) |
|---|---|---|---|---|
| planktivores | 0.38 | 0.38 | 0.85 | ~1 |
| mackerel | 0.42 | 0.42 | 0.89 | ~1 |
| water column | 0.48 | 0.48 | 0.93 | ~1 |
| whiting | 0.31 | 0.31 | 0.77 | ~1 |
| benthivores | 0.19 | 0.19 | 0.57 | 0.997 |

Caveats [J]:
1. **Independence fails for home-ranging fish.** A cod with a 5 ha home range near the west edge of its cell crosses
   into the west neighbour repeatedly and never into the east one. For residents the set of neighbours visited is fixed
   by where the home range lies, and memory cannot raise coverage beyond the neighbours that home range touches. Under
   this reading a resident's long memory raises w on its one or two "home" neighbours, and its share of neighbours known
   stays near the memoryless value. For ranging fish (herring, saithe, mackerel) independence is closer to true.
2. **Food memory is long against the revisit interval.** With food spans of days and revisit intervals of 1-5 ticks,
   every ranging group would know all four neighbours. Its w would then be set by persistence, not memory.
3. **Predator memory is short.** At <= 1 tick for pelagic fish, it adds nothing beyond the memoryless reading B. At
   ~4 ticks for benthivores, it would add something, but only on the neighbours the home range touches (caveat 1).
4. **The predator is a moving point target, so the target's persistence binds.** A remembered predator position is
   worth something only while the predator stays. The goby study's authors make exactly that argument for the 1-2 d
   span with a sedentary cod.
5. **Cell biomass is not one fish.** Memory belongs to individuals, while the model's cell biomass reshuffles by
   movement. A remembered neighbour belongs to fish that may since have left the cell; pooling (section 3) partly
   compensates for schooling fish.

## 5. Gaps

1. No measurement of food-place memory over hours to weeks for any model member except cod (daily returns, one 8 d
   return) and saithe (farm landmarks). Haddock, whiting, Norway pout, blue whiting, sprat, mackerel, hake, ling and
   spurdog have nothing.
2. No study of herring or any clupeid returning to, or avoiding, a place over hours to days. Every herring memory
   result is either minutes (school structure) or years (migration culture).
3. No study of a model fish avoiding a place after a single encounter with a seal, a porpoise or a fish predator. The
   only marine value is a goby avoiding a cod's site for ~1 d.
4. Retention after feeding stops is not available in what I could read for conditioned or herd cod (Midling 1987,
   Bjornsson 1999/2002/2019) or for saithe at fallowed farms (Ottera & Skilbrei 2014 report only a qualitative fall).
   This is the cleanest possible test of food-place memory in gadoids, and a lead.
5. Not read in full (abstracts or citations only): Milinski 1994 (tested interval range disputed between abstracts),
   Huntingford & Wright 1989 (retention after attacks stopped), Ferno & Huse 1983 (cod hook avoidance after hooking,
   Fish Res 2:19, doi 10.1016/0165-7836(83)90100-5, no abstract seen), Helfman & Schultz 1984 (days of following),
   Laland & Williams 1997, Lokkeborg & Ferno 1999 (doi 10.1023/A:1007504712163), Engas 1996, Slotte et al. 2025.
   Unpaywall found no open copy for most of them.
6. The lab spans come from sticklebacks and wrasse in tanks of < 1 m to a few m. Whether a fish's place memory at
   1 km (a neighbour cell) has the same span as at 1 m is untested [J].
7. Memory span is itself matched to habitat stability (Brydges 2008; Mackney & Hughes 1995). Treating memory and
   persistence as independent factors in w may double-count persistence [J].
