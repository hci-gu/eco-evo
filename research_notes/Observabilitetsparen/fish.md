# Observation pairs of the fish observers - own cell and neighbour cells (2026-11-13)

Research note (literature only). Nothing in the code, library, manifest, tests, AGENTS.md or the resume was
changed; no value is proposed from model behaviour. Grades: [F] measured (field or lab), [J] judgement or
inference (incl. my own range arithmetic). Reading of each source: "full" = full text read (here or in the
cited project note), "abs" = abstract, index record or secondary citation only. Sibling notes in this folder:
`code.md` (layout and cost of a per-pair scope).

## 0. The question and the model

Each decision maker reads, for every group it observes, that group's biomass density in its own 1 km cell
and in the four neighbours N/E/S/W (centres 1 km away); one switch per (observer, target) turns both on
(`<obs>_observes_<target>` in `fgconfig/fg_library.yaml`). The observation is meant as a diffuse blend of all
senses (smell, sound, lateral line, sight), not one sense (user; research_notes/"Gom och sinnen").

Two readings of a neighbour channel are documented side by side, without choosing:
- **A (sensing at distance)**: the fish senses the target 0.5-1.5 km away.
- **B (ranging within the tick)**: an animal that covers several km in 6 h has itself sampled the
  neighbouring km, so the channel stands for what it met there.

Fish pairs switched on today (library read 2026-11-13; seabirds muted in `mareld2.yaml`). Role = the model's
checked predation pairs.

| observer | prey | predators | no food link | self |
|---|---|---|---|---|
| planktivorous_fish | zooplankton, krill | benthivores, whiting, water column, mackerel, porpoises, seals, (seabirds, muted) | - | yes |
| mackerel | zooplankton, krill, planktivores | water column | benthivores, whiting, porpoises, seals | yes |
| whiting | zooplankton, krill, mobile epifauna, planktivores, benthivores (J) | benthivores, water column, porpoises, seals | mackerel | yes |
| benthivorous_fish | infauna, mobile epifauna, Nephrops, krill, planktivores, whiting | water column, whiting (on J), porpoises, seals | mackerel | yes |
| predatory_fish (water column) | zooplankton, krill, planktivores, benthivores, whiting, mackerel | porpoises, seals | - | yes |

48 active pairs, 5 of them self-observation. **The 5 self pairs are inert in the code [F, code]**:
`state.py:316-328` drops the observer's own index from `obs_others_idx`, and the own biomass `B_own` is
always present in the centre and in every neighbour (`state.py:353-361`); the own juvenile channel is added
whatever the switch says (`obs_juvenile_idx` includes `own_idx`). Self pairs are left out of the table below.

## 1. Senses and ranges, written once per cue class

Settled elsewhere (cited, not redone):
- **Vision**: metres. R_nv 0.1-0.3 m in darkness, R_sat <= 2 predator lengths in light; no gadoid reactive
  distance measured (research_notes/"Synregel i vattenpelaren", "En synregel for fiskparen") [F/J].
  A whole school may be seen a little farther in bright surface water, but at the box's 100-450 m beds it is
  dark (research_notes/"Ljus under 50 m") [J].
- **Lateral line**: about one to two body lengths [J, textbook; not reviewed here].
- **Smell**: predator cues cm-m, minutes-hours, downstream filaments only; FOOD plumes are followed up-current
  by cod from "several hundred m" (700 m cited), cod and haddock approach baits upstream; a cue from the
  surface spreads 1-21 m vertically per tick, never to a 100-300 m bed (research_notes/"Lukt och
  rovdjursundvikande" E1-E29; "Sal och tumlare golv" D3) [F/A/J].

New here: **hearing**, the only sense that could reach a neighbour cell.

### 1.1 Audiograms of the members (detection, not startle)

| member (group) | range / best | method | grade, read | source |
|---|---|---|---|---|
| Atlantic herring (planktivores 0.55) | responses 30-4000 Hz (to 8 kHz at high level); flat low threshold -20 to -25 dB re 1 ubar (= 75-80 dB re 1 uPa) at 30-1200 Hz, ~+20 dB re 1 ubar (120 dB re 1 uPa) at 3 kHz, +35 at 4 kHz | medullary single-unit recordings, tentative audiogram | [F] abs | Enger 1967, Comp Biochem Physiol 22:527 |
| Pacific herring (proxy) | 100-5000 Hz, best 200-500 Hz; NO response to 40/60/80 kHz at 172/179/185 dB re 1 uPa | ABR, n 7 | [F] full | Mann, Popper & Wilson 2005, Biol Lett 1:158, doi 10.1098/rsbl.2004.0241 |
| Atlantic herring | 50 % startle threshold at 4 kHz, none reported above | tank, startle (not hearing) | [F] abs | Kastelein et al. 2008, Mar Environ Res 65:369, doi 10.1016/j.marenvres.2008.01.001 |
| sprat, Norway pout, blue whiting | no audiogram found. Sprat: clupeid prootic bulla, assume herring-like [J]. Pout and blue whiting: gadoids with a swim bladder, assume cod-like [J] | - | [J] | - |
| "pout" in Kastelein 2008 | startle 0.1-0.25 kHz | startle | [F] abs | species is probably Trisopterus luscus (bib/pouting), NOT Norway pout T. esmarkii - unverified (only the common name was seen) |
| Atlantic cod (benthivores 0.31) | 30-470 Hz, best 60-310 Hz, thresholds -18 to -26 dB re 1 ubar (= 74-82 dB re 1 uPa), ambient-noise limited at sea | conditioned (cardiac), field, mid-water | [F] abs | Chapman & Hawkins 1973, J Comp Physiol 85:147, doi 10.1007/BF00696473; masking: Hawkins & Chapman 1975, doi 10.1007/BF00617122 |
| cod, ultrasound | 38 kHz pulses detected only at ~194 dB re 1 uPa (possibly not via the ear); no heart-rate, flight or impairment response to clicks at 204-213 dB | conditioned / unconditioned | [F] abs | Astrup & Mohl 1993, JEB 182:71; Schack, Malte & Madsen 2008, JEB 211:2079, doi 10.1242/jeb.015081 ("most probably play no ecophysiological role in wild cod") |
| haddock (benthivores 0.39) | masking by sea noise confirmed as for cod; no audiogram values seen | field | [F] abs | Chapman 1973, Helgol wiss Meeresunters, doi 10.1007/BF01609527 |
| dab, plaice (benthivores, flatfish ~0.24) | particle-motion hearing, upper limit ~200 Hz, plaice down to 0.1 Hz; narrower and less sensitive than cod | field, conditioned | [F] abs | Chapman & Sand 1974, Comp Biochem Physiol 47A:371; Karlsen 1992, JEB 171:173 |
| pollack (water column, minor) | no 50 % startle reached at 0.1-64 kHz | startle | [F] abs | Kastelein 2008 |
| saithe (water column 0.80), whiting | no audiogram found; gadoids with swim bladder, assume cod-like (< ~500 Hz) | - | [J] | - |
| Atlantic mackerel (0.94) | no audiogram found; no swim bladder, so particle-motion only, assume flatfish-like (low, < ~300 Hz, insensitive) | - | [J] | null result of search |
| horse mackerel (0.06) | startle 0.1-2 kHz | startle | [F] abs | Kastelein 2008 |
| spurdog (water column ~0.1) | sharks: 40-~800 Hz, particle motion; attracted from "large distances" to erratic low-frequency pulsed sound (struggling prey) | review | [F] abs | Myrberg 2001, Env Biol Fish 60:31, doi 10.1023/A:1007647021634 |
| hake, anglerfish, ling | nothing found | - | gap | - |

### 1.2 What the model's targets emit, and how far it carries

Range arithmetic [J]: spherical spreading (20 log r, the conservative end; cylindrical spreading beyond the
water depth would give longer ranges), threshold ~78 dB re 1 uPa for a gadoid or herring at its best band
(ambient-limited, Chapman & Hawkins 1973). Skagerrak ambient noise is shipping-dominated, which would shorten
every range below [J, not measured here].

| emitter (target) | signal | source level | who can hear it | range [J] | grade |
|---|---|---|---|---|---|
| harbour porpoise, echolocation | NBHF clicks 130-140 kHz, 44-175 us | 178-205 dB pp (mean 191) | no model fish (all < 5 kHz) | none | [F] Villadsgaard et al. 2007 (abs, via "Sal och tumlare golv" E11) |
| harbour porpoise, LF by-product | weak ~2 kHz component, 59 dB below the HF peak; some earlier "LF components" were tape-recorder artifacts | ~130 dB at 1 m if SL 191 [J] | herring only (gadoids, mackerel, flatfish stop < 0.5 kHz) | herring threshold at 2 kHz ~100-110 dB re 1 uPa (between Enger's 1.2 and 3 kHz points) gives 10-40 m; a 0.1 ms click is far shorter than a fish's integration time, so less [J] | [F] Hansen, Wahlberg & Madsen 2008, JASA 124:4059 (abs; DOI not verified); arithmetic [J] |
| harbour seal, male roar | 40-500 Hz (< 4 kHz), < 20 s, adult males, breeding season (Moray Firth 1 Jul-12 Aug) | 144 dB (129-149) re 1 uPa @ 1 m; 153 dB rms in 40-500 Hz (Alaska) | herring, gadoids (in band), mackerel and flatfish maybe at short range | 0.35-3.5 km by spreading (central ~2 km) | SL [F] abs (Matthews et al., PMC7211458; Van Parijs & Kovacs 2002, doi 10.1139/z02-088); range [J]; **no study of any fish responding to a seal roar found** |
| harbour seal, otherwise | "among the least vocal pinnipeds"; swimming and breathing sounds unmeasured as fish cues | - | - | none known | [F] abs / gap |
| herring (prey), gas-release chirp / FRT | pulsed chirp 3-4 kHz, 32-133 ms; FRT bursts 1.7-22 kHz, more per capita at high density | 73 dB rms @ 1 m (one fish) | not gadoids or mackerel (out of band); seals and porpoises yes | a school of 10^4 fish emitting together would add ~40 dB [J], still out of the fish predators' band | [F] abs: Wahlberg & Westerberg 2003, Aquat Living Resour 16:271, doi 10.1016/S0990-7440(03)00017-2; Wilson, Batty & Dill 2004, Proc R Soc B 271 Suppl 3:S95 |
| cod / haddock (spawning) | grunts and knocks 50-500 Hz, mainly Jan-Apr spawning | cod 123 dB @ 1 m (median, sea cage) | cod, haddock, herring | ~100-300 m (123-78 dB = 45 dB -> ~180 m) | SL [F] abs (Bioacoustics 2024 cod grunt study); haddock calls "may mediate aggregation at > 10 m", ranges cut by vessel noise (Stanley, Van Parijs & Hatch 2017, Sci Rep, doi 10.1038/s41598-017-14743-9, abs); range [J] |
| killer whale feeding sounds (not a model group) | low-frequency tail slaps / calls | - | herring | overwintering herring moved vertically and horizontally on playback; sonar 1-2 and 6-7 kHz at 127-209 dB caused no escape | [F] abs: Doksaeter et al. 2009, JASA 125:554, doi 10.1121/1.3021301 |

### 1.3 The clupeid ultrasound dispute (per member)

- For ultrasound: Alosinae (American shad, Gulf menhaden, allis shad) detect it and flee from simulated
  odontocete clicks [F] (Mann et al. 2001, JASA 109:3048, doi 10.1121/1.1368406, abs; Wilson et al. 2011,
  allis shad, abs, via "Sal och tumlare golv" E27). Pacific herring changed behaviour to simulated echolocation
  clicks [F] (Wilson & Dill 2002, CJFAS 59:542, doi 10.1139/f02-029; not read, via Mann 2005). Pingers reduced
  Atlantic herring catches in one study, not in two others [F, as cited by Mann 2005; originals not read].
- Against: bay anchovy, scaled and Spanish sardine hear only to ~4 kHz (Mann et al. 2001); Pacific herring show
  no ABR to 40-80 kHz at up to 185 dB (Mann et al. 2005, full). Mann 2005: the Wilson & Dill stimuli were
  broadband, "<4 kHz up to ultrasonic", and "it was not clear whether the herring were responding to the
  lower-frequency components or to the ultrasound" (full). Ultrasound detection seems limited to Alosinae.
- Per member: **Atlantic herring** - Enger's neural data stop at 4 kHz (8 kHz at high level); startle at
  4 kHz and not above (Kastelein); no test with real porpoise clicks found. **Sprat** - nothing measured; a
  clupeine, not an alosine, so no ultrasound expected [J]. **Norway pout, blue whiting, cod, haddock, whiting,
  saithe** - gadoids; cod has no ecologically relevant ultrasound sense (Schack 2008) [F for cod, J for the
  rest]. **Mackerel** - no swim bladder, no ultrasound expected [J].
- Crux: a real porpoise click carries only a weak ~2 kHz by-product (-59 dB, Hansen et al. 2008). A herring
  might hear it at tens of metres at most [J]; no other model fish can. Verdict on the claim in
  "Sal och tumlare golv" ("no model fish hears porpoise clicks"): **holds at the model's scales**, with the
  nuance that herring could in principle hear the low-frequency by-product at very short range [F/J].

### 1.4 Checked: "at km scale smell and sound carry the blend"

Several project notes rest on the judgement that at the observation's km scale the blend is carried by smell
and sound, "schools are heard at hundreds of m" ([J, not reviewed] in research_notes/"Gom och sinnen" section 3;
also used in "Planktivorernas golv", "Vitlingens golv", "Bottenfiskens och vattenpelarens golv",
"Makrillens antipredatorbeteende"). Checked here, it **does not hold for fish targets**: herring sounds are
3-4 kHz at ~73 dB @ 1 m, out of the band of every gadoid and of mackerel; no sound of saithe, whiting,
mackerel, hake or flatfish usable as a cue was found; the only fish sound reaching hundreds of m is gadoid
spawning grunts (Jan-Apr). Porpoise clicks are inaudible to all model fish (a herring might hear the ~2 kHz
by-product within tens of m). Smell reaches hundreds of m only as food plumes, downstream. So at 1 km the
blend reduces to (a) downstream food odour for bed feeders and (b) seasonal male seal roars, for which no fish
response is documented [F/J, section 1.2]. Those notes' floor decisions concern what a hide cuts, not range;
whether this changes any of them is for the user.

## 2. Movement per 6 h (per observer, member-weighted)

Unit: 0.1 m/s = 2.2 km per 6 h. "Path" = distance swum; "net" = start-to-end displacement. A 3x3 km
neighbourhood is crossed by ~1.5 km of net movement from the cell centre.

| observer (members) | speed and displacement | per 6 h | home range / tortuosity | crosses into neighbours? | grade, sources |
|---|---|---|---|---|---|
| planktivores: herring 0.55 | gross 0.7-0.9 m/s; per-school net 0.5-0.66 m/s over minutes; population net 0.10-0.15 m/s; overwintering 0.1-0.3 m/s; Ums 4.1 BL/s | path 15-20 km; net 2-11 km; winter 2-6 km path | very tortuous: status change every 5.5 min, split/join every 35 min; deep tight schools by day, disperse at night | yes, most ticks | [F] full/abs via research_notes/"Nettoforflyttning per funktionsgrupp"/herring.md (Misund 1997; Kvamme 2000; Pitcher 1996; Zedel 2003); He & Wardle 1988, J Fish Biol 33:255 (abs) |
| Norway pout 0.35 | no telemetry or speed found; 15-20 cm at 1-2 BL/s would be 0.15-0.4 m/s [J] | path 3-9 km [J] | within 1.5 m of the bed by day (Albert 1994, via "Bottenfiskens hojdprofil") | probably [J] | gap |
| blue whiting, sprat (~0.10) | nothing measured; sprat 10-14 cm at 1-3 BL/s = 0.1-0.4 m/s [J] | 2-9 km [J] | DVM | probably [J] | gap |
| mackerel 0.94 | 1.33 m/s through water (1.44 over ground), Norwegian Sea Jul-Aug 2010, multibeam sonar; ~3.8 BL/s in ~250 tracked schools; 0-6 m/s over ground (63 schools, >= 30 s tracks); Ums 3.5 BL/s (33 cm); obligate swimmer | path ~29 km | "did not swim in a straight course, but often changed direction, suggesting active feeding"; schools swim with the current | yes, every tick | [F] Diaz et al. 2013, ICES CM 2013/N:05 (full); Nottestad 2016, doi 10.3389/conf.FMARS.2016.05.00095 (abs); Godo et al. 2004, ICES JMS 61:1093, doi 10.1016/j.icesjms.2004.06.009 (abs); He & Wardle 1988 (abs) |
| horse mackerel 0.06 | no field speed; jack mackerel (T. symmetricus, 10-12 cm) sustained 8.4 BL/s for 6 h in the lab | - | - | probably [J] | [F, other species] Hunter 1971 (abs) |
| whiting | no telemetry, DST or speed found; conventional tags: < 110 km over months-years | path 2-8 km if 0.1-0.35 m/s [J] | benthopelagic | unknown [J] | gap; gadoids.md |
| benthivores: cod 0.31 | through water 0.2-0.6 m/s directed (0.3-0.9 BL/s); mostly < 1 BL/s (17 s fixes); smaller cod move more, cold winters less | net while resident 0.1-0.5 km; directed bouts 4-11 km | 95 % MCP 1-25 ha (median 5 ha) Skagerrak fjord; weekly 95 % KUD 0.06 +- 0.06 km2; > 55 % of North Sea DST days < 2 km/day; winter movement 100s of m | **no** in most ticks (home range << 1 km2) | [F] gadoids.md (Kristensen 2021; Righton 2008; Arnold 1994; Staveley 2019); Villegas-Rios et al. 2018, J Anim Ecol 87:1309, doi 10.1111/1365-2656.12872 (full); Freitas et al. 2015, Ecol Evol 5:2070 (PMC4449760, full); Ferno et al. 2011, Mar Biol Res 7:310, doi 10.1080/17451000.2010.492223 (abs) |
| haddock 0.39 | no telemetry found; lab maximum sustainable speed measured in an annular tank (16-40 cm) | - | large haddock off the bed by day | unknown [J] | gap; Breen et al. 2004 ICES JMS (abs, values garbled in the index) |
| flatfish ~0.24 | plaice resident around wind-farm foundations (1.37 km2 array); tidal-stream transport irrelevant in the near-tideless Skagerrak [J] | - | bed-bound, often buried | **no** [J] | [F] abs (wind-farm telemetry); gadoids.md |
| water column: saithe 0.80 | mean 1.05 m/s (median 0.93), 0.72 by day / 1.08 at night (25-29 cm, hydroacoustic); Ums 3.5 BL/s at 25 cm, 2.2 at 50 cm; fjord saithe tripped between farms 1.6-4.7 km apart, 15 of 24 fish 2-21 times in 3 months, 8-10 h/day at a farm; Icelandic DSTs: less movement at night and in winter | path 16-23 km | schooling, ranging | yes, most ticks | [F] Pedersen 2001 (via gadoids.md); He & Wardle 1988; Uglem et al. 2009, MEPS 384:251, doi 10.3354/meps08001 (abs); Armannsson & Jonsson 2012, ICES JMS 69:1372, doi 10.1093/icesjms/fss121 (abs) |
| spurdog ~0.10 | no km/day found (US pop-up tags: seasonal, local; Loch Etive residency) | - | - | unknown | gap |
| anglerfish ~0.03 | ~95 % of hourly DST records show < 5 m vertical change; more active at night and in winter; 82 % of summer recaptures < 30 km | ~0 | ambush, sedentary | **no** | [F] Ofstad et al. 2022, Front Mar Sci 9:823066, doi 10.3389/fmars.2022.823066 (full) |
| hake, ling | hake DVM (DST pilot, de Pontual 2012); nothing horizontal | - | - | unknown | gap |

What reading B buys [J]. The area an animal actually sensed in a tick is path x 2R, R the sense range. With
R = 1-20 m (vision, lateral line, generous for a school seen in light): herring 15 km -> 0.03-0.6 km2,
saithe 20 km -> 0.04-0.8 km2, mackerel 29 km -> 0.06-1.2 km2, against 4 km2 of neighbour cells. So:
- **Diffuse targets** (zooplankton and krill layers, the bed's infauna and epifauna): a transect through a
  neighbour samples its density well if the patch scale is >= 1 km. Copepod and krill patches are reported
  at km to tens of km, and whale-krill association peaked at 5 km among nested 1-20 km scales [F abs,
  whale/krill: search records only]. B carries these for ranging observers.
- **Point targets** (fish schools, single mammals): a ranging fish meets only the schools or animals within
  metres of its track, a few % of the neighbour area. B carries "did I meet predators / prey schools out
  there", not a density. Weak, but not nothing.
- The model's cell holds many individuals and schools; if they spread over the neighbourhood during the tick,
  their pooled experience is larger than one track. Fish do not pool information between schools, so this is
  a statement about the group's biomass distribution, not about what any fish knows [J].
- **Resident observers** (cod, flatfish, anglerfish) do not leave the cell in most ticks: B gives nothing.

## 3. Field evidence at the 0.1-few km scale

- Prey finding, plankton: herring density tracks the ascending overwintering Calanus over the basin
  (Broms et al., abs); prespawning herring prefer zooplankton-rich water and fronts at survey (km) scale
  (Maravelias & Reid 1997, MEPS 147:1; Maravelias 2001, abs) [F, coarse scale]. Herring schools make side-to-side
  movements interpreted as searching gradients of food, temperature and current (Misund 1997, via herring.md)
  [F/J]. No study found of a fish detecting a plankton patch from 1 km away.
- Krill swarms: dense Meganyctiphanes swarms in the Norwegian Sea were "patrolled" by large piscivores hunting
  the planktivores that feed on them; "krill swarms govern small-scale patchiness of large piscivores"
  (Kaartvedt et al. 2005, MEPS 299:1, doi 10.3354/meps299001, abs) [F].
- Piscivores and fish prey: cod and haddock "stayed close to" herring schools in the Norwegian Sea, their attacks
  changed school shape without dispersing it; fast saithe made herring dive steeply (Pitcher et al. 1996, ICES
  JMS 53:449, doi 10.1006/jmsc.1996.0063, abs) [F]. Cod-capelin correlation negative at 2-3 km, positive at
  > 4-10 km (Rose & Leggett 1990, via "Lukt" E20) [F]. Porpoise distribution (satellite tags) and mackerel
  distribution in Skagerrak-Kattegat were explained by herring abundance from acoustic surveys (Sveegaard et al.
  2012, MEPS 468:245, doi 10.3354/meps09959, abs; resolution not seen) [F]. Herring and mackerel negatively
  correlated in the Norwegian Sea in July (Langoy et al. 2012, Mar Biol Res 8:442, doi
  10.1080/17451000.2011.642803, abs) [F, basin scale].
- Predator avoidance: herring schools attacked by killer whales became more circular and denser (Nottestad &
  Axelsen 1999, Can J Zool 77:1540, doi 10.1139/z99-124, abs) [F, school scale]; a wild-caught herring school
  of ~60 000 reacted most strongly to combined visual and water-displacement cues (PLoS ONE 2014, PMC3906054,
  abs) [F, metres]. Over 1971-2012, as grey seals rose > 10-fold, cod, white hake and thorny skate shifted to
  areas of lower seal predation risk while non-prey fishes did not (Swain, Benoit & Hammill 2015, J Anim Ecol
  84:1286, doi 10.1111/1365-2656.12391, abs) [F] - a decadal, population-scale redistribution, which may be
  differential mortality or habitat choice; it says nothing about a per-tick km-scale sense [J].
- Conspecifics: Atlantic herring shoals on Georges Bank formed by a wave of synchronised movement once a
  critical density was reached, imaged over a 100 km area (Makris et al. 2009, Science; press coverage only,
  DOI not verified) [F abs]. The self switch is inert anyway (section 0).
- Null results: no fish response to harbour seal sounds; no fish at all reacting to a porpoise or seal at km
  scale; no gadoid, mackerel or herring prey-odour or predator-sound detection range beyond the bait-plume
  hundreds of m; no Rindorf 2006 / Hinz 2005 per-km prey-predator analysis retrieved (Rindorf & Gislason 2005:
  whiting intake vs local prey density at five North Sea locations, abs only).

## 4. The table

**Superseded in part by section 7** (read summary.md for the current matrix): seal roars carry ~50-140 m
masked, not km, so every "partial, seasonal" nbr A entry for seals is **no** (7.4); spawning grunts carry
metres to ~20 m (7.4); mackerel -> planktivores nbr B is "partial" (Sveegaard's co-distribution is ~870 km2 cells,
7.2); whiting nbr B "uncertain, leaning no" (7.1, 7.3); water column nbr B member-weighted "partial-yes" with
saithe's site attachment (7.1).

Cue-class shorthand: V = vision (m), LL = lateral line (~1-2 BL), O = smell / taste, H = hearing.
"Same cell" means: within one tick the observer meets a share of the target in its own cell (encounter-scale
senses integrated over 6 h of activity). Neighbour A = sensing at 0.5-1.5 km. Neighbour B = the observer
ranged through the neighbours (section 2).

### 4.1 planktivorous_fish (ranging: B possible)

| target (role) | same cell | sense, range, grade | nbr A | sense, range, grade | nbr B | reason |
|---|---|---|---|---|---|---|
| zooplankton (prey) | yes | V + LL, cm; feeds in it all tick [F] | no | none reaches 1 km; gradient-following via temperature/fronts is a correlate, not a sense of the copepods [J] | yes | diffuse layer, patches >= km; herring path 15-20 km |
| krill (prey) | yes | V, R 0.1-0.9 m [F/J] | no | as above | yes | diffuse deep layer |
| benthivores (predator) | partial | V, LL, O at m; cod/haddock stay close to herring schools (Pitcher 1996) [F]; vertical overlap only by day near the bed (pout) or in deep herring layers [J] | no | O cm-m downstream; H only spawning grunts ~100-300 m, Jan-Apr [J] | partial | point targets; meets them only along its track |
| whiting (predator) | partial | V, LL, O at m [J] | no | as benthivores (whiting sounds unknown) | partial | point targets |
| water column (predator) | partial | V, LL at m; herring dive steeply from fast saithe (Pitcher 1996) [F] | no | no saithe sound found | partial | point targets; both range |
| mackerel (predator) | partial | V, LL at m [J] | no | mackerel silent as far as known [J] | partial | point targets |
| porpoises (predator) | partial | V, LL at m; H of the click by-product at <= tens of m, herring only [F/J] | no | H: clicks inaudible; LF by-product <= 40 m [F/J] | partial | single animals, metres |
| seals (predator) | partial | V, LL at m [J] | partial, seasonal | H: male roars 40-500 Hz audible ~0.4-3.5 km, Jul-Aug, males only; no fish response study [F SL, J range] | partial | as porpoises |

### 4.2 mackerel (ranging: B yes)

| target (role) | same cell | sense, range, grade | nbr A | sense, range, grade | nbr B | reason |
|---|---|---|---|---|---|---|
| zooplankton (prey) | yes | V, cm (daylight biter and filter) [F/J] | no | none | yes | diffuse, path ~29 km |
| krill (prey) | yes | V [J] | no | none | yes | diffuse |
| planktivores (prey) | yes | V m; LL; herring chirps out of band [F/J] | no | herring sounds at 3-4 kHz, mackerel hearing low and insensitive [J] | partial-yes | point targets, but mackerel co-distribute with herring at survey scale (Sveegaard 2012) [F] |
| water column (predator) | partial | V, LL at m [J] | no | none | partial | point targets |
| benthivores (none) | no | no link; mackerel at 0-50 m, benthivores at the bed: no vertical overlap [J] | no | - | no use | no food link |
| whiting (none) | no | no link (competitor for planktivores, krill, zooplankton, whose own channels carry the depletion) [J] | no | - | no use | no food link |
| porpoises (none) | no | no booked link; clicks inaudible [F/J] | no | - | no use | no food link |
| seals (none) | no | no booked link [J] | no | (roar audible only at short range for a bladderless fish [J]) | no use | no food link |

### 4.3 whiting (ranging unknown: B uncertain)

| target (role) | same cell | sense, range, grade | nbr A | sense, range, grade | nbr B | reason |
|---|---|---|---|---|---|---|
| zooplankton (prey) | yes | V cm [J] | no | none | uncertain | whiting movement unmeasured |
| krill (prey) | yes | V [J] | no | none | uncertain | as above |
| mobile epifauna (prey) | yes | V, O, LL cm-m [J] | partial | O: food odour plumes hundreds of m, downstream only (cod/haddock baits; whiting untested) [F/J] | uncertain | diffuse bed field |
| planktivores (prey) | yes | V m; prey sounds out of band [J] | no | none | uncertain | point targets |
| benthivores (prey J and predator) | partial | V, LL, O at m; whiting avoids a predator stimulus in a tank (Atkinson 2004, "Lukt" E4) [F/A] | no | O cm-m; H spawning grunts ~100-300 m Jan-Apr [J] | uncertain | point targets |
| water column (predator) | partial | V, LL at m [J] | no | none | uncertain | point targets |
| porpoises (predator) | partial | V, LL at m; clicks inaudible [F]; porpoises reach the bed only over < 130-150 m beds ("Sal och tumlare golv") | no | none | uncertain | single animals |
| seals (predator) | partial | V, LL, O at m; seals forage on the bed [F] | partial, seasonal | H: male roars, as 4.1 [J] | uncertain | single animals |
| mackerel (none) | no | no link; mackerel 0-50 m, whiting near the bed [J] | no | - | no use | no food link |

### 4.4 benthivorous_fish (resident: B no)

| target (role) | same cell | sense, range, grade | nbr A | sense, range, grade | nbr B | reason |
|---|---|---|---|---|---|---|
| infauna (prey) | yes | O, taste, touch cm (buried bodies found by cue; "Exponering av bottenbyten") [F/J] | partial | O plume hundreds of m downstream, for strong sources (baits); diffuse infauna odour unmeasured [F/J] | no | cod home range 0.05-0.08 km2, flatfish bed-bound |
| mobile epifauna (prey) | yes | V, O, LL cm-m [J] | partial | as infauna [F/J] | no | resident |
| Nephrops (prey) | yes | O, V at the burrow [J] | partial | as infauna [J] | no | resident |
| krill (prey) | yes | V; meets the layer only over beds < 150 m [F/J] | no | none | no | resident |
| planktivores (prey) | partial | V, LL m; cod and haddock stay close to herring schools (Pitcher 1996) [F]; overlap only when prey are near the bed [J] | no | H: herring sounds out of the cod band [F/J] | no | resident; Rose & Leggett: cod-capelin negative at 2-3 km, positive > 4-10 km [F] |
| whiting (prey and predator of J) | yes | V, LL, O m; shared bed layer [J] | no | O cm-m; whiting sounds unknown | no | resident |
| water column (predator) | partial | V, LL, O at m; saithe near the bed 0.125 of the time ("Sejens hojd over botten") [J] | no | O cm-m downstream; spurdog/saithe sounds unknown | no | resident |
| porpoises (predator) | partial | V, LL at m; clicks inaudible to gadoids and flatfish [F]; meet only over < 130-150 m beds | no | none [F/J] | no | resident |
| seals (predator) | partial | V, LL, O m; seals take buried flatfish at 80 % success (Bowen 2002, "Sal och tumlare golv" E15) [F] | partial, seasonal | H: male roars 40-500 Hz in the cod band, ~0.4-3.5 km [J]; flatfish less sensitive [F] | no | resident; Swain 2015: cod shifted away from grey-seal risk over decades [F] |
| mackerel (none) | no | no link, no vertical overlap [J] | no | - | no use | no food link |

### 4.5 predatory_fish / water column (saithe-weighted: B yes; anglerfish no)

| target (role) | same cell | sense, range, grade | nbr A | sense, range, grade | nbr B | reason |
|---|---|---|---|---|---|---|
| zooplankton (prey) | yes | V cm [J] | no | none | yes | saithe path 16-23 km |
| krill (prey) | yes | V; piscivores patrol krill swarms (Kaartvedt 2005) [F] | no | none | yes | diffuse layer |
| planktivores (prey) | yes | V, LL m; fast saithe attacks on herring schools (Pitcher 1996) [F] | no | herring sounds out of the gadoid band [F/J]; spurdog: low-frequency pulses of struggling prey, "large distances", untested for these prey [F abs] | partial-yes | point targets; ranging |
| benthivores (prey) | yes | V, LL m; saithe near the bed ~0.125 of the time [J] | no | H: cod/haddock spawning grunts ~100-300 m, Jan-Apr [J] | partial | point targets |
| whiting (prey) | yes | V, LL m [J] | no | none | partial | point targets |
| mackerel (prey) | partial | V m; vertical overlap only 0.02 of saithe's day time ("Overlapp i fiskparen") [F/J] | no | none | partial | point targets |
| porpoises (predator) | partial | V, LL m; clicks inaudible [F] | no | none | partial | single animals |
| seals (predator) | partial | V, LL m [J] | partial, seasonal | H: male roars [J] | partial | single animals |

## 5. Pairs with no cue or no use (candidates for "off")

1. **The five self pairs** (`<fish>_observes_<same fish>`): inert in the code, the own biomass is always in
   centre and neighbours [F, code]. Switching them changes nothing.
2. **No food link (6 pairs)**: mackerel -> benthivores, mackerel -> whiting, mackerel -> porpoises,
   mackerel -> seals, benthivores -> mackerel, whiting -> mackerel. No booked predation either way; the
   mackerel and bed groups also do not share a layer [J]. Competition (shared prey) is already carried by the
   observer's own prey channels [J].
3. **Neighbour channels of resident observers under reading B**: every benthivore pair (cod, flatfish,
   anglerfish are sub-cell residents) [F]. Under reading A the benthivores' neighbour channel has a cue only
   for benthic prey (food odour, downstream only, hundreds of m) and, seasonally, for seals (male roars) [F/J].
4. **Neighbour channel for porpoises under reading A, every fish observer**: no sense reaches (clicks
   inaudible; LF by-product <= tens of m, herring only) [F/J].
5. **Neighbour channel for every fish predator target under reading A**: predator odour cm-m, no predator
   fish sound found outside gadoid spawning grunts [F/J].

What no pair loses on this evidence: the **same-cell** channel of every prey pair (the observer feeds in it
all tick) and, as "partial", of every predator pair (encounters at metres integrated over 6 h).

## 6. Open points and disputed evidence

1. **Clupeid ultrasound**: herring and sprat show no ultrasound hearing (Mann 2001 on clupeines, Mann 2005 on
   Pacific herring, full), but Wilson & Dill 2002 behavioural responses to simulated clicks remain unexplained
   in detail (broadband stimuli); pinger effects on Atlantic herring catches are mixed. No test of Atlantic
   herring or sprat with real porpoise clicks exists. Real clicks carry a -59 dB ~2 kHz by-product; whether a
   herring hears it at > 10 m is untested [J].
2. **Seal roars**: in the band of herring and gadoids and loud enough for km ranges by spreading, but male,
   seasonal (summer breeding, probably near haul-outs east of the box rather than offshore [J]), and no study
   shows a fish reacting. Ambient shipping noise in the Skagerrak would shorten the range [J].
3. **Kastelein 2008 "pout"**: probably Trisopterus luscus, not Norway pout; its startle data do not transfer
   to the planktivores' pout without that check.
4. **Audiograms missing** for mackerel, sprat, Norway pout, blue whiting, whiting, saithe, hake, ling,
   anglerfish; all ranges for them are family-level judgements.
5. **Movement gaps**: whiting, haddock, Norway pout, blue whiting, sprat, hake, ling, spurdog have no
   per-hour displacement; whiting's B column is therefore "uncertain". Saithe's day/night contrast differs
   between sources (Pedersen 2001 faster at night; Armannsson & Jonsson 2012 less movement at night).
6. **Direction**: food plumes reach only downstream; the model's four neighbours are isotropic and carry no
   current. A food-odour neighbour channel would in reality be one neighbour (or none) per tick [J].
7. **Reading B and point targets**: a ranging fish senses a few % of the neighbour area (section 2), so for
   schools and mammals B means an encounter record, not a density. Whether the group-level biomass in a
   cell can stand for pooled experience is a modelling question, not a literature one [J].
8. **Swain et al. 2015**: the one field result of fish distributions tracking a mammal's predation risk is
   decadal and could be differential mortality; read as abstract only.
9. **Not retrieved**: Rindorf 2006 and Hinz 2005 (named in "Lukt" gaps; a Hinz 2005 haddock-benthos MEPS paper
   could not be found under that name), Langoy 2012 full text, Sveegaard 2012 resolution, Makris 2009 original,
   Hansen et al. 2008 full text.

## 7. Addendum after the resumed searches (2026-11-13)

### 7.1 Movement per 6 h (corrects section 2)

Unit 0.1 m/s = 2.16 km per 6 h; leaving the own cell needs > 0.5 km net, crossing the 3x3 ~1.5 km.

- **Whiting and Norway pout (measured, replaces the [J] rows)**: Onsrud, Kaartvedt & Breien 2005, CJFAS
  62:1822, doi 10.1139/f05-090 [F, abs]: split-beam target tracking at a 120 m site, deep fish "mainly
  Norway pout and whiting" (trawl) swam 14-16 cm/s by day, modes ~10-11 cm/s at night (switch from visual to
  non-visual feeding); "most tracks were short, but some long tracks revealed complex and cyclic swimming
  paths". Per 6 h path 3.0-3.5 km by day, 2.2-2.4 km at night [J]; 3D instantaneous speeds, cyclic, so net
  displacement well below; the pout/whiting split is [J]. Whiting U_opt 1.0 BL/s, cost of transport 0.15 J/N/m
  (Steinhausen, Steffensen & Andersen 2005, Mar Biol 148:197, doi 10.1007/s00227-005-0055-9) [F, snippet]; 13 cm
  whiting max sustained 3-3.5 BL/s (Hammer & Schwarz 1994, ICES CM 1994/G:41) [F]. **Whiting nbr B: "uncertain,
  leaning no"** (was "uncertain"); Norway pout likewise (was "probably").
- **Saithe day/night resolved [J]**: horizontal ranging is greater by day. Armannsson & Jonsson 2012 (full):
  51 DSTs, vertical activity highest in daylight, reduced at night in winter. Sarno, Glass & Smith 1994, J Fish
  Biol 45:811 [F, abs]: two saithe ranged over a whole reef by day, excursions to a reef ~250 m away, stayed on
  the main reef at night. Bjordal & Johnstone 1993, Bjordal & Skar 1992 (full): active by day, core areas, 1-7
  months at farm cages. Pedersen 2001's night speed is a ping-to-ping speed of 25-29 cm fish over seconds, not
  displacement. **Flag**: every coastal telemetry shows saithe attached to structures (hundreds of m; Uglem 2009
  farm trips 1.6-4.7 km); the "path 16-23 km, crosses most ticks" of section 2 rests on swimming speed, and net 6 h
  displacement of offshore saithe is unmeasured. Saithe nbr B stays "yes" on path length, with that caveat.
- **Spurdog**: Sulikowski et al. 2010, MEPS 418:249, doi 10.3354/meps08821 [F, abs]: PSAT mean horizontal speed
  1.36-2.49 km/h (8-15 km per 6 h, light-geolocation track means, an upper bound [J]); Junge et al. 2025, J Fish
  Biol 106, doi 10.1111/jfb.16000 [F, full]: 84 % popped up within 40 km after ~8 months; Klocker et al. 2024,
  Mov Ecol 12:62, doi 10.1186/s40462-024-00498-2 [F, full]: DVM ~138 m (198.5 m day, 60 m night median); Loch Etive
  year-round residency (doi 10.1007/s10641-024-01640-0, abs). Site-attached at tens of km; 6 h displacement a gap.
- **Hake**: de Pontual et al. 2013, ICES JMS 70:1416, doi 10.1093/icesjms/fst102 [F, full, papers/]: 1199 of 27 690
  recaptured, "most fish were recovered near their release locations" - resident [F]; per 6 h rate a gap.
- **Ling**: Lokkeborg, Skajaa & Ferno 2000, JEMBE 247:195, doi 10.1016/S0022-0981(00)00148-9 [F, abs]: home range,
  ~65 % of the time in a small core area, less active at night and more restricted than cod: resident, **no**.
- **Haddock**: no field telemetry found; Breen et al. 2004 tank only. Proxy Onsrud's deep gadoids 0.10-0.16 m/s [J].
- **Blue whiting, sprat, horse mackerel**: horizontal movement still a gap (Johnsen & Godo 2007 doi
  10.1093/icesjms/fsm110: more dispersed and higher at night; Solberg & Kaartvedt 2014/2017 vertical only).
- **Plaice**: Griffiths et al. 2018, Ecol Evol 8:7031, doi 10.1002/ece3.4223 [F, full]: HMM steps 6.4-12.9 km/day
  "resident", 12.9-19.5 migrating, between daily tidal geolocations (inflated as Righton's Table 1b, tidal southern
  North Sea); does not contradict sub-cell residency from acoustic positioning [J].
- **Cod**: Freitas et al. 2015 hourly distances only in its Dryad set (doi 10.5061/dryad.8m65n, not opened).

Net effect on the water column's B column: saithe yes (caveat above); spurdog probably; hake, ling and
anglerfish no. A member-weighted "partial-yes" is the honest grade [J].

### 7.2 Field evidence at 0.1 to a few km (adds to and corrects section 3)

Retrieved items from 6.9:
- **Rindorf, Gislason & Lewy 2006**, MEPS 325:243, doi 10.3354/meps325243 [F, full, papers/]: > 36 000 cod and
  whiting stomachs vs IBTS prey per 60 x 60 nmi and quarter; negative switching. Scale chosen because haul-level
  comparison is "greatly affected by random noise"; the authors: it "cannot be used to determine the underlying
  biological mechanisms acting at the local scale" and is "far greater than the perceptive abilities of the
  predators". A null by design for km scale.
- **Rindorf & Gislason 2005**, JEMBE 324:1, doi 10.1016/j.jembe.2005.04.013 [F, full, papers/]: five locations tens
  to hundreds of km apart, hauls every 4 h for 48-72 h; "no aggregative response of the predators was detected"
  (n = 5 location means; prey vs whiting negative in 7 of 9).
- **"Hinz 2005"**: no haddock-benthos MEPS paper exists. Candidates: Hinz, Kroncke & Ehrich 2005, J Fish Biol 67,
  doi 10.1111/j.0022-1112.2005.00918.x (dab, one research box, prey choice follows prey density) [F, abs];
  Hiddink, Jennings & Kaiser 2005, J Fish Biol 67 Suppl B:111 (haddock condition at 24 sites; no prey density,
  no km scale) [F, abs]. Neither speaks to km-scale finding.
- **Langoy et al. 2012**: abstract only (403); herring and mackerel tied to different water masses, low diet overlap
  even where they overlap [F, basin scale].
- **Sveegaard et al. 2012** (full): 0.25 x 0.5 deg cells, ~870 km2, 76 cells, porpoise fixes Jun-Aug 1998-2009,
  acoustic herring and mackerel averaged 2000-2009; "smaller-scale analysis impossible". **Correction to 4.2**:
  mackerel -> planktivores "co-distribute at survey scale" means ~30 km cells pooled over a decade, not km.
  Lead: Bailey & Thompson 2009, porpoise-habitat relations at 1-2 km grids, not at 4 km [S].
- **Makris et al. 2009**, Science 323:1734, doi 10.1126/science.1169441 [F, abs]. Follow-up **Pednekar, Jain,
  Godo & Makris 2024**, Commun Biol 7:1386, doi 10.1038/s42003-024-06952-6 [F, full, papers/]: Barents Sea, OAWRS
  over 75 km. Capelin reached critical density (1.4 /m2) ~10 min after dawn twilight and formed > 10 km shoals by
  density waves faster than the fish swim. Diffuse cod reached their own critical density (0.06 /m2) at the same
  sites and engulfed the capelin shoal as a wave an order of magnitude faster than cod swim. ~2.5 million cod ate
  ~10.6 million capelin in ~4 h. Mechanism: metre-scale "spheres of perception" chained through the group, not
  km-range sensing [F/J]. **The one measured km-scale predator convergence on prey within a 6 h tick**; supports
  reading B and collective transmission, not A. At the box's benthivore mean (1.53 t/km2, ~0.0015-0.005 fish/m2)
  that is 12-40x below the cod critical density: probably inactive at box mean densities [J].
- **Swain, Benoit & Hammill 2015**: still abstract only; tow-by-tow risk index over 75 000 km2, decadal. 6.8 stands.

New items:
- **Engelhard et al. 2008**, MEPS 369:213, doi 10.3354/meps07575 [F, full, papers/]: Dogger Bank, grid points
  ~5.5 x 3.3 km. Within grids whiting abundance and per-capita sandeel consumption rose with daytime sandeel
  numbers at grid points - an aggregative response at a few km. Sense not stated.
- **Temming et al. 2004**, J Fish Biol 64:1351, doi 10.1111/j.0022-1112.2004.00400.x [F, abs]: whiting aggregated
  "pronouncedly" where sandeel catches were highest within a ~30 km box; haddock less.
- **Kaiser & Spencer 1994**, MEPS 112:41, doi 10.3354/meps112041 [F, full, papers/]: after beam trawling, side-scan
  fish marks (presumed whiting shoals, within 1 m of the bed) 11 vs 1 over fished vs unfished ground, later x30 vs
  x11; whiting ate the damaged fauna. Aggregation within hours at hundreds of m [F]; odour is inferred [J].
- **Kaartvedt et al. 2005** now read in full (papers/10.3354_meps299001.pdf): krill swarms ~120 m x 50 m (~11 t)
  at 100-200 m, ~0.7 km apart; saithe 60-110 cm stacked up to ~100 m below most swarms. Patchiness at the 100 m
  swarm scale [F].
- **Fall et al. 2021**, Ecography, doi 10.1111/ecog.05473 [F, abs]: Barents cod stomachs at 2 km resolution; capelin
  depth mattered more than its abundance; cod density no effect.
- **Maravelias 2001**, Fish Oceanogr, doi 10.1046/j.1365-2419.2001.00172.x [F, abs]: finest unit 2.5 nmi (4.6 km);
  zooplankton and seabed determine herring aggregation, equally at all three scales.
- **Nottestad et al. 2002**, ICES JMS 59:393, doi 10.1006/jmsc.2001.1172 [F, abs]: fin whale attacks change herring
  school structure; no avoidance distance.
- **De Robertis & Handegard 2013**, ICES JMS 70:34, doi 10.1093/icesjms/fss155 [F, abs]: fish avoid vessels at
  ~100 m scale, noise the presumed stimulus (not a model target).

Nulls: no fish-zooplankton or fish-krill correlation finer than ~2-4.6 km; no model fish shown to detect a plankton
patch, a school or a predator from ~1 km; every km-scale aggregation found is an outcome (sense unstated, or metre
senses chained); no km-scale avoidance of seals or porpoises by prey fish; nothing on Norway pout.

### 7.3 Onsrud 2005 in full (refines the whiting / pout row of 7.1)

Onsrud, Kaartvedt & Breien 2005, doi 10.1139/f05-090 [F, full, papers/10.1139_f05-090.pdf]: Oslofjord
59 48'N 10 34'E, 120 m, November 2000, one diel cycle; currents 1-5 cm/s. Bottom transducer 8623 tracks, 2290 after
size and density filters; 80-110 m targets whiting and Norway pout by TS; whiting mostly 14-19 cm (juveniles).
Tracks mostly 10-20 pings at ~2 pings/s (5-10 s, ~0.5-1.6 m of path); speeds 3D but mostly horizontal by the
paths' horizontal net-to-gross ratio (qualitative); horizontal mode ~12 cm/s at night at the same site
(Klevjer & Kaartvedt 2003). Modes 14-16 cm/s by day, 10-11 at night (~1 and ~0.7 BL/s). Swimming direction
uniform in 65 % of the 1-2 h periods; one long track was a vertical spiral; after midnight fish sank out of the
layer "with little horizontal movement". Net displacement beyond seconds is not measured: the 2.2-3.5 km per 6 h
of 7.1 is an extrapolation of path, and uniform directions argue for small net movement [J]. Adult whiting
(25-35 cm) at 0.7-1 BL/s would swim 0.2-0.3 m/s, 4.5-6.5 km path per 6 h - extrapolation [J]. Verdict for whiting
and pout nbr B unchanged: **uncertain, leaning no**. Freitas 2015's Dryad file (Data.xlsx) needs a login: not
retrieved.

### 7.4 Hearing: corrections to 1.1-1.4 and the section 4 tables

Corrections (they change values above):
- **Ranges must be masked, not absolute.** Stanley, Van Parijs & Hatch 2017, Sci Rep, doi 10.1038/s41598-017-14743-9
  [F full]: r = 10^((SL - MSL - DT)/20), measured band ambient at three spawning sites (medians 88.7-102.3 dB re
  1 uPa), DT 15 dB (masked cod thresholds, Chapman 1973; Hawkins & Chapman 1975). Effective radii: cod 15.3 m
  (7.2-21.6) spring site, 2.7 m winter site; haddock 1.8-3.5 m. Cod grunt SL 127 dB (Nordeide & Kjellsby 1999,
  doi 10.1006/jmsc.1999.0473), f1 53 Hz; haddock knocks ~258 Hz, on 62 % of Jan-Mar days. **The ~100-300 m
  spawning-sound range of 1.2 is 10-100x too long: metres to ~20 m.**
- **Seal roars, recomputed the same way [J]**: SL 144-153 dB vs ~95 dB band ambient, DT 15: ~50-140 m; 160-450 m in
  a quiet 85 dB sea. **Hundreds of m at most, not 0.4-3.5 km.** The "partial, seasonal" nbr A entries for seals in
  4.1, 4.3, 4.4 and 4.5 become **no**. Still no study of a fish responding to a seal sound.
- **Skagerrak noise**: Tougaard, Ladegaard & Griffiths 2024, DCE fagligt notat 36 (Aarhus Univ., full, Danish):
  JOMOPANS median ship-noise excess 15-20 dB over natural background at 20-200 Hz in the Danish North Sea,
  highest on the route round Skagen, larger in summer [F, model + monitoring]. Every fish-hearing range in the
  box is cut accordingly [J].
- **Herring at 2 kHz ~96 dB re 1 uPa** (Enger 1967 via Nedwell 2004): porpoise LF by-product 132 - 96 = 36 dB ->
  ~60 m before the short-click penalty; still tens of m [J].

Audiogram additions (Nedwell, Edwards, Turnpenny & Gordon 2004, Subacoustech 534R0214, full, tabulating originals;
dB re 1 uPa): herring 75-79 at 30 Hz-1 kHz, 96 at 2 kHz, 136 at 4 kHz (n 36); cod best 75.3 at 160 Hz, 110 at
450 Hz; haddock best 80-81 at 110-310 Hz, 104 at 470 Hz (Chapman 1973, doi 10.1007/BF01609527); pollack best 80.8
at 160 Hz; ling best 80.8 at 200 Hz, 109 at 550 Hz (n 1); dab to ~260 Hz, particle motion. Sprat: no audiogram;
bulla and utricle respond to ~1 kHz, utricle threshold ~74 dB implied (Denton & Gray 1979, 1980; doi
10.1017/S0025315400046154, 10.1017/S002531540004039X) [F abs, J]. Horse mackerel proxy T. japonicus 70-3000 Hz, best
~1-1.5 kHz (Chung et al. 1995) [F abs]. Mackerel: no audiogram; wild schools responded to pile-driving impulses at
50 % level 163 dB pp, as did sprat (Hawkins, Roberts & Cheesman 2014, JASA 135:3101, doi 10.1121/1.4870697) [F abs].
(Nedwell's table wrongly gives Scomber scombrus a swim bladder.) Elasmobranchs: ~10 species, none above 1 kHz
(Parmentier et al. 2020, J Anat, doi 10.1111/joa.13212) [F full]; shark attraction to pulsed low-frequency sound
from tens to "a few hundred m" (Myrberg et al. 1969, 1972) [abs]; no Squalus audiogram. Kastelein 2008 "pout" is
Trisopterus luscus (confirmed; closes open point 3). Still no audiogram: mackerel, saithe, whiting, hake,
anglerfish, blue whiting, Norway pout.

Ultrasound dispute, additions: Kastelein et al. 2007, Mar Environ Res 64:160, doi 10.1016/j.marenvres.2006.12.012
[F full]: herring pinger record is one reduction (Kraus 1997) against four nulls (Trippel 1999, Culik 2001, Hughes
1999, Wilson & Dill 2002 tank); herring responded in the tank only to a ~10 kHz pinger with low harmonics, and the
authors trace responses to 0.5-5 kHz energy. Culik et al. 2001, MEPS 211:255, doi 10.3354/meps211255 [F full]: 25 407
Baltic herring in gillnets, PICE 20-160 kHz no effect, a 2.7 kHz pinger raised catches (attraction). Allis shad
paper is Wilson et al. 2008 (not 2011), JASA 124:EL243, doi 10.1121/1.2960899: threshold 161-167 dB pp at
70-120 kHz; a 191 dB click falls to that at ~20 m [J]. Hansen, Wahlberg & Madsen 2008 doi verified:
10.1121/1.2945154. Killer whale playbacks: Doksaeter 2009 (full) used 0.8-20 kHz calls at 150-160 dB rms; Sivle et
al. 2012, ICES JMS 69:1078, doi 10.1093/icesjms/fss080: herring dived to killer whale sounds at ~150 dB received,
not to sonar - reaction levels at metres to tens of m [J]. The verdict of 1.3 stands and is strengthened.

**Net effect on the section 4 tables**: under reading A the only remaining neighbour cue for any fish observer is
the downstream food-odour plume of bed feeders (benthivores and whiting on bed prey; directional, hundreds of m).
Every predator target is **no** under A.
