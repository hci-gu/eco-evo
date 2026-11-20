# What holds field piscivorous gadoids at 0.2-0.5 of Cmax? Literature on four mechanisms (2026-11-20)

Research subagent. Research only: nothing outside this folder was edited. Tags: [F] full text read, [A] abstract
only, [S] secondary (search summary or an earlier note), [D] derived here (arithmetic shown), [J] judgement.
Evidence rows: `tables/foraging_limits_raw.csv` (written by `scripts/make_foraging_limits_csv.py`).
Downloads: `data/<source>/`. Parser for the Ecosim table: `scripts/ewe_vuln_parse.py`, output
`tables/ewe_vuln_ns1991_stdout.txt`. OpenAlex helper: `scripts/oa_search.py`.

"Prior, cited" means the source is already in research_notes/"Födonivå och mättnad i fält"
(`tables/B_mechanisms_evidence.csv`) or "Mättnad hos bottenbyten" and was not re-read here.

Standing constraint from the brief (prior derived result): the gut buffers 1-4 days of meals. A short daily
feeding window alone therefore cannot hold a fish below ~0.6-0.9 of Cmax. A level near 0.3 needs a LOW intake
rate while foraging, or non-foraging periods lasting days.

## Headline

1. **The most discriminating new source is a natural experiment: saithe beside salmon farms (Dempster et al. 2011, [F]).**
   - Wild saithe with an energy-dense, easily caught subsidy hold **2.35x the stomach content** of saithe away
     from farms ([D] 20.2 g / 8.6 g).
   - Their share of empty stomachs halves (0.16 vs 0.31), and their liver index is 1.4-1.8x higher.
   - So natural saithe in summer are not held down by appetite, a full liver or gut capacity. Their intake is
     limited by acquisition: search, encounter or capture.
   - Farmed cod give the same signal: livers > 12 % of body weight on ad lib feed against 2-6 % wild ([S]).
2. **What that rules out, and what it leaves open.**
   - Ruled out: appetite or reserves as the cap (C, lipostatic form).
   - Too weak alone: a diel feeding window (A, time-budget form). The gut buffer covers it.
   - Not separated by this evidence: risk-restricted foraging (the foraging-arena form of A), per-prey capture
     limits (B) and patchiness at the km scale (D).
     - The farm removes search, capture failure and exposure all at once (Skilbrei & Otterå: farms give shelter
       as well as feed).
     - All three act as a low intake rate while foraging.
   - C (digestion) sets meal timing and the empty-stomach shares (hake: 0.47 empty), not the daily mean.
   - The only sized number for A remains the prior one: a predator cue cut juvenile cod's consumption to
     0.22-0.44 of control (Persson et al. 2012).
     - research_notes/"Mättnad hos bottenbyten" reads the field level as time/risk on that basis.
     - This note does not contradict that reading. It finds no adult-gadoid evidence for or against it.

## A. Foraging time limited by predation risk or other activities

**New evidence**
- **Armannsson & Jónsson 2012**, ICES J Mar Sci 69:1372-1381, doi:10.1093/icesjms/fss121 [A].
  - Hourly depth from 51 saithe DSTs off Iceland.
  - Hourly depth changes are smaller at night and in winter ("reduced activity").
  - Saithe can change depth fast (< 1 min).
  - The full text was blocked (OUP returned HTML), so no activity-share number.
- **Skilbrei & Otterå 2016**, ICES J Mar Sci 73:1186-1195, doi:10.1093/icesjms/fsv261 [A]. Acoustic tags on
  farm-associated saithe.
  - Reduced activity at night.
  - No DVM in 0.38 of observations; a daytime up or down move in 0.46.
  - Away from the farm in daytime in 0.16 of observations, "probably to feed elsewhere".
  - These are activity patterns, not a foraging-time share.
- **de Pontual, Jolivet, Bertignac & Fablet 2012**, J Fish Biol 81:728-734 (doi not checked) [S].
  - First DST evidence of hake DVM, with thermocline crossings and temperature shifts > 7 C.
  - Search summary only; no time budget.
- **Proxies for hake**:
  - **Juveniles** (Módica, Bozzano, Velasco, Albertelli & Olaso 2011, MEPS 440:177-189, doi:10.3354/meps09341 [A];
    9-12 cm, NW Mediterranean). Feeding starts in the early night and peaks in the early morning. There is a
    rest phase before sunset.
  - **Cape hakes** (Pillar SC et al. 1997, ICES J Mar Sci 54:485, doi:10.1006/jmsc.1996.0169 [A]). Little
    diel periodicity: up to 1/3 of the fed fish had fed recently at any hour.
  - Neither gives a fraction of the day spent feeding for adults.
- **Foraging-arena vulnerabilities, North Sea Ecosim**: Mackinson & Daskalov 2007, Cefas Sci Ser Tech Rep 142,
  Table 4.5, fitted 1991 model [F].
  - Each predator has ONE v on all its prey:

    | predator | v |
    |---|---|
    | adult whiting | 3 |
    | adult saithe | 6 |
    | hake | 3 |
    | adult cod, adult haddock | 6 |
    | juvenile cod | 4 |
    | juvenile saithe | 2 |
    | juvenile whiting | 1.1 |
    | juvenile haddock | 1 |

  - How the values were read: position-parsed from the text layer, then checked by eye on printed page 75
    (pdf page 77), e.g. the Norway pout row.
  - Initial values were set by reasoning: groups far below their historical biomass get a higher v, and juveniles
    and low-trophic-level feeders a low v.
  - The optimiser searched "for those species that we had good time series data for (i.e. mostly the MSVPA
    species) or were identified as being particularly sensitive". The report warns that "the model is being
    calibrated against another model" (MSVPA).
  - The report's own definition [F, appendix]:
    - v close to 1 is donor-controlled, with flow C = v B_prey, "so v can be interpreted as the maximum possible
      instantaneous mortality rate that a predator can cause on a prey".
    - High v (over 28) is fully top-down (Lotka-Volterra).
  - It is not a feeding level. **A conversion to a fraction of Cmax is not possible** [J].
  - Read qualitatively, the modellers placed adult gadoid predators away from strong arena (flux) limitation
    (v 3-6) and the juveniles at it (v ~1).
  - Weak, model-derived evidence.
- **State-dependent risk taking**: no new gadoid experiment found.
  - Prior, cited: a predator cue cut cod intake to 0.22-0.44 of control (Persson et al. 2012, MEPS 456:245,
    doi:10.3354/meps09752); Werner & Anholt 1993; Ahrens, Walters & Christensen 2012.
  - Counter-example with cod as the predator, not the prey: juvenile corkwing wrasse habituated and did not eat
    less (Norin et al. 2021, Behav Ecol Sociobiol 75:14, doi:10.1007/s00265-020-02947-5 [S]).
- **Predation on ADULT whiting, saithe and hake**: not sized here. An SMS "eaten weight" table in the repository
  has no age split and no checked unit, so it was not used. See research_notes/"Vitlingens predatorer". Gap.

**Verdict A, split.**
- **Diel time-budget form: weak.**
  - Diel rest and DVM exist (saithe, hake), but no source gives a foraging-time share for adults.
  - With the gut buffer, a one-night rest cannot hold intake near 0.3 of Cmax.
- **Risk / foraging-arena form: open.**
  - No source shows adult saithe, whiting or hake trading food for safety, and none rules it out.
  - The farm experiment (C) cannot separate it from B or D.
  - The only number is the prior juvenile-cod cue result (0.22-0.44 of control intake).
- **Size: unquantified for adults.**

## B. Per-prey-group intake plateaus

- **Prior, cited**:
  - Rindorf & Gislason 2005 [F]: plateaus of 0.06-0.17 of Wmax per prey group; total 0.14-0.35; fish prey
    taken at dawn and dusk.
  - Rindorf 2003, MEPS 249:265, doi:10.3354/meps249265 [A here]: bottom prey eaten at night; free-swimming prey
    eaten by day.
  - Onsrud et al. 2004: no day/night difference on krill.
  - Fall et al. 2021: cod eat capelin by its depth, not its abundance.
- **Rindorf A, Gislason H, Lewy P 2006**, "Prey switching of cod and whiting in the North Sea", MEPS 325:243-253,
  doi:10.3354/meps325243 (citation verified on Crossref; result from an abstract snippet [S]).
  - **Negative switching**: preference falls with the prey's relative density, strongest in cod of 40-100 cm.
  - This is what a per-prey plateau predicts: when intake on a prey saturates, its diet share falls as the prey
    gets abundant [J].
- **Kempf, Floeter & Temming 2008**, MEPS 367:295-308, doi:10.3354/meps07555 [F]. GAMs for cod and whiting at
  the population scale (ICES rectangles in roundfish areas 1-7; Schoener overlap across rectangles).
  - The prey's share of the stomach falls fast below a relative prey abundance of 0.1 and below a
    predator-prey overlap of 0.1. Above 0.1 overlap it reaches a plateau.
  - There is a "saturation effect" at high prey abundance.
  - This is **diet composition**, not feeding level: no Cmax number. It supports saturation per prey and an
    overlap-driven Type III at large scale.
- **Reilly et al. 2014**, ICES J Mar Sci 71:406-416, doi:10.1093/icesjms/fst137 [A]. "Gadoid fish predators"
  (species not named in the abstract): their biomass and
  "feeding performance" off the Firth of Forth do not depend on 0-group sandeel biomass.
- **Durant JM, Langangen Ø, Dolgov AV, Lindstrøm U et al. 2025**, MEPS, doi:10.3354/meps15078 [A]. In every fish predator the capelin share falls as krill
  abundance rises: krill substitutes for capelin. That is diet share, not intake.
- **Whether an abundant single prey such as krill can fill a gadoid to Cmax in the field: not found.**

**Verdict B: supported as phenomenology, but measured only once.**
- Only Rindorf & Gislason measure plateau levels as a fraction of Wmax: 0.06-0.17, up to 0.22 per prey group.
- The independent tests (negative switching; saturating diet share) are consistent with them but give no Cmax
  number.
- The mechanism the authors offer is a capture window at twilight for fish prey.
- **Best size**: the per-prey plateau of 0.06-0.22 Wmax, and a total of 0.14-0.35 from two to three prey groups.

## C. Digestion, meal dynamics and lipostatic regulation

- **Hake**: Guichet 1995, ICES J Mar Sci 52:21-31, doi:10.1016/1054-3139(95)80012-3 [A].
  - 8293 stomachs from the northern Bay of Biscay, 1991-92: **0.33 with food, 0.47 empty, 0.07 regurgitated,
    0.13 food remains**.
  - Empty is 0.47, or [D] 0.47 + 0.07 = 0.54 if the regurgitated count as empty. Regurgitation means trawl-caught hake overstate
    true emptiness [J].
  - Mahé et al. 2007 (doi:10.1093/icesjms/fsm100) has no empty share in its abstract.
- **Juvenile hake (proxy)**: Módica et al. 2011 [A]. Daily ration 1.5 % BW/d with the exponential model and
  4.09-5.09 % BW/d with the square-root model. The ration depends on the evacuation model by about 3x.
- **Saithe and cod at salmon farms**: Dempster et al. 2011, PLoS ONE 6:e15646, doi:10.1371/journal.pone.0015646 [F].
  Nine Norwegian locations, June-September.

  | | saithe, away from farms | saithe, at farms | cod, away from farms | cod, at farms |
  |---|---|---|---|---|
  | empty stomachs | 0.31 | 0.16 | 0.19 | 0.18 |
  | mean stomach content | 8.6 g | 20.2 g | 23.2 g | 32.9 g |
  | ratio, at / away [D] | | 2.35x | | 1.42x |
  | HSI, at / away | | 1.4-1.8x (2 of 3 areas) | | 2.0-2.8x |
  | condition index, at / away | | 1.06-1.12x | | 1.06-1.11x |

  - Waste feed was 0.71 of the farm saithe's stomach mass.
  - **Reading**: given energy-dense food that needs little search or capture, and shelter, wild saithe eat about
    twice as much and fatten. The natural level is therefore not a satiation or lipostatic set point.
  - **Caveats**:
    - One summer season; fish lengths differ between groups; pellets are not prey; a stomach snapshot is not a
      ration.
    - The control group is coastal fish of 34-54 cm, many immature, with 0.31 empty in June-September. The brief's
      North Sea Q2-Q4 saithe have 0.02-0.15 empty.
    - The 2.35x probably overstates the gap for offshore adult saithe in the box [J].
- **Farmed versus wild cod liver**: HSI > 12 % on ad lib feed against 2-6 % in the wild (Jobling 1988, via a
  University of Bergen thesis, core.ac.uk/download/9048951.pdf [S]). One farmed sample was 14.4 (Nofima,
  nofima.com/publication/1139654 [S]).
  - If wild livers sat at a lipostatic ceiling, ad lib cod would not overshoot them [D] 12/6 = 2x to 12/2 = 6x
    [J for the reading].
- **Prior, cited**:
  - Seyhan et al. 1998: captive whiting resume feeding at a stomach fill of 0.40, at intervals of about 21 h.
  - Jobling et al. 1994: compensatory hyperphagia.
  - Armstrong & Schindler 2011: excess digestive capacity.
  - Jeschke et al. 2002: digestion limits the timing of meals, not the asymptote.

**Verdict C.**
- Digestion and meal dynamics: **supported for the timing of meals and for the empty shares** (hake 0.47,
  saithe 0.31 coastal in summer; 0.27 North Sea Q1, prior). **Not a limiter of the daily mean**: a digestion-limited fish that has food still averages
  about Cmax.
- Lipostatic (reserve) regulation as the reason wild intake sits at 0.2-0.5: **contradicted for saithe and cod**
  (Dempster 2011; farmed cod HSI).
- **Best size**: wild saithe stomach content is about 0.43 of that of saithe with a food subsidy ([D] 8.6 / 20.2).
  This is a snapshot ratio, not a Cmax fraction.

## D. Food patchiness and search below 1 km

- **Rose & Leggett 1990**, Ecology 71:33-43, doi:10.2307/1940245 [A]. Acoustic transects of cod and capelin.
  - Densities are positively correlated at scales > 4-10 km and negatively at scales below the aggregation size
    of about 3.5 km, most negatively at 2-3 km.
  - Feeding cod tracked capelin at scales down to 3.5 m.
  - At the 1-km scale, then, a predator and its prey can be anti-correlated except while the predator is
    actively feeding. A mean-field encounter would overstate the realised one [J].
- **Kempf et al. 2008** [F]: the stomach share collapses below a Schoener overlap of 0.1.
  - The overlap is a basin-wide index across ICES rectangles (30 x 30 nm).
  - It says nothing about patchiness below 1 km.
- **Prior, cited**:
  - Rindorf & Gislason 2005: no aggregative response of whiting to prey; catches slightly negatively correlated.
  - Fall et al. 2021: the vertical overlap with capelin matters more than the horizontal.
  - The earlier Jensen/occupancy arithmetic: occupancy p acts like a cut in time budget.

**Verdict D: weak to moderate.**
- The direction is supported: anti-correlation at small scales and no aggregative response in whiting.
- **No source converts it into a ration fraction.**

## Synthesis [J]

- The farm natural experiment is the only source that tests "could the fish eat more if food were easier and safer
  to get". The answer is yes, by about 2x in stomach content, for coastal saithe in summer.
- Together with the gut-buffer constraint, this places the 0.2-0.5 level on the **intake rate while foraging**,
  not on appetite or reserves (C, contradicted) or on a diel window (A time-budget form, weak).
- Which form of low rate the evidence cannot separate:
  - per-prey capture limits (B; whiting plateaus 0.06-0.22 Wmax per prey group);
  - small-scale spatial and vertical mismatch with prey (D; Rose & Leggett; Fall);
  - risk-restricted foraging, the arena form of A (sized only by the juvenile-cod cue, 0.22-0.44).
- Season (Q1 empties, spawning) remains the documented cause of non-foraging periods lasting days (prior notes).

## Sought, not found

- Ecosim/EwE vulnerabilities for saithe, whiting or hake in Skagerrak/Kattegat models. Only the North Sea 1991
  model was found.
- A foraging-time share for whiting from tags (no whiting DST or acoustic study found), and saithe activity
  numbers (Armannsson full text blocked).
- Natural mortality (M2) of ADULT whiting, saithe or hake from seals, porpoises or larger gadoids, by age.
- Experiments in which adult gadoids reduce foraging as their reserves rise, or under predator cues (only a
  juvenile cod cue study, prior).
- Jobling & Johansen 1999 (hyperphagia) and a primary source for gadoid lipostatic regulation.
- Andersen 1998, 1999, 2001 maximum meal size against Cmax for whiting and saithe (not obtained).
- Field evidence that krill or Meganyctiphanes alone fills a gadoid to Cmax.
- The DOI of de Pontual 2012.
- The Rindorf 2006 abstract text: int-res.com returned HTTP 429. The citation is verified on Crossref.
