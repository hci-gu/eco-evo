# The payoff of rest - summary (2026-11-20)

Research only: a main research agent and four literature subagents; saved by the main agent.
- Nothing in the library, manifest, code, tests, AGENTS.md or the resume was changed. No model was run and nothing
  was committed.
- No value is chosen from model output. The frozen fix105 decomposition is used only as a yardstick for how much
  rest a full fish could take.

Files:
- **Trail:** `evidence.md` (model facts M1-M8, literature L1-L4, derived D1-D7) and the subagent trails `lit_*.md`.
- **Tables:** one row per sourced number in `tables/*_raw.csv`; derived numbers in `tables/derived.csv`.
- **Scripts:** `rest_payoff.py`, `candidate_forms.py`, `vertical_refuge.py`, `fill_tradeoff.py`, with stdout in
  `tables/*_stdout.txt`.
- **`data/`:** untrusted downloads, not to be committed.

Tags: [F] full text, [A] abstract, [S] secondary, [D] derived, [J] judgement. PROXY = not a model species, or not
marine.

Two of the candidate forms reopen decisions taken on 2026-11-11: form V brings back the "resting prey is
motionless" convention the user declined, and the seal floors argued from wake trails would lower the decided
seals -> benthivores 0.9 and -> whiting 0.85.

## Question

- **The model's situation:**
  - Fish could eat about 1.5-2.3x their field ration.
  - The reserve-room cap holds them at need with fill about 1, and they forage 89-97 % of the time.
  - Rest has no payoff: foraging cost is refilled by surplus food, and every fish -> fish floor has been 1.0 since
    section 270.
- **The question:** does field evidence support an activity-dependent predation risk, against these fishes' actual
  predators?
  - If so, how large is it, and by which physical mechanism? The aim is a rest payoff that can be derived.
- **Scope addition:** the vertical, layer-based refuge as a state-dependent choice.

## Answers

### Q1. Does activity raise predation risk?

No measured ratio r = risk(active) / risk(rest) exists for any model species against any predator [J; several
searches, not exhaustive]. Every number below is a PROXY.

**(a) Encounter through the prey's own speed.**
- Gerritsen-Strickler gives r = 1 + u²/(3v²) <= 1.33 [D].
- In model terms, a motionless prey against one cruising at 0.10 m/s:

| predator (search speed) | r |
|---|---|
| cod, 0.16 m/s | 1.13 |
| whiting | 1.07 |
| saithe | 1.01 by day, 1.04 at night |
| seals, porpoises | 1.00 |

- Herring at 0.20 m/s against cod would give r 1.5. But herring schools never stop (0.25-0.42 m/s when not feeding
  [A]), so r = 1 for them.
- For a pure ambusher such as anglerfish, r has no upper bound. Anglerfish carries about 3-4 % of the water
  column's a.
- This term is already the a's own convention. It is the only form that can be derived library-wide [D].

**(b) Detection of motion.**
- The ratios are large only where the prey would otherwise be cryptic, or where the predator hunts by touch or wake:
  - Merganser on coho: r 10.6 per look on a white background [F].
  - Catfish on guppy wakes in the dark: r 5.3 [F].
  - Crappie on copepods: 3x the detection distance [S].
  - Ioannou & Krause 2009: motion has no effect on a conspicuous background [F].
- No model species has a number.
- **Seals:**
  - A swimming fish leaves a trail that can be followed 15-35 s later over tens of metres. A still fish is sensed
    only through its gill current, within about 0.36 m [F, lab, 1-2 seals].
  - Following success on a gliding trail is 0.6-0.7 of that on a continuous one [F].
  - Seals also find still prey well. Bowen 2002 [F]: buried flounder caught at 79.9 % per attempt, and captures per
    unit time about equal for cryptic and schooling prey.
  - Synthesis [J]: r about 2 (1-5) on bed gadoids, 1.3 on flatfish, 1.5 on whiting, 1.1 on schools.
- **Porpoises:**
  - Sonar needs no prey motion: blindfolded animals caught dead fish in 67 of 71 trials [F]. So r is about 1-1.1,
    except for buried flatfish (about 2).
  - Porpoises forage most at night on dispersed, non-feeding herring [A/S], which may mean r < 1.

**(c) Exposure by position.** Covered under the vertical refuge below.

**(d) Vigilance.** The evidence runs both ways and gives no magnitude:
- Foraging fish react later (Krause & Godin 1996) [A].
- Resting, sleep-like fish have raised response thresholds (zebrafish, tilapia) [F, PROXY].
- Fed Menidia swim slower and are more vulnerable [A].
- Net r = 1 (0.8-1.25) [J].

**Whole-system proxy.** Trout in food-rich lakes were less active and survived 1.6x better on an
instantaneous-mortality basis [D on F]. Habitat use and water clarity are confounded with activity.

### Q2. Activity budgets and state

- **No adult foraging-time share exists for any model species.** Tags record vertical activity, not feeding [J].
- **Cod:** about 0.70 of days on the seabed without vertical movement (Hobson et al. 2007) [F], yet travelling
  horizontally on those days (Hobson 2009) [F]. A seabed day is not a rest day.
- **Other species:** plaice spend less than 5 h a day in midwater [F]; haddock feed mostly by day, with dusk and
  dawn peaks [A].
- **Satiation lowers activity:**
  - Cod swam 0.2 body lengths/s with food unlimited, against 0.6 at half ration (Bjornsson 1993) [A].
  - Activity peaks at intermediate food, low when satiated and when starved; the same shape holds in walleye
    pollock [F].
  - Juvenile whiting activity falls with ration [S].
- **State-dependent risk:**
  - Fed juvenile pollock stay grouped for up to an hour after a threat [A].
  - Hungry salmonids take more risk, but only the direction is reported.
- **Overall:** the direction is supported (fed fish search less and take fewer risks). The size is unmeasured for
  any adult gadoid or clupeid.

### Vertical refuge (scope addition)

**No state-dependent split is documented.**
- No box prey group has a documented within-period split into a foraging profile and a refuge profile, and no
  source gives the mix share.
- Every switch found follows the period (light). That is the form the engine already has: fixed day/night f(z)
  plus rule A [J on F/A].
- Herring's deep day layer is a feeding layer [F]. Whiting eat krill day and night [F].

**Norway pout is the one named case** (Albert 1994, Norwegian Deep) [F].
- Near the bed (about 1.5 m) by day; in the lowest 5-10 m at night, feeding more at night.
- The refuge reading is only the author's hypothesis.
- At a 140 m bed, the near-bed layer is more exposed to the box's own predators [D/J]:

| predator | exposure, refuge layer / foraging layer |
|---|---|
| saithe | 1.17 |
| hake | 1.45 |
| cod (benthivores -> planktivores is a checked pair) | 4.1 |
| whiting | 0.93 |
| seals | 2.0 |
| porpoises | 1.8 |

The seal and porpoise height profiles are judgement.

**Juvenile gadoids** (0-group on the bed against 3-5 m up; PROXY stage):
- Against their checked predators the result is near neutral: saithe 1.14, hake 1.42, whiting 0.93, seals and
  porpoises 1.0.
- Benthivores -> benthivores is unchecked, so cod does not count here.

**Check against the library profile.** A mix of Albert's two layers cannot reproduce the library's pout profile, so
the check fails [D].
- Both layers lie entirely in the bottom 10 m, so any mix has a bottom-10-m share of 1.0.
- The library has 0.7 by day and 0.4 at night.

**Cross-predator trade-offs are real.** Refuge from saithe at the bed means exposure to cod, and for pout also to
seals.
- Expressing this would need an exposure multiplier above 1, or a second, rest-state band profile per pair
  (a_rest / a_forage).
- The engine's floor is a share <= 1 and cannot do either.

### Q3. Arithmetic

Booked M2 and rho, the rest share a full fish could take (`rest_payoff_stdout.txt`):

| fish group | fish-predator M2 /yr | mammal M2 /yr | rest payoff today per unit rest | rho | maximum payoff at rho if rest were safe |
|---|---|---|---|---|---|
| planktivores | 0.349 (WC 0.17, benth 0.10, whiting 0.05, mackerel 0.02) | 0.071 | 0.003 | 0.55 | 23 %/yr |
| benthivores | 0.165 (whiting 0.11, WC 0.06) | 0.085 | 0.016 | 0.35 | 8.7 %/yr |
| whiting | 0.337 (benth 0.18, WC 0.16) | 0.155 | 0.016 | 0.53 | 26 %/yr |
| water column | 0 | 0.018 | 0.003 | 0.49 | 0.9 %/yr |
| mackerel | 0.021 (pair OFF) | 0 | 0 | 0.51 | 1.1 %/yr |

- **Booked, not realised.** All M2 are booked, so every payoff is an upper bound.
- **Juveniles vs one rest share.** Most fish-pair M2 falls on juveniles (stage_access), but the engine has one rest
  share per cell.
- **At need, rest never pays by risk.** Complete safety at rest would be worth only 0.005-0.095 of one foraging tick
  [D].
- **At exact surplus** the marginal foraging tick is worth about 0, so in theory any r > 1 pays.
- **Today's mammal floors already pay a little.** At rho they give whiting 0.9 %/yr and benthivores 0.6 %/yr (on
  booked M2).
  - fix105 still forages 89-97 % of the time; fix97 whiting rested 0.066.
  - Whether a payoff this small is too small to learn is [J]: realised mammal M2 is lower than booked, and there is
    no run without those floors to compare against.

### Activity-risk ratio r per fish group, by predator class and mechanism [J on PROXY and F components]

| prey | fish predators | seals | porpoises |
|---|---|---|---|
| herring / sprat schools | 1.0 (never stop; no refuge state) | ~1.1 | 1.0 (0.6-1.2) |
| Norway pout | V: 1.01-1.13; vertical refuge < 1 (riskier) vs cod, saithe, hake | ~1.5 [J] | ~1.1 |
| juvenile gadoids (benthivores) | V: 1.05-1.07 (x M ~1.1 [J]); bed refuge near neutral vs checked predators | ~2 (1-5) | ~1.1 |
| flatfish (benthivores) | burial unmeasured (dropped in 270) | ~1.3 | ~2 |
| whiting | V: 1.13 vs cod, 1.05 vs WC if resting whiting stop; 1.0 if they hover | ~1.5 | ~1.05 |
| water column, mackerel | no fish predation of consequence | ~1.3 | ~1.05 |

## Candidate model forms

Sources: `candidate_forms_stdout.txt`, `vertical_refuge_stdout.txt`. None of the forms caps foraging time T and none
is keyed on season.

**V: motionless rest in v_rel.** The pair floor is v_rel(v, 0) / v_rel(v, u), taken over the predator's and prey's
members.

| pair | floor under V |
|---|---|
| benthivores -> whiting | 0.885 |
| WC -> whiting | 0.951 (1.0 if whiting hover) |
| whiting -> benthivores | 0.938 |
| WC -> benthivores | 0.950 |
| benthivores -> planktivores | 0.931 |
| WC -> planktivores | 0.971 |
| whiting -> planktivores | 0.963 |
| mackerel pairs | 1.0 |
| mammals | about 1.0 |

- **Payoff at rho:** whiting 2.4, planktivores 1.0, benthivores 0.9 %/yr.
- **It reopens a declined decision:** the "resting prey is motionless" convention of 2026-11-11.
- **Consistent with section 270.** It adds no refuge, only the a's own speed term.
- **It needs a field datum** that resting whiting and pout actually stop. That datum is missing.

**M: motion detection.** Floor = 1 - m(1 - d) per pair.
- **Fish predators:** m 0.2 and d 0.5 [J, PROXY]. Combined with V, the floors are 0.80-0.91.
- **Seals**, from the wake-trail mechanism: -> benthivores 0.61 (today 0.9), -> whiting 0.70 (0.85), -> WC 0.75
  (0.70).
- **Porpoises:** -> benthivores 0.76 (0.75), -> whiting and WC 0.95 (0.9), -> planktivores 1.0.
- **Payoff at rho:** whiting 3.7, planktivores 1.9, benthivores 1.8, WC 0.1 %/yr. At the low-floor corner: 10.3, 5.7
  and 4.1 %/yr.
- **Fish pairs stay at 1.0.** m and d have no datum for any model species, so 1.0 remains the evidence-based floor
  for fish pairs (section 270).
- **The seal values cannot be sized end to end.**
  - Seal a's are field averages (M4), so a lower floor at rest would also cut seal intake unless a were renormalised
    by a field rest share. No such share was found.
  - Any change would act on seals, which are currently the weakest group.

**Q: vigilance split** (q_rest / q_feed at a fixed mean q). The sign is unknown, so the floor is 1. Not a candidate.

**R: vertical refuge** (rest = a refuge profile in the band overlap).
- In the box, the one named refuge is more exposed to the dominant predators, so rest would be riskier.
- That needs a floor above 1, or a per-pair rest profile; the engine has neither.
- Consistent with section 270: vertical protection is in f(z) as a fixed average, and no state-dependent split is
  documented.

**Option A coupling.** Any floor below 1 also cuts what that predator observes of resting prey.
- For V this is defensible: the prey is the same, only met less often.
- For seals under M it over-applies: a seal's km-scale observation is not wake detection at metres.

## Verdict [J]

**1. Is there a field-supported rest payoff?** In direction, yes; in measured size, no.
- Activity raises risk through the prey's own speed (theory, and the a's own convention) and through wake and motion
  detection (lab, PROXY, seals). Fed fish are less active (cod, pollock).
- For the model's species against their actual predators, no r has been measured.
- The one derivable term, V, is 1-13 % per pair: at most about 2.4 %/yr of biomass for whiting at the full surplus
  rest share, and less for the other groups.
- Seal wake detection supports lower seal floors on resting bed gadoids and whiting. But seal M2 is small
  (0.014-0.037/yr), so this is worth under 1 %/yr, and it cannot be sized without a field rest share.

**2. Which groups could it apply to?**
- Whiting, the planktivores (through pout only) and the benthivores' juveniles carry the booked fish predation.
- The water column and mackerel suffer about zero predation in the box (M2 0.018 and 0.021). No r gives them more
  than about 1 %/yr, so a risk payoff of rest cannot remove their roughly 2x spare capacity [D].

**3. Does it do what the decomposition needs? Only half.**
- A rest payoff lowers foraging time T, not fill.
- A full reserve does cost foraging time: whiting need 0.388 of the ceiling at fill 1 against 0.275 at fill 0.1,
  which is 0.137 of time.
- But that time buys about 0.59/yr of production. Resting it instead would save 0.6 (V), 1.0 (VM central) or 2.7
  (VM low-floor) %/yr of biomass, only 0.01-0.05 of what is at stake (`fill_tradeoff_stdout.txt`) [D].
- For the water column it is 0.03-0.06 %/yr against 0.41/yr.
- At any r in the evidence range, the fish rests to need and keeps its reserve full.
- Lean spring fish therefore need acquisition limits (research_notes/"Mättnad hos rovfisk", "Intag under födosök")
  or a cost of carrying reserve, not a rest payoff alone [D/J].

**4. The vertical refuge is not a source of rest payoff in the box.**
- The one named refuge sits closer to the box's main predators.
- State-dependent layer use is unmeasured for the box's species.

## Settling data missing

- **Fish predators:** predation per encounter or per unit time by cod or saithe on resting vs swimming juvenile
  gadoids or Norway pout.
  - Conditions: box light (dim, 100-200 m), with prey speed logged.
  - Setting: a mesocosm, or animal-borne video in the field.
- **Seals:** capture rate on gadoids by prey state, from seal-borne video or accelerometers in the North Sea.
- **Whether box fish stop at rest:** accelerometer time shares for whiting and pout. Without these, form V has no
  support for those groups.

## Load-bearing sources

- Hobson et al. 2007, doi:10.3354/meps07047 [F]; Hobson et al. 2009, doi:10.3354/ab00144 [F].
- Bjornsson 1993, doi:10.1139/f93-277 [A]; Sogard & Olla 1996, doi:10.3354/meps133043 [F].
- Wieskotten et al. 2010, doi:10.1242/jeb.041699 [F]; Niesterok et al. 2017, doi:10.1242/jeb.158055 [F]; Bowen et al.
  2002, doi:10.3354/meps244235 [F].
- Martel & Dill 1995, doi:10.1111/j.1439-0310.1995.tb01095.x [F, PROXY]; Pohlmann et al. 2001,
  doi:10.1073/pnas.121026298 [F, PROXY]; Biro, Post & Parkinson 2003, doi:10.1890/02-0416 [F, PROXY].
- Albert 1994, doi:10.1006/jmsc.1994.1005 [F]; Onsrud et al. 2004, doi:10.1016/j.icesjms.2004.09.005 [F]; Lough et al.
  1989, doi:10.3354/meps056001 [F].

## Gaps

**Not found:**
- An r for any marine gadoid or clupeid.
- Any adult foraging-time share.
- Any whiting activity tagging.
- Depth by stomach fullness within one period, for any box species.
- Seal or porpoise capture rates on resting vs active fish of the same species.

**Not obtained:**
- Lankford 2001 fed-vs-unfed numbers.
- Biro 2006.
- Robinson & Pitcher 1989 (herring hunger and swimming).
- Pearre 2003 full text.
- Wieskotten 2011.

**Judgement inputs:**
- Predator speeds and member shares (from the chain).
- Pout's share of the planktivore take (0.6).
- m and d in form M.
- Predator and mammal height profiles over a 140 m bed.
- Light at the bed.
- Every payoff uses booked M2 and is an upper bound.
