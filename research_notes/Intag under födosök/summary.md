# Intake rate while foraging in piscivorous gadoids (2026-11-20)

Research subagent; saved by the main agent. Research only. Nothing in the library, manifest, code, tests, AGENTS.md
or the resume was changed. No value is chosen from model output; the resume 289 decomposition (frozen fix105) is
the yardstick only. Files: `evidence.md` (E0-E8), `tables/mechanisms.csv`, `tables/intake_arithmetic_stdout.txt`
(from `scripts/intake_arithmetic.py`), `tables/budget_gate_stdout.txt` (live gate, read-only), `data/` (downloads,
untrusted, not to be committed). Tags: [F] full text, [A] abstract, [S] secondary, [D] derived, [J] judgement.

## Question
The model's whiting and water column could eat about 0.75 of the ceiling while foraging. Field fish eat about 0.4
and are limited by acquiring food ("Mättnad hos rovfisk"). What limits the field rate below encounter x capture x
digestion, and by how much?

## 0. What has to move [D]
Uncapped intake = T x X/(1 + X), where X = sum of a h B_vis.

| | whiting | water column |
|---|---|---|
| X | 5.45 | 3.55 |
| krill share of handling time | 0.82 | 0.46 |
| potential with every non-krill pair at 0 (fish and epifauna) | **0.74** (0.72-0.76) | 0.60 |
| potential with krill at 0 | 0.45 | 0.64 |
| generic factor on every a for 0.40 / 0.45 / 0.50 | x0.15 / 0.18 / 0.23 | x0.20 / 0.24 / 0.30 |
| or a factor on the foraging-time share | x0.52-0.66 | x0.53-0.66 |

- No mechanism specific to fish prey can bring whiting down: its spare capacity is krill.
- For the water column, neither a fish-only nor a krill-only factor reaches 0.5.

## 1. Per mechanism

| # | Mechanism | Verdict | Size | In the model? |
|---|---|---|---|---|
| 1 | Pursuit and handling time per item | No, for fish prey | Whiting 2-4 s per shrimp; cod 15-30 s per 0-group flatfish [F]. Digestion time per fish item 4-80 h. Loss under 0.3 % at 30 s per capture for every predator; at 10 min about 1 % for whiting, 1.2-4.2 % for saithe and hake. Under SSS the asymptote is 1/max(b, c), so about 0 [D]. Krill: a 10 % loss needs more than 4-6 min per Meganyctiphanes (whiting), about 1 min (saithe), about 13 s (saithe on Thysanoessa-size) | No, and not needed. Saithe on small krill is open, second order |
| 2 | Field capture success and pre-strike escape | No field datum for any gadoid | Sized corrections x0.6-1.4 on a. The fullness effect on strike success (x0.6, Staniland) is the Holling digestion term under Jeschke's SSS derivation; stacking it counts satiation twice [D on F] | Yes (p, q 0.8; krill p 0.5) |
| 3 | Search efficiency in the field | Encounter theory over-predicts in every case found | Cod-capelin: model stomachs fuller than observed (Fall & Fiksen 2020 [F]). Krill: library a 50-260x the field-implied a, January, 100-150 m beds ("Krillparen" [S]). Bass: 0.5 Cmax at very high prey density (Essington 2000 [S]). Whiting: fish plateaus 0.07-0.29 of the ceiling per prey group (Rindorf & Gislason 2005 [F]). Search speed: the chain uses whiting 0.8 BL/s, in situ about 0.6 (night x0.7; Onsrud et al. 2005 [S]), so x0.75 (night 0.525), new; saithe x0.5-1 [J] | Speed: no (sized, new). The rest: no, and not decomposed |
| 4 | Prey availability windows | Supported pattern, partly in the model | Fish "almost exclusively during dawn and dusk", midday low (Rindorf & Gislason 2005 [F]). Krill at dawn and in the afternoon (Rindorf 2003 [F]), but no diel trend in the Oslofjord (Onsrud 2004 [F]). Window 0.19-0.34 of the day at 58.5 N (-12 to +6 deg, definition [J]). Residual after F and the vision rule: fish x0.25-1, krill x0.5-1 [J] | Partly (vision rule at night, F for daytime schools) |
| 5 | Meals, intervals, appetite | Hunger-dependent search yes, already in h. Lower appetite contradicted in summer, suggested in winter | Whiting resume feeding at stomach fill 0.4, about every 21 h [S]. Rindorf 2002: fewer fresh meals with partly digested food present [S]. Bass forage past the sustainable fullness (13 % of diets) [S]. Dempster: wild saithe eat 0.43-0.65 of farm-fed intake [S]. Winter fish in the krill layer eat about maintenance ("Krillparen") | Yes (Holling h = E/Cmax is the SSS limit for a linear hunger term) |
| 6 | Interference, kleptoparasitism, unprofitable attacks | Minor | No aggregative response; predator and prey catches slightly anti-correlated (Rindorf & Gislason 2005 [F]), x0.9-1 [J]. Kleptoparasitism: nothing found. Prey refusal lowers intake only when gut-limited (Fall & Fiksen [F]) | w 0.1; fish attack every prey met (section 265) |

## 2. Overall answer [D/J]

**Forward combination (not back-solved).** Low / central / high potential:
- whiting 0.38 / 0.67 / 0.80 (need 0.388);
- water column 0.27 / 0.67 / 0.78 (need 0.435).

The central estimate removes about 0.1 of the roughly 0.3 excess. The all-low corner overshoots into starvation, so
the corners are not confidence limits. The separable, sized mechanisms do not explain the field level.

**By prey type.**
- **Fish prey:** the evidence points to finding vulnerable prey at the patch and school scale, not to processing
  prey once found:
  - twilight vulnerability;
  - schools met rarely;
  - prey avoiding predators at the km scale.
  When a dense shoal is met, gadoids fill up within hours (cod on capelin, Pednekar 2024 [S]; the gut-limited
  reading is [J]).
- **Krill:** the low rate is inside the layer. Whiting and pout sat in the krill layer day and night with half-full
  stomachs (Onsrud 2004 [F]), in line with "Krillparen" verdict C. Whether the in-layer limit is appetite (winter) or
  krill vigilance and escape (no fish-euphausiid capture datum) is open.

**Generic or fish-specific?** Generic in effect. In the model it must include krill, because krill carries 0.82 of
whiting's spare capacity. The pattern also holds in a non-gadoid piscivore (bass).

**What the model lacks.** A sub-cell encounter structure: prey met as patches, with at most one meal per patch, and
vulnerability windows. Candidate forms:
- a per-pair cap on the ceiling side (the Rindorf plateaus);
- an empirical field-level a, which absorbs p, q and speed, so nothing can be stacked on it;
- a foraging-time share that pays.

None is sized mechanistically. The January field-implied krill a, applied to krill all year (an extrapolation),
gives whiting 0.45-0.47 and leaves the water column at 0.64. The knife edge ("Mättnad hos rovfisk" M1) means any
choice decides starvation versus a full liver within its own uncertainty.

## 3. Load-bearing sources
1. Rindorf & Gislason 2005, JEMBE 324:1, doi:10.1016/j.jembe.2005.04.013 [F]
2. Fall & Fiksen 2020, Fish and Fisheries 21:63, doi:10.1111/faf.12415 [F]
3. Jeschke, Kopp & Tollrian 2002, Ecological Monographs 72:95 [F]
4. Onsrud et al. 2004, ICES JMS 61:1278, doi:10.1016/j.icesjms.2004.09.005 [F], with "Krillparen mot vintermagar" [S]
5. Ellis & Gibson 1997, MEPS 149:83 [F], and Staniland 1995, PhD thesis, Leicester [F]

## 4. Settling datum
The attack rate per hour, with success, of an adult gadoid inside a measured prey field. The CODFUN acoustic target
tracking of cod on capelin (Skaret et al. 2016, IMR report 41-2016) was designed to measure this.
- If attacks inside prey layers match encounter theory and success is about 0.5, the limit is finding patches.
- If attacks are far below theory inside the layer, the limit is search effort or prey vigilance. For krill this
  branch already has the Oslofjord evidence.

## 5. Gaps
- Not obtained:
  - the Rindorf 2002 full text (DTU and CORE blocked);
  - the CODFUN report (host unreachable);
  - the Essington 2000 full text;
  - Onsrud et al. 2005 (repository record only).
- Not found anywhere:
  - field capture success for any gadoid;
  - a foraging-time budget for adult gadoids from tags.
