# Herring winter energy budget at the Mareld bank (Skagerrak, ~58.1 N 10.8 E, ~100 m)

Question: the trained run's `pelagic_fish` holds flat Jun-Dec and loses ~30 % of biomass Jan-Apr every
year. Is the winter fat drawdown realistic? Is the winter cost too high, is the starvation rule too
harsh, or should part of the winter stock not be on the site at all?

Model state checked (supplied by the caller; library values re-read from `fgconfig/fg_library.yaml`,
`pelagic_fish`, and the growth code in `lib/environments/ecosystem_env/population_change.py`,
2026-10-20):
- Rest/hide 94-100 % of the time Nov-Feb. Ration 0.00-0.36 % BW/d Dec-Feb. Model zooplankton is
  3-6x below the A17 observations in Nov-Dec (caller's statement; this supersedes the "realistic"
  winter-zooplankton verdict of the sibling note, which was made on an earlier run).
- Reserve fill s: Oct 0.69, Nov 0.54, Dec 0.39, Jan 0.28, Feb 0.21, May 0.86.
- Starvation 0.07-0.09 %/d of biomass in Feb-Mar. M1 0.58 /yr all year. Predation by gadoids and
  seals 0.10-0.25 %/d.
- Library: `energy_content` 4000 MJ/t (lean), `max_energy_reserve` 7000 MJ/t, `maintenance_level`
  u = 0.2, `resting_metabolism` 8 MJ/t/tick (32 kJ/kg/d at 14 C), `metabolism_q10` 2.2,
  `metabolism_t_ref` 14 C, `feeding_cost` 1.9, `movement_cost` 1.5, `starve_rate` 0.025,
  derived `growth_rate` g = 0.00160628 per tick.
- Code: when s >= u, dB = B g (s - u) per tick, paid from the reserve at 4000 MJ/t (mass balance on
  by default); when s < u, dB = -B starve_rate (u - s).
- Fat mapping used throughout (library comment): fat % = s x 7000/396 = 17.7 s. So s 0.69 = 12.2 %,
  0.21 = 3.7 %, u = 0.2 = 3.5 %. Energy density ED = 4.05 + 0.396 x fat % (kJ/g).

Builds on (sibling notes; not redone, cited as [2nd]):
- `research_notes/Sillens vinterföda/winter_food_and_feeding.md`: winter ration 0.4-1.0 % BW/d
  (Lohmeyer & Hempel 1977), at or below 0.5 % BW/d, about maintenance (Bernreuther 2007). Herring feed
  day and night (Onsrud et al. 2004).
- `research_notes/Sillens vinterföda/zooplankton_refuge.md`, `herring_functional_response.md`,
  `herring_interference.md`.
- `reports/Vintermetabolism sill och tumlare.md` and
  `research_notes/Vintermetabolism sill och tumlare/herring_fat_cycle.md`, `q10_metabolism.md`:
  Kenyon et al. 2022 fat cycle, JNCC 633 energy densities, Rojbek et al. 2014, Q10 2.2 and the 14 C
  reference, Rudstam/Rose respiration constants, and the negative finding on adult winter starvation.
- `research_notes/Nettoförflyttning per funktionsgrupp/herring.md`: WBSS Skagerrak -> Sound migration
  (Nielsen 1996, DFU report 11-96).

Source tags: **[F]** full text read (pdftotext, or article page fetched in full); **[A]** abstract
only; **[2nd]** second-hand, via a sibling note or another paper that cites the original;
**[D]** my own data analysis (raw data or model arithmetic); **[L]** laboratory study;
**[S]** synthesis, i.e. my own combination of numbers from different sources; **[Sn]** search-engine
snippet only, not verified. Everything under "Inference" is mine.

---

## Evidence table

| # | Source | Tag | What it gives |
|---|---|---|---|
| 1 | Model budget from the caller's numbers + library + growth code | [D] | Where the Oct-Feb fat goes; what share of the Jan-Apr loss is starvation |
| 2 | ICES DATRAS, NS-IBTS Q1 and Q3, CPUE per age, 2014-2024, box around the site | [D] | Herring present in winter by age; Q1/Q3 biomass index |
| 3 | ICES DATRAS, NS-IBTS CA records, 2019-2024, rectangles 44F9-46G0 | [D] | Weight-at-length (condition) Q1 vs Q3; weight and length at age; maturity |
| 4 | Payne, Clausen & Mosegaard 2009, ICES JMS 66:1673 | [F] | WBSS distribution by age and quarter; adults overwinter in the Sound and Køge Bay |
| 5 | Berg, Slotte, Johannessen et al. 2017, PLoS ONE 12:e0187374 | [F] | NSAS dominate the Skagerrak in Q1; WBSS added in summer |
| 6 | Clausen, Stæhr, Rindorf & Mosegaard 2014/2015, J Fish Biol 86:228 (doi 10.1111/jfb.12571) | [A] | NSAS leave the Skagerrak-Kattegat at age 1-2; condition by area |
| 7 | Eggers et al. 2015, MEPS 520:245 | [A] | Tagged Skagerrak-coast herring; some may overwinter in coastal areas |
| 8 | Nielsen 1996, DFU report 11-96 (the Sound acoustics) | [2nd] | Sound occupied (late) Aug-Mar, up to 130,000 t; exit Feb-Mar |
| 9 | Slotte 1996, ICES CM / IMR (NORA record) | [A] | NSS herring fasting in winter lose only ~2 fat-% Sep-Dec; ~10 % during spawning migration |
| 10 | Slotte & Fiksen 2000, J Fish Biol 56:138 | [F] | NSS do not feed Sep-Oct to Mar-Apr; ~10 % fat left after spawning; 2 % fat limit for anadromous fish |
| 11 | Slotte 1999, J Fish Biol 54:338 | [2nd] | Energy loss mainly fat; 2-3x (absolute) and 3-4x (relative) higher on spawning migration than in wintering |
| 12 | Huse & Ona 1996, ICES JMS 53:863 | [A] | Overwintering NSS herring swim constantly at 0.25-0.42 m/s; not quiescent |
| 13 | Sewall, Norcross, Vollenweider & Heintz 2019, MEPS 623:195 | [F] [L] | Juvenile Pacific herring: lethal whole-body lipid 1.28 +- 0.10 % (lab fasting deaths); first-winter mortality model estimates up to 95 % |
| 14 | Vollenweider et al. 2011, J Mar Biol 2011:926812 | [A] [L] | Juvenile herring "use most of their energy reserves to survive their first winter" |
| 15 | Kenyon et al. 2022, ICES JMS 79:88 | [2nd] | NSAS muscle fat: min 4.5 % late May, 16.1 % late June, 9.1 % Sep, 5.3 % late Nov; NSS 0.03 %/d |
| 16 | Pedersen & Hislop 2001 / Wanless et al. 2005 via JNCC Report 633 | [2nd] | North Sea ED by quarter and length (Q1 4.4-8.5, Q3 3.9-11.9 kJ/g) |
| 17 | Rojbek et al. 2014, ICES JMS 71:56 | [2nd] | SD 25 whole-body lipid 2.1-13.6 %; ~10 % Nov, 3-5 % March; March and May the joint low |
| 18 | Lohmeyer & Hempel 1977; Bernreuther 2007; Onsrud et al. 2004 | [2nd] | Winter ration ~ maintenance; feeding day and night |
| 19 | Rose et al. 2008 / Bachiller et al. 2018 respiration constants | [2nd] | R = 0.0033 W^-0.227 e^(0.0548 T) g O2/g/d (resting term), 13,560 J/g O2 |
| 20 | Varpe & Fiksen 2010, Ecology 91:311 (doi 10.1890/08-1817.1) | [A] | NSS spring condition increase tied to returning day length more than to Calanus peaks |
| 21 | Bradford 1991 thesis (Dalhousie) | [Sn] | Spring spawners spend more storage energy on non-reproductive metabolism; winter distribution favours cool habitat |
| 22 | MKB Mareld (EIA, 2023), local repo PDF | [F] | Spawning periods of Skagerrak herring stocks; no seasonal presence data for the site |

---

## 0. What the model's winter budget actually does [D]

### 0.1 Where the Oct-Feb fat goes

Per day, with g_d = 4 x 0.00160628 = 0.006425 per day and q = s - u, the growth term moves
B g_d q of biomass per day and costs 4000 g_d q MJ/t/d. It also dilutes s, because new biomass arrives
with no reserve: ds/dt from dilution = -s g_d q. Resting cost = 32 kJ/kg/d x the Q10 multiplier of the
sibling report (Oct 0.91, Nov 0.76, Dec 0.64, Jan 0.54, Feb 0.51). Month-to-month drawdown of s is
converted to kJ/kg/d by x 7000 / 30.5. Mortality removes reserve with biomass, so it does not change s.

| Interval | mean s | s drawdown (kJ/kg/d) | growth payment | dilution | resting cost | implied assimilated intake | biomass growth (%/d) |
|---|---|---|---|---|---|---|---|
| Oct -> Nov | 0.615 | 34.4 | 10.7 | 11.5 | 26.7 | 14.4 (= 0.40 % BW/d of zooplankton) | 0.27 |
| Nov -> Dec | 0.465 | 34.4 | 6.8 | 5.5 | 22.4 | 0.3 (~0 %) | 0.17 |
| Dec -> Jan | 0.335 | 25.2 | 3.5 | 2.0 | 18.9 | ~0 | 0.09 |
| Jan -> Feb | 0.245 | 16.1 | 1.2 | 0.5 | 16.8 | 2.4 (0.07 %) | 0.03 |

Implied intake = drawdown balance + cost, with zooplankton at 4.5 kJ/g x 0.8 assimilation = 36 kJ/kg/d
per 1 % BW/d. The implied intakes match the caller's 0.00-0.36 % BW/d, so the table is consistent with
the run. The cost column assumes rest only. That holds for Nov-Feb (94-100 % rest) but was not
supplied for October. If October eating is non-trivial (feeding cost 1.9x), the Oct -> Nov cost and
implied ration are higher than shown.

- Over Oct -> Feb the reserve falls by 0.48 x 7000 = 3,360 MJ/t. Resting cost accounts for
  ~2,590 MJ/t. Growth payment plus dilution accounts for ~1,270 MJ/t (**~38 %**; 64 % of the Oct-Nov
  drawdown). Intake returns ~500 MJ/t.
- So the model's herring keep growing in biomass at 0.27 %/d (Oct-Nov) and 0.17 %/d (Nov-Dec), i.e.
  ~8 % and ~5 % per month. They pay for it out of their fat. This is what holds biomass "flat" through
  autumn against M1 (0.16 %/d) plus predation (0.10-0.25 %/d).

### 0.2 What share of the Jan-Apr loss is starvation

Over 120 days (Jan-Apr), with M1 0.58/yr, predation 0.10-0.25 %/d over the whole period and
starvation 0.07-0.09 %/d over ~60 days (Feb-Mar), and assuming no growth:

| predation (%/d) | starvation (%/d) | total loss | starvation share | M1 share |
|---|---|---|---|---|
| 0.10 | 0.07 | 29.7 % | 12 % | 54 % |
| 0.10 | 0.09 | 30.6 % | 15 % | 52 % |
| 0.25 | 0.07 | 41.3 % | 8 % | 36 % |
| 0.25 | 0.09 | 42.0 % | 10 % | 35 % |

The 0.10 %/d rows reproduce the ~30 % observed loss without any growth. **Starvation is only ~8-15 %
of the Jan-Apr loss.** The rest is M1 and predation acting on a pool that has stopped growing, because
s is near u from December. A pool with Z ~1/yr (M1 0.58 + checked M2 ~0.31 + F 0.10; sibling note
`Täthetsberoendets styrka per grupp/pelagic_fish.md`) and no production for four months *should* lose
~28 % (1 - e^(-1/3)).

### Inference
- Fixing the starvation rule alone cannot remove more than ~4-5 percentage points of the ~30 % winter
  loss.
- The year-on-year decline is not a winter problem in the narrow sense. It means that Apr-Sep
  production does not refill what M1 and predation take in the four months without growth. That is a
  question for the annual growth budget (r_max, g, s*), not for any of options (a)-(d).
- A fifth contributor outside the (a)-(d) menu: **autumn somatic growth paid from fat**. Real adult
  herring add no somatic mass from October to February (sibling note Q5; section 1 below for the
  population at the site). In the model, about a third of the autumn fat drawdown is converted to new
  biomass that M1 and predation then remove over winter.

---

## 1. Winter condition and lipid loss, Sep-Apr

### Literature
- NSAS muscle fat: 9.1 % in mid-September, 9.1-9.7 % late October, 5.3 % late November, about
  0.11 %/day during spawning. Minimum 4.5 % in late May. No Jan-Apr data, because there is no fishery
  then (Kenyon et al. 2022) [2nd].
- North Sea energy density (kJ/g WW) [2nd]:
  - Q1: 4.4 at 12-19.9 cm, 6.5 at 20-24.9 cm, 8.5 at 25-39.9 cm (Wanless).
  - Q3: 5.2 at 12-14.9 cm, 10.1 at 15-19.9 cm, 11.0 at 20-24.9 cm, 11.9 at 25-39.9 cm.
- SD 25 (central Baltic) whole-body lipid, spring spawners: ~10 % in November, 3-5 % in March
  (Rojbek et al. 2014) [2nd].
- NSS herring arrive at the wintering fjord with ~21-22 % fat and do not feed. "The decrease in fat
  content during the wintering period (September-December) was only around 2%". It fell by ~10 % more
  during the spawning migration and spawning (Jan/April) (Slotte 1996) [A]. Herring "have on the
  average c. 10% fat left after spawning" (Slotte & Fiksen 2000, citing Slotte 1999b) [F].
- Bradford's thesis: in NW Atlantic herring, spring spawners (long maturation period without feeding)
  spend proportionally more storage energy on routine and active metabolism than autumn spawners
  [Sn; the 1993 J Fish Biol paper was not read].
- No monthly Sep-Apr fat or energy series was found for WBSS, for Skagerrak/Kattegat juveniles, or for
  NSAS in winter (see Not found).

### Data: IBTS condition at the site [D]
Source: ICES DATRAS NS-IBTS CA records, 2019-2024, herring individuals with weight, rectangles 44F9,
44G0, 44G1, 45F9, 45G0, 45G1, 46F9, 46G0 (eastern Skagerrak around the site). Q1 = Jan-Feb (n = 2,919
fish), Q3 = August (n = 2,368). Length class midpoints are converted from the LngtCode. The
length-weight fit is lnW = a + b lnL.

| | Q1 (Jan-Feb) | Q3 (Aug) | Q1/Q3 |
|---|---|---|---|
| W at 12 cm (g) | 8.8 | 10.3 | 0.85 |
| W at 15 cm | 18.3 | 21.6 | 0.85 |
| W at 20 cm | 46.9 | 56.5 | 0.83 |
| W at 25 cm | 97.4 | 119.0 | 0.82 |
| Fulton K, 10-14 cm | 0.515 | 0.597 | 0.86 |
| Fulton K, 22-26 cm | 0.616 | 0.760 | 0.81 |
| Fulton K, 26-30 cm | 0.655 | 0.822 | 0.80 |

Kattegat rectangles (41G1-43G2) give the same picture: W at 20 cm 47.1 vs 54.2 g, and at 25 cm 96.5
vs 113.5 g.

Mean length and weight at age (site rectangles; ages are winter rings in Q1, so the Q3 age-0 cohort
reappears as Q1 age 1):

| Q3 (Aug) age | L (cm) | W (g) | -> Q1 (Jan-Feb) age | L | W | cohort weight change |
|---|---|---|---|---|---|---|
| 0 | 11.9 | 10.9 | 1 | 14.0 | 16.6 | +52 % |
| 1 | 18.6 | 45.3 | 2 | 20.9 | 56.6 | +25 % |
| 2 | 23.8 | 107.7 | 3 | 24.4 | 89.9 | -17 % |
| 3+ | 27.0 | 161.9 | 4 / 5 | 26.1 / 27.3 | 118.6 / 148.9 | -10 to -27 % |

### Synthesis: what the site population's mean fat looks like [S]
I combined the IBTS biomass composition (section 4) with the North Sea quarter- and length-specific
energy densities [2nd] and the model's mapping fat % = (ED - 4.05)/0.396.
- **Q1 (Jan-Feb).** Biomass shares are 0.42 (1-wr, 14.0 cm, ED 4.4 -> ~0.9 % fat), 0.47 (2-wr, 20.9 cm,
  at the 4.4/6.5 bin boundary -> ~3.5-6.2 %) and 0.11 (3+, 25.2 cm, 6.5-8.5 -> 6.2-11.2 %). The
  weighted mean is **2.7-4.5 % fat, s ≈ 0.15-0.25**.
- **Q3 (Aug).** Biomass shares are 0.28 (0-group, ~0.9-2.9 %), 0.36 (age 1, 18.6 cm, 10.1 -> 15.3 %),
  0.25 (age 2, 23.8 cm, 11.0 -> 17.6 %) and 0.12 (3+, 11.9 -> 19.8 %). The weighted mean is
  **12.4-13.0 %, s ≈ 0.70-0.73**, or 16.8 % without the 0-group.
- **Energy at fixed length**, Aug -> Jan-Feb: 20 cm 56.5 g x 11.0 = 622 kJ -> 46.9 g x 6.5 = 305 kJ
  (-51 %); 25 cm 119 x 11.9 = 1,416 kJ -> 97.4 x 8.5 = 828 kJ (-42 %).
- Caveats: the ED values are North Sea, not Skagerrak. Q3 is August, not October. The composition
  comes from a demersal trawl (section 4). The bin boundaries are coarse.

### Inference
- **For a single adult fish, 12 % in October -> 4 % in February is too steep.** NSS herring that do not
  feed at all lose ~2 fat-points from September to December. North Sea adults still hold 6.5-8.5 kJ/g
  (~6-11 %) in Q1. The model's drawdown of 0.087 fat-%/d in Oct-Nov is ~3x the NSS fasting rate
  (0.03 %/d, Kenyon via sibling; ~0.02 %/d from Slotte 1996).
- **For the population actually over the bank, the model's numbers are realistic.** The site's winter
  stock is dominated by lean juveniles. The mean is ~12-13 % fat in August and ~3-4.5 % in Jan-Feb,
  which brackets the model's Oct 12.2 % and Feb 3.7 %. Much of that real drop is a change in *who is
  there* (fat adults leave, lean juveniles stay; section 4). Within-fish condition loss is real
  (weight at length -15 to -20 %, energy at length -40 to -50 %), but in the data it is spread from
  August to Jan-Feb, not October to February.
- Answer: "plausible as a site-population mean, too steep as an individual adult trajectory". The
  model has one pool, so it cannot be both.

---

## 2. Do herring die of starvation in winter?

### Literature
- Adults: no source reports winter starvation as a component of adult natural mortality for NSAS,
  WBSS or Baltic herring. Adult M2 on 2+ herring is predation-dominated and stable (sibling report,
  Dickey-Collas et al. 2010) [2nd]. Nothing found in this session changes that negative finding.
- Adult fat floors that live fish reach:
  - 2.1 % whole-body (SD 25, spent fish; Rojbek) [2nd].
  - ~10 % left after spawning in NSS (Slotte & Fiksen 2000) [F].
  - Slotte & Fiksen 2000 [F] quote a 2 % fat limit (Jonsson et al. 1997) and a 40 % remaining-energy
    limit (Glebe & Leggett 1981) for anadromous fish that die after spawning. They chose 60 % remaining
    energy as the conservative floor for herring.
- Juveniles (first winter), Pacific herring as the analogue species:
  - Model estimates of YOY winter mortality "can be as high as 95%" (Norcross & Brown 2001 via Sewall
    et al. 2019) [2nd].
  - Laboratory YOY herring fasting for 10 weeks at 3 temperatures: those that died had **1.28 +- 0.10 %
    whole-body lipid** (wet), "the minimum lipid level needed for survival" (Vollenweider et al. 2011a
    via Sewall et al. 2019) [F] [L].
  - Small fish had "exhausted their lipid stores by March" [Sn of the same paper].
  - Juveniles "use most of their energy reserves to survive their first winter" (Vollenweider et al.
    2011) [A].
- Atlantic herring: no study of first-winter starvation mortality of juveniles was found for the
  North Sea, Skagerrak/Kattegat or Baltic.

### Calculation [D]
- Model onset: s = u = 0.2, i.e. ~3.5 % fat. That is 2.7x the lab-lethal 1.28 % and 1.7x the 2.1 %
  that live SD 25 herring reach.
- The lethal lipid corresponds to s = 1.28 x 396 / 7000 = 0.072.
- The model loss rate at a given s below u is 4 x 0.025 x (0.2 - s) per day: 0.1 %/d at s = 0.19 and
  1 %/d at s = 0.10.

### Inference
- Adults lose condition, they do not starve. Juveniles in their first winter are the only case where
  starvation mortality at the population level is plausible, and the evidence for it is Pacific.
- The model's threshold sits above any lethal or observed-survivable lipid level. Lowering it toward
  s ≈ 0.07-0.12 (1.3-2.1 % fat) is defensible. But section 0.2 shows this changes the winter loss by
  only a few percentage points. Also, `maintenance_level` is u in the growth term and the hunger gate
  too, so it cannot be lowered on its own without moving growth and appetite (sibling report).
- An alternative that keeps u: a separate starvation threshold s_starve < u, with neither growth nor
  starvation in the band between them. This is my inference, not a literature model.

---

## 3. Winter energy expenditure

### Already established (sibling report, not redone) [2nd]
- No respirometry of adult Atlantic herring at 4-7 C exists.
- Clupeid Q10 1.8-2.2.
- The library's 32 kJ/kg/d is the Rudstam/Rose resting term for a 150 g herring at ~14 C.
- Winter 0-50 m is 5.2-6.3 C (Jan-Apr).
- The Rudstam total including activity at 5 C is ~32 kJ/kg/d.
- Winter ration ~ maintenance: gross ~11 kJ/kg/d in Mar-Apr (Bernreuther); 12-25 kJ/kg/d assimilated
  in Jan-Feb (Lohmeyer).

### New
- **Field fasting loss, NSS.** About 2 fat-points over September-December without feeding (Slotte
  1996) [A]. Energy loss was "attributed mainly to fat metabolism", and relative loss decreases with
  length (Slotte 1999) [2nd]. Wintering-fjord temperature is not given in the abstract.
- **Activity.** Overwintering NSS herring are not quiescent. They swim constantly at 0.25-0.42 m/s
  to offset negative buoyancy at depth (Huse & Ona 1996) [A]. Skagerrak/Kattegat herring feed through
  winter, by day and by night (Onsrud, Lohmeyer) [2nd]. So "reduced activity" is supported (no feeding
  migrations, no spawning), but "torpor" is not.

### Calculation [D]/[S]
- NSS field rate: 2 fat-points x 0.396 kJ/g over ~90-120 days = **6.6-8.8 kJ/kg/d** for adult NSS
  (~300 g, assumed). This is an upper bound on routine metabolism, because some energy also moved
  into gonads.
- Mass scaling with the Rudstam exponent -0.227: x1.17 for 150 g, x1.58 for 40 g. That gives
  **~8-10 kJ/kg/d at 150 g and ~10-14 kJ/kg/d at 40 g** [S].
- Rudstam resting term, R = 0.0033 W^-0.227 e^(0.0548 T) x 13,560 J/g O2: 150 g at 6 C 19.9 kJ/kg/d;
  300 g at 7 C 18.0; 20 g at 6 C 31.5. The Wisconsin resting term alone is therefore ~2-3x the NSS
  field fasting loss for the same size. It is an overestimate for wintering herring, or the field loss
  misses protein catabolism. I cannot separate these.
- Model: resting cost 32 x 0.76/0.64/0.54/0.51 = **24.3 / 20.5 / 17.3 / 16.3 kJ/kg/d** (Nov/Dec/Jan/
  Feb). The model's herring rest 94-100 % of the time, so this is the total cost.

### Inference
- At the body size of the winter population at the site (biomass-weighted ~40-60 g in Q1; section 4),
  the model's winter cost is **~1.2-2x** the NSS-derived field estimate. Against Bernreuther's
  maintenance ration (~11 kJ/kg/d gross, Mar-Apr) it is ~1.5x.
- For adult-size fish it is ~2-2.5x.
- The parameter is `resting_metabolism` (8 MJ/t/tick, a 14 C summer value), together with how the Q10
  is anchored. A winter multiplier of ~0.25-0.35 instead of 0.51-0.64 would match the NSS fasting
  loss; scaled to juvenile size the band is ~0.3-0.45.
- Evidence strength: low-moderate. The field number is from another stock, at an unknown fjord
  temperature, from one abstract.

---

## 4. Where are the herring in winter?

### Literature
- **WBSS adults leave.** "Catches of mature (3+ wr) WBSS herring peak in midsummer in the eastern
  North Sea and outer Skagerrak". In Q3 "large concentrations of 2+-wr herring are found in the
  southern Kattegat", and mature 2+-wr herring overwinter in the Øresund and Køge Bay. In Q1 "the
  adults are in the overwintering and spawning areas and the juveniles are in the nursery areas", and
  the Q1 IBTS covers feeding grounds "where few WBSS fish are expected". "Younger fish mostly remain
  in the Skagerrak and Kattegat" (Payne et al. 2009) [F].
- The Sound holds herring from (late) August to March, up to 130,000 t (Sep 1993), with emigration in
  Feb-Mar (Nielsen 1996) [2nd].
- **NSAS dominate the Skagerrak in winter.** "NSAS dominated during the first quarter of a year,
  followed by an increased occurrence of WBSS in summer". The Skagerrak and Kattegat also hold local
  spring-spawning populations (Berg et al. 2017) [F]. NSAS adults spawn Aug-Oct in the North Sea
  (Kenyon, sibling) [2nd].
- **NSAS juveniles leave at age 1-2.** "NSAS were found to leave the study area at the age between 1
  and 2 years and WBSS 1 year olds were encountered in the Skagerrak". Condition was lower in the
  Kattegat than in the North Sea and Skagerrak (Clausen et al. 2014) [A]. A 1990s HAWG practice
  treated all 1-ringers in IIIa as NSAS [Sn].
- **Local coastal fish stay.** Tagged herring on the Norwegian Skagerrak coast left the tagging fjord
  by August. Later detections "suggest that some herring may overwinter in coastal areas" (Eggers et
  al. 2015) [A].
- The EIA for Mareld gives Skagerrak spawning periods (spring spawners Jan-Jul, mostly Apr-May;
  autumn spawners Sep-Feb, mostly Oct-Dec) but no seasonal presence at the site [F].

### Data: IBTS herring at the site, Q1 vs Q3 [D]
Source: ICES DATRAS NS-IBTS `getCPUEAge` (numbers per hour by age), 2014-2024. Box: 57.6-58.6 N,
9.8-11.5 E, depth 40-200 m. Q1: 165 hauls (130 Jan, 35 Feb). Q3: 99 hauls (95 Aug, 4 Sep). All but one
haul are daytime. Biomass index = CPUE x the site mean weight at age from the CA data (table in
section 1, 2019-2024, applied to all years).

| | age 0 | age 1 | age 2 | age 3+ | total excl. age 0 |
|---|---|---|---|---|---|
| Q1 mean n/h | 0 | 3,743 | 1,227 | 152 | |
| Q3 mean n/h | 15,516 | 4,819 | 1,401 | 460 | |
| Q1 mean kg/h | 0 | 62 | 69 | 16 | **148** (median 18) |
| Q3 mean kg/h | 169 | 218 | 151 | 75 | **444** (median 23) |
| Q1/Q3 (means) | - | 0.28 | 0.46 | **0.21** | **0.33** (medians 0.8) |

- Year-to-year variation is very large: the Q1 annual mean ranges 8-343 kg/h and Q3 10-2,031 kg/h.
- Kattegat (56.0-57.5 N) Q1 age-1 CPUE averages 24,667 n/h, i.e. the main juvenile nursery is further
  south.
- Maturity of age 3+ in Q1 at the site: regressing/spent (SMSF Da/Db) 147 fish; developing (B/Ba/Bb)
  134; spawning-capable (C/Cb) 43; the rest on the M6 scale. Ages 1-2 are mostly immature (A, Ba, 61).

### Inference
- The winter herring over the bank are mostly **1- and 2-ringer juveniles (~90 % of the Q1 biomass
  index)**, plus a small adult fraction. The adult fraction is a mix of spent autumn spawners and
  maturing spring spawners (from the maturity stages; stock identity not tested).
- Adults (3+) are at ~1/5 of their August index. That fits WBSS adults leaving for the Sound in
  Aug-Sep and NSAS adults spawning in the North Sea.
- Total winter presence is ~1/3 of August by the mean and similar by the median. That is not
  "everybody leaves".
- Catchability caveat: the GOV is a demersal trawl. Pelagic schools above the headline in summer are
  missed, so the Q1/Q3 ratio may be biased in either direction. These are relative indices within one
  gear, not abundance.
- For the model: if the August stock (adults included) is kept on the site through winter, the
  winter stock is overestimated by perhaps 1.5-3x, and its adult share is overestimated ~5x. Adults
  that left would spend winter elsewhere (at sea temperatures and on a ration the model does not
  charge) and return fat in summer. Today the model charges them a full winter on the bank.

---

## 5. Spring recovery

### Literature
- NSAS muscle fat 4.5 % in late May -> 16.1 % by late June/early July, a plateau to July, and a
  maximum of 16-22 % (Kenyon et al. 2022) [2nd].
- SD 25 spring spawners: March and May are the joint low, 3-5 % (Rojbek) [2nd].
- NSS feed April-August and store "a considerable amount of fat"; ~21-22 % on arrival in the wintering
  area in Aug-Sep (Slotte 1996) [A].
- The NSS spring condition increase follows returning day length more than the Calanus peaks of
  March-April and June-July (Varpe & Fiksen 2010) [A].
- Bornholm Basin rations exceed maintenance from April to July/August (Bernreuther, sibling) [2nd].
- No Mar-Jun monthly fat series was found for Skagerrak/Kattegat or WBSS.

### Calculation [D]
- Model: s 0.21 (Feb) -> 0.86 (May) = 3.7 -> 15.2 % fat over ~90 days, 0.13 fat-%/d. The refill
  starts in February (sibling: Feb ration 1.4 %, Mar 3.5 % in an earlier run).
- NSAS May-June: 4.5 -> 16.1 % in ~5-6 weeks, ~0.3 %/d.

### Inference
- The model reaches the right summer level (15 % by May vs 16 % by late June) but ~2-3 months early
  and at half the observed rate. The observed recovery starts in May (NSAS) or after spawning in
  Apr-May (spring spawners).
- A model minimum in February instead of April-May is consistent with the herring entering winter too
  early and refilling on an appetite-driven spring (sibling Q3/Q4).

---

## Recommendation

**Note first:** the Jan-Apr loss is mostly not starvation [D, section 0.2]. About 85-92 % of it is M1
and predation on a pool that has stopped growing, which is the expected shape for Z ~1/yr with no
winter production. The year-on-year decline means Apr-Sep production is too small for the annual
mortality (inference).

**The gate on all four options (inference from the growth code, [D]):** model growth is gated only by
s - u. The model reproduces "no winter growth" through *fat exhaustion* (s falls to u by January),
not through seasonal allocation. Anything that keeps s above 0.2 through winter, i.e. lower cost (b),
more autumn food (a) or adults leaving before they pay for winter (d), reopens winter growth at
~0.1-0.2 %/d (section 0.1). That offsets M1 and predation and flattens the Jan-Apr curve. A probe
will therefore show (a), (b) or (d) "removing the decline", but for the wrong reason. The IBTS
cohorts aged 2+ lose 10-27 % of their mass from August to Jan-Feb [D], and adults add no somatic mass
Oct-Feb (sibling Q5). A cost or food fix without a seasonal growth brake trades a realistic winter
shape for an unrealistic one. The honest verdict on the year-on-year decline: the summer surplus is
too small, *and* the model has no mechanism to stop spending it as autumn growth. The growth brake
(below) should come first, and (a)-(d) be judged with it in place.

Ranked by strength of evidence:

1. **(d) Part of the winter stock should leave the site. Supported, moderately strong
   (qualitatively strong, quantitatively weak).** WBSS adults overwinter in the Sound/Køge Bay
   (Payne 2009 [F], Nielsen 1996 [2nd]). NSAS adults spawn in the North Sea, and NSAS dominate the
   Skagerrak in Q1 (Berg 2017 [F]). The IBTS adult index in Jan-Feb is ~0.2 of August, total ~0.33
   (mean) [D]. Juveniles stay (~90 % of winter biomass). Implication (inference): the model
   overstates the winter stock on the bank and especially its adult part. The minimal version is an
   Aug/Sep emigration of ~2/3 of the adult-size biomass with return in May-June. That needs the
   `--migration` machinery, not a parameter change.
2. **(b) The winter cost is too high. Supported, low-moderate.** NSS herring that do not feed lose
   ~2 fat-points Sep-Dec, ~7-9 kJ/kg/d (Slotte 1996 [A]); ~8-14 kJ/kg/d after mass scaling [S]. The
   model's resting cost is 16-24 kJ/kg/d Nov-Feb. Parameter: `resting_metabolism` 8 MJ/t/tick, or its
   winter multiplier, too high by **~1.2-2x** at juvenile size and ~2-2.5x at adult size. Caveat: other
   stock, unknown fjord temperature, abstract only. A full read of Slotte 1999 (kJ/fish/d in wintering)
   is the test.
3. **(c) The starvation threshold is too harsh. Supported on level, irrelevant to the decline.** Onset
   at ~3.5 % fat vs a lab-lethal 1.28 +- 0.10 % lipid in juvenile herring (Sewall 2019 [F] [L]) and
   2.1 % in live adults (Rojbek [2nd]). The evidence for a lower onset (s ≈ 0.07-0.12) is good. But
   starvation is only ~8-15 % of the Jan-Apr loss [D], so this is a small correction. It should be a
   separate threshold, not a lower `maintenance_level` (sibling report).
4. **(a) The drawdown is realistic, so the fix is autumn food. Half supported.** As a *site
   population mean*, 12 % (Oct) -> 4 % (Feb) is realistic. The observed site mean goes from
   ~12-13 % in August to ~3-4.5 % in Jan-Feb [S], partly through turnover of who is present. As an
   *individual adult* trajectory it is ~3x too fast. Autumn food matters: model Nov-Dec zooplankton is
   3-6x below A17 (caller), and the Dec-Jan ration is 1.5-3x below the literature (sibling). But at
   s ≈ 0.5-0.7 the model's extra autumn intake would partly go into more autumn somatic growth
   (section 0.1), not into winter fat. This is inference from the growth code, and more Oct-Dec food
   is not sufficient on its own.

**Also outside the menu (inference, [D]):** about 38 % of the Oct-Feb fat drawdown is the growth term
turning fat into biomass at 0.17-0.27 %/d in Oct-Dec. The IBTS data show cohort weight *falling*
Aug -> Jan-Feb for ages 2+ (-10 to -27 %) and rising only for 0-1-group juveniles. A seasonal brake
on DM growth outside the feeding season (e.g. the existing `growth_q10`, or growth only above a
higher reserve in autumn) would keep that fat in the fish. This is the first thing to probe (see
"The gate on all four options" above); (a)-(d) should be evaluated with it in place.

---

## References

- Berg F, Slotte A, Johannessen A, Kvamme C, et al. 2017. Comparative biology and population mixing
  among local, coastal and offshore Atlantic herring in the North Sea, Skagerrak, Kattegat and western
  Baltic. PLoS ONE 12:e0187374. https://doi.org/10.1371/journal.pone.0187374 [F]
- Bradford RG 1991. PhD thesis, Dalhousie University (storage lipid/protein use in NW Atlantic
  herring). https://dalspace.library.dal.ca/handle/10222/55273 [Sn]; paper: Bradford 1993, J Fish Biol
  43:811 (not read).
- Clausen LW, Stæhr K-J, Rindorf A, Mosegaard H 2014. Effect of spatial differences in growth on
  distribution of seasonally co-occurring herring stocks. J Fish Biol 86:228-247 (online 2014).
  https://doi.org/10.1111/jfb.12571 [A]
- Eggers F, Moland Olsen E, Moland E, Slotte A 2015. Individual habitat transitions of Atlantic herring
  in a human-modified coastal system. MEPS 520:245-256. https://doi.org/10.3354/meps11103 [A]
- Huse I (Ona E as co-author in common citation; Crossref lists Huse only) 1996. Tilt angle distribution and swimming speed of overwintering Norwegian spring
  spawning herring. ICES J Mar Sci 53:863-873. https://doi.org/10.1006/jmsc.1996.9999 [A]
- ICES DATRAS, NS-IBTS Q1 and Q3, CPUE per age (getCPUEAge) 2014-2024 and CA records (getCAdata)
  2019-2024, downloaded 2026-10-20. https://datras.ices.dk/WebServices/DATRASWebService.asmx [D]
- Kenyon S, et al. 2022. Intra- and inter-annual variability in the fat content of Atlantic herring.
  ICES J Mar Sci 79:88-99. https://doi.org/10.1093/icesjms/fsab244 [2nd, via sibling note]
- Kenyon S, Kvamme C, Pastoors M, Mackinson S, Marshall CT 2025. Fat dynamics at the onset of sexual
  maturation in Atlantic herring. MEPS 763:115-126. https://doi.org/10.3354/meps14890 [F, skimmed; no
  winter data, not used for numbers]
- Mareld Green Energy AB 2023. Miljökonsekvensbeskrivning Mareld vindkraftpark, Bilaga C (repo file
  `MKB Mareld.pdf`) [F]
- Nielsen JR 1996. Acoustic monitoring of herring in the Sound. DFU report 11-96. [2nd, via
  `Nettoförflyttning per funktionsgrupp/herring.md`]
- Payne MR, Clausen LW, Mosegaard H 2009. Finding the signal in the noise: objective data-selection
  criteria improve the assessment of western Baltic spring-spawning herring. ICES J Mar Sci
  66:1673-1680. https://doi.org/10.1093/icesjms/fsp185 [F]
- Pedersen J, Hislop JRG 2001. J Fish Biol 59:380-389, via JNCC Report 633 (Ransijn, Booth & Smout
  2019). [2nd]
- Røjbek MC, Tomkiewicz J, Jacobsen C, Støttrup JG 2014. Forage fish quality: seasonal lipid dynamics
  of herring and sprat. ICES J Mar Sci 71:56-71. https://doi.org/10.1093/icesjms/fst106 [2nd]
- Rose KA, et al. 2008. Trans Am Fish Soc 137:278-297; Bachiller E, et al. 2018. PLoS ONE
  13:e0190345. https://doi.org/10.1371/journal.pone.0190345 [2nd]
- Sewall F, Norcross B, Vollenweider J, Heintz R 2019. Growth, energy storage, and feeding patterns
  reveal winter mortality risks for juvenile Pacific herring in Prince William Sound. MEPS 623:195-208.
  https://doi.org/10.3354/meps13013 [F]
- Slotte A 1996. Relations between seasonal migrations and fat content in Norwegian spring spawning
  herring. IMR/ICES CM paper, NORA record https://hdl.handle.net/11250/105591 [A]
- Slotte A 1999. Differential utilization of energy during wintering and spawning migration in
  Norwegian spring-spawning herring. J Fish Biol 54:338-355.
  https://doi.org/10.1111/j.1095-8649.1999.tb00834.x [2nd]
- Slotte A, Fiksen Ø 2000. State-dependent spawning migration in Norwegian spring-spawning herring.
  J Fish Biol 56:138-162. https://doi.org/10.1111/j.1095-8649.2000.tb02091.x [F; PDF from https://bio.uib.no/te/papers/Slotte_2000_State-dependent_spawning.pdf]
- Varpe Ø, Fiksen Ø 2010. Seasonal plankton-fish interactions: light regime, prey phenology, and
  herring foraging. Ecology 91:311-318. https://doi.org/10.1890/08-1817.1 [A]
- Vollenweider JJ, Gregg JL, Heintz RA, Hershberger PK 2011. Energetic cost of Ichthyophonus infection
  in juvenile Pacific herring. J Mar Biol 2011:926812. https://doi.org/10.1155/2011/926812 [A]
- Sibling notes listed under "Builds on" (Lohmeyer & Hempel 1977; Bernreuther 2007; Onsrud et al.
  2004; Dickey-Collas et al. 2010; JNCC Report 633) [2nd].

## Failed fetches and not found

Failed fetches:
- Ulrich et al. 2012, "Modelling the mixing of herring stocks between the Baltic and the North Sea
  from otolith data" (DTU Orbit, https://orbit.dtu.dk/files/123045975/J0212.pdf): HTTP 403. This is
  the paper with quarterly NSAS/WBSS composition in IIIa, i.e. the missing Q4 numbers.
- Thünen Clupea cruise report 391 (2024), Sound overwintering: too many redirects.
- Plymouth MBA "Variations in the chemical composition of herring" (plymsea.ac.uk/963): HTTP 401.
- IMR Brage (imr.brage.unit.no) for the full text of Slotte 1996/1999: DNS failure; the repository has
  moved to nva.sikt.no, not followed up.
- Huse & Ona 1996 on paperity: HTTP 403 (abstract read via OpenAlex instead).
- EVOS Trustee Council reports (evostc.state.ak.us): connection refused. They carry Pacific herring
  winter energy-loss rates (search snippet: 0.25-0.42 %/d), not used.
- Henderson & Almatar 1989 (JMBA, Clyde spring spawners, monthly lipid): paywalled, only the
  introduction seen.

Not found:
- Any monthly Sep-Apr fat or energy series for herring in the Skagerrak/Kattegat, for WBSS, or for
  NSAS in Jan-Apr.
- A measured routine or standard metabolic rate for adult Atlantic herring at 4-8 C.
- The temperature of the NSS wintering layer in Slotte 1996.
- Atlantic herring first-winter starvation mortality (North Sea, Skagerrak/Kattegat, Baltic).
- Quarterly stock composition and quarterly catches of herring in Division IIIa (HAWG tables).
- Hurst TP 2007. Causes and consequences of winter mortality in fishes. J Fish Biol 71:315-345,
  https://doi.org/10.1111/j.1095-8649.2007.01596.x: located via Crossref, not read.
- Wood 1958 (J Cons 23:390), "Fat cycles of North Sea herring": record only, no data.
