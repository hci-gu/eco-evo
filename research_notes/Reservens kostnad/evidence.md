# The cost of carrying reserve - evidence trail (2026-11-20)

Research only. Nothing in the library, manifest, code, tests, AGENTS.md or the resume was changed; no
model was run and nothing was committed. No value is chosen from model output: the frozen fix105 fill
decomposition ("Mättnad hos rovfisk"/model/tables/fill_decomp_all_s*_fix105.txt) and the brief's
"potential ~2x need" are yardsticks only.

Tags: [F] full text, [A] abstract, [S] secondary (an earlier note, or a claim about a paper read in
another), [D] derived here, [J] judgement. PROXY = not a model member species.

Literature trails: `lit_risk.md` (risk vs reserve), `lit_cost.md` (metabolic, buoyancy, drag and
storage costs), `lit_excess.md` (fish on excess food; why spring fish are lean). One row per sourced
number: `tables/risk_raw.csv`, `tables/cost_raw.csv`, `tables/excess_raw.csv`. Arithmetic:
`scripts/model_reasons.py` -> `tables/model_reasons_stdout.txt`, `scripts/optimal_fill.py` ->
`tables/optimal_fill_stdout.txt`.

## M. What a full reserve costs and earns in the model (read from code and library) [D]

M1. Reward. Training (fix105 `gpu_run.json`: ticks 1460, integral_reward True, legacy_reward False,
    warm-up 60 d, random start day) scores the mean over a one-year rollout of
    log((B ec + R) / E0) (lib/gpu/rollout.py). A stored MJ counts exactly like a MJ of lean tissue.
    A full reserve is ME/(ec + ME) of the group's energy: planktivores 0.555, benthivores 0.333,
    whiting 0.339, water column 0.324, mackerel 0.599.
M2. Growth. The material-law ceiling is B g (s - u) p (lib/environments/ecosystem_env/movement.py,
    income_growth) with g derived at s* = 1 (appetite: reserve_room). The somatic production P is
    reached only at fill 1; at fill 0.3 the ceiling is (0.3 - 0.07)/(0.93) = 0.25 of it (whiting).
    A model fish therefore needs a full reserve to grow at the field rate.
M3. Costs of a full reserve that already exist:
    - losses (M1, M2, emigration) take the reserve pro rata, so the fraction of E lost per tick does
      not depend on fill: no cost in log E;
    - benthivores only: a new tonne costs ec + c (ec + s ME) with c 0.69, x1.128 at fill 1 vs 0.33
      ("Bottenfiskens budget" finding 5 [S]);
    - the reserve-room cap discards surplus at fill 1: a ceiling, not a cost;
    - foraging time to build the reserve: whiting need 0.388 of the ceiling at fill 1 against 0.275
      at fill 0.1; resting that time would save 0.01-0.05 of the production at stake ("Vilans
      avkastning" verdict 3 [S]).
    No metabolic, swimming, buoyancy or risk term depends on the reserve.
M4. Observations. A DM sees the light level of its tick (daylight on); the planktivores also see
    their set-point room (section 271). No other group observes the day of year or the day length
    directly [S AGENTS.md]. Whether a feed-forward policy can infer the season from one tick's light
    level is [J] (the 6-h ticks fall at fixed clock times, so the light at the dawn and dusk ticks
    carries day length).

## D. Arithmetic

### D1. What a full reserve is worth to the learner, and the cost that would flip it
(`tables/model_reasons_stdout.txt`)

Holding fill 1 instead of the field annual-mean fill raises the one-year integral reward by:

| group | field mean fill | reward gain (log units) | = months of P |
|---|---|---|---|
| planktivores | 0.63 | +0.363 | 6.6 |
| benthivores | 0.62 | +0.291 | 4.7 |
| whiting | 0.56 | +0.308 | 6.1 |
| water column | 0.71 | +0.164 | 4.7 |
| mackerel | 0.67 | +0.291 | 8.9 |

About half of the marginal value is the reward currency (M1) and half the growth ceiling (M2).

Hazard-type cost, predation M2 x (1 + beta s), booked M2 [S "Vilans avkastning"]: the learner's
optimum leaves fill 1 only above beta* and reaches the field annual mean at:

| group | booked M2 /yr | beta* | beta at field mean | full / empty hazard |
|---|---|---|---|---|
| planktivores | 0.420 | 4.4 | 5.0 | 6.0x |
| benthivores | 0.250 | 5.9 | 6.3 | 7.3x |
| whiting | 0.492 | 2.7 | 3.0 | 4.0x |
| water column | 0.018 | 61 | 65 | 66x |
| mackerel | 0.021 | 77 | 91 | 92x |

As a fill-proportional extra loss independent of M2: 1.2-2.1 /yr at fill 1. Energy-type cost
k RM s: with potential ~2x need it is paid out of the spare intake and binds only above
k_crit = 1.3-2.5 (a full reserve would have to double or triple resting metabolism).

### D2. Optimal fill of a field fish (`tables/optimal_fill_stdout.txt`)

Dynamic programme for whiting (library ec, ME, RM, feeding_cost, P, M1, booked M2), maximising
survival x mass, decision = foraging share, growth paid before the reserve, activity risk r 1.1
["Vilans avkastning" S]. Results [D, on J inputs for food patchiness]:

1. Without food gaps a reserve has no insurance value: its level is arbitrary at beta 0 and any
   beta > 0 empties it. The field's full autumn fish need a reason to insure (gaps, winter).
2. Flat potential 2x need (the yardstick), food in bad spells of 5-20 d (bad share 0.3):
   beta 0 -> fill 0.85 (5-d spells) to 0.94 (20-d); beta 0.3 -> 0.30-0.81; beta 1 -> 0.25-0.62; beta 3 -> 0.21-0.46.
   The fill is flat over the year at every beta.
3. A metabolic carrying cost is a weak lever even for a field fish: k = 1 x RM per unit fill moves
   fill only 0.94 -> 0.87 (it is paid from the surplus; only the extra foraging risk counts).
4. The activity risk r (1.0-1.5) does not move the optimum (0.61-0.62 at beta 1).
5. Seasonal potential (SMS whiting ration shape) with 10-d gaps:
   - beta 0: fill 0.75-0.80 in June (0.85-0.88 in July), 0.91-0.96 in Sep-Nov, at every kpot (1.0-2.0);
   - kpot 1.0 (potential = need on the annual mean, the field closure): beta 0.3 -> 0.54 (Jun) to
     0.79 (Oct), mean 0.67; beta 1 -> 0.41 to 0.61, mean 0.51; beta 3 -> 0.31 to 0.45;
   - kpot 2.0: beta 0.3 -> 0.48-0.53, beta 1 -> 0.40-0.43: a flat fill.
   Field whiting: 0.30 (May-Jul) to 0.73 (Sep-Nov), mean 0.56, swing 0.43.
6. Field growth timing (growth x0.3 / 1.5 / 1.5 / 0.7 by quarter [J]) changes the trough by
   <= 0.04.

Reading [D/J]:
- The phase of the cycle (low in early summer, high in autumn) comes from the seasonal potential, not
  from the cost. A cost lowers the level and, at potential 2x need, removes the cycle.
- No beta reproduces both the field mean and the field swing: the best (beta 0.3-1 at kpot 1.0)
  gives a swing of 0.20-0.25 against 0.43. The rest needs a forced drawdown: winter or spawning
  intake below cost (Q1 fast of the mature 0.34-0.43 of whiting, "Lek i rutan" [S]), which the
  dynamic programme's fish avoid by stopping growth.
- The field optimum needs beta ~0.3-1 (a full fish 1.3-2x the hazard of an empty one) when food gaps
  are 10-20 d. The model learner needs beta 3-7 (D1), because its currency and growth ceiling pay
  for a full reserve and a field fish's do not.

## L. Literature (details, quotes and pages in the lit_*.md trails)

### L1. Predation risk and reserve (`lit_risk.md`, `tables/risk_raw.csv`)

- No study in a target species, or any fish, shows predation risk or escape performance getting worse as
  lipid or liver reserve rises within the natural range [J, ~30 sources looked at, 3 in full text].
- Lean is riskier at the low end:
  - Starved cod (K 0.54 vs 0.81): endurance x0.30; Ucrit and sprint lower (Martinez 2003, 2004 [A]).
    This is at or below the cod starvation line, with muscle wasted.
  - PROXY smolts with a negative mass residual are taken by auklets at RR 1.8-3.0 (Tucker 2016 [F/D]).
  - PROXY damselfish: low-lipid fish taken at RR 3.0 (Booth & Beretta 2004 [A/D]).
  - PROXY sea bass take more risk in proportion to the mass they lost (Killen 2011 [A]).
- In range:
  - Cod K 0.68 -> 0.92, which the authors call the wild spring and autumn levels: Ucrit up, sprint and
    endurance unchanged, cost of transport down, SMR per kg unchanged (Lapointe 2006 [A]).
  - PROXY Pacific herring with twice the lipid: Ucrit unchanged (Sewall 2021 [F]).
  - The one contrary datum is mixed: PROXY predators selected high Fulton K but low total lipid (Hoey &
    McCormick 2004 [A]).
- Meal and gonad mass have measured costs, but they are not reserve:
  - Meal: Ucrit -9 to -15 %, transient over hours to 3 days [A, PROXY].
  - Gonad: -0.37 % Ucrit per % body mass (PROXY mosquitofish [A/D]).
- Where wild fish keep reserves low, the cost named is acquiring them (forgone growth while small: Biro,
  Post & Abrahams 2005 [F, PROXY]), not carrying them.
- The theory papers (Lima 1986, McNamara & Houston 1990, Houston et al. 1993, Witter & Cuthill 1993)
  were not opened. Their mechanism, weight in flight, does not transfer to near-neutral fish [J].
  Inertia in a fast start would transfer; it is unmeasured for lipid.

### L2. Metabolic, buoyancy, drag and storage costs (`lit_cost.md`, `tables/cost_raw.csv`, `tables/derived_out.txt`)

- **Gadoids: k ~ +0.05 RM at fill 1 (0-0.11)** [D on PROXY F].
  - Derivation: liver respiration per g (PROXY porgy, 3.7x the body mean, Oikawa & Itazawa 2003 [F])
    applied to the doubling of non-lipid liver mass from HSI 2 % to 8 % (Davie 2005 cod liver
    composition [F, D]).
  - Gas secretion costs <= 1 % of SMR (Strand et al. 2005 [F]).
  - Buoyancy and drag are each < 0.5 % of RM [D].
  - Cod SMR per kg is unchanged with condition (Lapointe 2006 [A]).
- **Herring and sprat: k ~0 at the surface; -0.15 to +0.04 at depth (central -0.06)** [D on S/A].
  - Fat replaces bladder gas (Brawn 1969 [S]). Wintering herring are negatively buoyant and swim to
    hold depth (Huse & Ona 1996 [A]).
  - A full reserve saves lift at depth.
- **Mackerel: k -0.45 to -0.02 (central -0.21)** [D, order of magnitude].
  - No swim bladder; denser fish tilt more at slow speed (He & Wardle 1986 [A]); bladderless
    scombrid density 1.08-1.09 (Magnuson 1973 [F, PROXY]).
  - The saving reverses in bursts.
- **Storage cycling, all classes** (PROXY):
  - Deposition loses 9-24 % of the lipid energy (Lupatsch 2003 [A]); retrieval ~2 %.
  - This cost is per MJ cycled, not per tick held. It favours a steady reserve, not a low one [J].

### L3. Excess food and the lean spring (`lit_excess.md`, `tables/excess_raw.csv`)

- **Immature cod on satiation feeding show no spring low.**
  - Natural light: HSI 7.3 % (Oct) -> 8.4-8.7 (Jan-Apr) -> ~7.5 (Jul-Oct). Davie 2005 [F; values read
    by eye, D], who notes a "lack of detectable fluctuations" in non-maturing groups.
  - In April, immatures are at 9.3 % HSI against 4.4 % in spawners of the same age.
  - In fill units (library convention: a full gadoid reserve ~ HSI 8 % at 60 % liver lipid) [D]:
    immatures at HSI 9.3 % and ~65 % lipid sit at fill ~1.26, and the natural-light series
    (7.3-8.7 %) at ~1.0-1.2. Excess-food cod sit at or above s = 1 all year.
  - Under continuous light, non-maturing fish dip in summer, the opposite phase to the wild (Karlsen
    2006 [F via fetch summary]).
  - Caveat: one cohort, 4-11 fish per point, sampled 3-monthly; most immature groups were under
    continuous light.
- **Wild cod.**
  - Intake is ~nil Dec-Apr and liver dry weight falls 76-84 %; 81-96 % of the winter loss goes to
    metabolism, 4-19 % to gonads (Schwalme & Chouinard 1999 [A]).
  - Condition is lowest in spring and highest by autumn (Mello & Rose 2005 [A]).
- **Whiting.** HSI 2.6-2.9 % in a poor forage year vs 3.7-6.1 % in a good one, with stomach contents
  60 % lower (Lauerburg 2018 [F]); spawning-stage sampling differs between the years.
- **Six North Sea species.** Seasonal energy-density differences are "obvious in mature fish" (Pedersen
  & Hislop 2001 [A]). The full text, which would give immature vs mature by month, is the top gap.
- **Herring** is the only target species with a cycle under excess food: Berg 2024 ("Ljusstyrd aptit"
  [S]).
- **Model optimum** (a cod model under limited food): immature cod keep "only minor energy reserves"
  (Jørgensen & Fiksen 2006 [F]).
- **Lipostat.** Catch-up growth "is a response to restore lipid levels" (Ali et al. 2003 [A]): a
  defended lower level, in line with the salmonid lipostat already reviewed ("Höstens tillväxt och
  sommarens aptit" 3.1, "Mättnad hos rovfisk" C [S]).
- **Refill.**
  - Wild cod: ~5-6 months to the autumn peak.
  - On excess food: 1-4 months after spawning (Davie: spawning weight loss made up in 1-3 months [F]).
  - That the wild refill is food-limited is [J]: the measures differ.

## Synthesis [J]

1. **No evidenced carrying cost is anywhere near what would lower fill, for either a field fish or the
   model learner.**
   - In the model's units the gadoid k is 0.02-0.03 of k_crit; herring and mackerel have a saving
     (negative k).
   - Beta has no in-range measurement, and its measured sign at the low end is negative.
   - Near-full autumn fills argue against a large beta for herring (0.96) and mackerel (1.0). Gadoid
     autumn fills of 0.71-0.81 are compatible with beta up to ~0.3 in D2, so for gadoids the
     argument against a cost is the absence of evidence (L1, L2), not the autumn level.
2. **The farm and fattening evidence fits.**
   - Fish fatten without limit on easy food because carrying the reserve costs them almost nothing. The
     only regulation found is a lower, defended level (lipostat) plus the herring photoperiod set point.
   - Optimal-reserve theory predicts lean fish only where a carrying cost exists. Fish lack the bird
     cost, so the theory's prediction for fish is "store what you can", limited by acquisition.
3. **The carrying cost is excluded as the reason spring fish are lean.** Acquisition explains part
   of the cycle: cod winter intake is ~nil, and whiting HSI follows forage. Spawning explains part in
   matures. A residual remains open for whiting and the immature gadoids:
   - the whiting SMS ration is flat at 0.42-0.49 of Cmax;
   - "Lek i rutan" item 5 calls the May-Jul trough unresolved: somatic condition recovers in about a
     month, the liver in 3-6;
   - Lauerburg's 25-29.9 cm whiting class went against forage;
   - the dynamic programme's swing is 0.20-0.25 against the field 0.43, and growth timing moves it by
     <= 0.04.
   That residual looks like an allocation order (grow first, refill later), which is choice-like but
   not a carrying cost [J]. Choice in the narrow sense, a cycle under excess food, is shown only for
   herring, through the light-set clock already in the model (section 271).
4. **The model's fill-1 behaviour does not come from a missing cost.** It comes from three things
   (not ranked; D1 splits the learner's marginal value about half currency, half growth):
   - potential intake above need all year, the acquisition gap of "Mättnad hos rovfisk";
   - the currency, which pays a stored MJ like lean tissue (M1);
   - the growth ceiling g (s - u), which pays for a full reserve, while field fish recover somatic
     condition before the liver (M2; "Lek i rutan" item 5) [J].
   A field-sized cost would change nothing (D1, evidenced_sizes item 1).
