# Addendum: validation and behaviour gaps for the encounter-theory attack rates

Follow-up to `summary.md` (2026-10-25). Limited search (15 web searches, 25 page/API fetches, all
spent). Biology-first: nothing was tuned to model output; no `results/` file was read, the simulator
was not run, and no code, library, manifest, resume, existing note or script was edited. Downloads
are in `data/gap_validation/ (in this folder)` (not for commit). Tags: [F] full text/data,
[A] abstract, [S] secondary (search snippet, citing text), [D] computed here by hand, J judgement.

Engine reminder (from `summary.md`): intake = min(eat, phi) x hunger x a B/(1 + a h B + w B_pred), so the half-saturation of one pair is 1/(a h) in t/km2 of visible (accessible) prey,
whatever the eat share, hunger or phi. A field half-saturation is therefore compared with 1/(a h)
directly. A field plateau below Cmax maps onto eat share x hunger x phi, not onto a.

## Bottom line

| Gap | Obtained | Verdict / implied change |
|---|---|---|
| 1 Rindorf & Gislason 2005 | [F] postprint | Half-saturation test **passes but cannot discriminate**: field-equivalent 1/(a h) 0.57 t/km2 of pool (corner span 0.035-7.1) vs derived 0.64 (dispersed 0.30). Type II shape **passes**. Diel: fish eaten "almost exclusively during dawn and dusk", little in darkness -> night/day < ~1, **disfavours the central 1.25**. Gadoid prey half-saturate at ~20x (3-150x) the clupeid catch -> predators -> benthivores a 0.18 looks **~6x high** (field-implied ~0.03; weak). |
| 2 Speeds | Løkkeborg [A]; Onsrud et al. 2005 [S]; Solberg & Kaartvedt 2017 [A] (sprat, not pout) | Whiting ~0.6 BL/s and night speed ~0.7 of day -> predators -> planktivores a x 0.74-0.97 (0.0041-0.0054), night/day 1.25 -> ~1.0. Pout v 0.33 -> ~1.3 BL/s (x3.9), planktivore-pair a +2 %. Cod: night slightly slower, benthos pair unchanged (saturated). |
| 3 Vertical distribution | sprat winter [A]; ICES factsheet [S]; Baltic [S] | Sprat winter day ~50 m: day overlap with the 0-25 m pool lower than 0.45; planktivore-pair a about -1 %. No Skagerrak herring night depth. Albert 1994 not obtained. |
| 4 Herring consumption | context only [S] | Utne et al. 2012 not obtained; no field feeding-level vs zooplankton-density relation found. Unfilled. |
| 5 Hiddink 2016 | [A] only (Dryad 401) | Limitation acts through the prey/consumer ratio (interference and depletion), not through a. The 0.035 t/km2 accessible half-saturation stays untested. |

## Gap 1. Rindorf & Gislason 2005 (whiting functional response)

Source: Rindorf A, Gislason H 2005. Functional and aggregative response of North Sea whiting. J Exp
Mar Biol Ecol 324:1-19. doi:10.1016/j.jembe.2005.04.013. [F] DTU Orbit peer-reviewed postprint
(`data/gap_validation/rindorf2005/postprint.pdf`, `.txt`; served for private study, so quotes are
kept short). Sampling detail from Rindorf 2003, MEPS 249:265 [F] (`rindorf2003.pdf`).

### 1.1 What the paper measured

- Five North Sea locations, early September 1988-89; bottom depth 44, 46, 53, 73, 137 m; bottom
  temperature 16.7, 13.7, 10.6, 8.2, 8.6 C (Table 1) [F].
- Bottom trawl, one haul every 4 h for 48-72 h per location [F]. Haul duration, gear and swept area
  are given in Rindorf 2002 (J Fish Biol 61:579), which was not obtained.
- Predators: whiting 20-24.9, 25-29.9, 30-34.9 cm [F].
- Intake: 48-h stomach series plus the Andersen evacuation model, expressed as P = intake / Wmax. The
  maximum ration Wmax is 2.7-6.1 %BW/d, mean 3.3 %BW/d [F].
- Prey density: geometric-mean trawl catch of prey-sized fish per haul (numbers), averaged over the
  diel cycle. Catchability is unknown, so the density is relative [F].
- Prey sizes: maximum length in stomachs 14 cm for clupeids and 11 cm for gadoids; modal 12-14 cm and
  7-10 cm, matching the trawl modes [F].
- Model: P = K T^beta / (q rho^beta + T^beta). K is the plateau as a share of Wmax. The half-saturation
  (q rho) is in catch per haul. beta = 1 is Type II [F].

### 1.2 Fitted values (Table 2; type II unless marked)

| Prey | Whiting cm | K (share of Wmax) | q rho (catch/haul) | beta | r2 |
|---|---|---|---|---|---|
| sprat | 25-30 / 30-35 | 0.019 / 0.020 | 9.5 / 11 | 1 | 0.76 / 0.59 |
| herring | 25-30 / 30-35 | 0.085 / 0.17 | 36 / 57 | 1 | 0.87 / 0.88 |
| herring III | 25-30 / 30-35 | 0.082 / 0.16 | 15 / 14 | 1.5 / 1.8 | 0.92 / 0.98 |
| clupeids | 20-25 / 25-30 / 30-35 | 0.014 / 0.10 / 0.17 | 159 / 59 / 95 | 1 | - / 0.95 / 0.93 |
| clupeids III | 25-30 / 30-35 | 0.095 / 0.16 | 24 / 20 | 1.4 / 1.8 | 0.97 / 0.99 |
| gadoids | 25-30 / 30-35 | 0.12 / 0.063 | 2907 / 965 | 1 | 0.90 / 0.79 |
| gadoids III | 25-30 | 0.055 | 514 | 1.7 | 0.93 |

Other findings [F]:
- "a saturation level was reached for all types of fish prey". None of the plateaus reached the
  digestive maximum, and no single prey group exceeded 20 % of Wmax.
- Clupeids saturate at 10-17 % of Wmax and gadoids at about 6 %.
- Saturation sets in over a wide range of catches: from 20 herring per haul (the lowest saturated
  catch) up to more than 9000.
- Fish were ingested "almost exclusively during dawn and dusk". Midday intake was low, and
  "Ingestion during darkness was low for all species but whiting at location 5".
- There was no aggregative response; predator and prey catches were slightly negatively correlated.
- The multispecies Type II fitted poorly (r2 0.27-0.45).

### 1.3 Unit conversion to the engine's 1/(a h) [D, three J factors]

Half-saturation in t/km2 of vulnerable clupeids = q rho x w / (A x c):
- q rho: clupeid Type II 59-95 per haul for 25-35 cm whiting, geometric central 75 [F].
- w: weight of one prey, 15 g. The survey length-weight fits (`../Fiskgrupper i rutan/tables/
  length_weight_all.csv`) give herring 9.7-16.3 g at 12-14 cm and sprat 11.8-19.8 g [D].
- A: swept area per haul, 0.09 km2 (0.05-0.18), e.g. about 30-60 min at 3.5-4 kn with a 15-25 m wing
  spread. **J**: gear not obtained.
- c: catch efficiency of the trawl for the column's clupeids, 0.2 (0.05-1). **J**: a bottom trawl
  samples about 5 m of a 44-137 m column.

To the engine's planktivore pool (3.33 t/km2), divide by the share that 25-35 cm whiting can eat:
- herring <= 14 cm is about 0.10 of herring biomass (pooled L_b10 14.25 cm,
  `tables/size_structure_stdout.txt`): 0.79 x 0.10 = 0.079;
- plus sprat 0.03;
- vulnerable share 0.11 (0.08-0.14) [D].

| Corner | vulnerable clupeids t/km2 | pool-equivalent 1/(a h) | implied a (h 279.4) |
|---|---|---|---|
| central | 0.062 | **0.57** | **0.0063** |
| low (q rho 59, A 0.18, c 1, share 0.14) | 0.0049 | 0.035 | 0.10 |
| high (q rho 95, A 0.05, c 0.05, share 0.08) | 0.57 | 7.1 | 0.0005 |

**Verdict, half-saturation: passes, but the test is not discriminating.**
- The central field-equivalent value of a, 0.0063, almost equals the derived central 0.0056
  (1/(a h) 0.64).
- The corner span (0.0005-0.10) contains:
  - the dispersed variant 0.012 (1/(a h) 0.30);
  - the library value 0.04 (0.09);
  - every MC value (0.0019-0.018).
- No value of a can be rejected until the gear's swept area and the clupeid catchability are known
  (Rindorf 2002; Sparholt 1990 is cited for equal herring and sprat catchability).
- The Type III fits (q rho 20-24) would move the central to a ~0.02. That is still inside the span.

**Verdict, shape: passes.**
- The clupeid fits are close to Type II (beta <= 1.8; "only slightly different from a type II").
- The gadoid fits are Type II.
- Both support section 209's Type II on every fish -> fish pair.

**Verdict, diel pattern: the one firm result.**
- Fish intake is concentrated at dawn and dusk. It is low at midday and low in darkness.
- That gives night/day < ~1 for whiting on fish, with the real peak in twilight, which the engine
  cannot represent (convention 6 in `summary.md`).
- The predator note's central night/day 1.25 (schools inside a) is therefore disfavoured. The
  dispersed variant's 0.35, and the library's dark_ratio < 1, are consistent.
- This supports convention 2 (leave schooling to the prey's hide action and visibility floor, use the
  dispersed variant 0.012).

**Ordering test (weak but directional): predators -> benthivores looks too saturated.**
- Gadoid prey reached half-saturation at a ~20x (3-150x) higher catch than clupeids (514-2907 vs
  20-159).
  Rindorf herself declines to conclude because catchabilities differ.
- The same conversion gives a field value of 0.19 t/km2 (0.014-0.97), using w 5 g for 7-10 cm
  gadoids and c 0.5 (0.3-1, J) for demersal juveniles.
- Rindorf's gadoid prey are already prey-sized (<= 11 cm), so the field value compares with the
  derived 1/(a h) on the accessible stock: 0.032 (MC 0.009-0.098).
- Implied a = 1/(0.19 x 176.5) = **0.030** (0.006-0.40), against the derived 0.18 (MC 0.058-0.65):
  about 6x lower, below the MC 5 %.
- At a 0.03, x = a h B_acc = 1.2 instead of 7.3. The pair would move from saturated to the knee.
- Suspect factor: the 2-m near-bed layer of settled juveniles. Rindorf mentions shelter use by
  juvenile gadoids. J.
- Caveat: Rindorf's "gadoids" are juvenile whiting (within-group cannibalism, unchecked in the
  engine), haddock and Norway pout. Pout is in the engine's planktivore pool, so the gadoid fit spans
  two engine pairs and the cannibal pair.

**Plateau K (a training check, not a test of a).**
- K 0.10-0.17 for clupeids and 0.06 for gadoids, with all fish <= 0.2 of Wmax.
- With the engine's phi 0.533 on predators -> planktivores, whiting-like predators should end at
  eat x hunger of about 0.2-0.3 on that pair at saturation [D].
- The plateau fits a twilight time budget: 2-4 h of feeding at about Wmax/24 per hour gives
  0.08-0.17 of Wmax [D, J].
- Saithe (0.41 of the group) were not studied, so this checks whiting-sized stages only.

## Gap 2. Foraging swimming speeds

**Løkkeborg 1998.** Feeding behaviour of cod, Gadus morhua: activity rhythm and chemically mediated
food search. Anim Behav 56:371-378. doi:10.1006/anbe.1998.0772. [A] PubMed 9787028
(`data/gap_validation/pubmed/lok.txt`).
- Eight tagged cod, late spring, northern Norwegian fjord, 3-8 days each.
- Swimming speed was "slightly lower" during a 5-h period at night, "indicating that they searched
  more actively for food during the day".
- Cod detected bait odour "from a distance of several hundred metres" and answered with a rheotactic
  search "at a moderately higher swimming speed".
- The abstract gives no speed numbers.
- Confirms the chemo-tactile search mode and the 0.4 BL/s modal choice (no contradiction).
- The cod night/day of 1.0 (1-1.5) on benthos should lean slightly below 1, J ~0.9.
  - Effect on the annual a: cod is 0.25 of the pair, so 0.25 x 0.05 gives about -1 %.
  - The pair is saturated either way. **No change.**

**Onsrud MSR, Kaartvedt S, Breien MT 2005.** In situ swimming speed and swimming behaviour of fish
feeding on the krill Meganyctiphanes norvegica. Can J Fish Aquat Sci 62(8):1822-. [S] (abstract
text via search snippets; journal from a citing reference list; full text not fetched).
- Acoustic target tracking at a 120-m site of "deep-living fish, mainly Norway pout ... and whiting".
- Speeds were "highest during the day (speeds centred around 14-16 cm/s)" and "decreased somewhat by
  night (modes around 10-11 cm/s)".
- Fish in the upper 10-30 m swam at 16-24 cm/s.
- The authors note a speed vs range artefact and smoothing. Species and lengths are not separated in
  the snippet.

Implied changes (hand arithmetic, J because the species mix is unknown):
- **Whiting:** 15 cm/s at L_num 25.2 cm = 0.60 BL/s, the lower half of the predator note's 0.8
  (0.5-1.3). The night factor is 10.5/15 = 0.7, where the note assumed none.
  - Adult and juvenile whiting carry 0.55 of the predators -> planktivores a.
  - Night factor alone: pair a x 0.87-0.97, giving 0.0049-0.0054. The range depends on the night
    share of whiting's a (0.2-0.8).
  - With the day speed too (x0.75 by day, x0.53 by night): pair a x 0.74-0.86, giving 0.0041-0.0048.
  - Night/day 1.25 -> ~1.04 and dispersed 0.35 -> ~0.29. This fits the Rindorf diel result above.
  - The zooplankton and benthivore pairs, where whiting is 19-68 % of a, scale by the same factor on
    whiting's share; in-range and second order.
- **Norway pout:** 15 cm/s at L_num 11.8 cm = 1.27 BL/s vs the planktivore note's 0.33 BL/s, a 3.9x
  raise on pout's own v (feeding fish, so the speed is not transit).
  - Pout is 0.7 % of the planktivores -> zooplankton a, so the pair a changes by +2 %
    (0.0070 -> 0.0071).
  - As prey of predators, a faster pout raises v_rel only by the Gerritsen-Strickler term (< 10 %).
- **Blue whiting:** no individual speed found; keep 0.33 BL/s (J). It is 0.9 % of the planktivore
  a.

**Solberg I, Kaartvedt S 2017.** The diel vertical migration patterns and individual swimming
behavior of overwintering sprat Sprattus sprattus. Prog Oceanogr 151:49-61.
doi:10.1016/j.pocean.2016.11.003.
- Correction to the task description: this paper is about **sprat**, not Norway pout or blue
  whiting.
- The OA postprint is registered at NVA (Sikt; handle 11250/2478119 -> nva.sikt.no). The download
  API answered "Forbidden" and the Brage NINA mirror failed on DNS.
- [A via search snippet]: four months of a bottom-mounted upward echosounder cabled to shore (inner
  Oslofjord). Sprat mainly used the upper 65 m and schooled at ~50 m by day. They started an upward
  migration ~1 h before sunset and showed a "rise and sink" pattern at depth. Vertical speed differed
  between behavioural tasks.
- No horizontal speed numbers were obtained. The sprat search speed 0.6 BL/s (0.3-1.0) stays.

## Gap 3. Vertical distribution

- **Sprat, winter day** (Solberg & Kaartvedt 2017 [A], above): day schools at ~50 m within the upper
  65 m. With the 0-25 m pool this gives a winter day overlap of ~0.1-0.2 (J) against the
  season-blind 0.45 in `evidence_vertical_overlap.md` 4.2.
  - Sprat is 5.6 % of the planktivore a, and the change covers half the year and half the day:
    pair a about -1 %. Negligible.
  - Caveat: inner Oslofjord is hypoxic at depth, an unusual site.
  - Night: dense layers near the surface after ice-up (related Kaartvedt abstract [S]). Consistent
    with sprat night O 0.8.
- **Norway pout:** the ICES FishMap factsheet [S] says pout are benthopelagic, "frequently in midwater
  off the bottom", progressively deeper with age, with the pelagic 0-group close to the seabed by
  day and in midwater at night. Consistent with the 5-10 m above-bottom layer (Albert 1994 [A],
  already used) and the night rise. Albert 1994 (ICES J Mar Sci 51:45-61) full text was not obtained.
  **No change.**
- **Herring and sprat, Baltic** (Cardinale et al. 2003, Aquat Living Resour 16:283-292,
  doi:10.1016/S0990-7440(03)00007-X): the ALR PDF returned a bot-check page.
  - Snippets [S]: pelagic fish "spread out near the surface at night and gathered near the bottom
    during the day"; larger herring deeper. The same issue suggests the pelagic DVM mirrored the
    zooplankton DVM.
  - Qualitatively the same as the overlap table. **No change.**
- **Skagerrak herring night depth:** nothing beyond Falk 1968 ("upper limit of nightly herring
  traces ... between 0 and 100 m", already used). **Still missing.**

## Gap 4. Herring consumption

- Utne KR, Hjøllo SS, Huse G, Skogen M 2012. Estimating the consumption of Calanus finmarchicus by
  planktivorous fish in the Norwegian Sea using a fully coupled 3D model system. Mar Biol Res
  8:527-547. **Not obtained.**
  - [S]: herring, mackerel and blue whiting together ate ~82 Mt of zooplankton in 1997, of which
    ~35 Mt was C. finmarchicus. No herring-only figure and no biomass were given, so no ration can be
    formed.
  - The VLIZ PDF that was fetched (`data/gap_validation/utne2012/`) is the INFERNO overview (Huse et
    al. 2012, Mar Biol Res 8:415-419), which has no herring ration.
- Bachiller et al. 2018 [S]: three-species mean ~135 Mt/yr (2005-2010). This is already used in the
  planktivore note through Bachiller's 0.6-1 of Cmax.
- Field feeding level vs zooplankton density: none found.
  - Spatial co-occurrence only [S]: Olsen et al. 2005 found herring fullness and zooplankton higher
    in the west; Eliasen et al. 2022 found higher fullness in high-influx years; Dalpadado et al. 2000
    found a higher stomach content in waters with higher zooplankton biomass.
  - None gives a fitted curve. **Unfilled.**

## Gap 5. Hiddink et al. 2016 (trawling and flatfish food limitation)

Hiddink JG et al. 2016. Bottom trawling affects fish condition through changes in the ratio of prey
availability to density of competitors. J Appl Ecol 53:1500-1510. doi:10.1111/1365-2664.12697. [A]
- Kattegat trawling gradient, including a closed area.
- For plaice, prey biomass fell more slowly than fish biomass. Food per plaice therefore rose and
  then fell above about 5 trawl passes per year, and condition and stomach contents followed. Long
  rough dab showed the same pattern; dab showed no effect. Nephrops condition rose as its biomass
  fell.
- Conclusion: "the ratio of the prey to consumer biomass" decides the response.

Mapping (J):
- The limitation runs through prey per consumer, i.e. the engine's depletion of the benthos pool
  (trawl removal d x SAR plus grazing) and the benthivore interference term w (0.45). It does not run
  through a.
- The field data do not give the prey level at which plaice became food-limited.
- The derived half-saturation, 0.035 t/km2 accessible (about 0.17 t/km2 of total benthos at f 0.2),
  therefore stays **untested**. The earlier "potential contradiction" in `benthivores_encounter.md`
  section 3 is neither confirmed nor removed.
- Next step: the Dryad data set doi:10.5061/dryad.04dv4 ("Infauna Kattegat Biomass all stations" and
  otter-trawl catches). The download API answered 401 without a token.

## Still missing

1. Rindorf 2002 (J Fish Biol 61:579): trawl gear, haul duration and swept area. These would cut the
   gap-1 corner span by about one decade. A clupeid bottom-trawl catchability would cut it further.
2. Onsrud et al. 2005 full text: speeds per species and length (whiting vs pout).
3. Any numeric cod or haddock search speed from Løkkeborg 1998 (full text).
4. Solberg & Kaartvedt 2017 full text (sprat horizontal speeds).
5. Albert 1994 full text (height above bottom by size and time of day).
6. A Skagerrak herring night depth.
7. Utne et al. 2012 herring-only consumption, and a field herring feeding level vs zooplankton
   density.
8. Hiddink 2016 prey biomass per station (Dryad), to test the benthos pair's saturation under
   trawling.

## Downloads (`data/gap_validation/ (in this folder)`)

- `rindorf2005/postprint.pdf`, `postprint.txt`: Rindorf & Gislason 2005 [F].
- `rindorf2005/rindorf2003.pdf`, `.txt`: Rindorf 2003 [F].
- `pubmed/lok.txt`: Løkkeborg 1998 abstract.
- `utne2012/v235456.pdf`, `u.txt`: INFERNO overview, not Utne 2012.
- `crossref/`, `openalex/`, `s2/`: lookup JSON.
- `solberg2017/nva.json`: file record, download refused.
- `cardinale2003/`, `hiddink2016/`: failed responses only.
