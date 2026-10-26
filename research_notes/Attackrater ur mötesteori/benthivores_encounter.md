# Attack rates of `benthivorous_fish` from encounter theory

This note derives the Holling attack rate `a` (`max_intake_rate`) of the three `benthivorous_fish`
pairs from encounter theory and species biology. The inputs are search speed, detection width or
reactive distance, capture success, vertical overlap and the number of individuals per tonne.
No model output was used. Nothing in the code, `fg_library.yaml`, `mareld2.yaml` or
`mareld_resume.txt` was changed. The note derives `a` and does not set it.

**Tags.** [F] full text or data read. [A] abstract only. [S] secondary (the citing source is named).
[D] computed here. **J** marks a judgement (no measurement; my choice, stated).

**Files.** The computation is `scripts/benth_encounter.py`, with stdout in
`tables/benth_encounter_stdout.txt`. The size moments are in `scripts/benth_size_moments.py` and
`tables/benth_size_moments_stdout.txt`. Downloads are in `data/benth_flatfish/`,
`data/benth_gadoid_search/`, `data/benth_light_layers/` and `data/benth_validation/`. Run them with
`python3 -I` (see the script headers).

**Engine definition.** Intake per t of predator per 6-h tick is
`min(eat, phi) x hunger x a B / (1 + a h B + w B_pred)`.
- `a` is the area searched per tonne of foraging predator, counting only the share phi that can take
  this prey, per tick, in km2 t-1 tick-1.
- 1 m2 per g of fish equals 1 km2 per t.
- The following stay outside `a`: the time spent foraging (eat share), phi, `accessible_fraction` 0.2,
  prey hiding, interference and h.
- **Correction to the earlier swept-area estimate.** The 0.21 km2/t/tick in
  `../Bentos åtkomlighet för torskfiskar/evidence_B_gadoid_foraging.md` included a factor "fraction of
  time searching" (0.25-1). That factor is the eat share and is dropped here, so the two numbers
  are not directly comparable.

---

## 1. Bottom line

### 1a. Pair values (annual mean, at the layer's annual temperature)

| Pair | `a` central | `a` 5-95 % (MC) | Corners | 1/(a h), central (5-95 %) | Against today's prey | Night/day ratio of `a` | Library now |
|---|---|---|---|---|---|---|---|
| `-> benthic_community` (h 110.7) | **0.26** | 0.12-0.63 | 0.011-2.3 | **0.035** t/km2 accessible (0.014-0.073); **0.18** t/km2 of total stock | accessible f B = 20.3 gives x = a h f B = 570, so 0.998 of the ceiling. At 1 % of today's stock it is still 0.85 | **0.84** (0.71-1.28). Unmodulated is defensible | a 0.04, 1/(a h) 0.23 |
| `-> planktivorous_fish` (h 262.9) | **0.016** | 0.0013-0.069 | 0.00005-1.1 | **0.24** t/km2 (0.055-2.9) | 3.33 all visible: 0.93 of the ceiling (0.54-0.98); 0.89 with interference I = 0.45 x 1.53 = 0.69 in the denominator. At the 0.25 visibility floor (0.83): 0.78 (0.22-0.94); 0.68 with I **0.1-0.35, about 0.2 J**. A static predator in its bottom band gives 0.08 (MC 0.01-0.36); cod that follow prey up at dusk raise it (section 2.3) | a 0.04, 1/(a h) 0.10; dark_ratio 0.6 |
| `-> predatory_fish` (h 193.8) | **0.022** | 0.0018-0.089 | 0.00007-1.4 | **0.24** t/km2 (0.058-2.9) | 1.20 all visible: 0.84 (0.29-0.95); 0.75 with I. At floor 0.25: 0.56 (0.09-0.84); 0.43 with I **about 0.15-0.4 J**. The static corner is 0.12 (0.02-0.62); the pair has no dark_ratio now | a 0.04, 1/(a h) 0.13 |

- MC means a Monte Carlo with every factor log-uniform between its low and high, independently.
- Corners are all factors at low or all at high. They are not plausible combinations.
- For the fish pairs `a` includes the prey's vertical share inside the predator's bottom layer.
  The same-layer `a` before that share is applied is **0.12** (planktivores) and **0.10**
  (predators), with annual-mean shares of 0.13 and 0.21. See section 4, point 2.

### 1b. Benthic pair per member (central; range = MC 5-95 %)

| Member (weight in pair) | N/t | Mode | Search speed v | Width w | Capture p (infauna / epifauna) | a, central (5-95 %) | Night/day |
|---|---|---|---|---|---|---|---|
| Haddock (0.447) | 6 300 | chemo-tactile + short visual | 0.4 BL/s (0.2-0.8), about 9 cm/s at 22 cm | 0.15 L (0.07-0.4) | 0.5 (0.25-0.9) / 0.3 (0.1-0.7) | 0.23 (0.07-0.82) | 1.0 (1-1.5) |
| Cod (0.254) | 2 030 | same | same | same | same | 0.14 (0.05-0.51) | 1.0 (1-1.5) |
| Long rough dab (0.146) | 16 900 | visual pause-travel, mobile prey | 0.12 BL/s (0.04-0.3) | 0.6 L (0.25-1.2) | 0.5 (0.25-0.9) / 0.4 (0.15-0.8) | 0.33 (0.07-1.0) | 0.8 (0.5-1) |
| Dab (0.052) | 16 900 | visual pause-travel | 0.12 BL/s | 0.6 L | 0.7 (0.4-1.0) / 0.4 | 0.55 (0.11-1.5) | 0.5 (0.2-1) |
| Plaice (0.044) | 6 270 | visual pause-travel | 0.12 BL/s | 0.6 L | 0.7 / 0.4 | 0.42 (0.09-1.1) | 0.5 (0.2-1) |
| Witch (0.019) | 5 320 | visual (large optic lobes) | 0.12 BL/s | 0.6 L | 0.7 / 0.4 | 0.51 (0.11-1.4) | 0.5 (0.2-1) |
| Starry ray (0.017) | 1 320 | slow, electro/olfactory **J** | 0.15 BL/s (0.05-0.3) | 0.4 L (0.2-0.8) | 0.5 / 0.3 | 0.17 (0.04-0.47) | 1.0 |
| Grey gurnard (0.010) | 12 000 | walks on free pectoral rays **J** | 0.2 BL/s (0.1-0.4) | 0.3 L (0.15-0.6) | 0.5 / 0.3 | 0.34 (0.11-0.98) | 1.0 |
| Lemon sole (0.010) | 7 970 | visual, roams with head raised | 0.12 BL/s | 0.6 L | 0.7 / 0.4 | 0.44 (0.09-1.2) | 0.5 |
| **Group** | | | | | p weighted 0.92 infauna / 0.08 epifauna (0.04-0.15) of the accessible stock | **0.26 (0.12-0.63)** | **0.84** |

- **Per-tonne `a` hardly depends on size.** Area per fish scales as v w, roughly L^2, so area per
  tonne scales as L^2/W, roughly L^-1. The per-tonne sum of L^2 (M2) is 210-650 m2/t for every member
  (`tables/benth_size_moments_stdout.txt`).
- The member differences come from feeding mode, not from size.
- Hagfish (1.3 %), weever (0.8 %) and the other minor members (2.5 %) have no length data. They are
  left out and the weights renormalised.

### 1c. Fish pairs per member stage (central)

The central inputs are the same for every predator:
- R_sat = 2 L in saturating light.
- R_nonvisual = 0.6 L.
- Light half-saturation Ke = 0.02 lx.
- v = 0.4 BL/s.
- p = 0.3.
- Bottom search band H = 10 m.

| Stage (weight in pair) | N/t | Biomass-mean L | R_sat | v | a, planktivores | a, predators |
|---|---|---|---|---|---|---|
| Cod > 50 cm (P 0.53, W 0.77) | 373 | 68 cm | 1.36 m | 0.27 m/s | 0.014 | 0.021 |
| Cod 30-50 (P 0.23, W 0.16) | 1 774 | 41 cm | 0.81 m | 0.16 m/s | 0.017 | 0.024 |
| Haddock < 40 (P 0.10) | 7 710 | 30 cm | 0.60 m | 0.12 m/s | 0.021 | - |
| Haddock >= 40 (P 0.07) | 1 168 | 46 cm | 0.92 m | 0.18 m/s | 0.016 | - |
| Cod < 30, ray, LRD, gurnard (P 0.07, W 0.07) | 1 200-23 000 | 16-51 cm | 0.3-1.0 m | 0.06-0.2 m/s | 0.015-0.026 | 0.030-0.038 |

- **`a` per tonne is nearly independent of predator size here, but only because R is taken as
  proportional to L.** pi R^2 v then scales as L^3, as W does.
- Visual R to a given prey is set more by prey image size and light than by predator length.
- **Sensitivity [D, arithmetic on the rows below]:** with R fixed at about 1 m for every stage, the
  stages change as follows:
  - Large cod: x 0.54.
  - Haddock < 40 cm: x 2.8.
  - LRD: x 4.5-9.8. This overstates them, because small fish have small eyes.
- The pair `a` becomes 0.023 for planktivores (x 1.46) and 0.020 for predators (x 0.93), both
  inside the MC range.
- The rays are treated like cod (cruising). That is generous for an ambusher **J**, but their weight
  is under 5 %.

**Overlap with the predator's 10 m bottom band** (central; low-high in section 2.3):

| Prey (pool share) | Day | Night |
|---|---|---|
| **Planktivores (pool)** | **0.21** | **0.06** |
| Herring (0.79) | 0.15 | 0.02 |
| Blue whiting (0.11) | 0.3 | 0.15 |
| Norway pout (0.08) | 0.7 | 0.4 |
| Sprat (0.03) | 0.1 | 0.01 |
| **Predators (pool)** | **0.28** | **0.14** |
| Saithe (0.41) | 0.15 | 0.1 |
| Whiting (0.37) | 0.5 | 0.2 |
| Mackerel and horse mackerel (0.12) | 0.05 | 0.02 |
| Spurdog, hake and other demersal (0.10) | 0.3 | 0.2 |

### 1d. Dominant uncertainty, per pair

- **Benthos.** Search width w and search speed v (0.74 and 0.67 decades one-at-a-time, `tables/benth_encounter_stdout.txt` section 4)
  dominate, then capture success (0.5 decades).
- **Benthos is irrelevant to intake today.** Every combination saturates the pair: x is at least 30
  even at the all-low corner. What `a` governs is how far a cell can be depleted before intake falls.
  The answer is to about 1 % of today's accessible stock, so `a` is the depletion floor and not a
  ration lever.
- **Fish pairs.** Four factors carry about 0.6-0.8 decades each: capture success p (no measurement
  for gadoids; 0.78 decades), the deep light profile (0.74), the prey's share near the bottom by day
  (0.65, herring), and v and H (0.6 each).
- **The fish pairs sit near the knee.** 1/(a h) is about 0.24 against 1.2-3.3 t/km2 of prey, so
  unlike benthos, `a` matters for these two pairs.

---

## 2. Derivation

### 2.1 Individuals per tonne and size scaling [D]

`scripts/benth_size_moments.py` reads the long-run (2001-25) NS-IBTS GOV length-biomass table of the
group (`../Fiskgrupper i rutan/benthivores/tables/length_biomass_longrun.csv`; Q1+Q3,
depth-stratified, 1-cm classes).
- It computes N/t = sum_L s_L 1e6/W_L and the per-tonne moments of L, L^2 and L^3.
- Every encounter term is summed per 1-cm class, not taken at a mean weight.

N/t per species and stage:

| Species | N/t | By stage |
|---|---|---|
| Cod | 2 030 | 13 500 (< 30 cm), 1 770 (30-50), 373 (> 50) |
| Haddock | 6 300 | 7 700 (< 40), 1 170 (>= 40) |
| LRD | 16 900 | |
| Dab | 16 900 | |
| Plaice | 6 270 | |
| Witch | 5 320 | |
| Lemon sole | 7 970 | |
| Gurnard | 12 000 | |
| Starry ray | 1 320 | |

The weights per pair are group biomass share x stage share x the pair's phi stage score
(`../Fiskgrupper i rutan/benthivores/scripts/group_params.py`, `SC`), renormalised. `a` is
therefore per tonne inside phi:
- **Benthos:** gadoids 0.70, visual flatfish incl. LRD 0.27, gurnard and ray 0.03.
- **Planktivores:** cod 0.79, of which > 50 cm 0.53.
- **Predators:** cod > 50 cm 0.77.

### 2.2 Benthic pair: 2-D swept area

a = sum over classes of n_L x (v_BL L) x (w_rel L) x p x 21 600 s, in km2/t/tick.

**Search speed v, gadoids.**
- **Ramfjord tracking** (N Norway, 50-70 m, 5-6 C, the Fernö et al. 2011 data set) [F, Winger
  2004 PhD ch. 4]:
  - Hourly means were 0.67-0.90 BL/s (23-39 cm/s for 30-60 cm cod).
  - Modes were 0.1-0.4 BL/s.
  - Quote: "Swimming speeds below 1.0 bl/s were most common (>70%)".
  - "Each fish remained active throughout its diel cycle."
- **Other measurements:**
  - North Sea cod at 0.3-0.9 L/s (Arnold et al. 1994 [F], migration).
  - Newfoundland cod: summer means 10.6-25 cm/s at 6-7 C; winter 2.1-10.7 cm/s at about 0 C [F,
    Winger 2004 ch. 5].
  - Lab Uopt: cod 1.21 BL/s, haddock 1.14 BL/s (Norin et al. 2019 [F]).
- **No in situ haddock speed** was found. Haddock is given the cod values **J**.
- **Chosen:** central 0.4 BL/s (the modal search speed, below the hourly means, which include
  transit), range 0.2-0.8.

**Search speed v, flatfish.**
- Holmes 1979 [F], lab, 8-23 cm, 15 C:
  - The searching gait ("shuffle") "has an upper limit of about 10 cm per second".
  - Pleuronectids "pause frequently to scan around them to locate prey".
  - Plaice feeding on worms were active 59 % of the time; on Corophium, 26 %.
- Hill, Burrows & Hughes 2000-2003 [A]: plaice "search more efficiently for prey when stationary
  than while moving".
- **Chosen:** mean speed over move plus pause of 0.12 BL/s (0.04-0.3), about 3 cm/s at 25 cm **J**.
  Paused scanning is part of searching, so the pauses are inside v and not inside the eat share.

**Width w, gadoids.**
- Brawn 1969 [A], cod in tanks: "Large pieces of food on the bottom were found by sight but smaller
  pieces were taken after being detected by taste buds on the trailing barbel and pelvic fin rays."
  Also: "Food covered by sand was not found. Some live food was captured in the dark."
- Harvey & Batty 2002 [A] report pelvic fins "trailed over the sea bed when searching for prey".
- Detection of small infauna is therefore by contact. The path is the barbel plus the pelvic-ray
  span, about 0.1 L, plus short-range sight or smell of larger epifauna.
- **Chosen:** 0.15 L (5 cm at 35 cm), range 0.07-0.4 L **J**. No measured distance exists.

**Width w, flatfish.**
- de Groot 1969 [F]: "Plaice, flounder and dab are day feeders", "visual feeders, [which] may use
  chemical clues". Visual test models were moved past at about 20 cm. A dab bit a 2-cm ball at
  10 cm.
- Steven 1930 [S via Holmes 1979], lemon sole: "scans the ground with its very prominent and
  moveable eyes".
- Witch has "very large optic lobes" (Evans 1937 [S via de Groot 1969]).
- **Chosen:** w = 2R, with R 10-20 cm from a raised head, giving 0.6 L (0.25-1.2) **J**. No reaction
  distance to siphons was found; Holmes & Gibson 1983 (Anim Behav 31:1244) probably has it but was
  not read.

**Capture success p.**
- Holmes 1979 [F], Table 35, per attack: plaice on worms 100 %, plaice on Corophium 99 %, turbot and
  brill on Crangon 72-73 %, on mysids 84-89 %.
- Holmes adds that "many more incomplete feeding cycles in the sea" would occur.
- Per encounter, I take the following **J** (the gadoid values have no measurement):
  - Flatfish on infauna: 0.7 (0.4-1.0).
  - Gadoids on infauna: 0.5 (0.25-0.9); the prey retract when touched.
  - Epifauna: 0.3-0.4 (0.1-0.8).
- The prey mix is weighted by the accessible stock: epifauna is 0.25 x 6.4 = 1.6 t/km2 of 20.3
  (8 %; 4-15 %) (`../Bottensamhällets epifauna/summary_standing_stock.md`).
- p does not include "too deep". That is the accessible fraction 0.2, already outside `a`.
- **Conflict with the diet data.** The project's diet data show about 28 % crabs, shrimp and
  Nephrops in the fish diet, against about 8 % of the reachable stock
  (`../Bottensamhällets epifauna/summary_standing_stock.md`).
  - So the effective `a` on epifauna is about 4x that on infauna, not the 0.6x that p_epi 0.3
    against p_inf 0.5 implies.
  - The mechanism is detection width: mobile shrimp and crabs are seen, felt by the lateral line or
    smelt at a distance, while a siphon tip is found only by contact.
  - With w x 4-6 on the 8 % epifauna, the pooled `a` rises by less than 30 %, so nothing is re-run.
  - It strengthens the Hiddink 2016 flag (section 3): the preferred epifauna can be depleted while
    the pooled pair stays saturated, and one pooled pair cannot show that.

**Result.** The group `a` is 0.26 (0.12-0.63).
- By mode, gadoids contribute 55 % of `a`, flatfish 24 % and LRD 19 %.
- **Spot check, 35 cm haddock** (390 g): 0.14 m/s x 0.053 m x 0.5 x 21 600 s = 79 m2 per tick,
  i.e. 0.20 km2/t.

**Night/day.**
- **Gadoids: 1 (1-1.5).**
  - North Sea whiting: bottom prey eaten more at night (Rindorf 2003 [A]).
  - Brawn: live food taken in the dark.
  - Cod chemical search day and night (Løkkeborg & Fernö 1999 [A]).
  - All in `../Torskfiskars ljusberoende vid jakt/`.
- **Plaice, dab, lemon sole, witch: 0.5 (0.2-1).**
  - Day feeders: de Groot 1969 [F]; plaice feeding "largely restricted to the daylight period"
    [S via Holmes].
  - Against that, juvenile plaice ate as many items by night (Gibson et al. 1998 [F]).
  - Juvenile plaice attacked siphons in the dark (Burrows, Gibson & Maclean 1994 [A]).
- **LRD: 0.8.**
  - American plaice is "a daytime feeder, and no marked differences were noted over a 24 hour
    period", and fullness peaked at night in fish of 30 cm and over (González et al. 2003 [F]).
- **Group: 0.84 (0.71-1.28) [D].**
- **Depth note:** at 100-300 m the daytime light at the bottom is dim (section 2.3). The visual
  advantage of the flatfish shrinks there, so the group ratio should be nearer 1. Leaving the pair
  unmodulated is supported.

### 2.3 Fish pairs: 3-D encounter in a bottom band

The encounter is computed per predator class, then converted to an area rate:
- Volume rate per fish = pi R(t)^2 v_rel p x 21 600 s.
- v_rel is the Gerritsen & Strickler (1977, J Fish Res Board Can 34:73) mean relative speed, with
  prey at about 0.1 m/s **J**. It adds 0-5 %.
- Prey density in the predator's band = B x F_j(t) / H. So
  **a = sum_classes n_L pi R^2 v_rel p 21 600 x sum_j s_j F_j(t) / H**, averaged over the year.
- The library `a` is the annual mean, as the daylight normalisation requires.

**Light at the bottom [D].**
- **The surface curve** is the same construction as the sibling `scripts/light_at_depth.py`: AMS
  twilight values, Haurwitz-type day values, cloud factor 0.5 **J**.
- **Three Kd profiles** over 0-20 / 20-50 / > 50 m:
  - low K: 0.20 / 0.08 / 0.04
  - central: 0.30 / 0.12 / 0.06
  - high K: 0.45 / 0.15 / 0.08
- **Evidence for the deep K:**
  - Masfjorden 470-500 nm: K 0.192 (0-25 m), 0.092 (25-50 m), 0.060 (> 50 m) (Røstad, Kaartvedt &
    Aksnes 2016 [F]).
  - Aksnes et al. 2009 Eq. 2-3 [F] give Kd(500) 0.024-0.037 for Skagerrak deep water. That is
    derived, with an ASSUMED O2 of 5-6 ml/l.
  - On the clear outer Norwegian shelf the isolume sits about 3x deeper than in front water,
    roughly 1e-3 µmol at 230 m (Kaartvedt et al. 1996 [F]). This fits Kd of about 0.037.
- **Cell depths:** each grid cell's depth from `maps/depth.npz` (mean 167 m, median 142 m).
- **Result, bottom light by day** (median cell; p10-p90 across cells):

| Profile | Daylight hours | June noon | Night |
|---|---|---|---|
| Central | 3e-3 lx (7e-7 to 0.2) | 0.013 lx | about 1e-10 lx |
| Low K | 0.5 lx | | |
| High K | 1e-5 lx | | |

- **Vision near the bottom is dim-light limited all day.** The gadoid half-detection light is
  0.01-0.05 lx (`../Torskfiskars ljusberoende vid jakt/gadoid_vision_thresholds.md`, Q4).
- **Effective threshold.** In the central profile the median cell never reaches 0.02 lx. In the
  low-K profile it reaches it at about +1 deg of sun elevation. The effective threshold for this
  predator in this place is therefore >= 0 deg, not -5. [D]

**Reactive distance R(t)^2.** R(t)^2 = R_sat^2 [q + (1 - q) E/(E + Ke)], with q = (R_nv/R_sat)^2.
This is the saturating form of Aksnes & Utne (1997) [S via the vision note].
- **R_sat = 2 L (1.5-3).**
  - Lindholm, Auster & Kaufman 1999 [F]: a 38-40 cm cod reacted to 8.4 cm 0-group cod at 60 ± 16 cm
    over sand under lab lights. That is about 1.5 L and capped by the 125 cm arena, so a lower
    bound.
  - Fall & Fiksen 2020 [F] use R = 2 m for cod on capelin.
- **R_nv = 0.6 L (0.3-1.0) J.** This is the lateral-line and olfactory range in darkness. No gadoid
  measurement was found. Meager et al. 2005 [A]: chemical cues detected "from significantly longer
  distances" than vision, and predation was "significantly lower in dark conditions".
- **Ke = 0.02 lx (0.005-0.1) J**, from the vision note.
- **Central result:** mean R^2/R_sat^2 is 0.36 by day and 0.09 by night.

**Speed and capture.**
- v = 0.4 BL/s (0.2-0.8), as for benthos.
- **p = 0.3 (0.1-0.6) J.** No cod or saithe capture success per attack on fish was found:
  - Pitcher et al. 1996 [A]: cod and haddock attacks on herring schools caused "frequent
    modifications to, but not dispersal of, school structure"; no rates.
  - Ellis & Gibson 1997 [F] give capture efficiency only in a figure.
  - Holmes's 72-73 % for turbot on Crangon is a per-attack upper bound for a visual ambusher.

**Band thickness H = 10 m (5-20).**
- Stensholt & Stensholt 2004 [F]: 19-34 % of tag depths within 2 m of the daily maximum depth.
- Aglen et al. 1997 [F]: "All size groups of cod generally seemed to stay on the bottom both day
  and night". Large haddock reach up to 100 m above the bottom by day.
- Righton et al. 2000 [F]: North Sea cod mostly on the seabed in June-July, leaving it at nightfall
  in August-September.
- Engås & Godø 1986 [F]: "largest concentrations were near the bottom", with fish up to about 50 m.

**Prey shares inside the band.**
- **Herring:**
  - Danish HERAS logs (sibling script `heras_herring_depth.py`, `tables/heras_herring_depth_stdout.txt`
    [D]): share of herring NASC in the bottom 10 m was 0.34 (box) / 0.26 (region) with the sun above
    6 deg, 0.13-0.14 at low sun, and 0.01-0.04 in twilight.
  - Those logs average about 90-120 m bottom depth. The grid's median is 142 m and herring stay
    mostly above 100 m (`../Ekolodsdata planktivorer/tables/nasc_depth_profile_box.csv`, 1-5 % at
    100-150 m). So day 0.15 (0.05-0.34) and night 0.02 (0.005-0.05) **J**.
  - HERAS integrates by day only (Solea 764 cruise report [F]: "clupeids disperse at night").
  - Schools dispersed at sunset in the Oslofjord (Onsrud & Kaartvedt 1998 [F]).
- **Norway pout:**
  - Norwegian Deep: "generally 5-10 m above the sea bed, but somewhat closer by day"; feeding is
    more intense at night (Albert 1994 [A]).
  - It stays in the benthic boundary zone by day in clear water (Kaartvedt et al. 1996 [F]).
  - In the Oslofjord its night maximum was in midwater (Onsrud et al. 2004 [S]).
  - Day 0.7 (0.4-0.9), night 0.4 (0.1-0.7) **J**.
- **Blue whiting:**
  - It lives in the deep scattering layer, with dense shelf-break aggregations near the bottom day
    and night (Torgersen et al. 1997 [F]; Bergstad 1990 [A]).
  - Day 0.3, night 0.15 **J**.
- **Sprat:** midwater by day and in the upper 50 m at night (Onsrud 2004 [S]). Day 0.1, night 0.01
  **J**.
- **Predatory fish:** these are judgements on habitat (whiting benthopelagic and nearer the surface at
  night in the Oslofjord; saithe and mackerel pelagic). No 0-1-group heights were found. Bromley et
  al. 1997 [F]: "Few fish were caught near the bottom" of the 0-group.
- **Prey day/night switch:** at sunset (elevation 0), following the HERAS twilight logs. The
  engine's day is elevation > -5 deg.

**Night/day ratio of `a`, planktivore pair: central 0.08.**
- The vision part (R^2 night/day) is 0.25 and the overlap part (F night/day) is 0.30.
- One-at-a-time (section 4 of stdout), the ratio spans 0.02-0.24. It is highest with high-K
  (dark) water, where vision matters little, and with a large non-visual R.
- **The 0.6 on the pair is not supported by this mechanism.**
- **But the static band model omits two field signals:**
  - Cod stomachs hold herring with two peaks, morning and evening (Adlerstein & Welleman 2000 [S]).
  - Gadoids catch sprat during its dusk ascent in fjords (Kaartvedt et al. 2009 [A]).
  - Both are crepuscular encounters of prey crossing the band. The predator also follows its prey:
    - North Sea cod "left the seabed at nightfall and returned to it before dawn" in Aug-Sep
      (Righton et al. 2000 [F]).
    - "When cod were pelagic they often mixed with capelin" (Michalsen et al. 1996 [F]).
  - If cod follow dispersed herring up, the night overlap is the dilution of herring over a thicker
    layer. That is about 0.5 rather than the static 0.3, and in Aug-Sep it may exceed 1.
  - **Defensible range: 0.1-0.35, about 0.2 J.** The static 0.08 is the corner case.
  - This sits between the static model and the diel-stomach note's 0.5-1.

**The annual-mean `a` is robust to the same objection.**
- Bottom band: the herring F/H is 0.15/10 m = 0.015 per m.
- A cod inside a uniform 50-70 m herring layer meets 1/50-1/70 = 0.014-0.020 per m.
- So the central `a` hardly depends on whether cod wait at the bottom or swim up into the layer.
  Only the day/night split does.

**Predator pair:** static central 0.12 (0.02-0.62); with predator following, about 0.15-0.4 J. Whiting and saithe also leave the bottom at night.

---

## 3. Validation (not used to set `a`)

| Datum | Value, converted | What it tests | Verdict |
|---|---|---|---|
| Persson et al. 2012 [F], MEPS 456:245; 1-yr cod, about 18 cm, 8-9 C, Type II on sand | a = 1.8 (Crangon), 21 (Palaemon), 57 (Neomysis) cm2/s, i.e. 3.9 / 45 / 123 m2 per fish per 6 h. At about 55 g that is 0.07 / 0.82 / 2.2 km2/t/tick | Gadoid benthic `a` on epibenthic crustaceans. Per-tonne `a` falls with size, so this is an upper bound for adults | This note's model gives 0.37 for an 18 cm cod (20 m2 per fish per tick): inside the measured span. **Consistent** |
| Holmes 1979 [F], Table 33, plaice at about 194 prey/m2 | inter-attack intervals give 0.33-2.1 m2/h per 8-23 cm fish, i.e. 2-13 m2 per tick, about 0.07-0.4 km2/t for a 15 cm plaice | Flatfish `a`. This is a lower bound: at this density the time is mostly handling | Model: about 4 m2/h for a 15 cm plaice. Above the lower bound; **consistent** |
| Persson 2012 h; Ellis & Gibson 1997 [F] (15-30 s per fish prey) | behavioural 1/h is about 440x Cmax | That h is digestion-limited, not handling-limited | Supports the engine's Cmax-based h |
| MULTSPEC, Bogstad, Hauge & Ulltang 1997 [F] | half-saturation G3 = 1.57 t/km2 of suitability-weighted food; effective a = Hmax/G3, about 0.002 km2/t/tick for a 1 kg cod at 4 C, fitted on areas of about 10^2 km | Half-saturation of cod on fish | This note gives 0.24 t/km2 at 1 km. The field value is 6-7x higher. That is plausible from patchiness at the 10^2-km scale and stays within the MC 5 % of `a`. A **soft warning** that 0.016 may be high for a pooled, patchy pool |
| Fall & Fiksen 2020 [F] (model) | beta = pi(R sin theta)^2 v with R 2 m: about 0.27 km2/t/tick in a 50 m capelin layer. Cod "gut-limited, not encounter-limited" | The same-layer `a` | The same-layer `a` here is 0.12 per 10 m band, i.e. 0.024 per 50 m. Theirs is about 10x higher (full-light R, v 0.5 m/s, p = 1). Same structure; the gap is light and capture |
| Fall et al. 2021 [F], Barents cod | consumption "increased sharply with capelin density before levelling off"; median 0.73 % BW/d | Fish pair near saturation at low density | Consistent with 1/(a h) of about 0.2 against 1-3 t/km2 |
| Hall 1987 [A] | wild haddock feed at 15-22 % of the lab ad-lib maximum | Realised ration vs ceiling | The engine's benthos pair is saturated while foraging (section 1). The realised 15-26 % (the group's Q/B 3.48 is 26 % of the ceiling, params note) must come from eat share, hunger and interference, **not from `a`**. Check this after training |
| Johnson et al. 2015 [F], Irish Sea | plaice and dab "maintained constant levels of stomach fullness" over a trawling gradient that cut prey biomass | Saturation of the benthos pair | **Consistent** |
| Hiddink et al. 2016 [A], Kattegat | food per plaice and the condition of plaice and LRD fell above about 5 trawl passes per year | Whether benthos intake becomes food-limited | **Potential contradiction.** With 1/(a h f) = 0.18 t/km2 total, intake would stay saturated until preferred prey fell below about 1 t/km2. The field limitation implies the accessible preferred prey falls further, or `a` is lower, under heavy trawling. Worth a test with the trawl maps |
| Duineveld & Van Noort 1986 [A] | dab eat 0.84 g/m2/yr of Amphiura arms (6 % of arms) | Benthos M2, not `a` | n/a |
| Braaten & Gokstad 1980 [F] (ICES CM: cite only with the authors' consent) | ad lib cod 1.9 % BW/d at 4-5 C, 2.5 % at 6-7 C | The ceiling h | as in the params note |

---

## 4. Judgements, conflicts with engine conventions, dominant uncertainty

1. **Benthos `a` is about 6x the library value.** Mechanistic 0.26 (0.12-0.63) against 0.04. The
   benthos pair is saturated under either value (0.998 against 0.98 of the ceiling at today's
   stock). The half-saturation in total stock falls from 1.2 to 0.18 t/km2. Raising `a` changes
   only how deep cells can be grazed down. The 0.04 also fell inside the old 0.007-5.8 range by
   coincidence.
2. **Fish pairs: the prey's vertical share has no home in the engine.**
   - The engine has no `accessible_fraction` on the fish pairs, so the share of the pool near the
     bottom (0.13 / 0.21 annual mean) is folded into `a` here.
   - The cleaner convention is `a` = same-layer 0.12 / 0.10 with `accessible_fraction` 0.13 / 0.21.
     That gives the same half-saturation and is how benthos is already handled.
   - **Double-count check.** phi's stage scores are "access x capability". The cod scores for this
     pool were set mainly on gape (capability). If they also discounted cod for herring being in
     midwater, folding F in here counts that twice.
3. **The fish pairs are below the library value and near the knee.** Planktivores 0.016 against
   0.04, predators 0.022 against 0.04. 1/(a h) is about 0.24 against 0.10 / 0.13 now.
   - With interference in the denominator (I = 0.69 at 1.53 t/km2), full prey visibility gives
     0.89 / 0.75 of the ceiling.
   - At the prey's visibility floor it gives 0.68 / 0.43, so the pair is not saturated.
4. **`dark_ratio` 0.6 at -5 deg is a 30-50 m construct.**
   - At the grid's bottom (median 142 m), daytime light is below the gadoid half-detection level
     for most of the day and year.
   - The step to "day" for this predator lies at a sun elevation >= 0 deg (central profile: never),
     not at -5.
   - The evidence gives night/day of about 0.1-0.35 (central about 0.2 J) for planktivores and
     about 0.15-0.4 for predators.
   - The static-band corners are 0.08 and 0.12. About half of the static value is vertical overlap
     (herring and whiting leave the bottom at dusk), not vision.
   - Cod following prey up at night (Righton 2000, Michalsen 1996) and crepuscular capture lift it.
   - Either way it is below the inherited 0.6.
   - A logistic in sun elevation with a 2-deg width cannot represent a depth-dimmed daylight
     factor. A threshold of about 0 to +5 deg with a low ratio is the closest fit **J**.
   - The predator pair has no light dependence now; the evidence says it should.
5. **Benthos stays light-independent.** Group 0.84 (0.71-1.28): gadoids feed by contact at night as
   much as by day, and flatfish lean diurnal. Leaving the pair unset is within the evidence.
6. **Temperature.**
   - The speeds come from 5-6 C (Ramfjord), close to the bottom water and below the at_50m annual
     mean of 8.4 C.
   - Winter speeds at about 0 C are 2-5x lower (Winger 2004). No speed swing over 6-12 C was found.
   - `a` scales with v, so an `ingestion_q10` would move `a` and 1/h together. The params note
     suggests about 2 for Cmax.
7. **The numbers are judgement in places.** Capture success (every pair), the search width of
   gadoids on benthos, the flatfish reaction distance, the non-visual R and all prey band shares
   except herring (HERAS) have no direct measurement. The ranges carry this.

**Dominant uncertainty.**
- **Benthos:** gadoid search width (barbel/pelvic-ray contact vs short-range vision), about 0.7
  decades. It is irrelevant to intake today.
- **Fish pairs:** cod capture success on fish (unmeasured, 0.8 decades) and the deep light profile
  (0.7 decades), then the daytime near-bottom share of herring.

---

## 5. References

**Gadoid search, vertical position**
- Aglen A, Engås A, Godø OR, McCallum B, Stansbury D, Walsh SJ 1997. ICES CM 1997/Y:16 [F]
  (`data/benth_gadoid_search/aglen1997_icescm_Y16.pdf`).
- Arnold GP, Greer Walker M, Emerson LS, Holford BH 1994. ICES J Mar Sci 51:207-232 [F].
- Brawn VM 1969. J Fish Res Board Can 26:583-596, doi:10.1139/f69-054 [A].
- Engås A, Godø OR 1986. J Northw Atl Fish Sci 7:35-42 [F].
- Hjellvik V, Godø OR, Tjøstheim D 2004. ICES CM 2004/R:33 [F]; CJFAS 61:2237 [A].
- Lindholm JB, Auster PJ, Kaufman LS 1999. MEPS 180:247-255, doi:10.3354/meps180247 [F].
- Meager JJ et al. 2005. CJFAS 62:1978-1984, doi:10.1139/f05-104 [A].
- Michalsen K, Godø OR, Fernö A 1996. ICES J Mar Sci 53:389-395 [F].
- Norin T et al. 2019. PeerJ 7:e7784, doi:10.7717/peerj.7784 [F].
- Pitcher TJ et al. 1996. ICES J Mar Sci 53:449-452, doi:10.1006/jmsc.1996.0063 [A].
- Righton D, Turner K, Metcalfe JD 2000. ICES CM 2000/Q:09 [F].
- Stensholt BK, Stensholt E 2004. In: GIS/Spatial Analyses in Fishery and Aquatic Sciences,
  pp. 109-126 [F]; Stensholt, Michalsen & Godø 2000, ICES CM 2000/K:34 [F].
- Winger PD 2004. PhD thesis, Memorial Univ. Newfoundland, ch. 4-5 (Fernö et al. 2011 Mar Biol Res
  7:310 data) [F].

**Flatfish**
- Burrows MT, Gibson RN, Maclean A 1994. J Fish Biol 45(Suppl A):171-180 [A]; Burrows MT et al.
  1994 MEPS 115:31 [F]; JEMBE 177:251 [A].
- de Groot SJ 1969. J Cons int Explor Mer 32:385-395 [F].
- Gibson RN et al. 1998. MEPS 165:145-159 [F].
- González C, Román E, Paz X 2003. NAFO SCR Doc 03/23 [F].
- Hill S, Burrows MT, Hughes RN 2000, 2002, 2003. J Fish Biol 56:1497; 61:1255; 63(Suppl A):117 [A].
- Holmes R 1979. Studies on the feeding behaviour of flatfish. PhD thesis, Univ. Stirling,
  hdl 1893/35071 [F]. The thesis restricts publication of derived information without the author's
  consent; used here internally.
- Johnson AF et al. 2015. Proc R Soc B 282:20142336 [F].
- Ntiba MJ, Harding D 1993. Neth J Sea Res 31:189 [A].

**Light, prey layers**
- Aksnes DL et al. 2009. MEPS 387:39, doi:10.3354/meps08120 [F].
- Aksnes DL, Utne ACW 1997. Sarsia 82:137 [S via the vision note].
- Albert OT 1994. ICES J Mar Sci 51:45, doi:10.1006/jmsc.1994.1005 [A].
- Bergstad OA 1990. Neth J Sea Res 25:237 [A].
- Bromley PJ, Watson T, Hislop JRG 1997. ICES J Mar Sci 54:846 [F].
- Kaartvedt S et al. 1996. MEPS 136:51, doi:10.3354/meps136051 [F]; Kaartvedt, Røstad & Klevjer
  2009, MEPS 390:237 [A].
- Onsrud MSR, Kaartvedt S 1998. MEPS 171:209 [F]; Onsrud et al. 2004, ICES J Mar Sci 61:1278 [S].
- Røstad A, Kaartvedt S, Aksnes DL 2016. Deep-Sea Res I 113:1-6, doi:10.1016/j.dsr.2016.04.014 [F].
- Solea cruise 764 report 2019 (HERAS manual practice) [F].
- Torgersen T et al. 1997. Sarsia 82:87 [F].
- Sibling outputs in this folder: `tables/heras_herring_depth_stdout.txt`,
  `tables/light_at_depth_stdout.txt` [D].

**Validation**
- Bogstad B, Hauge KH, Ulltang Ø 1997. J Northw Atl Fish Sci 22:317 [F].
- Braaten B, Gokstad SL 1980. ICES CM 1980/F:20 [F].
- Duineveld GCA, Van Noort GJ 1986. doi:10.1016/0077-7579(86)90064-5 [A].
- Ellis T, Gibson RN 1997. MEPS 149:83 [F].
- Fall J, Fiksen Ø 2020. Fish Fish 21:63, doi:10.1111/faf.12415 [F]; Fall J et al. 2021, Ecography
  44:802 [F].
- Hall SJ 1987. J Fish Biol 31:479 [A].
- Hiddink JG et al. 2016. J Appl Ecol 53:1500 [A].
- Kempf A, Floeter J, Temming A 2008. MEPS 367:295 [F] (shape only); Rindorf A, Gislason H, Lewy P
  2006, MEPS 325:243 [F] (relative switching, no absolute a).
- Ljungberg P et al. 2012. MEPS 466:215 [F].
- Persson A et al. 2012. MEPS 456:245, doi:10.3354/meps09752 [F].

**Method:** Gerritsen J, Strickler JR 1977. J Fish Res Board Can 34:73-82 (relative-speed encounter
formula, standard form) [S].

**Not obtained:**
- Holmes & Gibson 1983.
- de Groot 1971.
- Løkkeborg 1998 speeds.
- Meager 2010 values.
- Any gadoid lateral-line range or capture success on fish.
- Albert 1994 full text.
- Hiddink 2016 prey-biomass levels.
