# Literature-based growth_rate and natural_mortality for every Mareld FG

Status: **implemented** (2026-10-03) in `fgconfig/fg_library.yaml`,
`lib/world/growth_budget.py`, the loader and the FG editor (section 6). It
supersedes the earlier proposals (see "Superseded" at the end).

The main change is structural: **`growth_rate` is no longer a free parameter.** It
is derived from a per-FG literature budget plus a configurable predation matrix.
Checking a predator in the FG editor raises the prey's growth_rate by that
predator's predation share; unchecking lowers it by the same amount.

The hunger gate is **h = min(1, (1 - s)/(1 - u))** for every FG (full appetite up to maintenance, section 5 and `reports/Zooplankton energibudget.md`); `satiation_scale` is removed, so the maximum gate
(s - u) is 0.5 and g = gross P/B / 730.

Provenance tags: **[V]** value read in the primary source, **[S]** value seen only in
a search snippet or abstract (unverified), **[C]** computed for this report,
**[M]** model fact verified in the code.

## 1. Values in the library

growth_rate as the loader derives it for `mareld2.yaml` (seals and seabirds muted,
so their predation is not paid for).

| FG | r_max (1/yr) | **natural_mortality** M1 per tick (per yr) | checked predation on it (1/yr) | **growth_rate** per 6 h tick | Before g / M1 / K |
|---|---|---|---|---|---|
| phytoplankton | 75.2 | - | zooplankton 22.63 | **0.0670** (0.055-0.067) | 0.125 / - / 24 |
| zooplankton | 33.4 | **0.0134** (19.7) | pelagic_fish 5.24 | **0.0799** (0.068-0.118) | 0.0913 / 0.0025 |
| benthic_community | (proposal 0.82; **reverted**) | - | (gadoids 0.0048, inert) | **0.001** hand-set, **K 10** (proposal 5.65e-4, K 100) | 0.001 / - / 10 |
| pelagic_fish | 0.648 | **3.97e-4** (0.580) | gadoids 0.300, porpoises 0.012 (+ seals 0.008, seabirds 0.024 when unmuted) | **0.00211** (0.0019-0.0026) | 0.004 / 1.0e-4 |
| gadoids | 0.511 | **4.68e-4** (0.683) | porpoises 0.056 (+ seals 0.030 when unmuted) | **0.00171** (0.0015-0.0021) | 0.0025 / 7.0e-5 |
| porpoises | 0.10 | **6.85e-5** (0.10) | - | **2.74e-4** | 0.001 / 6.0e-5 |
| seals | 0.11 | **6.85e-5** (0.10) | - | **2.88e-4** | 0.001 / 6.0e-5 |
| seabirds | 0.10 | **1.03e-4** (0.15) | - | **3.42e-4** | 0.001 / 8.0e-5 |

Other changes in the library:
- `satiation_scale` removed from the model, the GUI and the library. It was 0.8 by
  default, and 1.37 for porpoises; the gate is now h = 1 - s.
- Porpoise `feeding_cost`: 1.4 -> 1.1 under h = 1 - s (section 5), and back to 1.4
  when the appetite became full up to maintenance (2026-10-03, section 133 of the
  resume).
- Phytoplankton `energy_content` 2000 -> 4800 MJ/t (Ecopath convention; see
  `reports/Zooplankton energibudget.md`).
- Benthos: K 10 -> 100 t/km2 with the spawn scaled x10 was applied, then **reverted
  on 2026-10-03 at the user's request** (resume section 136). Benthos is back to
  its pre-session state: hand-set growth_rate 0.001, no r_max (so not derived),
  K 10, spawn 12,000-20,000 t (inference 16,000). Section 3.3 is kept as the
  literature proposal.

Variants (gadoids, not applied):
- adults only: r_max + M = 1.21, so g 0.0017;
- including Norway pout and blue whiting: g 0.0027.

The three findings that matter most:
- **Mortality, not growth, is where the library was furthest off.** M1 rose 4-7x
  for the fish and 5x for zooplankton. Most natural mortality of gadoids comes from
  sources Mareld does not model (M0, other fish, crabs); that part stays in M1.
- **Mammals and seabirds fall to 0.27-0.34x of their old g.** Two independent routes
  agree within about 10 % on the gross P/B: r_max + M, and age-structured production
  budgets.
- **Benthic K was 7-15x too low.** The benthos r was in range.

## 2. Definitions (decided by the user)

### 2.1 Decision makers

- **growth_rate** is the maximum growth under ideal conditions, taken from the
  literature. It includes recruitment and individual growth, and excludes natural
  mortality:

      gross P/B (per yr) = r_max + M1 + sum of checked M2_ij
      g = gross / ((s-u)_max * 1460)

  - r_max is the literature maximum net population growth at low density. It is
    measured net of all natural mortality, which is why the mortality is added back.
  - **M1** (`natural_mortality`) holds every loss that is not predation by another
    FG: M0, predation by groups that are not FGs (sharks, rays, flatfish, other fish,
    carnivorous zooplankton, larvae), and predation by NDMs, which cannot hunt in the
    tick (crabs eating juvenile gadoids). Per tick = 1 - exp(-M1 / 1460).
  - **M2_ij** (`predation_mortality` on the pair) is the literature predation
    mortality of prey i caused by FG j (section 4). It counts while the pair is
    checked, the predator is a decision maker, and the predator is active (not
    muted) in the project.
  - Ideal conditions mean a full reserve, so (s - u)_max =
    1 - u = 0.5 for every DM.
  - Net growth at low density is always r_max, whatever the matrix looks like:
    removing a predator removes both its mortality and its growth compensation.

### 2.2 Non decision makers (logistic, no separate mortality term)

- **growth_rate (r)** is growth in Kattegat/Skagerrak net of natural mortality and
  of predation by predators that are not modelled. It is **gross of the predation
  the model applies itself** (zooplankton on phytoplankton, gadoids on benthos):
  r = (r_max + sum of checked M2) / 1460.
- r is the logistic per-capita rate at low biomass, not the mean realised rate.
  - Near equilibrium the mean net rate r(1 - B/K) tends to zero for any stable
    population.
  - Equating r with that mean (an intermediate version of this proposal) gave
    benthos r = 4e-6 per tick, which is meaningless. That version is withdrawn.
  - Benthos r is therefore the observed low-density biomass recovery rate.
  - Phytoplankton r is likewise the highest in situ growth, net of unmodelled
    losses (section 3.1). An annual-mean version (0.0194) was withdrawn for the
    same reason as the benthos one.

### 2.3 What each FG contains

- **zooplankton = mesozooplankton.** Sources:
  - `fg_library.yaml` lines 23-24.
  - Resume section 124.1: a mixed Skagerrak copepod assemblage, 2/3 small neritic
    copepods and 1/3 Calanus.
  - Microzooplankton is not represented, so its grazing is an unmodelled loss of
    phytoplankton.
  - Pelagic fish take about 70 % of their diet as mesozooplankton and about 5 % as
    protozoa [C, NS Ecopath diet matrix], which supports this definition.
- **pelagic_fish and gadoids** include every species under the umbrella term that
  eats the foods the library gives the FG.
  - pelagic_fish eats zooplankton.
  - gadoids eat pelagic_fish and benthic_community.

## 3. Derivations per FG

### 3.1 Phytoplankton: r = 0.067 per tick

The logistic r is the specific growth at LOW biomass, net of the losses the model does
not resolve, and gross of mesozooplankton grazing (the zooplankton pair).

Low-density growth:
- Dilution experiments measure in situ phytoplankton growth separately from
  microzooplankton grazing. The highest rate measured was **0.67 d-1** (range
  0.13-0.67, highest offshore) [V, Stelfox-Widdicombe et al. 2004, southern North Sea,
  April, 7-9.5 C].
- Microzooplankton grazing, 60 % of daily growth in coastal waters [V abstract,
  Calbet & Landry 2004], is not represented in the model and is removed.
- 0.67 x 0.40 = **0.27 d-1 = 0.067 per tick = 97.8 /yr** [C].
- The zooplankton pair supplies 20 % x 0.31 d-1 = 0.062 d-1 = **22.63 /yr** of it
  (mesozooplankton share of mean PP: North Sea Ecopath 20.1 %, Calbet 2001 22.6 %).
  So **r_max = 97.8 - 22.63 = 75.2 /yr**, and with the pair checked
  r = (75.2 + 22.63) / 1460 = **0.0670 per tick**.
- Sinking and lysis are not subtracted. Removing 10-20 % gives about **0.055**, the
  low end of the range.

Why not the annual mean (withdrawn):
- The earlier value, 0.0194, took annual 14C PP over mean biomass in Kattegat/
  Skagerrak, 190-290 g C m-2 yr-1 over 2.16 g C m-2 = 0.31 d-1 [V, Richardson &
  Christoffersen 1991; Heilmann et al. 1994; Lindahl et al. 1998; Scotti et al.
  2022]. From that it subtracted every loss.
- What is left is a steady-state remainder. A logistic population near K realises
  r(1 - B/K), not r, so the remainder is close to zero by construction. This is the
  same error that gave benthos 4e-6.
- It cut the maximum supply rK/4 to 0.12 t/cell/tick, against 0.40 now and 0.75 with
  the old 0.125. In the closed phyto-zoo test, zooplankton fell to 0.027x in 200 ticks.

Diagnostic only (the values were not chosen from it): the same test gives
zooplankton 1.07x (minimum 0.70x) at 0.067, and 2.04x at 0.125.

Caveats:
- Single source, southern North Sea in spring, not Kattegat/Skagerrak. No dilution
  study from Kattegat/Skagerrak was found.
- The model has no seasonal forcing (`seasonal_amplitude` 0), so spring-like growth
  applies all year. Setting the amplitude from the literature is the companion step.
- K = 24 t/cell was not reviewed.

### 3.2 Zooplankton: g = 0.0799, M1 = 0.0134 per tick

growth_rate (ideal conditions = food-unlimited):
- The copepod weight-specific growth rate is g = 0.0445 e^(0.111 T) d-1 [Huntley &
  Lopez 1992; coefficients via secondary source, S]. It includes egg production.
- At 10 / 11.5 / 15 C that is 0.135 / 0.16 / 0.235 d-1.
- Per year that is 49 / 58.4 / 86 /yr. The budget is additive, so it is written
  as r_max = 58.4 - M (25) = **33.4 /yr**; with M1 and the pelagic-fish pair the
  gross is 58.3 /yr, giving g = **0.0799** (0.068-0.118) [C].

Mass-balance bound:
- Under the default mass-balanced growth, g can never exceed
  max_energy_reserve / energy_content = 675 / 4500 = **0.150** [M]. Section 124 raised
  the reserve from 450 to 675; the 0.100 quoted in section 115.3 is the old value.
- The bound does not depend on the gate (cap and wish are both linear in s - u), so
  it stays 0.150 under h = 1 - s. The whole range (0.068-0.121) is below
  it; the central value uses 55 % of the bound. Under today's 0.8 the same biology
  needs g 0.114-0.202, and the 15 C value would be capped.

natural_mortality:
- Total copepod mortality at steady state equals realised P/B, about 20-30 per yr.
  This is the best-supported band [C from Peterson et al. 1991 V, Hirst & Bunker 2003
  S, Mackinson & Daskalov 2007 V]. The 15-40 band quoted in section 111 has no source.
- In the NS Ecopath model the expanded pelagic pool eats 20 % of copepod production,
  or 23 % if carnivorous zooplankton is included [C, section 4].
- The FG predators take 21 % (pelagic fish 20.9 %, gadoids 0.36 %, seabirds 0.04 %),
  so M1 = 25 x 0.787 = **19.7 per yr** (15.7-23.6) = **0.0134 per tick**.
  Today's value is 3.655 per yr.
- The rest of copepod mortality is carnivorous zooplankton, sandeel larvae and other
  fish larvae, jellyfish and non-predation losses. That is consistent with predation
  being 2/3-3/4 of copepod mortality [S, Hirst & Kiorboe 2002].

### 3.3 Benthic community: r = 5.6e-4 per tick, K about 100 t/km2

growth_rate:
- Logistic fit to whole-community macrofauna biomass recovery after trawling: **r =
  0.82 per yr** (5-95 %: 0.42-1.53) [V, Hiddink et al. 2017].
  - That is **5.6e-4 per tick** (2.9e-4-1.05e-3).
  - It includes recruitment.
  - It is net of all background mortality in the field.
  - Gadoid predation changes it negligibly: gadoids take only 0.3-1.3 % of benthic
    production [C, section 4].
- In Kattegat the right part of the range depends on what the group stands for:
  - long-lived brittle stars (Amphiura), which dominate the soft bottoms, pull toward
    about 0.3 per yr;
  - short-lived polychaetes and crustaceans, which are what gadoids eat, pull toward
    1.5 per yr or more;
  - Hiddink et al. 2019 give r ~ 5.31 / longevity [S, constant unverified].
- Today's 0.001 (1.46 per yr) is in the upper part of the range and not unreasonable;
  5.6e-4 is the best-supported central value.

natural_mortality: no separate term. Under the NDM definition it is inside r, and
density dependence against K does the rest. Gross P/B, about 1.0 per yr [V, NS Ecopath],
is only a check on what gadoids can harvest long term.

K is the real problem:
- Literature standing stocks:
  - Kattegat soft bottoms: 71-146 g WW m-2 [S].
  - North Sea: about 7 g AFDW m-2, i.e. 45-120 t/km2 WW [V via Mackinson & Daskalov
    2007, restating Heip et al. 1992; conversion via Ricciardi & Bourget 1998].
  - NS Ecopath total benthos: about 500 t/km2, but that includes sessile and large
    epifauna.
- Modelled predators take about 1 % of production, so the equilibrium sits close to K,
  and **K ~ observed mean stock ~ 70-150 t/km2**. For prey-sized fauna only (no large
  bivalves, sessile fauna or large echinoderms) it would be about 40-100.
- Today's K = 10 is 7-15x too low.
- Maximum surplus production rK/4 shows the consequence:
  - today: 1.46 x 10 / 4 = 3.7 t/km2/yr;
  - gadoid demand: about 0.7 t/km2 at spawn x Q/B 3.5-6, of the same order, so
    gadoids can graze the benthos down;
  - with r 0.82 and K 100: about 20 t/km2/yr.

### 3.4 Pelagic fish: g = 0.00211, M1 = 3.97e-4 per tick

The pool covers herring (juvenile and adult), sprat, mackerel, horse mackerel, sandeel,
and "miscellaneous filter-feeding pelagic fish" (anchovy, sardine, shads, Maurolicus).
All eat zooplankton (28-93 % of the diet) [V, NS Ecopath Table 3.4].

| Species / group | B (t/km2, NS 1991) | r_max (FishBase CMSY prior, 95 % CL) | P/B | F | M = P/B - F |
|---|---|---|---|---|---|
| herring juvenile | 0.63 | 0.44 (0.29-0.65) | 1.31 | 0.24 | 1.07 |
| herring adult | 1.966 | 0.44 (0.29-0.65) | 0.80 | 0.44 | 0.36 |
| sprat | 0.579 | 0.48 (0.32-0.72) | 2.28 | 0.31 | 1.97 |
| mackerel | 1.72 | 0.43 (0.28-0.64) | 0.60 | 0.32 | 0.28 |
| horse mackerel | 0.579 | 0.39 (0.26-0.58) | 1.20 | 0.30 | 0.90 |
| sandeel | 3.122 | 1.02 (0.67-1.53) * | 2.28 | 0.47 | 1.81 |
| misc. filter feeders | 0.030 | ~0.64 ** | 4.0 | 0.34 | 3.66 |
| **pool, biomass weighted** | **8.63** | **0.65** (0.43-0.97) | 1.47 | 0.40 | **1.08** |

\* FishBase gives the prior under *A. tobianus*, but it is based on the North Sea
sandeel stocks, which are mainly *A. marinus*. The *A. marinus* page itself has
resilience "medium" only.
\** Mean of anchovy 0.59, sardine 0.63, shads and Maurolicus from their resilience
classes (medium 0.4, high 0.95).

Results:
- M = 1.08 splits into M0 0.39, predation by non-FG groups 0.19, and predation by
  FGs: gadoids 0.30, pelagic fish (mackerel and horse mackerel on juveniles and
  sandeel) 0.14, seabirds 0.024, porpoises 0.012, seals 0.008 [C].
- **M1 = 0.39 + 0.19 = 0.580 per yr = 3.97e-4 per tick.**
- Predation by FGs is in the matrix. The pelagic self-pair (0.142) is unchecked in
  the library, so it is neither paid for nor applied.
- g = (0.648 + 0.580 + 0.300 + 0.012) / 730 = **0.00211** for mareld2. With seals
  and seabirds unmuted it is 0.00215; the CL range of r gives 0.0019-0.0026.
- Sandeel is 36 % of the biomass, with both high r and high M.

### 3.5 Gadoids: g = 0.00171, M1 = 4.68e-4 per tick

The pool:
- cod, whiting, haddock, saithe (juvenile and adult), hake;
- "other large gadoids": pollack, tusk, ling, greater forkbeard;
- "other small gadoids": poor cod, bib, silvery pout, four-bearded, five-bearded,
  three-bearded and shore rockling.

All eat fish and/or benthic invertebrates.

Excluded:
- **Norway pout** ("zooplanktivorous, trophic level 3.59" [V, NS Ecopath 13.21]).
- **Blue whiting** (euphausiids).
- Both eat mainly zooplankton, i.e. pelagic_fish food, not gadoid food.
- Norway pout is the largest gadoid biomass in the North Sea (1.39 t/km2), so this
  choice matters; see the variant in section 1.

r_max (FishBase CMSY priors):
- cod 0.51, whiting 0.49, haddock 0.50, saithe 0.52, hake 0.48;
- large others about 0.55 (pollack 0.63, tusk 0.45, ling 0.55, forkbeard 0.55);
- small others about 0.57.
- Biomass weighted: **0.51** (CL about 0.33-0.76).
- Cross-check: Myers et al. 1997 cod r_m is 0.53 for Kattegat, 0.82 for Skagerrak and
  0.56 for the North Sea [V].

M and M2:
- Pool B = 1.96 t/km2, P/B 1.46, F 0.40, so **M = 1.06** [C]. It is high because
  juvenile whiting and haddock and the small gadoids have P/B 2.0-2.4.
- M = 1.06 splits into M0 0.39, predation by non-FG groups 0.17, by benthos
  (crabs; NDMs cannot hunt, so it stays in M1) 0.12, and predation by FGs: gadoids
  (cannibalism) 0.217, pelagic fish 0.080, porpoises 0.056, seals 0.030, seabirds
  0.008 [C].
- **M1 = 0.39 + 0.17 + 0.12 = 0.683 per yr = 4.68e-4 per tick.**
- Only porpoises and seals are checked in the library, so
  g = (0.511 + 0.683 + 0.056) / 730 = **0.00171** for mareld2 (0.00175 with seals
  unmuted); range 0.0015-0.0021. Checking the cannibalism pair would add 0.217 /yr,
  i.e. +3.0e-4 to g.

### 3.6 Porpoises: g = 2.7e-4, M1 = 6.8e-5 per tick

- r_max:
  - about 0.10 per yr: Lockyer 2003 "probably 9.4 %" [S]; Caswell et al. 1998 median
    lambda about 1.10 [S]; Morro Bay recovery 9.6 %/yr after bycatch ended [V, Forney
    et al. 2021];
  - 0.04 is the HELCOM/Wade management default [V/S] and is used as the low end.
- M (biomass weighted, no bycatch) is 0.10 (0.08-0.15). It comes from Leslie models in
  the notes, built from adult survival 0.90-0.95 excluding bycatch plus calf mortality
  [C].
- Gross = 0.20 (0.12-0.25), so g = **0.20 / 730 = 2.7e-4** [C].
- The Leslie gross P/B of 0.14-0.21 (75-85 % juvenile growth) agrees.
- Nothing preys on porpoises in Mareld, so M1 = M = **0.10 per yr**.
- The porpoise g is the same under the old `satiation_scale` 1.37 and under h = 1 - s,
  because the reserve is clipped at s = 1 either way (gate 0.5). What changes is the
  ration at the maintenance level (section 5).

### 3.7 Seals: g = 2.9e-4, M1 = 6.8e-5 per tick

- r_max:
  - Kattegat-Skagerrak harbour seals grew 10.2-13.6 %/yr after 1988, with intrinsic
    maximum "slightly less than 13 %" [V, Carroll et al. 2025];
  - HELCOM uses 12 % for harbour seals and treats 10 % as the grey seal ceiling [V].
  - Pool: 0.11.
- M (biomass weighted) is 0.10 (0.08-0.15) from the Leslie models [C].
- Gross = 0.21 (0.18-0.24), so g = 0.21 / 730 = **2.9e-4** [C]. The Leslie gross P/B is 0.17-0.20.
- M1 = M = **0.10 per yr**.
- Seals are muted in `mareld2.yaml`.

### 3.8 Seabirds: g = 3.4e-4, M1 = 1.0e-4 per tick

- r_max uses the Niel & Lebreton demographic invariant on JNCC 552 rates [C]:
  guillemot 0.076-0.085, razorbill/gannet about 0.10, kittiwake and large gulls
  0.11-0.16. Community: 0.10.
- Observed sustained colony growth is lower: Skomer guillemots about 5 %/yr [V],
  UK gannets 1.3-2.2 %/yr [V].
- M (biomass weighted, juveniles included) is 0.15 (0.11-0.20). Leslie Z is 0.145 for
  guillemot, 0.16 for kittiwake and 0.20 for gannet [C].
- Gross = 0.25 (0.19-0.36), so g = 0.25 / 730 = **3.4e-4** [C].
- M1 = M = **0.15 per yr**.
- Seabirds are muted in `mareld2.yaml`. The offshore Skagerrak is mainly a foraging
  and wintering area, so even a closed-pool r_max is generous.

## 4. Predation matrix (default predation_mortality, 1/yr)

These are the defaults stored on each pair in `interaction_definitions`. **Bold** =
checked in the library; *italic* = literature value, pair unchecked.

| prey \ predator | zooplankton | pelagic_fish | gadoids | porpoises | seals | seabirds |
|---|---|---|---|---|---|---|
| phytoplankton | **22.63** | *0.0566* | - | - | - | - |
| zooplankton | - | **5.24** | *0.089* | - | - | *0.010* |
| benthic_community | - | *0.015* | **0.0048** | - | - | - |
| pelagic_fish | - | *0.142* | **0.300** | **0.0117** | **0.0076** | **0.0236** |
| gadoids | - | *0.080* | *0.217* | **0.0557** | **0.0299** | *0.0078* |

No FG preys on porpoises, seals or seabirds in the literature used. Adding such a
pair needs its own value.

The values are budget entries, not the realised predation of the tick. The tick's
predation comes from the Holling response, and the realised M2 can differ, in which
case net growth no longer equals r_max. Logging realised M2 per pair is the check.

## 4b. M2 decomposition from the North Sea Ecopath model

Source: Mackinson & Daskalov 2007 (Cefas Tech. Rep. 142), North Sea 1991:
- Table 3.3: B, P/B, Q/B, EE.
- Table 3.4: the diet matrix.
- Table 3.5: catches.
- Area: 570,000 km2.

Predation mortality of prey i by predator j = DC_ij x (Q/B)_j x B_j / B_i [C].
Extraction scripts and the parsed matrix are in
`research_notes/Litteraturbaserad growth rate per FG/ecopath_ns1991/`.

Validation:
- Summed over all predators, the computed M2 matches the report's EE-based
  M2 = P/B x EE - F.
- herring juvenile 0.63 vs 0.64, adult 0.116 vs 0.116;
- sprat 1.50 vs 1.52, sandeel 1.31 vs 1.32;
- cod juvenile 1.06 vs 1.02, whiting juvenile 1.97 vs 1.92.

| Prey pool | M | M2 by Mareld predators | of which | M2 by unmodelled predators | M0 |
|---|---|---|---|---|---|
| pelagic_fish | 1.08 | 0.34 | gadoids 0.30, seabirds 0.024, porpoises 0.012, seals 0.008 | 0.34 | 0.39 |
| gadoids | 1.06 | 0.086 | porpoises 0.056, seals 0.030 | 0.58 | 0.39 |
| copepods | ~P/B | 20 % of production by the pelagic pool | sandeel 1.20, herring 0.48, sprat 0.17 per yr | carnivorous zooplankton 1.93, fish larvae 0.32 per yr | 52 % of P/B |
| macrobenthos | ~P/B 0.8-1.0 | 0.3-1.3 % of production by gadoids | - | flatfish, crabs, starfish, Nephrops (about 59 % of P) | about 41 % of P |

Data-quality problems in the source table (confirmed on the rendered page):
- Rows 24-25 ("other gadoids") are duplicated from another page.
- Several gadoid predator columns (13-24) do not sum to 1.

Consequences:
- Predation on "other gadoids" is scaled down to the EE-based M2.
- Gadoid predation on benthos cannot be read reliably. The 0.3-1.3 % range combines
  the table (0.3-0.6 %) with an order-of-magnitude check (about 1 %).
- Prey rows 13-33 validate, so the fish M2 values above are sound.

## 5. satiation_scale removed: h = 1 - s for every FG (superseded by full appetite up to maintenance, section 133)

What the parameter does: intake per tick is scaled by the hunger factor
h = max(0, 1 - s / satiation_scale) (`predation.py`), where s is the reserve fill.
The default 0.8 is "a pure model constant (no literature anchor)"
(`energy_balance.py`); porpoises override it with 1.37 (section 92).

| satiation_scale | feeding stops at | h at u = 0.5 | (s - u)_max |
|---|---|---|---|
| 0.8 (default today) | s = 0.8 | 0.375 | 0.3 |
| **1.0 (proposed, all FGs)** | s = 1, full reserve | 0.5 | 0.5 |
| 1.37 (porpoises today) | never (s clipped at 1) | 0.635 | 0.5 |

Why 1:
- **"Full reserve" means satiated.** With 0.8 the top 20 % of the
  literature-anchored `max_energy_reserve` (storage lipid, blubber store) can never
  be reached.
- **It removes a parameter with no literature anchor.** What remains is a linear
  appetite that falls with body condition and ends at a full reserve.
- **Porpoises stop wasting energy.** With 1.37 a porpoise at a full reserve keeps
  eating, but the excess is clipped away in `movement.py`: prey is killed and the
  energy is lost.
- **The zooplankton mass-balance bound stops binding** (section 3.2).

Consequences, none implemented:
- **Every DM g except porpoises is 0.6x** what the same biology needs under 0.8
  (0.3 / 0.5). Section 1 already uses the new values. M1 is unaffected.
- **Ration at the maintenance level changes.**
  - h(u) rises from 0.375 to 0.5 (+33 %) for every DM except porpoises.
  - Section 68 found that raising the scale for zooplankton deepened the
    zooplankton-phytoplankton overshoot (zoo floor 0.36x, phyto 0.70x). Check that
    first.
  - Porpoise h(u) falls from 0.635 to 0.5. The realised ration goes from 8.9 % to
    about 7 % of body mass per day (0.035 x 0.5 x 4 ticks), still inside Kastelein et
    al.'s 4-9.5 %.
  - That alone breaks the porpoise budget (118 MJ/t/tick from pure herring against
    126 MJ feeding cost). Chosen fix (user): `feeding_cost` 1.4 -> **1.1**. Net eat is
    then +19 MJ/t/tick, the break-even clupeid share is 0.49 (as before), and the
    break-even visible herring density is 5.2 t/cell (5.25 before). The multiplier
    itself has no literature anchor.
- **Fasting endurance on the surplus rises 67 %**, because the surplus is
  0.5 x ME instead of 0.3 x ME.
- **Code, tests and training.**
  - `HUNGER_SATIATION_SCALE` (0.8) and the porpoise entry in `fg_library.yaml`
    must change; the GPU engine reads the same value.
  - The GUI energy-balance gate and `tests/test_energy_balance_gate.py` use the
    scale.
  - Behaviour changes, so policies need retraining. Checkpoints still load.
- **Removed (section 130 of the resume).** `HUNGER_SATIATION_SCALE`,
  `resolve_satiation_scale`, the `hunger_scale` argument of
  `evaluate_energy_balance`, `FunctionalGroup.satiation_scale`, the GPU satiation
  vector, the FG-editor field and every library entry are gone. h = max(0, 1 - s)
  in `get_hunger`, `hunger_at` and the GPU engine. A stale key in an old library is
  ignored, and the FG editor drops it on apply.

## 6. Implementation (2026-10-03)

- **`lib/world/growth_budget.py`** is the single definition of the budget, used by
  the loader, the FG editor and the tests.
- **Loader (`lib/config/config_loader.py`).** Both FG builders derive growth_rate
  before the tick-length rescale. Active predators are the project's unmuted FGs, or
  every library FG on the library-only path. An FG without `r_max` keeps its stored
  growth_rate.
- **`fg_library.yaml`.**
  - `r_max` on every FG and `predation_mortality` on every literature pair.
  - New M1 values, porpoise `feeding_cost` (1.1, later back to 1.4) and benthos K
    100, each with a comment giving the source.
  - The stored growth_rate is the derived value for mareld2. A test guards it
    against drift.
- **FG editor (`fgconfig/fgconfig.py`).**
  - A new "Predation Mortality M2" matrix sits under the predation checkboxes. Its
    cells are active only when the pair is checked.
  - A "Derived growth_rate" panel shows every FG's budget. It updates **immediately
    on check/uncheck** and on M2 edits.
  - The FG editor has an `r_max` field. The Max Growth field is read-only and live
    while r_max > 0, and shows the budget breakdown.
  - "Apply All Matrix Changes" writes the derived growth rates to the library.
- **`lib/diagnostics/viability.py`.** `colocate_spawn` normalises each prey map
  before summing. With benthos at K 100 the raw-tonne sum let benthos alone decide
  where gadoids were placed.
- **Tests.**
  - `tests/test_growth_budget.py` (12 cases) covers the formula, check/uncheck
    symmetry, r_max invariance, active-predator rules, library drift and the loader.
  - Suite: 777 passed, 0 failed (after section 3.1). A one-generation training run
    passed.
- **Resolved: the closed phyto-zoo test.** `test_phyto_zoo_sanity::test_zoo_bounded`
  failed with the annual-mean phytoplankton r: zooplankton fell to 0.027x. The
  driver was phytoplankton r; the zooplankton M1 alone accounted for little (0.066x
  when restored). After the definition was corrected in 3.1 (user decision,
  low-density in situ rate), the test passes at 1.07x.
- **`--mortality off`** (the CLI default): the loader now leaves M1 out of g
  (`apply_natural_mortality` on `load_project_config` / `setup_full_mareld_mvp`).
  The flag is passed from every env builder: train.py (both builders),
  `inference.build_env`, `training_progress.build_inference_env` and the GPU
  `EnvironmentBuilder` / `ProjectSpec`. For mareld2 with mortality off: zooplankton
  0.0529, pelagic fish 0.00131, gadoids 0.00078, porpoises 1.37e-4. NDMs are
  unchanged. The FG editor and the stored library values assume mortality on.

## 6b. Consequences and follow-ups

1. **Zooplankton g is well inside the mass-balance bound** (0.080 against 0.150),
   under h = 1 - s. Under the old 0.8 it would have been close to the bound.
2. **Benthos K = 100 was reverted** (section 136). With K 10, gadoids fed by the
   pre-133 policies decline to 0.54x in 1000 ticks (54 % starvation) and still eat
   about 95 % benthos. Every DM's growth and mortality changed anyway, so **all
   policies in `results/` need retraining**. Checkpoints still load.
3. **NDMs have no natural_mortality term.** Under the definition in 2.2 none is
   needed, because the losses are inside r.
4. **Optional structural improvements for benthos**, in order of value:
   - K as a habitat map (depth and sediment) instead of a constant;
   - coupling benthos to phytoplankton sedimentation (in Kattegat benthos is driven by
     the spring bloom; in Mareld it is decoupled from phytoplankton);
   - enabling a bottom-trawling impact (the main pressure in Kattegat);
   - setting `seasonal_amplitude` from recruitment timing.
5. **Phytoplankton r** rests on one southern North Sea spring study and applies all
   year. Review it together with K = 24 and a literature `seasonal_amplitude`.
6. **Higher feeding requirement.** With lower g and higher M1 the groups must keep a
   higher mean feeding gate just to persist. Before adopting, measure the realised
   biomass-weighted `<s-u>` per DM and the realised M2 each predator imposes.

## 7. Unverified or missing

Unverified [S]:
- Huntley & Lopez coefficients (secondary source);
- Kiorboe & Nielsen 1994 Kattegat copepod figures;
- Kattegat benthos biomass 71-146 g WW m-2;
- Hiddink 2019 constant 5.31;
- Lockyer 2003, Caswell 1998, Wade 1998 (abstracts only);
- the Stora Karlsoe guillemot growth rate;
- the guillemot departure mass;
- sinking/lysis share for phytoplankton (scaled from a spring figure).

Missing:
- Kattegat Ecopath model (ICES WGINOSE 2020, HTTP 403);
- western Baltic Ecopath Table S6;
- numeric ICES SMS M1/M2 (figures only);
- a primary-source Kattegat phytoplankton carbon biomass.

All fish M and M2 come from a single model (North Sea 1991), not Kattegat/Skagerrak.
The species mix differs there: less sandeel, more sprat.

## 8. Sources

- Bissinger, J.E. et al. 2008. Limnol. Oceanogr. 53:487-493.
- Calbet, A. 2001. Limnol. Oceanogr. 46:1824-1830.
- Calbet, A. & Landry, M.R. 2004. Limnol. Oceanogr. 49:51-57.
- Carroll, D. et al. 2025. PLoS ONE (harbour seals, Kattegat-Skagerrak). https://pmc.ncbi.nlm.nih.gov/articles/PMC12208499/
- Caswell, H. et al. 1998. Ecol. Appl. 8:1226-1238.
- Eppley, R.W. 1972. Fish. Bull. 70:1063-1085.
- FishBase species summaries (CMSY prior r), accessed 2026-10-02. https://www.fishbase.se
- Forney, K.A. et al. 2021. NOAA (Morro Bay harbor porpoise).
- Heilmann, J.P., Richardson, K. & Aertebjerg, G. 1994. Mar. Ecol. Prog. Ser. 112:213-223.
- HELCOM 2013. Core indicator: population growth rate, abundance and distribution of marine mammals.
- Hiddink, J.G. et al. 2017. PNAS 114:8301-8306.
- Hirst, A.G. & Bunker, A.J. 2003. Limnol. Oceanogr. 48:1988-2010.
- Hirst, A.G. & Kiorboe, T. 2002. Mar. Ecol. Prog. Ser. 230:195-209.
- Horswill, C. & Robinson, R.A. 2015. JNCC Report 552.
- Huntley, M.E. & Lopez, M.D.G. 1992. Am. Nat. 140:201-242.
- Lindahl, O. et al. 1998. ICES J. Mar. Sci. 55:723-729.
- Lockyer, C. 2003. NAMMCO Sci. Publ. 5:71-89.
- Mackinson, S. & Daskalov, G. 2007. Cefas Sci. Ser. Tech. Rep. 142. https://www.cefas.co.uk/publications/techrep/tech142.pdf
- Myers, R.A., Mertz, G. & Fowlow, P.S. 1997. Fish. Bull. 95:762-772.
- O'Brien, S.H., Cook, A.S.C.P. & Robinson, R.A. 2017. J. Environ. Manage. 201:163-171.
- Peterson, W.T., Tiselius, P. & Kiorboe, T. 1991. J. Plankton Res. 13:131-154.
- Ricciardi, A. & Bourget, E. 1998. Mar. Ecol. Prog. Ser. 163:245-251.
- Richardson, K. & Christoffersen, A. 1991. Mar. Ecol. Prog. Ser. 78:217-227.
- Scotti, M. et al. 2022. Front. Mar. Sci. (western Baltic EwE), supplement.
- Skoeld, M. et al. 2025. Conserv. Sci. Pract. doi:10.1111/csp2.70037.
- Sparholt, H. et al. 2021. ICES J. Mar. Sci. 78:55-69.
- Wade, P.R. 1998. Mar. Mammal Sci. 14:1-37.

Per-topic notes with full citations: `research_notes/Litteraturbaserad growth rate per FG/`
(plankton.md, benthos.md, fish.md, mammals.md, seabirds.md).

## Superseded

Phytoplankton r 0.0194 (annual mean, r_max 5.66) was replaced by 0.067 (low-density
in situ rate, r_max 75.2) on 2026-10-03.

The proposal versions of 2026-10-03 kept predation by modelled FGs inside the prey's
M (and so inside g and M1) whatever the matrix said. The implemented version moves
it to the matrix: M1 now holds only non-FG losses, and g follows the checked pairs.
For pelagic fish M1 went from 0.73 to 0.58 /yr, and for gadoids from 0.98 to
0.68 /yr.


The 2026-10-03 version before this one assumed today's `satiation_scale` (0.8, and
1.37 for porpoises), so its DM g values were 5/3 of the current ones (porpoises
unchanged).

The first version of this report (2026-10-02) set each DM ceiling to
r_max + M1 + an assumed share of M2 by modelled predators. It kept M1 at today's
values and left the NDMs near their current r. The user then fixed the definitions in
section 2:
- maximum growth under ideal conditions, gross of all natural mortality;
- M1 from the literature;
- NDM r net of natural mortality and of unmodelled predation;
- umbrella-term species pools.

Every number in that version is replaced by section 1.
