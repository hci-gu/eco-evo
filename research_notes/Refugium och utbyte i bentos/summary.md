# Refuge and exchange in the benthos: biology for the mechanism (resume 223.9, decision 1)

Written by the research subagent, saved by the main agent (2026-10-26). Research only: no library, manifest
or code change. Scripts in `scripts/`, output and evidence in `tables/` (`refuge_parameters.csv` is the
recommended table), downloads in `data/`. J = judgement.

## 0. Headline

1. **Epifauna: no biological slow exchange exists.** Every measurable exchange is diel (Nephrops emerge
   1-2 times a day, the same animals every day; shrimp migrate dusk to dawn; crabs and small shrimps
   diel): k ~365-730 /yr, 150-300x the rate (r - F)/phi = 2.39 /yr below which a refuge survives
   unlimited predation. A single A/R pool at a biological k reproduces today's engine (equilibrium in 2-4
   ticks at today's f). The only slow structure is taxonomic: Nephrops versus crabs and shrimps (k = 0).
   Recommended form **E1: two compartments inside the group, no exchange** (exposed taxa; Nephrops), each
   with its own r_max, K, M2 and trawl term. E1 protects Nephrops and the group total; it does NOT make the
   predators' epifauna intake sustainable (the budget flux 1.40 t/km2/yr is 90 % of the exposed
   compartment's net maximum surplus 1.56; 1/(a h) unchanged, so saturation to ~2 % of the compartment).
2. **Infauna: the slow exchange is real** (cropping with regrowth, settlers growing out of reach): k ~2.5
   /yr (1.5-3), k_in ~0.5. It slows local depletion but does not stop it: under unlimited cropping the
   refuge holds 0.38 of K_R; a cell with doubled benthivores at their ceiling falls 93.6 -> ~37 t/km2 in 5
   years (vs ~0.7 in 3 years at fast exchange).
3. **The engine's Beddington-DeAngelis interference is already the limit of a fast refuge** (Berardo et
   al. 2020, after Geritz & Gyllenberg 2012): a fast A/R pool on top would count the same behaviour twice.

## 1. Model form (per cell, per benthic group; rates /yr, /1460 per tick)

```
dA/dt = g A (1 - A/K_A) + k_in R - k_out A - F_A A - sum_j removal_j(s_j A)
dR/dt = g R (1 - R/K_R) - k_in R + k_out A - F_R R
K_A = phi K(c),  K_R = (1 - phi) K(c);  k_in = phi k,  k_out = (1 - phi) k
```

- With no predation A/B relaxes to phi = k_in/(k_in + k_out), today's accessible_fraction.
- Both compartments grow, each logistic on its own capacity (variant S; a saturating refuge, Krivan
  1998). Only-R growth leaves the refuge unregulated, only-A growth makes R inert, pooled density
  dependence (P) lets R take over K when A is eaten down.
- Refuges are not saturated in the visited range (FU 3-4: 0.21 Nephrops burrows/m2 vs a Clyde maximum
  of 1.48; burrows re-dug in 1-2 days); the deep infauna layer's capacity is unmeasured.
- Exchange as exact relaxation `A <- phi B + (A - phi B) exp(-k dt)` after growth (or exchange and
  predation together by a 2x2 matrix exponential, checked against RK4). Predation removes from A only; the
  Holling term and harvest cap see s_ij x A.
- Trawl: infauna proportional (Hiddink's d is whole-community, in-path); epifauna on the exposed animals
  (Nephrops trawls catch emerged animals, 0.4-2.5 % of burrows per haul, Tuck 1997; d_u in burrows 0.01).
- Observations: the fish observe A (or, in E1, each pair's compartment); checkpoints invalid either way.
- Growth budget: exact per compartment in E1 (k = 0); for the infauna pool at k 2.5 the loader's rule
  lands B* within 0.1 % (r needed 1.031 vs 1.029); fails only at k < ~1 /yr or a 300 m variant (k 0.3
  needs r 1.045). A/B at equilibrium falls below phi (infauna 0.193 vs 0.20).
- Ecosim (foraging arena) uses the same algebra in the fast limit, V = v B/(v + v' + aP); its v = K x
  M2_baseline is a fitted effective rate (~0.06-0.35 /yr here), not a behavioural one - ruled out by the
  biology-not-fit rule. Today's per-tick f is Walters & Christensen's "bout" model with v* ~290 /yr.

## 2. Rates and shares (full table with sources: tables/refuge_parameters.csv)

Infauna (A/R pool):

| Quantity | Value (range) | Basis |
|---|---|---|
| phi | 0.20 (0.08-0.45) | Dauwe 1998, Skagerrak 280 m: 27 % of biomass in 0-2 cm; Amphiura arm tips; reach depth unmeasured (J) |
| Refill time of the exposed layer | ~146 d (120-250) | Amphiura arm regrowth 9.2-13.1 mm in 77 d (Skold & Gunnarsson 1996); 0.6-3.3 mm/week at 14 C (Dupont & Thorndyke 2006); slower at 6 C (J) |
| k | 2.5 /yr (1.5-3.0) | 1/refill time; must stay below 4.47 for the refuge to persist |
| k_in | 0.5 /yr (0.3-0.6) | production-limited: lugworm tails cropped at 0.39 /yr, ~60 % of somatic production (Beukema & de Vlas 1979); no compensatory growth (Bergman 1988) |
| k_out | 2.0 /yr | (1 - phi) k |
| 300 m deposit-feeder variant k | 0.3 /yr (0.2-0.5) | no size-depth refuge; refill by settlement over 2-5 yr (Rosenberg 2002) |
| Benthivores' share of A | 1.0 | the reach is already in phi |
| Recolonisation of an emptied cell | 2-5 yr | adults ~1 m/yr, postlarvae ~1 km, larvae further |

Epifauna taxa (inputs to E1): Nephrops exposed share 0.07 (0.03-0.35; camera 0.06-0.12 visible per burrow
at peak, 0.037-0.066 fully emerged, Aguzzi 2021; berried females in burrows 7-10 months; no hard size
refuge; ~0.29 of the biomass at or below 35 mm CL); northern shrimp 0.75 (0.65-0.85) of the day on the
bottom; crabs and hermits 0.6; Munida 0.5; small shrimps and mysids 0.6. Group phi 0.29 (0.27-0.46) vs the
library's 0.31; Nephrops contributes ~0.04. Correction: L. holsatus is day-active (Abello 1991).

E1 compartments:

| Compartment | B | r_max | sum M2 | F | Eq. share | K | bf100 |
|---|---|---|---|---|---|---|---|
| Exposed taxa (crabs, hermits, Munida, northern shrimp, small shrimps, mysids) | 3.06 | 1.11 | 0.456 | 0.21 | 0.575 | 5.32 | 0.00477 |
| Nephrops | 4.57 | 0.34 | 0.0007 | 0.087 | 0.743 | 6.15 | 0.00198 |

Pair shares: benthivores -> exposed 0.63 (0.45-0.8); predators -> exposed 0.33 (0.2-0.45) (= today's 0.13
and 0.31 of B); benthivores -> Nephrops 0.07 (~0.02 with gape); predators -> Nephrops unchecked. k between
compartments 0 (upper bound 0.3 if small Nephrops count as exposed). NB: the M2 values in this table predate
the 223.10 library change (benthivore M2 on the epifauna 0.141 -> 0.276).

## 3. Independent checks (not used to set values)

- Mackinson & Daskalov 2007 recomputed (`scripts/md2007_fish_share.py`): the PDF's diet page for
  predator columns 13-24 repeats columns 1-12, so the gadoid values are lost for every invertebrate prey.
  Bounds on the share of production eaten by fish: infaunal macrobenthos 2.9-5.5 %, polychaetes 8-20 %,
  large crabs 60-92 %, Nephrops 65-94 %, shrimp 15-39 %; the high ends are Ecopath closure (EE 0.93-0.99).
  The earlier "4 % / 82 % / 71 %" (evidence_C) predate the defect; the "4 % / 70-80 %" pair is from M&D,
  not Heath 2005 (Heath 2005 Table 4: fish predation on benthic carnivores 72 / 53 / 33 % by decade).
- Infauna: fish eat 1.3-6.5 % of production (Dauvin 1988); cage experiments show no fish-driven
  depletion; the budget's 2.72 t/km2/yr is ~3 % of P/B 1 x 94.
- Trawled-patch recovery 2-5 yr (Hiddink 2017: 0.82 /yr, 3.6 yr from 0.5 K; Rosenberg 2002).
- Nephrops: cod M2 0.04-0.25 /yr in the western Irish Sea; the budget's 0.0007 is far below (in-area diet,
  cod rare; flagged J). Pandalus (Skagerrak) total M 0.9-1.2 /yr, predation at least twice the catch.
- Group: an epifauna M2 of ~0.2-0.4 /yr fits; the 223 removal of 5.4 t/km2/yr exceeds the whole-pool
  maximum surplus 1.91 and can only be a transient.

## 4. Judgements and conflicts

1. E1 is a taxon split inside one group; it replaces "exchange" for the epifauna because biology gives no
   slow exchange there. An Ecosim-style k = K x M2 is a fitted rate, ruled out.
2. One infauna k vs depth-dependent k, and the 0-2 cm reach depth, are judgements (fish feeding depth in
   mud never measured).
3. Cropping is an R -> A transfer charged to R (the prey's own production), not added on top of r_max.
4. Seasonal Nephrops (berried females) and Macoma burial depth left out (a seasonal pair share later).
5. Recolonisation of a completely emptied cell (A and R zero) not covered: NDMs do not move, seed_rate 1e-6.
6. The predators' saturation and E1's thin margin remain: the refuge does not resolve 223.3.
7. The Nephrops budget M2 is low against the Irish Sea; the crab stock rests on mass balance (221.5).

## 5. Code and test changes (list only)

State: a second per-cell field per benthic group (the NDM `energy_reserve` slot is free, as the producers'
nutrient pool uses it), or two compartment fields for E1. `interactions.apply_pair_access` / `access_mat`
(share on A, shared cap from s_ij x A); `predation.py` (remove from A, harvest cap on s x A);
`population_change.py` (per-compartment logistic, one derived g, exact relaxation, clipping per
compartment); `impacts.py` (per-compartment tables or a compartment key); `observations.py`;
`growth_budget.py`; `habitat.py` (two capacity maps, two-compartment equilibrium, spawn of A and R);
`lib/gpu/ecosystem.py` (mirror); library/manifest keys (e.g. `refuge_exchange_rate`, `exposed_share`);
fgconfig rows; viewer and plots; `lib/diagnostics/viability.py:331`; `tools/probes/budget_gate.py`;
`frozen_years.py` codes. Tests: test_pair_shares, test_habitat, test_gpu_ecosystem parity, growth-sync and
growth-budget, test_fgconfig_depth_map; new: exchange conserves B, equilibrium share = phi without
predation, k -> infinity reproduces today, refuge persistence when k_in < r - F.

Not obtained: Walters & Juanes 1993; Walters et al. 1997; Chapman 1980; Bell et al. 2006; Bergstrom 2000;
de Vlas 1979/1985; Mattson 1992; any measured fish feeding depth in mud; the deep layer's capacity; a
Skagerrak measurement of shrimp night ascent.
