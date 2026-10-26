# Exposure of benthic prey: mechanistic e and revised benthivore Cmax (resume 223.14)

Written by the research subagent, saved by the main agent (2026-10-26). Scripts in `scripts/`
(`exposure_e.py`, `bottom_temperature.py`, `C_cmax_bottom.py`, `D_crab_window.py`, `D_epi_shares.py`),
tables and evidence in `tables/`, data in `data/` (SHARK in `data/E_shark/`). J = judgement. Holmes 1979
values are for internal use only.

## Headline (contradicts P1 of "Fodoniva och mattnad i falt")

Mechanistic e is ~0.15-0.3 on infauna and ~0.4-0.5 on exposed epifauna, not 0.002-0.01. Every benthic pair
stays saturated (x = a h A 19-66 at today's prey), so the field feeding levels (0.22-0.53 of Cmax) cannot
come from a; at x >> 1 the engine's feeding level is the policy's eat x hunger. The field data point to
foraging time and exposure windows. e alone does not stop the 223 eat-out: it moves the depletion floor
1/(a h) from 0.20 to 1.4 t/km2 (infauna), 0.025 to 0.07 (benthivores -> epifauna), 0.06 to 0.16 (predators
-> epifauna).

## Decomposition (largest overstated steps first)

A = phi B mixes exposed parts (arm tips), whole bodies buried in 0-2 cm and small mobile crustaceans.
(a) Buried bodies are found only via a cue (Brawn 1969: "Food covered by sand was not found"; Abra siphon
out 25 % at 10 C, ~13 % at 6 C J; spionids 24-79 %). (b) Yield per strike is often a cropped part (siphon
tips ~3 % of biomass; dab graze arms and leave the disc, Kaiser & Ramsay 1997); Amphiura retract arms when
disturbed. (f) Flatfish vision fails at the dim bottom (Hurst 2007): chemo-tactile width 0.1-0.3 L (width
x0.80 infauna, x0.92 epifauna). Epifauna: crab size window (carapace <= 0.06-0.09 fish length, Arnott & Pihl
2000; share inside: cod 0.45, others ~0.03); small-shrimp burial (Persson 2012, Ljungberg 2012). Not
overstated: search share (cod active round the clock), behavioural handling, gadoid a on exposed epifauna.

## e and a_eff (km2 per t predator per 6 h tick, annual mean; MC median, 5-95 %)

| Pair | a now | e | a_eff |
|---|---|---|---|
| benthivores -> benthic_infauna | 0.25 | 0.165 (0.11-0.27) | 0.036 (0.016-0.092) |
| benthivores -> mobile_epifauna (exposed, E1) | 0.44 | 0.43 (0.37-0.48) | 0.15 (0.05-0.53) |
| predators -> mobile_epifauna (exposed) | 0.33 | 0.39 (0.31-0.49) | 0.13 |
| benthivores -> norway_lobster | 0.44 | 1.0 (J; emergence and size in f) | 0.44 |

Composition variant (Skold & Gunnarsson 90 m, Amphiura 58 %): infauna e 0.27, a_eff 0.056. Flag: the exposed
Amphiura tip is 1-5 % of body mass vs 6 % inside phi (phi at 100 m 0.185 -> ~0.168; an exposed_share
correction, not part of e).

## Independent tests

Diet vs availability, benthivores: a_eff gives infauna 0.63, epifauna 0.27, planktivores 0.06, predators
0.03 vs field 0.54 / 0.31 / 0.12 / 0.02 (library a: 0.83 / 0.15) - passes. Predators' epifauna share: a_eff
0.29, library 0.51, field 0.072 (most of the rest is bed access). Hiddink 2016: fails for both (needs x
~1-4); its signature is exchange-limited supply (the infauna refuge). Share of infauna production eaten tests
the feeding level, not a.

## Benthivore Cmax and handling times

Bottom temperature (SHARK A16/A17 1991-2020, each cell at its depth): monthly 7.80 6.99 6.59 6.58 6.69 6.96
7.22 7.41 7.70 8.04 8.36 8.50 C, annual 7.40 (the at_50m layer used now: 8.4). Central group Cmax 104.7
kJ/kg/d, ceiling 26.18 MJ/t/tick (was 108.4 / 27.1): bottom temperature -7 %, haddock x1.3 -> x1.0 -12 %, cod
FB4 -> Bjornsson et al. 2001 +20 % (J). Alternatives 87.2 (21.8) to 119.9 (30.0). h = E / 26.18: infauna 99.3
(95.9), exposed epifauna 143.2 (138.4), Nephrops 137.6, planktivores 272.1 (262.9), predators 200.6 (193.8).
Armour option on crustacean pairs h x1.5-3 (J; Couturier 2013, Andersen 1999).

## Implementation (list)

Library a on the three benthic pairs (or a pair key exposure_factor); handling_time on the five benthivore
pairs; a `bottom` temperature layer in mareld2.yaml for the benthivores; growth_budget moves with h only
(ceiling -3.4 %); budget_gate; tests (test_half_saturation rate locks, pair shares, growth sync). No
observation change.

Gaps: strike success on retracting arms/siphons; Abra activity at 5-8 C; haddock digging depth; flatfish
light thresholds; adult haddock ad-lib intake; in-area crab sizes.
