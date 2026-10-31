# Benthivores' planktivore prey: the field take is Norway pout, not herring (2026-10-31)

Research only. Nothing in the library, manifest, code or tests was changed. Tags: [F] field data or stated in a source, [D] derived here, [J] judgement. Written by the research subagent, saved by the main agent.
- `evidence.md`: the audit trail, quotes and the per-section derivations.
- `scripts/` and `tables/`: every script with its stdout.
- `data/README.md`: the one download, plus the command lines.

## Verdict

1. **The booked M2 0.145 is supported in the box.** This is the opposite of the whiting audit's 0.203.
   - In-area stomachs give a group planktivore share of the checked pairs of 0.08-0.17 (evidence C), 0.076-0.187 (ICES DB post-2000, Skagerrak >= 80 m / NNS >= 100 m) and 0.104 (SMS 1981-91 at the box length mix). The booked share is 0.075.
   - At Q/B 3.5-4.46 those shares mean M2 0.12-0.30 (central ~0.17) [D].
   - What the field changes is the species. In the box, 83-99 % of the take is Norway pout. Herring plus sprat are 0-7 % of cod's fish prey [F/D].

2. **The pair is 15x (booked) to 16-45x (field diets) too weak against the bed anchors** (infauna and epifauna, which agree within 6 %).
   - Per species, almost all of the deficit is the pout term: model / field 0.014-0.039.
   - Herring is roughly consistent with the in-area stomachs (0.9-2.4). It is 0.12 against SMS North Sea 1981-91.
   - Source: `tables/clearance_test_benthivores_stdout.txt`.

3. **Four bookkeeping and geometry errors together give x2.0 (x2.9 with the herring-specific window ratio).**
   - **Gape counted twice (the whiting fault-2 analogue).** In 223.26, phi 0.32 went into a (a = 0.019 x 0.32). phi's planktivore scores are gape shares ("cod 40 cm swallows 96 % of the Norway pout biomass and 45 % of herring ... cod 0.3 / 0.7 / 0.9"). Section 223.32 then added the prey-size window [0.49, 0.14] on the same axis.
     - The fix is a membership share without gape: cod >= 20 cm at 1, haddock with its own window x a non-gape capability of 0.3-1.
     - That gives 0.377 (0.35-0.41) instead of 0.32: x1.18 (1.08-1.27).
   - **Vertical geometry (the whiting fault-1 analogue, opposite sign).** Herring's near-bed daytime share was cut from HERAS's box value 0.34 to 0.15 by a depth judgement.
     - The box GOV hauls, at the grid's depths, contradict that cut. Raw GOV herring 0.70 t/km2 against the column's 2.60 t/km2 means >= 0.27 of the herring is inside the ~5 m trawl opening, and the GOV under-catches small pelagic fish.
     - So herring is 0.34 (0.27-0.54) by day [D]. Sprat is 0.5 (0.3-0.9), blue whiting 0.6 (0.3-1).
     - The pool overlap sum rises x1.87 (1.6-2.6).
   - **Window weighted by pool biomass, while the a weights by encounter:** x1.09 [D].
   - **Blue whiting** has no rows in the stage table, but it is outside the cod window: x0.86 [D].
   - **The window ratio 0.30 itself is right.** A length-dependent p95 gives the same f_eff. The herring-specific p95 gives x1.42 (high end).

4. **The rest of the gap is the pout biomass convention, not the a.**
   - Box pout is 0.25 t/km2 (raw GOV q = 1, acoustic 0.2-0.3). Walker et al. 2017 put GOV pout q at 0.07-0.19, which gives 0.6-2.7 t/km2 after correction [S/D].
   - Mortality check: the booked benthivore take plus whiting's take would put M2 on 0.25 t/km2 of pout at ~3 /yr. SMS pout M2 is 1.89 and M1 0.2-0.3. So 0.25 t/km2 cannot carry the booked diets [D].
   - With the model side moved too, at pout 0.6 / 1.3 t/km2 the remaining gap is x5.6 / x3.5 against booked and x8-10 / x5-6 against the field variants. At 1.3 that is inside the encounter's own MC (95th percentile x3.7).
   - Even then the closure is partly compensation. Per species, pout stays 5-14x low, and the revised herring term is 2-5x above the in-area stomachs.

5. **The diet gate stays OFF after the proposal:** planktivore share 0.010-0.015 (0.023 at pout 1.3) against 0.075. Benthivore monophagy is the structural cause (exclusive per-prey allocation with saturated bed pairs, section 232) and is not fixed through a.

## Parameters: current vs proposed

| Key | Current | Proposed (central, range) | Tag | Basis |
|---|---|---|---|---|
| ben -> plk `max_intake_rate` | 0.0062 | **0.013** (0.010-0.020) | D | 0.019 (per capable tonne) x overlap x1.87 (GOV) x membership 0.377 instead of the gape phi 0.32. Caveat: this raises the herring term, which the in-area stomachs do not show |
| ben -> plk `stage_access` | [0.49, 0.14] | **[0.45, 0.14]** (band to [0.64, 0.19]) | D / J | Encounter-weighted window [0.519, 0.160] x0.86 for blue whiting. The band is the herring-specific p95. The f_J/f_A ratio is about preserved. a x f_eff 0.0017 -> 0.0034 (0.0049) |
| ben -> plk `predation_mortality` | 0.145 | unchanged; **supported** | F/D | In-area 0.12-0.30. Re-derive it if the pout biomass changes |
| ben -> plk `dark_ratio` (0 deg) | 0.06 | 0.2 (0.1-0.35), low priority | J | Annual a is unaffected (normalised); monthly multiplier 0.56-1.44 now vs ~0.69-1.33 at 0.2. The encounter note's own field-based range is 0.1-0.35: crepuscular herring peaks in cod stomachs, cod leaving the bed at night. Pure static vision x overlap with the revised overlaps gives ~0.04 |
| ben -> plk `visibility_floor` | 0.25 | unchanged; flagged | J | The field clearance already includes prey evasion. If prey hiding stays on top, the share falls another ~4x (0.003 at 25 % visible) |
| planktivore pool: Norway pout B | 0.25 t/km2 | open: 0.6-1.3 | D / J | Walker q, stock TSB per habitat area, the mortality check (section 4 of evidence.md) |

## Concurrent-search diet shares (all prey visible)

| Model | infauna | epifauna | Nephrops | planktivores | whiting |
|---|---|---|---|---|---|
| current | 0.629 | 0.328 | 0.038 | **0.005** | 0.000 |
| proposed | 0.625 | 0.326 | 0.038 | 0.011 | 0.000 |
| proposed, herring-specific ratio | 0.623 | 0.325 | 0.037 | 0.015 | 0.000 |
| proposed, pout 1.3 t/km2 | | | | 0.023 | |
| field V1 booked | 0.578 | 0.318 | 0.016 | 0.075 | 0.013 |
| field V3 evidence C | 0.554 | 0.305 | 0.015 | 0.102 | 0.023 |
| field V4 SMS (box length mix) | 0.551 | 0.303 | 0.015 | 0.104 | 0.027 |
| field V5 ICES deep | 0.541 | 0.298 | 0.015 | 0.123 | 0.024 |

Single-pair returns (share of the pair's own ceiling):
- Planktivores: 0.57 now -> 0.73 proposed with all prey visible; 0.25 -> 0.41 at 25 % visible.
- Bed pairs: 0.99.
- So under free time allocation the fish pair still pays less than the bed.

## Open points (user's call)

1. **Norway pout biomass and the pool composition.** This decides most of the remaining gap. Its catchability convention, q = 1 against Walker q, applies to pout only here.
2. **The herring overlap revision.** It is supported by the GOV/acoustic ratio but makes the herring term 2-5x higher than the thin in-area stomachs. Adopting it without the pout change moves the species mix the wrong way.
3. **visibility_floor**: the same double-discount convention question as for whiting.
4. **Housekeeping.**
   - `research_notes/Vitlingens fiskbyten/scripts/whiting_diet_ices.py` `area()` has no eastern longitude bound, so Baltic hauls at 17-21 E fall into "Skagerrak". Checked: no whiting record is affected (905 of 905 at lon <= 13), so the whiting results stand. The new script uses the bounded rule.
   - In the clearance test, the bed split is fixed across variants, so the test is planktivores against the bed only.
