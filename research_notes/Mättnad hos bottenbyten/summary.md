# Saturation on benthic prey: why the two earlier notes disagree (after resume 223.27)

Written by the research subagent, saved by the main agent (2026-10-27). Research only: no library,
manifest, code, resume or AGENTS.md edit. Scripts in `scripts/`, outputs in `tables/`, downloads in `data/`.
[A] field/experimental measurement, [F] full text read, [D] data analysed here, abs = abstract only,
J = judgement. Sub-notes: `tables/Q1_notes.txt` (field/lab responses), `tables/Q2Q3_notes.txt` (search,
handling, armour).

## Headline

1. Both notes are right, about different quantities. The Exposure note's a and e stand: no search-side
   step lowers a by 5-50x (Q2); lab juvenile cod on sand Crangon give a = 0.065-0.13 km2/t/tick (Persson
   et al. 2012 [F][A]), the library's 0.15. The Feeding-level note is right that the response to ONE prey
   rises at field densities: cod on Pandalus at stock scale sit on the lower limb, x ~0.25-0.5 (Hvingel &
   Kingsley 2006, WebFetch extract); diets track local prey over 5-250x contrasts while total intake stays
   flat (Smith et al. 2013 [F]; Berenboim et al. 2000 [F]).
2. They fit together only if co-occurring bed prey share one search and one gut (the multi-prey Holling
   form, Murdoch 1973; one ceiling per gut, Jeschke et al. 2002 [F]): I_i = a_i f_i B_i / (1 + X_bed),
   X_bed = sum of x_j over the accepted bed prey. The engine gives each pair its own denominator
   1 + a_i h_i f_i B_i + wP and the policy splits `eat` over pairs (`predation.py`,
   `interactions.holling_a_eff`), i.e. bed prey as separate patches. Benthivores today: X_bed = 123 [D];
   per pair x_eff = x_i/(1 + X - x_i) = 1.2 (infauna), 0.7 (epifauna), 0.05 (Nephrops), elasticities
   0.46, 0.59, 0.96; at 10 % of a prey the pair keeps 0.10-0.20 of its intake vs 0.42-0.88 per-pair.
   Structural point, not a library value.
3. Three mechanisms, each on its own quantity (J): (i) shared search and gut set each pair's elasticity;
   (ii) renewal limits the supply of exposed infauna parts (aP 80 /yr vs k 1.5-3 /yr; the refuge carries
   it); (iii) consumer density (Kattegat) stays open. None explains the field feeding level 0.22-0.53 of
   Cmax (X_bed still fills the gut); that level is time/risk (a predator cue cut cod intake to 0.22-0.44,
   Persson 2012 [F]).
4. Whiting stays saturated: its only checked bed pair is epifauna (x_eff 35); with a diet-ratio infauna
   term on the same bed it would be 2.5 (J; unchecked pair). Armour, which halves the energy ceiling on
   crustaceans, is the only non-structural lever found.

## Q1. Direct evidence

No field functional response in density units exists for ADULT gadoids or flatfish on benthic invertebrates.
- Kattegat (Hiddink et al. 2016, abs only; stomach data not public): food per plaice rose then fell above
  ~5 passes/yr, condition and stomach contents "mirrored" it (also long rough dab; not dab). Hinz et al.
  2017 [F], same stations: prey biomass not related to trawling (p 0.67), consumers fell (R2 0.51), the
  prey:consumer ratio domed (R2 0.47). Dryad reanalysis [D] (`tables/kattegat_station_analysis_stdout.txt`,
  15 fine-sediment stations, excluded set J): prey (large infauna excl.) 98-302 t/km2, ~65 % Amphiura
  bodies, x0.90 low to high trawling; flatfish 0.11-0.45 t/km2 at q = 1 (lower bound), x0.46-0.62;
  condition independent of B (r2 <= 0.08), tracks trawling and P weakly (LRD r2 0.26). A response to B
  alone fails at any x (predicted change <= 2.5 %).
- Irish Sea (Johnson et al. 2015 [F]; Hinz data [D], `tables/field_density_x_stdout.txt`): prey 2.4-18.5
  t/km2; low to high trawling prey x0.45, flatfish x0.46; plaice and dab stomach fullness and content flat
  (p 0.47-0.95, 15 stations). The shared form at X ~6 predicts 0.84 of intake; 15 stations would not
  detect it (J).
- Cod on Pandalus, West Greenland (Hvingel & Kingsley 2006): Holling III, P50 3.6-3.8 B_MSY vs observed
  0.96-1.70 -> x ~0.25-0.5; P50 rests on the prior.
- Barents Sea (Berenboim 2000 [F]): shrimp occurrence in cod stomachs falls with cod biomass; stomachs hold
  "those prey species which are most abundant".
- Georges Bank (Smith 2013 [F]): epifauna differed 5-250x between sites; haddock diets tracked it; total
  intake no consistent sign.
- Baltic cod: intake responds only at collapse scale (Neuenfeldt et al. 2020, extract); weak within normal
  variation (Lindmark et al. 2024 [F]).
- Lab, Persson 2012 [F][A]: a = 1.8 (Crangon), 21 (Palaemon), 57 (Neomysis) cm2/s; x against digestion
  4-1000 at field densities; habitat structure cuts a 5-9x.
- Verdict: saturated per foraging fish in a patch; rising per pair and per cell; total bed intake flat.

## Q2. Search side

Time budget: cod near the bed 57-90 % of the time (Hobson 2007, 2009 [F]); foraging share 0.5-0.9 gadoids,
0.3-0.6 flatfish (J): <= 3x, absorbed by the gut buffer. Burial 12-32x (Persson 2012 [F]), already in e.
Missing: strike success on retracting prey, giving-up densities, offshore per-fish predation rates.
Prey-side rates (`tables/saturation_arithmetic_stdout.txt`): infauna aP/k 27-54, cropping roughly equals
regeneration (de Vlas 1979 via Lees 2024 [S]); epifauna diel k (365-730 /yr), arena factor only 0.5-0.8.
The step both notes missed is shared search time over co-occurring bed prey; it changes the per-pair
response, not a.

## Q3. Handling

Per item (`tables/Q3_item_handling_stdout.txt`; ICES stomach item masses [D]; Arnott & Pihl 2000 [F]
times): whole items need <= 3 h/d at Cmax (ratio <= 0.2), so h = E/Cmax holds. Cropped parts (2-3 mg, 3 s
J) need 7 000-40 000 items/d = 6-34 h/d, ratio 0.23-1.4 for fish of 0.5-2 kg: a large fish cannot fill
its ration from parts (Jeschke 1/max(b, c)); this raises x, it does not unsaturate. Armour (Andersen 1999
abs; Andersen et al. 2016 abs; Arnott & Pihl 2000 [F], satiation meal 2.8 % WW on crabs vs 9.1 % Crangon
vs 17.8 % fish): h factor (J) shrimps x1.7-2, crabs and hermits x2-3, Nephrops ~x2 (unmeasured), mysids
~x1. It flips the library's epifauna advantage (assimilation 0.8 vs 0.7) into a disadvantage per gut-time.
Flagged.

## Q4. Per-pair synthesis (`tables/synthesis_pairs.csv`)

| Pair | per-pair x | x_eff shared (h x2) | elasticity | carried by | independent test |
|---|---|---|---|---|---|
| benthivores -> infauna | 67 | 1.2 (0.6) | 0.46-0.63 | shared search and gut; refuge renewal | Hiddink 2011 (abs); Kattegat condition flat over B [D] |
| benthivores -> epifauna | 50 | 0.7 (1.3) | 0.44-0.59 | shared search and gut; armour | Smith 2013; Berenboim 2000 |
| whiting -> epifauna | 35 | 35 (2.5 with infauna term, J) | 0.03 | unresolved; armour | whiting feeding level 0.25 Cmax, epifauna 13.7 % of diet |
| benthivores -> Nephrops | 5.5 | 0.05 (0.07) | 0.95 | minority prey in the shared denominator | Irish Sea cod M2 on Nephrops 0.04-0.25 /yr |

The diet split (0.63/0.33/0.04 vs field 0.54/0.31/0.015) is the Exposure note's ratio test, not new
evidence. Diagnostic only (not a test): shared-form M2 at a feeding level 0.36-0.54 (J) vs budget M2 -
infauna 0.05-0.07 vs 0.040; epifauna 0.60-0.90 (0.42-0.62 with armour) vs 0.54; Nephrops 0.06-0.09 vs
0.0225. Consumer density: a HYPOTHETICAL 20 % effect would need BD w = 4-14 km2/t at x 0.5-4, or 30-180 at
x 35-67, vs 0.45; candidates local depletion and area-restricted search at home-range scale (27 ha,
Espeland 2007 abs; Hill 2002 abs), unquantified.

## Gaps

Hiddink 2016 full text and stomach data (Bangor behind Cloudflare, IEO behind Anubis; not circumvented);
Kattegat tow length (J) and catchability q; lab responses on infauna or cropped parts; strike success;
offshore predation rates on crabs, Pandalus and Nephrops; accelerometer budgets; a Skagerrak Pandalus index
vs cod/whiting stomach occurrence (a local test of the shared form). Prey selectivity: Johnson's Chesson
indices decide which pairs enter X, not whether denominators are shared (J).
