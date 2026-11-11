# Evidence - ling's inputs in the water-column pairs (2026-11-11)

Grades: [F] field data parsed here or full text / figure read, [L] lab, [A] abstract only, [S] secondary / via
another note, [D] derived here, [J] judgement. Model facts M1-M5 are read from the exec'd derivations and the library.

## Model facts (where ling's inputs live)

- M1 Membership and weight: ling 0.014 of the water column's 0.617 t/km2 (0.0086 t/km2), library member list
  (research_notes/"Rovfiskens uppdelning" params_split_mack.md). Capability scores in research_notes/"Fiskgrupper i
  rutan"/predators/inputs/pred_scores.csv: ling benthos 0.40, plank_fish 0.60, zoo_krill 0.05, zoo_cop 0.00,
  benth_fish 0.60 ("judgement: demersal piscivore"); the whiting-prey pair reads benth_fish (split_encounter_223.py
  sc()). Membership is a 0/1 switch at score >= 0.1, so ling is in WC -> benthivores, -> whiting, -> planktivores
  and -> mackerel (EATERS), and NOT in WC -> krill (0.05 < 0.1).
- M2 Encounter (research_notes/"Attackrater ur motesteori" pred_encounter_a.py): TYPE "midambush", SPEED key
  v_hake (0.2, 0.5, 1.0) BL/s "J: ambusher in midwater, no measurement", no night speed factor (NIGHTV has no ling);
  N/t 550, number-mean L 62.9 cm, biomass-median 77.5 cm (tables/pred_numbers_per_tonne.csv, from
  research_notes/"Fiskgrupper i rutan"/predators/tables/pred_length_biomass.csv: GOV Q1+Q3 2001-25, Skagerrak + box
  rectangles, depth-stratified, CA length-weight 0.00371 L^3.088).
- M3 Band profile (research_notes/"Overlapp i fiskparen" band_pairs.py): ling exp(5 m) above the bed day and night,
  "[J] demersal; no field profile found"; rule A vision: R_sat = min(R0, 2 L) = 1.26 m binds.
- M4 Windows: the library window sets (research_notes/"Fonsterviktning" aw_stage_access.py WC_BEN / WC_WHT / WC_PLK)
  do not list ling; research_notes/"Vattenpelarens vuxna bottenfisk" decompose_members.py gives ling anglerfish's
  windows (p95 PPR 0.57): benthivores [0.970, 0.587], planktivores [0.993, 0.930], whiting [0.990, 0.834].
- M5 Q/B: no ling value; research_notes/"Fiskgrupper i rutan"/predators/inputs/species_values.csv carries ling with
  hake as physiology proxy and Q/B 2.0 (1.5-2.5); the sibling note used 2.0 (1.5-3.0) [J].

## Field and literature

- E1 NS-IBTS GOV (ICES DATRAS extracts on disk: research_notes/"Fiskgrupper i rutan"/data/extract and
  research_notes/"Torskfiskarnas artsammansattning" (a-umlaut)/data/extract), Q1 + Q3 2000-25, parsed here
  (`scripts/ling_gov_box.py`) [F data, D]. Box (57.7-58.4 N, 9.9-11.3 E): 436 hauls, ling in 73 (17 %), ~192 fish
  summed n/h; length 8-90 cm, number-mean 56.6 cm, biomass-median 64.5 cm, 40 % of numbers >= 60 cm; kg/h by bed
  class < 100 / 100-150 / 150-250 / >= 250 m: 0.16 / 0.45 / 1.53 / 0 (3 hauls in the last). Wider window
  (57.3-59.2 N, 8-12 E): 1533 hauls, ling in 160; L 8-150 cm, number-mean 58.1, biomass-median 69.5 cm. Ling's class
  shares (box 0.06 / 0.20 / 0.74 / 0) match the water-column group's GOV shape (saithe, 0.04 / 0.13 / 0.78 / 0.05):
  co-occurrence with the gate's benthivore J / A shapes relative to the group's 1.25 / 1.12 (box) and 0.89 / 0.98
  (window) - no member-level spatial correction is indicated. Ling IS on the box's trawlable ground, mostly 150-250 m.
- E2 ICES stomach database on disk (research_notes/"Torskfiskarnas bytesandelar" (a-umlaut)/data/diet/
  ices_stomach, CC BY 4.0), all years and areas, parsed here (`scripts/ling_ices_all.py`, `tables/ling_ices_items.csv`)
  [F data, anecdotal]: 84 ling stomachs in the North Sea + Skagerrak (50 recorded non-empty), 13 prey rows in 12
  stomachs of 30-88 cm ling: Norway pout (6, 15.5 cm), Trisopterus 17 cm, herring 24 cm, a 16.6 cm flatfish, a 6.2 cm
  cod (length from weight), Pontophilus, crustaceans. Fish-prey PPR (n 7): 0.07, 0.07, 0.20, 0.22, 0.25, 0.26, 0.28
  (median 0.22, max 0.28). By mass (235 g): planktivore members 0.61, benthivore members 0.39 (the one flatfish and
  the small cod). Ling outside the North Sea: none in the database.
- E3 IMR Toktrapport 2022-15 (de Jong et al., Hywind Tampen before development; HTML report and figures,
  hi.no/hi/nettrapporter/toktrapport-en-2022-15) [F text + figure, counts read from bars]: gillnets (74 and 90-93 mm
  half mesh), northern North Sea 61.3 N, stations 252-286 m and 136-141 m, late March - early April 2022. Ling the
  most numerous species (589 caught; 326 measured; 114 stomach samples listed); ling more abundant on the shallow
  stations; deep ling larger; more spawning ling on the shallow stations; medians ~1.0-1.1 m (Figure 12). Figure 14
  (`tables/hywind2022_ling_stomach_counts.csv`): of ~128 category records 58 empty, 50 unidentified teleost, 8
  pelagic teleost, 3 demersal teleost, 7 misc. invertebrates, 1 elasmobranch; no crustaceans. Identified fish: 8 of
  11 pelagic (the key's pelagic teleosts are mackerel, herring, blue whiting, garfish), 3 demersal (the key's demersal
  teleosts are long rough dab, haddock, Norway pout, whiting, redfish, rockfish, sculpin; which ones ling ate is not
  given). Mackerel "found in stomachs of hake, saithe, ling and cod". Counts, not mass; gillnet soak (feeding in the
  net cannot be excluded); spawning aggregation.
- E4 IMR Toktrapport 2024-5 (Tenningen et al., Hywind Tampen 2023 follow-up; HTML report and figures,
  hi.no/hi/nettrapporter/toktrapport-en-2024-5) [F text + figure]: 651 ling, raised length frequency 0.5-1.45 m, mode
  ~0.95 m; stations 251-280 m (slope, soft bed, nets drifted) and 136-142 m (even bed); 56 spawning ling, all
  shallow. Figure 15: ling stomachs mostly empty, inverted or unidentified teleost; a few pelagic and demersal
  teleosts; no crustaceans - while cod in the same hauls ate crustaceans, invertebrates and demersal teleosts.
- E5 MFRI HV 2024-01 (Solmundsson et al., Faeda 36 tegunda botnfiska a Islandsmidum 1996-2023), pp. 138-139 read here
  from the excerpt on disk (research_notes/"Sejens diet pa unga torskfiskar" (a-ring)/data/excerpts) [F figure]:
  haddock ~8 % of ling's prey weight ("um 8% af thyngd faedu ... longu", Fig. 159c, 8.2 % read from the bar), cod
  ~6.7 % (Fig. 160c); ling haddock + cod ~15 % by weight, all sizes, Icelandic waters 1996-2023. Flatfish, prey sizes
  and ling n are in the report's ling chapter, not on these pages (not fetched: no DOI).
- E6 Lokkeborg, Skajaa & Ferno 2000, J. Exp. Mar. Biol. Ecol. 247:195-208, doi:10.1016/S0022-0981(00)00148-9,
  PubMed 10742504 abstract read in full (E-utilities) [A]: 5 ling tagged in situ (transmitter in bait), tracked 6-11
  days by a fixed 3-hydrophone array (position every 3 min); crepuscular activity, higher swimming at dawn and dusk; a
  home range throughout, "during inactive periods (about 65% of the time) they remained within a small core area";
  with mackerel-baited gear present ling moved more slowly in a restricted area; "responded to and located baits in
  both their active and inactive periods during the day, but were not observed to react at night"; compared with
  cod, ling "less active at night", "more restricted use of space", "a higher proportion of mobile prey and being
  more dependent on visual stimuli". Swimming speeds, fish sizes, depth and bed type: in the closed full text.
- E7 ICES CM 1997 (doi:10.17895/ices.pub.27150858, abstract via OpenAlex) [A]: cod and ling tagged by swallowed
  bait-wrapped transmitters "close to the sea bed", movements from a stationary positioning system; expected
  gillnet encounters per fish per 12-h night soak 0.077-0.614 for ling (0.127-0.626 cod). No speeds in the abstract.
- E8 Andrews & Brooks 2026, Front. Mar. Sci., doi:10.3389/fmars.2026.1743207 [F full text]: 98 ling (rod and line
  and gillnet, Hywind Tampen / Snorre, March-May 2024; size outliers < 55 and > 150 cm excluded); isotopes place ling
  between saithe (most pelagic, lowest d15N) and tusk (most benthic); "ling diet comprising of greater proportions
  of fish than tusk (Bergstad, 1991; Tenningen et al., 2024)"; ling and tusk "more sedentary" than saithe.
- E9 Sara et al. 2009, Helgol. Mar. Res. 63:309-315, doi:10.1007/s10152-009-0159-6 [F full text, n 5 ling]: SW
  Iceland, ling the highest d15N (16.2) of the demersal fish; their text reads earlier work as ling "preying almost
  exclusively on larger demersal species (i.e. cod, haddock, saithe, wolffish) and benthic fish (mostly plaice)"
  [S for that reading].
- E10 ICES stock annex lin.27.346-91214 (2026; on disk in research_notes/"Fiskgrupper i rutan"/predators/data/
  lit_life_history) [S/F]: "prefers hard seabed, or sandy seabed with large rocks", 60-1000 m, mainly 300-400 m
  (Pethon 2005); "alone or in small schools" (Gordon et al. 1995); feeds "mainly on fish, but also on crustaceans,
  cephalopods, and echinoderms"; growth Linf / K: northern North Sea 189 / 0.080, 4.a + 6.ab 183 / 0.118, BHAC 160 /
  0.09; W = 0.0055 TL^3.012 (Norwegian fleet); M 0.15 assumed; L50 63.6 cm; NS-IBTS occurrence rising in the
  northern North Sea and Skagerrak.
- E11 FishBase (on disk: summary and growth tables; FishBase v24.07 parquet tables) [S]: "mainly on rocky bottoms",
  100-400 m usual; length-weight 0.0038 L^3.08 (Bayesian); swimming mode subcarangiform; no Q/B, ration, speed,
  oxygen or caudal aspect-ratio record for Molva molva; food items: cod (adults), flatfish (adults), herring, blue
  whiting, saithe, Norway pout, poor cod, gurnard, Nephrops, cephalopods, echinoderms.
- E12 Palomares & Pauly 1998 empirical Q/B (the Ecopath default; equation in `scripts/qb_ling.py`) applied to
  E10 growth and length-weight at the box bottom 7.4 C [D]: Q/B 1.23-1.59 (A 0.8-1.6 [J]), 1.22-1.61 over 6-9 C.
- E13 SMS North Sea key run 2023 cod ration a_q W^0.786 (consum_ab.in on disk, research_notes/"Fiskgrupper i rutan"/
  predators/data/diet/sms_consumption) at ling's box length mix [S proxy, D]: Q/B 3.8 (biomass-weighted), 4.1 at 2
  kg, 2.7 at 15 kg - the ration of a more active gadoid (E6: cod more active and more mobile than ling).
- E14 Via research_notes/"Vattenpelarens vuxna bottenfisk" evidence E11-E15 [S]: ICES extract by area (ling NNS
  2000+ 25 fed < 40 cm: Norway pout 92 %, cod 5.7 %; Skagerrak 16 fed: herring 81 %, Trisopterus 19 %); Bergstad
  1991 (Norwegian Deep: cod juveniles and adults, saithe, blue whiting, Norway pout named; no shares reachable);
  Brooks, Beylich & Hawley 2026 (tags at ~300 m: three just above the floor at all hours, two rising 250-280 m at
  night; species attribution ambiguous between ling and tusk).
- E15 Mauchline & Gordon 1984, JMBA 64:657, doi:10.1017/S0025315400030320 [A]: Rockall Trough ling "primarily fish
  eaters".

Not found anywhere: a ling Q/B, ration, gastric evacuation or bioenergetics model; an in-situ or tunnel swimming
speed (the tracking speeds are in the closed Lokkeborg et al. 2000 text); a ling vertical profile from depth tags
(E14 is ambiguous); a size-resolved ling diet with prey lengths beyond the 7 ICES items; an Ecopath group for ling
(EcoBase models on disk: North Sea 413 / 457 / 680, West Scotland 461, Faroe 46, Barents 63 / 64, Malangen 751 -
blue ling only, Porsanger 761 - none).
