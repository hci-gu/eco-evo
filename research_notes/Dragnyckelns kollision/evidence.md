# Evidence: inputs, keys and collision counts per script (2026-11-09)

All runs: `nice python3 -I`, from the note's folder unless stated. R = /home/simon/eco-evo/research_notes.
Extracts: E00 = R/"Fiskgrupper i rutan"/data/extract (NS-IBTS Q1/Q3 2000-20; HH all hauls, CPUEL 56.5-59.5 N,
6.5-12.5 E), E21 = R/"Torskfiskarnas artsammansättning"/data/extract (2021-25, same extractor),
E91 = R/"Tumlarens byten"/data/extract (1991-96). XML = R/"Fiskgrupper i rutan"/data/datras (HH and CPUEL XML
2000-2020, whole North Sea).

## Collision census [D]
- HH keys (Y, Q, Ship, HaulNo) with more than one position (> 0.02 deg apart):
  - E00: 765 keys, years 2011-2020, mainly 26D4 DK/SE (DK 760, SE 723, DE 77 rows);
  - E21: 165 keys (2021, 2024, 2025; DK/DE);
  - E91: 0.
- Box (57.7-58.4 N, 9.9-11.3 E), valid GOV Q1/Q3 2000-20: 131 of 354 collide (SE 127, DK 4).
  - Partners lie at 51.5-60.2 N, -1.3 to 11.5 E (DK 125, DE 10, SE 2); 9 of 137 are inside the CPUEL extract
    window.
  - Region 57.5-58.9 N, 8.5-11.8 E: 375 of 972 collide; 5 pairs lie both inside it.
- HH vs CPUEL positions agree exactly at 4 decimals: 0 mismatches of 1877 (E00) and 450 (E21) CPUEL hauls.
- Closest pair: 2015 Q3 26D4 haul 1 is SE 57.4870 N 9.1378 E (21 Aug) and DK 57.4871 N 9.1254 E (28 Aug).
- Independent check (main session): 2016 Q1, 26D4 haul 18 is DK 54.4449 N 1.6753 E and SE 57.8805 N 11.0015 E
  in HH_2016_Q1.xml.gz, both valid GOV.

## Per script

### Tumlarens byten/scripts/then_now_densities.py: CONTAMINATED
- Inputs: E91, E00, E21, R/"Fiskgrupper i rutan"/tables/length_weight_all.csv. The original reproduces
  `tables/then_now_densities_stdout.txt` byte for byte with these inputs, so the scratch-only 2021-25 extract of
  the note's README equals E21.
- Key: 4-field `drop_duplicates` on HH (before region selection) and a 4-field CPUEL join.
- Hauls lost by the dedup [D]:
  - E00: Skagerrak 218 / 980, Kattegat 136 / 713, IIIa 354 / 1693, box rectangles 105 / 539, all 2013-2020;
  - E21: Skagerrak 7 / 239 (2025);
  - E91: 0.
- Region hauls given another haul's catch [D]: E00 Skagerrak 24, Kattegat 6, IIIa 30, box rectangles 5; E21 0;
  E91 0.
- Fixed: `scripts/then_now_densities_fixedkey.py` (dedup on the key plus ShootLat/ShootLong; CPUEL row ->
  nearest same-key HH haul, < 0.05 deg).
  Command: `python3 -I scripts/then_now_densities_fixedkey.py data/extract "$R/Fiskgrupper i rutan/data/extract"
  "$R/Torskfiskarnas artsammansättning/data/extract" "$R/Fiskgrupper i rutan/tables/length_weight_all.csv"
  > tables/then_now_densities_fixedkey_stdout.txt`.
- Check:
  - E91 hauls are identical (511 / 511), and so are E21 hauls with a unique key (401 / 401); E21 gains the 7
    hauls.
  - E00 gains 355 hauls, and 31 kept hauls change (the ones that had received other catch).
- Downstream: `scripts/clearance_test.py` hard-codes the Skagerrak ratios.
  - Copy: `scripts/clearance_test_fixedkey.py` (the ratios only: herring 0.98, blue whiting 0.90, pout 4.65,
    sprat 0.51, ben 1.06, wht 0.91, wc 0.31, wc < 40 cm 0.88).
  - Command: `python3 -I scripts/clearance_test_fixedkey.py > tables/clearance_test_fixedkey_stdout.txt`.

### Torskfiskarnas artsammansättning/scripts/gadoid_mix.py: haul loss only
- Inputs: E21 and /home/simon/eco-evo/maps/depth.npz. The original reproduces `tables/gadoid_mix_stdout.txt`
  and all its CSVs byte for byte.
- Key: 4-field dedup before region selection, then a 4-field join. 7 of 272 region hauls are lost: Danish
  Skagerrak 2025 hauls 1-3 (Q1) and 1-4 (Q3), whose German partners lie at 55.7-57.6 N, 4.5-6.2 E. 0 box hauls
  are lost, and 0 catch is added (E21 CPUEL has no far rows on region keys).
- Fixed: `scripts/gadoid_mix_fixedkey.py`. Its CSVs go to `tables/fixedkey/` (a new folder, so the original
  tables stay).
  Command: `python3 -I scripts/gadoid_mix_fixedkey.py data/extract /home/simon/eco-evo/maps/depth.npz
  tables/fixedkey > tables/gadoid_mix_fixedkey_stdout.txt`.
- Result: box rows identical; Skagerrak 232 -> 239 hauls; < 100 m bin 177 -> 184 hauls.

### Hauls_region.csv consumers (2021-25; inherit the 7-haul loss; own 4-field join clean)
Each was rerun UNCHANGED on `tables/fixedkey/hauls_region.csv`, and each original reproduced its stored stdout
first.
- `Fiskgrupper i rutan/scripts/fish_guilds.py`
  - Inputs: E21, the haul list, `inputs/guilds.csv`, depth.npz.
  - Output: `Fiskgrupper i rutan/tables/fish_guilds_fixedhauls_stdout.txt`.
  - Planktivores pooled 261.0 -> 275.0 kg/h; benthivores 292.3 -> 292.2; predators 252.9 -> 252.4.
- `Attackrater ur mötesteori/scripts/size_structure.py`
  - Inputs: E21 CPUEL, the haul list, length_weight_all.csv, depth.npz.
  - Output: `Attackrater ur mötesteori/tables/size_structure_fixedhauls_stdout.txt`.
  - Herring pooled N/t 2.885e4 -> 2.709e4, M2 7.949e6 -> 7.804e6; other species < 0.1 %.
- `Interferens hos bottenfisk/scripts/ibts_patchiness.py`
  - Run on a scratch dir with symlinks (tables/hauls_region.csv -> the fixed list; data/extract -> E21).
  - Output: `Interferens hos bottenfisk/tables/ibts_patchiness_fixedhauls_stdout.txt`.
  - Benthivore guild Lloyd median 3.06 (unchanged), pooled 2.95 -> 2.93.
- Not rerun: `catchability.py`, and the `Torskfiskarnas bytesandelar` scripts `length_biomass.py`,
  `pelagic_prey_sizes.py` and `stock_age_length.py` (4-field merges on E21 and the same haul list).
  - `prey_shares.py` merges its own region-only outputs on a key that is unique within them.
  - Same inherited 7-haul effect; box rows unchanged.

### Fiskgrupper i rutan/benthivores/scripts/prey_length_gape.py: clean
- Inputs: E00 and E21 CPUEL only (no HH). Rows are filtered on their own ShootLat/ShootLon (57.0-59.2 N,
  7.5-11.5 E, 50-300 m), so no outside haul enters.
- The dedup on 4 fields + AphiaID + Sex + LngtClas meets 2 keys shared by two in-region hauls (2017 Q3 26D4 51;
  2018 Q3 26D4 2). It drops 49 of 107 225 rows, under 0.1 % of numbers for every species.
- The stored "hauls per quarter 574 / 438" = 1012 of 1014.

### Vitlingens predatorer/scripts/gov_overlap.py: CONTAMINATED (known)
- Inputs: XML 2000-2020. Box: 131 / 354 collide, 2 merged (352). Region: 375 / 972 collide, 5 merged (967).
- The fixed copy already exists: `../Vitlingens uteblivna predation/scripts/gov_overlap_fixedkey.py`
  (key + "%.4f" ShootLat). Box stdout there.
- Region run here, from "Vitlingens predatorer": `python3 -I "../Vitlingens uteblivna
  predation/scripts/gov_overlap_fixedkey.py" "../Fiskgrupper i rutan/data/datras" region
  > tables/gov_overlap_fixedkey_region_stdout.txt`.

### Krill i Skagerrak/a_revisit/scripts/gov_by_bed.py: CONTAMINATED (known)
- Inputs: XML 2000-2020 (not 2000-19 as its docstring says).
- Fixed: `scripts/gov_by_bed_fixedkey.py` (key + "%.4f" ShootLat).
- Commands, from a_revisit: `python3 -I scripts/gov_by_bed_fixedkey.py "$R/Fiskgrupper i rutan/data/datras"
  > tables/gov_by_bed_fixedkey_stdout.txt` and the same with `box` > `tables/gov_by_bed_fixedkey_box_stdout.txt`.
- The box run reproduces 0.769 / 0.204 / 0.027 / 0.001 and 354 hauls.
- `realised_krill.py` has no haul key. `weighted_gate.py` hard-codes the gov_by_bed shares (downstream; not
  rerun).

### Krill över grunda bottnar/stomachs/scripts/link_datras.py: clean
- It links 1980-91 stomach hauls; the linked ships are 58M2, 06DA, 749S, 74EX and others, with no 26D4.
- Match A needs the same or an adjacent ICES rectangle, and the HH dedup includes ShootLat.

### Other scripts with HaulNo: clean (key carries position, Country or StNo)
- Fiskgrupper i rutan: long_series, long_species, benthivores/length_structure, predators/pred_season_length.
- Single scripts: Fönsterviktning/blue_whiting_lengths; Årsungarnas andel/juvenile_share_by_quarter and
  datras_q2q4_fetch; Makrillens bytesstorlek/mackerel_planktivore_gape; Storleksfönster som standard/box_stages;
  Bottenfiskens sillbyte/reconcile_gov_nearbed.
- Sejens diet på unga torskfiskar: box_gape_by_quarter and the three datras_* scripts.
- Beståndsreglering: share_survey and the mig_* scripts.
- Sejens djupläge: extract_gov, gov_saithe_box, gov_prey_overlap, gov_season_route.
- Vitlingens uteblivna predation: all scripts.
- Extractors without a join: extract_datras, share_extract, mig_extract_cpue_age.

## Trace: water column -> whiting 0.157 and field 0.13
- Booked 0.157 = sum over members of (diet share of whiting) x Q/B x B_member / B_whiting [S, "Vitlingens
  predatorer" evidence 2; `../Fiskgrupper i rutan/predators/inputs/diet_consumption_central.csv`]:
  - saithe 0.02 x 1.0 x 3.0 x 0.492 / 0.446 = 0.066;
  - hake 0.21 x 0.9 x 3.5 x 0.030 / 0.446 = 0.044;
  - spurdog 0.22 x 0.5 x 2.0 x 0.060 / 0.446 = 0.030;
  - anglerfish + ling 0.017.
  - B: 0.617 x 0.797 saithe etc., member shares from `pred_season.csv` (position-keyed); whiting 0.446 =
    1.20 x 0.3715 (long-run, position-keyed) [S, params_split_223 2].
- Field 0.13 (0.06-0.20) = saithe 0.066 (0.023-0.10) + hake 0.030 (0.008-0.057) + spurdog 0.030 + anglerfish
  and ling 0-0.017 [S]. The SMS saithe -> whiting 1+ 0.085 is a comparison in the library comment.
- Neither has an overlap or spread step, and neither touches the collided key or the bed shares.
- The only overlap step is the note's alternative ("box segregation on saithe") -> recomputed in summary.md.
