# Evidence: bed depth of the benthivores' recruits and juveniles (2026-11-11)

Grades: [F] field data or full text, [S] secondary (read via another note), [A] abstract only, [D] derived
here, [J] judgement. No PDF was downloaded for this note; literature is read through sibling notes, named
with each item.

## Data

- **E1. NS-IBTS GOV, Q1 + Q3 2000-25** [F data]. ICES DATRAS (CC BY 4.0), HH and CPUE-per-length extracts with
  every species: research_notes/"Fiskgrupper i rutan"/data/extract (2000-20) and research_notes/"Torskfiskarnas
  artsammansattning" (a-umlaut)/data/extract (2021-25). Read by `scripts/gov_juv_bed.py`. Valid GOV hauls with
  all species recorded (BySpecRecCode 1), haul key year/quarter/ship/haul/shoot position. Box (57.7-58.4 N,
  9.9-11.3 E): Q1 127 / 103 / 26 / 0 hauls on < 100 / 100-150 / 150-250 / >= 250 m beds, Q3 83 / 40 / 54 / 3;
  window (57.3-59.2 N, 8-12 E): Q1 542 / 138 / 69 / 24, Q3 561 / 74 / 112 / 13. Every Q3 haul is a day haul.
  Members: haddock, cod, long rough dab, dab, plaice, witch, starry ray, hagfish, lemon sole, grey gurnard
  (library member shares; lemon sole and gurnard 0.009 each from params_benthivorous_fish).
- **E2. NS-IBTS GOV, Q2 + Q4 1991-97** [F data]. research_notes/"Sejens djuplage" (a-umlaut)/data/gov_q2q4_1991_97
  (haddock and cod only). Box Q2 22 / 20 / 5 / 1 hauls, Q4 4 / 8 / 6 / 0. Read by `scripts/gov_q2q4_check.py`.
- **E3. DATRAS CA 2021-25** [F data]. research_notes/"Torskfiskarnas artsammansattning"/data/datras/CA_*.xml.gz.
  Mean length at age 1 in Q3, Skagerrak rectangles 43-47 F7-G1: haddock 27.0 (465 aged fish), cod 27.1 (822),
  plaice 16.0 (250), witch 14.8 (25) cm - the library's cuts, reproduced. Dab, long rough dab and lemon sole
  have no age readings at age 1 in Q3 anywhere in the NS-IBTS 2021-25 CA (dab 5 fish, lemon sole 3, long rough
  dab 0): no field cut exists for them (`scripts/flatfish_age1_length.py`).
- **E4. maps/depth.npz** [D]. The model's 60 x 60 bed depth (44-529 m): cell shares 0.2317 / 0.2992 / 0.3236 /
  0.1456 per class. Used for the equal-cell normalisation (the saithe note's script hard-coded 0.21 / 0.31 /
  0.33 / 0.15).

## Model and library facts

- **M1. J definition** [library; research_notes/"Arsungarnas andel" (A-ring) parameters_B.md section 0]: J = the
  fish below the species' mean length at age 1 in Q3 (the size at their second summer), haddock 27.0, cod 27.1,
  plaice 16.0, witch 14.8 cm; species without an age key (long rough dab, dab, lemon sole, rays, gurnard,
  hagfish) count as adult; box J/B 0.15 (Q1) / 0.16 (Q3), "juveniles ~90-95 % haddock" (library comment).
- **M2. Recruit import** [library juvenile_stage comment; research_notes/"Bestandsreglering" (a-ring)
  box_share.md, Oppen population summary.md]: recruit_import 0.183 t/km2/yr = SMS stock recruits x the box's
  survey share, haddock age 0 in Q3 (< 20 cm) 190-323 t/yr and cod age 1 in Q1 (10-27 cm) 336-469 t/yr
  (midpoints 256.5 / 402.5: 0.39 / 0.61 of the import tonnes). The box's share of cod age 0 in Q3 is only
  0.3-0.5 % of the stock against 3.7-5.2 % for age 1 in Q1: most cod recruits enter between August and January.
  recruitment months Jun-Sep 0.2 / 0.5 / 0.5 / 0.2.
- **M3. Engine** [code]: `stages.exchange` adds I dt shape(t) per cell into J, uniformly (`per_cell` a scalar;
  `imported = per_cell x cells`), at the month's start_fill; GPU mirror `lib/gpu/ecosystem.py`
  `_stage_exchange` (`stage_import` [T, N_all, 1] broadcast over cells; the record sums `imported x C`). J moves
  with B (the same movement fractions) and is scaled with B by every whole-group loss (stages.py docstring);
  `initialise` sets J = B x start_share uniformly. `open_pool.depth_weight_map` renormalises a depth table to an
  equal-cell mean 1 on the run's grid; `open_pool._parse_rows` validates rows (values >= 0, one > 0).
- **M4. Gate** [code]: `tools/probes/budget_gate.py` places a prey by `field_diet.field_prey_shape` (open-pool
  shape, FIELD_BED_SHARES, else uniform) times `_visible_share` (the annual-mean start J/B mixing f_J and f_A),
  so the benthivores are uniform; only predatory_fish and whiting have field spreads.
- **M5. Pair depth tables** [library]: predatory_fish -> benthivores 0.0693 / 0.057 / 0.0234 / 0.0128 by class
  (mean-1 on the 60x60 map 1.63 / 1.34 / 0.55 / 0.30); whiting -> benthivores 1.28 / 1.15 / 0.80 / 0.69 (mean 1).
  Stage access [0.392, 0.0388] and [0.038, 0.0064].

## Literature (via sibling notes)

- **E5. Albert 1994**, ICES J. Mar. Sci. 51: 31-44, doi:10.1006/jmsc.1994.1004 [S, read as scanned images in
  research_notes/"Sejens diet pa unga torskfiskar" (a-ring) evidence_settlement.md item 8]. Norwegian Deep slopes
  incl. the Skagerrak, 1984-87, bottom trawl at 80-300 m only. "All catches of more than 500 specimen h-1 were
  taken in the depth interval 80 to 200 m"; 0-group summer frequency of occurrence 40.0 / 38.9 / 21.4 % in
  80-149 / 150-199 / 200-250 m, autumn 89.3 / 57.1 / 15.9 %, 1-group winter 81.3 / 75.0 / 46.7 %; "all age-groups
  of haddock were distributed deeper in winter than in summer and autumn". Summer 0-9 cm haddock stomachs held
  ophiuroids and polychaetes (already on the bed). No stations shallower than 80 m.
- **E6. Settlement timing** [S, same file, entries 6-9, 13]: haddock pelagic 0-group in June (4 cm), settlement
  transition July (6-8 cm), demersal in August (IBTS 12.4 cm, p10-p90 10-15), 13.7 cm in September; 0-group cod
  4-6 cm settle in July in the central North Sea. Arsungarnas andel section 2: no age-0 fish in the May survey,
  all in by August; haddock settle mid-Jul to mid-Aug.
- **E7. Munk et al. 1995**, MEPS 125: 21, doi:10.3354/meps125021 [S, via box_share.md]: cod larvae and juveniles
  along the 80-140 m isobaths of the inner Skagerrak slope front in May, 58-58.5 N east of 10 E (overlaps the
  box).
- **E8. GOV gear by depth** [S, research_notes/"Torskfiskarnas artsammansattning" evidence_B_catchability.md A2-A4]:
  Walker et al. 2017 q-at-length (haddock 0.012 below 11 cm, 0.12 above 46 cm; cod 0.16-0.67) is by length only;
  the NS-IBTS manual: standard stations to 250 m; Norway and Sweden vary sweep length with depth in Q1 (60 m
  below 70 m, 110 m deeper), Q3 uses 60 m everywhere - "since sweep length drives herding, Q1 and Q3 q may
  differ for herded species". No depth-dependent q is published. Wing spread in E1 rises with bed depth
  (class medians 20 / 22 / 23 / 24 m; 40 % of hauls record it).
- **E9. Saithe note** [S, research_notes/"Sejens hojd over botten" summary.md finding 3, gov_juv_overlap.py]:
  in-gape haddock (< 19 cm) box Q3 class shares 0.82 / 0.14 / 0.04 / 0; F_gate box Q3 0.19, Q1 0.55,
  presence-weighted 0.225, window 0.49, haul level 0.086 / 0.296; combined-central gate WC -> benthivores 2.00
  (2.33 at the field spread).
- **E10. Combined capture-p x schooling factors** [D, research_notes/"Fangst och stim i fiskparen"
  tables/gates/combined.json, central]: x2.96 on WC -> benthivores, x2.61 on whiting -> benthivores, others on the
  nine fish-prey pairs.
