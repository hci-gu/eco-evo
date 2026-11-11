# Saithe height above the bed and the water column -> juvenile benthivores pair - evidence (2026-11-11)

Research only. Nothing in the library, code, tests, manifest, AGENTS.md, resume or existing notes changed; no
training; no value set from model output (budget_gate ran on library copies in the session scratchpad).
Grades: [F] field data or full text read, [L] lab, [A] abstract only, [S] secondary / read from a figure via
another note, [D] derived here, [J] judgement. Scripts and their stdout: `scripts/`; tables: `tables/`.

## Saithe vertical position relative to the bed

E1. **HERAS / IESSNS pelagic hauls, whole stored region (54.6-59.1 N, 5-12.4 E), June-July** [F data, D fit]
(`scripts/heras_saithe_height.py`; data research_notes/"Ekolodsdata planktivorer"/data/derived, ICES acoustic-trawl
database, CC BY 4.0). Adult saithe (>= 35 cm, by measured weight), bed >= 100 m, day:

| footrope height above bed | hauls | with adults | mean kg/h | kg/h per m opening | bed median |
|---|---|---|---|---|---|
| 0-3 m | 12 | 10 | 39.4 | 7.1 | 129 m |
| 3-10 m | 4 | 4 | 160 | 23.8 | 167 m |
| 10-30 m | 4 | 2 | 458 | 20.1 | 115 m |
| 30-60 m | 3 | 2 | 9.1 | 0.44 | 180 m |
| 60-150 m | 15 | 0 | 0 | 0 | 130 m |
| > 150 m (surface) | 60 | 0 | 0 | 0 | 298 m |

Night: no bed-referenced haul; 11/77 night surface hauls over beds >= 100 m caught adults (0/60 by day).
Exponential fit (scale Hs, per-gear level; quasi-Poisson qp / presence-absence pa), with the GOV bed-class
density as a known offset (E7): bed >= 60 m qp 15 m, pa 30 m; bed >= 100 m qp 14 m, pa 31 m; bed >= 140 m
qp 9 m, pa 14 m (no offset: 7-31 m; 95 % profile intervals 5.5-85 m). Shape comparison (2 dll/phi vs exp):
a hump h exp(-h/s) (mode s 4.5-7 m) is preferred by the catch-rate likelihood (+5 to +14) but not by
presence (-5.7 to +1.0); bottom-2-m share 0.12-0.20 (exp, qp) vs 0.03-0.07 (hump, qp). Rests on 18 hauls with
adults (8 over beds >= 140 m), one of 1834 kg/h; hauls aimed at herring marks; nominal footrope heights.

E2. **Cross-gear check on the near-bed density** [D/J] (`scripts/gov_juv_overlap.py`). GOV (bottom trawl,
headline ~4-5 m) in the box, Q3 2000-25: adult saithe 172 kg/h over 150-250 m beds (54 hauls), against the HERAS
pelagic hauls at footrope 0-3 m over beds >= 140 m 21.6 kg/h (4 hauls) and at 3-10 m 184 kg/h (3 hauls). The
bottom trawl at the bed catches at the rate the pelagic trawl reaches only 3-10 m up, so the HERAS on-bed
deficit that drives the hump fit is plausibly a pelagic-gear artefact (footrope held off the bed, mouth areas
differ ~x1-2) [J]. Not a calibrated comparison.

E3. **Neilson et al. 2003** [F, full text on disk: research_notes/"Sejens diet pa unga torskfiskar"/data/
settlement/neilson2003_pollock_prespawning_acoustics.pdf], ICES J Mar Sci 60:860-871,
doi:10.1016/S1054-3139(03)00068-7. Scotian Shelf pollock (= saithe), pre-spawning aggregations, 30 Sep-3 Oct
1999, 38 kHz, Sa in 1-5, 6-10, ... 26-30 m off bottom, first metre dead zone: "while pollock can occur up to 30
m off bottom, the greatest proportion remains within 1-5 m off bottom during both day and night"; Sa larger in
the deepest layer and at night at both sites. 1-5 m shares read from Fig. 7 via research_notes/"Vattenpelarens
bottenfiskbyte" A1 [S]: Banana 27 % day / 37 % night, Shelf Edge 68 / 80 %. Implied exponential Hs
(`scripts/neilson_hs.py`) [D]: Banana 15.4 m day / 9.1 m night; Shelf Edge 3.5 / 2.5 m.

E4. **DFO CSAS 2003/075** [S via research_notes/"Vattenpelarens bottenfiskbyte" A2]: echo logger, 75 % of pollock
biomass within 10 m of the bed in Dec-Jan, 75 % height 3-13 m by bimonth -> Hs 2.2-9.4 m [D].

E5. **Armannsson & Jonsson 2012** [F, full text on disk: same folder, armannsson2012_saithe_dst_iceland.pdf],
ICES J Mar Sci 69:1372-1381, doi:10.1093/icesjms/fss121. 51 DSTs, 42-83 cm, Iceland: depth only, no bed
reference. Median depth 140 m Mar-Apr, < 50 m summer-autumn; vertical activity highest by day, "much
reduced" at night and in winter; saithe share of commercial bottom-trawl catches highest in darkness; they
cite "Compared with other gadoid species, saithe spend more time in the pelagic habitat and less time at the
bottom (Scott and Scott, 1988)". Gives no height-above-bed time budget.

E6. **Kaartvedt et al. 2005** [F, PDF in `papers/`], MEPS 299:1-5, doi:10.3354/meps299001: large saithe in
midwater below krill swarms (Norwegian Sea, deep water); they preyed on the planktivores attracted to the
krill. Saithe use midwater where prey layers are; no bed-referenced share.

E7. **GOV bed-class distribution of adult saithe** [F data, D]: 0.024 / 0.268 / 0.646 / 0.062 over < 100 /
100-150 / 150-250 / > 250 m beds (0.1 Q1 + 0.9 Q3; research_notes/"Krill i Skagerrak"/wc_depth/evidence.md 1.2);
box GOV 0.043 / 0.127 / 0.782 / 0.048 (lib/diagnostics/field_diet.py FIELD_BED_SHARES).

E8. Not obtained: Stensholt et al. 2002 (doi:10.1006/jmsc.2002.1249; publisher page: saithe "mainly
distributed within the bottom half of the water column", no layer shares) [A]; Skilbrei & Ottera 2016
(doi:10.1093/icesjms/fsv261; farm-attracted, ~100 m of column, already in the sibling notes) [S]; the 2018
Norwegian-German saithe acoustic-trawl survey cruise report (Thunen WH 414; fetch failed twice; a search
summary attributes to a North Sea survey report that saithe "mainly stay within the bottom channel (10 m above
seabed)" - unverified, not used) [S, unverified]; Nedreaas 1997 NEA saithe acoustic survey evaluation (no
vertical numbers in the abstract).

## The juvenile prey

E9. **Lough et al. 1989** [F, full text on disk: same folder, lough1989_georgesbank_juvenile_haddock.pdf], MEPS
56:1-12, doi:10.3354/meps056001. Georges Bank, 70-100 m, submersible: settled 0-group cod and haddock "resided by
day within a few centimeters of the bottom"; "At night, most juveniles were on the bottom, but some fish were
several meters up" (3-5 m); patchy, "at times strongly clustered"; coloration mimics the gravel.

E10. **Bromley & Kell 1999; Bromley et al. 1997; Aglen et al. 1997** [S via research_notes/"Sejens diet pa unga
torskfiskar"/evidence_settlement.md entries 6-7 and "Bottenfiskens hojdprofil"]: pelagic 0-group gadoids
migrate to the surface at night; small fish "demersal by day and pelagic at night"; cod > 8 cm wholly
demersal; Barents small haddock (< 20 cm) lifted 30-40 m at night.

E11. **NS-IBTS GOV, in-gape haddock (< 19 cm) vs adult saithe by bed depth** [F data, D]
(`scripts/gov_juv_overlap.py`; research_notes/"Sejens djuplage"/data/gov, DATRAS CC BY 4.0). Box Q3: in-gape
haddock class shares 0.823 / 0.138 / 0.039 / 0 against saithe 0.005 / 0.166 / 0.801 / 0.028; Q1 0.545 / 0.199 /
0.176 / 0.08 against 0.005 / 0.055 / 0.646 / 0.294. In-gape haddock density met by the mean adult saithe
relative to the box mean (F_gate): box Q3 0.19, Q1 0.55, presence-weighted 0.225; Skagerrak/Trench window
(57.3-59.2 N, 8-12 E) 0.49 / 0.50, 0.49; haul level 0.086 (box) and 0.296 (wide). Agrees with the sibling's
haul-level Williamson overlap (NE Trench/Skagerrak Q1 0.06, Q3 0.26; research_notes/"Sejens diet pa unga
torskfiskar"/data/settlement/datras/saithe_haddock_overlap_summary.txt) and the IBTS Q3 settlement density by
depth band (50-100 m 761 n/h, 150-200 m 52, 200-300 m 20; evidence_settlement.md entry 14).

## Saithe diet and the booking

E12. **Booking origin** [F library + notes]: `predatory_fish_preys_on_benthivorous_fish` predation_mortality
0.0833 = Q x diet share / prey B, the water column's consumption-weighted member diet share 0.061 on benthivores
(research_notes/"Rovfiskens uppdelning"/params_split_223.md 3.1-3.2), kept at section 228 (mackerel not a member).
Field (research_notes/"Sejens diet pa unga torskfiskar" summary 1): saithe 5 % (2-9) of food, almost all 0-1
group haddock; group ~4 % (2-9). The booked 6.1 % is at the top of the central field value.

E13. **SMS 2023 key run, saithe partial M2 on haddock 2001-22** [F model output of ICES, D]
(`scripts/prey_side.py`): age 0 0.385, age 1 0.519, age 2 0.132 /yr at a North Sea saithe Q1 density 0.66 t/km2
(575 000 km2 [J]); scaled to the box's saithe (0.617 x 0.80 = 0.49 t/km2): 0.29 / 0.39 / 0.10 /yr. SMS modelled
9.1 % haddock in saithe food vs observed stomachs 4.1 % in 1980-91 (diet note) -> x0.45 for the stomach level.

E14. **Where and when saithe take juvenile haddock** [S via the diet note]: haddock share ~0 in Q1 (1991) and
peaks in Q3 when the 0-group settles; Q3 haddock share 1.1 % in a poor year class, 6.9-8.5 % in good ones; saithe
diel feeding mostly by day (Bjordal & Johnstone 1993; Sarno et al. 1994, via E5) [S]. No stomach study separates
bed vs off-bed captures or day vs night captures of juvenile haddock by saithe (searched in the sibling notes;
none found here) - gap.

## Model facts used

M1. Section 269 central (research_notes/"Overlapp i fiskparen", "Bottenfiskens hojdprofil"): saithe/pollack exp
15 m by day, night 0.07 in 0-30 m + exp(10.5 m); juveniles uniform bottom 2 m with slab clipping; WC ->
benthivores a 0.0418, window [0.392, 0.0388] (saithe 0.89 of the band a). Reproduced exactly by
`scripts/band_hs.py` (ratios 2.904 / 7.604 / 2.764 / 0.155; krill 0.01169).
M2. Combined capture-p x schooling factors (research_notes/"Fangst och stim i fiskparen" combined.json, central):
WC -> benthivores x2.96, -> whiting x2.92, -> planktivores x1.61, -> mackerel x0.24; Hs-independent.
M3. budget_gate places benthivores uniformly in the field-spread form (lib/diagnostics/field_diet.py: only the
water column and whiting have FIELD_BED_SHARES); juveniles are a fixed share of B in every cell (engine).
