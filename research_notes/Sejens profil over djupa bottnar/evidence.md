# Saithe's profile over deep beds and the water column's bed-class gradient: evidence (2026-11-18)

Research only. Nothing in the library, code, tests, manifest, AGENTS.md, resume or existing notes was changed; no
training; no value set from or fitted to model output. Scratch files only in the session scratchpad.
Grades: [F] field data or full text read, [L] lab, [A] abstract only, [S] secondary / read via another note or a
summarising fetch, [D] derived here, [J] judgement. Scripts and stdout: `scripts/`, `tables/`.

## Model facts (read from the repo)

M1. Library pair tables (`fgconfig/fg_library.yaml`, `predatory_fish_preys_on_*`, depth_access_table, classes
< 100 / 100-150 / 150-250 / >= 250 m), normalised to the equal-cell mean [D]: krill 2.62 / 1.21 / 0.22 / 0.006;
planktivores 2.21 / 1.21 / 0.41 / 0.16; benthivores 1.66 / 1.36 / 0.56 / 0.31; whiting 1.67 / 1.38 / 0.55 / 0.27;
mackerel 1.68 / 0.93 / 0.75 / 0.75. They are the section 269 A_band class a's; section 270 rescaled only
`max_intake_rate` (capture p x schooling, constants on a). `scripts/class_terms.py` reproduces them exactly from
research_notes/"Overlapp i fiskparen"/scripts/band_pairs.py (exec'd unchanged): ratios 2.9040 / 7.6039 / 2.7636, krill
0.03055 / 0.01407 / 0.002616 / 6.9e-05 (file 0.03059 / 0.01409 / 0.00262 / 6.91e-05).

M2. Saithe/pollack search profile (krill_pairs_wc.saithe_field): day exp(-h/15 m) above the bed, never above 10 m
depth; night 0.07 in 0-30 m + exp(10.5 m). Prey profiles: juvenile benthivores uniform in the bottom 2 m; whiting
bottom 10 m; pout 0.7 in the bottom 10 m by day; blue whiting below 150 m by day (bed - 50 m to bed over deep beds);
herring by day 10-100 m depth except a HERAS bottom share on beds < 95 m; krill day layer centred 95 m over beds
<= 150 m, 140 m at 160-170 m, 120 m at 250 m (section 262). Planktivore pool shares fixed in every cell: herring
0.554, pout 0.350, blue whiting 0.075, sprat 0.021.

M3. Rule A (section 269): R^2 = R_nv^2 (1 - g) + R_sat^2 g, g = E/(E + Ke), Ke 0.02 lx [J], R_nv 0.3 m on fish /
0.1 m on krill [J], R_sat = min(R0, 2 L). Light: section 253, measured K to ~65 m, K_deep 0.06 /m below [J; 0.05-0.09].

M4. Policy-free saturation (`tables/class_diet_stdout.txt`, budget_gate's env, start densities, field prey shapes):
S = sum a h B_vis per class 50.1 / 29.3 / 7.6 / 2.0, share of the Holling ceiling 0.98 / 0.97 / 0.88 / 0.67. The
water-column ceiling is 1/h = 1460/167.8 = 8.7 t krill/t/yr (6.7 on planktivores); field Q/B 2.83 is 0.33-0.42 of
it [D].

## Task 1. Saithe's vertical position over deep beds

E1. **HERAS / IESSNS pelagic hauls, 54.6-59.1 N, 5-12.4 E, June-July** [F data, D] (`scripts/heras_deep_windows.py`,
head of research_notes/"Sejens hojd over botten"/scripts/heras_saithe_height.py exec'd unchanged; ICES acoustic-trawl
database, CC BY 4.0). Beds >= 150 m:

| day, window | hauls | with adult saithe | adult kg/h | bed median | window | prey in the same hauls (share with; kg/h) |
|---|---|---|---|---|---|---|
| footrope 0-3 m above bed | 4 | 2 | 21.5 | 204 m | 200-205 m | blue whiting 0.75 (129), pout 0.50 (69) |
| 3-30 m | 4 | 4 | 595 | 171 m | 152-162 m | blue whiting 1.00 (613), pout 1.00 (96), euphausiids 0.50 (115), herring 0.50 (17) |
| 30-60 m | 3 | 2 | 9.1 | 180 m | 140-148 m | herring 0.33 (272), blue whiting 0.33 (209) |
| > 60 m up, 60-160 m depth | **1** | 0 | 0 | 255 m | 140-155 m | none |
| > 60 m up, above 60 m | 63 | 0 | 0 | 290 m | 0-29 m | herring 0.90 (1343) |

Night: 82 surface hauls (0-20 m) over beds >= 150 m, **11 with adults** (3/26 over 150-250 m, 8/56 over > 250 m),
mean 1.7 kg/h, fish 70-106 cm (median lengths), with herring in every one. No night haul fished near the bed.
Reading [D/J]: by day the adults over 150-250 m beds are within ~30 m of the bed, where blue whiting, pout and
(in 2/4 hauls) euphausiids were caught with them; the krill/blue-whiting day layer 60-160 m depth over deep beds
was fished once (a gap, not an absence); 0/63 surface hauls by day. Large adults reach the surface at night
(13 % of night surface hauls).

E2. **Stensholt, Aglen, Mehl & Stensholt 2001**, ICES CM 2001/Q:24, doi:10.17895/ices.pub.25636107.v1 [F, read by
the literature subagent; the quotes on pp. 15-16 and 22-23 re-checked here in the PDF text; figure values read by
eye]. IMR acoustic surveys Oct-Nov 1992-2000, Norwegian coast 62-72 N (coastal saithe survey, 2-5-year-olds, part
may be < 35 cm) and North Sea south of 62 N; beds 30-600 m; 10 m bins plus a 0-10 m bottom channel; dead zone
excluded; species allocated per scrutiny sub-block (so profiles partly the scrutinizer's allocation). Vertical
axis RPRL = 1 - (d + 10)/(b + 10), i.e. height above bed = RPRL x (bed + 10) [D].
- p. 15: "Along the Norwegian coast, saithe mainly stay in the bottom half of the water column for all bottom
  depths, but with deeper bottom they spread relatively higher into the water column ... For depth less than 200m
  the saithe stay close to seabed, and for depth greater than 300m the median of cumulative relative sA-values in
  some years is well below 100% at 50%RPRL."
- p. 16: "Saithe in the North Sea mainly stay within the bottom channel (10 m above seabed) (Figure 13a,b) and
  Table 6", mainly N and NW of the survey area, west of the Norwegian Trench. Table 6: 69.5-96.1 % of saithe sA in
  the 0-10 m channel by year (beds pooled; no depth or day/night split).
- Fig. 12a (beds 30-200 m): ~60-75 % within 0.1 RPRL, ~93-98 % within 0.2 [F, eye] -> over a 175 m bed ~60-75 %
  within 18.5 m, ~95 % within 37 m [D]. An exponential from the bed with scale 0.07-0.11 RPRL fits the first number
  (Hs 13-20 m at 175 m) [D].
- Fig. 12b (beds 300-600 m): ~40-75 % within 0.1 RPRL, ~55-90 % within 0.2, ~90-100 % within 0.5 [F, eye]; scale
  0.07-0.20 RPRL, Hs 22-61 m at a 300 m bed [D]. The 200-300 m class is not shown.
- p. 18: saithe DVM "not clear from the combined data"; day/night samples from different places.
- Abstract: pelagic living "especially in areas with high acoustic sA-values and where the bottom is deeper than
  200m".

E3. **Armannsson & Jonsson 2012**, ICES J Mar Sci 69:1372-1381, doi:10.1093/icesjms/fss121 [F; PDF in
research_notes/"Sejens diet pa unga torskfiskar" (a-ring)/data/settlement/]. 51 DSTs, 42-83 cm, Iceland, depth
only (no bed reference). Free vertical range FVR from Dmax = 1.5 (D + 10) - 10, Dmin = 0.75 (D + 10) - 10 (cod
tolerance); within the FVR on 63 % of days; daily vertical range rises with median depth (Fig. 5a, R^2 0.26; ~1.1 m
of range per m of depth, eye); "visual feeders that forage in the daylight hours"; night activity "much reduced
in winter". Height above the bed not measurable. Supports a range that scales with depth [J], not a time budget.

E4. **Skilbrei & Ottera 2016**, ICES J Mar Sci 73:1186-1195, doi:10.1093/icesjms/fsv261 [S: two summarising
fetches of the full text]. Farm-associated saithe 38-59 cm, Ryfylke fjords (farm depth ~100 m, 150-600 m within
1 km): median depth 25-50 m summer, 60-90 m winter; ~100 m of column; ~38 % without DVM. Not bed-referenced, food
subsidised. Introduction: "trawl catches have demonstrated that saithe may stay close to the bottom at 150-250 m
depth" (citing Rosen & Holst 2013, doi:10.1016/j.fishres.2013.08.002, not obtained).

E5. **Kaartvedt et al. 2005**, MEPS 299:1-5, doi:10.3354/meps299001 [F]. Norwegian Sea 70 N, May (no darkness):
stacks of fish up to ~100 m below krill swarms at 100-200 m; trawls at 200-300 m took four 60-110 cm saithe holding
fish and Gonatus. Saithe in midwater over very deep water, under krill, eating fish.

E6. **Nedreaas 1997**, ICES CM 1997/Y:20 [F, via the subagent]: saithe "dive in response to ship noise, so even
those well above bottom may be sampled with a bottom trawl" (an acoustic/trawl bias towards the bed); "not
generally found" at 200-300 m in the Barents Sea; bottom channel 0-10 m.

E7. **Neilson et al. 2003; DFO 2003/075** [F/S, in "Sejens hojd over botten" E3-E4]: pre-spawning pollock mostly
1-5 m off the bottom; Hs 2.5-15 m. **ICES FishMap saithe** [F, in "Sejens djuplage" item 6]: adults "hundreds of
metres above the bottom during night time". **Homrum et al. 2013** [F]: conventional tags, no depth data.

E8. Not obtained: Stensholt et al. 2002 ICES J Mar Sci 59:679 (subscriber; adds no North Sea data beyond E2 as far
as the abstract shows) [A]; Thunen WH III cruise 414 (third failed fetch); Bergstad 1990/1991 (no open copy);
ICES CM 2006/Q:21; Rosen & Holst 2013. No saithe stomach or diel evidence on feeding in darkness at 150-300 m.

## Task 2. Stomachs by bed class

E9. **ICES stomach database, adult saithe (>= 35 cm), N of 57 N** [F data, D] (`scripts/saithe_stomach_bed.py`;
records from research_notes/"Krill over grunda bottnar" (o-umlaut)/stomachs/tables/records_linked.csv.gz, prey rows
from the full download in research_notes/"Fodoniva och mattnad i falt" (umlauts)/data/A_ices_stomach/all). 1980-87
records are rectangle-only (spread over classes by sea-area share) and carry no prey weights in this download: FO
only. 2020s records have haul positions (Q1 only, 34 stations). Station-bootstrap 5-95 % in brackets.

FO (share of fed records with the prey), NE North Sea + Skagerrak, spread:

| set | stations | krill | pout | juvenile gadoids (haddock, cod, whiting id.) | Gadidae unid. |
|---|---|---|---|---|---|
| Q3, < 100 m | 57 | 0.65 | 0.26 | 0.16 | 0.19 |
| Q3, 100-150 m | 64 | 0.72 (0.59-0.83) | 0.37 (0.28-0.50) | 0.20 (0.10-0.32) | 0.18 |
| Q3, 150-250 m | 34 | 0.79 (0.68-0.89) | 0.29 (0.18-0.40) | 0.08 (0.03-0.15) | 0.07 |
| Q3, > 250 m | 28 | 0.79 | 0.32 | 0.08 | 0.06 |
| all q, 100-150 m | 109 | 0.65 (0.55-0.74) | 0.42 (0.32-0.52) | 0.19 (0.13-0.26) | 0.15 |
| all q, 150-250 m | 69 | 0.76 (0.68-0.84) | 0.41 (0.34-0.49) | 0.18 (0.10-0.27) | 0.12 |
| all q, > 250 m | 62 | 0.73 | 0.44 | 0.14 | 0.09 |

All N of 57 N (0.6 NW North Sea), all quarters, < 100 / 100-150 / 150-250 / > 250 m (202 / 298 / 203 / 84 stations):
krill 0.55 / 0.39 / 0.51 / 0.72; pout 0.33 / 0.52 / 0.54 / 0.45; juvenile gadoids 0.14 / 0.21 / 0.18 / 0.14. Herring
and sprat 0.02-0.05, blue whiting 0.01-0.04 in every class. By length (NE, all quarters) the krill FO stays
0.77-0.87 at 35-50 cm and 0.61-0.77 at 50-70 cm across classes; no class falls.

True positions, Q1 2020s (N57; 24 / 5 stations at 100-150 / 150-250 m): fullness 8.1 (3.8-13.7) against 2.2
(0.6-7.3) per mille of body weight; mass shares at 100-150 m pout 0.82, krill 0.05, identified gadoids 0; at
150-250 m pout 0.26, unidentified fish 0.73, krill 0. Q1 is the spawning season; 5 stations [J].

E10. **GOV prey density by bed class, box 2000-25** [F data, D in research_notes/"Sejens djuplage"/tables/
gov_prey_overlap_stdout.txt]: Q3 kg/h < 100 / 100-150 / 150-250 / > 250 m: pout 73.5 / 111.3 / 60.4 / 16.3; blue
whiting 0.1 / 117 / 153 / 20; herring 706 / 62 / 50 / 0.6; haddock < 25 cm 55.9 / 16.2 / 2.6 / 0; whiting < 25 cm
265 / 22.6 / 1.0 / 0.2. Krill: no field density by bed class; the library depth shape is used (class means 2.08 /
5.02 / 6.11 / 7.12 t/km2) [model input].

E11. **Clearance test** [D] (`scripts/field_vs_model.py`): occurrence index -ln(1 - FO) over prey density, as a
ratio to 100-150 m, against the model's class a ratio and its predicted intake ratio (a x density, with or without
the policy-free saturation 1/(1 + S)). 150-250 / 100-150 m:

| prey | field occurrence ratio (NE Q3; NE all q; N57 all q) | density ratio | model a ratio | model intake ratio, a-limited / saturated |
|---|---|---|---|---|
| krill | 1.23; 1.36; 1.44 | 1.22 | 0.186 | 0.23 / 0.79 |
| pout | 0.74; 0.98; 1.08 | 0.54 | 0.384 | 0.21 / 0.73 |
| juvenile gadoids | 0.40; 0.93; 0.85 | 0.09 | 0.411 | 0.04 / 0.13 |

> 250 / 100-150 m: krill field 1.22-2.61 vs model a 0.005 (intake 0.01 / 0.07); pout 0.82-1.06 vs 0.159 (0.02 /
0.24). < 100 / 100-150 m: krill field 0.81-1.63 vs model a 2.17 (field clearance 2.0-3.9 at the library density
shape); pout 0.56-0.66 vs a 1.21 (clearance 0.85-0.99).

Caveats [J]: pooled 1980s records compress FO towards 1; capture bed is not feeding bed (saithe swim tens of km a
day, gastric evacuation 1-3 days), pulling every field ratio towards 1; the rectangle spread is ecological inference;
stomachs 1980s and GOV densities 2000-25; krill density by class is a model input; the juvenile-gadoid FO counts
prey of any size while the density counts < 25 cm; the densities are GOV Q3 in every row (in Q1 the box pout ratio
150-250 / 100-150 m is 8.3, not 0.54). The model's own prediction at start densities is the saturated column
(M4). A field feeding level of 0.33-0.42 of Cmax does not by itself show that wild saithe are a-limited (a time
budget gives the same), so the a-limited column is a bracket, not the like-for-like value. The saturation-free
check, the krill : pout occurrence ratio 150-250 / 100-150 m, is 1.3-1.7 in the field and ~1.1 in the model (a
ratios x density ratios), so it does not discriminate either. Net: between 100-150 and 150-250 m the stomachs
cannot test the gradient; beyond 250 m they contradict it (model 0.07-0.24 saturated, 0.01-0.02 a-limited, against
field 0.8-2.6 for krill and pout); juvenile gadoids disagree at 150-250 m too (field 0.40-0.93 against 0.04-0.13),
with the prey-size and dilution caveats.

Krill ids: Euphausiidae 110671, Meganyctiphanes 110690, Euphausiacea 1128 and congeners; AphiaID 1131 (Isopoda) was
in the first krill list by mistake (2 saithe prey rows, 0 HERAS rows) and was removed before the final runs; no
printed value changed.

## Task 3. Decomposition (`scripts/class_terms.py`, `tables/class_terms_stdout.txt`)

E12. Geometry vs vision [D]. "dark" = Ke 1e9 lx (every visual member at R_nv in every class: overlap geometry and
member mix only); vision factor = central / dark per class. G = a(< 150 m, cell-weighted) / a(150-250 m).

| pair | central shape | G | geometry shape (G) | vision shape (G) |
|---|---|---|---|---|
| krill | 2.62 / 1.21 / 0.22 / 0.006 | 7.96 | 2.58 / 1.10 / 0.33 / 0.03 (5.16) | 1.26 / 1.36 / 0.84 / 0.23 (1.57) |
| planktivores | 2.21 / 1.21 / 0.41 / 0.16 | 3.94 | 1.46 / 0.95 / 0.84 / 0.80 (1.38) | 1.68 / 1.41 / 0.54 / 0.23 (2.81) |
| - herring + sprat term | 3.33 / 0.76 / 0.15 / 0.08 | 12.4 | (4.00) | (2.95) |
| - pout term | 1.69 / 1.40 / 0.54 / 0.22 | 2.83 | (1.03) | (2.75) |
| - blue whiting term | 1.78 / 1.47 / 0.48 / 0.10 | 3.35 | (1.49) | (2.25) |
| benthivores (J and A alike) | 1.66 / 1.36 / 0.56 / 0.31 | 2.65 | 1.01 / 0.99 / 1.00 / 0.99 (1.00) | 1.64 / 1.37 / 0.56 / 0.31 (2.65) |
| whiting | 1.67 / 1.38 / 0.55 / 0.27 | 2.70 | (1.00) | (2.70) |

The pout term is 0.45 / 0.68 / 0.77 / 0.79 of the planktivore pair's a by class, herring + sprat 0.47 / 0.19 / 0.11
/ 0.14. The stage window [0.3745, 0.01112] multiplies every class alike, so in the engine J and A share the pair's class
shape. That is an engine convention, not a derivation result: band_pairs.py derives the shape on the juvenile slab,
while most of the adult part rides on bed members with fixed R (anglerfish, ling), whose own class shape would be
flat (section 270.4 put the adult profile and capture p into f_A, a constant).

E13. Light at the bed [D] (`scripts/light_at_bed.py`, K_deep 0.06; 0.05 / 0.09 in the stdout): daytime median
light at the bed < 100 / 100-150 / 150-250 / >= 250 m 2.6 / 0.18 / 0.0056 / 1e-5 lx; day-mean g (Ke 0.02 lx) 0.94 /
0.75 / 0.17 / 0.00 (Ke 2e-4: 1.00 / 0.99 / 0.76 / 0.05); 50 m above the bed g 1.00 / 0.96 / 0.56 / 0.01. At K_deep
0.09 the 150-250 m bed has 0.0003 lx, g 0.02. Everything below ~65 m is the judged K_deep.

E14. Profile and vision variants [D] (class shape; G; G1 = a(100-150)/a(150-250)):

| variant | krill G (G1) | planktivores G (G1) | benthivores G (G1) | whiting G (G1) |
|---|---|---|---|---|
| exp15, library | 7.96 (5.38) | 3.94 (2.95) | 2.65 (2.43) | 2.70 (2.49) |
| expHsz: Hs 15 x (bed + 10)/150 (E2's relative profile) | 7.65 (4.50) | 4.96 (3.39) | 4.19 (3.32) | 3.73 (3.14) |
| freerange: uniform to 0.25 (bed + 10) above the bed (E3) | 5.50 (4.06) | 5.81 (3.91) | 4.11 (3.23) | 4.37 (3.41) |
| mixK0.10 / 0.25: share of day search in the krill layer [J bracket] | 5.62 / 4.15 | 4.19 / 4.64 | 2.70 / 2.78 | 2.83 / 3.05 |
| nightcol: night uniform 10 m-bed (FishMap) | 7.44 (5.09) | 4.05 (3.02) | 3.03 (2.78) | 3.04 (2.78) |
| Hs 5 / 40 m (band_hs.json, sibling) | G1 ~ 13.8 / 2.9 | ~ unchanged | ~ unchanged | ~ unchanged |
| Ke 2e-4 lx | 5.23 (3.62) | 1.75 (1.41) | 1.30 (1.26) | 1.29 (1.25) |
| K_deep 0.05 / 0.09 | 6.67 / 18.2 | 2.96 / 6.91 | 2.06 / 3.76 | 2.07 / 4.17 |

E16. Saithe search time inside each prey's layer (prey support phi > 0) [D] (`scripts/prey_layer_time.py`),
< 100 / 100-150 / 150-250 / >= 250 m, day:
- exp 15 m: krill 0.63 / 0.48 / 0.15 / 0.00; pout 0.88 / 0.87 / 0.87 / 0.87; blue whiting 0.75 / 0.74 / 0.86 / 0.96;
  juvenile slab (2 m) 0.13 in every class; whiting band (10 m) 0.49; herring 1.00 / 0.78 / 0.50 / 0.49, of which over
  deep beds almost all is the herring profile's small bottom-10-m share (~0.05 of the herring), not the herring layer.
- expHsz: krill 0.74 / 0.48 / 0.19 / 0.007; pout 0.96 / 0.90 / 0.78 / 0.59; juvenile slab 0.20 / 0.14 / 0.10 / 0.06.
- Night (exp 15 m): krill 0.87 / 0.49 / 0.15 / 0.05; blue whiting 0.97 / 0.86 / 0.10 / 0.07 (the 17-28 cm fish rise).

E17. Condition by bed class [F data, D] (`scripts/condition_bed.py`): only 15 adult saithe in the stomach database
carry an individual weight (Q1 2024-26, haul positions), none over 150-250 m beds; no test possible. No other
condition or liver-index data by bed depth were found in the notes or the literature searched.

E15. Policy-free intake per tonne by class [D] (`tables/class_diet_stdout.txt`; start densities, field prey
shapes, fixed attack, shared handling): ratio to 150-250 m for < 100 / 100-150 / >= 250 m: library 1.11 / 1.09 /
0.75; expHsz 1.11 / 1.09 / 0.68; Ke 2e-4 1.04 / 1.03 / 0.78; K_deep 0.09 1.24 / 1.22 / 0.84. Diet share of
benthivores by class (library) 0.20 / 0.12 / 0.05 / 0.04, whiting 0.14 / 0.04 / 0.007 / 0.001, krill 0.24 / 0.45 /
0.41 / 0.05; at the GOV saithe spread benthivores 0.10, krill 0.39, planktivores 0.46, whiting 0.04 (booked 0.055 /
0.37 / 0.51 / 0.043).
