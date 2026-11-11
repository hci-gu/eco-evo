# Field metabolism of haddock and the flatfish

Research subagent, 2026-11-10/11 (saved by the main agent). Literature and published data only; no
library, code or resume edit. Evidence ids refer to `evidence.md` (E1-E19). Arithmetic in `scripts/`
(`python3 -I`): `oa.py`, `alewijnse_species.py`, `otolith_fmr.py`, `group_fmr.py`, `budget_fmr.py`; output
in `tables/`. `data/README.md` lists the licences; every download is flagged do-not-commit.

Tags: [M] measured, [S] secondary, [I] inference / arithmetic here. Every otolith-based FMR value is [M] for
the isotope and [I] for the conversion: a cod-only calibration applied to other species, with assumed diet
and DIC baselines. 1 mg O2/kg/h = 119.1 MJ/t/yr = 0.0816 MJ/t/tick.

## Correction to earlier notes

`Tillväxt mot ranson` W.1 and `Bentivorernas energibudget` Q2 say there is no field metabolism for haddock
or the North Sea flatfish (searched 2026-11-04). That is wrong. Own-species otolith-carbon FMR exists for
haddock (52 fish, one Barents site, 27-4 300 g), long rough dab (22 fish, same site), witch (5 fish,
northern North Sea, 8.4 C, 215-1 027 g), dab (4 fish, Channel, 17.8 C) and lemon sole (5 fish, 14.6 C), all
in the Alewijnse 2022 thesis dataset (E2, doi:10.6084/m9.figshare.22110335); 281 adult North Sea plaice,
200-800 g at a mean 8 C (Jones et al. 2023, E3, doi:10.3389/fevo.2023.1161105); a seasonal plaice study
(Jones et al. 2025, E4, abstract only). The FMR weight on own-species data rises from 0.32 (cod only) to
0.90.

## 1. Field metabolic rate

**Otolith route (E1-E3).** Cod, haddock and long rough dab at one Barents site (72.8N) share one
seawater-DIC baseline and one calibration (Chung 2019: C_resp = 0.243 (1 - exp(-0.0088 VO2))). Fit:
ln VO2 = species + b ln mass + c T on fish >= 100 g (`tables/otolith_fmr_stdout.txt`).

Within-site ratios [M/I]:
- haddock/cod at equal mass and temperature: 0.72-0.76 (95 % CI 0.63-0.85);
- long rough dab/cod: 0.84-0.95 (CI 0.68-1.18), on only 8 adults; 0.75 with juveniles;
- the same ranges with any of the three species' diet values;
- temperature fit Q10 2.7; FMR barely falls with mass (b about 0.03), as Jones 2023 found in plaice,
  against an SMR slope of about -0.2.

Absolute values from the cod calibration:
- plaice 54 mg O2/kg/h at 8 C, about 6 430 MJ/t/yr; per-fish SD 36 [M/I] (E3). Its authors call this
  "similar to standard metabolic rates ... lower than expected" and flag the cod calibration;
- witch 36 mg O2/kg/h at 8.4 C, about 4 290 MJ/t/yr [M/I];
- Barents fish extrapolated to 7.5 C [I]: cod (1.2 kg) 108-136 mg O2/kg/h (2.5-3.2 x SMR; Chung 1.9-3.5),
  haddock (275 g) 79-91, long rough dab (170 g) 98-118;
- dab and lemon sole: summer values at 15-18 C only, at most a bound;
- plaice seasonality (E4, abstract): FMR peaks in early summer before spawning, "decoupled from seasonal
  variation in experienced temperature"; the maxima approach maximum metabolic rate.

**Field-ration route, an independent check (E8-E14).** R = 0.8 x consumption - production:
- haddock, Jones 1978 stomach contents and growth, 8 C, 250-600 g (doi:10.1093/icesjms/38.1.18):
  R 9 400-13 300 MJ/t/yr, 2.0-2.5 x Tytler's SMR [M/I]; with single-meal evacuation rates 4 900-6 900
  (a lower bound). The SMS ration gives R 0.5 x SMR, impossible for a growing fish, so the SMS
  consumption or its prey energy is too low;
- plaice: winter spawning-fast R 5 330-7 590 MJ/t/yr at 6.8 C [M] (E11); annual about 7 700-8 100 [I,
  weak: guild ration, borrowed prey energy];
- long rough dab about 5 000 MJ/t/yr from a model-based population ratio and an assumed mass [S/I] (E13);
- dab, witch: nothing. Flatfish stomach-content consumption is unreliable (E14).

The two routes overlap for haddock and plaice; both put haddock above the library's judged 6 205 and
plaice at or above 6 400.

## 2. SMR at 5-10 C (denominators)

- Cod: 35.5 / 57.0 mg O2/kg/h at 5 / 10 C (Schurmann & Steffensen 1997, juveniles) [S]; Arctic cod
  reportedly 2 x this.
- Haddock: Norin 2019 haddock/cod 0.54 at 12 C, 63-68 g [S]; Perez-Casanova 2009 0.64-0.75 at 2-11 C [S];
  Tytler 1969 via Jones 1978, 0.16 W^0.8 exp(0.081(T-10)), gives 44 mg O2/kg/h = 3.6 MJ/t/tick at 275 g and
  8 C [M] (E6, E8), 1.44 x the library's judged 2.50.
- Plaice: Fonds 1985 DOCmin 38.7 (150 g) and 31.2 (400 g) mg O2/kg/h at 8 C [S]; Rijnsdorp & Ibelings
  resting 3-4 kJ/d for a 438 g fish at 6.8 C [M].
- Witch: Moser 1996, 5 g fish scaled, 11-15 mg O2/kg/h at 6 C [S, weak].
- Long rough dab: no lab SMR obtained (MacIsaac 1997 refused, abstract only). The library's 1.50
  MJ/t/tick is MacKinnon's field winter fasting metabolism, not an SMR.
- Group: weighted SMR 2.69 MJ/t/tick (3 925 MJ/t/yr) in `Tillväxt mot ranson`; engine RM 3.33 = 1.2 x 2.78.

## 3. Field consumption and growth

See section 1, ration route. Assimilation 0.8 (Winberg / Jones; metabolisable-type, so R includes SDA);
0.75 lowers R about 8 %. Field production: haddock 0.43-0.61 /yr at 217-362 g (Jones 1978 Table 2) [M];
plaice somatic about 0.34 /yr in energy for an adult female, eggs 594-790 kJ/yr extra (E11) [M].

## 4. SDA, cost of growth and foraging activity

- SDA at realistic meals: cod at satiation with repeated meals 12.9 % of intake at 7 C, rising with ration,
  up to 21 % at 18 C, including excited feeding activity (E15) [M]; other cold demersal species at a single
  satiating meal 15-23 % (E15) [M]; no haddock or flatfish value at meals >= 3 % BM (juvenile haddock
  small-meal values 3-5 %); cod SDA flat between 2.5 and 5 % BM; halibut SDA falls with meal size.
- Synthesis inside SDA: whole-body protein synthesis is 23-44 % of the cod post-meal O2 rise (Lyndon 1992
  [A]; corrects the earlier 23-51 %) (E16); SDA is now read mainly as protein synthesis and turnover
  (Goodrich 2024 [F]; Jobling 1983); upper bound for synthesis plus turnover from Houlihan 1988: 0.37-0.80
  J per J of protein deposited [I].
- A realistic-meal SDA explains at most about a third of the haddock and plaice residual (0.06-0.35 J/J) [I].
- Activity: swimming plaice cost about 0.35-0.70 kJ/kg/h at 0.6-1 BL/s, the same per unit as haddock;
  plaice resting metabolism is lower (Priede & Holliday) [M]. Tank flatfish rest 76-94 % of the day; motor
  activity is 9-20 % of resting energy [M]. Wild plaice rarely leave the bed in summer (E17-E18).
- Gaps: no haddock time budget; no study separating fed-fish activity from synthesis in a growth-ration
  budget.

## 5. Seasonal activity (E10-E12, E18-E19)

- Plaice (3.8 %): mature fish hardly feed while spawning (females about 5 weeks late Jan-early Mar, males
  about 11 weeks; E18), but winter is their most active season: 1/4 to 1/3 of the time swimming in
  midwater in Dec-Jan, almost nothing by May (E18); not torpor. Winter metabolism 5.3-7.1 kJ/kg/d
  (Rijnsdorp 1983). The "0.7 x SMR" in earlier notes was an inference, not in the source.
- Haddock (38.6 %): no winter fast. Barents fullness peaks in March, July and September; empty stomachs
  most frequent in Nov-Jan and April. North Sea spawning Feb-May.
- Cod: feed on the spawning grounds, which corrects the "wild cod fast Dec-Apr" premise.
- Dab: feeding constant on the Dogger Bank; reduced but not suspended inshore in winter.
- Long rough dab: feeds all year; 31 % feeding in Dec-Jan against 90 % in May.
- Witch: nothing. No measured winter metabolic depression for any species.
- Guild: Greenstreet's Q1 rations are 0.64 (benthivores) and 0.79 (piscivores) of the annual mean.
- A feeding/fasting split of annual FMR can be built for plaice only. The otolith FMR is already
  time-averaged (1-2.3 month windows across all quarters), so no seasonal factor goes on top of it.

## Implications

### (a) FMR/SMR multipliers per species, and the group value

Route B, cod-anchored: cod kept at 11 338 MJ/t/yr (2.2 x SMR, Chung 2019); haddock and long rough dab
scaled by the within-site ratios; plaice and witch at face value (low case anchored, high case mass-scaled
to 150-190 g). `tables/group_fmr_stdout.txt`.

| Species (weight %) | FMR MJ/t/yr at ~7.5 C, central (low-high) | Multiplier | Method |
|---|---|---|---|
| Cod (31.4) | 11 338 | 2.2 (1.9-3.5) | Chung 2019 (unchanged) |
| Haddock (38.6) | 7 940 (6 800-9 300) | 2.2 on the judged SMR 2.50; 1.5 on Tytler's measured 3.6 | Otolith ratio to cod |
| Long rough dab (12.6) | 9 640 (7 370-12 470); fallback = plaice 6 430 | none valid (its 1.50 "SMR" is a field fasting rate) | Otolith ratio, n = 8 adults |
| Plaice (3.8) | 6 430 (4 490-7 980) | 1.4-1.7 on Fonds 150-400 g | Otolith absolute, North Sea, 8 C |
| Witch (1.7) | 4 290 (2 990-5 290) | 1.1 (plaice SMR) to 2.6 (Moser) | Otolith absolute, North Sea, 8.4 C |

Proxies: dab, lemon sole and weever take the plaice value; gurnard takes haddock; ray and hagfish keep the
earlier values (2.8 %). The site's haddock adults are 406-4 300 g, so 275 g is an extrapolation, small
given FMR's weak mass slope.

Group (weighted FMR / weighted SMR 3 925 MJ/t/yr): with long rough dab from its otoliths 8 890 (7 920-9 980)
MJ/t/yr = 2.27 (2.02-2.54) x SMR; with long rough dab = plaice 8 480 (7 550-9 400); the library derivation
7 260 (1.85 x). Route A (every species at the calibration's face value) gives 10 200-12 200, an upper view.
Own-species weight 0.90 (0.84 from one site). Main uncertainties: the cod-only calibration; extrapolation
from 2-4 C to 7.5 C (cancels in the cod-anchored ratios); long rough dab's n = 8.

### (b) The haddock/plaice residual: feeding cost inside FMR, or synthesis?

FMR size cannot settle it. With each species' own field production (`tables/budget_fmr_stdout.txt`,
section 5), haddock's c_max (the largest growth cost that leaves activity >= RM): otolith low 0.31-0.44,
central 0.70-1.00, high 1.2-1.7; ration route 1.2-2.6 (continuous evacuation) or < 0 (single-meal). Haddock's
residual 0.84 fits the central otolith FMR, not the low case. Plaice does not discriminate (c_max ~0 at the
otolith face value, 0.86-0.90 at the high and ration values; its otolith value is the one its authors flag
as probably low).

The structure is settled [I from E15-E16]: protein synthesis is largely inside measured SDA; a
realistic-meal SDA explains at most a third of the residual; whatever the rest is, a wild fish growing at
the field rate pays it inside its FMR. A feeding_cost derived from total FMR (240.2) plus a separate c is
therefore a double count by construction.

The consistent fix is Option B of `Bottenfiskens budget`: feeding_cost = (FMR - SDA - synthesis(c)) / RM.
With the new FMR it is feasible at the central value (FC 1.02 at c 0.69, 1.16 at c 0.49), not at the low
value (FC 0.82 at c 0.69). Own-species bounds on c at the central FMR: haddock up to about 0.7-1.0; group up
to 0.72 with SDA 10 %, 0.58 with a realistic 13 %. The library's 0.69 sits at the bound; the FMR low/central
split would decide it and stays open.

### (c) Group FMR and the budget

| Quantity | Value |
|---|---|
| Group FMR at 7-8 C, central (low-high) | 8 890 (7 920-9 980) MJ/t/yr = 6.09 (5.43-6.83) MJ/t/tick |
| Library | 7 260 |
| 223.9 note | 8 200 |

In `Bottenfiskens budget` (`tables/budget_fmr_stdout.txt`):
1. Finding 3 reversed at the central value: activity left at c 0.69 is 1.02 x RM (was 0.68); low 0.82,
   high 1.24.
2. Field-closure Q/B (223.9 form, FMR + P x 4 500, redistributed diet): 4.94 (4.55-5.38) at the box's
   P 0.74, 4.76 at P 0.64 (was 4.46 / 4.66).
3. Engine need with a re-derived feeding_cost at the central FMR: Q/B 5.05 at fill 1, 4.76 at fill 0.62
   (c 0.69); 4.75 at c 0.49 and fill 0.62; all against the new field 4.94. The budget closes at the field
   fill; the remaining excess is fill 1 (the food surplus, finding 5). At the low FMR it closes only with
   c <= ~0.4.
4. Published values (3.48; SMS cod 4.6, haddock 2.8) stay below; the SMS haddock ration implies an
   impossible R (section 3), so it is not a binding comparison.

## Gaps

- No species-specific C_resp-to-O2 calibration for any gadoid but cod, or any flatfish.
- Jones 2025 monthly plaice FMR: full text refused (HTTP 403).
- Long rough dab: 8 adult otoliths, no lab SMR (MacIsaac 1997 not obtained).
- Dab: no FMR at a usable temperature. Witch: n = 5, no reliable SMR.
- No haddock or flatfish SDA at realistic meals; no haddock activity budget.
- No measured winter metabolic depression for any species.
- Paywalled or blocked: Metcalfe 1990, Fonds 1992, Peck 2003/2005, Lyndon 1992, MacKinnon 1973.

## Correction 2026-11-11 (main agent)

`scripts/budget_fmr.py` copied LOSS 0.40 from `Bottenfiskens budget`, which missed M1 (0.29 /yr; a
per-tick value read as per year in closure.py). Fixed to 0.694 and re-run: in Implications (c) item 3 the
engine need with the re-derived feeding_cost at the central FMR is Q/B 4.91 at the field fill (was 4.76) and
5.29 at fill 1 (was 5.05), c 0.69; the field closure 4.94 is unchanged. Items 1, 2 and the FMR values are
unaffected.
