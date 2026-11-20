# Evidence trail: why field demersal piscivores are not full

Research only (2026-11-20). Nothing in the library, manifest, code, tests, AGENTS.md or the resume was
changed. No value here is chosen from model output; fix104 feeding levels (resume 278.4) are used only
as the yardstick of the gap.

Tags: [F] full text, [A] abstract, [S] secondary, [D] derived here, [J] judgement.

Literature notes from three subagents: `lit_fill_series.md` (field reserve series), `lit_foraging_limits.md`
(mechanisms 1-4), `lit_costs_sinks.md` (mechanisms 5-6). Raw CSVs: `tables/fill_series_raw.csv`,
`tables/foraging_limits_raw.csv`, `tables/costs_sinks_raw.csv`. Model-side arithmetic:
`scripts/model_side.py` -> `tables/model_side_stdout.txt`.

## M0. Common currency [D]

- Net energy per ingested MJ = assimilation 0.80 - SDA 0.12 = 0.68 on every whiting and water-column pair.
- Whiting: ceiling 31.34 MJ/t/tick = 125.4 kJ/kg/d (bottom layer, 7.4 C); cost RM 3.87 x 1.5 = 5.80
  MJ/t/tick = 0.272 of the ceiling in ingested terms. Need at fill 1 (cost + P 0.61 /yr at lean 3900 +
  ME 2000) = 0.388 of the ceiling; fix104 realised 0.383, uncapped potential 0.664.
- Water column: ceiling 29.79 = 119.2 kJ/kg/d (at_50m, 8.4 C); cost 3.29 x 2.14 = 7.04 = 0.348 of the
  ceiling; need at fill 1 = 0.435; fix104 realised 0.438, uncapped 0.731.
- So the model's realised ration is exactly cost + field somatic production at fill 1, and its
  potential is 1.7x that.
- Field whiting rations (SMS 2024 by quarter, E 4.5 kJ/g, 120-250 g) are 41-52 kJ/kg/d annual mean =
  0.33-0.41 of the model ceiling: the model's realised whiting ration IS the field ration in absolute
  terms (also "Vitlingens ranson", "Ransonen i energi"). The field "0.25 of Cmax" is against the
  Andersen & Riis-Vestergaard lab Wmax (136-165 kJ/kg/d at 7 C for 250-120 g), ~1.1-1.3x the model
  ceiling. The question is therefore not the ration but the POTENTIAL: what keeps the field fish from
  eating more when its liver is not full.

## M1. The fill is a knife edge in the potential feeding level [D]

`model_side.py` section 2: with the derived growth law (G proportional to (s - u), s* = 1) the
equilibrium fill s needs a feeding level of
- whiting 0.275 (s 0.1) / 0.317 (0.5) / 0.347 (0.73) / 0.388 (1.0);
- water column 0.350 / 0.382 / 0.404 / 0.435.
The whole fill range sits inside ~0.11 of the ceiling. Any mechanism that leaves the potential above
~0.39 (whiting) / ~0.44 (water column) in a month leaves the fish full in that month; one that holds it
near 0.30-0.35 / 0.37-0.41 gives field-like fills. The field fish is not full simply because its
potential is close to its need; the model's is 0.27-0.30 of the ceiling (~9 MJ/t/tick ingested) above
it.

## M2. Refill speed bounds the field surplus [D]

`model_side.py` section 3: the model's surplus at fill 1 is ~6 MJ/t/tick net, which would refill
whiting's proxy swing (0.30 -> 0.73, 860 MJ/t) in ~36 days. Field liver/condition refill takes ~3-4
months (whiting, Cooper 1983 [F via Lek i rutan]) to ~5-6 months (saithe, NIFES [F via Svaltgrans]).
That puts the field potential during the refill season at ~0.44-0.50 (whiting) and ~0.47-0.51 (water
column) of the model ceiling, and that is while somatic growth also peaks, so it is an upper bound.
The field surplus is ~1/4-1/2 of the model's.

## M3. The field seasonal ration reproduces whiting's cycle [D]

`model_side.py` section 4: SMS quarterly rations (stomach-based) minus library costs at the layer's
temperature minus P (flat), with the annual imbalance removed:
- whiting 250 g, E 4.5: annual imbalance +3 % of cost (the SMS ration closes the library budget);
  reserve trajectory min ~1 Jul, max Oct-Jan, swing 0.36 ME (0.31-0.60 over E 4.0-5.5 and 120-250 g).
  Field proxy swing 0.43 (min May-Jul, max Sep-Nov); Cooper 1983 whiting condition min Apr-Jun, max
  Aug-Sep. Phase and amplitude agree.
- saithe 1.5 kg (North Sea stock, mostly spawning adults): annual imbalance -5 to -34 % of the model
  cost (the SMS ration does not pay the model's water-column cost; "Ransonen i energi" found the same);
  min ~1 Apr, max ~1 Jul, swing 0.75-0.98 ME. NIFES saithe liver: lowest Q1-Q2, highest Sep. Phase
  partly agrees (Q1 trough), the swing is larger than any measured saithe liver cycle.
- Caveat: SMS consumption contains a temperature-dependent evacuation model, so part of its Q3 rise
  is North Sea temperature, not food; growth spread evenly; saithe SMS is the adult stock.
Reading [J]: if the field ration is supply- (or time-) limited, its measured seasonal shape alone
produces a whiting reserve cycle of the observed size and phase. In the model the ration is set by
the reserve room instead, so the same food produces no cycle.

## M4. Mechanism sizes on the model scale [D] (`scripts/mechanism_sizes.py`)

- 1 Wmax (Andersen & Riis-Vestergaard, whiting) = 1.10-1.34 model ceilings (250-120 g; 1.20 at 180 g).
- Per-prey plateaus (Rindorf & Gislason 2005 [F]) on the model scale: clupeids 0.12-0.20 and gadoids
  0.07-0.14 are FITTED, density-independent plateaus. "Other food" (krill, epifauna, copepods) 0.10-0.26
  is an OBSERVED intake range, not a fitted plateau, so a krill-alone bound is [J].
  - All three saturated and additive: 0.29-0.61.
  - Observed totals: 0.17-0.42. These do not discriminate mechanisms: any field total equals the need
    (M1). What supports mechanism 2 is the fitted plateaus' independence of prey density and of the
    other prey.
  - Whiting's need band is 0.275-0.388.
- Dempster et al. 2011 [F]: farm-associated saithe stomach content x2.35, HSI x1.4-1.8; with linear to
  square-root evacuation, the wild intake is 0.43-0.65 of the farm-fed intake.

## M5. Literature (subagent notes; tags as given there)

- `lit_foraging_limits.md`: the four candidate mechanisms (time and risk, per-prey plateaus,
  digestion and lipostasis, patchiness), each with sources and numbers. Key sources: Dempster 2011 [F];
  Rindorf, Gislason & Lewy 2006 [A]; Kempf 2008 [F]; Guichet 1995 [A]; Mackinson & Daskalov 2007 [F];
  Rose & Leggett 1990 [A]; Skilbrei & Ottera 2016 [A]; Armannsson & Jonsson 2012 [A].
- `lit_costs_sinks.md`: field metabolic rates (otolith route, ration route, swimming bounds) and the
  seasonal sinks (saithe migration, the winter dip, warming, hake batch spawning).
- `lit_fill_series.md`: reserve series. Mathers 1992 [F] (saithe, northern North Sea, condition);
  Uglem 2020 [F] and Dempster 2011 [F] (saithe HSI); Erdogan Saglam & Saglam 2012 [F] (whiting HSI by
  month, Black Sea); Jonsson 2011 [F] (whiting, Iceland); Lauerburg 2016 [F] (North Sea whiting HSI by
  length); Costa 2013 [F] (hake HSI by month, Portugal).

## M6. Double counting checked [J]

- **Per-prey plateau vs the 270 capture p and schooling F.** Both act inside a, and a is inert while
  pairs are saturated: halving intake needs every prey x0.03-0.05 ("Fodobegransning" model table). A
  plateau has to act on the ceiling side, as a per-pair maximum, and must not be stacked on a cut in a.
- **dark_ratio.** Inside a too, so it is inert for the same reason. A dawn and dusk capture window is
  the field's explanation for the plateau's size; it is not a second factor.
- **The 279 spawning cut (whiting 0.028).** It overlaps the Q1-Q2 low in the seasonal ration. Any
  ration-season term would contain the spawners' fast.
- **Q10 on intake and cost.** These are matched, so summer warming is already charged. The SMS
  quarterly shape contains North Sea temperature through its evacuation model, so only the
  non-temperature part is new.
- **Water-column exchange.** Arrivals come in at start_fill 0.71 and are topped up within days. A
  lower potential would keep them near the outside fill instead.
- **Growth capped at field P.** In the field, surplus food goes to faster somatic growth and to the
  liver (Dempster: farm saithe are fatter). In the model it can only go to the reserve. So the
  potential cannot simply be raised and growth made ration-dependent without the runaway growth seen
  in 244.11.

## M7. Fill series on the library's own conversion [D/J] (`scripts/mechanism_sizes.py` section 4)

The conversion is from "Svaltgrans och startreserv" model/derive_u.py: fill = HSI x liver lipid
share x 39.5 MJ/kg / 2000 MJ/t. The library's saithe 0.71 is 6 % HSI x 60 % lipid on this rule.

**Saithe:**
- NIFES Norwegian Sea: Q1 0.63, Apr-Jun 0.67, Sep 1.02.
- Box size class (43 cm NIFES Oct-Dec, 41-43 cm Mathers August; lipid 0.6 [J]): 0.47-0.71.
- Coastal wild controls (Dempster, Uglem): 0.62-0.81.
- Farm-fed: 1.09, above the library maximum.

**Whiting** (Black Sea; lipid share 0.5-0.6 [J]):
- May-Aug 0.27-0.32;
- Sep-Oct 0.40-0.49;
- Nov-Apr 0.59-0.71.

Spot checks against the PDFs:
- Dempster 2011: 8.6 g against 20.2 g stomach content (lines 343-345 of pdftotext) - confirmed.
- Mathers 1992: condition, CSI and GLI defined, summer the highest feeding/growth - confirmed.
Email check: no file in this directory, data/ included, contains the Unpaywall address or "mailto".
