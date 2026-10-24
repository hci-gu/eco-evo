# Evidence C - Gadoid diets (benthos share by mass) and fish consumption vs benthic production

Scope: Q4 (benthos share by mass of gadoid diets, Skagerrak / Kattegat / northern North Sea) and
Q5 (demersal-fish consumption of benthos as a fraction of benthic production). Tags: [F] read in
full text / full data file, [A] abstract only, [S] secondary citation (via the named source).
1 t/km2 = 1 g/m2. All "%" are by mass (weight) unless stated.

## Short answers

**Q4.** By mass, benthic invertebrates make up roughly 30-50 % of the diet of cod (falling with
size), 40-65 % for haddock, 10-30 % for whiting, ~1-3 % for saithe and ~6-24 % for Norway pout.
The best size-structured data set is the 1991 Year of the Stomach (ICES CRR 219, Table 8, total
North Sea, % weight by age class) [F]. Benthos share (crustacea minus euphausiids, hyperiids and
copepods, plus annelids, molluscs other than cephalopods, echinoderms and so on), annual,
weighted by stomach content:
cod age 1 47 %, age 2 29 %, age 3 21 %, age 4 17 %, age 5-6+ 12 %;
haddock age 1-6+ 38-45 % (0-group 63 %); whiting 8-18 %; saithe (ages 3-9+) 0-1.4 %.
The North Sea 1991 Ecopath model (Mackinson & Daskalov 2007) gives juvenile/adult cod 0.40/0.26,
juvenile/adult haddock 0.67/0.54, whiting 0.29/0.12, saithe 0.03/0.01, Norway pout 0.24 [F, EcoBase
deposit of the model file]. **The printed diet table (Table 3.4) is broken for predators 13-24**
(see the warning below), which is why an earlier parse got 6 % for adult haddock and 0 % for adult cod.
Community share for today's Skagerrak IBTS mix (no Norway pout): **0.28-0.42, consumption-weighted**
(CRR219 0.28, Greenstreet 1997 0.31, M&D 0.35-0.42). Adding Norway pout at 0.5-1.0x the biomass of
the other five species lowers it to **0.17-0.36**. The model's ~40 % is at the top of the no-pout
range and above every with-pout variant. Skagerrak-specific diets by mass were not retrieved; the
abstracts only say what dominates (haddock in the Norwegian Deep: polychaetes, crustaceans,
ophiuroids; Norway pout: euphausiids) [A].

**Q5.** In the North Sea 1991 Ecopath model, fish ate 30 g WW/m2/yr of benthos. That is
**5.6 % of total benthic production** (532 g WW/m2/yr) and 7.7 % of production excluding sessile
epifauna and meiofauna. **Gadoids alone ate 3.7 g WW/m2/yr = 0.7 %.** Flatfish (16.0) and sandeel
(8.1) eat most of it [F, computed]. Fish take a large share only of the large mobile prey: large
crabs 82 %, Nephrops 71 %, shrimp 30 %. For the rest, invertebrate predation dominates the
ecotrophic efficiency (EE). Heath (2005): fish ate 72 / 53 / 33 % (decadal means 1970s / 80s / 90s) of
the production of *macrobenthos carnivores* (crabs, shrimps, Nephrops...), "between 50% and 60%
... during the 1970s, but had declined to 15-20% by the late 1990s" [F]. Cramer & Daan (1986):
cod + haddock alone ate 0.5-0.8 g C/m2/yr of benthos, equal to "10-25% of the total benthos
production" when set against Steele's 2-5 g C/m2/yr [F]. No Skagerrak or Kattegat budget was found.

## WARNING - Mackinson & Daskalov 2007, printed Table 3.4

I parsed the PDF from word coordinates and checked it against rendered page images (pp. 31-32 of
the report, PDF pp. 33-34). The parse is faithful to the print: adult haddock (col 18) shows
0.013 + 0.010 + 0.009 + 0.008 + 0.019 + 0.002 = 0.061 benthos, and adult cod (col 14) has no benthos
cells. The print itself is defective for predators 13-24, though. Its column sums (all printed
cells, rows 1-68) are 13: 0.56, 14: 0.78, 15: 0.62, 16: 1.29, 17: 0.93, 18: 0.64, 19: 0.40,
20: 1.53, 21: 1.54, 22: 0.39, 23: 0.81, 24: 1.23. The other blocks sum to ~1.00 (1-12 and 25-68,
except seabirds 0.80 + import 0.34, 49 0.97, 53 1.07, 54 1.47, 55 1.12, 58 1.03). The misplaced cells are biologically absurd: hake
24 % small mobile epifauna, Norway pout 36 % large crabs, "Import 0.324" in adult whiting, adult
saithe 30 % epifauna + 18 % infauna. The "Sum = 1" row is nominal. **Do not use the printed
Table 3.4 for gadoids.** Use instead the model file deposited in EcoBase (model 457, "North Sea
1991", Mackinson & Daskalov). Its diet columns sum to 1, and my predation/production recomputation matches its EE for the unfished
benthic groups (e.g. infauna 0.276 vs 0.277, small infauna 0.908 vs 0.908); for crabs, Nephrops and
shrimp the gap (0.83 vs 0.84, 0.89 vs 0.94, 0.37 vs 0.43) is the fishery catch, which EE also contains.
Version caveat: the deposit is the model as modified up to 2009 (its description field lists edits
from 2006-2009), not exactly the 2007 print. For example, juvenile whiting B is 0.022 vs 0.222 in
printed Table 3.3, juvenile haddock B 0.0067 vs 0.284, juvenile cod B 0.124 vs 0.079 and Q/B 8.0 vs 5.96. The diet entries that the print keeps intact are close but not
identical (juvenile cod: Norway pout 0.121 printed vs 0.127 EcoBase; sandeel 0.136 vs 0.152).

## Table - benthos share of diet by mass

"Benthos" = all benthic invertebrates (crabs, Nephrops, shrimp, epi- and infaunal macrobenthos,
small mobile epifauna, small infauna, sessile epifauna, meiofauna). "excl. S" = without
shrimp/Caridea and small mobile epifauna (amphipods, mysids, cumaceans, isopods). "Euph" =
euphausiids, counted as zooplankton everywhere.

| Species / stage | Area, period | Benthos | excl. S | Euph | Source | Tag |
|---|---|---|---|---|---|---|
| Cod juv (0-2 y, <40 cm) | North Sea 1991 | 0.401 | 0.323 | (zoo 0.079) | M&D EcoBase 457 | [F] |
| Cod adult (3+, >40 cm) | North Sea 1991 | 0.263 | 0.253 | (zoo 0.001) | M&D EcoBase 457 | [F] |
| Whiting juv / adult | North Sea 1991 | 0.290 / 0.124 | 0.237 / 0.093 | (zoo 0.17 / 0.05) | M&D EcoBase 457 | [F] |
| Haddock juv (0-1) / adult | North Sea 1991 | 0.673 / 0.535 | 0.592 / 0.517 | (zoo 0.15 / 0.06) | M&D EcoBase 457 | [F] |
| Saithe juv (0-3) / adult | North Sea 1991 | 0.028 / 0.008 | 0.009 / 0.001 | (zoo 0.57 / 0.24) | M&D EcoBase 457 | [F] |
| Norway pout | North Sea 1991 | 0.238 (infauna 0.191!) | 0.191 | (zoo 0.762) | M&D EcoBase 457 | [F] |
| Hake / blue whiting | North Sea 1991 | 0.001 / 0.149 | - | - | M&D EcoBase 457 | [F] |
| Other gadoids small / large | North Sea 1991 | 0.553 / 0.075 | - | - | M&D EcoBase 457 | [F] |
| Cod age 0 / 1 / 2 / 3 / 4 / 5 / 6+ (mean L 10/27/45/57/73/86/99 cm) | North Sea 1991, 4 quarters | 0.78 / 0.47 / 0.29 / 0.21 / 0.17 / 0.12 / 0.12 | 0.40 / 0.38 / 0.26 / 0.19 / 0.16 / 0.12 / 0.12 | 0.036 / 0.003 / 0.002 / ~0 | CRR 219 Table 8-A | [F] |
| Haddock age 0 / 1 / 2 / 3 / 4 / 5 / 6+ (L 11-44 cm) | same | 0.63 / 0.43 / 0.45 / 0.38 / 0.45 / 0.40 / 0.38 | 0.52 / 0.38 / 0.41 / 0.37 / 0.41 / 0.39 / 0.34 | 0.048 / 0.025 / 0.013 / 0.021 / <0.01 | CRR 219 Table 8-B | [F] |
| Whiting age 0 / 1 / 2 / 3 / 4 / 5 / 6+ (L 12-30 cm) | same | 0.19 / 0.18 / 0.12 / 0.10 / 0.10 / 0.09 / 0.08 | 0.11 / 0.13 / 0.09 / 0.08 / 0.07 / 0.07 / 0.06 | 0.06 / 0.017 / 0.011 / 0.012 | CRR 219 Table 8-C | [F] |
| Saithe age 3 / 4 / 5 / 6 / 7 / 8 / 9+ (L 42-95 cm) | same | 0.013 / 0.014 / 0.008 / 0.006 / 0.002 / 0.001 / 0.000 | ~same | 0.20 / 0.15 / 0.10 / 0.12 / 0.07 / 0.05 / 0.03 | CRR 219 Table 8-E | [F] |
| Cod (annual, all sizes) | North Sea, 1991 data, quarterly | 0.340 (deposit 0.065, filter 0.009, carnivores 0.266) | 0.074 without "carnivores" | carn. zoo 0.049 | Greenstreet et al. 1997 via Heath 2005 Table 3 | [F] of Heath; [S] for Greenstreet |
| Haddock | same | 0.422 (0.157 / 0.072 / 0.193) | 0.229 | 0.109 | same | same |
| Whiting | same | 0.244 (0.057 / 0.017 / 0.170) | 0.074 | 0.097 | same | same |
| Saithe | same | 0.002 | 0.001 | 0.354 | same | same |
| Norway pout | same | 0.059 (0.023 / 0.013 / 0.024) | 0.036 | 0.576 | same | same |
| Cod / haddock combined | North Sea 1981 | "approximately 45% of the food requirements ... derived from benthic evertebrates" | - | - | Cramer & Daan 1986 | [F] |
| Cod; whiting; haddock; saithe; Norway pout | North Sea 1981 (Year of the Stomach) | strict 0.10; 0.06; 0.34; 0; 0. With "other crustaceans + other invertebrates": 0.44; 0.19; 0.51; 0.02; 0.08 | - | euph 0.05; 0.10; 0.10; 0.36; 0.52 | Christensen 1995 model, EcoBase 680 | [F] model file |
| Haddock | Norwegian Deep (Skagerrak), 1985-87 | "Polychaetes, crustaceans, and ophiuroids dominated the diet, but teleosts were of less importance"; no % | - | euphausiids important in summer | Albert 1994, ICES JMS 51:31 | [A] |
| Norway pout | Norwegian Deep | "The diet was highly dominated by euphausiids"; no % | - | dominant | Albert 1994, ICES JMS 51:45 | [A] |

Notes on the CRR 219 rows. The values are annual means over the quarters sampled, weighted by the
mean stomach content ("Total weight all prey") of each quarter (the unweighted mean over quarters
differs by at most 0.1 for age 0-1, <=0.06 otherwise). Q1 has no 0-group. Table 8 covers the total
North Sea, roundfish areas 1-7 weighted; area 8, outside the North Sea, is excluded, so the
Skagerrak is not in it. The table shows the haddock Hyperiidea peak (Q2, age 3: 23.7 % of diet);
I classed it as pelagic. A few cells were OCR-damaged ('.' read as '0', e.g. "29049" = 29.49);
I repaired them, and every major-taxa column then sums to 100 +/- 1 %.

## Weighted share for a model-like gadoid community (computed)

Within species (M&D juvenile + adult):
`d_sp = (Bj*QBj*dj + Ba*QBa*da) / (Bj*QBj + Ba*QBa)`.
Example, cod with printed Table 3.3 B and Q/B: (0.079*5.96*0.401 + 0.161*3.5*0.263) /
(0.079*5.96 + 0.161*3.5) = (0.189 + 0.148)/1.035 = **0.326**. The same formula gives:
- printed B, Q/B (Table 3.3): cod 0.326, whiting 0.196, haddock 0.642, saithe 0.021
- EcoBase B, Q/B: cod 0.352, whiting 0.151, haddock 0.559, saithe 0.020

Community: `share = sum_i(m_i*QB_i*d_i) / sum_i(m_i*QB_i)` (consumption-weighted), or with
QB_i = 1 (biomass-weighted). m = IBTS mix cod 0.058, haddock 0.543, whiting 0.267, saithe 0.121,
hake 0.011 (sum 1). Hake d = 0. For Norway pout, m_NP is its biomass relative to the five
species together. In the North Sea 1991 model, NP : (cod+whiting+haddock+saithe) = 1.394/1.70 =
0.82. CRR219 species values used: cod 0.324 (mean of ages 1-3, assuming Skagerrak cod biomass is
mostly young fish), haddock 0.421 (ages 1-5), whiting 0.118 (ages 1-5), saithe 0.010 (ages 3-6).
Q/B for the Greenstreet and CRR rows = Greenstreet daily ration x 365 (cod 0.0144, haddock
0.0130, whiting 0.0148, saithe 0.0127, NP 0.0122 /d); hake 2.2. NP d = 0.059 (Greenstreet), or
0.238 in the M&D rows; in the "excl. shrimp + small epifauna" row NP d = 0.036 = Greenstreet's NP
deposit + filter feeders (0.023 + 0.013), i.e. without benthic carnivores.
Worked example (CRR, no NP, biomass-weighted): 0.058*0.324 + 0.543*0.421 + 0.267*0.118 +
0.121*0.010 + 0.011*0 = 0.019 + 0.229 + 0.032 + 0.001 = **0.280**.

| Diet source | NP = 0 | NP = 0.25 | NP = 0.5 | NP = 0.82 | NP = 1.0 |
|---|---|---|---|---|---|
| M&D EcoBase, printed B/QB weights | 0.42 / 0.42 | 0.39 / 0.39 | 0.36 / 0.36 | 0.34 / 0.34 | 0.33 / 0.33 |
| M&D EcoBase, EcoBase B/QB weights | 0.37 / 0.35 | 0.34 / 0.33 | 0.32 / 0.32 | 0.31 / 0.30 | 0.30 / 0.30 |
| Greenstreet 1997 (via Heath 2005) | 0.31 / 0.32 | 0.26 / 0.27 | 0.23 / 0.24 | 0.20 / 0.21 | 0.19 / 0.19 |
| CRR 219 (1991), central ages | 0.28 / 0.28 | 0.24 / 0.24 | 0.21 / 0.21 | 0.18 / 0.18 | 0.17 / 0.17 |
| CRR 219 excl. shrimp + small epifauna (NP 0.036) | 0.25 / 0.25 | 0.21 / 0.21 | 0.18 / 0.18 | 0.16 / 0.16 | 0.15 / 0.15 |

(cells: biomass-weighted / consumption-weighted). Weighting makes little difference because the
Q/B values are similar. The result is driven by haddock (54 % of the mix, d 0.42-0.64) and by how
much Norway pout is included. Sensitivity: cod at its age-1 value (0.47) instead of 0.32 adds only
+0.009, since cod is 6 % of the mix.

## Krill / shrimp classification per source

- M&D 2007: euphausiids sit in "Carnivorous zooplankton (Euphasiids, chaetognaths (arrow worms,
  eg sagitta), amphipods, mysiids, ichthyoplankton)" (section 10.2) = zooplankton. "Shrimp" (58;
  Crangon crangon, C. allmanni, Pandalus montagui, P. borealis) is benthic. "Small mobile epifauna
  (swarming crustaceans)" (59): "Crustaceans, molluscs, and polychaetes that live on the benthic
  interface and mysids, gammarids and amphipods that swarm off the bottom" (dominated by the
  amphipod Bathyporeia elegans) is benthic [F].
- Greenstreet 1997 / Heath 2005: carnivorous zooplankton = "Euchaeta spp., Tomopteris spp., hyperiid
  amphipods, euphausiids, chaetognaths" (zooplankton). "Macrobenthos carnivores" include the
  commercial "common shrimp (Crangon spp.), Norway lobster ..., northern prawn (Pandalus borealis),
  and edible crab" (benthos) [F].
- CRR 219: taxonomic. I counted Euphausiacea, Hyperiidea, Calanoida and Lepadomorpha as pelagic, and
  Caridea (shrimps incl. Pandalus), Mysida, Gammaridea and Cumacea as benthic (reported separately
  under "excl. S"). Krill matters only for saithe (euph 3-20 % of diet) and 0-group fish.
- Christensen 1995 (1981 model): Euphausiacea is its own group (zooplankton). "O. crustaceans"
  (feeding on detritus/benthos; crabs, shrimps, amphipods) is ambiguous, hence the strict/wide range.

## Q5 - consumption of benthos vs benthic production

M&D 1991 model (EcoBase 457, balanced). Fish consumption = sum over fish groups 5-48 of
B*QB*DC; production = B*P/B; units t/km2/yr = g WW/m2/yr [F, computed by me]:

| Benthic group | B | P/B | P | eaten by fish | by gadoids | by flatfish | all predators / P | fish/P |
|---|---|---|---|---|---|---|---|---|
| Large crabs | 1.35 | 0.55 | 0.75 | 0.62 | 0.17 | 0.24 | 0.83 | 0.82 |
| Nephrops | 1.10 | 0.37 | 0.41 | 0.29 | 0.10 | 0.11 | 0.89 | 0.71 |
| Epifaunal macrobenthos | 78 | 0.39 | 30.3 | 4.31 | 0.55 | 3.41 | 0.43 | 0.14 |
| Infaunal macrobenthos | 136 | 1.0 | 136 | 5.62 | 1.87 | 3.41 | 0.28 | 0.04 |
| Shrimp | 0.5 | 3.0 | 1.5 | 0.45 | 0.22 | 0.11 | 0.37 | 0.30 |
| Small mobile epifauna | 30 | 1.9 | 57 | 5.58 | 0.44 | 3.91 | 0.92 | 0.10 |
| Small infauna (polychaetes) | 150 | 0.9 | 135 | 10.96 | 0.36 | 4.10 | 0.91 | 0.08 |
| Sessile epifauna | 105 | 0.26 | 27.3 | 0.46 | 0.01 | 0.45 | 0.03 | 0.02 |
| Meiofauna | 4.1 | 35 | 144 | 1.78 | 0.00 | 0.23 | 0.99 | 0.01 |
| **Total** | 506 | - | **532** | **30.1** | **3.7** | **16.0** | 0.70 | **0.056** |
| excl. sessile + meiofauna | - | - | 361 | 27.8 | 3.7 | - | 0.63 | **0.077** |

("all predators / P" = predation by all consumers / production; EcoBase EE is slightly higher for
the fished groups crabs 0.84, Nephrops 0.94, shrimp 0.43 because EE includes catch.)

Gadoid groups 13-25 (B 2.87 t/km2) eat 15.5 t/km2/yr in total, of which benthos is 3.7 (24 %).
That is 1.3 t of benthos per t of gadoid per year. The Mareld gadoids (0.69 t/km2, Q/B ~4.8, 40 %
benthos) imply ~1.3 g WW/m2/yr. Top fish consumers of benthos: dab 9.9, sandeel 8.1, plaice 2.4,
Norway pout 1.7, lemon sole 1.3 g/m2/yr. The high EE of small infauna, small mobile epifauna and
meiofauna comes mainly from invertebrate predators (macrobenthos eating macrobenthos), not from fish.

Heath 2005, Table 4 (decadal means 1973-79 / 1980-89 / 1990-99, g C/m2/yr; North Sea, 602 535 km2)
[F]:
- Gross production of macrobenthos carnivores: 1.10 / 1.24 / 1.49. Eaten by fish: 0.789 / 0.657 /
  0.485, i.e. **72 / 53 / 33 %** (text: "between 50% and 60% of estimated production during the
  1970s, but had declined to 15-20% by the late 1990s").
- Macrobenthos deposit- and filter-feeders eaten directly by fish: 2.62 / 2.47 / 1.91. Fish
  demand on deposit/filter-feeder production, direct plus indirect: 6.59 / 5.80 / 4.42. Their
  production is not tabulated, so no fraction is given.
- Carbon conversions (from Greenstreet 1996): "macrobenthos filter-feeders, 0.04; macrobenthos
  deposit-feeders and carnivores, 0.08" g C/g WW. So fish ate ~10 -> 6 g WW/m2/yr of benthic
  carnivores and ~33 -> 24 g WW/m2/yr of deposit feeders (at 0.08). This is of the same order as the
  M&D 30 g WW/m2/yr.
- Benthic carnivore production 1983-85 = "1.3 g C m-2 y-1", estimated by Greenstreet et al. (1997)
  from the ICES North Sea Benthos Survey [S via Heath 2005].
- Heath 2007 summarises Heath 2005: fish consumption 10-20 g C/m2/yr (mean 14.3), of which
  "benthos (average 2.4 gC m-2 year-1)" [F of Heath 2007].

Cramer & Daan 1986 (ICES CM 1986/G:56), 1981 stomach data [F]: benthos eaten by the cod + haddock
stocks "decreases progressively from the southern (0.8 g.C.m-2.y-1) to the northern North Sea
(0.5)"; total North Sea 0.51 g C (~6 g WW at 0.08). Against "STEELE (1974) ... 2-5 g C.m2 y-1 ...
only cod and haddock would already take care of 10-25% of the total benthos production". Rachor
(1982) gives 5.0 / 0.4 / 0.3 g C for the S / C / N North Sea, which would make the
northern-central mismatch "even worse" [S via Cramer & Daan]. Jones (1982, 1984) put cod +
haddock consumption at only "0.24 and 0.4 g C" and benthos production at "0.8 g C" [S via Cramer &
Daan]. Their Table 4 is OCR-garbled (its components do not add to the total). I used only the
quoted text values.

Greenstreet et al. 1997, ICES JMS 54:243 [A]: secondary production "is sufficient to supply the
food requirements of ... benthivorous demersal fish"; "a detritivore trophic level can be inserted
between organic settlement and production of macrobenthos". No numbers in the abstract.

## Evidence list (URLs)

- Mackinson & Daskalov 2007, Cefas Sci. Ser. Tech. Rep. 142 (Tables 3.3, 3.4; sections 10.2, 11,
  13.15-13.23) [F]: https://www.cefas.co.uk/publications/techrep/tech142.pdf
- EcoBase model 457 "North Sea 1991" (Mackinson & Daskalov), full model file incl. diet matrix [F]:
  http://sirs.agrocampus-ouest.fr/EcoBase/php/webser/soap-client.php?no_model=457
- EcoBase model 680 "North Sea 1981" (Christensen 1995, "Dana. pp 1-28" as given in EcoBase) [F model file]:
  http://sirs.agrocampus-ouest.fr/EcoBase/php/webser/soap-client.php?no_model=680
- ICES 1997, Database report of the Stomach Sampling Project 1991, CRR 219 (Table 8 A-C, E; section
  6.3) [F]: https://doi.org/10.17895/ices.pub.4626 (PDF https://ndownloader.figshare.com/files/33403490).
  Section 6.3: "The diet of haddock was more varied, and included a larger proportion of
  echinoderms, annelids and molluscs".
- Heath 2005, ICES JMS 62:847-868 (Tables 3, 4) [F]: https://strathprints.strath.ac.uk/18596/6/strathprints018596.pdf
- Heath 2007, ICES JMS (zooplankton consumption by early life stages; Table 8) [F]:
  https://strathprints.strath.ac.uk/18621/6/strathprints018621.pdf
- Cramer & Daan 1986, ICES CM 1986/G:56 [F]: https://www.vliz.be/imisdocs/publications/ocrd/274395.pdf
- Greenstreet et al. 1997, ICES JMS 54:243-266 [A]: https://academic.oup.com/icesjms/article/54/2/243/634560
- Albert 1994, haddock, Norwegian Deep, ICES JMS 51:31-44 [A]:
  https://academic.oup.com/icesjms/article-lookup/doi/10.1006/jmsc.1994.1004
- Albert 1994, Norway pout, Norwegian Deep, ICES JMS 51:45-61 [A]:
  https://academic.oup.com/icesjms/article-lookup/doi/10.1006/jmsc.1994.1005
- ICES WGINOSE 2020 report [F, searched]: the "Kattegat EwE" it cites is Niiranen et al. 2012
  (a Baltic food-web model); no benthos consumption figures. https://ndownloader.figshare.com/files/33484958

## Gaps

- No Skagerrak or Kattegat diet by mass retrieved. Bergstad 1991 (Sarsia 75:269-325, Norwegian
  Deep gadoids), Albert 1994/1995 full texts, Mattson 1990, Pihl 1994 (Kattegat, CJFAS 51:321)
  and Hüssy et al. 1997 were not accessible (paywall / no PDF found). CRR 219 Table 8 excludes
  the Skagerrak (roundfish area 8 is unweighted). Its Table 10 by roundfish area covers only
  commercial prey and was not mined.
- Daan (ed.) 1989, CRR 164 (1981): not opened. 1981 values come via Christensen 1995 (EcoBase 680)
  and Cramer & Daan 1986. DAPSTOM, Hislop 1991 (haddock), Schückel et al. 2010, the North Sea SMS
  key-run diets and WGSAM stomach data were not mined. SMS has fish prey only, plus "other food".
- Greenstreet et al. 1997 full text (benthos production by group, g C) not obtained; numbers are via
  Heath 2005 only. Collie 1987, Jennings et al. 2001/2002, Hiddink et al. and Rijnsdorp &
  Vingerhoed 2001 were not retrieved, so no further production-vs-consumption fractions from them.
- No Skagerrak/Kattegat EwE or mass-balance budget with benthos consumption was found.
- Norway pout's benthos share is uncertain: 0.06 (Greenstreet), 0.08 wide (Christensen) and 0.24
  in M&D (where 0.19 "infaunal macrobenthos" looks odd for a euphausiid feeder). Its biomass in
  the model box is an assumption, so the with-pout community values are a sensitivity, not an
  estimate.
- The age weighting of the CRR values (cod ages 1-3, etc.) is my assumption about the Skagerrak
  age structure. It is not measured.
