# Nitrogen recycling inside the euphotic layers (Skagerrak / Kattegat / North Sea) and what it means for the Mareld plankton

Question: how can the model's gross primary production (PP) rise from 68 to about 150 g C m-2 yr-1, at roughly unchanged biomass, with an annual f-ratio (new / total production) of 0.2-0.4 instead of 0.55?

Grades:
- [F] full text read
- [A] abstract only
- [S] search snippet or secondary citation only
- [N] not found or no access
- [D] own analysis of SHARK data

Markers:
- "(Own calculation)" marks my arithmetic.
- "(Own judgement)" marks my interpretation.

Units: 1 t WW/km2 over 10 m = 10 ug C/l at C:WW 0.1.

Detailed quotes, tables and URLs are in the five evidence files in this folder (`evidence_A` to `evidence_E`). This note is the synthesis.

Builds on, does not redo:
- `Detritus och remineralisering/detritus_remineralisation.md` (ERSEM / NEMURO / Fennel detritus rates)
- `Mikrozooplankton/microzooplankton.md` (Q6 regeneration, Q7 model parameters)
- `Näringspool för växtplankton/nutrient_parameters.md` (Q2: copepod N fate)
- `Näringspool för växtplankton/phytoplankton_losses.md` (lysis and sinking)
- `Primärproduktion och C:chl i Skagerrak/pp_and_c_chl.md` (PP target 150, f 0.2-0.4)
- Resume sections 177, 179, 201-205

---

## 0. Summary

1. **The gap is one number: the share r of phytoplankton N losses that comes back to the euphotic pools.**
   - Gross PP = new N / (1 - r). New N is about 38 g C-equivalent per year.
   - Today r is about 0.45. PP 150 needs r 0.75; PP 120-200 needs r 0.68-0.81. (Own calculation)
2. **In the model about 80 % of PP goes through the copepods, and that path returns only about 0.43.** The literature has it the other way round.
   - Model: microzooplankton graze about 7 % of PP, and their path returns about 0.77. (Own calculation from the saved fix70 run, section 1.)
   - Kattegat: ciliates eat 20-60 % and heterotrophic dinoflagellates 57 % of annual PP [F]. Copepods take a minority.
3. **Copepods: the dissolved N released is 0.21-0.50 of ingested N as NH4 alone, rising as ration falls** (Kiørboe et al. 1985 [F]).
   - Counting urea, sloppy feeding and pellet leaching, the total is 0.49-0.58 (Saba 2010/2011 [F]).
   - Egestion is 0.05-0.40 of ingested N; N gross growth efficiency is 0.3-0.4 [F].
   - The model's 0.3 equals the NH4 excretion at moderate ration only.
4. **Faecal pellets are not 100 % export.**
   - Retention in the upper 15-50 m is 60-97 % for small-copepod summer communities (North Sea, August, 3-39 % reached 50 m [F]).
   - It is above 80 % of all Acartia egesta in a 15 m mixed layer [F].
   - It is only about 16 % for large calanoid pellets at 30 m in Gullmar Fjord [F].
   - In the Kattegat, N is recycled preferentially: trap C/N 8.6 against 6.4 suspended [A].
5. **f-ratio.**
   - SE Kattegat: the March bloom is about 80 % nitrate; in summer NH4 is about 50 % and urea about 30 % of uptake; annual uptake 2.26 mol N m-2 yr-1, about 180 g C (Own calculation) [S].
   - Western Channel: annual f 0.32 [F].
   - Norwegian Trench: July f below 0.02 [F].
   - NH4 turns over in about 1 h to a few hours in summer [A].
   - In a one-pool model, f is purely a supply budget, so k_N does not set it (evidence C, section 4).
6. **Summer surface community.**
   - Cells under 8-11 um are 30-85 % of biomass and up to 85 % of PP [F].
   - Heterotrophic protists hold 5-15 ug C/l on a prey field of 12-65 ug C/l (central Skagerrak, June [F]).
   - Microzooplankton, not bacteria, are the main summer NH4 regenerators. Bacteria take 10-35 % of NH4 uptake [A].
   - A 12 ug C/l threshold is low by lab standards (6-20, tail to 300 [F]).
   - Starving ciliates die in 1-3 d, heterotrophic dinoflagellates last 30-71 d [F]. Nothing supports a lower microzooplankton M1.
   - The model's microzooplankton starve in summer because their prey is at 4-7 ug C/l, below their break-even of about 12 ug C/l (Own calculation, section 6.5). The threshold is not what kills them.
7. **Ranking** (static budget, Own calculation; details in section 8):
   - **Parameter-only:** `grazing_regeneration` 0.3 -> 0.5 gives PP about 95-100 and f about 0.39.
   - **Microzooplankton at their literature share:** together with the first change, PP about 110-115, f about 0.34. This needs them to survive the summer.
   - **Routing the mixed-layer producer's sinking to the subsurface layer, plus a modest zooplankton M1 return:** PP about 120-135, f about 0.28-0.32.
   - Reaching 150 needs every lever at the upper end of its range. 120-135 is inside the 120-200 target band.

---

## 1. The budget frame (Own calculation)

At an annual steady state, every unit of N the producers take up either returns to the euphotic pools, with share r, or leaves the 0-30 m system. So uptake = new + r x uptake, and:

- PP = New / (1 - r)
- f = 1 - r

Each loss path i of phytoplankton N (grazing by copepods, grazing by microzooplankton, phytoplankton M1) takes a share x_i of production and returns a share rho_i, counting what comes back downstream through the grazer's own M1. Then r = sum of x_i x rho_i.

**Model today** (the section-205.4 baseline of `tools/probes/a17_score.py`: frozen fix70 policies, seed 20260530, 3 years after 1 burn-in; PP 68.4, f 0.552; its monthly stocks are used below):

| Path | Flux, g C m-2 yr-1 (rough) | x_i | rho_i | Basis |
|---|---|---|---|---|
| Phytoplankton M1 (sinking) | 8.3 | 0.12 | 0 | 0.005/tick x 1460 = 7.3 /yr x 1.14 g C/m2 mean column stock |
| Microzooplankton grazing | ~5 | ~0.07 | ~0.77 | M1 0.77 g C/yr on a 0.42 t/km2 mean stock, plus predation, / GGE 0.3; returns 0.40 + 0.30 at once, plus 0.5 of its M1 |
| Copepod grazing | ~55 | ~0.81 | ~0.43 | remainder; returns 0.30 at once, plus 0.5 x M1. M1 is 19.6 /yr x 7.4 t/km2 = 14.5 g C/yr, so 7.3 g C returned |
| Sum of returns | ~27.7 | | r ~0.41 | against 0.445 implied by f = 0.552; close enough for a rough partition |

What this implies:
- Copepod production is 25 /yr x 0.74 g C/m2, about 18.5 g C/yr, against an intake of about 55-60. That puts the model copepods' gross growth efficiency at about 0.33, within the literature N-GGE of 0.3-0.4 (section 2). (Own calculation)
- Column P/B is 68.4 / 1.14, about 60 /yr. 150 at the same biomass needs about 130 /yr. Doubling the turnover means the producers' losses must double as well, and the grazers have to take that extra growth. Recycling and grazer survival are therefore one problem, not two.

Caveats on this frame:
- New N is held at 38. Higher subsurface pools would lower the reservoir exchange, which is proportional to N_res - N_sub, and therefore lower the new supply.
- The x_i shift when the food web shifts.
- These are steady-state annual numbers, not a simulation.

---

## 2. Copepod nitrogen budget (task item 1)

| Fate, share of ingested N | Value | Source (grade) |
|---|---|---|
| NH4 excreted, Acartia tonsa, 18 C, acclimated 2 d | 0.50 / 0.245 / 0.21 at low / intermediate / saturating food (19 / 75 / 187 ug N/l) (Own calculation from Table 3) | Kiørboe, Møhlenberg & Hamburger 1985, MEPS 26:85, doi:10.3354/meps026085 [F] |
| Respired C, share of ingested C, same | 0.32 / 0.14 / 0.12 | same [F] |
| Urea share of excreted N | < 10 %; NH4 > 80-90 % | same [F] |
| NH4 excretion, A. tonsa on Thalassiosira, 20 C | 0.34 | Saba, Steinberg & Bronk 2011, JEMBE 404:47, doi:10.1016/j.jembe.2011.04.013 (dissertation version) [F] |
| Sloppy feeding NH4 + urea, same | 0.08 + 0.10 | same [F] |
| Urea leached from pellets, same | 0.06 | same [F] |
| Total dissolved N, all routes | 0.58 of N ingested (Own calculation); 0.49 of N removed from suspension | same [F] |
| Egestion (particulate) | 0.05 (low food) to 0.40 (high food); 0.17-0.23 (Saba); Calanus 1 - 0.62 = 0.38 | Kiørboe 1985 [F]; Saba [F]; Corner et al. 1967 via Ikeda & Motoda 1978 [F, secondary] |
| N assimilation efficiency | Acartia 0.60-0.95 (falls with ration); Calanus 0.54-0.68 | same |
| N gross growth efficiency | Acartia 0.40-0.49; Calanus 0.14-0.38 | same |
| Sloppy feeding, Calanus on bloom diatoms | 49 % of the C removed lost as DOC; pellets lose > 20 % of C in the first hour | Møller, Thor & Nielsen 2003, MEPS 262:185, doi:10.3354/meps262185 [F] |
| Temperature | NH4 excretion Q10 1.8-2.0 | Ikeda et al. 2001, Mar Biol 139:587, doi:10.1007/s002270100608 [S] |
| Mesozooplankton excretion as a share of phytoplankton N demand | typically 10-30 %; 4 % in a herbivorous bloom, 30 % in a multivorous food web (Iberian shelf); none found for the North Sea, Kattegat or Skagerrak | evidence A, section 3 ([A]/[S]) |

How to read the table:
- **The two budgets cannot be averaged.** Kiørboe gets assimilation by difference, with any DOM loss hidden in "assimilated". Saba derives egestion from an assumed pellet C:N. Use Kiørboe for the ration curve of NH4 excretion, and Saba for the urea, sloppy-feeding and leaching routes that come on top of it.
- **The share is food-dependent.** The excreted share is highest at low ration, because basal excretion is a fixed cost. In the stratified summer, copepods sit on the low-food side, so the share there is towards 0.4-0.5 [F]. With Q10 about 2 for both excretion and ingestion, the share should be roughly temperature-independent at a given relative ration. (Own judgement; no source measured it.)
- **Time scale.** NH4 and urea are released during feeding, peaking in the first 20 min [F]. Sloppy-feeding DOM is labile, and dissolved free amino acids turn over in 11-36 h [S]. At a 6 h tick, all of it can enter the pool at once. A separate DON pool would turn over in about 0.5-2 d (Own judgement), which section 201.5 found makes no difference.

**Translation to the model** (Own calculation):
- The model copepod assimilates 0.65 and grows at about 0.33 of intake (section 1).
- The N of the respired share, 0.65 - 0.33 = 0.32, is what today's `grazing_regeneration` 0.3 returns. It is the Kiørboe NH4 excretion and nothing more.
- The N of the egesta (0.35) is all exported today. A mass-consistent copepod return is 0.32 + retention x 0.35:

| Pellet retention | Return |
|---|---|
| 0.2 (calanoid spring) | 0.39 |
| 0.6 (summer, small copepods) | 0.53 |
| 0.8 | 0.60 |

Raising the share is not the double counting that section 179.1 warned against, as long as the zooplankton M1 return stays tied to the growth share. The 0.5/0.5 run there added respiration on top of the 0.3, which counted the respired N twice.

---

## 3. Faecal pellets: export or retention (task item 1)

| Site, season, depth | Exported / produced | Copepods | Source (grade) |
|---|---|---|---|
| Northern North Sea, August 2002, 50 m | 3-39 %; degradation 1-3 /d; residence 0.3-0.8 d | Oithona similis | Poulsen & Kiørboe 2006, MEPS 323:195, doi:10.3354/meps323195 [F] |
| Acartia, 15 m mixed layer (lab rates) | about 40 % of pellets, < 20 % of all egesta; > 50 % of egesta not in pellets; pellets sink about 5 m/d, degrade about 0.5 /d at 18 C | Acartia tonsa | Olesen, Strake & Andrushaitis 2005, MEPS 293:131, doi:10.3354/meps293131 [F] |
| Gullmar Fjord (Skagerrak coast), October / March, 30 m | about 84 % (regression slope 0.84); copepod pellets 6.0 % of PP below 30 m | Pseudocalanus, Temora | Vargas et al. 2002, MEPS 241:125, doi:10.3354/meps241125 [F] |
| Southern Kattegat, March-October 1989, pycnocline (about 10-20 m) | "Copepod defecation nearly equaled the sedimentation of fecal pellets" | calanoids | Lundsgaard & Olesen 1997, L&O 42:1001, doi:10.4319/lo.1997.42.5.1001 [A] |
| Disko Bay, June, euphotic zone | 65 % (35 % retained) | Calanus | Juul-Pedersen et al. 2006, MEPS 314:239, doi:10.3354/meps314239 [A] |
| Northern Baltic | "> 99 % of the fecal matter was recycled in the upper water column" | Acartia, small copepods | Viitasalo et al. 1999, L&O 44:1388, cited by Olesen 2005 [S] |
| Review | retention 30-98 %; pellets mostly < 40 % of POC flux; sinking 5-220 m/d | various | Turner 2015, Prog Oceanogr 130:205, doi:10.1016/j.pocean.2014.08.005 [F]; Turner 2002, AME 27:57, doi:10.3354/ame027057 [F] |
| Øresund, seasonal degradation | 0.52 /d (late winter) to 2.5 /d (spring bloom); protozoa 15-53 % of it; mesozooplankton not significant | - | Poulsen & Iversen 2008, MEPS 367:1, doi:10.3354/meps07611 [F] |
| Coprophagy | copepods fragment pellets (coprorhexy) more than they eat them: about 5 % ingested (Acartia, Temora); < 37 % (C. helgolandicus) | - | Poulsen & Kiørboe 2005, MEPS 299:217 [F]; Iversen & Poulsen 2007, MEPS 350:79 [F] |
| Kattegat trap C/N | 8.6 trapped against 6.4 suspended: N recycled preferentially in the euphotic zone | - | Olesen & Lundsgaard 1995, ECSS 41:475 [A] |

**The Kattegat tension, resolved as far as the sources allow** (Own judgement; the 1997 full text was not available):
1. **Pellets only.** The 1997 "defecation" term probably counted pellets. The same first author wrote in 2005 that "only a minor fraction of copepod defecation appears to leave the upper water column as fast-sinking fecal pellets in coastal waters" [F].
2. **Shallow traps.** The traps sat at a 10-20 m pycnocline, so "export" there means leaving the mixed layer. In the model, that matter would enter the subsurface layer, not leave 0-30 m.
3. **Community.** Calanoid pellets (spring, Gullmar) are exported; summer small-copepod and Oithona communities retain theirs.

**For the model** (Own judgement):
- **Annual retention within 0-30 m: about 0.5** (range 0.2-0.8). Roughly 0.2-0.3 in the calanoid spring and 0.6-0.8 in summer.
- **Steady-state escape fraction from a layer: s / (s + r_deg)**, with s = v/H. For example, v 5-25 m/d and r_deg 0.5-2.5 /d over H 30 m give 5-45 %; for v 100-200 m/d at r_deg 0.5 it is about 85-93 %. (Own calculation)
- **Where it returns.** Egesta from feeding in the mixed layer remineralise largely in the subsurface layer.

---

## 4. Microzooplankton N budget and who regenerates NH4 (task item 2)

- **Microzooplankton budget** (not redone; `Mikrozooplankton/microzooplankton.md`, Q6):
  - NH4 excreted is 0.4 of ingested N (range 0.15-0.55; tintinnids 53 % [F]).
  - Gross growth efficiency on carbon is 0.3, which by stoichiometry is about 0.4 on nitrogen.
  - So about 0.6 of the N is released. The model returns 0.40 + 0.30 = 0.70 at once, with all of the egesta returned (`egestion_regeneration` 1.0). That is at the generous end and should not be raised.
  - ERSEM sends only half of microzooplankton egesta to DOM; the other half goes to POM sinking at 5 m/d (evidence E [F]).
- **Partition of NH4 regeneration in coastal and shelf seas:**
  - Organisms under 200 um dominate:
    - Oslofjord summer: mostly the 45-200 um fraction, including copepod nauplii (Paasche & Kristiansen 1982, Mar Biol 69:55, doi:10.1007/BF00396961 [A]).
    - Western Channel, summer: about 50 % from microplankton and 20-45 % from organisms under 1 um (Maguer et al. 1999, CSR 19:1755, doi:10.1016/S0278-4343(99)00037-0 [S]).
    - A Coruña: zooplankton under 10 % (Bode et al. 2004, JPR 26:11, doi:10.1093/plankt/fbh003 [A]).
  - Mesozooplankton supply 12-33 % in oceanic estimates [S].
  - Bacteria are more often net NH4 consumers, taking 10-35 % of NH4 uptake (Hoch & Kirchman 1995, L&O 40:886 [A]; Baltic about 20 %, Klawonn et al. 2019, ISME J, doi:10.1038/s41396-019-0386-z [A]).
  - Central Skagerrak, June: "herbivorous micrograzers (viz. ciliates and heterotrophic dinoflagellates) were responsible for a substantial excretion of nutrients which had not been processed by bacteria" (Bjørnsen et al. 1993, MEPS 95:279, doi:10.3354/meps095279 [F]).
- **Synthesis for summer** (Own judgement, not a measured partition): protists 50-80 %, bacteria / picoplankton 10-45 %, mesozooplankton 10-20 % of NH4 regeneration.
- **Why the model is lopsided.** In the model the copepods carry about 80 % of the grazing, and therefore nearly all of the recycling (section 1).

---

## 5. Regenerated production, NH4 kinetics and the single nutrient currency (task item 3)

### 5.1 f-ratio by season

| Site, season | f, or the form of N taken up | Source (grade) |
|---|---|---|
| SE Kattegat (Laholm Bay), March bloom | NO3 about 80 % of uptake | Sahlsten, Sörensson & Pettersson 1988, JEMBE 121:227, doi:10.1016/0022-0981(88)90090-1 [S] |
| SE Kattegat, summer / early autumn | NH4 about 50 %, urea about 30 %, so NO3 + NO2 at most about 20 % (Own calculation) | same [S] |
| SE Kattegat, annual | uptake 2.26 mol N m-2 yr-1, about 180 g C (Own calculation, Redfield); bloom 16 % of it; "~25 % new" unconfirmed | same [S] |
| Western English Channel (permanently mixed), annual | f 0.32; summer NO3 27 %, NH4 55 %, urea 13 % | L'Helguen, Madec & Le Corre 1993, Oceanol Acta 16:653 [F] |
| Oslofjord, April-October | urea 19 % of uptake (mean) | Kristiansen 1983, Mar Biol 74:17, doi:10.1007/BF00394270 [A] |
| Norwegian Trench, July | < 0.02 | Bendtsen & Richardson 2018 [F] (pp_and_c_chl.md) |
| Central Skagerrak, June, surface | 0.04-0.11; nitrate uptake 4-11 % of N demand | Bjørnsen 1993 [F] |
| Skagerrak, late summer | 0.18-0.36 | Richardson 2003 via the Maar thesis [F]/[N] |
| POLCOMS-ERSEM, northern North Sea shelves | pelagic recycling / uptake 0.86-0.90 (whole column, includes nitrified N, so an upper bound on 1 - f) | Holt et al. 2012, BG 9:97, doi:10.5194/bg-9-97-2012 [F] |

Measured f-ratios are probably upper bounds: routine 15N work underestimates NH4 uptake about 2x (Glibert et al. 1982, L&O 27:639 [F]), and including urea lowers f (Harrison et al. 1987, JPR 9:235 [A]). An annual f of **0.25-0.35** is the best-supported target. The summer surface value is below 0.1-0.2.

### 5.2 NH4 turnover and concentrations
- **Turnover:** about 1 h in the N-depleted summer Baltic [A]; "within a few hours" in the summer Oslofjord [A].
- **SMHI July 2022:** Skagerrak and Kattegat surface DIN at or below 0.1 umol/l, the detection limit [F].
- **SHARK A17** [D], 0-10 m, 2001-2009 (detection limit 0.05; values below it set to half of it; later years have a limit of 0.20 and are not used). Monthly means:

| Depth | Quantity | J | F | M | A | M | J | J | A | S | O | N | D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-10 m | NH4, umol/l | 0.17 | 0.08 | 0.24 | 0.25 | 0.16 | 0.11 | 0.16 | 0.13 | 0.11 | 0.30 | 0.31 | 0.46 |
| 0-10 m | NO2+NO3 (2001-2024), umol/l | 5.4 | 5.7 | 2.6 | 0.78 | 0.43 | 0.12 | 0.10 | 0.10 | 0.10 | 0.30 | 1.4 | 3.7 |
| 15-30 m | NH4 (2001-2009), umol/l | 0.07 | 0.10 | 0.21 | 0.43 | 0.42 | 0.23 | 0.21 | 0.12 | 0.24 | 0.45 | 0.31 | 0.42 |

  - 24-41 % of the summer NH4 values are below the detection limit.
  - The summer nitrate means of 0.10 are mostly detection-limit values as well.
  - The 15-30 m maximum of 0.4 in April-May is bloom decay.
- **Reading** (Own judgement): A17 summer surface DIN is about 0.1-0.25 umol/l, and NH4 is at least as large as nitrate.
  - The model's single pool stands for DIN, so its summer target is NH4 + nitrate. `tools/probes/a17_score.py` compares the pool with nitrate alone.
  - The model's summer mixed-layer pool of 0.05-0.09 is therefore 1.5-4x below A17's DIN, which points the same way as the f-ratio: too little recycling.

### 5.3 Half-saturation and one currency
- **Fasham et al. 1990** (J Mar Res 48:591, doi:10.1357/002224090784984678 [F]): K1 (NO3) = K2 (NH4) = 0.5 mmol m-3, with K2 "given the same value as K1" for lack of data, and inhibition psi 1.5. Even with two pools, that model's annual f was 0.71 against 0.31 observed.
- **Size dependence:** K is proportional to cell size, at or below 0.5 uM for small cells (Eppley et al. 1969, L&O 14:912 [A]).
- **Oceanic values:** K_NH4 is about equal to K_NO3, both 20-30 nM; NH4 is preferred at all concentrations; inhibition is never complete (Harrison et al. 1996, L&O 41:16 [A]).
- **Field inhibition, Channel:** NO3 uptake halves at NH4 0.5 umol/l [F].
- **Other models' choices:**
  - ERSEM has no NH4 preference: uptake is proportional to affinity x concentration [F].
  - NEMURO: K_NH4 0.1 against K_NO3 1.0 for small cells, psi 1.5 [F].
  - ECOSMO: psi 3.0 [F].
  - NORWECOM appears to have a single DIN pool [S].
- **Verdict** (Own judgement, agreeing with evidence C, section 4): one currency is acceptable.
  - In stratified summer water there is no nitrate left to inhibit, and the pool turns over in hours.
  - In a one-pool model, f is the ratio of supply from below to total uptake. It is fixed by the return shares, not by k_N.
  - Section 203.2 agrees: k_N 0.5 -> 0.2 added only 1 g C of PP.
  - What one pool loses is a size-selective advantage for small cells at low N. That matters only together with a small-cell producer (section 8, item 9).

---

## 6. The summer surface community and microzooplankton survival (task item 4)

### 6.1 Small cells dominate the summer
- **Skagerrak, May 1987:** "small flagellates" in the central stratified area; cells under 8 um about 30 % of chlorophyll, about 50 % by counts (Kiørboe et al. 1990, MEPS 59:19, doi:10.3354/meps059019 [F]).
- **Dogger Bank, May:** cells under 11 um were 82 % of biomass and 85 % of production (Nielsen et al. 1993, MEPS 95:115 [F]).
- **Kiel Bight:** cells under 20 um were 70-100 % of biomass and production outside the blooms; picoplankton 8-33 % in midsummer; summer production "ammonia-maintained" (Jochem 1989, MEPS 53:153 [A]).
- **Gullmar Fjord:** Synechococcus up to 9.3e4 cells/ml in late summer (Laber et al. 2022, Front Microbiol 13:828459 [F]). At an assumed 100-250 fg C per cell, that is about 9-23 ug C/l. (Own calculation; the conversion factor is not from the paper.)

### 6.2 Summer protist stocks and grazing
- **Central Skagerrak, June 1990, 0-15 m, nitrate below the detection limit** (Bjørnsen et al. 1993 [F]):
  - autotrophs about 65 ug C/l by microscopy (12-18 by chl x 30; unresolved in the paper)
  - heterotrophic nanoflagellates 3.6-5.4, heterotrophic dinoflagellates 3.4-3.5, ciliates 0.8-1.4 ug C/l
  - bacteria 16-23 ug C/l
  - PP 9-14 ug C/l/d
- **Kattegat** (Levinsen & Nielsen 2002, L&O 47:427 [F]):
  - Ciliates grew at near-maximal, temperature-limited rates "independent of various measures of food availability".
  - Heterotrophic dinoflagellates were food-limited, fell to pre-bloom levels after the bloom, and rose again in late summer. Neither group disappeared.
  - Ciliates ate 20-60 % and heterotrophic dinoflagellates 57 % of annual PP (290 g C).
- **Late-summer Skagerrak:** micrograzers potentially took 43-166 % of daily PP (Maar et al. 2004, Mar Biol 144:917 [S]).

### 6.3 Thresholds and mortality
- **Lab thresholds:** Balanion 11 and Gymnodinium 17-18 ug C/l. The compiled list mostly runs 6-20, with a tail to 246-325 that the authors suspect is a lab artefact (Jakobsen & Hansen 1997, MEPS 158:75 [F]). Oligotrichs: 10-100 (Montagnes 1996, MEPS 130:241 [F]).
- **Models:** ERSEM 15.06 `minfood` is 12 mg C m-3 for Z4, Z5 and Z6, a Michaelis-Menten food-perception constant [F]. The functional form was not checked against the ERSEM source; that it matches the model's F^2/(F + F0) comes from the library comment only. ERSEM III listed 50 [F].
- **Dilution experiments** usually find no threshold (microzooplankton.md).
- **Starvation:** naked oligotrich ciliates die in 1-3 d; Balanion is gone in 60 h. Heterotrophic dinoflagellates survive 30 d (Gymnodinium) to 71 d (Protoperidinium) and cut respiration by 25-70 % (Montagnes 1996; Menden-Deuer et al. 2005, AME 41:145; Calbet et al. 2013, MEPS 483:67 [F]). 30-71 d to 1 % survival is a loss of 0.065-0.15 /d (Own calculation).
- **Verdict** (Own judgement): **no evidence for a lower M1 than 0.05 /d.** Mortality at low food is higher in the literature, not lower. The 12 ug C/l threshold is at the low end of the lab values.

### 6.4 Mixotrophy
- **Kattegat:** Mesodinium rubrum is more than 50 % of ciliate biomass in February-April; Laboea is present; mixotrophic Ceratium in late summer (Levinsen & Nielsen 2002 [F]).
- **Elsewhere:** plastidic oligotrichs are up to 90 % of summer oligotrichs (Woods Hole); M. rubrum is 58-73 % of annual ciliate biomass in the Gdańsk Basin (Stoecker et al. 2009, AME 57:279, doi:10.3354/ame01340 [F]).
- **Gap:** no Kattegat or Skagerrak summer share was found [N].
- **Effect:** mixotrophy lowers the prey level at which net growth is zero.

### 6.5 Why the model's microzooplankton starve in summer (Own calculation)

Net growth = 0.3 x Imax(T) x F/(K + F) - 0.05 /d, with:
- F = P^2/(P + 12) ug C/l (the threshold term)
- K = 70 ug C/l (the library's 7 t/km2 at 10 m; the field value is about 30)
- Imax = 1.7 /d at 10 C, Q10 2.6

Predation by copepods is ignored here, so these break-even prey levels are lower bounds.

| Temperature used | K, ug C/l | Break-even P with threshold, ug C/l | Break-even P without threshold, ug C/l |
|---|---|---|---|
| July mean_0_50m, 12.1 C (as the manifest assigns) | 70 | 12.1 | 6.1 |
| same | 30 | 7.0 | 2.6 |
| July A17 0-10 m, 16.3 C [D] | 70 | 9.2 | 4.0 |
| same | 30 | 5.5 | 1.7 |

What the table shows:
- **The model's summer surface prey is 4-7 ug C/l**, so microzooplankton cannot grow there with today's K and temperature.
- **At A17's 32-56 ug C/l** they grow at about +0.1 /d.
- **The half-saturation 70 and the cold 0-50 m temperature** together raise the break-even about 2x.
- **Why the threshold tests failed.** Section 177 raised K from 3 to 7 t/km2 because 3 over-grazed the bloom in the one-layer world, which had no Type III copepods, no second layer and no regeneration. Lowering the threshold alone failed three times (sections 179.3, 201.4, 203.2): the microzooplankton ate the spring bloom and the summer stayed empty. The threshold is not the summer limit; prey level, K and temperature are.
- **A17 surface temperature [D]**, 0-10 m: 5.4 3.8 4.2 6.1 9.7 13.8 16.3 17.3 15.7 12.8 10.3 7.6 C. At 15-30 m: 6.7 5.8 5.6 5.9 6.8 8.3 10.7 12.6 14.1 13.2 10.9 8.7 C.

---

## 7. How comparable models carry the recycling (task item 5)

Detritus and DOM rates are in `detritus_remineralisation.md`, and microzooplankton Z5 and NORWECOM / ECOSMO microzooplankton values in `microzooplankton.md` Q7. Source code and parameter files were read for all rows (evidence E [F], except where marked):

| Model / group | To biomass | To NH4 at once | To DOM | To POM / detritus | NH4 handling |
|---|---|---|---|---|---|
| ERSEM Z4 mesozooplankton (pu 0.6, pu_ea 0.5, pe_R1 0.5) | 0.6 (minus rest respiration 0.015 /d) | about 0.2 (the N of respired C, by stoichiometry) | 0.1 | 0.1 (R8, 10 m/d) | uptake proportional to affinity x concentration; no NH4 preference |
| ERSEM Z5 / Z6 | quota-dependent | excess N at stempn 0.5 /d | half of egesta | half of egesta (R6 / R4) | same |
| ERSEM B1 bacteria | growth efficiency 0.6 minus srs 0.1 /d | release above the N quota, uptake below it | - | - | the only DON -> NH4 route |
| NEMURO zooplankton (all three) | 0.30 | 0.40 | 0 | 0.30, plus all mortality | psi 1.5; K_NH4 0.1, K_NO3 1.0 (small cells) |
| Fasham, Kühn & Radach 1997 FABM port | 0.625, then losses 0.7 to NH4 / 0.1 to DON / 0.2 to detritus | via the losses | via the losses | 0.375 plus 0.2 of the losses | k1 0.2, k2 0.8 |
| Fennel et al. 2006 | 0.75 | 0.1 + 0.1 /d of biomass | 0 | 0.25 | NO3 inhibition 1/(1 + 2 NH4) |
| ECOSMO operational | 0.75 | 0.06 / 0.08 /d of biomass | 0.10 | 0.15 | psi 3.0 |
| ERGOM CGT port | 1.0 | respiration 0.01 /d | - | mortality 0.03 /d | - |

Reading (Own judgement):
- The models that resolve copepods send 0.2-0.4 of intake to NH4 at once, plus 0.1 to DOM in ERSEM. Over a short time their total in-layer return is 0.3-0.5. Their POM is effectively exported from a 6-17 m layer.
- A North Sea model gets its recycling from microzooplankton, bacteria and DOM; none of them returns more than about 0.5 from the mesozooplankton.
- POLCOMS-ERSEM recycles 0.86-0.90 of uptake over the whole column (Holt 2012 [F]).
- No model paper gives a North Sea, Skagerrak or Kattegat f-ratio, or a micro / meso grazing split [N].

---

## 8. Recommendation (ranked; parameter-only first)

Each Δr is a static estimate (Own calculation, section 1) on the current partition, unless the change itself moves the partition. PP = 38/(1 - r). Base: r 0.445, PP 68, f 0.55.

### A. Parameter-only (library or manifest; no new state, no observation change, checkpoints stay valid; the zooplankton should be retrained to judge)

1. **`grazing_regeneration` 0.3 -> 0.5 on `phytoplankton` and `subsurface_phytoplankton`.**
   - Range: 0.4-0.6. Evidence: moderate.
   - Basis: the N of the copepod's respired share (0.32 at the model's own AE 0.65 and GGE about 0.33; NH4 excretion 0.21-0.50, Kiørboe [F]) plus the retained share of the egesta (0.35 x about 0.5; section 3 [F]/[A]).
   - Cross-check: total dissolved release with urea and sloppy feeding is 0.49-0.58 (Saba [F]). At 0.5, the copepod's N still balances: about 0.5 returned + 0.33 growth + 0.17 exported egesta.
   - Do not go above about 0.6 unless the copepod GGE in the model falls.
   - Effect: Δr about +0.16, PP about 97, f about 0.39 (Own calculation).
   - Caveat: in summer the copepods feed mostly in the subsurface layer, so most of this returns there. On its own, it may not fix the 6-8x low summer surface carbon.
   - Spring risk: a uniform 0.5 also raises the return in April-May, when copepods eat most and the model's spring is already 1.25-1.78x A17 and early (section 204.2). Earlier "more return" tests smeared the bloom (r(P,chl) fell) and over-filled the summer nitrate: the instant / share-1 rows of 179.3 and the share 0.7 / 1.0 rows of 201.5. The "unchanged biomass" condition is most at risk in spring, not summer. If that happens, item 10 (low return in the calanoid spring, high in summer) is what makes item 1 acceptable, not an optional refinement.
2. **Keep the microzooplankton `food_threshold` at 1.2** (12 ug C/l). Lab 6-20 [F]; ERSEM 15.06 12 [F]. Do not lower it: three tests (179.3, 201.4, 203.2) showed it only moves grazing into the spring.
3. **Keep the microzooplankton M1 at 0.0125 /tick (0.05 /d)**, and do not make it lower at low food. Starvation losses in the literature are 0.065-0.15 /d for heterotrophic dinoflagellates and 1-2 /d for oligotrichs [F]. Evidence: moderate.
4. **Re-test the microzooplankton half-saturation 7 -> 3-4 t/km2 at 10 m** (30-40 ug C/l; field about 30, range 10-80, microzooplankton.md).
   - Evidence: moderate for the value. The earlier over-grazing of the bloom was in a different world.
   - Effect: lowers the summer break-even from about 12 to about 7 ug C/l (section 6.5).
   - Judge it together with item 1: the goal is microzooplankton surviving May-December and taking a larger share of PP, not a stronger spring.
5. **Give the microzooplankton a warmer temperature table** (manifest `group_layers`). For example, a 0-30 m layer table from A17 [D], or 0-10 m in summer.
   - Q10 2.6 at 16-17 C instead of 12-13 C raises the grazing rate about 1.5x in July-August.
   - Caveat: one temperature serves its grazing in both layers, and the subsurface layer is 4-6 C colder in summer. A 0-30 m mean is the compromise. Evidence: weak to moderate.
6. **Zooplankton `mortality_regeneration` 0.5 -> 0.6-0.7.**
   - M1 is mostly predation by unmodelled carnivores (jellyfish, chaetognaths, fish larvae), whose excretion stays in the layer. ERSEM pe_R1 0.5 counts only the dissolved part of dead matter.
   - Evidence: weak (Own judgement). Effect: Δr about +0.02-0.04.
7. **Not recommended:**
   - Lowering k_N: f is not kinetic in a one-pool model (section 5.3; 203.2: +1 g C).
   - Raising the microzooplankton returns (0.70 is already at the generous end, section 4).
   - Putting `mortality_regeneration` on the producers while their M1 stands for sinking.

**Combined estimate:** if items 1 and 4-5 bring microzooplankton grazing to its literature share (x_m about 0.5, replacing copepod grazing), Δr is 0.43 x (0.77 - 0.63) = +0.06 on top of item 1. Total r about 0.66-0.67, PP about 110-115, f about 0.34. Without item 1, the same shift alone gives +0.15. (Own calculation)

### B. New mechanisms, no new FG (engine work; checkpoints stay valid if no observation changes)

8. **Mixed-layer producer M1 (sinking) into the subsurface layer**, as subsurface biomass or into the subsurface pool, instead of export.
   - Cells sinking at 0.3-1 m/d spend weeks in the 6-30 m layer. Kattegat traps show N recycled preferentially [A].
   - Effect: Δr about +0.03 (x about 0.034 for the mixed-layer producer). Evidence: moderate.
9. **Edibility on the DM pair `zooplankton_preys_on_phytoplankton`.**
   - The section-182 term e = floor + (1 - floor) N/(N + K) exists only for NDM grazing today (`interactions._edibility`).
   - Reason: at low nitrate the surface community is small cells that copepods eat poorly and protists eat well (section 6.1 [F]).
   - It would let the summer surface stock rise and feed the microzooplankton without a new FG.
   - Values: none sourced; floor 0.2-0.4 and K about 1 umol/l are only a starting guess (Own judgement). Evidence for the direction: moderate. For the values: none.
10. **Seasonal or layer-routed pellet retention**, instead of folding it into item 1. Two parts:
    - A share of DM egesta goes to the subsurface pool, with an escape fraction s/(s + r_deg) per layer.
    - The degradation rate r_deg follows the season: 0.5 /d in late winter to 2.5 /d in spring (Øresund [F]).
    - This gives the calanoid spring a low return and the summer a high one. Worth doing only if item 1 smears the bloom (cf. 179.3).

### C. New functional groups (invalidate every policy checkpoint)

11. **A small-cell (pico/nano) producer** in the mixed layer: low k_N (0.05-0.1 umol/l for NH4, Eppley 1969 [A]; NEMURO 0.1 [F]); grazed by microzooplankton, hardly at all by copepods.
    - This is the community the literature describes: 30-85 % of summer biomass, up to 85 % of PP [F].
    - It is the structural fix if items 1-9 cannot hold 30-50 ug C/l at the surface in summer.
12. **A separate bacteria pool or heterotrophic nanoflagellates: not needed.**
    - Bacteria are a net NH4 consumer and a carbon sink, with growth yield about 0.2 [A]. Their net effect is already inside the return shares.
    - Section 201.5 showed that an explicit DOM / detritus pool behaves like an instant return.
13. **Mixotrophic ciliates (Mesodinium)** as a separate FG or a photosynthetic subsidy on the microzooplankton.
    - Evidence for importance: strong for the Kattegat spring [F]; none for the Skagerrak summer [N].
    - Lowest priority.

### Bottom line (Own judgement)
- **Parameter-only changes plausibly reach f about 0.35-0.4 and PP about 95-115.** Most of that comes from item 1; the rest depends on the microzooplankton surviving the summer (items 4-5).
- **f 0.28-0.32 and PP 120-135** need item 8 and probably item 9: two engine changes, no new FG.
- **150 needs every lever at the upper end of its range.** It is not required, because 120-135 lies inside the 120-200 target band.
- **If the summer surface carbon stays far below A17 after items 1, 4, 5, 8 and 9, the missing piece is the small-cell producer (item 11).** It is a new FG and invalidates every checkpoint.
- **Score the result against DIN, not nitrate** (section 5.2). The model's summer mixed-layer pool should sit at about 0.1-0.25 umol/l.

---

## 9. Not found / not accessed [N]
- **Copepod excretion:**
  - Steinberg & Saba 2008 chapter.
  - Møller 2005, 2007.
  - Miller & Glibert 1998 and Checkley (originals).
  - Gaard, Daro, Gentsch 2009 (regional copepod excretion).
  - Ikeda et al. 2001 regression coefficients.
- **Pellets:**
  - Full text of Lundsgaard & Olesen 1997, and why it disagrees with Olesen & Lundsgaard 1995 (pellets < 5 % against 11 g C m-2).
  - Viitasalo et al. 1999 (original).
  - Wexels Riser 2007/2008.
  - Any open-Skagerrak sediment trap study.
- **Regeneration and f-ratio:**
  - Any Kattegat or Skagerrak 15N isotope-dilution regeneration study.
  - Rees et al. 2002 (North Sea f by size).
  - Maguer 1999 and Glibert 1982 primary numbers.
  - Sahlsten 1988 full text (the "25 % new" figure).
  - Litchman 2007 K-size exponent.
- **Summer community:**
  - A Kattegat or Skagerrak summer mixotroph share.
  - A size-fractionated summer PP share.
  - Montagnes et al. 2008 threshold review.
- **Models:**
  - The original Fasham 1990 parameters other than K and psi.
  - ECOHAM, Moll 1998.
  - NORWECOM rates.
  - Any North Sea model f-ratio or micro / meso grazing split.
