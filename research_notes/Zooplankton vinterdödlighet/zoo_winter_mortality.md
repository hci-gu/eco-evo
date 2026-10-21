# Winter mortality of mesozooplankton: literature for the Å17 zooplankton M1

Tags: [F] field, [L] lab, [D] derived by me. Access: **FT** = full text read, **ABS** = abstract or search snippet only, **SRC** = model source code or parameter file read.
Model reference: M1 = 19.7 /yr = 0.054 /d. Å17 temperatures: 5.3 C (Feb-Mar), 13.8 C (Sep). I assume an annual mean of about 9.5 C; recompute it from the monthly table.

## Table of values

| # | Quantity | Value | Tag | Source (access) |
|---|---|---|---|---|
| 1 | Post-hatch mortality, broadcast spawners | ln b = 0.0725 T - 3.415 (r2 0.72, n 885); eggs ln b = 0.0725 T - 1.112 | [F]* | Hirst & Kiørboe 2002, MEPS 230:195-209, Fig. 4a (FT) |
| 2 | Mortality, sac spawners (all stages) | ln b = 0.0707 T - 3.157 (r2 0.87, n 166) | [F]* | same, Fig. 4b (FT) |
| 3 | Q10 of field mortality | 2.06 (broadcast), 2.03 (sac). Authors: "approximately double with a 10 C increase". Field data are corrected with Q10 = 2.0 | [F]* | same, text (FT) |
| 4 | Rows 1-2 at Å17 temperatures | broadcast 0.048 /d (5.3 C), 0.089 /d (13.8 C); sac 0.062 /d, 0.113 /d | [D] | from rows 1-2 |
| 5 | Body size | mortality falls with body weight in broadcast spawners, is size-invariant in sac spawners | [F]* | H&K 2002 (FT) |
| 6 | Predation share of adult mortality | about 2/3 to 3/4, independent of temperature. Non-predatory share 1/4 to 1/3 | [F]/[L] | H&K 2002 (FT) |
| 7 | Predation-free (lab) adult mortality | 0.019 /d at 5 C, 0.065 /d at 25 C (implied Q10 about 1.85 [D]). Field total 0.062 /d at 5 C, 0.190 /d at 25 C | [L]/[F]* | H&K 2002, Fig. 7 text (FT) |
| 8 | C. helgolandicus CV-adult female, L4 (English Channel, 50 m, about 7.5-17 C) | ln m = 0.092 T - 3.581, so Q10 2.5 [D]; 0.045 /d at 5.3 C, 0.099 /d at 13.8 C [D]. CV-adult male: ln m = 0.112 T - 3.001. Temperature explained r2 0.65, chaetognaths 0.28, siphonophores 0.13 | [F] | Hirst, Bonnet & Harris 2007, MEPS 340:189-205 (FT) |
| 9 | Fladen Ground (N North Sea), spring bloom | Pseudocalanus elongatus: N1-N2 0.11 /d, N3-N5 <0.03, C1 about 0.10 (doubtful), C2-C4 about 0, C5 0.05 /d. Oithona similis: 0.03-0.07 /d in N1-N2, then <=0.001 /d. Survival N1 to end of C5: C. finmarchicus 4.3 %, P. elongatus 10.3 %, O. similis 44 % | [F] | Eiane & Ohman 2004, MEPS 268:183-193 (FT) |
| 10 | C. finmarchicus at 5 N Atlantic sites (spring-summer only) | rates and stage patterns differ between regions; density-dependent early-stage mortality (egg cannibalism). No winter rates | [F] | Ohman et al. 2004, ICES JMS 61:687-697 (ABS) |
| 11 | Diapause respiration, C. finmarchicus | 10-20 % of active conspecifics; swimming activity down 88.5 ± 3.4 % | [F]/[L] | Grigor et al. 2022, Front. Mar. Sci. 9:909528, citing Ingvarsdóttir et al. 1999 (ABS, secondary) |
| 12 | Diapause depth | below 600 m in the open ocean; in fjords under 200 m the animals stay active but rest in mid-winter | [F] | Coguiec et al. 2023, MEPS 712:49-65 (ABS) |
| 13 | Oithona similis in winter (N Atlantic) | egg production 1.60 ± 0.15 eggs/female/d in winter, 1.61 in summer, 2.13 in spring. Gross growth efficiency >100 % in winter, which suggests carnivory and/or euphausiid faecal pellets | [F] | Castellani et al. 2005, MEPS 288:173-182 (ABS) |
| 14 | O. similis, W Greenland winter | ciliates preferred; high egg production and efficiency in food-limited winter | [F] | Zamora-Terol et al. 2013, MEPS 483:85-102 (ABS) |
| 15 | Mnemiopsis, Gullmar Fjord | 8-yr study: in ctenophore years copepods "nearly disappeared" in **autumn**, with a trophic cascade to chlorophyll | [F] | Tiselius & Møller, ICES ZP6 2016 slides (ABS, not peer-reviewed) |
| 16 | Aurelia aurita, Gullmar Fjord | scyphistoma growth Aug-Sep, strobilation in Oct; ephyrae abundant in late autumn (100x the winter-spring level); some overwinter | [F] | Gröndahl 1988, Mar. Biol.; Hernroth & Gröndahl 1983, Ophelia (ABS) |
| 17 | NEMURO zooplankton mortality | quadratic in biomass, `MorZ0 * exp(KMorZ * T) * Z^2`, with MorZ0 = 0.0585 (mmol N m-3)-1 d-1 at 0 C and KMorZ = 0.0693 /C (Q10 = 2.0) for small, large and predatory zooplankton | model | Kishi et al. 2007, Ecol. Model. 202:12-25; ROMS `nemuro.in` (SRC) |
| 18 | ERSEM mesozooplankton | basal mortality sd = 0.05 /d, **not** temperature-scaled (`rd = sd*cP`; only uptake and rest respiration get Q10 = 2, reference 10 C). When depth-integrated prey < 300 mg C m-2 it switches to an **overwintering** state: mortality 0.0025 /d and respiration 0.0025 /d, 20x lower | model | Butenschön et al. 2016, GMD 9:1293; pmlmodelling/ersem `src/mesozooplankton.F90` and `testcases/ersem-15.06/...L4-ben-docdyn-iop.yaml` (SRC) |
| 19 | ECOSMO | form of mesozooplankton mortality not verified | - | Daewel & Schrum 2013 (not read) |

\*H&K's "predicted" rates are inferred from field fecundity and development times (life-history steady state), not measured directly. They agree with the direct field estimates they compile.

### Derived checks [D]
- The model M1 (0.054 /d) is about 3x H&K's predation-free floor at 5 C (0.019 /d). It roughly equals their **total** field mortality at 5 C (0.062 /d), predation included, and herring predation is added on top of it.
- Zero-growth decay over 90 winter days: at 0.054 /d, exp(-4.9) leaves 0.8 %. A Q10 = 2 M1 with T_ref = 9.5 C gives 0.040 /d in Feb (14.7 /yr) and 0.073 /d in Sep (26.5 /yr), and still leaves only exp(-3.6) = 2.7 % after 90 d.
- Å17 Nov to Jan (3.5 to 1.2 t/km2 over about 61 d) implies a net loss of about 0.018 /d, and about 0 from Dec to Jan.

## Answers by question

**Q2 (unmodelled predators).** Verified for the Gullmar Fjord: Mnemiopsis predation is an autumn phenomenon (row 15), and Aurelia ephyrae are produced in October, with medusae from spring to summer (row 16). I could not verify month-by-month data for Pleurobrachia, chaetognaths or fish larvae here. The general pattern is ctenophores in summer-autumn, chaetognaths peaking in autumn, and herring and sprat larvae in spring. Two consequences: predation pressure on copepods peaks from Aug to Nov, and from Dec to Mar it is low.

**Q3 (overwintering).** The diapausing C. finmarchicus stock sits at 200-700 m in the deep Skagerrak basin and does not appear in a 0-25 m sample. So the observed 1.2 t/km2 winter stock at Å17 already **is** the active small-copepod population (Pseudocalanus, Oithona, Acartia, Temora). I found no source for the Skagerrak diapause biomass share. Diapause physiology (10-20 % respiration, row 11) matters only if the model group is meant to include the deep stock, and in a 0-50 m model it is not. The right comparison is the low mortality of the small species (row 9: Oithona about 0.001 /d after N2, Pseudocalanus copepodids 0-0.05 /d).

**Q4 (winter food).** Small copepods keep feeding and reproducing in winter, on ciliates and heterotrophic dinoflagellates and probably on faecal pellets and detritus (rows 13-14). I did not find a verified source for Oithona feeding on marine snow (often cited to Kiørboe and colleagues, but unchecked). The model's zooplankton already grazes microzooplankton (section 177), so the real mechanism exists in the code.

## Recommendation

A Q10-scaled M1 is **defensible**. The field literature gives a consistent Q10 of about 2 for total copepod mortality (2.03-2.06 globally, 2.5 for C. helgolandicus at L4), and a Q10 of about 1.85 for the predation-free component. NEMURO uses the same form with a Q10 of 2.0. Use **Q10 = 2.0**, and set **T_ref to the annual mean of the 0-50 m Å17 temperature (about 9.5 C)**, not H&K's 15 C. Otherwise the annual-mean M1 that the growth budget was calibrated on shifts silently. Because exp(kT) is convex, the annual mean of M1 rises slightly, by about 2 % for a ±4.25 C seasonal cycle [D]; renormalise if exactness matters.

**However, Q10 alone does not reproduce the Å17 winter stock.** It cuts winter M1 by only about 25 % (0.054 to 0.040 /d), while the observations imply a net loss of 0.02 /d or less. The bigger lever is that M1 mostly stands for **unmodelled predators**, which are strongly seasonal (autumn ctenophores, spring fish larvae) and largely absent from Dec to Mar. H&K's "predation share is constant with temperature" is a cross-site result, not a seasonal one. Two literature-backed options:

- **(a) Split M1.** Use a non-predatory part, about 0.02 /d at 5 C with Q10 about 2 (H&K lab longevity), plus an unmodelled-predator part with a seasonal or temperature factor steeper than Q10 = 2.
- **(b) ERSEM-style state switch.** When food (or day length) falls below a threshold, mortality and respiration drop to a low overwintering rate (ERSEM: 0.0025 /d each, threshold 300 mg C m-2 of prey).

Option (a) fits the project's "biology drives" convention better. In either case, check that zooplankton has non-phytoplankton winter food (microzooplankton) with which to offset the remaining loss.
