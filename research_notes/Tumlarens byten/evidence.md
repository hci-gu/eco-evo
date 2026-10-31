# Porpoise prey pairs: evidence (2026-10-31)

Research only; nothing in the library, manifest, code or tests was changed. Tags: [F] field / stated in a
source (full text or table read), [A] abstract only, [S] secondary or snippet, [D] derived here, [J]
judgement. Scripts in `scripts/`, their stdout in `tables/`, downloads in `data/` (`data/README.md`).

## 1. How the library's porpoise attack rates came about (the derivation chain)

There is no encounter derivation of any porpoise a. Every value is the end of a chain of fits:

| Step | Section | What happened | Basis |
|---|---|---|---|
| 1 | 151 (2026-10-10) | a_herring 0.035 -> 0.015, a_gadoids 0.035 -> 0.2 (ratio 13) | fitted on frozen fix51 policies to an Ecopath 1991 diet (49 % gadoids); the resume says "The a ratio 13 is a fit to diet data, not a mechanism: nothing says porpoises catch gadoids 13x more easily; it is needed because herring saturate the porpoise and porpoises have learned to sit on herring" |
| 2 | 206 (2026-10-24) | both x1.5: 0.0225 / 0.3 | porpoises fell to 0.27-0.37 of start in frozen 15-50 year runs; x1.5 held them at ~1.0 (model output) |
| 3 | 209 (2026-10-25) | gadoid 0.3 -> benthivores 0.12, predators 0.20; planktivores 0.0225 kept | "a x B invariance" (old gadoid pool ~4x too small) then split by a per-tonne selectivity, with Ransijn et al. 2021 read as "whiting ~1.2x and cod ~0.6x herring per unit availability" |
| 4 | 223.26 | predators split; a 0.2 copied onto whiting and onto the water column | J |
| 5 | 228 / 228.3 | mackerel left the water column; porpoise a 0.2 kept on the rest (saithe 0.80, spurdog, hake, anglerfish, ling, velvet belly, pollack); mackerel pair unchecked | J |
| 6 | 223.32 / 230 | stage access benthivores [1, 0.55], whiting [1, 0.95] (porpoise window 0-40 cm) | D (size windows) |

[F: mareld_resume.txt 151.3, 206.3-206.4, 209.3; params_top_predators.md 3.2; library comments.]

Internal inconsistency in step 3 [D]: params_top_predators.md 3.2 derived the field per-tonne selectivity
relative to planktivores as benthivores 0.50 and predators 0.86, but the a values it then set are 5.3x
(0.12 / 0.0225) and 8.9x (0.2 / 0.0225) the planktivore a. The a x B rescale carried the step-1 fit ratio
(13, from a world with ~4x too few gadoids) forward; the selectivity it quoted would have given
a_ben = 0.0225 x 0.50 / 0.62 = 0.018 and a_pred = 0.0225 x 0.86 = 0.019.

## 2. Ransijn et al. 2021 cannot be read as per-tonne selectivity

Ransijn et al. 2021 (Ecol. Evol. 11: 17458, doi 10.1002/ece3.8380) [F, full text,
`../Fiskgrupper i rutan/data/top_predators/ransijn2021_PMC8668753.xml`]:
- Availability = NS-IBTS BPUE of fish < 40 cm within a foraging buffer around each stranded animal
  (southern North Sea, Dutch coast, 2006-15).
- Eq. 7-9: "N_i = q_i x B_i ... defining constant a_i = alpha_i q_i^m ... from diet composition and BPUE
  data we can estimate the value of the a_i parameters without needing to correct for catchability."
- "relative prey abundance was rescaled so that the maximum observed value was 100 ... The estimated
  values of a_i are thus a measure of prey 'preference' or attack rate in relation to an index of
  abundance and not to absolute estimates of biomass."
- Best model Type III, m = 1.5 (DIC 97 202 vs 117 216). Relative a (sandeel = 1): sprat 0.238, whiting
  0.120, herring 0.101, other 0.089, cod 0.058.

So the a_i absorb each species' GOV catchability to the power 1.5, and possibly a per-species
rescaling. GOV under-samples herring 3-7x relative to acoustics (`../Ekolodsdata planktivorer`) and has
wing q ~1.7 for whiting, ~0.2 for cod (Fraser et al. 2007 via `../Fiskgrupper i rutan/gadoid_level.md`).
Read literally per true tonne, the whiting : herring ratio would be 1.19 x (q_h / q_w)^m, anywhere from
~1 to ~20 depending on q and m [D]. The numbers neither support nor refute any per-tonne ratio; 209's
"whiting 1.2x, cod 0.6x herring per unit availability" is a misreading. They are not used as a test here.
They still show the porpoise eating whiting and cod in proportion to their survey index at roughly herring's
rate, with a strong sandeel preference that has no counterpart in the box.

## 3. Field diets (Kattegat-Skagerrak)

Sveegaard 2011 PhD, Paper V, Table 2 [F, table read; `../Fiskgrupper i rutan/data/top_predators/sveegaard2011_phd.pdf`]:

| Study | Cod | Whiting | Gadidae spp. | Saithe | Norway pout | Herring | Sprat | Sandeels | Gobies | Eelpout | Pearlsides | Hagfish | Fish < 5 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Borjesson et al. 2003 %N | 0.6 | 17.4 | 9.0 | 2.0 | 8.2 | 8.3 | 8.6 | 5.6 | 19.0 | | 8.9 | 7.8 | 14.8 |
| Borjesson et al. 2003 %W | 5.5 | 5.4 | 6.8 | 5.0 | 4.1 | 56.2 | 9.0 | 1.8 | 0.8 | | 0.4 | 1.4 | 12.7 |
| Aarefjord et al. 1995 %W (est.) | 10.6 | 8.2 | 11.1 | | | 42.1 | 1.1 | 0.1 | 13.1 | 2.2 | | | 11.5 |

- Borjesson: 112 bycaught porpoises, Kattegat-Skagerrak 1988-96 [A]; Aarefjord: 39 porpoises, Kattegat 1985-90.
- Sveegaard: "Gadidae spp. that represent unspecified species of the cod family, here presumably mainly cod and whiting".
- Mean prey weight ratios from %W / %N (Borjesson) [D]: herring 6.8, saithe 2.5, cod 9.2, whiting 0.31,
  Norway pout 0.50 (units of the mean prey weight). Saithe eaten weigh ~0.37 of an eaten herring; with
  herring ~26 cm (~130 g, Borjesson mean via Sveegaard), saithe ~48 g, i.e. ~18 cm [D, rough: otolith
  counts and %W are not from the same digestion state].

On the model's four pools (gobies, sandeels, pearlsides, hagfish, eelpout and "fish < 5 %" dropped;
Gadidae spp. split g to cod, 1 - g to whiting, g 0.5 [J], 0.25-0.75) [D, `tables/clearance_test_stdout.txt`]:

| Diet | planktivores | benthivores | whiting | water column |
|---|---|---|---|---|
| booked (library M2 x B) | 0.648 | 0.150 | 0.130 | 0.073 |
| Borjesson 2003 | 0.753 | 0.097 (0.078-0.115) | 0.096 (0.077-0.114) | 0.054 |
| Aarefjord 1995 | 0.591 | 0.221 | 0.188 | 0 |

## 4. Prey densities in the stomach years (NS-IBTS 1991-96)

`scripts/then_now_densities.py` -> `tables/then_now_densities_stdout.txt` [D]. GOV hauls, Q1 and Q3
averaged, members weighted by today's pool composition; the box's acoustic herring level is scaled by GOV
herring's own change, never compared with GOV gadoids in absolute terms.

| Pool ratio 1991-96 / 2001-25 | Skagerrak, whole | Skagerrak, < 40 cm | IIIa | box rectangles |
|---|---|---|---|---|
| planktivores (GOV herring) | 1.22 | 1.22 | 1.03 | 1.05 |
| benthivores | 1.00 | 1.22 | 1.06 | 1.08 |
| whiting | 0.87 | 0.88 | 1.09 | 0.74 |
| water-column eaten taxa (saithe, hake, ling, pollack) | 0.30 | 0.84 | 0.31 | 0.21 |

- Norway pout was 4.5x today's level in 1991-96; GOV herring 0.97x (Skagerrak).
- Herring by stock assessment (ICES SAG 2025; `data/sag/`): western Baltic spring spawners SSB 1991-96 /
  2001-25 = 2.90, North Sea autumn spawners 0.58 [F/D]. The Skagerrak's summer adults are largely WBSS,
  its juveniles NSAS; a 50/50 mix gives herring x1.74 and the planktivore pool x1.82 [D, J on the mix].
- Saithe < 40 cm was 0.29 of the Skagerrak GOV saithe biomass in 1991-96 against 0.14 in 2001-20, and
  < 30 cm 0.08 vs 0.02 [D]. Saithe < 20 cm, the size the stomachs suggest, is 0.1-0.4 % of GOV saithe
  biomass in either period: GOV does not sample the coastal 0- and 1-group saithe.

So for planktivores, benthivores and whiting the 1990s prey field was within ~0.7-1.8x of today's box
densities; the clearance ratios below are robust to the decade. The water column changed most
(adult saithe rose ~3-7x since the 1990s).

## 5. The water-column rest and the porpoise size window

`scripts/wc_window.py` -> `tables/wc_window_stdout.txt` [D]. Box long-run (2001-25) stratified GOV length
structure (`../Fiskgrupper i rutan/predators/tables/pred_length_biomass.csv`), member weights of section 228.

| Member (weight) | < 30 cm | < 40 cm | < 45 cm |
|---|---|---|---|
| saithe (0.797) | 0.016 | 0.049 | 0.169 |
| spurdog (0.097) | 0.001 | 0.006 | 0.012 |
| hake (0.048) | 0.097 | 0.299 | 0.433 |
| anglerfish (0.027) | 0.002 | 0.021 | 0.039 |
| ling (0.014) | 0.001 | 0.005 | 0.010 |
| velvet belly (0.009) | 0.031 | 0.463 | 0.877 |
| pollack (0.008) | 0.000 | 0.016 | 0.043 |
| **group f (eaten taxa only)** | **0.017** | **0.054** | **0.156** |

- Sharks and anglerfish never appear in any porpoise stomach set read here, so they count 0.
- The accessible part is ~0.73 saithe, 0.27 hake; 0.033 t/km2 (119 t on the grid) at < 40 cm.
- Unstratified box-rectangle means for 2001-20 give saithe < 40 cm 0.23, hake 0.25 -> f 0.19: the 2021-25
  saithe peak was large fish. Range of f: 0.017 (< 30 cm) - 0.19 (2001-20 box rectangles, < 40 cm).
- The ramp selectivity (1 below 30 cm, linear to 0 at 50 cm) gives f 0.10 [D, J on the ramp].

## 6. Literature on depth use, bottom feeding and prey size (literature subagent, 2026-10-31)

Downloads in `data/lit/` (sources in the table at the end). Grades as above.

Depth use:
- Teilmann, Larsen & Desportes 2007, J Cetacean Res Manage 9: 201-210 [F]: 14 satellite-dive-recorder
  porpoises from Belt-sea pound nets. "Maximum dive depth recorded was 132m from animals moving north into
  Skagerrak"; daily maximum depths "6-132m, with median and mean values of 26m and 30m"; 68 % of time at
  0-5 m, 2 % below 20 m (Belt seas, < 50 m water).
- van Beest et al. 2018, Mar Biol 165: 95 [F]: 6 porpoises, 3 from northern Jutland (Skagerrak); dive depth
  "mean = 15.5 m ... max = 151.5 m"; "Some individuals consistently dove down to the seabed which is
  indicative of a hunting strategy focussed on demersal fish species".
- Westgate et al. 1995, Can J Fish Aquat Sci 52: 1064 [S]: maxima 83-226 m, mean depths 14-41 m.
- Chambault et al. 2024, Sci Rep [F]: West Greenland shelf, mean daily dive depth 64 m, frequent dives
  > 100 m; ~45 % "benthic" dives (within 70 m of the charted depth).
- Fontaine et al. 2007, BMC Ecol 7: 1 [F]: "It is currently assumed that harbour porpoises forage mainly
  within the first 200 m"; in deep Nordic waters "the higher bathymetry ... reduces the availability of
  demersal prey species".
- Wisniewska et al. 2016, Curr Biol 26: 1441 [F]: pelagic, near-surface and benthic foraging by day,
  "primarily pelagic dives at night"; up to 550 prey/h of 3-10 cm fish, near-continuous foraging (shallow
  Danish waters).
- Bottom prey in Kattegat-Skagerrak stomachs: hagfish (Borjesson 2003 abstract, "may be an important food
  resource, at least for adult porpoises") [A]; Esmark's eelpout in IIIa (Kinze 2003, NAMMCO SP 5, Table 5) [F].

Reading [J]: on the box's 100-300 m slope the bed is reachable (132-152 m recorded in the Skagerrak) but
not routinely; the coastal stomach sets come from animals feeding in shallower water. A depth factor on the
two bed pairs in the box is therefore <= 1, plausibly 0.3-1; no source gives a number. It is not applied
here; it would lower the benthivore and whiting pairs further.

Prey size and saithe:
- Santos & Pierce 2003, OMBAR 41: 355 [F]: "mainly small schooling fish < 400 mm long, indeed in most cases
  < 300 mm"; whiting eaten ~23 cm, "probably younger than 2-yr old".
- Borjesson et al. 2003 [A; full text not reachable]: herring "the dominating prey species for both
  juveniles and adults", eaten in "size classes containing mature or maturing individuals". Cod 28.1 +/-
  5.8 cm, "majority ... below 30 cm" (via Wallin 2017) [S]. No saithe lengths found.
- Aarefjord et al. 1995 via Santos & Pierce 2003 and Bjorge 2003 [S/F secondary]: saithe, blue whiting and
  capelin frequent in Norwegian-coast porpoises; North Sea / Skagerrak top prey herring, gobies, sandeels,
  sprat, whiting, cod - saithe not among them. Pelagic prey more important in Norway, benthic prey in
  Danish and Swedish waters.
- Saint-Andre 2019, UiT MSc [S]: 134 bycaught porpoises, northern Norway; "saithe (juvenile) was by far the
  most important prey species".
- Leopold 2015, PhD Wageningen [F]: no diet-vs-survey selectivity estimate; starving animals eat gadoids and
  gobies, well-fed ones more clupeids and sandeels.
- No study estimates porpoise prey preference per catchability-corrected tonne.

Reading [D/J]: every saithe record with a size or age is juvenile, and the Borjesson %W/%N ratio puts the
Kattegat-Skagerrak saithe at ~18 cm. GOV holds 0.1-0.4 % of its Skagerrak saithe biomass below 20 cm and
the box ~1.6 % below 30 cm. The stomach saithe are coastal 0-/1-group fish that the offshore box does not
contain at that size.

| File | Source |
|---|---|
| teilmann2007.pdf | https://journal.iwc.int/index.php/jcrm/article/download/668/393 |
| wisniewska2016_accepted.pdf | https://research-repository.st-andrews.ac.uk/bitstream/handle/10023/10866/ultra_high_foraging_rates_final.pdf |
| wisniewska2016_proof.pdf | https://ecos.au.dk/fileadmin/user_upload/CURBIO12775_proof.pdf |
| ransijn2021.pdf | https://research-repository.st-andrews.ac.uk/bitstream/10023/24441/1/Ransijn_2021_Integrating_disparate_datasets_ECE_8380_CCBY.pdf |
| leopold2015_thesis.pdf | https://edepot.wur.nl/361729 |
| santos_pierce2003.pdf | PCA exhibit R-0047 (copy of OMBAR 41) |
| kinze2003_nammco.pdf | https://septentrio.uit.no/index.php/NAMMCOSP/article/download/2745/2594 |
| bjorge2003_nammco.pdf | https://septentrio.uit.no/index.php/NAMMCOSP/article/download/2749/2598 |
| wallin2017_slu.pdf | https://stud.epsilon.slu.se/10643/ |
| vanbeest2018.xml, fontaine2007.xml, chambault2024.xml | Europe PMC fullTextXML PMC5924767, PMC1781931, PMC10873309 |

## 7. Clearance test results

See `tables/clearance_test_stdout.txt` and the summary. Field clearance per whole-group tonne relative to
planktivores (range over Borjesson g 0.25-0.75 and Aarefjord, at today's and at 1990s densities):
benthivores 0.23-1.49, whiting 0.77-4.98, water column 0-2.4 (0.39-2.4 where saithe was eaten). Library
a x f relative: 3.29, 8.51, 8.89. Implied a with a_plk 0.0225 kept (geometric mean of five central cases,
booked plus Borjesson and Aarefjord at both 1990s herring readings): benthivores 0.024 (0.008-0.054),
whiting 0.054 (0.018-0.12).
