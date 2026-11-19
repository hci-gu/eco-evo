# Item 1 - full texts behind Ke (working table)

| # | source | what the number is | value | read | tag |
|---|---|---|---|---|---|
| 1a | Aksnes & Utne 1997, Sarsia 82:137-147, p. 142-143, Fig. 5 (UiB lab copy) | Ke fitted to Anthony's (1981) cod contrast thresholds, one parameter set over 1e-7 to 1e-1 W m-2 sr-1 | "gave an average T2 = 3.19 10-2 and Ke = 2.9 10-6 W m-2 sr-1. As demonstrated in Fig. 5, use of a single parameter set seems inconsistent with the observations." Fig. 5 broken line = "increased contrast sensitivity (by lowering T3 and Ke) at low irradiance levels" (value not printed) | full text | [F] |
| 1b | same, Fig. 5 digitised here (scripts/digitise_anthony_fig5.py) | cod contrast threshold Ct per background radiance | level means 0.46 / 0.22 / 0.094 / 0.054 / 0.046 / 0.033 / 0.023 at log L -7.1 / -6.3 / -5.5 / -4.4 / -3.3 / -2.1 / -1.0 | figure | [D] |
| 1c | fits to 1b (scripts/fit_cod_ke.py) | Ke of Ct = T3 (Ke + L)/L, i.e. half-saturation of R^2 (R^2 ~ 1/Ct in the A&U model, c = 0) | all levels 1.3e-6; scotopic (<= -5.5) 3.3e-7; photopic (>= -5.5) 5.8e-6; photopic (>= -4.4) 3.2e-5 W m-2 sr-1 | - | [D] |
| 1d | Fiksen et al. 1998, Fish Oceanogr 7:355 (UiB lab copy), Models section | Ke used for LARVAL cod and herring | "The parameter Ke was assigned the same value (5 uE m-2 s-1; Aksnes and Utne, 1997) as found earlier for gobies (Gobiusculus flavescens)" - borrowed, not fitted to cod; only E'(l) was fitted (optomotor angle) | full text | [F] |
| 1e | Aksnes & Utne 1997 Fig. 2 | goby (shallow, photopic) Ke | Ke = 5 uE m-2 s-1 (~270 lx at 54 lx/umol) | full text | [F] (non-gadoid) |
| 1f | Fiksen et al. 2002, Hydrobiologia 484:49 | planktivore predator in the same model | uses eq. with Ke, value not printed; "E' with a constant (10^4)" | full text | [F] |
| 1g | Meager et al. 2010 | juvenile cod R vs light, prey = mysid Praunus neglectus, 5 levels, PFD | full text closed (Unpaywall closed, no repository copy; USC record links publisher only); prey and design from Rohan et al. NOAA manuscript Table B3 | abstract + secondary | [A]/[S] |
| 1h | Utne-Palm et al. 2018 Table 1 | "Possible light threshold of cod" 0.01-0.001 umol m-2 s-1, "Behav. resp. to trawl", ref. Blaxter J. 1970. Light, animals, fishes. Marine Ecology 1(1):213-320 (Kinne ed., book chapter; not online) | as quoted | secondary | [S] |
| 1i | Anthony 1981 J Fish Biol 19:87 | method: cod trained to respond to a projected spot pattern, 7 background levels, min contrast ~2.0 %, two populations differed | abstract (scilit) | [A] |
