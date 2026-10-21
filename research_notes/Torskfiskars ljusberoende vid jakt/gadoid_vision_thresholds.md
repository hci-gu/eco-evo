# Gadoid prey detection vs light: sensory basis, thresholds and translation to sun elevation (eastern Skagerrak, 58 N)

Scope: Atlantic cod (*Gadus morhua*), whiting (*Merlangius merlangus*), saithe (*Pollachius virens*), haddock (*Melanogrammus aeglefinus*). Proxies are flagged: walleye pollock (*Gadus chalcogrammus* / *Theragra chalcogramma*), sablefish, larval and age-1 cod.
Model context (from the repo): `lib/world/daylight.py` uses `f(elev) = 1/(1+exp(-(elev-theta)/2))` (TRANSITION_DEG = 2, so 10 % -> 90 % detection spans about +-4.4 deg around theta), attack rate `a(t) = a * (rho + (1-rho) f)` normalised to the annual mean; manifest `mareld2.yaml` has `light_attenuation_per_m: 0.29` (used for producer growth over the mixed layer). The herring -> zooplankton pair uses `dark_ratio 0.1`, `light_threshold_deg -8.0`, derived in the library comment with Kd ~0.15 /m at 10-40 m.

---

## Q1. Lab and field evidence: foraging success / reactive distance / capture vs light, and residual non-visual feeding in darkness

### Takeaway
Gadoids are visual feeders that keep some capacity to detect and catch prey in total darkness (chemoreception, lateral line, taste/barbel). Juvenile cod still had long reactive distances at 0.01 umol m-2 s-1 (the lowest light tested), and larval cod attacked prey at ~3.7e-6 W m-2; predation on mobile prey is "significantly lower" in darkness, but no paper found gives a numeric dark/light ratio for adult gadoids. Field diel diets show piscivory concentrated at dusk/dawn, free-swimming prey eaten by day, and benthic prey eaten as much or more at night.

### Cited Findings
**Cod - laboratory**
- Meager et al. (2010): reactive distances of age-1 juvenile cod were measured over 0.01-64 umol m-2 s-1. "The reactive distance at 0.01 umol m-2 s-1 indicated high sensitivity to low light conditions. The reactive distance was less at intermediate light levels (1.5-6.5 umol m-2 s-1) and increased thereafter." Only a model with different parameter sets above and below 5 umol m-2 s-1 fitted, and validation against another dataset showed "the generality of the model was poor". The authors interpret it as a change in foraging behaviour at twilight light levels. — [Meager, Moberg, Strand & Utne-Palm 2010, Mar Freshw Behav Physiol 43:99-108](https://doi.org/10.1080/10236241003798910)
- Meager et al. (2005): juvenile cod feeding on mysids (*Praunus neglectus*): using chemical cues they located mysids "from significantly longer distances than when only visual cues were available"; turbidity (0.4-17 m-1 beam attenuation) had only a weak negative effect on visual reactive distance; "predation rates were significantly lower in dark conditions than in well lit conditions"; cod use chemoreception together with vision at close range. (Abstract gives no numeric dark/light ratio.) — [Meager, Solbakken, Utne-Palm & Oen 2005, CJFAS 62](https://doi.org/10.1139/f05-104)
- Vollset et al. (2011): larval cod tested at dusk ~1.36e-3 W m-2, night ~1.38e-4 W m-2 and darkness ~3.67e-6 W m-2. At 6 dph pursuit only at dusk; attacks under night conditions from 26 dph; at 53 dph cod "attacked prey in darkness", i.e. "larvae are still able to feed at light intensities of 3.67e-6 W/m2". — [Vollset, Folkvord & Browman 2011, Mar Biol](https://doi.org/10.1007/s00227-011-1635-5)
- Løkkeborg & Fernö (1999): field (acoustic telemetry) on cod; higher swimming speeds and range by day than night. Baits set by day vs night: more fish localized the bait during the high-activity (day) period (70 % vs 45 % of observations) and time to localization was 50 % shorter; but olfactory response threshold showed no diel difference (rheotactic response distances 240 vs 227 m). Authors: taste/olfactory detection is "relatively independent of light", visual detection distance and capture of active prey depend on light. — [Løkkeborg & Fernö 1999, Env Biol Fish 54](https://doi.org/10.1023/A:1007504712163) (numbers as reported in the abstract quoted by search; full text not accessed)
- Utne-Palm et al. (2018), Table 1, compile cod visual data: lambda_max 446 and 517 nm (microspectrophotometry, Bowmaker 1990), 450/490 nm (conditioning, Anthony & Hawkins 1983); change from scotopic to photopic vision at 8 x 10^-6 W sr-1 m-2 (Anthony 1981); "possible light threshold of cod" 0.01-0.001 umol m-2 s-1 from behavioural response to trawls (Blaxter 1970). — [Utne-Palm et al. 2018, PLoS ONE 13:e0190918](https://doi.org/10.1371/journal.pone.0190918)
- Strand et al. (2020): lab tanks (Porsangerfjorden predators), "Sculpins and cod were most effective predators on 0-group cod and saithe at dusk/night/dawn"; sculpins alone under 24 h light caught zero 0-group fish, but 29 % under a day/night cycle. — [Strand, Pedersen, Christie & Moy 2020, JEMBE 151356](https://doi.org/10.1016/j.jembe.2020.151356)

**Walleye pollock (gadoid proxy) - laboratory**
- Ryer & Olla (1999): juvenile (0+) walleye pollock and sablefish feeding on *Artemia*: sigmoidal relation between log illumination and prey consumed; threshold below which fish no longer forage visually approximately 5 x 10^-7 uE m-2 s-1 for walleye pollock and 5 x 10^-5 uE m-2 s-1 for sablefish (exponents taken from a search-engine summary; the OCR'd abstract garbles them - VERIFY against the original); pollock "better adapted for visual feeding at depth or at night"; "both species were capable of detecting and capturing prey in darkness". — [Ryer & Olla 1999, MEPS 181:41-51](https://doi.org/10.3354/meps181041)

**Field diel feeding (cod, whiting, haddock)**
- Whiting, North Sea: "occurrence of bottom dwelling prey increased significantly during the night at 4 of 5 locations. In contrast, free swimming prey and prey migrating towards the demersal layer during the day were eaten mainly in the daylight hours." No diel pattern for larger fish prey (long digestion time). — [Rindorf 2003, MEPS 249:265](https://doi.org/10.3354/meps249265)
- Whiting feeding on sandeels (North Sea): single feeding peak with maximum 22:00-24:00 h, minimum 08:00-10:00 h. — [Mergardt & Temming 1997, ICES JMS](https://doi.org/10.1006/jmsc.1996.0190)
- Haddock (15-26 cm, SW Norway coastal bank, July): ate polychaetes and isopods "mainly in the early morning and from noon until a few hours after sunset"; poor-cod fed on decapods, isopods and polychaetes mainly at night. — [Albert 1995, ICES JMS 52](https://doi.org/10.1006/jmsc.1995.0084)
- Cod on sprat (Bornholm Basin): "Predation by cod took place primarily at dusk and dawn during ascent and descent of sprat associated with school dissolution and formation"; cod stayed near bottom outside these windows. — [Andersen, Lundgren, Neuenfeldt et al. 2017, MEPS](https://doi.org/10.3354/meps12319)
- Cod in a subarctic Greenland fjord: "Cod diets were dominated by Euphausiacea during night and capelin in the day, reflecting vertical overlap with prey." — [Jansen et al. 2025, Rev Fish Biol Fish](https://doi.org/10.1007/s11160-025-09989-w)
- Saithe: reduced activity at night (acoustic tags at fish farms; data storage tags off Iceland). — [Skilbrei & Otterå 2016, ICES JMS](https://doi.org/10.1093/icesjms/fsv261); [Armannsson & Jónsson 2012, ICES JMS](https://doi.org/10.1093/icesjms/fss121)
- Haddock use the chin barbel to detect buried benthic prey (worms, molluscs, brittle stars) in soft sediment. — [Wikipedia: Haddock](https://en.wikipedia.org/wiki/Haddock) (secondary, low quality; primary barbel taste-bud work not retrieved)

### Inferences
- Cod (and pollock as proxy) vision reaches into nautical twilight / moonlit-night levels: the 0.01 umol m-2 s-1 result (~0.35-1.5 lx, spectrum-dependent, see Q3) is NOT a threshold - reactive distance was already high there, so half-detection lies below it. The larval-cod darkness attack level (3.67e-6 W m-2) is of order 1e-3 lx, i.e. starlight; larvae are less sensitive than juveniles/adults, so adults should be at least as sensitive.
- A residual "dark" attack rate clearly exists for every gadoid (pollock in darkness, cod chemoreception, whiting/haddock night benthic feeding), so `dark_ratio` should be > the herring value of 0.1 for most gadoid pairs. No measured ratio exists in what was found; any number is an estimate.
- Prey type matters more than gadoid species: mobile/schooling prey -> crepuscular peak (dusk/dawn), consistent with a threshold in nautical/civil twilight and a low dark_ratio; benthic infauna/epifauna -> taste/olfaction dominated, little or no light dependence.

### Gaps
- No numeric dark/light capture ratio found for adult cod, whiting, saithe or haddock (Meager 2005 says "significantly lower" without a number in the abstract; full text not accessible).
- No laboratory light-threshold study found for whiting, saithe or haddock. Saithe: nothing on light thresholds - only activity patterns. Whiting/haddock evidence is diel stomach data only.
- Meager et al. 2010 fitted Ke/threshold parameters are not in the abstract; full text not accessible.
- Ryer & Olla (1999) exponents need verification against the original PDF (access blocked).
- The often-quoted "larval cod cease foraging below ~0.1-0.4 lx" came from a search summary without a traceable primary source; not used.

---

## Q2. Retinal / visual physiology (rods, cones, absolute and scotopic thresholds)

### Takeaway
Cod have a duplex retina with rhodopsin-based scotopic vision peaking at ~490 nm and dichromatic photopic vision (SWS2 blue + RH2 green opsins). The behavioural photopic/scotopic transition sits at a radiance of ~8e-6 W sr-1 m-2 (Anthony 1981), roughly late nautical twilight at the surface; minimum contrast threshold ~2 % in bright light. No absolute scotopic threshold in lux was found for adult cod or whiting.

### Cited Findings
- Cod contrast discrimination (cardiac conditioning) at 7 background levels: minimum detectable contrast fell with light to ~2.0 %; "The contrast threshold curve showed a discontinuity at a light level of approximately 8.0 x 10^-6 W sr-1 m-2", thought to be the photopic-scotopic change; differences between cod from two locations. — [Anthony 1981, J Fish Biol 19:87-103](https://doi.org/10.1111/j.1095-8649.1981.tb05814.x)
- Dark-adapted behavioural and ERG spectral sensitivity peak at 490 nm (rhodopsin); secondary behavioural peak at 550 nm suggesting yellow cones contribute to scotopic vision; rods and cones "both active over a wide range of light intensities". — [Anthony & Hawkins 1983, Mar Behav Physiol 10:145-166](https://doi.org/10.1080/10236248309378614)
- Cod colour vision driven by only two opsin subfamilies (SWS2 blue, RH2 green) — cited in [Utne-Palm et al. 2018](https://doi.org/10.1371/journal.pone.0190918) (Valen et al. 2014, PLoS ONE e115436).
- Cod "possible light threshold" 0.01-0.001 umol m-2 s-1 (Blaxter 1970, behavioural response to trawls), as tabulated in [Utne-Palm et al. 2018, Table 1](https://doi.org/10.1371/journal.pone.0190918).
- Krill (*Meganyctiphanes norvegica*) bioluminescence at 468 nm and visual peak 488-490 nm overlap cod sensitivity (446/517 nm pigments). — [Utne-Palm et al. 2018](https://doi.org/10.1371/journal.pone.0190918)

### Inferences
- Converting Anthony's radiance to irradiance needs assumptions (not a cited fact): for a roughly isotropic field E ~ pi*L = 2.5e-5 W m-2; at 490-500 nm that is ~1.0e-4 umol m-2 s-1 and ~0.004-0.006 lx (using the photopic V(lambda) at 490-500 nm; scotopic lux would be higher). Downwelling radiance is not isotropic, so this is order-of-magnitude only. It suggests the shift to rod vision - and the steep loss of acuity/contrast - happens around 1e-4 umol m-2 s-1 (~0.005 lx) at the fish's depth.
- Blaxter's 0.001-0.01 umol m-2 s-1 "threshold" (~0.035-1.5 lx depending on spectrum) and Anthony's ~1e-4 umol transition bracket the "half-detection" region for cod at ~1e-4 to 1e-2 umol m-2 s-1 (~0.005-0.5 lx). Confidence: low-moderate.
- Because cod rods peak at 490 nm and deep coastal water transmits best at ~550-570 nm (Q3), the "effective" attenuation for cod scotopic vision lies between Kd(490) and Kd(555).

### Gaps
- No absolute rod threshold (lux or quanta) found for adult cod, whiting, haddock or saithe; no retinal studies found for whiting/saithe/haddock.
- Blaxter (1970) primary text not retrieved; the 0.01-0.001 umol value is second-hand.

---

## Q3. Converting thresholds to sun elevation at 30-60 m in the eastern Skagerrak

### Takeaway
Surface illuminance falls ~0.5 decades per degree of solar depression in twilight (-3 to -11 deg), from ~400-600 lx at sunset to ~3 lx at -6 deg and ~0.01 lx at -11 deg, down to a ~0.001 lx starlight floor (moon up to ~0.2 lx). Skagerrak water: green/photopic attenuation ~0.18-0.30 m-1, blue (490 nm) ~0.29 m-1, PAR (0-Secchi) ~0.37-0.52 m-1 in the surface layer; deeper Atlantic water is clearer. With effective Kd 0.2 m-1 a gadoid half-detection light of 0.01-0.03 lx is reached at a sun elevation of about -6 to -5 deg at 30 m, -4 to -3 deg at 40 m, and -1.4 to +0.8 deg at 50 m. With Kd 0.29 through the whole column, gadoids at 50 m would be near-blind even at winter noon - an artefact, so Kd 0.29 is an upper bound, not a recommendation.

### Cited Findings
**Surface illuminance vs sun elevation**
- Twilight sky brightness (Koomen et al. 1952): "For H from about -3 deg to -11 deg the entire sky changed in brightness at about the same rate of a factor of 10 for each 2 deg change in H"; clearer air (Sacramento Peak) gave skies ~12-23 times dimmer than Maryland. — [Koomen, Lock, Packer, Scolnik, Tousey & Hulburt 1952, JOSA 42:353](https://opg.optica.org/josa/abstract.cfm?uri=josa-42-5-353)
- Civil twilight (0 > elev > -6), nautical (-6 to -12), astronomical (-12 to -18). "During a clear evening's civil twilight, horizontal illuminance decreases from ~585-410 lux to ~3.5-2 lux"; starlight ~0.001 lux, moonlight ~0.2 lux, sunlight up to 100,000 lux. — [Spitschan, Aguirre, Brainard & Sweeney 2016, Sci Rep 6:26756](https://doi.org/10.1038/srep26756) (lux values from the paper's introduction and search-snippet; Fig. 2 shows log irradiance vs elevation is sigmoidal, steepest in twilight)
- End of civil twilight ~3.2-3.4 lux; beginning ~330 lux. — [Visual Expert: twilight criterion (Owens, Francis & Leibowitz 1989)](https://www.visualexpert.com/Resources/twilightcriterion.html)
- Sunlight luminous efficacy: 1 W m-2 = 122 +- 1 lx (outdoor solar measurement); rule of thumb 120 lx per W m-2 (1 sun ~ 120,000 lx). — [Michael, Johnston & Moreno 2020, J Meas Eng 8(4)](https://www.extrica.com/article/21667/pdf)

**Attenuation in the Oslofjord-Skagerrak**
- Skagerrak (57-59 N) all-year white Secchi depth 8.3 +- 2.8 m (Outer Oslofjord: 4.3 m summer, 11.1 m winter). — [Aas, Høkedal & Sørensen 2014, Ocean Sci 10:177](https://doi.org/10.5194/os-10-177-2014), Table 1
- Mean attenuation of photopic luminance KL,phot = 0.26 +- 0.19 m-1 (32 stations, averaged over 0-Secchi depth); Kd/KL "usually is close to 1"; relationships to 1/Z_D: Kd(490) ~ 2.4/Z_D (mean 0.46 +- 0.39 m-1), Kd(555) ~ 1.5/Z_D (mean 0.30 +- 0.21 m-1); PAR (quanta) Kq = 0.18 + 1.6/Z_D, mean 0.52 +- 0.20 m-1 (205 stations, mostly fjord). In-water luminance spectra narrow to ~80 nm half-width with peaks at 550-570 nm. — [Aas et al. 2014](https://doi.org/10.5194/os-10-177-2014), Tables 3, 6, 8
- Kd(PAR) Aarhus Bay (North Sea-Baltic transition) 30-yr mean 0.30 +- 0.08 m-1 (max March 0.38, min June 0.26). — [Lund-Hansen 2024, Reg Stud Mar Sci](https://doi.org/10.1016/j.rsma.2023.103367)

### Inferences
**Surface curve used (clear sky, no moon)** - log-linear anchors: 0 deg 500 lx; -3 deg ~110 lx; -6 deg 3.4 lx; then x10 per 2 deg (Koomen) -> -8 deg 0.34 lx, -10 deg 0.035 lx, -11 deg 0.012 lx, -12 deg ~0.004 lx; + 0.001 lx night floor. Above +5 deg: Haurwitz clear-sky global radiation (as in `daylight.py`) x 115 lx per W m-2 (e.g. +5 deg ~5,200 lx; midwinter noon at 58.15 N, +8.4 deg, ~11,400 lx). 0 to +5 deg bridged log-linearly. Cloud (manifest 0.66-0.78 transmission) moves all thresholds ~0.3 deg later in twilight (factor 1.5 = 0.17 decade).

**Effective Kd at 30-60 m** - Skagerrak Secchi 8.3 m gives (Aas relations) Kd(555) ~0.18, Kd(490) ~0.29, Kq(PAR) ~0.37 m-1 for the surface layer. Below the surface layer the spectrum narrows to the 500-570 nm window and Atlantic water is clearer, so the effective Kd for gadoid-relevant light at 30-60 m is plausibly 0.15-0.22 m-1. The manifest's 0.29 is a mixed-layer Kd(PAR) for producer growth; applied over 50 m it implies 5e-7 transmission and gadoids blind at noon - not supported by any evidence. Note the herring pair was derived with Kd 0.15; using 0.2 for gadoids is a deliberate (slightly conservative) choice - if both should be consistent, use 0.15-0.2 for both.

**Unit conversion (spectrum dependent)**: 1 umol m-2 s-1 = 34.7 lx at 490 nm, 52.8 lx at 500 nm, 147 lx at 555 nm (photopic V(lambda)); for daylight ~55-60 lx (122 lx per W m-2 global, PAR = 0.45 x 4.57 umol/J). Gadoid lab thresholds converted with 50-150 lx per umol:
| Source | Value | ~lx |
|---|---|---|
| Meager 2010, lowest tested, cod still "highly sensitive" | 0.01 umol m-2 s-1 | 0.35-1.5 |
| Blaxter 1970 via Utne-Palm 2018, "possible threshold" | 0.001-0.01 umol | 0.035-1.5 |
| Anthony 1981, scotopic/photopic transition (pi*L assumption) | ~1e-4 umol | ~0.005 |
| Vollset 2011, larval cod attack in "darkness" | 3.67e-6 W m-2 (~1.5e-5 umol) | ~0.001 |
| Ryer & Olla 1999, pollock visual floor (verify) | ~5e-7 uE | ~3e-5 |

**Sun elevation (deg) at which light at depth equals a threshold** (E(z) = E0 x 0.9 x exp(-Kd z), clear sky):
| Light at depth | 30 m, Kd .15 / .20 / .25 / .29 | 40 m, Kd .15 / .20 / .25 / .29 | 50 m, Kd .15 / .20 / .25 / .29 |
|---|---|---|---|
| 0.3 lx | -4.1 / -2.6 / +0.4 / +3.0 | -2.6 / +1.5 / +6.2 / +21.7 | +0.4 / +6.2 / +56 / never |
| 0.1 lx | -5.1 / -3.8 / -1.8 / +0.6 | -3.8 / -0.8 / +3.4 / +8.8 | -1.8 / +3.4 / +18.1 / never |
| 0.03 lx (half-point candidate) | -6.1 / -4.8 / -3.5 / -1.8 | -4.8 / -3.1 / +0.8 / +4.2 | -3.5 / +0.8 / +7.1 / +39 |
| 0.01 lx (half-point candidate) | -7.1 / -5.8 / -4.5 / -3.4 | -5.8 / -4.0 / -1.4 / +1.9 | -4.5 / -1.4 / +3.8 / +14.0 |
| 0.003 lx | -8.1 / -6.8 / -5.5 / -4.5 | -6.8 / -5.1 / -3.3 / -0.6 | -5.5 / -3.3 / +1.2 / +5.8 |
| 0.001 lx (visual floor) | -9.1 / -7.8 / -6.5 / -5.4 | -7.8 / -6.0 / -4.3 / -2.8 | -6.5 / -4.3 / -1.0 / +3.2 |

**Sanity check (midwinter noon, 58.15 N, +8.4 deg, ~11,400 lx clear, ~7,500 lx with 0.66 cloud)**: light at 50 m = 3.7 lx (Kd .15), 0.31 lx (Kd .20), 0.025 lx (Kd .25), 0.0034 lx (Kd .29). Any threshold above ~+8 deg would make gadoids non-visual all winter - nothing found supports that, so combinations giving > +5 deg indicate a wrong Kd/depth assumption, not a result.

**Logistic width**: twilight surface light changes ~0.5 decade per degree, so `TRANSITION_DEG = 2` (10 % -> 90 % over +-4.4 deg) spans ~4.4 decades of light, versus ~2 decades for a Michaelis-Menten response E/(Ke+E). A logistic in elevation equivalent to MM-in-light would have a scale of ~0.87 deg. The fixed 2-deg width therefore partly stands in for the spread of gadoid depths (30-60 m = ~2-3 decades of light at Kd 0.2) and individual variation - which is reasonable for a population-level f.

### Gaps
- No direct measurement of Kd at 30-60 m in the open eastern Skagerrak was retrieved (Aas 2014 coefficients are 0-Secchi averages).
- No primary table of illuminance vs solar elevation (e.g. Bond & Henderson 1963; Seidelmann 1992; Austin et al. 1976) was retrieved; the curve above is built from Koomen's slope and Spitschan's end-points. Uncertainty ~+-0.5-1 deg in the -6 to -12 deg range.
- Moonlight (up to ~0.2 lx full moon) is not in the model; at 30 m with Kd 0.2 it gives ~4e-4 lx - just below the half-point candidates, so ignoring it is acceptable.

---

## Q4. How visual predation is parameterised in models (Aksnes & Giske 1993; Aksnes & Utne 1997; Fiksen; Varpe)

### Takeaway
The standard is the Aksnes-Giske/Aksnes-Utne visual range model: r^2 exp(c r) = |C0| Ap E' Eb/(Ke+Eb), so reactive distance saturates with background irradiance via a half-saturation Ke, with only two fitted eye parameters (sensitivity threshold E' and saturation Ke). Encounter/attack rate scales ~r^2, so attack rate ~ Eb/(Ke+Eb) in clear water: "half detection" corresponds to light at depth ~ Ke, which is 1-2 decades above the absolute visual floor. No adult-gadoid Ke value was retrieved; Meager et al. (2010) tried to fit the model for juvenile cod and needed two parameter sets.

### Cited Findings
- Aksnes & Utne (1997) revised visual range model, adds saturation at high light; "Given the values of only two parameters, one sensitivity threshold and one saturation parameter, the model is capable of predicting visual ranges for relatively large intervals of light intensity, prey size and turbidity." Tested on *Gobiusculus flavescens* (proxy, not a gadoid). — [Aksnes & Utne 1997, Sarsia 82:137-147](https://doi.org/10.1080/00364827.1997.10413647)
- Parameter definitions as used downstream: Ke = half-saturation level for processing of retinal irradiance (uE m-2 s-1); Eb = background irradiance; Emax; dSe = irradiance sensitivity threshold. — [Rohan 2021 PhD thesis, Univ. Washington](https://digital.lib.washington.edu/server/api/core/bitstreams/241f514a-9768-4b02-9fec-7db45851b99f/content) (secondary)
- Fiksen et al. (1998): ingestion model for larval cod and herring combining turbulence, irradiance, light attenuation and visual characteristics; the visual range model "was calibrated with observed estimates in larval herring and cod"; larvae predicted to ingest more near the surface due to exponential light decay. — [Fiksen, Utne-Palm, Aksnes, Eiane & Helvik 1998, Fish Oceanogr 7](https://doi.org/10.1046/j.1365-2419.1998.00068.x)
- Varpe & Fiksen (2010): mechanistic ingestion model including solar irradiance predicted herring seasonal intake/condition; visual predators at high latitude "vulnerable to changes in the light regime, such as water clarity". — [Varpe & Fiksen 2010, Ecology 91](https://doi.org/10.1890/08-1817.1)
- Meager et al. (2010): for juvenile cod "Only a model with a different set of parameters above and below 5 umol m-2 s-1 fitted the data", generality poor. — [Meager et al. 2010](https://doi.org/10.1080/10236241003798910)
- Langbehn & Varpe (2017) used the same visual-search framework to show sea-ice loss boosts visual search of fish (parameter values not retrieved). — [Langbehn & Varpe 2017, Glob Change Biol](https://doi.org/10.1111/gcb.13797)

### Inferences
- In the model's terms, the logistic half-point theta should be placed where light at the gadoid depth ~ Ke (half of saturated r^2), not where feeding stops. With no gadoid Ke available, the cod data in Q2/Q3 (reactive distance still high at 0.01 umol ~ 0.5 lx; scotopic transition ~0.005 lx) put the half-point at roughly 0.01-0.05 lx at depth. Confidence low-moderate.
- The model also implies prey size dependence: larger, higher-contrast prey (fish) stay detectable at lower light than small zooplankton - so for the same predator the threshold for zooplankton prey should be ~1-2 deg higher (earlier in dusk) than for fish prey.
- The visual-range model has no non-visual term; `dark_ratio` is the model's stand-in for lateral line + chemoreception + taste, and must come from behavioural evidence (Q1), not from the eye model.

### Gaps
- No published Ke / E' values for adult cod, whiting, saithe or haddock retrieved (Fiksen et al. 1998 larval cod values and Meager 2010 juvenile values exist but full texts were inaccessible).

---

## Q5 (objective). Recommended dark_ratio and light_threshold_deg per gadoid prey pair

### Takeaway
Suggested starting values (all estimates, not measured; Kd_eff 0.2 m-1, gadoids feeding at ~30-50 m, half-point 0.01-0.03 lx at depth): fish prey (pelagic_fish, gadoids/cannibalism) dark_ratio 0.25 (range 0.15-0.35), threshold -4 deg (range -6 to -1); zooplankton/krill dark_ratio 0.3 (0.2-0.5), threshold -3 deg (-5 to 0); benthic_community unmodulated (dark_ratio 1 or absent), or at most 0.8 with threshold -6. All gadoid thresholds sit 3-5 deg above the herring value (-8) because gadoids feed deeper, not because their eyes are worse.

### Cited Findings
(Evidence base, all cited in Q1-Q4.)
- Crepuscular piscivory: cod on sprat mostly at dusk/dawn — [Andersen et al. 2017](https://doi.org/10.3354/meps12319); cod most effective on 0-group cod/saithe at dusk/night/dawn — [Strand et al. 2020](https://doi.org/10.1016/j.jembe.2020.151356); whiting eat free-swimming prey mainly by day — [Rindorf 2003](https://doi.org/10.3354/meps249265).
- Residual non-visual capture: dark predation by juvenile cod "significantly lower" but present, chemical cues detected from longer distance than visual — [Meager et al. 2005](https://doi.org/10.1139/f05-104); pollock capture prey in darkness — [Ryer & Olla 1999](https://doi.org/10.3354/meps181041); cod olfactory response independent of light, bait localization 45 % (night) vs 70 % (day) — [Løkkeborg & Fernö 1999](https://doi.org/10.1023/A:1007504712163).
- Benthic prey at night: whiting bottom-dwelling prey more at night (4/5 locations) — [Rindorf 2003](https://doi.org/10.3354/meps249265); haddock polychaetes until after sunset — [Albert 1995](https://doi.org/10.1006/jmsc.1995.0084).
- Krill: cod eat euphausiids mostly at night (vertical overlap, not detection) — [Jansen et al. 2025](https://doi.org/10.1007/s11160-025-09989-w); krill bioluminescence (468 nm) within cod's sensitivity — [Utne-Palm et al. 2018](https://doi.org/10.1371/journal.pone.0190918).

### Inferences
| Pair (library key) | dark_ratio | light_threshold_deg | Reasoning / confidence |
|---|---|---|---|
| `gadoids_preys_on_pelagic_fish` | 0.25 (0.15-0.35) | -4 (-6 to -1) | Visual pursuit of schooling clupeids peaks at dusk/dawn; residual close-range lateral-line/olfactory capture of dispersed night schools justifies > herring's 0.1. Threshold: 0.01-0.03 lx at 30-40 m, Kd 0.2 -> -6 to -3 deg. Low-moderate. |
| `gadoids_preys_on_gadoids` (cannibalism, 0-group) | 0.3 (0.2-0.5) | -4 | Strand 2020: cod most effective at dusk/night/dawn on 0-group gadoids - suggests night capture is substantial (prey near bottom, tactile/olfactory cues). Low. |
| `gadoids_preys_on_zooplankton` (krill proxy) | 0.3 (0.2-0.5) | -3 (-5 to 0) | Small prey needs more light (r^2 ~ prey area), so half-point ~1-2 deg earlier than fish; krill bioluminescence and night vertical overlap raise effective night intake. Jansen 2025 conflates overlap with detection - the model's spatial overlap should not be double-counted via dark_ratio. Low. |
| `gadoids_preys_on_benthic_community` | 1.0 (no modulation) or 0.8 | (-6 if modulated) | Benthic feeding is taste/olfaction dominated (barbel in cod/haddock), eaten as much or more at night (whiting, haddock, poor-cod). The model cannot express dark_ratio > 1. Moderate. |

- Species weighting: the `gadoids` FG mixes cod (generalist piscivore), whiting (piscivore, night benthic feeding), saithe (pelagic, diurnal activity, no light data) and haddock (benthivore, barbel). Saithe-heavy assemblages would argue for a lower dark_ratio on fish; haddock-heavy for leaving benthos unmodulated.
- If the modeller keeps Kd 0.29 for the gadoid conversion, thresholds at 40-50 m move to +2 to +14 deg and gadoids become blind all winter - contradicting the field diel data. Use Kd_eff 0.15-0.2 (consistent with herring's 0.15) or place gadoids at ~30 m.
- Comparison with herring: herring's 0.001 lx at 10-40 m (Kd 0.15) -> -8 to -12 deg. Gadoids with comparable or slightly lower sensitivity at 30-50 m land at -6 to -1 deg; a 3-5 deg difference = ~1.5-2.5 decades of light, i.e. mostly the depth difference.

### Gaps
- None of the dark_ratio values above are measured; a numeric dark/light capture-rate ratio for adult gadoids on any prey type was not found.
- Seasonal depth changes (saithe deeper in winter 60-90 m vs 25-50 m in summer, Skilbrei & Otterå 2016) would shift the threshold seasonally; the model uses one threshold per pair.
