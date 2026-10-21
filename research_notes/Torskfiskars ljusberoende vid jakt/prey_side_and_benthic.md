# Prey side and benthic contrast: light-dependent predation by gadoids (cod, whiting, saithe, haddock) in the North Sea / Skagerrak

Scope note for the report writer: the target is a per-prey `dark_ratio` (night attack rate / day attack rate) for the `gadoids` FG at 58 N in the eastern Skagerrak. In the current code (`lib/world/daylight.py`, `attack_multiplier`) a `dark_ratio` >= 1 is treated as "no light dependence", so a prey type that gadoids eat MORE at night cannot currently be represented as anything above 1. Each finding below is tagged with region; anything outside the North Sea/Skagerrak/Norwegian coast is flagged. Several primary PDFs (Springer, Int-Res, ScienceDirect) were blocked to automated fetching; where only the abstract or a search snippet was available this is stated.

## 1. Herring/sprat: does night-time school dispersal and upward migration make clupeids more or less vulnerable to gadoids?

### Takeaway
The direct evidence from Norwegian fjords (Oslofjord, i.e. Skagerrak-adjacent at 59-60 N) is that gadoids keep eating clupeids at night, and that clupeid schools disperse at dusk; for sprat, gadoids were observed foraging on the vertically migrating sprat specifically at night. Herring stomach contents of gadoid-eaten clupeids are not resolved by time of day because fish prey digest slowly. There is no evidence for a large day/night collapse of gadoid predation on clupeids comparable to herring's own visual plankton feeding; the evidence leans toward roughly equal or night-shifted encounter, offset by worse visual capture in the dark (net effect: dark_ratio near 1, low confidence, in the range ~0.5-1).

### Cited Findings
- Oslofjord (59 N): "Fish schools dispersed at sunset, and fish invaded the krill layer at night." (Krill DVM study; the abstract does not name the fish species.) — [Onsrud & Kaartvedt 1998, MEPS 171:209](https://doi.org/10.3354/meps171209)
- Norwegian fjord (Bunnefjorden, inner Oslofjord), 4 winters of acoustics: sprat overwinter in hypoxic deep water (down to ~7% O2 saturation); "Gadoid predators seemed to avoid waters with oxygen contents below ~15 to 20% O2 saturation so that few predators occurred in the deep, low-oxygen sprat habitat. However, gadoids were foraging on the vertically migrating sprat at night." — [Kaartvedt, Røstad & Klevjer 2009, MEPS 390:237](https://doi.org/10.3354/meps08196)
- Same fjord, 4 winters: "particularly large concentrations of krill appeared to mitigate predation pressure from gadoids, which then preferred krill as prey"; sprat carried out DVM in all winters; part of the population schooled at mid-depth by day. — [Solberg, Røstad & Kaartvedt 2015, Prog. Oceanogr.](https://doi.org/10.1016/j.pocean.2015.08.003)
- Same fjord: when ice covered the fjord, sprat switched from near-bottom 'rise and sink' to midwater schooling by day and "aggregated in dense layers near the surface at night"; interpreted as a shift in antipredator strategy "from hiding at depth to hiding in schools in the darker waters below the ice" (i.e. darkness itself is treated as a refuge from visual predators). — [Solberg, Klevjer & Kaartvedt 2012, MEPS 464:245](https://doi.org/10.3354/meps09877)
- Master thesis, Bunnefjorden winter 2005/06: by day krill and sprat scattering layers were restricted to below 75 m and 130 m respectively; krill and sprat ascended at dusk; whiting was the main gadoid studied, with fish sampled day and night (abstract only; detailed diel stomach results not retrieved). — [Brun 2007, UiO thesis](http://hdl.handle.net/10852/11664)
- Norwegian Sea (offshore, May): individual cod and haddock "intimately accompanied the herring schools, their attacks causing frequent modifications to, but not dispersal of, school structure"; "rapid approach by fast-moving schooling predators such as saithe ... caused the herring to dive steeply"; >230 behavioural events at ~5 min intervals. No time-of-day breakdown in the abstract. — [Pitcher, Misund, Fernö, Totland & Melle 1996, ICES JMS 53:449](https://academic.oup.com/icesjms/article/53/2/449/792847)
- Norwegian Sea (April): herring schools averaged 987 m2, at 148 m depth, 1.1 BL/s, "reflecting a risk-averse, anti-predator behaviour"; the paper contrasts this with "herring attacked by gadoid predators closer to the coast in a similar study in May" (Pitcher et al. 1996). Cod and saithe were not caught or detected in that offshore area. — [Nøttestad 2002, ICES JMS 59:393](https://doi.org/10.1006/jmsc.2001.1172)
- Overwintering Norwegian spring-spawning herring (northern Norwegian fjords): downward migration at dawn, upward migration at dusk, with partial descent through the night; "attributed to schooling and predator avoidance during the day, and energy conservation and swimbladder refilling during the night." — [Huse & Korneliussen 1995, IMR/NORA](http://hdl.handle.net/11250/105417)
- Same system: daytime layers aggregated with near-horizontal swimming; at night tilt angles up to 40 deg and "rise and glide" swimming, i.e. loose, non-schooling individuals at night. — [Huse 1996, ICES JMS 53:863](https://doi.org/10.1006/jmsc.1996.9999)
- Overwintering herring (Dec/Jan, Norway) ADCP: higher swimming speeds in daylight, increased activity at dawn and dusk, "no well-defined velocity" at night. — [Zedel, Knutsen & Patro 2003, ICES JMS 60:846](https://doi.org/10.1016/s1054-3139(03)00067-5)
- Western Norway local herring (Lindåspollene): during pre-spawning herring increased distance to the bottom "in the presence of gadoid predators"; packing density increased when the school moved to the bottom (home range of cod) for spawning, interpreted as increased vigilance. — [Fatnes 2011, UiB thesis](http://hdl.handle.net/1956/5587)
- Analogous gadoid prey (juvenile walleye pollock, lab, NE Pacific - FLAG region/species): shoals gradually disperse as light falls with no sharp threshold; below ~2.8 x 10^-? uE s-1 m-2 (exponent garbled in the OCR'd abstract) fish no longer formed a shoal; in the dark, a predator made prey swim faster and disperse further; "fish which have become widely scattered during the darkness may take longer to reform shoals at dawn, resulting in greater predation risk." — [Ryer & Olla 1998, MEPS 167:215](https://doi.org/10.3354/meps167215)
- Herring-eating killer whales (Iceland, winter - FLAG: mammal predator, not gadoid) fed at night as well as day: feeding in 50% of night time vs 73% of day time, with more acoustic herding calls at night. Relevant only as evidence that herring remain catchable in darkness by a non-visual-capable predator. — [Richard et al. 2017, Mar Biol](https://doi.org/10.1007/s00227-016-3059-8)

### Inferences
- Two opposing effects: at night clupeid schools break up (loss of dilution/confusion and collective vigilance, which favours the predator) and migrate up into the gadoids' foraging range in fjords; but gadoids lose visual detection range and pursuit accuracy. Fjord evidence (Kaartvedt 2009) suggests the first effect is at least as strong in practice: the night ascent of sprat is when gadoids catch them. This argues against a low (e.g. 0.1, herring-on-zooplankton style) dark_ratio for gadoids -> pelagic_fish.
- Herring and sprat in the eastern Skagerrak are largely pelagic and, in the model, gadoids are partly demersal; the night ascent of clupeids could *reduce* vertical overlap with bottom-associated cod/haddock while *increasing* it with midwater whiting/saithe. Because the model has no vertical dimension, the net is best represented as dark_ratio ~ 0.5-1 (low confidence), not << 1.
- Since the gadoids -> pelagic_fish pair already has a Type III response (section 141.4 of the project), a light effect that is near-neutral avoids double-counting refuge effects.

### Gaps
- No quantitative day vs night attack or capture rate of cod/saithe on herring schools was found; Pitcher et al. 1996 and Nøttestad 2002 abstracts give no time-of-day split. The claim "low predation risk at night associated with demersal herring layers" appeared only in an uncited search snippet and could not be verified - not used.
- Rindorf 2003 (below) found no diel pattern for larger fish prey in whiting stomachs "presumably due to their longer digestion time", so stomach data cannot resolve the fish-prey dark_ratio.
- Utne-Palm's and Fernö's direct work on herring-predator light interactions was not retrieved.

## 2. Benthic invertebrates: chemosensory/tactile feeding by cod and haddock - does benthic feeding continue or increase at night?

### Takeaway
This is the strongest contrast in the evidence. For whiting in the North Sea, bottom-dwelling prey occur significantly MORE often in stomachs at night (4 of 5 locations), while free-swimming prey are eaten mainly in daylight. Cod locate food by smell over hundreds of metres day and night (more slowly at night), gadoids carry dense taste buds on barbel and pelvic fin rays used to search the sea bed, and cod and haddock keep feeding through continuous polar-night darkness. Benthic prey should get dark_ratio >= 1 (no reduction); the evidence for whiting supports a night increase that the current code cannot express.

### Cited Findings
- North Sea whiting, round-the-clock stomach sampling at 5 locations, GLMs: "the occurrence of bottom dwelling prey increased significantly during the night at 4 of 5 locations. In contrast, free swimming prey and prey migrating towards the demersal layer during the day were eaten mainly in the daylight hours. No diel pattern in the presence of larger fish prey could be found"; no general diel pattern in predator catch rates; "numerous studies have analysed the feeding periodicity of North Sea gadoids, [but] no general diel pattern has been found", attributed to differences in prey composition. — [Rindorf 2003, MEPS 249:265](https://doi.org/10.3354/meps249265)
- Northern Norway fjord, acoustic telemetry of 8 cod over 3-8 days: diurnal rhythm with "slightly lower swimming speed during a 5-h period at night"; a larger share of active prey taken by day with vision more important then; no significant day/night difference in range occupied; cod detected bait odour "from a distance of several hundred metres" and responded with rheotactic search. — [Løkkeborg 1998, Animal Behaviour](https://doi.org/10.1006/anbe.1998.0772)
- Cod respond to bait odour both day and night and locate bait at night "but more slowly"; chemically mediated search at night uses lower swimming speeds than visual search by day; cod responding from 700 m located baits 7 h after line setting (search-snippet summary drawn from Løkkeborg & Fernö 1999 and/or Løkkeborg et al. 2014; full text not retrieved and attribution of the 700 m / 7 h numbers and "slower at night" to a specific paper is uncertain - verify before citing a DOI; medium confidence) — [Løkkeborg & Fernö 1999, Env Biol Fish](https://doi.org/10.1023/a:1007504712163); [Løkkeborg et al. 2014, Rev Fish Biol Fish](https://doi.org/10.1007/s11160-014-9360-z)
- Responses of cod, haddock and whiting to baited hooks increase at dawn and decrease at dusk, following the seasonal photoperiod (search-snippet summary; this concerns bait/longline catchability, which mixes activity and feeding motivation) — [Løkkeborg et al. 2014](https://link.springer.com/article/10.1007/s11160-014-9360-z)
- Juvenile gadoids, west of Scotland, 5 species: highest cutaneous taste bud densities on the anterior naris flap, the barbel, pelvic fin rays, snout tip and upper lip; cod and poor cod had significantly higher densities on the first two pelvic fin rays, "correspond[ing] with their more benthic lifestyle, in which the pelvic fins are frequently trailed over the sea bed when searching for prey." — [Harvey & Batty 2002, J Fish Biol](https://doi.org/10.1111/j.1095-8649.2002.tb01686.x)
- Svalbard fjords (78-80 N, FLAG: Arctic, January polar night, 2012-2016): across polar cod, Atlantic cod and haddock ">70% of individuals were feeding"; empty stomachs 40% polar cod, 25% Atlantic cod, 21% haddock; few stomachs well filled; diets opportunistic with heavy overlap on krill (Thysanoessa) in krill-rich years. Shows boreal cod and haddock feed in near-continuous darkness. — [Larsen et al. 2023, Polar Biology](https://doi.org/10.1007/s00300-023-03167-z)
- Northern North Sea haddock: stomach contents dominated by echinoderms and polychaetes, mirroring the benthic community; prey selection driven by field prey density, "highly opportunistic" (no diel analysis). — [Schückel et al. 2010, J Fish Biol](https://pubmed.ncbi.nlm.nih.gov/20646141)
- Juvenile cod (lab, FLAG: not North Sea): foraging rate little affected by turbidity up to 28 beam attenuation m-1, attributed to chemoreception used together with vision; prey odour raised activity at all turbidity levels. — [Meager & Batty 2007, Phil Trans R Soc B](https://doi.org/10.1098/rstb.2007.2104)
- North Sea IYFS trawl data 1976-1993 (Scottish): day-night catch differences for adult haddock and dab and juvenile herring/dab, but not reported for whiting; haddock length composition unaffected by light. (Catchability, not feeding.) — [Petrakis et al. 2001, ICES JMS 58:50](https://doi.org/10.1006/jmsc.2000.0989)

### Inferences
- The mechanism for a night increase is on the prey side: many benthic crustaceans (e.g. crangonids, amphipods, Nephrops) and polychaetes emerge from the sediment at night, so encounter rises even as vision fails; gadoids can exploit this because their benthic search is chemo-tactile (barbel, pelvic fin taste buds, olfaction). Rindorf 2003 is the only North Sea study found that ties diel intake to prey type directly, and it points this way.
- Recommendation: gadoids -> benthic_community dark_ratio = 1 (leave unset) as the conservative, code-compatible choice; if the code is extended to allow rho > 1, the evidence supports a modest night increase (order 1.2-2, low confidence: Rindorf reports significance of occurrence, not a rate ratio).
- Benthic feeding by haddock is the most chemo-tactile of the four gadoids; whiting and saithe are more visual/pelagic. Since the model's gadoid FG lumps them, the benthic link (mainly cod/haddock) should be the least light-limited.

### Gaps
- No study found giving a quantitative night/day consumption ratio for cod or haddock on benthos in the North Sea/Skagerrak; specific nocturnal haddock feeding literature (e.g. older Aberdeen stomach-chronology work) was not retrieved.
- Rindorf 2003 effect sizes per prey taxon (which bottom prey, odds ratios) could not be read (PDF blocked); only the abstract.
- Pawson's and Hara's chemoreception work was not retrieved.

## 3. Juvenile gadoids / cannibalism: light dependence of cod and whiting predation on 0-group gadoids

### Takeaway
Experimental evidence from northern Norway shows large cod (and sculpin) are most effective predators on 0-group cod and saithe at dusk, night and dawn, not in full light; Newfoundland lab work shows 0-group cod lose some anti-predator spacing in the dark. Predation on small juvenile gadoids therefore does not collapse at night; if anything it peaks in low light. dark_ratio for gadoids -> gadoids should be ~1 (no reduction), low-medium confidence.

### Cited Findings
- Porsangerfjorden (northern Norway, 70 N) tank experiments with video: large cod (300-400 g) and shorthorn sculpin preying on newly settled 0-group cod and saithe. Under 24 h light (midnight sun), sculpins alone caught no 0-group fish and did not even attack; switching to a day/night cycle raised sculpin consumption from 0 to 29% of the 0-group fish; "Sculpins and cod were most effective predators on 0-group cod and saithe at dusk/night/dawn"; adding a large cod increased sculpin success because 0-group fish fled into cobble/vegetation. — [Strand et al. 2020, JEMBE 524:151356](https://doi.org/10.1016/j.jembe.2020.151356)
- Newfoundland (FLAG: NW Atlantic), lab and field: diel changes in behaviour and habitat use of age-0 cod; search-snippet summary: under darkness age-0 cod kept less distance from a larger gadoid predator than in light regardless of habitat, and stayed below predators in high light but above them in low light (attribution to this paper from a search snippet; full text not retrieved, medium confidence). — [Anderson, Laurel & Brown 2007, JEMBE](https://doi.org/10.1016/j.jembe.2007.07.001)
- Newfoundland (FLAG): juvenile cod hide in cobble interstices when a larger conspecific predator is present (lab substrate choice). — [Gotceitas & Brown 1993, Oecologia](https://pubmed.ncbi.nlm.nih.gov/28313770)
- NW Atlantic (FLAG): 3+ cod preying on 0-group cod over 5 habitats; predator reaction distance decreased with habitat complexity "presumably due to the obstruction of visual cues by complex relief" - i.e. cannibalistic cod rely partly on vision. — [Lindholm, Auster & Kaufman 1999, MEPS 180:247](https://doi.org/10.3354/meps180247)
- Juvenile (age-1) cod reactive distance measured from 0.01 to 64 umol m-2 s-1: high sensitivity at 0.01 umol m-2 s-1, lower reactive distance at intermediate light (1.5-6.5) and higher above; interpreted as a change of foraging behaviour at twilight light levels (Norway, lab). — [Meager, Moberg, Strand & Utne-Palm 2010, Mar Freshw Behav Physiol 43:2](https://doi.org/10.1080/10236241003798910)
- Cod larvae at 53 dph attacked prey in darkness (~3.67 x 10^-6 W m-2); attacks from 26 dph at "night" light (~1.38 x 10^-4 W m-2) (Norway, lab). — [Vollset, Folkvord & Browman 2011, Mar Biol](https://doi.org/10.1007/s00227-011-1635-5)

### Inferences
- Juvenile gadoids are near-bottom, cover-seeking prey: by day they hide in structure (vision lets them see the predator), at dusk/night their vigilance drops while cod switch to chemo-tactile/low-light search. That produces a crepuscular/nocturnal peak, consistent with Strand et al. 2020. A dark_ratio < 1 for gadoids -> gadoids is not supported; 1 (unset) is the defensible value, with a possible night increase if the code allowed rho > 1.
- Cod visual sensitivity is high at very low light (Meager et al. 2010; Vollset et al. 2011), so twilight at 58 N does not blind them; any threshold for gadoid visual search should sit well below civil twilight.

### Gaps
- No North Sea/Skagerrak field estimate of diel timing of cod or whiting cannibalism was found.
- Strand et al. 2020 numbers for cod alone by light phase were not available beyond the abstract.

## 4. Krill/Calanus: do gadoids eat krill when krill ascend at night (visual vs non-visual)?

### Takeaway
In the Oslofjord, whiting and Norway pout ate Meganyctiphanes krill with no systematic day/night difference in stomach fullness or digestion state, and whiting tracked the krill layer up to the surface at night. Svalbard cod and haddock feed on krill in polar night. Gadoid predation on krill/zooplankton should get dark_ratio ~1 (medium confidence, Skagerrak-adjacent evidence).

### Cited Findings
- Inner Oslofjord (59 48 N, 120 m site, 1997-2001): "Herring, whiting, and Norway pout were foraging on krill during both day and night." Whiting: krill up to 99% of stomach weight, "on average ~82% during both day and night", up to 40 krill per stomach (21 cm fish), "No systematic trends ... comparing proportion of stomachs containing krill or state of digestion during the day and night"; whiting in the upper/middle krill layer by day and near the surface at night. Norway pout: diet dominated by krill and Calanus, no significant day/night difference in fullness; deeper than the krill by day, attacking from below; "deep nocturnal distribution rendered visual predation unlikely". Herring: krill in stomachs day and night, no clear diel pattern. Sprat: no krill. — [Onsrud, Kaartvedt, Røstad & Klevjer 2004, ICES JMS 61:1278](https://doi.org/10.1016/j.icesjms.2004.09.005)
- Oslofjord krill layers stay below ~70 m by day, ascend at dusk following isolumes but mostly stop at ~10-30 m; "fish invaded the krill layer at night". Light at top of the krill layer by day ~10^-? to 2.3 x 10^-4 umol m-2 s-1 (lower bound exponent garbled in OCR). — [Onsrud & Kaartvedt 1998, MEPS 171:209](https://doi.org/10.3354/meps171209)
- Meganyctiphanes swim faster at night (Oslofjord acoustics) - relevant to encounter rates with non-visual predators. — [Klevjer & Kaartvedt 2011, L&O 56:765](https://doi.org/10.4319/lo.2011.56.3.0765)
- Bunnefjorden: large krill concentrations diverted gadoid predation from sprat to krill. — [Solberg et al. 2015](https://doi.org/10.1016/j.pocean.2015.08.003)
- Svalbard polar night (FLAG: Arctic): cod and haddock diets overlapped heavily on Thysanoessa krill in krill-rich years, with 75-79% of cod/haddock stomachs non-empty in January. — [Larsen et al. 2023](https://doi.org/10.1007/s00300-023-03167-z)
- Northern Norway (FLAG: artificial light): lit pots caught 17x more cod than bait-only controls at the highest intensity (9744 mW m-2), ~5x at 23 mW m-2, no effect at 8 mW m-2 green; cod in lit pots had more krill and arrowworms in stomachs; authors conclude cod were attracted by the dense krill swarms, not light per se. Shows cod feeding on krill is strongly density/aggregation driven and can be visually enhanced. — [Humborstad et al. 2018, ICES JMS 75:2257](https://doi.org/10.1093/icesjms/fsy099)
- Rindorf 2003 (North Sea whiting): "prey migrating towards the demersal layer during the day were eaten mainly in the daylight hours" - i.e. DVM prey (likely euphausiids/Norway pout-type) are eaten when they are near the bottom, by day. — [Rindorf 2003](https://doi.org/10.3354/meps249265)

### Inferences
- The krill/zooplankton pair is shaped by vertical overlap, not light-limited detection: demersal gadoids meet DVM crustaceans near the bottom by day (Rindorf 2003), pelagic gadoids (whiting) follow them up at night (Onsrud et al. 2004). Averaged over a lumped gadoid FG, the diel signal cancels -> dark_ratio ~1.
- This contrasts with the existing herring -> zooplankton dark_ratio 0.1 in `fg_library.yaml`: herring are particulate visual planktivores, whereas gadoids that eat krill do so with no detectable diel difference in the Skagerrak-adjacent fjord data.

### Gaps
- No Skagerrak open-water (non-fjord) diel stomach data for saithe or cod on krill found.
- Calanus specifically as gadoid prey (Norway pout) has no diel resolution beyond Onsrud et al. 2004.

## 5. How other ecosystem models treat light-dependent feeding by gadoids

### Takeaway
None of the standard end-to-end/ecosystem frameworks checked applies a light-dependent attack rate to gadoids. Atlantis handles day/night only through separate day and night vertical distributions (which change predator-prey spatial overlap); light otherwise enters primary production and an optional human light-pollution scalar. EwE and OSMOSE (from documentation knowledge, not re-verified here) have no diel light term. Light-dependent visual feeding is the domain of IBMs for larvae and planktivores, not adult gadoids.

### Cited Findings
- Atlantis: "VERTnight_XXX and VERTday_XXX parameters define vertical distribution"; after horizontal distribution "it will then apply the vertical distribution VERTnight_XXX and VERTday_XXX parameters to distribute the individuals (or biomass) in each box among the vertical layers." — [Atlantis User's Guide Part I (CSIRO, 2025)](https://research.csiro.au/atlantis/wp-content/uploads/sites/52/2025/05/AtlantisUserGuide_PartI.pdf)
- Atlantis predation requires: "1) Physical overlap - prey and predator must be in the same cell at the same time (determined by vertical and horizontal distribution parameters) ... 2) Diet connection matrix (pPREY) ... 3) Gape limitation ... 4) Habitat refuge. 5) Environmental factors (temperature, salinity, pH) that can modify predator's feeding rates" - light is not among the listed feeding modifiers; most models use a modified Holling type II. — [Atlantis User's Guide Part I](https://research.csiro.au/atlantis/wp-content/uploads/sites/52/2025/05/AtlantisUserGuide_PartI.pdf)
- Atlantis optional light/noise pollution: for groups flagged `isLightEffected`, a scalar makes boxes less attractive and scales consumption (C) and growth (mum), with mortality scaled by the inverse - anthropogenic light only, not the natural diel cycle. — [Atlantis User's Guide Part I, section 4.6](https://research.csiro.au/atlantis/wp-content/uploads/sites/52/2025/05/AtlantisUserGuide_PartI.pdf)
- Light-based visual-predation models exist for larval cod and planktivores (lab parameterisation of cod reactive distance vs light for "a predictive visual model", which generalised poorly). — [Meager et al. 2010](https://doi.org/10.1080/10236241003798910)
- Stoner's review: light level is among the variables (with temperature, current, prey density) "likely to have largest effects on fish catchability, potentially affecting variation in CPUE by a factor of ten" - an argument that light effects on gadoid feeding behaviour are real but mostly documented as catchability, not as diet rates. — [Stoner 2004, J Fish Biol](https://doi.org/10.1111/j.0022-1112.2004.00593.x)

### Inferences
- A per-pair dark_ratio for gadoids goes beyond common practice. The closest analogue in Atlantis is a day/night change in overlap, which in a 2D grid model maps onto a per-pair multiplier that can go either way (down for visually hunted pelagic prey, up or neutral for benthic/DVM prey). This supports keeping gadoid pairs at or near 1 unless there is pair-specific evidence.

### Gaps
- OSMOSE, EwE, NEMURO and Barents/North Sea cod IBM documentation were not fetched in this session; the statement that they lack a light term is from prior knowledge and should be verified before citing.

## 6. Synthesis: suggested dark_ratio by gadoid prey type (answering the objective)

### Takeaway
Evidence ranks gadoid prey from most to least light-limited as: pelagic fish (possible mild reduction at night) > krill/zooplankton (~neutral) ~ juvenile gadoids (neutral to night-peak) > benthic invertebrates (neutral to night increase). Nothing supports a herring-style 0.1 for any gadoid pair.

### Cited Findings
- Free-swimming prey eaten mainly by day, bottom prey more at night, larger fish prey no detectable diel pattern (North Sea whiting). — [Rindorf 2003](https://doi.org/10.3354/meps249265)
- Gadoids caught vertically migrating sprat at night in a Skagerrak-adjacent fjord. — [Kaartvedt et al. 2009](https://doi.org/10.3354/meps08196)
- No day/night difference in krill feeding by whiting and Norway pout (Oslofjord). — [Onsrud et al. 2004](https://doi.org/10.1016/j.icesjms.2004.09.005)
- Cod most effective on 0-group gadoids at dusk/night/dawn (northern Norway, lab). — [Strand et al. 2020](https://doi.org/10.1016/j.jembe.2020.151356)
- Cod locate food chemically day and night, slower at night; swimming only slightly slower for ~5 h at night. — [Løkkeborg 1998](https://doi.org/10.1006/anbe.1998.0772)

### Inferences
- Suggested values (all estimates, not measured ratios):
  - gadoids -> pelagic_fish: 0.5-1.0 (low confidence). Visual pursuit worse at night, but schools disperse and clupeids migrate into gadoid range. If a value < 1 is used, a high-ish ratio (~0.6-0.7) is more defensible than anything near 0.1. Note interaction with the existing Type III response on this pair.
  - gadoids -> zooplankton (krill proxy): 1.0, unset (medium confidence).
  - gadoids -> gadoids (cannibalism / juvenile gadoids): 1.0, unset (low-medium confidence); evidence hints at a twilight/night peak.
  - gadoids -> benthic_community: 1.0, unset (medium confidence); evidence favours a night increase (>1) which the code currently cannot represent (rho >= 1 is treated as "no effect" in `lib/world/daylight.py`).
- If the project adds light dependence only to the gadoids -> pelagic_fish pair, the annual-mean normalisation means the day rate rises above the library `a`; with rho ~0.7 that rise is small.

### Gaps
- No study gives a measured night/day attack-rate ratio for any gadoid-prey pair; every value above is an inference from stomach-occurrence, behaviour or lab data.
