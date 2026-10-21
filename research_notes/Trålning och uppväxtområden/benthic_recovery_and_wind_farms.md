# Benthic recovery when trawling stops, and whether wind farms or trawl closures raise juvenile gadoids (Q4)

Scope:
- This note answers Q4 of the trawling/nursery review: how fast the benthos recovers when trawling stops, and whether
  wind farms or trawl closures raise juvenile gadoid abundance or recruitment.
- It builds on `research_notes/Bottentrålning och bentos/depletion_and_recovery.md`, which already covers:
  - Hiddink et al. 2017: community r 0.82 /yr, 3.6 yr from 0.5K to 0.95K;
  - Pitcher et al. 2022: R 0.29-0.68 /yr by sediment, the RBS equation;
  - Rijnsdorp et al. 2020: the PD method, quoting Hiddink et al. 2019 r x longevity = 5.31;
  - Sköld et al. 2018: Kattegat Nephrops grounds.
  Those values are not re-derived here.

Labels:
- **[Primary]**: peer-reviewed literature.
- **[Report]**: assessments, EIAs and grey literature.
- **[Own]**: my computation or inference.

Fetch status, where it matters:
- **FULL**: full text read.
- **ABS**: verbatim abstract read.
- **SNIP**: only a search snippet, so the number is unverified.

DOIs were checked against Crossref.

## Site facts that decide what transfers

The grid holds two parks, Mareld (west) and Poseidon Nord (east). Each has its own EIA. [Report]
- Mareld: *Miljökonsekvensbeskrivning Mareld vindkraftpark*, Bilaga C till ansökan om tillstånd, 17 April 2023, Mareld
  Green Energy AB (`MKB Mareld.pdf` in the repo).
- Poseidon: *Poseidon. MKB för etablering av havsbaserad vindkraft*, KonTiki Vind AB / WSP Sverige AB, 16 December 2022
  (`Poseidon/Poseidon MKB.pdf`).

- **Depth.**
  - Mareld: 180-450 m, deepest in the NW (Mareld EIA 5.2, 6).
  - Poseidon Nord: "från ca 55 till ca 230 m". About 15 % of it is shallower than 100 m (Poseidon EIA 1.4, 4.2).
  - The task brief's "100-500 m" is the whole grid.
  - A17 itself is about 345 m deep (`research_notes/Zooplankton vinterdödlighet/winter_stock.md`).
- **Foundations: floating turbines in both parks.**
  - Mareld: three to nine mooring lines per turbine with drag, suction, pile or torpedo anchors; anchor radius up to
    1,400 m.
  - Poseidon: semi-submersibles are the likely choice; transformer stations "troligen" on fixed foundations. Scour
    protection "enbart" below 100 m depth.
- **Seabed.**
  - Mareld: "framför allt mjukbotten bestående av lera" (mainly soft clay bottom), probably accumulation bottom.
  - Poseidon: "till största del av mjukbotten" (mostly soft bottom).
  - Fauna includes sea pens (Funiculina quadrangularis, Virgularia, Stylatula, Pennatula), Nephrops and Pandalus.
  - The EIA states that the red-listed sea pens now survive mainly where trawling is banned or impossible (9.1).
- **Fishery:** the park lies largely inside the national-interest area for shrimp trawling, "Norra Skagerrak
  utsjöområde" (5.9.1.1).
- **Reef effect:** the EIA expects it to be "betydligt mindre för de flytande vindkraftverken" (much smaller for the
  floating turbines), because only the mooring chains touch the bottom. It rates the consequence for bony fish as
  negligible (8.10, 9.2.2.5).

From the repo grid [Own]:
- The park polygons cover **521 km2, 14.5 % of the 3,600 km2 grid**: area-weighted from `vindparker.npz`, 627 cells
  touched and 429 cells full.
- Per park:
  - Mareld: 369 km2 inside the grid, mean depth 264 m, 22 % of grid effort (6.6 MW-h/km2/yr).
  - Poseidon Nord: 152 km2, mean depth 157 m, 5 % of grid effort (3.3 MW-h/km2/yr).
  - Depths are from the EMODnet DTM; see `nursery_areas.md`.
- Together they carry **27 % of the grid's bottom-trawl effort**:
  - mean effort 5.6 MW-h/km2/yr inside, against 2.5 outside and 3.0 for the grid;
  - SAR 4.2 /yr inside against 1.9 outside, at 0.75 SAR per MW-h/km2.
- The park is therefore the most trawled part of the box, on the habitat most sensitive per pass (mud, high d).

## A. How fast does the benthos recover after trawling stops?

### Takeaway
- **Community biomass on mud recovers within years.**
  - Logistic r is 0.68 /yr for mud (Pitcher 2022; lower CL 0.48) and 0.82 /yr for an untrawled community (Hiddink
    2017).
  - That means about 4-6 yr from half to 95 % of K.
- **Long-lived emergent epifauna takes decades:**
  - sea pens, gorgonians and sponges, the "structure" relevant to fish shelter;
  - longevity above 10 yr gives r below about 0.5 /yr.
- The only wind-farm closure with a benthos test (OWEZ, sand, 20 m) found **no community difference after 5 years**.
- The closest analogue to Mareld, the Kattegat no-take zone on Nephrops mud, found **Nephrops biomass more than
  doubled** in about 10 years.
- No recovery data exist for deep (>150 m) Skagerrak mud or Pandalus grounds.

### Cited findings
- **Hiddink et al. 2017** [Primary, FULL in the earlier note].
  - Hiddink JG, Jennings S, Sciberras M, Szostek CL, Hughes KM, Ellis N, Rijnsdorp AD, McConnaughey RA, et al. 2017.
    Global analysis of depletion and recovery of seabed biota after bottom trawling disturbance. *PNAS*
    114:8301-8306. doi [10.1073/pnas.1618858114](https://doi.org/10.1073/pnas.1618858114).
  - r 0.82 /yr (0.42-1.53); 3.6 yr (1.9-6.4) from 0.5K to 0.95K.
- **Pitcher et al. 2022** [Primary, FULL in the earlier note].
  - Pitcher CR, Hiddink JG, Jennings S, Collie J, Parma AM, Amoroso R, Mazor T, Sciberras M, et al. 2022. Trawl
    impacts on the relative status of biotic communities of seabed sedimentary habitats in 24 regions worldwide.
    *PNAS* 119:e2109449119. doi [10.1073/pnas.2109449119](https://doi.org/10.1073/pnas.2109449119).
  - R 0.29 (gravel) to 0.68 (mud) /yr; Skagerrak-Kattegat regional RBS 0.63.
- **Hiddink et al. 2019** [Primary, ABS here; the 5.31 coefficient comes via Rijnsdorp et al. 2020, FULL in the
  earlier note].
  - Hiddink JG, Jennings S, Sciberras M, Bolam SG, Cambiè G, McConnaughey RA, Mazor T, Hilborn R, et al. 2019.
    Assessing bottom trawling impacts based on the longevity of benthic invertebrates. *J. Appl. Ecol.* 56:1075-1084.
    doi [10.1111/1365-2664.13278](https://doi.org/10.1111/1365-2664.13278).
  - Biota living more than 1 yr "decreased by ~9% immediately following a trawl pass".
  - "a 2-3x larger effect on biota living >10 years than on biota living 1-3 years".
  - r scales with 1/longevity (r x L ≈ 5.31).
- **Lindeboom et al. 2011** [Primary, ABS].
  - Lindeboom HJ, Kouwenhoven HJ, Bergman MJN, Bouma S, Brasseur S, Daan R, et al. 2011. Short-term ecological
    effects of an offshore wind farm in the Dutch coastal zone; a compilation. *Environ. Res. Lett.* 6:035101.
    doi [10.1088/1748-9326/6/3/035101](https://doi.org/10.1088/1748-9326/6/3/035101).
  - OWEZ is sandy, about 20 m deep, and was closed to fishing.
  - "no short-term effects on the benthos in the sandy area between the generators".
  - Recruit abundance differences are "most likely to be attributed not to the presence of the farm but to the
    absence of fisheries".
  - "Some fish species, such as cod, seem to find shelter inside the farm". The abstract gives no numbers.
- **Bergman et al. 2015** [Primary, ABS]. This is the only direct wind-farm trawl-exclusion benthos test.
  - Bergman MJN, Ubels SM, Duineveld GCA, Meesters EWG. 2015. Effects of a 5-year trawling ban on the local benthic
    community in a wind farm in the Dutch coastal zone. *ICES J. Mar. Sci.* 72:962-972.
    doi [10.1093/icesjms/fsu193](https://doi.org/10.1093/icesjms/fsu193).
  - "Multivariate analysis did not reveal any difference ... with respect to abundance, biomass, and production after a
    5-year closure".
  - Diversity was higher, and Spisula was more abundant, with Tellina and Ensis larger.
  - Closures "will not automatically imply that restoration of benthic assemblages can be expected within a relatively
    short period of years".
- **Sköld et al. 2022, Kattegat no-take zone** [Primary, ABS verbatim; full-text quotes are second-hand].
  - Sköld M, Börjesson P, Wennhage H, Hjelm J, Lövgren J, Ringdahl K, et al. 2022. A no-take zone and partially
    protected areas are not enough to save the Kattegat cod, but enhance biomass and abundance of the local fish
    assemblage. *ICES J. Mar. Sci.* 79:2231-2246. doi [10.1093/icesjms/fsac152](https://doi.org/10.1093/icesjms/fsac152).
  - Design: a 426 km2 year-round no-take zone on Nephrops mud from 2009; surveys 2008-2021, 470 hauls.
  - "The biomass of N. norvegicus in the NTZ has more than doubled over time relative to both NTZ control and PPA
    north". Dab and lemon sole also increased.
  - For cod, see section B.
- **Lambert et al. 2014** [Primary, ABS]. Isle of Man scallop grounds, coarse substrate.
  - Lambert GI, Jennings S, Kaiser MJ, Davies TW, Hiddink JG. 2014. Quantifying recovery rates and resilience of seabed
    habitats impacted by bottom fishing. *J. Appl. Ecol.* 51:1326-1336.
    doi [10.1111/1365-2664.12277](https://doi.org/10.1111/1365-2664.12277).
  - Recovery "<1 to >10 years, depending on the group".
  - Large species recovered faster "when conspecifics were abundant within a radius of 6 km". Recovery inside a closure
    therefore depends on remnant source populations.
- **Kaiser et al. 2018** [Primary, ABS]. Lyme Bay reef closure.
  - Kaiser MJ, Hormbrey S, Booth JR, Hinz H, Hiddink JG. 2018. Recovery linked to life history of sessile epifauna
    following exclusion of towed mobile fishing gear. *J. Appl. Ecol.* 55:1060-1070.
    doi [10.1111/1365-2664.13087](https://doi.org/10.1111/1365-2664.13087).
  - Soft corals and king scallops recovered in less than 3 yr.
  - Ross coral and pink sea fan had projected recovery of 17-20 yr; low-dispersal species "could take >20 years".
- **Sea pens on mud** [Primary, ABS].
  - Chimienti G, Ferrigno F, et al. 2026. *Mar. Pollut. Bull.* 222:118906.
    doi [10.1016/j.marpolbul.2025.118906](https://doi.org/10.1016/j.marpolbul.2025.118906).
    - Adriatic, before/after with no control.
    - Funiculina density rose 466 %, "from 0.0047 colonies m-2 ... to 0.0266 colonies m-2 after a two years-long
      fishing ban".
  - Martinelli M, et al. 2023. *Fishes* 8:347. doi [10.3390/fishes8070347](https://doi.org/10.3390/fishes8070347).
    Density rose in the Pomo Pit closure; no numbers.
  - Funiculina longevity of about 30 yr is SNIP (IUCN).
  - Caution: the often-quoted "sea pens re-establish within 72 h" (Eno et al. 2001, *ICES J. Mar. Sci.* 58:11-20,
    doi [10.1006/jmsc.2000.0984](https://doi.org/10.1006/jmsc.2000.0984)) concerns creels and pots, not trawls.
- **Coolen et al. 2022, Degraer et al. 2020, Dannheim et al. 2020** [Primary, ABS]. These are syntheses of wind-farm
  benthos. They describe fouling on hard structures and season-driven soft-sediment change. They give no recovery rate
  for trawl exclusion.
  - Coolen JWP, et al. 2022. *J. Environ. Manage.* 315:115173.
    doi [10.1016/j.jenvman.2022.115173](https://doi.org/10.1016/j.jenvman.2022.115173).
  - Degraer S, et al. 2020. *Oceanography* 33(4):48-57.
    doi [10.5670/oceanog.2020.405](https://doi.org/10.5670/oceanog.2020.405).
  - Dannheim J, et al. 2020. *ICES J. Mar. Sci.* 77:1092-1108.
    doi [10.1093/icesjms/fsz018](https://doi.org/10.1093/icesjms/fsz018).

### [Own] Recovery time in the park
- Logistic recovery from H0 to 0.95: t = ln[(0.95/0.05)/(H0/(1 - H0))]/r.
- The park's present equilibrium is H0 ≈ 0.49 at d 0.10, r 0.82, or 0.38 at Pitcher's mud r 0.68 (see
  `quantitative_links.md`). Recovery to 0.95 then takes:

  | r (/yr) | Basis | Time (yr) |
  |---|---|---|
  | 0.82 | the model's own benthos (library growth_rate 0.000562 x 1460) | 3.6 |
  | 0.68 | Pitcher mud | 4.3-5 |
  | 0.48 | Pitcher mud, lower CL | 6.1 |
  | ≈ 0.18 | sea-pen-like taxa with longevity about 30 yr (5.31/30) | ≥ 15 |

- So within a 25-30 yr park lifetime the community biomass index recovers almost fully. The tall sea-pen "structure"
  probably recovers only partly.

## B. Do wind farms or trawl closures raise juvenile gadoids or their recruitment?

### Takeaway
- **Attraction is well documented. Production is at most local. No regional recruitment effect has ever been
  shown.**
- Every gadoid aggregation result is from **fixed** foundations with rock scour protection on shallow (<40 m) sand,
  within metres to tens of metres of the structure.
- The one North Sea **floating** farm study, Hywind Tampen at 266-300 m, saw no clear fish effect.
- The closest analogue to Mareld (Kattegat NTZ, Nephrops mud, 426 km2) did **not** raise cod, despite two good year
  classes.
- No source gives the share of a stock held at a wind farm.

### Cited findings: fixed-foundation wind farms (attraction)
- **Reubens et al. 2013, Fish. Res.** [Primary, FULL]. Belgian monopiles and jackets.
  - Reubens JT, Braeckman U, Vanaverbeke J, Van Colen C, Degraer S, Vincx M. 2013. Aggregation at windmill artificial
    reefs: CPUE of Atlantic cod (*Gadus morhua*) and pouting (*Trisopterus luscus*) at different habitats in the
    Belgian part of the North Sea. *Fish. Res.* 139:28-34.
    doi [10.1016/j.fishres.2012.10.011](https://doi.org/10.1016/j.fishres.2012.10.011).
  - Line fishing 1-10 m from the turbines. Cod CPUE (fish per angler-hour) was 4.6 ± 0.9 at turbines, 1.1 ± 0.2 at
    wrecks and 0.05 ± 0.03 on sand.
  - Peak 13.6 in July; near zero in winter (spawning migration).
  - "This argument only applies for fish aggregation, not for production."
- **Reubens et al. 2013, Mar. Environ. Res. 90:66-74** [Primary, ABS].
  doi [10.1016/j.marenvres.2013.05.013](https://doi.org/10.1016/j.marenvres.2013.05.013).
  - "Mainly I- and II-group Atlantic cod" were found at the turbines; 0- and I-group dominated for pouting.
  - No evidence of an ecological trap.
- **Reubens et al. 2013, Mar. Environ. Res. 90:128-135** [Primary, ABS].
  doi [10.1016/j.marenvres.2013.07.001](https://doi.org/10.1016/j.marenvres.2013.07.001).
  - 22 tagged cod showed strong summer-autumn residency.
- **Reubens, Degraer & Vincx 2014** [Primary, SNIP], and the Reubens PhD summary [Report, FULL, VLIZ].
  - Reubens JT, Degraer S, Vincx M 2014, *Hydrobiologia* 727:121-136,
    doi [10.1007/s10750-013-1793-1](https://doi.org/10.1007/s10750-013-1793-1).
  - Local production "can be assumed", but "no changes in productivity were observed on a regional scale".
- **Bergström et al. 2013, Lillgrund, Sweden** [Primary, ABS].
  - Bergström L, Sundqvist F, Bergström U. 2013. *Mar. Ecol. Prog. Ser.* 485:199-210.
    doi [10.3354/meps10344](https://doi.org/10.3354/meps10344).
  - Cod were denser near the foundations, "mainly ... local changes in distribution rather than to immigration or
    increased local productivity".
- **Stenberg et al. 2015, Horns Rev 1** [Primary, ABS].
  - Stenberg C, et al. 2015. *Mar. Ecol. Prog. Ser.* 528:257-265.
    doi [10.3354/meps11261](https://doi.org/10.3354/meps11261).
  - Whiting was most abundant away from the turbines.
- **van Hal et al. 2017** [Primary, ABS].
  - van Hal R, Griffioen AB, van Keeken OA. 2017. *Mar. Environ. Res.* 126:26-36.
    doi [10.1016/j.marenvres.2017.01.009](https://doi.org/10.1016/j.marenvres.2017.01.009).
  - Cod and pouting are attracted, but the structures are "only used temporarily for shelter or feeding". Whiting
    prefers sand.
- **Gimpel et al. 2023** [Primary, ABS].
  - Gimpel A, et al. 2023. *Sci. Total Environ.* 878:162902.
    doi [10.1016/j.scitotenv.2023.162902](https://doi.org/10.1016/j.scitotenv.2023.162902).
  - Cod feed at the piles; spawning is "suggested". This is the only pointer toward production, and it has no numbers.
- **Methratta & Dardick 2019** [Primary, ABS].
  - Methratta ET, Dardick WR. 2019. Meta-analysis of finfish abundance at offshore wind farms. *Rev. Fish. Sci.
    Aquac.* 27:242-260. doi [10.1080/23308249.2019.1584601](https://doi.org/10.1080/23308249.2019.1584601).
  - The overall effect is positive; the abstract gives no number.
  - Methratta 2021 (*Front. Mar. Sci.* 8:674594, doi [10.3389/fmars.2021.674594](https://doi.org/10.3389/fmars.2021.674594))
    finds the effect largest near the foundations, fading with distance.
- **Wilber et al. 2022, Block Island** [Primary, ABS].
  - Wilber DH, et al. 2022. *ICES J. Mar. Sci.* 79:1274-1288.
    doi [10.1093/icesjms/fsac051](https://doi.org/10.1093/icesjms/fsac051).
  - No significant change for the common species. The minimum detectable effect was 40-63 %.
- **Buyse et al. 2022, 2023** [Primary, ABS]. Plaice in Belgian farms.
  - Plaice was 4x higher on scour protection (radius 16.5 m).
  - Inside the farm the fish were larger and more often female, but showed no gain in condition or fecundity. This is a
    fishing-refuge effect.
  - Buyse 2022: doi [10.1093/icesjms/fsac107](https://doi.org/10.1093/icesjms/fsac107).
  - Buyse 2023: doi [10.1016/j.scitotenv.2022.160730](https://doi.org/10.1016/j.scitotenv.2022.160730).

### Cited findings: floating wind
- **Hestetun et al. 2024** [Report, FULL].
  - Hestetun JT, Mayers K, Mugu S, Dahlgren TG. 2024. *Environmental DNA monitoring of fish communities at the Hywind
    Tampen floating offshore wind farm*. NORCE report 4-2024, ISBN 978-82-8408-350-6.
  - "we could not see clear influence of the FOWF, either positive or negative".
  - "the floating construction ... and comparatively high water depth in the area (266-300 m) could very well mean that
    any influence on demersal communities is light".
  - Caveat: turbines were still being installed during sampling.
- **Preprints (not peer reviewed)**:
  - Andrews & Brooks 2025, bioRxiv doi 10.1101/2025.10.20.683384. Saithe, tusk and ling at Hywind Tampen: effects
    smaller than natural variation.
  - Tsuchida et al. 2026, bioRxiv. Japan; no acoustic biomass difference.
- No study of fish at mooring lines or anchors was found.

### Cited findings: trawl closures and gadoids (production test)
- **Kattegat NTZ** (Sköld et al. 2022, above).
  - Cod: "neither cod abundance nor biomass showed a significant increase ... despite two relatively strong year
    classes in 2012 and 2013".
  - Reasons given:
    - discards were about 63 % of catch weight;
    - the zone is "too small to protect cod";
    - the effort rule was dropped in 2016.
  - Lindegren M, Eero M. 2013. *Mar. Ecol. Prog. Ser.* 490:223-232, doi [10.3354/meps10437](https://doi.org/10.3354/meps10437)
    [Primary, ABS]: "only a drastic reduction in total mortalities ... may promote a recovery".
  - Note: the Kattegat NTZ was a cod *spawning* area. The closure aimed at mortality, not habitat.
- **Georges Bank closed areas** [Primary, SNIP].
  - Murawski SA, Brown R, Lai H-L, Rago PJ, Hendrickson L. 2000. Large-scale closed areas as a fishery-management tool
    in temperate marine systems: the Georges Bank experience. *Bull. Mar. Sci.* 66:775-798. No DOI.
  - About 17,000 km2 closed; scallops rose 14-fold. The closures gave "less year-round protection to migratory age
    groups of Atlantic cod and haddock".
  - Murawski et al. 2005 (*ICES J. Mar. Sci.* 62:1150-1167, doi [10.1016/j.icesjms.2005.04.005](https://doi.org/10.1016/j.icesjms.2005.04.005))
    [ABS]: 25 % of groundfish effort lay within 5 km of the boundaries ("fishing the line"). This is relevant to
    displaced effort around a park.
- **Plaice Box** [Primary, ABS].
  - Beare D, Rijnsdorp AD, et al. 2013. Evaluating the effect of fishery closures: lessons learnt from the Plaice Box.
    *J. Sea Res.* 84:49-60. doi [10.1016/j.seares.2013.04.002](https://doi.org/10.1016/j.seares.2013.04.002).
  - Beam-trawl effort fell by more than 90 % in a nursery closure, yet "plaice landings and biomass declined". Juveniles
    moved deeper.
- **Small Norwegian Skagerrak MPAs** [Primary, ABS].
  - Moland E, Olsen EM, Knutsen H, et al. 2013. Lobster and cod benefit from small-scale northern marine protected
    areas. *Proc. R. Soc. B* 280:20122679. doi [10.1098/rspb.2012.2679](https://doi.org/10.1098/rspb.2012.2679).
  - Cod density and size rose; "on average 5 cm longer".
  - This is a positive case, but for small resident **coastal** cod inside the fjords. The mechanism is fishing
    mortality, not offshore habitat.

### [Own] Implications for the model
1. **The only park effect on the seabed is trawl exclusion.** A floating park has no hard-substrate reef, so the
   Reubens and Degraer reef numbers should not be used. A reef-attraction term would also be redistribution, not new
   K.
2. **Benthos recovery inside the park is real and fast for biomass** (r 0.5-0.8 /yr). The model already represents
   it: setting the bottom-trawling map to 0 inside the park lets `benthic_community` recover at its own r 0.82 /yr.
   The prey pathway to gadoids is therefore already in the engine.
3. **No field study has measured gadoid recruitment rising after a trawl closure or a wind farm.** The direct mud
   analogue (Kattegat NTZ) is negative for cod. There is nothing to calibrate a habitat-to-K gain against.
4. **Effort displacement.** Georges Bank shows that effort moves to the boundary. The `projekt` effort map should
   decide explicitly whether the park's 27 % of effort is removed or redistributed (`quantitative_links.md`, last
   section).
