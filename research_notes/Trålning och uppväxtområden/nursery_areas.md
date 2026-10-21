# Nursery and juvenile habitat of cod, haddock, whiting and saithe, and the share the Mareld box could hold (Q3)

Labels:
- **[Primary]**: peer-reviewed literature.
- **[Report]**: ICES, EIA and grey literature.
- **[Own]**: my computation or inference.

Fetch status:
- FULL: full text read.
- ABS: verbatim abstract.
- ABS-P: abstract seen only as a paraphrase.
- SNIP: search snippet only.
- BIB: bibliographic record only.

DOIs were checked against Crossref.

## Takeaway

- **Cod, whiting and saithe have coastal nurseries.**
  - Cod settles in shallow eelgrass and the sheltered Skagerrak coast.
  - Whiting uses sheltered coast and shelf down to about 200 m.
  - Saithe uses inshore water and sea lochs for its first 2-4 years.
  - The offshore trench slope is at most a transit, feeding and overwintering area for their older juveniles (age 1-3+).
- **Haddock is the exception.** It is 54 % of today's pool.
  - The Norwegian Deep surveys of 1984-87 found 0- and 1-group haddock *deeper* than adults, with substantial numbers
    at 150-300 m. They were "comparatively abundant in the Skagerrak".
  - Below 300 m haddock were nearly absent.
  - So the 150-300 m part of the box, which holds most of the park and most of the trawling, is genuine juvenile
    haddock habitat.
  - It is a feeding habitat on mud with no known structural requirement: haddock show "no sediment preference".
- **Hake (1 %).** An MSc-level hint says the smallest hake concentrate in the Skagerrak. This is weak.
- **No published Skagerrak nursery area in km2 exists for any of the four species.** The share estimate below is an
  order-of-magnitude inference.
  - **Grid:** roughly 0.5-1 % of the haddock stock's juvenile habitat, and ≈ 0 % (< 0.1 %) of the cod, whiting and saithe
    settlement nurseries.
  - **Park:** about one third of the grid value.
  - **Pool-weighted, by today's biomass mix:** grid about 0.3-0.6 %, park about 0.1-0.2 %.
- **Main data gap.** NS-IBTS standard stations reach at most 250 m, so no modern depth-resolved juvenile data cover
  180-450 m at the site. A DATRAS extraction (Q1 age-1 and Q3 age-0 by haul depth in rectangles 44G0, 45G0, 44G1 and
  45G1), plus fish bycatch from the IMR Skagerrak shrimp survey, is the direct way to settle it.

## Site depth and substrate [Own, Report]

**Depth.** I recomputed the grid depth from the EMODnet Bathymetry DTM ("emodnet__mean", the same WCS request as
`make_djup.py`), sampled at the cell centres.
- This was a scratch run; `djup.png` and the repo were not touched. It correlates 0.99998 with `djup.png`.
- Grid depth: max 533 m, min 43 m, mean 168 m. Park area-weighted mean 232 m.

| Depth band | Grid km2 | Park km2 | Share of grid trawl effort | Mean effort (MW-h/km2/yr) |
|---|---|---|---|---|
| < 50 m | 25 | 0 | 0.3 % | 1.4 |
| 50-100 m | 807 | 21 | 11.7 % | 1.6 |
| 100-150 m | 1,079 | 39 | 14.2 % | 1.4 |
| 150-200 m | 689 | 77 | 25.6 % | 4.0 |
| 200-300 m | 689 | 321 | 41.2 % | 6.5 |
| 300-400 m | 193 | 51 | 6.1 % | 3.4 |
| > 400 m | 118 | 12 | 0.9 % | 0.8 |

- Two thirds of the trawl effort is on the 150-300 m slope. That is the Pandalus ground, and it is also the haddock
  juvenile band.
- **By park:**
  - Mareld: 369 km2 in the grid; full cells 191-426 m; mean 264 m; 291 km2 at 200-300 m.
  - Poseidon Nord: 152 km2; full cells 66-219 m; mean 157 m; 101 km2 at 100-200 m, 21 km2 shallower than 100 m.
  - These match the EIAs: Mareld 180-450 m on clay/mud (`MKB Mareld.pdf`, 5.2, 9.1); Poseidon Nord about 55-230 m on
    mostly soft bottom (`Poseidon/Poseidon MKB.pdf`, 1.4) [Report].
  - Poseidon Nord is therefore the only part of the park area inside whiting's 10-200 m range and haddock's shallower
    80-150 m band. It holds 5 % of the grid's trawl effort, against Mareld's 22 %.

**Substrate.**
- Mud and clay, probably accumulation bottom, with sea pens, Nephrops and Pandalus (EIA 9.1).
- Bratten, west and north-west of the park, has rocky ravines with gorgonians and sponges.
- Bergstad 1990 (below) puts a "rather sharp boundary at about the 200 m isobath" between Norwegian Deep and plateau
  assemblages. The Skagerrak below 300 m is characterised by greater argentine and roundnose grenadier, not gadoids.

**Sampling caveat** [Report, FULL].
- ICES 2020, *Manual for the North Sea International Bottom Trawl Survey*, Series of ICES Survey Protocols SISP 10-IBTS
  10 Rev. 11, doi [10.17895/ices.pub.7562](https://doi.org/10.17895/ices.pub.7562): "The maximum fishing depth for
  standard stations in the entire survey area is 250 m".
- The Swedish coastal trawl survey, Norwegian Flødevigen beach seine and Swedish tagging work are all coastal.
- Only the University of Bergen Norwegian Deep surveys of 1984-87 (Bergstad, Albert) trawled the slope systematically.
  They predate the gadoid declines and the hake expansion.

## Cod

Nursery evidence:
- **Pihl L, Ulmestrand M. 1993.** Migration pattern of juvenile cod (*Gadus morhua*) on the Swedish west coast.
  *ICES J. Mar. Sci.* 50:63-70. doi [10.1006/jmsc.1993.1007](https://doi.org/10.1006/jmsc.1993.1007). [Primary, ABS-P]
  - Juveniles were tagged at 5-10 m and stayed near the release sites through summer and autumn.
  - At age 2 (30-50 cm) they moved offshore to deeper water, mainly south and west.
- **Pihl L, Baden S, Kautsky N, Rönnbäck P, Söderqvist T, Troell M, Wennhage H. 2006.** Shift in fish assemblage
  structure due to loss of seagrass *Zostera marina* habitats in Sweden. *Estuar. Coast. Shelf Sci.* 67:123-132.
  doi [10.1016/j.ecss.2005.10.016](https://doi.org/10.1016/j.ecss.2005.10.016). [Primary, SNIP of abstract]
  - Skagerrak eelgrass "decreased by 60 % over two decades".
  - "juvenile 0-group cod density was reduced by 96 % at sites where Z. marina had disappeared".
  - This is the strongest *habitat-loss to juvenile cod* number in the region. It concerns coastal eelgrass, not
    trawling.
- **Svedäng H, Svenson A. 2006.** Cod *Gadus morhua* L. populations as behavioural units: inference from time series
  on juvenile abundance in the eastern Skagerrak. *J. Fish Biol.* 69:151-164.
  doi [10.1111/j.1095-8649.2006.01272.x](https://doi.org/10.1111/j.1095-8649.2006.01272.x). [Primary, ABS]
  - Fish of offshore origin settle on the coast and "eventually leave the coast for the open Skagerrak or the North
    Sea".
- **Larval supply** [Primary, ABS]:
  - Knutsen H, André C, Jorde PE, Skogen MD, Thuróczy E, Stenseth NC. 2004. Transport of North Sea cod larvae into the
    Skagerrak coastal populations. *Proc. R. Soc. B* 271:1337-1344.
    doi [10.1098/rspb.2004.2721](https://doi.org/10.1098/rspb.2004.2721).
  - Stenseth NC, Jorde PE, Chan K-S, et al. 2006. Ecological and genetic impact of Atlantic cod larval drift in the
    Skagerrak. *Proc. R. Soc. B* 273:1085-1092. doi [10.1098/rspb.2005.3290](https://doi.org/10.1098/rspb.2005.3290).
  - Jonsson PR, Corell H, André C, Svedäng H, Moksnes P-O. 2016. *Fish. Oceanogr.* 25:210-228,
    doi [10.1111/fog.12146](https://doi.org/10.1111/fog.12146): the Kattegat now supplies 30 % of the larvae recruiting
    to the Swedish Skagerrak, against 72 % in the 1970s.
  - Henriksson S, et al. 2023. *ICES J. Mar. Sci.* 80:145-157,
    doi [10.1093/icesjms/fsac220](https://doi.org/10.1093/icesjms/fsac220): mixed ecotypes along the coast; "the offshore
    ecotype dominates in offshore areas".
  - North Sea larvae drift into the Skagerrak and **settle on the coast**.
- **Fromentin JM, Stenseth NC, Gjøsæter J, et al. 1997.** *Mar. Ecol. Prog. Ser.* 155:209-222,
  doi [10.3354/meps155209](https://doi.org/10.3354/meps155209). [Primary, ABS]
  - Beach seine, 1919-94: juvenile cod and whiting are "more abundant in sheltered areas".

Offshore and deep evidence:
- **ICES-FishMap species factsheet, Cod** (ICES c. 2006, IBTS-based). [Report, FULL]
  - "In Skagerrak/Kattegat, 1-group fish is most abundant".
  - "Catches have been reported from the deepest parts of the Norwegian Deeps at 500 m".
  - Juveniles are "often found in more shallow waters than the adults".
- **Bergstad OA. 1991a.** Distribution and trophic ecology of some gadoid fish of the Norwegian Deep. 1. Accounts of
  individual species. *Sarsia* 75:269-313. doi [10.1080/00364827.1991.10413455](https://doi.org/10.1080/00364827.1991.10413455).
  [Primary, ABS]
  - "most of the species utilize the area as late juveniles and adults, some also as young juveniles". The abstract
    does not say which species.
- **Ellis JR, Milligan SP, Readdy L, Taylor N, Brown MJ. 2012.** *Spawning and nursery grounds of selected fish
  species in UK waters*. Sci. Ser. Tech. Rep. Cefas Lowestoft 147. [Report, FULL]
  - It covers UK waters only, with no Skagerrak polygons.
  - Juvenile cod (2-22 cm) were caught at 7-450 m, so deep occurrence is possible. Cod "utilises estuarine habitats
    and other coastal waters as nursery grounds".

**Verdict [Own]**
- The 0-group nursery is coastal and shallow (< 30 m, eelgrass). The grid has 25 km2 shallower than 50 m and no
  coastline.
- The box is plausibly a transit and feeding area for 1-2 group cod, but no density data exist for 150-450 m.
- **Grid share of the cod settlement nursery: ≈ 0. Share of the 1-2 group feeding range: unknown, probably ≤ 1 %.**

## Haddock

- **Albert OT. 1991.** *Distribution and trophic ecology of haddock Melanogrammus aeglefinus in the Norwegian Deep.*
  ICES CM 1991/G:45, 15 pp. (ICES CM archive). [Report, FULL]
  - 13 bottom-trawl surveys, 1984-87. Subarea 4 is the Skagerrak.
  - "An age-specific depth distribution was found, with the 0 and I group distributed in deeper parts of the trough than
    their elder conspecifics."
  - "catches fell sharply deeper than 200 m ... the species was generally not found beneath 300 m."
  - "In subarea 4 (Skagerrak), however, the proportion of zero-catches was low."
  - Autumn catches were 65-98 % 0-group.
  - Numbers measured per depth band (Fig. 4, subareas pooled; read from the figure):

    | Depth band | Winter | Summer | Autumn |
    |---|---|---|---|
    | 80-149 m | 1,943 | 898 | 1,196 |
    | 150-199 m | 1,683 | 648 | 1,096 |
    | 200-300 m | 760 | 143 | 292 |

  - At 200-300 m the catch was almost all the two youngest year classes.
  - Caveat: these are measured numbers, not cpue, and the shallowest stratum is 80 m.
- **Albert OT. 1994.** Ecology of haddock (*Melanogrammus aeglefinus* L.) in the Norwegian Deep. *ICES J. Mar. Sci.*
  51:31-44. doi [10.1006/jmsc.1994.1004](https://doi.org/10.1006/jmsc.1994.1004). [Primary, ABS]
  - "seasonal migration between relatively shallow areas in summer and deeper parts in winter. The distribution became
    gradually more shallow with increasing age".
- **ICES-FishMap factsheet, Haddock** (c. 2006). [Report, FULL]
  - "juveniles tend to be more abundant in the Skagerrak than adults".
  - Habitat envelope 75-125 m, above 6 °C, salinity above 34.5, **no sediment preference**.
  - "In the Norwegian Deeps, haddock is generally not found in waters more than 300 m deep."
- **Wright PJ, Tobin D, Gibb FM, Gibb IM. 2010.** Assessing nursery contribution to recruitment: relevance of closed
  areas to haddock *Melanogrammus aeglefinus*. *Mar. Ecol. Prog. Ser.* 400:221-232.
  doi [10.3354/meps08384](https://doi.org/10.3354/meps08384). [Primary, ABS] Scottish waters only.
- No source describes a coastal or vegetated haddock nursery.

**Verdict [Own]**
- The 150-300 m band of the box (1,378 km2 in the grid, 398 km2 in the park) is real juvenile haddock habitat, at
  least as of the 1980s.
- The juveniles are plausibly spawned in the North Sea and advected in. That is my inference; no source states it.
- Haddock juveniles eat benthic invertebrates (polychaetes, crustaceans, echinoderms; general knowledge, not fetched
  here, though Albert 1991/1994 covers diet). A benthos change is therefore a *food* channel for them. No study links haddock habitat to emergent epifauna or shelter on mud.

## Whiting

- **ICES-FishMap factsheet, Whiting.** [Report, FULL]
  - "commonly found near the bottom in waters from 10 to 200 m".
  - "catches in the Skagerrak/Kattegat are dominated by 1-year-old whiting".
- **Fromentin et al. 1997** (above): juvenile whiting are "more abundant in sheltered areas" on the Norwegian Skagerrak
  coast.
- **Ellis et al. 2012**: coastal and estuarine nurseries in UK waters.
- **Bergstad 1991a**: whiting are among the gadoids using the Norwegian Deep; the life stage is not specified.
- Horns Rev and Dutch wind-farm studies find whiting preferring open sand away from structures
  (`benthic_recovery_and_wind_farms.md`).

**Verdict [Own]**
- Shelf and coastal nursery. The box's < 200 m part (2,600 km2) is within the 1-group range, the park mostly is not.
- No evidence either way at 200-450 m.
- **Grid share of the whiting nursery: small, probably < 0.5 %.**

## Saithe

- **ICES-FishMap factsheet, Saithe.** [Report, FULL]
  - "juveniles occupy inshore habitats and recruit to the exploitable stock at an age of 2-3 years old".
  - "O-group saithe ... move to sea lochs along the Scottish and Scandinavian coastline ... remain there close to the
    shore during the first three years of life".
  - Larger saithe (≥ 30 cm) have a "clear zone of high abundance along the edge of the Norwegian Deep".
- **UK DTI SEA5, Appendix 4** (2004). [Report, FULL] "Young saithe remain in these nursery areas up to the age of 3 or 4
  before slowly migrating into deeper water"; adults occur at 80-450 m.
- **Bergstad OA. 1990.** Ecology of the fishes of the Norwegian Deep: distribution and species assemblages. *Neth. J.
  Sea Res.* 25:237-266. doi [10.1016/0077-7579(90)90025-C](https://doi.org/10.1016/0077-7579(90)90025-C). [Primary, ABS]
  - "The western and southern slopes appear to be feeding and overwintering areas ... particularly populations of saithe
    ... and Norway pout".
- **Bergstad OA. 1991b.** *Sarsia* 75:315-325, doi [10.1080/00364827.1991.10413456](https://doi.org/10.1080/00364827.1991.10413456).
  [Primary, ABS] "Small saithe" are major euphausiid predators in the Deep.

**Verdict [Own]**
- The nursery is coastal. The slope is a feeding and overwintering area for fish of 30 cm and more.
- **Grid share of the saithe nursery: ≈ 0.**

## Hake (1 % of the pool)

- Werner K-M. 2015. MSc thesis, University of Bergen, hdl 1956/10408. [Report, ABS]
  - "Smallest individuals from 6.5-10.5 cm were mainly located in the Skagerrak".
  - Based on IMR survey data, 2006-14.
- Baudron AR, Fernandes PG. 2015. Adverse consequences of stock recovery: European hake, a new "choke" species under a
  discard ban? *Fish Fish.* 16:563-575. doi [10.1111/faf.12079](https://doi.org/10.1111/faf.12079). [Primary, ABS]
  - Sharp North Sea expansion, 2004-11.
- **Verdict:** a weak hint that the deep Skagerrak may matter for juvenile hake. With 1 % of the pool it does not move
  the pooled share.

## [Own] Order-of-magnitude share of the regional nursery

Every number here is inference. No source gives nursery area in km2 for these stocks in the Skagerrak.

Areas:
- An ICES rectangle at 58 N is about 3,300 km2. The Skagerrak (IIIa north) is of the order of 3 x 10^4 km2. That
  figure is my approximate value and was not fetched.
- The juvenile ranges of the North Sea stocks span 10^5 km2 or more. For haddock that is the northern North Sea
  plateau at 75-200 m plus the Skagerrak and Norwegian Deep margins.

Per-species estimates:

| Species (pool weight) | Juvenile-habitat area in the grid | Share of the stock's juvenile habitat: grid | Share: park | Basis |
|---|---|---|---|---|
| Haddock (0.54) | 1,378 km2 at 150-300 m, plus part of the 100-150 m band | 0.5-1 % (range 0.2-2 %) | 0.2-0.4 % | Albert 1991; 10^5-3 x 10^5 km2 juvenile range |
| Whiting (0.27) | at most 2,600 km2 under 200 m, not a known hotspot | 0.1-0.5 % | < 0.05 % | FishMap 10-200 m; coastal nursery |
| Cod (0.06) | settlement nursery about 0 | < 0.1 % (1-2 group feeding: ≤ 1 %) | ≈ 0 | coastal eelgrass; offshore move at age 2 |
| Saithe (0.12) | about 0 | ≈ 0 | ≈ 0 | coastal and sea-loch nursery to age 3-4 |
| Hake (0.01) | unknown | unknown, maybe a few % | unknown | MSc thesis |
| **Pool, weighted** | | **≈ 0.3-0.6 %** (upper bound about 1.5 %) | **≈ 0.1-0.2 %** (upper bound about 0.5 %) | |

Caveats:
- Weighting by today's biomass mix assumes each species' share of K is proportional to its share of biomass.
- Recruitment of these stocks is set over 10^5 km2 by larval drift, temperature and spawning stock. One 3,600 km2 box
  cannot hold more than a few percent of any of them, even if it were prime habitat.
