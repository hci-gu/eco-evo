# Zooplankton refuge from herring: how much copepod biomass is really out of reach? (Mareld, ~100 m)

Model state checked (supplied by the caller; library values re-read from `fgconfig/fg_library.yaml`
and `lib/environments/ecosystem_env/interactions.py`, 2026-10-15):

- `zooplankton` = mesozooplankton, ~2/3 small neritic copepods (Pseudocalanus, Paracalanus, Temora,
  Acartia, Oithona), ~1/3 Calanus. Decision maker: Eat / Move / Rest, Rest = HIDE.
- Visible fraction to a predator: `visible = 1 - pi_rest * (1 - floor)` (interactions.py docstring);
  `pelagic_fish_preys_on_zooplankton.visibility_floor = 0.2` (also the FG's own floor).
- Trained policy rests/hides 81-88 % of the time all year (eat 12-23 %).
- `feeding_cost 4.0` (feeding costs 4x resting, resume section 135).
- Herring -> zooplankton: `dark_ratio 0.1`, `light_threshold_deg -8` (night attack rate = 0.1 x day;
  daylight calendar), `predation_mortality 5.24` /yr, `handling_time 25`. The library comment uses
  Kd ~0.15 m-1 for the water the herring feed in.
- Zooplankton M1 = 0.0134 per tick = 0.0134 x 1460 = 19.6 /yr.

Conventions: **[F]** fulltext read, **[A]** abstract read, **[S]** search snippet only (NOT verified),
**[2nd]** secondary (read in another document or a sibling note), **[D]** computed by me.
Each question is split into *Literature*, *Calculation* and *Inference*.

---

## Sources and how they were read

| Source | Read as | Link |
|---|---|---|
| Daro 1988, Bull Mar Sci 43:710, North Sea summer 1985, copepod vertical distribution and diel grazing | [F] (OCR text layer, letter-spaced) | https://www.vliz.be/imisdocs/publications/263496.pdf |
| Irigoien, Conway & Harris 2004, MEPS 267:85, DVM of the whole plankton community, Irish Sea (central stations 93-130 m) | [F] | https://doi.org/10.3354/meps267085 (PDF https://www.int-res.com/articles/meps2004/267/m267p085.pdf) |
| Bernreuther et al. 2013, MEPS 489:177, herring/sprat vertical prey selectivity, Bornholm Basin (~90 m), June 2001 | [F] (results and discussion) | https://doi.org/10.3354/meps10405 |
| Tiselius & Jonsson 1990, MEPS 66:23, foraging behaviour of six calanoid copepods (Kosterfjorden, Swedish Skagerrak coast) | [F] | https://www.int-res.com/articles/meps/66/m066p023.pdf |
| Bollens & Frost 1991, J Plankton Res 13:1359, Acartia hudsonica DVM response to fish in enclosures | [F] (scanned; pages rendered and read as images) | https://s3.wp.wsu.edu/uploads/sites/1116/2016/07/BollensFrost-DVM-in-zpl-rapid-resp-to-pred-JPR-1991.pdf |
| Dale, Bagøien, Melle & Kaartvedt 1999, MEPS 179:113, overwintering depth of Calanus vs predators | [F] (abstract and introduction) | https://www.int-res.com/articles/meps/179/m179p113.pdf |
| Kaartvedt 2000, ICES JMS 57:1819, Calanus life history vs planktivorous fish, Norwegian Sea | [F] | https://doi.org/10.1006/jmsc.2000.0964 |
| Hirst & Batten 1998, MEPS 171:307, North Sea Calanus DVM (CPR) unrelated to herring | [F] | https://www.int-res.com/articles/meps/171/m171p307.pdf |
| Dale & Kaartvedt 2000, ICES JMS 57:1800, stage-specific DVM of C. finmarchicus | [A] via OpenAlex | https://doi.org/10.1006/jmsc.2000.0961 |
| Bagøien, Kaartvedt, Aksnes & Eiane 2001, L&O 46:1494, overwintering Calanus mortality in Norwegian fjords | [A] via OpenAlex | https://doi.org/10.4319/lo.2001.46.6.1494 |
| Bollens & Frost 1989, L&O 34:1072, Calanus pacificus DVM vs zooplanktivorous fish (Dabob Bay) | [A] via OpenAlex | https://doi.org/10.4319/lo.1989.34.6.1072 |
| Ohman, Frost & Cohen 1983, Science 220:1404, reverse DVM of Pseudocalanus | [A] via OpenAlex | https://doi.org/10.1126/science.220.4604.1404 |
| Hays 2003, Hydrobiologia 503:163, review of DVM | [A] (marineinfo record) | https://doi.org/10.1023/B:HYDR.0000008476.23617.b0 ; https://marineinfo.org/doc/publication/58432 |
| Varpe & Fiksen 2010, Ecology 91:311, light regime and herring foraging on Calanus | [A] via OpenAlex | https://doi.org/10.1890/08-1817.1 |
| Batty, Blaxter & Richard 1990, Mar Biol 107:383, light intensity and herring feeding | [A] (Springer abstract page) | https://doi.org/10.1007/BF01313419 |
| Batty, Blaxter & Libby 1986, Mar Biol 91:371, herring filter-feeding in the dark | [A] (Springer abstract page) | https://doi.org/10.1007/BF00428631 |
| Darbyson, Swain, Chabot & Castonguay 2003, J Fish Biol 63:1235, herring diel feeding | [A] via OpenAlex | https://doi.org/10.1046/j.1095-8649.2003.00247.x |
| Saiz 1994, L&O 39:1566, Acartia tonsa free-swimming behaviour vs food | [A] via OpenAlex | https://doi.org/10.4319/lo.1994.39.7.1566 |
| Durbin, Durbin & Wlodarczyk 1990, MEPS 68:23, diel feeding of Acartia tonsa | [A] via OpenAlex | https://doi.org/10.3354/meps068023 |
| Kiørboe, Saiz, Tiselius & Andersen 2018, L&O 63:308, adaptive feeding behaviour and functional responses | [A] via OpenAlex | https://doi.org/10.1002/lno.10632 |
| Almeda, van Someren Gréve & Kiørboe 2017, Ecosphere 8:e01668, behaviour and predation risk | [A] via OpenAlex | https://doi.org/10.1002/ecs2.1668 |
| Bandara et al. 2021, Biol Rev, 200 years of vertical migration research | [A] via OpenAlex | https://doi.org/10.1111/brv.12715 |
| Onsrud et al. 2004 (Oslofjord herring day/night), Falkenhaug et al. 2022 (Skagerrak Calanus overwintering) | [2nd] via sibling note `winter_food_and_feeding.md` (read there as [F]) | see that note |

**Failed fetches / not obtained (recorded, not used for numbers):**
- Hays 2003 fulltext: closed (Deakin repository "permanent embargo"; Semantic Scholar abstract elided).
- Bollens & Frost 1989 L&O fulltext (Wiley pdfdirect and aslopubs): HTTP 403. The abstract says
  migration strength was measured as the proportion crossing 75 m, but the values were not obtained.
- Varpe & Fiksen 2010 fulltext (Wiley/ESA): HTTP 403. Their visual-range parameters for herring
  (half-saturation light etc.) were therefore NOT obtained.
- Batty et al. 1990 and 1986 fulltexts: Springer paywall (abstracts only).
- Kiørboe et al. 2018 fulltext not attempted after repeated Wiley 403s; abstract only.
- Bagøien, Kaartvedt & Øverås 2000 (Calanus seasonal migration in Oslofjorden, Sarsia): only the
  bibliographic listing found, no abstract. Tiselius 1988 (Skagerrak/Kattegat diel feeding and
  vertical migration): not found.
- **No Skagerrak/Kattegat-specific day/night vertical-distribution study of copepods was obtained.**
  The answer below triangulates from the northern/central North Sea (Daro 1988), the central Irish
  Sea at 93-130 m (Irigoien et al. 2004), the Bornholm Basin at ~90 m (Bernreuther et al. 2013) and
  Norwegian fjords/sea (Kaartvedt, Dale, Bagøien).
- Literature on herring light-saturation for feeding on copepods: not obtained. Only Batty's biting
  threshold (0.001 lx) is sourced.

---

## Q1. What fraction of copepod biomass is out of reach of herring (day/night, summer/winter)?

### Literature
- **Mechanism.** Hays 2003 [A]: for many mesozooplankton "minimising the risk of predation from fish
  seems to be the ultimate factor behind DVM"; migrants "use deep water as a dark daytime refuge".
  Bandara et al. 2021 [A]: DVM lets herbivores feed shallow "during the night when light-dependent
  (visual) predation risk is minimal". In both, the refuge is *darkness at depth by day*, not
  inactivity.
- **North Sea summer: most copepods do not migrate.** Daro 1988 [F], whole North Sea, summer 1985:
  "In most areas, copepods did not vertically migrate and their vertical distribution followed that of
  the phytoplankton which was restricted to the upper 20-30 m." "Many species remain all day and all
  night at approximately the same depth ... When migration occurs, it covers only 10 m and lasts a very
  short time." Acartia: "no significant difference at different times of the day"; along the Norwegian
  coast (her zone D, nearest the Skagerrak) Acartia and Pseudocalanus were "close to the surface at all
  times of day". C. finmarchicus in the northern North Sea: minimal numbers in the upper 10 m, highest
  at 20-40 m. Daro links non-migration to "barely sufficient or limiting food" (~200 mg C m-3).
- **Irish Sea, central stations 93-130 m (Mareld-like depth).** Irigoien et al. 2004 [F]: "Generally,
  more than 60% of the plankton did not perform significant DVM"; "mean depth of 60 to 80% of the
  organisms did not vary by more than 10 m". Acartia, Temora longicornis, Oithona and nauplii showed
  "little or no DVM". Only chaetognaths, Calanus and Pseudocalanus migrated significantly, with
  "great flexibility in both the amplitude and direction" (reverse DVM where chaetognaths stayed
  shallow by day). Caveat by the authors: a single day/night pair can underestimate twilight or
  midnight-sinking migrations.
- **Bornholm Basin (~90 m), June.** Bernreuther et al. 2013 [F]: Temora longicornis, Centropages
  hamatus and Podon had >60 % of their numbers in 0-10 m at night versus 10 % (Temora), 40 %
  (Centropages) and <20 % (Podon) by day. But "T. longicornis and P. acuspes are present in the
  entire water column", and herring and sprat themselves "dwell during the daytime" at 60-80 m.
- **Calanus.** Dale & Kaartvedt 2000 [A], Norwegian and Greenland Seas: CI-CIII "confined to the
  upper 30 m both day and night"; CIV and males non-migratory; CV and females migrated "within a narrow
  depth range"; older stages "consistently occurred in low numbers in the upper 10 m, both day and
  night"; DVM amplitude was largest where herring and mackerel were scarce. Kaartvedt 2000 [F]: herring
  "may stay relatively deep in the water mass by day, but later appear to prey in upper waters through
  the diel cycle"; Calanus overwinters "below depths where these fish can locate prey visually".
  Hirst & Batten 1998 [F]: the reported North Sea correlation between Calanus DVM (CPR, 7 m) and herring
  biomass disappears when day and night are defined by actual sunrise/sunset; not significant.
- **Plasticity.** Bollens & Frost 1989 [A]: strength of Calanus pacificus DVM regressed on abundance
  of fish feeding on it. Bollens & Frost 1991 [F], Acartia hudsonica in 3 m enclosures: migration
  stopped within a day without fish and resumed within 12-24 h when fish were added; noon-vs-midnight
  mean depth differed by only ~0.5-1 m (Fig. 1), and 69 % of the copepods died within that day with
  fish present. Ohman et al. 1983 [A]: Pseudocalanus migrates in reverse to avoid nocturnally rising
  invertebrate predators.
- **Metridia.** Irigoien et al. 2004 [F]: Metridia lucens was among the taxa that moved deeper at night
  at the stratified station. It is a strong migrator but not part of the model's composition
  (~2/3 small neritic copepods + ~1/3 Calanus), so it does not move the accessible fraction.
- **Winter / diapause.** Dale et al. 1999 [F]: oceanic C. finmarchicus overwinter "between 500 and
  1500 m" on the Atlantic side. Sibling note (Falkenhaug et al. 2022 [2nd]): the Skagerrak
  overwintering stock sits in the Norwegian Trench; not in substantial numbers on the shelf. Bagøien
  et al. 2001 [A]: overwintering Calanus in 380-1300 m fjords still died at up to 0.024-0.027 d-1, and
  "predation by M. muelleri and B. glaciale alone could explain" it; Calanus select habitats "below
  depths where planktivorous fish can forage efficiently by sight".
- **Herring by day are deep too.** Bernreuther 2013 [F]: herring at 60-80 m by day. Onsrud et al. 2004
  [2nd]: Oslofjord herring by day in and above the krill layer below ~70 m, at 0-50 m at night.

### Calculation (light at depth at a 100 m site)
Biting threshold 0.001 lx (Batty et al. 1990 [A]: herring "fed by biting (snapping) at light
intensities above a threshold of 0.001 lx"). `E(z) = E0 exp(-Kd z)`. Surface midday illuminance is my
assumption (typical clear-to-overcast values, not sourced): December (sun ~8.5 deg at 58.1 N)
2,000-10,000 lx; June (sun ~55 deg) 20,000-100,000 lx. [D]

| Season | Kd (m-1) | 25 m | 50 m | 75 m | 100 m | depth of 0.001 lx |
|---|---|---|---|---|---|---|
| Dec | 0.10 | 160-820 | 13-67 | 1.1-5.5 | 0.09-0.45 | 145-161 m |
| Dec | 0.15 | 47-240 | 1.1-5.5 | 0.03-0.13 | 0.0006-0.003 | 97-107 m |
| Dec | 0.20 | 13-67 | 0.09-0.45 | 0.0006-0.003 | ~1e-5 | 73-81 m |
| Dec | 0.25 | 4-19 | 0.008-0.04 | ~1e-5 | ~1e-7 | 58-64 m |
| Jun | 0.10 | 1600-8200 | 130-670 | 11-55 | 0.9-4.5 | 168-184 m |
| Jun | 0.15 | 470-2400 | 11-55 | 0.26-1.3 | 0.006-0.03 | 112-123 m |
| Jun | 0.20 | 130-670 | 0.9-4.5 | 0.006-0.03 | ~1e-4 | 84-92 m |
| Jun | 0.25 | 39-190 | 0.08-0.37 | ~5e-4 | ~1e-6 | 67-74 m |

With the library's Kd 0.15 the whole 100 m column is at or above the herring biting threshold at
midday in both seasons; only the bottom ~25 m is dim (<~1 lx). A full dark refuge by day on the
Mareld bank exists only if Kd >= ~0.2 (then below ~75-90 m). The threshold is where biting stops,
not where it is efficient, so the partial refuge starts shallower; with no sourced saturation light
for herring I do not put a number on that.

Fish-predation mortality even in the "safe" deep winter habitat (Bagøien 2001; mesopelagic fish, not
herring): 0.024-0.027 d-1 x 365 = 8.8-9.9 /yr (if sustained), larger than the model's herring -> zooplankton
`predation_mortality` of 5.24 /yr. [D]

### Inference
- From the plasticity studies: the copepod "refuge" is plastic and species-specific. It is not a
  fixed fraction of the population, and in shallow water it buys little (Bollens & Frost 1991).
- On a ~100 m bank the copepod refuge from herring is mainly a **daytime light refuge**, and a weak
  one: the column is mostly lit above threshold at midday (Kd 0.15), herring themselves go to 60-80 m
  by day, and most small copepods (60-80 % of taxa; the 2/3 of the model's biomass) barely migrate in
  shelf seas in summer.
- **Summer, day:** the protected share is roughly the late-stage Calanus and part of the Pseudocalanus
  that sit deeper (Daro: Calanus at 20-40 m; Irigoien: Calanus/Pseudocalanus the only significant
  migrants). My estimate: small copepods 0.8-1.0 accessible, Calanus 0.3-0.7 accessible, giving
  2/3 x (0.8-1.0) + 1/3 x (0.3-0.7) = **0.63-0.90 accessible**.
- **Summer, night:** almost everything is in the upper layer (Bernreuther: >60 % of Temora in 0-10 m),
  i.e. spatially ~0.9-1.0 accessible; the limit is herring vision, not copepod hiding.
- **Winter, day:** the local Calanus overwintering stock is mostly absent (Norwegian Trench), and the
  copepods present are spread through a weakly stratified column (my assumption; no winter profile
obtained for the site) whose lower half is dim at December midday
  (1-5 lx at 50 m, <0.13 lx below 75 m at Kd 0.15). If copepods are spread roughly evenly, ~0.25-0.5
  of them sit in dim-to-dark water: **~0.5-0.8 accessible** by day. Day length (~6.5 h at 58 N in
  December) is a separate limit the daylight calendar already handles.
- **Comparison:** the model's visible fraction is 1 - 0.85 x 0.8 = **0.32** (0.30-0.35 over the 81-88 %
  rest range), and the mechanism is not light-gated, so it applies day and night, summer and winter
  (unless the trained policy has learned a diel rest pattern; not checked here). That is about half of the literature-implied
  daytime accessibility in winter and about 0.35-0.5 of it in summer.

---

## Q2. Copepod time budget: feeding vs not feeding; is "rest = hidden" sensible?

### Literature
- Tiselius & Jonsson 1990 [F], adult females from Kosterfjorden (Swedish Skagerrak), video at 12 C,
  chlorophyll <1 ug/l, Table 1 (% of time mouthparts moving, low/high light): Paracalanus parvus 47/48,
  Pseudocalanus elongatus 55/84, Temora longicornis 99/99, Centropages typicus 58/42, C. hamatus 41/27,
  Acartia clausi 0/0.6. Non-feeding time was spent **sinking and swimming in place**: Paracalanus
  "subsequently sank"; Pseudocalanus "when not suspension feeding, copepods sank at 1.1 mm s-1";
  Acartia "spent 87 to 99 % of its time sinking ... The swift jumps ... usually kept the copepod in the
  same vertical position", and its sinking periods included raptorial catching movements (Acartia's
  ~0 % reflects ambush feeding, not resting). Lab beaker, not a field rest fraction.
- Saiz 1994 [A]: Acartia tonsa "spent most of its time feeding"; time in feeding bouts fell only "from
  98% to 81%" across food concentrations.
- Kiørboe et al. 2018 [A]: copepods that can switch modes "modify foraging effort in response to prey
  density and predation risk"; obligate feeding-current feeders kept a type II response, contrary to
  theory. Almeda et al. 2017 [A]: active (feeding-current, cruising) copepods suffered ~2-8 times higher
  mortality than ambush feeders, **but the predators were predatory copepods** (mechanosensory), so
  "motionless = safer" does not transfer to visual herring: a still copepod at 20 m in daylight is
  visible.
- Diel feeding rhythms: Daro 1988 [F]: "Most zooplankton groups are nocturnal feeders"; rhythms were
  "greatly depressed" or lost below ~200 mg C m-3 (Acartia, Pseudocalanus, young Temora); Calanus
  females shifted their peak to 16:00 at low food. Durbin et al. 1990 [A]: Acartia tonsa kept "about a
  3-fold amplitude in gut contents and feeding rate over the diel period", even when food-limited.
  Dale & Kaartvedt 2000 [A]: diel feeding rhythms in Calanus "throughout the ocean", but "both enhanced
  and reduced nocturnal gut fluorescence were recorded".

### Calculation
A 3-fold night:day feeding amplitude (Durbin; direction inferred from the abstract: day values "skewed
towards lower values") with a 12 h/12 h day means daytime feeding is
~1/(1+3) = 25 % of the daily ration, not 0. Feeding-time fractions measured on Skagerrak species: 27-99 %
(the five non-ambush species; Centropages is a swim-and-sink feeder, Tiselius & Jonsson's category
2), median of their 10 values 51.5 %. The model's 12-23 % eat
share is below every suspension-feeder value in Table 1. [D]

### Inference
- In nature **not feeding is not hiding**. The refuge is a place (depth, darkness) and a time (day),
  not an activity. A copepod that stops feeding near the surface stays where it is (sinks/jumps back)
  and stays visible. Low motility reduces encounter with *mechanosensory* predators (Almeda), which is
  what the model's tactile M1 term (section 139) is about, not with herring.
- The 81-88 % rest share is best read as an energetic outcome of `feeding_cost 4.0`. As a time budget
  it is not copepod-like in summer (non-ambush Skagerrak species fed 27-99 % of the time, median ~50 %). In winter, at
  low food (<200 mg C m-3, Daro), reduced feeding is plausible, but that reduced feeding does not
  make the copepods less visible.
- Coupling "rest = hidden" is therefore not biologically sensible as a general rule. What the biology
  supports: protection that (a) applies by day and lapses at night, and (b) is bought by moving deeper,
  not by not eating. In the model the hide also applies at night, where the literature gives no refuge,
  but the effect on night predation is small because herring night attack is already x 0.1.

---

## Q3. Is the refuge mainly a daytime phenomenon, and does herring night feeding meet unhidden copepods?

### Literature
- Refuge is daytime: Hays 2003 [A]; Bandara et al. 2021 [A]; Bernreuther 2013 [F] (Temora 10 % vs >60 %
  in 0-10 m, day vs night).
- Herring at night: Batty et al. 1986 [A]: "in darkness, only filtering was observed", schools
  dispersed, swimming 0.11 m s-1 (filtering) vs 0.34 m s-1 in light; nauplii were removed more slowly
  than in light "except at the highest prey concentrations"; night filtering matters mainly in dense
  patches. Bernreuther 2013 [F]: herring stomach content rose from ~48-54 mg (night) to a peak at
  16:30 (196 mg small, 313 mg large herring); comparison with gastric evacuation showed herring "were
  either not feeding or feeding only to a minor extent" at night in the upper layer (sprat fed
  intensively at night). Darbyson et al. 2003 [A]: two stomach-fullness peaks, morning and evening.
  Onsrud 2004 [2nd]: Oslofjord herring fullness day/night 2.91/2.29 (Nov) and 3.43/2.70 (Mar) on a
  1-5 scale, with copepods present at both times.
- Herring light constraint dominates seasonally: Varpe & Fiksen 2010 [A]: herring condition follows
  "the return of longer days in spring" more than prey peaks.

### Calculation
Night/peak stomach content in Bernreuther: 48/196 = 0.24 (small) and 54/313 = 0.17 (large herring).
These are standing contents, not intake rates; evacuation shows night intake was well below that. So
a night:day intake ratio of order 0.1 (the model's `dark_ratio`) is consistent with the field data.
No sourced night:day intake ratio was found. [D]

### Inference
- Yes: on a shelf bank the copepod refuge is a daytime phenomenon. At night the copepods are exposed;
  what protects them is that herring see poorly and only filter-feed, which the model already has as
  `dark_ratio 0.1`.
- Literature "accessible fraction" should therefore be compared with the model's **daytime** visible
  fraction. At night the model applies both hiding (x 0.32, if the policy hides at night as much as by
  day; its diel rest pattern was not checked here) and dark (x 0.1) = x 0.032 of the daytime
  potential. The literature gives about x 0.1 (dark) x ~0.9-1.0 (no spatial refuge). That is a
  structural double count, but small in absolute terms because night intake is small anyway.

---

## Q4. Defensible floor and overall accessible fraction; what the model's visible zooplankton would be

### Literature (bounds only)
- The only places where a near-complete (floor ~0) refuge from herring is documented are deep
  overwintering habitats below visual range (Dale 1999, Kaartvedt 2000, Bagøien 2001: hundreds of
  metres), and even there mortality is not zero (0.024-0.027 d-1).
- On a 100 m bank by day: most small copepods do not migrate (Daro, Irigoien); Calanus late stages sit
  at 20-40 m or migrate within a narrow range (Daro, Dale & Kaartvedt); herring follow to 60-80 m by day
  (Bernreuther, Onsrud).

### Calculation
Model now (pi = 0.85, floor 0.2): visible = 1 - 0.85 x 0.8 = 0.32 (range 0.30-0.35 for pi 0.81-0.88).
- December: 1.47 x 0.32 = **0.47 t/km2** (0.43-0.52).
- Summer: 12 x 0.32 = **3.8 t/km2** (3.6-4.2).

Literature-implied daytime accessible fraction (Q1 inference): winter 0.5-0.8, summer 0.63-0.90 (use
0.65-0.90).
- December: 1.47 x (0.5-0.8) = **0.74-1.18 t/km2**, i.e. 1.6-2.5 x the model's 0.47.
- Summer: 12 x (0.65-0.90) = **7.8-10.8 t/km2**, i.e. 2.0-2.8 x the model's 3.8.

What floor is implied? The model's single "hide" stands for "being in the dim lower column by day".
- If hiders really sit in the bottom ~25 m (Kd 0.15: 0.03-1.3 lx at 75 m), a floor ~0.2 for those
  individuals is not absurd; then the hiding share must be small: visible 0.65-0.90 with floor 0.2
  needs pi = (1 - v)/0.8 = **0.13-0.44**.
- If hiders sit at mid-depth (40-60 m, ~1-55 lx at midday in either season at Kd 0.15), detection
  is reduced much less; a floor of ~0.5 or more is more realistic. With pi 0.81-0.88 kept,
  visible 0.65-0.90 needs floor = 1 - (1 - v)/pi = **0.57-0.88**.
- Middle option: pi ~0.4-0.5 by day with floor ~0.4-0.5 gives 1 - 0.45 x 0.55 = 0.75. [D]

### Inference
- **Defensible overall accessible fraction (daytime, herring):** summer ~0.65-0.9, winter ~0.5-0.8.
  Night: spatially ~0.9-1.0, with the herring vision limit carried by `dark_ratio`.
- **Defensible floor:** 0.2 matches a deep-basin or Norwegian Trench refuge, or copepods pressed into
  the darkest bottom layer at Kd >= 0.2. For a generic hiding copepod on a 100 m bank with Kd ~0.15
  the biology supports ~0.4-0.6. Which pair of (pi, floor) is right cannot be pinned down from these
  sources; the *product* is what the literature constrains: visible ~0.65-0.9 by day.
- The model's 0.32 is too protective by a factor of ~1.6-2.8 in daytime, and the hiding is not tied to
  light, so it can also protect at night where nature does not (whether the trained policy actually
  hides at night was not checked here).

---

## Verdict

- **Hiding level (81-88 %, all year): not supported as a refuge fraction.** It reflects the
  energetics of `feeding_cost 4.0` (rest is cheap), not a copepod antipredator distribution. Shelf-sea
  copepods mostly do not migrate in summer (Daro 1988; Irigoien 2004: >60 % no significant DVM), and
  non-ambush copepods feed 27-99 % of the time (Tiselius & Jonsson 1990).
- **Floor 0.2: a deep-water number, too low for a 100 m bank** unless the hiders are the few that sit in
  the darkest bottom layer. With Kd 0.15 the whole column is above the herring biting threshold at
  midday in both seasons.
- **Visible zooplankton:** model 0.47 t/km2 (Dec) and 3.8 t/km2 (summer); literature-implied daytime
  0.7-1.2 and 7.8-10.8 t/km2.
- **Structure:** a copepod refuge from herring is a daytime, depth/light refuge. Not eating is not
  hiding. The hide mechanism is not light-gated; if the policy hides at night (not checked here), that
  night protection double-counts `dark_ratio`, small in absolute terms.
- Weakest links: no Skagerrak-specific day/night copepod profiles; Kd and surface illuminance are
  assumptions; no sourced herring light-saturation value; the per-taxon accessibility fractions in
  Q1 are my estimates from qualitative descriptions.
