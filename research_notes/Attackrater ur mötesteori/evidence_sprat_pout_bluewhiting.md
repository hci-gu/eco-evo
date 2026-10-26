# Encounter factors for sprat, Norway pout and blue whiting (planktivorous_fish -> zooplankton)

Companion evidence for `planktivores_encounter.md` (herring and the group derivation are done
there). Written 2026-10-25. Covers only the three non-herring members (annual-mean biomass
shares: blue whiting 0.105, Norway pout 0.075, sprat 0.03). No code, library or manifest changed.

Tags: [F] full text or data read, [A] abstract only, [S] secondary / search snippet, [D] computed
here, J = judgement (stated). Script: `scripts/members_encounter.py` (stdout in
`tables/members_encounter_stdout.txt`); DOI/OA lookup `scripts/members_lookup.py`. Downloads:
`data/members/` (README there). Earlier evidence reused, not redone:
`../Fiskgrupper i rutan/planktivores/evidence_pout_bluewhiting.md` (diet, depth, diel feeding,
Q/B), `evidence_cmax_energy.md` (Cmax, field rations) and the theses in
`../Fiskgrupper i rutan/planktivores/data/ediss1765` (Bernreuther 2007), `ediss7589` (Kulke 2018),
`ediss4862` (Meskendahl 2013), and `../Torskfiskars ljusberoende vid jakt/` (gadoid vision, diel feeding).

---

## 1. Bottom line: factor values per member

Encounter form used: clearance beta = f_field x pi r^2 v x p, r = k_v sqrt(A_prey) at saturating
light (Aksnes & Utne 1997 form for small beam attenuation; image area A = L x (w/L x L) x 0.75 as in
Langbehn et al. 2019, Suppl. S1). f_field = 1 (full circle) or 0.5 (half the visual area scanned,
the convention of Varpe & Fiksen 2010 used by Langbehn et al. 2019) - the choice is a factor 2 and
must be made once for all members (caller's decision; both are printed by the script).
Prey for all three: copepod-sized pool prey (1-2.5 mm), because the model's zooplankton is the
resident 0-25 m WP2 community, which excludes krill (section 4.1 below).

| Factor | Sprat | Norway pout | Blue whiting | Key sources |
|---|---|---|---|---|
| Length, cm (central; low-high) | 12 (10-14) | 15 (11-20) | 25 (20-30) | sibling notes (box sizes); caller does N/t from the survey |
| Weight g, N per t (central) [D] | 11.8 g, 84 500 | 24 g, 41 700 | 88 g, 11 400 | NS-IBTS CA length-weight fits (`length_weight_all.csv`) |
| Feeding mode | strictly particulate (biting); no filter feeding | visual in dim light + partly tactile/lateral line; ambush-like at night | visual in dim light, night feeding on migrating krill; non-visual component | Kulke 2018 [F]; Bernreuther 2007 [F]; Onsrud et al. 2004 [F]; Kaartvedt et al. 1996 [F] |
| k_v = r / sqrt(A) at saturating light (dimensionless) | 100 (60-155) | 100 (60-155) | 100 (60-155) | anchors: Langbehn 57-65 [F, model parameters]; Kulke sprat back-calc 96-149 [D]; Utne 1997 goby on Calanus 152 (on Acartia 259) [A] |
| r day (saturating light) on 1.2-2 mm copepods | 6 cm (3-12) | 10 cm (4-19) | 10 cm (4-19) | [D] from k_v and prey size; no species measurement exists |
| r on 20 mm krill (not pool prey) | - | 0.77 m (0.46-1.2) | 0.77 m (0.46-1.2) | [D] |
| Light dependence | visual threshold ~0.01 lx on small prey, 0.001 lx on Calanus (herring values; none for sprat); little or no night feeding | dim-light specialist (large eyes); fullness night/day 0.9 on krill (Oslofjord), 0.5 for small pout on copepods | night-biased on krill (Skagerrak); night/day 1-1.5 | Batty et al. 1990 via Kulke [S]; Onsrud et al. 2004 [F]; Albert 1995 [F]; Degnbol & Munch-Petersen 1985 [S] |
| Night/day intake ratio | **0.1** (0.05-0.3) | **0.9** (0.5-1.0) | **1.0** (1.0-1.5; engine caps at 1) | section 4 |
| Foraging speed v | 0.6 BL/s = 7 cm/s (0.3-1.0 BL/s) | 0.33 BL/s = 5 cm/s (0.2-0.6) | 0.33 BL/s = 8 cm/s (0.2-0.6) | Meskendahl 2013 lab [F]; Kaartvedt et al. 2009 deep gadoid-sized fish in situ [F] |
| Search mode | cruise (continuous, horizontal at low prey; vertical zig-zag above ~15 prey/L) | slow cruise / stop-and-go; "ambush-feeding" sinking layer at night | slow cruise (~1/3 BL/s) | Kulke 2018 [F]; Onsrud et al. 2004 [F]; Kaartvedt et al. 2009 [F] |
| Capture success p on copepods | 0.7 (0.33-0.86) | 0.7 (0.33-0.86) | 0.7 (0.33-0.86) | Kulke 2018 tank/video 0.70 +/- 0.35 [F, D]; Viitasalo et al. 2001 siphon 9-100 % by species [F]; Langbehn 2019 assumption 0.7 [F] |
| beta per fish, f = 1, central [D] | 2.0 m3/h | 3.9 m3/h | 6.5 m3/h | script |
| Clearance per t per 6 h tick, f = 1 (f = 0.5), central [D] | 1.0e6 (5.2e5) m3 | 9.7e5 (4.9e5) m3 | 4.4e5 (2.2e5) m3 | script; divide by overlap thickness H (m) x 1e6 for km2/t/tick |
| same, low - high (all extremes multiplied, f = 1) | 9.3e4 - 5.5e6 | 6.9e4 - 4.5e6 | 2.7e4 - 2.5e6 | script; the multiplied extremes overstate the plausible range |

Conversion for the caller: a [km2/t/tick] = (clearance per t per tick, m3) / H [m] / 1e6, with H the
effective overlap thickness of fish and pool prey. Example, sprat central at H 25 m: 1.0e6/25/1e6 =
0.041 (f = 1) or 0.021 (f = 0.5). Pout and blue whiting are deep (sections 4.2-4.3): most of their
search volume lies below the 0-25 m pool, so H for them is large or their overlap fraction small.

**Validation anchor (not used above):** measured sprat clearance (Kulke 2018) is 0.4-1.3 m3/h per
fish for 10-14 cm field sprat, i.e. 2.0e5-7.3e5 m3 per t per tick, = a 0.008-0.029 km2/t/tick at
H 25 m (section 5.1). The central encounter value with f = 0.5 (5.2e5) sits inside it; with f = 1
it is ~1.4-5x above.

---

## 2. Feeding mode and visual range

### 2.1 Sprat
- Strictly particulate: "Sprat were strictly sticking to the particulate-feeding mode, whereas herring
  occasionally started to filter-feed from a prey concentration of 50 L-1" (6 cm sprat, 8 cm herring,
  16 C, 1-160 prey/L) [F] Kulke 2018, summary. "Filter-feeding has not been reported in sprat"
  [F] Bernreuther 2007, ms 2. Field: sprat "preyed upon older copepodite stages, indicating
  size-selective particulate feeding" [A] Moellmann et al. 2004. So no filter mode for sprat.
- No measured reactive distance for sprat exists (searched; none found). Back-calculation [D] from
  Kulke's lab functional response of 8.7 cm sprat on Artemia nauplii (0.5-0.9 mm) at 16 C:
  clearance BRmax/k = 2.0/10.7 L/s = 0.187 L/s (k = -0.11 T + 12.45, Fig. 5.1). With search speed
  0.3-0.5 BL/s and p 0.7-1: r = 3.7-5.7 cm, k_v = 96-149.
- Same-size comparison at 16 C (Kulke ms 1; initial slope BRmax/k) [F/D]: sprat 6.3 cm on Artemia
  0.072 L/s, on the evasive copepod Acartia tonsa 0.042 L/s; herring 8 cm on Artemia 0.060 L/s, on
  A. tonsa 0.079 L/s. Summary text: "at lower prey concentrations up to 50 L-1 sprat exhibit
  significantly higher feeding rates than herring" (sprat k 7.9 vs herring 36.4 on Artemia).
  So, per individual of similar size, sprat is not a weaker particulate searcher than herring.
- Light: Kulke quotes Batty et al. 1990: particulate-feeding threshold 0.01 lx for herring on
  Artemia-sized prey, ~0.001 lx on Calanus [S, via Kulke 2018 ms 3]; salmonids need ~3.4 lx for
  maximum feeding rate [S, same]. Nothing sprat-specific. Kulke: in the Bornholm Basin "light
  intensity in the deep is probably just high enough to detect prey but not to feed on it with a
  high rate" (0.07-0.37 lx at 60 m on summer days) [F].

### 2.2 Norway pout
- Oslofjord, krill (Meganyctiphanes) and Calanus: "Norway pout have particularly large eyes,
  presumably enabling them to feed visually in relatively deep water during day"; but "their deep
  night-time distribution renders visual predation unlikely"; the lateral line "suggests that it may
  be a partially tactile predator", and the slowly sinking night layer "would correspond to the
  description of non-visual, ambush-feeding fish" [F, article page] Onsrud et al. 2004.
- Norwegian shelf, April: pout left the benthic boundary layer and "ascended into the lower part of
  the krill layer inside the front, foraging on Thysanoessa inermis" where algal shading lowered the
  light; stomachs held only T. inermis 17-21 mm [F] Kaartvedt et al. 1996. Reading: a dim-light
  visual forager whose depth tracks light.
- No reactive distance, eye sensitivity or retina study for pout (searched; none). Gadoid proxies
  (`../Torskfiskars ljusberoende vid jakt/gadoid_vision_thresholds.md`): juvenile cod reactive
  distance still high at 0.01 umol m-2 s-1 (~0.5 lx) (Meager et al. 2010 [A]); juvenile cod use
  chemical cues to locate mysids from longer distances than vision, and turbidity barely reduces
  visual reactive distance (Meager et al. 2005 [A]).

### 2.3 Blue whiting
- Skagerrak: feeding "increases during the evening and is at a maximum around and after midnight",
  rising towards the surface at night (Degnbol & Munch-Petersen 1985 via Post et al. 2021) [S, sibling
  note]. Irminger Sea: highest intake from noon to late evening [F, sibling note]. Krill swarms in the
  Norwegian Sea are patrolled by blue whiting and saithe (acoustic targets ascribed to blue whiting
  beneath swarms at 100-200 m) [F] Kaartvedt et al. 2005.
- No visual range, eye sensitivity or speed measurement for the species (searched; none).

### 2.4 The shared visual-range scale k_v [D]
r = k_v sqrt(A) is the small-c limit of the Aksnes & Utne (1997) model at saturating light; k_v
lumps prey contrast and eye capacity. Anchors (`members_encounter.py`):
- Langbehn et al. 2019, Table S1 [F, model parameters, not measurements]: Benthosema glaciale (6 cm)
  r_max 0.06 m on a 2.7 mm copepod -> k_v 57; generic piscivore (50 cm) r_max 1.5 m on 6 cm fish -> 65.
  Their clearance uses half the visual area, "beta = 0.5 pi r^2 v", capture success 0.7, swimming
  velocity 0.06 m/s (Benthosema, 1 BL/s) and 0.25 m/s (piscivore, 0.5 BL/s); non-visual detection
  threshold 3 mm (Benthosema) and 0.1 m (piscivore); eye half-saturation K_e 1e-8 (mesopelagic) and
  4e-4 mW m-2 nm-1 (piscivore), "unknown and therefore" chosen to match the observed scattering layer.
- Utne 1997 [A]: Gobiusculus flavescens (4-5 cm hovering planktivore) "The longest reaction distance,
  18.9 cm for larger prey (Calanus finmarchicus), occurred at a light level of 80 umol m-2 s-1
  compared to 12.9 cm for a smaller prey (Acartia clausi) at 8 umol m-2 s-1. Above a light saturation
  level of 10 umol m-2 s-1, additional light had little influence" -> k_v 152 (Calanus), 259 (Acartia).
- Sprat back-calculation (2.1) -> 96-149.
- J: central 100, range 60-155 for all three members. Fish size does not enter k_v here; Langbehn's
  two very different fish give the same k_v, which supports that. If r grows with fish size beyond
  this (Aksnes-Giske eye capacity), pout and blue whiting are underestimated (by up to ~2x in r).
- Light below saturation: r^2 scales as E/(K_e + E). For these members K_e is unknown; the gadoid
  proxies (cod reactive distance high at 0.01 umol m-2 s-1; Langbehn piscivore 4e-4 mW m-2 nm-1)
  put pout and blue whiting among dim-light feeders. The light at their depth is the caller's part.

## 3. Swimming speed and search mode

- **Sprat, lab** [F] Meskendahl 2013 ms 4 (YOY sprat and herring, 16 C, Artemia and A. tonsa):
  horizontal speed rose with prey concentration to a plateau at 30-39 prey/L; maxima 0.92 BL/s on
  copepods (1.07 on Artemia); herring 0.81 (0.96). Spontaneous swimming "dominated by low swimming
  speeds (~ 0.57 body length s-1)"; minimum activity 0.28 BL/s (used by Kulke 2018 for non-feeding
  hours). Vertical speeds up to 1.6x horizontal at high prey density. Temperature (10-19 C) "had no
  direct effect on the performed swimming speeds".
- **Sprat search pattern** [F] Kulke 2018: below ~15 prey/L "fish increasingly searched" in a
  horizontal plane; above, a vertical zig-zag with 2-12 bites per upward run. Cruise search.
- **Sprat in the field**: no foraging speed found. Overwintering fjord sprat descended at ~3.5 cm/s
  (vertical) and alternated up and down ("rise and sink") [F] Solberg et al. 2012; the 2017 paper
  with individual speeds (Solberg & Kaartvedt, Prog. Oceanogr. 151:49) could not be fetched.
- **Deep gadoid-sized fish, in situ** [F] Kaartvedt et al. 2009 (Masfjorden, bottom-mounted 38 kHz,
  target tracking): fish below 300 m (TS -40 dB ~ 23 cm gadoid) median 7-9 cm/s, max ~25 cm/s; fish
  at 200-300 m (TS -36 dB ~ 37 cm) median 12-15 cm/s, max 30-40 cm/s; "the median swimming speed
  would correspond to ~1/3 body length s-1"; species not identified. Proxy for pout and blue whiting
  (J). Kaartvedt et al. note slow swimming would keep a predator hydrodynamically "silent" and
  that stop-and-go (saltatory) movement occurs in the mesopelagic Benthosema.
- Langbehn et al. 2019 assumed 0.5 BL/s for their piscivore and 1 BL/s for Benthosema [F, model].
- Central values: sprat 0.6 BL/s (0.3-1.0); pout 0.33 BL/s (0.2-0.6); blue whiting 0.33 BL/s
  (0.2-0.6). J.

## 4. Diel feeding and depth (night/day ratio check)

### 4.1 Which prey counts
The model's zooplankton is the resident 0-25 m community calibrated to SHARK A17 WP2 90 um 0-25 m
carbon, which covers copepods, cladocerans, appendicularians and chaetognaths; "meroplankton,
medusae and krill are not in it" (`../Sillens vinterföda/winter_food_and_feeding.md`, Q1). Blue
whiting (krill ~60 % of the diet) and larger pout (krill, Pandalus) feed largely on prey outside
the pool. For `a` on the pool, only their copepod feeding counts: copepods ~25 % of blue whiting
diet (Bachiller 2018), 0.48 of pout diet (EcoBase 457) (sibling note 4.1-4.2).

### 4.2 Norway pout
- Oslofjord (120 m site), stomach fullness (1-5 scale) day vs night [F]: Nov 3.27 vs 2.93
  (ratio 0.90), Mar 4.00 vs 3.62 (0.91); "no significant day/night difference"; krill per stomach
  equal day and night (1.54/1.54; 2.17/2.10). Depth: below ~90 m by day in November, approaching the
  krill from below; "deepest nocturnal distribution, with maximum abundance in midwater and no
  catches in the upper 25 m" Onsrud et al. 2004.
- SW Norway bank (140 m, July), small pout on copepods: "Mean total dry weight by day was twice that
  at night" (0.5); larger pout no regular diel variation [F] Albert 1995 (sibling note 5.1).
- Night/day intake **0.9 (0.5-1.0)** [D/J]. Vertical habitat: 5-10 m above the bottom by day, rarely
  in the upper 25 m at night - i.e. little overlap with the 0-25 m pool at any time.

### 4.3 Blue whiting
- Night-biased feeding in the Skagerrak on krill [S]; in the box a Q3 visitor below 100 m (sibling
  note 5.2). **Night/day 1.0 (1.0-1.5)**; the engine treats dark_ratio >= 1 as no light effect.

### 4.4 Sprat
- Bornholm Basin, June 2001, 11-15 cm sprat [F] Bernreuther 2007 ms 2: mean stomach content from
  30-40 mg WW at 02:00 to a peak of ~100 mg at 14:30; "Two sprat feeding peaks, one in the afternoon
  and one close to midnight"; "both species feed at night, but to a lower extent than during the
  day". Shoals at 60-80 m by day, ~20 m at night.
- Central Baltic [F] Kulke 2018 ms 3: 71 +/- 13 % of the daily ration taken "in the crepuscular
  phases in the upper layers and/or during the downwards/upwards migration"; feeding there 3.16x the
  deep daytime rate; at night sprat "do not eat at all or only very slightly" (Koester 1994;
  Bernreuther et al. 2013, cited there).
- Cardinale et al. 2003 [A]: "sprat seemed to continue feeding during the whole day time" (daytime
  only; sibling note).
- Night/day intake **0.1 (0.05-0.3)** [J]: night standing contents ~0.3-0.4 of the day peak are
  partly undigested day food, so intake is lower than that ratio.

## 5. Validation data (not used to set a)

| Datum | Value | What it tests | Source |
|---|---|---|---|
| Sprat lab functional response, 8.7 cm, Artemia, 5-20 C | Type II; BRmax = 2.16/(1+11.17 exp(-0.28 T)) bites/s (1.0 at 8 C, 2.0 above ~16 C); k ~11 /L, no T effect; clearance 0.19 L/s at 16 C | per-individual clearance at saturating lab light; temperature scaling of the ceiling but not of the clearance | Kulke 2018 ms 2 [F] |
| Sprat field k vs length | k = 291 L^-1.5 prey/L (CIV-adult copepods + cladocerans, mean prey 32 ug WW), i.e. 7.0 /L at 12 cm -> clearance 0.5-1.0 m3/h (BRmax 1-2), = 2.6e5-5.2e5 m3 per t per tick [D] | the whole encounter product (r, v, p, light, overlap) for sprat at field conditions; a 0.010-0.021 at H 25 m | Kulke 2018 ms 2, Fig. 5.5 [F]; the BRmax used in Eq. 5.6 is not stated unambiguously, hence the 1-2 range |
| Sprat lab: tank depletion vs video biting | depletion rates 0.70 +/- 0.35 of the video model; at low prey density the video rate exceeds depletion up to 4-7x: "sprat cannot perform this behavior continuously, especially at lower prey concentrations" | capture success, and whether the clearance holds at low density (it falls) | Kulke 2018 [F] |
| Herring vs sprat lab clearance, 6-8 cm | sprat 0.042-0.072 L/s, herring 0.060-0.079 L/s (16 C) | relative member weighting per individual | Kulke 2018 ms 1 [F/D] |
| Sprat field daily ration | 3.4-4.8 %BW/d (new method), 2.4-3.6 (established), 11-13 cm, Baltic, Aug; YOY 2-11 %BW/d | feeding level vs Cmax (sibling Cmax 16.6 %BW/d on 4.5 kJ/g prey -> ~0.2-0.3 of Cmax) | Kulke 2018 [F]; Arrhenius 1998 [A]; `evidence_cmax_energy.md` 2.3 |
| Norway pout daily ration | 1.1 %BW/d (11-15 cm, July, 140 m, copepods); Heath 2005 1.2 %/d; Ecopath Q/B 5.05 /yr | feeding level vs the cod-proxy Cmax 3.7-3.9 %/d -> ~0.3 | Albert 1995 [F]; sibling 2.4 |
| Blue whiting ration | 0.02 g/g/d (Norwegian Sea Apr-Sep, mainly krill); Q/B 6-9 /yr | feeding level ~0.6-0.7 of the cod-proxy Cmax; mostly krill, not pool prey | Bachiller et al. 2018 [F]; sibling 2.5 |
| Pout stomach fullness day/night | 0.90-0.91 on krill (Oslofjord) | night/day ratio of a | Onsrud et al. 2004 [F] |
| Absolute clearance or functional response of pout or blue whiting | none found | - | searched |

## 6. Judgement and uncertainty

- J: k_v 100 (60-155) for all three. Largest single uncertainty for the members: no reactive
  distance exists for any of them; the visual-range scale spans ~2.6x (r), i.e. ~7x in clearance.
- J: f_field (0.5 or 1). A factor 2, shared with herring; must be one convention for the group.
- J: speeds for pout and blue whiting from unidentified gadoid-sized targets (Kaartvedt et al. 2009).
- J: capture success 0.7 from sprat on Artemia and siphon experiments with Baltic copepods
  (Calanus not tested); krill would be lower (0.3-0.8, J).
- Structural: pout and blue whiting feed deep (pout 5-10 m above the bottom by day, no catches above
  25 m at night; blue whiting below 100 m in the box) and mostly on krill/shrimp outside the pool.
  Their effective attack rate on the 0-25 m pool is set more by vertical overlap than by r or v.
  This favours a small contribution from these two members to the group `a`, regardless of their
  per-volume clearance.
- Products of independent extremes (table rows "low - high") span ~60x; the central values carry
  the information, the extremes are bounds.

## 7. Failed fetches and gaps
- Solberg & Kaartvedt 2017 (Prog. Oceanogr. 151:49, doi 10.1016/j.pocean.2016.11.003; OA copy at
  hdl.handle.net/11250/2478119): brage.nina.no did not resolve (DNS) via curl or WebFetch.
- Onsrud et al. 2004 OUP PDF: returned an HTML challenge page; the article page was read with
  WebFetch (first 100 000 of 110 874 characters) instead.
- Meager et al. 2005 (cdnsciencepub 403) and 2010, Utne 1997, Cardinale et al. 2003: abstracts
  only (OpenAlex); no numbers beyond the abstracts.
- No reactive-distance, eye-sensitivity, retina or foraging-speed study found for sprat, Norway pout
  or blue whiting; no blue whiting gill-raker filtering evidence; no functional response for pout or
  blue whiting.

## 8. References
- Albert OT 1995. ICES J Mar Sci 52:873. (local copy, sibling data/albert1995) [F]
- Aksnes DL, Utne ACW 1997. A revised model of visual range in fish. Sarsia 82:137-147. doi 10.1080/00364827.1997.10413647 (form only, via Langbehn 2019 S1)
- Bachiller E et al. 2018. PLoS ONE 13:e0190345. [F, sibling]
- Bernreuther M 2007. PhD thesis, Univ. Hamburg. https://ediss.sub.uni-hamburg.de/bitstream/ediss/1765/1/DissertationMatthiasBernreuther.pdf [F]
- Bromley PJ, Watson T, Hislop JRG 1997. ICES J Mar Sci 54:846-853. [F, first page]
- Cardinale M, Casini M, Arrhenius F 2003. Aquat Living Resour 16:283. doi 10.1016/S0990-7440(03)00007-X [A]
- Casini M, Cardinale M, Arrhenius F 2004. ICES J Mar Sci 61:1267. doi 10.1016/j.icesjms.2003.12.011 [A]
- Degnbol P, Munch-Petersen S 1985. ICES CM 1985/H:46. [S via Post et al. 2021]
- Kaartvedt S, Melle W, Knutsen T, Skjoldal HR 1996. MEPS 136:51-58. doi 10.3354/meps136051 [F]
- Kaartvedt S, Rostad A, Fiksen O, Melle W, Torgersen T, Breien MT, Klevjer TA 2005. MEPS 299:1-5. doi 10.3354/meps299001 [F]
- Kaartvedt S, Rostad A, Klevjer TA, Staby A 2009. MEPS 395:109-118. doi 10.3354/meps08174 [F]
- Kulke R 2018. PhD thesis, Univ. Hamburg. https://ediss.sub.uni-hamburg.de/bitstream/ediss/7589/1/Dissertation.pdf [F]
- Langbehn TJ, Aksnes DL, Kaartvedt S, Fiksen O, Jorgensen C 2019. MEPS 623:161-174 and Supplement. doi 10.3354/meps13024 [F]
- Meager JJ, Solbakken T, Utne-Palm AC, Oen T 2005. CJFAS 62. doi 10.1139/f05-104 [A]
- Meager JJ, Moberg O, Strand E, Utne-Palm AC 2010. Mar Freshw Behav Physiol 43:99-108. doi 10.1080/10236241003798910 [A]
- Meskendahl L 2013. PhD thesis, Univ. Hamburg. https://ediss.sub.uni-hamburg.de/bitstream/ediss/4862/1/Dissertation.pdf [F]
- Moellmann C, Kornilovs G, Fetter M, Koester FW 2004. J Fish Biol 65:1563. doi 10.1111/j.0022-1112.2004.00566.x [A]
- Onsrud MSR, Kaartvedt S, Rostad A, Klevjer TA 2004. ICES J Mar Sci 61:1278. doi 10.1016/j.icesjms.2004.09.005 [F, article page]
- Solberg I, Klevjer TA, Kaartvedt S 2012. MEPS 464:245-256. doi 10.3354/meps09877 [F]
- Utne ACW 1997. J Fish Biol 50:926. doi 10.1111/j.1095-8649.1997.tb01619.x [A]
- Varpe O, Fiksen O 2010. Ecology 91:311. doi 10.1890/08-1817.1 (half-visual-field convention, via Langbehn 2019) [S]
- Viitasalo M, Flinkman J, Viherluoto M 2001. MEPS 216:191-200. doi 10.3354/meps216191 [F]
