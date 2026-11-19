# Gadoid vision in near-darkness: Ke and R_nv on fish prey (2026-11-19)

Research note, literature and field evidence only. Nothing in the library, code, tests, manifest, resume,
AGENTS.md or other notes was changed. No value is chosen or recommended to hit a stomach share, booked M2, diet
or clumping result. Skagerrak stomachs by bed depth stay an independent test and are not used here.

Tags: [F] primary source read (full text), [D] derived here, [J] judgement, [A] abstract only, [S] secondary.
Units: 1 W m-2 at 490-500 nm = 4.10-4.18 umol m-2 s-1; 35-54 lx per umol (photopic V at 490-500 nm; the notes'
deep blue-green convention), i.e. 145-226 lx per W m-2. Radiance L -> irradiance E = pi L (isotropic field)
unless stated. All Ke below are half-saturation of R^2, the library's g = E/(E + Ke).

Files: `scripts/digitise_anthony_fig5.py` -> `tables/anthony1981_fig5_digitised*.csv`; `scripts/fit_cod_ke.py`
-> `tables/fit_cod_ke_stdout.txt`; `scripts/field_light_at_depth.py` -> `tables/field_light_at_depth_stdout.txt`;
`tables/item1_full_texts.md`. PDFs in `papers/` (licences in `papers/LICENCES.md`; never commit).

## Verdict

- **A gadoid-specific Ke exists, and the earlier notes missed it.**
  - Aksnes & Utne (1997) fitted their visual model to Anthony's (1981) cod contrast thresholds. They got Ke =
    2.9e-6 W m-2 sr-1, and said themselves that one parameter set fits poorly [F].
  - The thresholds were redigitised and refitted here [D]. Cod contrast sensitivity keeps improving over 6 decades
    of light, so no single hyperbola fits it. The fitted Ke depends on the light range chosen: all levels 9e-4 lx,
    photopic 2e-3 to 2e-2 lx, and a scotopic 3-point branch 2.4e-4 lx. These values assume E = pi L. The scotopic
    branch's plateau R^2 is only ~1/3 of the bright one, so it is not the library's Ke 2e-4 variant. It is merely
    consistent with the pollock half-way point.
- **The geometry decides the result, and it is unresolved [J].**
  - The cod saw a background radiance. A fish looking horizontally in a downwelling field has E_d / (pi L_bg) = F,
    which is unknown, likely between 1 and ~30.
  - The empirical cod R^2 ratio between 0.1 and 0.006 lx (an illustrative pair, not measured bed lights) is
    1.48 / 1.81 / 2.42 / 3.11 at F = 1 / 3 / 10 / 30 [D].
  - That spans the model's 1.3x to 2.65x. **The cod data cannot yet separate Ke 2e-4 from 0.02 lx.**
  - In downwelling lx the A&U fit is 2e-3 (F 1) to 6e-2 lx (F 30). The library's 0.02 lx equals that fit at
    F ~10.
- **Ke bracket, gadoid on fish prey: ~2e-4 to ~0.07 lx [D/J]; confidence low.**
  - The bracket is unchanged in substance, but now has a gadoid basis instead of only the pollock and salmonid
    analogues.
  - It narrows only if F is fixed: to ~1e-3 to 2e-2 lx at F = 1, and to 3e-3 to 7e-2 lx at F = 3-30.
  - Meager 2010, the only gadoid R-vs-light curve, used mysid prey and is closed.
- **The absolute visual floor is ~1e-5 to 1e-4 lx [D/S].**
  - Cod still discriminated contrasts of 0.2-0.6 at 4-6e-5 lx (pi L) [D from F]. The pollock floor is ~1-3e-5 lx.
  - A model eye with an 8 mm pupil sees 1 cm prey ~2 m away horizontally at 195 m in coastal water by day (Nilsson
    et al. 2014 Fig. 7; figure read by sub-agent). A 50 cm cod's lens is 8.4 mm [F].
  - So shelf gadoids at 150-250 m by day are dim, not blind [J].
- **R_nv on fish prey: no gadoid measurement exists. The analogue bracket is 0.05-0.5 m, and it depends on the
  scaling [D/J]; confidence low.**
  - Gadiform burbot in the dark: mean 5.6 cm, max 12 cm [F].
  - Scaled by predator length to a 40-70 cm gadoid: ~0.05-0.3 m.
  - Scaled by prey size (r ~ a U^(1/3); analogue prey 3-8 cm against ~15 cm fish prey): ~0.1-0.5 m.
  - The library's 0.3 m sits inside the prey-size scaling and above the burbot length scaling. No measurement
    places it outside the bracket.
- **Field feeding in darkness constrains the combined dark intake, not Ke [J].**
  - Svalbard cod and haddock keep eating in January at 90-455 m, cod mostly fish (98 % of cod with prey) [F].
  - Light at >= 150 m is <= 1e-5 lx without the moon [D]. At full moon it is visual-range light down to ~150 m in
    clear water. Only depths >= 250 m with K >= 0.05 m-1 stay below the floor whatever the moon.
  - Barents bank cod (bed 161-175 m) took pelagic fish mostly around dawn, when the prey come down to the bed [F].
  - No same-size lit reference and no encounter data turn this into an R_nv.
- **Bioluminescence:** no gadoid evidence. Berge 2015 only suggests krill luminescence "might act as a cue" in the
  polar night [F].

Sub-agent readings: full texts were downloaded and read by research sub-agents. Quotes rechecked here by grep in the
local copies: Aksnes & Utne, Fiksen 1998, Bengtsson, Larsen, Berge, Cohen, Skaret (K, empty share, quotes), Michels
(burbot, bass, walleye), Pohlmann, Leifsdottir Table 1, the Nilsson Fig. 7 caption. Figure readings not rechecked:
Nilsson Fig. 7 ranges, the Skaret Fig. 3a isolume depth, the Larsen Fig. 5 fish shares, the Bengtsson Table S3 sum.

## Evidence tables

### E1. Ke and visual floor (what each number is)

| # | source | species | kind of quantity | value | lx [D] | tag |
|---|---|---|---|---|---|---|
| 1 | Aksnes & Utne 1997, Sarsia 82:137, p. 142-143, Fig. 5 | cod (Anthony 1981 data) | Ke, one hyperbola over 1e-7 to 1e-1 W m-2 sr-1 | "Ke = 2.9 10-6 W m-2 sr-1 ... use of a single parameter set seems inconsistent with the observations" | 1.3-2.1e-3 | [F] |
| 2 | same, Fig. 5 broken line | cod | lower Ke at low light ("lowering T3 and Ke") | not printed | - | [F] |
| 3 | Fig. 5 digitised, refitted (`fit_cod_ke_stdout.txt`) | cod | Ke by fitted range: scotopic (<= 1e-5.5) / all / photopic (>= 1e-5.5) / photopic (>= 1e-4.4) | 3.3e-7 / 1.3e-6 / 5.8e-6 / 3.2e-5 W m-2 sr-1 | 1.5-2.4e-4 / 0.6-1.0e-3 / 2.6-4.1e-3 / 1.4-2.2e-2 | [D] |
| 4 | same data, relative R^2 = Ct_min/Ct (A&U: R^2 ~ 1/Ct) | cod | empirical R^2 ratio between two lights (no Ke) | 0.1/0.006 lx 1.48; 0.1/0.003 1.75; 0.03/0.001 2.3; 0.01/3e-4 3.5; 0.01/1e-4 5.2 | - | [D] |
| 5 | Anthony 1981, J Fish Biol 19:87 | cod | contrast threshold, 7 levels; min ~2 %; scotopic/photopic break 8e-6 W m-2 sr-1 | as quoted | break 3.6-5.7e-3 | [A] |
| 6 | Fiksen et al. 1998, Fish Oceanogr 7:355 | larval cod and herring | Ke used | "Ke was assigned the same value (5 uE m-2 s-1; Aksnes and Utne, 1997) as found earlier for gobies"; not fitted to cod (larvae: cone-only eyes) | ~270 | [F] |
| 7 | Ryer & Olla 1999 (prior note "Synregel i vattenpelaren") | juvenile walleye pollock | visual-feeding floor; half-way of the visual increment | 5e-7 uE; ~4e-6 uE | ~3e-5; ~2e-4 | [S] |
| 8 | Meager et al. 2010 (closed; prey from Rohan et al. NOAA ms Table B3) | juvenile cod on mysids | R at 0.01-64 umol, 5 levels, two-part fit; "high" at 0.01 | no values available | 0.01 umol = 0.35-0.54 | [A]/[S] |
| 9 | Blaxter 1970 via Utne-Palm et al. 2018 Table 1 | cod | "possible light threshold", behaviour at trawls | 0.01-0.001 umol | 0.035-0.54 | [S] (book chapter, not online) |
| 10 | Walsh & Hickey 1993, ICES MSS 196:68, p. 72 | cod, haddock, saithe, silver hake at trawls, 44-53 m | assumed vision threshold, taken from Glass & Wardle 1989; no light measured. At night still oriented: haddock 87 %, hake 100 %, cod 67 % | "vision threshold value of 10^-6 lux" | 1e-6 | [F] (sub-agent, page images); threshold [S] |
| 11 | Nilsson, Warrant & Johnsen 2014, Phil Trans B 369:20130038, Fig. 7 | model eye, 8 mm pupil | detection range of 1 cm prey, coastal water, day | ~+-3 m at 120 m, ~+-2 m at 195 m horizontally | - | [F] (model) |
| 12 | Leifsdottir & Campana 2023, PLoS ONE, Table 1 | cod, haddock (173-308 m) | lens diameter (mm) vs TL (cm) | cod 4.034 + 0.087 TL; haddock 3.434 + 0.124 TL | 50 cm cod 8.4 mm | [F] |
| 13 | Bozzano & Catalan 2002; Mas-Riera 1991 | hake | ganglion-cell peak 47,900 mm-2 (juvenile); hake species' sensitivity tracks depth and night feeding | optical sensitivity S ~8-13 um2 sr [D], near an epipelagic eye (Landgren 2014: 6.4) | - | [A] |

Notes on E1:
- In the A&U model (c = 0) R^2 is proportional to C0 Ap / Ct. Rows 3-4 assume that holds for the cod data. Anthony's
  targets were a projected spot pattern, not prey. Fish size and the lamp spectrum are not known here, and a broadband
  radiometric W would give fewer lx per W, which lowers the lx values.
- Library form for comparison [D]: g(0.1)/g(0.006) is 3.6 at Ke 0.02, 1.31 at 2e-3, 1.16 at 1e-3 and 1.03 at
  2e-4.
- The cod curve gives 1.48 / 1.81 / 2.42 / 3.11 at F = 1 / 3 / 10 / 30 (visual R^2 only, no R_nv;
  `fit_cod_ke_stdout.txt`). Ratios of a pair do not depend on how Ct is normalised.
- Ke in downwelling lx for F = 1 / 3 / 10 / 30:

  | fit | F 1 | F 3 | F 10 | F 30 |
  |---|---|---|---|---|
  | A&U single | 2.1e-3 | 6.2e-3 | 2.1e-2 | 6.2e-2 |
  | all levels | 9.4e-4 | 2.8e-3 | 9.4e-3 | 2.8e-2 |
  | scotopic | 2.4e-4 | 7.1e-4 | 2.4e-3 | 7.1e-3 |

- The scotopic branch (T3 0.100 against 0.035-0.041) has ~1/3 of the bright plateau R^2.

### E2. Non-visual detection on fish prey (R_nv)

| # | source | predator / prey | condition | distance or ratio | tag |
|---|---|---|---|---|---|
| 1 | Michels 2020 MS thesis (= TAFS 10.1002/tafs.10293), p. 34 | burbot 285-490 mm TL / round goby, sculpin 60-78 mm | night, 8.45e8 photons m-2 s-1 (~7e-8 lx [D]) | "maximum reactive distance for Burbot during dark trials was 12 cm"; mean 5.6 +- 1.4 cm (0.11-0.20 BL); peak 8.7 cm vs 44.8 cm at sun -4 deg (0.19) | [F] |
| 2 | same, Table 1 (counts derived by sub-agent) | burbot | prey retained, dark vs civil twilight -6 deg | 3 vs 12 (0.25); burbot moved 11 vs 87 min per trial | [F/D] |
| 3 | same | smallmouth bass 290-425 mm / same prey | dark | mean 14.2 cm (0.33-0.49 BL), max 55 cm; retained dark/sunrise 0.53 | [F] |
| 4 | Michels 2024 PhD diss. ch. 3 | YOY walleye 103-162 mm (tapetum) / fathead minnow 31-42 mm | dark (<6.65e8 photons) | 12.5 +- 2.0 cm (0.94 BL) vs 29.3 cm in light (0.43); undirected reaction angles | [F] |
| 5 | Pohlmann, Grasso & Breithaupt 2001, PNAS | European catfish 20-25 cm / guppy 2-5 cm | IR darkness | wake tracking began at 6-40 cm, median 12 cm (~0.5 BL); 59 of 94 attacks captured; this is wake following, not point detection | [F] |
| 6 | Ryer & Olla 1999 (prior note) | juvenile pollock / Artemia | total dark vs 1e-3 uE | consumption ratio ~0.2-0.3 | [S] |
| 7 | McMahon & Holanov 1995, J Fish Biol | largemouth bass | 243 lx to total dark | success > 95 % to 3e-3 lx, 62 % at starlight, ~0 % in total darkness | [A] |
| 8 | Brawn 1969, JFRBC 26:583 | cod 5-90 cm, captive | dark | "Some live food was captured in the dark"; small items found by taste buds on barbel and pelvic fin rays (contact senses) | [A] |
| 9 | Palmer et al. 2005; reviews (Braun 2002; Coombs 1999) | toadfish; general | lateral-line range | 0.4 BL measured; reviews 1-2 BL or < 1 BL | [S] |

Reading of E2 [D/J]:
- Three scalings:
  - **Absolute.** Means 5.6 / 12.5 / 14.2 cm (burbot / walleye / bass); maxima 12 / 55 cm (burbot / bass);
    catfish wake onset median 12 cm.
  - **By predator length.** Burbot mean 0.11-0.20 BL, max 0.24-0.42 BL; bass mean 0.33-0.49, max 1.3-1.9;
    walleye mean 0.94; catfish ~0.5. For a 40-70 cm gadoid: burbot 0.05-0.3 m, the others 0.15-1.3 m.
  - **By prey (source) size**, r ~ a U^(1/3) as in the prior note, with analogue prey 3-8 cm against ~15 cm fish
    prey: burbot mean 5.6 -> ~12 cm and max 12 -> ~26 cm; walleye 12.5 -> ~50 cm; catfish 12 -> ~50 cm.
- **Which statistic maps to R_nv [J]:** the effective encounter radius lies between the mean and the maximum
  reaction distance, as in A&U's r0 = R + 2s.
- **Weight of the analogues:** burbot is the only gadiform, but it is a sit-and-wait ambusher in fresh water.
- A dark/light rate ratio (0.19-0.3) converts to a distance ratio only if rate ~ R^2 and speed, p, prey behaviour
  and encounter limitation are unchanged. Burbot moved 8x less in the dark, so no square-root conversion is
  offered.

### E3. Field feeding in near-darkness

| # | source | species, size | depth, season | light at depth | metric | tag |
|---|---|---|---|---|---|---|
| 1 | Bengtsson et al. 2024, MEPS 747:117, p. 122, 125 | cod mean 39 cm (polar night) vs 67 cm (Sept); n 211 / 39 | 114-349 m bottom, 30-183 m pelagic; Kongsfjorden 79 N, mid-January | none measured; <= 5e-5 lx at >= 90 m [D] | "207 (98.1 %) of the Atlantic cod had identifiable stomach contents during polar night"; fish ~68 % of prey mass (sub-agent sum of Table S3) vs ~96 % in summer; sizes confound | [F] |
| 2 | Larsen et al. 2023, Polar Biol 46:783, Table 5 | cod n 692, haddock n 293 | > 90 m (basins 180-455 m), 6 fjords 78-80 N, January | none | "almost empty" (< 5 % full): cod 25.2 %, haddock 21.8 %, polar cod 40.1 %; "Well-filled stomachs ... were rare"; "limited evidence that the low-light conditions ... inhibited feeding"; no lit reference | [F] |
| 3 | Berge et al. 2015, Curr Biol 25:2555 | cod, haddock | trawl to 250 m, Kongsfjorden, January; one trawl | none | "approximately half ... exhibited a stomach fullness over 50%" | [F] |
| 4 | Cohen et al. 2015, PLoS ONE | (light) | Kongsfjorden, January noon, in air | 1-1.5e-5 umol m-2 s-1 (0.4-0.8e-3 lx) | basis of the depth estimates | [F] |
| 5 | `field_light_at_depth_stdout.txt` | - | polar-night noon, K 0.03-0.10 m-1 [J]; no moon / full moon (0.2 lx in air) | no moon: 90 m <= 5e-5, 150 m <= 9e-6, 250 m <= 4e-7 lx; full moon: 90 m 2e-5 to 1e-2, 150 m 6e-8 to 2e-3, 250 m 3e-12 to 1e-4 lx | without moon below the cod contrast range (4e-5 lx) and the pollock floor at >= 90-150 m; with a full moon visual-range light reaches ~150 m in clear water | [D] |
| 6 | Skaret et al. 2020, MEPS 638:13, p. 18-21 | cod > 25 cm, n 239 | bed 161-175 m, Great Bank 77 N, October; K 0.064 m-1 (CTD PAR) | noon ~4e-3 to 4e-2 lx; civil dawn ~3e-5 to 1e-4 lx at the bed [D] | 14 % empty, 79 % of fed held fish; fresh pelagic fish more often in the morning; "an average cod fed on pelagic fish only every third day"; pelagic fish "avoided light intensities above 0.001 umol"; ~45 h to digest half a meal | [F] |
| 7 | Klemetsen 1982, ICES JMS 40:101 | cod, Balsfjord 69 N | year round | none | "Neither spawning nor the polar darkness seemed to influence food intake" | [A] |
| 8 | Kaartvedt et al. 2005, MEPS 299:1, p. 4 | saithe 60-110 cm | 200-300 m, May, light night, 70 N | none | stomachs "had fish and the cephalopod Gonatus" | [F] |

Reading of E3 [J]:
- Gadoids keep eating, fish included, at depths where moonless light is below any measured gadoid visual limit.
  Moon phase at sampling is not reported.
- Below ~250 m (K >= 0.05) that holds whatever the moon. So a non-visual and/or bioluminescent floor supports a
  real ration there, at least at cold-water demand (slow evacuation).
- This bounds the dark intake from below but not Ke. No study gives same-size lit-season fullness, prey density
  or encounter data to convert it into R_nv.
- Skaret shows timing set by prey position (capelin reach the bed at dawn), not a light optimum alone.

### E4. Bioluminescence

| # | source | finding | tag |
|---|---|---|---|
| 0 | Berge et al. 2015, Curr Biol | "Thysanoessa spp. (krill) are known to be bioluminescent [37], which might act as a cue for detecting these organisms in the dark" (said of seabird prey); Cohen 2015 likewise names bioluminescence as a polar-night cue | [F] (suggestion, no data) |
| 1 | Walsh & Hickey 1993, p. 68 | bioluminescence discussed (second-hand) as letting fish detect a towed net | [F]/[S] |
| 2 | Fleisher & Case 1995 | squid use dinoflagellate luminescence to locate non-luminous prey ("burglar alarm") | [A], not fish |
| 3 | Warrant 2000; Nilsson et al. 2014 Fig. 2 | point flashes of 1e10-1e11 quanta s-1 (assumed emission) visible at ~10-20 m for a 10 mm pupil | [A]/[F] (model) |
| 4 | Utne-Palm et al. 2018 (prior notes) | krill luminescence (468 nm) lies within cod sensitivity | [S] |

No gadoid study of prey luminescence was found. Krill, Maurolicus and Benthosema emission rates were not found.

## Gaps

1. Meager et al. 2010 (cod R vs light; fitted parameters) is closed, with no repository copy. Glass & Wardle 1989
   (origin of the 1e-6 lx figure) is closed. Blaxter 1970 is a book chapter. These are the best sources for a
   measured cod R curve.
2. No R-vs-light measurement for any gadoid on fish prey, and none for saithe, whiting, hake, pout or ling. The
   only adult-gadoid Ke rests on contrast thresholds of an unknown-size cod. The lamp spectrum and the background
   geometry (pi L vs horizontal radiance) are unresolved, which is a factor of up to ~30 in lx [J].
3. No gadoid retina data: rod density, rod:cone ratio, outer segments, summation, tapetum. Closed: Valen 2016,
   Jonsson 2014 (cod lens), Wagner 1998 (deep retinas), Mas-Riera 1991.
4. No lateral-line, olfactory or tactile detection distance for any marine gadoid on fish prey, and no dark capture
   rate of a piscivorous gadoid. Closed: Enger, Kalmijn & Sand 1989; Pohlmann 2004; New 2001; Harvey & Batty.
5. No light measured at the capture depth of any polar-night gadoid. No same-size lit-season reference. Nothing on
   daytime fish-prey feeding against depth at 150-400 m in the Skagerrak or Norwegian Trench (Bergstad 1991 and
   Giske 1990 are closed).
6. No evidence on gadoid use of prey bioluminescence. No emission rates for krill, Maurolicus or Benthosema.
