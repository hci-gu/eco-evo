# Spawning of the pelagic members in the box: planktivores and mackerel group (2026-11-18)

Research only, by a literature subagent. Nothing in the library, code, tests, manifest, resume or AGENTS.md was
changed. No model output was used to set or select any value. PDFs are in `papers/pelagics/` (not to be
committed); licences are in `papers/README_pelagics.md`.

Context read first and not re-derived: `research_notes/"Ljusstyrd aptit"/summary.md` (herring photoperiod
condition; Norway pout winter feeding, first spawning at age II, post-spawning death) and
`research_notes/"Svaltgrans och startreserv"/fish.md` section 1 (Swedish letters; herring, pout, sprat and
blue whiting fat by season).

## Tags

- [F] full text read by me (pdftotext or Europe PMC XML).
- [Ff] full text read through a web-fetch summariser. Paraphrased; numbers are not verified verbatim, and the
  page was cut at 100 000 characters.
- [A] abstract only (Crossref or publisher page).
- [S] secondary: cited inside another source, or taken from an earlier project note.
- [D] my own derivation; the arithmetic is shown.
- [J] judgement.
- snip = search-engine snippet only.

## Headline table

Reading guide:
- "MJ/t mature/yr" is the reproductive product energy (eggs or milt) per tonne of MATURE fish per spawning
  season. It excludes the extra metabolism of migration and spawning behaviour.
- For comparison, the model reserve ceilings are 4990 MJ/t (planktivores) and 6980 MJ/t (mackerel).

| | Herring (0.554) | Norway pout (0.35) | Mackerel (0.941 of mackerel group) | Blue whiting (0.075) | Sprat (0.021) | Horse mackerel (0.059) |
|---|---|---|---|---|---|---|
| Spawns in box? | **No evidence of spawning in the box.** Box biomass is mostly immature NSAS juveniles; adults are WBSS that build gonads here and leave. Local Skagerrak spring spawners spawn on the adjacent coast [F/D]; that they spawn inshore, not at 50-300 m, is [J] | **Builds gonads in the box; most leave to spawn.** In Jan-Feb, 72 % of age-2 females are maturing, but "few spawners" occur in the Skagerrak [D/Ff/F] | **Box share of stock egg output: small** [F]: main spawning is in the central North Sea, and Skagerrak spawning is "relatively small", on the Norwegian side. **Per fish:** adults present in May-Jul are in their spawning season [J]. May-Jul presence comes from the brief; arrival timing relative to the late-June peak was not found | No evidence. Adults are present Aug-Sep and absent in Jan-Feb [D/snip] | **Probably yes.** Skagerrak-Kattegat peak May-Jul; 47-91 % of 10-14 cm fish maturing in Jan-Feb [snip/D] | No evidence for the box; not researched in depth [J] |
| Peak GSI, F / M | F ~17-20 % of body mass, M ~15-20 % [F/S/D] | Not found. January ovaries 2.0-5.8 % of body mass (not peak) [F/D] | Not found (batch spawner, so peak GSI is not the season total) | Not found | Not found (GSI highest in spring, snip) | Not found |
| Gonad kJ/g wet | Ovary 6.4-7.8; testis 4.5-5.2 [D from F; lower %DW bounds J] | Egg 1.34 J/mm³ (no oil globule) [D from F] | Egg 1.80 J each (modelled) [F] | - | - | - |
| MJ/t mature/yr | F 1100-1560, M 680-1040, sex-averaged ~900-1300 [D] | F 800-1500 (central ~1100); M not found [D] | F 2000-2700; M not found; [J] sex-averaged 1300-2700 | - | - | - |
| Timing (box-relevant component) | WBSS: build-up autumn-winter, spawn Mar-May (outside box). NSAS adults: build-up summer, spawn Aug-Oct / Dec-Jan (outside box) [S/J] | Build-up Oct-Feb, spawn mid-Feb to Apr (peak mid-Feb), most die Q1->Q2 [F/Ff] | North Sea spawning May to July, peak late Jun-early Jul [F] | Spawning Mar-Apr west of the British Isles; eggs in Norwegian fjords Mar-Jul [snip] | May-Jul peak; Norwegian Skagerrak coast Feb-Jul [snip] | Long season, 8 months for the population [snip] |
| Reserve share of gonad energy | Capital: ~1 for spring spawners; autumn spawners partly financed from food [A/F] | Unknown; pout feed all winter on krill [S] | Low: income breeder, and the lipid budget was negative in 18 of 30 years [F] | Unknown | Unknown (indeterminate batch spawner) [F] | First batch from stores, the rest from income [snip] |
| Feeding reduction x duration | **No measured intake for any species.** NSS herring do not feed from Sep-Oct to end of spawning (stock-specific). Elsewhere "lower stomach contents while developing" [F/S] | Weak dip: "low feeding intensity" Jan and Mar-Apr ("inconclusive") [F] | **None.** Feeding is highest during the spawning season [F, field stomach fullness] | - | Not found | Not found |
| Recovery | Fat 2-5 % -> 16-24 % muscle or fillet within ~2-3 months of spawning, on feeding [S] | "Gain rapidly in condition in the spring", for survivors [F] | Growth until late Sep [F] | - | K highest in autumn [snip] | - |
| Weight loss at spawning | ~17-20 % (female, the ovary shed) [D] | Most age-2+ spawners die (non-predation M1 1.7-2.0 at age 2) [A] | Weight declines from Oct to spawning; % not given [F] | - | - | - |
| Confidence | Medium (components, energy); low (feeding) | Medium (where they spawn); low (energy) | Medium-high (income breeding); low-medium (share spawned in box) | Low | Low | Very low |

**Bottom line [J].**
1. **No pelagic member is a clear "spawns in the box" case with a large biomass share.**
   - Herring: the box holds mostly immature fish.
   - Pout: mature fish build gonads in the box in winter, and most then leave or die.
   - Mackerel: an income breeder. The Skagerrak's share of stock egg output is small, but adults present in
    May-July are presumably spawning fish [J].
2. **The case with the most mass is Norway pout.**
   - The age-2 cohort (the bulk of spawners) is ~72 % maturing in January-February. It is semelparous in effect.
   - Its gonad cost is followed by emigration and death, not by reserve recovery.
3. **The scratch-test problem ("the drain is eaten back") is consistent with the literature for mackerel.**
   - Mackerel eat most while spawning.
   - Non-feeding during maturation is documented only for capital-breeding herring stocks that are not resident in
     the box (NSS).

## Q1. Spawning in the eastern Skagerrak; components in the box

### Box DATRAS check (NS-IBTS GOV, rectangles 44G0, 44G1, 45G0, 45G1, 2000-2025) [D]

Source: the files already in `data/datras/` (fetched by another agent); I only read them.
- Final run on the completed fetch: hh 1834, hl 265 448 and ca 115 771 lines; box hauls 2000-2025.
- The separately refetched 2001 Q1 CA (`data/datras_2001q1_ca`) is not included.
- The scripts were scratch only; the method is below.

Method:
- Each CA record was mapped to a box haul via HH.
- Maturity was collapsed to: I (1/61/A), M (2/62/B), S (3/63/C), P (4/64/D), R (5/65/E).
- Q1 hauls are January (215) and February (138); Q3 hauls are August (186) and September (134).
- Biomass shares use HL counts x SubFactor x a L^3.1. The a, b values are my placeholders [J] (herring a 0.0068,
  pout 0.0067, sprat 0.0058).
- **GOV catch is not the pelagic biomass composition.** It is indicative only.
- **Q3 stage I at adult length is very likely resting adults** (the old 4-stage scale has no resting stage). Q3
  "mature shares" are therefore not reported as maturity.
- **Q1 hauls (Jan-Feb) precede the pout and sprat spawning peaks.** A lack of running fish is not evidence that
  they leave.

Herring, Q1 maturing share (M+S+P+R) by length:

| length | share maturing |
|---|---|
| < 20 cm | ~0.01 |
| 20-24 cm | 0.20 |
| 25-29 cm | 0.71 (spawning stage 0.08) |
| 30-34 cm | 1.00 (spawning stage 0.14) |

- GOV weight share in Q1: >= 20 cm 0.23, >= 23 cm 0.08, >= 25 cm 0.03.
- Biomass-weighted maturing share in Q1: **0.06**.
- In Q3 (Aug-Sep), 25-29 cm fish are 0.73 "maturing" and 0.07 resting; 30-34 cm fish are 0.89 maturing.

Norway pout:

| group | Q1 maturing share |
|---|---|
| 10-14 cm | 0.20 |
| 15-19 cm | 0.93 (spawning stage 0.01) |
| 20-24 cm | 1.00 (spawning stage 0.05) |
| age-2 females | 172 maturing + 2 spawning of 240 (0.73) |
| age-2 males | 67 of 84 (0.80) |
| age-3 females | 24 maturing + 2 spawning of 27 |

- 4 spawning-stage pout (all females, ages 2-3) were found in 2000-2025 Q1. Q1 hauls end in February.
- Females outnumber males about 2.3:1 at age 2+, as Lambert et al. report for the Skagerrak.
- Biomass-weighted Q1 maturing share: 0.22. Q1 GOV weight is 87 % 10-14 cm (the 1-group).

Sprat, Q1 maturing share by length:

| length | share maturing |
|---|---|
| 8-9 cm | 0.09 |
| 10-11 cm | 0.42 |
| 12-13 cm | 0.73 |
| 14-15 cm | 0.90 |

- Biomass-weighted: 0.27.

Mackerel:
- CA has no usable maturity records in the box.
- In Q3, 0.78 of the GOV weight is >= 28 cm and 0.96 is >= 25 cm. North Sea mackerel are > 90 % mature at age 2
  (ICES FishMap [F]), so the box mackerel are mostly adults [D/J].

Blue whiting:
- Q1: n = 15, essentially absent.
- Q3: 0.91 of the weight is >= 22 cm, i.e. adult [D/J: L50 not checked].

### Herring

- **Components** (Bekkevold et al. 2023, doi:10.1093/icesjms/fsac223 [F]; genetic mixed-stock analysis of HERAS
  2019-2021 and time series 2002-2021):
  - In the North Sea-Skagerrak-Kattegat, the NSAS and Downs fish "were largely juveniles", while "WBSS, BAS, and CBH
    were largely adults".
  - "Two collections of WBSS representing populations spawning in Norwegian and Swedish Skagerrak coastal waters
    showed strong divergence from other WBSS (Fst > 0.17)". These are spawning samples from Skagerrak East
    (Öckerö, 57.60 N) and Skagerrak West (Hövåg, 58.15 N).
  - This local Skagerrak sub-stock made up 14 % of WBSS assignments in mixed catches. The authors: "a limited
    contribution of coastal Skagerrak herring to mixed stocks", "a recent increase".
  - Baltic autumn spawners (BAS) contribute < 10 % in most hauls.
- **Spawning in the box: no evidence found.**
  - Not checked: a spawning-ground map, a Swedish coastal spawning study and MIK larval data.
  - The local spring spawners spawn on the coast adjacent to the box [F]. Öckerö lies ~20 km south of the box.
  - [J, unsourced here] Herring spawn on benthic substrate in shallow coastal water, which the 50-300 m offshore
    box lacks.
  - WBSS adults spawn in the western Baltic (Rügen, the main sub-stock).
  - NSAS spawn off Scotland, on the banks and in the Channel (Downs).
  - The DATRAS spawning-stage herring in Jan-Feb (6-13 % of 25-34 cm fish) are few, and their stock is unknown
    (Downs winter spawners or early spring spawners) [D/J].
- **Classification [J]:**
  - Immature in the box: ~0.94 of the GOV herring weight.
  - Builds gonads in the box but leaves to spawn: the WBSS adults present in winter.
  - The local Bohuslän coastal spring spawners spawn adjacent to the box, not in it.
- Not done: IBTS MIK larval data were not examined.

### Norway pout

- Lambert et al. 2009, doi:10.1093/icesjms/fsp153 [Ff]:
  - Spawning is mainly in Q1 around mid-February, in RFAs 1 and 3 (Viking Bank, Scottish east coast), and around
    the 120 m isobath along the Norwegian Trench.
  - "Few spawners were found in the Skagerrak and the Kattegat".
  - Males mature earlier (A50 1.2 yr, L50 11.7 cm; females 1.5 yr, 13.1 cm) and "migrate to spawn before females
    and ... neither return".
  - Females dominate in the Skagerrak (age-2 female share 0.86 in Q1).
- Raitt 1968 (FAO Fisheries Synopsis 33 Rev. 1) [F]:
  - "In the Skagerrak there is a large scale spawning migration out into the North Sea and the subsequent larvae
    and juveniles drift back into the Skagerrak and Kattegat (Poulsen, 1964 and 1966)".
  - Collett found "specimens spawning in the Oslofjord in February-March".
- Nash et al. 2012, doi:10.1093/icesjms/fss130 [F], stage-I eggs identified by TaqMan:
  - The principal spawning is east and south-east of Shetland and along the western slope of the Norwegian
    Trench. Larger larvae lie to the south-east, "toward the Skagerrak".
  - Caveat: "the distribution to the east is unknown as no sampling was carried" out (eastern Trench and
    Skagerrak unsampled).
- Albert 1994, doi:10.1006/jmsc.1994.1005 [A, and S via the earlier note]:
  - In the Norwegian Deep the mature part was most abundant to the north-west in the spawning season.
  - "Close to 100% matured for the first time as II-group".
- **Classification [J]: builds gonads in the box (Oct-Feb); most leave to spawn west or north-west.** Minor local
  spawning (Norwegian Trench slope, Oslofjord) is not excluded. A survivor does not return (Lambert's hypothesis)
  or dies.

### Mackerel

- Iversen 1977, ICES CM 1977/H:17 [F], egg surveys 1968-1975:
  - "The most important spawning area is located in the central part of the North Sea. The spawning along the
    coasts both in Skagerrak and the North Sea is scarce."
  - "The spawning in Skagerrak during these years is also relatively small compared to the area further west.
    The concentration of mackerel eggs in this area is higher along the Norwegian coast than along the Danish
    coast."
  - Spawning starts in late May.
- Jansen & Gislason 2013, doi:10.1371/journal.pone.0064744 [F]:
  - NE Atlantic mackerel spawn "from Hatton Bank in the West to Kattegat in the East".
  - Their Fig. 1 marks every egg observation >= 50 eggs m-2 d-1 in North Sea surveys 2002-2011. All dots lie in
    54-60 N, 2 W-4 E; none are in the Skagerrak, although Skagerrak coverage by those surveys was limited.
  - "Spawning in the North Sea begins in May, peaks in late June or early July and ceases during July".
- ICES FishMap factsheet [F]: spawning "mainly situated in the central North Sea, with extensions along the
  southern coast of Norway and in the Skagerrak". Myrberget (1957-1959) found eggs in practically all Skagerrak
  areas [snip].
- Lindquist (1968) describes mackerel spawning while migrating from the outer Skagerrak eastward, with eggs and
  larvae across the whole Skagerrak, densest off the Norwegian coast [snip].
- **Classification, two separate statements:**
  - (a) The box's share of the stock's egg output is small [F: Iversen, Fig. 1 of Jansen & Gislason]. These are
    eggs per area, which track how many fish are present.
  - (b) Per tonne of mature mackerel present, a > 90 % mature batch spawner present in late May-July is in its
    spawning season and is presumably shedding batches, unless it arrives spent [J].
    - May-July presence is from the brief, not DATRAS: the Q3 adult share is Aug-Sep, after spawning.
    - Arrival timing relative to the late-June peak was not found.
- No quantitative share of the season's batches shed in the box was found.

### Sprat (short)

- Vitale et al. 2015, Aquatic Living Resources 28:127-137, doi:10.1051/alr/2016007 [snip only]:
  - Skagerrak, Kattegat and the Uddevalla fjords: "highest proportion of spawning fish was recorded from May to
    July".
  - "K being the highest in autumn and lowest in spring, while the GSI index was highest in spring".
- The Norwegian Skagerrak coast spawns Feb-Jul [snip].
- IMR sprat narrative 2021 [F]: an "indeterminate batch spawner", "up to 10" batches (George & Alheit 1987). In
  3.a and 4, sprat are a mixture of North Sea sprat and hybrids near the Swedish coast.
- With the DATRAS Q1 maturing shares above: **probably spawns in or near the box** [J].

### Blue whiting (short)

- Main spawning is west of the British Isles in Mar-Apr. An ICES 1983 paper found eggs and larvae in Norwegian
  fjords and over the deeper shelf, Mar-Jul [snip]. Spawning needs > 8-9 C and salinity > 35.2-35.3 (Henderson
  1957 via a Faroese paper) [snip].
- Nothing found on spawning in the Skagerrak.
- DATRAS: adults in Aug-Sep only.
- **No evidence [J].**

### Horse mackerel (short)

- Not researched in depth; no box-specific evidence.
- Income breeder with indeterminate fecundity: "GSI, fecundity, and body condition [increase] after the onset of
  spawning"; the "first batch of oocytes is developed on stored energy" (van Damme et al. 2014) [snip].
- Individual season 65-94 days, 5-16 batches (Abaunza) [snip].

## Q2. Gonad investment

### Herring

- **Female GSI:**
  - NSS: "gonad mass just prior to spawning averages 19% of the body mass" (Varpe et al. 2005 via Warren et al.
    2026) [S].
  - Slotte & Fiksen 2000 (J Fish Biol 56:138-162) [F]: the mean ovary weight of fully mature NSS females off Møre
    is Gmax(l) = 3387.8 - 322.82 l + 10.106 l^2 - 0.1023 l^3 (g, l in cm; n 568).
  - [D] At 32 cm, Gmax = 3387.8 - 10330.2 + 10348.5 - 3352.2 = 53.9 g. Body weight is ~315 g ([J] W = 0.0068
    L^3.1), so ~17 % of body mass.
- **Male GSI:**
  - Slotte 1999, doi:10.1111/j.1095-8649.1999.tb00834.x [A]: at the onset of wintering the GSI "was higher ... in
    male than in female herring"; the subsequent rise was faster in females.
  - Warren et al. 2026, doi:10.1111/jfb.70400 [F] (Gulf of Maine): males reach GSI > 20 %, and "testes energy did
    not exceed 17% of the somatic energy and plateaued as GSI increased further". Female ovary energy kept rising
    linearly above GSI 17 %. GSI is defined as gonad / (wet weight - gonad).
- **Energy density:**
  - Warren et al. 2026 [F]:
    - ovary ED = 0.03 + 0.212 %DW (adj. r2 0.961), maximum %DW 36.9;
    - testis ED = -0.54 + 0.212 %DW, maximum %DW 27.1;
    - lipid 39.565 and protein 23.64 kJ/g.
  - [D] Ovary at %DW 30-36.9 = 6.4-7.85 kJ/g. Testis at %DW 24-27.1 = 4.5-5.2 kJ/g. The lower %DW bounds (30, 24)
    are my choice for ripe gonads [J]; the maxima are measured.
  - Pacific herring ovaries are 5-7 kJ/g before spawning (Paul, ADF&G) [snip], consistent with this.
- **Per tonne mature [D]:**

| | GSI (as share of total mass) | ED (kJ/g) | MJ/t |
|---|---|---|---|
| female | 0.17-0.20 | 6.4-7.8 | 1090-1560 |
| male | 0.15-0.20 | 4.5-5.2 | 680-1040 |

  - Sex ratio ~1:1 [J; not measured for box herring]. Sex-averaged ~890-1300 MJ/t mature, i.e. ~0.18-0.26 of
    the planktivore reserve ceiling.
  - Herring are total (single-batch) spawners: Pacific herring expend ~97 % of ovarian energy at spawning
    [snip]. So the peak gonad energy is roughly the season's output.
- **At box scale [D]:** the Q1 maturing share is 0.06 of the GOV herring weight. 0.06 x ~1100 = ~65 MJ per t of
  box herring per year, and even that is paid on spawning grounds outside the box.

### Norway pout

- **Ovaries:** Fulton 1904 (in Raitt 1968, Table 2) [F]: January ovaries of 15-20 cm fish weigh 0.5-3.4 g on
  24.7-67 g fish, i.e. 2.0-5.8 % [D] (maximum 1.8/31.0). This is mid-maturation, not peak.
- **Fecundity:** Raitt 1966b in Raitt 1968 [F]: age-2 fish of 15, 17 and 19 cm have mean fecundities of 27 000,
  38 000 and 51 200.
- **Egg energy [D]:**
  - Pout eggs have no oil globule (Raitt: "without oil globule"). Stage-I egg diameter is 1.03-1.28 mm (Nash et
    al. 2012 [F]).
  - The general egg-energy model (Riis-Vestergaard 2002, as used in Jansen et al. 2021 [F]) is E = V (1.34 +
    40.61 P_oil) J/mm3, which is V x 1.34 with no oil.
  - d 1.03 mm: V 0.572 mm3, 0.77 J. d 1.15: 0.796, 1.07 J. d 1.28: 1.098, 1.47 J.
- **Female energy [D]:**
  - Weights from Fulton's table: 15 cm ~25 g, 17 cm ~35 g, 19 cm ~50 g [J interpolation]. Relative fecundity is
    1020-1090 eggs/g.
  - 1050 eggs/g x 0.77-1.47 J = 0.81-1.54 kJ/g = **800-1500 MJ/t female** (central ~1100).
  - Raitt's counts are probably potential fecundity, so realised output is lower by any atresia [J].
- **At box scale [D]:** the Q1 maturing share is 0.22 of the GOV pout weight (age-2+ ~0.73-0.80, the 1-group
  ~0.20). 0.22 x ~1100 x ~0.5 (female share of energy, males unknown) to 0.22 x ~1100 = **~120-240 MJ per t of box
  pout per year**, paid in Oct-Feb. The spawning act and the deaths are mostly outside the box.
- **Males:** not found.
- **Weight loss:** Lambert [Ff] reports a "notable decrease in MWA [mean weight at age] in the western North Sea
  from Q1 to Q2 for age 2" (the spawning ground).

### Mackerel

- Jansen et al. 2021, doi:10.1016/j.pocean.2021.102658 [F]:
  - Egg energy is 1.80 J. This is modelled (oil globule 0.31 mm plus cod-like yolk), not measured.
  - The ICES mean relative realised fecundity is 1134 eggs/g female (1998-2019). The authors assume a maximum of
    1500 eggs/g. Relative batch fecundity is ~40 eggs/g, so 1134-1500 eggs/g means ~28-38 batches; "from around
    20 up to an unknown maximum".
  - The protein lost over the reproductive cycle (Oct-May) is 15 g per fish at age 2, rising to 22 g at age 9.
  - The sex ratio is 0.5 in every month (N = 90 606).
- ICES FishMap [F]: about 20 batches; 255 000 ripe eggs for a 30 cm, 200 g fish = 1275 eggs/g.
- **Per tonne [D]:**
  - Female: 1134-1500 eggs/g x 1.80 J = **2.0-2.7 kJ/g = 2040-2700 MJ/t female per season**.
  - Male: no testis data found. [J] At 0.3-1.0 of the female value, sex-averaged is ~1300-2700 MJ/t mature, i.e.
    ~0.19-0.39 of the 6980 MJ/t mackerel reserve ceiling.
  - Only the part shed while in the box (May-July) would be a box term, and its share is unknown.

### Sprat, blue whiting, horse mackerel

- No gonad energy found.
- Sprat absolute fecundity in a Skagerrak fjord (1985-1986) was 2 900-79 000 eggs, mean 21 000 and 12 300, and
  probably an underestimate because spawning is indeterminate [snip].

## Q3. Timing

| Component | Gonad build-up | Spawning | Evidence |
|---|---|---|---|
| WBSS herring (adults in the box in winter) | Autumn to winter; fillet fat falls from 10.9-12.7 % in Jan to 4.7-5.7 % in Mar and 2.1-7.1 % in Apr (Rügen, Munkner 1999) | Mar-May, outside the box | [S, earlier note], [J] |
| NSAS herring adults | During the summer feeding season (Bradford: autumn spawners mature while feeding) | Aug-Oct (northern); Dec-Jan (Downs) | [S], [J] |
| Local Skagerrak spring spawners | Not found | Spring, coastal | [F Bekkevold: spawning samples], months not found |
| Norway pout | Germ cells develop "most active during late winter and early spring"; in the box, maturing in Jan-Feb | Mid-Feb (peak) to April; newly hatched larvae imply spawning "through to the end of April" | [F Raitt; F Nash; Ff Lambert; D DATRAS] |
| Mackerel | Starts 1 Oct (Jansen's cycle start); 98 % resting mid-Aug to end-Nov | North Sea late May to July, peak late Jun-early Jul; western stock from Feb | [F] |
| Sprat | Winter to spring | May-Jul peak (Kattegat-Skagerrak); Feb-Jul on the Norwegian coast | [snip], [D] |
| Blue whiting | - | Mar-Apr, west of the British Isles | [snip] |

## Q4. Source of gonad energy (capital vs income)

- **Herring.**
  - Slotte 1999 [A]: NSS herring are non-feeding. "The distinct energy loss was attributed mainly to fat
    metabolism, whereas the storage protein only was transferred from the soma to the gonads".
  - Slotte & Fiksen 2000 [F]: "NSS herring do not feed during the period from the arrival in the wintering area
    in September-October until the end of spawning in March-April"; herring "have on the average c. 10% fat left
    after spawning". The 60 % energy-retention rule in their model is an input, not a measurement.
  - Warren et al. 2026 [F]: early spawners "occupy the extreme capital end of the spectrum", while fall spawners
    "may be able to reduce energy losses to some extent through feeding". This is consistent with Bradford 1993 [S]:
    spring spawners have a long non-feeding maturation, autumn spawners a brief maturation with feeding.
  - Reserve share [J]: ~1 for spring spawners (WBSS, the box's adults); lower for autumn spawners, not quantified.
- **Mackerel** (Jansen et al. 2021 [F]):
  - "Mackerel feed intensively during the spawning season (mainly after March) regardless of the maturity stage";
    an "indeterminate, income breeder".
  - "The lipid energy budget was negative in 18 of the 30 years indicating that additional energy was needed to
    sustain egg production"; "The mackerel must therefore depend on food intake".
  - Their framework charges eggs to surplus protein and lipid to maintenance and migration. That split is an
    assumption.
  - Reserve share [J]: low. The lipid store does not finance the eggs.
- **Norway pout.**
  - No source found on capital vs income.
  - Feeding continues all winter (Albert 1994 [S]). A "pronounced reduction in condition factor and liver weight
    during the winter season" coincides with first spawning [S].
  - [J] Mixed, not quantified.
- **Sprat, horse mackerel:** income-leaning batch spawners [snip for horse mackerel; J for sprat].
- **General:** McBride et al. 2015, doi:10.1111/faf.12043 [A]: "Many species exhibit mixed capital- and
  income-breeding patterns", and the position on the continuum "may shift with ontogeny or in relation to
  environmental conditions".

## Q5. Feeding during maturation and spawning

**No captive or other intake measurement during maturation or spawning was found for any species.** No fraction x
duration can be given from data.

- **Herring.**
  - Non-feeding is documented for NSS (above [F]), a stock not resident in the box.
  - Link & Burnett 2001, doi:10.1111/j.1095-8649.2001.tb00149.x [A, and S via Warren]: NW Atlantic species fall
    into "ramper" and "rester" feeding-reproduction patterns. Herring increased feeding after spawning; "the amount
    of prey in the stomach was significantly lower during the developing stage compared to the post-spawning
    stages". This is stomach contents, not intake.
  - Warren et al. [F] give a mechanism hypothesis: "the enlarged gonads likely limit gut capacity due to spatial
    constraints within the coelomic cavity" (Weeks 1996).
  - Ma, Kjesbu & Jørgensen 1998, doi:10.1139/f97-305 [A]: captive NSS-stock herring were fed 12 mg g-1 d-1 krill
    (high ration) through 1.5 years of maturation. Half ration cut potential fecundity by 26 % and modelled
    realised fecundity by 40 %. The abstract does not say whether the fish reduced intake during late maturation.
  - For box herring (~0.94 immature by GOV weight) a maturation feeding stop would touch ~6 % of the herring
    biomass, and only in winter [D/J].
- **Mackerel:** feeding is highest during spawning (field stomach fullness, IMR 1993-2003, Jansen et al. 2021 [F]).
  This is the opposite of a stop. The fraction of empty stomachs has risen since 2010.
- **Norway pout:**
  - Raitt 1968 [F]: in adults "there may be three periods of low feeding intensity, in January, March-April (the
    main spawning period) and August"; the data are "inconclusive". Juveniles showed "no marked variations" from
    September to April.
  - Albert 1994 [S]: stomach contents relative to fish weight were higher in winter than in summer (Norwegian
    Deep).
- **Sprat, blue whiting, horse mackerel:** not found.

## Q6. Post-spawning recovery

- **Herring** [S, the earlier fish.md series]:
  - Downs: muscle fat 2.0 % in Feb (post-spawn) to 15.6 % in May and 19.6-20 % in Jun-Jul (NIFES).
  - Downs, whole body minus gonads: 4.7 % in early May to 23.6 % in early Jul (Wood 1958).
  - WBSS fillet: 2.1-7.1 % in April; Kattegat fillet in autumn 15.9 %.
  - Recovery to near peak takes ~2-3 months of summer feeding.
  - Spring spawners show a secondary energy-density peak in late October (Warren [F]).
- **Norway pout:**
  - Survivors "gain rapidly in condition in the spring" (Raitt [F]).
  - Lambert [Ff], maturity at age 4: ~100 % in Q1 and 5 % in Q2. This suggests that few survivors remain or that
    survivors skip spawning [J].
- **Mackerel:** weight and length grow until late September (Jansen et al. 2021 [F]).

## Q7. Weight loss at spawning; post-spawning mortality

- **Herring:** the shed ovary is ~17-20 % of female body mass [D/S]. Pacific herring spend ~97 % of ovarian energy
  [snip]. The post-spawning fat floor is ~10 % (NSS, Slotte & Fiksen [F]). Male loss is not quantified.
- **Norway pout:**
  - Sparholt, Larsen & Nielsen 2002, doi:10.1006/jmsc.2002.1305 [A via fetch], non-predation natural mortality M1:

| age | year classes 1977-1981 | year classes 1987-1991 |
|---|---|---|
| 1 | 0.10 | 0.10 |
| 2 | 1.74 | 2.03 |
| 3 | 2.58 | 3.04 |
| 4 | 3.05 | 4.39 |

    "the main mortality occurs between the first and second quarters", i.e. from before to after spawning. The
    time unit is not stated in what I saw; per year is presumed [J]. Per year, age-2 survival from M1 alone is
    exp(-1.74 to -2.03) = 0.13-0.18 [D].
  - Nielsen et al. 2012, doi:10.1093/icesjms/fss001 [A + Ff]:
    - "natural mortality (M) is significantly correlated with sexual maturity, sex, growth, and intraspecific stock
      density", "consistent with a greater mortality occurring mainly from the first to the second quarter".
    - The assessment still uses M = 0.4 per quarter.
    - In 2005 (fishery closed), M at ages 1-3 was 1.59, 1.81 and 2.11.
    - Males have higher Q1 mortality than females (difference "0.2", unit not confirmed).
    - Predation mortality on the spawning grounds during spawning is low.
- **Mackerel:** no % given. Weight declines from October to spawning [F]. No spawning mortality reported.

## Not found (explicit)

- Any intake measurement (captive or bioenergetic) of herring, pout, mackerel or sprat during maturation or
  spawning.
- Peak GSI and gonad energy of Norway pout. Testis data for pout and mackerel.
- Gonad energy of sprat, blue whiting and horse mackerel.
- Spawning months of the local Skagerrak coastal herring.
- Any quantitative share of North Sea mackerel egg production in the eastern Skagerrak.
- Blue whiting spawning in the Skagerrak.
- IBTS MIK larval data for the box (not examined).

Full texts not obtained (paywalled or blocked):
- Kurita et al. 2003, doi:10.1016/S1385-1101(03)00004-2.
- Ma et al. 1998 (the repository copy is behind NVA access control).
- Link & Burnett 2001.
- Albert 1994.
- Lambert et al. 2009 and Nielsen et al. 2012 (read through a fetch summariser only).
- Vitale et al. 2015 (site refused).
- Bradford 1993; Varpe et al. 2005; van Damme et al. 2014.

## Implications for the model's spawning term [J]

1. **Herring (0.554):** a gonad term inside the box applies to only ~6 % of the box herring weight (WBSS adults in
   winter), and they spawn elsewhere. If included, it is a capital drain of ~900-1300 MJ/t mature over Oct-Mar,
   with a feeding stop supported only for a non-resident stock.
2. **Norway pout (0.35):**
   - The decisive pelagic case. ~72 % of age-2 females mature in Jan-Feb, the gonad is ~800-1500 MJ/t female, and
     a mass loss (emigration or spawning death; Sparholt M1 ~1.7-2.0 at age 2) follows in Q1-Q2.
   - That is a biomass loss, not a reserve drain that can be eaten back.
   - Feeding continues in winter, so a feeding stop is not supported.
3. **Mackerel (0.941 of its group):**
   - An income breeder that feeds hardest while spawning. A reserve drain will be eaten back, and the literature
     says that is what happens.
   - The season's ~2000-2700 MJ/t female is shed over ~20-38 batches. The part shed while in the box (May-July)
     is unknown: possibly most of it per fish present [J], but the arrival timing was not found.
4. **Sprat (0.021):** likely spawns in or near the box in May-Jul, but it is a negligible share.
