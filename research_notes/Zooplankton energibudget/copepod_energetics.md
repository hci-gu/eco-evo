# Copepod (mesozooplankton) feeding energetics, mortality and stocks - literature check of the Mareld `zooplankton` parameters

Assemblage being judged: Skagerrak/Kattegat mix, ~2/3 small neritic copepods (Acartia, Temora,
Pseudocalanus, Centropages, Oithona) and ~1/3 Calanus finmarchicus/helgolandicus, 8-12 C.

## Conventions

- 1 t/km2 = 1 g/m2. Model tick = 6 h, so per tick = per day / 4 and per year = per day x 365.
- Verification marks: **[V]** = read by me in the full text, a postprint, a data file or (where
  stated) the abstract; **[S]** = taken only from a search-engine snippet/summary or a secondary
  quotation, NOT read in the source.
- Conversion factors used in the Inferences (all are assumptions unless cited):
  - Copepod C/WW = 0.084: Brey (2001) WW->DW 0.186 x DW->C 0.451, as tabulated in Mackinson &
    Daskalov (2007) Appendix 2 [V]. The appendix row is labelled "Dry Mass to Wet Mass (Copepods)
    1 to 0.451", which is almost certainly a mislabelled DM->C factor; it agrees with the 45 %
    C of DW (Kiorboe et al. 1985) that the same report quotes for Nielsen & Richardson (1989) [V].
    Kiorboe (2013) gives 5-10 % C of WW for non-gelatinous zooplankton [S].
  - Phytoplankton C/WW = 0.1: Ecopath convention "0.1 g C = 0.2 g dry weight = 1 g wet weight"
    (Mackinson & Daskalov 2007, section 7 and Appendix 2) [V]. Real cell-level values range
    ~0.03-0.16 (Q7).
  - Energy per carbon: ~45-48 kJ/g C for both phytoplankton and copepods (Q7).
- "Body C d-1" = carbon-specific rate (ug C ingested per ug body C per day). In the model,
  energy is the currency, so the comparable model quantity is ingested energy / body energy.

Model values being judged (from the task brief and `fgconfig/fg_library.yaml`): max ingestion
`max_intake_rate` 0.25 t phyto WW / t zoo WW / tick; `handling_time` 4 (a*h = 1); assimilation
0.65; phyto energy 2000 MJ/t WW; zoo energy 4500 MJ/t WW; reserve max 675 MJ/t; resting
metabolism 22.5 MJ/t/tick (x2 when feeding); appetite h = 1 - s, maintenance level s = 0.5;
non-fish natural mortality 19.7 /yr (0.0134 /tick).

---

## Q1. Maximum ingestion rate vs temperature and body size; is 100 % body WW per day plausible?

### Takeaway
Laboratory maximum ingestion of the relevant species, temperature-corrected to 10 C with
Q10 = 2.8, has a median of ~0.44 body C d-1 for small neritic copepods (range 0.12-2.3),
~0.29 (0.13-0.50) for Calanus CV/adult females and ~0.9 (0.6-2.0) for Calanus nauplii/early
copepodites. Saiz & Calbet (2007) found lab maximum feeding to be temperature-independent; read
that way (no correction), the medians are ~0.75, ~0.48 and ~1.5 body C d-1. The model's
"100 % body WW per day" equals **0.44 body energy d-1** in the model's own units
(0.25 x 2000/4500 x 4), which sits at the Q10-corrected median and about half the uncorrected
values: plausible to low. Expressed instead with Ecopath-style C/WW factors (0.1 phyto, 0.084
copepod), the same number is 1.0-1.2 body C d-1, which is high (Acartia-at-18-20-C level).
The verdict depends on the phytoplankton C/WW that the 2 kJ/g WW energy density implies (Q7).

### Cited Findings
- Kiorboe, Mohlenberg & Hamburger (1985), Acartia tonsa, food-acclimated to 0-1700 ug C/L:
  "Ingestion and egg production rates increased sigmoidally with food concentration approaching
  plateaus equivalent to 180 and 64 % body C d-1, respectively"; clearance peaked at 150 ug C/L;
  respiration and excretion of copepods fed at saturating food were "more than 4 times higher
  than those for starved individuals" [V, abstract; the temperature is not stated in the
  abstract] - [Kiorboe et al. 1985, MEPS 26:85-97](https://doi.org/10.3354/meps026085)
- Kiorboe, Saiz, Tiselius & Andersen (2018) derive from Kiorboe et al. (1985) for A. tonsa on
  Rhodomonas: Fmax = 0.65 ug C (ug dry body weight)-1 d-1, max clearance beta = 1.65 mL
  (ug DW)-1 d-1, starvation metabolism m0 = 0.015 and feeding metabolism mf = 0.1 ug C
  (ug DW)-1 d-1 (mf includes processing costs and defecation losses) [V, postprint] -
  [Kiorboe et al. 2018, L&O 63:308-321, postprint at DTU Orbit](https://backend.orbit.dtu.dk/ws/files/141852543/Postprint.pdf)
- Durbin, Durbin & Wlodarczyk (1990), A. tonsa adult females: daily ingestion at high
  Thalassiosira weissflogii at 20 C ~148 % body C (104 % body N); 74 and 85 % body C at ~13 C on
  natural food in mesocosms; gut evacuation rate 0.09 min-1 at 20 C, 0.042-0.043 min-1 at
  12.7-13.3 C [V, abstract] - [Durbin et al. 1990, MEPS 68:23-45](https://doi.org/10.3354/meps068023)
- Saiz & Calbet (2007), literature review: "Maximum feeding rates of copepods, determined in the
  laboratory, were temperature independent and scaled in conformity to three-quarters universal
  law"; field rates depended on food, temperature and size (81 % of variance), with a much lower
  mass slope "indicating severe food limitation in the larger copepods" [V, abstract; regression
  coefficients not retrieved] - [Saiz & Calbet 2007, L&O 52:668-675](https://doi.org/10.4319/lo.2007.52.2.0668)
- Hansen, Bjornsen & Hansen (1997): maximum ingestion and clearance from lab functional responses
  of ~60 species; within-group exponent -0.23 (+-0.12) on body volume; "calanoid copepods have
  maximum ingestion and growth rates that exceed those of filter-feeding cladocerans and
  meroplankton larvae by a factor of 10" [V, abstract; coefficients not retrieved] -
  [Hansen et al. 1997, L&O 42:687-704](https://doi.org/10.4319/lo.1997.42.4.0687)
- Kiorboe & Hirst (2014): ingestion and growth follow a near-universal ~3/4 mass-scaling across
  ~10^15 in body mass; rates standardised to 15 C with Q10 = 2.8 [S for the paper itself;
  the Q10 is confirmed in Brun et al. 2017 [V]] - [Kiorboe & Hirst 2014, Am. Nat. 183:E118-E130](https://doi.org/10.1086/675241)
- Brun, Payne & Kiorboe (2017) copepod trait database: adult carbon-specific maximum ingestion at
  15 C ranges from 15 (Calanus pacificus) to 116 ug C h-1 mg C-1 (Euterpina acutifrons); max
  growth at 15 C from 5 to 19 ug C h-1 mg C-1 (C. finmarchicus highest); temperature corrections
  with Q10 = 2.8 [V, full text] - [Brun et al. 2017, Earth Syst. Sci. Data 9:99-113](https://doi.org/10.5194/essd-9-99-2017)
- The same database (PANGAEA xlsx, sheet "Ingestion rates", records from Kiorboe & Hirst 2014;
  "Specific Imax (15 C)" in ug C h-1 mg C-1) [V, data file] -
  [PANGAEA doi:10.1594/PANGAEA.862968](https://doi.org/10.1594/PANGAEA.862968). Selected records
  (my unit conversion x24/1000 to d-1; 10 C column = 15 C value / 2.8^0.5):

| Taxon (stage) | Exp. T (C) | Body (ug C) | Imax 15 C (d-1) | Imax at 10 C, Q10 2.8 (d-1) |
|---|---|---|---|---|
| Acartia clausi | 15 | 5.0 | 1.30 | 0.77 |
| Acartia hudsonica | 4.5 / 8 / 12 / 16 | 3.9-6.8 | 0.74-1.38 | 0.44-0.82 |
| Acartia tonsa (F) | 18 | 2.5-3.0 | 0.85-3.90 | 0.51-2.33 |
| Temora longicornis | 15 | 3.3-8.8 | 0.65-1.70 | 0.39-1.02 |
| Centropages hamatus | 17 | 9.9 | 0.86 | 0.52 |
| Oithona similis (F) | 8.5 | 0.36 | 0.70-0.83 | 0.42-0.50 |
| Oithona nana (F/M) | 10 | 0.2 | 0.20-0.77 | 0.12-0.46 |
| Calanus finmarchicus | 13 | 104 | 0.21-0.29 | 0.13-0.18 |
| Calanus helgolandicus (F) | 15 | 66-97 | 0.56-0.83 | 0.33-0.50 |
| Calanus helgolandicus (CV) | 15 | 65 | 0.41 | 0.24 |
| Calanus helgolandicus (NIII-CIV) | 15 | 0.35-29 | 1.0-3.3 | 0.6-2.0 |

- Meyer, Irigoien, Graeve, Head & Harris (2002), C. finmarchicus / C. helgolandicus: adult female
  C. finmarchicus daily ration ~19 % body C in lab experiments; average intake 35-40 % body C,
  maximum 50 % on diatoms at high food (these numbers are from a search summary that may mix
  this and another source) [S] - [Meyer et al. 2002, Helgol. Mar. Res. 56:169-176](https://doi.org/10.1007/s10152-002-0105-3)
- Mackinson & Daskalov (2007), North Sea Ecopath: Daro & Gijsegem (1984) consumption of
  copepodite stages II-IV ~4-5 ug C d-1 ind-1, with a mean copepod weight of 25.76 ug C, "gives an
  estimate of Q/B 0.19 d-1 or 30 y-1 (year May-Sept 153 days)"; Cushing & Vucetic (1963) suggested
  Calanus daily intake "may be as much as 390% body weight", considered too high; Paffenhofer:
  daily ingestion ~3-5 x daily production [V] -
  [Mackinson & Daskalov 2007, Cefas Sci. Ser. Tech. Rep. 142, section 10.1](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Brey (2001) conversions as tabulated by Mackinson & Daskalov (2007) Appendix 2: copepod WW->DW
  1:0.186, DW->C 1:0.451; phytoplankton C:WW 1:10 [V].

### Inferences (my computations)
- Database summary (my medians over the PANGAEA records): small neritic genera (Acartia, Temora,
  Centropages, Oithona, Paracalanus; n = 29; measured at 4.5-22 C): median 0.44 body C d-1 at 10 C
  (Q10 2.8), 0.75 at the measurement temperature. Calanus fin./helg. adults and CV (n = 6,
  13-15 C): 0.29 at 10 C, 0.48 uncorrected. Calanus nauplii-CIV (n = 10, 15 C): 0.90 at 10 C,
  1.5 uncorrected.
- A 2/3 : 1/3 assemblage weighting gives ~0.45-0.5 body C d-1 at 10 C with the Q10 correction and
  ~0.8-0.9 without it. Durbin et al.'s 0.74-0.85 at ~13 C on natural food (realised, not maximal)
  falls between.
- Kiorboe 2018's Fmax 0.65 ug C (ug DW)-1 d-1 equals ~1.4-1.6 body C d-1 only via C/DW 0.40-0.45;
  this is A. tonsa, a warm-water estuarine species, at high temperature.
- Model, energy units: 0.25 t phyto WW/t zoo WW/tick x 4 ticks/d x 2000/4500 MJ per t = 0.44
  body energy d-1. If phytoplankton and copepods both carry ~45-48 kJ/g C, this is also
  ~0.44 body C d-1, i.e. at the Q10-corrected literature median for the assemblage.
- Model, Ecopath WW units: 1.0 t/t/d x (0.1 C/WW phyto) / (0.084 C/WW copepod) = 1.2 body C d-1
  (1.0 if copepod C/WW is 0.1). That is the Acartia-at-18-20-C maximum, high for 8-12 C.
- The two readings differ only through the phytoplankton energy density: 2 kJ/g WW implies
  C/WW ~0.042 at 47.7 kJ/g C (Q7). For a diatom-dominated diet that is defensible, so the
  energy-unit reading (0.44 d-1, plausible to low) is the consistent one inside the model.
- Holling II: a*h = 0.25 x 4 = 1, so the half-saturation prey biomass is 1/(a*h) = 1 t WW/km2
  (column-integrated). Literature half-saturation is of order 50-150 ug C/L (Kiorboe et al.
  1985: clearance peaks at 150 ug C/L). Over an assumed 10-20 m feeding layer that is 0.5-3 g C
  m-2, i.e. ~5-60 g WW m-2 at C/WW 0.05-0.1. The model's half-saturation is therefore 1-2 orders
  of magnitude lower, so the saturation term should rarely limit model zooplankton intake at
  realistic phytoplankton stocks. (The layer depth is my assumption.)

### Gaps
- Saiz & Calbet (2007, 2011) and Hansen et al. (1997) regression coefficients were not retrieved
  (Wiley blocked), so the temperature question (Q10 2.8 vs temperature-independent maxima) cannot
  be settled numerically here.
- No functional-response maximum was retrieved for Pseudocalanus specifically.
- Meyer et al. (2002) full text not read; the Calanus ration numbers are snippet-only.

---

## Q2. Gross growth efficiency (GGE), net growth efficiency (NGE) and assimilation efficiency (AE)

### Takeaway
Copepod GGE (growth/ingestion) averages 26 % (median 22 %) in Straile's (1997) meta-analysis,
with interquartile ranges of 13-35 % across taxa; Acartia's egg-production efficiency at
saturation is ~36 % (64/180). Carbon AE is typically ~0.6-0.85 (63 % average for C; up to 85 % at
low food, 68 % at high food in Calanus pacificus) and falls with increasing food. The model's
AE 0.65 is fine. At the model's own maximum, however, implied GGE is ~0.55, about twice the
literature mean, because model metabolism is low (Q4).

### Cited Findings
- Straile (1997), compilation of protozoan and metazoan zooplankton: "Mean and median GGE of all
  taxa scattered around 20-30%"; copepods mean 26 %, median 22 % (n = 122 in the regression
  table); interquartile ranges 13-35 % across taxa; copepod GGE correlated positively with food
  concentration (r = 0.25) and negatively with temperature (-0.33) but these relationships were
  "sensitive to the exclusion or consideration of single sources"; maximum NGE of heterotherms
  70-80 % (Calow 1977) or 82-88 % (Schroeder 1981); measured AE can exceed 80 %; "a rather low GGE
  of ~10-20% may be more appropriate" under bloom conditions [V, full text] -
  [Straile 1997, L&O 42:1375-1385, KOPS full text](https://kops.uni-konstanz.de/server/api/core/bitstreams/70485015-5601-4554-9daa-1812adeeba6e/content)
- Same source, quoting Landry et al. (1984): Calanus pacificus "reached AE up to 85% compared to
  68% at higher food concentrations"; declining AE with increasing food found "in many studies of
  metazoan zooplankton"; low AE at high food is interpreted as an adaptation to maximise
  absorption rate (Jumars et al. 1989) [V, via Straile 1997].
- Kiorboe et al. (1985): plateaus of 180 % (ingestion) and 64 % (egg production) body C d-1 in
  A. tonsa; theoretical minimum biosynthesis costs accounted for 50-116 % of SDA; "the efficiency
  of egg production in this species is near its theoretical maximum" [V, abstract] -
  [MEPS 26:85](https://doi.org/10.3354/meps026085)
- Thor, Koski, Tang & Jonasdottir (2007): 14C/51Cr carbon AE of A. tonsa 44 % (Dunaliella),
  37 % (Amphidinium), 49 % (Phaeocystis), rising to 61 % on a Dunaliella + Amphidinium mix [V,
  abstract] - [Thor et al. 2007, MEPS 331:131-138](https://doi.org/10.3354/meps331131)
- Conover (1966): a 3-5 C increase within 2-11 C had no effect on AE of Calanus hyperboreus on
  diatoms; "Percentage of assimilation was not related to amount of food offered nor to the
  amount of food ingested"; "'superfluous feeding' does not normally occur in nature" [V,
  abstract] - [Conover 1966, L&O 11:346-354](https://doi.org/10.4319/lo.1966.11.3.0346)
- Mackinson & Daskalov (2007), North Sea Ecopath copepods: P/Q = 0.30 set from Daro & Gijsegem
  (1984) "daily net production efficiency versus ingestion was 20-30% for young copepod stages
  and perhaps even higher for adults"; unassimilated fraction 0.38, "close to an estimate of 33%
  for copepods given by Nielsen and Richardson (1989)" (balanced P/Q 0.3067) [V] -
  [Cefas Tech. Rep. 142, section 10.1 and Table 3.3](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Search summaries: copepods assimilate N (77 %) more efficiently than C (63 %) on average;
  Besiktepe & Dam (2002) found AE decreasing with food only on diatom and autotrophic
  dinoflagellate diets [S] - [Besiktepe & Dam 2002, MEPS 229:151](https://doi.org/10.3354/meps229151)

### Inferences (my computations)
- Model at maximum intake and h = 1: assimilated = 0.44 x 0.65 = 0.286 body energy d-1;
  metabolism while feeding = 2 x 22.5 x 4 / 4500 = 0.040 d-1; net = 0.246 d-1; GGE = 0.246/0.44
  = 0.55. NGE = 0.246/0.286 = 0.86, at the biochemical ceiling (82-88 %).
- Literature-consistent alternative at the same growth ceiling: ingestion ~0.6-1.0 body C d-1
  with GGE ~0.25-0.35 gives 0.15-0.35 d-1, matching the maximum growth rates in Q4. The model
  reaches a plausible growth ceiling by the wrong route (low intake x very high efficiency).
- Ecopath P/Q 0.30 with unassimilated 0.38 implies NGE = 0.30/0.62 = 0.48.

### Gaps
- No AE measurement specific to Pseudocalanus, Temora or Calanus finmarchicus on natural
  Skagerrak food was retrieved.

---

## Q3. Shape of the functional response; hunger and satiation; should appetite fall linearly with reserve fill?

### Takeaway
Copepod ingestion is set by food concentration through a type II (sometimes type III)
functional response that saturates because of gut processing capacity. Hunger effects are
real but short-lived and gut-scale: food-deprived small copepods raise clearance by 14-60 % after
6-14 h without food, and the effect fades within ~1-3 h of refeeding. No study found here shows
feeding declining gradually with energy-reserve fill. A linear appetite h = 1 - s that halves
intake at the maintenance level s = 0.5 has no support for income-breeding small copepods.
Appetite should stay near 1 until gut-limited, which the Holling term already represents.
Reserve state should change allocation (growth vs storage vs eggs), not intake. Calanus
lipid storage is a seasonal/diapause phenomenon. I found no direct evidence for or against
condition-dependent intake in active Calanus.

### Cited Findings
- Kiorboe et al. (2018): passive ambush feeders show invariant behaviour and a type II response;
  the switching species changes from type II to III when foraging actively; obligate active
  feeders "follow a type II response"; a literature survey "finds consistent type II response in
  ambush feeding copepods, but variable (II or III) responses in active feeders". "In suspension
  feeding zooplankton ... the handling of prey is rarely, if ever, the limiting factor ... Rather,
  ingestion is limited by the capacity of the gut to process food". On starvation: "starvation
  may result in significantly increased feeding in Acartia spp, but the effect is reduced or has
  disappeared within 100 min of feeding (Tiselius 1998), consistent with gut turnover times at the
  experimental temperature of just 20-25 min" [V, postprint] -
  [Kiorboe et al. 2018](https://backend.orbit.dtu.dk/ws/files/141852543/Postprint.pdf)
- Tiselius (1998), A. tonsa, A. clausi, Centropages hamatus deprived of food 1-14 h: A. tonsa
  clearance +?7 % (6 h; tens digit lost in the OCR'd abstract) and +44 % (14 h); A. clausi +60 % only after 14 h; C. hamatus +14 % after
  6 h; the stimulating effect "only lasted ~1 h in the case of 1 h of starvation but lasted more
  than 3 h after 14 h without food"; hunger responses let copepods "intermittently search for food
  or avoid predators and still maintain the same overall ingestion rate as constantly feeding
  animals" [V, abstract] -
  [Tiselius 1998, MEPS 168:119-126](https://doi.org/10.3354/meps168119)
- Mackas & Burns (1986), starved Calanus pacificus and Metridia pacifica re-exposed to
  phytoplankton: gut fullness shows "a strong initial peak 1-2 h after food is supplied and drops
  to about 20% of the peak level after 7-9 h of exposure to high food levels"; feeding involves
  "episodic on-off switching" and regulation of "satiation and hunger thresholds" [V, abstract] -
  [Mackas & Burns 1986, L&O 31:383-392](https://doi.org/10.4319/lo.1986.31.2.0383)
- Runge (1980): starved Calanus pacificus females "fed at higher rates than previously fed
  animals, even at low food concentrations"; strong seasonal changes in maximum clearance tied to
  the spring bloom [V, abstract] - [Runge 1980, L&O 25:134-145](https://doi.org/10.4319/lo.1980.25.1.0134)
- Durbin et al. (1990): A. tonsa kept a ~3-fold diel feeding rhythm "even when the copepods were
  food limited and lost about 20 % body carbon during the course of the 24 h experiment" [V,
  abstract] - [MEPS 68:23](https://doi.org/10.3354/meps068023)
- Mayzaud & Poulet (1978): over a year, feeding of five neritic copepod species was linear in
  natural particulate food, but saturating over 18-20 h; digestive enzymes acclimate over weeks,
  so the short-term response is curvilinear and the long-term (acclimated) response linear [V,
  abstract] - [Mayzaud & Poulet 1978, L&O 23:1144-1154](https://doi.org/10.4319/lo.1978.23.6.1144)
- Kiorboe et al. (1985): ingestion and egg production in food-acclimated A. tonsa both rise
  sigmoidally with food to plateaus (180 and 64 % body C d-1), i.e. intake and reproduction track
  food, not stored reserves [V, abstract].
- Conover (1966): no superfluous feeding; AE independent of ration [V, abstract].
- Pearre (2003) reviews gut-content evidence that individual vertical migrations are driven by
  hunger and satiation (gut-fullness timescale) [V, abstract] -
  [Pearre 2003, Biol. Rev. 78:1-79](https://doi.org/10.1017/S146479310200595X)
- Lee, Hagen & Kattner (2006): diapausing copepods store wax esters after feeding in
  spring/summer blooms; coastal zooplankton accumulate large lipid stores, tropical omnivores only
  small reserves [V, abstract] - [Lee et al. 2006, MEPS 307:273-306](https://doi.org/10.3354/meps307273)

### Inferences (my reasoning)
- The observed hunger/satiation signal operates on the gut-passage timescale (gut evacuation
  0.04-0.09 min-1, i.e. turnover 10-25 min at 13-20 C; effects gone within 1-3 h). It is shorter
  than one 6 h tick. Over a tick the relevant regulator is food concentration (functional
  response), which the Holling term already supplies.
- With h = 1 - s, intake at the maintenance level (s = 0.5) is 50 % of the food-determined rate:
  0.22 instead of 0.44 body energy d-1 at saturating food. Combined with the Q2 numbers,
  net = 0.22 x 0.65 - 0.04 = 0.10 d-1 at saturation, and 0.144 f - 0.04 at Holling fraction f.
  Balancing a total mortality of 25 /yr (0.068 d-1) then needs f >= 0.75. This is my reading of
  why the model zooplankton cannot cover its mortality.
- A form more consistent with the evidence keeps h ~ 1 for s below ~0.8-0.9 and drops it only
  as the reserve approaches full (a "full-gut / full-store" cut-off). An alternative leaves intake
  alone and lets the reserve level steer allocation. Acartia-type income breeders turn
  assimilated food into eggs within ~a day (Kiorboe et al. 1985), so they do not "fill and stop".

### Gaps
- No study was found that measured ingestion as a function of lipid/energy reserve in actively
  feeding Calanus (outside diapause entry). Whether lipid-full CV reduce intake gradually or
  abruptly is unresolved here.
- Thor (2002), elevated respiration during recovery from starvation (JEMBE 283:133), was found
  but not read.

---

## Q4. Respiration / metabolic rates at 8-12 C and starvation tolerance

### Takeaway
Routine respiration of field-collected small copepods converts to ~0.03-0.07 body C d-1 at 10 C
(Ikeda et al. 2007 data via the trait database, RQ 0.8 and Q10 2.8 assumed). Starvation
metabolism of A. tonsa is ~0.035 body C d-1, and feeding raises respiration more than 4-fold
(SDA tracks biosynthesis). The model's resting rate of 2 % body energy d-1 is at or below the
low end. Its x2 feeding multiplier (4 % d-1) is well below the >4x observed. Maximum growth rates
at 8-12.5 C are 0.13-0.34 d-1. Starvation tolerance: Acartia dies after ~6-10 d without food,
other species survive up to ~20 d [S]; the model's full reserve lasts 7.5 d at resting rate.

### Cited Findings
- Brun et al. (2017) / PANGAEA sheet "Respiration rates" (records from Ikeda et al. 2007,
  "Specific respiration at 15 C" in ul O2 h-1 mg C-1), medians by genus [V, data file]: Temora
  5.4 (n = 7), Acartia 9.4 (n = 24), Centropages 10.3 (n = 55), Oithona 10.4 (n = 28),
  Pseudocalanus 12.1 (n = 20), C. finmarchicus 7.1 (n = 1) - [PANGAEA.862968](https://doi.org/10.1594/PANGAEA.862968).
  Database-wide range at reference temperature 0.3-53.8 ul O2 h-1 mg C-1 [V] -
  [Brun et al. 2017](https://doi.org/10.5194/essd-9-99-2017)
- PANGAEA sheet "Growth rates" (Kiorboe & Hirst 2014 records; "Specific growth" in mg C h-1
  mg C-1, i.e. per HOUR), at measured temperature [V, data file]: Pseudocalanus 8 C 0.0055-0.0065
  h-1 (0.13-0.16 d-1), 12 C 0.0083-0.0094 h-1 (0.20-0.23 d-1); Temora longicornis 12.5 C
  0.0088-0.0142 h-1 (0.21-0.34 d-1); Calanus finmarchicus 8 C 0.0083-0.0105 h-1 (0.20-0.25 d-1);
  Paracalanus 10.3 C 0.0048-0.0100 h-1 (0.12-0.24 d-1); Oithona davisae 12 C 0.0027 h-1
  (0.065 d-1).
- Kiorboe et al. (2018): A. tonsa m0 (starvation) 0.015 and mf (feeding, incl. defecation)
  0.1 ug C (ug DW)-1 d-1 [V, postprint].
- Kiorboe et al. (1985): respiration of saturation-fed A. tonsa >4x that of starved; SDA mostly
  biosynthesis and transport [V, abstract].
- Durbin et al. (1990): food-limited A. tonsa lost ~20 % body C in 24 h at 20 C [V, abstract].
- Dagg (1977): A. tonsa needed fairly continuous food and starved within 6-10 d with no or only
  intermittent food; co-occurring species survived up to 20 d of constant starvation;
  Centropages typicus intolerant of starvation [S] -
  [Dagg 1977, L&O 22:99-107](https://doi.org/10.4319/lo.1977.22.1.0099)
- Ikeda, Kanno, Ozaki & Shinada (2001) give multiple regressions of copepod O2 consumption on
  body mass and temperature [V, bibliographic only; coefficients not retrieved] -
  [Ikeda et al. 2001, Mar. Biol. 139:587-596](https://doi.org/10.1007/s002270100608)
- Moreno & Sasaki (2023): starvation up to 2 d did not change A. tonsa CTmax; after 3 d CTmax
  fell [V, abstract] - [Ecol. Evol. 13:e10586](https://doi.org/10.1002/ece3.10586)

### Inferences (my computations; RQ = 0.8 and Q10 = 2.8 are my assumptions)
- 1 ul O2 at RQ 0.8 = 0.8 x 12/22.4 = 0.43 ug C respired. Median routine respiration at 10 C:
  Temora 0.033, C. finmarchicus 0.044, Acartia 0.058, Centropages 0.063, Oithona 0.064,
  Pseudocalanus 0.074 body C d-1.
- A. tonsa m0 0.015 per ug DW = ~0.033-0.038 body C d-1 at C/DW 0.40-0.45, measured warm (lab
  culture, ~18 C); at 10 C with Q10 2.8 ~0.015-0.02. mf 0.1 per ug DW = 0.22-0.25 body C d-1
  (incl. defecation).
- Model: 22.5 MJ/t/tick x 4 / 4500 MJ/t = 0.020 body energy d-1 resting; 0.040 when feeding.
  Resting is plausible at 10 C for a starving copepod but at or below field routine rates; the
  feeding increment (x2) is far below the >4x of Kiorboe et al. (1985). A literature-shaped term
  would be ~0.02-0.03 d-1 basal plus an SDA proportional to assimilation (on the order of
  20-30 % of assimilated energy).
- Starvation: full reserve 675 MJ/t / 90 MJ/t/d = 7.5 d to empty at resting rate; from the
  maintenance level (s = 0.5) 3.75 d. Acartia's 6-10 d is comparable; Calanus/Pseudocalanus
  (up to ~20 d [S]) outlast it. The reserve size (15 % of body energy) is reasonable for the small
  copepod fraction and low for lipid-storing Calanus.

### Gaps
- Ikeda et al. (2001) regression coefficients not retrieved; values above rely on the trait
  database and my RQ/Q10 assumptions.
- Thor (2002) and starvation-respiration time courses for Calanus not read.

---

## Q5. Standing stocks in Kattegat/Skagerrak (and North Sea / western Baltic); zoo:phyto ratio; P/B; share of PP grazed

### Takeaway
Summer (June-August) total mesozooplankton biomass at the Swedish monitoring stations Anholt E
(Kattegat) and Slaggo (Gullmar Fjord, Skagerrak) was ~10-55 g WW m-2 in 1998-2017, declining to
~5-15 g WW m-2 in 2018-2023. Copepods make up ~20-30 % of it at Anholt E. These are approximate
figure readings from a non-peer-reviewed presentation. That gives summer copepod stocks of
~3-10 g WW m-2 (~0.2-0.8 g C m-2). The North Sea Ecopath uses 16 g WW m-2 for herbivorous
copepods. Zoo:phyto biomass ratios in carbon are ~0.1-0.5 in productive coastal systems (western
Baltic Ecopath 0.32). Mesozooplankton graze a small share of primary production: Calbet's global
mode is 6 % and mean 22.6 %; spring Skagerrak <3 %; Skagerrak/Kattegat summer up to 48 % of
daily PP but <4 % of standing stock.

### Cited Findings
- Casties (University of Gothenburg), "Djurplankton", Swedish water-environment seminar, January
  2025 (presentation slides, not peer-reviewed): "Totalbiomassa djurplankton (medelvarde for
  juni-augusti)", g WW m-2, stations Anholt E (1998-2017), N14, Slaggo, A17 (2007-2017), Alsback,
  BroA E (2018-2023); a second panel shows Anholt E and Slaggo with a declining trend; a third shows
  "% hoppkraftor av totalbiomassan" at Anholt E [V, read from rendered figures - all numbers below
  are approximate readings]. Anholt E summer totals ~10-57 g WW m-2 (peaks 2004 ~57, 2009 ~50);
  Slaggo ~14-42 g WW m-2 (2008-2017); all stations ~3-17 g WW m-2 in 2018-2023; trend line
  ~33 -> ~13 g WW m-2 over 1998-2023; copepod share at Anholt E 10-57 %, mostly 20-30 %, trend
  ~29 -> ~19 %. A Slaggo time series (2018-2023) plots "Calanoid copepods mg C m-3" with peaks of
  ~25-75 mg C m-3 and a baseline of a few mg C m-3 (my axis assignment from legend colour). The
  slides do not say what "total biomass" includes (e.g. gelatinous taxa, meroplankton) -
  [Casties 2025, havsmiljo.se](https://havsmiljo.se/pdf/presentationer/vms25/zooplankton-casties-vattenmiljoseminariet-jan-2025-final.pdf)
- Kiorboe & Nielsen (1994), southern Kattegat: copepod production "episodic", in bursts tied to
  three net-phytoplankton blooms; copepod biomass unimodal with peak in June-July; biomass declined
  Aug-Oct during the largest production event, implying elevated mortality [V, abstract]. Annual
  production ~12 g C m-2 with <10 % net seasonal biomass increase [S, carried over as
  unverified from the sibling notes] -
  [Kiorboe & Nielsen 1994, L&O 39:493-507](https://doi.org/10.4319/lo.1994.39.3.0493)
- Mackinson & Daskalov (2007), North Sea Ecopath 1991, herbivorous + omnivorous zooplankton
  (copepods) [V, Tables 3.3, 3.6, 10.1-10.3, section 10.1] -
  [Cefas Tech. Rep. 142](https://www.cefas.co.uk/publications/techrep/tech142.pdf):
  - B = 16 t WW km-2, P/B = 9.2, Q/B = 30, EE = 0.4852, P/Q = 0.3067, unassimilated 0.38;
    flows: production 147.2, consumption 480.0, respiration 150.4, consumed as food 71.4
    t km-2 y-1.
  - Biomass back-calculated: annual production 12.35 g C m-2 yr-1 (147 g WW; Fransz & Gieskes
    1984) / P/B 9.2. The P/B is the May-September mean of Temora 8.667, Acartia 7.667,
    Pseudocalanus 11.167 on a "year = 153 days".
  - FLEX 1976 (northern North Sea) summer peak standing stock ~12.5 g C m-2 (~125 g WW m-2),
    dominated by C. finmarchicus (11.2 g C m-2); Calanus biomass rose to 4 g C m-2 by end of May;
    population ~0.4 g C m-2 at the end-April phytoplankton peak; coastal mixed-area production
    5-20 g C m-2 yr-1.
  - Nielsen & Richardson (1989): "Total copepod biomass ranged from 2.5-8.8 mg C m-3 (approx.
    2.7-9.5 g ww m-2 ...)".
  - Phytoplankton in the same model: B = 7.5 g WW m-2 from an average March-June standing stock
    of ~750 mg C m-2 (FLEX); Hannon & Joiris (1989) southern North Sea microplankton 3.7 g C m-2
    (43 g WW m-2); P/B 286 y-1.
- Western Baltic Ecopath (SD 22+24): phytoplankton 2.161 g C m-2; pooled zooplankton (macro +
  meso + micro) 0.697 g C m-2 [V via the sibling notes, Scotti et al. 2022 supplement] -
  [Scotti et al. 2022 supplement](https://oceanrep.geomar.de/id/eprint/57232/2/DataSheet_1_Ecosystem-based%20fisheries%20management%20increases%20catch%20and%20carbon%20sequestration%20through%20recovery%20of%20exploited%20stocks%20The%20western%20Baltic%20Sea.pdf)
- Gasol, del Giorgio & Duarte (1997): the heterotroph:autotroph biomass ratio (bacteria +
  protozoa + mesozooplankton vs phytoplankton) declines with phytoplankton biomass and production;
  productive areas have "a normal biomass pyramid with a broad autotrophic base"; coastal
  communities support less heterotrophic biomass per unit autotroph than open-ocean ones [V,
  abstract] - [Gasol et al. 1997, L&O 42:1353-1363](https://doi.org/10.4319/lo.1997.42.6.1353)
- Calbet (2001): mesozooplankton grazing impact mode 6 %, mean 22.6 % of PP per day, decreasing
  with productivity [V via sibling notes, abstract] -
  [Calbet 2001, L&O 46:1824-1830](https://doi.org/10.4319/lo.2001.46.7.1824)
- Tiselius (1988), Skagerrak and Kattegat May-October: copepod grazing <4 % of phytoplankton
  standing stock d-1 but up to 48 % of daily PP [V via sibling notes] -
  [Ophelia 28:215](https://doi.org/10.1080/00785326.1988.10430814)
- Maar et al. (2002), Skagerrak spring diatom bloom: "Despite the Calanus population, the copepod
  community only grazed < 3% of daily primary production"; protozooplankton ingested 2-4x more;
  total zooplankton grazing 17 % of daily PP [V, abstract] -
  [Maar et al. 2002, MEPS 239:11-29](https://doi.org/10.3354/meps239011)
- Nielsen et al. (1993), Dogger Bank, May: ~15 % (on/south of the bank) and ~30 % (north) of
  phytoplankton production went directly to copepods; grazable (>11 um) production could not alone
  meet copepod carbon demand, so ciliates were likely important food [V, abstract] -
  [Nielsen et al. 1993, MEPS 95:115-131](https://doi.org/10.3354/meps095115)
- Peterson, Tiselius & Kiorboe (1991), Skagerrak August: community copepod production 3-8 mg C
  m-3 d-1; specific growth 0.10 d-1 (females), 0.27 d-1 (juveniles) [V via sibling notes] -
  [J. Plankton Res. 13:131](https://doi.org/10.1093/plankt/13.1.131)
- Tonnesson, Nielsen & Tiselius (2006), Skagerrak: the carnivorous copepod Pareuchaeta norvegica
  ate 2.0-6.5 % of the copepod population daily, equal to 6-16 % of copepod production; its
  impact was 10-100x that of chaetognaths [V, abstract] -
  [MEPS 314:213-225](https://doi.org/10.3354/meps314213)
- Lindahl & Hernroth (1988), Gullmar Fjord: autumn inflows of Skagerrak water bring highly
  variable numbers of Calanus CIV-V; advection was "the major factor regulating zooplankton
  biomass in the fjord" [V, abstract] - [MEPS 43:161-171](https://doi.org/10.3354/meps043161)
- Falkenhaug et al. (2022), Skagerrak coast (Arendal station, 1994-2019): C. finmarchicus peaks in
  spring at 6-8 C, C. helgolandicus in autumn at 11-16 C [V, abstract] -
  [Front. Mar. Sci. 9:779335](https://doi.org/10.3389/fmars.2022.779335)

### Inferences (my computations)
- Kattegat/Skagerrak summer copepods: total 13-33 g WW m-2 (trend range) x copepod share
  0.2-0.3 = 2.6-10 g WW m-2 = 0.22-0.84 g C m-2 (C/WW 0.084). Summer is near the seasonal peak
  (Kiorboe & Nielsen 1994), so the annual mean is lower. Recent years (2018-2023, totals
  ~5-15 g WW m-2) give ~1-4.5 g WW m-2 of copepods.
- Production check: ~12 g C m-2 yr-1 (Kattegat [S]; North Sea 12.35) / P/B ~25 yr-1 (HK2002
  total mortality at 10 C, Q6; sibling notes plankton.md Q4 range 20-30 yr-1) = ~0.5 g C
  m-2 = ~6 g WW m-2. With Ecopath's P/B 9.2 the same production would need 16 g WW m-2.
- Zoo:phyto (carbon): Kattegat copepods 0.2-0.8 g C m-2 against a phytoplankton proxy of
  ~2.2 g C m-2 (western Baltic Ecopath; no Kattegat primary value found) gives 0.1-0.4. Western
  Baltic pooled zooplankton/phyto is 0.32. North Sea Ecopath: 16 g WW x 0.084 = 1.3 g C copepods
  vs 0.75 g C phyto gives 1.8. That high ratio arises only because the phyto biomass is a
  March-June mean through a conversion the report itself flags; with the 3.7 g C m-2 southern
  North Sea value it is 0.36.
- In model WW units (phyto C/WW ~0.042 implied by 2 kJ/g WW, copepod 0.084): 2.2 g C m-2 phyto =
  ~50 t WW km-2; 0.5 g C m-2 copepods = ~6 t WW km-2; WW ratio ~0.1-0.2.
- Diet check: B 0.5 g C m-2 x P/B 25 / GGE 0.25 = Q ~50 g C m-2 yr-1. With PP 190-290 g C m-2
  yr-1 (sibling notes), copepods eating 10-25 % of PP gives 20-70 g C m-2 yr-1 of phyto. That is
  enough only if a substantial part of the diet is microzooplankton, as Nielsen et al. (1993)
  and Maar et al. (2002) indicate.

### Gaps
- Kiorboe & Nielsen (1994) annual mean copepod biomass (mg C m-3, g C m-2) not obtained; their
  annual production is snippet-only.
- No primary depth-integrated Kattegat/Skagerrak phytoplankton carbon stock retrieved.
- The SMHI/SHARK monitoring data (Anholt E, Slaggo; mg WW m-3, copepods by species) were not
  downloaded; the presentation readings are approximate.
- Zervoudaki, Nielsen & Carstensen (2009, J. Plankton Res. 31:1475) has Danish-waters biomass and
  production along a eutrophication gradient but was not read.

---

## Q6. Copepod mortality: total vs non-predation, at 8-12 C

### Takeaway
Hirst & Kiorboe (2002) give post-hatch mortality of broadcast spawners as
ln(beta) = 0.0725 T - 3.415 (d-1). Evaluated at 8/10/12 C this is 0.059/0.068/0.079 d-1
= 21/25/29 yr-1. For sac spawners (ln beta = 0.0707 T - 3.157) it is 27/32/36 yr-1. Predation is
~2/3-3/4 of the total, so non-predation (lab longevity) is only ~0.023-0.029 d-1 (~8-11 yr-1) at
8-12 C. The model's 19.7 yr-1 "non-fish" mortality is consistent if total mortality is ~25-30 yr-1
and invertebrate predators (carnivorous zooplankton, fish larvae, jellyfish) take roughly half of
predation, as in the North Sea Ecopath diet matrix. It is not a non-predation rate. Because these
mortality rates are derived from in situ fecundity at steady state, they equal realised field
P/B: the zooplankton must produce ~25 yr-1 to persist.

### Cited Findings
- Hirst & Kiorboe (2002), Fig. 4 caption (minus signs restored; pdftotext dropped them):
  broadcasters "loge beta = 0.0725T - 1.112 (r2 = 0.723, n = 885, p < 0.001) for eggs, and
  loge beta = 0.0725T - 3.415 (r2 = 0.723, n = 885) for post-hatch individuals"; sac spawners
  "loge beta = 0.0707T - 3.157 (r2 = 0.873, n = 166)"; species means for broadcasters
  0.0730T - 3.453 (post-hatch). beta in d-1, T in C. Mortality "declines with body weight in
  broadcast spawners, while mortality in sac spawners is invariant with body size"; body-weight
  analyses were standardised to 15 C with Q10 = 2.0 [V, full text] -
  [Hirst & Kiorboe 2002, MEPS 230:195-209, DTU Orbit copy](https://backend.orbit.dtu.dk/ws/files/3696723/Kiorboe5.pdf)
- Same source, predation: "At 5 C, the predicted field longevity is 16.2 d, which translates to a
  mortality of 0.062 d-1; laboratory longevity is 52.2 d, equating to a mortality of 0.019 d-1.
  At 25 C, predicted field mortality is 0.190 d-1 and the laboratory mortality 0.065 d-1 ...
  this would account for about 2/3 of the total adult mortality regardless of temperature";
  field-longevity slopes 0.056 (broadcast) and 0.071 (sac), laboratory 0.061. At ~10 ug DW
  (data corrected to 15 C), predicted field longevity 9.2 d (broadcasters) and 8.1 d (sac),
  mortality 0.109 and 0.123 d-1; average lab longevity 38.8 d (0.026 d-1); "predation mortality
  accounts for 3/4 of the total". The data set is "dominated by estuarine and coastal studies"
  [V, full text].
- Kiorboe et al. (2018) assume ~0.1 d-1 total mortality for a small feeding copepod, "a magnitude
  typical for mm-sized feeding-current feeding copepods (Hirst and Kiorboe 2002)" [V, postprint].
- Tang et al. (2014): dead zooplankton average 11.6 (minimum) to 59.8 (maximum) % of individuals in
  marine field samples; causes include senescence, temperature, physical/chemical stress,
  parasitism and food [V, abstract] - [Tang et al. 2014, J. Plankton Res. 36:597-612](https://doi.org/10.1093/plankt/fbu014)
- da Cruz et al. (2023), tropical estuaries: mean adult copepod non-predatory mortality
  0.15 d-1 (0.01-2.80) [V, abstract; tropical, for context only] -
  [Mar. Ecol. 44:e12775](https://doi.org/10.1111/maec.12775)
- Eiane & Ohman (2004), FLEX 1976 Fladen Ground: stage-specific mortality of C. finmarchicus,
  Pseudocalanus elongatus and Oithona similis "changes substantially over the life span"; O.
  similis shows negligible losses after NI-NII [V, abstract; numbers not retrieved] -
  [MEPS 268:183-193](https://doi.org/10.3354/meps268183)
- Ohman & Hirche (2001): egg mortality of C. finmarchicus at Ocean Station M (Norwegian Sea,
  March-June 1997) was density-dependent on adult female and juvenile abundance [S, abstract via
  search; numbers not retrieved] - [Nature 412:638-641](https://doi.org/10.1038/35088068)
- Mollmann et al. (2004), Bornholm Basin (Baltic): high C4-C6 mortality of Pseudocalanus
  vs low in Acartia longiremis, in line with planktivorous-fish predation; rates "in the range
  observed in other areas"; the figure axis spans 0-0.5 d-1 [V, poster; values not readable] -
  [ICES CM 2004/L:33](https://www.ices.dk/sites/pub/CM%20Doccuments/2004/L/L3304.pdf)
- North Sea Ecopath copepods: EE 0.4852 (so 51.5 % of production is unexplained "other
  mortality"), consumed as food 71.4 of 147.2 t km-2 y-1 production [V] -
  [Cefas Tech. Rep. 142, Tables 3.3 and 3.6](https://www.cefas.co.uk/publications/techrep/tech142.pdf)

### Inferences (my computations)
- Q10 implied by the slope: e^(0.0725 x 10) = 2.06, matching the paper's Q10 = 2.0.
- Regressions evaluated (d-1 -> yr-1 x 365):

| T (C) | Broadcast post-hatch | Sac spawners | Lab (non-predation), 0.019 e^(0.061 (T-5)) |
|---|---|---|---|
| 8 | 0.059 (21.4 /yr) | 0.075 (27.4 /yr) | 0.023 (8.3 /yr) |
| 10 | 0.068 (24.8 /yr) | 0.086 (31.5 /yr) | 0.026 (9.4 /yr) |
| 12 | 0.079 (28.7 /yr) | 0.099 (36.3 /yr) | 0.029 (10.6 /yr) |

  The adult-longevity route (0.062 d-1 at 5 C, slope 0.056) gives 0.082 d-1 (30 /yr) at 10 C.
  The lab regression 0.019 e^(0.061 (T-5)) reproduces the paper's 0.065 d-1 at 25 C.
- Consumers of copepods in the North Sea Ecopath diet matrix (computed from the sibling notes'
  `ecopath_ns1991/diet.json` x B x Q/B; total 67.8 vs 71.4 t km-2 y-1 in Table 3.6): carnivorous
  zooplankton 30.8 (45 %), sandeel 19.1, adult herring 5.4, fish larvae 5.1 (7.5 %), sprat 2.8,
  juvenile herring 2.2, others <0.5 each. Fish take ~46 % and invertebrates + larvae ~53 % of
  predation. Of production 147.2: predation by invertebrates/larvae ~36, by fish ~31, unexplained
  ~76 t km-2 y-1. Non-fish losses are thus ~76 % of production, close to the library's 79 %.
- Applying HK2002 at 10 C (total ~25 /yr, non-predation ~9 /yr, predation ~16 /yr) with the
  Ecopath split (fish ~46 % of predation) gives fish ~7 /yr and non-fish ~18 /yr. The model's
  19.7 /yr is in that range, but most of it must be read as invertebrate predation, not
  senescence/disease.
- Per tick: 19.7 /yr = 0.0135 /tick (library 0.0134); 25 /yr = 0.0171 /tick.

### Gaps
- Eiane & Ohman (2004) and Ohman & Hirche (2001) numeric rates not retrieved (int-res and
  Nature full text not accessible).
- No Kattegat/Skagerrak-specific in situ copepod mortality estimate found.

---

## Q7. Energy density of copepods and phytoplankton

### Takeaway
Small neritic copepods carry 18.7-21.9 kJ/g DW and C. finmarchicus 26.9 kJ/g DW (Laurence 1976,
Formalin-preserved). That is ~3.5-4.1 kJ/g WW for small copepods and ~5 kJ/g WW for Calanus at
DW/WW 0.186. Lipid-rich C. finmarchicus CV reach 5.8-6.8 kJ/g WW; direct WW measurements in
the Bay of Biscay span 0.5-6.7 kJ/g WW (size classes 0.74-1.26). The model's 4.5 kJ/g WW is
plausible for a 2/3 : 1/3 mix, slightly above the small-copepod value. Phytoplankton carries
~47.7 kJ/g C [S]; per WW that is ~1.6-3.8 kJ/g WW for diatoms (C/WW 0.03-0.08) and ~6-8 for
flagellates (C/WW 0.12-0.16). The model's 2 kJ/g WW implies C/WW ~0.042, a large-diatom value,
well below the Ecopath 0.1 convention (4.8 kJ/g WW).

### Cited Findings
- Laurence (1976), bomb calorimetry, copepods off Narragansett Bay (Formalin-preserved), Table 1
  cal/g DW: Calanus finmarchicus 6425.1, Tortanus discaudatus 5398.3, Centropages typicus 5244.7,
  Acartia tonsa 5160.0, Pseudocalanus minutus 5070.9, Centropages hamatus 4998.6, Temora
  longicornis 4466.3; mean of the seven species 5251.9 cal/g DW, 5626.3 cal/g AFDW, 6.70 % ash;
  ash 4.1-10.4 % (Temora highest); higher Calanus value attributed to lipid [V, full text] -
  [Laurence 1976, Fish. Bull. 74:218-220](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/fish-bull/laurence.pdf)
- McKinstry, Westgate & Koopman (2013), Bay of Fundy C. finmarchicus CV, July-September
  2006-2010: mean 6.77 +- 0.65 kJ/g WW (2007, highest) to 5.82 +- 0.90 kJ/g WW (2009, lowest);
  energy correlated with lipid [V, abstract] - [Endang. Species Res. 20:195-204](https://doi.org/10.3354/esr00497)
- Dessier et al. (2017), Bay of Biscay spring: copepod species (Centropages typicus, Anomalocera
  patersoni, Calanus helgolandicus, Labidocera wollastoni) and anchovy eggs 0.5-6.7 kJ/g WW;
  mesozooplankton size classes 0.74-1.26 kJ/g WW [V, abstract] -
  [Prog. Oceanogr. 166:121-128](https://doi.org/10.1016/j.pocean.2017.10.009)
- Michaud & Taggart (2007): 84 % of zooplankton energy density in Grand Manan Basin came from wax
  esters of C. finmarchicus CV [V, abstract] - [Endang. Species Res. 3:77-94](https://doi.org/10.3354/esr003077)
- Platt & Irwin (1973): phytoplankton calorific value predictable from carbon content; conversion
  11.4 kcal (g C)-1 [S for the number; abstract V] -
  [L&O 18:306-310](https://doi.org/10.4319/lo.1973.18.2.0306)
- Finlay & Uhlig (1981): protozoa ~46 J (mg C)-1; ~45 J (mg C)-1 across biological materials [S] -
  [Helgol. Meeresunters. 34:401-412](https://doi.org/10.1007/BF01995913)
- Mackinson & Daskalov (2007): phytoplankton production conversions "based on the conversion
  factors used by Christensen (1995) (1 g C = 15 kcal; 1 g wet wt = 1.3 kcal; Jones 1984 ...)";
  alternative factors gave 400-8000 g WW m-2 y-1 from 170 g C m-2 y-1 [V] -
  [Cefas Tech. Rep. 142, section 7](https://www.cefas.co.uk/publications/techrep/tech142.pdf)
- Menden-Deuer & Lessard (2000): diatoms pg C cell-1 = 0.288 x volume^0.811; other protists
  (excluding diatoms) 0.216 x volume^0.939; dinoflagellates 0.760 x volume^0.819; carbon density
  declines with cell volume; diatoms are less C-dense than dinoflagellates [V, abstract] -
  [L&O 45:569-579](https://doi.org/10.4319/lo.2000.45.3.0569)
- Kiorboe (2013): non-gelatinous zooplankton 5-10 % C of WW, gelatinous ~0.5 % [S] -
  [L&O 58:1843](https://doi.org/10.4319/lo.2013.58.5.1843)

### Inferences (my computations)
- Laurence values x DW/WW 0.186 (Brey): Temora 3.5, Centropages hamatus 3.9, Pseudocalanus 3.9,
  Acartia 4.0, C. typicus 4.1, C. finmarchicus 5.0 kJ/g WW. Formalin preservation may lower these.
  Lipid-rich Calanus probably also has a higher DW/WW than 0.186, so 5.0 is a floor for it.
- Per carbon: 21-22 kJ/g DW / 0.45 C/DW = ~47-49 kJ/g C for small copepods, the same as Platt &
  Irwin's ~47.7 kJ/g C for phytoplankton [S]. Energy per carbon is therefore ~equal for prey and
  predator, and the model's WW energy densities are effectively C/WW statements.
- Assemblage: 2/3 x ~3.9 + 1/3 x ~5-6.8 = 4.3-4.9 kJ/g WW. The model's 4.5 is in range.
- Phytoplankton C density from Menden-Deuer & Lessard (pg C per um3 ~ g C per g WW at density
  ~1; density assumption mine): diatoms 0.078 (10^3 um3), 0.050 (10^4), 0.033 (10^5); other
  protists 0.163 (10^2), 0.142 (10^3), 0.124 (10^4). Times 47.7 kJ/g C: diatoms 1.6-3.7 kJ/g WW,
  flagellates 5.9-7.8 kJ/g WW. The model's 2.0 kJ/g WW implies C/WW = 0.042 (a diatom of ~3 x 10^4
  um3). The Ecopath 0.1 gives 4.8 kJ/g WW; Christensen's generic 1.3 kcal/g WW = 5.4 kJ/g WW.
- Consequence: if model phytoplankton represents a mixed spring/summer community (flagellates and
  small diatoms, plus the microzooplankton copepods also eat), its energy per WW is probably
  2-3x higher than 2 kJ/g WW. That would double or triple the energy delivered by the same
  0.25 t/t/tick intake. The WW intake cap and the phyto energy density must be set together.

### Gaps
- Platt & Irwin (1973) and Finlay & Uhlig (1981) values are snippet-only.
- No energy-density measurement of Skagerrak/Kattegat copepods (fresh, not Formalin) was found.
- Kerambrun (1987, Mar. Biol. 95:115), the energy equivalent of Acartia clausi, was not read.

---

## Summary table: model parameters vs literature (inference)

| Parameter (model) | Model value | Literature value (8-12 C) | Verdict |
|---|---|---|---|
| Max ingestion | 0.25 t phyto WW/t WW/tick = 100 % WW d-1 = 0.44 body energy d-1 | Small neritic median 0.44 body C d-1 at 10 C (Q10 2.8) or 0.75 uncorrected; Calanus CV/F 0.13-0.5; Calanus juveniles 0.6-2.0; Acartia 1.5-1.8 at 18-20 C | Plausible to low in energy terms; high if read with Ecopath C/WW (1.0-1.2 body C d-1) |
| Holling half-saturation | 1/(a h) = 1 t WW km-2 | ~50-150 ug C/L, i.e. ~5-60 g WW m-2 over 10-20 m (layer depth assumed) | 1-2 orders of magnitude too low; saturation is not the limiting factor |
| Assimilation | 0.65 | C AE ~0.6-0.85 (63 % average; 68 % high food, 85 % low food); Ecopath 0.62 | OK |
| Implied max GGE | ~0.55 (NGE 0.86) | Mean 26 %, median 22 % (Straile); Acartia egg efficiency 0.36; Ecopath P/Q 0.30 | ~2x too high, because metabolism is low |
| Resting metabolism | 22.5 MJ/t/tick = 2.0 % body energy d-1 | Starved A. tonsa ~1.5-3.5 % body C d-1; field routine 3-7 % d-1 at 10 C | Low end |
| Feeding metabolism | x2 (4 % d-1) | Fed >4x starved (Kiorboe 1985); mf/m0 = 6.7 (Kiorboe 2018) | Too low; SDA should scale with assimilation |
| Max growth (energy ceiling) | ~0.25 d-1 at h = 1; ~0.10 d-1 at s = 0.5 | 0.13-0.34 d-1 (Pseudocalanus, Temora, Calanus, Paracalanus at 8-12.5 C) | OK only with h = 1 |
| Appetite h = 1 - s, s = 0.5 | Intake halved at maintenance level | Hunger effects last 1-3 h (gut scale); intake follows food (functional response); no superfluous feeding | Not supported; keep h ~1 until reserves nearly full |
| Reserve max | 675 MJ/t (15 % of body energy), 7.5 d at rest | Acartia starves in 6-10 d, others up to ~20 d [S]; Calanus stores much more lipid | OK for small copepods, low for Calanus |
| Energy density zoo | 4.5 kJ/g WW | Small 3.5-4.1, Calanus ~5-6.8 kJ/g WW | OK |
| Energy density phyto | 2 kJ/g WW (C/WW ~0.042) | Diatoms 1.6-3.7, flagellates 5.9-7.8, Ecopath convention 4.8 kJ/g WW | Low end (large diatoms only) |
| Non-fish mortality | 19.7 /yr | Total 21-36 /yr at 8-12 C (HK2002); non-predation only 8-11 /yr; invertebrate predation ~half of predation (NS Ecopath) | Plausible as "invertebrate predation + non-predation", not as non-predation alone |
| Copepod stock (for checks) | - | Kattegat/Skagerrak summer ~3-10 g WW m-2 (approx.); NS Ecopath 16 g WW m-2; zoo:phyto (C) ~0.1-0.5 | - |
