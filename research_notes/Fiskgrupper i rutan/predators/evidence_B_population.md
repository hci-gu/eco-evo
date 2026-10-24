# Evidence B: population dynamics of the `predatory_fish` members (r, F, M, residence)

Scope: the ten members of `predatory_fish` (long-run 2001-25 shares, `../split_brief.md`): saithe 41 %, whiting 37 %,
mackerel 8 %, spurdog 5 %, horse mackerel 3.4 %, hake 2.5 %, anglerfish 1.4 %, ling 0.7 %, velvet belly 0.5 %,
pollack 0.4 % (sum 99.9 %; normalised before weighting). Target 1.20 t/km2.

Tags: [F] full text read / data downloaded, [A] abstract only, [S] secondary, [D] computed here.
Downloads: `data/population/`, `data/lit_life_history/`, `data/lit_residence/` (README in `data/README.md`).
Scripts: `scripts/sms_m2_split.py`, `scripts/sag_productivity.py`, `scripts/bottom_line_weights.py`. Tables:
`tables/sms2023_m2_by_predator_class.csv`, `tables/sag_dgs_productivity.csv`, `tables/sag_anf_productivity.csv`,
`tables/bottom_line_weighted.csv`. Life-history and residence searches were run by two helper agents; I
spot-checked their key quotes against the saved files.

Reused, not redone (cite these): r_max / d / r_now for saithe, whiting, hake
(`../../Torskfiskarnas reglering/r_pool_present.md`), mackerel and horse mackerel
(`../../Den pelagiska poolens reglering/productivity.md`); F for saithe, whiting, hake, mackerel, horse mackerel
(`../../../reports/Fiskeridödlighet i Skagerrak idag.md`, `../../Fiskeridödlighet i Skagerrak idag/*.md`); gear split
(`../../Fiskeridödlighet i Skagerrak idag/catch_by_gear.md`); gadoid M1/M2 (`../../Restdödlighet M0 hos torskfiskar/*.md`).

## 1. Intrinsic rate r and present-day productivity

Method as in the reused notes: r_max = FishBase CMSY prior; r_now = r_max (1 - ln(d)/3), d = fall in recruits per
spawner. For the new stocks d cannot be computed (no R/SSB) or shows no decline; then d = 1, said per species.

### 1.1 FishBase CMSY priors and life history [F]
Source: `fishbase.se/summary/<Genus-species>.html` (saved in `data/lit_life_history/`), Resilience field.

| Species | Prior r (95 % CL) | Resilience text (verbatim) | tmax | Linf / K (NE Atl.) |
|---|---|---|---|---|
| spurdog | **0.07** (0.05-0.10), "Based on 10 full stock assessments" | "Very Low, minimum population doubling time more than 14 years (rm=0.034; K=0.03-0.07; tm=10-30; tmax=75; Fec=1)" | 75 | ICES annex: F 110.7 / 0.086, M 81.4 / 0.17 |
| anglerfish | **0.47** (0.31-0.70), 8 full assessments | "Medium ... (K=0.07-0.33; tm=4.5-14; tmax=24; Fec=1 million)" | 24 | N Sea 106 / 0.18 |
| ling | **0.55** (0.36-0.82), 9 full assessments | "Medium ... (K=0.12; tm=5-6; tmax=25; Fec=60 million)" | 25 | ICES advice 148.8 / 0.11 |
| pollack | **0.63** (0.42-0.95), "3 data-limited stock assessments" | "Medium ... (K=0.19; tmax=8)" | 8 (too low) | S Ireland 85.6 / 0.186 |
| velvet belly | none | "Very Low, minimum population doubling time more than 14 years (Fec=6)" | - | S Portugal 55.8 / 0.14 (Coelho 2007) |

### 1.2 Spurdog: three independent lines
- **Life tables** (NE Pacific and NW Atlantic stocks, not NE Atlantic), from Chapple & Botsford 2013, Conserv Biol,
  doi:10.1111/cobi.12053, Table 1: Smith, Au & Show 1998 rebound r2M 0.017 (British Columbia), 0.034 (NW Atlantic);
  Frisk et al. 2001 0.04 / 0.12; Cortés 2002 stochastic λ 0.893 / 0.938 (below 1 as published; Cortés called λ "a
  relative index of population growth"). Table read [F]; the originals not [S]. FishBase rm 0.034.
- **Pardo et al. 2016 equation with NE Atlantic inputs** [D, helper]: age at maturity ~11.6 yr (80 cm, annex female
  growth), litter 6.6 pups at 90 cm, 2-yr gestation: rmax 0.089 (0.04-0.14) at M 0.1.
- **The assessment itself** [F/D]: B0 = Blim/0.2 = 1.224 Mt. Total biomass fell to 194 kt in 2004 (0.16 B0) and was
  652 kt in 2025 (0.53 B0). Surplus production (ΔB + C)/B, 2010-25: 0.05-0.085, mean 0.077; it falls as B rises
  (2023-25: 0.050-0.061). HRMSY 0.042. With no catch, "HR = 0" gives Btot +4.3 % (2028 on 2027) and +8.5 % (2029).
  Caveat: these are model outputs; "The magnitude of the recruitment spikes in 2016 and 2022 are likely an artefact".
- **d = 1**: recruits per tonne of biomass rose (median 0.24 in 1949-90, 0.41 in 2006-25) [D]. Before 1975 recruitment
  "was therefore derived from a single stock–recruitment relationship", so this is not independent evidence. The
  case for no decline rests on the biomass trend: ~5-7 %/yr growth under u < 0.01.
- **Reading**: the prior 0.07, the Pardo-type 0.09 and the realised 0.05-0.08 at half of B0 agree. r_now = **0.07**
  (0.05-0.10). The stock is recovering at ~5 %/yr (+4.9 % in 2024-25) under u < 0.01. A logistic r(1 - B/B0) would
  give 0.10-0.13, but elasmobranch production curves are skewed (MSY Btrigger 690 kt = 0.56 B0), so I do not use it.
- Stock annex: "Population productivity is low, with low fecundity and a protracted gestation period." [F]

### 1.3 The other new members
- **Anglerfish**: SSB fell from ~232 kt (1949, near virgin) to 24 kt (2004) and is 47 kt in 2025 (0.20 of 1949) [D,
  SAG]. R/SSB rose (median 1.3 in 1949-90, 4.7 in 2006-25): compensation at low SSB, no sign of lower productivity.
  The 2023 and 2025 year classes are very strong (428 and 590 million). d = 1, r_now = **0.47** (judgement on d).
- **Ling**: category 3, no R/SSB; the CPUE index halved from 2020 to 2024 under F slightly above F = M. d unknown;
  r_now = prior **0.55** (d = 1 by default, gap).
- **Pollack**: category 5, no data on productivity. r_now = prior **0.63** (gap).
- **Velvet belly**: no prior. Coelho 2007 (PhD thesis, S Portugal, Table 8.5) [F]: r 0.045-0.126 with natural M only
  (-0.017 with the Jensen-age M); negative with fishing. Pardo-type rmax 0.24-0.30 [D, helper]. Use ~**0.10**
  (0.05-0.25), judgement.
- **Reused**: saithe 0.52 -> 0.44 (d 1.6), whiting 0.49 -> 0.49, hake 0.48 -> 0.26 (d 3.9), mackerel 0.43 -> 0.25
  (d 3.6), horse mackerel 0.39 -> 0.33 (d 1.96) [S, project notes].


## 2. Fishing mortality today, and its gear split

Convention (as in the F report): F = u = catch / total biomass at 1 January. Fbar or a harvest rate is shown
separately. Bottom = bottom-otter-trawl map; pelagic = pelagic map (OTM/PTM/PS). Gillnets, longlines, hooks and
seines are on neither map.

### 2.1 New stocks
- **Spurdog** (dgs.27.nea, category 1, age-length-sex model, benchmark 2021) [F]
  - SAG "SSB" column is total biomass: the advice Table 1 gives "Total biomass (Btot; 2026) 679893", the same number.
  - u = 4963 / 652 023 = **0.0076** (2025); mean 2023-25 **0.0073**; 2016-25 0.0063 [D]. Reported harvest rate
    (ages 5-30, by numbers) 0.011 (2025); HRMSY 0.042. — [ICES Advice 2026 dgs.27.nea](https://doi.org/10.17895/ices.advice.30932039)
  - Management: zero TAC 2010-2022, with "Landings of bycatch permitted up to 10% of the 2009 quota" (2010) and
    later "spurdog was listed as a prohibited species in EU waters of subareas 2–10 for the years 2017–2022" except
    a 270 t avoidance-programme bycatch quota. TACs again from 2023 (15 453 t), 2024 (15 900), 2025 (19 866),
    2026 (10 875 t, "54.05% of the advised catch"). Catches stay far below the TAC: 4.2-5.0 kt in 2023-25. [F]
  - Gear, 2025 landings (2532 t): "Set nets 14%, All bottom trawls 62%, Hooks and lines 5%, Other gears 20%".
    Discards (2431 t, half the catch, "All discards are assumed to be dead") are not split by gear. Pelagic gear 0 %.
    Bottom share of the catch ~0.6-0.75 if discards are mainly trawl (judgement). [F]
  - Status: "Fishing pressure on the stock is below HRMSY. Stock size is below MSY Btrigger, and above BPA and Blim." [F]
- **Anglerfish** (anf.27.3a46, both Lophius species, category 1 SS3 since WKBFLATFISH 2024) [F]
  - Fbar(5-10) **0.092** (2025), 0.10-0.16 in 2015-2022; FMSY 0.137. — [ICES Advice 2026 anf.27.3a46](https://doi.org/10.17895/ices.advice.30931943)
  - Total biomass is not published (SAG and the advice give SSB only). Catch / SSB = 17 254 / 46 997 = 0.37 [D]. This
    is an upper bound only: most of the catch is immature. Judgement: catch / total biomass ~**0.08** (0.06-0.10).
  - Gear, 3.a + 4, 2025 landings (11 393 t): "Demersal trawls 65%", "Gillnets 20%", "Nephrops directed trawls 12.3%",
    "Other gears 2.5%". Bottom trawl 77 %, pelagic 0 %. [F]
  - "L. budegassa consistently constituted approximately 10% of the stock" (the IBTS share here is L. piscatorius). [F]
- **Ling** (lin.27.346-91214, category 3, CPUE index) [F]
  - No absolute F. Length-based proxy: "Fishing pressure proxy (LF=M 2024/Lmean) 1.04", i.e. F a little above F = M.
    "Fishing pressure on the stock is above the FMSY proxy". Index fell from 171 (2020) to 82 kg/1000 hooks (2024).
  - With the advice's own "Linf = 148.81 cm; k = 0.11" and its M/K = 1.5: M = 0.165, so F ≈ **0.17** (0.15-0.20) [D].
  - Gear, 2025 catch (13 942 t): longline 38 %, bottom trawl 40 %, gillnets 11 %, other 11 %. Pelagic 0 %. —
    [ICES Advice 2025 lin.27.346-91214](https://doi.org/10.17895/ices.advice.27202680) [F]
- **Pollack** (pol.27.3a4, category 5) [F]
  - "ICES cannot assess the stock and exploitation status". "pollack is taken solely as bycatch". "Recreational
    catches are known to take place and might be substantial but are currently not quantified." No F (gap).
  - Gear, 2023 (2077 t): "64% otter trawl", "23% gillnet", "4% seine", "4% longline", "5% other". —
    [ICES Advice 2024 pol.27.3a4](https://doi.org/10.17895/ices.advice.25019474)
- **Velvet belly** (Etmopterus spinax): no ICES assessment. Discarded bycatch of Nephrops/shrimp trawls; F unknown (gap).

### 2.2 Western horse mackerel: which number
- u = 61 434 / 1 596 010 = **0.038** (2025); 2021-25: 0.070, 0.056, 0.009, 0.010, 0.038, mean **0.037**
  (`ices_pelagic_f.md`, SG key 25892) [D]. Fbar(1-10) 0.065, FMSY 0.08.
- The "~0.053" in the brief is the Baranov F at the pelagic pool's M 0.58 (`ices_pelagic_f.md` table, column
  "Baranov F, M=0.58" = 0.052). It is not a horse mackerel value. With horse mackerel's own M (§3) the Baranov
  conversion gives ~0.04. Use **0.038** (convention: catch / TSB).
- Gear, 2025 landings: "Pelagic trawl 80 %", "Otter trawl 7%", "Purse-seine 12 %", "Unspecified* and other gears
  1 %" ("Most of these catches are taken by pelagic trawls"). — [ICES Advice 2026 hom west](https://doi.org/10.17895/ices.advice.30932114) [F]
  In the EU North Sea FDI (27.4, 2022-24, 2.0 kt) the bottom-otter share was 9.9 % (`catch_by_gear.md`) [S].
  Use 7-10 % bottom / 90-93 % pelagic.

### 2.3 Gear check for the reused species (`catch_by_gear.md`, FDI 3.a.20 2022-24 and ICES Table 7 2025) [S]
| Species | F (u) | Local 3.a.20 | Stock level | Use bottom / pelagic |
|---|---|---|---|---|
| saithe | 0.30 (0.11-0.38) | OTB 96.8 %, pelagic 0 % | OTB 85.2 %, purse seine 6.1 % (Norway, outside) | 100 / 0 (local) |
| whiting | 0.08 (0.035-0.15) | OTB 95.0 %, pelagic 0.7 % | + industrial bycatch 6.1-6.4 % | 94 / 6 (with IBC) |
| hake | 0.27 | OTB 59.3 %, rest gillnet/longline/seine | gillnet 49.6 %, trawl 25.1 % | 100 / 0 (59 % really on the map) |
| mackerel | 0.215 | 82 % hooks/static (neither map), OTB 13.4 %, pelagic 2.9 % (503 t) | bottom gear 1.1 % | 1 / 99 (stock level) |

- Saithe is not 100 % bottom trawl at stock level, but the purse-seine part is a Norwegian fishery far from the box.
  Mackerel is ~99 % pelagic at stock level; the local catch is hook and line. Both checks hold.


## 3. Natural mortality: M0, M2 by model groups, M2 by non-model groups

### 3.1 SMS key run 2023 (North Sea), biomass-weighted [D]
- Files: `partial_m2.out`, `summary.out` (N, BIO, M1, M2 by year, quarter, species, age), `species_names.in`
  ([NorthSeaKeyRun_2023](https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2023)) [F].
- Method: annual rate = Σ_quarters Σ_ages rate × BIO / mean quarterly BIO, 2018-2022 mean. The predator classes sum
  exactly to the SMS M2 total (check passed). Script `scripts/sms_m2_split.py`; full table in
  `tables/sms2023_m2_by_predator_class.csv`.
- Predator classes. Model: cod, haddock -> `benthivorous_fish`; harbour porpoise -> `porpoises`. Within
  `predatory_fish`: whiting, saithe, mackerel, hake, horse mackerel. Non-model: grey seal; the eight seabirds
  (`seabirds` is a model group, but muted, so non-model while muted). Ambiguous: grey gurnard and starry ray
  (the task lists them as non-model; `split_brief.md` lists them as `benthivorous_fish` members).
- SMS has no harbour seal. None of its seal M2 is the model's harbour seal. Grey seals are rare in the Skagerrak, so
  the North Sea grey-seal term is an upper figure for the box.

**Whiting, North Sea, per year (ages 1+ / ages 3+):**

| Term | 1+ | 3+ |
|---|---|---|
| M1 (residual, fixed 0.05/quarter) | 0.200 | 0.200 |
| M2 total | 0.485 | 0.272 |
| within `predatory_fish` (saithe 0.085 + whiting cannibalism 0.094) | 0.178 | 0.031 |
| model: harbour porpoise | 0.191 | 0.182 |
| model: cod (`benthivorous_fish`) | 0.050 | 0.038 |
| non-model: seabirds (guillemot 0.030) | 0.031 | 0.000 |
| non-model: grey seal | 0.022 | 0.022 |
| ambiguous: grey gurnard (+ starry ray) | 0.011 | 0.000 |
| **non-model M2** (seal + birds; + gurnard) | **0.053** (0.064) | **0.022** |

- Ages 0+ add a large 0-group term (M2 0.82 incl. grey gurnard 0.24). The model's group is the survey biomass,
  mostly ages 1+, so 1+ is the right basis.
- The within-group term (0.18 at 1+) cannot be a pair: section 160 keeps within-group predation out of the budget.
  Leaving it out lowers whiting's total M by 0.18; putting it into `natural_mortality` keeps the real total. This is
  the user's decision. It moves the group's M by ~0.07 (37 % weight).
- **Saithe**: M2 = 0 at all ages, and saithe has no prey rows. This is structural: SMS treats saithe as a predator
  only, and its 0-2 year olds live on the Norwegian coast outside the model area (`ices_sms_assessments.md`).
- **Mackerel**: M1 = 0.0375/quarter = 0.15/yr; M2 = 0 and no prey rows at all (checked: `awk '$5==20'` gives 0 rows).
  Structural, not a filter artefact. SMS M for mackerel = the assessment M 0.15.
- **Hake, horse mackerel**: external predators in SMS, no M estimated.
- For context (prey of this group, model cross-group pairs): haddock 1+ M2 0.40, of which saithe 0.19 and whiting
  0.075; cod 1+ M2 0.42, of which porpoise 0.19 and grey seal 0.15 (table).

### 3.2 Assessment M and Then et al. 2015 for the members SMS does not cover
Then 2015: M = 4.899 tmax^-0.916 and M = 4.118 K^0.73 Linf^-0.33 (formulas as in
`../../Restdödlighet M0 hos torskfiskar/empirical_estimators.md`). Then values are [D].

| Species | Assessment M (quote, source) [F] | Then tmax / growth [D] | Adult M used |
|---|---|---|---|
| saithe | "scaled to 0.2 at 5kg (≈mean weight of age 9)"; ages 3-10+: 0.384, 0.335, 0.294, 0.259, 0.232, 0.212, 0.197, 0.177 (pok.27.3a46 annex 2026, doi:10.17895/ices.pub.32791221) | 0.257 / 0.238 | **0.29** [D]: weighted by SMS Q1 biomass at age 2018-22 (3-6 only: 0.32) |
| mackerel | "Natural mortality (M) has been fixed at 0.15 for decades" (mac.27.nea annex) | - | **0.15** |
| horse mackerel | Gislason et al. 2010 M at age: age 1 0.66, 2 0.44, 3 0.37, 5 0.29, 10 0.22, 15 0.18 (hom west annex 2025) | - | **0.30** (0.22-0.37), judgement for the adult mix |
| hake | Table 4: age 0 1.19, ages 1-4 0.64, ages 5-14 0.34 (F) / 0.415 (M), 15+ 0.2 / 0.279 (hke annex 2024; the old "M=0.4." line is legacy) | 0.315 / - | **0.45** (0.34-0.64), judgement: Skagerrak hake are many juveniles |
| spurdog | "WGEF has assumed a length dependent natural mortality with a value of 0.1 for a large range of ages"; "Madult ... 0.1 ... expert opinion" (dgs annex 2022, doi:10.17895/ices.pub.18623396) | 0.094 / 0.15 (F) | **0.10** |
| anglerfish | now "Age-specific natural mortalities (Lorenzen, 1996; Thorson et al, 2017)" (advice 2026); earlier annex: "tmax method of Then et al., (2015) is preferable ... (0.25–0.4)" | 0.267 / 0.12-0.25 | **0.25** (0.2-0.4) |
| ling | "a natural mortality of 0.15 is assumed" (lin annex 2026; used only for the LBI reference) | 0.257 / 0.12-0.19 | **0.15** |
| pollack | none (category 5) | 0.73 (tmax 8, implausible) / 0.28 | **0.28** (growth estimator) |
| velvet belly | none; Coelho 2007 Table 8.3: 0.21-0.42 | 0.545 / 0.23-0.27 | **0.30** (0.21-0.42) |

### 3.3 Split into M0, model M2 and non-model M2
- **Whiting**: from SMS (§3.1). M0 = 0.20 (convention). Non-model M2 0.053 (0.064 with gurnard and ray).
  `natural_mortality` part = **0.25**; **0.43** if the within-group 0.18 is added.
- **Saithe, mackerel, spurdog, anglerfish, ling**: adult predators are few. The assessment M is a total M, and I put
  all of it in M0 + non-model. Non-model predators exist but are not quantified: grey seals and minke whales on
  saithe; dolphins, gannets and returned Atlantic bluefin tuna on mackerel (44 Skagerrak bluefin stomachs 2020-22:
  mainly garfish, herring and mackerel [S, search summary of DTU Aqua news 2025](https://www.aqua.dtu.dk/english/newsarchive/2025/12/tuna-and-diet)).
  Model M2 on adults (porpoise, harbour seal) is taken as ~0 (judgement); a small double count remains if the
  seal/porpoise pairs also eat saithe.
- **Hake, horse mackerel**: the assessment M includes cannibalism and predation by hake/saithe (within-group) and
  by cetaceans. Not separable; all in M0 + non-model (judgement).
- Weighted over the group (normalised shares, `tables/bottom_line_weighted.csv`) [D]: total M 0.42; model M2 0.09
  (whiting only); within-group 0.07 (whiting only); **M0 + non-model M2 = 0.26** (0.32 with within-group).
  Saithe + whiting only: 0.27 (0.36 with within-group).


## 4. Seasonal residence in the box (literature context; the project computes Q1 vs Q3 itself)

- **Saithe** [F]: one stock, 4 + 6 + 3.a.20. "smaller/younger fish are found in the Skagerrak compared to
  Subareas 4 and 6" (pok annex 2019). "Juvenile saithe are mainly distributed along the coast and in the fjords of
  western and southern Norway"; they "migrate from nursery areas to the North Sea within the ages of 2–5". FishMap:
  "a clear zone of high abundance along the edge of the Norwegian Deep"; "density in the northeastern North Sea is
  markedly higher during summer than during winter"; adults "close to the bottom, down at depths of more than 300 m
  during day time", rising at night. Spawning "in February–March" on the shelf edge; tagging (Homrum et al. 2013,
  ICES JMS 70:782, doi:10.1093/icesjms/fst048) shows southward spawning migrations from Norway into the North Sea.
  Reading [D]: immatures (1-4 yr) all year; adults mainly Apr-Dec, low in Jan-Mar.
- **Mackerel** [F]: the North Sea component "overwinters in the deeper parts of the Skagerrak and north-eastern North
  Sea from where it ascends to the surface layer in spring"; "main spawning period is mid-May to late June", mostly
  in the central North Sea (Jansen & Gislason 2011, Cont. Shelf Res. 31:64, in the
  [Jansen PhD thesis](https://backend.orbit.dtu.dk/ws/files/52281725/Teunis_Jansen_PhD.pdf)). Norwegian Skagerrak
  fishery (Revheim 1969, ICES CM 1969/H:6): trolling "starts in June and lasts to August - September"; "In autumn
  the mackerel start to withdraw from the coastal waters and congregate in larger shoals in the Norwegian Channel and
  its slope"; purse seine "peak of catch is reached in October". Acoustic surveys in August: "relative high
  concentrations of mackerel" (Degnbol et al. 1988, ICES CM 1988/H:40). "Mackerel stop feeding almost completely
  during winter" (FishMap). Reading [D]: in the box Jun-Nov (feeding above the thermocline, then deep on the trench
  slope in autumn); Dec-May sparse and deep, not feeding. Today's North Sea component is small, so winter presence
  is probably low.
- **Western horse mackerel** [F]: since 2024 "catches in divisions 3.a and 4.a are now considered to belong to the
  western stock"; large fish "may reach areas around the Shetland Islands, the Norwegian coast, and the northern North
  Sea by September" (hom annex 2025). Autumn catches follow the winter Atlantic inflow (Iversen, Skogen & Svendsen
  2002, Fish. Oceanogr. 11:245) [A]. Norwegian memo 2025: bycatch "har ... tiltatt noe fra slutten av august", taken
  until November "langs «kanten» i Nordsjøen" [F]. Reading [D]: Aug/Sep-Dec, some into Feb; absent in spring (spawning
  west of the British Isles and in Biscay).
- **Spurdog**: old tagging: fish "regularly recaptured off the coast of Norway, indicating a winter migration from
  Scotland, returning in summer (Aasen, 1960; 1962)" (dgs annex 2025) [F]. Swedish surveys 1967-2020: spurdog the one
  elasmobranch with a seasonal shift, offshore Skagerrak and the Norwegian side in Q1, nearer the Swedish coast in
  Q3-Q4; bycatch "in the deeper parts (> 200 m) of Skagerrak and the Norwegian trench" (Börjesson et al. 2022, SLU
  Aqua Reports 2022:14) [S, PDF not opened]. Tagged pregnant females (W Norway fjords) used depths "down to 300 m
  at temperatures between 8 and 14 °C" and in summer-autumn "predominantly selected warm waters above the
  thermocline" (Klöcker et al. 2024, Mov. Ecol. 12:62, doi:10.1186/s40462-024-00498-2) [F]. Reading [D]: all year,
  richest on the offshore Norwegian side in Q1.
- **Whiting**: "Part of the Skagerrak population is thought to migrate into the north-eastern North Sea to spawn"
  (FishMap); "Spawning occurs from January to July at 30–100 m depth" (whg.27.3a annex 2026) [F]. Largely resident [D].
- **Hake**: the North Sea increase since 2004 is in an area "where hake have been largely absent for over 50 years";
  "high densities occur in the North Sea only between April and September" (Baudron & Fernandes 2015, Fish Fish.
  16:563) [A]. In the Swedish Skagerrak survey (30 Sep-19 Oct 2024) hake were in 28 of 41 hauls, saithe in 23 (606 kg),
  horse mackerel 13, mackerel 11 (bottom trawl under-samples it), spurdog 9 [F]. Reading [D]: at least Apr-Oct.


## Bottom line

Per year. F = catch / total biomass. F_bot / F_pel: the project convention puts every non-pelagic gear (gillnet,
longline, seine) on the bottom map. natmort = M0 + M2 by non-model predators (the `natural_mortality` part);
"within" = predation inside `predatory_fish`, which no pair can carry (section 160). Residence: months in the box.
Weighted rows use the long-run shares normalised to 100 % (`scripts/bottom_line_weights.py`,
`tables/bottom_line_weighted.csv`) [D].

| Species (share) | r_max prior | d | r_now | F total | F_bot / F_pel | M total | M2 model | within | natmort (M0 + non-model) | Residence | Tags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| saithe (41 %) | 0.52 | 1.6 | 0.44 | 0.30 (0.11-0.38) | 0.30 / 0 | 0.29 | ~0 | 0 | 0.29 (0.20-0.32) | immatures all year; adults Apr-Dec | [S] r, F; [F]/[D] M |
| whiting (37 %) | 0.49 | 1.0 | 0.49 | 0.08 (0.035-0.15) | 0.075 / 0.005 | 0.685 (1+) | 0.24 (porpoise 0.19, cod 0.05) | 0.18 | 0.25 (0.20 + 0.053) | all year | [S] r, F; [D] SMS |
| mackerel (8 %) | 0.43 | 3.6 | 0.25 | 0.215 | 0.002 / 0.213 | 0.15 | 0 | 0 | 0.15 | Jun-Nov | [S] r, F; [F] M |
| spurdog (5 %) | 0.07 (0.05-0.10) | 1 | 0.07 (0.05-0.10) | 0.0076 | 0.0076 / 0 | 0.10 | ~0 | 0 | 0.10 | all year, Q1 offshore | [F] [D] |
| horse mackerel (3.4 %) | 0.39 | 1.96 | 0.33 | 0.038 | 0.003 / 0.035 | 0.30 | ~0 | n.s. | 0.30 | Aug/Sep-Dec | [S] r, F; [F] M |
| hake (2.5 %) | 0.48 | 3.9 | 0.26 | 0.27 | 0.27 / 0 | 0.45 | ~0 | n.s. | 0.45 | at least Apr-Oct | [S] r, F; [F] M |
| anglerfish (1.4 %) | 0.47 | 1 | 0.47 | ~0.08 (Fbar 0.092) | 0.08 / 0 | 0.25 | ~0 | 0 | 0.25 | resident (assumed) | [F] [D] judgement F |
| ling (0.7 %) | 0.55 | ? (1) | 0.55 | ~0.17 (F ≈ M) | 0.17 / 0 | 0.15 | ~0 | 0 | 0.15 | resident (assumed) | [F] [D] |
| velvet belly (0.5 %) | none | ? | ~0.10 (0.05-0.25) | ? (0.10 assumed) | 0.10 / 0 | 0.30 | ~0 | 0 | 0.30 | resident (assumed) | [F] thesis; judgement |
| pollack (0.4 %) | 0.63 | ? (1) | 0.63 | ? (0.20 assumed) | 0.20 / 0 | 0.28 | ~0 | 0 | 0.28 | resident (assumed) | [F] prior; judgement |
| **Weighted, all** | **0.47** | | **0.42** | **0.18** | **0.16 / 0.020** | 0.42 | 0.09 | 0.07 | **0.26** (0.32 with within) | ~0.87 of the year present | [D] |
| Weighted, saithe + whiting | 0.51 | | 0.46 | 0.20 | 0.19 / 0.002 | 0.48 | 0.11 | 0.08 | 0.27 (0.36) | | [D] |
| Weighted, no elasmobranchs | 0.50 | | 0.44 | 0.19 | 0.17 / 0.021 | 0.44 | 0.09 | 0.07 | 0.27 (0.34) | | [D] |

n.s. = not separable from the assessment M. Residence fraction [D, judgement]: saithe 0.85, whiting 1, mackerel 0.5,
spurdog 1, horse mackerel 0.42, hake 0.8, the small residents 1.

Reading:
- The group's r_now (~0.42) is set by saithe and whiting. The elasmobranchs lower it by only ~0.02.
- F 0.18 is ~90 % bottom-map. Saithe (0.30) carries two thirds of it; its F range (0.11-0.38) is the largest single
  uncertainty. The pelagic part (0.02) is mackerel and horse mackerel at stock level.
- M0 + non-model M2 is ~0.26. The open choice is whiting's within-group 0.18 (+0.07 on the group).

## Gaps
- Saithe F: stock-level F on a seasonal, schooling visitor; the local share is unknown.
- Anglerfish total biomass is not published (SS3 output not public on TAF); catch/TB 0.08 is a judgement.
- Ling and pollack have no absolute F or R/SSB; velvet belly has no assessment at all (F and r are judgements).
- Spurdog: discards (half the catch) are not split by gear; life-table r values are from NE Pacific / NW Atlantic
  stocks (Smith 1998, Cortés 2002 [S] only); the realised production is model output.
- Non-model predation on adult saithe, mackerel, hake and horse mackerel (grey seal, minke whale, dolphins,
  bluefin tuna) is not quantified anywhere I found. SMS gives saithe and mackerel M2 = 0 by construction.
- SMS is North Sea-wide. Grey seals are rare in the Skagerrak (their 0.022 on whiting is an upper figure) and the
  porpoise density in the Skagerrak is high; local M2 will differ.
- Grey gurnard and starry ray are non-model here but `benthivorous_fish` members in `split_brief.md`. If they are
  model predators, whiting's non-model M2 falls from 0.064 to 0.053 (1+) and its model M2 rises by 0.011.
- Hake and horse mackerel adult M are judgements over age vectors (no box age structure used).
- "Resident (assumed)" for anglerfish, ling, pollack and velvet belly has no source; it is a default.
- Residence months are literature readings; Jakobsen 1987, Lindquist & Hannerz 1974 and the Norwegian shrimp survey
  were not obtained. The SLU 2022:14 elasmobranch report was read only via search summaries.

