# One vision rule for the water column's krill and fish pairs (2026-11-11)

Research note, written by a research subagent. Literature only. Nothing in the library, code, tests,
manifest, resume or AGENTS.md was changed, no training was started, and no value was set from model
output.

Files:
- `scripts/one_rule.py` -> `tables/one_rule_stdout.txt`: R_krill and R_fish under one rule at the box
  light (100 / 150 / 200 / 250 m), two ways of getting the saturated R, four eye sensitivities.
- `papers/`: local PDFs and text extracts; `papers/README.md` lists source and licence. **Never commit.**

Tags: [S] source (read here, or an abstract where marked), [D] derived here, [J] judgement.

Units: lx and umol m-2 s-1 (= uE m-2 s-1) are converted at ~40-55 lx per umol for deep blue-green water,
the convention of "Ljus under 50 m". The salmonid tanks measured photopic lux under lamps at 67 lx per uE
(Hansen 2014). This conversion moves the gadoid Ke bracket by at most x1.5.

## Findings

1. **Nobody has measured a gadoid's reactive distance (R) on krill or on fish prey.**
   - The only gadoid R-vs-light study is Meager et al. 2010 (juvenile cod, 0.01-64 umol m-2 s-1). Its
     abstract says R was still high at the lowest level tested, 0.01 umol (~0.4-0.5 lx). The full text
     is closed (no repository copy), so its R values in cm are not available here.
   - The numbers therefore come from other sources:
     - gadoid feeding-vs-light thresholds (juvenile walleye pollock, adult cod contrast thresholds);
     - piscivorous salmonids, used as analogues. Their R has been measured against light, turbidity
       and prey size.
2. **Under one visual rule, R_fish >= R_krill at every light level, so the library has the sign
   backwards.**
   - Piscivore R does not change with prey size over 55-139 mm (lake trout). It falls only below about
     50 mm, as FL^0.42 in low light and FL^0.71 in high light (Chinook, 23-50 mm prey) [S].
   - A 30 mm krill therefore gets 0.53-0.61 of the R of a >= 50 mm fish, before any contrast term [D].
     Krill are partly transparent and counter-illuminate [S], which lowers the krill side further [J].
   - In darkness the non-visual floor scales with the size and swimming speed of the hydrodynamic
     source (the prey), which again favours fish [D/J].
   - Today the library gives the krill pair R_sat 0.87 m under the measured light, against a fixed
     0.2 m for the fish pairs. That is a daytime R^2 ratio of krill to fish of 10-18 at 100-150 m.
     The sibling note's x5-7.7 is the overlap- and time-weighted mean of the same ratio, nights
     included.
   - One rule gives a daytime R^2 ratio of krill to fish of 0.3-0.6 at 100-150 m. At 200-250 m,
     where vision fades, it falls to 0.12-0.27, and at night to 0.11.
3. **The krill pair's own R_sat (0.87 m) is consistent with this rule. The fish pairs' 0.2 m is the
   outlier.**
   - Salmonid RD_max on 50 mm fish is 0.5-1.9 m [S]. Langbehn et al. 2019 give a piscivore a visual
     range of 1.5 m on a 6 cm fish [S, a model value].
   - Scaled to a 30 mm krill, both give 0.5-1.2 m [D].
   - The correction therefore points at the fish pairs (the sibling note's V_one direction), not at the
     krill R.
4. **Eye sensitivity (Ke): the gadoid evidence supports the library's Ke_gad of 0.02 lx or a more
   sensitive value. It rules out the salmonid ~10 lx.**
   - Ryer & Olla 1999 (original read here; the sibling note's "verify" flag is resolved): juvenile
     pollock stop feeding visually between 1e-7 and 1e-6 uE m-2 s-1 (geometric mean 5e-7), and
     sablefish at 5e-5 uE.
   - Pollock consumption plateaus by 1e-4 to 1e-3 uE (~0.005-0.05 lx). The half-way point of the
     visual increment is ~4e-6 uE (~2e-4 lx) [D, read from Fig. 1].
   - Salmonid lab fits put the half-saturation of R^2 at 7-12 lx [D]. With that Ke, every pair would sit
     at its non-visual floor over every bed >= 100 m by day.
   - The cod data rule that out: R still high at ~0.5 lx (Meager 2010), the scotopic transition at
     ~0.005 lx (Anthony 1981). Salmonids are shallow-water, photopic-lux tank fish.
5. **Non-visual floor (darkness).**
   - Measured gadoid dark feeding: juvenile pollock ate 28-40 Artemia in darkness against ~150 in light
     per 10 min, a ratio of ~0.2-0.3 [S, text and figure].
   - Analogues:
     - Chinook R levels off at ~0.5 m at the lowest light (0.03 lx). Rohan et al. read this as
       non-visual detection or an artefact.
     - The lateral-line operating range is put at 1-2 body lengths in reviews; a toadfish measured
       0.4 body length.
   - Floor [J]: krill 0.1 m (0.05-0.2), fish 0.3 m (0.2-0.5).
6. **Capture success (strike -> capture): no fish-on-euphausiid value exists.**
   - The library's krill p of 0.5 against fish p of 0.2 (x2.5) rests on judgement for krill and on one
     lab study for fish.
   - The review figure is piscivores ~0.5 against planktivores 0.70-0.80 (x1.3-1.6). The sibling note's
     field-p variant (krill 0.6, fish 0.45, x1.33) sits inside that range. Whiting on shrimp is 0.4-0.8
     per strike.
   - Light lowers the combined detection-and-capture rate at night to 0.1-0.35 of day in every analogue
     found. That is a dark ratio, not a p.
7. **The diets show the krill share falling with predator length in every member with data.**
   - Saithe: from ~0.5-0.76 at 30-50 cm to <= 0.1 at >= 70 cm.
   - Velvet belly (Skagerrak): 0.89 below 20 cm, 0.07 above.
   - Hake: fully piscivorous above 23 cm (Biscay / Celtic Sea; no North Sea value found).
   - Large saithe stacked under krill swarms at 100-200 m had eaten fish and squid, not krill.
   - A size-capped one rule reproduces the direction: fish R grows with predator length up to the cap,
     krill R does not. The library's fixed fish R cannot.
   - Rough odds arithmetic on the sibling note's V_one [D, not a re-run]: the krill share at 150-250 m
     moves from 0.69 to about 0.42-0.57 with library p, and about 0.28-0.42 with review p. That
     brackets the field Q3 value of 0.47 (0.32-0.69) and the booked 0.36.
8. **What vision cannot fix: the juvenile benthivore pair.**
   - Under V_one it goes to 1.1-1.6x its booking, against a field diet share of ~5 %.
   - Those prey are > 50 mm, so no prey-size or contrast term in a visual rule lowers them, and the fish
     floor of 0.3 m raises them.
   - That pair needs a different mechanism or booking (near-bed refuge, prey vigilance, size window).
     That is outside this note.

## Item 1. Reactive distance against light, prey size and contrast

| # | value | units / conditions | species | source | tag |
|---|---|---|---|---|---|
| 1a | R "high" at 0.01 umol m-2 s-1 (lowest tested); lower at 1.5-6.5 umol; a two-part fit with a break at 5 umol; poor generality | lab, 0.01-64 umol; prey not given in the abstract | juvenile cod, age 1 | Meager et al. 2010, 10.1080/10236241003798910 | [S abstract; full text closed] |
| 1b | contrast threshold falls to ~2 % in bright light; scotopic/photopic break at ~8e-6 W sr-1 m-2 (~0.005 lx [D] at pi L) | cardiac conditioning | cod | Anthony 1981, 10.1111/j.1095-8649.1981.tb05814.x | [S via sibling note] |
| 1c | visual-feeding threshold between 1e-7 and 1e-6 uE m-2 s-1 (geometric 5e-7); consumption ~37-40 in total dark (< 1e-8), rising from 1e-6, ~105-120 at 1e-5 to 1e-4, ~150 at 1e-3 Artemia per 10 min | 5 fish x 200 Artemia, 10 min | juvenile walleye pollock (gadoid) | Ryer & Olla 1999, 10.3354/meps181041 | [S, abstract and Fig. 1 read] |
| 1d | sablefish threshold 5e-5 uE ("pollock better adapted for visual feeding at depth or at night") | as 1c | sablefish (non-gadoid) | Ryer & Olla 1999 | [S] |
| 1e | RD = 2.68 I + 55.4 cm up to 24.9 lx, RD_max 122 cm (Chinook); RD = 5.66 I + 85.3 up to 18.0 lx, RD_max 187 cm (cutthroat); no effect of predator length; tested 0.03-250 lx (surface lux, ~67 lx per uE) | tethered 50 mm rainbow trout prey, 1 h trials | yearling Chinook, adult coastal cutthroat (analogue) | Hansen, Beauchamp & Schoen 2013, 10.1080/00028487.2013.785978 (thesis ch. 2) | [S full text] |
| 1f | prey size: RD = FL^0.415 x 10^1.288 (low light, < SIT), FL^0.709 x 10^0.935 (high light), FL 23-50 mm; RD falls 30-40 % from 50 to 23 mm; low- and high-light RD converge at ~23 mm | as 1e | Chinook (analogue) | Hansen et al. 2013 | [S] |
| 1g | RD does not change with prey size over 55-139 mm | lab | lake trout (analogue) | Vogel & Beauchamp 1999, 10.1139/f99-071 (via Hansen 2014) | [S secondary] |
| 1h | RD = 0.2549 I^0.4747 m up to 17.8 lx, 1.01 m above | lab, fish prey | lake trout (analogue) | Vogel & Beauchamp 1999 via Eckmann 2016 (sibling note) | [S] |
| 1i | rainbow and cutthroat RD_max ~0.5 m; lake char 65 % farther at high light, the same at low light; lake char predation low in darkness, rising fast at 0.50-0.75 lx | lab | salmonids (analogue) | Mazur & Beauchamp 2003, 10.1023/A:1025807711512 | [S abstract] |
| 1j | generalised piscivore RD = 44.56 L^0.266 cm (L <= 24.2 lx), 104 cm above; planktivore 11.46 L/(0.23 + L) cm | synthesis of 1e and 1i | salmonids | Hansen 2014 thesis ch. 3 (eq. 3.5-3.6) | [S] |
| 1k | Aksnes-Utne form R^2 exp(cR) = \|C0\| A_p E' E/(Ke + E); in behavioural data the generalised model beats it in every case (turbidity is not monotonic); Chinook RD levels off at ~0.5 m at low light | four lab systems | Chinook, gobies, moroko | Rohan, Beauchamp, Essington & Hansen 2021, 10.1016/j.ecolmodel.2021.109688 | [S manuscript] |
| 1l | piscivore r_max 1.5 m on a 6 cm fish (\|C0\| 0.5, A = 0.75 L x 0.2 L); Ke 4e-4 mW m-2 nm-1 at 486 nm (~1.6e-4 umol, ~0.006 lx [D], assuming a ~100 nm effective band around 486 nm), set from scattering-layer light, not measured; non-visual threshold 0.1 m; capture p 0.7 | model | generic piscivore | Langbehn et al. 2019, 10.3354/meps13024 (Suppl. Table S1) | [S model value] |
| 1m | krill "partly transparent"; dense swarms cast a shadow that may favour spotting fish from below | field acoustics, Norwegian Sea | saithe under M. norvegica swarms | Kaartvedt et al. 2005, 10.3354/meps299001 | [S] |
| 1n | M. norvegica photophores are ventral, normally directed downwards (countershading); light production is triggered by falling light (dusk) and turbulence | lab | krill | Fregin & Wiese 2002, 10.1007/s10152-002-0104-4 | [S] |

Derived from the table [D]:
- **Ke as the half-saturation of R^2:**
  - Chinook (1e): R = 122/sqrt 2 = 86 cm at 11.5 lx.
  - Lake trout (1h): 0.716 m at 8.8 lx.
  - Generalised salmonid (1j): 6.6 lx.
  - Salmonids therefore give Ke ~7-12 lx (photopic lux, shallow tanks).
  - Pollock (1c) gives the half-way point of the visual increment at ~4e-6 uE (~2e-4 lx), with
    saturation by 1e-4 to 1e-3 uE.
  - Cod (1a, 1b) puts Ke below ~0.5 lx, with the scotopic break at ~0.005 lx.
  - Gadoid bracket: 2e-4 to 0.1 lx. The library's 0.02 lx sits inside it.
- **Prey-size ratio, 30 mm krill to >= 50 mm fish:** (30/50)^0.415 = 0.81 (dim light) and
  (30/50)^0.709 = 0.70 (bright). With a contrast ratio sqrt(0.4/0.7) = 0.76 [J] this gives 0.61 / 0.53.
- **The mechanistic rule alone (no cap) is not supported** by 1f/1g. For 15 cm vs 30 mm prey it gives R
  ratios of ~5 (3.24 m vs 0.68 m at c 0.25). The behavioural data say R stops growing above ~50 mm, so
  a cap is required. The library's 2 L_pred cap (0.86 m for 43 cm saithe, 1.40 m for 70 cm) falls in
  the salmonid RD_max range of 0.5-1.9 m.

## Item 2. Non-visual detection in darkness (the floor R_nv)

| # | value | conditions | species | source | tag |
|---|---|---|---|---|---|
| 2a | dark/light consumption ~0.2-0.3 (text: 28.3 total dark, 39.5 dark + IR, n.s.; Fig. 1: ~35-40 at 1e-8 to 1e-7 against ~150 at 1e-3 Artemia per 10 min) | total dark < 1e-8 uE vs 1e-3 uE | juvenile walleye pollock | Ryer & Olla 1999 | [S, figure read] |
| 2b | cod locate mysids by chemical cues from longer distances than by vision alone; predation "significantly lower" in the dark | lab, mysids 14-23 mm | juvenile cod | Meager et al. 2005, 10.1139/f05-104; Meager & Batty 2007, 10.1098/rstb.2007.2104 | [S abstract] |
| 2c | olfactory rheotactic response distance 240 m (day) vs 227 m (night); more bait localisation by day (70 vs 45 %) | field, bait | cod | Lokkeborg & Ferno 1999, 10.1023/A:1007504712163 | [S via sibling note]; bait, not live mobile prey |
| 2d | RD levels off at ~0.5 m at the lowest light (intercept 55 cm at 0.03 lx) | lab, 50 mm fish prey | Chinook (analogue) | Hansen et al. 2013; Rohan et al. 2021 | [S] |
| 2e | toadfish detect mobile prey only within ~0.4 body length (28 cm SL) by lateral line | lab | oyster toadfish (analogue) | Palmer et al. 2005, 10.1242/jeb.01766 | [S abstract-level, via search] |
| 2f | lateral-line operating range "1-2 body lengths" | reviews (Denton & Gray 1983, Kalmijn 1988, Coombs 1999) | general | as cited by Palmer et al. 2005 | [S secondary] |
| 2g | lateral line needed to detect invertebrate prey in the dark | lab | peacock cichlid (analogue) | Schwalbe et al. 2012, 10.1242/jeb.065920 | [S abstract] |
| 2h | whiting krill numbers equal day and night; Norway pout krill in fewer stomachs at night (44-48 % vs 59-75 %) | Oslofjord field | whiting, Norway pout | Onsrud et al. 2004, 10.1016/j.icesjms.2004.09.005 (sibling note) | [S] |
| 2i | krill bioluminescence (468 nm) lies within cod sensitivity | review table | cod / krill | Utne-Palm et al. 2018, 10.1371/journal.pone.0190918 | [S] |

**Size scaling of the floor [D/J].**
- A dipole source of size a moving at U gives a flow ~U a^3 / r^3, so at a fixed detection threshold
  r ~ a U^(1/3).
- 15 cm fish at 0.15 m/s against 3 cm krill at 0.04 m/s: r ratio ~5 x 1.55 = ~8. The receiver's 1-2
  body lengths (0.4-0.9 m for 43 cm saithe) caps it.
- So R_nv(fish) is several times R_nv(krill). Stimulated krill bioluminescence (1n, 2i) partly offsets
  this for krill.
- Proposed floor [J]: krill 0.1 m (0.05-0.2; library value), fish 0.3 m (0.2-0.5). The fish value lies
  between the toadfish 0.4 BL (~0.17 m for 43 cm saithe) and the Chinook ~0.5 m.
- The lake-trout power law (1h) has no floor (R -> 0); it is the low bracket.

## Item 3. Capture success

| # | value | conditions | species | source | tag |
|---|---|---|---|---|---|
| 3a | no measured capture probability of any fish on any euphausiid, in light or dark | - | - | sibling note "Krillens flykt och energibudget" G1 | gap |
| 3b | 39 % of advances successful alone, 55 % in groups of six (0.62-0.73 per attack if the remainder are failed attacks) | lab, tail-flipping mysid swarms | Australian salmon Arripis trutta (analogue) | sibling note E14 | [S abstract] |
| 3c | 0.8 per strike on small shrimp (0.2-0.34 of predator length), 0.4 on large; falls 0.05 per unit of stomach content; ~1 attack probability when empty | lab video, lit tank | whiting 23-30 cm | Staniland 1995 thesis (sibling note) | [S] |
| 3d | "about half of piscivore attacks are successful compared with 70-80 % for planktivorous fishes"; falls with prey : predator size | review | piscivores | Juanes, Buckel & Scharf 2002, 10.1002/9780470693803.ch12 | [S] |
| 3e | perch on single roach 0.20, pike 0.12 (pursuit) / 0.36 (sit-and-wait), pikeperch 0.085 | lab | percids, pike | Turesson & Bronmark 2004, 10.1111/j.0022-1112.2004.00455.x | [S via sibling note] |
| 3f | 0.5-1 per attack at prey : predator 0.20-0.33; 0.05-0.3 at 0.35-0.45 | lab, bright | 0-group cod on flatfish | Ellis & Gibson 1997, 10.3354/meps149083 | [S] |
| 3g | feeding efficiency at full moon ~35 % of day; < 10 % typical with the moon down | lab, drift prey | juvenile Atlantic salmon (analogue) | Fraser & Metcalfe 1997, 10.1046/j.1365-2435.1997.00098.x | [S abstract-level, via search] |
| 3h | large saithe (60-110 cm) under krill swarms had fish and squid, not krill | field | saithe | Kaartvedt et al. 2005 | [S] |

Reading:
- **Krill to fish p ratio:** the review gives 0.70-0.80 against ~0.5, i.e. x1.3-1.6 [S]. Whiting on shrimp gives
  0.4-0.8 against cod/percids on fish at 0.1-0.5. The library's x2.5 (0.5 vs 0.2) is at the top of what
  any source allows. A krill p of 0.6 (0.4-0.8) with a fish p of 0.45 (0.25-0.6) is the sibling note's
  field-p variant; nothing found here moves it.
- **Light acts on detection, not on p**, in every study found. The dark reductions (2a, 3g, 1i) are
  combined detection-and-capture rates of 0.1-0.35. A rule that puts light into R alone should keep p
  light-independent, so the dark is not counted twice [J].

## Item 4. Diet evidence

| # | value | where / when | predator | source | tag |
|---|---|---|---|---|---|
| 4a | euphausiid mass share by length, winter: 48 % (30-40 cm), 76 % (40-50), 21 % (50-70), 9 % (70-100), 0.1 % (>= 100); summer 28 / 37 / 46 / 34 / 13 % | North Sea 1981 | saithe | ICES CRR 164 (sibling note "Vattenpelarens par per djupklass") | [S] |
| 4b | Q2 euphausiids 49 % at 42 cm down to 7 % at 95 cm | North Sea 1991 | saithe | ICES CRR 219 (sibling note) | [S] |
| 4c | krill share all seasons 0.19 (0.08-0.27); NE North Sea Q3 0.55 (0.36-0.78); 2026 Q1 at true depth 0.0003 (5 stations); within a length class no drop from 100-150 to 150-250 m beds (odds ratio 0.98) | N of 57 N | saithe | sibling note (ICES stomach database; Fujii 2016, 10.3354/meps11534) | [S/D] |
| 4d | under krill swarms at 100-200 m (day and the light night, ~70 N, May): saithe stacked below; stomachs held fish and squid, not krill; blue whiting the likely krill eater and saithe prey | Norwegian Sea | saithe 60-110 cm | Kaartvedt et al. 2005 | [S] |
| 4e | M. norvegica 88.5 %W (< 20 cm, n 19) vs 7.1 %W (>= 20 cm, n 128); fish >= 55 %W in the larger (blue whiting 29.5, teleosts 16.4, Maurolicus 6.5, gadids 2.4) | Skagerrak deep, all seasons | velvet belly Etmopterus spinax | Bergstad, Wik & Hildre 2003, 10.2960/J.v31.a12 (Table 4, from Wik 1994) | [S] |
| 4f | small hake almost only crustaceans (mainly euphausiids); fully piscivorous above 23 cm | Bay of Biscay, Celtic Sea, autumn 2001 | hake | Mahe et al. 2007, 10.1093/icesjms/fsm100 | [S abstract]; no North Sea / Trench hake diet found |
| 4g | Bergstad 1991 (Norwegian Deep gadoids): "small saithe ... and the euphausids as prey" | Norwegian Deep | saithe | 10.1080/00364827.1991.10413455 (closed; sibling note) | [S abstract] |
| 4h | no length-resolved Norwegian-coast saithe diet found (IMR 1987 young-saithe report, not open) | - | - | gap | - |

Reading:
- Every member with data switches from krill to fish with length, at 20-60 cm.
- Within the box's saithe sizes (most 30-70 cm) the field krill share spans ~0.1-0.76 by length and
  season.
- A pair with a fixed fish R and a light-driven krill R gives the same krill share at every predator
  length (the sibling note's 0.78 / 0.79). A one rule with the size cap gives a share falling with
  length.

## Proposed single rule

One eye per predator member, one function for every prey:

    R_j^2(E) = R_nv,j^2 (1 - g) + R_sat,j^2 g,    g = E / (E + Ke)
    R_sat,j  = min(R0_j, cap_L L_pred),  R0_j^2 exp(c R0_j) = |C0_j| A_j k_v^2,  A_j = 0.75 L_j (0.2 L_j)

| parameter | value | basis |
|---|---|---|
| Ke | 0.02 lx (2e-4 to 0.1) | the library value. Inside the gadoid bracket (1a-1c, 1l); the salmonid 7-12 lx is rejected for gadoids |
| k_v, c | 100, 0.25 /m | library (section 245) |
| cap_L | 2 L_pred | library (section 220). Supported by salmonid RD_max 0.5-1.9 m with RD size-free above 50 mm (1e-1g) |
| \|C0\| krill / fish | 0.4 / 0.7 [J] | krill partly transparent and counter-illuminating (1m, 1n); fish opaque in silhouette. Langbehn uses 0.5 for both. Not chosen to close any gap |
| L krill / fish | 30 mm / 10-25 cm | |
| R_nv krill / fish | 0.1 / 0.3 m [J] | item 2 |
| p krill / fish | keep light-independent; ratio x1.3-1.6 (review), not x2.5 | item 3 |

Variant B (empirical, no image-area model): R_sat(fish) = min(RD_max, cap), RD_max 1.0-1.9 m (salmonids);
R_sat(krill) = R_sat(fish) x (30/50)^b x sqrt(0.4/0.7), b 0.415-0.709.

**Numbers** (`tables/one_rule_stdout.txt`):
- Saithe 43 cm. Daytime-median box light: 100 and 150 m are section 253 values; 200 and 250 m are
  extended with K_deep 0.06 (0.09 dark bound to 0.05 bright bound).
- Central Ke 0.02 lx.
- The last column is the library's daytime instantaneous R^2 ratio: the krill rule against the fixed
  0.2 m fish R.

| depth | E (lx), best (bracket) | g | R_krill A / B (m) | R_fish 15 cm (m) | R^2 krill/fish A / B | library, daytime |
|---|---|---|---|---|---|---|
| 100 m | 0.43 (0.15-0.84) | 0.96 | 0.66 / 0.48 | 0.84 | 0.61 / 0.32 | 17.7 |
| 150 m | 0.022 (0.0017-0.069) | 0.52 | 0.49 / 0.36 | 0.66 | 0.57 / 0.30 | 9.8 |
| 200 m | 1.1e-3 (1.9e-5 to 5.7e-3) | 0.05 | 0.18 / 0.15 | 0.35 | 0.27 / 0.18 | 1.2 |
| 250 m | 5.5e-5 (2.1e-7 to 4.6e-4) | 0.00 | 0.11 / 0.10 | 0.30 | 0.12 / 0.12 | 0.3 |
| night, any depth | ~0 | 0 | 0.10 | 0.30 | 0.11 | 0.25 |

- **Ke sensitivity:**
  - At Ke 2e-4 lx (the pollock half-point) vision stays saturated to 200 m by day. R_krill is
    0.62-0.68 m and R_fish 0.80-0.86 m; the ratio is 0.6 (A) / 0.32 (B).
  - At the salmonid 10 lx, both prey sit at the floors at every depth.
- **Larger saithe:** for 70 cm saithe the fish cap rises to 1.40 m. The saturated ratio falls to ~0.24
  (A); krill R is unchanged.
- **The x2 between A and B** comes from two things:
  - the [J] contrast ratio, which is the whole difference inside A;
  - B carrying a fish-prey size exponent (measured on fish 23-50 mm) down to a translucent
    crustacean.
  - Neither is measured for a gadoid. No contrast value was picked to close the gap.

**Effect on the gate** [D, crude odds arithmetic on the sibling note's V_one, not a re-run]:
- V_one is rule A with a contrast ratio of 1 and the krill floor applied to the fish.
- Rule A's contrast factor (x0.57 on krill R^2) moves the 150-250 m krill share from 0.69 to ~0.57.
  Rule B's (x0.32) moves it to ~0.42.
- With review p (krill 0.6, fish 0.45; V_one_p 0.548) the share becomes ~0.42 / ~0.28.
- Field Q3 is 0.47 (0.32-0.69), booked 0.36. The fish floor of 0.3 m lowers the krill share further at
  night, but it raises the juvenile-benthivore pair (V_one already 1.08-1.62x booked).

## Caveats

1. **No gadoid R on fish or krill exists.** The saturated R values and the prey-size scaling are
   salmonid analogues: shallow tanks, tethered or single prey, photopic lux. Gadoid evidence fixes only
   the light sensitivity (Ke bracket) and the dark feeding ratio, from juvenile pollock on Artemia.
2. **The light below 150 m is an extrapolation of an extrapolation.**
   - The sensor floor is at ~65 m and K_deep is judged.
   - The bracket is 2-3 decades at 200-250 m (table), which is wider than the Ke bracket's effect.
   - For the krill pair the relevant light is at the krill layer, ~100-160 m by day over 150-250 m beds
     (sibling note, section 262). It is not the light at the bed. Saithe at the bed meet krill only
     where the layer reaches them.
3. **The Aksnes-Utne form is violated in every behavioural dataset tested** (Rohan et al. 2021: the
   turbidity response is not monotonic, and a non-visual floor is needed). It is used here only as a
   consistent bookkeeping of light, size and contrast. The cap does most of the work.
4. **Contrast values are judgement.** The view geometry differs between prey: fish seen from below as
   opaque silhouettes, or horizontally as silvered and near-invisible; krill partly transparent and
   counter-illuminating. A direct measurement of M. norvegica transparency was not found.
5. **Schools and swarms.** R to a herring school or a krill swarm is not the single-prey R. The library
   handles schools separately (section 220 school variant); krill swarms are not modelled. Large saithe
   under swarms ate fish (4d).
6. **Not fitted.** Every value comes from the sources or is judged. Nothing was chosen to hit the
   booked 0.36 or the field 0.47.

## Gaps

- Meager et al. 2010 and 2005 full texts (juvenile cod R in cm against light; mysid prey), closed with
  no repository copy.
- Any gadoid or hake R on krill or on fish; any fish-on-euphausiid capture probability.
- Hake and spurdog diets by length in the northern North Sea or Trench; a length-resolved
  Norwegian-coast saithe diet (IMR 1987 report).
- Onsrud et al. 2005 (in-situ gadoid search on krill, 10.1139/f05-090): the repository record has no
  file.
- M. norvegica optical transparency or contrast.

## References

- Aksnes DL, Giske J 1993. A theoretical model of aquatic visual feeding. Ecol Model 67:233-250. doi:10.1016/0304-3800(93)90007-F
- Aksnes DL, Utne ACW 1997. A revised model of visual range in fish. Sarsia 82:137-147. doi:10.1080/00364827.1997.10413647
- Anthony PD 1981. Visual contrast thresholds in the cod Gadus morhua L. J Fish Biol 19:87-103. doi:10.1111/j.1095-8649.1981.tb05814.x
- Bergstad OA 1991. Distribution and trophic ecology of some gadoid fish of the Norwegian Deep. Sarsia 75:269-313, 315-325. doi:10.1080/00364827.1991.10413455, 10.1080/00364827.1991.10413456
- Bergstad OA, Wik AD, Hildre O 2003. Predator-prey relationships and food sources of the Skagerrak deep-water fish assemblage. J Northw Atl Fish Sci 31:165-180. doi:10.2960/J.v31.a12
- Du Buit MH 1991. Food and feeding of saithe (Pollachius virens L.) off Scotland. Fish Res 12:307-323. doi:10.1016/0165-7836(91)90015-8 (not read)
- Ellis T, Gibson RN 1997. Predation of 0-group flatfishes by 0-group cod: handling times and size selection. MEPS 149:83-90. doi:10.3354/meps149083
- Fraser NHC, Metcalfe NB 1997. The costs of becoming nocturnal: feeding efficiency in relation to light intensity in juvenile Atlantic salmon. Funct Ecol 11:385-391. doi:10.1046/j.1365-2435.1997.00098.x
- Fregin T, Wiese K 2002. The photophores of Meganyctiphanes norvegica (M. Sars) (Euphausiacea): mode of operation. Helgol Mar Res 56:112-124. doi:10.1007/s10152-002-0104-4
- Fujii T 2016. Potential influence of offshore oil and gas platforms on the feeding ecology of fish assemblages in the North Sea. MEPS 542:167-186. doi:10.3354/meps11534 (via sibling note)
- Hansen AG 2014. Persisting in the pelagic: environmental, behavioral, and morphological controls on predator-prey interactions. PhD thesis, Univ. Washington.
- Hansen AG, Beauchamp DA, Schoen ER 2013. Visual prey detection responses of piscivorous trout and salmon: effects of light, turbidity, and prey size. Trans Am Fish Soc 142:854-867. doi:10.1080/00028487.2013.785978
- ICES CRR 164 (Daan 1989) and CRR 219 (1997): North Sea stomach sampling 1981 and 1991 (via sibling notes).
- Juanes F, Buckel JA, Scharf FS 2002. Feeding ecology of piscivorous fishes. In: Handbook of Fish Biology and Fisheries 1:267-283. doi:10.1002/9780470693803.ch12
- Kaartvedt S, Rostad A, Fiksen O, Melle W, Torgersen T, Breien MT, Klevjer TA 2005. Piscivorous fish patrol krill swarms. MEPS 299:1-5. doi:10.3354/meps299001
- Langbehn TJ, Aksnes DL, Kaartvedt S, Fiksen O, Jorgensen C 2019. Light comfort zone in a mesopelagic fish emerges from adaptive behaviour along a latitudinal gradient. MEPS 623:161-174. doi:10.3354/meps13024
- Lokkeborg S, Ferno A 1999. Diel activity pattern and food search behaviour in cod. Env Biol Fish 54:345-353. doi:10.1023/A:1007504712163
- Mahe K, Amara R, Bryckaert T, Kacher M, Brylinski JM 2007. Ontogenetic and spatial variation in the diet of hake in the Bay of Biscay and the Celtic Sea. ICES J Mar Sci 64:1210-1219. doi:10.1093/icesjms/fsm100
- Mazur MM, Beauchamp DA 2003. A comparison of visual prey detection among species of piscivorous salmonids: effects of light and low turbidities. Env Biol Fish 67:397-405. doi:10.1023/A:1025807711512
- Meager JJ, Solbakken T, Utne-Palm AC, Oen T 2005. Effects of turbidity on the reactive distance, search time, and foraging success of juvenile Atlantic cod. CJFAS 62:1978-1984. doi:10.1139/f05-104
- Meager JJ, Batty RS 2007. Effects of turbidity on the spontaneous and prey-searching activity of juvenile Atlantic cod. Phil Trans R Soc B 362:2123-2130. doi:10.1098/rstb.2007.2104
- Meager JJ, Moberg O, Strand E, Utne-Palm AC 2010. Effects of light intensity on visual prey detection by juvenile Atlantic cod. Mar Freshw Behav Physiol 43:99-108. doi:10.1080/10236241003798910
- Onsrud MSR, Kaartvedt S, Rostad A, Klevjer TA 2004. Vertical distribution and feeding patterns in fish foraging on the krill Meganyctiphanes norvegica. ICES J Mar Sci 61:1278-1290. doi:10.1016/j.icesjms.2004.09.005
- Palmer LM, Deffenbaugh M, Mensinger AF 2005. Sensitivity of the anterior lateral line to natural stimuli in the oyster toadfish. J Exp Biol 208:3441-3450. doi:10.1242/jeb.01766
- Rohan SK, Beauchamp DA, Essington TE, Hansen AG 2021. Merging empirical and mechanistic approaches to modeling aquatic visual foraging using a generalizable visual reaction distance model. Ecol Model 457:109688. doi:10.1016/j.ecolmodel.2021.109688
- Ryer CH, Olla BL 1999. Light-induced changes in the prey consumption and behavior of two juvenile planktivorous fish. MEPS 181:41-51. doi:10.3354/meps181041
- Schwalbe MAB, Bassett DK, Webb JF 2012. Feeding in the dark: lateral-line-mediated prey detection in the peacock cichlid Aulonocara stuartgranti. J Exp Biol 215:2060-2071. doi:10.1242/jeb.065920
- Staniland IJ 1995. PhD thesis, Univ. Leicester (whiting feeding behaviour; via sibling note).
- Turesson H, Bronmark C 2004. Foraging behaviour and capture success in perch, pikeperch and pike and the effects of prey density. J Fish Biol 65:363-375. doi:10.1111/j.0022-1112.2004.00455.x
- Utne-Palm AC, Breen M, Lokkeborg S, Humborstad OB 2018. Behavioural responses of krill and cod to artificial light in laboratory experiments. PLoS ONE 13:e0190918. doi:10.1371/journal.pone.0190918
- Vogel JL, Beauchamp DA 1999. Effects of light, prey size, and turbidity on reaction distances of lake trout to salmonid prey. CJFAS 56:1293-1297. doi:10.1139/f99-071
- Vollset KW, Folkvord A, Browman HI 2011. Foraging behaviour of larval cod (Gadus morhua) at low light intensities. Mar Biol 158:1125-1133. doi:10.1007/s00227-011-1635-5
