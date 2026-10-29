# Spurdog (Squalus acanthias) and anglerfish (Lophius piscatorius) on juvenile gadoids and flatfish

Same question as `evidence_saithe.md`: share of the predator's food (by mass) that is the model's
`benthivorous_fish` group (haddock, cod, long rough dab, dab, plaice, witch, other flatfish, starry ray).
Whiting, Norway pout, poor cod, bib, rocklings, blue whiting are NOT benthivores in the model.
Marks: [S] from the source (table/page), [J] my derivation or judgement.

In the water-column predator group (`predatory_fish`, member biomass shares from
`../Rovfiskens uppdelning/params_split_223.md`: saithe 0.651, mackerel 0.128, spurdog 0.080, horse mackerel
0.055, hake 0.040, anglerfish 0.022, ling 0.011, velvet belly 0.008, pollack 0.007) spurdog and anglerfish
are small, so their diets move the group number little (section 3).

---

## 1. Spurdog

### 1.1 Ecopath North Sea 1991 (Mackinson & Daskalov 2007)

- Citation: Mackinson, S., Daskalov, G. 2007. Cefas Sci. Ser. Tech. Rep. 142; EcoBase model 457
  (`../Fiskgrupper i rutan/benthivores/data/ecobase/model_457.xml`), read with
  `scripts/ecopath1991_benthivore_prey.py`, output `data/ecopath1991_benthivore_prey_stdout.txt`. Source
  of the elasmobranch diet rows inside the Ecopath report not checked here (the earlier file attributes
  them to Ellis et al. 1996 and the 1991 stomachs).
- Spurdog diet fractions [S]: herring adult 0.187, other gadoids (small) 0.132, whiting adult 0.119,
  Norway pout 0.099, mackerel 0.082, plaice 0.0475, sprat 0.041, sandeels 0.039, gelatinous zooplankton
  0.038, dragonets 0.027, squid 0.023, blue whiting 0.022, spurdog 0.021, shrimp 0.020, horse mackerel
  0.019, ..., dab 0.0068, witch 0.0068, haddock adult 0.0061, haddock juvenile 0.0003. Cod: 0.
- Benthivores (plaice + dab + witch + haddock) = **6.7 %** [J]. Mostly plaice (4.8 %); haddock 0.6 %.
- Correction [J]: the earlier file's spurdog "Bf 0.22-0.24" counted "Other gadoids (small)" (0.13; poor
  cod, bib, rocklings, silvery pout). On the model's benthivore definition the Ecopath number is ~6-7 %.
- Weight: medium (balanced model; the only North Sea spurdog diet by mass I could reach).

### 1.2 Ellis, Pawson & Shackley 1996 (via FishBase)

- Citation: Ellis, J.R., Pawson, M.G., Shackley, S.E. 1996. The comparative feeding ecology of six species of
  shark and four species of ray (Elasmobranchii) in the north-east Atlantic. J. Mar. Biol. Ass. U.K.
  76:89-106. doi:10.1017/S0025315400029039. Full text not accessed (closed, Unpaywall: no OA copy).
- FishBase v25.04 diet record 222 ("Northeastern Atlantic, 1981-1985") [S,
  `data/fishbase/fishbase_v25.04_diets_saithe_spurdog_anglerfish.txt`]: Gadiformes 82.3 %, *Pagurus* 8.7,
  benthic invertebrates 5.7, cephalopods 2.0, amphipods 1.0, annelids 0.3. Method and sample size not in
  the record. Gadiformes is not split (whiting, Norway pout, poor cod and haddock all fall in it).
- Weight: low for this question (no split, method unknown).

### 1.3 ICES stomach data (earlier file)

- `../Fiskgrupper i rutan/predators/evidence_C_diet_consumption.md`, sect. 2.1 [S]: central North Sea 1991,
  7 pooled records: P 50.7, Bf 24.8, W 23.8 % (Bf there = cod, haddock, poor cod - split not given);
  Skagerrak August 2024, 7 fish, 2 g of prey (unidentified fish, *Pasiphaea*).
- Weight: very low (n 7).

### 1.4 Other spurdog sources

- Holden, M.J. 1966. The food of the spurdog, Squalus acanthias (L.). J. Cons. int. Explor. Mer
  30:255-266. doi:10.1093/icesjms/30.2.255. Not accessed (closed). Abstract content via search: 1080
  stomachs from British waters, 69.1 % empty, sandeels the most common prey, herring in ~11.5 % of stomachs
  with food; by occurrence, not weight [S, abstract as reported by the search tool - not verified against
  the PDF].
- IMR 2021 spurdog climate narrative (`data/imr2021_spurdog_narrative.pdf`, p. 1): "Adult spurdog mostly
  feed on small schooling fish (herring, mackerel), but also molluscs, crustaceans, jellyfish, and solitary
  fish as flatfish and haddock (Burgess, 2002)" [S].
- Klocker et al. 2024, Movement Ecology 12:62 (Norwegian fjords, on disk in
  `../Fiskgrupper i rutan/predators/data/diet/lit_gadoids_elasmo/spurdog2024_movecol.pdf`): prey in Irish and
  Norwegian spurdog stomachs include pearlside, herring, whiting, Norway pout and sprat ("CJ, unpublished
  data") [S, Discussion]. No haddock/cod/flatfish named, no shares.
- NW Atlantic, Bowman et al. 2000 via FishBase records 3347-3350 (1977-80) [S]: Gulf of Maine n 417
  sandlance 40.9 %, squid 13.9 %; Georges Bank n 783 red hake (*Urophycis chuss*) 49.9 %; Scotian Shelf n 238
  mackerel 30.6 %. No cod, haddock or flatfish in the transcribed rows.
- Cortes 1999 (ICES J. Mar. Sci. 56:707), standardized diet over studies, n 19 259 [S, FishBase record 221]:
  teleosts 43.7 %, decapods/benthic crustaceans 35.2 %, cephalopods 5.2 %. No gadoid split.

### 1.5 Spurdog bottom line [J]

- Benthivore share **~6 % (range 2-15 %)**: Ecopath 1991 6.7 % (mostly plaice); flatfish and haddock named
  as occasional prey; the dominant prey everywhere is pelagic schooling fish, sandeel, whiting/Norway pout
  and small gadoids. The 15 % upper end allows for the unsplit "Gadiformes"/"Bf" rows of Ellis 1996 and
  ICES 1991 containing some haddock. No density-dependence evidence found.

---

## 2. Anglerfish

### 2.1 Ecopath North Sea 1991 (monkfish group)

- Same source as 1.1 [S]: sandeels 0.170, Norway pout 0.132, herring 0.119, whiting adult 0.117, squid
  0.091, mackerel 0.068, **haddock adult 0.0659, long rough dab 0.0452**, small demersal fish 0.0415,
  **dab 0.0302, cod adult 0.0115, cod juvenile 0.0115, lemon sole 0.0113**, ..., **juvenile rays 0.0076,
  plaice 0.0037, witch 0.0037, haddock juvenile 0.0028**.
- Benthivores = haddock 6.9 + cod 2.3 + flatfish 9.4 (long rough dab, dab, lemon sole, plaice, witch) +
  juvenile rays 0.8 = **19.3 %** [J].
- Weight: medium.

### 2.2 Laurenson & Priede 2005 - Shetland (via FishBase)

- Citation: Laurenson, C.H., Priede, I.G. 2005. The diet and trophic ecology of anglerfish Lophius
  piscatorius at the Shetland Islands, UK. J. Mar. Biol. Ass. U.K. 85:419-424. Full text not accessed.
  Abstract (Aberdeen research portal): samples from sandeel trawling and from demersal whitefish trawling;
  "the main prey was Norway pout (Trisopterus esmarkii) and lesser sandeel (Ammodytes marinus)" [S].
- FishBase v25.04 records 6852-6856 (Q2-Q3 1999, Q1-Q2 2000, commercial whitefish trawlers) [S]:
  all sizes: Norway pout 53.0, lesser sandeel 16.4, dragonet 6.5, **dab 6.4**, teleosts 4.6, *Hyperoplus*
  3.7, herring 3.6, argentines 3.0, **Pleuronectiformes 1.5**, cephalopods 1.3. Size rows: 30-40 cm witch
  30.1 + Norway pout 69.9; 40-50 cm witch 27.2 + whiting 70.4; 50-60 cm dab 11.0 + whiting 84.5; > 60 cm
  **cod 93.5** + cephalopods 6.5.
- Benthivores, all sizes: flatfish 7.9 %, haddock/cod 0 [J]. The size rows hold only 2-4 items each and
  look like a handful of stomachs (n not in the record); the > 60 cm "cod 93.5 %" is likely one or two
  large prey and should not be used as a share [J]. Method (weight vs other) not stated in the record.
- Weight: medium for the pooled row, low for the size rows.

### 2.3 Crozier 1985 - northern Irish Sea (via FishBase)

- Citation: Crozier, W.W. 1985. Observations on the food and feeding of the angler-fish, Lophius
  piscatorius L., in the northern Irish Sea. J. Fish Biol. 27:655-665. Not accessed; FishBase records
  5893-5895, May 1983-April 1984 [S]:
  9-25 cm (n 86): fish unid. 76.3, *Lumpenus* 7.0, gadoid 5.8, Norway pout 4.2, herring 3.3 - no
  benthivores; 26-60 cm (n 1457): fish unid. 44.9, Norway pout 14.8, gadoid unid. 8.3, *Nephrops* 8.1,
  *Lumpenus* 8.0, whiting 5.5, herring 4.7, poor cod 4.3 - no cod/haddock/flatfish identified;
  61-129 cm (n 78): herring 42.0, *Nephrops* 20.7, whiting 10.3, Norway pout 7.2, **cod 6.3**, bib 4.4, poor
  cod 3.5, *Lumpenus* 2.2, **dab 1.9**.
- Benthivores: 0 % (identified) at < 60 cm, 8.2 % at > 60 cm [J]. Large unidentified fish shares hide
  some.
- Weight: low-medium (Irish Sea, old).

### 2.4 Icelandic MFRI 2024 (1996-2023)

- Source as in `evidence_saithe.md` entry 9, excerpt `data/excerpts/hv2024_01_iceland_pp63-65_138-139.pdf`.
  Anglerfish (skotuselur): haddock ~4.7 % of diet weight (Fig. 159c, p. 138); cod ~16 % (Fig. 160c, p. 139)
  [S figure; values J, read from the bars]. Text: "Of other species that eat cod, mainly anglerfish, ling,
  ... cod was 1-16 % of the diet weight of these species" [S, p. 139, my translation].
- Benthivores (haddock + cod only; flatfish not shown) ~21 % [J].
- Weight: medium (independent, post-2000, large program), Icelandic prey field.

### 2.5 Other anglerfish sources

- ICES stomach data 2022-24, Skagerrak + northern North Sea, 9 fish: Bf 0-92 % between samples [S, earlier
  file `../Fiskgrupper i rutan/predators/evidence_C_diet_consumption.md` sect. 2.1]. Very low weight.
- Pierre et al. 2017, Food Webs 13:33-37 (Celtic Sea, `data/pierre2017_anglerfish_celtic.pdf`):
  "demersal and pelagic fishes such as Gadiformes, Trachurus trachurus and Scomber scombrus accounted for
  more than 60% of the diet of large L. piscatorius" [S, Results]; occurrence-based, Gadiformes not split.
- Thangstad et al. 2002 (Nordic anglerfish report, on disk `imr2002_anglerfish_nordic.pdf`, sect. 2.7):
  "Gadoid fish were likewise the main food for L. piscatorius from the Irish Sea (Crozier 1985) and
  Norwegian waters (Woll et al. 1995a)"; "Stomach contents reflect spatial and/or temporal prey
  availability or abundance (Crozier 1985)" [S].

### 2.6 Anglerfish bottom line [J]

- Benthivore share **~15-20 % (range 8-25 %)**: Ecopath 1991 19 %, Iceland ~21 % (haddock + cod), Shetland
  8 % (flatfish), Irish Sea > 60 cm 8 %. Anglerfish is the one member for which haddock, cod and flatfish
  are a substantial part of the diet. It is 2.2 % of the group's biomass and has the lowest Q/B (1.8).
  Density dependence: only the qualitative "contents reflect prey availability" (Crozier 1985 via
  Thangstad et al. 2002).

---

## 3. Group-level check [J]

Consumption-weighted benthivore share of `predatory_fish` with member biomass shares above, Q/B from the
earlier file (saithe 3.0, mackerel 2.2, spurdog 2.0, horse mackerel 3.6, hake 3.5, anglerfish 1.8; ling,
velvet belly, pollack assumed 2-3), and benthivore shares saithe 0.05, spurdog 0.067, anglerfish 0.19,
mackerel 0.005, horse mackerel 0.01, hake 0.01, ling 0.19 (assumed like anglerfish, no data), velvet belly
0, pollack 0.05:

- central **4.4 %** of group consumption; 2.4 % with saithe at 0.02; 7.2 % with saithe at 0.09.
- Saithe supplies 0.78 of the group's benthivore predation in the central case (0.58-0.86).

This brackets the model's booked 6.1 % and is well below the 15-23 % the trained model realises.

## 4. Weak numbers

- No spurdog or anglerfish diet by mass from the Skagerrak or the Norwegian Trench (Skagerrak spurdog n 7
  in 2024, anglerfish n 9 in 2022-24, both too thin).
- Spurdog rests on one balanced Ecopath vector; the primary Ellis et al. 1996 and Holden 1966 texts were
  not accessed.
- Anglerfish values are FishBase transcriptions (Laurenson & Priede, Crozier) and bar-chart readings
  (Iceland); methods (weight vs volume) unverified.
