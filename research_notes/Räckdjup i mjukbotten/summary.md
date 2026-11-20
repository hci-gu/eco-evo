# Reach depth into soft sediment and the infauna's exposed share phi

Research note, 2026-11-19. Written by the research subagent; saved by the main agent. Research only: no library, manifest, code or test change. Scripts are in `scripts/`, tables and evidence logs in `tables/`, downloads in `data/` (one subdirectory each). Tags: F = full text read, A = abstract only, S = secondary, D = database extract, J = judgement.

## 0. Headline

1. **No one has measured how deep haddock, cod, plaice, dab, long rough dab or witch reach into mud.** About 25 new targeted searches found nothing: feeding pits, excavation, buried-prey experiments, sidescan, fish bioturbation trait lists, flatfish feeding mechanics. Every reach is still a judgement. What is new are bounds and ordinal facts:
   - Dab leave the Amphiura disc and take the arms (F). Haddock do the same (S, Sköld et al. 1994 citing Mattson 1992).
   - Haddock: tube worms withdraw "below the feeding depths" (A), and feeding depth rises with fish size (A).
   - Plaice feed head-down by suction on whole buried shallow prey (F).
   - Dab take only the chelae of burrowing crabs (F), and buried fauna is "less likely to be ingested" (A).
   - Witch eat surface and shallow polychaete families, Abra and Nuculana (F diet, J classification).
   - Cod: "food covered by sand was not found" (A), and cod in the box eat almost no infauna (D).
2. **Recomputed phi, purely geometric** (`scripts/phi_reach.py`): biomass above each member's reach, plus parts exposed above the surface. The 100-300 m blend is weighted by the library's K over `maps/depth.npz`.

   | Weighting | low | central | high |
   |---|---|---|---|
   | Biomass-weighted, haddock and dab arms only (Sköld bound) | 0.07 | **0.17** | 0.29 |
   | Biomass-weighted, discs taken above the reach | 0.07 | 0.17 | 0.40 |
   | Infauna-consumption-weighted (sensitivity) | 0.10 | 0.22 | 0.37 / 0.54 |

3. **Recommendation: keep phi = 0.20, range 0.07-0.29.** The upper end becomes 0.40 if haddock take discs.
   - The new central 0.17 differs from 0.20 only through judged reach depths, so it is not a measured correction. Its position in the range cannot be narrowed.
   - The swing input is haddock's reach (39 % of the biomass): 1 to 4 cm moves phi from 0.12 to 0.26, or to 0.205 under the arms-only bound.
   - Group phi 0.20 corresponds to a haddock reach of about 3.2 cm, with the other members at central.

## 1. Sources the earlier notes already used (not redone)

Dauwe et al. 1998; Sköld, Loo & Rosenberg 1994; Sköld & Gunnarsson 1996; Duineveld & Van Noort 1986; Loo et al. 1996; Zwarts & Wanink 1993; Zwarts et al. 1994; Grémare et al. 2004; Zanzerl & Dufour 2017; Rosenberg et al. 1997, 2002; Kaiser & Ramsay 1997; Brawn 1969; Mattson 1990, 1992 (abstracts); Solan & Kennedy 2002 (withdrawal only); de Groot 1969; Josefson 1990, 1995; Beukema & de Vlas 1979; Hiddink 2016/2017; Mackinson & Daskalov 2007; Heath 2005; MarLIN and SeaLifeBase; Buchanan 1964 and Muus 1981 (S).

New in this note:
- Schückel et al. 2010: the sentence on spionids withdrawing below the feeding depth (A).
- Hinz, Kröncke & Ehrich 2005 (A).
- Johnson et al. 2015 (F; the file sits in "Fiskgrupper i rutan").
- Bergstad, Wik & Hildre 2003, witch prey families (F).
- Jensen c. 1960, US Fish & Wildlife Service Fishery Leaflet 489 (a popular leaflet).
- Gonzalez-Iglesias et al. 2003 (A).
- Flach & Heip 1996 (F, figure digitised).
- Solan & Kennedy 2002, burrow and disc depths (F).
- Dobson et al. 1991, arm and disc mass split (F, analogue species).
- Fischer et al. 2025 (press release only).

## 2. Reach per member (`tables/reach_evidence.txt`)

| Member (biomass share) | Reach low / central / high, cm | Basis | Tag |
|---|---|---|---|
| Haddock (0.386) | 1 / 2 / 4 | Crops arms and does "not generally consume the entire brittle star" (Sköld et al. 1994 citing Mattson 1992). Spionids and Phoronis withdraw "below the feeding depths" (Schückel et al. 2010). "The prey size and the feeding depth in the sediment increased with fish size" (Mattson 1992). "Grub out of the bottom with their muscular lips" (Jensen; anecdote, no number). | J on S/A |
| Cod (0.314) | 0 / 0.5 / 1 | "Food covered by sand was not found", although groups dig in gravel (Brawn 1969). Feeds "on and below the bottom surface" by suction (Mattson 1990). Box diet: infauna 10 %, mostly the sea mouse Aphrodita (ICES DB). | J on A/D |
| Long rough dab (0.126) | 0 / 0.5 / 1 | Groups with the turbot-like visual hunters of mobile prey (de Groot 1969). Prey are epifaunal ophiuroids and crustaceans (ICES DB). A daytime feeder (Gonzalez-Iglesias et al. 2003). | J |
| Dab (0.045) | 0 / 0.5 / 1 | Arms, not discs, in undisturbed sediment; discs only after trawling (Kaiser & Ramsay 1997). Burrowing crabs "as whole chelae only" (Johnson et al. 2015). "Buried living fauna was less likely to be ingested" (Hinz et al. 2005). | J on F bound |
| Plaice (0.038) | 1 / 2 / 4 | "Well-developed suction capability, horizontal mouth/head down foraging position". Prefers Abra alba, Glycera and Nephtys and forages "for buried food items" (Johnson et al. 2015). Crops siphons and tails (de Vlas, S). | J on F |
| Witch (0.017) | 0.5 / 1 / 2 | Skagerrak prey families (156-490 m) are mostly surface and shallow: Polynoidae, Sigalionidae, Cirratulidae, Ampharetidae, Opheliidae, Scrobiculariidae, Nuculanidae. Capitellids are the one deep family (Bergstad et al. 2003). | J on F |
| Others (0.074) | group mean | Not searched; renormalised over the six named members. | J |

**Stomach taxa against living depth** (`tables/stomach_taxa_living_depth.csv`):
- In Skagerrak plaice (ICES DB, 2024), deep-living families are about 14 % of the identified family mass: Priapulida, Maldanidae, Capitellidae and Scalibregmatidae, 3.35 of 23.4 g. That is 6 % if Scalibregmatidae are counted as shallow.
- Dauwe's SK station has 49 % of the biomass below 5 cm.
- So plaice take the shallow fraction out of proportion to the stock. Deep taxa in stomachs may be cropped parts, so this bounds the reach from above only loosely.

## 3. Vertical distribution of biomass (`tables/vertical_evidence.txt`, `tables/vertical_rows.csv`)

**New sources:**
- **Flach & Heip 1996, MEPS 141:55** (Goban Spur; figure digitised ±3 %; stacking order inferred, so weaker than a clean F).
  - At 185 m: 0-1 cm 31 %, 1-5 cm 35 %, 5-10 cm 33 %, 10-15 cm 1 %.
  - At 208 m, fine sand: 32 / 11 / 9 / 48 %.
  - Most individuals (40-80 %) sit in the top 1 cm.
- **Solan & Kennedy 2002, MEPS 228:179** (Galway Bay, 22 m): A. filiformis burrows reach a maximum depth of 8.95 ± 1.6 cm (n = 8). Disc chamber at 9.15 cm (n = 1). Arms extend more than 20 mm.

No new sliced-core biomass profile exists for Skagerrak or Kattegat mud. Rosenberg, Hellman & Lundberg 1996 and Rosenberg et al. 2000 (the Alkor Deep, not the Gullmarsfjord) were not obtained.

**Synthesis for 70-300 m Skagerrak-like mud (J):**

| Layer | Low | Central | High | Note |
|---|---|---|---|---|
| 0-2 cm | 0.27 | 0.27 | 0.42 | High is a ceiling |
| 0-5 cm | 0.43 | 0.51 | 0.66 | |
| 5-10 cm | 0.09 | 0.30 | 0.33 | |
| Below 10 cm | 0.01 | 0.19 | 0.48 | |

**Inputs used in the calculation:**
- **300 m, and the polychaetes at 100 m:** the Dauwe SK profile. Cumulative share is 0.27 at 2 cm, 0.51 at 5 cm, 0.81 at 10 cm and 1.0 at 20 cm, linear within each layer (J). The low and high profiles shift the top layers by ±5 pp.
- **100 m composition:** SK3 standing stock without heart urchins (D).
  - Amphiura 65.4 %, polychaetes 16.6 %, bivalves 12.2 %, priapulids and nemerteans 4.0 %, crustaceans 1.0 %, gastropods 0.8 %.
  - The bivalves are 86 % Abra nitida (0-2 cm, F). The rest is split evenly between thyasirids (1.2-6.4 cm) and Nucula (5-10 cm).
  - The polychaetes are mostly Scalibregma inflatum, "buried deep" (MarLIN, S).
  - This replaces the earlier note's judged weights of 50/30/15/5.
- **Amphiura disc depth:** uniform 3-6 cm central (Rosenberg et al. 1997, F); 4-9 cm deep scenario (Solan & Kennedy 2002, F); 2-5 cm shallow scenario (Gullmarsfjord, A).
- **Grid weighting:** K-weighted over 3600 cells (median depth 143 m), the 100 m composition carries 0.74 of the weight.
- **Top-heavy sensitivity:** if 0-1 cm holds 0.20 of the biomass and 0.75 of Abra, the central phi goes from 0.166 to 0.178 (+0.012).

## 4. Croppable parts

- **Amphiura arms:** 0.85-0.90 of the body's organic mass. Measured on the amphiurid Microphiopholis gracillima (Dobson et al. 1991, F) and transferred to A. filiformis (J). The earlier note used 40-70 %.
- **Exposed arm tips while feeding:** 0.03-0.09 of the body, central 0.05 (J; 2 arms out 10-20 mm). This answers the exposure note's flag: it is consistent with the 6 % inside phi.
- **Arms above a buried disc:** raised arm segments inside the bite depth, about 0.031 of the body per cm of reach (J).
- **Cropped piece size:** about 2 mg wet weight per arm piece (Duineveld & Van Noort, S).
- **Bivalve siphons:** 0.5-2 % of body mass (Zwarts et al. 1994, A).
- **Polychaete palps and tails:** 0-3 % (J).

## 5. Combined phi (`tables/phi_reach_stdout.txt`, `tables/phi_rows.csv`)

Grid phi by reach depth, central profile (low / high profiles in brackets):

| Reach | phi |
|---|---|
| 0.5 cm | 0.093 (0.072-0.127) |
| 1 cm | 0.145 (0.120-0.184) |
| 2 cm | 0.251 (0.217-0.298) |
| 3 cm | 0.298 (0.264-0.476) |
| 4 cm | 0.479 (0.311-0.645) |

The jump between 3 and 5 cm is the Amphiura disc layer at 100-150 m.

Per member, at its own reach (central profile, low / central / high reach):

| Member | phi |
|---|---|
| Haddock | 0.15 / 0.25 / 0.48 (discs taken) |
| Plaice | 0.15 / 0.25 / 0.48 |
| Cod, long rough dab, dab | 0.04 / 0.09 / 0.15 |
| Witch | 0.09 / 0.15 / 0.25 |

Group values are given in section 0. Applying the earlier note's uniform reach (1 / 2 / 5 cm) to today's profiles gives 0.12 / 0.25 / 0.80. The geometric definition raises the value of each cm of reach, and the member-specific shallow reach of cod, long rough dab and dab pulls it back down.

## 6. Judgements and conflicts

1. **phi stays purely geometric.** Siphon activity, cue detection and capture already sit in the exposure factor e = 0.165. Counting them in phi as well would count them twice.
2. **Weighting by NS-IBTS biomass puts 70 % on cod and haddock.** Cod barely eats infauna, which lowers the biomass-weighted phi. The consumption-weighted variant, at 0.22, shows how large that effect is.
3. **Haddock's high reach is capped at the disc (4 cm, was 5 cm in the earlier note).** The arms-only branch also removes discs that lie shallower than 4 cm. Its tag is S, because Mattson 1992 itself was not read.
4. **Brawn 1969 points both ways.** Cod did not find food covered by sand, but groups dug food out of gravel. I take mud as no digging.
5. **The Jensen leaflet is anecdote and sets no number.**
6. **The Amphiura disc depth conflicts between sources:** about 4 cm in the Kattegat-Skagerrak (Rosenberg) against 9 cm in Galway Bay (Solan & Kennedy). Rosenberg is the central case; Galway is the deep scenario.
7. **Seasonal burial changes at 100-300 m are unknown.**

**What would cut the range:** whether haddock in the box or the Skagerrak eat Amphiura discs or only arm pieces. Sources: Mattson 1992 in full, or species-level counts in the ICES stomach database or the SLU and Danish stomach programmes. If haddock take only arms, phi lies in 0.15-0.21. If they take discs, it is 0.26-0.30 at a 4 cm reach.

## 7. Not obtained

- Any measured bite, dig or pit depth for the six members.
- Full texts of:
  - Mattson 1990 and 1992;
  - Schückel et al. 2010 and 2012;
  - Hinz et al. 2005, and Hinz's PhD thesis (blocked);
  - Hill, Burrows & Hughes 2000;
  - Pihl 1994;
  - Fischer et al. 2025;
  - Rosenberg, Hellman & Lundberg 1996, and Rosenberg et al. 2000;
  - Nilsson & Sköld 1996, Munday 1993, Bowmer & Keegan 1983, O'Connor et al. 1986.
- de Groot 1971 and 1973: the server's TLS certificate did not verify, so not downloaded.
- A Skagerrak 0-1 cm biomass slice.
- The A. filiformis arm/disc split itself (an amphiurid analogue was used).
- The living depths of Scalibregma inflatum and Amphilepis norvegica.
