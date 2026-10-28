# Gastric evacuation and the energy ceiling: does fat herring fill a benthivore faster than bed prey?

Written by the research subagent, saved by the main agent (2026-10-28, after resume 223.28). Research
only: no library, code, resume or AGENTS.md edit. Scripts in `scripts/` (run with `python3 -I`), printed
output in `tables/`, downloads and their URLs in `data/` (`data/README.md`). Tags: [A] measured/
experimental value, [F] full text read, abs = abstract only, [S] secondary (cited by another paper or a
search extract), [D] data analysed here, J = judgement.

## Headline

1. **The energy-density exponent is not one number; two schools disagree.** Write gastric evacuation of
   wet mass as ~E^-b, so energy leaves the stomach as ~E^(1-b). b = 1 is the library's energy ceiling
   (h = E/Cmax_E); b = 0 is a mass ceiling.
   - DTU (Andersen) gadoid models: cod b 0.85 (Andersen 2012) [F via Andersen et al. 2017 suppl. eq.
     S2]; whiting and other gadoids b 0.875 (Andersen pers. comm. in Velasco & Olaso 2000) [F]; whiting
     b ~1.0 (Andersen 1999: evacuation time x1.6-1.7 for E 3.4 -> 5.6 kJ/g; abs; [D] b 0.94-1.06); plaice
     on formulated diets b 1 (Jobling 1980, abs).
   - Cod data from Tromso and Hamburg: b 0.16 among six whole-fish prey (dos Santos & Jobling 1995 Table
     3) [F, D]; b 0.37 among the Temming & Herrmann 2003 fish clusters [D from Funk et al. 2023 Table
     G1]; T&H's own fitted exponent reported as 0.45 [S]. Not independent: T&H re-analysed the dos Santos
     Barents data.
   - Even at the low end fatty fish give more energy per gut-hour: cod, dos Santos [D]: herring/haddock
     2.0x, capelin/long rough dab 2.0x, fatty/lean geometric means 1.8x; cod, T&H [D]: clupeids/flatfish
     1.6x, clupeids/(whiting + goby) 1.3x.
2. **Armour is the larger and better-measured effect.** At equal energy density shrimps evacuate x1.3
   slower in cod (T&H, Crangon + Pandalus) to x1.45 (dos Santos, Pandalus); in whiting Crangon "took
   almost twice as long" (Andersen 1999, abs). Crabs x1.9-2.1 slower (Liocarcinus, Chionoecetes; T&H
   formula fitted by Funk et al. 2023 to Andersen et al. 2016 data) [D]. Andersen et al. 2016 (abs): a
   shell-cracking stage of 0-33 h, and both stage rates halved as armour (chitin, ash) increases. Krill
   and mysids evacuate like fish (Andersen 1999 abs; dos Santos: krill faster than fish) [A].
3. **Soft polychaetes are fast; the rest of the infauna is unmeasured.** Arenicola evacuates ~2x faster
   than fish of equal E in cod (T&H rho 0.0141 vs a fish fit of 0.0073 at 2.5 kJ/g [D]; dos Santos alpha
   30 h vs 58-88 h for fish [F]), but Arenicola went "twice as rapidly as the Nereis used by Jones" (dos
   Santos & Jobling 1995 [F]), so errant polychaetes sit near lean fish. Brittle stars and bivalves:
   only the direction (ash and carbonate lengthen every stage; Couturier et al. 2013, abs). Andersen's
   model reportedly has a floor: no E effect below ~3.5 kJ/g [S].
4. **Model translation** (`tables/prey_ceilings_stdout.txt`): ceiling_j = 26.18 (E_j/4.184)^(1-b) / A_j
   MJ gross per t predator per tick; h_j = E_j / ceiling_j. Herring : infauna energy per tick at a full
   gut: 1.00 today (1.14 after assimilation); central (b 0.75, no armour on infauna) 1.29 gross, 1.47
   assimilated; range 1.0-1.9 gross (b 1.0-0.37); 1.5 in Andersen's branch with the floor; 2.3 gross /
   2.7 assimilated only at the dos Santos b 0.16. A mass ceiling (2.74) is outside every gadoid estimate.
5. **Recommendation (J): keep the energy ceiling as the form, add armour on the crustacean pairs, and
   optionally relax the exponent to b ~0.75 (0.4-1.0).** The armour factors (epifauna x1.75, Nephrops
   x2) change bed returns more than any defensible b. Confidence: moderate for armour; low for b (a
   judgement between datasets that disagree); low for infauna.

## Q1. Evacuation rate vs prey energy density, size and exoskeleton

All forms are square-root or close to it (rate ~ S^0.5 at a given stomach content).

| Source | Predator | Form (wet mass, g/h) | E term | Basis |
|---|---|---|---|---|
| Andersen 2012 (via Andersen et al. 2017 suppl. eq. S2) | cod | dS/dt = -rho_LTE L^1.30 e^0.083T E_t^-0.85 d sqrt(S) | b 0.85, on the chyme's E | [F] (`data/andersen2017_meps583/supp_page1.png`) |
| Andersen pers. comm. in Velasco & Olaso 2000 | whiting, "other gadoids" | dS/dt = -0.00132 L^1.44 e^0.078T E^-0.875 S^0.5 | b 0.875 | [F] (ICES CM 2000/Q:11); `../Fiskgrupper i rutan/params_benthivorous_fish.md` l.73 calls this a "Temming & Herrmann 2003 type model"; it is Andersen's |
| Andersen 1999 | whiting | power model with E | time x1.6-1.7 for E 3.4 -> 5.6 | abs; [D] b 0.94-1.06 |
| Andersen 2001 | whiting, saithe, cod | "negative power function of energy density of fish prey over ... 3.4 to 11.5 kJ g-1" | exponent not seen | abs |
| Floor | gadoids | no E effect below ~3.5 kJ/g (lean prey's low lipid, no intestinal feedback) | | [S] |
| Temming & Herrmann 2003 | cod | C = 24 rho W^0.305 e^0.11T S^0.5 (curvature 0.534) | prey-specific rho; "for fish prey a large part of the variation ... explained by differences in energy density" | abs [F via marineinfo]; rho [F] Funk et al. 2023 Table G1 |
| dos Santos & Jobling 1995 | cod | S_t = S_0 2^-(t / (alpha S_0^0.52 B^-0.26 e^-0.13T))^xi; restricted xi = 1 | prey-specific half-life alpha | [F] Tables 1, 3 |
| Jobling 1980 | plaice | volume model, formulated diets | "rate of evacuation (of energy) ... independent of energy level and the composition of the diet" -> b 1 | abs |
| dos Santos & Jobling 1988 | cod | minced herring +/- fish meal and oil | oil "reduced the percentage of the meal emptied", but "rates of energy throughput on the enriched herring diet were high"; "physical form ... more important than energy content" -> b < 1 | abs |
| Jobling 1987 | fish (review) | | "emptying pattern in fish ... rather less sensitive to changes in dietary composition than ... in mammals" | abs |
| Knutsen & Salvanes 1999 | juvenile cod | surface (digestion velocity) model | goby (3.7 kJ/g) d 2.6e-3 cm/h vs herring (7.9) 2.2e-3 (Salvanes 1995, another population) -> b ~0.2; "high lipid content is known to retard" | [F]; cross-study comparison J |

dos Santos & Jobling 1995, cod, restricted form, relative to capelin; mass rate ~ 1/alpha ([D]
`scripts/evac_exponent.py`, `tables/evac_exponent_stdout.txt`):

| Prey | E kJ/g | alpha h | mass rate | energy per gut-hour |
|---|---|---|---|---|
| Lugworm (Arenicola) | 2.5 | 30 | 1.93 | 0.88 |
| Krill | 4.0 | 41 | 1.41 | 1.03 |
| Capelin | 5.5 | 58 | 1.00 | 1.00 |
| Herring | 6.8 | 88 | 0.66 | 0.81 |
| Polar cod | 3.3 | 59 | 0.98 | 0.59 |
| Squid mantle | 4.4 | 78 | 0.74 | 0.59 |
| Redfish | 3.5 | 68 | 0.85 | 0.54 |
| Long rough dab | 3.0 | 62 | 0.94 | 0.51 |
| Pandalus | 4.9 | 103 | 0.56 | 0.50 |
| Haddock | 3.2 | 84 | 0.69 | 0.40 |

Fish only: b 0.16 (0.17 without the spiny redfish). Pandalus x1.45 slower than the fish fit at 4.9
kJ/g; krill x0.60. Item sizes differed (haddock 16-40 g, LRD 6-17 g) and the model has no prey-size
term, so size and bones are confounded with E (J).

Temming & Herrmann 2003 rho for cod (Funk et al. 2023 Table G1), relative to clupeids at equal stomach
mass:

| Cluster | E kJ/g | rho | mass rate | energy rate |
|---|---|---|---|---|
| Annelids (Arenicola) | 2.50 | 0.01414 | 2.79 | 1.06 |
| Clupeids (herring + sprat) | 6.58 | 0.00507 | 1.00 | 1.00 |
| Other fish (whiting + goby) | 4.23 | 0.00597 | 1.18 | 0.76 |
| Flatfish (LRD) | 3.00 | 0.00677 | 1.34 | 0.61 |
| Crangon + Pandalus | 4.58 | 0.00445 | 0.88 | 0.61 |
| Liocarcinus (fit to Andersen 2016) | 3.54 | 0.0034 | 0.67 | 0.36 |
| Chionoecetes (fit to Andersen 2016) | 4.53 | 0.0028 | 0.55 | 0.38 |
| Carcinus (Funk's regression) | 2.81 | 0.00381 | 0.75 | 0.32 |

Fish: b 0.37. Armour vs the fish fit at the same E: shrimps x1.30, Liocarcinus x1.87, Chionoecetes
x2.08; Arenicola x1.95 faster. Funk et al. (2023 discussion [S]): lean flatfish evacuate faster than
crustaceans and carry more energy, so a flatfish diet gives "significantly higher consumption, energy
intake and growth than a crab-based diet".

Other measured rates: Andersen 2012 basic rates (Andersen et al. 2017 Table S2 [F]): clupeids 2.43e-3,
sandeel and goby 2.24e-3, Saduria 1.22e-3, "like brown shrimp Crangon crangon (i.e. at c. half the rate
of clupeids)". Temming & Andersen 1992 [F]: curvature lowest for prawn (B 0.27) vs herring 0.43, capelin
0.47, "related with the delaying effect of a robust exoskeleton".

Why the schools differ (J): Andersen's cylinder model carries prey length and geometry and puts E on the
chyme; dos Santos and T&H fit one constant per prey with no size term, so their "E effect" also absorbs
size, bones, scales and skin, and their fatty prey (herring, capelin, sprat) are also the soft,
small-boned ones. Andersen's b is the cleaner E effect at fixed geometry; the Tromso/Hamburg contrasts
describe whole natural prey as eaten. Both agree fatty fish give more energy per gut-hour than lean fish:
1.06x at b 0.85 to 2x at b 0.16, herring vs whiting-type prey.

## Q2. Maximum consumption on prey of different energy density

No study measures the sustained ad-lib ration of the same gadoids on fatty fish and on crustaceans or
benthos, in energy units, under the same conditions. What exists:
- The cod Cmax functions are rations of fish prey: Jobling 1988 Cmax "established ... from data on
  satiation rations of capelin" (Andersen et al. 2017 [F]); Hansson et al. 1996 (FB4) mostly fish food,
  converted at 4.184 kJ/g in the benthivore note. So the library's 26.18 MJ/t/tick is an energy ceiling
  measured on fish of ~4.2-5.5 kJ/g, exact only there.
- dos Santos & Jobling 1995 Fig. 1 [F, figure reading, `data/dossantos1995/fig1_page215.png`]: cod fed to
  satiation once daily for 23-26 days ate ~10-55 g/d of prawn (4.4 C, 420-3510 g) vs ~5-48 g/d of
  capelin (3.3-3.7 C, 290-3790 g); not size-standardised; similar wet mass at 4.9 vs 5.5 kJ/g ~ similar
  energy (J). The single-meal model predicted only 0.41-0.50 of the observed prawn intake (Table 4);
  authors blame falling temperature and spawning. J: with repeated feeding the shell-cracking lag may
  overlap the previous meal, so the sustained armour cost may be below the single-meal factor.
- dos Santos, Burkow & Jobling 1993 (Aquaculture 110:173; closed; [S]): cod on whole herring and enriched
  herring paste grew fastest and stored liver lipid; lean prawn gave the best conversion. Fatty fish
  bring more energy in, not compensated away by eating less mass.
- Compensation: plaice evacuate energy at a constant rate whatever the diet energy (Jobling 1980, abs);
  cod on oil-enriched mince keep high energy throughput (dos Santos & Jobling 1988, abs), only partial
  compensation; on formulated diets cod "adjusted its feed intake to both protein and energy levels"
  ([S]).
- Single-meal bulk limit (Arnott & Pihl 2000 [F], prior note): 1-group cod satiation meals 2.8 % WW on
  crabs, 9.1 % on Crangon, 17.8 % on plaice; in energy ~0.1 / 0.34 / 0.7 kJ/g BW (J): a crab meal is
  limited by volume and handling long before the gut's energy rate binds.
- Field (Fall & Fiksen 2019 [F], citing Hussy et al.): North Sea cod eating more Norway pout grew faster,
  more sandeel slower; capelin has by far the highest "digestive quality" (energy per digestion time) of
  the Barents prey.

Bottom line: indirect but consistent. Energy intake at satiation is higher on fatty fish than on lean
fish or crustaceans, by less than the 2.7x a mass ceiling would give.

## Q3. Armour and the indigestible fraction; double counting

Not double counting (J): library energies are whole-body as eaten, shells included
(`../Infauna och rörlig epifauna/params_split.md` 1.5: "Wet weights of molluscs and echinoderms include
shells"; crabs 3.6, northern shrimp 4.7 kJ/g); the armour factors are measured at equal E against fish,
on wet mass of whole prey in all three datasets; assimilation (0.7 infauna, 0.8 the rest) is the
absorbed share of gross energy. Three separate terms: dilution (in E), time per gram (armour),
absorption (assimilation), as long as armour is applied relative to fish of the same E.

Armour factors, time per gram relative to soft fish of equal E:

| Prey type | Factor | Basis |
|---|---|---|
| Small shrimps (Crangon type) | 1.5 (1.3-2.0) | T&H 1.30; dos Santos Pandalus 1.45; Andersen 1999 whiting ~2 (abs) |
| Mysids, krill | 1.0 (0.6-1.0) | Andersen 1999 abs; dos Santos krill 0.60 |
| Northern shrimp | 1.45 (1.3-1.8) | dos Santos Pandalus [F] |
| Crabs, hermits, Munida | 2.0 (1.8-2.5) | T&H fit to Andersen 2016 data 1.87-2.08 [D]; plus a 0-33 h lag (abs) |
| Cephalopods | ~1.1 | squid mantle alpha 78 vs the fish fit (J) |
| Nephrops | 2.0 (1.5-2.5) | unmeasured; claws and carapace like crabs/lobster (J) |
| Epifauna composite | 1.75 (1.4-2.3) [D/J] | stock shares 0.65 crabs/hermits/Munida, 0.15 northern shrimp, 0.14 small shrimps/mysids, 0.06 cephalopods/small fish |
| Infauna | 1.0 (0.6-1.5) | Arenicola 0.5; Nereis ~1; ophiuroids/bivalves >1, unmeasured; cropped arm tips and siphons are soft (J) |

The prior satiation note's "armour h x1.7-3": the data support the lower half for shrimps (1.3-1.45 in
cod) and confirm ~2 for crabs; x3 needs the crab lag on top (J).

## Q4. Per-prey ceilings and h for the benthivore pairs

ceiling_j = 26.18 (E_j/4.184)^(1-b) / A_j MJ gross per t predator per tick (bottom layer ~7.4 C);
h_j = E_j / ceiling_j; b = 1, A = 1 reproduces the library (`scripts/prey_ceilings.py`).

Central b (J, `tables/prey_ceilings_stdout.txt` last block): haddock 0.39 and others 0.10 at Andersen's
0.875, flatfish ~0.20 at 1.0, cod 0.31 at its own data -> b 0.68 (cod at 0.16), 0.74 (0.37), 0.77
(0.45), 0.81 (0.60), 0.89 (0.85). Central 0.75 (cod at T&H); band 0.4-1.0.

h, with the ceiling in MJ/t/tick in brackets:

| Prey (E MJ/t) | Library h | b 1.0 + armour | **b 0.75 + armour (central)** | b 0.85 + floor + armour (DTU) | b 0.37 + armour (T&H) |
|---|---|---|---|---|---|
| Planktivores/herring (7124) | 272.1 | 272.1 (26.2) | **238 (29.9)** | 251 (28.4) | 195 (36.6) |
| Whiting (4800) | 183.3 | 183.3 (26.2) | **177 (27.1)** | 180 (26.7) | 168 (28.6) |
| Mobile epifauna (3750), A 1.75 | 143.2 | 251 (15.0) | **258 (14.6)** | 255 (14.7) | 269 (14.0) |
| Nephrops (3600), A 2.0 | 137.6 | 275 (13.1) | **286 (12.6)** | 281 (12.8) | 302 (11.9) |
| Infauna (2600), A 1.0 | 99.3 | 99.3 (26.2) | **112 (23.2)** | 137 (18.9) | 134 (19.4) |

- Assimilated energy per tick at a full gut, central: herring 23.9, whiting 21.7, infauna 16.3,
  epifauna 11.6, Nephrops 10.1 MJ/t/tick (library today 20.9 / 20.9 / 18.3 / 20.9 / 20.9).
- Armour ranges at b 0.7: epifauna h 207-341, Nephrops 216-360; infauna A 0.6-1.5 -> h 69-172.
- Capelin (5.5 kJ/g) instead of 4.184 as the Cmax reference lowers every b<1 ceiling by
  (4.184/5.5)^(1-b) (x0.93 at b 0.75); ratios unchanged.

Notes (J): herring E varies 4000-11000 MJ/t over the year; with b < 1 its h should scale as E^b, so a
fat autumn herring fills the gut more slowly per tonne but delivers more energy per tick. Andersen's E
acts on the mixed chyme, so additive per-prey h_j is an approximation (exact at b = 1 and 0). Changing h
moves s* and every derived growth_rate paid at the Holling ceilings (growth_budget, section 160);
lowering one pair's ceiling pushes an energy maximiser to the other pairs (208.5).

Temperature: all models multiply the temperature term with the E/armour terms, so prey ratios hold at
any temperature [F forms]. Evacuation Q10: 2.2 (Andersen whiting), 2.3 (Andersen 2012 cod), 3.0 (T&H),
3.4 (Knutsen & Salvanes), 3.7 (dos Santos) - steeper than the Cmax Q10 ~1.9-2 of the benthivore note, so
the gut limits intake more in winter than the Cmax functions say (Fall & Fiksen 2019: at 1.5 C a full
stomach takes 3 d to empty on amphipods, 11 d on Pandalus). No temperature x armour interaction
reported.

## Q5. Energy, mass or in between

- Energy ceiling (b 1): supported for whiting (Andersen 1999) and plaice on formulated diets (Jobling
  1980); nearly by Andersen's gadoid and cod models (b 0.85-0.875: herring 1.06-1.16x lean fish or
  infauna per gut-hour); not by the cod whole-prey data (b 0.16-0.45).
- Mass ceiling (b 0): no gadoid dataset; the lowest estimate (b 0.16) puts herring at 2.3x infauna, not
  2.7x.
- In between: central b 0.75 (band 0.4-1.0).
- Herring : infauna energy per tick at a full gut: today 1.00 gross (1.14 assimilated); central 1.29
  (1.47); range 1.0-1.9 (1.14-2.2); 1.5 in the DTU branch with the 3.5 kJ/g floor. Herring : epifauna,
  assimilated, with armour: 1.75 at b 1, 2.05 at b 0.75, 2.6 at b 0.37.
- The user's intuition holds in direction: fat herring is worth ~1.3-1.9x infauna and ~2x crustaceans
  per gut-hour; most of the crustacean gap is armour, not fat.

## Gaps

Andersen 2001 exponent (closed); primary source for the 3.5 kJ/g floor; Andersen 2016 and Couturier 2013
full texts; T&H 2003 per-species rho (only Funk et al. cluster means); haddock evacuation vs E;
ophiuroid, bivalve and Nephrops evacuation; sustained ad-lib rations of one gadoid on fatty fish vs
crustaceans in energy units.

## Sources

- Andersen NG 1999 J Fish Biol 54:287-301, doi 10.1111/j.1095-8649.1999.tb00830.x (abs)
- Andersen NG 2001 J Fish Biol 59:1198-1217, doi 10.1111/j.1095-8649.2001.tb00186.x (abs)
- Andersen NG 2012 J Fish Biol 80:595-612, doi 10.1111/j.1095-8649.2011.03195.x (abs; parameters via the 2017 suppl.)
- Andersen NG, Beyer JE 2005a,b J Fish Biol 67:1392-1412, 1413-1433, doi 10.1111/j.0022-1112.2005.00834.x, .00835.x (abs)
- Andersen NG, Chabot D, Couturier CS 2016 J Fish Biol 88:1886-1903, doi 10.1111/jfb.12976 (abs)
- Andersen NG, Lundgren B, Neuenfeldt S, Beyer JE 2017 MEPS 583:195-209, doi 10.3354/meps12319; suppl. https://www.int-res.com/articles/suppl/m583p195_supp.pdf [F]
- Arnott SA, Pihl L 2000 (via `../Mättnad hos bottenbyten/`) [F]
- Couturier CS, Andersen NG, Audet C, Chabot D 2013 J Fish Biol 82:789-805, doi 10.1111/jfb.12005 (abs)
- dos Santos J, Jobling M 1988 J Fish Biol 33:511-516, doi 10.1111/j.1095-8649.1988.tb05495.x (abs)
- dos Santos J, Jobling M 1995 ICES J Mar Sci 52:209-219, doi 10.1016/1054-3139(95)80036-0 [F]
- dos Santos J, Burkow IC, Jobling M 1993 Aquaculture 110:173-189, doi 10.1016/0044-8486(93)90271-Y ([S])
- Fall J, Fiksen O 2019 Fish Fish 21, doi 10.1111/faf.12415 [F]
- Funk S et al. 2023 Ecol Evol 13:e10751, doi 10.1002/ece3.10751 (suppl. Table G1 [F])
- Jobling M 1980 J Fish Biol 17:187-196, doi 10.1111/j.1095-8649.1980.tb02752.x (abs)
- Jobling M 1987 J Fish Biol 30:299-314, doi 10.1111/j.1095-8649.1987.tb05754.x (abs)
- Knutsen I, Salvanes AGV 1999 MEPS 181:61-79, doi 10.3354/meps181061 [F]
- Neuenfeldt S et al. 2020 ICES J Mar Sci 77:624, doi 10.1093/icesjms/fsz224: C = 24 rho0 L^1.30 e^0.083T E^0.15 S (kJ/d), same b 0.85 ([S])
- Temming A, Andersen NG 1992 ICES CM 1992/G:61, https://www.vliz.be/imisdocs/publications/ocrd/274816.pdf [F]
- Temming A, Herrmann J-P 2003 Fish Res 63:21-41, doi 10.1016/S0165-7836(03)00041-9 (abs)
- Velasco F, Olaso I 2000 ICES CM 2000/Q:11, https://www.vliz.be/imisdocs/publications/ocrd/276039.pdf [F]
