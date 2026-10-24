# Evidence B - Gadoid benthic foraging: who can feed on benthos, how fast, and how much

Scope: (1) benthivory classified from morphology and habitat, not from diet; (2) mechanistic
measures of feeding (search speed, search width, foraging time, rations, Cmax, gastric
evacuation); (3) a mechanistic attack rate for the gadoids -> benthos pair, and its ceiling.
Tags: [F] full text (or full data file) read, [A] abstract/snippet only, [S] secondary
citation (the secondary source is named). Wet weight throughout. 1 t/km2 = 1 g/m2.

## Short answers

**Q1. Benthivore share: min 0.54-0.57, central 0.60, max 0.68 of today's gadoid biomass.**
Basis: morphology and habitat only (FishBase mouth position, FAO 1990 jaw and barbel
diagnoses, taste-bud studies). Haddock is the only species in the IBTS mix with a
"sub-terminal/inferior" mouth and a lower jaw shorter than the upper; it is demersal at
80-200 m (score 1). Cod has a terminal mouth but a well-developed barbel and benthic taste-bud
equipment at all sizes (score 0.5-1). Whiting (terminal mouth, barbel absent or small), saithe
(lower jaw longer, pelagic) and hake (large oblique mouth, lower jaw projecting) score 0-0.3.
Norway pout has a "superior" mouth (planktivore build); including it at 0.5-1x the mix biomass
lowers the share to 0.27-0.52 (central 0.30-0.40). The trait database (Beukhof et al. 2019) and the guilds are
**diet-derived**, so they are circular for this purpose and serve only as a cross-check. They agree
on the ranking.

**Q2. Measured rations at 6-8 C, 0.2-2 kg.** Cod Cmax (Hansson et al. 1996 via FB4):
2.1 % BW/d at 500 g and 1.9 % at 1000 g (7 C). Cod lab satiation at 6.3-6.8 C: mean 2.2-3.1 %,
single days 5.7-7.9 %. Field rations for cod and haddock are 0.9-1.7 % BW/d. No haddock Cmax
function was found. Hall 1987 says wild haddock feed at "15-22% of their maximum capacity" [A]. That is a stomach-capacity measure, not a daily ration, and cannot be compared with Cmax. No gadoid
search rate, reaction distance to infauna, capture probability or per-item handling time was
found. Cod in situ swim mostly below 1 body length/s [A].

**Q3. Attack rate.** Swept area (speed x width x fraction of time searching x capture
probability / body mass) gives a = 0.007-5.8, **central 0.21 km2 per t per 6 h**, against the
model's 0.04. The model value lies inside the range, 5x below the central value. Width and
capture probability are assumptions, so a can only be tightened by a measurement or a fit. It
hardly matters: at 90 t/km2 any a in this range saturates the pair unless less than ~0.3 %
(model a) of the stock is accessible. The accessible fraction and the ceiling set intake, not a.
**Ceiling.** The model's 4.6 % BW/d of benthos is 138 kJ/kg/d. That is ~1.6x (0.9-1.8x) the cod
Cmax at 7 C and 2-3x the field rations. h derived from Cmax is 135-154 ticks for 0.5-1 kg at 7 C,
and 87 only at the top of the satiation data. The pairs do not
add up. One action distribution splits each tick between prey, so the whole-group ceiling is the
same 138 kJ/kg/d (checked in code, section 3b).

## 1. Benthivory from morphology and habitat (Q1)

### 1a. The trait databases and guilds are diet-based (circular here)

| Source | What it says | Basis | Tag |
|---|---|---|---|
| Beukhof, Dencker, Palomares & Maureaud 2019, fish trait collection, PANGAEA doi:10.1594/PANGAEA.900866 (data file + SupplementaryInformationTraitCollection.pdf) | `feeding.mode` for LME 22-24 (North Sea, Baltic, Celtic-Biscay): haddock **benthivorous**; cod, whiting, saithe, hake **generalist**; Norway pout **planktivorous**; plaice benthivorous | SI: "Feeding mode is the type of food consumed and reflects species diet ... generalist (... assigned when the major food type comprised no more than approximately two-thirds of the diet). Feeding modes were assigned consulting the feeding type and feeding habit from the 'Ecology' table and the 'Diet' table of FishBase." **Diet-based.** | [F] |
| Garrison & Link 2000 (Georges Bank; ICES JMS 57:723; MEPS 202:231 also cited, volume/pages not confirmed) | five "trophic guilds spanning benthic to pelagic feeders" | guilds are clusters of diet similarity ("diet-defined guilds", Collie/Gifford/Steele preprint) | [A]/[S] |
| Greenstreet et al. 1997 guilds; Ecopath diet matrices (see `../Bentos och torskfiskar/evidence_C_diets_and_consumption.md`) | benthos 30-50 % of cod diet, 40-65 % haddock, 10-30 % whiting, 1-3 % saithe | stomach contents | [S] |

None of these can be used to set a benthic attack rate without circularity. They are listed only as a
cross-check: the morphology ranking below and the diet ranking agree (haddock > cod > whiting >
Norway pout > saithe, hake).

### 1b. Morphology and habitat per species

Mouth position: FishBase Morphology table ("Position of mouth"), https://www.fishbase.se/physiology/MorphDataSummary.php?genusname=<G>&speciesname=<s> [F, page read].
Jaws, barbel, habitat, depth: FAO Species Catalogue vol. 10, Gadiform fishes of the world
(Cohen, Inada, Iwamoto & Scialabba 1990), https://www.fao.org/4/t0243e/t0243e.pdf [F].

| Species (IBTS mix) | Mouth (FishBase) | Jaws / barbel (FAO 1990 unless noted) | Habitat, depth (FAO 1990) | Benthic capacity score (0-1) |
|---|---|---|---|---|
| Haddock (0.54) | "sub-terminal/inferior"; "Mouth relatively small; lower jaw shorter than the upper jaw; chin barbel rather small" | "Chin barbel rather small. Lower jaw shorter than upper." | "A demersal species found from 10 to 450 m depth, more common from 80 to 200 m" | **1.0** (only gadoid of the mix with an inferior mouth) |
| Cod (0.06) | "terminal"; "Protruding upper jaw, a conspicuous barbel on the lower jaw" | genus *Gadus*: "Lower jaw shorter than upper ... Chin barbel well developed." Cod is in the barbel group of "feeders on the infauna and large, slow-moving epifauna; such fish being provided with stout barbels, well equipped with taste-buds, e.g. cod" (Nagabhushanam 1965, JMBA 45:615, https://www.cambridge.org/core/product/01F81175C106A770198CF2B91CED8BB6 [A]). Juvenile cod have "significantly higher taste bud densities ... on the first two pelvic fin rays ... their more benthic lifestyle, in which the pelvic fins are frequently trailed over the sea bed" (Harvey & Batty 2002, J Fish Biol, pp. 583-592, vol. not shown on page, https://pure.uhi.ac.uk/es/publications/cutaneous-taste-buds-in-gadoid-fishes/ [A]) | "generally considered a demersal fish, although its habitat may become pelagic"; "mostly ... from 150-200 m"; "gregarious during the day, forming compact schools that swim between 30 and 80 m above the bottom, and scatter at night" | **0.5-1.0**: equipped for infauna at all sizes; the adult shift to fish is a preference/behaviour, not a loss of capacity |
| Whiting (0.27) | "terminal"; "Chin barbel small or absent"; FishBase "Barbels 0" | "Chin barbel absent or small. Upper jaw projecting slightly." In the barbel-less group feeding "on the smaller active epifauna and nektonic prey, and possessed no, or only rudimentary, barbels, e.g. whiting" (Nagabhushanam 1965 [A]) | "Benthopelagic at depths from 10 to 200 m, but more common from 30 to 100 m, mainly on mud and gravel bottoms" | **0-0.3** (the upper jaw overhangs only "slightly" and there is no chemo-tactile barbel: built for visual epibenthic/nektonic prey, not infauna) |
| Saithe (0.12) | "terminal"; "Chin barbel small" | genus *Pollachius*: "Lower jaw longer than upper"; "A small barbel at tip of lower jaw" | "An active, gregarious, pelagic fish ... to about 200 m depth" | **0** |
| Hake (0.01) | "terminal"; "Barbels 0"; gill rakers 7-9 on lower limb, 8-12 total | genus *Merluccius*: "Mouth large and oblique ... lower jaw projecting below the upper ... teeth in both jaws well developed, sharp" | "usually found between 70 and 370 m depth ... close to the bottom during day-time, but moves off-bottom at night" | **0** (piscivore build) |
| Norway pout (lumped, not in mix) | "superior" | "Lower jaw slightly longer than upper"; genus *Trisopterus*: "Chin barbel well developed" | "Benthopelagic to pelagic over muddy bottoms at depths of 50-300 m, but mostly found between 100 and 200 m"; "It is a pelagic feeder" | **0-0.2** (upturned mouth: planktivore) |
| Flatfish (plaice, dab; lumped) | not checked per species here | asymmetric bottom-dwellers | benthic | 1.0 if they are counted in the group |

Depth: all five species of the mix occur at the model's 100-300 m, haddock and cod at their
modal depths, whiting at its deep edge. Bergstad 1990 (Neth J Sea Res 25:237, abstract via
https://marineinfo.org/id/publication/16334 [A]): "a rather sharp boundary at about the 200 m isobath
between species assemblages of the Norwegian Deep and those of the shallow plateaus of the North Sea
and Skagerrak"; the slopes "appear to be feeding and overwintering areas for ... saithe ... and Norway
pout". So below ~200 m the gadoid mix shifts towards saithe/Norway pout, both scored ~0.

### 1c. Share of today's gadoid biomass that is effectively benthivorous

Share = sum over species of (IBTS biomass share x capacity score). Basis: mouth position, jaw
relation, barbel/taste-bud equipment and demersal habitat; **no diet data**.

| Variant | Haddock | Cod | Whiting | Saithe + hake | Share |
|---|---|---|---|---|---|
| min | 0.54 x 1 | 0.06 x 0.5 | 0 | 0 | **0.57** (0.54 if cod is scored 0) |
| central | 0.54 x 1 | 0.06 x 1 | 0 | 0 | **0.60** |
| max | 0.54 x 1 | 0.06 x 1 | 0.27 x 0.3 | 0 | **0.68** |

Adding Norway pout at 0.5-1x the mix biomass with score 0-0.2 lowers the share to 0.27-0.52
(central 0.30-0.40: 0.60/2 and 0.60/1.5);
adding flatfish raises it. The weighting is by biomass, not numbers. The share says which fraction
of the group can use the benthic pair at a haddock-like rate; it says nothing about what fraction
of the diet ends up being benthos, which must stay emergent.

## 2. Mechanistic feeding measures (Q2)

Retrieved partly by a delegated web lookup. Its [F] tags mean that lookup read the full text;
the FB4 parameter row and the Cmax arithmetic were re-checked here.

### 2a. Maximum consumption (Cmax, satiation) - % body weight per day, wet prey / wet fish

| Species, size, T | Value | Type | Source | Tag | Quote |
|---|---|---|---|---|---|
| Cod, North Sea; Cmax = CA W^CB f(T), CA 0.099, CB -0.195, CQ 2.41, CTO 13.7 C, CTM 21 C (Kitchell f) | 7 C: 200 g **2.55**, 500 g **2.13**, 1000 g **1.86**, 2000 g 1.63; 6 C 500/1000 g 1.97/1.72; 8 C 2.30/2.01; at optimum 13.7 C x1.38 | lab-based bioenergetic Cmax (g prey/g fish/d; ED 4184 J/g) | Hansson et al. 1996 via Fish Bioenergetics 4 `Parameters_official.csv`, https://raw.githubusercontent.com/jim-breck/FB4/master/Parameters_official.csv | [F] file, [S] Hansson | row "North Sea cod, Hansson et al. 1996,2,0.099,-0.195,2.41,13.7,21" |
| Cod 357-668 g, 6.3-6.8 C, chopped capelin ad lib | mean **2.2-3.1** %BW/d; single-day max 5.7-7.9 % | lab satiation | Braaten & Gokstad 1980, ICES CM 1980/F:20, https://www.ices.dk/sites/pub/CM%20Doccuments/1980/F/1980_F20.pdf | [F] | "Maximum daily food intake ranged between 5.7% (4-hr intervals) to 7.9% (24-hr intervals) for cod between 496 and 638 g." |
| Cod 117-400 g, 6.2-6.5 C (4.1-5.0 C) | 2.5 (1.9) %BW/d | lab satiation | same | [F] | "At 6.2 - 6.5 C the daily ration ranged from 1.9 to 3.1 % per day with an average of 2.5%." |
| Cod ~500 g adults | ~1.9 %/d (T not given) | Cmax | Karamushko & Reshetnikov 1994 via Peck et al. 2003, MEPS 251:233, https://int-res.com/articles/meps2003/251/m251p233.pdf | [S] | "...and 500 g adults ... ~55, 13.1, 8.9, and 1.9% d-1" |
| Cod, NE Arctic, upper envelope of stomach data | Cmax = 68.84 W^0.934 cal/d -> 500 g: 2.7 %BW/d of capelin (1670 cal/g), 3.4 % of herring | field envelope, no T term | Ajiad et al. 1994, ICES CM 1994/O:7, https://www.vliz.be/imisdocs/publications/ocrd/275273.pdf | [F] | "Cmax(cal/day) = 68.84 * W^0.934" |
| Cod 600-4000 g, 8 C / 4 C, moist pellet | peak ~1.6 %BW/d (8 C, Aug-Oct), 0.4-0.8 in winter; maintenance 0.27 / 0.22 % | lab satiation (scan's basis ambiguous) | Waiwood 1978, CAFSAC Res Doc 78/16, https://waves-vagues.dfo-mpo.gc.ca/library-bibliotheque/75642.pdf | [F] | "Food intake varied seasonally despite constancy of temperature." |
| Haddock, wild | feeds at "between 15 and 22% of their maximum capacity"; Cmax value not in abstract | stomach-capacity based | Hall 1987, J Fish Biol 31:479, https://marineinfo.org/doc/publication/244624 | [A] | as quoted |
| Dab, 9.2-10.0 C | mean 2.23 %BW/d (day to day 0-7.5) | lab satiation | Bagge et al. 1995, ICES CM 1995/J:6, https://www.vliz.be/imisdocs/publications/ocrd/275364.pdf | [F] | - |

No haddock, whiting or saithe Cmax function was found (FB4 has none).

### 2b. Field daily rations and evacuation

| Species, size, T | Value | Source | Tag | Quote |
|---|---|---|---|---|
| Cod 30-60 cm, North Sea (5-9 C) | 30 cm 1.71, 40 cm 1.28, 50 cm 1.03, 60 cm 0.85 %BW/d (R = 2S/D, S = 0.00016 L^3, D = 0.06 L d) | Daan 1973, ICES CM 1973/F:38, https://www.vliz.be/imisdocs/publications/ocrd/274191.pdf | [F] | cited by Durbin: "from 1.3% BW/d in a 40 cm cod to 0.8% BW/d in a 60 cm fish" |
| Cod >30 cm, NW Atlantic | 1.42 (6.8 C), 1.66 (9.3 C); bounds 0.9-1.5 %BW/d | Durbin et al. 1983, Fish Bull 81:437, https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1983/813/durbin.pdf | [F] | "Upper and lower daily ration estimates were ... 1.5 and 0.9% BW per day for cod >30 cm." |
| Cod, Newfoundland | 1.0-2.5 %, occasionally 3 % | Orlova & Chumakov 1991, NAFO SCR 91/100, https://www.nafo.int/Portals/0/PDFs/sc/1991/scr-91-100.pdf | [F] | "daily rations make up 1.0-2.5% on the average" |
| Cod / haddock, North Sea 1991, annual | 1.44 / 1.30 %BW/d | Greenstreet et al. 1997 via `../Bentos och torskfiskar/evidence_C_diets_and_consumption.md` | [S] | - |
| Haddock, cod, whiting evacuation (Jones 1974) | dR/dt = Q (L/40)^1.4 S^0.46, Q 0.15 /h at 6 C; 40-cm fish with 10-20 g in the stomach -> 1.6-2.2 %BW/d | Ursin et al. 1985 (Dana 5:63) and Durbin 1983 App. Table 1 | [S] | "Q = 0.15 ... on an average at 6 C" |
| Haddock vs cod energy demand | intake ~ W^b, b 0.77-0.80 haddock, 0.80-1.3 cod | Jones 1978, J Cons 38:18, https://academic.oup.com/icesjms/article/38/1/18/645562 | [A] | - |
| Evacuation vs temperature | R = 0.0406 e^(0.111 T) (small prey); cod (Tyler 1970) b 0.106 over 2-15 C; fish flesh evacuated ~10 % as fast as shrimp tails | Durbin 1983 | [F] | "R = 0.0406 e^0.111t" |

Search and capture: in situ cod speeds "below 1.0 body lengths (bl) s-1 were most common (70%)"
(Fernö, Jørgensen, Løkkeborg & Winger 2011, Mar Biol Res 7:310, abstract via
https://marineinfo.org/id/publication/204763 [A]). **No measured search rate, clearance area,
reaction distance to infauna, capture probability or per-item handling time was found for any
gadoid or flatfish on infauna.** The plaice studies of Hill, Burrows & Hughes (J Fish Biol 2000,
2002, 2003; pause-travel, area-restricted search for buried bivalves) are the closest but their
movement parameters were not retrievable [A].

## 3. Derivations (Q3)

### 3a. Attack rate as swept area

Encounter model: a = v x w x T x f x p / W, the area cleared per tonne of fish per 6-h tick
(km2 t^-1 tick^-1, which is the model's unit since intake = a x B_prey).

| Input | min | central | max | Source / status |
|---|---|---|---|---|
| Body length L | - | 35 cm | - | ASSUMPTION (typical Skagerrak IBTS haddock; size distribution not retrieved) |
| W = 0.0083 L^3.028 | - | 0.39 kg | - | FishBase haddock length-weight, North Sea Q2 1995-2007, n 161, https://www.fishbase.se/popdyn/LWRelationshipList.php?ID=1381 [F] |
| v, search speed | 0.10 m/s (0.3 L/s) | 0.15 m/s (0.43 L/s) | 0.35 m/s (1 L/s) | Cod, 30-60 cm, in situ: "Swimming speeds below 1.0 body lengths (bl) s-1 were most common (70%)" (Fernö et al. 2011 [A]); slower at night (Løkkeborg 1998 [S via `../Torskfiskars ljusberoende vid jakt/prey_side_and_benthic.md`]). Search speed within that range is an ASSUMPTION |
| w, search width for benthos | 0.05 m | 0.10 m | 0.30 m | ASSUMPTION: chemo-tactile reach of barbel, lips, pelvic rays (Harvey & Batty 2002 [A]). No measured reaction distance of a gadoid to infauna was found. Far-field odour detection (bait, 240 m; Løkkeborg & Fernö 1999 [S]) concerns carrion plumes, not infauna |
| T | 21 600 s | 21 600 s | 21 600 s | one tick |
| f, fraction of time searching the bottom | 0.25 | 0.5 | 1.0 | ASSUMPTION; haddock and cod feed day and night on benthos ([S] same light-dependence notes) |
| p, capture per encounter | 0.1 | 0.5 | 1.0 | ASSUMPTION (prey retract; Amphiura arms are nipped, not whole animals) |
| Area per fish per tick | 2.7 m2 | 81 m2 | 2 268 m2 | computed |
| **a** | **0.007** | **0.21** | **5.8** km2 t^-1 tick^-1 | computed (= m2/kg/tick / 1000) |

Model: a = 0.04 km2 t^-1 tick^-1 = 40 m2 per kg per 6 h, i.e. a 0.39-kg haddock clearing
16 m2 per 6 h (0.7 m2 per hour). That sits inside the mechanistic range, 5x below the central
value. With only today's benthivore share (section 1c) eating benthos at the haddock rate, the
pooled-group a is 0.6 x a_haddock (0.004-3.5, central 0.12).

What a can and cannot be. Every input except W is either an assumption or a cod speed
transferred to haddock. The two biggest unknowns, w and p, are behavioural quantities nobody has
measured for gadoids on infauna, so **a for this pair is a range, not a number; tightening it
needs either a lab clearance-rate measurement or a fit**, and a fit to diet would be circular.

Saturation. The intake fraction of the ceiling is x/(1+x), x = a h phi B, with B = 90 t/km2,
h = 87, phi = accessible fraction of the benthic stock.
- Model a 0.04: x = 313 phi. phi 1 -> 0.997, 0.1 -> 0.97, 0.01 -> 0.76, 0.003 -> 0.48,
  0.001 -> 0.24.
- Mechanistic central a 0.21: half-saturation at phi = 0.0006.
So with any a in the mechanistic range, **a does not control today's benthos intake**; the
ceiling and phi do. Benthos intake drops below the ceiling only if less than ~0.3 % (model a)
or ~0.06 % (central a) of the 90 t/km2 is within reach. phi (burial depth, size window,
exposed arms and siphons) is the parameter that matters and is for the sibling accessibility
evidence; without measured burial-depth/size data it can only be fitted.

### 3b. Ceiling on benthos

Model: 1/h = 1/87 per tick = 4.6 % BW/d of benthos (wet). Library comment: "Energy-limited
ceiling, h = 130 * 3000 MJ/t / 4500" (`fgconfig/fg_library.yaml`, resume section 148), so in
energy the pair ceiling is 4 x 3000/87 = 138 kJ per kg fish per day, the same for every gadoid pair.

Compare on an energy basis (benthos 3000 J/g as in the library; FB4 cod prey 4184 J/g):

| Quantity at 6-8 C | % BW/d of its own prey | kJ/kg/d | Model ceiling / it |
|---|---|---|---|
| Model ceiling, one pair | 4.6 (benthos) | 138 | 1 |
| Cod Cmax, Hansson/FB4, 500 g, 7 C | 2.13 (4184 J/g) | 89 | **1.55** |
| Cod Cmax, 1000 g, 7 C | 1.86 | 78 | **1.78** |
| Cod lab satiation, Braaten & Gokstad, 6.3-6.8 C, capelin (~4-5 kJ/g, ASSUMED) | 2.2-3.1 | 90-155 | 0.9-1.5 |
| Field rations, cod/haddock 0.3-2 kg | 0.9-1.7 | 40-75 (at 4.4 kJ/g, ASSUMED) | 1.8-3.4 |

Expressed in benthos mass, the cod Cmax at 7 C, 0.5-1 kg, is 3.0-2.6 % BW/d (89/3.0 = 3.0 %),
against the model's 4.6 %. So the single-pair ceiling is about 1.6x (range 0.9-1.8x) the best
measured maximum at Skagerrak temperatures, and 2-3x the field ration. Three further points:

1. Shared, not summed. The task context says each prey pair saturates independently. In the
   code each pair does have its own a_eff = a B/(1 + a h B + I), but intake is
   `predator_biomass * actions.eat * a_eff * hunger` (`lib/environments/ecosystem_env/predation.py`),
   and `actions.eat` is a slice of ONE action distribution over move/rest/eat-per-prey
   (`decisions.actions_from_probabilities`), so the eat shares sum to at most 1. Total intake is
   therefore at most max_j(1/h_j) in energy = 138 kJ/kg/d for every gadoid pair (all pairs use the
   same energy-limited ceiling). The group ceiling is ~1.6x Cmax, not 3x 138. This is a time-sharing
   ceiling, consistent with one shared gut.
2. Temperature. The FB4 cod function puts 6-8 C at 0.67-0.78 of the optimum-temperature Cmax.
   A ceiling taken at the optimum (13.7 C) would be 2.9 % BW/d of 4184-J prey for 500 g, still
   below the model's 4.6 % of benthos once converted (4.1 %).
3. Handling. The model's h is a digestion (gastric evacuation) limit, not a per-item handling
   time; per-item handling of a small polychaete is seconds and was not found measured. Deriving
   h from Cmax is the defensible route: h = 1/(Cmax per tick), with Cmax from a Cmax function.

Implied h (ticks) from Cmax, benthos mass, 7 C: 500 g: 4/0.0297 = 135; 1000 g: 4/0.0259 = 154;
optimum-temperature 500 g: 4/0.041 = 97; Braaten & Gokstad upper (3.1 % of ~4.5-kJ capelin =
4.65 % of benthos): 86. The model's 87 is the edge of the measured range, reached only at the top
of the cod satiation data (6.3-6.8 C).

## Not found / could not open

- Mattson 1990/1992 (Sarsia, "Food and feeding habits of fish species over a soft sublittoral
  bottom", cod/haddock): not located online.
- Bergstad 1991 Sarsia 75:269-325 (Norwegian Deep gadoids): only bibliographic records. Albert
  1994 (ICES JMS 51:31, haddock, Norwegian Deep): publisher page blocked, abstract not read here.
- Jones 1954 (whiting vs haddock food), Du Buit, Ellis et al., Greenstreet & Rogers guilds,
  Garrison & Link 2000 full text (int-res returned 429): not read.
- No gadoid jaw-protrusion or suction-kinematics study was found. Nagabhushanam 1965 and Harvey &
  Batty 2002 were read as abstracts only.
- Løkkeborg 1998 (Anim Behav 56:371) and Løkkeborg & Fernö 1999 (Env Biol Fish 54:345) full texts:
  the speeds in m/s were not obtained. Hill, Burrows & Hughes plaice-search parameters: paywalled.
- Haddock Cmax function; Hall 1987 full text; Jones & Hislop 1978; Jones 1974/1978 full texts;
  Jobling 1988 equation; Bjornsson 2001/2007 intake tables; Hislop 1991 full text; Temming &
  Herrmann 2003 rho table; Andersen 1999/2001 (Wiley 403); Bromley 1994; dos Santos & Jobling
  1995; Hop & Tonn 1998; Tyler 1970 original; Lankester; Greenstreet 1997 original.
- IBTS length distribution of Skagerrak haddock. The 35 cm / 0.39 kg used in 3a is an assumption.
- Any measured reaction distance, capture probability or per-item handling time of a gadoid on infauna.
