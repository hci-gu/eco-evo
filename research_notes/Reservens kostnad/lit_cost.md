# Does carrying an energy reserve cost a fish energy? - literature trail

Question: in the model a reserve costs nothing (maintenance is charged per
tonne of lean biomass only). Is there a carrying cost of a lipid reserve
(muscle/mesenteric lipid, or a fatty gadoid liver), and how big is it per
unit fill, relative to resting metabolism? Four mechanisms: (1) metabolic
(the stored tissue respires), (2) buoyancy (lipid is light; swim bladder
gas; hydrodynamic lift), (3) drag / cost of transport (girth), (4) the cost
of making and retrieving a store (deposition and mobilisation efficiency,
turnover).

Species classes kept apart: GADOIDS (physoclist swim bladder, liver store:
cod, haddock, whiting, saithe, Norway pout, hake), CLUPEIDS (physostome
swim bladder, muscle and mesenteric lipid: herring, sprat), MACKEREL (no
swim bladder, muscle lipid). Anything else is PROXY.

Tags: [F] full text read (quotes with page/section), [A] abstract only,
[S] secondary (claim about a paper read in another), [D] derived here,
[J] judgement. One row per sourced number in tables/cost_raw.csv.

Conversion conventions (from the task): gadoid reserve maximum
1800-2000 MJ per tonne lean (~HSI 8 % at 60 % liver lipid); planktivore
maximum 4990 MJ/t (whole-body fat ~14 % above a 2 % floor at 39.5 MJ/kg);
mackerel 6980 MJ/t; resting metabolism ~3.3-3.9 MJ/t per 6 h for gadoids
(13-16 MJ/t/d), 8 for planktivores (32 MJ/t/d, summer reference), 4.6 for
mackerel (18.4 MJ/t/d).

## Source trail

Downloads are in data/c_* (this note's own; other data/ folders belong to a
parallel note and were only read). Arithmetic is in scripts/derive.py, its
output in tables/derived_out.txt.

### 1. Strand, Jorgensen & Huse 2005 - swim bladder bioenergetics, cod [F]
Ecological Modelling 185: 309-327, doi 10.1016/j.ecolmodel.2004.12.013.
Theory/model parameterised for Atlantic cod (target species). Full text
(data/c_strand2005).
- Tissue density table (p. 312, parameter list): "Fish tissue density; water
  density 1081; 1026 kg m-3"; secretion efficiency Eeff 0.10.
- Gas secretion cost (p. 319): "When gas secretion is at its maximum, the
  estimated energetic cost is 0.0017 J kg-1 s-1. This corresponds to 1.0% of
  the standard metabolism and is ... close to negligible compared to the
  energy required for swimming."
- Abstract: "Regulation of swimbladder volume is energetically cheap but
  slow." Compensatory lift is the expensive mode; optimal tilt ~7 deg.
- Free band (p. 323): "A 2-kg cod that swims at a speed of 0.25 BL s-1 ...
  would, without increasing its energy budget, be able to freely deviate
  within 89 and 106% of optimal swimbladder volume."
- Fins (p. 314): "Experimental studies have shown that large cod, by using
  their pectoral fins as hydrofoils while swimming freely, can compensate
  for 70-90% of the lift from the swimbladder (Ona, 1990)." (a claim about
  Ona 1990: [S]).
- Tagged cod mostly sat 0 to -40 % below optimal bladder volume (p. 323).
Use here: a cod's bladder at neutral buoyancy is (1 - 1026/1081) = 5.1 % of
body volume [D]. The liver lipid of a full gadoid reserve offsets ~10 % of
the lean excess weight (derive.py), i.e. the same as an 10 % smaller bladder
- inside the free band of 89-106 %. Mechanism buoyancy: sign = small
BENEFIT (less gas to secrete and keep; a smaller bladder also changes less
in absolute volume with depth), size <= 0.1 % of SMR [D].

### 2. Lapointe, Guderley & Dutil 2006 - cod metabolic rate vs condition [A]
Physiol Biochem Zool 79: 109-119, doi 10.1086/498290. Lab, 24 Gulf of
St Lawrence cod, condition K 0.676 -> 0.923 after 9 wk feeding. Abstract
only (no OA copy; publisher blocked).
- "The improved condition did not change mass-specific SMR." AMR, aerobic
  scope and EMR per kg decreased. "Ucrit increased and cost of transport at
  1.3 and 1.5 body lengths s(-1) decreased with improved condition", COT at
  other speeds unchanged.
- Caveats [J]: the K gain is refeeding after a low-condition (spring) state,
  so it mixes restored muscle water/protein, liver growth and lipid, and any
  release of starvation metabolic depression; SMR is per WET mass. An
  unchanged mass-specific SMR means the added mass respired at about the
  body mean - it cannot separate a lipid cost from a lean cost. The COT
  result (per kg) shows no drag penalty of better condition at 1.3-1.5 BL/s.

### 3. Martinez, Guderley, Dutil et al. 2003; Martinez et al. 2004 - cod
condition and swimming [A]
J Exp Biol 206: 503-511, doi 10.1242/jeb.00098; J Exp Biol 207: 2979-2990,
doi 10.1242/jeb.01142. Abstracts only (journal site refuses scripted
download).
- 2003: K 0.54 (starved 16 wk) vs 0.81 (fed): starved cod swam "only 30% of
  the time (and distance) of fed fish".
- 2004: K 0.5 vs 1.0 (12 wk); Ucrit and sprint both lower in starved fish.
Use: the performance cost is on the LOW-condition side; no sign of a
carrying cost of a full reserve. Starved fish are lean AND reserve-depleted,
so these are not clean tests of reserve alone [J].

### 4. He & Wardle 1986 - mackerel tilting at low speed [A]
J Fish Biol 29 (Suppl A): 223-232, doi 10.1111/j.1095-8649.1986.tb05013.x.
Scomber scombrus (target), 30-39 cm, lab film records. Abstract:
"Negatively-buoyant Atlantic mackerel ... tilt their bodies with the head up
while swimming at speeds below 0.8 body length per second"; max attack angle
27 deg at 0.45 BL/s; "larger body attack angles ... were associated with
denser bodies at each speed"; "Lift from the fins is insufficient at low
swimming speeds." Use: direct evidence that body density sets a lift
requirement in mackerel; lipid lowers density, so a fat mackerel needs less
lift (BENEFIT). Densities themselves are in the full text, not read.

### 5. Magnuson 1973 - scombrid density and minimum speed [F] (PROXY)
Fishery Bulletin 71(2): 337-356 (no doi; NOAA SWFSC copy, data/c_magnuson1973).
Table 1 (p. 340), densities Df (g/ml): Scomber japonicus 0-2 kg mean 1.054
(SD 0.011, range 1.024-1.069; S. japonicus HAS a gas bladder); bladderless
scombrids "All sizes (without gas bladders only)": Sarda 1.078, Auxis 1.086,
Euthynnus 1.087, Katsuwonus 1.091 g/ml. Minimum speed for hydrostatic
equilibrium rises with density and falls with pectoral lift area (abstract
and model). PROXY for Scomber scombrus (no gas bladder): lean scombrid
tissue ~1.08-1.09 g/ml, used as D_lean 1.085 in derive.py.

### 6. Brawn 1969 - herring buoyancy vs oil content [S]
J Fish Res Board Can 26: 2077-2091, doi 10.1139/f69-193. Not read; claims
as summarised in search results/citing literature: Atlantic and Pacific
herring adjusted near the surface had sinking factors 1000-1008 and an
inverse relation between oil content and swim bladder volume up to 12 %
oil; above that the bladder was kept at 2.6-3.0 % of body volume. Use: from
these numbers the lean (0 % oil) herring tissue density is 1.073-1.088 g/ml
(central 1.080) [D] (derive.py) - the same as cod tissue (1.081) and
bladderless scombrids. Fat replaces swim bladder gas, i.e. buoyancy is a
BENEFIT of lipid in herring.

### 7. Ona 1990 - natural variation in swim bladder (herring, gadoids) [A]
J Mar Biol Assoc UK 70: 107-127, doi 10.1017/S002531540003424X. Abstract:
"Evidence for both periodic variations, as from uncompensated vertical
migrations, and seasonal variations, caused by the fat cycle and gonad
development, are presented." Use: the bladder volume follows the fat cycle
(qualitative; numbers in the full text, not read).

### 8. Huse & Ona 1996 - overwintering herring tilt and speed [A]
ICES J Mar Sci 53: 863-873, doi 10.1006/jmsc.1996.9999 (Crossref record
with this doi; OUP full text refused scripted download; abstract verbatim
from OpenAlex). Norwegian spring-spawning herring at wintering depth:
"Lacking the ability to refill the swimbladder they are constantly
negatively buoyant." "Negative buoyancy seems to be controlled through
constant swimming at speeds between 0.25-0.42 m s-1 because these are
sufficiently high to generate lift when the pectoral fins are used as
spoilers." Night tilt up to 40 deg, "rise and glide".
Use: herring below a few tens of metres carry a compressed (physostome)
bladder and pay for lift by swimming; lipid reduces that excess weight.

### 9. Ona 2003 - herring target strength vs depth [A]
ICES J Mar Sci 60: 493-499, doi 10.1016/S1054-3139(03)00031-6. Abstract:
mean TS "significantly dependent on the depth (pressure) and the
gonadosomatic index" - confirms bladder compression with depth (no gas
compensation at depth).

### 10. Gleiss, Potvin & Goldbogen 2017 - buoyancy vs drag in sharks [F] (PROXY)
Proc R Soc B 284: 20171345, doi 10.1098/rspb.2017.1345 (PMC5698638 HTML,
data/c_gleiss2017; sections quoted). 32 shark species + hydrodynamic model
(parasite + induced + acceleration-reaction drag; lift set equal to negative
buoyancy).
- Results: "Our hydrodynamic model predicts the lowest steady-state drag when
  buoyancy is close to neutral, whereas negative buoyancy appears
  particularly costly at low speeds ... a result of induced drag dominating
  at slow cruising speeds compared to the increased parasite drag resulting
  from a decreased FR [fineness ratio]."
- "Beyond 1.5 m s-1, induced drag becomes negligible during steady swimming
  ... sharks with larger livers generate higher drag compared to those with
  smaller livers."; negative buoyancy lowers drag during accelerations.
- Intro: lipids "provide a comparatively inefficient solution for the
  production of positive buoyancy; ... large livers are required ... at the
  potential expense of increased drag because of reduced streamlining."
Use: the sign of a lipid store's hydrodynamic effect depends on speed: a
BENEFIT (less induced drag) at slow cruising, a COST (girth) at fast
cruising and a cost in bursts. Not a fish with a bladder; mechanism only.

### 11. Sewall, Norcross & Heintz 2021 - juvenile Pacific herring [F] (PROXY)
Can J Fish Aquat Sci 78: 881-893, doi 10.1139/cjfas-2020-0293 (PDF read from
the parallel note's data/cjfas2021_pacherring). Clupea pallasii YOY, winter
rations. At the April swim test wet-mass lipid was 8.72 (full), 7.73
(reduced) and 6.92 % (fasted), not significantly different (p = 0.228;
table p. 887-888). p. 890: "Critical swimming speeds (BL s-1) did not
significantly differ among treatments in April following the winter ration
period (ANOVA; F[2,27] = 0.14, ... p = 0.870) or in May following spring
re-feeding (... p = 0.360)". Use: weak - Ucrit did not differ where lipid
barely differed (7-9 % wet); neither a drag penalty nor a benefit shown.

### 12. Oikawa & Itazawa 2003 - summed tissue respiration, porgy [F] (PROXY)
Fisheries Science 69: 687-694, doi 10.1046/j.1444-2906.2003.00675.x
(J-STAGE, data/c_oikawa2003). Pagrus major, in vitro tissue QO2.
- Table 1 (p. 689), QO2 = c W^d (umol O2 g-1 h-1): white muscle c 0.80,
  d -0.035; hepatopancreas c 13.39, d -0.130; hepatopancreas mass
  0.0170 W^1.008; summed whole body Min vitro = 5.30 W^0.816 (umol/h).
  At 1 kg [D]: liver 5.45, white muscle 0.63, whole body 1.49 umol/g/h;
  liver 1.8 % of body mass = 6.6 % of whole-body O2; liver per g = 3.7 x
  the body mean, white muscle 0.42 x.
- p. 693: "tissues less essential to life (e.g. those of white muscle and
  fat) have a lower metabolic rate. Fat was not examined in the present
  study, but the QO2 of fat was negligibly low in the carp." (carp claim =
  [S], their ref. 27).

### 13. Itazawa & Oikawa 1983 - excised tissues of carp [S] (PROXY)
Experientia 39: 160-161, doi 10.1007/BF01958874. Not read; via search
summary: summed in vitro tissue O2 matched in vivo whole-body O2 in carp,
which validates the summing approach of item 12.

### 14. Wang et al. 2010 (Elia 1992 values) - human tissue rates [A] (PROXY)
Am J Clin Nutr 92: 1369-1377, doi 10.3945/ajcn.2010.29885 (PubMed
abstract, data/c_wang2010). Specific resting rates (kcal kg-1 d-1): "200
for liver, 240 for brain, 440 for heart and kidneys, 13 for skeletal muscle,
4.5 for adipose tissue, and 12 for residual organs", validated against
measured REE. Use: adipose tissue respires at 0.35 x skeletal muscle per kg
(and its lipid itself not at all); an endotherm, so only the ratio is used.

### 15. Davie 2005 - cod liver composition vs HSI [F]
PhD thesis, Institute of Aquaculture, University of Stirling (no doi; PDF
in the parallel note's data/davie2005_thesis). Farmed Gadus morhua.
Section 4.4.3.2 (p. 207-208): liver "%lipid = -1.1536(%moisture) + 95.775
(r2 = 0.98, n = 22)"; HLSI (liver lipid as % of body) vs HSI r2 = 0.76;
"HSI accurately reflects liver lipid content". Use [D]: the non-lipid,
non-water liver fraction is 14.3 % at 20 % lipid and 9.0 % at 60 % lipid,
so from a lean liver (HSI 2 %, 20 % lipid; 159 MJ/t lean) to a full one
(HSI 8 %, 60 % lipid; 1992 MJ/t lean, = the model's gadoid maximum) the
non-lipid liver mass doubles, 1.6 -> 3.2 % of body mass, and the
non-lipid-non-water part rises 0.29 -> 0.72 % of body mass. Stated as the
pattern in Lambert & Dutil 1997 (CJFAS 54(S1): 104-112, doi
10.1139/f96-149, [S]): liver energy changes are mainly lipid, and liver
energy rises exponentially with HSI.

### 16. Bernard, Reidy, Zwingelstein & Weber 1999 - trout lipolysis [A] (PROXY)
J Exp Biol 202: 279-288, doi 10.1242/jeb.202.3.279. Rainbow trout, tracer
infusions. Abstract: trout "maintain Ra glycerol and Ra NEFA well in excess
of oxidative fuel requirements even at rest. More than two-thirds of the
NEFAs produced are re-esterified"; high re-esterification proposed to serve
membrane restructuring. Use: a stored fatty acid is continuously cycled;
each re-esterification costs ~2 ATP of the ~106 ATP it yields when
oxidised (palmitate, standard stoichiometry [D]). The cost relative to RMR
and whether the cycling rate scales with store size were not in what I read.
A Weber-lab thesis summary (search snippet, not read; [S]) states that only
~13 % of the resting lipolytic rate is needed for resting fuel.

### 17. Lupatsch, Kissil & Sklan 2003 - deposition costs of protein and lipid [A] (PROXY)
Aquaculture 225: 175-189, doi 10.1016/S0044-8486(03)00288-6. Gilthead sea
bream, European sea bass, white grouper; comparative slaughter, multiple
regression. Via the abstract as summarised by the search tool: cost of
protein deposition 1/kP = 1.79-1.90 kJ DE per kJ deposited; of lipid
deposition 1/kL = 1.10-1.31. [D] on one basis: extra energy per kJ
deposited 10-31 % (lipid) vs 79-90 % (protein); share of the input lost
9-24 % (lipid, kL 0.76-0.91) vs 44-47 % (protein, kP 0.53-0.56).
Companion values (Sklan, Lupatsch & Kissil 2003, Isr J Aquacult Bamidgeh
55, doi 10.46989/001c.20354, [A]): kDEg 0.67, kDPg 0.47 in sea bream; and
Lupatsch et al. 1998 (Aquacult Nutr 4: 165-173, doi
10.1046/j.1365-2095.1998.00065.x, [A]) DE for growth 0.46 above maintenance.

### Looked at, nothing usable or not reachable
- Sousa, Domingos, Poggiale & Kooijman 2010 (Phil Trans R Soc B 365: 3413,
  doi 10.1098/rstb.2010.0166): DEB theory; the abstract does not state the
  reserve-maintenance rule and the full text was not reachable. From
  general knowledge of DEB [J]: reserve is assumed to need no somatic
  maintenance (maintenance is paid on structure) - the model's current
  assumption.
- Lannig et al. 2003 (cod liver lipid and mitochondrial enzymes; AWI
  copy timed out), Fassler et al. 2008 (Fish Res 92: 314,
  doi 10.1016/j.fishres.2008.01.013; no abstract, no OA), salmonid SMR vs
  lipid (no specific study found in two searches), any measured mackerel
  density vs fat (none found).

## Synthesis by mechanism (numbers from derive.py unless sourced)

Common constants [D]: lipid 0.93 g/ml lifts 0.103 kgf per kg in seawater
(1.026). Lean tissue ~1.08 g/ml in all three classes (cod 1.081 [F] item 1;
herring 1.073-1.088 [D] from Brawn [S]; bladderless scombrids 1.078-1.091
[F] PROXY item 5) carries an excess weight of ~50 kgf per tonne lean; ~490
kg lipid per tonne lean (33 % of the body) would be needed for neutral
buoyancy without gas. Full-fill reserve lipid per tonne lean: gadoid 51 kg
(2000 MJ), herring 126 kg (4990 MJ), mackerel 177 kg (6980 MJ).

### M1 Metabolic (does the stored tissue respire?)
- Lipid itself: no respiration; adipose tissue 0.35 x muscle per kg in man
  [A PROXY item 14]; "QO2 of fat was negligibly low in the carp" [S via F,
  item 12]. Whole-body cod SMR per wet kg unchanged from K 0.68 to 0.92 [A
  item 2] - added mass respired about as the body mean, not clean for lipid.
- GADOIDS: the store is a liver, a metabolically active organ. Lean porgy
  liver respires 3.7 x the body mean per g and is 6.6 % of whole-body O2 at
  1.8 % of body mass [F PROXY item 12]. In cod the non-lipid liver mass
  doubles from a lean (HSI 2 %, 20 % lipid) to a full liver (HSI 8 %, 60 %
  lipid): +1.6 % of body mass of water+protein, +0.43 % of non-water
  non-lipid matter [F item 15 -> D]. If that matrix respires as lean liver
  per g: +6.5 % of SMR at full fill; scaled by protein-like matter instead:
  +11 %; if the extra is mostly lipid-droplet cytoplasm and water: ~0 [D].
  Central [J]: +5 % of SMR at full fill (range 0-11 %), i.e. ~0.65-0.8
  MJ/t/d at 13-16 MJ/t/d, = 0.03-0.04 % of the stored energy per day.
- HERRING/SPRAT, MACKEREL: the store is lipid in muscle and mesenteric
  adipose tissue. Upper bound with the human adipose/muscle ratio (0.35) x
  fish white muscle/whole (0.42): +2.2 % (herring) and +3.0 % (mackerel) of
  RM at full fill; lower bound ~0 (carp) [D, PROXY inputs].

### M2 Buoyancy (lipid is LIGHT: the sign is a benefit wherever the fish
pays for lift)
- GADOIDS (physoclist, secrete gas): a full liver offsets ~10 % of the lean
  excess weight, equal to a 10 % smaller bladder [D]. The bladder adjusts in
  ~2 days and its maximum secretion cost is 1 % of SMR [F item 1]; a 2 kg cod
  at routine 0.25 BL/s can sit at 89-106 % of the optimal bladder volume at
  no extra cost [F item 1]. Net: small BENEFIT, <= 0.1 % of SMR [D].
  Liver lipid matters for buoyancy only when the bladder is off-volume after
  vertical moves (Strand: mostly 0 to -40 %); then a fat liver cuts the
  excess by ~10 % of the lean excess - again inside the free band.
- HERRING/SPRAT (physostome: gas only by gulping at the surface, no
  secretion): near the surface lipid only replaces gas (Brawn [S]: bladder
  volume falls with oil content up to 12 %), no cost either way. At depth
  the bladder is compressed (Ona 2003 [A], Huse & Ona 1996 [A]) and the fish
  swims 0.25-0.42 m/s to make lift. A full reserve cuts the deep excess
  weight from 48 to 35 kgf/t lean (-27 %; -47 % if lift cost goes as lift
  squared at fixed speed) [D]. Lift cost at 0.30 m/s with L/D 5-10 and
  efficiency 0.2 (both assumed [J]; 0.2 is Strand's Omega at MAXIMUM
  sustained speed, 0.05 at zero speed, so the costs and savings here are
  lower bounds): 6-12 MJ/t/d [D]; the reserve
  then SAVES ~1.6-5.7 MJ/t/d at full fill, 0.05-0.18 of the summer RM (32
  MJ/t/d), relatively more in winter. Applies only while deep and
  uncompensated.
- MACKEREL (no bladder): always negatively buoyant, lift from body tilt and
  fins (He & Wardle [A]: denser fish tilt more). A full reserve cuts the
  excess weight 53 -> 35 kgf/t lean (-34 %; -57 % squared) [D]. Lift cost at
  1 BL/s (0.35 m/s, 35 cm), L/D 5-10 and efficiency 0.2 (assumed [J], a
  lower bound as above): 8-16 MJ/t/d, i.e. 0.4-0.9 of the model's 18.4
  MJ/t/d floor [D, wide]. Saving at full fill ~2.7-9 MJ/t/d = 0.15-0.5 of
  RM. No measured S. scombrus density-vs-fat or lift-cost data found: order
  of magnitude only. The model's mackerel rest cost (4.59 MJ/t per 6 h, the
  obligate swimming floor) already contains lift at some unknown mean
  fatness, so a reserve term would MODULATE that floor around the mean fill
  (cheaper when fat, dearer when lean), not add to RM.

### M3 Drag / girth
- Volume added by the store at full fill: gadoid +5.9 %, herring +14 %,
  mackerel +20 % of the lean volume; at fixed length the wetted area grows
  as sqrt(volume): +2.9, +6.9, +9.7 % -> friction drag up by about that
  fraction per fish, i.e. per tonne lean, at a given speed [D].
- As a share of RM this is (fraction) x (swimming drag cost / RM): cod at
  routine 0.25 BL/s swims little, so <~0.5 % of RM; herring and mackerel at
  ~1 BL/s, with thrust cost of order 0.5-1 RM (assumed [J]), +3-10 % of
  RM [D].
- Measured: cod COT per kg FELL with better condition at 1.3-1.5 BL/s and
  was unchanged at other speeds [A item 2]; Ucrit rose with condition in cod
  [A items 2, 3] and did not differ in juvenile Pacific herring whose lipid
  barely differed (7-9 % wet) [F PROXY item 11; weak]. In sharks a big liver costs drag only above
  ~1.5 m/s; below that the lift saving dominates [F PROXY item 10].
- Net hydrodynamic effect (M2 + M3): mackerel and deep herring NET BENEFIT
  at cruising speed (lift saving 0.15-0.5 / 0.05-0.18 RM vs girth cost
  ~0.03-0.1 RM); gadoids ~0.

### M4 Making and retrieving the store
- Deposition: 1/kL 1.10-1.31 for lipid vs 1/kP 1.79-1.90 for protein in
  three marine fishes [A PROXY item 17]: laying down lipid costs 10-31 %
  extra per kJ deposited (9-24 % of the input lost; protein 79-90 % extra,
  44-47 % lost). Lower end for dietary lipid stored directly, upper end
  includes synthesis from other substrates [J].
- Retrieval: ~2 % (fatty acid activation, 2 of ~106 ATP per palmitate) [D].
  Trout re-esterify >2/3 of the fatty acids they release, at rest [A PROXY
  item 16]: the store turns over continuously; its running cost relative to
  RMR was not found, and whether it scales with store size is unknown.
- These are one-off costs per MJ cycled through the store (a round-trip
  efficiency ~0.75-0.89 [D] = kL 0.76-0.91 x ~0.98 retrieval), not a daily
  carrying cost. Double counting:
  the model already charges SDA (benthivores 0.10, other fish 0.11-0.12 of
  ingested energy) and a growth_cost on benthivore new tissue; fish SDA
  includes biosynthesis, so part of the deposition cost may already be in
  the SDA. Whether the model's reserve deposition is 1:1 with assimilated
  energy should be checked in the engine before adding a kL.
- Value loss over time: no evidence of a passive decay of stored lipid
  beyond the cycling above [J].

## Verdict per species class (carrying cost of reserve per unit fill;
fill 1 = the class maximum above; RM = resting metabolism)

- GADOIDS (cod, haddock, whiting, saithe, Norway pout, hake): sign COST,
  small. The liver's own respiration: ~+5 % of RM at full fill (range
  0-11 %), linear in fill to first order -> ~0.05 RM per unit fill, i.e.
  ~0.0004 MJ/t/d per MJ/t of reserve (0.03-0.04 % of the store per day).
  Buoyancy (benefit <= 0.1 %) and drag (cost < 0.5 %) are negligible. The
  evidence is indirect: PROXY liver respiration and an assumed respiring
  non-lipid matrix; the one direct cod test (Lapointe 2006) sees no change
  of SMR per wet kg - consistent with this size, not a test of it.
- HERRING / SPRAT: metabolic cost 0-2 % of RM per unit fill. Hydrodynamics
  dominate and change sign with place: near the surface ~0 (gas replaces
  fat); at depth (compressed physostome bladder) a NET BENEFIT of ~0.05-0.18
  RM per unit fill (lift saving) minus a girth cost of ~0.03-0.07 RM. Net
  sign: ~0 to a modest benefit. No evidence of a carrying COST.
- MACKEREL: metabolic cost 0-3 % of RM per unit fill; girth cost ~0.05-0.1
  RM; lift saving ~0.15-0.5 RM. Net sign: BENEFIT (a fat mackerel is cheaper
  to keep up), size very uncertain (order-of-magnitude arithmetic, no
  Scomber scombrus measurement found), and it reverses at high speed and in
  bursts (Gleiss 2017, PROXY).
- All classes: the cost of filling and emptying the store (deposition
  loss 9-24 % of the input, retrieval ~2 %) is larger than any daily carrying cost over
  a season, and is a per-MJ-cycled cost, not a per-tick cost of holding.

## Gaps
- No full text for the cod condition papers (Lapointe 2006, Martinez
  2003/2004) - their HSI and per-tissue data could tighten the gadoid
  number; no direct cod/gadoid liver tissue respiration per g (lean vs
  fatty liver).
- No herring or mackerel SMR vs lipid study; no measured herring lean
  density (derived from Brawn's secondary numbers) or S. scombrus density
  vs fat (He & Wardle full text would have it).
- Lift and drag costs are first-principles arithmetic with assumed L/D and
  efficiency; the mackerel and deep-herring benefits span a factor ~3.
- The running cost of TAG:FA cycling relative to RMR and its scaling with
  store size were not found.
- All gadoid evidence is cod (Strand, Lapointe, Martinez, Davie); nothing
  for haddock, whiting, saithe, Norway pout or hake, and nothing for sprat
  (herring and Pacific herring stand in).
- Named in the task but not consulted: Pedersen & Hislop (North Sea energy
  densities), Jobling (fish bioenergetics texts), Fassler et al. (herring
  bladder, no OA), Magnuson 1978 (Magnuson 1973 used instead), a salmonid
  SMR-vs-lipid study (none found in two searches).
