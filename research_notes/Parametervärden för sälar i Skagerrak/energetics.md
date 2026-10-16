# Harbour seal (and grey seal) energetics for the Skagerrak seal group

Conventions used in every "Inferences" block below (my conversions, not literature):

- 1 kcal = 4.184 kJ; 1 W = 86.4 kJ/day; 1 L O2 = 19.7-20.1 kJ (Paterson et al. 2021 use 19.7, Howard 2009 uses 20.1).
- Kleiber (1975) BMR = 70 kcal kg^-0.75 d^-1 = 3.39 W kg^-0.75 = 0.293 MJ kg^-0.75 d^-1.
  Mass-specific Kleiber: 41.8 kg -> 0.115 MJ/kg/d; 44.7 kg -> 0.113; 63 kg -> 0.104; 80 kg -> 0.098; 84 kg -> 0.097; 89 kg -> 0.095.
- Model units: MJ per tonne of seal = kJ per kg; 4 ticks/day, 1460 ticks/yr. X MJ/kg/day = 250*X MJ/t/tick.
- Current library values (fg_library.yaml): resting_metabolism 45 MJ/t/tick (= 0.18 MJ/kg/d), feeding_cost 1.3, movement_cost 1.5, max_energy_reserve 5200 MJ/t, maintenance_level 0.5, starve_rate 0.03, energy_content 10000 MJ/t, handling_time 20 (ceiling 0.05 t/t/tick = 20 % BM/d), assimilation 0.86 (gadoids, energy 5500 MJ/t) and 0.92 (pelagic fish/herring, lean 4000 + reserve up to 7000 MJ/t).
- "Net" below means gross energy (GE) intake minus faecal, urinary and heat-increment-of-feeding (HIF) losses; Olesiuk (1993) takes these as 30 % of GE (6 % faeces + 7 % urine + 17 % HIF).

## 1. Resting / basal metabolism and the REST cost (resting_metabolism)

### Takeaway
Measured harbour seal RMR/BMR is about 1.1-1.6x Kleiber (mean 1.37x); hauled-out captive adult males (~84 kg, fed, post-moult) consume 0.53-0.70 L O2/min, i.e. 1.85-2.2x Kleiber = 45-52 MJ/t/tick. The model's 45 MJ/t/tick is therefore a "resting but fed, hauled-out" cost for an ~80 kg adult (1.84x Kleiber), not a basal cost; at the Skagerrak population mean mass (41.8 kg) it is only 1.56x Kleiber.

### Cited Findings
- Review of marine-mammal metabolic rates: harbour seal BMR/RMR = 1.37 +/- 0.17 (range 1.1-1.6) x Kleiber (sources: Matsuura & Whittow 1973; Innes 1984; Davis et al. 1985; Rosen & Renouf 1998); grey seal 1.26 +/- 0.47 (0.7-2.3) x Kleiber; phocid grand mean 1.32 +/- 0.31. Mass-specific RMR of harbour seals declines with age (Rosen & Renouf 1998). Body masses are in the review's supplementary appendices (not read). — [Review, PMC10545007](https://pmc.ncbi.nlm.nih.gov/articles/PMC10545007/)
- Paterson et al. 2021 (captive wild-caught male harbour seals, Scotland; adults 85.6 +/- 6.0 kg during moult, 84.2 +/- 4.9 kg post-moult; one subadult 61-63.5 kg; April-October): post-haul-out VO2 at 0 min 0.70 L/min (moult) vs 0.64 L/min (post-moult); peak at ~40 min 0.90 vs 0.70 L/min; at 180 min 0.65 vs 0.53 L/min. Mean over 3 h: 0.76 L/min (moult) vs 0.62 L/min (post-moult); the moult adds ~500 kJ per haul-out (+18.4 %). Animals were fed ad lib before trials; 29-35 % of measurements were post-absorptive (>15 h after feeding). Air temperature 13.5-16.9 C in the chamber. — [Paterson et al. 2021, PBZ 94:152-161](https://research-repository.st-andrews.ac.uk/bitstream/10023/25035/1/Moss_2021_PBZ_Increased_metabolic_rate_VoR.pdf)
- The same paper notes that RMR has been reported LOWER during the moult in harbour seals (Ashwell-Erickson et al. 1986; Rosen & Renouf 1998) but those animals were post-absorptive or fasting; and that captive harbour seals voluntarily reduce food intake during the moult (Rosen & Renouf 1998). — [Paterson et al. 2021](https://research-repository.st-andrews.ac.uk/bitstream/10023/25035/1/Moss_2021_PBZ_Increased_metabolic_rate_VoR.pdf)
- Captive harbour seals (P. v. concolor, Newfoundland): RMR varied seasonally from 45 to 129 MJ/week and accounted for 10-90 % of available energy over the year; adult body mass varied 16-30 % (15-32 kg) over the year. — [Rosen 1995 PhD thesis abstract](https://memorial.scholaris.ca/items/34a99c4f-8bfb-4e29-a209-2354848b7fd4)
- "Metabolic rates of adult harbor seals resting in water and measured in captivity were 1.2 times the Kleiber prediction (Hoelzel 2002)"; Howard used a phocid BMR relation BMR = 1.93 M^0.87 and, for resting in water, 1.9 x Kleiber (from grey seals, Sparling & Fedak 2004). Hauled-out metabolic rate of wild harbour seals has never been measured; ~75 % of haul-out time is spent sleeping (Sullivan 1979). — [Howard 2009 MSc thesis, pp. 26-29](https://faculty.aceveda.wwu.edu/PDFs/theses/Howard%20thesis.pdf)
- Davis et al. (1985): resting metabolic rate of an adult harbour seal (63 kg) was 1.1x expected BMR. — as reported in [Olesiuk 1993, Fish. Bull. 91:491-515](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1993/913/olesiuk.pdf)
- Grey seals (captive, adult females): RMR was highest in spring and declined through summer, not explained by water temperature; surface apnoea lowered RMR estimates ~25 % and periodic submergence ~35 %. — [Sparling PhD thesis abstract, St Andrews](https://research-repository.st-andrews.ac.uk/handle/10023/3722)

### Inferences
- Paterson's hauled-out values converted: 0.62 L/min x 19.7 kJ/L = 17.6 MJ/d / 84 kg = 0.21 MJ/kg/d = 52 MJ/t/tick (2.2x Kleiber); the lowest value (0.53 L/min at 180 min) = 15.0 MJ/d = 0.18 MJ/kg/d = 45 MJ/t/tick (1.85x Kleiber); moulting mean 0.76 L/min = 0.25 MJ/kg/d = 63 MJ/t/tick (2.6x Kleiber).
- Review BMR 1.37x Kleiber at 80 kg = 0.134 MJ/kg/d = 34 MJ/t/tick (range 1.1-1.6x -> 27-39 MJ/t/tick).
- The library's 45 MJ/t/tick = 0.18 MJ/kg/d = 1.84x Kleiber at 80 kg, 1.56x at the Skagerrak mean mass 41.8 kg. It coincides almost exactly with the measured cost of a fed adult harbour seal lying on a haul-out (Paterson, 180 min post-haul-out). It is ~30 % above true BMR, which is reasonable because the model's REST tick also has to cover digestion spill-over, arousals and haul-out entries/exits.
- Which mass does the FG represent? A population's per-tonne cost is total energy / total biomass, which equals per-capita energy / per-capita mass; that ratio is dominated by juveniles' elevated rates (Olesiuk: juvenile maintenance up to 1.8x adult at weaning, see section 2). So the per-tonne REST cost of a whole population sits above an adult's: 45 is adult-like; 50 is a defensible population value.

### Gaps
- No direct RMR measurement of wild or Skagerrak harbour seals; the Rosen & Renouf (1998) Can J Zool paper with mass-specific numbers was not retrievable (only the thesis abstract).
- Rosen thesis gives RMR in MJ/week but the abstract does not state the body masses, so it cannot be converted to MJ/kg.

## 2. Field metabolic rate and the daily energy budget (does the 45 / 1.3 / 1.5 triplet have the right level?)

### Takeaway
Free-ranging harbour seal FMR is 2.3-6.0x Kleiber (mean 3.76x); bioenergetic models put the population's per-tonne gross requirement at 0.33 MJ/kg/d (Strait of Georgia) to 0.47 MJ/kg/d (Skagerrak, Harkonen & Heide-Jorgensen 1991). With HIF charged to eat ticks (porpoise convention) the model's expenditure should match metabolisable intake (~0.87 x gross) minus ~5 % for growth/reproduction: 0.26-0.39 MJ/kg/d. The current triplet gives 0.22-0.25 MJ/kg/d, i.e. at or slightly below the low (Olesiuk) end.

### Cited Findings
- Reilly & Fedak (1991), free-living male harbour seals in the mating season (n = 3, each caught twice; DLW for one animal): 52.5 MJ/day, "about six times" Kleiber BMR; all three seals were in negative energy balance. Body masses not given in the abstract. — [Reilly & Fedak 1991, J Zool 223:461-468](https://research-portal.st-andrews.ac.uk/en/publications/rates-of-water-turnover-and-energy-expenditure-of-free-living-mal/)
- Review: harbour seal FMR = 3.76 +/- 1.65 (2.3-6.0) x Kleiber (Davis et al. 1985; Reilly & Fedak 1991; Bowen et al. 1992; Coltman et al. 1998); grey seal FMR = 3.24 +/- 1.99 (1.3-4.6) x Kleiber (Anderson & Fedak 1985; Sparling et al. 2008). The review also computes for harbour seals a swimming cost of 2.8x Kleiber vs an at-sea FMR of 5.0x Kleiber. — [Review, PMC10545007](https://pmc.ncbi.nlm.nih.gov/articles/PMC10545007/)
- DLW validated against simultaneous respirometry in wild grey seals held 4-5 days in simulated foraging: mean difference 0.7 %, individual errors -39 % to +44 %. — [Sparling et al. 2008, Funct Ecol 22:245-254](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/j.1365-2435.2007.01368.x)
- Harkonen & Heide-Jorgensen (1991), Skagerrak harbour seal model: mean per-capita gross requirement 227 W at a mean body mass of 41.8 kg; 73.1 % of metabolisable energy to maintenance, 19.0 % activity, 2.0 % growth, 4.5 % reproduction, 1.5 % moult. Olesiuk attributes their 32 % higher estimate (vs his 172 W) to scaling juvenile requirements linearly with mass (with a 1.4 correction) rather than to M^0.75. — as reported in [Olesiuk 1993](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1993/913/olesiuk.pdf); the original ("The harbour seal Phoca vitulina as a predator in the Skagerrak", Ophelia 34:191-207, 1991) is listed in [marineinfo.org](https://marineinfo.org/doc/publication/17893) (the search summary showed Harkonen as sole author there, whereas Olesiuk, Paterson and Marine Scotland cite Harkonen & Heide-Jorgensen; not resolved, full text not read)
- The Marine Scotland harbour seal consumption estimate uses "the estimated average daily energy requirement of 4,680 Kcals (Harkonen and Heide-Jorgensen 1991)". — [Marine Scotland SMFS 7(21)](https://data.marine.gov.scot/sites/default/files/SMFS%20Vol%207%20No%2021.pdf). Paterson et al. 2021 cite the same work as "daily energy requirement (20,000 kJ) of an adult harbor seal" — [Paterson et al. 2021](https://research-repository.st-andrews.ac.uk/bitstream/10023/25035/1/Moss_2021_PBZ_Increased_metabolic_rate_VoR.pdf). Note the conflict: Olesiuk reports 227 W as a per-capita population mean at 41.8 kg, Paterson calls it an adult value.
- Olesiuk (1993), Strait of Georgia (population growing 12.5 %/yr, skewed to young): mean per-capita gross requirement 172 W at mean mass 44.7 kg; 30 % lost in faeces, urine and HIF, 42.3 % basal metabolism, 23.4 % activity, 1.2 % growth, 3.2 % reproduction. Daily energy requirement ranged from 150 W (yearlings) to 215 W (full-grown males; male growth-curve asymptote 89.16 kg). Captive-derived gross maintenance of adults = 1.85-1.94x (mean 1.90x) BMR; activity-budget-derived gross maintenance (40 % hauled out at BMR, 60 % swimming at 2.3x BMR) = 2.5x BMR (net 1.8x). Juvenile maintenance assumed to fall from 1.8x adult at the onset of feeding to adult levels at maturity; gross maintenance (MR1) of seals aged 0-3 yr was 2.9x predicted adult BMR, similar to 2.8x BMR in captive harbour seals aged 0-3 yr (Markussen et al. 1990). Lactating females' requirement was 2.8x that of non-lactating females of equal mass. — [Olesiuk 1993](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1993/913/olesiuk.pdf)
- Ashwell-Erickson & Elsner (1981), Bering Sea harbour seals: 216-238 W per capita at mean mass 67.7 kg. — as reported in [Olesiuk 1993](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1993/913/olesiuk.pdf)
- Howard (2009), San Juan Islands harbour seals: per-capita NET energy use 112 +/- 12 W (breeding season) and 108 +/- 14 W (non-breeding); masses 24-66 kg (females), 24-89 kg (males). Time budgets (TDR): females breeding season 16.5 % hauled out / 43.6 % diving / 40.0 % shallow water; non-breeding 16.9 / 67.0 / 16.2 %. — [Howard 2009](https://faculty.aceveda.wwu.edu/PDFs/theses/Howard%20thesis.pdf)
- Grey seals, UK: "estimated average daily energy requirement of 5,497 Kcals (Sparling and Smout 2003)" per seal, used for North Sea consumption estimates (body mass not stated in the report). — [Marine Scotland SMFS 7(20)](https://data.marine.gov.scot/sites/default/files/SMFS%20Vol%207%20No%2020.pdf)

### Inferences
- Conversions of the budgets (per tonne of population = per capita / per-capita mass):
  - H&HJ 1991 Skagerrak: 227 W = 19.6 MJ/d gross / 41.8 kg = 0.47 MJ/kg/d gross (117 MJ/t/tick; 4.1x Kleiber). ME (x0.87) = 0.41; net of HIF (x0.70) = 0.33 MJ/kg/d (82 MJ/t/tick; 2.85x Kleiber).
  - Olesiuk 1993: 172 W = 14.9 MJ/d / 44.7 kg = 0.33 MJ/kg/d gross (2.9x Kleiber); ME 0.29; net 0.23 MJ/kg/d (58 MJ/t/tick; 2.05x Kleiber).
  - Olesiuk full-grown male: 215 W = 18.6 MJ/d / ~89 kg = 0.21 MJ/kg/d gross; net 0.146 MJ/kg/d (37 MJ/t/tick; 1.5x Kleiber).
  - Howard: 110 W net = 9.5 MJ/d; with masses 24-89 kg the per-kg value is ~0.2 MJ/kg/d (population mass not extracted; rough).
  - Grey seal: 5,497 kcal/d = 23.0 MJ/d gross (mass unknown; at a guessed 150 kg this would be 0.15 MJ/kg/d - unverified).
  - Reilly & Fedak: 52.5 MJ/d at ~6x Kleiber implies a ~90 kg male, i.e. ~0.58 MJ/kg/d during the mating season - an upper bound, animals were losing mass.
- Model expenditure, triplet weighted by a time budget (REST = hauled out, EAT = diving, MOVE = shallow-water/transit):
  - Olesiuk 40 % rest / 60 % at sea (half eat, half move): (0.4 + 0.3*1.3 + 0.3*1.5) = 1.24 x 0.18 = 0.22 MJ/kg/d (56 MJ/t/tick).
  - Howard non-breeding 17/67/16, current eat 1.3 / move 1.5: 1.28 x 0.18 = 0.23 MJ/kg/d.
  - Same budget, eat 1.5 / move 1.3: 1.38 x 0.18 = 0.25 MJ/kg/d; with resting_metabolism 50: 0.28 MJ/kg/d.
- Comparison basis: if feeding_cost carries the HIF (porpoise convention) and assimilation_factor is ME, the model's total spend should equal ME intake minus production (~5 %): Olesiuk 0.29 x 0.95 = 0.27; H&HJ 0.41 x 0.95 = 0.39 MJ/kg/d. The current triplet (0.22-0.25) is ~10-20 % below Olesiuk and ~40 % below H&HJ. If instead HIF were folded into assimilation, compare with "net" values (0.23-0.33): the current triplet is at the low end.
- Growth and reproduction are only 4.4 % (Olesiuk) to 6.5 % (H&HJ) of the budget; they belong in the derived growth_rate, not in resting_metabolism.
- The model's realised time budget is learned by the policy; if trained seals rest much more than 17-40 % of ticks, their spend falls further below the literature.

### Gaps
- No DLW FMR for harbour seals outside the mating season and none for the Skagerrak; Reilly & Fedak's masses not available from the abstract.
- Absolute grey seal FMR values of Sparling et al. 2008 and Sparling & Smout 2003 (body mass behind 5,497 kcal/d) not retrieved.
- The H&HJ 1991 paper itself was not available; its loss fractions (faeces/urine/HIF) are unknown, so the 0.70 net factor is Olesiuk's, applied by me.

## 3. Cost of foraging vs travelling vs hauled-out rest; heat increment of feeding (feeding_cost, movement_cost)

### Takeaway
Pure locomotion (swimming at the preferred 1.4 m/s) costs 2.2x RMR (2.3-2.4x BMR), and grey seal travelling dives cost more than foraging dives, but diving itself is cheap (1.6-1.7x Kleiber, below surface rest) and feeding raises diving metabolism 1.4-1.6x; HIF is up to ~15-17 % of gross energy and lasts ~15 h. Whether eat > move therefore depends on where the model charges HIF: if the eat tick carries it (porpoise convention), the literature supports eat >= move, roughly eat 1.4-1.6 and move 1.2-1.4 relative to the 45 MJ/t/tick rest.

### Cited Findings
- Davis et al. (1985): swimming metabolic rate of a 63 kg adult harbour seal at its preferred 1.4 m/s was 2.3x expected BMR, i.e. 2.2x its RMR (RMR = 1.1x BMR). Williams (1987): net SMR of an 85 kg adult at the same speed 210 W = 2.4x predicted BMR. A 33 kg yearling at 1.4 m/s: 170 W (3.6x adult-equivalent BMR, 2.3x yearling-corrected BMR). — as reported in [Olesiuk 1993](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1993/913/olesiuk.pdf)
- Grey seals, captive, voluntary diving (n = 8): diving metabolic rate (DMR) ~1.7x predicted standard metabolic rate for terrestrial animals of equal size, lower than the rate measured while resting at the surface for every animal; DMR falls with dive duration and rises with swim speed. — [Sparling & Fedak 2004, J Exp Biol 207:1615-1624 (abstract via St Andrews portal)](https://research-portal.st-andrews.ac.uk/en/publications/metabolic-rates-of-captive-grey-seals-during-voluntary-diving/)
- Sparling thesis: mean DMR 1.6x predicted BMR; "travelling dives cost more energy than foraging dives"; feeding during a dive raised DMR to 1.4-1.6x pre-feeding levels. — [Sparling PhD thesis abstract](https://research-repository.st-andrews.ac.uk/handle/10023/3722)
- Grey seals show very high metabolic rates during extended surface intervals, mostly at night and often many hours after feeding, interpreted as digestion deferred until after foraging ("eat now, pay later"). — [Sparling, Fedak & Thompson 2007 (St Andrews portal)](https://research-portal.st-andrews.ac.uk/en/publications/eat-now-pay-later-evidence-of-deferred-food-processing-costs-in-d/)
- Markussen, Ryg & Oritsland (1994), harbour seals fed herring: HIF up to 14.9 % of gross energy intake; metabolic rate up to 46 % above post-absorptive RMR for ~15 h in a ~40 kg seal; magnitude and duration depend on meal size; herring energy 6,575-12,560 kJ/kg depending on season. A later paper cites 5.1 % of GE for high-energy herring (secondary citation, unverified). — [Markussen et al. 1994, J Comp Physiol B (abstract via search)](https://link.springer.com/article/10.1007/BF00301648)
- Olesiuk used HIF = 17 % of GE (Gallivan & Ronald 1981, harp seals); Howard used 1.8-13.7 % for prey of ~3,000-11,000 J/g, stating HIF is inversely related to prey energy density and "can be up to 15 % when harbor seals consume fish" (Markussen et al. 1994). — [Olesiuk 1993](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1993/913/olesiuk.pdf); [Howard 2009](https://faculty.aceveda.wwu.edu/PDFs/theses/Howard%20thesis.pdf)
- Olesiuk argues thermoregulation is covered by waste heat: 17 % of GE released as HIF plus 75 % of the 23 % of GE spent swimming (muscle ~25 % efficient) = ~59 W for an average seal. — [Olesiuk 1993](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1993/913/olesiuk.pdf)
- Hauled-out cost rises during the moult (+18.4 % over a 3 h haul-out) and the first ~40 min after hauling out are the most expensive. Haul-out share: 0.72 of time during the moult vs 0.34 post-moult (Lonergan et al. 2013; Cunningham et al. 2009). — [Paterson et al. 2021](https://research-repository.st-andrews.ac.uk/bitstream/10023/25035/1/Moss_2021_PBZ_Increased_metabolic_rate_VoR.pdf)

### Inferences
- Relative to the model REST (1.84x Kleiber at 80 kg):
  - Transit swimming 2.3-2.4x Kleiber -> move ~1.25-1.3 (Davis/Williams). The review's 2.8x Kleiber "swimming" -> 1.5.
  - Foraging dive without digestion 1.6-1.7x Kleiber -> ~0.9; with the feeding increment (x1.4-1.6) -> 2.2-2.7x Kleiber -> eat ~1.2-1.5.
  - HIF budget: at a gross intake of 0.33-0.47 MJ/kg/d (section 2) and HIF 10-17 %, HIF = 0.033-0.08 MJ/kg/d = 18-44 % of the daily REST cost. Charged entirely to eat ticks that make up ~50-67 % of the day, that alone adds +0.3 to +0.9 to the eat multiplier. Even with half of it deferred into rest ticks ("eat now, pay later"), eat >= 1.4 follows.
  - Suggested: feeding_cost 1.5 (range 1.4-1.8), movement_cost 1.3 (range 1.2-1.5) - the porpoise ordering. The current library ordering (eat 1.3, move 1.5) is only defensible if HIF is NOT charged to eat ticks (i.e. if assimilation_factor is net of HIF).
- Haul-out rest is not much cheaper than rest at sea in this species (hauled-out fed adults 1.85-2.2x Kleiber vs 1.2-1.9x in water), so a single REST cost for "rest/hide/haul-out" is acceptable.

### Gaps
- No harbour-seal-specific diving metabolic rate; grey seal DMR is the proxy everyone uses (Howard 2009).
- Howard prints the Sparling & Fedak DMR equation as "0.016 M^0.76 ml O2/min", which cannot be right in those units for an 80 kg seal; the absolute equation was not verified and is not used here.
- Two "resting in water" figures conflict and should not be averaged: 1.2x Kleiber (harbour, captive, Hoelzel 2002 via Howard) vs 1.9x (grey seal, surface, fed, Sparling & Fedak via Howard) - different species and feeding state.

## 4. Daily food intake, typical and maximum (intake ceiling, handling_time)

### Takeaway
Routine rations are 3-5 % of body mass per day for adults and up to ~8 % for yearlings (energy-based models), and captive adults eat ~2.6-3.3 kg/day (~1000-1200 kg/yr). The highest sustained mass-specific gross intake found is ~0.57 MJ/kg/d (spotted seal yearlings, upper quartile). The library ceiling of 20 % BM/day gives 1.1 (gadoids) to 1.5 (herring) MJ/kg/d gross, ~2-2.7x the highest routine intake found. With the porpoise rule (1.5x the highest ration) the ceiling should be ~0.85 MJ/kg/d gross: about 15 % BM/d on gadoids (h ~ 26) and ~11 % BM/d on herring (h ~ 35).

### Cited Findings
- Olesiuk 1993 (Strait of Georgia, diet mean 6.65 MJ/kg): energy-based daily food requirement 2.0 kg (yearlings, 8.2 % BM) to 2.8 kg (full-grown males, 3.1 % BM); population mean 2.2 kg = 5.0 % of mean body mass (FR1, energetic life tables). Stomach-volume estimate (FR2, from Boulva & McLaren 1979: FR = 0.089 M^0.76) = 1.6 kg (73 % of FR1); adults 2.1 kg (stomachs) vs 2.5 kg (energetics). The abstract gives "1.9 kg, or 4.3 % of mean body mass" (apparently the mean of the two). Boulva & McLaren found no seasonal pattern in the share of empty stomachs. Even a seasonal blubber swing of 21 <-> 55 % would only move daily requirement to 125 % / 75 % of the annual mean. — [Olesiuk 1993](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1993/913/olesiuk.pdf)
- Howard (2009) model: predicted consumption 1.0-2.9 kg per seal per day, highest for adult females. — [Howard 2009](https://faculty.aceveda.wwu.edu/PDFs/theses/Howard%20thesis.pdf)
- Kastelein et al. 2005 (captive harbour seals, Harderwijk): annual food consumption of males rose from ~950 kg at age 1 to ~1,200 kg at age 13; non-reproductive females ~1,000 kg/yr; intake as % of body mass declined with body mass; reproduction amplified seasonal swings in female intake; immatures showed no seasonal change; seals usually lost mass during the moult. Diet energy was not measured. — [Kastelein, Staal & Wiepkema 2005, Aquatic Mammals 31:34-42 (abstract via search)](https://www.aquaticmammalsjournal.org/article/vol-31-iss-1-kastelein/)
- Captive harbour seals (Rosen 1995): gross energy intake ranged 30-300 MJ/week over the year (a tenfold seasonal range). — [Rosen 1995 thesis abstract](https://memorial.scholaris.ca/items/34a99c4f-8bfb-4e29-a209-2354848b7fd4)
- Spotted seals (Phoca largha, sister species of P. vitulina), captive, up to 9 yr: mass-specific gross energy intake median 109.6 kcal/kg/d in year 1 (quartiles 92-135), 87-90 in years 2-4, ~62 in years 5-9 (quartiles ~43-86). By year 9 the monthly maximum intake was 2.5x the following minimum (mass-specific max 380 % of min). Hyperphagia in late spring/early summer, hypophagia in autumn-winter; intake peaks often coincided with mass declines. — [Rosen et al. 2021, Aquatic Mammals 47:559-573](https://mmru.ubc.ca/wp-content/pdfs/Rosen%202021%20AM%2047.6_1.pdf)
- Lactating harbour seals: food intake rose sharply by late lactation, less so in heavier mothers. — [Bowen et al. 2001, Funct Ecol (Smithsonian record)](https://repository.si.edu/items/aa8cc841-8293-4376-b656-995dac687ba7)
- Harbour seals eating lean pollock ate more by mass (greater gut fill) and still obtained more digestible energy than on herring - a flexible gut compensates for low-energy prey; lipid digestibility of herring fell from 90 % to 50 % when lipid intake exceeded 60 g kg^-0.75 d^-1. — [Trumble 2003 thesis (abstract via search)](https://scholarworks.alaska.edu/handle/11122/8669)

### Inferences
- Gross-energy routine rations converted: H&HJ 0.47 MJ/kg/d, Olesiuk 0.33 MJ/kg/d (population means, i.e. per tonne of population); spotted seal adults ~0.26 MJ/kg/d (median), yearlings 0.46 (median) to 0.57 (upper quartile).
- Break-even rations at the model's spend (0.25-0.28 MJ/kg/d, section 2) with ME ~0.87: herring at 6 MJ/kg -> 4.8-5.4 % BM/d; at 7.5 MJ/kg -> 3.8-4.3 %; at 11 MJ/kg -> 2.6-2.9 %; gadoids at 4.0 MJ/kg -> 7.2-8.0 %; at 5.5 MJ/kg (library) -> 5.2-5.9 %. Consistent with the 3-8 % routine rations above.
- Ceiling: 1.5 x 0.57 MJ/kg/d = 0.85 MJ/kg/d gross. In mass: gadoids (5.5 MJ/kg) 15.5 % BM/d = 0.039 t/t/tick -> handling_time ~26; herring at reference fill (~7.5 MJ/kg) 11.3 % BM/d = 0.028 t/t/tick -> handling_time ~35. A single h for both pairs: 25 (range 20-35). The current h = 20 (20 % BM/d) lets a seal take in 1.1-1.5 MJ/kg/d gross, ~2-2.7x anything routine that I found, and ~4x the model's eat-tick cost on gadoids (0.05 x 5500 x 0.86 = 237 vs 58.5 MJ/t/tick).
- Seasonality in intake is large (2.5-10x between monthly extremes) but is endogenous (moult, breeding) rather than prey-driven; a fixed ceiling cannot express it.

### Gaps
- No measured single-day maximum intake for harbour seals was found; Kastelein 2005 full text (daily maxima, body masses) and Markussen et al. 1990 (ad-lib intake in captive harbour seals) were not accessible. Bekkby & Bjorge 1998 (meal size from stomach temperature) only as a citation.
- The "5-6 % (4.5-8.2 kg)" and "7-10 %" figures on popular sites (SeaWorld, Whale Trail) are uncited and were not used.

## 5. Energy reserve, blubber, fasting tolerance and starvation (max_energy_reserve, maintenance_level, starve_rate)

### Takeaway
Adult harbour seals carry ~27-30 % of body mass as blubber (extremes 21-55 %), blubber holds 37.8 MJ/kg, and the seasonal swing of adult body fat is ~11 percentage points (Alaska). That swing is worth ~4,000-4,200 MJ per tonne of seal, and lactation draws down ~17 points (~6,500 MJ/t); the library's 5,200 MJ/t (~14 % of BM as blubber) is in the middle. Fasting harbour seals depress RMR ~20 % within 24 h and take ~50 % of lost mass (~75-77 % of energy) from fat. True fasts last ~2-3 weeks (pups post-weaning 15-17 d; grey seal mothers 18-21 d at ~2 % BM/d).

### Cited Findings
- Blubber energy density 37.8 MJ/kg, proteinaceous tissue 6.5 MJ/kg (Olesiuk & Bigg unpubl.); adult body ~30 % blubber, "consistent with the 27-30 % reported for free-ranging harbor seals (Pitcher 1986)"; individual blubber content ranged between extremes of 21 % and 55 % of body mass (Pitcher 1986); blubber is thickest in winter, thins in summer, and females are thinnest by the autumn moult; at weaning ~50 % of pup body mass is blubber; nursing females modelled to use an extra 16.8 kg of blubber (22 % of body mass) during lactation. — [Olesiuk 1993](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1993/913/olesiuk.pdf)
- Alaska (harvested seals 1972-2004, n = 874; % body fat from sternal blubber depth): all cohorts leanest in September (during/after the moult); adults (>5 yr) fattest in January (+10.68 +/- 2.59 % vs September), subadults in March (+6.11 +/- 2.25 %), pups just after weaning in July (+7.15 +/- 2.41 %). — [Karpovich, Pendleton & Blundell (abstract)](https://www.adfg.alaska.gov/static/home/library/pdfs/wildlife/research_pdfs/differences_harbor_seal_percent_body_fat_predict_population_trends.pdf)
- Harbour seal lactation (Sable Island, 17 females, 24-day lactation): females lost about one third of their body mass in the first 19 days; maternal body fat fell from ~24 % at parturition to ~7 % at day 19. CAVEAT: this abstract text was found on a ResearchGate page carrying a harp seal title. — [Bowen, Oftedal & Boness 1992, Physiol Zool 65:844-866 (via ResearchGate)](https://www.researchgate.net/publication/32248088_Lactation_Performance_and_Nutrient_Deposition_in_Pups_of_the_Harp_Seal_Phoca_groenlandica_on_Ice_Floes_off_Southeast_Labrador); the 33 % figure is repeated in [a lactation review (IntechOpen)](https://www.intechopen.com/chapters/68765). Lactating harbour seals begin foraging trips by mid-lactation, so this is not a strict fast — [Bowen et al. 2001 record](https://repository.si.edu/items/aa8cc841-8293-4376-b656-995dac687ba7)
- Harbour seal pups after weaning: all lost body mass during the first 4-6 weeks while total body water rose (lean growth continued); heavier pups were fatter at weaning (41.0 % vs 37.1 % fat); estimated food intake rose in weeks 3-4 but stayed below expenditure for most pups. — [Muelbert, Bowen & Iverson 2003, PBZ (DalSpace)](https://dalspace.library.dal.ca/handle/10222/29092). A related Sable Island study: post-weaning fast averaged 15-17 days (stomach lavage); body composition unchanged in the first 14 days despite mass loss, then body fat declined to ~12 %. — [search summary pointing to Tethys record](https://tethys.pnnl.gov/publications/duration-lactation-postweaning-changes-mass-body-composition-harbor-seal-phoca)
- Fasting juvenile harbour seals: RMR 20 % lower than during normal feeding; ~50 % of mass lost was fat, supplying ~77 % of the energy; metabolic rate returned to the initial value about 1 week after refeeding. — [Markussen, Ryg & Oritsland 1992, Can J Zool 70:220-224 (abstract via search)](https://www.researchgate.net/publication/238025962_Metabolic_rate_and_body_composition_of_harbour_seals_Phoca_vitulina_during_starvation_and_refeeding); [Markussen 1995 chapter](https://www.sciencedirect.com/science/article/abs/pii/S0163699506800405)
- Markussen (1995): "rapid (within 24 h) 20 % decrease in mass-specific metabolism in experimentally fasted harbour seals"; harbour seals fed 10 % of their normal diet showed metabolic rates equal to or greater than control (no depression); harbour seals lost mass at similar rates during fasting and food restriction; metabolism still depressed at least a week after refeeding. Comparison: Steller sea lions lost 1.5 % of body mass per day during 9-14-day fasts. — as reported in [Rosen & Trites 2002, Comp Biochem Physiol B 132:389-399](https://mmru.ubc.ca/wp-content/pdfs/metdeprostrites.pdf)
- Grey seal mothers (GREY SEAL): fast throughout an ~18-21-day lactation, losing ~40 % of initial mass and 61-84 % of fat reserves (Fedak & Anderson 1982; Pomeroy et al. 1999) — [PMC10469160](https://pmc.ncbi.nlm.nih.gov/articles/PMC10469160/); a recent estimate is 37 % of pre-partum mass (39 % in Iverson et al. 1993) — [Sanchez et al. 2026, Ecol Evol](https://onlinelibrary.wiley.com/doi/10.1002/ece3.73224); initial maternal mass 149-256 kg, daily mass loss 3-9 kg/d — [maternal size and milk output (ResearchGate)](https://www.researchgate.net/publication/32248074_The_effect_of_maternal_size_and_milk_output_on_pup_growth_in_grey_seals_Halichoerus_grypus)
- Pups of heavier harbour seal mothers survived better. — [Bowen et al. 2001 record](https://repository.si.edu/items/aa8cc841-8293-4376-b656-995dac687ba7)

### Inferences
- Mobilisable reserve per tonne of seal: Alaska adult seasonal swing 10.7 points of body fat x 37.8-39.3 MJ/kg = 4.0-4.2 MJ/kg = 4,000-4,200 MJ/t; harbour seal lactation 17 points (24 -> 7 %) = 6,400-6,700 MJ/t plus lean tissue; Olesiuk's 22 % BM of blubber over lactation = 8,300 MJ/t; Pitcher's extreme 55 -> 21 % = 12,900 MJ/t (extreme, not typical). 5,200 MJ/t = 13.8 % of BM as blubber; recommend keeping 5,200 (range 4,000-7,000).
- Energy density of mass lost while fasting (Markussen: 50 % fat by mass, 77 % of energy from fat) ~ 20-25 MJ/kg lost; at the model REST cost 0.18 MJ/kg/d that is ~0.8 % BM/d, rising to 1-2 % BM/d with activity - consistent with Steller sea lion (1.5 %/d) and grey seal mother (~2 %/d = 40 % / 18-21 d, my calculation) rates.
- Fasting time implied by the library: 5,200 / 180 = 29 days at pure REST from a full reserve (36 days with a 20 % metabolic depression, which the model does not implement); from the maintenance level (half reserve) ~14-18 days. This is the right order compared with real fasts of 15-21 days (pups, grey seal mothers), which happen at higher-than-resting cost.
- starve_rate 0.03 /tick = 11.4 % of biomass per day once the reserve is empty: biomass halves in ~6 days. No literature value exists for when starvation mortality sets in for adult seals; the only quantitative anchor is the earlier project target T_total 30-60 d (resume section 46), which 0.03 meets (~57 d). Keep 0.03 (range 0.02-0.05).
- maintenance_level 0.5 cannot be calibrated from the literature: seal appetite follows an endogenous calendar (hyperphagia in late spring/summer, voluntary hypophagia during moult and breeding, intake peaks coinciding with mass loss - Rosen et al. 2021, Paterson et al. 2021), not reserve level alone. Keep 0.5.

### Gaps
- No per-day mass-loss rate for fasting harbour seals (Markussen 1992 full text not available); no data on when starvation mortality begins in adults.
- Lipid content of harbour seal blubber (as opposed to 37.8 MJ/kg blubber from Olesiuk's unpublished data) not found.
- Bowen et al. 1992 numbers need checking against the original (wrong title on the page that carried them).

## 6. Whole-body energy density (energy_content)

### Takeaway
A 30 %-blubber adult carcass holds ~15.9 MJ/kg (Olesiuk). 10,000 MJ/t only fits if energy_content means "body without the mobilisable reserve", with the 5,200 MJ/t reserve stacked on top (the convention pelagic_fish already uses). Since nothing eats seals, the choice is cosmetic, but the convention should be named.

### Cited Findings
- Mean carcass energy density 15.9 (printed "KJ.kg-1", clearly MJ/kg) from 30 % blubber at 37.8 MJ/kg + 70 % proteinaceous tissue at 6.5 MJ/kg; foetus/neonate 7.2 MJ/kg (harp seal value, neonates nearly fat-free); mass gained by nursing pups 33.1 MJ/kg (~85 % fat). — [Olesiuk 1993](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1993/913/olesiuk.pdf)

### Inferences
- Whole-body values by fatness: 15 % blubber -> 11.2 MJ/kg; 21 % -> 13.1; 30 % -> 15.9; 40 % -> 19.0; weaned pup (50 %) -> 22.2 MJ/kg.
- Reading A (whole body): energy_content ~15,000-16,000 MJ/t (range 11,000-19,000).
- Reading B (lean + structural blubber, reserve added separately, as pelagic_fish's library comment describes): 0.15 x 37.8 + 0.85 x 6.5 = 11.2 MJ/kg, so 10,000-11,500 MJ/t, and 10,000 + 5,200 = 15,200 at a full reserve matches Olesiuk's 15.9. Under reading B the current 10,000 is right.

### Gaps
- No direct bomb-calorimetry of whole harbour seal carcasses was found; the 37.8 / 6.5 MJ/kg tissue values are Olesiuk & Bigg unpublished data.

## 7. Assimilation efficiency on gadoids vs herring

### Takeaway
Apparent digestible energy of fish is ~91-97 % in phocids, metabolisable energy (after ~6-7 % urinary loss) ~85-89 %. The library values sit on different bases: 0.92 (herring) is a digestibility value, 0.86 (gadoids) a metabolisable one. If feeding_cost carries the HIF, both pairs should be ME: herring ~0.88 (0.86-0.90), gadoids ~0.85 (0.83-0.87). The herring-gadoid gap is ~3 points (or none), not 6.

### Cited Findings
- Harp seals (12 captive), digestive efficiency: Atlantic cod 93.5 %, Arctic cod 93.5 %, halibut 94.7 %, capelin 95.7 %, herring 96.6 %; digestive efficiency positively correlated with prey energy density. — [Lawson, Miller & Noseworthy 1997, Can J Zool 75:1285-1291 (abstract via search)](https://cdnsciencepub.com/doi/10.1139/z97-152)
- Juvenile harp seals on Atlantic herring: assimilation efficiency 92.5-95.0 % of GE (72.2 % on shrimp); metabolisable energy 85.5-88.7 % of GE on herring; urinary energy loss rises directly with apparent digestible nitrogen intake. — [Keiver, Ronald & Beamish 1984, Can J Zool (abstract via search)](https://doi.org/10.1139/z84-110)
- Olesiuk assumed 13 % of GE (herring diet) not metabolisable: 6 % faeces + 7 % urine (Keiver et al. 1984), plus 17 % HIF -> 70 % overall efficiency. — [Olesiuk 1993](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1993/913/olesiuk.pdf)
- Harbour seals: assimilation efficiency 87.4-93.2 % (Trumble et al. 2003); digestive efficiency ~90 % in carnivores and 90.3 % for harbour seals on a mixed fish diet; with urinary losses the overall efficiency term is roughly 85 % of GE (Lavigne et al. 1982). — [Howard 2009](https://faculty.aceveda.wwu.edu/PDFs/theses/Howard%20thesis.pdf)
- Harbour seals took in more digestible energy from pollock than from herring (more gut fill, consistent dry-matter digestibility); herring lipid digestibility fell from 90 % to 50 % at lipid intakes above 60 g kg^-0.75 d^-1. — [Trumble 2003 thesis (abstract via search)](https://scholarworks.alaska.edu/handle/11122/8669)
- A Bering Sea energy-flow dissertation (1981; species and author not confirmed from the snippet) reported digestible energy of pollock 96.7 +/- 0.2 % vs herring 91.2 +/- 0.7 % of GE, and net energy ~80.3 % of GE for both. — [UAF thesis record (search snippet only)](https://scholarworks.alaska.edu/uaf_unassigned_theses/65)

### Inferences
- The ordering herring > gadoids is not unanimous (Lawson: herring 96.6 > cod 93.5; Trumble and the Bering Sea data: pollock >= herring). Lean, protein-rich gadoids also give higher urinary losses (urinary energy scales with digestible N, Keiver) and a higher HIF (inverse to prey energy density, Markussen/Howard), so their net value is lower even when apparent digestibility is equal.
- Recommended (ME basis, HIF in feeding_cost): seals_preys_on_pelagic_fish 0.88 (0.86-0.90); seals_preys_on_gadoids 0.85 (0.83-0.87). If the project wants HIF inside assimilation instead (and feeding_cost reduced to ~1.2-1.3): herring ~0.75-0.80, gadoids ~0.70-0.75.

### Gaps
- No harbour-seal ME trial on cod/whiting/saithe specifically; gadoid values are from harp seals (Atlantic cod) and harbour seals on walleye pollock.

## 8. Is temperature dependence (Q10) relevant for seals in 4-18 C water?

### Takeaway
No. Harbour seals are treated as thermoneutral in 6-17 C water in every bioenergetic model found, and seasonal RMR changes in both harbour and grey seals are not explained by water temperature. Give seals no Q10, like porpoises. There is still an endogenous seasonal signal (moult, breeding) that a Q10 would not capture.

### Cited Findings
- Olesiuk assumed negligible thermoregulatory cost: no change in harp seal swimming metabolism at 8.5-26 C (Oritsland & Ronald 1975) or resting metabolism at 1.8-28.2 C (Gallivan 1977); lower critical air temperature of harbour seals 2 C (Hart & Irving 1959); local sea temperatures 6.2-17.0 C. Hart & Irving's in-water lower critical temperatures (20 C summer, 13 C winter) came from restrained, probably post-absorptive seals; waste heat from HIF and swimming (~59 W) would cover any deficit. — [Olesiuk 1993](https://spo.nmfs.noaa.gov/sites/default/files/pdf-content/1993/913/olesiuk.pdf)
- Howard: water temperatures 7.1-16.6 C recorded by tagged seals, "a range in which seal metabolic rates are not expected to change"; thermoregulatory costs assumed minimal or met by HIF. — [Howard 2009](https://faculty.aceveda.wwu.edu/PDFs/theses/Howard%20thesis.pdf)
- Seasonal BMR variation in harbour seals (Rosen & Renouf 1998) and grey seals (Sparling et al. 2006) is attributed to moult, body mass, energy expenditure or food intake "rather than environmental temperatures per se". — [Review, PMC10545007](https://pmc.ncbi.nlm.nih.gov/articles/PMC10545007/); grey seal female RMR seasonality "was not explained by water temperature" — [Sparling thesis abstract](https://research-repository.st-andrews.ac.uk/handle/10023/3722)
- Captive harbour seal rectal temperature varied circannually by 2.0-2.8 C and tracked water temperature more closely than energy intake. — [Rosen 1995 thesis abstract](https://memorial.scholaris.ca/items/34a99c4f-8bfb-4e29-a209-2354848b7fd4)

### Inferences
- No metabolism_q10 for seals. The real seasonal pattern (RMR range 45-129 MJ/week in captivity, +18 % hauled-out cost during the moult, voluntary hypophagia at the moult, haul-out share 0.72 vs 0.34) is calendar-driven. If the model ever wants it, the right tool is a month-indexed multiplier on the daylight calendar, not a temperature Q10.

### Gaps
- No measurement of free-ranging harbour seal metabolism across water temperatures in the Skagerrak range.

## 9. Recommended seal parameters (summary for the model)

### Takeaway
Keep the level of the REST cost (45-50 MJ/t/tick), swap the activity multipliers to the porpoise ordering (eat 1.5, move 1.3) if HIF lives in feeding_cost, tighten the intake ceiling to ~12-16 % BM/day, put both assimilation factors on an ME basis, and keep the reserve, starvation and energy-content values with their conventions named.

### Cited Findings
- All values are derived from the cited findings in sections 1-8; no additional sources.

### Inferences
| Parameter | Current | Recommended | Range | Main basis |
|---|---|---|---|---|
| resting_metabolism (MJ/t/tick) | 45 | 45 (adult-like) to 50 (population incl. juveniles) | 35-65 | Hauled-out fed adults 45-52 (Paterson 2021); BMR 1.37x Kleiber = 34; population ME budgets need ~0.27-0.39 MJ/kg/d total spend |
| feeding_cost (x rest) | 1.3 | 1.5 | 1.4-1.8 | Diving 1.6-1.7x Kleiber x feeding 1.4-1.6; HIF 10-17 % of GE = 18-44 % of daily rest cost |
| movement_cost (x rest) | 1.5 | 1.3 | 1.2-1.5 | Swimming at 1.4 m/s = 2.2x RMR / 2.3-2.4x BMR (Davis 1985, Williams 1987) |
| handling_time (ticks) | 20 (20 % BM/d) | 25 (16 % BM/d); per pair: gadoids ~26, herring ~35 | 20-35 | 1.5 x highest routine gross intake (0.57 MJ/kg/d, spotted seal yearlings) |
| max_energy_reserve (MJ/t) | 5200 | 5200 | 4000-7000 | Adult seasonal fat swing ~10.7 points (4,000-4,200 MJ/t); lactation ~17 points |
| maintenance_level | 0.5 | 0.5 | - | Not calibratable; appetite is calendar-driven |
| starve_rate (/tick) | 0.03 | 0.03 | 0.02-0.05 | No literature anchor; meets the project's T_total 30-60 d |
| energy_content (MJ/t) | 10000 | 10000 if reserve stacked on top (reading B); 15,900 if whole body | 10,000-16,000 | Olesiuk 15.9 MJ/kg at 30 % blubber |
| assimilation, herring | 0.92 | 0.88 | 0.86-0.90 | ME on herring 85.5-88.7 % (harp), harbour AE 87-93 % |
| assimilation, gadoids | 0.86 | 0.85 | 0.83-0.87 | DE cod 93.5 % minus ~7-9 % urinary (higher for lean, protein-rich prey) |
| metabolism_q10 | none | none | - | Thermoneutral 6-17 C; seasonality is endogenous |

- With rest 50 and eat 1.5 / move 1.3 on Howard's non-breeding budget the model spends ~0.28 MJ/kg/d (69 MJ/t/tick), matching Olesiuk's population ME budget; reaching the Skagerrak H&HJ level (~0.39 MJ/kg/d) would need rest ~65-70, which Olesiuk's critique of H&HJ's mass scaling argues against.
- Break-even rations with these values: herring (7.5 MJ/kg) ~4.3 % BM/d, gadoids (5.5 MJ/kg) ~6 % BM/d - inside the literature's routine 3-8 %.

### Gaps
- The three biggest gaps: (1) a measured maximum daily intake for harbour seals (Kastelein 2005, Markussen 1990 full texts); (2) the Skagerrak-specific Harkonen & Heide-Jorgensen 1991 budget itself (loss fractions, age structure, whether 20 MJ/d is per capita or per adult); (3) any data on when starvation mortality sets in for adult harbour seals and on fasting mass-loss rates (Markussen 1992 full text).
