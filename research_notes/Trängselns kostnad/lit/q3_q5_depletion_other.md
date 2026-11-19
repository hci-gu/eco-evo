# Q3 and Q5: prey depletion under feeding aggregations, other crowding costs

Literature pass, 2026-11-19. Research only; nothing in the model was changed.
Grades:
- [F] full text read (page or table given).
- [A] abstract only (OpenAlex or a marineinfo record).
- [S] secondary: cited in another paper, or seen only as a search snippet.
- (computed) my arithmetic. Inputs are named and every assumption is flagged.

PDFs are in `papers/q35/` and their licences in `papers/q35/LICENCES_q35.md`.

## Verdict first

1. **Not found: intake per fish falling with local fish density.** No source I read shows intake rate per fish (stomach content, fullness or ration) falling as LOCAL fish density rises, for a gadoid or flatfish on benthic prey at the scale of an aggregation (a cell, over days). The only haul-level test of that kind I met is the Baltic cod-flounder stomach GLMM, which appears to be the Lindmark et al. work already covered (not redone). **Elasticity of intake per fish on local fish density: not computable from the sources read.**
2. **What the literature does have is mostly the inverse experiment.** Fish converge on a local food pulse: at baited carrion within 30-50 min, on trawl tracks within 2-3 h. They stay above the background density for 1-3 days, eat it out ("consumed within a few days"), and disperse. Dab intake was significantly raised on day 1 only (Kaiser & Ramsay 1997, Fig. 3; the asterisks mark day -1, control higher, and day 1, treatment higher).
   - That shows that demersal fish track and leave ephemeral patches on the time scale of the model's clump hold (< 4 d).
   - It does not show a cost of crowding on a normal bed.
3. **The best-documented field channel is a cropped, renewable exposed fraction.** Dab (and haddock [S]) graze the exposed arms of *Amphiura* (and flatfish crop bivalve siphons and lugworm tails), leaving the animal alive to regrow.
   - The renewal is slow: an arm regrows about 0.24 mm/d (9.68 mm in 41 d), so a cropped tip takes weeks to regrow. The tip size was not read (Skold et al. 1994).
   - The croppable stock is large relative to a 4-day clump ration (computed, below).
   - So within one hold of under 4 days, a clump at 30-90x the mean density removes only a few to about 50 % of the croppable arm tips. The top of that range applies only if arms are its whole diet.
   - The cost would therefore build up over **revisits within the regrowth time (weeks)**, not inside one hold.
4. **Behavioural depression of the prey is documented only qualitatively and at short range.** Macoma retracts its siphon on direct tactile contact with Crangon [S]. Nipped Macoma lose siphon mass and change burial depth [S]. I found nothing measured for *Amphiura* or for deep mud prey against fish. The *Amphiura* alarm-response paper (Rosenberg & Selander 2000) was not accessible, and a later test found no alarm signal [S].

## Q3 key numbers

| Study | Species / site | Spatial scale | What was measured | Magnitude | Time scale | Grade |
|---|---|---|---|---|---|---|
| Kaiser & Spencer 1994, MEPS 112:41-49, doi:10.3354/meps112041 | Gurnards, dogfish, whiting. Irish Sea off Anglesey; coarse sand, gravel, shell | 4 m beam-trawl line, a few km long; side-scan swath 200 m | Stomachs 3 h after repeated trawling vs before; fish marks over vs beside the tracks | Fish marks per 2 min (Table 4):<br>- after tow 3: 0.58 fished vs 0.05 unfished (about 12x)<br>- after tow 6: 20.4 vs 6.41 (about 3.2x)<br>- 80 of 111 marks within 50 m of the tracks<br><br>Gurnard fullness 3.36 -> 5.04 (0-10 scale; H = 6.83, p = 0.009, Table 1).<br>Whiting prey width 6.37 -> 4.43 mm (Table 3): smaller prey after trawling | 2-3 h | [F] p. 44-45, Tables 1, 3, 4 (page images checked). The text's "factors of 30 and 11" is the rise from tow 3 to tow 6 and does not reconcile exactly with Table 4; the table values are quoted |
| Kaiser & Ramsay 1997, MEPS 152:307-310, doi:10.3354/meps152307 | Dab, Irish Sea, 40 m, fine mud/sand (the habitat closest to the box) | 1500 m line, 30-40 m wide strip, controls 250 m away | Dab density and stomach index (SC/FL) daily, from day -1 to day 3 | Background dab 12.6 +- 2.2 per 1000 m2.<br>Dab did NOT aggregate here: "The mean density of dab was significantly lower on the treatment line for 48 h". This contrasts with "3 times more abundant in trawled areas for at least 48 h" at a 14-20 m site (Kaiser & Spencer 1996a [S]).<br>SC/FL: "Thereafter, the SC/FL index did not vary significantly for dab collected from either treatment or control lines". In Fig. 3 the treatment line is significantly higher on day 1 only.<br>Oral discs of *Amphiura* were eaten on 3 consecutive days. Normally dab "graze the exposed arms, while leaving the oral disc intact" | 1-3 d | [F] p. 308-309 |
| Kaiser & Hiddink 2007, MEPS 350:267-276, doi:10.3354/meps07194 (review) | Dab, whiting, other scavengers; Irish and North Sea | Trawl strips; baited cameras | Arrival, persistence, consumption | "Dabs, whiting and hermit crabs arrived between 30 and 50 min".<br>Scavenger abundance raised "up to 72 h after the trawl disturbance occurred and thereafter declines".<br>Carrion consumption time "varies from 3 to 72 h" | 0.5 h - 3 d | [F] for the review text, p. 269-270. The primary numbers are [S] (Kaiser & Spencer 1996; Ramsay et al. 1996, 1997; Demestre et al. 2000; Groenewold & Fonds 2000) |
| Groenewold & Fonds 2000, ICES JMS 57:1395-1406, doi:10.1006/jmsc.2000.0914 | Dab, plaice, whiting, dragonet, grey gurnard; southern North Sea | Experimentally trawled strip | Immigration; clearance of discards and moribund benthos | "Discards as well as moribund benthos were consumed within a few days".<br>One beam trawl pass makes 1.27 g AFDW m-2 available (6-13 % of annual macrobenthos production) | days | [A] |
| Skold, Loo & Rosenberg 1994, MEPS 103:81-90, doi:10.3354/meps103081 | *Amphiura filiformis*, Gullmarsfjord, Skagerrak, 40 m | Station | Production allocated to arm regeneration; regrowth rate | 280 adults m-2.<br>Regeneration 0.34 g AFDW m-2 yr-1 = 13.3 % of total production.<br>"Mean regenerated biomass in percent of total biomass ... 12 and 30 % (mean 22 %)"; P/B 0.46 yr-1.<br>Arm growth after cutting "9.68 mm 41 d-1 (SE = 1.06, n = 15)" (p. 83, as printed). Read as 9.68 mm per 41 d, about 0.24 mm/d, "estimated over 7 wk"; i.e. 0.0048 mg DW arm-1 d-1. Check (computed): the paper's regression DW = 0.01056 L^1.288 (p. 84) gives 0.196 mg DW at 9.68 mm; 0.196 / 41 d = 0.0048, which reproduces the paper's rate, so the per-41-d reading holds.<br>Predators "crop only the arms"; the haddock link is cited to Mattson 1992 [S] | weeks (regrowth) | [F] p. 81, 83, 88-89 |
| Duineveld & van Noort 1986, Neth J Sea Res 20:85-94 | *A. filiformis* and dab, Oyster Ground, southern North Sea | Station | Arm-regeneration frequency; dab consumption | "Only 4 % of the individuals in a sample had intact arms ... 55 % of the animals (or 20 % of all arms) had one or more 'recent' distal regenerations".<br>Dab eat 0.84 g WW m-2 yr-1 = 420 arms = "6 % of the arm population" per year | 1 yr | [A] marineinfo record 15896 |
| Albert 1994, ICES JMS 51:31-44, doi:10.1006/jmsc.1994.1004 | Haddock, Norwegian Deep (next to the box) | Survey | Diet | "Polychaetes, crustaceans, and ophiuroids dominated the diet". Ophiuroids are important in summer | seasons | [A] |
| Toresen 1991, ICES JMS 48:15-21, doi:10.1093/icesjms/48.1.15 | Haddock aggregated on a herring spawning bed, west Norway | One spawning ground | Egg removal | Haddock "the main predator on eggs"; about 4 % of the egg production eaten over 50 days. That the aggregation did not deplete its patch is my inference | 50 d | [A] |
| Hill, Burrows & Hughes 2000 (J Fish Biol 56:1497, doi:10.1111/j.1095-8649.2000.tb02160.x) and 2002 (J Fish Biol 61:1255, doi:10.1111/j.1095-8649.2002.tb02469.x) | Juvenile plaice, laboratory arena | cm - m | Search mode after finding prey | Area-restricted search triggered "by a high local density of prey but not by isolated prey"; "win-shift behaviour when prey were dispersed".<br>A local-cue mechanism, with no memory needed | seconds - minutes | [A] |
| de Vlas 1979, Neth J Sea Res 13:362-393 | Plaice, flounder on *Arenicola* tails; Wadden Sea | Tidal flat | Tail cropping | Each attack removes "one or a few tailtip segments". About 120 mg AFDW per lugworm per year is cropped. Companion paper: about 1/3 of flatfish food at Balgzand was siphon tips, tail ends and similar parts | yr | [S] (search snippets of the abstract) |
| de Goeij et al. 2001, Oecologia 126:500-506, doi:10.1007/s004420000526 | Juvenile plaice nipping *Macoma* siphons | Mesocosm / field plots | Siphon mass, burying depth | Nipped clams "lost siphon mass and came closer to the surface"; recovery speed "varied greatly between experiments" | weeks | [S] |
| Macoma vs Crangon (Wageningen and Derby repository records; I did not identify the paper) | *Macoma* | Lab | Siphon behaviour | "Immediate retraction of the feeding siphon and a reduced feeding activity" on tactile encounter | minutes | [S] |
| Espeland et al. 2007, ICES JMS 64:920-928, doi:10.1093/icesjms/fsm028 | Coastal cod, Skagerrak coast, acoustic telemetry | Inshore basin | Home range | "The average home range of the adult cod was 27 ha". This is a spawning-season range, not a foraging residence time | season | [A] |

### Not accessed (paywalled; no open copy; only the title or a citation was seen)

- **Hall, Raffaelli, Basford & Robertson 1990**, JEMBE 136:65-76, doi:10.1016/0022-0981(90)90100-Q. A dab caging experiment; title only.
- **de Vlas 1985**, Neth J Sea Res 19:147-164 (Macoma siphon regeneration production).
- **Peterson & Quammen 1982**, JEMBE 63:249-268.
- **Peterson & Skilleter 1994**, Oecologia 100:256-267.
- **Skilleter & Peterson 1994**, Oecologia 100:268-278 [S]: "greater siphon regeneration at lower density" when protected from nipping fish. This is a prey-density effect.
- **Kamermans & Huitema 1994**, JEMBE 175:59-75.
- **Bonsdorff, Norkko & Sandberg 1995**, JEMBE 192:125-144.
- **Rosenberg & Selander 2000**, Mar Biol 136:43-48 (*Amphiura* alarm response).
- Full texts of **Groenewold & Fonds 2000** and **Demestre et al. 2000** (book chapter).
- The **Hinz 2005 PhD thesis** (Bangor), which was refused.
- No telemetry study linking demersal fish residence to measured patch depletion was found.

Frid & Hall 1999 (MEPS 184:183) was downloaded but not used: it compares dab diets between decades and is not about depletion.

## The discriminating computation (computed)

The question for the model: can a clump at 50-150 t/km2, held under 4 days, crop the exposed, renewable fraction of the bed down far enough to lower its intake rate in the field?

**Inputs**
- Oyster Ground arm stock: 420 arms = 6 % of the arm population per year (Duineveld & van Noort [A]).
  - So the arm population is about 7 000 arms m-2.
  - The mean cropped piece is 0.84 g / 420, about 2.0 mg WW.
- Upper-bound croppable stock: one mean piece per arm, 7 000 x 2.0 mg, about 14 g WW m-2 = 14 t WW/km2.
  - This is an upper bound because only arms held above the sediment are exposed. The sources read do not quantify that share.
- Clump demand at the model's booked benthivore ration (Q/B 4.46 t/t/yr, AGENTS section 223.9):
  - 50-150 t/km2 x 4.46/365 = 0.61-1.83 t WW/km2/d.
  - Over a 4-day hold: 2.4-7.3 t/km2.
- Regrowth: about 0.24 mm/d (Skold [F]; the per-41-d reading is checked against the paper's length-weight regression). The size of a cropped tip was not read. A 10 mm tip would take about 41 d, so regrowth is on the order of weeks.

**Results**

| Case | Share of croppable arm tips removed in one 4-day hold |
|---|---|
| Arms = 100 % of the diet | 17-52 % |
| Arms = 20 % of the diet (illustrative share) | 3.5-10 % |
| Scaling Duineveld's background cropping (6 %/yr) by 30-90x for 4 d | 2.0-5.9 % of arms |

The last row assumes Duineveld's dab density equals the model group's mean density. The abstract does not give that density.

**Reading**
- Within one hold of under 4 days, arm-tip depletion at clump density is modest unless arms dominate the diet.
- Regrowth takes weeks (tip size not read), so a cell revisited within the regrowth time offers less.
- In the field, depletion through a cropped fraction therefore acts on the REVISIT interval and on cumulative cropping, not within the hold.
- The field intake-rate cost by this channel is a renewal-limited supply of the exposed fraction, not a lower Holling term on total biomass.

Caveats:
- Skold's site is at 40 m and Duineveld's in the southern North Sea. *A. filiformis* peaks at 20-50 m, and the box's 100-300 m mud has other ophiuroid species and lower densities (not quantified here).
- The *Amphiura* arm is only one prey item.

## Q5. Other crowding costs (brief)

| Item | Finding | Grade |
|---|---|---|
| Oxygen | A16 and A17 bottles at 100-300 m (SHARK, repo file `research_notes/Ljus under 50 m/data/shark_physchem_a17_slaggo/physchem.tsv`):<br>- A17 100-200 m: n 851, median 6.0 ml/l, minimum 4.56<br>- A17 200-300 m: n 566, median 5.98, minimum 4.72<br>- A16 100-200 m: n 316, median 6.06, minimum 5.07<br><br>Saturation (Garcia & Gordon 1992 solubility; A16 + A17 pooled, 1990-2024, n 1098): median 89 %, p05 81 %, minimum 72 %; median T 7.2 C (computed).<br><br>Chabot & Dutil 1999 (J Fish Biol 55:472, doi:10.1111/j.1095-8649.1999.tb00693.x): cod growth reduced "at 45% and 56% O2 saturation" but not at 65 % or above.<br><br>Clump draw-down (computed): FMR 8 891 MJ/t/yr (repo section 267) / 13.6 MJ per kg O2 (assumed oxycalorific value) gives about 1.8 kg O2/t/d. At 150 t/km2 that is 0.27 g O2/m2/d, against about 86 g O2/m2 in a 10 m bottom layer: about 0.3 %/d with no exchange.<br><br>Irrelevant for the box | SHARK data computed; Chabot [A] |
| Parasites | *Lernaeocera branchialis* transmits through an intermediate flatfish host (flounder). Whiting infection is heaviest where flounder density is highest (ICES CM 1977 [S]). That is an intermediate-host density effect, not gadoid crowding.<br><br>Seal-transmitted anisakids are a predator-density effect, not crowding. This is life-cycle knowledge; I read no source and give no number.<br><br>No measured relationship of a directly transmitted parasite or of cod ulcers with local host density was found | [S] |
| Aggression and kleptoparasitism | Brawn 1961 (Behaviour 18:107-147, doi:10.1163/156853961x00295): captive cod aggressive in Sep-Nov and before spawning; dominance orders form.<br>"When food was given all aggressive behaviour was lost even though the fish came into bodily contact with each other while feeding"; it resumed 5-10 min later.<br>Brawn speculates that autumn aggression disperses shoals on the feeding grounds.<br>No measure of intake lost to interference. No field kleptoparasitism data for gadoids or flatfish found | [A] |
| Group membership energetics | Nothing measured for demersal gadoids or flatfish.<br>Snippets: cod stay mostly within about 10 m of the bottom (Skaret et al. 2020 [S]); juvenile haddock stay nearer the bottom than cod (Lough & Potter 1993 [S]).<br>Hydrodynamic schooling savings concern continuously swimming schools; whether they apply to bottom-feeding haddock or flatfish is untested (inference) | [S] |

## Sources consulted

- Kaiser & Spencer 1994: https://doi.org/10.3354/meps112041
- Kaiser & Ramsay 1997: https://doi.org/10.3354/meps152307
- Kaiser & Hiddink 2007: https://doi.org/10.3354/meps07194
- Skold et al. 1994: https://doi.org/10.3354/meps103081
- Duineveld & van Noort 1986: https://marineinfo.org/id/publication/15896
- Groenewold & Fonds 2000: https://doi.org/10.1006/jmsc.2000.0914
- Albert 1994: https://doi.org/10.1006/jmsc.1994.1004
- Toresen 1991: https://doi.org/10.1093/icesjms/48.1.15
- Hill et al. 2000: https://doi.org/10.1111/j.1095-8649.2000.tb02160.x
- Hill et al. 2002: https://doi.org/10.1111/j.1095-8649.2002.tb02469.x
- Espeland et al. 2007: https://doi.org/10.1093/icesjms/fsm028
- Chabot & Dutil 1999: https://doi.org/10.1111/j.1095-8649.1999.tb00693.x
- Brawn 1961: https://doi.org/10.1163/156853961x00295
- de Goeij et al. 2001: https://doi.org/10.1007/s004420000526
- de Vlas 1979 record: https://marineinfo.org/doc/publication/14153
- Lernaeocera snippets: https://www.vliz.be/imisdocs/publications/317652.pdf and https://www.ices.dk/sites/pub/CM%20Doccuments/1977/E/1977_E20.pdf
