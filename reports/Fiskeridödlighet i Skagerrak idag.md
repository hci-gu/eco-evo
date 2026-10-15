# Dagens fiske tar runt en tiondel per år av både torskfiskar och sill

Med ICES bedömningar från 2026 och data till och med 2025 blir den biomassaviktade fiskeridödligheten **F ≈ 0,12/år för gadoiderna** (intervall 0,07–0,19) och **F ≈ 0,10/år för `pelagic_fish`** (intervall 0,005–0,19). I modellens enhet är det ungefär **8,2·10⁻⁵ respektive 6,9·10⁻⁵ per tick** av 6 h. Värdena gäller uttaget på hela poolen, alltså fångst delad med medelbiomassan över alla åldrar. ICES Fbar avser bara de fullt rekryterade åldrarna och ligger 15–25 % högre. Rekommendationen är att lägga in fisket som impacts med en baseline-karta: `bottom_trawling` för gadoiderna och `pelagic_trawling` för sillen. Kartan anger ansträngningen per cell, och tabellen översätter den till förlust per tick.

"Restriktivt fiske" stämmer för sillen. Fångsten i Skagerrak (3a.20) har legat under 1 kt per år sedan 2022, mot 10–15 kt 2019–2021. För torskfiskarna stämmer det bara delvis. Kolja och vitling fiskas lågt, men torskens delbestånd ligger över Fpa och gråsejen över FMSY. Bottentrålningen per km² i rutnätet är dessutom 1,5–2 gånger så hög som i grannrutorna, även om den mest gäller räka och sker med sorteringsrist.

Undersökningen ger också en signal som inte gäller fisket: trålundersökningen IBTS tyder på 2,7 t/km² gadoider i modellens område, medan modellen startar på 0,7 t/km².

Rapporten skiljer på tre slags uppgifter. **Litteratur** anges med källa. **Beräkning** avser egna räkningar ur källornas tabeller. **Inferens** avser egna slutsatser och är uttryckligen markerade. Underlaget finns i `research_notes/Fiskeridödlighet i Skagerrak idag/`: `ices_gadoid_f.md`, `ices_pelagic_f.md` och `local_fishery.md`. Alla tre bygger på ICES-råd från juni–oktober 2026, HAWG 2026, SAG-databasen och IBTS-rådata.

## Rekommendation

| FG | F per år (intervall) | Per tick (6 h) | Impact | Karta |
|---|---|---|---|---|
| `gadoids` | **0,12** (0,07–0,19) | 8,2·10⁻⁵ | `bottom_trawling`, baseline-karta | ansträngning per cell (Bratten-cellerna 0), eller jämn |
| `pelagic_fish` | **0,10** (0,005–0,19) | 6,9·10⁻⁵ | `pelagic_trawling`, baseline-karta | jämn, eller medel över flera EMODnet-fönster |

`biomass_factor` är andelen av den stående biomassan som tas per tick. Med en jämn karta blir det effektiva F exakt tabellvärdet, oavsett var policyerna placerar fisken. En ansträngningskarta per cell gör det effektiva F beroende av hur väl fisken och trålarna överlappar. Då bör q kontrolleras mot fiskens fördelning i proben efter träning.

## Torskfiskar: 0,12 per år, med torsk och gråsej högst

**Litteratur och beräkning.** Värdena nedan är Fbar för 2025 ur ICES-råden 2026. Den biomassaviktade raten är beräknad som fångst delad med medelbiomassan för alla åldrar. För hela biomassan har förhållandet mellan de två tagits ur ICES flerartsmodell SMS (2023), utom för kummel som har en publicerad totalbiomassa.

| Bestånd | Fbar 2025 (åldrar) | FMSY | Biomassaviktad F (intervall) | Källa |
|---|---|---|---|---|
| Torsk, Viking-delbeståndet (där ICES placerar Skagerrak) | 0,426 (2–4) | 0,198 | 0,39 (0,35–0,50) | [cod.27.46a7d20](https://doi.org/10.17895/ices.advice.30932018) |
| Kolja | 0,079 (2–4) | 0,167 | 0,06 (0,045–0,075) | [had.27.46a20](https://doi.org/10.17895/ices.advice.30932057) |
| Vitling, Nordsjön | 0,093 (2–5) | 0,35 | 0,08 (0,035–0,15) | [whg.27.47d](https://doi.org/10.17895/ices.advice.30932456) |
| Gråsej | 0,414 (4–7) | 0,279 | 0,30 (0,11–0,38) | [pok.27.3a46](https://doi.org/10.17895/ices.advice.30932291) |
| Kummel | 0,236 (1–7) | 0,24 | 0,27 (0,24–0,31) | [hke.27.3a46-8abd](https://doi.org/10.17895/ices.advice.30932108) |

Vitlingen i Skagerrak–Kattegatt saknar absolut F och har bara en längdbaserad indikator något över sin MSY-proxy ([whg.27.3a](https://doi.org/10.17895/ices.advice.30932453)). Därför används Nordsjöbeståndet som ankare.

**Sammanvägning (beräkning).** Vikterna är biomassaandelarna i Skagerrak enligt IBTS kvartal 1 och 3, 2021–2025: kolja 54 %, vitling 27 %, gråsej 12 %, torsk 6 % och kummel 1 %. Det ger **0,116**, och 0,10–0,13 med andra uppsättningar vikter. Gråsejens andel är den största osäkerheten. Den kommer säsongsvis i stim, och med enbart kvartal 1-vikter sjunker värdet till 0,10.

**Oberoende kontroll (beräkning).** Gadoidfångsten i Skagerrak 2023–25, cirka 13,8 kt per år med utkast uppräknade, delad med biomassan enligt IBTS skalad med trålvingarnas spännvidd (107,5 kt) ger **0,13/år**. Det stämmer med 0,12. Lokalt i ICES-rutan 45G0 landar svenska, danska och norska fartyg enligt MKB:n (2010–2020) cirka 0,18 t/km² gadoider per år. Mot IBTS-biomassan i rutorna (2,7 t/km²) ger det ~0,07.

## Sill och skarpsill: 0,10 per år för beståndet, nästan noll inne i Skagerrak

**Litteratur och beräkning.** Uttaget u är fångst delad med totalbiomassan, 2025. Modellen tar F/1460 per tick av den aktuella biomassan, så modellens F motsvarar ungefär u.

| Bestånd | Fbar 2025 (åldrar) | Referenspunkt | u | Källa |
|---|---|---|---|---|
| Nordsjösill (NSAS) | 0,191 (2–6) | FMSY = Fpa 0,23 | 0,145 | [her.27.3a47d](https://doi.org/10.17895/ices.advice.32133961) |
| Västra Östersjöns vårlekande sill (WBSS) | 0,011 (2–5) | FMSY = Fpa 0,05 | 0,009 | [her.27.20-24](https://doi.org/10.17895/ices.advice.32133847) |
| Skarpsill | 1,51 (1–2) | Fcap 0,71 | 0,45 | [spr.27.3a4](https://doi.org/10.17895/ices.advice.32134120) |
| Makrill | 0,285 (4–8) | FMSY 0,191 | 0,215 | [mac.27.nea](https://doi.org/10.17895/ices.advice.30932144) |
| Tobis (område 3r, 6) | 0 | – | ~0 | [san.sa.3r](https://doi.org/10.17895/ices.advice.31430314) |

WBSS ligger under Blim, och ICES råder till nollfångst 2027–28. Sammanvägt för `pelagic_fish` blir det **0,10/år** med två oberoende vikter som ger samma svar. Sillandelarna i Skagerrak-undersökningen ger 0,101, och projektets egna andelar från Nordsjö-Ecopath 1991 (tobis med F = 0) ger 0,106.

**Inne i Skagerrak fiskas nästan ingenting (litteratur och beräkning).** All sill som fångades i 3a.20 år 2025 var **0,46 kt** (Danmark 0,09, Norge 0,27, Sverige 0,09), mot 10–15 kt 2019–2021 och 109–168 kt i början av 1990-talet. Fisket har varit nästan stängt sedan 2022 för att skydda WBSS. Akustikundersökningen HERAS mätte 98 kt sill i Skagerrak i juli 2025. Uttaget i området blir alltså **~0,005/år** ([HAWG 2026, tabell 3.1.1 och 3.3.2](https://ices-library.figshare.com/articles/report/Report_of_the_Herring_Assessment_Working_Group_for_the_Area_South_of_62_N_HAWG_/31424315)). De 98 kt motsvarar ~3,0 t/km², vilket stämmer med modellens 3,3 t/km² för `pelagic_fish`.

**Vilket F gäller för modellen (inferens).** Sillen i Skagerrak är en blandning. 2025 var 32 % av biomassan WBSS (14 % 2024). Av ettåringarna är bara 12 % WBSS. Resten är mest NSAS, som växer upp i Skagerrak och flyttar som vuxna ut i Nordsjön, där de fiskas med F ≈ 0,19. Modellens migration lämnar tillbaka den biomassa som går ut över kanten, så poolen i rutnätet är sluten och ersätts bara av sin egen produktion. Under sin livstid möter sillen beståndets fiske, inte områdets. Jag rekommenderar därför **0,10** för den slutna poolen. Värdet står då både för fisket utanför och för den utvandring modellen saknar. 0,005 passar bara om poolen verkligen stannade i rutnätet hela livet.

## "Restriktivt" stämmer för sillen men bara delvis för torskfiskarna

**Regler (litteratur, `local_fishery.md`).**
- Rutnätet ligger i två ICES-rutor, 45G0 (54 %) och 44G0 (34 %). Gränsen mellan Sverige och Danmark från 1984 skär diagonalt genom det, så ungefär hälften är dansk zon.
- Svenska, danska och norska fartyg får fiska bortom 4 nm. EU:s maskregler för Skagerrak gäller: 120 mm som bas, 90 mm med sorteringsrist, och räktrål med 35 mm maska och 19 mm rist.
- Enda permanenta stängningen i rutnätet är södra kanten av Bratten-zonerna i nordväst, där allt fiske är förbjudet ([EU 2017/118](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32017R0118)). Det gäller högst rad 0–5 och kolumn 0–11 och 19–26.
- Ingen flytt av trålgränsen till 12 nm i Skagerrak hittades. Besluten 2025–2026 gäller Östersjön.

**Ansträngning (EMODnet, beräkning).** Bottentrålningen per km² i rutnätet är 1,5–1,8 gånger så hög som i de nio omgivande rutorna, och så har det sett ut i alla fem fyraårsfönster 2015–2024. Mest är det räktrålning med rist som sorterar bort fisk, så kvoten överskattar F för gadoiderna. Den pelagiska ansträngningen är 0,75–1,4 gånger grannskapets, och passiva redskap förekommer knappt.

**Inferens.** För sillen stämmer "restriktivt" mycket väl: fisket i området är i praktiken stängt sedan 2022. För torskfiskarna är fisket lokalt inte lägre än i beståndet. Det restriktiva gäller torsken i Kattegatt och kustnära områden, inte rutnätet.

## Fördelning på redskap: 98 % på huvudkartan, 2 % på den andra

**Litteratur och beräkning (`catch_by_gear.md`).** Varje grupps F fördelas på de två kartorna efter vilka redskap som tar fångsten. Det är redskapskoden som avgör, inte målarten: bottentrål (OTB/OTT, även industritrål OTB_SPF) hör till bottentrålkartan, och flyttrål och not (OTM/PTM/PS) till den pelagiska kartan.

- **Gadoider:** 98 % bottentrål, 2 % pelagisk trål (intervall 0,1–4,5 %). Lokalt i 3a.20 (STECF FDI 2022–24) tar pelagiska redskap 0 % av torsk, kolja, gråsej och kummel och 0,7 % av vitlingen. Danmarks skarpsillsfiske, som loggas som OTM_SPF, ger dessutom en industriell bifångst av vitling. På beståndsnivå (ICES-råden) blir andelen 3,4–4,2 %. Snurrevad och garn, cirka 15 % av den lokala gadoidfångsten, saknar egen karta och ligger kvar på bottentrålkartan.
- **Pelagisk fisk:** 98 % pelagisk trål och not, 2 % bottentrål (intervall 0,2–11 %). Det rör sig om sill som bifångst i havskräft- och fisktrål. Andelen är 0,2–1 % på beståndsnivå och 4–11 % lokalt för sill. Den lokala siffran väger lite, eftersom bara ~5 % av sillens F tas inne i området.

Båda minoritetsandelarna ryms inom osäkerheten i F, men uppdelningen ger rätt plats åt dödligheten. Totalerna är oförändrade (0,12 och 0,10).

## Startbiomassan för gadoiderna kan vara fyra gånger för låg

**Beräkning (`ices_gadoid_f.md`).** IBTS skalad med vingarnas spännvidd ger **3,3 t/km²** gadoider i hela Skagerrak och **2,7 t/km²** i rutorna kring modellområdet. Modellen startar på 0,7 t/km² (2500 t). Skalas undersökningen med trålbordens spännvidd i stället blir biomassan ~4 gånger lägre, 0,83 t/km², nära modellens värde. Men då blir uttaget ~0,5 per år, vilket inte går ihop med ICES bedömningar. Spännvidden mellan vingarna är därför den rimligare skalningen. IBTS-trålen underskattar dessutom stor torsk, stimmande gråsej och djup kummel.

**Inferens.** Detta knyter an till rapporten om täthetsberoende. Med frysta fix55-policyer var gadoidernas överskott vid 2500 t +0,36 per år. Fisket tar 0,12. Den täthetsberoende M1-termen från SMS (0,0875 per t/km²) tar ~0,12 vid dagens lokala täthet, så det återstår ~0,12. Det stängs när den lokala tätheten har ökat med ~1,4 t/km², till ungefär **2,8 t/km²**. Det är nästan exakt vad IBTS visar för området.

Räkningen bygger på frysta policyer och bortser från att bytesbristen också växer med tätheten, så den är bara en indikation. Den talar ändå för att gadoidernas tillväxt i fix55 delvis är modellen som söker sig mot en realistisk nivå, och att det är startvärdet som är lågt.

## Tumlarnas bifångst (sidonot)

Danska och svenska garnfisken fångade i medel **2088 tumlare per år** 2010–2020 (95 % KI 667–6798), och för hela Nordsjöregionen uppskattas bifångsten 2020 till ~1,7 % av populationen per år ([Kindt-Larsen et al. 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10336385/), bara sökutdrag). Bibliotekets M1 för tumlare bygger på överlevnad *utan* bifångst. En garnfiske-impact på ~0,017 per år vore konsekvent, men ingen siffra för Skagerrak hittades.

## Så läggs det in

Inget av detta är gjort.

1. **Kartor:** ansträngning per km² och cell som npz, från EMODnet (`fishingbottomottertrawls` för gadoiderna, `fishingpelagic` för sillen), helst ett medel över flera fyraårsfönster. Lägg in dem som baseline-karta i Impact Editor (avsnitt 161.8). Lägg samma filer under inferensfliken om dagens fiske också ska finnas i inferensen.
2. **Tabeller:** linjära, med fångstbarheten q = F / medel(ansträngning), så att 0 → 0 och ett tak långt över dagens högsta värde ger q·tak/1460. `interp_impact` klämmer vid sista raden, så tabellen måste räcka till scenarier med höjda kvoter. Platshållaren för sill (100 % per tick vid 1000) ersätts. Gadoiderna får en post `gadoids_impacted_by_bottom_trawling`.
3. **Täthetsberoende M1** för gadoider enligt den förra rapporten.
4. **Kontroll:** kör proben med frysta policyer: var planar gadoiderna ut, och vad händer med sillen? Sillen producerade redan för lite i fix55, och 0,10 i fiske gör det värre. Det behöver lösas innan omträningen.
5. **Startbiomassan:** avgör om den ska höjas mot IBTS-nivån innan F och k kalibreras mot 2500 t.

## Källäget

Alla F-värden är lästa ur ICES rådgivningsblad 2026 och SAG-databasen i fulltext. De är nyare än de 2024/2025-råd jag efterfrågade.

Det som inte kunde verifieras är:
- totalbiomassan 2025 för torsk, kolja, vitling och gråsej (uppskattad med kvoter ur SMS),
- utkasten i Skagerrak (uppräknade med beståndens genomsnitt),
- landningar per ICES-ruta efter 2020,
- om norska VMS-data ingår i EMODnets lager (Norge landar ~44 % av tonnaget i 45G0),
- en siffra för tumlarnas bifångst i just Skagerrak.

MKB:n ger två svenska landningssiffror för 45G0 som skiljer sig med 30 % (3024 mot 2285 t per år). I EMODnet minskar ansträngningen i hela Skagerrak med ~60 % efter fönstret 2018–21, och den sydvästra (danska) delen av rutnätet nästan töms 2021–24. Ingen regel som förklarar något av det hittades.

## Slutsats

Fisket ska in som en ren förlust utanför budgeten: 0,12 per år på gadoiderna och 0,10 på sillen, med jämna fält. Sillens "restriktiva" fiske i området är verkligt, men för en sluten pool är det beståndets uttag som räknas. Den viktigaste bifångsten av den här kontrollen är att gadoidernas startbiomassa på 0,7 t/km² ligger långt under vad trålundersökningen visar. Med fiske och täthetsberoende M1 pekar räkningen mot ungefär den nivå undersökningen anger.
