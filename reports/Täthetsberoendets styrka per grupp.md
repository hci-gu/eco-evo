# Täthetsberoende dödlighet: torskfiskar och pelagisk fisk γ ≈ 0,25–0,3, övriga grupper ≈ 0

Frågan är hur starkt den naturliga restdödligheten M1 (`natural_mortality`) bör bero på gruppens egen biomassa. Litteraturen ger olika svar för olika grupper:

- **Fiskpoolerna:** ett måttligt γ. Det fungerar som ersättning för den juvenila kompensationen, som en pool utan åldrar saknar.
- **Zooplankton, tumlare och sjöfåglar:** γ ≈ 0. Det täthetsberoende som är belagt för dem sitter i stadier eller processer som modellen redan har explicit (föda och svält) eller inte har alls (ägg).
- **Sälar:** γ ≈ 0, med ett litet γ bara om epizootier och trängsel på kolonier ska ligga i M1 i stället för som pulser.
- **Grupperna utan policy:** har redan sitt täthetsberoende genom logistisk tillväxt.

| Grupp | M1 nu (/år) | **γ** | Försvarbart | β_M = γ·M1 (/år) | Huvudskäl |
|---|---|---|---|---|---|
| Gadoider | 0,415 | **0,25** | 0,1–0,5 | 0,10 | Kannibalismen saknas (+0,06). Juvenil kompensation saknas delvis. Modellens eget täthetsberoende är svagt under referensen. |
| Pelagisk fisk | 0,58 | **0,3** | 0–0,5 | 0,17 | Total kompensation ~0,3/år. Modellen ger själv 0,10 under och 0,33 över referensen. |
| Zooplankton | 19,6 | **0** | 0–0,25 | 0 (≤ 4,9) | Belagt täthetsberoende bara hos ägg. Rovdjuren mättas inte, och maneter följer klimatet. Modellen återgår själv inom veckor. |
| Tumlare | 0,10 | **0** | 0–0,1 | 0 (≤ 0,01) | Vuxenöverlevnad svarar sist och minst. Belagt täthetsberoende går via föda, som modellen redan har. |
| Sälar (avstängda) | 0,10 | **0** | 0–0,2 | 0 (≤ 0,02) | Vuxenöverlevnad konstant. Epizootier är pulser och hör inte hemma i ett jämnt γ. 0,2 bara om de ska ligga i M1. |
| Sjöfåglar (avstängda) | 0,15 | **0** | 0–0,1 | 0 (≤ 0,015) | Täthetsberoendet sitter i häckning och rekrytering, inte i vuxendödlighet. |
| Fytoplankton, bentos | – | ej tillämpligt | – | – | Logistisk tillväxt mot K, alltså redan täthetsberoende. |

Ett andra resultat är lika viktigt. **Inget av γ-värdena löser gadoidernas överskott.** Med fix57-policyerna och γ = 0,25 hamnar gadoiderna i jämvikt nära dubbla referensen, ungefär 5000 t (1,4 t/km²). Det är fortfarande under IBTS-nivån på ~2,7 t/km². γ är en lutning och flyttar bara var jämvikten hamnar. Överskottet på +0,24/år vid referensen är en nivåfråga: antingen är produktionen vid referensen för hög, eller så är referensen för låg.

Rapporten skiljer på fyra slags uppgifter:
- **Litteratur** anges med källa.
- **Beräkning** avser egna räkningar på publicerade värden.
- **Modellprob** avser mätningar med en fryst kopia av fix57-policyerna (kopierad 15:37) i probvärlden: 60×60 celler, mortalitet, migration, strömmar, dagsljus, temperatur och dagens fiskekartor. Varje grupp startades på 0,5×, 1× och 2× sin referensbiomassa medan övriga låg på referensen, och kördes två år.
- **Inferens** avser egna slutsatser och är uttryckligen markerade.

Underlaget finns i fem anteckningar i `research_notes/Täthetsberoendets styrka per grupp/`: `theory_and_syntheses.md`, `gadoids.md`, `pelagic_fish.md`, `zooplankton.md` och `mammals_and_seabirds.md`.

## Definition: γ är en lutning, och den räknas som litteraturens totala minus modellens egen

- **β_M** = dM1/d ln B vid referensbiomassan, per år. Det är hur mycket dödligheten per individ ökar när biomassan ökar med faktorn e.
- **γ** = β_M / M1(B_ref) är samma sak utan enhet, alltså elasticiteten. Formen M1·(B/B_ref)^γ har elasticiteten γ överallt.
- **Totalen** (β_tot) är hur snabbt nettotillväxten per individ faller med ln B i ett verkligt bestånd, från alla källor. Där ingår rekrytering, tillväxt, dödlighet och predation.
- **Den explicita delen** (β_explicit) är det modellen redan har: födobegränsad produktion, interferens, svält och den explicita predationens respons.

Bara skillnaden hör hemma i M1:

    γ = (β_tot − β_explicit) / M1,   golv 0

Tidigare noter räknade ofta som om modellen saknade eget täthetsberoende. Därför mättes β_explicit här direkt (avsnittet *Vad modellen redan har*).

**Litteratur.** Kompensation hos fisk sker framför allt genom juvenil dödlighet efter bottenfällningen, därefter genom tillväxt. Direkt täthetsberoende vuxendödlighet är nästan odokumenterat:
- Rose et al. ger inget exempel på vuxendödlighet bland processerna ([Rose et al. 2001](https://aquaticecology.tamu.edu/wp-content/uploads/sites/3/2012/07/Rose_etal_FishFisheries.pdf), fulltext).
- Horbowy et al. skriver att de "lack empirical observations" och härleder täthetsberoende M från täthetsberoende tillväxt ([Horbowy et al. 2025](https://doi.org/10.1093/icesjms/fsaf002), sammanfattning av fulltext).
- Den kanalen, M ∝ W^−0,3, ger bara γ ≈ 0,03–0,07 för torskfiskar (beräkning i `theory_and_syntheses.md` Q4).

**Inferens.** För fisk är ett γ > 0 alltså inte vuxenbiologi. Det fyller platsen för den juvenila kompensationen, som en ålderslös pool inte har någon annanstans att lägga.

## Hur stark är kompensationen totalt?

Tre oberoende vägar ger samma storleksordning för fiskpoolerna:

1. **Produktionsform (Pella–Tomlinson).**
   - **Litteratur.** Thorson et al. 2012 anpassade 147 bestånd. B_MSY/B_0 är 0,44 för Gadiformes och 0,26 för Clupeiformes, vilket är lägst av alla ordningar ([Thorson et al. 2012](https://jensen.limnology.wisc.edu/wp-content/uploads/sites/147/2021/11/Thorson_etal_2012b_CJFAS.pdf), fulltext).
   - **Beräkning.** För g(B) = (r/m)[1 − (B/K)^m] är −dg/d ln B = r(B/K)^m. Vid fiskad jämvikt blir det r − mF.
   - Med bibliotekets r_max (CMSY-priorer, Schaefer-förankrade) ger det **0,32–0,39/år för gadoider** i jämvikt och 0,18–0,27 om dagens bestånd ligger på B/K 0,3–0,5. **Pelagisk fisk** får 0,22–0,32 med Thorsons form, eller 0,55 med Schaefer.
   - ICES-rådgivningen för 2026 visar att torskfiskbestånden inte ligger i jämvikt: nordlig torsk och gråsej ligger under Blim, kolja långt över.
2. **MSY-identiteten.**
   - **Beräkning.** Maximering av F·B ger exakt dp/d ln B = −F_MSY vid B_MSY, oavsett modellform.
   - För den pelagiska poolen ger ICES F_MSY/Fcap, viktat efter biomassa, **≈ 0,33/år** (`pelagic_fish.md` Q0). Hästmakrillens F_MSY 0,15 är antagen och inte verifierad.
3. **Steepness (Beverton–Holt).**
   - **Litteratur.** Myers et al. 1999 anger h = 0,79 för Gadidae och 0,71 för Clupeidae ([Myers et al. 1999](https://sedarweb.org/documents/s8rd21-maximum-reproductive-rate-of-fish-at-low-population-sizes/), fulltext enligt teorinoten). FishLife ger torsk 0,79, kolja 0,66, vitling 0,71 och gråsej 0,79 ([Thorson 2020](https://doi.org/10.1111/faf.12427) och data ur FishLife-paketet).
   - **Beräkning.** Via Euler–Lotka motsvarar en rekryteringsfaktor c en förlust −ln c / T på en pool utan åldrar (T = generationstid). För Beverton–Holt blir β = (1/T)·(α̂−1)d / (1 + (α̂−1)d), med α̂ = 4h/(1−h).
   - Med h 0,66–0,79 och T = ln α̂ / r_max = 4–5 år ger det **0,14–0,19/år för gadoider** (taket är 1/T). För den pelagiska poolen blir det ≈ 0,33 när Z_eff kalibreras med MSY-identiteten.

**Inferens.** Totalen vid referensen ligger kring **0,2–0,3/år för gadoider** (spann 0,14–0,39) och **~0,3/år för pelagisk fisk** (0,16–0,55). Två sorters syntes går inte att använda som parametrar:
- **GPDD-synteserna** (medianlutning fisk −0,7, däggdjur −0,4; [Herrando-Pérez et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3433995/), fulltext). Rent mätfel ger ensamt −1 ([Sibly et al. 2005](https://doi.org/10.1126/science.1110760), fig. 4). För arter med r_max ≈ 0,1 är däggdjursmedianen aritmetiskt omöjlig.
- **Theta-logistiska θ-skattningar.** θ och r_m byts mot varandra längs en åsyta, och bara produkten r·θ, alltså lutningen vid K, är identifierad ([Clark et al. 2010](https://doi.org/10.1111/j.2041-210X.2010.00029.x), endast utdrag).

## Vad modellen redan har

**Modellprob, år 1.** Lutning i nettotillväxt per individ (per år) mot ln medelbiomassa, mellan startnivåerna. Positiv = täthetsberoende.

| Grupp | 0,5× → 1× | 1× → 2× | Vad som bär det |
|---|---|---|---|
| Gadoider | **0,08** | **0,25** | Under referensen: lägre bruttoproduktion. Över: produktion 0,17 och svält 0,09, eftersom sillen tar slut. |
| Pelagisk fisk | **0,10** | **0,33** | Produktion +0,15 och gadoidpredation −0,05 (minskar per individ med tätheten). Över referensen tillkommer svält. |
| Tumlare | 0,15 | 0,10 | Svält. År 2 är lutningen ≈ 0. |
| Zooplankton | – | – | Alla tre starterna konvergerar till samma bana. Skillnaden i ln B halveras på ~10 dagar, en återgångstakt på **23–27/år**. |

Nettotillväxten vid referensen var +0,24/år för gadoider och −0,13/år för sill, i linje med träningens prob (1,28 och 0,85).

**Inferens.** Proben ändrar bilden i två avseenden:
- **Modellen är asymmetrisk.** Under referensen har fiskpoolerna svagt eget täthetsberoende (0,08–0,10). Över referensen är det starkt (0,25–0,33), eftersom sillen tar slut och svälten slår till. Ett γ på M1 gör därför mest nytta under referensen.
- **Gadoidpredationen på sill minskar per individ med tätheten.** Lutningen är −0,05 till −0,09, mindre än de −0,29 som pelagisk-noten räknade fram analytiskt för jämnt fördelad täthet. Den fiskade sillen saknar alltså inte eget täthetsberoende, men den har inte heller något extra.

Förbehåll:
- Policyerna är frysta och omoptimeras inte mot den nya tätheten.
- År 1 innehåller ett insvängningsförlopp.
- Proben har ett enda slumpfrö.
- Gruppen skalades men resten av världen gjorde det inte.

## Gadoider: γ ≈ 0,25

**Litteratur, var det verkar.**
- **Skagerraks kusttorsk.** Täthetsberoende dödlighet från 0- till 1-grupp är signifikant i 9 av 11 fjordområden: a = −0,25 till −0,50, median −0,28, med observationsfel modellerat ([Fromentin et al. 2001](https://doi.org/10.1890/0012-9658(2001)082[0567:EODDAS]2.0.CO;2), fulltext via Archimer).
  - Formen är log-linjär, inte linjär ([Bjørnstad et al. 1999](https://doi.org/10.1890/0012-9658(1999)080[1278:ANTFDD]2.0.CO;2), abstract). Den rumsliga skalan är under 20 km ([Fromentin et al. 2002](https://doi.org/10.3354/meps241231), fulltext).
  - Efter första året är M ≈ 0,4, men något täthetsberoende där är inte visat ([Julliard et al. 2001](https://doi.org/10.1890/1051-0761(2001)011[0540:NMAFMI]2.0.CO;2), abstract).
- **Torsk och vitling.** Starka belägg för juvenil täthetsberoende dödlighet, "related to the log of juvenile abundance". Kolja finns inte med bland arterna med starka belägg ([Myers & Cadigan 1993](https://doi.org/10.1139/f93-179), endast abstract; koefficienterna sågs inte).
- **Kannibalism och predation mellan torskfiskar.** I Nordsjö-SMS (key run 2023, egen beräkning) är den 0,016 + 0,0875·B/år, med B i t/km². Logaritmisk anpassning ger β = 0,24/år kring 2,7 t/km². Lutningen är delvis inbyggd i SMS, där ransonen är fast. Paret är avbockat i modellen, så **förlusten saknas helt i dag**.

**Beräkning, bidrag till β_M vid 0,69 t/km²** (`gadoids.md` avsnitt 6):

| Komponent | β (/år) | Grund |
|---|---|---|
| M0 (sjukdom, parasiter, ålder; 0,20) | +0,01 | Inget belägg för täthetsberoende sjukdomsdöd hos torskfisk. Bara storlekskanalen bidrar. |
| Rovdjur utanför modellen (0,214) | −0,02 (−0,08 till 0) | Mättade generalister: dödligheten per individ sjunker med tätheten. Inget stöd för positivt tecken. |
| Saknad kannibalism (avbockat par) | +0,06 | SMS 0,0875·B vid 0,69 t/km². 0,12 vid 1,4 t/km² lokalt. |
| Juvenil kompensation utöver modellens egen | +0,07 (0,01–0,18) | Beverton–Holt 0,14–0,24 minus modellens 0,06–0,13. Kontrolleras mot Fromentin: −a/T = 0,06–0,11. |
| **Summa** | **+0,12** (0,04–0,25) | |

**Kontroll mot proben (beräkning).** Total enligt Beverton–Holt (0,14–0,24) plus kannibalismen, som ligger utanför S-R-kurvorna (0,06), ger 0,20–0,30. Modellens egen lutning är 0,08 under och 0,25 över referensen. Skillnaden blir 0,12–0,22 under referensen och ~0–0,05 över. Komponentsumman 0,12 ligger alltså i den nedre änden av vad som behövs under referensen och är för hög över den.

**Inferens.** γ = **0,25** (β_M ≈ 0,10/år; försvarbart 0,1–0,5). Värdet ligger något under komponentsumman (0,12, γ 0,29). Det är en kompromiss med en enda potenslag: under referensen behövs 0,12–0,22, över den nästan inget. Formen bör vara konkav med golv, eftersom de juvenila skattningarna är log-linjära. Den linjära SMS-formen k·B är fyra gånger svagare vid 0,69 än vid 2,7 t/km². Ett γ är skalfritt och därför robust mot att B_ref kan ligga ~4× under IBTS.

## Pelagisk fisk: γ ≈ 0,3

**Litteratur.**
- **Nordsjösill (NSAS).** Kompensation fanns och var starkare under återhämtningen. Inget belägg för depensation ([Dickey-Collas et al. 2010](https://doi.org/10.1093/icesjms/fsq033), fulltext). Samtidigt följde årsklasstyrkan lekbiomassan under 80 % av åren 1976–2000 ([Nash & Dickey-Collas 2005](https://doi.org/10.1111/j.1365-2419.2005.00336.x), abstract). Beståndet har varit i en lågproduktiv regim sedan 2002 (ICES 2025, fulltext).
- **Västra Östersjösillen (WBSS), Skagerrak–Kattegatts vårlekare,** ligger på 0,63 × Blim. **Tobis område 3r** ligger under Blim. Båda ligger alltså under ICES brytpunkter, där den segmenterade S-R-kurvan inte har någon kompensation (ICES-råd 2025–2026, fulltext).
- **Tobis.** Rekryteringen regleras av 1-gruppen, inte av lekbiomassan. Det är en överkompensatorisk kohortinteraktion ([Arnott & Ruxton 2002](https://www.int-res.com/articles/meps2002/238/m238p199.pdf); [van Deurs et al. 2009](https://int-res.com/articles/meps_oa/m381p249.pdf), båda fulltext).
- **Täthetsberoende dödlighet efter rekryteringen** saknar belägg hos småpelagiska arter:
  - Äggpredation verkar på äggstadiet.
  - Stimbildning späder ut risken lokalt ([Beare et al. 2002](https://archimer.ifremer.fr/doc/00000/10113/7347.pdf), fulltext).
  - Ichthyophonus-utbrottet 1991–93 var episodiskt, och något samband med täthet visades inte.
- **Mättade rovdjur** ger sjunkande dödlighet per individ över mättnadspunkten ([Bakun 2006](https://www.mafmc.org/s/Bakun_PredatorPit.pdf), fulltext). Tumlare på tobis följer typ III ([Ransijn et al. 2021](https://doi.org/10.1002/ece3.8380), abstract). Tecknet för de 0,19/år från rovdjur utanför modellen är därför osäkert, centralt −0,05.

**Beräkning.**
- Totalen är ~0,3/år och modellens egen lutning 0,10 under och 0,33 över referensen. Det ger γ = (0,30 − 0,10)/0,58 ≈ 0,35 under referensen och ≈ 0 över.
- Pelagisk-notens rena litteraturvärde, 0,45, förutsatte att modellen inte har något eget täthetsberoende. Proben visar att den har det.

**Inferens.** γ = **0,3** (β_M ≈ 0,17/år; försvarbart 0–0,5). Att de två uttömda komponenterna ligger under brytpunkten talar för den nedre halvan, liksom 80 %-resultatet för NSAS. Clupeiformes låga B_MSY/B_0 talar för den övre. Ett "rekrytering oberoende av lekbiomassa" (61 % av 224 bestånd; [Szuwalski et al. 2015](https://doi.org/10.1111/faf.12083), abstract) är inget argument för γ = 0. I en poolmodell betyder det tvärtom full kompensation.

## Zooplankton: γ ≈ 0

**Litteratur.**
- Allt direkt belagt täthetsberoende i dödlighet hos copepoder gäller **ägg** av *Calanus finmarchicus*, genom kannibalism från honor och C5. På stadienivå är elasticiteten 0,27–0,72 ([Ohman & Hirche 2001](https://doi.org/10.1038/35088068); [Ohman et al. 2002](https://doi.org/10.1023/A:1021228900786); [Ohman et al. 2004](https://doi.org/10.1016/j.icesjms.2004.03.016), alla fulltext). På Georges Bank är sambandet villkorat och plant under ~2000 honor/m².
- Hos *Pseudocalanus*, som bär äggen, finns inget sådant täthetsberoende. Ingen studie visar täthetsberoende dödlighet hos copepoditer eller vuxna, som bär nästan all biomassa.
- De globala dödlighetsmönstren styrs av temperatur (Q10 ≈ 2) och kroppsstorlek. Tätheten testas aldrig ([Hirst & Kiørboe 2002](https://doi.org/10.3354/meps230195), fulltext).
- **Rovdjuren bakom M1** mättas inte vid fältkoncentrationer. *Sagitta setosa* i Gullmarsfjorden ökar nästan linjärt upp till ~50 copepoder/L, medan fälttätheten är ~6/L ([Vergara-Soto et al. 2010](https://doi.org/10.4067/S0718-19572010000100003), fulltext). Maneternas förekomst i Nordsjön följer NAO och atlantiskt inflöde mer än copepoderna ([Attrill et al. 2007](https://doi.org/10.4319/lo.2007.52.1.0480), abstract).
- **Den kvadratiska slutningstermen** i NPZ-modeller (γ = 1) står för alla rovdjur och för saknad födovävstruktur ([Anderson et al. 2015](https://doi.org/10.5194/gmd-8-2231-2015), fulltext). Den är "mathematically (rather than biologically) justified" ([Mitra 2009](https://doi.org/10.1016/j.ecolmodel.2008.12.008), abstract). Påståendet att den stabiliserar har motbevisats ([Edwards & Yool 2000](https://doi.org/10.1093/plankt/22.6.1085), abstract).

**Beräkning.**
- Äggens täthetsberoende motsvarar ≤ 0,04 i γ för poolen. Ungefär 98 % av äggen dör redan vid referensen, och äggförlusterna ingick aldrig i M1, som kalibrerades på dödlighet efter kläckningen.
- Om rovdjuren följer copepoderna med elasticiteten k blir γ = (1 − M0/M1)(k − σ): 0,54–0,69 vid k = 1, ~0,25 vid k = 0,5 och ≈ −0,04 vid k = 0.
- Temperaturens Q10 kan ensam se ut som γ ≈ 0,27 över en tiofaldig säsongsvariation i biomassa. Den variationen är antagen och inte verifierad för Skagerrak.

**Inferens.** γ = **0**, försvarbart upp till 0,25.
- Zooplankton-notens centralvärde 0,25 bygger på antagandet k = 0,5. Det finns inga Skagerrak-data för det, och klimatstyrningen av maneterna talar emot.
- Proben visar att modellen själv drar tillbaka zooplankton med 23–27/år. Ett γ på 0,25 skulle lägga till 4,9/år, ungefär 20 %, i en grupp som redan är den snabbast reglerade.
- Om ett γ ändå väljs ska det räknas på en utjämnad, regional täthet, eftersom rovdjuren svarar med veckors fördröjning. Det får inte kombineras med en Q10 på M1.

## Tumlare: γ ≈ 0

**Litteratur.**
- Hos stora däggdjur svarar juvenilöverlevnad först, sedan ålder vid första reproduktion och fruktsamhet. Vuxenöverlevnaden svarar sist ([Eberhardt 1977](https://doi.org/10.1017/S0376892900025819); [Eberhardt 2002](https://doi.org/10.1890/0012-9658(2002)083[2841:APFPAO]2.0.CO;2); [Gaillard et al. 2000](https://doi.org/10.1146/annurev.ecolsys.31.1.367), abstracts).
- I Fowlers genomgång hade bara 1 art täthetsberoende vuxenöverlevnad. Formen kallas "purely speculative", och MNPL/K ligger troligen på 0,60–0,65, vilket motsvarar θ ≈ 2,4–3,5 ([Taylor & DeMaster 1993](https://storage.googleapis.com/nmfs_odp_swfsc/Publications/Files/CR/1993/9388.pdf), fulltext).
- För tumlare går det belagda täthetsberoendet via föda:
  - dräktighet följer bytets energitäthet ([IJsseldijk et al. 2021](https://doi.org/10.1038/s41598-021-98629-x), abstract);
  - svält ökade när tobisen saknades ([MacLeod et al. 2007](https://doi.org/10.1098/rsbl.2006.0588), abstract);
  - 68 % av sjukdomsdöden sker hos magra djur ([Neimanis et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8833703/), fulltext).
- Ingen studie testar täthetsberoende vitala tal mot populationstäthet.
- Nordsjöpopulationen, där platsen (58,1° N) ingår, har legat stilla i 28 år: 289 000, 355 000, 345 000 och 339 000 ([OSPAR QSR 2023](https://oap.ospar.org/en/ospar-assessments/quality-status-reports/qsr-2023/indicator-assessments/abundance-distribution-cetaceans/); [Gilles et al. 2023](https://tethys.pnnl.gov/sites/default/files/publications/Gilles_etal_2023.pdf), fulltext).

**Beräkning.**
- Med bifångst på 0,5–1,7 %/år ligger beståndet på B/K ≈ 0,83–0,98. Litteraturens total blir då 0,14–0,24/år vid θ = 2,4, eller 0,08–0,10 vid θ = 1.
- Den del som inte går via föda är ~5 % (0–15 %). Det ger β ≈ 0,01/år, alltså γ ≈ 0,1.
- Proben gav 0,10–0,15/år genom svält under år 1, och ≈ 0 under år 2.

**Inferens.** γ = **0** (högst 0,1). Det enda tänkbara värdet, 0,01/år, går inte att skilja från noll. Det som återstår är föda, och den finns redan explicit. Två modellfrågor bör utredas oberoende av γ:
- **Ingen bifångst.** Modellen har ingen bifångst av tumlare och drar därför mot sitt eget K.
- **Låg referens.** B_ref = 12 t är bara 1/10–1/17 av vad SCANS-III-tätheten ger för området: 1,08 djur/km² i block P1, ungefär 200 t på 3600 km² ([Hammond et al. 2021](https://scans3.wp.st-andrews.ac.uk/files/2021/06/SCANS-III_design-based_estimates_final_report_revised_June_2021.pdf), tabell 6).

## Sälar och sjöfåglar (avstängda)

**Litteratur, sälar.**
- Knubbsälen i Kattegatt–Skagerrak följer en logistisk kurva med K = 13 965 och lutningen vid K μ = 0,28 ± 0,10/år ([Carroll et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12208499/), fulltext med tabell S2).
- PDV dödade 50–66 % 1988 och 2002, med högst dödlighet i norra Skagerrak 2002. Att svårighetsgraden beror på tätheten är inte testat ([Härkönen et al. 2006](https://doi.org/10.3354/dao068115), fulltext).
- Hos gråsäl ligger allt täthetsberoende i ungöverlevnaden, medan vuxenöverlevnaden är konstant 0,96 ([SCOS 2022](https://www.smru.st-andrews.ac.uk/files/2023/09/SCOS-2022.pdf), fulltext).

**Inferens, sälar.**
- γ = **0** (försvarbart upp till 0,2). Skälen är desamma som för tumlarna: vuxenöverlevnaden är över 0,95 hos knubbsäl och hålls konstant hos gråsäl, och det täthetsberoende som är belagt går via föda och ungar.
- Det som skiljer sälarna från tumlarna är epizootierna och kolonierna. Att PDV:s svårighetsgrad beror på tätheten är inte testat, och utbrotten är pulser (i snitt ~0,05/år). Om de ska vara med är en stokastisk puls rätt form. Bara om de i stället ska ligga i M1 är γ ≈ 0,2 rimligt, vilket motsvarar ~10 % av totalen 0,21/år vid 0,9 K.

**Litteratur, sjöfåglar.**
- Täthetsberoendet verkar under häckningen genom att födan kring kolonierna utarmas: Ashmoles halo och kolonistorlekens tak ([Lewis et al. 2001](https://doi.org/10.1038/35090566), *Nature*; [Jovani et al. 2016](https://doi.org/10.1111/oik.02781), abstracts).
- Det slår mot häckningsframgång, rekrytering och juveniler. JNCC-genomgången finner kompensation i rekrytering i 16 studier ([Horswill & Robinson 2015](https://data.jncc.gov.uk/data/897c2037-56d0-42c8-b828-02c0c9c12d13/JNCC-Report-552-REVISED-WEB.pdf), fulltext).
- Vuxenöverlevnaden påverkas svagt och mest under kalla vintrar, till exempel hos skarv i Kattegatt ([Frederiksen & Bregnballe 2000](https://doi.org/10.1046/j.1365-2656.2000.00435.x), abstract).

**Inferens, sjöfåglar.** γ = **0** (högst 0,1). Det saknade täthetsberoendet hör till häckningsproduktionen, alltså tillväxttermen, och inte till vuxnas M1.

## Grupperna utan policy: redan täthetsberoende

Fytoplankton och bentos växer logistiskt, r·B·(1 − B/K). Det motsvarar en theta-logistisk kurva med θ = 1: förlusten per individ växer linjärt med B och har lutningen r·B/K. För bentos är logistisk återhämtning just den form som trålningslitteraturens r = 0,82/år är skattad med (rapporten *Bottentrålning och bentos*). En separat M1 finns inte för dessa grupper, så γ är inte tillämpligt.

## Därför ska vissa grupper ha γ = 0

Litteraturen ger fyra skäl. Varje grupp med γ = 0 har minst två av dem.

1. **Vuxenöverlevnad är den sista och svagaste kanalen.** Det gäller för stora däggdjur och sjöfåglar (Eberhardt, Fowler, Gaillard, Taylor & DeMaster, SCOS, JNCC). M1 är en vuxendödlighet för dessa grupper. Det gäller tumlare, sälar och sjöfåglar.
2. **Det belagda täthetsberoendet går via föda, och modellen har föda och svält explicit.** Tumlarnas dräktighet och svält samt sjöfåglarnas halo i den mån fåglarna är bundna till kolonin. Att lägga samma sak i M1 vore att räkna den två gånger. Det som inte går via föda är ~5 % för tumlare, ≈ 0,01/år. Proben stöder inte slutsatsen åt något håll: svälten gav 0,10–0,15/år under år 1 men ≈ 0 under år 2. Det gäller tumlare, sälar och sjöfåglar.
3. **Det belagda täthetsberoendet sitter i ett stadium som M1 aldrig innehöll.** För zooplankton är det ägg. M1 kalibrerades på dödlighet efter kläckningen, och 98 % av äggen dör redan. Det gäller zooplankton.
4. **De rovdjur som M1 står för ger inte stigande dödlighet per individ.** Generalister där gruppen är ett byte bland många ("bycatch") ger ≈ 0 ([McLellan et al. 2010](https://wildlifescience.ca/wp-content/uploads/2022/11/McLellan-et-al.-2010-Ecology.pdf), fulltext). Mättade rovdjur ger negativt tecken. Rovdjur som styrs av klimatet ger ≈ 0. Det gäller zooplankton och den del av fiskarnas M1 som är predation.

Det fjärde skälet gäller även fiskarna. Därför ska den del av deras M1 som kommer från rovdjur utanför modellen (gadoider 0,214, pelagisk fisk 0,19) ligga i golvet som inte beror på tätheten. γ ska bara verka på M0-delen, som ersättning för den juvenila kompensationen.

## Form, golv och skala

**Golv.** En ren potenslag låter M1 gå mot noll när B går mot noll, och det är fel för M0. Följande form behåller golvet och ger samma γ vid referensen:

    M1(B) = M_golv + (M1_ref − M_golv) · (B/B_ref)^κ,   κ = γ · M1_ref / (M1_ref − M_golv)

| Grupp | M_golv | κ | M1 vid 0,5× / 2× referensen |
|---|---|---|---|
| Gadoider (γ 0,25) | 0,214 (rovdjur utanför modellen) | 0,52 | 0,353 / 0,500 |
| Pelagisk fisk (γ 0,3) | 0,19 | 0,45 | 0,476 / 0,721 |

κ < 1 ger den konkava formen som de juvenila skattningarna pekar på.

**Skala (inferens).** Varje mekanism som bär γ verkar på beståndsnivå:
- rekryteringskompensation över ~500 km ([Myers 2001](https://doi.org/10.1006/jmsc.2001.1109), abstract);
- uppväxtområden för kusttorsk mest utanför rutnätet;
- sillens lekområden utanför domänen.

B bör därför vara gruppens totala biomassa på rutnätet, inte cellens. En term per cell straffar stim och belönar utspridning (`pelagic_fish.md` Q6). Det enda lokala inslaget är kannibalismen hos gadoider, som är liten (+0,06). Det avviker från rekommendationen i *Täthetsberoende och interferens hos torskfiskar*, som föreslog en term per cell med motiveringen att den ger en signal som går att lära sig. Med rekryteringskompensationen inräknad är den motiveringen svagare än argumentet om skala.

**Förankring.** Om M1(B_ref) sätts till dagens M1 blir tillväxtbudgeten oförändrad vid referensen, och termen verkar bara på avvikelser. Den förra rapporten argumenterade för att lutningen ska vara obetald av budgeten. Det är ett separat beslut. Det ändrar inte γ, men det avgör om termen kan ta bort ett överskott vid referensen. Det kan den inte med förankring vid referensen.

## Vad γ inte löser

**Beräkning med fix57-proben.**
- **Gadoider.** Nettot är +0,24/år vid referensen och +0,087 vid 2× under år 1. Med γ = 0,25 och golvformen ovan ökar M1 vid 2× med 0,085. Nettot vid 2× blir då ≈ 0, och jämvikten hamnar nära **5000 t (1,4 t/km²)**. Det är dubbla referensen men fortfarande hälften av IBTS-nivån. Värdet är en övre gräns: under år 2 var nettot vid 2× redan −0,086 utan γ, eftersom sillen då var uppäten. Modellens egen jämvikt ligger alltså redan under 2×, och γ flyttar den bara måttligt nedåt. Det som reglerar gadoiderna i modellen är fortfarande att sillen tar slut.
- **Sill.** Nettot är −0,13/år vid referensen och −0,06 vid 0,5×. Med γ = 0,3 minskar M1 vid 0,5× med 0,105. Jämvikten hamnar kring **0,6–0,65 × referensen (7000–8000 t)**.

**Inferens.** γ ändrar var bestånden planar ut, inte varför de drar iväg. För sillen döljer det delvis underproduktionen: nedgången stannar på en lägre nivå i stället för att fortsätta. Den bör därför utredas med γ = 0 först. För gadoiderna kvarstår frågan från förra rapporten: är 2500 t för lågt satt, eller är produktionen vid referensen för hög? Att jämvikten med ett litteraturgrundat γ hamnar på 1,4 t/km² talar för att referensen är låg snarare än att modellen är fel.

## Rättelser till vad jag sa i chatten

- Jag skrev att gadoiderna med γ = 1 skulle landa på ~1,6 × referensen, att det behövs γ ≈ 5 för att få ned dem nära 1,0, och att sillen skulle landa på ~0,72 ×. De räkningarna antog att modellen saknar eget täthetsberoende. Proben visar att den har det, 0,08–0,33/år beroende på grupp och riktning, så de siffrorna gäller inte.
- Jag föreslog en term förankrad vid referensen och räknad på hela gruppens biomassa. Att räkna på hela gruppen står sig. Att förankringen gör termen budgetneutral vid referensen är ett separat beslut (avsnittet *Form, golv och skala*). Det beslutet avgör om termen kan ta bort ett överskott där.

## Källäget

Wiley, PNAS, Royal Society, OUP och NOAA returnerade ofta 403. Därför bygger flera källor bara på abstract, och varje anteckning markerar fynd för fynd hur källan lästes. Det viktigaste:

| Källa | Läst | Används för |
|---|---|---|
| [Thorson et al. 2012](https://jensen.limnology.wisc.edu/wp-content/uploads/sites/147/2021/11/Thorson_etal_2012b_CJFAS.pdf) | fulltext | B_MSY/B_0 och form per ordning |
| [Myers et al. 1999](https://sedarweb.org/documents/s8rd21-maximum-reproductive-rate-of-fish-at-low-population-sizes/) | fulltext (teorinoten); pelagisk-noten fick 404 och använde Froese & Luna 2004 | steepness per familj och art |
| [Thorson 2020](https://doi.org/10.1111/faf.12427), FishLife-data | fulltext och extraherade data | h, r och G per art |
| [Rose et al. 2001](https://aquaticecology.tamu.edu/wp-content/uploads/sites/3/2012/07/Rose_etal_FishFisheries.pdf) | fulltext | var kompensationen verkar |
| [Horbowy et al. 2025](https://doi.org/10.1093/icesjms/fsaf002) | sammanfattning av fulltext | att belägg för täthetsberoende vuxen-M saknas |
| [Sibly et al. 2005](https://doi.org/10.1126/science.1110760), [Herrando-Pérez et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3433995/), [Brook & Bradshaw 2006](https://conservationbytes.com/wp-content/uploads/2014/12/brook-bradshaw-2006-ecology.pdf) | fulltext | form och styrka över taxa, mätfelsproblemet |
| [Clark et al. 2010](https://doi.org/10.1111/j.2041-210X.2010.00029.x), Polansky 2009, Ross 2006/2009, Knape & de Valpine 2011 | utdrag eller sekundärt | kritik av θ-skattningar |
| [Fromentin et al. 2001](https://doi.org/10.1890/0012-9658(2001)082[0567:EODDAS]2.0.CO;2), [2002](https://doi.org/10.3354/meps241231) | fulltext | Skagerraktorskens juvenila täthetsberoende, skala |
| [Bjørnstad et al. 1999](https://doi.org/10.1890/0012-9658(1999)080[1278:ANTFDD]2.0.CO;2), [Myers & Cadigan 1993](https://doi.org/10.1139/f93-179) | abstract | log-linjär form |
| [SMS key run 2023](https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2023) | rådata, egen beräkning | kannibalism 0,0875·B, icke-FG-predationens tecken |
| [Dickey-Collas et al. 2010](https://doi.org/10.1093/icesjms/fsq033), [Payne et al. 2009](https://oceanrep.geomar.de/id/eprint/2933/1/538_Payne_2009_RecruitmentInAChangingEnvironment_Artzeit_pubid11565.pdf), ICES-råd 2025–2026 | fulltext | sillens kompensation, brytpunkter, F_MSY |
| [Nash & Dickey-Collas 2005](https://doi.org/10.1111/j.1365-2419.2005.00336.x), [Szuwalski et al. 2015](https://doi.org/10.1111/faf.12083) | abstract | 80 % följer lekbiomassan; 61 % oberoende |
| [Arnott & Ruxton 2002](https://www.int-res.com/articles/meps2002/238/m238p199.pdf), [van Deurs et al. 2009](https://int-res.com/articles/meps_oa/m381p249.pdf) | fulltext | tobisens kohortreglering |
| [Bakun 2006](https://www.mafmc.org/s/Bakun_PredatorPit.pdf), [Hilborn et al. 2014](https://www.arlis.org/docs/vol2/hydropower/Hilborn%202014.pdf), [McLellan et al. 2010](https://wildlifescience.ca/wp-content/uploads/2022/11/McLellan-et-al.-2010-Ecology.pdf), [Hixon & Jones 2005](https://researchonline.jcu.edu.au/6235/1/6235_Hixon_%26_Jones_2005.pdf) | fulltext | tecknet för predation från rovdjur utanför modellen |
| [Ohman & Hirche 2001](https://doi.org/10.1038/35088068), [Ohman et al. 2002](https://doi.org/10.1023/A:1021228900786), [2004](https://doi.org/10.1016/j.icesjms.2004.03.016), [Hirst & Kiørboe 2002](https://doi.org/10.3354/meps230195) | fulltext | copepoders täthetsberoende dödlighet och temperatur |
| [Anderson et al. 2015](https://doi.org/10.5194/gmd-8-2231-2015), [Vergara-Soto et al. 2010](https://doi.org/10.4067/S0718-19572010000100003) | fulltext | NPZ-slutning, rovdjurens respons |
| Steele & Henderson 1992, [Edwards & Yool 2000](https://doi.org/10.1093/plankt/22.6.1085), [Mitra 2009](https://doi.org/10.1016/j.ecolmodel.2008.12.008), [Attrill et al. 2007](https://doi.org/10.4319/lo.2007.52.1.0480) | abstract | slutningskritik, maneter och klimat |
| [Taylor & DeMaster 1993](https://storage.googleapis.com/nmfs_odp_swfsc/Publications/Files/CR/1993/9388.pdf), [Carroll et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12208499/), [Härkönen et al. 2006](https://doi.org/10.3354/dao068115), [SCOS 2022](https://www.smru.st-andrews.ac.uk/files/2023/09/SCOS-2022.pdf), [OSPAR QSR 2023](https://oap.ospar.org/en/ospar-assessments/quality-status-reports/qsr-2023/indicator-assessments/abundance-distribution-cetaceans/), [Neimanis et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8833703/), [JNCC 552](https://data.jncc.gov.uk/data/897c2037-56d0-42c8-b828-02c0c9c12d13/JNCC-Report-552-REVISED-WEB.pdf) | fulltext | däggdjurs och sjöfåglars täthetsberoende, status |
| Eberhardt 1977/2002, Fowler 1981, Gaillard 2000, IJsseldijk 2021, MacLeod 2007, Lewis 2001 | abstract | ordningen bland vitala tal, födokanalen |

Inte verifierat:
- Lorenzen & Camp 2019 (bara sekundärt).
- Myers & Cadigans koefficienter.
- Stenseth et al. 1999 och Bjørnstad et al. 1999 i PNAS (bara abstract).
- Nash et al. 2009 och Kotterba et al. 2014 (bara utdrag).
- Hästmakrillens F_MSY (antagen).
- Steepness för tobis (hittades inte).
- Kol-till-torrvikt och individvikter i zooplanktons enhetsomräkningar (antagna, påverkar inte γ).

Proben använder skriptet `dd_probe.py` i scratchpad, som inte ligger i repot.
