# Litteraturen halverar torskfiskarnas restdödlighet M0

M0 = 0,392/år går inte att försvara som restdödlighet för `gadoids`. Det gäller oavsett modellens nedgång. Rekommendationen är **M0 = 0,20/år (försvarbart 0,15–0,30)**. Dessutom bör **bentosdelen 0,121 strykas helt**, eftersom den bygger på ett sättningsfel i källans dietmatris. Icke-FG-predationen bör räknas om till **0,214**. Tillsammans ger det **`natural_mortality` ≈ 0,41/år = 0,000284 per tick** i stället för 0,683/år = 0,000468 (intervall 0,36–0,51/år). Tre jämförbara mått pekar åt samma håll. ICES flerartsmodell SMS sätter restdödligheten M1 till **0,20/år för torsk, kolja, vitling och gråsej i alla åldrar**. Den siffran är en konvention och har aldrig skattats. Den vuxna delen av Ecopath-modellen för Nordsjön 1991 har ett biomassviktat M0 på **0,18**. Livshistorieskattningarna ger en *total* naturlig dödlighet för vuxna på **0,25–0,40**, och 0,392 skulle ensamt nästan förbruka hela den. De 0,392 är en balanseringsrest i Ecopath. **66 % av den kommer från två juvenilgrupper**, kolja (EE 0,45) och gråsej (EE 0,31), vars låga EE bryter mot praxis för hur modellen ska balanseras. Säkerheten är måttlig–hög för att värdet ska ner. För själva värdet är den låg–måttlig, eftersom även 0,20 är en konvention. Intervallets bredd speglar framför allt hur stor juvenilandel den ålderslösa poolen antas ha. I modellen ger sänkningen gadoiderna ungefär **+0,55 × ΔM1** i nettotillväxt (modellprob). Det beror på att M1 också ingår i tillväxtbudgeten. Sänkningen bör ändå göras för att litteraturen kräver det, inte som justering för att hejda nedgången. Den egentliga orsaken till nedgången är att budgeten betalas vid reservfyllnaden s = 1, som aldrig nås, och den ska åtgärdas först.

Rapporten skiljer på tre slags uppgifter. **Litteratur** anges med källa. **Modellprob** avser mätningar från denna session: den frysta fix53-checkpointen från 2026-10-11 22:27 i den deterministiska probvärlden, 60×60 celler, med mortalitet, migration, strömmar, massbalans, dagsljus och temperatur påslagna. Rater anges per år på årets medelbiomassa. **Inferens** avser egna slutsatser och egna beräkningar på publicerade tabeller. Dit hör omräkningen av dietmatrisen i Mackinson & Daskalov 2007. Inferenser står under "tolkning" eller är uttryckligen markerade.

## Rekommendation per komponent

| Komponent i `gadoids.natural_mortality` | Nu (/år) | Rekommenderat (/år) | Försvarbart | Säkerhet | Nyckelkällor |
|---|---|---|---|---|---|
| M0, icke-predationsrest | 0,392 | **0,20** | 0,15–0,30 | riktning måttlig–hög; värde låg–måttlig | [SMS stock annex 2023](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/StockAnnex/NorthSea/2023_NorthSea.pdf), [Christensen 1995](https://www.aqua.dtu.dk/-/media/institutter/aqua/publikationer/dana/dana_vol_11_1__pp_1_28.pdf), [Mackinson & Daskalov 2007](https://www.cefas.co.uk/publications/techrep/tech142.pdf), [Heymans et al. 2016](https://repository.library.noaa.gov/view/noaa/31751/noaa_31751_DS1.pdf) |
| Predation från grupper som inte är FG | 0,171 | **0,214** (pool med juveniler, som idag) | 0,17–0,22; ~0,01–0,02 om poolen definieras som vuxna | låg–måttlig | [Mackinson & Daskalov 2007, dietmatris](https://www.cefas.co.uk/publications/techrep/tech142.pdf) (inferens: omräknad) |
| Predation från bentos (krabbor) | 0,121 | **0** | 0 | hög (sättningsfel, se nedan) | [Mackinson & Daskalov 2007, dietmatris](https://www.cefas.co.uk/publications/techrep/tech142.pdf) (inferens: kolumnsummor) |
| **M1 totalt** | 0,683 | **≈ 0,41** | 0,36–0,51 | låg–måttlig | summan ovan |
| **per tick**, 1 − exp(−M1/1460) | 0,000468 | **0,000284** | 0,000249–0,000352 | | |
| härlett `growth_rate` (mareld2, `--mortality on`) | 0,00200951 | ≈ 0,00164 | 0,00157–0,00178 | | `lib/world/growth_budget.py` |

Följdändringar från samma rättelse av dietmatrisen gäller par i `fg_library.yaml` och inte M1. De redovisas för fullständighetens skull (inferens, låg–måttlig säkerhet): `gadoids_preys_on_gadoids` 0,2168 → 0,202, `porpoises_preys_on_gadoids` 0,0557 → 0,065, `seals_preys_on_gadoids` 0,0299 → 0,042, `seabirds_preys_on_gadoids` 0,0078 → 0,009 och `pelagic_fish_preys_on_gadoids` 0,0799 → 0,150. Det sista paret är avbockat och påverkar därför inte körningen.

Biblioteksposten kan skrivas så här: *"M1 = M0 (0.20; SMS residual M1 convention for cod, haddock, whiting and saithe, Ecopath NS 1991 adults 0.18) + predation by non-FG groups (0.214; squid, halibut, gurnards, long-rough dab, megrim, rays, monkfish) = 0.41 /yr. The Ecopath pool residual 0.392 is not used: 66 % of it is juvenile haddock and saithe with EE 0.45/0.31. No benthos part: the crab/Nephrops/shrimp entries on 'other gadoids' in Mackinson & Daskalov 2007 Table 3.4 duplicate the megrim/halibut/large demersal columns (column sums). Was 0.000468."*

## 0,392 är en balanseringsrest som två juvenilgrupper bär

**Litteratur.** I Ecopath sätts produktionen per biomassa lika med totaldödligheten, P/B = Z = F + M2 + M0. Restdödligheten beräknas sedan som M0 = P/B × (1 − EE), där EE är den del av produktionen som modellen förklarar. För de bedömda torskfiskarna togs Z från ICES flerarts-VPA, och F räknades som fångst/biomassa. **EE är rödmarkerad för alla torskfiskgrupper utom "other gadoids"**, vilket betyder att Ecopath skattade den: M0 blir det som dietmatrisen och fångsterna lämnar oförklarat ([Mackinson & Daskalov 2007, avsnitt 13.3, 13.14 och tabell 3.3](https://www.cefas.co.uk/publications/techrep/tech142.pdf)). Någon numerisk M0-tabell finns inte i rapporten, bara stapeldiagrammet i figur 3.9. Författarna skriver dessutom att biomassan för vuxen gråsej och marulk höjdes för att få ned F/Z. Skälet var att Ecosim annars gav "unrealistic" återhämtning utan fiske ([avsnitt 4.1](https://www.cefas.co.uk/publications/techrep/tech142.pdf)). M0 är alltså delvis ett resultat av justeringar i modellen.

**Inferens (beräknat från tabell 3.3 och 3.5, area 570 000 km²).** Poolen med grupperna 13–21, 24 och 25 ger exakt bibliotekets **0,392**. Fördelningen per grupp visar varifrån värdet kommer:

| Grupp | B (t/km²) | P/B | EE | F | M0 | Bidrag till poolens 0,392 |
|---|---|---|---|---|---|---|
| Kolja juvenil | 0,284 | 2,00 | 0,45 | 0,22 | **1,09** | 0,159 (41 %) |
| Gråsej juvenil | 0,281 | 1,00 | 0,31 | 0,20 | **0,69** | 0,098 (25 %) |
| Gråsej vuxen | 0,220 | 0,95 | 0,62 | 0,53 | 0,36 | 0,041 (10 %) |
| Vitling juvenil | 0,222 | 2,36 | 0,86 | 0,11 | 0,33 | 0,037 (10 %) |
| Torsk vuxen | 0,161 | 1,19 | 0,75 | 0,77 | 0,30 | 0,025 (6 %) |
| Vitling vuxen | 0,352 | 0,89 | 0,93 | 0,52 | 0,06 | 0,011 |
| Kolja vuxen | 0,104 | 1,14 | 0,97 | 0,92 | 0,03 | 0,002 |
| Övriga (torsk juv., kummel, other gadoids) | 0,333 | | | | 0,06–0,29 | 0,020 |

De vuxna i torsk, vitling, kolja, gråsej och kummel har tillsammans **M0 0,183**. Om de två avvikande juvenilgrupperna får EE 0,95, som "other gadoids" fick som indata, sjunker poolen till **0,157**. Gråsejens biomassjustering lade till 0,09–0,16 på vuxen gråsej: med datavärdet B 0,19 i stället för 0,22 blir M0 0,20–0,27 i stället för 0,36. Samtidigt fick gråsejen nya predatorer i samma steg, så nettoeffekten är okänd.

**Litteratur om varför det är en rest.** I Nordsjömodellen för 1981 jämförde Christensen Ecopath-M0 direkt med flerartsmodellens antaganden. Ecopath gav **torsk 0,27, vitling 0,34, gråsej 0,21 och kolja 0,50**, medan MSVPA antog M0 0,20. Förklaringen var att MSVPA:s antagna "other predation" M1 inte täcks av Ecopaths predatorer. Den hamnar då i Ecopaths M0, tillsammans med utkast. MSVPA:s icke-predationsdödlighet beskrivs som parametriserad "based on very limited information, especially M0 which is assumed to take values between 0.1 and 0.2 y⁻¹" ([Christensen 1995, tabell 5 och s. 15–16](https://www.aqua.dtu.dk/-/media/institutter/aqua/publikationer/dana/dana_vol_11_1__pp_1_28.pdf)). Enligt praxis för hur Ecopath ska balanseras bör de flesta grupper ha EE "approaching, but lower than 1" ([Heymans et al. 2016](https://repository.library.noaa.gov/view/noaa/31751/noaa_31751_DS1.pdf)). Juvenilerna med EE 0,45 och 0,31 är därför de grupper en sådan granskning skulle flagga. I det anpassade Ecosim-läget gjordes fiskarnas M0 dessutom beroende av födosökstiden ([Mackinson & Daskalov 2007, avsnitt 4.1](https://www.cefas.co.uk/publications/techrep/tech142.pdf)). Inte ens källmodellen använde alltså ett konstant M0.

För Östersjön 1974 ger Tomczak et al. torsk-M0 **0,28 för vuxna** och 0,31 viktat över tre stadier. Juvenilstadiet har där EE 0,058 och samma slags artefakt (inferens, beräknat från [Tomczak et al. 2013, tabell S1](https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0075439.s001)). Harvey et al. ger för samma system och år bara 0,13 ([Harvey et al. 2003](https://academic.oup.com/icesjms/article/60/5/939/769206); värdena kommer från tidigare noter och är inte omkontrollerade). Ingen EwE-modell för Kattegatt eller Skagerrak med publicerade parametrar för torskfiskar hittades.

**Tolkning.** Ecopaths M0 är ingen biologisk mätning av sjukdom och ålderdom. Den samlar MSVPA:s antagna 0,1–0,2, oförklarad predation på juveniler, fel i F och justeringar. Spannet mellan modellerna, 0,13–0,50 per art och stadium, mäter mest balanseringen.

## ICES räknar med 0,20 i restdödlighet för alla fyra arterna

**Litteratur.** SMS stock annex säger ordagrant: "M1 (residual natural mortality) by quarter is set to 0.05 for the species cod, whiting, haddock, saithe". Det blir **0,20/år**. Det enda undantaget är torsk från 3 års ålder sedan 2011, som har 0,353. Det är ett ad hoc-tillägg för utvandring till västra Skottland och ingen biologi ([SMS stock annex 2023, avsnitt 3.1.3 och 3.4.4](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/StockAnnex/NorthSea/2023_NorthSea.pdf); [natmor1.in](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/natmor1.in)). Samma 0,2 gällde i alla nyckelkörningar 2011–2020 och i den klassiska MSVPA:n, där värdena "were taken from the last MAWG (ICES 1997a) and copied forwards in time" ([ICES CM 2002/D:04](https://www.ices.dk/sites/pub/CM%20Doccuments/2002/D/D0402.PDF)). Inget dokument motiverar 0,2 biologiskt. WGSAM behandlar själv M1 som en osäker indataparameter som ska känslighetstestas ([WGSAM Review Criteria](https://ices-eg.github.io/wg_WGSAM/ReviewCriteria.html), endast utdrag).

Konventionen innehåller känd predation. MSVPA:s grupp "Other predators" var tänkt att ta över en del av M1, men "these values were not lowered" ([ICES CM 2002/D:04](https://www.ices.dk/sites/pub/CM%20Doccuments/2002/D/D0402.PDF)). För rödspätta används en M1 där sälpredationen redan ingår ([SMS stock annex 2023](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/StockAnnex/NorthSea/2023_NorthSea.pdf)). Lorenzen skriver att M1 "subsumes all causes of mortality not explicitly accounted for by predation terms (including any predators not explicitly modeled)". Han konstaterar också att termen "has received scant scientific attention" och borde vara storleksberoende ([Lorenzen 2022](https://sedarweb.org/documents/sedar76rd13_size-and-age-dependent-natural-mortality-in-fish-populations-biology-models-implications-and-a-generalized-length-inverse-mortality-paradigm-pdf)).

**Predationen på vuxna är liten utom för vitling.** Medel för 2018–2022 i 2023 års nyckelkörning (summerat över kvartal ur [natmorm1m2.out](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/natmorm1m2.out) och [partial_m2.out](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/NorthSeaKeyRun_2023/partial_m2.out)):

| Art | M2 juvenil (ålder 0 / 1) | M2 vuxen | Dominerande predatorer på vuxna |
|---|---|---|---|
| Torsk | 1,89 / 1,02 | ålder 3: 0,24; ålder 4: 0,04; 5+: 0 | gråsäl |
| Kolja | 1,02 / 0,92 | ålder 3: 0,15; 4: 0,10; 5–8: 0,02–0,07 | gråsäl, torsk |
| Vitling | 1,54 / 0,86 | ålder 3–5: 0,27–0,31 | tumlare (66–74 %) |
| Gråsej | 0 | 0 | ingen (bara predator i SMS) |

Den publicerade tabellen i annexet stämmer med filerna. Torsk 2018 har där M1 + M2 = 2,146, 1,300, 1,028, 0,594 och 0,404 för åldrarna 0–4 ([SMS stock annex 2023, tabell 5-4](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/StockAnnex/NorthSea/2023_NorthSea.pdf)).

**Enartsbedömningarna** importerar samma tal. Kolja och vitling använder SMS-M, och WGSAM rekommenderar det ([WGSAM 2023](https://ices-library.figshare.com/articles/report/Working_Group_on_Multispecies_Assessment_Methods_WGSAM_outputs_from_2023_meeting_/25020968), endast utdrag; [WGSAM 2024](https://archimer.ifremer.fr/doc/00873/98452/107912.pdf)). Torsk på norra shelfen använder utjämnad SMS-M ([WKBCOD 2023](https://archimer.ifremer.fr/doc/00841/95303/103049.pdf)). Gråsej gick 2024 från fast M = 0,2 till en Lorenzen-kurva skalad till **0,2 vid ålder 9**, med motiveringen "0.2 as residual mortality may be a reasonable assumption for the terminal age". Profillikelihooden föredrog svagt ett fast M kring 0,3 ([WKBGAD 2024](https://ices-library.figshare.com/articles/report/Benchmark_workshop_on_selected_haddock_and_saithe_stocks_WKBGAD_/25002470)). Kattegatttorsken har "constant natural mortality of 0.2". Ökande säldödlighet antas finnas men tas upp av fångstmultiplikatorer i SAM ([Stock annex Kattegatt 2024](https://ices-library.figshare.com/articles/report/Stock_Annex_Cod_in_Subdivision_21_Kattegat_/18622304)).

**Tolkning.** 0,20 är ICES standardvärde för allt utom modellerade predatorer och fiske. Som ren icke-predationsdödlighet är det snarare en övre gräns, eftersom det rymmer predatorer som SMS saknar. Det är den direkta motsvarigheten till Mareld-poolens M0, och 0,392 är dubbelt så stort.

## Livshistorieskattningar ger total vuxendödlighet, inte en rest

**Litteratur.** De empiriska skattarna beräknar total naturlig dödlighet M, alltså predation plus allt annat. Rekommenderade former är Then et al. 4,899·tmax^−0,916 ([Then et al. 2015](https://www.usgs.gov/publications/evaluating-predictive-performance-empirical-estimators-natural-mortality-rate-using), endast abstract) och Hamel & Cope 5,40/tmax ([Hamel & Cope 2022](https://sedarweb.org/documents/sedar76rd15_development-and-considerations-for-application-of-a-longevity-based-prior-for-the-natural-mortality-rate-pdf)). Därtill kommer de storleksberoende skattarna av Gislason, Charnov och Lorenzen. Deras koefficienter är hämtade ur [fishmethods](https://raw.githubusercontent.com/cran/fishmethods/master/R/M.empirical.R) och inte ur originalartiklarna. Maxåldrarna kommer från FishBase: torsk och gråsej 25 år, kolja, vitling och kummel 20 år ([FishBase torsk](https://www.fishbase.se/summary/Gadus-morhua.html), [kolja](https://www.fishbase.se/summary/Melanogrammus-aeglefinus.html), [vitling](https://www.fishbase.se/summary/Merlangius-merlangus.html), [gråsej](https://www.fishbase.se/summary/Pollachius-virens.html)). Tillväxtparametrarna kommer från FishBase och [mizers Nordsjöuppsättning](https://raw.githubusercontent.com/sizespectrum/mizer/master/inst/extdata/NS_species_params.csv).

**Inferens (beräknat, SSB-viktat).** Vikterna är lekbiomassor ur ICES SAG ([vitling](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19484), [kolja](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19508), [gråsej](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19578), torsk som summan av [19661](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19661), [19662](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19662) och [19663](https://sag.ices.dk/SAG_API/api/SummaryTable?assessmentKey=19663)). År 1991 var fördelningen torsk 16 %, kolja 13 %, vitling 45 % och gråsej 26 %.

| Metod | Vuxen total M, vikter 1991 | Medelvikter 2005–24 |
|---|---|---|
| Hoenig 1983, global tmax | 0,19 | 0,19 |
| Hamel & Cope 2022, global tmax | 0,25 | 0,25 |
| Then 2015, global tmax | 0,29 | 0,29 |
| Pauly 1980 (9 °C) / Then 2015, tillväxtform | 0,34 / 0,38 | 0,33 / 0,37 |
| SMS total M (M1 + M2), vuxna | 0,33 | 0,31 |
| Lorenzen 1996 vid vuxenvikter | 0,45 | 0,43 |
| Then 2015, tmax 12–15 år | 0,45 | 0,44 |
| **SMS M1, icke-predation** | **0,20** | **0,20** |

**Tolkning.** Den jämförbara siffran för M0 är M1 = 0,20. Mot den totala vuxendödligheten, centralt 0,25–0,40, skulle 0,392 som rest lämna nästan inget utrymme åt predation. Med Mareld-poolens icke-FG-predation och FG-predation ovanpå skulle totalen hamna över nästan alla skattningar.

**Motsatt synpunkt.** Fyra observationer drar uppåt. Med fiskeförkortade maxåldrar på 12–15 år ger maxåldersskattarna 0,36–0,50. Märkningsförsök på kusttorsk i Skagerrak ger ett implicit M på ~0,49: årlig överlevnad 33 %, varav 44 % av dödligheten är "dead by other causes" ([Kleiven et al. 2016](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0149595)). I södra S:t Lawrencebukten steg vuxen torsks M till ~0,8 ([CBC 2019](https://www.cbc.ca/lite/story/1.5255266); [Swain et al.](https://int-res.com/abstracts/meps/v442/p149-167), endast abstract). Lorenzen menar att M1 kan vara högre och mer storleksberoende än predationsmodellerna antar ([Lorenzen 2022](https://sedarweb.org/documents/sedar76rd13_size-and-age-dependent-natural-mortality-in-fish-populations-biology-models-implications-and-a-generalized-length-inverse-mortality-paradigm-pdf)). Inget av detta är ändå en icke-predationsrest. Märkningsvärdet är en övre gräns för ung fisk med en median på 40 cm, där tagg-förlust och utvandring ingår. S:t Lawrence-ökningen tillskrivs gråsälspredation, med 18 % "normal" dödlighet = 0,20/år. Maxåldersskattningarna är total M. Lorenzens argument talar för storleksberoende, inte för en högre rest hos vuxna. De motiverar den övre änden 0,30 men inte 0,39.

## Ett vuxendominerat torskfiskförråd landar på M0 0,15–0,20

`gadoids` har ingen åldersstruktur. Biomassvikterna i Ecopath ger vitling 29 %, gråsej 26 %, kolja 20 %, torsk 12 % och "other gadoids" 12 %, varav **44 % är juveniler** av de fyra huvudarterna. Det är ett fiskat tillstånd. Biblioteket tar M = P/B − F, så modellen är ofiskad per konstruktion. Utan fiske förskjuts åldersstrukturen mot vuxna, och då ska vikterna förskjutas likadant.

**Inferens (beräknat från Mackinson & Daskalov 2007).** Biomassviktat M0 för olika avgränsningar:

| Avgränsning | M0 (/år) | Kommentar |
|---|---|---|
| Vuxna torsk, vitling, kolja, gråsej och kummel | **0,183** | gråsejens 0,36 är uppblåst av biomassjusteringen |
| Samma, med gråsejens databiomassa (M0 0,20–0,27) | 0,14–0,16 | |
| Juveniler, med EE 0,95 för kolja och gråsej | 0,144 | |
| Hela poolen med samma juvenilrättelse | **0,157** | |
| Hela poolen som den trycktes | 0,392 | 66 % från två EE-avvikare |
| SMS M1 (konvention) | 0,20 | |
| Nordsjön 1981 (åldrar sammanslagna, inklusive utkast) | 0,34 | |
| Östersjön 1974, vuxen torsk | 0,28 | |

**Tolkning.** Rättar man de två juvenilgrupperna blir M0 nästan oberoende av åldersvikten. Vuxna ligger på 0,14–0,18 och rättade juveniler på 0,14, så varje blandning hamnar på 0,15–0,18. SMS-konventionen ligger strax över med 0,20. **Rekommendationen 0,20** väljer ICES etablerade värde och ligger i den övre halvan av de Ecopath-interna skattningarna. Det är rimligt, eftersom Mareld saknar Ecopaths sjukdoms- och utkastkomponenter som separata termer. Intervallet har tre ankare. **0,15** gäller ren vuxenpool och juveniler enligt praxis. **0,20** är SMS. **0,30** motsvarar en viss kvarvarande juvenilandel eller Östersjöns och 1981 års nivå.

Åldersfrågan påverkar de andra två komponenterna mer än M0. Icke-FG-predationen är nästan helt juvenil. I den rättade matrisen (se nästa avsnitt) står juveniler för 0,119 och "other small gadoids" för 0,086 av poolens 0,214, medan de vuxna står för 0,009. Definieras poolen som vuxen faller icke-FG-delen till ~0,01–0,02 och M1 till ~0,22. Men då måste parens `predation_mortality` också räknas om för vuxna: kannibalism 0,108, tumlare 0,024 och sälar 0,042. Det är en större omkalibrering. Rekommendationen behåller därför poolen som den är, med juveniler, så att M1, M2 och `r_max` vilar på samma avgränsning.

## Bentosdelen 0,121 är ett sättningsfel i dietmatrisen

Raderna 24–25 ("other gadoids") i Mackinson & Daskalovs dietmatris (tabell 3.4) ser kopierade ut från en annan sida. **Inferens (kontrollerat mot pdf:ens textlager och den tolkade matrisen).** Raderna trycks med identiska värden på två sidpar. För raderna 24–25 har predatorkolumnerna 1–12 samma värden som 13–24, och för raderna 23–25 har kolumnerna 37–48 samma värden som 49–60. Rad 25 har till exempel 0,227 både under megrim (kolumn 42) och large crabs (kolumn 54), och 0,125 både under spurdog (6) och vuxen kolja (18). Kolumnsummorna avgör vilken kopia som är äkta:

| Predator (kolumn) | Kolumnsumma med de dubblerade raderna | utan dem |
|---|---|---|
| Megrim (42) | 1,001 | 0,534 |
| Halibut (43) / Turbot (41) | 1,001 / 1,002 | 0,886 / 0,872 |
| Toothed whales (2) / Spurdog (6) | 1,001 / 0,981 | 0,953 / 0,856 |
| **Large crabs (54)** | **1,468** | **1,001** |
| **Nephrops (55)** / **Shrimp (58)** | **1,115** / **1,026** | **1,000** / **1,000** |

Kolumnerna 37–48 summerar till 1 bara med raderna, och bentoskolumnerna summerar till 1 bara utan dem. Bentosvärdena är alltså kopior. För kolumnerna 1–12 mot 13–24 är beviset svagare: 1–12 summerar till ~1 med raderna, men torskfiskkolumnerna 13–24 summerar inte till 1 i någon variant (vuxen gråsej 1,53, kummel 1,54). Hela bentosdelen av M1 (krabbor, havskräfta och räkor på "other small gadoids", 1,235/år på den gruppen) är en artefakt. Den drabbade aldrig torsk, vitling, kolja eller gråsej, eftersom deras M2 från bentos är 0. Samma fel ger falsk kannibalism på "other gadoids". Det motsvarar ~0,03 av parets 0,217.

Rad 25:s totala M2, 2,04/år, ligger fast, eftersom EE 0,95 var indata. Stryks kopiorna fördelas den om på de äkta predatorerna. Med den tidigare rapportens metod blir poolens fördelning: **icke-FG 0,214** (bläckfisk 0,036, hälleflundra 0,030, knot 0,029, lerskädda 0,029, glasvar 0,019, rockor 0,03, marulk 0,013 med flera), **bentos 0**, kannibalism 0,202, makrill och taggmakrill 0,150, tumlare 0,065, sälar 0,042 och sjöfåglar 0,009. Bibliotekets 0,171 innehöll dessutom blåvitling och vitlinglyra (0,030), och båda försvinner med rättelsen. Säkerheten är hög för att artefakten finns. Den är låg–måttlig för omfördelningen, eftersom den följer en skalningskonvention och inte källans egen balansering.

## I modellen verkar M1 via tillväxtbudgeten, och den ska rättas först

**Modellprob (fix53, 22:27, probvärlden).**

| År | Biomassa (t) | Medel-s | Produktion | M1-förlust | Predation | Svält |
|---|---|---|---|---|---|---|
| 1 | 2500 → 2152 | 0,707 | 0,609 | 0,683 | 0,077 | 0,000 |
| 2 | 2152 → 1968 | 0,724 | 0,661 | 0,683 | 0,066 | – |
| 3 | 1968 → 1840 | 0,728 | 0,673 | 0,683 | 0,056 | – |

Bokföringen går ihop (inferens): produktion − förluster = −0,151, −0,088 och −0,066. Det motsvarar ln(B₁/B₀) = −0,150, −0,089 och −0,067. M1 är **9–12 gånger** den realiserade predationen och den klart största förlusten.

**Modellprob, budgeten.** Vid s = 1 är budgeten 1,467/år = `r_max` 0,511 + M1 0,683 + avbockad M2 0,273 (kannibalism 0,2168 + tumlare 0,0557). Produktionen blir budget × (s − u)/(1 − u) med u = 0,5. s = 1 nås aldrig, eftersom hungergrinden stänger intaget när s → 1 medan ämnesomsättningen fortsätter. Jämvikten ligger vid s ≈ 0,74. Den följer av intagstaket ~25 MJ/t/tick, kostnaderna 9–13 MJ/t/tick och tillväxtdebiteringen ~11(s − 0,5) MJ/t/tick. Den realiserade kannibalismen är **~0,001/år mot budgeterade 0,217**: policyn lägger 0,02 % av sin ätsannolikhet på gadoider.

Eftersom M1 ingår i budgeten flyttar en ändring ΔM1 nettotillväxten med ungefär −ΔM1 × (1 − f), där f = (s − u)/(1 − u) ≈ 0,45. Det ger ~0,55 × ΔM1. Statiska uppskattningar vid f 0,45 och realiserad predation 0,06:

| M1 (/år) | Innehåll | Netto (/år) | Typ |
|---|---|---|---|
| 0,683 | nuvarande | −0,08 | modellprob |
| 0,606 | utan bentos, M0 kvar på 0,392 | −0,04 | inferens |
| 0,562 | utan bentos (bibliotekets 0,171) | −0,02 | modellprob |
| 0,512 | | +0,01 | modellprob |
| **0,414** | **rekommenderat: 0,20 + 0,214** | **+0,07** | inferens |
| 0,392 | bara M0 | +0,08 | modellprob |

**Tolkning.** Två kopplingar gör att sänkningen inte ska användas som justeringsratt. Den första är att budgeterad men orealiserad kannibalism ger gadoiderna ~0,217 × 0,45 ≈ **0,10/år i gratis produktion** för en förlust de aldrig tar. Det är lika stort som effekten av M0-sänkningen, fast med motsatt tecken. Den andra är att när budgeten betalas vid en nåbar referensfyllnad i stället för s = 1, blir en M1-ändring nästan neutral vid den fyllnaden. Både produktion och förlust flyttas då lika mycket. M0 styr därefter främst omsättningen och hur snabbt nettot reagerar när s avviker från referensen, inte nettonivån. Ordningen som diskuterats med användaren håller därför: rätta referensfyllnaden först, och lägg sedan in M0 = 0,20 och bentosrättelsen på litteraturgrund. **Litteraturen ensam motiverar sänkningen.** Den skulle göras även om gadoiderna växte i modellen. Observera att `--mortality` som standard är av på `train.py` och `inference.py`. Med `--mortality off` lämnar laddaren M1 utanför `growth_rate`, och då har ändringen ingen effekt alls.

## Så testas det i modellen

Ändra `natural_mortality` under `gadoids` i `fgconfig/fg_library.yaml` från 0.000468 till **0.000284** och uppdatera kommentaren enligt förslaget ovan. Ändra inte `growth_rate` för hand. Laddaren härleder det, men det lagrade dokumentationsvärdet bör uppdateras till vad laddaren räknar fram för mareld2 med `--mortality on`, ≈ 0.00164. Kör `python3 -m pytest tests/ -q`, eftersom tester kan jämföra lagrat och härlett värde. Kör sedan `python3 tools/probes/budget_gate.py` för att se att gadoiderna fortfarande kan stänga sin energibudget.

Upprepa därefter treårsprobet med samma frysta fix53-checkpoint. Inga observationskanaler ändras, så den laddar. Förväntat statiskt utfall är att nettot går från ~−0,07–−0,08 till ~+0,07/år om s och predationen står still (inferens). I praktiken stiger biomassan och sänker s via födobegränsning, så utfallet bör ligga lägre. Gör samma körning efter rättelsen av referensfyllnaden. Då bör skillnaden mellan M1 0,683 och 0,414 krympa kraftigt, och det är testet på att M0 inte längre fungerar som justeringsratt. Rättelserna av parvärdena (kannibalism, tumlare, sälar, pelagisk fisk) bör läggas in i ett separat steg, så att effekterna går att skilja åt. Avsluta med `python3 tools/viability.py` som mekanikkontroll och en 1-generations träningsrunda. Omdömet kräver en omtränad jämförelse med `--mortality on`.

## Källäget: vad som bara bygger på utdrag

Ingen källa delar upp vuxen torskfiskdödlighet i sjukdom, parasiter, ålderdom och lekstress för Nordsjön. Alla icke-predationsvärden är konventioner eller balanseringsrester.

| Källa | Läst | Kommentar |
|---|---|---|
| [Then et al. 2015](https://www.usgs.gov/publications/evaluating-predictive-performance-empirical-estimators-natural-mortality-rate-using) | abstract | koefficienterna via fishmethods |
| [Gislason et al. 2010](https://research.wur.nl/en/publications/size-growth-temperature-and-the-natural-mortality-of-marine-fish) | abstract | |
| [Lorenzen 1996](https://www.gov.uk/dfid-research-outputs/the-relationship-between-body-weight-and-natural-mortality-in-juvenile-and-adult-fish-a-comparison-of-natural-ecosystems-and-aquaculture) | abstract | koefficienten 3,0 bara via fishmethods |
| [Swain et al.](https://int-res.com/abstracts/meps/v442/p149-167), [AFS 2014](https://afs.confex.com/afs/2014/webprogram/Paper16609.html), [AFS 2016](https://afs.confex.com/afs/2016/webprogram/Paper24555.html) | abstract | S:t Lawrence; 18 %/55–57 % via [CBC](https://www.cbc.ca/lite/story/1.5255266) |
| [WGSAM 2023 (figshare)](https://ices-library.figshare.com/articles/report/Working_Group_on_Multispecies_Assessment_Methods_WGSAM_outputs_from_2023_meeting_/25020968) | utdrag | rekommendationen om SMS-M |
| [WKBALT 2017](https://archimer.ifremer.fr/doc/00415/52643) | utdrag | sälar och Kattegatttorsk |
| [WGSAM Review Criteria](https://ices-eg.github.io/wg_WGSAM/ReviewCriteria.html) | utdrag | |
| [Julliard et al. 2001](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.1890/1051-0761(2001)011[0540:NMAFMI]2.0.CO;2), [fjordtorsk](https://www.researchgate.net/publication/222045226_Sampling_and_a_mortality_model_of_a_Norwegian_fjord_cod_Gadus_morhua_L_population), [ICES JMS 2025](https://academic.oup.com/icesjms/article/82/8/fsaf137/8237823) | utdrag | används inte för siffrorna |
| Tretyak (ICES CM 2000/N:09), [PMC3352523](https://pmc.ncbi.nlm.nih.gov/articles/PMC3352523/) | utdrag | NEA-torsk 0,15–0,25, ej verifierat, används inte |
| [Baudron et al. 2011](https://onlinelibrary.wiley.com/doi/10.1111/j.1095-8649.2011.02940.x) | utdrag | |
| [Harvey et al. 2003](https://academic.oup.com/icesjms/article/60/5/939/769206) | tidigare noter | ej omkontrollerat |

Lästa i fulltext eller som rådata är [SMS stock annex 2023](https://raw.githubusercontent.com/ices-eg/wg_WGSAM/master/StockAnnex/NorthSea/2023_NorthSea.pdf) och nyckelkörningens indata- och utdatafiler, [WGSAM 2024](https://archimer.ifremer.fr/doc/00873/98452/107912.pdf), [WKBCOD 2023](https://archimer.ifremer.fr/doc/00841/95303/103049.pdf), [WKBGAD 2024](https://ices-library.figshare.com/articles/report/Benchmark_workshop_on_selected_haddock_and_saithe_stocks_WKBGAD_/25002470), [Kattegatt-annexet](https://ices-library.figshare.com/articles/report/Stock_Annex_Cod_in_Subdivision_21_Kattegat_/18622304), [ICES CM 2002/D:04](https://www.ices.dk/sites/pub/CM%20Doccuments/2002/D/D0402.PDF), [Mackinson & Daskalov 2007](https://www.cefas.co.uk/publications/techrep/tech142.pdf), [Christensen 1995](https://www.aqua.dtu.dk/-/media/institutter/aqua/publikationer/dana/dana_vol_11_1__pp_1_28.pdf), [Tomczak et al. 2013 S1](https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0075439.s001), [Heymans et al. 2016](https://repository.library.noaa.gov/view/noaa/31751/noaa_31751_DS1.pdf), [Hamel & Cope 2022](https://sedarweb.org/documents/sedar76rd15_development-and-considerations-for-application-of-a-longevity-based-prior-for-the-natural-mortality-rate-pdf), [Lorenzen 2022](https://sedarweb.org/documents/sedar76rd13_size-and-age-dependent-natural-mortality-in-fish-populations-biology-models-implications-and-a-generalized-length-inverse-mortality-paradigm-pdf), [Kleiven et al. 2016](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0149595) och ICES SAG-serierna. WGSAM 2015 Annex 6 (EwE-nyckelkörningen för Nordsjön 1991–2013) öppnades inte. Det är alltså okänt om den ombalanserade 1991 års bas och rättade dietmatrisen.

## Slutsats

Den viktigaste lärdomen är att bibliotekets M1 för gadoider ärvde tre olika slags fel från Ecopath. M0 bygger på en balanseringsrest som domineras av två juvenilgrupper. Bentosdelen bygger på ett tryckfel. Icke-FG-delen bygger delvis på samma felaktiga rad. Ingen av dem är biologi om torskfiskar. Rättat landar M1 nära 0,41. Det sammanfaller nästan med det gamla M0, men nu med rätt innehåll: ICES restkonvention plus predation från bläckfisk, plattfisk, knot och rockor som Mareld inte modellerar. Samma derivation gav pelagisk fisk M0 0,39 i den tidigare rapporten. Den bör granskas på samma sätt, både EE per grupp och dietmatrisens dubblerade rader, eftersom raderna 23–25 också omfattar vitlinglyra (inferens).

För modellens nedgång är M0 en sekundär fråga. Med budgeten betald vid ett s som aldrig nås, och 0,217 i kannibalism som aldrig äts, kan M0 bara förskjuta en obalans som har en annan orsak. När referensfyllnaden väl är rättad blir M0 vad det borde vara: en omsättningsparameter förankrad i ICES 0,20, inte en ratt för gadoidernas trend.
