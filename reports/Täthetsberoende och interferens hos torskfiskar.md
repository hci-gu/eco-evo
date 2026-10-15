# Interferensen kan inte bära torskfiskarnas täthetsreglering

Litteraturen stöder att torskfiskarna har en interferens över noll, men en svag sådan. Dagens `interference: 1.0` ger redan ett täthetsberoende i intag och produktion som ligger inom de empiriska intervallen. **Rekommendationen är att låta `w` ligga kvar på 1,0, eller höja den måttligt till ~2 (försvarbart intervall 0,5–3)**. Den ska inte användas som populationsbroms. För att stoppa gadoidernas tillväxt med frysta policyer krävs `w` ≈ 11–30, och det ligger en storleksordning över allt litteraturen ger stöd för. Även då är tillväxten tillbaka andra året. Den broms som saknas är en **förlust**, inte ett sämre intag. Den förlust som gick ut med kannibalismen är i North Sea SMS nästan proportionell mot tätheten: 0,016 + 0,0875·B per år, med B i t/km². Den bästa kandidaten är därför en täthetsberoende M1 per cell. Ingen enskild mekanism som litteraturen stöder räcker dock för att hålla gadoiderna vid 2500 t. Antingen står 2500 t för ett fiskat läge som modellen saknar fiske för, eller så är det inte modellens jämviktsläge.

Rapporten skiljer på tre slags uppgifter. **Litteratur** anges med källa. **Modellprob** avser körningar med en fryst kopia av fix55-policyerna i probvärlden (60×60, mortalitet, migration, strömmar och dagsljus/temperatur), två simulerade år per körning. **Inferens** avser egna slutsatser och är uttryckligen markerade. Underlaget finns i fyra noter i `research_notes/Täthetsberoende och interferens hos torskfiskar/`: `predator_interference.md`, `gadoid_density_dependent_growth.md`, `regulation_and_models.md` och `model_context.md`.

## Rekommendation i korthet

| Fråga | Svar | Säkerhet | Nyckelkällor |
|---|---|---|---|
| Ska gadoidernas `w` vara > 0? | Ja | måttlig | [DeLong & Vasseur 2011](https://doi.org/10.1186/1472-6785-11-1), [Skalski & Gilliam 2001](https://doi.org/10.1890/0012-9658(2001)082[3083:FRWPIV]2.0.CO;2), [Arditi et al. 2004](https://doi.org/10.1016/j.crvi.2004.06.007) |
| Hur stor? | **1,0–2** (0,5–3 försvarbart) | låg | [Lorenzen & Enberg 2002](https://doi.org/10.1098/rspb.2001.1853), [Denechaud et al. 2020](https://doi.org/10.1111/gcb.15298), [Mackinson & Daskalov 2007](https://www.cefas.co.uk/publications/techrep/tech142.pdf), modellprob |
| Kan `w` hålla gadoiderna vid 2500 t? | Nej, det kräver `w` ≈ 11–30 och ger svält | hög (modellprob) | `model_context.md` |
| Vad gick förlorat med kannibalismen? | En täthetsproportionell förlust, ~0,23/år vid Nordsjöns tätheter | måttlig | SMS 2023 key run ([WGSAM](https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2023)) |
| Ersättning | Täthetsberoende M1 per cell, `M0 + k·B(c)`, k ≈ 0,09–0,12/år per t/km² | låg–måttlig | SMS 2023, [Fromentin et al. 2001](https://doi.org/10.1890/0012-9658(2001)082[0567:EODDAS]2.0.CO;2) |

## Interferens är vanlig, men ingen har mätt den hos torskfiskar

Det finns ett brett stöd för att rovdjurets egen täthet påverkar intaget per individ. Skalski & Gilliam fann att modeller där även rovdjurstätheten ingår passade bättre än ren Holling-respons i **18 av 19** dataserier ([Skalski & Gilliam 2001](https://doi.org/10.1890/0012-9658(2001)082[3083:FRWPIV]2.0.CO;2), endast abstract). DeLong & Vasseur gick igenom skattningar av Hassell–Varleys interferensexponent m. Av **51** skattningar bedömdes 35 som icke-snedvridna, med **median 0,7 och medel 0,8**. Bara en såg ut som ren bytesberoende respons och nio som kvotberoende ([DeLong & Vasseur 2011](https://doi.org/10.1186/1472-6785-11-1); tidskriften är *BMC Ecology*, inte *Ecology* som jag skrev i chatten). Teorin säger att svag till måttlig Beddington–DeAngelis-interferens stabiliserar dynamiken ([Arditi et al. 2004](https://doi.org/10.1016/j.crvi.2004.06.007)).

För torskfiskar är underlaget däremot tunt:

- **Fisk saknas nästan i synteserna.** DeLong & Vasseur nämner ingen fiskart. I Stouffer & Novaks sammanställning finns två fiskrovdjur, och bara i en serie (guldfisk på *Daphnia*) varierades rovdjurstätheten ([Stouffer & Novak 2021](https://doi.org/10.1111/ele.13670)). FoRAGE (2598 funktionella responser) anpassar bara typ II och säger därför inget om interferens.
- **Inget interferensexperiment på torsk, vitling, kolja eller gråsej hittades.**
- **Det bästa fälttestet på cellskala är ett nollresultat.** Omkring 2000 torskmagar från Barents hav, 12 år, analyserade med 2 km upplösning: torskens täthet hade *"no effect on capelin consumption"* ([Fall et al. 2021](https://doi.org/10.1111/ecog.05473), abstract).
- Essington & Hansson anpassade rovdjursberoende responser för Östersjötorsk på sill och skarpsill ([Essington & Hansson 2004](https://doi.org/10.1139/f04-146)). Parametervärdena gick inte att läsa. En sekundär uppgift om "0,6" kommer från en Crowley–Martin-modell och ska inte föras över till `w`.

Det går inte heller att räkna om skattningarna till modellens enhet (km²/t per cell). Läses m = 0,7 skalfritt vid 1 t/km² blir `w` ≈ 2,3. Förankras samma m i tanktätheter blir `w` ≈ 3·10⁻⁴ för en torskfisk på 2 kg. Det är fyra storleksordningar mellan tolkningarna. **Inferens:** synteserna säger att `w` > 0 och att interferensen typiskt är måttlig. Själva siffran går inte att härleda ur dem, och fältdata på torsk talar för den svaga änden.

## Täthetsberoende tillväxt är svag, och w = 1 ligger redan inom den

Det närmaste ett kalibreringsmål kommer är hur mycket tillväxten per individ sjunker när beståndet blir tätare. Fynden visar en svag effekt:

- **Lorenzen & Enberg (2002)** fann signifikant täthetsberoende tillväxt i **9 av 16** bestånd, med sambandet g = 3,3/B ([Lorenzen & Enberg 2002](https://doi.org/10.1098/rspb.2001.1853), fulltext). Det betyder att L∞ sjunker ~3,3 cm när biomassan går från medel till dubbla medel: ~3 % för torskstor fisk och 8–10 % för fisk på 30–40 cm. Nordsjötorsk (KI korsar 0), kolja och vitlinglyra var icke-signifikanta.
- **Zimmermann et al. (2018) och Rindorf et al. (2022)** finner täthetsberoende tillväxt i 26 % respektive 69 % av bestånden, beroende på metod, och i den sena tillväxten hos 54 % ([Zimmermann et al. 2018](https://doi.org/10.1111/1365-2656.12800), [Rindorf et al. 2022](https://doi.org/10.1111/faf.12650)).
- **Den renaste torskskattningen gäller nordostarktisk torsk:** otolittillväxten minskade **21,78 %** över torskbiomassor från 707 till 4999 kt, med temperatur, lodda och fiske i samma modell ([Denechaud et al. 2020](https://doi.org/10.1111/gcb.15298)). Det motsvarar ~8 % per fördubbling i tillväxt och ~3–5 % i intag (omräkning i noten).
- **Isländsk kolja:** konditionsindexet (Fulton K) håller sig inom 0,71–0,79 över ett 6–10-faldigt täthetsintervall ([Jonsdottir et al. 2024](https://doi.org/10.1093/icesjms/fsad108)).
- **Östersjötorsken duger inte som mål.** Svedäng & Hornborgs L∞-samband ger −55 % per fördubbling, men det är sammanblandat med syrebrist, bytesbrist och parasiter ([Svedäng & Hornborg 2014](https://doi.org/10.1038/ncomms5152), *Nature Communications*, inte *ICES JMS* som jag skrev i chatten). Casini et al. finner att torsktätheten **inte är signifikant** för konditionen 1976–2014 (p = 0,11, 2,3 % av devians), medan syrebristens utbredning ensam förklarar 67 % ([Casini et al. 2016](https://doi.org/10.1098/rsos.160416)). Det rättar det jag påstod i chatten, där jag använde Casini som stöd för täthetsberoende kondition.
- **De stora historiska viktfallen följer bytet, inte torsktätheten.** Barentshavstorskens vikt per ålder föll ~50 % på 1980-talet när loddan kollapsade vid *låg* torskbiomassa.

Noten sammanfattar det till att intaget per individ faller **3–6 % per fördubbling (högst ~10 %)** och produktionen per individ **5–15 %**.

**Modellprob.** Med `w` = 1 och frysta fix55-policyer gav en fördubbling av startbiomassan följande. Intaget per individ minskar med interferensfaktorn R, som mäts viktat med intaget.

| Fördubbling | R | Bruttoproduktion/år | Intaget minskar | Produktionen minskar |
|---|---|---|---|---|
| 1250 → 2500 t | 0,957 → 0,941 | 0,872 → 0,834 | 1,7 % | 4 % |
| 2500 → 5000 t | 0,941 → 0,915 | 0,834 → 0,741 | 2,8 % | 11 % |

Produktionsfallet innehåller också att sillen tar slut. **Inferens:** dagens `w` = 1 ger alltså ett täthetsberoende i den nedre delen av det empiriska intervallet. Det beror på att policyn sprider gadoiderna över nästan hela rutnätet (2000–3600 av 3600 celler; p50 1,4 t/km², p90 1,9). Litteraturens 3–6 % i intag motsvarar ungefär `w` 1–2. Över ~3 överskrider modellen även den övre gränsen på ~10 %.

## EwE-vulnerabiliteten pekar på w 2–7, inte under 1

I Ecosim beskriver "foraging arena" ett rovdjursberoende med samma matematiska form som en Beddington–DeAngelis-term. Vulnerabiliteten K anger hur många gånger bytets predationsdödlighet högst kan öka när rovdjursbiomassan växer obegränsat. Mackinson & Daskalovs anpassade Nordsjömodell för 1991 har K = **6** för vuxen torsk, kolja och gråsej och **3** för vuxen vitling ([Mackinson & Daskalov 2007, tabell 4.5](https://www.cefas.co.uk/publications/techrep/tech142.pdf)).

Två av noterna räknar om K till `w` med `w·B₀ = 1/(K−1)`, och den ena drar slutsatsen att dagens `w` = 1 redan är *starkare* än Ecosim. Den omräkningen glömmer mättnadstermen. I modellens form är rovdjurets totala intag `B·a·B_prey / (1 + S + w·B)`, med S = a·h·B_prey. När B går mot oändligheten närmar sig intaget a·B_prey/w. Kvoten mot basläget blir då

    K = (1 + S + w·B₀) / (w·B₀)   =>   w·B₀ = (1 + S) / (K − 1)

**Inferens (egen härledning).** Mättnadstermen S är stor i dagens modell: ~13 för zooplankton och ~18 för bentos (modellprob, årsmedel). Herring på ~100 är typ III och går inte att jämföra rakt av. Med B₀ = 1,37 t/km² ger K = 6 ett `w` ≈ 2,0 (zooplankton) till 2,8 (bentos). K = 3 ger `w` ≈ 5–7. Ecosim pekar alltså på ett *starkare* `w` än dagens, inte ett svagare. Två förbehåll gäller. Anpassade K-värden kodar också bytessidans refuger, inte bara interferens. Och spridningen är stor: isländska anpassningar ger allt från K = 1,19 (torsk) till över 100 (kolja och gråsej) ([Ribeiro et al. 2019](https://arxiv.org/abs/1810.00613)).

Avsnitt 86:s egen kalibreringsregel ("sat/p90, sedan en tredjedel") ger med dagens tal `w` ≈ 2,4–3,2 (modellprob). Handling-tiderna höjdes 3,5–8,7 gånger i avsnitt 147–148, så dagens `w` = 1 är 2–6 gånger svagare än avsnitt 86 avsåg.

Sammantaget pekar tillväxtdata på `w` ≈ 1–2, Ecosim på 2–7 och modellens egen designlogik på 2,4–3,2. Överlappet ligger kring **2**, och det är rekommendationen om `w` ska ändras alls.

## w kan inte bromsa gadoiderna

**Modellprob.** Gadoiderna startade på 2500 t med frysta fix55-policyer. Nettot är bruttoproduktion minus förluster per år.

| w | Biomassa efter år 1 | Medel s | R | Brutto/år | Svält/år | Netto år 1 | Sill uppäten år 1 | Netto år 2 |
|---|---|---|---|---|---|---|---|---|
| 0 | 3771 (1,51×) | 0,700 | 1,000 | 0,887 | 0 | +0,41 | 6222 t | +0,42 |
| **1** | **3574 (1,43×)** | 0,688 | 0,941 | 0,834 | 0 | **+0,36** | 5972 t | +0,38 |
| 2 | 3425 (1,37×) | 0,678 | 0,898 | 0,792 | 0 | +0,31 | 5798 t | +0,35 |
| 3 | 3299 (1,32×) | 0,670 | 0,865 | 0,757 | 0,002 | +0,28 | 5659 t | +0,33 |
| 5 | 3067 (1,23×) | 0,657 | 0,816 | 0,704 | 0,021 | +0,20 | 5439 t | +0,30 |
| 10 | 2607 (1,04×) | 0,633 | 0,743 | 0,620 | 0,097 | +0,04 | 5021 t | +0,27 |
| 20 | 1912 (0,76×) | 0,601 | 0,680 | 0,531 | 0,330 | −0,27 | 4259 t | +0,27 |

- **Nettot når noll först vid `w` ≈ 11,4** (interpolerat), och då med svält på ~0,12/år.
- **Att få bruttoproduktionen ned till förlusterna kräver `w` ≈ 27–30** (extrapolerat).
- **Andra året är nettot tillbaka på +0,26–0,27/år** för alla `w` mellan 10 och 20, eftersom sillen hinner byggas upp igen.

Ett högt `w` skjuter alltså bara upp överskjutningen ungefär ett år. Mekanismen förklarar varför: interferensen verkar per cell och på intaget. Ett utspritt bestånd kommer delvis undan den, och den rabatterar sillparet minst (typ III, hög mättnad), så ett stort `w` förskjuter dessutom dieten *mot* sill. Arditi et al. visar att ett mycket stort `w` driver rovdjurets jämvikt mot noll och släpper bytet fritt. I modellenheter ligger den gränsen kring `w` 30–90.

Det som reglerar gadoiderna i modellen i dag är att sillen tar slut. Vid `w` = 1 är nettot noll först kring 7500–10000 t startbiomassa, med sillen nere på 1400–3100 t.

Avsnitt 86.4 visar att tränade policyer omoptimerar mot `w` genom att sprida sig. Allt i tabellen är därför ett statiskt svar i första ordningen. Slutsatsen om storleksordningen påverkas inte av det.

## Det som gick förlorat med kannibalismen är en täthetsproportionell förlust

Kannibalismparet bockades av eftersom en belöning för gruppens biomassa aldrig lär sig äta sin egen grupp. Med det försvann också gadoidernas enda täthetsberoende förlust. Hur stor den är i verkligheten har räknats fram ur ICES Nordsjö-SMS (key run 2023, 1974–2022, 570 000 km²; egen beräkning i `regulation_and_models.md`):

- **Torskfiskar som äter varandra** (torsk, vitling, kolja, gråsej) tar i medel **0,233/år** av den sammanlagda biomassan, med intervallet 0,10–0,45 och 0,229 år 1991. Det stämmer med bibliotekets 0,2168.
- **Ren torsk-på-torsk-kannibalism** är bara **0,073/år** och faller mest på ålder 0–2.
- **Förlusten är nästan proportionell mot tätheten:** rate = **0,016 + 0,0875·B** (t/km²), r = 0,86, med exponenten 0,98 i log-log. För torsk på torsk är sambandet 0,008 + 0,169·B. Formen är delvis inbyggd i SMS och är därför inte helt oberoende belägg.
- **Andra regioner bekräftar åldersbilden.** I Östersjön tar vuxna torskar 25–38 % av 0-gruppen och 11–17 % av 1-gruppen per år. I Skagerraks kusttorsk syns täthetsberoende juvenil dödlighet i 8 av 11 populationer ([Fromentin et al. 2001](https://doi.org/10.1890/0012-9658(2001)082[0567:EODDAS]2.0.CO;2)).

Hur ekosystemmodeller gör: Atlantis har en generisk kvadratisk dödlighet (mQ·B), men handboken avråder från den utom för toppredatorer ([Atlantis User Guide](https://research.csiro.au/atlantis/wp-content/uploads/sites/52/2025/05/AtlantisUserGuide_PartI.pdf)). Ecosim låter Beverton–Holt-rekryteringen och "other mortality" kopplad till födosökstid sköta regleringen. Ingen modell hittades som använder M = M0 + k·B för vuxna torskfiskar. **Inferens:** i en biomassapool utan åldrar har de juvenila processerna ingen egen plats. En täthetsberoende M1 är det närmaste en sådan pool kan komma både kannibalism och juvenil täthetsdödlighet.

Förslaget blir `M1(c) = M0 + k·B_gadoid(c)` per cell, med **k ≈ 0,09–0,12/år per t/km²**, alltså ~6,0·10⁻⁵ till 8,2·10⁻⁵ per tick. Per cell ger termen policyn en lokal signal som går att lära sig. Om den räknas på hela gruppens biomassa ger den ingen sådan signal.

## Den nya förlusten ska inte vägas in fullt i budgeten

Noten `regulation_and_models.md` föreslår att `growth_budget.py` ska räkna termens värde vid referenstätheten in i g, så som kannibalismparet räknades in före avsnitt 160. **Det avråder jag från (inferens).** r_max är tillväxten vid låg täthet, och det är den budgeten ska betala. Räknas k·B_ref in i g blir termen budgetneutral vid referensen och bromsar bara över den. Rätt är att räkna in interceptet, ~0,016/år. Lutningen k·B ska vara obetald, eftersom det är den som gör nettotillväxten logistisk.

**Räkneexempel (inferens).** Med den lösningen ger termen vid dagens lokala täthet (~1,4 t/km²) ~0,12/år i extra förlust, vilket inte räcker mot det uppmätta nettot på +0,36. Interceptet är betalt av budgeten, så jämvikten utan andra effekter hamnar där k·B ≈ 0,36, alltså B ≈ 3–4 t/km². Det motsvarar 3–5 gånger dagens biomassa. Sillens utarmning och interferensen sänker den nivån i praktiken. Den ligger dock inom SMS:s observerade spann för sammanlagda torskfiskar i Nordsjön, **1,45–5,4 t/km²**. Det spannet gäller ett fiskat läge.

Här ligger den egentliga frågan, och den är ingen litteraturfråga. Modellens referens på 2500 t är 0,69 t/km² i medel, alltså under Nordsjöns fiskade tätheter. Modellen saknar fiske: M1 0,41 är ICES-konventionen 0,20 plus predation från grupper som inte modelleras. Reala torskfiskbestånd hålls av fisket långt under ofiskad nivå. Då är det biologiskt rimligt att en ofiskad modell drar uppåt. Två vägar är försvarbara:

- **Om 2500 t ska vara ett jämviktsläge** behöver modellen en icke-FG-förlust motsvarande fiske, en egen post i M1 som inte räknas in i budgeten.
- **Om modellen ska vara ofiskad** är 2500 t inte jämviktsläget. Då är frågan i stället om sillen tål den gadoidnivå som k·B och bytesbristen till slut sätter.

## Så testas det

Inget av följande är gjort. Inga observationskanaler ändras, så fix55-checkpointen laddar.

1. **Lägg till `M1(c) = M0 + k·B(c)` för gadoider bakom en biblioteksnyckel**, på CPU och GPU, med k = 0,0875/år per t/km² (6,0·10⁻⁵ per tick). Räkna bara in interceptet i `growth_budget.py`. Kör pytest.
2. **Kör proben i `model_context.md` med samma frysta fix55-policyer:** `python3 <scratchpad>/gadoid_interference_probe.py <ckpt> --years 2 --w 1` (skriptet ligger i scratchpad, inte i repot). Mät var nettot når noll och vad sillen landar på, både med och utan `w` = 2.
3. **Om gadoiderna planar ut på en nivå sillen tål, träna om (fix56) och jämför mot fix55.** Interferensen bör ändras i ett separat steg, så att effekterna går att skilja åt. En lägre interferens kan få tillbaka randartefakten från avsnitt 86. En högre behöver en A/B-körning och inget litteraturutslag.
4. **Avgör frågan om fiske innan k kalibreras mot 2500 t.**

## Rättelser till vad jag sa i chatten

- Casini et al. 2016 stöder *inte* täthetsberoende kondition hos Östersjötorsk. Torsktätheten var icke-signifikant och syrebristen dominerade.
- Svedäng & Hornborg 2014 är publicerad i *Nature Communications* 5:4152.
- DeLong & Vasseur 2011 är publicerad i *BMC Ecology* 11:1.
- Jag föreslog en högre interferens som den naturliga ersättningen för kannibalismen. Litteraturen och proben talar emot det. Interferensen beskriver täthetsberoende i intag, som redan är rätt dimensionerat. Kannibalismen var en förlust.

## Källäget

Wiley, Springer, CJFAS/CSP och Taylor & Francis returnerade 403, så flera källor bygger bara på abstract eller sökutdrag. Varje anteckning markerar fynd för fynd hur källan lästes.

| Källa | Läst | Används för |
|---|---|---|
| [DeLong & Vasseur 2011](https://doi.org/10.1186/1472-6785-11-1) | fulltext | fördelningen av m |
| [Skalski & Gilliam 2001](https://doi.org/10.1890/0012-9658(2001)082[3083:FRWPIV]2.0.CO;2) | abstract | 18/19 rovdjursberoende |
| [Stouffer & Novak 2021](https://doi.org/10.1111/ele.13670) | preprint i fulltext | att fisk saknas i underlaget |
| [Arditi et al. 2004](https://doi.org/10.1016/j.crvi.2004.06.007) | fulltext | stabilitet, gränser för `w` |
| [Fall et al. 2021](https://doi.org/10.1111/ecog.05473) | abstract | nolltest på cellskala |
| [Essington & Hansson 2004](https://doi.org/10.1139/f04-146) | abstract | rovdjursberoende hos Östersjötorsk (inga värden) |
| [Lorenzen & Enberg 2002](https://doi.org/10.1098/rspb.2001.1853) | fulltext | g = 3,3/B, 9/16 |
| [Zimmermann et al. 2018](https://doi.org/10.1111/1365-2656.12800), [Rindorf et al. 2022](https://doi.org/10.1111/faf.12650) | fulltext | förekomst och effektstorlek |
| [Denechaud et al. 2020](https://doi.org/10.1111/gcb.15298) | fulltext | NEA-torsk −21,78 % |
| [Casini et al. 2016](https://doi.org/10.1098/rsos.160416) | fulltext inkl. tabeller | rättelse: tätheten icke-signifikant |
| [Svedäng & Hornborg 2014](https://doi.org/10.1038/ncomms5152) | fulltext | sammanblandat, används inte som mål |
| [Jonsdottir et al. 2024](https://doi.org/10.1093/icesjms/fsad108) | delvis | koljans konditionsintervall |
| [Mackinson & Daskalov 2007](https://www.cefas.co.uk/publications/techrep/tech142.pdf) | fulltext (tabell 4.5) | K-värden i Nordsjön |
| [Ribeiro et al. 2019](https://arxiv.org/abs/1810.00613) | fulltext | Ecosims ekvation, isländska K-värden |
| [SMS key run 2023](https://github.com/ices-eg/wg_WGSAM/tree/master/NorthSeaKeyRun_2023) | rådata, egen beräkning | 0,233/år, 0,016 + 0,0875·B |
| [Fromentin et al. 2001](https://doi.org/10.1890/0012-9658(2001)082[0567:EODDAS]2.0.CO;2) | abstract | juvenil täthetsdödlighet i Skagerrak |
| [Atlantis User Guide](https://research.csiro.au/atlantis/wp-content/uploads/sites/52/2025/05/AtlantisUserGuide_PartI.pdf) | fulltext | mQ-praxis |

Inte verifierat: Nordsjökoljans årsklass 1999, Barentshavskoljans tillväxtfall 2004–06 (bara utdrag ur ICES-råd), något kvantitativt samband för vitlingens vikt per ålder mot täthet, och Marshall, Jørgensen och Ottersen om NEA-torsk.

## Slutsats

Litteraturargumentet för interferens räcker till att försvara dagens `w` = 1 och en måttlig höjning till ~2. Det räcker inte till att göra interferensen till populationsbroms, och proben visar att den inte heller fungerar som broms: värden som stoppar tillväxten svälter gadoiderna och håller bara ett år. Det som kannibalismen gav var en förlust som växer med tätheten. Den bör återinföras som en täthetsberoende M1 per cell, med lutningen från SMS och utan att lutningen betalas av budgeten. Även en sådan term landar på en högre gadoidnivå än 2500 t. Innan k kalibreras måste du bestämma om modellen ska föreställa ett fiskat eller ett ofiskat hav.
