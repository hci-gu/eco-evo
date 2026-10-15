# Bottentrålningen tar ungefär en femtedel av bentos per år

Bottentrålningen tar nu bentos direkt i modellen. Förlusten per tick är **d · SAR / 1460**:

- **d** är andelen av bentosbiomassan som dör per passage. Värdet är **0,10** (intervall 0,06–0,115).
- **SAR** är hur många gånger per år en cell sveps av hela redskapets kontaktyta. Det fås ur ansträngningskartan med **k = 0,75** svepningar per år per MW-h/km² (intervall 0,6–1,1).

Lutningen blir 5,14·10⁻⁵ per tick och ansträngningsenhet. Vid rutnätets medelansträngning (3,0 MW-h/km²/år) betyder det SAR 2,3 och en förlust på 0,23 per år. Med frysta fix55-policyer blev den realiserade förlusten 0,19–0,20 per år, och bentos låg efter två år 20 % lägre än utan trålning (16 501 mot 20 659 t). Gadoiderna påverkades knappt.

Rapporten skiljer på tre slags uppgifter. **Litteratur** anges med källa. **Beräkning** avser egna räkningar. **Inferens** avser egna slutsatser och är uttryckligen markerade. Underlaget finns i `research_notes/Bottentrålning och bentos/`: `depletion_and_recovery.md` och `swept_area.md`.

## Dödlighet per passage: 0,10 för kräft- och räktrål på mjukbotten

**Litteratur.**
- **Hiddink m.fl. (2017)** fann i en global metaanalys att bottentrål tar **d = 0,06** per passage (5–95 %: 0,02–0,16). Bomtrål tar 0,14, skrapa 0,20 och hydraulisk skrapa 0,41. Dödligheten följer hur djupt redskapet tränger ned (r = 0,98) och skiljer sig inte mellan biomassa och antal ([Hiddink m.fl. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5547586/)).
- **Pitcher m.fl. (2022)** räknar fram 0,115 för bottentrål på dy och 0,047 på sand, utifrån inträngningsdjupet ([Pitcher m.fl. 2022](https://europepmc.org/article/PMC/PMC8764683)).
- **Rijnsdorp m.fl. (2020)** sätter **0,10** för kräft- och räktrål (OT_CRU), det vill säga 0,06 skalat med den andel av redskapets bredd som gräver ned sig ([Rijnsdorp m.fl. 2020](https://edepot.wur.nl/535091)).
- **Eigaard m.fl. (2016)** klassar markrepet på havskräft- och räktrål på mjukbotten som fullt nedgrävande (≥ 2 cm). Maskstorlek och sorteringsrist styr fångsten, inte bottenkontakten ([Eigaard m.fl. 2016](https://archimer.ifremer.fr/doc/00310/42137/41505.pdf)).

**Inferens.** Bottnen i rutnätet är mest dy och sand, och trålningen är mest räk- och havskräfttrål. Därför används 0,10. Någon direkt mätning för räktrål hittades inte.

## Svept yta: 0,75 svepningar per år per ansträngningsenhet

**Litteratur och beräkning.**
- EMODnet publicerar ICES svepta yta per c-ruta i lagren `fishingsurface` och `fishingsubsurface`. Yt-SAR avser hela redskapets kontaktyta, och "subsurface" avser den del som tränger ned minst 2 cm ([OSPAR QSR 2023](https://oap.ospar.org/media/filer_public/0d/50/0d50ce44-f6ad-4a19-ac2a-dde5e1dffc87/p00948_bh3_benthic_habitats_damage_fisheries_qsr2023.pdf)).
- I rutnätets c-rutor var yt-SAR 2021–2024 i medel **2,74 per år**, för alla bottenredskap sammantaget.
- Där bottentrål dominerar ger ICES-data 0,74 yt-SAR per MW-h/km² för 2019–2024. Det passar fiskekartans nivå, som är 2021–2024.
- Eigaards bredder för en räktrålare på 300–500 kW ger 0,87–1,14. Den svenska räkflottan har i medel 368 kW.

**Inferens.** Det rätta paret är d enligt Hiddink och Rijnsdorp tillsammans med yt-SAR (hela kontaktytan). Om d = 0,10 för OT_CRU används ska andelen som gräver ned sig inte räknas in en gång till. All SAR är en undre gräns, eftersom norska fartyg och fartyg under 12 m saknas i VMS-underlaget.

## Återhämtningen: modellens r är nu litteraturens 0,82 per år

**Litteratur.** Hiddink m.fl. anger en återhämtningstakt r = **0,82 per år** (0,42–1,53), och Pitcher m.fl. 0,68 per år på dy. Modellens bentos hade **r = 1,46 per år**, det handsatta värdet från avsnitt 136. Efter den här rapporten är det ändrat till **0,82 per år** (avsnitt 161.13). Det är fortfarande handsatt, utan härledning från predationsmatrisen, och K = 10 t/km² är oförändrat.

**Beräkning.** Med enbart trålning blir jämvikten B*/K = 1 − d·SAR/r:

| r (per år) | SAR 1 | SAR 2 | SAR 5 |
|---|---|---|---|
| 1,46 (tidigare modellvärde) | 0,93 | 0,86 | 0,66 |
| 0,82 (Hiddink, modellen nu) | 0,88 | 0,76 | 0,39 |

**Inferens.** Med det gamla r gav samma trålning ungefär hälften så stor utarmning som litteraturen. Litteraturens r är skattad på bottnar där rovdjuren finns med, medan modellen har gadoidernas predation separat, så 0,82 kan vara något lågt för modellen. Det gamla 1,46 saknade däremot stöd.

**Modellprob med r = 0,82 (frysta fix55-policyer, tre år).** Bentos går 16 000 → 13 527 → 10 470 → 7 126 t med trålning och 16 000 → 16 031 → 14 161 → 10 718 t utan. Trålförlusten är 0,17–0,20 per år. Nedgången utan trålning drivs av gadoiderna, som växer från 2500 till ~5100 t och äter mer bentos. Den täthetsberoende regleringen av gadoiderna saknas fortfarande. Pitcher m.fl. rankar Skagerrak–Kattegatt bland de tre sämsta av 24 regioner: SAR 3,33 per år och ett medelvärde för bentosstatus på 0,63.

Ett enda snabbt återhämtande bentosförråd kan inte representera de känsligaste djuren, som sjöpennor (*Funiculina*).

## Vad som är ändrat

- **Motorn** tillämpar impact-dödlighet även på grupper utan policy, på CPU och GPU. För sådana grupper verkar bara biomassafaktorn, eftersom de inte har några handlingar att lägga kostnad på.
- **Biblioteket** har fått posten `benthic_community_impacted_by_bottom_trawling` med en linjär tabell: 0 → 0 och 100 → 0,005137 per tick.
- **FG-editorn** visar matriserna Impact Affects och Impact Tables för alla grupper, och en not förklarar att energifaktorn inte verkar på grupper utan policy.

## Källäget

Hiddink 2017 och Pitcher 2022 (med SI) är lästa i fulltext. Konfidensintervallet för d enligt Hiddink är bara läst som det citeras hos Rijnsdorp 2020. ICES SAR per métier krävde inloggning, så k bygger på EMODnets summa över alla redskap där bottentrål dominerar. Sköld m.fl. 2018 fann att den totala biomassan på Kattegatts havskräftbottnar knappt svarade på SAR 0–15 per år ([Sköld m.fl. 2018](https://backend.orbit.dtu.dk/ws/portalfiles/portal/190977577/Microsoft_Word_Skold_et_al_Kattegatt_Revised_171124clean.docx.pdf)). Det talar för den lägre änden av d för total biomassa.
