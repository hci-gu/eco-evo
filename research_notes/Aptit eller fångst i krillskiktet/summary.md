# Appetite or capture in the krill layer: why field whiting eat little krill (2026-11-20)

Research only (research agent with three literature subagents; saved by the main agent). Nothing in the library,
manifest, code, tests, AGENTS.md or the resume was changed; no commit. No model run, and no model output chose a
value: the yardstick (T 0.905, X 5.45 = krill 4.47 + other 0.98, need 0.388, maintenance 0.272) only sizes effects.
Files: `evidence.md` (main trail, spot checks), `lit_appetite.md`, `lit_inlayer.md`, `lit_seasonal.md` (subagent
trails with quotes), `tables/evidence.csv` (719 sourced numbers), `scripts/` -> `tables/*_stdout.txt`, `data/`
(untrusted, not to be committed). Tags: [F] full text, [A] abstract, [S] secondary, [D] derived, [J] judgement.
PROXY = not whiting.

## The question and the conflict

Whiting could eat ~0.75 of its ceiling while foraging; the field eats ~0.39, and krill carries 0.82 of the spare
capacity. "Krillparen mot vintermagar" (verdict C) read the Oslofjord winter fish as eating ~maintenance with krill
present (appetite); "Mättnad hos rovfisk" (Dempster 2011) found field gadoids acquisition-limited. The two imply
different model forms: an appetite term on realised intake, or an encounter/capture term on the krill pair.

First finding [D/J]: for whiting the "about maintenance" premise does not hold. On the model scale (krill 5 kJ/g):
Oslofjord whiting ate krill at 0.09-0.22 of the ceiling in November, 0.26-0.60 in March and 0.08-0.18 in June
(one site, homogeneous deep water); North Sea Q1 1981 whiting averaged 0.23-0.26 on krill alone, and the fish that
met krill 0.32-0.75 [D]. Verdict C was carried mainly by herring sources (PROXY), none of which involved krill
(Berg 2024 lab dry feed; Bernreuther 2007 and Lohmeyer & Hempel 1977 without krill), plus the Onsrud stomachs.

## Q1. Appetite or capture inside the krill layer?

| discriminator | appetite predicts | acquisition predicts | evidence | reads | tag |
|---|---|---|---|---|---|
| ration vs measured krill density in a layer | flat above need | rising | none for any gadoid in a layer. PROXY juvenile pollock: stomach krill rose with site krill abundance (Wilson 2009; 4th-root axes, krill only); icefish median content 0.27 -> 2.19 %BW with krill availability between years (Kock 1994) | acquisition at low / column-scale krill; in-layer open | F proxy |
| one site, three months, deep water homogeneous (equal deep T assumed) | winter low (seasonal), or June high (lipostat: fill minimum) | follows krill met | whiting Nov low, Mar high (x2.8), Jun as low as Nov (Onsrud 2004 Table 3) | neither winter-keyed nor lipostatic; krill density unreported | F, D, J |
| gorging in winter | rare | where krill is dense | 40 krill (8-16 %BW) in one 21-cm whiting; krill-dominated Q1 pools 0.32-0.75 of the ceiling; night food "eaten recently" | no winter shut-off; capture not near zero | F, D |
| spread among fish in the layer | even (each eats its demand) | overdispersed bouts | whiting variance/mean 3.5-22 among krill-bearing fish; PROXY pout: a nightly bout, empties 0 -> 72 % by 14 h (Albert 1994) | bouts; variable hunger fits too | F, D |
| addition vs substitution | equal totals | krill adds | Skagerrak Apr 1991 whiting: total x3.05, other food x0.48 (36 records); proxies mixed, cod reversed | weak support for acquisition | D |
| easy food outside summer | no rise | rise | PROXY saithe at farms: spring pellet meals 5.4 %BW, HSI x1.6; autumn weak; no Nov-Feb occasion (Uglem 2020). PROXY haddock x6 content on herring spawn, Mar-Apr (Høines 1992) | acquisition in spring; winter is a gap | F proxy |
| lab intake by season at constant T | falls in winter | no fall | none for whiting, pout or saithe. Whiting "fed well" in autumn-winter at 15 C (Mazlan 2001). PROXY cod: appetite falls 2-3 months before the fish's own spawning and moves with it (Skjæraasen 2004); spawning fast up to ~70 d; no depression in immature or vitellogenic cod Nov-Feb | the documented gadoid depression is spawning-bound | A, S proxy |
| condition vs ration in winter | liver kept while eating little | liver lost while eating little | PROXY saithe 1 Mar: stomachs at summer level, "clear signs of starvation", krill scarce (Nedreaas 1985). PROXY pout: more content in winter, yet liver loss (Albert 1994; spawning confound). Whiting stomach and liver track prey supply across years (Lauerburg 2016) | against a lipostatic set point; no case for appetite found | F, S |
| light | - | vision-limited: day > night | whiting night >= day in Nov and Mar; PROXY cod and haddock feed in the polar night | not vision-limited | F, S |
| capture, escape, vigilance | - | - | no gadoid datum; krill voids under fish schools by day (Onsrud 2004; Christiansen 2024) | open | F, S |

## Q2. Does the low krill intake hold outside winter?

| quarter | whiting ration / Cmax: SMS composite [S; T assumed, J] / Robb 1991 adults [D; upper-side, shape only] | krill on the model scale [D] | krill presence [S] |
|---|---|---|---|
| Q1 | 0.43 / 0.42 (about 50 % empty) | NE/NW North Sea 0.23-0.26; Skagerrak 0.05; Oslofjord Mar 0.26-0.60 | box season 1.0; large adults |
| Q2 | 0.42 / 0.83 | North Sea 0; Skagerrak Apr 0.30; Kattegat 0.04; Oslofjord Jun 0.08-0.18 | 0.6; mostly young |
| Q3 | 0.49 / 0.83 | North Sea 0.01 | 1.1; near the bed on 100-170 m beds |
| Q4 | 0.49 / 0.78 | North Sea 0.006-0.07; Oslofjord Nov 0.09-0.22 | 1.3 |

- **Year-round, not winter-only [J].** In the open North Sea whiting krill intake is highest in Q1 and near zero in
  Q2-Q4, when whiting eat sandeel, herring, juvenile gadoids and pout (Robb 1994 [F]). In the Oslofjord layer June is
  as low as November.
- **No appetite cycle independent of food [J].** Ration/Cmax at the season's temperature is flat in the SMS composite
  (0.42-0.49). In 1981 there were "no obvious between-quarter differences in the weights of food" (Hislop 1983 [F]).
- **One whiting-specific low, in one year.**
  - 1991 adults ate x0.52 of their Q2-Q4 level in Q1 (x0.60 with regurgitated stomachs counted as empty), mostly in the
    northern 200-350 mm classes (Robb 1994 [F/D]). Whiting spawn Feb-Jun, so a spawning-bound reading fits [J].
  - Against it, the Oslofjord March fish (27-28 Mar, inside the window; maturity unknown) were the high month [J].
  - The Q1 krill cells above are 1981 data; krill in Q1 1991 is not shown.
- **PROXIES:** saithe's Q1 low falls in its spawning quarter; Raitt's pout lows fall at spawning.

## Q3. Sizing on the model scale (`tables/model_scale_stdout.txt`, all [D] unless marked)

The yardstick potential T X/(1+X) is 0.765 (need 0.388; with krill removed 0.448). The field refill-season bound is
0.44-0.55 ("Mättnad hos rovfisk" [S]). f solved from realised field krill intake assumes that intake is
acquisition-limited; if appetite also acts, f is a lower bound.

| mechanism | model slot and form | learned / fixed | sized input [tag] | effect on the potential |
|---|---|---|---|---|
| capture p (escape) | factor f on the krill pair's a | fixed | p 0.4-0.75 vs 0.5 -> f 0.8-1.5 [S] | 0.74-0.80 |
| krill voids around fish | factor on a, by day | fixed | x0.32-0.88 by day [F/S] | 0.72-0.76 |
| patch occupancy | occupancy on a (patch tracking learnable) | fixed / learned | acoustic p 0.77-0.9; stomach p 0.1-0.3 [S] | 0.74; 0.53-0.63 |
| whiting's own in-area krill intake | empirical f on a (absorbs p, q, speed, occupancy) | fixed | Skagerrak Q1 1981 0.05; booked field flow 0.23 (M2 0.1873, from in-area diet share 0.50 x Q/B 4.2); Skagerrak Apr 1991 0.30; Oslofjord mean 0.14-0.33 -> f 0.026-0.26 [D] | 0.47 / 0.56 / 0.60 / 0.52-0.62 |
| North Sea quarterly mean (not in-area) | as above | fixed | 0.06-0.09 -> f 0.03-0.05 [D] | 0.48-0.49 |
| January local inversion | as above | fixed | f 0.004-0.013 [S] | 0.45-0.46, but krill intake only 0.01-0.03; does not transfer |
| krill-specific ceiling | h_krill / c, crowds out other prey under fixed attack | fixed | c ~1 (Andersen 1999 [A]); armament 0.67-1 [J] | >= 0.57; c 0.50 is needed for 0.45 |
| exclusive modes | time share tau in the layer | fixed here | no budget data | 0.45-0.74; double-counts the depth access [J] |
| winter appetite (lipostat or clock) | cap on realised intake | fixed | field winter 0.25-0.34 vs maintenance 0.272 [S/D] | potential unchanged; a cap at maintenance for 120 d leaves 1187 MJ/t of production unpaid |
| spawning cut (279) | reserve-room cut in a day-length window | fixed | share 0.344, fast_cut 0.5, 40 d | group intake x0.92 in the spawning quarter; mature adults x0.62-0.78 |

The brief's "0.45-0.47" is the off-season floor, not the field level [D/J]. Sized on whiting's own in-area krill
intake, any acquisition form gives 0.47-0.60, central 0.56 on the booked flow. That straddles the top of the
refill-season bound: krill acquisition alone does not bring whiting fully down, and the remainder sits with the other
prey (floor 0.448) [J].

## Verdict [J]

**For whiting's krill: acquisition, year-round.**
- The only appetite term with support is spawning-bound. It rests on PROXIES (cod, haddock, saithe) and on one whiting
  year (Robb 1991). The Oslofjord March sample, inside the spawning window, points the other way.
- The appetite is neither winter-keyed nor lipostatic. The strongest whiting data against a winter appetite are the
  high March ration and the gorged Q1 fish; June being as low as November argues against a lipostat. That lipostat
  argument needs the fill cycle, which is a Black Sea and haddock proxy.
- Within acquisition, capture or escape inside the layer cannot be separated from krill met in sub-layer bouts. The
  data size the product, not the form.

**Consequences for earlier notes and the model.**
- The whiting part of "Krillparen mot vintermagar" verdict C is revised. Its herring part (Berg 2024 photoperiod,
  Bernreuther, Lohmeyer & Hempel) is untouched.
- Pout is not re-judged. Its bouts and higher winter content point to acquisition, but spawning confounds them.
- **Supported model form:** a fixed acquisition term on the whiting -> krill pair (f ~0.03-0.26 in-area, central
  ~0.15), or patch structure with learned placement.
- **Not supported:** a winter or lipostatic appetite term, or a krill-specific ceiling.
- The spawning cut is already in section 279.
- An empirical field-level f fits field stomachs, not model output. It still stands in for the missing mechanistic
  derivation (no in-layer capture or encounter data exist), and nothing can be stacked on it.

## Reconciliation with Dempster 2011 [J]

- Dempster [F, read in "Mättnad"]: summer saithe eat x2.35 when food needs no search or capture.
- Uglem [F]: the same holds in spring (5.4 %BW pellet meals, HSI x1.6).
- Read for whiting alone, the krill data agree [D]:
  - fish that meet dense krill gorge in winter;
  - the March layer fish ate 0.26-0.60 of the ceiling;
  - the "maintenance" level was a population mean over fish with and without krill (krill occurrence 0.35-0.80),
    carried by herring.
- The one window that stays open is Nov-Feb: no easy-food datum exists for any gadoid there.

## Settling datum still missing

- **Field test.** Whiting trawled inside an acoustically identified krill layer (Sv converted to ind/m3 at trawl depth),
  at several layer densities, in Nov-Jan and May-Jul, with per-fish stomach content, digestion stage and HSI.
  - A ration that rises with density at low fill means capture or encounter.
  - A flat ration at high fill with dense krill means appetite.
- **Lab alternative.** Whiting at a constant 7-8 C offered live and thawed krill ad libitum, in winter and summer.
  - Thawed eaten far above live means capture or escape.
  - Both low in winter only means appetite.

## Load-bearing sources

1. Onsrud et al. 2004, ICES JMS 61:1278, doi:10.1016/j.icesjms.2004.09.005 [F]: Table 3, Nov / Mar / Jun.
2. Robb et al. 1994, ICES CM 1994/G:42 [F]; Hislop 1983, ICES CM 1983/G:59 [F]; ICES WGSAM 2024,
   doi:10.17895/ices.pub.25020968 [F].
3. Uglem et al. 2020, doi:10.1111/are.14519 [F]; Dempster et al. 2011, doi:10.1371/journal.pone.0015646 [S here].
4. Nedreaas 1985, ICES CM 1985/G:22 [F]; Albert 1994, doi:10.1006/jmsc.1994.1005 [F].
5. PROXIES: Wilson et al. 2009, doi:10.3354/meps08237 [F]; Kock et al. 1994, doi:10.3354/meps108043 [F].
6. The ICES stomach database as linked in "Krill över grunda bottnar" [D].
7. Skjæraasen et al. 2004, doi:10.1007/s10695-005-4319-6 [S]; Temming et al. 2004,
   doi:10.1111/j.0022-1112.2004.00400.x [A] (whiting aggregate on a sandeel patch in July).

## Gaps

- **Not found anywhere:**
  - a gadoid ration against in-layer krill density;
  - a capture, escape or vigilance datum for gadoids on euphausiids;
  - an easy-food datum for Nov-Feb;
  - lab seasonal intake for whiting, pout or saithe at constant temperature.
- **Not obtained:**
  - the Hislop et al. 1991 quarterly rations (download refused);
  - Greenstreet 1996 (SFRR 55);
  - Pedersen 2000 (abstract only);
  - the hake ration tables;
  - Dalpadado & Bogstad 2004.
- **Weak spots in the data:**
  - The Oslofjord series has no krill density, no fish lengths and no n per cell. Its stomach hauls are not tied to the
    layer, and fish were ~8x denser under the anchored vessel.
  - The 1981 counts do not separate Thysanoessa from Meganyctiphanes, which is a factor of ~3 in krill mass.
