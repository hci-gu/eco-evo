# Evidence D - the gear's direct KILL of bottom fauna it does not catch (d x SAR)

Scope: component (b). Does today's d = 0.10 per pass apply to the prey-sized infauna (polychaetes, small
bivalves, brittle stars)? Does it differ for epifauna? What is the kill rate per year at mean effort?
This note builds on `../Bottentrålning och bentos/depletion_and_recovery.md` (d) and `swept_area.md`
(k = SAR per MW-h/km2) and does not repeat them.
Tags: [F] full text, [A] abstract, [S] snippet, [D] my derivation.

## 1. What today's d measures

- **Hiddink et al. 2017 (PNAS 114:8301).** Otter trawls remove 0.06 per pass (5-95 % 0.02-0.16).
  "Depletion rates estimated from the experimental studies for biomass and numbers were not significantly
  different" [F, via sister note]. d is pooled over the whole sampled community. **There is no taxon or
  epi/infauna split in d.**
- **Pitcher et al. 2022 (PNAS 119:e2109449119).** The pooled data are "pooled relative biomass and numbers of
  epifauna and infauna" and are "representative of ... primarily biomass of epi-fauna, with some infauna"
  [F, via sister note]. d is therefore mainly an epifauna-biomass number.
- **Rijnsdorp et al. 2020.** OT_CRU d = 0.100 = 0.06 x 0.304/0.18, scaled by the subsurface share of the gear
  [F, via sister note]. This is a gear-geometry scaling, not a measured value for any taxon.

## 2. Taxon-level evidence for the prey-sized fraction

- **Kaiser et al. 2006, MEPS 311:1 (meta-analysis of 101 manipulations)** [F] (`data/lit/kaiser2006.pdf`):
  - "Of these 30 combinations, 19 indicated a significant initial response to fishing disturbance which
    ranged from -98% for Cnidaria and Porifera in response to ScD, to -24% for Annelida in response to OT in
    mud habitats".
  - "In mud communities, there were only data for OT and ID ... OT produced a significant, negative,
    short-term effect, but interestingly there was also a longer-term positive effect".
  - "OT had a uniformly less negative effect on both groups [deposit and suspension feeders] ... OT had the
    greatest impact on SFs in mud habitats".
  - Recovery: "biota with shorter life-spans such as polychaetes (<1 yr)".
  - The -24 % is per experiment, and experiments often apply several passes. It is not a per-pass d.
- **Sciberras et al. 2018, Fish Fish 19:698** [A, S]. All gears: "A gear pass reduced benthic invertebrate
  abundance by 26%". Per gear, mean initial community abundance response per pass: "Towed dredges, beam
  trawls and otter trawls ... -8%, -12%, -3% per gear pass" [S]. Recovery: sessile, long-lived biota
  (sponges, soft corals, bivalves) >3 yr; mobile, short-lived biota "such as polychaetes and malacostracans
  (<1 year)" [A].
- **Hinz, Prieto & Kaiser 2009, Ecol. Appl. 19:761.** Chronic gradient on a muddy Nephrops ground,
  NE Irish Sea [A, S]:
  - "Chronic otter trawling had a significant, negative effect on benthic infauna abundance, biomass, and
    species richness";
  - "Benthic epifauna abundance and species richness also showed a significant, negative response, while no
    such effect was evident for epibenthic biomass".
  - So on Nephrops mud, the long-run biomass loss is seen in the **infauna** rather than the epifauna.
- **Skagerrak/Kattegat:**
  - Sköld et al. 2018 (MEPS 586:41), Kattegat Nephrops mud, SAR 0-15 /yr: almost no detectable effect on
    total biomass (GAM R2 = 0.015), falling diversity, and burrowing brittle stars increasing at medium
    intensity (predator release) [F, via sister note].
  - Hansson et al. 2000 (MEPS 198:191), Gullmarsfjorden, experimental shrimp trawling: Amphiura brittle
    stars decreased significantly after 7-12 months, other taxa few significant changes [S, via sister note].
  - The Mareld EIA cites Sköld (2022) for a significant increase of Nephrops biomass inside trawl-free areas
    (`mkb.txt` 5115) [F, EIA; primary not read].
- **Epifauna.**
  - Large sessile epifauna (sea pens Funiculina/Kophobelemnon, Asteronyx on them) is the most sensitive
    part, and recovers slowly.
  - Mobile scavenging epifauna (Asterias, Pagurus, Liocarcinus, whelks) is attracted to trawl tracks and
    discards (Bergmann et al. 2002 MEPS 233:185 [F]: A. rubens, P. bernhardus, L. depurator and whelks were the most
    abundant scavengers at discard-baited traps).
  - The epifauna's net biomass response can therefore be small even when d per pass is high.

## 3. Conclusion on d for the model's prey fraction [D]

- Per-pass experimental evidence for small infauna on mud is weak and mixed:
  - Sciberras OT -3 % per pass in abundance;
  - Kaiser annelids -24 % per (multi-pass) experiment on mud;
  - Hiddink pooled 0.06, mostly epifauna;
  - Pitcher's mud prediction 0.115 comes from penetration depth, not from infauna data.
- The chronic evidence (Hinz 2009) shows that infauna biomass does fall under OT_CRU-type trawling on
  Nephrops mud. Sköld 2018 in the Kattegat found no net total-biomass signal.
- So d = 0.10 is not specifically confirmed for polychaetes, small bivalves and brittle stars. The evidence
  puts the per-pass loss of the prey-sized infauna **at or below** the pooled OT value:
  - infauna d: central 0.06, range 0.03-0.115;
  - epifauna d: 0.10 (0.06-0.16), with the fragile sessile part higher and the mobile scavengers lower.
- For a single pooled `benthic_community` that now also contains mobile epifauna (shrimps, crabs, Munida,
  Nephrops), keeping d = 0.10 is defensible as the whole-community value. The caught part of that epifauna
  is counted separately in (a), so with d = 0.10 there is partial double counting (see evidence C section 7).

## 4. Kill rate at today's mean effort and the per-unit-effort slope [D]

Kill rate = d x k x E. E = 3.014 MW-h/km2/yr (grid mean of `maps/fishing_bottom_otter_trawls.npz`), and
k = 0.75 SAR per MW-h/km2 (0.6-1.1, swept_area.md).

| Case | d | k | Rate at mean effort (/yr) | Slope, /yr per MW-h/km2/yr | Per tick per unit (biomass_factor at value 100) |
|---|---|---|---|---|---|
| Whole community, low | 0.06 | 0.6 | 0.109 | 0.036 | 2.47e-5 (0.00247) |
| Whole community, central (today's table) | 0.10 | 0.75 | **0.226** | 0.075 | 5.14e-5 (0.00514) |
| Whole community, high | 0.115 | 1.1 | 0.381 | 0.127 | 8.66e-5 (0.00866) |
| Prey-sized infauna, low | 0.03 | 0.6 | 0.054 | 0.018 | 1.23e-5 |
| Prey-sized infauna, central | 0.06 | 0.75 | **0.136** | 0.045 | 3.08e-5 |
| Prey-sized infauna, high | 0.115 | 1.1 | 0.381 | 0.127 | 8.66e-5 |

Arithmetic, central: 0.10 x 0.75 x 3.014 = 0.226 /yr; per tick 0.10 x 0.75 / 1460 = 5.137e-5 per unit of
effort. This matches the library's 0.005136986 at value 100. With the formal Hiddink 5-95 % range of d
(0.02-0.16), the rate spans 0.036-0.53 /yr.

**Fleet basis.** k and E both come from VMS data without Norway (and without vessels under 12 m), so this rate
covers the EU fleets only. In the grid, Norwegian bottom-trawl catch is small (evidence C section 2), so the
missing part is probably small too.

## Source status

| Source | Status |
|---|---|
| Kaiser et al. 2006 MEPS 311:1 (int-res.com, open access) | [F] |
| Sciberras et al. 2018 Fish Fish 19:698 (doi 10.1111/faf.12283) | [A]/[S]; full PDF not retrievable |
| Hinz, Prieto & Kaiser 2009 Ecol Appl 19:761 (doi 10.1890/08-0351.1) | [A]/[S] |
| Hiddink 2017, Pitcher 2022, Rijnsdorp 2020, Sköld 2018, Hansson 2000 | via `../Bottentrålning och bentos/depletion_and_recovery.md` |
| Bergmann et al. 2002 MEPS 233:185 (epic.awi.de/10388) | [F] |
