# Epifauna in `benthic_community`: standing stock, r and f (summary)

Written by the main agent from the standing-stock subagent's report (the
harness does not let subagents write summary files). Evidence:
`evidence_A1_pandalus.md`, `evidence_A2_nephrops.md`,
`evidence_B_other_epifauna.md`; `scripts/combine.py` reproduces every
number below. Units t WW/km2 (= g WW/m2).

## Bottom line
The mobile epifauna the benthivores eat (Nephrops, northern shrimp,
crabs, hermit crabs, Munida, small shrimps, mysids) add about 6.4 t/km2
(3.3-13.8) to the grid-mean infauna stock of about 92 t/km2: +7 %
(+4 to +15 %). Nephrops is most of it. It grows slowly and sits in
burrows, so the community's biomass-weighted r and f barely move:
r 1.0 -> 0.971, f 0.20 -> 0.203.

## Stock by taxon, central (low-high)
| Taxon | 100-150 m | 250-300 m | Grid mean |
|---|---|---|---|
| Nephrops | 4.7 (2.6-8.7) | 1.3 (0.7-2.5) | 4.57 |
| Pandalus | 0.30 (0.22-0.71) | 1.43 (1.04-3.35) | 0.56 |
| Crabs, hermit crabs, Munida, small shrimps, mysids | 1.4 (0.5-4.7) | 1.0 (0.3-3.3) | 1.27 |
| Total | 6.4 (3.3-13.9) | 3.8 (2.1-9.3) | 6.40 (3.3-13.8) |

Nephrops density is flat to about 250 m and near zero beyond 300 m;
Pandalus peaks at 200-300 m. A least-squares line over the 3600 cells
(it reproduces the grid mean) gives the epifauna anchors for the depth
table: 8.0 at 100 m and 3.1 at 300 m (low 4.0 / 1.75, high 16.5 / 8.1).

## r and f by taxon
| Taxon | r (/yr) | f |
|---|---|---|
| Nephrops | 0.34 (0.25-0.5): lifespan > 10 yr, M 0.2-0.3, Ecopath P/B 0.35-0.4 | 0.10 (0.05-0.2): ~0.09 visible animals per burrow at peak emergence (Aguzzi et al. 2021) |
| Pandalus | 0.85 (0.5-1.3): surplus-production r 0.53 [F]; Hiddink longevity rule (4 yr) 1.33 | 0.7 (0.4-0.9): surface-living, males partly off the bottom at night |
| Other epifauna | 1.2: Hiddink longevity classes, Brey P/B | 0.6: surface-living |
| Epifauna, weighted | 0.55 | 0.25 |

## Arithmetic (central)
- Stock at the anchors: 110 + 8.0 = 118.0 at 100 m; 55 + 3.1 = 58.1 at 300 m.
- Biomass-weighted over the grid: r 0.971, f 0.203 (low 0.983 / 0.201, high 0.955 / 0.214).
- K = stock / equilibrium share 1 - (F + M2)/r (the main agent's final
  calibration, with the new trawl loss and M2, is in mareld_resume.txt
  section 218).

## Biggest uncertainties
1. Nephrops (largest term): grid share from an overlay of the ICES UWTV
   strata (70 % of the grid inside surveyed grounds, none deeper than
   300 m); range from the +73 % 2024 -> 2025 jump and 23-45 g per animal;
   fewer than one animal per burrow would lower it.
2. Pandalus catchability differs ~3x between models (SS3 q 0.51-0.55;
   old surplus-production q 0.16, prior-driven); effect on K < 2 %. ICES
   publishes only relative biomass since 2023.
3. No in-area biomass for crabs, hermit crabs, Munida or mysids: North
   Sea proxies, ~3x either way.
4. One pooled group cannot hold taxon selectivity: the fish diet is
   ~28 % crabs, shrimp and Nephrops against ~8 % of the reachable stock,
   and the trawl fishes Pandalus at F ~0.5 and Nephrops at ~0.06 while
   the group has one loss rate.

## Key sources
ICES advice pra.27.3a4a 2025 and nep.fu.3-4 2025; ICES WGNEPS 2025;
NAFO SCR 15/059 and 21/001; Aguzzi et al. 2021 (doi:10.1038/s41598-021-
85240-3); Mackinson & Daskalov 2007 (Cefas Tech Rep 142); Reiss et al.
2006; Callaway et al. 2002; Dorman et al. 1992; Hiddink et al. 2019.
