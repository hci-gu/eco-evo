# Stock-scale regulation in place of the regional K - synthesis (2026-10-30)

Sources: `stock_recruitment.md` (WP1), `migration.md` (WP2), `box_share.md` (WP3), scripts and
tables in this folder. Grades as there: D data, F source, A analogy, J judgement. No engine or library
change; nothing fitted to model output.

## 1. What the three packages say together

1. **The box is an open nursery and feeding area, not a self-reproducing population** [D, F; WP3].
   It holds 0.1-0.6 % of the herring, haddock, whiting, saithe, Norway pout and sprat stocks (Q1),
   ~2 % of North Sea cod (5-7 % of the Viking substock), witch 6-9 %, long rough dab ~1.7 %. Its
   share of recruits is 2-5x its share of spawners, and almost none of its recruits are spawned by
   its own adults. Weight of the box's adults in the recruitment that supplies it: w = 0.003
   (planktivores), 0.03 (benthivores; flatfish up to 0.2), 0.004 (water column), 0.02 (whiting).
2. **Stock-scale recruitment is flat in SSB over today's range** [D; WP1]. Elasticity of R/S at
   today's SSB (post-2000 regime): planktivores -0.74, benthivores -0.97, water column -0.65,
   whiting -0.98 (near-full compensation); the origin slope is not identified for any stock.
   Exceptions: WBSS herring (e 0, below Blim), saithe and mackerel (trend-confounded).
   Productivity shifted down after ~2000 (haddock R/S x0.24, cod x0.27-0.46, herring x0.63-0.66).
3. **Fish leave the box as they grow** [D, F; WP2]. Net emigration relative to the stock: juveniles
   (stage J) benthivores 1.1 /yr (haddock-dominated), planktivores 0.1 (0.5-0.8 from age 1), whiting
   0.2; adults planktivores 0.6, benthivores 0.17, whiting 0.3, water column 0.1 plus a seasonal
   presence (Feb / Aug 0.09 for the water column, 0.6 planktivores, 0.73 benthivores, ~1 whiting).
   Weak density dependence of the outflow at the leaving age (c ~ 0.3 per e-fold, 3-4 of ~10 tests
   significant); none in adults. Survey rates are relative to the stock: upper bounds.

**Consequence.** The compensation that K stands for happens OUTSIDE the box (in the stock's
spawning and nursery dynamics, flat over today's SSB). Seen from the box, recruitment is an
external supply that the box's adults barely affect, and the box loses fish by emigration. An open
population with a donor-controlled supply I and per-capita losses L is regulated without any
density dependence inside it: dB/dt = I + (P - L) B has the stable equilibrium B = I / (L - P)
whenever the per-capita losses exceed the per-capita production P. K is then not needed where
L - P > 0 holds with the field rates.

## 2. The form this points to (for the engine; not built)

Per staged group (juvenile compartment J inside B, `stages.py`):

- **Recruitment = an imported flux into J**, I_J(t) in t/km2 per tick: (box share of the stock's
  recruits, WP3) x (stock recruitment in numbers, WP1, flat -> its post-2000 mean) x (weight at
  arrival), spread over the arrival window already in the library (the rho month shape Jun-Sep).
  It replaces the rho(t) A allocation of the group's own growth (which creates no mass, only labels
  it). The box's own reproduction (weight w) can be dropped or kept as the small rho term.
- **Somatic growth stays energy-funded** (income / reserve growth as today; J tonnes at kappa).
  The derived g must then pay the SOMATIC production only, not r_max (r_max is the whole life
  cycle's growth at low density, recruitment included: paying it and importing recruits would count
  recruitment twice).
- **Emigration: per-capita losses e_J on J and e_A on B - J** (WP2 table), optionally
  e_J + c ln(J/J_ref) on the juveniles. A boundary loss like M1, not a behaviour.
- **Seasonal visitors** (water column, planktivores): an adult import I_A(t) and per-capita
  departure d(t) from the monthly presence tables (WP2), or leave them out at first.
- **No regional K** where the open form regulates (section 3: benthivores; planktivores and the
  water column through the exchange term). Whiting is regulated but its level is not closed: keep
  its K for now. Seals keep their K (no exchange data; WP2).
- **Scenarios:** a wind-farm closure does not change the stock's SSB, so I is unchanged; the box
  responds through F only (B = I / (L - P)), not through a K that treats the box as the whole stock
  (fixes the scale problem of resume 185.9).

## 3. Closure test with field rates only, all four groups [D, J]

`scripts/closure.py` -> `tables/closure.csv`, `tables/closure_stdout.txt`. Is today's B* an
equilibrium of dB/dt = I + (P - L) B? P = somatic production per tonne (SMS 2023 key run 2013-22,
per-tonne growth from weight at age, ages 0-1 / 2+, at the box J/B); L = library M1 + booked M2 +
F + WP2 net emigration (upper bounds); I = SMS stock biomass at the survey recruit stage x the
box's share of the stock's recruits (WP3, survey low-high). Members outside SMS take the SMS
members' rates and import nothing (flatfish, blue whiting, the minor predators).

| group | P | L (M1 + M2 + F + emig.) | L - P | imported recruits I/B | covers |
|---|---|---|---|---|---|
| planktivores | 0.53 | 0.26 + 0.76 + 0.10 + 0.41 = 1.53 | 1.00 | 0.08-0.24 | 8-24 % |
| benthivores | 0.74 | 0.29 + 0.22 + 0.17 + 0.31 = 0.99 | 0.25 | 0.10-0.14 | 38-57 % |
| whiting | 0.79 | 0.25 + 0.37 + 0.08 + 0.27 = 0.97 | 0.18 | 0.27-0.50 | 153-282 % |
| water column | 0.36 | 0.26 + 0.02 + 0.24 + 0.10 = 0.62 | 0.26 | 0 (no juveniles) | 0 % |

(Correction: the first, hand-made benthivore estimate gave I/B 0.11-0.22 and "closes within 5-15
%"; with SMS numbers and weights at the survey stage the import is 0.10-0.14 and covers about half.)

Reading per group:
- **Benthivores (nursery, resident):** the recruit import covers 38-57 %; the remaining 0.11-0.16
  /yr is within the uncertainty of the emigration (0.31, an upper bound) and the flatfish's own
  recruitment (0.30 of the group, w up to 0.2, not counted). The open form can plausibly hold
  today's level; sensitive to e.
- **Whiting (nursery, resident):** L > P, so the open form is regulated (WP2's "fails" used the
  closed r_max - F; with the stage-structured local P it passes), but the import is 1.5-2.8x the
  need: its equilibrium I/(L - P) is 2 400-4 450 t vs today's 1 606. Missing loss of ~0.09-0.32
  /yr, or the share is inflated by the separate 3a whiting component (WP3: 3a stock identity
  unresolved). Not closed.
- **Planktivores (window on the stock):** recruits cover only 8-24 %; ~0.8 /yr of B (~9 000-11 000
  t/yr) must arrive as older fish. That is what the herring survey and acoustics say: HERAS ~5.4
  t/km2 in summer vs ~1.3 in winter (185.18), physical exchange of the box in 4-18 days
  (`../Den pelagiska poolens reglering/migration_scale.md` 3.5). The booked M2 (0.76) is 2.5x SMS
  herring M2 (0.26-0.32); with SMS M2 the need is 0.54 and recruits cover 15-45 %, still not closed.
- **Water column (window on the stock):** no recruits; the need 0.26 /yr (~700 t/yr) is ~10 % of the
  seasonal visitor turnover (-ln 0.09 = 2.4 /yr, WP2): the box's biomass is the stock's presence.

**So the open form splits in two.** Benthivores and whiting are resident nursery groups:
imported recruits into J plus per-stage emigration (whiting's level not yet closed). Planktivores
and the water column are windows on mobile stocks: their box biomass relaxes to an external,
seasonal density on a scale of days to weeks, dB/dt = local + m (p(t) B_ext - B) with m ~ 1/(4-18
d) (Ecosim I - eB, Atlantis migration), and the local terms only shade it. That regulates them
without K, but it also means their group biomass is mostly set outside the box, so a policy's
effect on its own reward shrinks (a learning-signal question to test).

## 4. Decisions for the user

1. Recruitment as an imported flux into J (replacing the rho allocation) - the main change.
2. The growth budget on somatic production instead of r_max (re-derive g; validate against
   weight-at-age increments, independent field data).
3. Emigration as fixed per-capita losses per stage (boundary process, like M1) - consistent with
   "behaviour must be learned"? It is not a behaviour of the box's fish but the stock's movement
   beyond the grid; flagged for the user. Same question for the seasonal-visitor tables (a forcing
   like the recruitment table already accepted).
4. Seals keep their K. Whiting: the open form is regulated but its level is 1.5-2.8x too high with
   the 4+3a share; keep K until the 3a share is resolved.
5. Planktivores and water column: an exchange term toward an external seasonal density (window)
   instead of recruits + emigration - a larger change, with a learning-signal cost.

## 5. Open

- Whiting's missing loss (0.09-0.32 /yr): SMS 2023 whiting M2 (2013-22, biomass-weighted) is 0.82
  all ages, 0.48 age 1+, 0.34 age 2+, vs the library's booked 0.369 - at the box J/B 0.3 the field
  M2 is ~0.5-0.6, enough to close it [D]. Also check an inflated 4+3a share, and the closure's
  overlap (recruits entered at age 1 Q1 while P also counts their age-0 growth as J growth); the planktivore M2: `scripts/sms_m2_check.py`
  (`tables/sms_m2_planktivores_stdout.txt`), SMS 2023 partial M2 2013-22 at the box stage mix,
  scaled to the box's predator / prey biomass (SMS diet): benthivores 0.145 (cod 0.132; booked
  0.245, realised 0.0004), water column 0.257 (saithe 0.21, hake 0.04; booked 0.34, realised 0.25),
  whiting 0.203 (booked 0.16, realised 0.006), porpoises 0.004 (0.013), seals 0.005 (0.007, grey
  seal analogy); total 0.61 vs booked 0.76 vs realised 0.27 [D]. Unscaled North Sea total 0.44.
- Weight at arrival per member and stage, and the arrival month shape per member.
- Survey emigration is an upper bound (relative to the stock, trawl to 280 m in a box reaching
  533 m): the closure will be sensitive to it.
- Whiting stock identity in 3a (local share 0.2-0.5, J).
- WBSS herring is below Blim and on the ascending limb: its import would follow its SSB.
