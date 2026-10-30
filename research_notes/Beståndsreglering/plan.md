# Stock-scale regulation in place of the regional K - research plan (2026-10-29)

Question (user): replace `regional_carrying_capacity` K (resume 185, 209: a logistic loss
r_max B_region/K on all biomass, K = B*/(1 - F/r)) by processes that can be validated one by one:
(1) density-dependent recruitment at stock scale acting on the juvenile compartment's recruitment
weight rho(t) (stages.py), e.g. rho / (1 + A_bar/K_R) (Beverton-Holt in the adult biomass), and
(2) immigration / emigration across the box edge. Plus (3) the box's share of each stock, which sets
how much of the recruitment arriving in the box is the box's own.

Box: 60 x 60 km, 57.81-58.31 N, 10.07-11.15 E (northern Skagerrak offshore of Bohuslan, Lysekil-Smogen; corrected by WP3), Norwegian
Trench slope, 43-533 m, mean 168 m).

Work packages (one agent each; files in this folder, English, grades D computed from data, F stated
in a source, A analogy, J judgement):
- WP1 `stock_recruitment.md`: S-R per member stock from ICES SAG (SSB, R, lag), BH / Ricker /
  hockey-stick fits, compensation at today's SSB, regime shifts in productivity.
- WP2 `migration.md`: immigration / emigration of each group's members in and out of the Skagerrak
  box, seasonal and ontogenetic, density dependence of it.
- WP3 `box_share.md`: the box's share of each stock (biomass, SSB, recruitment), where the box's
  recruits come from (local spawning, larval drift, nursery emigration), the closure problem.
Synthesis: `summary.md` (mapping to the engine, no engine change yet).
