# Evidence: water-column pairs by bed class (2026-11-10)

ASCII only. Folder names with Swedish letters are transliterated: "Vattenpelarens par i uppmatt ljus" (a-umlaut),
"Krill over grunda bottnar" (o-umlaut), "Sejens diet pa unga torskfiskar" (a-ring), "Attackrater ur motesteori"
(o-umlaut), "Sejens djuplage" (a-umlaut). Tags: [F] stated in the source, [D] derived here, [J] judgement.

## Model-side evidence (derivations, exec'd unchanged)

E1. Krill pair reproduction [D]. `scripts/wc_terms.py` exec's "Krill i Skagerrak"/wc_depth/scripts/krill_pairs_wc.py
    (--variant field) with the section 253 light and the section 262 layer and presence (copied from "Krill over grunda
    bottnar"/rederive/scripts/krill_geometry.py). Class a 0.06774 / 0.03144 / 0.005791 / 0.0001364 (library 0.068 /
    0.031 / 0.0058 / 0.00014); equal-cell 0.02596 (library 0.026). `tables/wc_terms_stdout.txt`.

E2. Krill pair terms [D/F/J], from krill_pairs_wc.py P table: L_krill 25 mm [J]; k_v 100 (planktivore note 2.2,
    Langbehn 2019) [F/D]; beam c 0.25 /m [J]; cap 2 L_pred (220; Lindholm 1999 >= 1.5 L) [F/J]; R_nv 0.1 m [J];
    Ke_gad 0.02 lx [J]; p_cap 0.5 [J, "tail-flip escape"]; scan 0.5 (Varpe & Fiksen 2010) [F/J]; u_krill 0.04 m/s
    (220); saithe 1.0 BL/s day, x0.6 at night [F/J]; saithe day position exponential from the bed, scale 15 m
    (HERAS in-box hauls, Neilson 2003) [F/D]; night 0.07 in 0-30 m [D/J]; spurdog and velvet belly non-visual (R_nv only).
    Resulting R_sat on krill 0.869 m (= the 2 L cap for saithe < 60 cm, 0.86 m).

E3. Fish pair terms [D/F/J], from "Attackrater ur motesteori"/scripts/pred_encounter_a.py (section 220) via
    "Rovfiskens uppdelning"/scripts/split_encounter_223.py: in-layer a3d (predator inside the prey layer, f = 1);
    R_fish_day_cruiser 0.5 m ("lake-trout RD(I) at 0.1-10 lx in the 30-70 m herring layer"); R_fish_dark 0.2 m
    ("non-visual floor 0.1 m (Langbehn), Chinook low-light asymptote 0.5 m, RD(0.01 lx) 0.03 m"), used day AND night on
    pout and blue whiting ("deep prey: dark at 150-300 m day and night"); R_bedfish_day 0.25 m ("bed at 44-529 m ... is
    below ~0.5 lx even at noon"); p_fish 0.2 ("Turesson & Bronmark 2004: single prey 0.20, pursuit 0.12, sit-and-wait
    0.36"); H_herring 70 m day / 40 m night, H_pout 15 m, H_bluewhiting 50 m, H_juv 2 m, H_whiting 10 m; D_DAY 0.5.
    Reproduction: planktivores 0.005948 (library 0.0060, 243.3 pool shares herring+sprat 0.575, pout 0.350, blue
    whiting 0.075); benthivores in-layer 0.06507 x member overlaps (243: saithe 0.05, spurdog 0.15, hake 0.25, velvet
    belly 0.10, anglerfish 1) = 0.00569 (library 0.0055); whiting 0.01301 (library 0.013, no overlap factor).

E4. Light at the krill layer [D on F] (measured box light, section 253): at each class's median bed (83 / 123 / 190 /
    325 m) the day layer lies 1 / 13 / 36 / 161 m above the bed; daytime median light at its centre 2.4 / 0.57 / 0.055 /
    0.043 lx; gadoid g = E/(E+0.02) 0.99 / 0.97 / 0.73 / 0.68. `tables/wc_terms_stdout.txt` block A.

E5. Factor decomposition, saithe < 60 cm [D] (`tables/gate_options_stdout.txt`): krill a / planktivore a per t/km2 =
    p x2.50 x O 0.746 / 0.325 / 0.093 / 0.005 x R^2 7.15 / 7.74 / 5.02 / 1.98 = x13.3 / 6.3 / 1.17 / 0.026 by class;
    member mix x0.82-0.88; group a ratio 11.3 / 5.2 / 0.97 / 0.023. Pool: O_eff 30.0e-3 /m, R_eff 0.266 m.

E6. Gate on the 60 x 60 map [D] (`tables/gate_options_stdout.txt`), V_lib krill share: equal cells 0.873, saithe
    spread 0.809, classes 0.915 / 0.922 / 0.728 / 0.070. budget_gate (16 x 16 probe grid, today): 0.870 / 0.811
    (`tables/budget_gate_stdout.txt` lines 86, 114). Krill encounter by class: equal cells 0.361 / 0.512 / 0.126 /
    0.002; saithe spread 0.114 / 0.369 / 0.517 / 0.001.

E7. Prey side, budget_gate today [D]: water column -> krill implied M2 0.287 vs booked 0.119 (whole krill pool 1.12 of
    booked); water column -> planktivores 0.037 vs 0.174 (field spread 0.055), -> benthivores 0.015 vs 0.083 (0.022),
    -> whiting 0.028 vs 0.157 (0.013), -> mackerel 0.002 vs 0.021. `tables/budget_gate_stdout.txt` lines 132-170.

E8. Size in the model [D] (`tables/member_size_stdout.txt`): V_lib krill share at 150-250 m saithe < 60 cm 0.779,
    >= 60 cm 0.788 (saithe spread 0.842 / 0.849). V_one: 0.745 / 0.573.

## Field evidence

E9. ICES CRR 164 (Daan ed. 1989; doi:10.17895/ices.pub.4533), Table 8-E saithe, northern North Sea 1980-82, % by
    weight, Euphausiacea by predator size class [F, read from the PDF text]:
    Q1+Q4 ("Quarter: 1+4" block): 30-40 cm 47.78; 40-50 76.43; 50-70 21.10; 70-100 9.35; >= 100 0.12
    (n stomachs 55 / 98 / 340 / 408 / 203).
    Q2+Q3: 30-40 cm 28.15; 40-50 37.13; 50-70 45.66; 70-100 33.56; >= 100 13.48 (n 97 / 208 / 403 / 354 / 21).
    About half the 1981 saithe records come from the NE North Sea ("Vattenpelarens krill mot fisk" saithe_1981_regions).

E10. ICES CRR 219 (ICES 1997; doi:10.17895/ices.pub.4626), Tables 8-E-1..4 saithe, "Total North Sea" but 3012 of
    3210 stomachs from roundfish area 1 (NW), 1991, Euphausiacea % weight by age (mean length, cm) [F]:
    Q1 (39-96 cm): 7.80, 5.87, 2.09, 2.06, 1.49, 0.69, -.
    Q2 (42-95 cm): 49.13, 42.66, 24.32, 25.94, 21.59, 12.83, 6.97.
    Q3 (42-93 cm): 8.85, 2.76, 2.54, 3.51, 2.63, 2.66, 2.87.
    Q4 (44-97 cm): 0.39, 0.53, 0.34, 0.38, 0.25, 0.09, 0.05.

E11. ICES stomach database, saithe by bed class ("Krill over grunda bottnar"/stomachs/summary.md table) [D there]:
    mass share N of 57 N, 1980-87, spread, all seasons: 0.22 / 0.10 / 0.19 (0.08-0.27) / 0.43 for < 100 / 100-150 /
    150-250 / > 250 m; NE North Sea Q3 spread: 0.31 / 0.58 / 0.55 (0.36-0.78) / 0.49; 2026 Q1 true depth (34-93 cm):
    0.020 / 0.052 / 0.0003 (5 stations) / -. Odds ratios against 100-150 m adjusted for length, quarter, area: 150-250 m
    0.98 (0.55-1.75), NE+Skagerrak 0.65 (0.22-1.92).

E12. Same database, saithe FO by length, NE North Sea + Skagerrak 1980-91 (`"Krill over grunda bottnar"/stomachs/
    tables/length_season_stdout.txt`) [D there]: 30-40 cm 0.90-1.00; 40-50 0.67-0.91; 50-60 0.61-0.71; 60-80
    0.54-0.72 across the four bed classes. Summary: "Bed class has no consistent effect within a length class."
    The NE database at the box's bed spread and presence gives saithe 0.452 annual (Q1 0.099, Q2 0.493, Q3 0.539,
    Q4 0.363) ("Krill over grunda bottnar"/rederive/tables/saithe_krill_share_stdout.txt).

E13. Fujii 2016 (Mar Ecol Prog Ser 542: 167-186; doi:10.3354/meps11534; CC-BY) [F]: abstract: "Saithe fed
    predominantly on euphausiids"; Table 5 (IBTS Q3 2012 saithe stomachs): "SIZE Body size (total length, cm) ... 0.155
    < 0.01", "DEPTH Water depth (m) ... 0.099 < 0.05" (Spearman rho with diet structure); text: euphausiid-associated
    saithe stations "generally constrained in the northern North Sea along the edge of the Norwegian Deep". Diet
    composition by size class is not tabulated.

E14. Bergstad 1991, Sarsia 75 (doi:10.1080/00364827.1991.10413456; abstract only) [A]: "the strongest interactions
    involving the gadoids are of predator-prey type and exist between the most abundant gadoids and their main prey:
    small saithe and blue whiting as predators and the euphausids as prey, and between saithe and Norway pout."
    "intraspecific resource overlap seemed rather strong among the life-stages found in this area."

E15. Saithe bed depth grows with length in the box ("Sejens djuplage" summary, section 263.7) [D there]: 25-35 cm
    mean bed 102 m, >= 55 cm about 210 m; adults >= 35 cm 0.91 (Q1) / 0.80 (Q3) on 150-250 m beds.

E16. Capture success [F review]: Juanes, Buckel & Scharf 2002 pp. 271-272, as quoted in "Vattenpelarens par i
    uppmatt ljus" A2: "about half of piscivore attacks on average are successful compared with 70-80% for
    planktivorous fishes", declining with the prey : predator size ratio. No gadoid-on-euphausiid capture or reactive
    distance has been found by any of the sibling notes ("Krill over grunda bottnar"/stomachs "Not obtained").

E17. Kaartvedt et al. 2005 (as read in "Krill over grunda bottnar"/stomachs) [F there]: four saithe of 60-110 cm
    below krill swarms at 200-300 m held fish and Gonatus, not krill.

E18. Placement and realised diet (sections 263.1-263.2) [D, model]: frozen fix94 water column 0.60 / 0.34 / 0.05 /
    0.01 over the four classes, krill 0.78 of the diet; re-placed at the GOV spread, realised krill 0.60, planktivores
    0.37 (budget_gate at that spread 0.81).
