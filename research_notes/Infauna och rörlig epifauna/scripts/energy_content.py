"""Energy content as eaten (MJ/t WW) of the two new bottom groups, weighted by what the fish eat.

Usage: python3 -I energy_content.py

kJ/g WW per taxon (low, central, high), from
  Ricciardi & Bourget 1998 (MEPS 163:245) Table 1 AFDW/WW medians [F]: polychaetes 15.6 %, bivalves 5.5 %
    (whole, shell on), prosobranchs 6.3 %, amphipods 16.5 %, isopods 14.2 %, cumaceans 7.5 %, decapods 16.5 %,
    mysids 15.5 %, ophiuroids 6.5 %, echinoids 2.7 %, asteroids 9.7 %;
  x energy per g AFDW, Steimle & Terranova 1985 (J Northw Atl Fish Sci 6:117) Table 1 [F]: polychaetes 21.85,
    bivalves 21.69, gastropods 21.17, benthic malacostraca 20.74, stelleroids 19.10, echinoids 25.82 kJ/g;
  checked against Steimle & Terranova's appendix wet-weight values [F]: Crangon septemspinosa 3.7, Dichelopandalus
    5.5, Cancer irroratus 3.7, Hyas araneus 2.8, Pelia 3.3, Libinia 5.4, Pagurus 6.0-6.9 (shell-free), Homarus 4.8,
    Neomysis 2.9 (dry 16.1 x 18 % DW), Ophiopholis 1.8, Ophioderma 2.1, urchins 0.6-2.0, polychaetes 2.4-7.6
    (Aphrodita 2.4), Nucula whole 2.2;
  Lawson et al. 1998 (MEPS 164:13) [F]: Pandalus borealis whole 4.72 kJ/g WW (females 4.57, males 5.19).
Diet weights: tables/benth_diet_split_stdout.txt (benthivores, consumption weighted) and
tables/pred_benthos_split_stdout.txt (predators), combined by the flows of split_budget.py.
"""
E = {  # kJ/g WW (low, central, high)
    "polychaete": (2.4, 3.4, 4.4),        # 0.156 x 21.85 = 3.41; Aphrodita 2.4; Steimle wet median ~4.2
    "ophiuroid": (1.0, 2.0, 3.0),         # Ricciardi 0.065 x 16.2 = 1.05 (Ophiura-type, whole); Steimle wet 1.8-2.1;
                                          # Amphiura filiformis AFDW 13.7-17.0 % WW (Rumohr et al. 1987, via
                                          # "Bentos och torskfiskar"/evidence_A) x 19.1 = 2.6-3.2; arms cropped
    "other_echinoderm": (0.6, 0.9, 2.0),  # echinoids 0.027 x 25.8 = 0.70; Steimle 0.6-2.0; asteroids 2.2-3.4
    "bivalve": (0.8, 1.2, 2.2),           # 0.055 x 21.69 = 1.19 whole with shell; Nucula whole 2.2
    "other_mollusc": (0.9, 1.3, 2.0),     # prosobranchs 0.063 x 21.17 = 1.33
    "amphipod": (2.5, 3.4, 3.8),          # 0.165 x 20.74 = 3.42
    "other_crust": (1.6, 2.9, 3.4),       # isopods 0.142 x 20.74 = 2.95; cumaceans 1.56; Calocaris ~decapod
    "other_inv": (1.0, 2.0, 3.0),         # J
    "crabs": (2.8, 3.6, 4.5),             # decapods 0.165 x 20.74 = 3.42; crabs 2.8-5.4, hermits shell-free 6-6.9
    "pandalus": (4.4, 4.7, 5.2),          # Lawson et al. 1998
    "nephrops": (3.4, 4.0, 4.8),          # J: decapod 3.4, Homarus 4.8
    "small_shrimp": (3.4, 3.7, 5.5),      # Crangon 3.7, Dichelopandalus 5.5
    "mysid": (2.9, 3.1, 3.4),             # 0.155 x 20.74 = 3.21; Neomysis 2.9
}
# infauna diet composition, benthivores (data species, consumption weighted; % of benthos)
INF = {"polychaete": 34.4, "ophiuroid": 21.4, "other_echinoderm": 3.1, "bivalve": 6.6, "other_mollusc": 2.0,
       "amphipod": 0.9, "other_crust": 2.1, "other_inv": 0.2}
# epifauna intake by taxon, t/km2/yr (split_budget.py central: benthivores + predators)
EPI = {"crabs": 0.969 + 0.029, "pandalus": 0.057 + 0.143, "nephrops": 0.003, "small_shrimp": 0.081 + 0.086,
       "mysid": 0.029}
# stock weights for the epifauna comparison (t/km2, central after the crab mass balance)
EPI_STOCK = {"crabs": 2.0, "pandalus": 0.56, "nephrops": 4.57, "small_shrimp": 0.3, "mysid": 0.2}


def mix(w):
    t = sum(w.values())
    return [sum(w[k] * E[k][i] for k in w) / t for i in range(3)]


for lab, w in (("infauna, as eaten by benthivores", INF), ("mobile epifauna, as eaten (benthivores + predators)", EPI),
               ("mobile epifauna, stock weighted (comparison)", EPI_STOCK)):
    lo, c, hi = mix(w)
    t = sum(w.values())
    print(f"{lab:<52} {1000 * c:6.0f} MJ/t (low {1000 * lo:.0f}, high {1000 * hi:.0f});  weights " +
          ", ".join(f"{k} {v / t:.3f}" for k, v in w.items()))
inf_c, epi_c = mix(INF)[1], mix(EPI)[1]
print(f"pooled check at the benthivore diet split 0.71 / 0.29: {1000 * (0.71 * inf_c + 0.29 * epi_c):.0f} MJ/t (library 3000)")
for pred, ceil in (("benthivorous_fish", 27.1), ("predatory_fish", 25.5)):
    print(f"h = E / ceiling, {pred} (ceiling {ceil} MJ/t/tick): infauna {1000 * inf_c / ceil:.1f}, epifauna {1000 * epi_c / ceil:.1f}")
