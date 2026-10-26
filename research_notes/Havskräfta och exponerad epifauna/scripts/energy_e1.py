"""Energy content as eaten (MJ/t WW) of the two E1 groups, and h = E / ceiling per pair.

Usage: python3 -I energy_e1.py <split_e1_central.json>

Taxon energies (kJ/g WW, low/central/high) are those of
research_notes/"Infauna och rorlig epifauna"/scripts/energy_content.py (221; Ricciardi & Bourget 1998
AFDW/WW x Steimle & Terranova 1985 kJ/g AFDW; Lawson et al. 1998 for northern shrimp), except:
  nephrops 3.6 (2.8-4.8) J, was 4.0 (3.4-4.8) J in 221. Basis: decapod AFDW/WW 16.5 % x 20.74 kJ/g
    AFDW = 3.42 (Ricciardi & Bourget; Steimle & Terranova); the burrowing thalassinid Calocaris
    macandreae, bombed whole (only bivalve shells and polychaete cases were removed) from Nephrops grounds in the Clyde, 11.22 kJ/g DW at 57.6 %
    organic (Parslow-Williams 1998 thesis, Table 2.3) = ~2.8 kJ/g WW at 25 % DW (J); Homarus 4.8
    (Steimle & Terranova appendix) the top. No whole-body Nephrops calorimetry was found.
Booked prey (223.10 redistribution, eaten as the exposed group in the engine; comparison only, J):
  cephalopods 3.9 (sepiolids/octopods; squid mantle ~4), small demersal fish 4.5, krill/hyperbenthos
  3.1 (mysid value), carrion/unidentified-fish slices 4.0.
Weights: the fish flows of split_e1.py (benthivores + predators; t/km2/yr). Small shrimps : mysids
split 0.85 : 0.15 within "small" (221 intake weights 0.120 : 0.021).
"""
import json
import sys

res = json.load(open(sys.argv[1]))
E = {"crabs": (2.8, 3.6, 4.5), "pandalus": (4.4, 4.7, 5.2), "small_shrimp": (3.4, 3.7, 5.5),
     "mysid": (2.9, 3.1, 3.4), "nephrops": (2.8, 3.6, 4.8),
     "cephalopod": (3.3, 3.9, 4.5), "small_demersal_fish": (4.0, 4.5, 5.0),
     "krill_hyperbenthos": (2.9, 3.1, 3.4), "unid_fish_E": (3.5, 4.0, 5.0), "carrion_E": (3.5, 4.0, 5.0)}
# taxon flows reconstructed from split_e1.py central (benthivores + predators)
QB_B, BB, NORM = 4.46, 1.53, 1.0 - 0.04325
C_B = QB_B * BB
measured = C_B * 0.720 * 0.290 / NORM
pred = 3.34 * 1.20 * 0.11 * 0.65
flows = {"crabs": measured * 0.872 + pred * 0.1, "pandalus": measured * 0.051 + pred * 0.5,
         "small": measured * 0.073 + pred * 0.4, "nephrops": measured * 0.003}
booked = res["checks"]["booked_flow"]


def mix(w):
    t = sum(w.values())
    return [sum(w[k] * E[k][i] for k in w) / t for i in range(3)]


members = {"crabs": flows["crabs"], "pandalus": flows["pandalus"],
           "small_shrimp": 0.85 * flows["small"], "mysid": 0.15 * flows["small"]}
stockw = {"crabs": 2.0, "pandalus": 0.56, "small_shrimp": 0.3, "mysid": 0.2}
withb = dict(members, **booked)
for lab, w in (("exposed group, members as eaten (central)", members),
               ("exposed group, stock weighted (comparison)", stockw),
               ("exposed group incl. the booked prey (comparison)", withb),
               ("norway_lobster", {"nephrops": 1.0})):
    lo, c, hi = mix(w)
    t = sum(w.values())
    print(f"{lab:<50} {1000 * c:6.0f} MJ/t (low {1000 * lo:.0f}, high {1000 * hi:.0f}); weights "
          + ", ".join(f"{k} {v / t:.3f}" for k, v in w.items()))
for pred_id, ceil in (("benthivorous_fish", 27.1), ("predatory_fish", 25.5)):
    print(f"h = E / ceiling, {pred_id} (ceiling {ceil} MJ/t/tick): exposed 3750 -> {3750 / ceil:.1f}; "
          f"Nephrops 3600 -> {3600 / ceil:.1f} (4000 -> {4000 / ceil:.1f})")
