"""Redistribute the benthivores' no-pool diet and recompute the pair M2 (resume 223.9 (2b, 2c)).

Usage: python3 -I diet_m2.py <qb_species.csv> <diet_species.csv> <out_dir>
  (the two inputs are research_notes/"Fiskgrupper i rutan"/benthivores/inputs/*.csv)

1. Reproduces the consumption-weighted group diet of section 209 (group_diet.py, central
   variant: cod Skagerrak with unidentified gadoids allocated): B 72.0, Z 2.5, P 8.4, W 1.9,
   Bf 4.3, Oth 10.9 % (sum of the last three = the 17.7 % with no model pair).
2. Splits Oth per species into prey categories (OTH_SPLIT, sources/judgements inline) and
   maps every no-pair category to the checked pools under three scenarios (MAPS).
3. M2 on each prey = Q/B x B_ben x diet share / B_prey (sections 209, 221), Q/B 4.2 (3.4-5.0)
   by mass, B_ben 1.53 t/km2, prey means: infauna 93.6 and epifauna 7.75 t/km2 (the
   engine's equilibrium start totals 337 000 / 27 900 t over 3600 km2; section 221 used 7.88
   from the two-row table), planktivores 3.33, predators 1.20 (section 209).
4. Energy check: the 4.2 comes from an energy budget (~11 000 MJ/t/yr assimilated) divided
   by the assimilated quality of the REAL mixed diet (~2620 MJ/t, no-pool prey valued as fish,
   4750 x 0.8). The redistributed diet has another real quality; the energy-equivalent mass
   Q/B is printed next to it.
"""
import csv
import os
import sys

qb_path, diet_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)

qb = {r["species"]: r for r in csv.DictReader(open(qb_path))}
D = {}
for r in csv.DictReader(open(diet_path)):
    D.setdefault(r["species"], {})[r["variant"]] = r
CHOICE = {"cod": "skag_gu"}          # the central variant of section 209
POOLS = ["B", "Z", "P", "W", "Bf", "Oth"]

# Oth split per species (fractions of that species' Oth). Tags: D derived from the measured
# rows in evidence_benthivores_C (ices_benth_diet_pools.csv), J judgement.
OTH_SPLIT = {
    # 2025 Q1 haddock (485 fish): unid. fish 7 (3 of it moved to P in the 'used' row), cephalopods
    # 9.6 (3 items), other invertebrates 0.3 -> 4 : 9.6 : 0.3 (D)
    "haddock": {"unid_fish": 4.0 / 13.9, "cephalopod": 9.6 / 13.9, "other_inv": 0.3 / 13.9},
    # Skagerrak cod box-weighted: other demersal fish (dragonets, sculpins) 5.6, unid. fish 5.7 (D)
    "cod": {"small_demersal_fish": 5.6 / 11.3, "unid_fish": 5.7 / 11.3},
    # 0-group fish and gobies on the bed (J; M&D 1991 'small fish')
    "long_rough_dab": {"small_demersal_fish": 1.0},
    # 'small gobies/sandeels' (J)
    "dab": {"small_demersal_fish": 0.5, "sandeel": 0.5},
    # ICES-DB Skagerrak 2024: sandeel 0.4 % (D)
    "plaice": {"sandeel": 1.0},
    "witch": {"small_demersal_fish": 1.0},                       # J
    # adult rays, fish > 50 %, sandeels 'only infrequently' (Skjaeraasen & Bergstad 2000) (J)
    "starry_ray": {"unid_fish": 1.0},
    # Malangen Ecopath 80 % detritus/carrion (J)
    "hagfish": {"carrion": 1.0},
    # sandeels/gobies/dragonets/unidentified (de Gee & Kikkert 1993 via FishBase) (J, equal)
    "grey_gurnard": {"sandeel": 1 / 3, "small_demersal_fish": 1 / 3, "unid_fish": 1 / 3},
    "lemon_sole": {},
    "greater_weever": {"unid_fish": 1.0},                        # J
    "others": {"unid_fish": 1.0},                                # J
}

# Unidentified fish follow the identified non-within-group fish prey of Skagerrak cod (the
# predator holding most of the fish diet): Norway pout etc. (P) 43.7, whiting (W) 3.7, other
# no-pool demersal fish 11.8 (-> where small_demersal_fish goes) (D, cod_fish_prey_pools).
_U = {"P": 43.7, "W": 3.7, "SD": 11.8}
_UT = sum(_U.values())
BENTHOS = {"I": 0.71, "E": 0.29}      # infauna : epifauna of the benthos diet (section 221)


def unid(sd_target):
    m = {"P": _U["P"] / _UT, "W": _U["W"] / _UT}
    for k, v in sd_target.items():
        m[k] = m.get(k, 0.0) + _U["SD"] / _UT * v
    return m


MAPS = {
    # Central (J): by foraging mode and habitat of the missing prey.
    "central": {
        "Z": {"E": 1.0},                     # krill/hyperbenthos -> mysids, small shrimps
        "Bf": None,                          # dropped: renormalised over the external diet
        "small_demersal_fish": {"E": 1.0},   # gobies, dragonets, sculpins: taken on the bed
        "sandeel": {"P": 1.0},               # small schooling zooplanktivore
        "unid_fish": unid({"E": 1.0}),
        "cephalopod": {"E": 1.0},            # sepiolids/octopods on the bed
        "carrion": dict(BENTHOS),            # hagfish: live prey polychaetes, shrimps
        "other_inv": {"I": 1.0},
    },
    # Benthos-heavy (J): every no-pair item replaced by benthos at the benthos split.
    "benthos_heavy": {k: (None if k == "Bf" else dict(BENTHOS)) for k in
                      ["Z", "Bf", "small_demersal_fish", "sandeel", "unid_fish",
                       "cephalopod", "carrion", "other_inv"]},
    # Fish-heavy (J): fish items -> the fish pools at the cod P:W ratio, within-group fish
    # -> P/W too (not dropped); cephalopods, carrion, krill as central.
    "fish_heavy": {
        "Z": {"E": 1.0},
        "Bf": {"P": 43.7 / 47.4, "W": 3.7 / 47.4},
        "small_demersal_fish": {"P": 43.7 / 47.4, "W": 3.7 / 47.4},
        "sandeel": {"P": 1.0},
        "unid_fish": {"P": 43.7 / 47.4, "W": 3.7 / 47.4},
        "cephalopod": {"E": 1.0},
        "carrion": dict(BENTHOS),
        "other_inv": {"I": 1.0},
    },
}

# 1. consumption-weighted diet and the no-pool categories
tot = {p: 0.0 for p in POOLS}
cat = {}
cons = 0.0
for sp, q in qb.items():
    d = D[sp][CHOICE.get(sp, "central")]
    s = float(q["share_pct"]) / 100.0
    Q = float(q["qb_central"])
    cons += s * Q
    for p in POOLS:
        tot[p] += s * Q * float(d[p]) / 100.0
    for c, f in OTH_SPLIT[sp].items():
        cat[c] = cat.get(c, 0.0) + s * Q * float(d["Oth"]) / 100.0 * f
diet = {p: tot[p] / cons for p in POOLS}
cat = {c: v / cons for c, v in cat.items()}
base = {"I": diet["B"] * BENTHOS["I"], "E": diet["B"] * BENTHOS["E"],
        "P": diet["P"], "W": diet["W"]}

lines = []


def say(s=""):
    print(s)
    lines.append(s)


say("Consumption-weighted diet (section 209 central): " +
    ", ".join(f"{p} {100 * diet[p]:.1f}" for p in POOLS))
say(f"No model pair: Z {100 * diet['Z']:.1f} + Bf {100 * diet['Bf']:.1f} + Oth "
    f"{100 * diet['Oth']:.1f} = {100 * (diet['Z'] + diet['Bf'] + diet['Oth']):.1f} %")
say("Oth by category (% of consumption): " +
    ", ".join(f"{c} {100 * v:.2f}" for c, v in sorted(cat.items(), key=lambda x: -x[1])))
say(f"Checked pools today: infauna {base['I']:.3f}, epifauna {base['E']:.3f}, "
    f"planktivores {base['P']:.3f}, predators {base['W']:.3f} (sum {sum(base.values()):.3f})")
say()


def redistribute(mapping):
    share = dict(base)
    items = dict(cat)
    items["Z"] = diet["Z"]
    items["Bf"] = diet["Bf"]
    dropped = 0.0
    for k, v in items.items():
        m = mapping[k]
        if m is None:
            dropped += v
            continue
        for pool, f in m.items():
            share[pool] += v * f
    total = sum(share.values())
    return {k: v / total for k, v in share.items()}, dropped


B_BEN = 1.53
PREY = {"I": ("benthic_infauna", 93.6), "E": ("mobile_epifauna", 7.75),
        "P": ("planktivorous_fish", 3.33), "W": ("predatory_fish", 1.20)}
# Real assimilated quality (MJ/t) as in the energy budget of "Bentivorernas energibudget":
# infauna 2600 x 0.7, epifauna 3750 x 0.8, fish/other 4750 x 0.8.
Q_REAL = {"I": 1820.0, "E": 3000.0, "P": 3800.0, "W": 3800.0}
# What the engine gives per tonne (static, assimilation x energy at the reference fill).
Q_MODEL = {"I": 0.7 * 2600, "E": 0.8 * 3750, "P": 0.8 * 7125, "W": 0.8 * 5252}
q_real_orig = (base["I"] * 1820 + base["E"] * 3000 + base["P"] * 3800 + base["W"] * 3800
               + (diet["Z"] + diet["Bf"] + diet["Oth"]) * 3800)
say(f"Real assimilated quality of the published diet: {q_real_orig:.0f} MJ/t "
    f"(energy budget 4.2 x {q_real_orig:.0f} = {4.2 * q_real_orig:.0f} MJ/t/yr)")
say()

rows = []
say("Reproduction (Q/B 3.48, today's shares): " + ", ".join(
    f"{PREY[k][0]} {3.48 * B_BEN * base[k] / PREY[k][1]:.4f}" for k in "IEPW"))
say("  library today: infauna 0.029, epifauna 0.141 (at 7.88), planktivores 0.14, predators 0.085")
say()
for name, mapping in MAPS.items():
    sh, dropped = redistribute(mapping)
    q_real = sum(sh[k] * Q_REAL[k] for k in sh)
    q_model = sum(sh[k] * Q_MODEL[k] for k in sh)
    qb_eq = 4.2 * q_real_orig / q_real
    say(f"=== {name}: shares " + ", ".join(f"{PREY[k][0]} {sh[k]:.3f}" for k in "IEPW")
        + f" (dropped within-group {dropped:.3f})")
    say(f"    real quality {q_real:.0f} MJ/t -> energy-equivalent Q/B {qb_eq:.2f} "
        f"(3.4 -> {3.4 * q_real_orig / q_real:.2f}, 5.0 -> {5.0 * q_real_orig / q_real:.2f}); "
        f"engine quality {q_model:.0f} MJ/t")
    for QB in (3.4, 4.2, 5.0, qb_eq):
        m2 = {k: QB * B_BEN * sh[k] / PREY[k][1] for k in "IEPW"}
        tag = "energy-eq" if QB == qb_eq else f"Q/B {QB}"
        say(f"    {tag:>10s}: " + ", ".join(f"{PREY[k][0]} {m2[k]:.4f}" for k in "IEPW")
            + f"   (consumption {QB * B_BEN:.2f} t/km2/yr)")
        rows.append([name, tag, round(QB, 3)] + [round(sh[k], 4) for k in "IEPW"]
                    + [round(m2[k], 5) for k in "IEPW"])
    say()

with open(os.path.join(out, "diet_m2.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["scenario", "qb_label", "qb", "share_infauna", "share_epifauna",
                "share_planktivores", "share_predators", "m2_infauna", "m2_epifauna",
                "m2_planktivores", "m2_predators"])
    w.writerows(rows)
with open(os.path.join(out, "diet_categories.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["category", "share_of_consumption", "central_map", "benthos_heavy_map",
                "fish_heavy_map"])
    allc = dict(cat, Z=diet["Z"], Bf=diet["Bf"])
    for c, v in sorted(allc.items(), key=lambda x: -x[1]):
        w.writerow([c, round(v, 5)] + [str(MAPS[s][c]) for s in MAPS])
