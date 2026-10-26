"""E1 split -> a library COPY and a manifest COPY (the live files are never touched).

Usage (from the repository root):
  python3 -I "research_notes/Havskräfta och exponerad epifauna/scripts/build_e1.py" \
      fgconfig/fg_library.yaml mareld2.yaml "<note>/data/split_e1_central.json" "<note>/data" [nephrops_f]

  nephrops_f: accessible_fraction of benthivorous_fish -> norway_lobster (default 0.02 = exposed
              share 0.07 x edible size window 0.29; 0.07 without the gape filter)

What it does
1. Reads the live library and manifest, and the numbers of split_e1.py (one variant).
2. In memory: mobile_epifauna becomes the exposed taxa (r_max, members, pairs' M2 and f, trawl slope),
   norway_lobster is added (species, every pair / observes / impact entry the old group had), the
   manifest copy gets norway_lobster in non_decision_makers.
3. Derives every active group's growth_rate with lib/world/growth_budget.py (as the loader does) and
   refits both groups' capacity_depth_table by non-negative least squares with the copy's own F(c), M2
   and r (lib/world/habitat.py), so the engine's start map B*(c) matches today's stock map of the members
   (split_e1.py target, *_today.npy) and its sum is today's total.
4. Writes text-edited copies (comments kept): <out>/fg_library_e1.yaml, <out>/mareld2_e1.yaml (the
   manifest copy; its relative map paths resolve from the repository root, not from <out>),
   <out>/mareld2_e1_snippet.yaml (non_decision_makers only), <out>/build_e1.json.
5. Verifies: both copies parse; stored growth_rate == derived for every active group; habitat.build
   equilibrium sums vs the manifest copy's totals; load_project_config builds every FG from the copies.
"""
import copy
import hashlib
import json
import os
import re
import sys

import numpy as np
from scipy.optimize import nnls

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
from lib.config.config_loader import load_config, load_project_config  # noqa: E402
from lib.world import growth_budget as gb                               # noqa: E402
from lib.world import habitat                                           # noqa: E402
from lib.world.impact_maps import project_baseline_maps                 # noqa: E402
from lib.world.tick_time import LIBRARY_TICK_HOURS, ticks_per_year      # noqa: E402

lib_path, project_path, split_json, out = sys.argv[1:5]
F_NEPH = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02
os.makedirs(out, exist_ok=True)
TPY = ticks_per_year(LIBRARY_TICK_HOURS)
GRID = (60, 60)
NOTE = 'research_notes/"Havskrafta och exponerad epifauna" (Swedish letters)'
sp = json.load(open(split_json))
variant = sp["variant"]
gE, gN = sp["groups"]["mobile_epifauna"], sp["groups"]["norway_lobster"]
today = {g: np.load(split_json.replace(".json", f"_{g}_today.npy")) for g in sp["groups"]}
lib_sha = hashlib.sha256(open(lib_path, "rb").read()).hexdigest()[:12]

# ---- proposal (rounded library values) ----
R_MAX = {"mobile_epifauna": round(gE["r_max"], 2), "norway_lobster": round(gN["r_max"], 2)}
M2 = {"benthivorous_fish_preys_on_mobile_epifauna": round(gE["M2"]["benthivorous_fish"], 3),
      "predatory_fish_preys_on_mobile_epifauna": round(gE["M2"]["predatory_fish"], 3),
      "benthivorous_fish_preys_on_norway_lobster": round(gN["M2"]["benthivorous_fish"], 4)}
F_PAIR = {"benthivorous_fish_preys_on_mobile_epifauna": round(sp["pairs"]["benthivorous_fish_preys_on_mobile_epifauna"]["f"], 2),
          "predatory_fish_preys_on_mobile_epifauna": round(sp["pairs"]["predatory_fish_preys_on_mobile_epifauna"]["f"], 2),
          "benthivorous_fish_preys_on_norway_lobster": F_NEPH}
# h rounded UP to one decimal (132.84 -> 132.9) so the pair's ceiling energy 0.8 E/h does not exceed the
# exposed pair's 0.8 x 3750/138.4 by a rounding digit (growth_budget would otherwise rank Nephrops "best"
# and move the benthivores' s* by 0.02 %)
E_NEPH = 3600.0
H_NEPH = float(np.ceil(E_NEPH / 27.1 * 10.0) / 10.0)
BF100 = {"mobile_epifauna": round(gE["bf100"], 6), "norway_lobster": round(gN["bf100"], 6)}
memb = gE["members"]
tot_e = sum(memb.values())
MEMBERS_E = [("crabs, hermit crabs and squat lobsters", memb["crabs"] / tot_e),
             ("northern shrimp", memb["pandalus"] / tot_e),
             ("small shrimps", 0.6 * memb["small"] / tot_e),     # 0.3 : 0.2 of "small" (221)
             ("mysids", 0.4 * memb["small"] / tot_e)]
if "booked" in memb:
    MEMBERS_E.append(("cephalopods and small demersal fish", memb["booked"] / tot_e))

old = load_config(lib_path)
project = load_config(project_path)
new = copy.deepcopy(old)
S, I = new["species_definitions"], new["interaction_definitions"]
S["mobile_epifauna"]["r_max"] = R_MAX["mobile_epifauna"]
S["mobile_epifauna"]["member_species"] = [{"name": n, "share": round(s, 2)} for n, s in MEMBERS_E]
neph = copy.deepcopy(S["mobile_epifauna"])
neph.update(display_name="Norway Lobster", member_species=[{"name": "Norway lobster", "share": 1.0}],
            r_max=R_MAX["norway_lobster"], energy_content=E_NEPH)
S["norway_lobster"] = neph
for k in ("benthivorous_fish_preys_on_mobile_epifauna", "predatory_fish_preys_on_mobile_epifauna"):
    I[k]["predation_mortality"] = M2[k]
    I[k]["accessible_fraction"] = F_PAIR[k]
I["benthivorous_fish_preys_on_norway_lobster"] = dict(
    preys_on=True, predation_mortality=M2["benthivorous_fish_preys_on_norway_lobster"],
    assimilation_factor=0.8, handling_time=H_NEPH, accessible_fraction=F_NEPH, max_intake_rate=0.44)
I["predatory_fish_preys_on_norway_lobster"] = dict(preys_on=False, assimilation_factor=1.0, handling_time=0.0)
I["mobile_epifauna_impacted_by_bottom_trawling"]["impact_table"][1]["biomass_factor"] = BF100["mobile_epifauna"]
for imp in ("bottom_trawling", "pelagic_trawling", "rotor", "windfarm_noise"):
    I[f"norway_lobster_impacted_by_{imp}"] = copy.deepcopy(I[f"mobile_epifauna_impacted_by_{imp}"])
I["norway_lobster_impacted_by_bottom_trawling"]["impact_table"][1]["biomass_factor"] = BF100["norway_lobster"]
I["benthivorous_fish_observes_norway_lobster"] = {"observes": True}
I["predatory_fish_observes_norway_lobster"] = {"observes": False}

# ---- manifest copy (dict) ----
proj_new = copy.deepcopy(project)
ndm = proj_new["non_decision_makers"]
idx = [g["group_id"] for g in ndm].index("mobile_epifauna")
ndm.insert(idx + 1, {"group_id": "norway_lobster"})


def active_of(p):
    return [g["group_id"] for key in ("decision_makers", "non_decision_makers")
            for g in (p.get(key) or []) if isinstance(g, dict) and not g.get("muted")]


act_old, act_new = active_of(project), active_of(proj_new)


def budgets(lib, active):
    out_ = {}
    for fid in active:
        b = gb.growth_budget(fid, lib["species_definitions"], lib["interaction_definitions"],
                             active_predators=active)
        if b is None:
            continue
        out_[fid] = dict(g=b.growth_rate, err=b.error, s_star=b.reference_fill, gross=b.gross_annual,
                         m2=dict(b.predation), best=b.best_prey, C=b.ideal_intake)
    return out_


b_old, b_new = budgets(old, act_old), budgets(new, act_new)
for fid, b in b_new.items():
    S[fid]["growth_rate"] = b["g"]

# ---- K tables: NNLS with the copy's own F(c), M2, r; total kept ----
depth = habitat.load_depth_map(proj_new, project_path, GRID)
maps = project_baseline_maps(proj_new, project_path, GRID)
ROWS = {"mobile_epifauna": [81.0, 123.0, 173.0, 221.0, 272.0, 387.0],
        "norway_lobster": [81.0, 123.0, 173.0, 221.0, 272.0, 300.0]}
hab = {}
for fid in ("mobile_epifauna", "norway_lobster"):
    loss = habitat.annual_impact_loss(fid, new, maps, GRID)
    m2 = habitat.predation_m2(fid, new, act_new)
    rg = S[fid]["growth_rate"] * TPY
    keep = np.clip(1.0 - (loss + m2) / rg, 0.0, 1.0)
    rows = np.array(ROWS[fid])
    basis = np.stack([habitat.capacity_map((rows, np.eye(len(rows))[i]), depth).ravel()
                      for i in range(len(rows))], axis=1)
    A = basis * keep.ravel()[:, None]
    tgt = today[fid].ravel()
    w = 1.0e3
    caps, _ = nnls(np.vstack([A, w * A.sum(axis=0) / A.shape[0]]),
                   np.concatenate([tgt, [w * tgt.sum() / A.shape[0]]]))
    caps = np.round(caps, 2)
    K = habitat.capacity_map((rows, caps), depth)
    eq = K * keep
    S[fid]["capacity_depth_table"] = [{"depth_m": float(d), "capacity": float(c)} for d, c in zip(rows, caps)]
    S[fid]["max_carrying_capacity"] = round(float(K.mean()), 1)
    bands = []
    for lo, hi in [(0, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 600)]:
        m = (depth >= lo) & (depth < hi)
        bands.append((lo, hi, float(today[fid][m].mean()), float(eq[m].mean()), float(K[m].mean())))
    hab[fid] = dict(rows=rows.tolist(), caps=caps.tolist(), m2=m2, r_gross=rg, F_mean=float(loss.mean()),
                    F_max=float(loss.max()), share=float(1 - (loss.mean() + m2) / rg), K_mean=float(K.mean()),
                    total=float(eq.sum()), today_total=float(tgt.sum()), eq_mean=float(eq.mean()),
                    rms=float(np.sqrt(np.mean((eq.ravel() - tgt) ** 2)) / tgt.mean()), bands=bands,
                    zero_cells=int((eq <= 0).sum()), keep_min=float(keep.min()))


def rnd_total(x):
    return int(round(x / 100.0) * 100)


START = {fid: rnd_total(h["total"]) for fid, h in hab.items()}
RANGE = {fid: (int(round(0.75 * t / 50.0) * 50), int(round(1.25 * t / 50.0) * 50)) for fid, t in START.items()}
for g in ndm:
    if g["group_id"] in START:
        g["initial_biomass_min"], g["initial_biomass_max"] = RANGE[g["group_id"]]
        g["inference_initial_biomass"] = START[g["group_id"]]

# ---------------------------------------------------------------------------------------------------
# text edits of the library (comments kept)
# ---------------------------------------------------------------------------------------------------
text = open(lib_path, encoding="utf-8-sig").read()


def block(text, name, start=0):
    m = re.compile(rf"^  {re.escape(name)}:\s*$", re.M).search(text, start)
    if m is None:
        raise KeyError(name)
    nxt = re.compile(r"^  \S|^\S", re.M).search(text, m.end())
    return m.start(), (nxt.start() if nxt else len(text))


def comments_above(text, pos):
    """Start of the run of '  #' comment lines directly above pos."""
    lines_before = text[:pos].split("\n")[:-1]
    i = len(lines_before)
    while i > 0 and lines_before[i - 1].startswith("  #"):
        i -= 1
    return len("\n".join(lines_before[:i])) + (1 if i > 0 else 0)


def replace_entry(text, name, new_text, keep_comments=False):
    a, b = block(text, name)
    a0 = a if keep_comments else comments_above(text, a)
    return text[:a0] + new_text + text[b:]


def insert_after(text, name, new_text):
    _, b = block(text, name)
    return text[:b] + new_text + text[b:]


def fmt_rows(rows, caps):
    return "".join(f"      - depth_m: {d:.1f}\n        capacity: {c:.2f}\n" for d, c in zip(rows, caps))


pE, pB, pP, pN = (sp["pairs"]["benthivorous_fish_preys_on_mobile_epifauna"],
                  sp["pairs"]["benthivorous_fish_preys_on_mobile_epifauna"],
                  sp["pairs"]["predatory_fish_preys_on_mobile_epifauna"],
                  sp["pairs"]["benthivorous_fish_preys_on_norway_lobster"])
hE, hN = hab["mobile_epifauna"], hab["norway_lobster"]
gdE, gdN = S["mobile_epifauna"]["growth_rate"], S["norway_lobster"]["growth_rate"]
mem_lines = "".join(f"      - {{name: {json.dumps(n) if ',' in n else n}, share: {s:.2f}}}\n" for n, s in MEMBERS_E)
booked_txt = ("stocks unchanged (223.10 convention)" if not variant.startswith("massbal") else
              "a member of its own, B = flow / (EE 0.9 x P/B) at Mackinson & Daskalov 2007 P/B")
species_E = f"""  mobile_epifauna:
    current_response: 0.0
    display_name: "Mobile Epifauna"
    # Section 223.12 decision (E1): the EXPOSED mobile epifauna only; Nephrops
    # is its own group, norway_lobster, below (no exchange between them: no
    # biological process turns burrowed Nephrops into crabs or shrimp; every
    # measured epifauna exchange is diel). Members by share of today's stock
    # (221 taxon stocks): crabs, hermit crabs and Munida 2.0 t/km2 (the North
    # Sea proxy 0.76 raised by mass balance of the fish's measured crab
    # consumption, section 221), northern shrimp 0.56 (SS3 strata), small
    # shrimps 0.3 and mysids 0.2 (J); grid mean {gE['B_today']:.2f} t/km2. In the fish
    # diets: crabs 0.75, northern shrimp 0.12, small shrimps and mysids 0.13.
    # The 223.10 redistribution (cephalopods, krill/hyperbenthos, small
    # demersal fish, the epifauna slices of unidentified fish and carrion;
    # 0.096 of the benthivores' consumption, {sp['checks']['extra_exposed'] or sum(sp['checks']['booked_flow'].values()):.2f} t/km2/yr) is booked on
    # this group: {booked_txt}.
    # {NOTE}; research_notes/"Refugium och utbyte i bentos" (Swedish letters).
    member_species:
{mem_lines}    is_decision_maker: false
    movement_speed: 0.0
    # Derived growth g = (r_max + sum M2)/1460. r_max {R_MAX['mobile_epifauna']} /yr (0.9-1.3): the
    # stock-weighted taxon r_max (crabs 0.9, northern shrimp 1.2, small
    # shrimps and mysids 1.86; section 221 and the refuge note); the one
    # logistic matching the sum of the taxon K would be {gE['r_max_self']:.2f} (J). r_gross
    # {hE['r_gross']:.2f} /yr against a stock-weighted literature P/B ~{sp['checks']['pb_bw_exposed']:.2f}: the M2 of the
    # booked prey (open flag).
    r_max: {R_MAX['mobile_epifauna']}
    growth_rate: {gdE:.11g}
    # K = today's stock / equilibrium share {hE['share']:.3f} (trawl {hE['F_mean']:.3f}, M2 {hE['m2']:.3f}),
    # rows fitted by non-negative least squares so the start map B*(c)
    # matches the members' depth structure (crabs 2:1 from 100 to 300 m,
    # northern shrimp peaking at 200-300 m, small shrimps and mysids flat)
    # under the effort map, total kept; rms {100 * hE['rms']:.0f} % of the mean.
    capacity_depth_table:
{fmt_rows(hE['rows'], hE['caps'])}    max_carrying_capacity: {S['mobile_epifauna']['max_carrying_capacity']}
    # 3.75 kJ/g WW as eaten (3.1-4.7; crustaceans with their shell; Lawson
    # et al. 1998 for northern shrimp); 3.74 recomputed on the E1 intake
    # weights without Nephrops, 3.77 with the booked prey (J energies).
    energy_content: 3750.0
    spawn:
      mode: equilibrium
    seed_rate: 1e-06
  norway_lobster:
    current_response: 0.0
    display_name: "Norway Lobster"
    # Section 223.12 decision (E1): Nephrops norvegicus, split out of
    # mobile_epifauna. 4.57 t/km2 grid mean: ICES UWTV 2024 stratum
    # burrow densities (FU 3-4; 0.26 / 0.09 / 0.55 per m2, one animal per
    # burrow) x 30 g per animal on
    # the per-cell survey (mud) share; flat to ~250 m, ~0 below 300 m. In
    # burrows ~93 % of the time (0.06-0.12 visible per burrow at peak
    # emergence, Aguzzi et al. 2021); berried females stay in for 7-10
    # months. 0.3 % of the in-area benthivores' epifauna diet (Skagerrak
    # cod 2023-25: 0.1 % of the benthos); none in whiting or saithe.
    # {NOTE}.
    member_species:
      - {{name: Norway lobster, share: 1.0}}
    is_decision_maker: false
    movement_speed: 0.0
    # Derived growth g = (r_max + sum M2)/1460. r_max 0.34 /yr (0.25-0.5):
    # lifespan > 10 yr (Hiddink et al. 2019 longevity class r 0.34), M
    # 0.2-0.3, Ecopath P/B 0.35-0.4 (Mackinson & Daskalov 2007).
    r_max: {R_MAX['norway_lobster']}
    growth_rate: {gdN:.11g}
    # K = today's stock / equilibrium share {hN['share']:.3f} (trawl {hN['F_mean']:.4f}, M2 {hN['m2']:.4f}),
    # non-negative least squares per depth row, total kept; the table ends
    # at 300 m (held below it), where the UWTV grounds end. Depth cannot
    # hold the mud patches (23 % of the cells have no Nephrops ground):
    # rms {100 * hN['rms']:.0f} % of the mean, band means within ~0.3 t/km2 above 300 m.
    capacity_depth_table:
{fmt_rows(hN['rows'], hN['caps'])}    max_carrying_capacity: {S['norway_lobster']['max_carrying_capacity']}
    # 3.6 kJ/g WW as eaten (2.8-4.8) J: decapod AFDW/WW 16.5 % x 20.74 kJ/g
    # AFDW = 3.4 (Ricciardi & Bourget 1998; Steimle & Terranova 1985);
    # Calocaris whole ~2.8 (Parslow-Williams 1998); Homarus 4.8. No
    # whole-body Nephrops calorimetry found. Was 4.0 J in section 221.
    energy_content: {E_NEPH}
    spawn:
      mode: equilibrium
    seed_rate: 1e-06
"""
text = replace_entry(text, "mobile_epifauna", species_E, keep_comments=True)

pair_BE = f"""  # E1 (section 223.12 decision): the exposed taxa only. Diet share 0.314
  # of the consumption (223.10 central, Q/B 4.46 x 1.53 = 6.82 t/km2/yr),
  # all of it except the Nephrops 0.003 of the measured epifauna diet:
  # {sp['groups']['mobile_epifauna']['flows']['benthivorous_fish']:.2f} t/km2/yr (crabs 1.30, northern shrimp 0.08, small shrimps
  # and mysids 0.11, booked cephalopods/krill/small demersal fish 0.66),
  # M2 = flow / B* {gE['Bstar_mean']:.2f} = {M2['benthivorous_fish_preys_on_mobile_epifauna']}; h = 3750 / 27.1 = 138.4; accessible
  # {F_PAIR['benthivorous_fish_preys_on_mobile_epifauna']} (crabs and hermits 0.6, northern shrimp 0.75, small shrimps and
  # mysids 0.6 of the time on the bed, by stock; 0.45-0.8); a 0.44
  # (0.15-1.2) per accessible tonne (2-D swept area, section 221). Half-
  # saturation 1/(a h f) {1 / (0.44 * 138.4 * F_PAIR['benthivorous_fish_preys_on_mobile_epifauna']):.3f} t/km2. No dark_ratio (f is a time
  # mean). {NOTE}.
  benthivorous_fish_preys_on_mobile_epifauna:
    preys_on: true
    predation_mortality: {M2['benthivorous_fish_preys_on_mobile_epifauna']}
    assimilation_factor: 0.8
    handling_time: 138.4
    accessible_fraction: {F_PAIR['benthivorous_fish_preys_on_mobile_epifauna']}
    max_intake_rate: 0.44
"""
text = replace_entry(text, "benthivorous_fish_preys_on_mobile_epifauna", pair_BE)
pair_PE = f"""  # E1: whiting, hake and saithe on northern shrimp, small shrimps,
  # mysids and small crabs: diet share 0.072 (benthos 0.11 x 0.65), {sp['groups']['mobile_epifauna']['flows']['predatory_fish']:.3f}
  # t/km2/yr, M2 = flow / B* {gE['Bstar_mean']:.2f} = {M2['predatory_fish_preys_on_mobile_epifauna']} (was 0.036 on the pooled 7.9);
  # h = 3750 / 25.5 = 147.1; accessible {F_PAIR['predatory_fish_preys_on_mobile_epifauna']} (northern shrimp 0.75, small
  # shrimps and mysids 0.6, a quarter of the crabs at 0.6; 0.2-0.45, = the
  # 0.13 of the pooled group); a 0.33 (0.10-1.5) per edible tonne, every
  # member at its own encounter rate. {NOTE}.
  predatory_fish_preys_on_mobile_epifauna:
    preys_on: true
    predation_mortality: {M2['predatory_fish_preys_on_mobile_epifauna']}
    assimilation_factor: 0.8
    handling_time: 147.1
    accessible_fraction: {F_PAIR['predatory_fish_preys_on_mobile_epifauna']}
    max_intake_rate: 0.33
  # E1: cod (and the few large haddock, rays) on emerged Nephrops: 0.003 of
  # the in-area measured epifauna diet ({sp['groups']['norway_lobster']['flows']['benthivorous_fish']:.4f} t/km2/yr), M2 = flow / B*
  # {gN['Bstar_mean']:.2f} = {M2['benthivorous_fish_preys_on_norway_lobster']} (northern North Sea cod diet: ~0.04; Irish Sea cod
  # 0.04-0.25); h = 3600 / 27.1 = {H_NEPH}; accessible {F_NEPH} = exposed share
  # 0.07 (0.03-0.35; UWTV camera, Aguzzi et al. 2021) x the edible size
  # window 0.29 (cod < 66 cm take 96 % <= 39 mm CL, De Clerck & Sterk 1980;
  # 0.29 of the biomass at <= 35 mm CL) (J; 0.07 without the gape filter);
  # a 0.44 (emerged Nephrops are moving decapods: the exposed-pair
  # detection width and capture). Saturated at today's stock. {NOTE}.
  benthivorous_fish_preys_on_norway_lobster:
    preys_on: true
    predation_mortality: {M2['benthivorous_fish_preys_on_norway_lobster']}
    assimilation_factor: 0.8
    handling_time: {H_NEPH}
    accessible_fraction: {F_NEPH}
    max_intake_rate: 0.44
  # E1: no Nephrops in whiting, saithe or hake stomachs in the box or the
  # Skagerrak (whiting elsewhere 0.04 of its benthos): unchecked.
  predatory_fish_preys_on_norway_lobster:
    preys_on: false
    assimilation_factor: 1.0
    handling_time: 0.0
"""
text = replace_entry(text, "predatory_fish_preys_on_mobile_epifauna", pair_PE)

imp_E = f"""  # E1: catch incl. dead discards (northern shrimp 0.336, crabs 0.037,
  # small shrimps 0.005 t/km2/yr; resume 218) + kill of the uncaught part
  # (d_u x SAR 2.26: crabs 0.05, shrimps 0.01-0.02; 0.26) over {gE['B_today']:.2f} t/km2 = {gE['F']:.3f}
  # /yr at the map's mean effort 3.0142 (northern shrimp alone F 0.62,
  # crabs 0.13, small shrimps 0.055). {NOTE}.
  mobile_epifauna_impacted_by_bottom_trawling:
    impact_affects: true
    impact_table:
      - value: 0.0
        biomass_factor: 0.0
        energy_factor: 0.0
      - value: 100.0
        biomass_factor: {BF100['mobile_epifauna']}
        energy_factor: 0.0
"""
text = replace_entry(text, "mobile_epifauna_impacted_by_bottom_trawling", imp_E)
imp_N = f"""  # E1: Nephrops landings + 75 % dead of the 17.6 % discards, 0.296 t/km2/yr
  # (STECF FDI, Norwegian ERS; resume 218), + kill of the uncaught part in
  # burrows d_u 0.01 x SAR 2.26 (0.10) over 4.57 t/km2 = {gN['F']:.4f} /yr at the
  # map's mean effort 3.0142. {NOTE}.
  norway_lobster_impacted_by_bottom_trawling:
    impact_affects: true
    impact_table:
      - value: 0.0
        biomass_factor: 0.0
        energy_factor: 0.0
      - value: 100.0
        biomass_factor: {BF100['norway_lobster']}
        energy_factor: 0.0
  norway_lobster_impacted_by_pelagic_trawling:
    impact_affects: false
    impact_table: []
  norway_lobster_impacted_by_rotor:
    impact_affects: false
    impact_table: []
  norway_lobster_impacted_by_windfarm_noise:
    impact_affects: false
    impact_table: []
"""
text = insert_after(text, "mobile_epifauna_impacted_by_windfarm_noise", imp_N)
obs_N = """  # E1: the benthivores see both benthic crustacean groups; the predators
  # only the exposed one (no pair on Nephrops).
  benthivorous_fish_observes_norway_lobster:
    observes: true
  predatory_fish_observes_norway_lobster:
    observes: false
"""
text = insert_after(text, "predatory_fish_observes_mobile_epifauna", obs_N)


def set_growth(text, fid, g):
    a, b = block(text, fid)
    blk = re.sub(r"^    growth_rate:.*$", f"    growth_rate: {g:.11g}", text[a:b], count=1, flags=re.M)
    return text[:a] + blk + text[b:]


changed_g = []
for fid, b in b_new.items():
    if fid in ("mobile_epifauna", "norway_lobster") or b["g"] is None:
        continue
    g_old = float(old["species_definitions"][fid].get("growth_rate", 0.0) or 0.0)
    if g_old > 0 and abs(b["g"] / g_old - 1.0) > 1e-5:
        text = set_growth(text, fid, b["g"])
        changed_g.append(fid)
text = (f"# RESEARCH COPY ({NOTE}, variant {variant}, Nephrops f {F_NEPH}):\n"
        f"# E1 split of mobile_epifauna into the exposed taxa and norway_lobster, built on the live library\n"
        f"# sha256 {lib_sha}. NOT the live library.\n" + text)
dst = os.path.join(out, f"fg_library_e1_{variant}.yaml")
open(dst, "w", encoding="utf-8").write(text)

# ---- manifest copy: text edit ----
mtext = open(project_path, encoding="utf-8-sig").read()
m = re.search(r"^  - group_id: mobile_epifauna\n(?:    .*\n)+", mtext, re.M)
nd = {g["group_id"]: g for g in ndm}
ent = (f"  # E1 (section 223.12 decision): mobile_epifauna is the exposed taxa,\n"
       f"  # norway_lobster the Nephrops; start = sum of today's equilibrium B*(c),\n"
       f"  # +/-25 % ({NOTE}).\n")
for fid in ("mobile_epifauna", "norway_lobster"):
    ent += (f"  - group_id: {fid}\n    initial_biomass_min: {nd[fid]['initial_biomass_min']}\n"
            f"    initial_biomass_max: {nd[fid]['initial_biomass_max']}\n"
            f"    inference_initial_biomass: {nd[fid]['inference_initial_biomass']}\n")
mtext = mtext[:m.start()] + ent + mtext[m.end():]
mdst = os.path.join(out, f"mareld2_e1_{variant}.yaml")
open(mdst, "w", encoding="utf-8").write(
    f"# RESEARCH COPY ({NOTE}): manifest with norway_lobster. Relative map paths resolve from the\n"
    f"# repository root (copy it there to use it). NOT the live manifest.\n" + mtext)
snip = os.path.join(out, f"mareld2_e1_{variant}_snippet.yaml")
open(snip, "w", encoding="utf-8").write("non_decision_makers:\n" + ent)

# ---------------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------------
chk, pchk = load_config(dst), load_config(mdst)
assert active_of(pchk) == act_new, (active_of(pchk), act_new)
Sx, Ix = chk["species_definitions"], chk["interaction_definitions"]
bad = []
for fid in act_new:
    g = gb.derived_growth_rate(fid, Sx, Ix, act_new)
    if g is not None and abs(float(Sx[fid]["growth_rate"]) - g) > 1e-6 * g:
        bad.append((fid, float(Sx[fid]["growth_rate"]), g))
_, hx = habitat.build(pchk, project_path, chk, GRID, act_new, baseline_maps=maps)
man = {g["group_id"]: g for g in pchk["non_decision_makers"]}
fgs = load_project_config(project_path, library_config=chk, project_config=pchk, seed=1, spawn_seed=1)
fg_ids = sorted(getattr(f, "name", getattr(f, "fg_id", str(f))) for f in (fgs[0] if isinstance(fgs, tuple) else fgs))

print(f"live library sha256 {lib_sha}; variant {variant}; Nephrops f {F_NEPH}")
print(f"active groups (copy): {act_new}")
print()
print(f"{'group':26s} {'g old':>14s} {'g new':>14s} {'ratio':>7s} {'s* old':>7s} {'s* new':>7s} {'gross new':>9s} best prey")
for fid in act_new:
    n_ = b_new.get(fid)
    if n_ is None:
        continue
    o_ = b_old.get(fid, {"g": None, "s_star": None})
    go = "-" if o_["g"] is None else f"{o_['g']:.10f}"
    ratio = f"{n_['g'] / o_['g']:.4f}" if o_.get("g") and n_["g"] else "-"
    so = "-" if o_.get("s_star") is None else f"{o_['s_star']:.4f}"
    sn = "-" if n_["s_star"] is None else f"{n_['s_star']:.4f}"
    print(f"{fid:26s} {go:>14s} {n_['g']:14.10f} {ratio:>7s} {so:>7s} {sn:>7s} {n_['gross']:9.4f} {n_['best'] or ''}"
          f" {n_['err'] or ''}")
print()
for fid, h in hab.items():
    print(f"{fid}: r_gross {h['r_gross']:.4f}, M2 {h['m2']:.4f}, box-mean F {h['F_mean']:.4f} (max {h['F_max']:.3f}), "
          f"share {h['share']:.4f}, keep min {h['keep_min']:.3f}")
    print(f"   K rows {dict(zip(h['rows'], h['caps']))}; box-mean K {h['K_mean']:.3f}")
    print(f"   start total {h['total']:.0f} t (today {h['today_total']:.0f}); B* mean {h['eq_mean']:.3f}; rms "
          f"{100 * h['rms']:.1f} %; zero cells {h['zero_cells']}")
    for lo, hi, t, e, k in h["bands"]:
        print(f"     {lo:>3}-{hi:<3} m: today {t:7.3f}  engine B* {e:7.3f}  K {k:7.3f}")
print()
print(f"growth rates re-stored for other groups: {changed_g or 'none'}")
print(f"copy written: {dst}; manifest copy {mdst}; snippet {snip}")
print(f"stored != derived: {bad or 'none'}")
for fid in ("benthic_infauna", "mobile_epifauna", "norway_lobster"):
    tot = float(hx[fid]["equilibrium"].sum())
    ref = float(man[fid]["inference_initial_biomass"])
    print(f"   {fid}: habitat.build equilibrium total {tot:.0f} t vs manifest copy {ref:.0f} "
          f"({100 * (tot / ref - 1):+.2f} %; test tolerance 2 %); range {man[fid]['initial_biomass_min']}-"
          f"{man[fid]['initial_biomass_max']}")
print(f"load_project_config built {len(fg_ids)} FGs: {fg_ids}")
json.dump(dict(variant=variant, lib_sha=lib_sha, f_neph=F_NEPH, active=act_new, budgets_old=b_old,
               budgets_new=b_new, habitat=hab, start=START, range=RANGE, m2=M2, f=F_PAIR, bf100=BF100,
               r_max=R_MAX, members=MEMBERS_E, bad=bad),
          open(os.path.join(out, f"build_e1_{variant}.json"), "w"), indent=1, default=str)
