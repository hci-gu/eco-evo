"""Proposed benthivore library values -> derived growth rates, K tables and a library COPY.

Usage (from the repository root):
  python3 -I "research_notes/Bentivorernas energi och diet/scripts/build_library.py" \
      fgconfig/fg_library.yaml mareld2.yaml "<out_dir>"

Never touches the real library. Reads it, applies the proposal (PROPOSAL below) to an
in-memory copy, derives every growth_rate with lib/world/growth_budget.py for the active
project groups (mareld2.yaml, muted groups out, --mortality on), re-fits the two benthic
capacity_depth_tables so the engine's start map B*(c) (lib/world/habitat.py) stays today's
map, and writes <out_dir>/fg_library_proposed.yaml as a text-edited copy of the library
(comments kept) plus build_library.json with every number.
"""
import copy
import json
import os
import re
import sys

import numpy as np

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
from lib.config.config_loader import load_config            # noqa: E402
from lib.world import growth_budget as gb                   # noqa: E402
from lib.world import habitat                               # noqa: E402
from lib.world.impact_maps import project_baseline_maps     # noqa: E402
from lib.world.tick_time import LIBRARY_TICK_HOURS, ticks_per_year  # noqa: E402

lib_path, project_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
TPY = ticks_per_year(LIBRARY_TICK_HOURS)
GRID = (60, 60)

# ---- the proposal (diet_m2.py central, Q/B 4.2; lean_reserve.py) ----
SPECIES = {"benthivorous_fish": {"energy_content": 3600.0, "max_energy_reserve": 1800.0,
                                 "prey_includes_reserve": True, "reserve_reference_fill": 0.5}}
PAIRS = {"benthivorous_fish_preys_on_benthic_infauna": 0.037,
         "benthivorous_fish_preys_on_mobile_epifauna": 0.26,
         "benthivorous_fish_preys_on_planktivorous_fish": 0.23,
         "benthivorous_fish_preys_on_predatory_fish": 0.12}
if len(sys.argv) > 4 and sys.argv[4] == "energy_only":
    PAIRS = {}
if len(sys.argv) > 4 and sys.argv[4] == "diet_only":
    SPECIES = {}
if len(sys.argv) > 4 and sys.argv[4] == "energy_eq":
    # central diet at the energy-equivalent Q/B 4.457 (diet_m2.csv row "central,energy-eq";
    # user's choice, resume 223.9)
    PAIRS = {"benthivorous_fish_preys_on_benthic_infauna": 0.0396,
             "benthivorous_fish_preys_on_mobile_epifauna": 0.276,
             "benthivorous_fish_preys_on_planktivorous_fish": 0.245,
             "benthivorous_fish_preys_on_predatory_fish": 0.128}
elif len(sys.argv) > 4 and sys.argv[4].startswith("qb"):
    # Q/B range: the central-diet M2 scaled by Q/B / 4.2 (diet_m2.py rows)
    _f = float(sys.argv[4][2:]) / 4.2
    PAIRS = {k: round(v * _f, 4) for k, v in PAIRS.items()}

old = load_config(lib_path)
project = load_config(project_path)
active = [g["group_id"] for key in ("decision_makers", "non_decision_makers")
          for g in (project.get(key) or []) if isinstance(g, dict) and not g.get("muted")]
new = copy.deepcopy(old)
for fid, kv in SPECIES.items():
    new["species_definitions"][fid].update(kv)
for key, m2 in PAIRS.items():
    new["interaction_definitions"][key]["predation_mortality"] = m2


def budgets(lib):
    S, I = lib["species_definitions"], lib["interaction_definitions"]
    res = {}
    for fid in active:
        b = gb.growth_budget(fid, S, I, active_predators=active)
        if b is None:
            continue
        res[fid] = dict(g=b.growth_rate, err=b.error, s_star=b.reference_fill,
                        gate=b.gate, gross=b.gross_annual, m2=dict(b.predation),
                        C=b.ideal_intake, cost=b.ideal_cost, best=b.best_prey,
                        pmax=b.pmax_annual)
    return res


b_old, b_new = budgets(old), budgets(new)
for fid in b_new:
    new["species_definitions"][fid]["growth_rate"] = b_new[fid]["g"]

# ---- habitat: keep today's start map B*(c) ----
depth = habitat.load_depth_map(project, project_path, GRID)
maps = project_baseline_maps(project, project_path, GRID)
BENTHIC = ("benthic_infauna", "mobile_epifauna")
hab = {}
for fid in BENTHIC:
    so, sn = old["species_definitions"][fid], new["species_definitions"][fid]
    table = habitat.parse_capacity_table(so)
    loss = habitat.annual_impact_loss(fid, old, maps, GRID)
    m2o, m2n = habitat.predation_m2(fid, old, active), habitat.predation_m2(fid, new, active)
    ro, rn = float(so["growth_rate"]) * TPY, b_new[fid]["g"] * TPY
    eq_old = habitat.equilibrium_map(habitat.capacity_map(table, depth), loss, m2o, ro)
    keep_new = np.clip(1.0 - (loss + m2n) / rn, 0.0, 1.0)
    # K(c) is linear in the table rows: basis maps from unit rows, then least squares
    depths, caps = table
    basis = np.stack([habitat.capacity_map((depths, np.eye(len(caps))[i]), depth).ravel()
                      for i in range(len(caps))], axis=1)
    A = basis * keep_new.ravel()[:, None]
    rows_new = np.linalg.lstsq(A, eq_old.ravel(), rcond=None)[0]
    eq_new = habitat.capacity_map((depths, rows_new), depth) * keep_new
    eq_unrefit = habitat.capacity_map(table, depth) * keep_new
    hab[fid] = dict(depths=depths.tolist(), caps_old=caps.tolist(), caps_new=rows_new.tolist(),
                    m2_old=m2o, m2_new=m2n, r_old=ro, r_new=rn, F_mean=float(loss.mean()),
                    share_old=float(1 - (loss.mean() + m2o) / ro),
                    share_new=float(1 - (loss.mean() + m2n) / rn),
                    total_old=float(eq_old.sum()), total_new=float(eq_new.sum()),
                    total_if_K_kept=float(eq_unrefit.sum()),
                    rms_rel=float(np.sqrt(np.mean((eq_new - eq_old) ** 2)) / eq_old.mean()),
                    K_mean_old=float(habitat.capacity_map(table, depth).mean()),
                    K_mean_new=float(habitat.capacity_map((depths, rows_new), depth).mean()))

# ---- report ----
print(f"active groups: {active}")
print()
print(f"{'group':22s} {'g old':>12s} {'g new':>12s} {'ratio':>6s} {'s* old':>7s} {'s* new':>7s} "
      f"{'gross old':>9s} {'gross new':>9s}  error")
for fid in b_new:
    o, n = b_old[fid], b_new[fid]
    so_ = "-" if o["s_star"] is None else f"{o['s_star']:.4f}"
    sn_ = "-" if n["s_star"] is None else f"{n['s_star']:.4f}"
    ratio = n["g"] / o["g"] if o["g"] and n["g"] else float("nan")
    print(f"{fid:22s} {o['g']:12.10f} {n['g']:12.10f} {ratio:6.3f} {so_:>7s} {sn_:>7s} "
          f"{o['gross']:9.4f} {n['gross']:9.4f}  {n['err'] or ''}")
print()
bo, bn = b_old["benthivorous_fish"], b_new["benthivorous_fish"]
print(f"benthivorous_fish ideal intake C {bn['C']:.3f} MJ/t/tick (best {bn['best']}), cost "
      f"{bn['cost']:.2f}; Pmax {bo['pmax']:.3f} -> {bn['pmax']:.3f} /yr; gross {bn['gross']:.4f} /yr")
print()
for fid, h in hab.items():
    print(f"{fid}: M2 {h['m2_old']:.4f} -> {h['m2_new']:.4f}, r_gross {h['r_old']:.4f} -> "
          f"{h['r_new']:.4f}, box-mean F {h['F_mean']:.4f}, share {h['share_old']:.3f} -> "
          f"{h['share_new']:.3f}")
    print(f"   K rows at {h['depths']}: {[round(c, 2) for c in h['caps_old']]} -> "
          f"{[round(c, 2) for c in h['caps_new']]}; box-mean K {h['K_mean_old']:.2f} -> "
          f"{h['K_mean_new']:.2f}")
    print(f"   start total {h['total_old']:.0f} t -> refit {h['total_new']:.0f} t "
          f"(K kept: {h['total_if_K_kept']:.0f} t); rms deviation of B*(c) {100 * h['rms_rel']:.2f} %")

# ---- library copy: text edits so the comments survive ----
text = open(lib_path, encoding="utf-8-sig").read()


def block(text, name, section_start=0):
    m = re.compile(rf"^  {re.escape(name)}:\s*$", re.M).search(text, section_start)
    if m is None:
        raise KeyError(name)
    nxt = re.compile(r"^  \S", re.M).search(text, m.end())
    nxt2 = re.compile(r"^\S", re.M).search(text, m.end())
    ends = [x.start() for x in (nxt, nxt2) if x is not None]
    return m.start(), min(ends) if ends else len(text)


def set_key(text, name, key, value, after=None):
    a, b = block(text, name)
    blk = text[a:b]
    pat = re.compile(rf"^    {re.escape(key)}:.*$", re.M)
    line = f"    {key}: {value}"
    if pat.search(blk):
        blk = pat.sub(line, blk, count=1)
    else:
        anchor = re.compile(rf"^    {re.escape(after)}:.*$", re.M).search(blk)
        blk = blk[:anchor.end()] + "\n" + line + blk[anchor.end():]
    return text[:a] + blk + text[b:]


def set_table(text, name, depths, caps):
    a, b = block(text, name)
    blk = text[a:b]
    m = re.compile(r"^    capacity_depth_table:\s*\n((?:      .*\n)+)", re.M).search(blk)
    rows = "".join(f"      - depth_m: {d:.1f}\n        capacity: {c:.2f}\n"
                   for d, c in zip(depths, caps))
    blk = blk[:m.start(1)] + rows + blk[m.end(1):]
    return text[:a] + blk + text[b:]


for fid, kv in SPECIES.items():
    text = set_key(text, fid, "energy_content", f"{kv['energy_content']:.1f}")
    text = set_key(text, fid, "max_energy_reserve", f"{kv['max_energy_reserve']:.1f}")
    text = set_key(text, fid, "prey_includes_reserve", "true", after="energy_content")
    text = set_key(text, fid, "reserve_reference_fill", f"{kv['reserve_reference_fill']}",
                   after="prey_includes_reserve")
for key, m2 in PAIRS.items():
    text = set_key(text, key, "predation_mortality", f"{m2}")
for fid in b_new:
    g_stored = float(old["species_definitions"][fid].get("growth_rate", 0.0) or 0.0)
    if (b_new[fid]["g"] is not None and g_stored > 0.0
            and abs(b_new[fid]["g"] / g_stored - 1.0) > 1e-5):
        text = set_key(text, fid, "growth_rate", f"{b_new[fid]['g']:.11g}")
for fid, h in hab.items():
    text = set_table(text, fid, h["depths"], h["caps_new"])
    text = set_key(text, fid, "max_carrying_capacity", f"{h['K_mean_new']:.1f}")
text = ("# RESEARCH COPY (research_notes/\"Bentivorernas energi och diet\"): proposed benthivore\n"
        "# lean/reserve energetics and redistributed diet, Q/B 4.2. NOT the live library.\n" + text)
tag = sys.argv[4] if len(sys.argv) > 4 else "proposed"
dst = os.path.join(out, f"fg_library_{tag}.yaml")
open(dst, "w", encoding="utf-8").write(text)

# verify the copy: parses, stored growth = derived, start totals
chk = load_config(dst)
Sx, Ix = chk["species_definitions"], chk["interaction_definitions"]
bad = []
for fid in active:
    g = gb.derived_growth_rate(fid, Sx, Ix, active)
    if g is not None and abs(float(Sx[fid]["growth_rate"]) - g) > 1e-5 * g:
        bad.append(fid)
_, hx = habitat.build(project, project_path, chk, GRID, active, baseline_maps=maps)
man = {g["group_id"]: g for g in project["non_decision_makers"]}
print()
print(f"copy written: {dst}; stored != derived: {bad or 'none'}")
for fid in BENTHIC:
    tot = float(hx[fid]["equilibrium"].sum())
    ref = float(man[fid]["inference_initial_biomass"])
    print(f"   {fid}: equilibrium total in the copy {tot:.0f} t vs manifest {ref:.0f} "
          f"({100 * (tot / ref - 1):+.2f} %; test tolerance 2 %)")

json.dump(dict(active=active, budgets_old=b_old, budgets_new=b_new, habitat=hab,
               species=SPECIES, pairs=PAIRS),
          open(os.path.join(out, f"build_library_{tag}.json"), "w"), indent=1, default=str)
