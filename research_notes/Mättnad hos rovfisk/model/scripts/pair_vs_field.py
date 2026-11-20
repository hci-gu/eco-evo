"""Per pair: the frozen world's uncapped feeding on each prey (fill_decomp_all.py) against the library's booked field
flow on that pair, both on the ceiling scale.
  booked level_j = M2_ij B_j / B_i / 1460 x h_ij          (B at the manifest's inference densities, as budget_gate's
                                                           total-ration row; h the engine's handling time per tick)
  field level_j  = booked level_j x field Q/B / booked Q/B (the group's field total ration, FIELD_QB; scales every
                                                           pair of a group alike)
  model unc_j    = share_j x unc;  realised_j = share_j x feed (the cap scales every prey of a cell alike)
Ratio unc_j / field_j per pair: equal ratios within a group = a group-level excess (time budget or a generic
factor); ratios that differ by prey type = a prey-specific excess. The booked flows are the library's field diets
x field Q/B (some pairs re-booked; see AGENTS.md); they set nothing here.
Usage: pair_vs_field.py fill_decomp_all_s0.json [fill_decomp_all_s1.json ...]"""
import json, os, sys
import numpy as np
ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT)
runs = [json.load(open(p)) for p in sys.argv[1:]]
os.chdir(ROOT)
import yaml
import train as train_mod
from lib.diagnostics.field_diet import FIELD_QB
builder = train_mod._ProbeEnvBuilder(project_path='mareld2.yaml', grid_size=(16, 16),
                                     apply_natural_mortality=True, migration=True)
env = builder(); env._build_static_caches()
man = yaml.safe_load(open('mareld2.yaml', encoding='utf-8-sig'))
meta = man['project_metadata']; area = float(meta['reference_grid_width']) * float(meta['reference_grid_height'])
dens = {}
for entries in man.values():
    if isinstance(entries, list):
        for e in entries:
            if isinstance(e, dict) and e.get('inference_initial_biomass') is not None and 'group_id' in e:
                dens[e['group_id']] = float(e['inference_initial_biomass']) / area
h = np.asarray(env.handling_time_mat, dtype=np.float64)
order = list(env.global_fg_order)
for g in runs[0]:
    i = env.dm_ids.index(g)
    pairs = env.fgs[g].params.get('interaction', {}) or {}
    prey = runs[0][g]['prey']
    unc = np.mean([r[g]['unc'] for r in runs]); feed = np.mean([r[g]['feed'] for r in runs])
    share = np.mean([r[g]['share'] for r in runs], axis=0)
    booked = {}
    for p in prey:
        j = order.index(p)
        m2 = float((pairs.get(f"{g}_preys_on_{p}") or {}).get('predation_mortality', 0.0) or 0.0)
        booked[p] = m2 * dens.get(p, 0.0) / dens[g] / 1460.0 * h[i, j]
    bq = sum(m2b for m2b in [float((pairs.get(f"{g}_preys_on_{p}") or {}).get('predation_mortality', 0.0) or 0.0)
                              * dens.get(p, 0.0) / dens[g] for p in prey])
    qb, src = FIELD_QB.get(g, (bq, 'booked'))
    sc = qb / bq if bq > 0 else float('nan')
    btot = sum(booked.values()); ftot = btot * sc
    print(f"\n{g}: uncapped {unc:.3f}, realised {feed:.3f}; booked total {btot:.3f}, field total {ftot:.3f} "
          f"(Q/B {qb:.2f} vs booked {bq:.2f}, {src}); realised / field {feed / ftot:.2f}, uncapped / field "
          f"{unc / ftot:.2f}")
    print(f"  {'prey':22s} {'unc':>7s} {'real':>7s} {'field':>7s} {'unc/field':>9s} {'real/field':>10s}"
          f"  {'diet model':>10s} {'field':>6s}")
    for k, p in sorted(enumerate(prey), key=lambda kv: -share[kv[0]]):
        f = booked[p] * sc
        u, r = share[k] * unc, share[k] * feed
        rr = lambda a, b: f"{a / b:9.2f}" if b > 1e-9 else f"{'unbooked':>9s}"
        print(f"  {p:22s} {u:7.3f} {r:7.3f} {f:7.3f} {rr(u, f)} {rr(r, f):>10s}  {share[k]:10.3f} "
              f"{booked[p] / btot if btot > 0 else 0:6.3f}")
