"""Does the passive current put the plankton, and the mackerel after it, in the south-east corner?
Frozen policies, the run's probe world (unfished as in training), the run's grid and seed, CPU. Two arms:
  on  - the run's current config (east/south drift only, movement.current_direction_fractions);
  off - currents=None, everything else equal.
Per group and quarter of each probe year (quarters counted from the inference start day):
  - the biomass centroid as an offset from the grid centre in cells (+row = south, +col = east);
  - the share of the biomass in the south-east quadrant (0.25 if spread evenly over the cells);
  - the group's mean total relative to its start total;
and the spatial correlation of mackerel with zooplankton.
usage: RUN=<run> POL=<dir> ARM=on|off current_onoff.py <years>"""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
years = int(sys.argv[1]); S = os.path.dirname(os.path.abspath(__file__))
import numpy as np, torch
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from inference import load_policies_and_stats
torch.set_num_threads(3)
arm = os.environ['ARM']
cfg = json.load(open('results/' + os.environ['RUN'] + '/progress/config.json')); seed = cfg['seed']
keys = ("project", "library", "grid", "mortality", "migration", "mortality_multiplier", "mass_balance", "currents", "tick_hours")
eco = {k: cfg[k] for k in keys if k in cfg}
if arm == 'off':
    eco['currents'] = None
with evaluation_randomness(seed): env = build_inference_env(eco, seed)
with evaluation_randomness(seed):
    pol, om, ov = load_policies_and_stats(env, os.environ['POL'], verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = om, ov
    env.softmax_temperature = cfg.get('temperature', 1.0)
print('arm', arm, '| currents', env.currents, flush=True)
H, W = env.H, env.W
rows, cols = np.indices((H, W)).astype(float)
se = (rows >= H // 2) & (cols >= W // 2)
groups = [f for f in env.global_fg_order]
start = {f: float(env.fgs[f].biomass.sum()) for f in groups}
QT = 365  # ticks per quarter at 6 h
out = []
with evaluation_randomness(seed):
    for y in range(years):
        for q in range(4):
            acc = {f: np.zeros((H, W)) for f in groups}
            for t in range(QT):
                env.tick()
                for f in groups:
                    acc[f] += env.fgs[f].biomass
            rec = dict(year=y + 1, quarter=q + 1, groups={})
            for f in groups:
                a = acc[f]; tot = a.sum()
                if tot <= 0:
                    continue
                rec['groups'][f] = dict(
                    d_row=float((a * rows).sum() / tot - (H - 1) / 2),
                    d_col=float((a * cols).sum() / tot - (W - 1) / 2),
                    se_share=float(a[se].sum() / tot),
                    rel=float(tot / QT / max(start[f], 1e-12)))
            if 'mackerel' in acc and 'zooplankton' in acc:
                m, z = acc['mackerel'].ravel(), acc['zooplankton'].ravel()
                rec['mack_zoo_corr'] = float(np.corrcoef(m, z)[0, 1]) if m.std() > 0 and z.std() > 0 else None
            out.append(rec)
            print(f"year {y+1} Q{q+1}: group d_row d_col se_share rel | mack~zoo r={rec.get('mack_zoo_corr')}", flush=True)
            for f, g in rec['groups'].items():
                print(f"  {f[:24]:24s} {g['d_row']:+6.2f} {g['d_col']:+6.2f} {g['se_share']:.2f} {g['rel']:.2f}", flush=True)
os.makedirs(f"{S}/../tables", exist_ok=True)
json.dump(out, open(f"{S}/../tables/current_onoff_{os.environ['RUN']}_{arm}.json", 'w'))
