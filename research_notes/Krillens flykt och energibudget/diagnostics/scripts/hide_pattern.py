"""When does each decision maker rest (= hide; the rested fraction is invisible to its predators)?
Frozen policies, the run's probe world (unfished as in training), the run's grid, seed and currents, CPU, nothing
altered. Per DM, month and tick of day (dark = the two ticks around midnight, slots 0 and 3 of the 6-h day;
bright = slots 1 and 2 around noon; a light threshold leaves Mar-Sep without any dark tick at 6 h, whose tick-mean
light is 0.1-0.74 then), the biomass-weighted shares of rest, move and eat in the policy's action distribution (after the sub-unit merge, as
applied). Also each DM's mean biomass by month (t/km2 per cell).
usage: RUN=<run> POL=<dir> hide_pattern.py <years>"""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
years = int(sys.argv[1]); S = os.path.dirname(os.path.abspath(__file__))
import numpy as np, torch
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from inference import load_policies_and_stats
from lib.environments.ecosystem_env import decisions, interactions
from lib.world import daylight
from lib.world.tick_time import ticks_per_day
torch.set_num_threads(3)
cfg = json.load(open('results/' + os.environ['RUN'] + '/progress/config.json')); seed = cfg['seed']
keys = ("project", "library", "grid", "mortality", "migration", "mortality_multiplier", "mass_balance", "currents", "tick_hours")
eco = {k: cfg[k] for k in keys if k in cfg}
with evaluation_randomness(seed): env = build_inference_env(eco, seed)
with evaluation_randomness(seed):
    pol, om, ov = load_policies_and_stats(env, os.environ['POL'], verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = om, ov
    env.softmax_temperature = cfg.get('temperature', 1.0)
dm = list(env.dm_ids); N = len(dm); cells = float(env.H * env.W)
seen = {}
orig = decisions.actions_from_probabilities


def patched(env_, probs, subthreshold):
    act = orig(env_, probs, subthreshold)
    b = np.stack([env_.fgs[f].biomass for f in dm]).astype(float)
    tot = np.maximum(b.sum(axis=(1, 2)), 1e-30)
    w = lambda x: (np.asarray(x, dtype=float) * b).sum(axis=(1, 2)) / tot
    seen['rest'] = w(act.rest); seen['move'] = w(act.move.sum(axis=1)); seen['eat'] = w(act.eat.sum(axis=1))
    seen['B'] = b.sum(axis=(1, 2)) / cells
    return act


decisions.actions_from_probabilities = patched
tpd = ticks_per_day(float(env.daylight['tick_hours'])); tpy = int(round(365 * tpd))
LIGHTS = ('dark', 'bright', 'all')
out = []
with evaluation_randomness(seed):
    for y in range(years):
        acc = {k: np.zeros((len(LIGHTS), 12, N)) for k in ('rest', 'move', 'eat')}
        n = np.zeros((len(LIGHTS), 12)); B = np.zeros((12, N))
        for k in range(tpy):
            tick = interactions.light_index(env)
            month = daylight.month_of_day(int(tick / tpd))
            slot = int(tick % tpd)
            env.tick()
            rows = [2, 0 if slot in (0, int(tpd) - 1) else 1]
            for r in rows:
                n[r, month] += 1
                for key in acc: acc[key][r, month] += seen[key]
            B[month] += seen['B']
        nn = np.maximum(n, 1)[:, :, None]
        rec = dict(year=y + 1, dm=dm, lights=list(LIGHTS),
                   share={key: (acc[key] / nn).tolist() for key in acc},
                   biomass=(B / np.maximum(n[2], 1)[:, None]).tolist())
        out.append(rec)
        print(f"=== year {y+1}: rest share (biomass-weighted) by month; rows dark (ticks around midnight) / bright (around noon) ===", flush=True)
        print(f"{'':22s}" + ''.join(f'{m:6d}' for m in range(1, 13)))
        for d, f in enumerate(dm):
            for r, lab in ((0, 'dark'), (1, 'bright')):
                print(f"{(f[:16] + ' ' + lab):22s}" + ''.join(f"{rec['share']['rest'][r][m][d]:6.2f}" for m in range(12)))
        print("move share (all light):")
        for d, f in enumerate(dm):
            print(f"{f[:22]:22s}" + ''.join(f"{rec['share']['move'][2][m][d]:6.2f}" for m in range(12)))
        print("mean biomass t/km2:")
        for d, f in enumerate(dm):
            print(f"{f[:22]:22s}" + ''.join(f"{rec['biomass'][m][d]:6.2f}" for m in range(12)), flush=True)
os.makedirs(f"{S}/../tables", exist_ok=True)
json.dump(out, open(f"{S}/../tables/hide_pattern_{os.environ['RUN']}.json", 'w'))
