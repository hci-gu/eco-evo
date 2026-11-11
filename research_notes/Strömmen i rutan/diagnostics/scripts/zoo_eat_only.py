"""What does the zooplankton's learned eat/rest choice do? Frozen policies, the run's probe world (unfished as in
training), the run's grid, seed and currents, CPU. Two arms:
  learned - the zooplankton policy as trained;
  eat     - the same policy with its rest (hide) mass moved to its eat actions in proportion to its learned eat shares
            (diagnostic override after the decision; every other DM untouched).
Booked in both arms (the rest share is read BEFORE the override, so both arms show the policy's wish):
  - per month: plankton densities (t/km2: grid mean per cell, the library density unit), the zooplankton's biomass-weighted
    rest share overall, at night (light observation < 0.01) and by day (> 0.5), and the A17 resident zooplankton for comparison;
  - per year: tonnes eaten per predator and prey (env.intake_by_pred_prey), the fish groups' mean biomass / start.
usage: RUN=<run> POL=<dir> ARM=learned|eat zoo_eat_only.py <years>"""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
years = int(sys.argv[1]); S = os.path.dirname(os.path.abspath(__file__))
import numpy as np, torch
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from inference import load_policies_and_stats
from lib.environments.ecosystem_env import decisions, interactions
from lib.environments.ecosystem_env.constants import REST_INDEX, EAT_START
from lib.world import daylight
from lib.world.tick_time import ticks_per_day
torch.set_num_threads(3)
arm = os.environ['ARM']
cfg = json.load(open('results/' + os.environ['RUN'] + '/progress/config.json')); seed = cfg['seed']
keys = ("project", "library", "grid", "mortality", "migration", "mortality_multiplier", "mass_balance", "currents", "tick_hours")
eco = {k: cfg[k] for k in keys if k in cfg}
with evaluation_randomness(seed): env = build_inference_env(eco, seed)
with evaluation_randomness(seed):
    pol, om, ov = load_policies_and_stats(env, os.environ['POL'], verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = om, ov
    env.softmax_temperature = cfg.get('temperature', 1.0)
AREA = float(env.H * env.W)
A17_ZOO = [0.21, 0.18, 1.56, 19.3, 26.0, 11.4, 6.9, 4.6, 3.7, 2.9, 3.5, 0.92]
z = list(env.dm_ids).index('zooplankton')
tick_rest = {}
orig = decisions.actions_from_probabilities


def patched(env_, probs, subthreshold):
    b = env_.fgs['zooplankton'].biomass
    tot = float(b.sum())
    rest = probs[z, REST_INDEX]
    tick_rest['rest'] = float((rest * b).sum() / tot) if tot > 0 else np.nan
    if arm == 'eat':
        probs = probs.copy()
        eat = probs[z, EAT_START:EAT_START + env_.N_all]
        s = eat.sum(axis=0)
        ok = s > 1e-12
        scale = np.where(ok, (s + rest) / np.where(ok, s, 1.0), 1.0)
        probs[z, EAT_START:EAT_START + env_.N_all] = eat * scale[None]
        probs[z, REST_INDEX] = np.where(ok, 0.0, rest)
    return orig(env_, probs, subthreshold)


decisions.actions_from_probabilities = patched
plank = [f for f in ('phytoplankton', 'subsurface_phytoplankton', 'microzooplankton', 'zooplankton', 'krill') if f in env.fgs]
fish = [f for f in ('planktivorous_fish', 'mackerel', 'whiting', 'predatory_fish', 'benthivorous_fish') if f in env.fgs]
start = {f: float(env.fgs[f].biomass.sum()) for f in fish}
print('arm', arm, '| start totals t/km2:', {f: round(start[f] / AREA, 3) for f in fish}, flush=True)
tpd = ticks_per_day(float(env.daylight['tick_hours'])); tpy = int(round(365 * tpd))
out = []
with evaluation_randomness(seed):
    for y in range(years):
        dens = np.zeros((12, len(plank))); cnt = np.zeros(12)
        rest_all = np.zeros(12); rest_n = np.zeros(12); rest_d = np.zeros(12); n_n = np.zeros(12); n_d = np.zeros(12)
        fishB = np.zeros(len(fish))
        book0 = {p: dict(r) for p, r in env.intake_by_pred_prey.items()}
        for k in range(tpy):
            tick = interactions.light_index(env)
            month = daylight.month_of_day(int(tick / tpd))
            light = float(interactions.light_level(env)) if env._has_daylight else 1.0
            env.tick()
            r = tick_rest.get('rest', np.nan)
            if np.isfinite(r):
                rest_all[month] += r
                if light > 0.5: rest_d[month] += r; n_d[month] += 1
                elif light < 0.01: rest_n[month] += r; n_n[month] += 1
            for i, f in enumerate(plank): dens[month, i] += float(env.fgs[f].biomass.sum()) / AREA
            cnt[month] += 1
            for i, f in enumerate(fish): fishB[i] += float(env.fgs[f].biomass.sum())
        c = np.maximum(cnt, 1)
        eaten = {}
        for p, row in env.intake_by_pred_prey.items():
            for q, v in row.items():
                d = v - book0.get(p, {}).get(q, 0.0)
                if d > 0: eaten.setdefault(p, {})[q] = d
        rec = dict(year=y + 1,
                   monthly={f: (dens[:, i] / c).tolist() for i, f in enumerate(plank)},
                   rest=(rest_all / c).tolist(), rest_night=(rest_n / np.maximum(n_n, 1)).tolist(),
                   rest_day=(rest_d / np.maximum(n_d, 1)).tolist(),
                   eaten=eaten, fish_rel={f: float(fishB[i] / tpy / max(start[f], 1e-12)) for i, f in enumerate(fish)})
        out.append(rec)
        fm = lambda v, p=2: ' '.join(f'{x:6.{p}f}' for x in v)
        print(f"year {y+1} ({arm}) month:       " + ' '.join(f'{m:6d}' for m in range(1, 13)))
        for f in plank: print(f"  {f[:22]:22s} {fm(rec['monthly'][f])}")
        print(f"  {'A17 zoo resident':22s} {fm(A17_ZOO)}")
        print(f"  {'zoo rest share':22s} {fm(rec['rest'])}")
        print(f"  {'  night':22s} {fm(rec['rest_night'])}")
        print(f"  {'  day':22s} {fm(rec['rest_day'])}")
        print('  fish mean B / start:', {f: round(v, 2) for f, v in rec['fish_rel'].items()})
        for p in ['zooplankton'] + fish:
            row = eaten.get(p, {})
            print(f"  eaten by {p[:20]:20s} t/km2/yr:", {q: round(v / AREA, 3) for q, v in sorted(row.items(), key=lambda kv: -kv[1])[:6]}, flush=True)
os.makedirs(f"{S}/../tables", exist_ok=True)
json.dump(out, open(f"{S}/../tables/zoo_eat_only_{os.environ['RUN']}_{arm}.json", 'w'))
