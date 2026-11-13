"""Starvation events of porpoise cells: the cell's state when s < u."""
import json, os, sys
import numpy as np
ROOT = '/home/simon/eco-evo'; sys.path.insert(0, ROOT); os.chdir(ROOT)
from inference import load_policies_and_stats
from lib.environments.ecosystem_env import population_change, movement
from lib.runners import training_progress as tp
from lib.world import daylight
RUN, LIB = os.environ['RUN'], os.environ['LIB']; TICKS = int(os.environ.get('TICKS', 1460)); G = 'porpoises'
config = json.load(open(os.path.join(RUN, 'progress', 'config.json'))); config['library'] = LIB
seed = int(config['seed'])
with tp.evaluation_randomness(seed):
    env = tp.build_inference_env(config, seed)
with tp.evaluation_randomness(seed):
    policies, mean, var = load_policies_and_stats(env, RUN, verbose=False)
    env.policies = policies; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = mean, var
    env.softmax_temperature = float(config.get('temperature', 1.0))
    fg = env.fgs[G]; u = float(fg.maintenance_level); me = float(fg.max_energy_reserve)
    st = {'prevB': fg.biomass.astype(np.float64).copy(), 'prevS': None, 'preMoveB': None, 'preMoveS': None, 't': 0}
    _mv = movement.apply_movement
    def mv(e, s):
        b = e.fgs[G].biomass.astype(np.float64); r = e.fgs[G].energy_reserve.astype(np.float64)
        st['preMoveB'] = b.copy(); st['preMoveS'] = np.where(b > 0, r / np.maximum(b * me, 1e-30), 0)
        return _mv(e, s)
    movement.apply_movement = mv
    _pc = population_change.apply_population_change
    events = []
    def pc(e, **kw):
        b = e.fgs[G].biomass.astype(np.float64); r = e.fgs[G].energy_reserve.astype(np.float64)
        s = np.where(b > 0, r / np.maximum(b * me, 1e-30), 1.0)
        for (y, x) in zip(*np.where((b > 0) & (s < u))):
            loss = b[y, x] * fg.starve_rate * (u - s[y, x])
            events.append(dict(t=st['t'], y=int(y), x=int(x), B=b[y, x], s=s[y, x], loss=loss,
                               preB=st['preMoveB'][y, x], preS=st['preMoveS'][y, x], prevB=st['prevB'][y, x],
                               plank=float(e.fgs['planktivorous_fish'].biomass[y, x]),
                               benth=float(e.fgs['benthivorous_fish'].biomass[y, x]),
                               whit=float(e.fgs['whiting'].biomass[y, x])))
        return _pc(e, **kw)
    population_change.apply_population_change = pc
    for t in range(1, TICKS + 1):
        st['t'] = t
        env.step()
        st['prevB'] = fg.biomass.astype(np.float64).copy()
    tot = sum(ev['loss'] for ev in events)
    print('events', len(events), 'loss (approx, t)', round(tot, 3), 'u', u, 'starve_rate', fg.starve_rate)
    # aggregate by cell episodes
    from collections import defaultdict
    by = defaultdict(list)
    for ev in events: by[(ev['y'], ev['x'])].append(ev)
    rows = sorted(by.items(), key=lambda kv: -sum(e['loss'] for e in kv[1]))[:15]
    for (y, x), evs in rows:
        ts = [e['t'] for e in evs]
        L = sum(e['loss'] for e in evs)
        first = evs[0]
        print(f"cell ({y:2d},{x:2d}) ticks {min(ts)}-{max(ts)} n {len(evs):4d} loss {L:.3f} | first: B {first['B']:.3f} s {first['s']:.2f} "
              f"preMove B {first['preB']:.3f} s {first['preS']:.2f} | plank {first['plank']:.2f} benth {first['benth']:.2f} whit {first['whit']:.2f} "
              f"| mean s {np.mean([e['s'] for e in evs]):.2f} mean plank {np.mean([e['plank'] for e in evs]):.2f} benth {np.mean([e['benth'] for e in evs]):.2f} whit {np.mean([e['whit'] for e in evs]):.2f}")
    # grid mean prey for reference
    for p in ['planktivorous_fish', 'benthivorous_fish', 'whiting']:
        print(p, 'grid mean t/cell', round(float(env.fgs[p].biomass.mean()), 3))
