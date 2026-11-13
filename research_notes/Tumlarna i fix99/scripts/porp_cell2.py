import json, os, sys
import numpy as np
ROOT = '/home/simon/eco-evo'; sys.path.insert(0, ROOT); os.chdir(ROOT)
from inference import load_policies_and_stats
from lib.environments.ecosystem_env import movement, decisions
from lib.runners import training_progress as tp
RUN, LIB = os.environ['RUN'], os.environ['LIB']; Y, X = int(os.environ['Y']), int(os.environ['X'])
T0, T1 = int(os.environ['T0']), int(os.environ['T1']); G = 'porpoises'
config = json.load(open(os.path.join(RUN, 'progress', 'config.json'))); config['library'] = LIB
seed = int(config['seed'])
with tp.evaluation_randomness(seed):
    env = tp.build_inference_env(config, seed)
with tp.evaluation_randomness(seed):
    policies, mean, var = load_policies_and_stats(env, RUN, verbose=False)
    env.policies = policies; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = mean, var
    env.softmax_temperature = float(config.get('temperature', 1.0))
    i = env.dm_ids.index(G); fg = env.fgs[G]; me = fg.max_energy_reserve
    st = {}
    _c = movement.apply_energy_costs
    def hook(e, a):
        st['eat'] = float(a.eat[i][:, Y, X].sum()); st['rest'] = float(a.rest[i][Y, X])
        st['move'] = [round(float(v), 3) for v in a.move[i][:, Y, X]]
        return _c(e, a)
    movement.apply_energy_costs = hook
    _m = decisions.merge_subunit_fractions
    def mhook(e, probs):
        st['raw'] = np.round(probs[i, :, Y, X].astype(float), 3).tolist()
        out = _m(e, probs)
        st['merged'] = np.round(out[i, :, Y, X].astype(float), 3).tolist()
        return out
    decisions.merge_subunit_fractions = mhook
    def win(name):
        b = env.fgs[name].biomass
        return np.round(b[max(Y-2,0):Y+3, max(X-2,0):X+3], 2)
    for t in range(1, T1 + 1):
        env.step()
        if t >= T0 and (t - T0) % 4 == 0:
            b = float(fg.biomass[Y, X]); s = float(fg.energy_reserve[Y, X]) / max(b * me, 1e-30)
            print('   raw', st.get('raw'), '\n   merged', st.get('merged')); print(f"t{t} B {b:.3f} s {s:.2f} eat {st['eat']:.3f} rest {st['rest']:.3f} move NSEW? {st['move']}")
            if False:
                for p in ['planktivorous_fish', 'benthivorous_fish', 'whiting', 'porpoises']:
                    print(' ', p); print(win(p))
