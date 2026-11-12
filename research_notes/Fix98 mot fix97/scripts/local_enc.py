"""Where does the benthivores' realised krill share come from, if the diet gate at their own bed
spread reads x1.3? Runs a run's frozen policies in its probe world and, per probe year, books:

- co-location: each prey's density where the benthivores are (benthivore-biomass weighted) over its
  grid mean;
- the encounter shares a * depth mult * B_vis summed over cells and ticks, without the light
  lever, (L) at the realised local prey densities and (G) at each prey's grid mean in every cell
  (the benthivores where they are);
- the realised diet shares from the engine's intake book.

B_vis = biomass x the pair's access (stage access at the start J share; an NDM with a refuge its
exposed part, kept in the reserve slot). RUN, LIB, YEARS from the environment."""
import json
import os
import sys

import numpy as np

ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from inference import load_policies_and_stats  # noqa: E402
from lib.runners import training_progress as tp  # noqa: E402
from lib.environments.ecosystem_env import interactions, predation  # noqa: E402

RUN, LIB = os.environ['RUN'], os.environ['LIB']
YEARS = int(os.environ.get('YEARS', 3))
P = 'benthivorous_fish'
config = json.load(open(os.path.join(RUN, 'progress', 'config.json')))
config['library'] = LIB
seed = int(config['seed'])
with tp.evaluation_randomness(seed):
    env = tp.build_inference_env(config, seed)
with tp.evaluation_randomness(seed):
    policies, mean, var = load_policies_and_stats(env, RUN, verbose=False)
    env.policies = policies
    env.rebuild_batched_weights()
    env.obs_mean, env.obs_var = mean, var
    env.softmax_temperature = float(config.get('temperature', 1.0))

    i = env.dm_ids.index(P)
    menu = [j for j in range(env.N_all) if env.eat_static_mask[i, j]]
    names = [env.global_fg_order[j] for j in menu]
    stage_rows = set(env.stage_rows.tolist()) if getattr(env, '_has_stages', False) else set()

    def vis(j, name):
        fg = env.fgs[name]
        b = np.asarray(fg.biomass, dtype=np.float64)
        if getattr(fg, 'exposed_share', None) is not None and not fg.is_decision_maker:
            return np.asarray(fg.energy_reserve, dtype=np.float64)  # exposed part A
        if j in stage_rows:
            js = float(env.stage_start[:, j].mean())
            return b * (js * float(env.stage_f_j[i, j]) + (1 - js) * float(env.stage_f_a[i, j]))
        return b * float(env.access_mat[i, j])

    mult = {j: (np.asarray(env.pair_depth_mult[i, j], dtype=np.float64)
                if getattr(env, '_has_pair_depth', False) else 1.0) for j in menu}
    a = {j: float(env.max_intake_mat[i, j]) for j in menu}
    ty = 1460
    coloc = np.zeros((YEARS, len(menu)))
    encL = np.zeros((YEARS, len(menu)))
    encG = np.zeros((YEARS, len(menu)))
    eaten = np.zeros((YEARS, len(menu)))
    encLm = np.zeros((YEARS, len(menu)))
    dem_pre = np.zeros((YEARS, len(menu)))
    dem_post = np.zeros((YEARS, len(menu)))
    encH = np.zeros((YEARS, len(menu)))
    conc = np.zeros(YEARS)
    hj = {j: float(env.handling_time_mat[i, j]) for j in menu}
    cur = {'y': 0}
    _cap = predation._reserve_room_cap

    def cap_hook(e, demand, *args, **kw):
        out = _cap(e, demand, *args, **kw)
        y = cur['y']
        lm = (e.light_mult_table[interactions.light_index(e)]
              if e._has_light_pairs else np.ones_like(e.max_intake_mat))
        bp = np.asarray(e.fgs[P].biomass, dtype=np.float64)
        lams = []
        for k, (j, name) in enumerate(zip(menu, names)):
            v = vis(j, name).reshape(bp.shape)
            m = mult[j] if np.isscalar(mult[j]) else mult[j].reshape(bp.shape)
            lams.append(a[j] * float(lm[i, j]) * m * v)
        busy = sum(hj[j] * l for j, l in zip(menu, lams))
        srt = np.sort(bp.reshape(-1))[::-1]
        conc[y] += float(np.searchsorted(np.cumsum(srt), 0.5 * srt.sum()) + 1) / srt.size
        for k, (j, name) in enumerate(zip(menu, names)):
            v = vis(j, name).reshape(bp.shape)
            m = mult[j] if np.isscalar(mult[j]) else mult[j].reshape(bp.shape)
            encH[y, k] += (bp * lams[k] / (1.0 + busy)).sum()
            encLm[y, k] += (bp * a[j] * float(lm[i, j]) * m * v).sum()
            dem_pre[y, k] += float(np.asarray(demand[i, j], dtype=np.float64).sum())
            dem_post[y, k] += float(np.asarray(out[i, j], dtype=np.float64).sum())
        return out

    predation._reserve_room_cap = cap_hook
    prev = dict(env.intake_by_pred_prey.get(P, {}))
    for t in range(YEARS * ty):
        y = t // ty
        cur['y'] = y
        env.step()
        bp = np.asarray(env.fgs[P].biomass, dtype=np.float64)
        bp = bp.reshape(-1) if bp.ndim == 1 else bp
        wsum = bp.sum()
        now = dict(env.intake_by_pred_prey.get(P, {}))
        for k, (j, name) in enumerate(zip(menu, names)):
            v = vis(j, name).reshape(bp.shape)
            m = mult[j] if np.isscalar(mult[j]) else mult[j].reshape(bp.shape)
            coloc[y, k] += (bp * v).sum() / max(wsum, 1e-30) / max(v.mean(), 1e-30)
            encL[y, k] += (bp * a[j] * m * v).sum()
            encG[y, k] += (bp * a[j] * m * v.mean()).sum()
            eaten[y, k] += now.get(name, 0.0) - prev.get(name, 0.0)
        prev = now
    for y in range(YEARS):
        print(f"year {y + 1}: cells holding half the benthivores {conc[y] / ty:.3f} of the grid")
        print(f"  {'prey':>20} {'coloc':>7} {'enc(G)':>7} {'enc(L)':>7} {'+light':>7} {'+satur':>7} {'demand':>7} {'roomcap':>7} {'real':>7}")
        for k, name in enumerate(names):
            print(f"  {name:>20} {coloc[y, k] / ty:7.2f} {encG[y, k] / encG[y].sum():7.3f} "
                  f"{encL[y, k] / encL[y].sum():7.3f} {encLm[y, k] / encLm[y].sum():7.3f} {encH[y, k] / encH[y].sum():7.3f} "
                  f"{dem_pre[y, k] / dem_pre[y].sum():7.3f} {dem_post[y, k] / dem_post[y].sum():7.3f} "
                  f"{eaten[y, k] / eaten[y].sum():7.3f}")
