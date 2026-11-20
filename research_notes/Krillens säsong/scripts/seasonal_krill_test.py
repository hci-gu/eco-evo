"""TEST ONLY - sizing a seasonal krill availability in the frozen world. Nothing here is an engine change: the
patches live in this process and wrap research_notes/"Mättnad hos rovfisk"/model/scripts/fill_decomp_all.py.

Arms (combine with commas):
  shallow  the pool's presence over beds < 100 m follows the field half-years instead of the library's annual 0.30
           of the >= 250 m level: Sep-Apr 0.43, May-Aug 0.05 (resume 262, research_notes/"Krill over grunda
           bottnar"/rederive: Bohle & Moksness 1991 winter nights; 0 of 14 near-bed day tows Jun-Jul) -> monthly
           multiplier 0.43/0.303 or 0.05/0.303 on the open pool's target in those cells, the target renormalised
           each month to the library's box total (the krill move to the deeper beds; the stock stays the season's).
  energy   krill energy by season: ~6 kJ/g WW in Q4-Q1, ~4.3 in Q2-Q3 (Buchholz & Prado-Fiedler 1987; Nowicki
           2023), normalised to the library's annual 5000 (x1.165 / x0.835). Applied as the column of
           env.energy_gain_mat AND env.handling_time_mat (h = E / ceiling), so energy per tonne and handling time
           per tonne both follow E. fill_decomp_all.py reads h once, so its unc / feed for these arms are on the
           reference E for the krill share; the fill is exact.
  flat     the pool's quarterly stock season 1.0 / 0.6 / 1.1 / 1.3 replaced by 1.0 all year (the Q2 trough is a
           judgement; field stomachs have Q2 pulses).
Usage: seasonal_krill_test.py ARMS POLICY_DIR SEED LIBRARY
"""
import sys, runpy, json
import numpy as np
import yaml
sys.path.insert(0, '/home/simon/eco-evo')
from lib.environments.ecosystem_env import open_pool, interactions
from lib.environments.ecosystem_env import environment as env_mod
from lib.world import habitat

arms = set(sys.argv[1].split(','))
D, seed, LIB = sys.argv[2], sys.argv[3], sys.argv[4]
cfg = json.load(open(D + '/config.json'))
project = yaml.safe_load(open(cfg['project'], encoding='utf-8-sig'))
H, W = cfg['grid']
depth = habitat.load_depth_map(project, cfg['project'], (H, W))
shallow = depth < 100.0
PRES = np.array([0.43] * 4 + [0.05] * 4 + [0.43] * 4)          # Jan-Apr, May-Aug, Sep-Dec
ANNUAL = PRES.mean()
E_Q = np.array([6.0] * 3 + [4.3] * 6 + [6.0] * 3)
E_M = E_Q / E_Q.mean()
SEASON_LIB = np.array([1.0] * 3 + [0.6] * 3 + [1.1] * 3 + [1.3] * 3)
print(f"TEST arms {sorted(arms)}; shallow cells {shallow.mean():.3f}; presence mult Sep-Apr "
      f"{0.43 / ANNUAL:.3f}, May-Aug {0.05 / ANNUAL:.3f}; energy mult {E_M[0]:.3f} / {E_M[5]:.3f}", flush=True)


def month(env):
    return min(11, int(interactions.light_index(env) / env.light_period * 12))


o_target = open_pool.target
_cache = {}


def target(env, fid):
    t = o_target(env, fid)
    if fid != 'krill':
        return t
    m = month(env)
    if 'flat' in arms and env.open_season.get(fid) is not None:
        t = t / env.open_season[fid][interactions.light_index(env)]
    if 'shallow' in arms:
        key = (id(env), m)
        if key not in _cache:
            base = env.open_target[fid].astype(np.float64)
            mult = np.where(shallow, PRES[m] / ANNUAL, 1.0)
            w = base * mult
            _cache[key] = (w * base.sum() / w.sum() / np.maximum(base, 1e-30)).astype(env.dtype)
        t = t * _cache[key]
    return t


open_pool.target = target

if 'energy' in arms:
    o_step = env_mod.EcosystemEnvironment.step

    def step(self, *a, **k):
        j = self.global_fg_order.index('krill')
        if not hasattr(self, '_krill_E0'):
            self._krill_E0 = (np.array(self.energy_gain_mat[:, j], copy=True),
                              np.array(self.handling_time_mat[:, j], copy=True))
        g0, h0 = self._krill_E0
        f = E_M[month(self)]
        self.energy_gain_mat[:, j] = (g0 * f).astype(self.energy_gain_mat.dtype)
        self.handling_time_mat[:, j] = (h0 * f).astype(self.handling_time_mat.dtype)
        return o_step(self, *a, **k)

    env_mod.EcosystemEnvironment.step = step

sys.argv = ['fill_decomp_all.py', D, seed, LIB]
runpy.run_path('/home/simon/eco-evo/research_notes/Mättnad hos rovfisk/model/scripts/fill_decomp_all.py',
               run_name='__main__')
