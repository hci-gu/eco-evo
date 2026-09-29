"""Same closed phyto+zoo scenario as tests/test_phyto_zoo_sanity.py,
run with the legacy growth term and with --mass-balance, so the
failure can be attributed."""
import sys, numpy as np
sys.path.insert(0, '.')
from lib.config.config_loader import setup_full_mareld_mvp
from lib.environments.ecosystem import EcosystemEnvironment

GRID = (30, 30)
CFG = {'width': 30, 'height': 30, 'cell_size': 1000.0, 'tick_duration': 6.0}

def run(mass_balance, seed=0, ticks=200, rm=None, greedy=False):
    all_fgs = setup_full_mareld_mvp(grid_size=GRID, seed=seed, spawn_seed=seed)
    fgs = {k: v for k, v in all_fgs.items() if k in ('phytoplankton', 'zooplankton')}
    if rm is not None:
        fgs['zooplankton'].resting_metabolism = float(rm)
        fgs['zooplankton'].params['resting_metabolism'] = float(rm)
    env = EcosystemEnvironment(CFG, fgs, {}, mass_balance=mass_balance)
    for iid in ('djup', 'windfarm_noise', 'bottom_trawling',
                'pelagic_trawling', 'rotor'):
        env.grid.add_map(iid, np.zeros(GRID, dtype=np.float32))
    prov = None
    if greedy:
        from lib.diagnostics import viability
        prov = viability.install_behaviour(env, 'greedy', seed=seed)
    z0 = float(env.fgs['zooplankton'].biomass.sum())
    p0 = float(env.fgs['phytoplankton'].biomass.sum())
    for _ in range(ticks):
        env.step(prov(env) if prov is not None else None)
    z1 = float(env.fgs['zooplankton'].biomass.sum())
    p1 = float(env.fgs['phytoplankton'].biomass.sum())
    return z0, z1, p0, p1

print(f"{'rm':>7} {'mass_bal':>9} {'behaviour':>10} {'zoo ratio':>10}"
      f" {'phyto ratio':>12}  verdict")
for rm in (3.5, 11.25, 22.5, 56.2):
    for mb in (False, True):
        for gr in (False, True):
            z0, z1, p0, p1 = run(mb, rm=rm, greedy=gr)
            print(f"{rm:>7} {str(mb):>9} {'greedy' if gr else 'default':>10}"
                  f" {z1/z0:>10.4f} {p1/p0:>12.3f}"
                  f"  {'PASS' if 0.03 <= z1/z0 <= 100 else 'FAIL'}")
