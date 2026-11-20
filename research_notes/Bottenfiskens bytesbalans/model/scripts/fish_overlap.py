"""Where the benthivores and their fish prey stand: bed-class shares and overlap (frozen world, 1 year, daily).
O_cell = n sum(B_be B_p) / (sum B_be sum B_p) (1 = no association); O_class = the same from class shares alone
(each group even within its class); field O_class from GOV shares where they exist.
Usage: fish_overlap.py POLICY_DIR SEED LIBRARY"""
import sys, json, numpy as np, yaml
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.world import habitat
from lib.diagnostics.field_diet import bed_class, FIELD_BED_SHARES, FIELD_STAGE_BED_SHARES
import inference
D, seed, LIB = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
man = yaml.safe_load(open(cfg['project']))
cls = bed_class(habitat.load_depth_map(man, cfg['project'], (env.H, env.W)))
f = np.array([(cls == k).mean() for k in range(4)])
G = ['benthivorous_fish', 'planktivorous_fish', 'whiting', 'predatory_fish', 'krill', 'benthic_infauna']
sh = {g: [] for g in G}; shJ = []; oc = {g: [] for g in G}; ocl = {g: [] for g in G}
with evaluation_randomness(seed * 13 + 5):
    for t in range(365 * 4):
        env.step()
        if t % 4: continue
        Bb = env.fgs['benthivorous_fish'].biomass.astype(np.float64)
        Jb = env.fgs['benthivorous_fish'].juvenile.astype(np.float64)
        sb = np.array([Bb[cls == k].sum() for k in range(4)]) / Bb.sum()
        shJ.append(np.array([Jb[cls == k].sum() for k in range(4)]) / max(Jb.sum(), 1e-30))
        for g in G:
            x = env.fgs[g].biomass.astype(np.float64)
            s = np.array([x[cls == k].sum() for k in range(4)]) / x.sum(); sh[g].append(s)
            oc[g].append(x.size * (Bb * x).sum() / (Bb.sum() * x.sum()))
            ocl[g].append(float((sb * s / f).sum()))
fmt = lambda s: '/'.join(f'{x:.2f}' for x in s)
print(f"seed {seed}; cells per bed class (<100 / 100-150 / 150-250 / >=250 m) {fmt(f)}")
for g in G:
    print(f"  {g:20s} class shares {fmt(np.mean(sh[g], 0))}; overlap with benthivores: cell {np.mean(oc[g]):.2f}, "
          f"class-only {np.mean(ocl[g]):.2f}")
print(f"  benthivore J           class shares {fmt(np.mean(shJ, 0))}")
fb = np.array(FIELD_STAGE_BED_SHARES['benthivorous_fish'][0]); fw = np.array(FIELD_BED_SHARES['whiting'][0])
fs = np.array(FIELD_BED_SHARES['predatory_fish'][0])
print(f"field (GOV): benthivore adults {fmt(fb)}, whiting {fmt(fw)}, saithe {fmt(fs)}; class-only overlap "
      f"benthivore adults x whiting {(fb * fw / f).sum():.2f}, x saithe {(fb * fs / f).sum():.2f}")
