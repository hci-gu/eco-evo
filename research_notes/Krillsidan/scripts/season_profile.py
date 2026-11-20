"""Seasonal profile of the krill factor: (field krill intake by quarter) / (chain a_eff(t) x D_lib(t)), each
normalised to its annual mean, per krill predator. The LEVEL cancels (and with it the unknown D level); what is
left is whether the factor needs to vary over the year, and in which quarter.
Chain side (policy-free; the probe env's own tables): a_eff(t) = library a x light_mult_table(t) x ingestion
q(t) (equal-cell depth factor constant over the year); D_lib(t) = 5 t/km2 x krill outside_season(t).
Field side [F/D/J, sources in ../evidence.md]: krill share of the stomach mass by quarter x a quarterly ration
index (1 = flat; the seasonal ration evidence is stated per row).
Usage: python3 season_profile.py
"""
import os, sys
import numpy as np
ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT); os.chdir(ROOT)
import train as train_mod
from lib.environments.ecosystem_env import interactions
env = train_mod._ProbeEnvBuilder(project_path='mareld2.yaml', grid_size=(16, 16), apply_natural_mortality=True,
                                 migration=True)()
env._build_static_caches()
order = list(env.global_fg_order); jk = order.index('krill')
season = np.array(env.fgs['krill'].params['outside_season'], dtype=float)
P = env.light_period
qidx = [np.arange(P)[(np.arange(P) * 4 // P) == k] for k in range(4)]
lm = np.asarray(env.light_mult_table, dtype=float) if env._has_light_pairs else None
qt = np.asarray(env.ingestion_temp_table, dtype=float) if getattr(env, '_has_ingestion_temperature', False) else None
month_of_tick = np.minimum(11, (np.arange(P) * 12) // P)
D = 5.0 * season[month_of_tick]
# field: krill share by quarter (Q1..Q4) and a ration index by quarter
FIELD = {
    'planktivorous_fish': [
        ("pout, Albert 1994 (winter = Feb-Apr; summer = Jul-Sep), ration flat", [0.80, 0.60, 0.52, 0.66], [1, 1, 1, 1]),
        ("pout, same, stomach content higher in winter (index 1.5 / 1) [J on F Fig. 10]", [0.80, 0.60, 0.52, 0.66],
         [1.5, 1.0, 1.0, 1.25]),
    ],
    'whiting': [
        ("whiting in-area/Oslofjord: Q1 Skag 1981 0.14 & Oslof Mar 0.94 (gm 0.36), Q2 Skag Apr 1991 0.70 & "
         "Oslof Jun 0.29 (gm 0.45), Q3 none (NS 0.01 of ceiling; Skag 2022 ~0) -> 0.05 [J], Q4 Oslof Nov 0.41;"
         " %BW/d directly", [0.36, 0.45, 0.05, 0.41], [1, 1, 1, 1]),
        ("whiting, Oslofjord only (Mar 0.94, Jun 0.29, Q3 = Jun, Nov 0.41)", [0.94, 0.29, 0.29, 0.41], [1, 1, 1, 1]),
    ],
    'predatory_fish': [
        ("saithe CRR 219 1991, ages 3-5 krill %W", [0.053, 0.387, 0.047, 0.004], [1, 1, 1, 1]),
        ("saithe CRR 164 1981, 400-699 mm, Q1+Q4 / Q2+Q3", [0.49, 0.41, 0.41, 0.49], [1, 1, 1, 1]),
    ],
    'mackerel': [
        ("mackerel CRR 219 1991 (Q1 ages 1-2 only)", [0.757, 0.047, 0.218, 0.004], [1, 1, 1, 1]),
        ("mackerel CRR 164 1981 ages 3-7", [0.824, 0.19, 0.198, 0.431], [1, 1, 1, 1]),
    ],
}
print("factor profile = field krill intake index / chain a_eff x D_lib, both over their annual mean; Q1..Q4")
for g, rows in FIELD.items():
    i = env.dm_ids.index(g)
    m = lm[:, i, jk] if lm is not None else np.ones(P)
    q = qt[:, i] if qt is not None else np.ones(P)
    chain = m * q * D
    cq = np.array([chain[ix].mean() for ix in qidx]); cq /= cq.mean()
    print(f"\n{g}: chain a_eff x D_lib by quarter (mean 1): " + ' '.join(f'{x:5.2f}' for x in cq)
          + f"   [light {' '.join(f'{m[ix].mean():.2f}' for ix in qidx)}; q {' '.join(f'{q[ix].mean():.2f}' for ix in qidx)}; "
            f"krill season {' '.join(f'{season[3 * k:3 * k + 3].mean():.2f}' for k in range(4))}]")
    for lab, share, rat in rows:
        fi = np.array(share) * np.array(rat); fi = fi / fi.mean()
        f = fi / cq; f = f / f.mean()
        print(f"  {lab}\n    field index {' '.join(f'{x:5.2f}' for x in fi)}; factor profile "
              + ' '.join(f'{x:5.2f}' for x in f) + f"; max/min {f.max() / max(f.min(), 1e-9):.1f}")
