"""Multi-year run with frozen policies in the probe world (resume section 219).

    python3 tools/probes/frozen_years.py run CKPT_DIR SEED YEARS OUT.json [LIBRARY]
    python3 tools/probes/frozen_years.py show OUT.json [OUT.json ...]

``run`` builds the probe world (mareld2.yaml on 60x60, M1 on, migration,
currents; the spawn seed is SEED) and lets the policies in CKPT_DIR act
for YEARS years of 1460 ticks. Per year it records the start, end and
mean biomass of every group, the booked losses (predation, starvation,
natural, impact), the intake per predator-prey pair (biomass eaten) and
the mean reserve fill of every DM. LIBRARY defaults to
fgconfig/fg_library.yaml. The JSON is rewritten after each year, so a
partial run can be read. (Section 220.7 metered the eat share against
predator_share here; the cap was removed in section 221.)

``show`` prints per seed: end/b0 per year, mean density (t/km2), losses
per mean biomass (/yr) in years 1, mid and last, the diet shares and
Q/B, the reserve fill per year, the realised M2 (intake by each
predator over the prey's mean biomass) to set against the library's
predation_mortality. With juvenile compartments (stages.py) also the
J/B of every staged group by calendar month and the juvenile part of each
pair's intake.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

TPY = 1460
CELLS = 3600.0
LOSS = ('loss_predation', 'loss_starvation', 'loss_natural', 'loss_impact')
AB = {'benthivorous_fish': 'ben', 'predatory_fish': 'pred', 'whiting': 'wht', 'mackerel': 'mack', 'planktivorous_fish': 'plk',
      'porpoises': 'porp', 'seals': 'seal', 'benthic_infauna': 'bent',
      'mobile_epifauna': 'epi', 'norway_lobster': 'neph',
      'phytoplankton': 'phy', 'subsurface_phytoplankton': 'sub',
      'microzooplankton': 'mic', 'zooplankton': 'zoo'}


def run(ckpt, seed, years, out_path, library=None):
    import numpy as np
    import torch
    import train as train_mod
    from inference import load_policies_and_stats
    from lib.environments.ecosystem_env.currents import CurrentConfig

    class Builder(train_mod._ProbeEnvBuilder):
        PROBE_SEED = seed

    env = Builder(project_path='mareld2.yaml', grid_size=(60, 60), apply_natural_mortality=True,
                  migration=True, currents=CurrentConfig(0.1, 20, 0, 12.0),
                  library_path=library)()
    env.policies, env.obs_mean, env.obs_var = load_policies_and_stats(env, ckpt, verbose=False)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dms = [f for f in env.fgs if f in env.policies]
    out = {"seed": seed, "ckpt": ckpt, "library": library, "years": []}
    for y in range(years):
        mean = {f: 0.0 for f in env.fgs}
        fill = {f: 0.0 for f in dms}
        start = {f: float(env.fgs[f].biomass.sum()) for f in env.fgs}
        l0 = {k: dict(getattr(env, k)) for k in LOSS}
        i0 = {p: dict(r) for p, r in env.intake_by_pred_prey.items()}
        j0 = {p: dict(r) for p, r in getattr(env, 'intake_juvenile_by_pred_prey', {}).items()}
        staged = [f for f, fg in env.fgs.items() if getattr(fg, 'juvenile', None) is not None]
        jb = {f: [[0.0, 0.0] for _ in range(12)] for f in staged}
        for t in range(TPY):
            for f, fg in env.fgs.items():
                mean[f] += float(fg.biomass.sum()) / TPY
            if staged:
                from lib.environments.ecosystem_env import interactions
                from lib.world.daylight import month_of_day
                month = month_of_day(interactions.light_index(env) * int(env.daylight['tick_hours']) // 24)
                for f in staged:
                    jb[f][month][0] += float(env.fgs[f].juvenile.sum())
                    jb[f][month][1] += float(env.fgs[f].biomass.sum())
            if t % 4 == 0:
                for f in dms:
                    fg = env.fgs[f]
                    B = float(fg.biomass.sum())
                    if B > 0:
                        fill[f] += (float(fg.energy_reserve.sum()) / (B * fg.max_energy_reserve)
                                    / (TPY / 4))
            env.step()
        loss = {k: {f: float(getattr(env, k).get(f, 0.0)) - float(l0[k].get(f, 0.0))
                    for f in env.fgs} for k in LOSS}
        intake = {p: {q: float(v) - float(i0.get(p, {}).get(q, 0.0)) for q, v in r.items()}
                  for p, r in env.intake_by_pred_prey.items()}
        juvenile_intake = {p: {q: float(v) - float(j0.get(p, {}).get(q, 0.0)) for q, v in r.items()}
                           for p, r in getattr(env, 'intake_juvenile_by_pred_prey', {}).items()}
        out["years"].append({"start": start, "mean": mean,
                             "end": {f: float(env.fgs[f].biomass.sum()) for f in env.fgs},
                             "loss": loss, "intake": intake, "fill": fill,
                             "juvenile_share_by_month": {
                                 f: [j / b if b > 0 else None for j, b in jb[f]] for f in staged},
                             "juvenile_intake": juvenile_intake})
        with open(out_path, 'w') as fh:
            json.dump(out, fh)
        print(f"year {y + 1} done", file=sys.stderr, flush=True)


def _per(x, m):
    """x per mean biomass m; nan once the group is extinct (m == 0)."""
    return x / m if m > 0 else float('nan')


def show(path):
    with open(path) as fh:
        d = json.load(fh)
    Y = d['years']
    b0 = Y[0]['start']
    gs = [g for g in AB if g in b0]
    pick = (Y[0], Y[len(Y) // 2], Y[-1])
    print('==', path, 'seed', d['seed'], len(Y), 'years')
    print('end/b0   yr ' + ' '.join(f"{AB[g]:>6}" for g in gs))
    for i, y in enumerate(Y):
        print(f"         {i + 1:2} " + ' '.join(f"{y['end'][g] / b0[g]:6.3f}" for g in gs))
    print('mean t/km2 yr ' + ' '.join(f"{AB[g]:>7}" for g in gs))
    for i, y in enumerate(Y):
        print(f"         {i + 1:2} " + ' '.join(f"{y['mean'][g] / CELLS:7.3f}" for g in gs))
    print('losses per mean biomass (/yr), pred/starv/nat/imp: year 1 | mid | last')
    for g in gs:
        row = ['/'.join(f"{_per(y['loss'][k][g], y['mean'][g]):.3f}" for k in LOSS) for y in pick]
        print(f"  {AB[g]:5}", ' | '.join(row))
    print('diet shares (% of intake) and Q/B: year 1 | mid | last')
    for p in [g for g in gs if g in Y[0]['fill']]:
        parts = []
        for y in pick:
            I = y['intake'].get(p, {})
            T = sum(I.values()) or 1.0
            parts.append(' '.join(f"{AB.get(q, q)}{v / T * 100:.0f}"
                                  for q, v in sorted(I.items(), key=lambda kv: -kv[1]))
                         + f" QB{_per(T, y['mean'][p]):.1f}")
        print(f"  {AB[p]:5}", ' | '.join(parts))
    print('fill per year:', {AB.get(f, f): [round(y['fill'][f], 2) for y in Y] for f in Y[0]['fill']})
    print('realised M2 (intake / prey mean biomass, /yr): year 1 | last')
    for q in gs:
        rows = []
        for y in (Y[0], Y[-1]):
            m = {AB.get(p, p): round(_per(I.get(q, 0.0), y['mean'][q]), 4)
                 for p, I in y['intake'].items() if I.get(q, 0.0) > 0}
            rows.append(m)
        if rows[0] or rows[1]:
            print(f"  {AB[q]:5}", rows[0], '|', rows[1])
    if Y[-1].get('juvenile_share_by_month'):
        print('juvenile compartments: J/B by calendar month, last year')
        for f, row in Y[-1]['juvenile_share_by_month'].items():
            print(f"  {AB.get(f, f):5}", ' '.join('  -  ' if v is None else f"{v:.3f}" for v in row))
        print('juvenile share of each pair\'s take (last year)')
        for p, I in Y[-1].get('juvenile_intake', {}).items():
            T = Y[-1]['intake'].get(p, {})
            print(f"  {AB.get(p, p):5}", {AB.get(q, q): round(v / T[q], 3) for q, v in I.items()
                                         if T.get(q, 0.0) > 0})


def main(argv):
    if len(argv) >= 5 and argv[0] == 'run':
        run(argv[1], int(argv[2]), int(argv[3]), argv[4],
            argv[5] if len(argv) > 5 else None)
    elif len(argv) >= 2 and argv[0] == 'show':
        for path in argv[1:]:
            show(path)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
