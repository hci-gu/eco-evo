"""Copy of the session scratchpad's wc_class_decomp.py (main agent, 2026-11-22) with ONE change: an optional library
copy passed to the probe builder (library_path). Policy-free decomposition of the water column's (WC) intake by bed class at the CURRENT library (after 280, 295,
296), per month: which pair's depth table (or the krill pool's depth-season layout) carries the shallow-bed gradient?
As research_notes/"Sejens profil over djupa bottnar"/scripts/class_diet.py (fixed attack, shared handling, T = 1,
no interference, prey at manifest start density x the budget_gate field-spread shape x visible share), plus: the
month's mean daylight multiplier per pair, the WC's ingestion q10 factor, the krill pool's month layout and season
(section 295) and krill's energy season on h. Usage: intake_by_class.py <grid n> [library.yaml or "live"] [predator id, default predatory_fish]

Research_notes/"Morkersyn i synparen" copy of research_notes/"Djupgradientens baerare"/scripts/intake_by_class.py
with ONE more change: the predator is an argument (whiting, benthivorous_fish), "live" = no library_path."""
import io, contextlib, os, sys
import numpy as np
ROOT = '/home/simon/eco-evo'; sys.path.insert(0, ROOT); _CWD = os.getcwd(); os.chdir(ROOT)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
LIBP = os.path.abspath(os.path.join(_CWD, sys.argv[2])) if len(sys.argv) > 2 and sys.argv[2] != 'live' else None
PRED = sys.argv[3] if len(sys.argv) > 3 else 'predatory_fish'
with contextlib.redirect_stdout(io.StringIO()):
    import train as train_mod, yaml
    from lib.diagnostics.field_diet import bed_class, field_prey_shape, field_stage_shapes
    from lib.world import habitat
    env = train_mod._ProbeEnvBuilder(project_path='mareld2.yaml', grid_size=(N, N), apply_natural_mortality=True,
                                     migration=True, library_path=LIBP)()
    env._build_static_caches()
man = yaml.safe_load(open('mareld2.yaml', encoding='utf-8-sig')); meta = man['project_metadata']
area = float(meta['reference_grid_width']) * float(meta['reference_grid_height'])
density = {e['group_id']: float(e['inference_initial_biomass']) / area for v in man.values() if isinstance(v, list)
           for e in v if isinstance(e, dict) and 'group_id' in e and e.get('inference_initial_biomass') is not None}
depth = habitat.load_depth_map(man, 'mareld2.yaml', (N, N)); cls = bed_class(depth)
area_sh = np.array([(cls == k).mean() for k in range(4)])
i = env.dm_ids.index(PRED); gi = env.global_fg_order.index(PRED)
menu = [j for j in range(env.N_all) if env.eat_static_mask[i, j]]
order = env.global_fg_order
kj = order.index('krill')

def visible_share(j, pid):
    if getattr(env, '_has_stages', False) and j in set(env.stage_rows.tolist()):
        js = float(env.stage_start[:, j].mean())
        return js * float(env.stage_f_j[i, j]) + (1 - js) * float(env.stage_f_a[i, j])
    f = float(env.access_mat[i, j]); phi = getattr(env.fgs[pid], 'exposed_share', None)
    return f * float(phi) if (phi is not None and not env.fgs[pid].is_decision_maker) else f

def prey_cells(j, pid):
    st = field_stage_shapes(pid, env.fgs[pid], depth)
    if st is not None and getattr(env, '_has_stages', False) and j in set(env.stage_rows.tolist()):
        js = float(env.stage_start[:, j].mean()); juv, ad = st
        return js * float(env.stage_f_j[i, j]) * juv + (1 - js) * float(env.stage_f_a[i, j]) * ad
    return field_prey_shape(pid, env.fgs[pid], depth) * visible_share(j, pid)

P = env.light_period
months = {'Jan': 15, 'Apr': 105, 'Jul': 196, 'Oct': 288}
def ticks(day):  # the month around this day
    return np.arange(int((day - 15) * 4), int((day + 15) * 4)) % P

def run(month_day, flat=None, krill_uniform=False):
    tk = ticks(month_day)
    q = float(env.ingestion_temp_table[tk, i].mean()) if getattr(env, '_has_ingestion_temperature', False) else 1.0
    E, Hh, names = [], [], []
    for j in menu:
        pid = order[j]
        if pid not in density and pid != 'krill':
            continue
        a = float(env.max_intake_mat[i, j]) * q
        if getattr(env, 'light_mult_table', None) is not None:
            a *= float(env.light_mult_table[tk, i, j].mean())
        m = np.asarray(env.pair_depth_mult[i, j], float) if getattr(env, '_has_pair_depth', False) else np.ones((N, N))
        if flat in (pid, 'all'):
            m = np.ones((N, N))
        h = float(env.handling_time_mat[i, j]) / q
        if pid == 'krill':
            mo = env.open_target_months['krill']                      # (12, H, W) t/km2? (density x month layout)
            w = env.open_month_weights[tk].mean(0)
            dens = (w[:, None, None] * mo).sum(0)
            if env.open_season.get('krill') is not None:
                dens = dens * float(env.open_season['krill'][tk].mean())
            if krill_uniform or flat == 'all':
                dens = np.full((N, N), dens.mean())
            bv = dens * visible_share(j, pid)
            if getattr(env, '_has_prey_energy', False):
                h *= float(env.prey_energy_table[tk, j].mean())
        else:
            bv = density[pid] * prey_cells(j, pid)
            if flat == 'all':
                bv = np.full((N, N), bv.mean())
        enc = np.array([float((a * m * bv)[cls == k].mean()) for k in range(4)])
        E.append(enc); Hh.append(h); names.append(pid)
    E, Hh = np.array(E), np.array(Hh)
    S = (E * Hh[:, None]).sum(0); I = S / (1 + S)
    return names, E, S, I

def g(I):
    lo = (I[0] * area_sh[0] + I[1] * area_sh[1]) / (area_sh[0] + area_sh[1])
    return lo / I[2], I[0] / I[2]

print(f'library {LIBP or "live"}; predator {PRED}'); print(f'{N}x{N}; area by class <100/100-150/150-250/>=250: ' + ' '.join(f'{x:.2f}' for x in area_sh))
print('pair depth tables (class mean of m):')
for j in menu:
    m = np.asarray(env.pair_depth_mult[i, j], float)
    print(f'  {order[j]:22s} ' + ' '.join(f'{m[cls == k].mean():7.3f}' for k in range(4)))
for mname, day in months.items():
    print(f'\n== {mname}: intake as share of the ceiling S/(1+S) by class; G = I(<150)/I(150-250), I(<100)/I(150-250); '
          'krill share of encounters by class')
    for arm, kw in [('library', {}), ('krill pool uniform', dict(krill_uniform=True))] + \
                   [(f'flat {order[j][:16]}', dict(flat=order[j])) for j in menu] + [('all flat + prey uniform', dict(flat='all'))]:
        names, E, S, I = run(day, **kw)
        ks = E[names.index('krill')] / E.sum(0)
        g1, g2 = g(I)
        dsh = E / E.sum(0)
        print(f'  {arm:26s} I ' + ' '.join(f'{x:5.3f}' for x in I) + f'  G {g1:5.2f} {g2:5.2f}  krill ' +
              ' '.join(f'{x:4.2f}' for x in ks) + ('   S ' + ' '.join(f'{x:6.2f}' for x in S) if arm == 'library' else ''))
        if arm == 'library':
            for n, row in zip(names, dsh):
                print(f'      share {n:22s} ' + ' '.join(f'{x:5.3f}' for x in row))
