"""Energy accounting of benthivorous_fish per tonne per year, as the engine charges it.

    python3 accounting.py FROZEN_YEARS_JSON SPATIAL_DIR [LIBRARY]

Builds the probe world once (to read every value the loader derives:
growth_rate, M1, the K-loss coefficient, the monthly Q10 multiplier of
the at_50m layer) and prints the budget the engine closes per tonne of
fish per year:
  costs at 100 % rest and at the realised action mix (from the spatial
  probe, year 1), the feeding surcharge, production cost per tonne
  produced (energy_content + s * max_energy_reserve: the debit plus the
  reserve the new tonne must carry), the intake needed to break even and
  to pay the budgeted / realised production, in MJ and tonnes of prey,
  and the energy the budget's own Q/B 3.48 delivers with its diet and
  with the engine's realised diet.
No repo file is written.
"""
import json
import math
import os
import sys

ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

TPY = 1460
BEN = 'benthivorous_fish'


def main(fy_path, spatial_dir, library=None):
    import numpy as np
    import train as train_mod
    from lib.environments.ecosystem_env.currents import CurrentConfig
    from lib.world.growth_budget import growth_budget
    import yaml

    env = train_mod._ProbeEnvBuilder(
        project_path='mareld2.yaml', grid_size=(60, 60),
        apply_natural_mortality=True, migration=True,
        currents=CurrentConfig(0.1, 20, 0, 12.0), library_path=library)()
    fg = env.fgs[BEN]
    i = env.dm_ids.index(BEN)
    p = fg.params
    rm = float(env.dm_resting_metabolism[i])
    c_eat, c_rest, c_move = (float(env.dm_cost_eat[i]), float(env.dm_cost_rest[i]),
                             float(env.dm_cost_move[i]))
    tm = np.asarray(env.metabolism_temp_table)[:, i]
    tm_mean = float(tm.mean())
    ec = float(p['energy_content'])
    me = float(fg.max_energy_reserve)
    u = float(fg.maintenance_level)
    g = float(fg.growth_rate)
    m1 = -TPY * math.log(1 - float(fg.natural_mortality))
    cK = float(env.dm_capacity_rate[i])
    K = float(p.get('regional_carrying_capacity'))
    rmax = float(p['r_max'])

    lib = yaml.safe_load(open(library or 'fgconfig/fg_library.yaml'))
    sd, inter = lib['species_definitions'], lib['interaction_definitions']
    active = [f for f in env.fgs]
    bud = growth_budget(BEN, sd, inter, active_predators=active)

    def qual(prey):
        e = inter[f'{BEN}_preys_on_{prey}']
        return float(e.get('assimilation_factor', 1.0)) * float(sd[prey]['energy_content'])

    q_inf, q_epi = qual('benthic_infauna'), qual('mobile_epifauna')
    q_plk = 0.8 * 7125.0  # planktivores at the reference fill (static quality)
    q_pred = 0.8 * 5252.0

    d = np.load(os.path.join(spatial_dir, 'year_01.npz'))
    Bsum = float(d['B'].sum())
    eat = float(d['eatB'].sum()) / Bsum
    rest = float(d['restB'].sum()) / Bsum
    move = float(d['moveB'].sum()) / Bsum
    fill = float(d['Bs'].sum()) / Bsum
    gains_obs = float(d['gains'].sum()) / Bsum * TPY
    cost_obs = float(d['cost'].sum()) / Bsum * TPY
    qb_inf_obs = float(d['inf_eat'].sum()) / Bsum * TPY

    fy = json.load(open(fy_path))
    Y = fy['years']

    def realised(y):
        mB = y['mean'][BEN]
        L = {k: y['loss'][k][BEN] / mB for k in y['loss']}
        growth = math.log(y['end'][BEN] / y['start'][BEN])
        I = y['intake'][BEN]
        T = sum(I.values())
        return L, growth, T / mB, {q: v / T for q, v in I.items()}

    out = []
    pr = out.append
    pr('# Benthivorous fish: engine energy accounting per tonne of fish per year')
    pr(f'library {library or "fgconfig/fg_library.yaml"}; probe year 1 of {spatial_dir}')
    pr('')
    pr('## Loaded values')
    pr(f'resting_metabolism {rm} MJ/t/tick; feeding_cost {c_eat}, resting_cost {c_rest}, movement_cost {c_move}')
    pr(f'Q10 multiplier at_50m: monthly-table mean {tm_mean:.4f} (min {tm.min():.3f}, max {tm.max():.3f})')
    pr(f'energy_content {ec}, max_energy_reserve {me}, maintenance u {u}, derived g {g:.6g}/tick')
    pr(f'M1 {m1:.4f}/yr, K-loss coefficient {cK:.6g}/tick per t/cell (r_max/(1460 K) = {rmax / (TPY * K):.6g}), K {K}')
    pr(f'budget: gross {bud.gross_annual:.4f}/yr = r_max {bud.r_max} + M1 {bud.m1_annual:.4f} + M2 {bud.predation_total:.4f} '
       f'({", ".join(f"{k} {v:.4f}" for k, v in bud.predation.items())}); s* {bud.reference_fill:.4f}; '
       f'ideal intake {bud.ideal_intake:.2f} MJ/t/tick on {bud.best_prey}; ideal cost {bud.ideal_cost} (no Q10)')
    pr(f'quality (assimilation x E, MJ per t eaten): infauna {q_inf:.0f}, epifauna {q_epi:.0f}, '
       f'planktivores {q_plk:.0f}, predators {q_pred:.0f}')
    pr('')
    pr('## Realised action mix and probe meters, year 1')
    pr(f'eat {eat:.4f}, rest {rest:.4f}, move {move:.4f}; biomass-weighted fill {fill:.3f}')
    pr(f'gains {gains_obs:.0f} MJ/t/yr assimilated; costs charged {cost_obs:.0f} MJ/t/yr; Q/B on infauna {qb_inf_obs:.2f}')
    for k in (0, 3, 5, 7):
        if k < len(Y):
            L, gr, qb, diet = realised(Y[k])
            pr(f'year {k + 1}: losses/yr ' + ', '.join(f'{a[5:]} {b:.3f}' for a, b in L.items())
               + f'; net growth {gr:+.3f}/yr; Q/B {qb:.2f}; diet '
               + ', '.join(f'{q} {v:.2f}' for q, v in sorted(diet.items(), key=lambda kv: -kv[1]) if v > 0.005))
    pr('')

    year_rm = rm * tm_mean * TPY
    cost_rest_all = year_rm * c_rest
    mix = eat * c_eat + rest * c_rest + move * c_move
    cost_mix = year_rm * mix
    surcharge = year_rm * eat * (c_eat - c_rest)
    unit_prod = ec + fill * me
    P_budget = bud.gross_annual
    P_fill = g * (fill - u) * TPY
    L1, gr1, qb1, diet1 = realised(Y[0])
    P_real = sum(L1.values()) + gr1

    rows = []

    def row(name, mj, note=''):
        rows.append((name, mj, mj / q_inf, note))

    row('A. resting cost, 100 % rest (RM x Q10 mean)', cost_rest_all, f'{rm} x {tm_mean:.3f} x 1460')
    row('B. action costs at the realised mix', cost_mix, f'mix factor {mix:.3f} x A')
    row('   of which the feeding surcharge (eat x (feeding_cost-1) x RM)', surcharge, '')
    row('C. production at the realised fill, g (s-u)', P_fill * unit_prod, f'P {P_fill:.3f}/yr x {unit_prod:.0f} MJ/t')
    row('   C1. budget gross production', P_budget * unit_prod, f'P {P_budget:.3f}/yr')
    row('   C2. realised year-1 losses + net growth', P_real * unit_prod, f'P {P_real:.3f}/yr')
    row('   C3. reserve carried by each new tonne (s ME), part of C', P_fill * fill * me, '')
    row('D = B + C: break-even + realised production', cost_mix + P_fill * unit_prod, '')
    row('E. B alone (costs, no production)', cost_mix, '')
    pr('## Required assimilated energy per t fish per yr (MJ) and its infauna equivalent (t infauna / t fish / yr)')
    pr(f'{"item":72} {"MJ/t/yr":>9} {"t inf":>7}  note')
    for n, mj, t, note in rows:
        pr(f'{n:72} {mj:9.0f} {t:7.2f}  {note}')
    pr('')
    budget_diet = {'benthic_infauna': 0.511, 'mobile_epifauna': 0.209,
                   'planktivorous_fish': 0.084, 'predatory_fish': 0.019}
    rest_share = 1.0 - sum(budget_diet.values())
    q_budget_listed = (budget_diet['benthic_infauna'] * q_inf + budget_diet['mobile_epifauna'] * q_epi
                       + budget_diet['planktivorous_fish'] * q_plk + budget_diet['predatory_fish'] * q_pred)
    q_mix_engine_pools = q_budget_listed / sum(budget_diet.values())
    pr('## What the budget Q/B 3.48 delivers (assimilated MJ/t/yr)')
    pr(f'budget diet in the model pools (inf 0.511, epi 0.209, plk 0.084, pred 0.019; the other {rest_share:.3f} '
       f'(within-group, sandeel, gobies, cephalopods, zooplankton) taken at the pools\' mean quality {q_mix_engine_pools:.0f})')
    pr(f'  3.48 x {q_mix_engine_pools:.0f} = {3.48 * q_mix_engine_pools:.0f} MJ/t/yr')
    pr(f'engine diet (99 % infauna): 3.48 x {q_inf:.0f} = {3.48 * q_inf:.0f} MJ/t/yr')
    pr(f'Q/B on infauna alone to cover B (costs only): {cost_mix / q_inf:.2f}; B + C: {(cost_mix + P_fill * unit_prod) / q_inf:.2f}')
    pr(f'Q/B on the budget mix to cover B: {cost_mix / q_mix_engine_pools:.2f}; B + C1: '
       f'{(cost_mix + P_budget * unit_prod) / q_mix_engine_pools:.2f}')
    pr(f'Holling ceiling Q/B on infauna (1/h x 1460): {TPY / float(inter[BEN + "_preys_on_benthic_infauna"]["handling_time"]):.1f}')
    pr(f'infauna M2 implied by the engine at B_ben 1.53 and infauna B* 93.6: '
       f'{(cost_mix + P_fill * unit_prod) / q_inf * 1.53 / 93.6:.3f}/yr (budget 0.029; realised 0.12-0.16)')
    pr('')
    pr('## The same budget with cost alternatives (MJ/t/yr; t infauna)')
    alts = [
        ('feeding_cost 1.0 (no surcharge; RM 3.6 = 1.2 x SMR routine)', 1.0, None),
        ('feeding_cost 1.5', 1.5, None),
        ('SDA 0.15 of ingested energy instead of the surcharge', 1.0, 0.15),
    ]
    for name, ce, sda in alts:
        cm = year_rm * (eat * ce + rest * c_rest + move * c_move)
        need = cm + P_fill * unit_prod
        if sda is not None:
            # ingested energy E_in = need / (assim - sda) per unit gross energy
            assim = q_inf / float(sd['benthic_infauna']['energy_content'])
            gross_E = need / (assim - sda)
            need_t = gross_E / float(sd['benthic_infauna']['energy_content'])
            pr(f'{name:72} {assim * gross_E:9.0f} {need_t:7.2f}  (assimilated; SDA paid out of it)')
        else:
            pr(f'{name:72} {need:9.0f} {need / q_inf:7.2f}')
    pr(f'{"production without the reserve carry (ec only)":72} {cost_mix + P_fill * ec:9.0f} '
       f'{(cost_mix + P_fill * ec) / q_inf:7.2f}')
    txt = '\n'.join(out)
    print(txt)
    return txt


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) < 2:
        print(__doc__)
        sys.exit(2)
    main(a[0], a[1], a[2] if len(a) > 2 else None)
