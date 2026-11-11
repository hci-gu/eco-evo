"""Can each fish group's energy budget close at its field Q/B? Library and manifest only - no policy, no rollout.

Per fish DM, per tonne-year (MJ):
  I_field = field Q/B x q, q = assimilated energy per tonne eaten (energy_gain_mat, net of SDA) averaged over
            the BOOKED diet (M2 x B_prey at the start densities), and, for comparison, over a given realised diet;
  RM      = resting_metabolism x the annual mean of its Q10 multiplier x ticks per year;
  costs   = RM x (e feeding_cost + (1 - e) resting_cost), e = the share of biomass-time spent eating
            (movement left out);
  G       = somatic P x the tonne price ec + c (ec + s ME);
  refill  = the reserve carried away by losses, s ME x (M1 + booked M2 on the group + adult emigration);
for s = the mean of the group's monthly start_fill (field) and s = 1 (the trained fish).
e* = the eating share at which I_field = costs + G + refill; e* < 0 means the budget cannot close even resting
all the time, e* > 1 means it closes with time to spare.
usage: closure.py"""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
import numpy as np
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.diagnostics.field_diet import FIELD_QB
S = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open('results/pzbpgp248nsrpfix96/progress/config.json'))
eco = {k: cfg[k] for k in ("project", "library", "grid", "mortality", "migration", "mortality_multiplier",
                            "mass_balance", "currents", "tick_hours") if k in cfg}
with evaluation_randomness(cfg['seed']): env = build_inference_env(eco, cfg['seed'])
TPY = 1460
dens = {f: float(env.fgs[f].biomass.mean()) for f in env.global_fg_order}
temp = getattr(env, 'metabolism_temp_table', None)
# Realised diets of fix96's probe 40, year 3 (probe_diets), for the q comparison.
REALISED = {
    'planktivorous_fish': {'krill': 0.500, 'zooplankton': 0.500},
    'benthivorous_fish': {'benthic_infauna': 0.746, 'krill': 0.111, 'mobile_epifauna': 0.035,
                          'norway_lobster': 0.053, 'planktivorous_fish': 0.055, 'whiting': 0.001},
    'whiting': {'benthivorous_fish': 0.011, 'krill': 0.771, 'mobile_epifauna': 0.095,
                'planktivorous_fish': 0.105, 'zooplankton': 0.017},
    'predatory_fish': {'benthivorous_fish': 0.017, 'krill': 0.874, 'mackerel': 0.003,
                       'planktivorous_fish': 0.083, 'whiting': 0.021, 'zooplankton': 0.003},
    'mackerel': {'krill': 0.672, 'planktivorous_fish': 0.015, 'zooplankton': 0.313},
}
order = list(env.global_fg_order)
out = {}
print(f"{'group':20s} {'QB':>5s} {'q_bk':>6s} {'q_re':>6s} {'I_fld':>7s} {'RM':>6s} {'ce':>5s} {'cr':>5s} "
      f"{'P':>5s} {'s_fld':>5s} {'loss':>5s} | s field: {'G':>6s} {'refill':>6s} {'e*':>6s} | s=1: {'G':>6s} "
      f"{'refill':>6s} {'e*':>6s} | e* with q_realised (s=1)")
for i, f in enumerate(env.dm_ids):
    if f not in FIELD_QB:
        continue
    fg = env.fgs[f]; p = fg.params
    qb = FIELD_QB[f][0]
    pairs = p.get('interaction', {}) or {}
    flow = {}
    for j, prey in enumerate(order):
        if not env.eat_static_mask[i, j]:
            continue
        m2 = float((pairs.get(f"{f}_preys_on_{prey}") or {}).get('predation_mortality', 0.0) or 0.0)
        if m2 > 0:
            flow[prey] = m2 * dens[prey]
    tot = sum(flow.values())
    q_bk = sum(v / tot * float(env.energy_gain_mat[i, order.index(k)]) for k, v in flow.items())
    rd = REALISED[f]; rs = sum(rd.values())
    q_re = sum(v / rs * float(env.energy_gain_mat[i, order.index(k)]) for k, v in rd.items())
    q10 = float(np.mean(temp[:, i])) if temp is not None else 1.0
    RM = float(env.dm_resting_metabolism[i]) * q10 * TPY
    ce, cr = float(env.dm_cost_eat[i]), float(env.dm_cost_rest[i])
    P = float(p.get('somatic_production', 0.0) or 0.0)
    ec = float(p.get('energy_content')); ME = float(env.dm_max_energy_reserve[i]); c = float(env.dm_growth_cost[i])
    fills = p.get('start_fill') or p.get('outside_fill')
    s_fld = float(np.mean(fills)) if fills else float('nan')
    m1 = float(p.get('natural_mortality', 0.0) or 0.0) * TPY   # per tick in the library -> per year (fixed 2026-11-11)
    m2_on = 0.0
    for pred in env.global_fg_order:
        pp = env.fgs[pred].params.get('interaction', {}) or {}
        m2_on += float((pp.get(f"{pred}_preys_on_{f}") or {}).get('predation_mortality', 0.0) or 0.0)
    emig = (p.get('juvenile_stage') or {}).get('emigration') or [0.0, 0.0]
    loss = m1 + m2_on + float(emig[1])
    row = dict(QB=qb, q_booked=q_bk, q_realised=q_re, I_field=qb * q_bk, RM=RM, q10_mean=q10, ce=ce, cr=cr, P=P,
               s_field=s_fld, M1=m1, M2_on=m2_on, emig_adult=float(emig[1]), booked_diet={k: v / tot for k, v in flow.items()})
    for tag, s in (('field', s_fld), ('full', 1.0)):
        G = P * (ec + c * (ec + s * ME)); refill = loss * s * ME
        def estar(I):
            return (I - G - refill - RM * cr) / (RM * (ce - cr))
        row[tag] = dict(G=G, refill=refill, e_star=estar(qb * q_bk), e_star_realised_q=estar(qb * q_re))
    out[f] = row
    print(f"{f[:20]:20s} {qb:5.2f} {q_bk:6.0f} {q_re:6.0f} {qb*q_bk:7.0f} {RM:6.0f} {ce:5.2f} {cr:5.2f} {P:5.2f} "
          f"{s_fld:5.2f} {loss:5.2f} | {row['field']['G']:13.0f} {row['field']['refill']:6.0f} {row['field']['e_star']:6.2f} "
          f"| {row['full']['G']:10.0f} {row['full']['refill']:6.0f} {row['full']['e_star']:6.2f} | "
          f"{row['full']['e_star_realised_q']:6.2f}")
os.makedirs(f"{S}/../tables", exist_ok=True)
json.dump(out, open(f"{S}/../tables/closure.json", 'w'), indent=1)
