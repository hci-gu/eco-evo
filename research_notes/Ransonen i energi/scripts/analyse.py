"""Tables from ledger.json: annual and monthly energy budgets per fish group.

Rates are per tonne of mean biomass per year (MJ/t/yr, t/t/yr). The
residual is gain - costs + reserve imported net - reserve carried out by
losses - change of the reserve = the energy spent on growth (and any
catabolism term). The field yardstick in energy is FIELD_QB x the gain per
tonne of the BOOKED diet (energy_gain_mat at the reference fills).
"""
import json
import os
import sys

import numpy as np

ROOT = '/home/simon/eco-evo'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from lib.diagnostics.field_diet import FIELD_QB  # noqa: E402
from lib.runners import training_progress as tp  # noqa: E402

path = sys.argv[1]
d = json.load(open(path))
meta, rows = d['meta'], d['rows']
FISH = ['planktivorous_fish', 'benthivorous_fish', 'whiting', 'predatory_fish', 'mackerel']
SHORT = dict(planktivorous_fish='plankt', benthivorous_fish='benthiv', whiting='whiting',
             predatory_fish='watercol', mackerel='mackerel')
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Gain per tonne eaten by pair (static quality at the reference fill) and the booked M2 x B shares.
config = json.load(open(os.path.join(meta['run'], 'progress', 'config.json')))
config['library'] = meta['lib']
env = tp.build_inference_env(config, int(config['seed']))
env.build_static_caches()
order = list(env.global_fg_order)
gain_t = {}
for f in FISH:
    i = env.dm_ids.index(f)
    gain_t[f] = {order[j]: float(env.energy_gain_mat[i, j]) for j in range(len(order))
                 if env.energy_gain_mat[i, j] > 0}

import yaml  # noqa: E402
lib = yaml.safe_load(open(meta['lib']))
inter = lib['interaction_definitions']
start = {f: float(np.asarray(env.fgs[f].biomass).sum()) for f in order}


def booked_shares(pred):
    out = {}
    for key, v in inter.items():
        if not key.startswith(pred + '_preys_on_') or not isinstance(v, dict):
            continue
        prey = key[len(pred + '_preys_on_'):]
        m2 = v.get('predation_mortality')
        if m2 and v.get('preys_on') and prey in start:
            out[prey] = float(m2) * start[prey]
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()} if tot else {}


def agg(sel):
    acc = {}
    for r in rows:
        if not sel(r):
            continue
        for f, g in r['groups'].items():
            a = acc.setdefault(f, {})
            for k, v in g.items():
                a[k] = a.get(k, 0.0) + v
    return acc


def rates(a):
    per = 1460.0 / a['B']          # x total -> per tonne-year
    tons = {k[2:]: v for k, v in a.items() if k.startswith('t:')}
    c = a['c_rest'] + a['c_eat'] + a['c_move']
    resid = a['gain'] - c + a['xR'] - a['lost_R'] - a['dR']
    grown = a['dB'] - a['xB'] + a['lost_t'] + a['starv_t']
    return dict(
        fill=a['R'] / a['B'] , ration=sum(tons.values()) * per, gain=a['gain'] * per,
        c_rest=a['c_rest'] * per, c_eat=a['c_eat'] * per, c_move=a['c_move'] * per, cost=c * per,
        xR=a['xR'] * per, lostR=a['lost_R'] * per, dR=a['dR'] * per, resid=resid * per,
        grown=grown * per, lost=a['lost_t'] * per, xB=a['xB'] * per,
        eat=a['sh_eat'] / a['B'], move=a['sh_move'] / a['B'], rest=a['sh_rest'] / a['B'],
        gpt=a['gain'] / max(sum(tons.values()), 1e-30), tons=tons)


print(f"run {meta['run']}  library {meta['lib']}  years {meta['years']}\n")
print("=== Annual budget per tonne of mean biomass (MJ/t/yr; ration t/t/yr) ===")
hdr = (f"{'group':>9} {'yr':>2} {'fill':>5} {'eat':>4} {'mov':>4} {'rst':>4} {'ration':>6} {'MJ/t':>5} "
       f"{'gain':>6} {'c_rest':>6} {'c_eat':>6} {'c_move':>6} {'xR_in':>6} {'lostR':>6} {'dR':>5} "
       f"{'growthE':>7} {'grown':>5} {'lost':>5} {'xB':>6}")
print(hdr)
annual = {}
for f in FISH:
    for y in range(meta['years']):
        sel = agg(lambda r, y=y: r['year'] == y)
        if f not in sel:
            continue
        a = sel[f]
        q = rates(a)
        annual[(f, y)] = q
        fill = q['fill'] / meta['ME'][f]
        print(f"{SHORT[f]:>9} {y + 1:>2} {fill:>5.2f} {q['eat']:>4.2f} {q['move']:>4.2f} {q['rest']:>4.2f} "
              f"{q['ration']:>6.2f} {q['gpt']:>5.0f} {q['gain']:>6.0f} {q['c_rest']:>6.0f} {q['c_eat']:>6.0f} "
              f"{q['c_move']:>6.0f} {q['xR']:>6.0f} {q['lostR']:>6.0f} {q['dR']:>5.0f} {q['resid']:>7.0f} "
              f"{q['grown']:>5.2f} {q['lost']:>5.2f} {q['xB']:>6.2f}")

print("\n=== Ration in tonnes vs energy against the field Q/B (years 2-3) ===")
print(f"{'group':>9} {'fieldQB':>7} {'bookMJ/t':>8} {'fieldMJ':>7} {'real t':>6} {'realMJ/t':>8} "
      f"{'realMJ':>7} {'x tonnes':>8} {'x energy':>8}")
for f in FISH:
    qb = FIELD_QB[f][0]
    shares = booked_shares(f)
    book_gpt = sum(s * gain_t[f].get(p, 0.0) for p, s in shares.items())
    a = agg(lambda r: r['year'] >= 1)[f]
    q = rates(a)
    print(f"{SHORT[f]:>9} {qb:>7.2f} {book_gpt:>8.0f} {qb * book_gpt:>7.0f} {q['ration']:>6.2f} "
          f"{q['gpt']:>8.0f} {q['gain']:>7.0f} {q['ration'] / qb:>8.2f} {q['gain'] / (qb * book_gpt):>8.2f}")
    print(f"{'':>9} gain/t eaten by prey: " + ", ".join(f"{p} {v:.0f}" for p, v in sorted(gain_t[f].items())))
    print(f"{'':>9} booked shares: " + ", ".join(f"{p} {v:.2f}" for p, v in sorted(shares.items())))

print("\n=== Monthly (years 2-3 pooled), MJ/t/yr ===")
for f in FISH:
    print(f"\n{f}  (ME {meta['ME'][f]:.0f}, rest metabolism {meta['rm'][f]} MJ/t/tick at t_ref)")
    print(f"{'mon':>4} {'B':>7} {'fill':>5} {'eat':>4} {'mov':>4} {'ration':>6} {'gain':>6} {'cost':>6} "
          f"{'c_move':>6} {'xR_in':>6} {'lostR':>6} {'dR':>6} {'growthE':>7} {'xB':>6}")
    for m in MONTHS:
        sel = agg(lambda r, m=m: r['year'] >= 1 and r['month'] == m)
        if f not in sel:
            continue
        a = sel[f]
        q = rates(a)
        meanB = a['B'] / a['ticks']
        print(f"{m:>4} {meanB:>7.0f} {q['fill'] / meta['ME'][f]:>5.2f} {q['eat']:>4.2f} {q['move']:>4.2f} "
              f"{q['ration']:>6.2f} {q['gain']:>6.0f} {q['cost']:>6.0f} {q['c_move']:>6.0f} {q['xR']:>6.0f} "
              f"{q['lostR']:>6.0f} {q['dR']:>6.0f} {q['resid']:>7.0f} {q['xB']:>6.2f}")
