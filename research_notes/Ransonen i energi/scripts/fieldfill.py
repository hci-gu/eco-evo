"""Counterfactual: the realised ledger with the box fish at the FIELD fill.

Per month (years 2-3): costs and grown / lost / exchanged tonnes as
realised; the reserve terms re-priced at the field fill f(m) (outside_fill,
else start_fill): exchange xB f ME (no refill of arrivals, emigrants at the
field fat), losses lost f ME, growth ec + c (ec + f ME). The difference to
the realised need is what the full reserve costs.
"""
import json, sys, yaml
d = json.load(open(sys.argv[1])); lib = yaml.safe_load(open(d['meta']['lib']))['species_definitions']
M = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
FIELD = dict(planktivorous_fish=(6.2, 3242), benthivorous_fish=(4.94, 2197), whiting=(4.2, 3370),
             predatory_fish=(2.83, 3903), mackerel=(2.75, 3269))
print(f"{'group':>18} {'gain':>6} {'cost':>6} {'growth':>6} {'lostR':>6} {'xR_out':>6} {'dR':>5} | "
      f"{'@field fill: growth':>19} {'lostR':>6} {'xR_out':>6} {'need':>6} | {'fieldE':>6} {'real/f':>6} {'need/f':>6}")
for f, (qb, gpt) in FIELD.items():
    sp = lib[f]; ME = sp['max_energy_reserve']; ec = sp['energy_content']; c = sp.get('growth_cost', 0.0)
    fills = sp.get('outside_fill') or sp.get('start_fill')
    tot = dict(B=0, gain=0, cost=0, growth=0, lostR=0, xR=0, dR=0, g_f=0, l_f=0, x_f=0)
    for r in d['rows']:
        if r['year'] < 1: continue
        g = r['groups'][f]; fm = fills[M.index(r['month'])]
        grown = g['dB'] - g['xB'] + g['lost_t'] + g['starv_t']
        cost = g['c_rest'] + g['c_eat'] + g['c_move']
        tot['B'] += g['B']; tot['gain'] += g['gain']; tot['cost'] += cost
        tot['growth'] += g['gain'] - cost + g['xR'] - g['lost_R'] - g['dR']
        tot['lostR'] += g['lost_R']; tot['xR'] += g['xR']; tot['dR'] += g['dR']
        tot['g_f'] += grown * (ec + c * (ec + fm * ME))
        tot['l_f'] += g['lost_t'] * fm * ME
        tot['x_f'] += g['xB'] * fm * ME
    k = 1460.0 / tot['B']
    q = {a: v * k for a, v in tot.items() if a != 'B'}
    need_f = q['cost'] + q['g_f'] + q['l_f'] - q['x_f']
    fe = qb * gpt
    print(f"{f:>18} {q['gain']:>6.0f} {q['cost']:>6.0f} {q['growth']:>6.0f} {q['lostR']:>6.0f} {-q['xR']:>6.0f} {q['dR']:>5.0f} | "
          f"{q['g_f']:>19.0f} {q['l_f']:>6.0f} {-q['x_f']:>6.0f} {need_f:>6.0f} | {fe:>6.0f} {q['gain']/fe:>6.2f} {need_f/fe:>6.2f}")
