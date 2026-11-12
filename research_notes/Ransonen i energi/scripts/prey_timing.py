"""Each fish group's prey flows if its monthly intake followed the field-fill
need (seasonal.py) with the realised monthly diet: flow_cf = sum_m t_prey(m)
x need(m)/gain(m) (need floored at 0). Years 2-3; t/yr per year."""
import json, sys, yaml
d = json.load(open(sys.argv[1])); lib = yaml.safe_load(open(d['meta']['lib']))['species_definitions']
M = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
tot_real, tot_cf = {}, {}
for f in ['planktivorous_fish','benthivorous_fish','whiting','predatory_fish','mackerel']:
    sp = lib[f]; ME = sp['max_energy_reserve']; ec = sp['energy_content']; c = sp.get('growth_cost', 0.0)
    fills = sp.get('outside_fill') or sp.get('start_fill')
    real, cf = {}, {}
    for mi, m in enumerate(M):
        acc = {}
        for r in d['rows']:
            if r['year'] >= 1 and r['month'] == m:
                for k, v in r['groups'][f].items():
                    acc[k] = acc.get(k, 0) + v
        fm = fills[mi]; dfdt = (fills[(mi + 1) % 12] - fills[mi - 1]) / 2 * 12
        k = 1460.0 / acc['B']
        grown = acc['dB'] - acc['xB'] + acc['lost_t'] + acc['starv_t']
        cost = acc['c_rest'] + acc['c_eat'] + acc['c_move']
        need = max(0.0, (cost + grown * (ec + c * (ec + fm * ME) + fm * ME)) * k + ME * dfdt)
        scale = need / (acc['gain'] * k)
        for key, v in acc.items():
            if key.startswith('t:'):
                p = key[2:]
                real[p] = real.get(p, 0) + v / 2; cf[p] = cf.get(p, 0) + v * scale / 2
    print(f"{f}: " + ", ".join(f"{p} {real[p]:.0f} -> {cf[p]:.0f} t/yr (x{cf[p] / real[p]:.2f})" for p in sorted(real) if real[p] > 1))
    for p in real:
        tot_real[p] = tot_real.get(p, 0) + real[p]; tot_cf[p] = tot_cf.get(p, 0) + cf[p]
print("all five fish: " + ", ".join(f"{p} {tot_real[p]:.0f} -> {tot_cf[p]:.0f} (x{tot_cf[p] / tot_real[p]:.2f})" for p in sorted(tot_real) if tot_real[p] > 1))
