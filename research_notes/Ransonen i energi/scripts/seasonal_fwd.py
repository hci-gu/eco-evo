"""Monthly realised gain vs the need of a fish on the FIELD fill cycle.

A box fish on the field fill f(m) has R = B f ME, so the reserve
balance gives gain = cost + G (price(f) + f ME) + B ME df/dt: losses and
the exchange carry the reserve out and in at the same fill and cancel
(the annual sum equals the fieldfill.py need). Realised costs and grown
tonnes G; df/dt centred from the mid-month fills. MJ/t/yr, years 2-3.
"""
import json, sys, yaml
d = json.load(open(sys.argv[1])); lib = yaml.safe_load(open(d['meta']['lib']))['species_definitions']
M = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for f in ['planktivorous_fish','benthivorous_fish','whiting','predatory_fish','mackerel']:
    sp = lib[f]; ME = sp['max_energy_reserve']; ec = sp['energy_content']; c = sp.get('growth_cost', 0.0)
    fills = sp.get('outside_fill') or sp.get('start_fill')
    print(f"\n{f}: realised gain vs need on the field fill cycle (MJ/t/yr); ratio; field fill; realised fill")
    print(f"{'mon':>4} {'fill_f':>6} {'fill_r':>6} {'gain':>6} {'need':>6} {'depos':>6} {'r/n':>5}")
    tg = tn = tb = 0
    for mi, m in enumerate(M):
        acc = {}
        for r in d['rows']:
            if r['year'] >= 1 and r['month'] == m:
                for k, v in r['groups'][f].items():
                    acc[k] = acc.get(k, 0) + v
        fm = fills[mi]; dfdt = (fills[(mi + 1) % 12] - fills[mi]) * 12  # per yr
        k = 1460.0 / acc['B']
        grown = acc['dB'] - acc['xB'] + acc['lost_t'] + acc['starv_t']
        cost = acc['c_rest'] + acc['c_eat'] + acc['c_move']
        need = (cost + grown * (ec + c * (ec + fm * ME) + fm * ME)) * k + ME * dfdt
        gain = acc['gain'] * k
        tg += acc['gain']; tn += need / k; tb += acc['B']
        print(f"{m:>4} {fm:>6.2f} {acc['R'] / acc['B'] / ME:>6.2f} {gain:>6.0f} {need:>6.0f} {ME * dfdt:>6.0f} {gain / need if need > 0 else float('nan'):>5.2f}")
    print(f"year: gain {tg * 1460 / tb:.0f} need {tn * 1460 / tb:.0f} ratio {tg / tn:.2f}")
