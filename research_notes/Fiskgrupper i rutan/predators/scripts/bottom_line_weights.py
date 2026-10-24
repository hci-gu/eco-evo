"""Biomass-weighted bottom-line values for predatory_fish (values typed from evidence_B_population.md).
Usage: python3 -I bottom_line_weights.py <out_csv>"""
import sys, csv
# species: weight %, r_max, r_now, F, F_pelagic, M_total(adult basis), M2_model, M2_within, natmort(M0+M2 nonmodel)
S = {
 'saithe':        (41.0, 0.52, 0.44, 0.30,   0.0,   0.29, 0.0,   0.0,   0.29),
 'whiting':       (37.0, 0.49, 0.49, 0.08,   0.005, 0.685,0.241, 0.178, 0.253),
 'mackerel':      (8.0,  0.43, 0.25, 0.215,  0.213, 0.15, 0.0,   0.0,   0.15),
 'spurdog':       (5.0,  0.07, 0.07, 0.0076, 0.0,   0.10, 0.0,   0.0,   0.10),
 'horse_mackerel':(3.4,  0.39, 0.33, 0.038,  0.035, 0.30, 0.0,   0.0,   0.30),
 'hake':          (2.5,  0.48, 0.26, 0.27,   0.0,   0.45, 0.0,   0.0,   0.45),
 'anglerfish':    (1.4,  0.47, 0.47, 0.08,   0.0,   0.25, 0.0,   0.0,   0.25),
 'ling':          (0.7,  0.55, 0.55, 0.17,   0.0,   0.15, 0.0,   0.0,   0.15),
 'velvet_belly':  (0.5,  0.10, 0.10, 0.10,   0.0,   0.30, 0.0,   0.0,   0.30),
 'pollack':       (0.4,  0.63, 0.63, 0.20,   0.0,   0.28, 0.0,   0.0,   0.28),
}
cols = ['r_max', 'r_now', 'F', 'F_pelagic', 'M_total', 'M2_model', 'M2_within', 'natmort']
def wmean(keys):
    W = sum(S[k][0] for k in keys)
    return [sum(S[k][0] * S[k][i + 1] for k in keys) / W for i in range(len(cols))]
sets = {'all': list(S), 'saithe+whiting': ['saithe', 'whiting'],
        'no_elasmobranchs': [k for k in S if k not in ('spurdog', 'velvet_belly')]}
with open(sys.argv[1], 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['set'] + cols + ['F_bottom'])
    for n, ks in sets.items():
        v = wmean(ks); w.writerow([n] + [round(x, 3) for x in v] + [round(v[2] - v[3], 3)])
        print(n, ' '.join(f'{c}={x:.3f}' for c, x in zip(cols, v)), f'F_bottom={v[2]-v[3]:.3f}',
              f'natmort+within={v[7]+v[6]:.3f}')
