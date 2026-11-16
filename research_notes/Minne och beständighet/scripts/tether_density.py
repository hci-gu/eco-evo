import memory_weight as mw
import numpy as np
L, D = 14.0, 5.5
for tether in (0.0, 8.0, 4.0, 2.0):
    pos = mw.tracks(L, D, 16, tether)
    net = lambda k: float(np.median(np.hypot(*(pos[:, k * mw.STEPS_PER_TICK] - pos[:, 0]).T)))
    row = [mw.weight('porpoises', 'diffuse', m, mw.rho_fn(1, 1e9, 1e9), tether=tether) for m in (1, 4, 8, 16)]
    rowp = [mw.weight('porpoises', 'planktivores', m, mw.rho_fn(1, 72, 72), tether=tether) for m in (1, 4, 8, 16)]
    print(f'tether {tether:4.1f} km: net 6h {net(1):.1f} 24h {net(4):.1f} 48h {net(8):.1f} 96h {net(16):.1f} km | '
          f'diffuse T=inf M1/4/8/16 ' + ' '.join(f'{v:.2f}' for v in row) +
          ' | plankt T=72h ' + ' '.join(f'{v:.2f}' for v in rowp))
print('planktivores -> porpoises own cell, M=1, rho=1, by local porpoise density factor')
for f in (1, 3, 6, 20):
    print(f'  x{f}: own {mw.weight("planktivores", "porpoises", 1, mw.rho_fn(1,1e9,1e9), own=True, density_factor=f):.3f}'
          f'  neighbour {mw.weight("planktivores", "porpoises", 1, mw.rho_fn(1,1e9,1e9), density_factor=f):.3f}')
