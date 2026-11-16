"""Sensitivity grid of reading C: memory span x persistence time (single exponential)."""
import memory_weight as mw

PAIRS = [('porpoises', 'planktivores', False), ('porpoises', 'diffuse', False),
         ('porpoises', 'benthivores', False), ('planktivores', 'porpoises', False),
         ('planktivores', 'porpoises', True), ('planktivores', 'diffuse', False),
         ('benthivores', 'diffuse', False), ('whiting', 'planktivores', False),
         ('seals', 'benthivores', False)]
MEM = [1, 2, 4, 8, 16]
T = [6, 24, 72, 168, 1e9]
print('rows: pair; columns: persistence time T (h, single exponential); cells: w for memory M ticks')
for o, t, own in PAIRS:
    print(f'\n{o} -> {t}{" (own cell)" if own else ""}')
    print('  M\\T ' + ' '.join(f'{("inf" if x > 1e8 else int(x)):>6}' for x in T))
    for m in MEM:
        row = [mw.weight(o, t, m, mw.rho_fn(1.0, x, x), own=own) for x in T]
        print(f'  {m:>3} ' + ' '.join(f'{v:6.3f}' for v in row))
