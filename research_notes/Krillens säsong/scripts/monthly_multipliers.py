"""Policy-free: each predator's krill encounter by month (linear regime, a x B_krill x E per tonne of predator),
relative to its library annual mean, for the field season terms switched on one by one. Predator placement:
whiting and the water column at the GOV bed-class shares in the box (lib/diagnostics/field_diet.py), the others
on equal cells. Bed depth from the manifest's depth map on the 30 x 30 grid. Terms:
  lib       library: stock season 1.0/0.6/1.1/1.3 (quarterly steps), annual depth shape, E 5000 all year
  +shallow  presence over < 100 m beds Sep-Apr 0.43, May-Aug 0.05 of >= 250 m (library annual 0.303), the target
            renormalised to the box total each month
  +energy   E 6 kJ/g Q4-Q1, 4.3 Q2-Q3 (normalised to 5000)
  +flat     stock season 1 all year instead of the quarterly table (with +shallow +energy)
The predators' ingestion Q10 is not multiplied in (it is the fish's, not the krill's).
"""
import sys, yaml, numpy as np
sys.path.insert(0, '/home/simon/eco-evo')
from lib.world import habitat
from lib.diagnostics.field_diet import FIELD_BED_SHARES, bed_class
man = yaml.safe_load(open('/home/simon/eco-evo/mareld2.yaml', encoding='utf-8-sig'))
depth = habitat.load_depth_map(man, '/home/simon/eco-evo/mareld2.yaml', (30, 30)).ravel()
L = yaml.safe_load(open('/home/simon/eco-evo/fgconfig/fg_library.yaml'))
I = L['interaction_definitions']
kr = next(s for s in L['species_definitions'] if s.get('group_id', s.get('id')) == 'krill') \
    if isinstance(L['species_definitions'], list) else L['species_definitions']['krill']


def table(rows, key, d):
    xs = [r['depth_m'] for r in rows]; ys = [r[key] for r in rows]
    return np.interp(d, xs, ys)


w = table(kr['outside_depth_table'], 'weight', depth); w = w / w.mean()
cls = bed_class(depth)
shallow = depth < 100.0
PRES = np.array([0.43] * 4 + [0.05] * 4 + [0.43] * 4); ANN = PRES.mean()
EQ = np.array([6.0] * 3 + [4.3] * 6 + [6.0] * 3); EM = EQ / EQ.mean()
SEAS = np.array([1.0] * 3 + [0.6] * 3 + [1.1] * 3 + [1.3] * 3)
M = 'JFMAMJJASOND'
print(f"cells: < 100 m {shallow.mean():.3f}; class shares (equal cells) "
      + ' '.join(f'{(cls == k).mean():.3f}' for k in range(4)))
for p in ['whiting', 'planktivorous_fish', 'predatory_fish', 'mackerel', 'benthivorous_fish']:
    pair = I[f'{p}_preys_on_krill']
    A = table(pair['depth_access_table'], 'factor', depth); A = A / A.mean()
    if p in FIELD_BED_SHARES:
        sh = np.array(FIELD_BED_SHARES[p][0]); cw = np.array([sh[c] / max((cls == c).sum(), 1) for c in cls])
        place = f"GOV {FIELD_BED_SHARES[p][1]}"
    else:
        cw = np.ones_like(depth) / depth.size; place = 'equal cells'
    rows = {}
    for lab in ['lib', '+shallow', '+shallow+energy', '+shallow+energy+flat']:
        v = []
        for m in range(12):
            wm = w.copy()
            if 'shallow' in lab:
                mult = np.where(shallow, PRES[m] / ANN, 1.0); wm = w * mult; wm = wm * w.sum() / wm.sum()
            s = 1.0 if 'flat' in lab else SEAS[m]
            e = EM[m] if 'energy' in lab else 1.0
            v.append((cw * A * wm).sum() * s * e)
        rows[lab] = np.array(v)
    ref = rows['lib'].mean()
    print(f"\n{p} ({place}); share of its krill encounter on < 100 m beds (lib): "
          f"{(cw * A * w)[shallow].sum() / (cw * A * w).sum():.2f}")
    print("  month                 " + ' '.join(f'{c:>5s}' for c in M) + "   Q1    Q2    Q3    Q4")
    for lab, v in rows.items():
        r = v / ref; q = r.reshape(4, 3).mean(1)
        print(f"  {lab:21s} " + ' '.join(f'{x:5.2f}' for x in r) + ' ' + ' '.join(f'{x:5.2f}' for x in q))
