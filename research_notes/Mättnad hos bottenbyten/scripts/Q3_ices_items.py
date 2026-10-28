"""Q3: benthic prey items per stomach and wet mass per item for North Sea / Skagerrak
gadoids and flatfish, from the ICES Stomach Content Database download already in
research_notes/"Fodoniva och mattnad i falt" (Swedish letters)/data/A_ices_stomach/all.
WoRMS classification of the prey AphiaIDs: data/Q3_ices_items/worms_records.json.

Run: python3 -I scripts/Q3_ices_items.py <ices_all_dir> <worms_records.json>
Box: lat 51-61 N, lon 4 W - 13 E. Individual stomachs only (Number == 1).
Mass per item: rows with Count > 0 and Weight > 0 in g, DigestionStage 1 (fresh) where
the stage is given, else all rows (flagged). Items per stomach: over stomachs holding any
item (non-empty), counting only benthic invertebrate groups.
"""
import json
import sys

import pandas as pd

D, W = sys.argv[1], sys.argv[2]
PRED = {'126436': 'cod', '126437': 'haddock', '126438': 'whiting', '127143': 'plaice',
        '127137': 'long_rough_dab', '127139': 'dab', '127160': 'lemon_sole'}

worms = {str(r['AphiaID']): r for r in json.load(open(W))}


def group(aid):
    r = worms.get(str(aid))
    if not r:
        return 'unknown'
    cls, order, fam, gen = (r.get('class') or ''), (r.get('order') or ''), (r.get('family') or ''), (r.get('genus') or '')
    name = r.get('scientificname') or ''
    if cls == 'Polychaeta':
        return 'polychaete'
    if cls == 'Ophiuroidea':
        return 'ophiuroid'
    if cls == 'Bivalvia':
        return 'bivalve'
    if fam in ('Nephropidae',) or gen == 'Nephrops':
        return 'nephrops'
    if fam in ('Paguridae', 'Diogenidae') or 'Pagur' in name:
        return 'hermit'
    if fam in ('Munididae', 'Galatheidae') or gen == 'Munida':
        return 'squat_lobster'
    if fam == 'Pandalidae':
        return 'pandalid'
    if fam in ('Crangonidae', 'Processidae', 'Hippolytidae', 'Palaemonidae') or order == 'Decapoda' and 'Caridea' in json.dumps(r):
        return 'small_shrimp'
    if order == 'Decapoda':
        return 'crab_other_decapod'
    if order == 'Mysida':
        return 'mysid'
    if order == 'Euphausiacea':
        return 'krill'
    if order == 'Amphipoda':
        return 'amphipod'
    if cls in ('Teleostei', 'Actinopteri', 'Actinopterygii', 'Elasmobranchii'):
        return 'fish'
    if cls == 'Cephalopoda':
        return 'cephalopod'
    if cls in ('Gastropoda',):
        return 'gastropod'
    if cls in ('Echinoidea', 'Holothuroidea', 'Asteroidea'):
        return 'other_echinoderm'
    if cls == 'Copepoda' or cls == 'Hexanauplia':
        return 'copepod'
    return 'other'


BENTHIC = {'polychaete', 'ophiuroid', 'bivalve', 'nephrops', 'hermit', 'squat_lobster', 'pandalid',
           'small_shrimp', 'crab_other_decapod', 'mysid', 'amphipod', 'gastropod', 'other_echinoderm'}

h = pd.read_csv(f'{D}/HaulInformation.csv', on_bad_lines='skip', low_memory=False)
pr = pd.read_csv(f'{D}/PredatorInformation.csv', on_bad_lines='skip', low_memory=False)
p = pd.read_csv(f'{D}/PreyInformation.csv', on_bad_lines='skip', low_memory=False)
h = h[h.ShootLat.between(51, 61) & h.ShootLong.between(-4, 13)]
pr['AphiaIDPredator'] = pr.AphiaIDPredator.astype(str)
pr = pr[pr.tblHaulID.isin(h.tblHaulID) & pr.AphiaIDPredator.isin(PRED) & (pr.Number == 1)]
pr['pred'] = pr.AphiaIDPredator.map(PRED)
p = p[p.tblPredatorInformationID.isin(pr.tblPredatorInformationID)].copy()
p['grp'] = pd.to_numeric(p.AphiaIDPrey, errors='coerce').fillna(0).astype(int).map(group)
p['Count'] = pd.to_numeric(p.Count, errors='coerce')
p['Weight'] = pd.to_numeric(p.Weight, errors='coerce')
p = p.merge(pr[['tblPredatorInformationID', 'pred', 'IndWgt', 'Length']], on='tblPredatorInformationID')

print('Individual stomachs in the box with >= 1 prey row, by predator:')
nst = p.groupby('pred').tblPredatorInformationID.nunique()
print(nst.to_string())
print()
print('Benthic invertebrate ITEMS per non-empty stomach (all benthic groups), median / mean / p90;')
print('and mean predator mass (g)')
b = p[p.grp.isin(BENTHIC)]
per = b.groupby(['pred', 'tblPredatorInformationID']).Count.sum()
allst = p.groupby(['pred', 'tblPredatorInformationID']).size()
per = per.reindex(allst.index, fill_value=0)
for pred, s in per.groupby(level=0):
    wt = pr[pr.pred == pred].IndWgt.mean()
    print(f'  {pred:15s} n {len(s):5d}  median {s.median():6.1f}  mean {s.mean():6.1f}  p90 {s.quantile(.9):6.1f}  '
          f'max {s.max():6.0f}   mean predator {wt:6.0f} g')
print()
print('Wet mass per item (g), by prey group, fresh rows (DigestionStage 1) if staged; '
      'median [p25-p75] (n rows)')
m = p[(p.Count > 0) & (p.Weight > 0) & (p.UnitWgt == 'g')].copy()
m['per_item'] = m.Weight / m.Count
fresh = m[m.DigestionStage.astype(str).str.strip() == '1']
rows = []
for grp in sorted(set(m.grp)):
    for lab, d in (('fresh', fresh[fresh.grp == grp]), ('all', m[m.grp == grp])):
        if len(d) >= 5:
            rows.append((grp, lab, d.per_item.median(), d.per_item.quantile(.25), d.per_item.quantile(.75), len(d)))
for r in rows:
    print(f'  {r[0]:20s} {r[1]:5s} {r[2]:8.3f} [{r[3]:.3f}-{r[4]:.3f}] (n {r[5]})')
print()
print('Same, by predator, fresh+all pooled, benthic groups only: median g/item (n)')
for pred in sorted(set(m.pred)):
    d = m[(m.pred == pred) & m.grp.isin(BENTHIC)]
    out = []
    for grp in sorted(set(d.grp)):
        dd = d[d.grp == grp]
        if len(dd) >= 5:
            out.append(f'{grp} {dd.per_item.median():.3f} ({len(dd)})')
    print(f'  {pred}: ' + '; '.join(out))
