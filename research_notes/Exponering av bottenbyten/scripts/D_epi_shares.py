import csv, sys, collections
# classify benthic prey taxa into model epifauna sub-pools
def cls(t):
    t=t.lower()
    if 'nephrops' in t: return 'nephrops'
    if 'pandalus borealis' in t: return 'pandalus_borealis'
    if any(k in t for k in ['crangon','philocheras','pontophilus','processa','pandalus montagui','pandalidae','caridea','eualus','spirontocaris','natantia','crangonidae','processidae','hippolyt','palaemon']): return 'small_shrimps'
    if any(k in t for k in ['mysi','erythrops','lophogaster','boreomysis','mysis']): return 'mysids'
    if any(k in t for k in ['brachyura','portun','liocarcinus','goneplax','atelecyclus','hyas','corystes','cancer','carcinus','macropipus','pagur','anapagur','paguridae','galathe','munida','anomura','decapoda reptantia','crab']): return 'crabs_hermits_munida'
    return None
def run(path, sp_col, area_col, per_col, pool_col, taxon_col, g_col, filt):
    agg=collections.defaultdict(lambda: collections.defaultdict(float))
    with open(path) as f:
        r=csv.DictReader(f)
        for row in r:
            if row[pool_col]!='B': continue
            key=(row[sp_col],row[area_col],row[per_col])
            if not filt(key): continue
            c=cls(row[taxon_col]) or 'other_benthos'
            agg[key][c]+=float(row[g_col])
    for key,d in sorted(agg.items()):
        tot=sum(d.values()); epi=sum(v for k,v in d.items() if k!='other_benthos')
        if epi<=0: continue
        print(key, 'B total g %.1f, epifauna g %.1f;'%(tot,epi), ', '.join('%s %.2f'%(k,v/epi) for k,v in sorted(d.items()) if k!='other_benthos'))
p=sys.argv[1]; q=sys.argv[2]
print('# whiting/predators table: shares WITHIN epifauna (incl. nephrops) of B grams')
run(p,'species','area','period','pool','taxon','g', lambda k: k[0] in ('whiting',) and k[1] in ('Skagerrak','Box','Kattegat','NNS','CSNS'))
print('# benthivores table')
with open(q) as f: hdr=next(csv.reader(f))
print('# header', hdr)
run(q,'species','area','period','pool','taxon','g', lambda k: k[0] in ('cod','haddock','whiting','long_rough_dab','lrd','plaice','dab','witch','lemon_sole') and k[1] in ('Skagerrak','Box','Kattegat','NNS','CSNS'))
