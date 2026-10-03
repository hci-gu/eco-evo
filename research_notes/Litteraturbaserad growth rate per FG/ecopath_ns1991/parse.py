import re, json
L=open('t.txt').read().split('\n')[1283:1920]
D={}  # (prey,pred)->dc
hdr=None
for ln in L:
    if 'Predator Prey' in ln:
        hdr=[(m.end(), int(m.group())) for m in re.finditer(r'\b\d+\b', ln)]
        continue
    m=re.match(r'^\s*(\d{1,2})\s+[A-Za-z]', ln)
    if not m or hdr is None: continue
    prey=int(m.group(1))
    if prey>68: continue
    for t in re.finditer(r'\d\.\d{3}', ln):
        e=t.end()
        col=min(hdr, key=lambda h: abs(h[0]-e))
        if abs(col[0]-e)>4: print('WARN', prey, t.group(), e, col); 
        D[f"{prey},{col[1]}"]=float(t.group())
json.dump(D, open('diet.json','w'))
# column sums check
from collections import defaultdict
s=defaultdict(float)
for k,v in D.items(): s[int(k.split(',')[1])]+=v
print({k:round(v,3) for k,v in sorted(s.items())})
