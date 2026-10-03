import re, json
from collections import defaultdict
txt=open('bb.html').read()
pages=txt.split('<page ')[1:]
W=re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>')
D={}
for pi,p in enumerate(pages):
    ws=[(float(a),float(b),float(c),float(d),t) for a,b,c,d,t in W.findall(p)]
    if not any('Diet' in w[4] for w in ws): 
        print('page',pi,'no diet'); continue
    # header: word 'Prey' y
    prey_w=[w for w in ws if w[4]=='Prey']
    if not prey_w: print('no hdr',pi); continue
    hy=prey_w[0][1]
    hdr=[(w,int(w[4])) for w in ws if abs(w[1]-hy)<2 and re.fullmatch(r'\d+',w[4])]
    hx=[((w[0]+w[2])/2,n) for w,n in hdr]
    # row labels: integers at left of 'Prey' x
    px=prey_w[0][0]
    labels=[(w[1],int(w[4])) for w in ws if re.fullmatch(r'\d+',w[4]) and w[2]<px and w[1]>hy+5 and int(w[4])<=68]
    nums=[w for w in ws if re.fullmatch(r'\d\.\d{3}',w[4])]
    for w in nums:
        yc=w[1]
        lab=min(labels,key=lambda l:abs(l[0]-yc))
        if abs(lab[0]-yc)>3: print('ywarn',pi,w,lab); continue
        xc=(w[0]+w[2])/2
        col=min(hx,key=lambda h:abs(h[0]-xc))
        if abs(col[0]-xc)>12: print('xwarn',pi,w[4],xc,col)
        D[f"{lab[1]},{col[1]}"]=float(w[4])
json.dump(D,open('diet.json','w'))
s=defaultdict(float)
for k,v in D.items(): s[int(k.split(',')[1])]+=v
print({k:round(v,3) for k,v in sorted(s.items())})
