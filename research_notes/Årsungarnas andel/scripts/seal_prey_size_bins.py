# Figure readings (Sorlie et al. 2020 Fig 3; Brown & Pierce 1998 Fig 4) typed in by hand; reads no downloaded file.
# Lengths in mm; mass ~ K L^3; shares by mass and number in <16, 16-25, >25 cm. Own calc.
# Biomass share of seal-eaten fish by length class from figure readings (no downloaded file is read).
import math
from statistics import NormalDist
N=NormalDist()
cuts=(160.0,250.0)
def lognorm_share(med,q1,q3,n,K=0.85):
    mu=math.log(med); s=max((math.log(q3)-math.log(q1))/1.349,0.03)
    # mass-weighted (L^3) lognormal: shift mu by 3 s^2
    m=mu+3*s*s
    a=N.cdf((math.log(cuts[0])-m)/s); b=N.cdf((math.log(cuts[1])-m)/s)
    mass=n*K*1e-5*math.exp(3*mu+4.5*s*s)*1e0  # g per fish * n with L in mm -> (L/10)^3*K/100
    mass=n*K/100*math.exp(3*(mu-math.log(10))+4.5*s*s)
    numa=N.cdf((math.log(cuts[0])-mu)/s); numb=N.cdf((math.log(cuts[1])-mu)/s)
    return mass,(a,b-a,1-b),(numa,numb-numa,1-numb)
def pts(L,K=0.85):
    m=sum(K/100*(l/10)**3 for l in L); c=[0,0,0]
    for l in L:
        w=K/100*(l/10)**3; c[0 if l<cuts[0] else 1 if l<cuts[1] else 2]+=w
    return m,[x/m for x in c],None
# Sorlie 2020 Fig 3 readings (mm): median,Q1,Q3,n  (+ explicit outliers)
gad={'Norway pout':(99,77,108,43),'Blue whiting':(114,112,151,4),'Poor cod':(117,92,134,113),
 'Saithe':(126,114,152,38),'Bib':(149,104,178,52),'Whiting':(151,131,188,96),'Hake':(144,129,198,8),
 'Haddock':(232,184,295,22),'Pollack':(285,234,340,17),'Cod':(286,275,312,6)}
flat={'Witch':(144,112,218,39,0.65),'Long rough dab':(163,131,207,78,0.75),'Dab':(237,195,254,21,0.9),
 'Plaice':(246,238,250,24,0.9),'Lemon sole':(267,259,286,21,0.9),'Flounder':(315,312,318,1,1.0)}
def group(d,extra=None,label=''):
    tot=0; c=[0,0,0]; cn=[0,0,0]; ntot=0
    for k,v in d.items():
        K=v[4] if len(v)>4 else 0.85
        m,sh,nsh=lognorm_share(*v[:4],K=K); tot+=m; ntot+=v[3]
        for i in range(3): c[i]+=m*sh[i]; cn[i]+=v[3]*nsh[i]
        print(f'  {k:15s} n={v[3]:4d} mass={m:7.0f} g  mass<16/16-25/>25: '+' '.join(f'{x:.2f}' for x in sh))
    if extra:
        m,sh,_=pts(extra); tot+=m; ntot+=len(extra)
        for i in range(3): c[i]+=m*sh[i]; cn[i]+=sum(1 for l in extra if (l<cuts[0] if i==0 else (cuts[0]<=l<cuts[1] if i==1 else l>=cuts[1])))
        print(f'  outliers       n={len(extra)} mass={m:7.0f} g')
    print(label,'MASS share <16 / 16-25 / >25 cm:',' '.join(f'{x/tot:.2f}' for x in c),'| NUMBER share:',' '.join(f'{x/ntot:.2f}' for x in cn))
    return tot,c
print('Sorlie 2020 (Norwegian Skagerrak, no digestion correction)')
tg,cg=group(gad,extra=[273,286,328,341,348],label='GADOIDS')
tf,cf=group(flat,label='FLATFISH')
print('GADOID+FLAT MASS share:',' '.join(f'{(a+b)/(tg+tf):.2f}' for a,b in zip(cg,cf)))
# Brown & Pierce 1998 Fig 4 (Shetland, digestion-corrected), bins (lo,hi,%):
def bins(name,b,n,K=0.85):
    m=0; c=[0,0,0]; cn=[0,0,0]
    for lo,hi,p in b:
        mid=(lo+hi)/2*10; w=n*p/100*K/100*(mid/10)**3; m+=w
        i=0 if mid<cuts[0] else 1 if mid<cuts[1] else 2; c[i]+=w; cn[i]+=p
    print(f'  {name:12s} n={n:5d} mass={m/1000:7.1f} kg  mass<16/16-25/>25: '+' '.join(f'{x/m:.2f}' for x in c)+' | num: '+' '.join(f'{x/sum(cn):.2f}' for x in cn))
    return m,c
bp={'Cod':([(0,10,6),(10,20,3),(20,30,57),(30,35,14),(35,45,20)],51),
 'Haddock':([(10,15,13),(20,25,20),(25,30,23),(30,35,30),(35,40,7),(40,45,7)],30),
 'Saithe':([(5,10,.5),(10,15,1),(15,20,7),(20,25,31),(25,30,20),(30,35,24),(35,40,8),(40,45,3),(45,50,3),(55,65,1)],307),
 'Whiting':([(4,8,1),(8,12,3.5),(12,16,6.5),(16,20,13),(20,24,8),(24,28,6.5),(28,32,16),(32,36,18),(36,40,17.5),(40,44,9),(44,48,2),(48,52,.5)],1075),
 'Norway pout':([(9,12,1),(12,15,15),(15,18,54),(18,21,15),(21,24,9),(24,27,.5)],1216),
 'Poor cod':([(6,9,3),(9,12,23),(12,15,23),(15,18,16),(18,21,4),(21,24,16),(24,27,14),(27,30,2)],50)}
print('Brown & Pierce 1998 (Shetland, digestion-corrected)')
T=0; C=[0,0,0]
for k,(b,n) in bp.items():
    m,c=bins(k,b,n); T+=m; C=[x+y for x,y in zip(C,c)]
print('GADOIDS MASS share <16/16-25/>25:',' '.join(f'{x/T:.2f}' for x in C))
