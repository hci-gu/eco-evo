# Figure readings (Sorlie et al. 2020 Fig 3; Brown & Pierce 1998 Fig 4) typed in by hand; reads no downloaded file.
# Lengths in mm; mass ~ K L^3; shares by mass and number in <16, 16-25, >25 cm. Own calc.
# Truncated-lognormal version: each species' length distribution cut at its whisker range (Sorlie 2020 Fig 3 readings).
import math
from statistics import NormalDist
N=NormalDist(); cuts=(160.0,250.0)
def share(med,q1,q3,n,lo,hi,K=0.85):
    mu=math.log(med); s=max((math.log(q3)-math.log(q1))/1.349,0.03)
    def F(x,m): return N.cdf((math.log(x)-m)/s)
    out={}
    for tag,m in (('num',mu),('mass',mu+3*s*s)):
        a,b=F(lo,m),F(hi,m); z=b-a
        c1=(F(min(max(cuts[0],lo),hi),m)-a)/z; c2=(F(min(max(cuts[1],lo),hi),m)-a)/z
        out[tag]=(c1,c2-c1,1-c2)
    # mean mass per fish: E[L^3] on truncated lognormal
    a,b=F(lo,mu+3*s*s),F(hi,mu+3*s*s); a0,b0=F(lo,mu),F(hi,mu)
    EL3=math.exp(3*mu+4.5*s*s)*(b-a)/(b0-a0)
    return n*K/100*EL3/1000, out
gad={'Norway pout':(99,77,108,43,36,175),'Blue whiting':(114,112,151,4,110,258),'Poor cod':(117,92,134,113,31,193),
 'Saithe':(126,114,152,38,88,197),'Bib':(149,104,178,52,14,247),'Whiting':(151,131,188,96,84,252),'Hake':(144,129,198,8,100,255),
 'Haddock':(232,184,295,22,63,399),'Pollack':(285,234,340,17,180,548),'Cod':(286,275,312,6,235,330)}
flat={'Witch':(144,112,218,39,58,314,0.65),'Long rough dab':(163,131,207,78,65,290,0.75),'Dab':(237,195,254,21,80,317,0.9),
 'Plaice':(246,238,250,24,180,330,0.9),'Lemon sole':(267,259,286,21,205,350,0.9),'Flounder':(315,312,318,1,310,320,1.0)}
def run(d,label,extra=()):
    T=0;C=[0,0,0];Nn=0;Cn=[0,0,0]
    for k,v in d.items():
        K=v[6] if len(v)>6 else 0.85
        m,o=share(*v[:6],K=K); T+=m; Nn+=v[3]
        for i in range(3): C[i]+=m*o['mass'][i]; Cn[i]+=v[3]*o['num'][i]
        print(f'  {k:15s} n={v[3]:3d} kg={m:6.2f} mass:'+' '.join(f'{x:.2f}' for x in o['mass'])+'  num:'+' '.join(f'{x:.2f}' for x in o['num']))
    for l in extra:
        w=0.85/100*(l/10)**3/1000; T+=w; Nn+=1; i=0 if l<cuts[0] else 1 if l<cuts[1] else 2; C[i]+=w; Cn[i]+=1
    print(label,'MASS <16/16-25/>25:',' '.join(f'{x/T:.2f}' for x in C),'| NUM:',' '.join(f'{x/Nn:.2f}' for x in Cn), f'| total {T:.1f} kg')
    return T,C
tg,cg=run(gad,'GADOIDS',extra=(273,286,328,341,348))
tf,cf=run(flat,'FLATFISH')
print('GADOID+FLAT MASS:',' '.join(f'{(a+b)/(tg+tf):.2f}' for a,b in zip(cg,cf)))
small={k:v for k,v in gad.items() if k in ('Norway pout','Blue whiting','Poor cod','Bib','Whiting','Saithe','Hake')}
run(small,'GADOIDS without haddock/pollack/cod',extra=(273,286,328,341,348))
