import json, math, sys
import numpy as np
from matplotlib.path import Path
fn, depthfn = sys.argv[1], sys.argv[2]
key = sys.argv[3] if len(sys.argv) > 3 else 'folk_7cl_txt'
# SWEREF99 TM inverse (Gauss-Kruger, GRS80, lon0 15, k0 0.9996, FE 500000)
def sweref_to_wgs(E, N):
    a=6378137.0; f=1/298.257222101; lon0=math.radians(15); k0=0.9996; FN=0; FE=500000
    e2=f*(2-f); n=f/(2-f); ah=a/(1+n)*(1+n*n/4+n**4/64)
    d1=n/2-2*n*n/3+37*n**3/96-n**4/360; d2=n*n/48+n**3/15-437*n**4/1440
    d3=17*n**3/480-37*n**4/840; d4=4397*n**4/161280
    xi=(N-FN)/(k0*ah); eta=(E-FE)/(k0*ah)
    xp=xi-d1*math.sin(2*xi)*math.cosh(2*eta)-d2*math.sin(4*xi)*math.cosh(4*eta)-d3*math.sin(6*xi)*math.cosh(6*eta)-d4*math.sin(8*xi)*math.cosh(8*eta)
    ep=eta-d1*math.cos(2*xi)*math.sinh(2*eta)-d2*math.cos(4*xi)*math.sinh(4*eta)-d3*math.cos(6*xi)*math.sinh(6*eta)-d4*math.cos(8*xi)*math.sinh(8*eta)
    phis=math.asin(math.sin(xp)/math.cosh(ep)); dl=math.atan(math.sinh(ep)/math.cos(xp))
    A=e2+e2**2+e2**3+e2**4; B=-(7*e2**2+17*e2**3+30*e2**4)/6; C=(224*e2**3+889*e2**4)/120; D=-(4279*e2**4)/1260
    s=math.sin(phis)
    lat=phis+s*math.cos(phis)*(A+B*s*s+C*s**4+D*s**6)
    return math.degrees(lat), math.degrees(lon0+dl)
def merc(lon, lat): return lon*20037508.34/180, math.log(math.tan((90+lat)*math.pi/360))*6378137
d=json.load(open(fn))
polys=[]
for f in d['features']:
    lab=f['properties'][key]
    g=f['geometry']; cs=g['coordinates'] if g['type']=='MultiPolygon' else [g['coordinates']]
    for poly in cs:
        outer=Path(np.array(poly[0])); holes=[Path(np.array(h)) for h in poly[1:]]
        polys.append((lab,outer,holes))
# 3x3 subsamples per cell
E0,N0=211000,6474000
labels=np.empty((60,60),dtype=object); frac={}
pts=[];idx=[]
for r in range(60):
    for c in range(60):
        for i in range(3):
            for j in range(3):
                E=E0+c*1000+(j+0.5)*1000/3; N=N0-r*1000-(i+0.5)*1000/3
                la,lo=sweref_to_wgs(E,N); pts.append(merc(lo,la)); idx.append((r,c))
pts=np.array(pts); lab=np.array(['none']*len(pts),dtype=object)
for l,outer,holes in polys:
    m=outer.contains_points(pts)
    for h in holes: m&=~h.contains_points(pts)
    lab[m]=l
from collections import Counter, defaultdict
dep=np.load(depthfn)['depth']
tot=Counter(lab); n=len(lab)
print('whole grid:',{k:round(v/n,3) for k,v in tot.items()})
bands=[(0,100),(100,150),(150,200),(200,250),(250,300),(300,600)]
for lo,hi in bands:
    sel=[k for k,(r,c) in enumerate(idx) if lo<=dep[r,c]<hi]
    cc=Counter(lab[sel]); m=len(sel)
    print(f'{lo}-{hi} m: share of grid {m/n:.3f}', {k:round(v/m,3) for k,v in cc.items()})
