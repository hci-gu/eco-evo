import json
P={int(k):v for k,v in json.load(open('params.json')).items()}
P[15].update(B=0.222,PB=2.36,QB=6.58); P[17].update(B=0.284,PB=2.0,QB=5.39)
P[59]=dict(name='Small mobile epifauna',B=30,PB=1.9,QB=5.4286); P[62]=dict(name='Meiofauna',B=4.1071,PB=35,QB=125)
P[63]=dict(name='Benthic microflora',B=0.105,PB=9470,QB=18940); P[64]=dict(name='Planktonic microflora',B=1.46,PB=571,QB=1142)
P[65]=dict(name='Phytoplankton',B=7.5,PB=286,QB=None)
D={tuple(map(int,k.split(','))):v for k,v in json.load(open('diet.json')).items()}
allp=[j for j in P if P[j].get('QB')]
def m2(prey,preds): return sum(D.get((prey,j),0)*P[j]['QB']*P[j]['B'] for j in preds if P.get(j) and P[j].get('QB'))/P[prey]['B']
# catches (landings+discards, t) from Table 3.5, area 570000 km2
C={13:29658,14:67431+3366,15:13886,16:84018+19416,17:35992,18:50046+4243,19:31504,20:66861,21:2337+662,22:34770,23:155895,24:18867+1828,25:16131,
   28:86214,29:487920+1854,30:99579+4218,31:197163+117941,32:98268+6,33:842574,48:5757+90}
A=570000.
def F(i): return C.get(i,0)/(P[i]['B']*A)
ZOO=[51,52]; BEN=list(range(54,62))
