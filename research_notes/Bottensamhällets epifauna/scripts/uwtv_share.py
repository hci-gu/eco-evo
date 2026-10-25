import sys; sys.path.insert(0,'/home/simon/eco-evo/tools')
from make_fishing_maps import wgs84_to_sweref99tm, E0, N0, CELL, GRID
import numpy as np
from PIL import Image
im=np.array(Image.open(sys.argv[1]).convert('RGB')).astype(int)
dep=np.load('/home/simon/eco-evo/maps/depth.npz')['depth']
H,W,_=im.shape
ys,xs=np.mgrid[0:H,0:W]
lon=10+(xs-175)/395; lat=58.5-(ys-118)/738
E,N=wgs84_to_sweref99tm(lon.ravel(),lat.ravel())
c=np.floor((E-E0)/CELL).astype(int); r=GRID-1-np.floor((N-N0)/CELL).astype(int)
p=im.reshape(-1,3)
grey=np.all(np.abs(p-230)<=8,axis=1); white=np.all(p>=248,axis=1)
ok=(c>=0)&(c<60)&(r>=0)&(r<60)
g=np.zeros((60,60)); w=np.zeros((60,60))
np.add.at(g,(r[ok&grey],c[ok&grey]),1); np.add.at(w,(r[ok&white],c[ok&white]),1)
sv=np.where(g+w>0,g/np.maximum(g+w,1),np.nan)
print('cells with info',np.isfinite(sv).sum(),'grid survey share',round(np.nanmean(sv),3))
for lo,hi in [(0,100),(100,150),(150,200),(200,250),(250,300),(300,600)]:
  m=(dep>=lo)&(dep<hi); print(lo,hi,'grid share',round(m.mean(),3),'survey share',round(np.nanmean(sv[m]),3))
np.save(sys.argv[2],sv)
# coarse ascii map, 6x6 km blocks
for R in range(0,60,4):
  print(''.join('#' if np.nanmean(sv[R:R+4,C:C+4])>0.5 else ('+' if np.nanmean(sv[R:R+4,C:C+4])>0.15 else '.') for C in range(0,60,3)))
