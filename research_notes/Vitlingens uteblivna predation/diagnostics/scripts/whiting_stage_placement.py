"""Whiting's juveniles (J) and adults (A = B - J) by bed depth under frozen policies (fished world, baseline maps unmuted
in memory, the run's grid, CPU). The model moves J with B under one policy (stages.py), so any difference between the
stages' placement comes from the uniform recruit import, graduation, emigration and the stage-specific losses.
Per year and field quarter (Q1 = Jan-Mar, Q3 = Jul-Sep on the daylight calendar), sampled every 10 ticks:
  J/B per class, class shares of J and of A (classes < 100 / 100-150 / 150-250 / >= 250 m, cls npy), shares of each
  stage in the < 100 m class's sub-bins < 60 / 60-80 / >= 80 m (depth npy), Lloyd's L of J and of A with the
  benthivores (cells / classes only / within class).
Per year, every tick: predation loss per tonne-year of J and of A per class (M2_J, M2_A).
usage: RUN=<run> POL=<dir> CLS=<cls npy> DEPTH=<depth npy> [LIB=<yaml>] whiting_stage_placement.py <years>"""
import os,sys,json,copy
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
years=int(sys.argv[1]); S=os.path.dirname(os.path.abspath(__file__))
import lib.world.impact_maps as im
orig=im.project_baseline_maps
def patched(project,project_path,grid_size,verbose=False):
    p=copy.deepcopy(project)
    for iv in (p or {}).get('impact_variables',[]) or []:
        if isinstance(iv,dict): iv.pop('baseline_map_muted',None)
    return orig(p,project_path,grid_size,verbose=verbose)
im.project_baseline_maps=patched
import numpy as np,torch
from lib.runners.training_progress import build_inference_env,evaluation_randomness
from inference import load_policies_and_stats
from lib.environments.ecosystem_env import environment as envmod
torch.set_num_threads(3)
cfg=json.load(open('results/'+os.environ['RUN']+'/progress/config.json')); seed=cfg['seed']
keys=("project","library","grid","mortality","migration","mortality_multiplier","mass_balance","currents","tick_hours")
eco={k:cfg[k] for k in keys if k in cfg}
if os.environ.get('LIB'): eco['library']=os.environ['LIB']
with evaluation_randomness(seed): env=build_inference_env(eco,seed)
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,os.environ['POL'],verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
cls=np.load(os.environ['CLS']); depth=np.load(os.environ['DEPTH']); K=4
f=np.array([(cls==k).mean() for k in range(K)])
sub=[(cls==0)&(depth<60),(cls==0)&(depth>=60)&(depth<80),(cls==0)&(depth>=80)]
print('library',eco.get('library'),'| cells per class',[int((cls==k).sum()) for k in range(K)],'| < 100 m sub-bins',[int(m.sum()) for m in sub],flush=True)
dl=None
for g in env.fgs.values(): dl=(getattr(g,'params',{}) or {}).get('daylight') or dl
start_tick=int(dl['start_tick']) if dl else 0; th=int(dl['tick_hours']) if dl else 6
W=env.fgs['whiting']; BEN='benthivorous_fish'
def stages():
    b=np.asarray(W.biomass,dtype=float); j=np.minimum(np.asarray(W.juvenile,dtype=float),b); return j,b-j
pc=lambda x: np.array([x[cls==k].sum() for k in range(K)])
acc={}
def reset(): acc.update(J=np.zeros(K),A=np.zeros(K),mJ=np.zeros(K),mA=np.zeros(K))
reset()
op=envmod.predation.apply_predation
def pred(*a,**k):
    j0,a0=stages(); r=op(*a,**k); j1,a1=stages()
    acc['J']+=pc(j0); acc['A']+=pc(a0); acc['mJ']+=pc(j0-j1); acc['mA']+=pc(a0-a1); return r
envmod.predation.apply_predation=pred
def L(x,y):
    cell=(x*y).mean()/(x.mean()*y.mean())
    mx=np.array([x[cls==k].mean() for k in range(K)]); my=np.array([y[cls==k].mean() for k in range(K)])
    clas=(f*mx*my).sum()/((f*mx).sum()*(f*my).sum()); return np.array([cell,clas,cell/clas])
out=[]
with evaluation_randomness(seed):
    for y in range(years):
        reset(); q=lambda: dict(n=0,jb=np.zeros(K),sJ=np.zeros(K),sA=np.zeros(K),subJ=np.zeros(3),subA=np.zeros(3),LJ=np.zeros(3),LA=np.zeros(3))
        qa={k:q() for k in ('year','Q1','Q3')}
        for t in range(1460):
            env.tick()
            if t%10: continue
            day=((start_tick+y*1460+t+1)*th/24.0)%365
            j,a=stages(); b=np.asarray(env.fgs[BEN].biomass,dtype=float)
            jb=pc(j)/np.maximum(pc(j+a),1e-12)
            for k in ['year']+(['Q1'] if day<90 else ['Q3'] if 181<=day<273 else []):
                s=qa[k]; s['n']+=1; s['jb']+=jb; s['sJ']+=pc(j)/j.sum(); s['sA']+=pc(a)/a.sum()
                s['subJ']+=np.array([j[m].sum() for m in sub])/max(j[cls==0].sum(),1e-12)
                s['subA']+=np.array([a[m].sum() for m in sub])/max(a[cls==0].sum(),1e-12)
                s['LJ']+=L(j,b); s['LA']+=L(a,b)
        rec=dict(y=y+1,M2_J=(acc['mJ']/np.maximum(acc['J'],1e-12)*1460).tolist(),M2_A=(acc['mA']/np.maximum(acc['A'],1e-12)*1460).tolist())
        fm=lambda v,p=3: ' / '.join(f'{x:.{p}f}' for x in v)
        print(f"year {y+1} M2 per tonne-year  J {fm(rec['M2_J'])} | A {fm(rec['M2_A'])}",flush=True)
        for k,s in qa.items():
            n=s['n']
            if not n: continue
            r={x:(s[x]/n).tolist() for x in ('jb','sJ','sA','subJ','subA','LJ','LA')}; rec[k]=r
            print(f"  {k:4s} J/B {fm(r['jb'])} | J shares {fm(r['sJ'])} | A shares {fm(r['sA'])}",flush=True)
            print(f"       < 100 m sub-bins (< 60 / 60-80 / >= 80 m) J {fm(r['subJ'])} A {fm(r['subA'])} | L x benthivores (cells / classes / within) J {fm(r['LJ'],2)} A {fm(r['LA'],2)}",flush=True)
        out.append(rec)
json.dump(out,open(f"{S}/../tables/whiting_stage_{os.path.basename(os.environ['POL'].rstrip('/'))}.json",'w'))
