"""Does whiting move shallower when the water column stands where it stands in the field? Frozen policies, fished world
(baseline maps unmuted in memory), the run's grid, CPU. After every tick the water column (predatory_fish) is
re-placed (biomass, reserve and juveniles, each at its own total), as in 263.2 (place94.py):
  learned  - not touched (control);
  within   - each bed class keeps this tick's learned total, spread evenly over the class's cells (removes the
             within-class co-location only);
  field    - the GOV group class shares 0.074 / 0.176 / 0.697 / 0.053 (resume 263.7), even within each class.
The re-placed state is what every policy observes at the next tick. Classes < 100 / 100-150 / 150-250 / >= 250 m.
Per year: whiting J and A class shares, M2 on whiting per predator and class (J | A; predation._stage_harvest wrapped),
whiting x benthivore and whiting x water-column Lloyd's L (cells / classes only / within class, every 10 ticks),
year-mean biomass over start for whiting, water column, benthivores.
usage: RUN=<run> POL=<dir> CLS=<cls npy> [LIB=<yaml>] whiting_wc_replace.py <arm> <years>"""
import os,sys,json,copy
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
arm=sys.argv[1]; years=int(sys.argv[2]); S=os.path.dirname(os.path.abspath(__file__))
assert arm in ('learned','within','field')
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
from lib.environments.ecosystem_env import predation
torch.set_num_threads(3)
cfg=json.load(open('results/'+os.environ['RUN']+'/progress/config.json')); seed=cfg['seed']
keys=("project","library","grid","mortality","migration","mortality_multiplier","mass_balance","currents","tick_hours")
eco={k:cfg[k] for k in keys if k in cfg}
if os.environ.get('LIB'): eco['library']=os.environ['LIB']
with evaluation_randomness(seed): env=build_inference_env(eco,seed)
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,os.environ['POL'],verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
cls=np.load(os.environ['CLS']); K=4; masks=[cls==k for k in range(K)]; f=np.array([m.mean() for m in masks])
pc=lambda x: np.array([np.asarray(x,dtype=float)[m].sum() for m in masks])
ids=list(env.global_fg_order); dm=list(env.dm_ids); WJ=ids.index('whiting'); W=env.fgs['whiting']
WC=env.fgs['predatory_fish']; BEN=env.fgs['benthivorous_fish']
dw=dm.index('whiting')
obs=[ids[j] for j in env.obs_others_idx[dw]] if hasattr(env,'obs_others_idx') else '?'
print('arm',arm,'| library',eco.get('library'),'| whiting observes',obs,'| water column staged',getattr(WC,'juvenile',None) is not None,flush=True)
preds=[i for i in range(len(dm)) if env.eat_static_mask[i,WJ]>0]
FIELD=np.array([0.074,0.176,0.697,0.053])
def replace():
    arrs=[('biomass',WC.biomass),('energy_reserve',WC.energy_reserve)]
    if getattr(WC,'juvenile',None) is not None: arrs.append(('juvenile',WC.juvenile))
    b=np.asarray(WC.biomass,dtype=float)
    for name,a in arrs:
        a=np.asarray(a,dtype=float); tot=a.sum(); new=np.zeros_like(a)
        for k,m in enumerate(masks):
            if arm=='within': new[m]=a[m].sum()/m.sum()
            else: new[m]=tot*FIELD[k]/m.sum()
        setattr(WC,name,new.astype(getattr(WC,name).dtype))
acc={}
def reset():
    acc.update(J=np.zeros(K),A=np.zeros(K),tJ=np.zeros((len(preds),K)),tA=np.zeros((len(preds),K)),n=0,
               wc=np.zeros(K),LB=np.zeros(3),LW=np.zeros(3),nl=0,bw=0.,bwc=0.,bb=0.)
reset()
osh=predation._stage_harvest
def sh(env_,demand,actual_intake,parts):
    ai,juv=osh(env_,demand,actual_intake,parts)
    b=np.asarray(W.biomass,dtype=float); j=np.minimum(np.asarray(W.juvenile,dtype=float),b)
    acc['J']+=pc(j); acc['A']+=pc(b-j); acc['n']+=1
    tot=ai[:,WJ]; tj=juv.get(WJ,np.zeros_like(tot))
    for r,i in enumerate(preds):
        acc['tJ'][r]+=pc(tj[i]); acc['tA'][r]+=pc(tot[i]-tj[i])
    return ai,juv
predation._stage_harvest=sh
def L(x,y):
    cell=(x*y).mean()/(x.mean()*y.mean())
    mx=np.array([x[m].mean() for m in masks]); my=np.array([y[m].mean() for m in masks])
    clas=(f*mx*my).sum()/((f*mx).sum()*(f*my).sum()); return np.array([cell,clas,cell/clas])
b0={k:float(np.asarray(env.fgs[k].biomass,dtype=float).sum()) for k in ('whiting','predatory_fish','benthivorous_fish')}
out=[]; fm=lambda v,p=3: ' / '.join(f'{x:.{p}f}' for x in v)
with evaluation_randomness(seed):
    for y in range(years):
        reset()
        for t in range(1460):
            env.tick()
            if arm!='learned': replace()
            wb=np.asarray(W.biomass,dtype=float); cb=np.asarray(WC.biomass,dtype=float); bb=np.asarray(BEN.biomass,dtype=float)
            acc['wc']+=pc(cb)/cb.sum(); acc['bw']+=wb.sum(); acc['bwc']+=cb.sum(); acc['bb']+=bb.sum()
            if t%10==0: acc['LB']+=L(wb,bb); acc['LW']+=L(wb,cb); acc['nl']+=1
        J=np.maximum(acc['J'],1e-12); A=np.maximum(acc['A'],1e-12)
        rec=dict(y=y+1,share_J=(acc['J']/acc['J'].sum()).tolist(),share_A=(acc['A']/acc['A'].sum()).tolist(),
                 wc_share=(acc['wc']/1460).tolist(),L_benth=(acc['LB']/acc['nl']).tolist(),L_wc=(acc['LW']/acc['nl']).tolist(),
                 ratio=dict(whiting=acc['bw']/1460/b0['whiting'],water_column=acc['bwc']/1460/b0['predatory_fish'],benthivores=acc['bb']/1460/b0['benthivorous_fish']),pred={})
        fJ=acc['J'].sum()/(acc['J'].sum()+acc['A'].sum())
        print(f"year {y+1} whiting J {fm(rec['share_J'])} | A {fm(rec['share_A'])} | water column {fm(rec['wc_share'])}",flush=True)
        print(f"  mean B / start: whiting {rec['ratio']['whiting']:.3f} water column {rec['ratio']['water_column']:.3f} benthivores {rec['ratio']['benthivores']:.3f} | J fraction {fJ:.3f}",flush=True)
        print(f"  L whiting x benthivores {fm(rec['L_benth'],2)} | x water column {fm(rec['L_wc'],2)} (cells / classes only / within)",flush=True)
        tot=0.
        for r,i in enumerate(preds):
            mj=acc['tJ'][r]/J*1460; ma=acc['tA'][r]/A*1460; g=(acc['tJ'][r].sum()+acc['tA'][r].sum())/(J.sum()+A.sum())*1460; tot+=g
            rec['pred'][dm[i]]=dict(M2_J=mj.tolist(),M2_A=ma.tolist(),M2=g)
            print(f"  M2 {dm[i][:18]:18s} group {g:.4f} | J {fm(mj)} | A {fm(ma)}",flush=True)
        rec['M2']=tot; print(f"  M2 total {tot:.4f}",flush=True)
        out.append(rec)
json.dump(out,open(f"{S}/../tables/whiting_wc_replace_{arm}_{os.path.basename(os.environ['POL'].rstrip('/'))}.json",'w'))
