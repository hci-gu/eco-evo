"""Who eats whiting on which beds, and which growth cap binds (frozen policies, fished world, baseline maps unmuted in
memory, the run's grid, CPU). Classes < 100 / 100-150 / 150-250 / >= 250 m (cls npy).
Per year:
  - M2 on whiting per predator and class, juveniles (J) and adults (A) apart: tonnes taken / tonne-years of the stage
    in the class (from predation._stage_harvest's per-cell intake, wrapped; nothing in the engine changed);
  - each predator's biomass share per class (time mean, every tick at predation);
  - the four terms of whiting's material growth law (movement.income_growth, recomputed from its inputs):
    ceiling B g (s - u) p, material kappa_m I p / ec, surplus max(I - C, 0), store above maintenance R' - u ME B';
    per class the sum of each (MJ per tonne-year of whiting) and the biomass share of the cells where it binds (is
    the minimum), plus the paid growth.
usage: RUN=<run> POL=<dir> CLS=<cls npy> [LIB=<yaml>] whiting_m2_caps.py <years>"""
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
from lib.environments.ecosystem_env import predation, movement
from lib.world import energy_balance
torch.set_num_threads(3)
cfg=json.load(open('results/'+os.environ['RUN']+'/progress/config.json')); seed=cfg['seed']
keys=("project","library","grid","mortality","migration","mortality_multiplier","mass_balance","currents","tick_hours")
eco={k:cfg[k] for k in keys if k in cfg}
if os.environ.get('LIB'): eco['library']=os.environ['LIB']
with evaluation_randomness(seed): env=build_inference_env(eco,seed)
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,os.environ['POL'],verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
cls=np.load(os.environ['CLS']); K=4; masks=[cls==k for k in range(K)]
pc=lambda x: np.array([np.asarray(x,dtype=float)[m].sum() for m in masks])
ids=list(env.global_fg_order); dm=list(env.dm_ids); WJ=ids.index('whiting'); dw=dm.index('whiting'); W=env.fgs['whiting']
preds=[i for i in range(len(dm)) if env.eat_static_mask[i,WJ]>0]
print('library',eco.get('library'),'| predators of whiting',[dm[i] for i in preds],flush=True)
acc={}
TERMS=('ceiling','material','surplus','store')
def reset():
    acc.update(J=np.zeros(K),A=np.zeros(K),tJ=np.zeros((len(preds),K)),tA=np.zeros((len(preds),K)),pB=np.zeros((len(preds),K)),n=0,
               B=np.zeros(K),term=np.zeros((4,K)),bind=np.zeros((4,K)),paid=np.zeros(K))
reset()
osh=predation._stage_harvest
def sh(env_,demand,actual_intake,parts):
    ai,juv=osh(env_,demand,actual_intake,parts)
    b=np.asarray(W.biomass,dtype=float); j=np.minimum(np.asarray(W.juvenile,dtype=float),b)
    acc['J']+=pc(j); acc['A']+=pc(b-j); acc['n']+=1
    tot=ai[:,WJ]; tj=juv.get(WJ,np.zeros_like(tot))
    for r,i in enumerate(preds):
        acc['tJ'][r]+=pc(tj[i]); acc['tA'][r]+=pc(tot[i]-tj[i])
        pb=np.asarray(env_.fgs[dm[i]].biomass,dtype=float); acc['pB'][r]+=pc(pb)/max(pb.sum(),1e-12)
    return ai,juv
predation._stage_harvest=sh
oig=movement.income_growth
def ig(env_,biomass,reserve,gains,cost,stationary_biomass,stationary_reserve):
    i=dw; fg=W
    b=biomass[i].astype(float); r=reserve[i].astype(float); g=gains[i].astype(float); c=cost[i].astype(float)
    sb=stationary_biomass[i].astype(float); sr=stationary_reserve[i].astype(float)
    me=float(env_.dm_max_energy_reserve[i]); u=float(fg.maintenance_level); ec=float(fg.params.get('energy_content',0.0) or 0.0)
    share=float(getattr(env_,'dm_material_share')[i]); grow=float(fg.growth_rate)
    gc=float(getattr(env_,'dm_growth_cost',np.zeros(env_.N_dm))[i])
    cap=b*me; fill=np.where(cap>0,r/np.where(cap>0,cap,1.0),0.0)
    price=np.asarray(energy_balance.tonne_price(np.float64(ec),np.float64(gc),fill,np.float64(me)),dtype=float)
    t=np.stack([b*grow*np.maximum(fill-u,0.0)*price, share*np.maximum(g,0.0)*price/ec,
                np.maximum(g-c,0.0), np.maximum(sr-u*me*sb,0.0)])
    arg=t.argmin(axis=0); fed=t.min(axis=0)>0
    acc['B']+=pc(b); acc['paid']+=pc(t.min(axis=0))
    for k in range(4):
        acc['term'][k]+=pc(t[k]); acc['bind'][k]+=pc(b*((arg==k)&fed))
    return oig(env_,biomass,reserve,gains,cost,stationary_biomass,stationary_reserve)
movement.income_growth=ig
out=[]
fm=lambda v,p=3: ' / '.join(f'{x:.{p}f}' for x in v)
with evaluation_randomness(seed):
    for y in range(years):
        reset()
        for t in range(1460): env.tick()
        n=acc['n']; J=np.maximum(acc['J'],1e-12); A=np.maximum(acc['A'],1e-12); TY=1460
        rec=dict(y=y+1,share_J=(acc['J']/acc['J'].sum()).tolist(),share_A=(acc['A']/acc['A'].sum()).tolist(),pred={})
        print(f"year {y+1} whiting class shares J {fm(rec['share_J'])} | A {fm(rec['share_A'])}",flush=True)
        print(f"  M2 /yr per predator and class (J | A) and the predator's biomass share per class",flush=True)
        for r,i in enumerate(preds):
            mj=acc['tJ'][r]/J*TY; ma=acc['tA'][r]/A*TY; ps=acc['pB'][r]/n
            rec['pred'][dm[i]]=dict(M2_J=mj.tolist(),M2_A=ma.tolist(),share=ps.tolist())
            print(f"    {dm[i][:18]:18s} J {fm(mj)} | A {fm(ma)} | biomass {fm(ps,2)}",flush=True)
        mj=acc['tJ'].sum(0)/J*TY; ma=acc['tA'].sum(0)/A*TY
        print(f"    {'all':18s} J {fm(mj)} | A {fm(ma)}",flush=True)
        B=np.maximum(acc['B'],1e-12)
        rec['terms']={TERMS[k]:(acc['term'][k]/B*TY).tolist() for k in range(4)}
        rec['bind']={TERMS[k]:(acc['bind'][k]/B).tolist() for k in range(4)}
        rec['paid']=(acc['paid']/B*TY).tolist()
        print(f"  growth law, MJ per tonne-year: "+' | '.join(f"{TERMS[k]} {fm(acc['term'][k]/B*TY,0)}" for k in range(4))+f" | paid {fm(acc['paid']/B*TY,0)}",flush=True)
        print(f"  binding (biomass share of fed cells): "+' | '.join(f"{TERMS[k]} {fm(acc['bind'][k]/B,2)}" for k in range(4)),flush=True)
        out.append(rec)
json.dump(out,open(f"{S}/../tables/whiting_m2_caps_{os.path.basename(os.environ['POL'].rstrip('/'))}.json",'w'))
