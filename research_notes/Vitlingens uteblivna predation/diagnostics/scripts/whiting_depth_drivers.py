"""What each bed-depth class offers whiting under frozen policies (fished world, baseline maps unmuted in memory, the
run's grid, CPU). Classes < 100 / 100-150 / 150-250 / >= 250 m (cls npy). Per class and year, per tonne of whiting in
the class (time-integrated):
  share      whiting biomass share
  F, M2      impact and predation losses /yr
  pop        net of the population step (growth, M1, starvation, emigration, recruit import) /yr
  move       net movement + currents /yr (in - out)
  intake     taken energy MJ/t/yr; pot = uncapped demand energy (before the reserve-room cap) MJ/t/yr
  pot_<prey> share of the uncapped demand energy per prey (encounter potential incl. depth access and hiding)
  fill       reserve fill
  benth      benthivore density relative to its grid mean
The temperature multipliers are per group and tick, not per cell (interactions.metabolism_temperature), so cost per
tonne does not differ between classes except through the action mix.
usage: RUN=<run> POL=<dir> CLS=<cls npy> [LIB=<yaml>] whiting_depth_drivers.py <years>"""
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
from lib.environments.ecosystem_env import environment as envmod, predation, movement
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
W='whiting'; w=env.fgs[W]; dm=list(env.dm_ids); dw=dm.index(W); ids=list(env.global_fg_order)
menu=[j for j in range(len(ids)) if env.handling_time_mat[dw,j]>0]
print('library',eco.get('library'),'| whiting menu',[ids[j] for j in menu],flush=True)
def bw(): return np.asarray(w.biomass,dtype=float).copy()
def per_class(x): return np.array([x[m].sum() for m in masks])
Z=lambda: np.zeros(K)
acc={}
def reset():
    acc.update(B=Z(),F=Z(),M2=Z(),pop=Z(),move=Z(),intake=Z(),pot=Z(),fill=Z(),benth=Z(),potp=np.zeros((len(menu),K)))
reset()
def wrap(mod,name,key):
    f=getattr(mod,name)
    def g(*a,**k):
        b0=bw(); r=f(*a,**k); acc[key]+=per_class(b0-bw()) if key in('F','M2') else per_class(bw()-b0); return r
    setattr(mod,name,g)
wrap(envmod.impact_module,'apply_impact_mortality','F')
wrap(envmod.predation,'apply_predation','M2')
wrap(envmod.population_change,'apply_population_change','pop')
for mod,name in ((envmod.movement,'apply_movement'),(envmod.current_module,'apply_currents')):
    wrap(mod,name,'move')
orc=predation._reserve_room_cap
def rc(env_,demand,predator_biomass,reserve_offset):
    out=orc(env_,demand,predator_biomass,reserve_offset)
    e=demand[dw]*env_.energy_gain_mat[dw][:,None,None]
    if reserve_offset is not None: e=e+demand[dw]*env_.assimilation_mat[dw][:,None,None]*reserve_offset
    e=np.asarray(e,dtype=float); acc['pot']+=per_class(e.sum(axis=0))
    for i,j in enumerate(menu): acc['potp'][i]+=per_class(e[j])
    return out
predation._reserve_room_cap=rc
oec=movement.apply_energy_costs
def ec(env_,actions):
    acc['intake']+=per_class(np.asarray(w.temp_energy_gains,dtype=float))
    b=bw(); acc['B']+=per_class(b)
    acc['fill']+=per_class(np.asarray(w.energy_reserve,dtype=float))/max(w.max_energy_reserve,1e-9)
    q=np.asarray(env_.fgs['benthivorous_fish'].biomass,dtype=float); acc['benth']+=per_class(q/max(q.mean(),1e-12)*b)
    return oec(env_,actions)
envmod.movement.apply_energy_costs=ec
TY=1460
out=[]
with evaluation_randomness(seed):
    for y in range(years):
        reset()
        for t in range(TY): env.tick()
        B=acc['B']; Bt=np.maximum(B,1e-12)/TY*TY  # biomass-ticks per class
        per=lambda x: x/np.maximum(B,1e-12)*TY      # per tonne-year
        rec=dict(y=y+1,share=(B/B.sum()).tolist(),F=per(acc['F']).tolist(),M2=per(acc['M2']).tolist(),pop=per(acc['pop']).tolist(),
                 move=per(acc['move']).tolist(),intake=per(acc['intake']).tolist(),pot=per(acc['pot']).tolist(),
                 fill=(acc['fill']/np.maximum(B,1e-12)).tolist(),benth=(acc['benth']/np.maximum(B,1e-12)).tolist(),
                 potp={ids[j]:(acc['potp'][i]/np.maximum(acc['pot'],1e-12)).tolist() for i,j in enumerate(menu)})
        out.append(rec)
        f=lambda v,p=2: ' / '.join(f'{x:.{p}f}' for x in v)
        print(f"year {y+1} share {f(rec['share'])} | F {f(rec['F'],3)} | M2 {f(rec['M2'],3)} | pop {f(rec['pop'])} | move {f(rec['move'])}",flush=True)
        print(f"        intake MJ/t/yr {f(rec['intake'],0)} | uncapped {f(rec['pot'],0)} | fill {f(rec['fill'])} | benthivores x mean {f(rec['benth'])}",flush=True)
        print('        uncapped shares '+' | '.join(f"{p[:6]} {f(v)}" for p,v in rec['potp'].items()),flush=True)
json.dump(out,open(f"{S}/../tables/whiting_depth_{os.path.basename(os.environ['POL'].rstrip('/'))}.json",'w'))
