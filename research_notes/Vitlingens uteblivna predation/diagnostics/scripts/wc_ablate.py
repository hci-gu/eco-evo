"""Which observation channels carry the water column's bed-depth choice? Frozen policies, the run's probe world
(baseline maps as the manifest has them: muted = unfished, as in training), the run's grid, CPU. Only the water
column's (predatory_fish) observation is altered; every other group and the engine are untouched.
Arms (channels set to their grid mean each tick):
  base          - nothing altered;
  no_direction  - the four neighbour slots read as the own cell (local state kept, no gradient);
  no_energy     - the own reserve fill flattened over occupied cells;
  no_own        - the own biomass, centre and neighbours;
  no_<group>    - that observed group's channels (biomass and its juvenile channel), centre and neighbours;
  no_others     - every other group's channels;
  only_energy   - every biomass channel (own and others) flattened, the energy kept.
Impact channels and the light level are left alone. Reports the water column's class shares (< 100 / 100-150 /
150-250 / >= 250 m) per year, its diet shares, ration and fill in year 2, and whiting's class shares.
usage: RUN=<run> POL=<dir> CLS=<cls npy> wc_ablate.py <arm> <out jsonl>"""
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
import numpy as np,torch
from lib.runners.training_progress import build_inference_env,evaluation_randomness
from lib.environments.ecosystem_env import observations
from inference import load_policies_and_stats
arm,out=sys.argv[1],sys.argv[2]
torch.set_num_threads(1)
cfg=json.load(open('results/'+os.environ['RUN']+'/progress/config.json')); seed=cfg['seed']
keys=("project","library","grid","mortality","migration","mortality_multiplier","mass_balance","currents","tick_hours")
eco={k:cfg[k] for k in keys if k in cfg}
with evaluation_randomness(seed): env=build_inference_env(eco,seed)
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,os.environ['POL'],verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
cls=np.load(os.environ['CLS'])
ids=list(env.global_fg_order); dm=list(env.dm_ids); P='predatory_fish'; i=dm.index(P)
oth=[int(x) for x in env.obs_others_idx[i]]; juv=[int(x) for x in env.obs_juvenile_idx[i]]
chan=[ids[j] for j in oth]+['J:'+ids[j] for j in juv]
n=len(chan); imp=int(env.n_obs_imp)
cdim=2+n+imp; ndim=1+n+imp
bases=[cdim+d*ndim for d in range(4)]
def slots(name):
    k=chan.index(name); return 2+k,[b+1+k for b in bases]
GROUPS={f'no_{g}':[c for c in chan if c.replace('J:','')==g] for g in set(c.replace('J:','') for c in chan)}
orig=observations.build_observation_batch
def patched(e):
    o=orig(e); w=o[i]
    assert w.shape[1]>=cdim+4*ndim, (w.shape,cdim,ndim)
    def flatten(cs,nb):
        v=w[:,cs].mean(); w[:,cs]=v
        for s in nb: w[:,s]=v
    if arm=='no_energy':
        occ=w[:,0]>0
        w[:,1]=w[occ,1].mean() if occ.any() else 0.0
    elif arm=='no_own': flatten(0,bases)
    elif arm in GROUPS:
        for c in GROUPS[arm]: flatten(*slots(c))
    elif arm=='no_direction':
        w[:,bases]=w[:,[0]]
        for c in chan:
            cs,nb=slots(c)
            for s in nb: w[:,s]=w[:,cs]
    elif arm=='only_energy':
        flatten(0,bases)
        for c in chan: flatten(*slots(c))
    elif arm=='no_others':
        for c in chan: flatten(*slots(c))
    elif arm!='base': raise SystemExit('unknown arm '+arm)
    return o
observations.build_observation_batch=patched
print(arm,'channels',chan,'| impact slots',imp,flush=True)
wc=env.fgs[P]; wh=env.fgs['whiting']
def share(b): b=np.asarray(b,dtype=float); return np.array([b[cls==k].sum() for k in range(4)])/b.sum()
yrs=[]; book0=None; Bs=0.; Rs=0.; n2=0
with evaluation_randomness(seed):
    for y in range(2):
        s=np.zeros(4); sw=np.zeros(4)
        if y==1: book0=dict(env.intake_by_pred_prey[P])
        for t in range(1460):
            env.tick(); s+=share(wc.biomass); sw+=share(wh.biomass)
            if y==1: Bs+=float(wc.biomass.sum()); Rs+=float(wc.energy_reserve.sum()); n2+=1
        yrs.append(dict(wc=np.round(s/1460,3).tolist(),whiting=np.round(sw/1460,3).tolist()))
row={q:v-book0.get(q,0.0) for q,v in env.intake_by_pred_prey[P].items()}; tot=sum(row.values()) or 1.0
res=dict(arm=arm,years=yrs,diet={q:round(v/tot,3) for q,v in sorted(row.items(),key=lambda x:-x[1]) if v/tot>0.005},
         ration=round(tot/(Bs/n2),2),fill=round(Rs/Bs/wc.max_energy_reserve,3))
open(out,'a').write(json.dumps(res)+'\n'); print(res,flush=True)
