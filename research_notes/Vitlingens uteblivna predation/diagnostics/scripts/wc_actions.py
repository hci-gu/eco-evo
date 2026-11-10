"""The water column's (predatory_fish) action mix under frozen policies in the run's probe world (unfished as in
training), the run's grid, CPU: biomass-weighted mean fractions of rest, eat and the four moves (N E S W; row 0 is
the north edge, the deep Norwegian Trench side; the shallow beds lie south and east), per bed class and overall,
per quarter of year 1, and the net drift (S - N, E - W). Arm `only_energy` flattens every biomass channel of the
water column's observation to its grid mean (as wc_ablate.py), so its actions vary only with the own reserve fill
and the light level.
usage: RUN=<run> POL=<dir> CLS=<cls npy> wc_actions.py <base|only_energy>"""
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
import numpy as np,torch
from lib.runners.training_progress import build_inference_env,evaluation_randomness
from lib.environments.ecosystem_env import observations, environment as envmod
from inference import load_policies_and_stats
arm=sys.argv[1]; S=os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(2)
cfg=json.load(open('results/'+os.environ['RUN']+'/progress/config.json')); seed=cfg['seed']
keys=("project","library","grid","mortality","migration","mortality_multiplier","mass_balance","currents","tick_hours")
eco={k:cfg[k] for k in keys if k in cfg}
with evaluation_randomness(seed): env=build_inference_env(eco,seed)
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,os.environ['POL'],verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
cls=np.load(os.environ['CLS']); K=4
dm=list(env.dm_ids); P='predatory_fish'; i=dm.index(P); WC=env.fgs[P]
n=len(env.obs_others_idx[i])+len(env.obs_juvenile_idx[i]); imp=int(env.n_obs_imp)
cdim=2+n+imp; ndim=1+n+imp; bases=[cdim+d*ndim for d in range(4)]
orig=observations.build_observation_batch
def patched(e):
    o=orig(e)
    if arm=='only_energy':
        w=o[i]
        cols=[0]+list(range(2,2+n))+[b+k for b in bases for k in range(0,1+n)]
        for c in cols: w[:,c]=w[:,c].mean()
    return o
observations.build_observation_batch=patched
acc=np.zeros((5,K,6)); wsum=np.zeros((5,K))   # quarter 0-3 and year (4)
oec=envmod.movement.apply_energy_costs
q=[0]
def ec(env_,actions):
    b=np.asarray(WC.biomass,dtype=float)
    mv=np.asarray(actions.move[i],dtype=float)            # (4, H, W) N E S W
    rest=np.asarray(actions.rest[i],dtype=float); eat=np.clip(1.0-rest-mv.sum(0),0,None)
    parts=np.stack([rest,eat]+[mv[d] for d in range(4)])  # (6, H, W)
    for k in range(K):
        m=cls==k
        for slot in (q[0],4):
            acc[slot,k]+= (parts[:,m]*b[m]).sum(axis=1); wsum[slot,k]+=b[m].sum()
    return oec(env_,actions)
envmod.movement.apply_energy_costs=ec
with evaluation_randomness(seed):
    for t in range(1460):
        q[0]=min(t*4//1460,3); env.tick()
lab=['rest','eat','N','E','S','W']
out={}
for slot,name in enumerate(['quarter 1','quarter 2','quarter 3','quarter 4','year']):
    print(f"{arm} {name}:",flush=True)
    rows=[]
    for k in range(K):
        f=acc[slot,k]/max(wsum[slot,k],1e-12); rows.append(f.tolist())
        print(f"  class {k} "+' '.join(f'{l} {v:.3f}' for l,v in zip(lab,f))+f" | drift S-N {f[4]-f[2]:+.3f} E-W {f[3]-f[5]:+.3f}",flush=True)
    a=acc[slot].sum(0)/max(wsum[slot].sum(),1e-12)
    print(f"  all     "+' '.join(f'{l} {v:.3f}' for l,v in zip(lab,a))+f" | drift S-N {a[4]-a[2]:+.3f} E-W {a[3]-a[5]:+.3f}",flush=True)
    out[name]=dict(per_class=rows,all=a.tolist())
json.dump(out,open(f"{S}/../tables/wc_actions_{arm}_{os.path.basename(os.environ['POL'].rstrip('/'))}.json",'w'))
