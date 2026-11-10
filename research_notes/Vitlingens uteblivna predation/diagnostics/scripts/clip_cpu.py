"""Reserve clip at B ME. Frozen policies, fished world (baseline maps unmuted in memory), the run's grid, CPU.
movement.apply_movement clips each DM's reserve at B_total ME after the move. Per DM and year:
  gain   = the predation intake energy (temp_energy_gains, summed over cells and ticks)
  clip   = energy above B ME removed by that clip
  post   = the part of the gain above the room left after this tick's predation (B ME - R with the
           predator's own losses to predators taken out; the cap reads the room before them)
  clip_bm = share of biomass-ticks in cells where the clip removed anything
The library is the one the policies were trained in (LIB, default the 262 library without attack_choice).
usage: POL=<dir> LIB=<yaml> clip_cpu.py <years>"""
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
from lib.environments.ecosystem_env import movement
torch.set_num_threads(4)
cfg=json.load(open('results/pzbpgp248nsrpfix94/progress/config.json')); seed=cfg['seed']
keys=("project","library","grid","mortality","migration","mortality_multiplier","mass_balance","currents","tick_hours")
eco={k:cfg[k] for k in keys if k in cfg}
if os.environ.get('LIB'): eco['library']=os.environ['LIB']
print('library',eco.get('library'),'grid',eco.get('grid'),flush=True)
with evaluation_randomness(seed): env=build_inference_env(eco,seed)
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,os.environ['POL'],verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
dm=list(env.dm_ids); N=len(dm)
print('reserve_room',[g for i,g in enumerate(dm) if bool(env.dm_reserve_room[i])],flush=True)
print('attack_choice',{g:getattr(env.fgs[g],'attack_choice','learned') for g in dm},flush=True)
Z=lambda: np.zeros(N)
acc=dict(gain=Z(),clip=Z(),post=Z(),clip_bm=Z(),bm=Z())
NP=np
class Proxy:
    def __getattr__(self,n): return getattr(NP,n)
    def clip(self,a,lo=None,hi=None,*args,**kw):
        if NP.ndim(hi)==3 and NP.shape(a)==NP.shape(hi) and NP.shape(a)[0]==N:
            over=NP.maximum(NP.asarray(a,dtype=float)-NP.asarray(hi,dtype=float),0.0)
            acc['clip']+=over.sum(axis=(1,2))
            b=NP.asarray(hi,dtype=float)/NP.maximum(NP.asarray(env.dm_max_energy_reserve,dtype=float),1e-30)[:,None,None]
            acc['clip_bm']+=(b*(over>0)).sum(axis=(1,2))
        return NP.clip(a,lo,hi,*args,**kw)
movement.np=Proxy()
oec=movement.apply_energy_costs
def ec(env_,actions):
    g=NP.stack([NP.asarray(env_.fgs[f].temp_energy_gains,dtype=float) for f in dm])
    b=NP.stack([NP.asarray(env_.fgs[f].biomass,dtype=float) for f in dm])
    r=NP.stack([NP.asarray(env_.fgs[f].energy_reserve,dtype=float) for f in dm])
    room=NP.maximum(b*NP.asarray(env_.dm_max_energy_reserve,dtype=float)[:,None,None]-r,0.0)
    acc['gain']+=g.sum(axis=(1,2)); acc['post']+=NP.maximum(g-room,0.0).sum(axis=(1,2)); acc['bm']+=b.sum(axis=(1,2))
    return oec(env_,actions)
movement.apply_energy_costs=ec
out=[]
with evaluation_randomness(seed):
    for y in range(years):
        for k in acc: acc[k]=Z()
        b0={g:float(env.fgs[g].biomass.sum()) for g in dm}
        for t in range(1460): env.tick()
        rec=dict(y=y+1,groups={})
        for i,g in enumerate(dm):
            mb=acc['bm'][i]/1460
            r=dict(gain=acc['gain'][i],clip=acc['clip'][i],post=acc['post'][i],clip_share=acc['clip'][i]/max(acc['gain'][i],1e-9),
                   post_share=acc['post'][i]/max(acc['gain'][i],1e-9),clip_bm=acc['clip_bm'][i]/max(acc['bm'][i],1e-9),
                   clip_per_t=acc['clip'][i]/max(mb,1e-9),B_end_over_start=float(env.fgs[g].biomass.sum())/max(b0[g],1e-9))
            rec['groups'][g]=r
            print(f"year {y+1} {g[:18]:18s} clip/gain {r['clip_share']:.4f} (after-predation room part {r['post_share']:.4f}) | clip {r['clip_per_t']:.1f} MJ/t/yr | cells clipped {r['clip_bm']:.2f} of biomass-ticks | B end/start {r['B_end_over_start']:.2f}",flush=True)
        out.append(rec)
json.dump(out,open(f"{S}/../tables/clip_{os.path.basename(os.environ['POL'].rstrip('/'))}.json",'w'),default=float)
