"""Porpoises at a scaled start (x factor on the manifest's), frozen policies, fished world (baseline maps unmuted in memory), 30x30.
Per year: porpoise mean biomass, ration, reserve fill, losses by cause per tonne, diet shares, M2 on each prey.
usage: POL=<dir> porp_cpu.py <factor> <years>"""
import os,sys,json,copy
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
fac,years=float(sys.argv[1]),int(sys.argv[2]); S=os.path.dirname(os.path.abspath(__file__))
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
torch.set_num_threads(4)
cfg=json.load(open('results/pzbpgp248nsrpfix94/progress/config.json')); seed=cfg['seed']
keys=("project","library","grid","mortality","migration","mortality_multiplier","mass_balance","currents","tick_hours")
eco={k:cfg[k] for k in keys if k in cfg}
with evaluation_randomness(seed): env=build_inference_env(eco,seed)
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,S+'/'+os.environ.get('POL','fix94_latest'),verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
P='porpoises'; p=env.fgs[P]
p.biomass=(np.asarray(p.biomass)*fac).astype(p.biomass.dtype); p.energy_reserve=(np.asarray(p.energy_reserve)*fac).astype(p.energy_reserve.dtype)
b0=float(p.biomass.sum()); print('porpoise start t',round(b0,2),'max reserve',p.max_energy_reserve,flush=True)
CA=['loss_starvation','loss_predation','loss_impact','loss_natural','loss_exchange']
def snap(): return {c:float(getattr(env,c)[P]) for c in CA}
PREY=['planktivorous_fish','benthivorous_fish','whiting','predatory_fish']
out=[]
with evaluation_randomness(seed):
    for y in range(years):
        s0=snap(); i0=dict(env.intake_by_pred_prey[P]); B=F=0.; mp={q:0. for q in PREY}; e0=float(p.biomass.sum())
        for t in range(1460):
            env.tick()
            b=float(p.biomass.sum()); B+=b; F+=float(np.asarray(p.energy_reserve).sum())/(p.max_energy_reserve*max(b,1e-9))
            for q in PREY: mp[q]+=float(env.fgs[q].biomass.sum())
        mB=B/1460; s1=snap(); r={q:v-i0.get(q,0.) for q,v in env.intake_by_pred_prey[P].items()}; tot=sum(r.values())
        rec=dict(y=y+1,ratio=mB/b0,end=float(p.biomass.sum()),ration=tot/mB,fill=F/1460,loss={c[5:]:(s1[c]-s0[c])/mB for c in CA},
                 diet={q:v/tot for q,v in r.items() if tot>0},m2={q:r.get(q,0.)/(mp[q]/1460) for q in PREY})
        print(f"year {y+1} mean/start {rec['ratio']:.2f} end {rec['end']:.1f} t ration {rec['ration']:.1f} fill {rec['fill']:.2f} | losses/t "+' '.join(f"{k} {v:.3f}" for k,v in rec['loss'].items())+" | diet "+' '.join(f"{q[:6]} {v:.2f}" for q,v in rec['diet'].items())+" | M2 "+' '.join(f"{q[:6]} {v:.4f}" for q,v in rec['m2'].items()),flush=True)
        out.append(rec)
json.dump(out,open(f"{S}/porp_cpu_x{fac:g}.json",'w'))
