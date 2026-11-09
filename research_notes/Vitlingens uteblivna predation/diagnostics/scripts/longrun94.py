"""Frozen fix94 policies over N years on the probe's world; fished = baseline maps unmuted, in memory only."""
import os,sys,json,copy
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
arm,years,S=sys.argv[1],int(sys.argv[2]),os.path.dirname(os.path.abspath(__file__))
import lib.world.impact_maps as im
if arm=='fished':
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
print(arm,'bottom trawl mean',float(np.mean(env.grid.get_map('bottom_trawling'))) if hasattr(env.grid,'get_map') else 'n/a',flush=True)
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,S+'/'+os.environ.get('POL','fix94_frozen'),verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
ids=list(env.fgs); b0={f:float(env.fgs[f].biomass.sum()) for f in ids}
out=[]; G=['predatory_fish','planktivorous_fish','whiting','benthivorous_fish','mackerel']
print('year '+' '.join(f[:6].rjust(6) for f in ids),flush=True)
with evaluation_randomness(seed):
    for y in range(years):
        book={g:dict(env.intake_by_pred_prey[g]) for g in G}; m={f:0. for f in ids}
        for t in range(1460):
            env.tick()
            for f in ids: m[f]+=float(env.fgs[f].biomass.sum())
        m={f:v/1460 for f,v in m.items()}
        print(str(y+1).ljust(4),' '.join(f"{m[f]/b0[f]:6.2f}" for f in ids),flush=True)
        d={}
        for g in G:
            r={q:v-book[g].get(q,0) for q,v in env.intake_by_pred_prey[g].items()}; s=sum(r.values())
            d[g]={'ration':s/m[g],**{q:v/s for q,v in r.items() if v/s>.005}}
        print('     WC',{q[:6]:round(v,2) for q,v in d['predatory_fish'].items()},'whit',{q[:6]:round(v,2) for q,v in d['whiting'].items()},'benth',{q[:6]:round(v,2) for q,v in d['benthivorous_fish'].items()},flush=True)
        out.append(dict(mean=m,diet=d))
json.dump(dict(b0=b0,years=out),open(f"{S}/longrun94_{arm}_{os.environ.get('POL','fix94_frozen')}.json",'w'))
