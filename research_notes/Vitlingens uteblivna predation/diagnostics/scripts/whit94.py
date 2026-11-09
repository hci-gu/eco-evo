"""Frozen fix94: whiting's realised flows per year (t/t/yr on mean B), stages, predators' overlap. Fished = baseline maps unmuted in memory."""
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
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,S+'/fix94_frozen',verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
cls=np.load(S+'/cls30.npy'); W='whiting'; w=env.fgs[W]
PRED=['benthivorous_fish','predatory_fish','seals','porpoises','mackerel']
CA=['loss_starvation','loss_predation','loss_impact','loss_natural','loss_exchange','gain_exchange']
def snap(): return {c:float(getattr(env,c)[W]) for c in CA}
def eaten(): return {p:float(env.intake_by_pred_prey[p].get(W,0)) for p in PRED}
print('whiting start t',float(w.biomass.sum()),'preds',{p:float(env.fgs[p].biomass.sum()) for p in PRED},flush=True)
out=[]
with evaluation_randomness(seed):
    for y in range(years):
        s0,e0,b0=snap(),eaten(),float(w.biomass.sum())
        B=J=0.; Pb={p:0. for p in PRED}; ov={p:0. for p in PRED}; ovJ={p:0. for p in PRED}; ovA={p:0. for p in PRED}; n=0; cl=np.zeros(4); fill=0.
        for t in range(1460):
            env.tick()
            b=np.asarray(w.biomass,dtype=float); j=np.asarray(w.juvenile,dtype=float) if w.juvenile is not None else np.zeros_like(b)
            B+=b.sum(); J+=j.sum(); fill+=float(np.asarray(w.energy_reserve,dtype=float).sum())/w.max_energy_reserve
            for p in PRED: Pb[p]+=float(env.fgs[p].biomass.sum())
            cl+=[b[cls==i].sum() for i in range(4)]
            if t%20==0:
                n+=1; a=b-j
                for p in PRED:
                    q=np.asarray(env.fgs[p].biomass,dtype=float)
                    ov[p]+=(q*b).mean()/(q.mean()*b.mean()+1e-12); ovJ[p]+=(q*j).mean()/(q.mean()*j.mean()+1e-12); ovA[p]+=(q*a).mean()/(q.mean()*a.mean()+1e-12)
        mB=B/1460; s1,e1,b1=snap(),eaten(),float(w.biomass.sum())
        d={c:(s1[c]-s0[c])/mB for c in CA}; me={p:(e1[p]-e0[p])/mB for p in PRED}
        G=(b1-b0)/mB+d['loss_starvation']+d['loss_predation']+d['loss_impact']+d['loss_natural']+d['loss_exchange']-d['gain_exchange']
        r=dict(year=y+1,mean_ratio=mB/float(1605),end_ratio=b1/1605,J_share=J/B,fill=fill/B,G=G,**d,eaten_per_B=me,pred_ratio={p:Pb[p]/1460 for p in PRED},overlap={p:ov[p]/n for p in PRED},overlapJ={p:ovJ[p]/n for p in PRED},overlapA={p:ovA[p]/n for p in PRED},cls=(cl/cl.sum()).tolist())
        out.append(r)
        print(y+1,f"B/start {r['mean_ratio']:.2f} J/B {r['J_share']:.3f} fill {r['fill']:.2f} | G {G:.3f} imp {d['gain_exchange']:.3f} | expt {d['loss_exchange']:.3f} M2 {d['loss_predation']:.3f} F {d['loss_impact']:.3f} M1 {d['loss_natural']:.3f} starv {d['loss_starvation']:.3f} | net {(b1-b0)/mB:+.3f}",flush=True)
        print('    eaten/B',{p[:6]:round(v,4) for p,v in me.items()},'L',{p[:6]:round(v,2) for p,v in r['overlap'].items()},'LJ',{p[:6]:round(v,2) for p,v in r['overlapJ'].items()},'LA',{p[:6]:round(v,2) for p,v in r['overlapA'].items()},'cls',np.round(r['cls'],2).tolist(),flush=True)
json.dump(out,open(f'{S}/whit94_{arm}.json','w'))
