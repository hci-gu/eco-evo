"""Satiation and krill flows. Frozen policies, fished world (baseline maps unmuted in memory), 30x30, CPU.
Per reserve_room DM and year: share of its biomass-ticks where the reserve-room cap binds, potential (uncapped)
energy / taken energy, ration, fill, T. Krill: mean biomass, import / export (exchange books), M1, M2 by predator,
own growth from the balance, per tonne of box krill.
usage: POL=<dir> satiation_cpu.py <years>"""
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
from lib.environments.ecosystem_env import predation,interactions
torch.set_num_threads(4)
cfg=json.load(open('results/pzbpgp248nsrpfix94/progress/config.json')); seed=cfg['seed']
keys=("project","library","grid","mortality","migration","mortality_multiplier","mass_balance","currents","tick_hours")
eco={k:cfg[k] for k in keys if k in cfg}
with evaluation_randomness(seed): env=build_inference_env(eco,seed)
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,S+'/'+os.environ.get('POL','fix94_latest'),verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
dm=list(env.dm_ids); RR=[g for i,g in enumerate(dm) if bool(env.dm_reserve_room[i])]
print('reserve_room DMs',RR,flush=True)
acc={g:dict(bind=0.,bm=0.,pot=0.,room=0.,take=0.) for g in RR}
orc=predation._reserve_room_cap
def rc(env_,demand,predator_biomass,reserve_offset):
    out=orc(env_,demand,predator_biomass,reserve_offset)
    def energy(d):
        e=(d*env_.energy_gain_mat[:,:,None,None]).sum(axis=1)
        if reserve_offset is not None: e=e+(d*env_.assimilation_mat[:,:,None,None]*reserve_offset[None]).sum(axis=1)
        return e
    pe,ce=energy(demand),energy(out)
    for g in RR:
        i=dm.index(g); b=np.asarray(predator_biomass[i],dtype=float)
        acc[g]['bind']+=float((b*(ce[i]<0.99*pe[i])).sum()); acc[g]['bm']+=float(b.sum())
        acc[g]['pot']+=float(pe[i].sum()); acc[g]['take']+=float(ce[i].sum())
    return out
predation._reserve_room_cap=rc
oac=interactions.attack_choice; Tacc={g:[0.,0.] for g in RR}
def ac(eat):
    f,a=oac(eat)
    for g in RR:
        i=dm.index(g); b=np.asarray(env.fgs[g].biomass,dtype=float); Tacc[g][0]+=float((np.asarray(f[i,0])*b).sum()); Tacc[g][1]+=float(b.sum())
    return f,a
interactions.attack_choice=ac
K='krill'; kf=env.fgs[K]
PRED=[g for g in env.intake_by_pred_prey if K in env.intake_by_pred_prey[g] or True]
CA=['loss_starvation','loss_predation','loss_impact','loss_natural','loss_exchange']
def ksnap(): return dict(imp=float(env.gain_exchange[K]),exp=float(env.loss_exchange[K]),nat=float(env.loss_natural[K]),pred=float(env.loss_predation[K]))
def keaten(): return {g:float(env.intake_by_pred_prey[g].get(K,0.)) for g in env.intake_by_pred_prey}
out=[]
with evaluation_randomness(seed):
    for y in range(years):
        for g in RR:
            acc[g]=dict(bind=0.,bm=0.,pot=0.,room=0.,take=0.); Tacc[g]=[0.,0.]
        i0={g:sum(env.intake_by_pred_prey[g].values()) for g in RR}; mb={g:0. for g in RR}; fill={g:0. for g in RR}
        k0=ksnap(); e0=keaten(); kb0=float(kf.biomass.sum()); KB=0.
        for t in range(1460):
            env.tick()
            KB+=float(kf.biomass.sum())
            for g in RR:
                f=env.fgs[g]; b=float(f.biomass.sum()); mb[g]+=b; fill[g]+=float(np.asarray(f.energy_reserve).sum())/(f.max_energy_reserve*max(b,1e-9))
        KB/=1460; k1=ksnap(); e1=keaten(); kb1=float(kf.biomass.sum())
        rec=dict(y=y+1,groups={},krill={})
        for g in RR:
            m=mb[g]/1460; a=acc[g]
            rec['groups'][g]=dict(bind=a['bind']/max(a['bm'],1e-9),pot_over_take=a['pot']/max(a['take'],1e-9),
                                  ration=(sum(env.intake_by_pred_prey[g].values())-i0[g])/m,fill=fill[g]/1460,T=Tacc[g][0]/max(Tacc[g][1],1e-9))
            r=rec['groups'][g]
            print(f"year {y+1} {g[:12]:12s} cap binds on {r['bind']:.2f} of biomass-ticks | uncapped/taken energy {r['pot_over_take']:.2f} | ration {r['ration']:.2f} fill {r['fill']:.2f} T {r['T']:.2f}",flush=True)
        fl={k:(k1[k]-k0[k])/KB for k in k0}; m2={g:(e1[g]-e0[g])/KB for g in e1 if e1[g]-e0[g]>0}
        growth=(kb1-kb0)/KB+fl['pred']+fl['nat']+fl['exp']-fl['imp']
        rec['krill']=dict(mean_t=KB,start_t=kb0,**fl,m2=m2,own_growth=growth)
        print(f"year {y+1} krill mean {KB:.0f} t | per tonne of box krill /yr: import {fl['imp']:.2f} export {fl['exp']:.2f} M1 {fl['nat']:.2f} M2 {fl['pred']:.2f} own growth {growth:.2f} | M2 by "+' '.join(f"{g[:6]} {v:.3f}" for g,v in m2.items()),flush=True)
        out.append(rec)
json.dump(out,open(f"{S}/satiation_{os.environ.get('POL','fix94_latest')}.json",'w'))
