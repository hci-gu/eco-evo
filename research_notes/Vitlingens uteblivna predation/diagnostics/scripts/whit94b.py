"""Diagnostic only, frozen fix94, fished world. Arms: base; p1 (cod and WC always attack whiting); mask (whiting blind to
benthivores); co16 (whiting re-placed each tick so Lloyd's L with benthivores = 1.6, the GOV cod-whiting value); co16wc (+ WC at saithe's GOV spread)."""
import os,sys,json,copy
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
arm,years,S=sys.argv[1],int(sys.argv[2]),os.path.dirname(os.path.abspath(__file__))
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
from lib.environments.ecosystem_env import interactions,decisions
torch.set_num_threads(3)
cfg=json.load(open('results/pzbpgp248nsrpfix94/progress/config.json')); seed=cfg['seed']
keys=("project","library","grid","mortality","migration","mortality_multiplier","mass_balance","currents","tick_hours")
eco={k:cfg[k] for k in keys if k in cfg}
with evaluation_randomness(seed): env=build_inference_env(eco,seed)
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,S+'/fix94_frozen',verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
cls=np.load(S+'/cls30.npy'); W='whiting'; w=env.fgs[W]
ids=list(env.global_fg_order); dm=list(env.dm_ids); iw=ids.index(W); dw=dm.index(W); ib=ids.index('benthivorous_fish'); db=dm.index('benthivorous_fish'); dc=dm.index('predatory_fish')
cap={'pb':0.,'pc':0.,'nb':0.,'nc':0.,'Tb':0.,'Tc':0.,'wb':0.,'wc':0.}
oac=interactions.attack_choice
def ac(eat):
    f,a=oac(eat)
    bw=np.asarray(w.biomass,dtype=float); bb=np.asarray(env.fgs['benthivorous_fish'].biomass,dtype=float); bc=np.asarray(env.fgs['predatory_fish'].biomass,dtype=float)
    cap['pb']+=float((a[db,iw]*bb*bw).sum()); cap['nb']+=float((bb*bw).sum()); cap['pc']+=float((a[dc,iw]*bc*bw).sum()); cap['nc']+=float((bc*bw).sum())
    cap['Tb']+=float((f[db,0]*bb).sum()); cap['wb']+=float(bb.sum()); cap['Tc']+=float((f[dc,0]*bc).sum()); cap['wc']+=float(bc.sum())
    if arm in('pall','pallco16','pallmask'):
        a=a.copy()
        for g in ['benthivorous_fish','predatory_fish','whiting','planktivorous_fish','mackerel']:
            i=dm.index(g)
            for j in range(len(ids)):
                if env.handling_time_mat[i,j]>0: a[i,j]=np.where(f[i,0]>0,1.0,a[i,j])
    return f,a
interactions.attack_choice=ac
if arm in('mask','pallmask'):
    n=len(env.obs_others_idx[dw]); others=[int(x) for x in env.obs_others_idx[dw]]
    jv=[int(x) for x in (env.obs_juvenile_idx[dw] if getattr(env,'obs_juvenile_idx',None) is not None else [])]
    chan=others+jv; n_other=len(chan); pos=[k for k,c in enumerate(chan) if c==ib]; print('whiting channels',[ids[c][:6] for c in chan],'benth pos',pos,flush=True)
    nimp=len(env.fgs and [])  # impacts counted below
    from lib.environments.ecosystem_env import impacts
    n_imp=len(impacts.observable_impact_layers(env))
    center=2+n_other+n_imp; nd=1+n_other+n_imp
    idx=[2+k for k in pos]+[center+d*nd+1+k for d in range(4) for k in pos]
    onorm=decisions.normalize_observations
    def nrm(e,x):
        o=onorm(e,x); o=np.array(o); o[dw,...,idx]=0.0; return o
    decisions.normalize_observations=nrm; print('masked',idx,'dim',env._in_dim,flush=True)
if arm in('co16','co16wc','pallco16'):
    def place(fg,m):
        tb=float(fg.biomass.sum()); tr=float(fg.energy_reserve.sum()); m=(m/m.sum())
        fg.biomass=(tb*m).astype(fg.biomass.dtype); fg.energy_reserve=(tr*m).astype(fg.energy_reserve.dtype)
        if fg.juvenile is not None:
            tj=float(fg.juvenile.sum()); fg.juvenile=(tj*m).astype(fg.juvenile.dtype)
    SA=(0.043,0.127,0.782,0.048); wc=np.zeros(cls.shape)
    for i in range(4): wc[cls==i]=SA[i]/(cls==i).sum()
PRED=['benthivorous_fish','predatory_fish','seals','porpoises','mackerel']
CA=['loss_starvation','loss_predation','loss_impact','loss_natural','loss_exchange','gain_exchange']
def snap(): return {c:float(getattr(env,c)[W]) for c in CA}
def eaten(): return {p:float(env.intake_by_pred_prey[p].get(W,0)) for p in PRED}
start=float(w.biomass.sum()); out=[]
series=[]
with evaluation_randomness(seed):
    for y in range(years):
        s0,e0,b0=snap(),eaten(),float(w.biomass.sum())
        for k in cap: cap[k]=0.
        B=J=F=0.; ov=0.;n=0;conb=conw=0.
        for t in range(1460):
            env.tick()
            q=np.asarray(env.fgs['benthivorous_fish'].biomass,dtype=float)
            if arm in('co16','co16wc','pallco16'):
                cv2=q.var()/(q.mean()**2+1e-12); lam=min(1.0,0.6/max(cv2,1e-9)); place(w,lam*q/q.mean()+(1-lam))
                if arm=='co16wc': place(env.fgs['predatory_fish'],wc.copy())
            b=np.asarray(w.biomass,dtype=float); j=np.asarray(w.juvenile,dtype=float)
            B+=b.sum(); J+=j.sum(); F+=float(np.asarray(w.energy_reserve,dtype=float).sum())/w.max_energy_reserve
            if t%10==0:
                L=(q*b).mean()/(q.mean()*b.mean()+1e-12); ov+=L;n+=1
                conb+=float(((q/q.sum())**2).sum()*q.size); conw+=float(((b/b.sum())**2).sum()*b.size)
                if y==0 and t<400 and t%40==0: series.append(round(L,2))
        mB=B/1460; s1,e1,b1=snap(),eaten(),float(w.biomass.sum())
        d={c:(s1[c]-s0[c])/mB for c in CA}; me={p:(e1[p]-e0[p])/mB for p in PRED}
        pb=cap['pb']/max(cap['nb'],1e-9); pc=cap['pc']/max(cap['nc'],1e-9)
        print(arm,y+1,f"B/start {mB/start:.2f} J/B {J/B:.3f} fill {F/B:.2f} L(b,w) {ov/n:.2f} conc b {conb/n:.1f} w {conw/n:.1f} pb {pb:.3f} pc {pc:.3f} Tb {cap['Tb']/cap['wb']:.3f} Tc {cap['Tc']/cap['wc']:.3f} | M2 {d['loss_predation']:.3f} benth {me['benthivorous_fish']:.3f} WC {me['predatory_fish']:.3f} seal {me['seals']:.4f} porp {me['porpoises']:.4f} | exp {d['loss_exchange']:.3f} imp {d['gain_exchange']:.3f} F {d['loss_impact']:.3f} net {(b1-b0)/mB:+.3f}",flush=True)
        out.append(dict(y=y+1,ratio=mB/start,J=J/B,fill=F/B,L=ov/n,pb=pb,pc=pc,**d,me=me))
        if y==0: print('  L series first 100 d (every 10 d)',series,flush=True)
print(arm,'end',{f[:6]:round(float(env.fgs[f].biomass.sum())/float(1),1) for f in ['whiting','benthivorous_fish','predatory_fish','seals','porpoises','planktivorous_fish','mobile_epifauna']},flush=True)
json.dump(out,open(f'{S}/whit94b_{arm}.json','w'))
print(arm,'ep/neph',{f[:6]:round(float(env.fgs[f].biomass.sum()),0) for f in ['mobile_epifauna','norway_lobster','benthic_infauna']},flush=True)
