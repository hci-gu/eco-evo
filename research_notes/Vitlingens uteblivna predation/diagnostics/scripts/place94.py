"""Diagnostic only: frozen fix94; the water column kept at a given bed-class spread by re-placing it every tick.
Asks whether it starves at the field spread (a real pull to the shallows) or the reward cannot tell the difference."""
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
arm=sys.argv[1]; S=os.path.dirname(os.path.abspath(__file__))
import numpy as np,torch
from lib.runners.training_progress import build_inference_env,evaluation_randomness
from inference import load_policies_and_stats
torch.set_num_threads(3)
cfg=json.load(open('results/pzbpgp248nsrpfix94/progress/config.json')); seed=cfg['seed']
keys=("project","library","grid","mortality","migration","mortality_multiplier","mass_balance","currents","tick_hours")
eco={k:cfg[k] for k in keys if k in cfg}
with evaluation_randomness(seed): env=build_inference_env(eco,seed)
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,S+'/fix94_frozen',verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
cls=np.load(S+'/cls30.npy'); wc=env.fgs['predatory_fish']; print(arm,'juvenile attr',getattr(wc,'juvenile',None) is not None,flush=True)
SP={'field':(0.043,0.127,0.782,0.048),'equal':tuple(float((cls==i).mean()) for i in range(4)),'learned':None,'deep':(0,0,1,0),'mid':(0,1,0,0),'shal':(1,0,0,0)}[arm]
w=None
if SP is not None:
    w=np.zeros(cls.shape)
    for i in range(4): w[cls==i]=SP[i]/(cls==i).sum()
def share(fg): 
    b=np.asarray(fg.biomass,dtype=float); return np.round([b[cls==i].sum()/b.sum() for i in range(4)],3)
print('start share',share(wc),flush=True)
ME=wc.max_energy_reserve
book0=None; B=0.;R=0.; months=[]; m=np.zeros(4); K={}
pl=env.fgs['planktivorous_fish']; PB=0.;KB=0.; kr=env.fgs['krill']
b0={f:float(env.fgs[f].biomass.sum()) for f in env.fgs}
with evaluation_randomness(seed):
    for t in range(2920):
        if t==1460: book0=dict(env.intake_by_pred_prey['predatory_fish'])
        env.tick()
        if w is not None:
            tb=float(wc.biomass.sum()); tr=float(wc.energy_reserve.sum())
            wc.biomass=(tb*w).astype(wc.biomass.dtype); wc.energy_reserve=(tr*w).astype(wc.energy_reserve.dtype)
        if t<1460:
            b=np.asarray(wc.biomass,dtype=float)
            m+=[b[cls==i].sum()/b.sum() for i in range(4)]
            if (t+1)%365==0: months.append(np.round(m/365,3).tolist()); m=np.zeros(4)
        if t>=1460:
            B+=float(wc.biomass.sum()); R+=float(wc.energy_reserve.sum()); PB+=float(pl.biomass.sum()); KB+=float(kr.biomass.sum())
row={q:v-book0.get(q,0) for q,v in env.intake_by_pred_prey['predatory_fish'].items()}; s=sum(row.values())
print('quarterly share y1',months)
print('y2 share',share(wc))
print(arm,'WC diet',{q[:6]:round(v/s,3) for q,v in sorted(row.items(),key=lambda x:-x[1]) if v/s>.002})
print(arm,'ration',round(s/(B/1460),2),'fill',round(R/B/ME,3),'WC mean B ratio',round(B/1460/b0['predatory_fish'],3))
print(arm,'WC M2 krill',round(row.get('krill',0)/(KB/1460),3),'on planktivores',round(row.get('planktivorous_fish',0)/(PB/1460),3))
print(arm,'end ratios',{f[:6]:round(float(env.fgs[f].biomass.sum())/b0[f],2) for f in env.fgs})
