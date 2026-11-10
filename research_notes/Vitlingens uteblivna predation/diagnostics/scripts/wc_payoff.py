"""What does each bed-depth placement pay the water column (predatory_fish) in the world it was trained in?
Frozen policies, the run's probe world (baseline maps as the manifest has them: muted = unfished, as in training),
the run's grid, CPU. After every tick the water column (biomass and reserve) is re-placed (as in 263.2):
  learned - not touched;
  within  - this tick's learned class totals, spread evenly inside each class;
  field   - the GOV group class shares 0.074 / 0.176 / 0.697 / 0.053 (263.7), even inside each class;
  equal   - even over all cells.
Reported per year:
  - the training reward of the water column: mean over ticks of log(E*/E0), E = B ec + R on the reward ledger
    (lib/runners/energy_ledger.py; the exchange with the outside taken out), as parallel_worker computes it;
  - mean B and E* over start;
  - per class (< 100 / 100-150 / 150-250 / >= 250 m) per tonne-year of the water column in the class: intake
    (assimilated MJ), M2 (all predators), the net exchange (open_pool.migrate), the population step's net (growth,
    M1, starvation), and the reserve fill; diet shares.
usage: RUN=<run> POL=<dir> CLS=<cls npy> wc_payoff.py <arm> <years>"""
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
arm=sys.argv[1]; years=int(sys.argv[2]); S=os.path.dirname(os.path.abspath(__file__))
assert arm in ('learned','within','field','equal')
import numpy as np,torch
from lib.runners.training_progress import build_inference_env,evaluation_randomness
from lib.runners import energy_ledger
from inference import load_policies_and_stats
from lib.environments.ecosystem_env import environment as envmod, open_pool
torch.set_num_threads(2)
cfg=json.load(open('results/'+os.environ['RUN']+'/progress/config.json')); seed=cfg['seed']
keys=("project","library","grid","mortality","migration","mortality_multiplier","mass_balance","currents","tick_hours")
eco={k:cfg[k] for k in keys if k in cfg}
with evaluation_randomness(seed): env=build_inference_env(eco,seed)
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,os.environ['POL'],verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
cls=np.load(os.environ['CLS']); K=4; masks=[cls==k for k in range(K)]
pc=lambda x: np.array([np.asarray(x,dtype=float)[m].sum() for m in masks])
P='predatory_fish'; WC=env.fgs[P]; ME=WC.max_energy_reserve; ec=float(WC.params.get('energy_content',0.0) or 0.0)
FIELD=np.array([0.074,0.176,0.697,0.053])
def replace():
    for name in ('biomass','energy_reserve'):
        a=np.asarray(getattr(WC,name),dtype=float); tot=a.sum(); new=np.zeros_like(a)
        for k,m in enumerate(masks):
            if arm=='within': new[m]=a[m].sum()/m.sum()
            elif arm=='field': new[m]=tot*FIELD[k]/m.sum()
            else: new[m]=tot/a.size
        setattr(WC,name,new.astype(getattr(WC,name).dtype))
acc={}
def reset(): acc.update(B=np.zeros(K),R=np.zeros(K),gain=np.zeros(K),m2=np.zeros(K),ex=np.zeros(K),pop=np.zeros(K))
reset()
def bw(): return np.asarray(WC.biomass,dtype=float).copy()
op=envmod.predation.apply_predation
def pred(*a,**k):
    b0=bw(); r=op(*a,**k); acc['m2']+=pc(b0-bw()); acc['gain']+=pc(WC.temp_energy_gains); return r
envmod.predation.apply_predation=pred
opc=envmod.population_change.apply_population_change
mig=open_pool.migrate
inside={'ex':0}
def migr(env_):
    b0=bw(); r=mig(env_); d=pc(bw()-b0); acc['ex']+=d; inside['ex']+=d.sum(); return r
open_pool.migrate=migr
def popc(*a,**k):
    b0=bw(); inside['ex']=0; r=opc(*a,**k); acc['pop']+=pc(bw()-b0)-0; return r
envmod.population_change.apply_population_change=popc
b0=float(WC.biomass.sum()); r0=float(WC.energy_reserve.sum()); e0=b0*ec+r0; eps=max(1e-6*e0,1e-9)
ledger=energy_ledger.EnergyLedger(env,{P:(b0,r0,e0)})
print('arm',arm,'| tracked by the ledger',ledger.tracks(P),'| ec',ec,'| start B',round(b0),flush=True)
out=[]; fm=lambda v,p=3: ' / '.join(f'{x:.{p}f}' for x in v)
with evaluation_randomness(seed):
    for y in range(years):
        reset(); book0=dict(env.intake_by_pred_prey[P]); logsum=0.; Bm=0.; Em=0.
        for t in range(1460):
            env.tick()
            if arm!='learned': replace()
            b=float(WC.biomass.sum()); r=float(WC.energy_reserve.sum())
            bi,ri,ei=ledger.update(P,b,r,b*ec+r)
            logsum+=np.log((ei+eps)/(e0+eps)); Bm+=b; Em+=ei
            acc['B']+=pc(WC.biomass); acc['R']+=pc(WC.energy_reserve)
        B=np.maximum(acc['B'],1e-12)
        row={q:v-book0.get(q,0.0) for q,v in env.intake_by_pred_prey[P].items()}; tot=sum(row.values()) or 1.0
        rec=dict(y=y+1,reward=logsum/1460,B_ratio=Bm/1460/b0,E_ratio=Em/1460/e0,share=(acc['B']/acc['B'].sum()).tolist(),
                 intake=(acc['gain']/B*1460).tolist(),M2=(acc['m2']/B*1460).tolist(),exchange=(acc['ex']/B*1460).tolist(),
                 pop_net_incl_exchange=(acc['pop']/B*1460).tolist(),fill=(acc['R']/B/ME).tolist(),
                 diet={q:round(v/tot,3) for q,v in sorted(row.items(),key=lambda x:-x[1]) if v/tot>0.005},ration=tot/(Bm/1460))
        out.append(rec)
        print(f"year {y+1} reward mean log(E*/E0) {rec['reward']:.4f} | B {rec['B_ratio']:.3f} E* {rec['E_ratio']:.3f} | share {fm(rec['share'])} | ration {rec['ration']:.2f}",flush=True)
        print(f"  per tonne-year: intake MJ {fm(rec['intake'],0)} | M2 {fm(rec['M2'])} | exchange {fm(rec['exchange'])} | population step {fm(rec['pop_net_incl_exchange'])} | fill {fm(rec['fill'])}",flush=True)
        print(f"  diet {rec['diet']}",flush=True)
json.dump(out,open(f"{S}/../tables/wc_payoff_{arm}_{os.path.basename(os.environ['POL'].rstrip('/'))}.json",'w'))
