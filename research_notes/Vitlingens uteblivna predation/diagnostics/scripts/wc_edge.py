"""The edge rule and the water column's (predatory_fish) placement. Frozen policies, the run's probe world (unfished as
in training), the run's grid, CPU. With --migration on, biomass a DM moves out over the grid edge re-enters on the
edge cells (movement._edge_emigration -> _concentrate_immigration, weights grid_masks.build_edge_immigration_weights:
every edge cell alike). Arms:
  base     - as the engine is; the water column's edge emigrants and where they re-enter are booked by class;
  reflect  - the water column's edge emigrants return to the cell they left (a closed edge for this group only;
             every other group and the currents untouched).
Reported per year: the water column's class shares (< 100 / 100-150 / 150-250 / >= 250 m), its edge emigration per
tonne-year by class of origin and re-entry, the open pool's net exchange by class (open_pool.migrate), the training
reward mean log(E*/E0) on the ledger, ration and fill; whiting's class shares.
usage: RUN=<run> POL=<dir> CLS=<cls npy> wc_edge.py <base|reflect> <years>"""
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
arm=sys.argv[1]; years=int(sys.argv[2]); S=os.path.dirname(os.path.abspath(__file__))
assert arm in ('base','reflect')
import numpy as np,torch
from lib.runners.training_progress import build_inference_env,evaluation_randomness
from lib.runners import energy_ledger
from inference import load_policies_and_stats
from lib.environments.ecosystem_env import movement, open_pool
from lib.environments.ecosystem_env.constants import NORTH, EAST, SOUTH, WEST
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
P='predatory_fish'; dm=list(env.dm_ids); i=dm.index(P); WC=env.fgs[P]; WH=env.fgs['whiting']
ec=float(WC.params.get('energy_content',0.0) or 0.0)
print('arm',arm,'| migration',env.migration,flush=True)
acc={}
def reset(): acc.update(B=np.zeros(K),out=np.zeros(K),inn=np.zeros(K),ex=np.zeros(K),sw=np.zeros(K))
reset()
last={}
oee=movement._edge_emigration
in_move=[False]
def ee(b_out,r_out):
    if not in_move[0]: return oee(b_out,r_out)    # the currents' call (drifting groups only)
    for name,a in (('b',b_out),('r',r_out)):
        o=np.zeros(a.shape[2:],dtype=float)
        o[0,:]+=a[i,NORTH,0,:]; o[-1,:]+=a[i,SOUTH,-1,:]; o[:,-1]+=a[i,EAST,:,-1]; o[:,0]+=a[i,WEST,:,0]
        last[name]=o
    return oee(b_out,r_out)
movement._edge_emigration=ee
oci=movement._concentrate_immigration
def ci(env_,b_emig,r_emig,base_weights,group_ids=None):
    b_imm,r_imm=oci(env_,b_emig,r_emig,base_weights,group_ids)
    if in_move[0] and group_ids is None and 'b' in last:      # the movement call, not the currents'
        if arm=='reflect':
            b_imm=b_imm.copy(); r_imm=r_imm.copy()
            b_imm[i]=last['b'].astype(b_imm.dtype); r_imm[i]=last['r'].astype(r_imm.dtype)
        # A staged group's J is settled in an earlier call of the same move; the biomass call is the last one,
        # so each call overwrites and apply_movement books the last.
        tick_book['out']=pc(last['b']); tick_book['inn']=pc(b_imm[i])
    last.clear()
    return b_imm,r_imm
movement._concentrate_immigration=ci
tick_book={}
oam=movement.apply_movement
def am(env_,settlement):
    tick_book.clear(); in_move[0]=True
    try: r=oam(env_,settlement)
    finally: in_move[0]=False
    if tick_book: acc['out']+=tick_book['out']; acc['inn']+=tick_book['inn']
    return r
movement.apply_movement=am
mig=open_pool.migrate
def migr(env_):
    b0=np.asarray(WC.biomass,dtype=float).copy(); r=mig(env_); acc['ex']+=pc(np.asarray(WC.biomass,dtype=float)-b0); return r
open_pool.migrate=migr
b0=float(WC.biomass.sum()); r0=float(WC.energy_reserve.sum()); e0=b0*ec+r0; eps=max(1e-6*e0,1e-9)
ledger=energy_ledger.EnergyLedger(env,{P:(b0,r0,e0)})
print('start class shares',np.round(pc(WC.biomass)/b0,3).tolist(),flush=True)
out=[]; fm=lambda v,p=3: ' / '.join(f'{x:.{p}f}' for x in v)
with evaluation_randomness(seed):
    for y in range(years):
        reset(); book0=dict(env.intake_by_pred_prey[P]); logsum=0.; Bm=0.; Rm=0.
        for t in range(1460):
            env.tick()
            b=float(WC.biomass.sum()); r=float(WC.energy_reserve.sum())
            _,_,ei=ledger.update(P,b,r,b*ec+r); logsum+=np.log((ei+eps)/(e0+eps)); Bm+=b; Rm+=r
            acc['B']+=pc(WC.biomass); acc['sw']+=pc(WH.biomass)/float(WH.biomass.sum())
        B=np.maximum(acc['B'],1e-12); Bt=acc['B'].sum()
        row={q:v-book0.get(q,0.0) for q,v in env.intake_by_pred_prey[P].items()}; tot=sum(row.values()) or 1.0
        rec=dict(y=y+1,share=(acc['B']/Bt).tolist(),edge_out_per_t=(acc['out']/B*1460).tolist(),edge_out_total=float(acc['out'].sum()/Bt*1460),
                 reentry_share=(acc['inn']/max(acc['inn'].sum(),1e-12)).tolist(),exchange_per_t=(acc['ex']/B*1460).tolist(),
                 exchange_share=(acc['ex']/max(abs(acc['ex']).sum(),1e-12)).tolist(),reward=logsum/1460,ration=tot/(Bm/1460),
                 fill=Rm/Bm/WC.max_energy_reserve,whiting_share=(acc['sw']/1460).tolist(),
                 diet={q:round(v/tot,3) for q,v in sorted(row.items(),key=lambda x:-x[1]) if v/tot>0.005})
        out.append(rec)
        print(f"year {y+1} water column shares {fm(rec['share'])} | whiting {fm(rec['whiting_share'])}",flush=True)
        print(f"  edge emigration /yr of the group {rec['edge_out_total']:.2f}; per tonne-year by origin class {fm(rec['edge_out_per_t'])}; re-entry class shares {fm(rec['reentry_share'])}",flush=True)
        print(f"  open-pool exchange per tonne-year {fm(rec['exchange_per_t'])} | reward {rec['reward']:.4f} | ration {rec['ration']:.2f} fill {rec['fill']:.3f} | diet {rec['diet']}",flush=True)
json.dump(out,open(f"{S}/../tables/wc_edge_{arm}_{os.path.basename(os.environ['POL'].rstrip('/'))}.json",'w'))
