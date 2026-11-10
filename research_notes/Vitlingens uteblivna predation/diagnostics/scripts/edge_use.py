"""How much does every decision maker use the grid edge? Frozen policies, the run's probe world (unfished as in
training), the run's grid, CPU, nothing altered. With --migration on, biomass a DM moves over the grid edge re-enters
evenly on the edge cells (movement._edge_emigration -> _concentrate_immigration). Per DM and year:
  - edge emigration as a share of the group's mean biomass per year (tonnes out / tonne-year);
  - the same for all in-grid moves (tonnes moved to a neighbour cell / tonne-year), for scale;
  - the class shares (< 100 / 100-150 / 150-250 / >= 250 m) of where the emigrants leave and where they re-enter;
  - the net relocation by the edge per class as a share of the group's mean biomass per year (re-entry - exit);
  - the group's mean class shares.
The currents' edge path (drifting groups) is not booked.
usage: RUN=<run> POL=<dir> CLS=<cls npy> edge_use.py <years>"""
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
years=int(sys.argv[1]); S=os.path.dirname(os.path.abspath(__file__))
import numpy as np,torch
from lib.runners.training_progress import build_inference_env,evaluation_randomness
from inference import load_policies_and_stats
from lib.environments.ecosystem_env import movement
from lib.environments.ecosystem_env.constants import NORTH, EAST, SOUTH, WEST
torch.set_num_threads(3)
cfg=json.load(open('results/'+os.environ['RUN']+'/progress/config.json')); seed=cfg['seed']
keys=("project","library","grid","mortality","migration","mortality_multiplier","mass_balance","currents","tick_hours")
eco={k:cfg[k] for k in keys if k in cfg}
with evaluation_randomness(seed): env=build_inference_env(eco,seed)
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,os.environ['POL'],verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
cls=np.load(os.environ['CLS']); K=4; masks=[cls==k for k in range(K)]
dm=list(env.dm_ids); N=len(dm)
pc=lambda x: np.array([np.asarray(x,dtype=float)[m].sum() for m in masks])
print('migration',env.migration,'| DMs',dm,flush=True)
acc={}
def reset(): acc.update(B=np.zeros((N,K)),out=np.zeros((N,K)),inn=np.zeros((N,K)),moved=np.zeros(N))
reset()
in_move=[False]; last={}; tick_book={}
oee=movement._edge_emigration
def ee(b_out,r_out):
    if in_move[0]:
        o=np.zeros((N,)+b_out.shape[2:],dtype=float)
        o[:,0,:]+=b_out[:,NORTH,0,:]; o[:,-1,:]+=b_out[:,SOUTH,-1,:]; o[:,:,-1]+=b_out[:,EAST,:,-1]; o[:,:,0]+=b_out[:,WEST,:,0]
        last['b']=o; last['all']=np.asarray(b_out,dtype=float).sum(axis=(1,2,3))
    return oee(b_out,r_out)
movement._edge_emigration=ee
oci=movement._concentrate_immigration
def ci(env_,b_emig,r_emig,base_weights,group_ids=None):
    b_imm,r_imm=oci(env_,b_emig,r_emig,base_weights,group_ids)
    if in_move[0] and group_ids is None and 'b' in last:
        # A staged group's J is settled in an earlier call of the same move: the last call (the biomass) wins.
        tick_book['out']=np.stack([pc(last['b'][d]) for d in range(N)]); tick_book['inn']=np.stack([pc(b_imm[d]) for d in range(N)])
        tick_book['moved']=last['all']-last['b'].sum(axis=(1,2))
    last.clear(); return b_imm,r_imm
movement._concentrate_immigration=ci
oam=movement.apply_movement
def am(env_,settlement):
    tick_book.clear(); in_move[0]=True
    try: r=oam(env_,settlement)
    finally: in_move[0]=False
    if tick_book:
        acc['out']+=tick_book['out']; acc['inn']+=tick_book['inn']; acc['moved']+=tick_book['moved']
    return r
movement.apply_movement=am
out=[]; fm=lambda v,p=3: ' / '.join(f'{x:.{p}f}' for x in v)
with evaluation_randomness(seed):
    for y in range(years):
        reset()
        for t in range(1460):
            env.tick()
            for d,fid in enumerate(dm): acc['B'][d]+=pc(env.fgs[fid].biomass)
        rec=dict(y=y+1,groups={})
        print(f"year {y+1}: per group - edge out /yr | in-grid moves /yr | exit classes | re-entry classes | net relocation by class (share of B /yr) | class shares",flush=True)
        for d,fid in enumerate(dm):
            Bt=acc['B'][d].sum()
            if Bt<=0: continue
            o=acc['out'][d]; n=acc['inn'][d]
            g=dict(edge_out=float(o.sum()/Bt*1460),moved=float(acc['moved'][d]/Bt*1460),
                   exit=(o/max(o.sum(),1e-12)).tolist(),reentry=(n/max(n.sum(),1e-12)).tolist(),
                   net=((n-o)/Bt*1460).tolist(),share=(acc['B'][d]/Bt).tolist())
            rec['groups'][fid]=g
            print(f"  {fid[:20]:20s} {g['edge_out']:7.3f} | {g['moved']:7.2f} | {fm(g['exit'],2)} | {fm(g['reentry'],2)} | {fm(g['net'])} | {fm(g['share'],2)}",flush=True)
        out.append(rec)
json.dump(out,open(f"{S}/../tables/edge_use_{os.path.basename(os.environ['POL'].rstrip('/'))}.json",'w'))
