"""Attack probabilities p by pair (encounter-weighted) and, arm pall, the world with p = 1 on every prey a fish group forages on. Frozen policies, fished world.
usage: pmat.py <arm base|pall|random> <ckpt dir or -> <years>"""
import os,sys,json,copy
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
arm,ck,years=sys.argv[1],sys.argv[2],int(sys.argv[3]); S=os.path.dirname(os.path.abspath(__file__))
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
from lib.environments.ecosystem_env import interactions
torch.set_num_threads(3)
cfg=json.load(open('results/pzbpgp248nsrpfix94/progress/config.json')); seed=cfg['seed']
keys=("project","library","grid","mortality","migration","mortality_multiplier","mass_balance","currents","tick_hours")
eco={k:cfg[k] for k in keys if k in cfg}
with evaluation_randomness(seed): env=build_inference_env(eco,seed)
if arm!='random':
    with evaluation_randomness(seed):
        pol,om,ov=load_policies_and_stats(env,ck,verbose=False)
        env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
else:
    env.rebuild_batched_weights() if hasattr(env,'rebuild_batched_weights') else None
ids=list(env.global_fg_order); dm=list(env.dm_ids)
FISH=['benthivorous_fish','predatory_fish','whiting','planktivorous_fish','mackerel']
mask=np.asarray(env.handling_time_mat)>0
num={f:np.zeros(len(ids)) for f in FISH}; den={f:np.zeros(len(ids)) for f in FISH}; top={f:np.zeros(len(ids)) for f in FISH}; T={f:[0.,0.] for f in FISH}
oac=interactions.attack_choice
def ac(eat):
    f,a=oac(eat)
    for g in FISH:
        i=dm.index(g); bp=np.asarray(env.fgs[g].biomass,dtype=float)
        T[g][0]+=float((f[i,0]*bp).sum()); T[g][1]+=float(bp.sum())
        for j,q in enumerate(ids):
            if env.handling_time_mat[i,j]>0:
                bq=np.asarray(env.fgs[q].biomass,dtype=float); w=bp*bq
                num[g][j]+=float((a[i,j]*w).sum()); den[g][j]+=float(w.sum()); top[g][j]+=float(((a[i,j]>=0.999)*bp).sum())
    if arm=='pall':
        a=a.copy()
        for g in FISH:
            i=dm.index(g)
            for j in range(len(ids)):
                if env.handling_time_mat[i,j]>0: a[i,j]=np.where(f[i,0]>0,1.0,a[i,j])
    return f,a
interactions.attack_choice=ac
b0={f:float(env.fgs[f].biomass.sum()) for f in env.fgs}; out=[]
print('year '+' '.join(f[:6].rjust(6) for f in ids),flush=True)
with evaluation_randomness(seed):
    for y in range(years):
        book={g:dict(env.intake_by_pred_prey[g]) for g in FISH}; m={f:0. for f in ids}; rfill={g:0. for g in FISH}
        for g in FISH:
            num[g][:]=0;den[g][:]=0;top[g][:]=0;T[g]=[0.,0.]
        for t in range(1460):
            env.tick()
            for f in ids: m[f]+=float(env.fgs[f].biomass.sum())
            for g in FISH: rfill[g]+=float(np.asarray(env.fgs[g].energy_reserve,dtype=float).sum())/env.fgs[g].max_energy_reserve
        print(str(y+1).ljust(4),' '.join(f"{m[f]/1460/b0[f]:6.2f}" for f in ids),flush=True)
        rec={}
        for g in FISH:
            r={q:v-book[g].get(q,0) for q,v in env.intake_by_pred_prey[g].items()}; s=sum(r.values())
            rec[g]=dict(ration=s/(m[g]/1460),fill=rfill[g]/m[g],T=T[g][0]/T[g][1],diet={q:v/s for q,v in r.items() if v/s>.003},p={ids[j]:num[g][j]/den[g][j] for j in range(len(ids)) if den[g][j]>0},topshare={ids[j]:top[g][j]/T[g][1] for j in range(len(ids)) if den[g][j]>0},M2={q:v/(m[q]/1460) for q,v in r.items() if q in m and v/s>.003})
            if y in(0,years-1):
                print(' ',g[:6],'ration %.2f fill %.2f T %.2f'%(rec[g]['ration'],rec[g]['fill'],rec[g]['T']),'p',{q[:6]:round(v,2) for q,v in rec[g]['p'].items()},flush=True)
                print('        diet',{q[:6]:round(v,3) for q,v in sorted(rec[g]['diet'].items(),key=lambda x:-x[1])},'M2',{q[:6]:round(v,3) for q,v in rec[g]['M2'].items()},flush=True)
        out.append(dict(mean={f:m[f]/1460/b0[f] for f in ids},rec=rec))
tag=os.path.basename(ck.rstrip('/')) if ck!='-' else 'rand'
json.dump(out,open(f'{S}/pmat_{arm}_{tag}.json','w'))
