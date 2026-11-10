"""Model side of bed_overlap_fixedkey.py. Frozen policies, fished world (baseline maps unmuted in memory), the run's
grid, CPU. Every 10 ticks: whiting and benthivore biomass per bed-depth class (cls = < 100 / 100-150 / 150-250 /
>= 250 m) and Lloyd's L over cells, over the class means (classes only, each class at its cell share) and their ratio
(within class). Averaged per year and per field quarter (Q1 = Jan-Mar, Q3 = Jul-Sep on the daylight calendar).
usage: RUN=<run> POL=<dir> CLS=<cls npy> [LIB=<yaml>] bed_overlap_model.py <years>"""
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
torch.set_num_threads(3)
cfg=json.load(open('results/'+os.environ['RUN']+'/progress/config.json')); seed=cfg['seed']
keys=("project","library","grid","mortality","migration","mortality_multiplier","mass_balance","currents","tick_hours")
eco={k:cfg[k] for k in keys if k in cfg}
if os.environ.get('LIB'): eco['library']=os.environ['LIB']
with evaluation_randomness(seed): env=build_inference_env(eco,seed)
with evaluation_randomness(seed):
    pol,om,ov=load_policies_and_stats(env,os.environ['POL'],verbose=False)
    env.policies=pol; env.rebuild_batched_weights(); env.obs_mean,env.obs_var=om,ov; env.softmax_temperature=cfg.get('temperature',1.0)
cls=np.load(os.environ['CLS']); f=np.array([(cls==k).mean() for k in range(4)])
dl=None
for g in env.fgs.values():
    dl=(getattr(g,'params',{}) or {}).get('daylight') or dl
start_tick=int(dl['start_tick']) if dl else 0; th=int(dl['tick_hours']) if dl else 6
print('library',eco.get('library'),'| start day',start_tick*th//24+1,'| cell shares per class',np.round(f,3).tolist(),flush=True)
W,B='whiting','benthivorous_fish'
def measure():
    w=np.asarray(env.fgs[W].biomass,dtype=float); b=np.asarray(env.fgs[B].biomass,dtype=float)
    sw=np.array([w[cls==k].sum() for k in range(4)])/w.sum(); sb=np.array([b[cls==k].sum() for k in range(4)])/b.sum()
    cell=(w*b).mean()/(w.mean()*b.mean())
    mw=np.array([w[cls==k].mean() if (cls==k).any() else 0 for k in range(4)]); mb=np.array([b[cls==k].mean() if (cls==k).any() else 0 for k in range(4)])
    clas=(f*mw*mb).sum()/((f*mw).sum()*(f*mb).sum())
    return sw,sb,cell,clas
out=[]
with evaluation_randomness(seed):
    for y in range(years):
        acc={q:dict(n=0,sw=0,sb=0,cell=0,clas=0) for q in ('year','Q1','Q3')}
        for t in range(1460):
            env.tick()
            if t%10: continue
            day=((start_tick+y*1460+t+1)*th/24.0)%365
            sw,sb,cell,clas=measure()
            qs=['year']+(['Q1'] if day<90 else ['Q3'] if 181<=day<273 else [])
            for q in qs:
                a=acc[q]; a['n']+=1; a['sw']=a['sw']+sw; a['sb']=a['sb']+sb; a['cell']+=cell; a['clas']+=clas
        rec=dict(y=y+1)
        for q,a in acc.items():
            if not a['n']: continue
            n=a['n']; cell,clas=a['cell']/n,a['clas']/n
            rec[q]=dict(sw=(a['sw']/n).tolist(),sb=(a['sb']/n).tolist(),cell=cell,clas=clas,within=cell/clas)
            print(f"year {y+1} {q:4s} whiting shares {' / '.join(f'{v:.3f}' for v in a['sw']/n)} | benthivores {' / '.join(f'{v:.3f}' for v in a['sb']/n)} | L cells {cell:.2f} classes only {clas:.2f} within class {cell/clas:.2f}",flush=True)
        out.append(rec)
json.dump(out,open(f"{S}/../tables/bed_overlap_{os.path.basename(os.environ['POL'].rstrip('/'))}.json",'w'))
