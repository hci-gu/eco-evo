"""Why do the fish rations run 35-50 % above the field Q/B? The energy budget of each fish DM.
Frozen policies, the run's probe world (unfished as in training), the run's grid, seed and currents, CPU, nothing
altered. Per fish DM and month (and the year), booked from the engine's own arrays:
  I  - assimilated intake, net of SDA (the gains movement.income_growth receives), MJ per tonne-year;
  C  - the action costs (rest + eat + move, with Q10 and impacts), MJ per tonne-year, and its split by action
       (from the policy's biomass-weighted shares times dm_cost_rest / _eat / _move);
  G  - the energy paid into new biomass by the material growth law, MJ per tonne-year, and the new tonnes per
       tonne-year (realised somatic P/B);
  ration - tonnes eaten per tonne-year (env.intake_by_pred_prey);
  fill - reserve fill R / (B ME), biomass-weighted;
  cut  - the share of the demand's energy removed by the reserve_room cap (predation._reserve_room_cap).
Optional ARM (diagnostic food arms, every policy as trained): base (default); nokrill - the krill pool's biomass
and reserve set to 0 before every tick (its import still runs inside the tick and is removed again); halfkrill -
the krill pool halved before every tick.
usage: RUN=<run> POL=<dir> [ARM=base|nokrill|halfkrill] ration_budget.py <years>"""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/home/simon/eco-evo'); os.chdir('/home/simon/eco-evo')
years = int(sys.argv[1]); S = os.path.dirname(os.path.abspath(__file__))
import numpy as np, torch
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from inference import load_policies_and_stats
from lib.environments.ecosystem_env import decisions, interactions, movement, predation
from lib.world import daylight
from lib.world.tick_time import ticks_per_day
torch.set_num_threads(3)
cfg = json.load(open('results/' + os.environ['RUN'] + '/progress/config.json')); seed = cfg['seed']
keys = ("project", "library", "grid", "mortality", "migration", "mortality_multiplier", "mass_balance", "currents", "tick_hours")
eco = {k: cfg[k] for k in keys if k in cfg}
with evaluation_randomness(seed): env = build_inference_env(eco, seed)
with evaluation_randomness(seed):
    pol, om, ov = load_policies_and_stats(env, os.environ['POL'], verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = om, ov
    env.softmax_temperature = cfg.get('temperature', 1.0)
ARM = os.environ.get('ARM', 'base')
if ARM in ('nokrill', 'halfkrill'):
    _k = 0.0 if ARM == 'nokrill' else 0.5
    _tick = env.tick
    def _arm_tick():
        kr = env.fgs['krill']
        kr.biomass = (kr.biomass * _k).astype(env.dtype, copy=False)
        kr.energy_reserve = (kr.energy_reserve * _k).astype(env.dtype, copy=False)
        return _tick()
    env.tick = _arm_tick
dm = list(env.dm_ids)
FISH = [f for f in ('planktivorous_fish', 'benthivorous_fish', 'whiting', 'predatory_fish', 'mackerel') if f in dm]
ix = [dm.index(f) for f in FISH]
tick = {}

o_dec = decisions.actions_from_probabilities
def p_dec(env_, probs, sub):
    act = o_dec(env_, probs, sub)
    b = np.stack([env_.fgs[f].biomass for f in dm]).astype(float)
    tick['w_rest'] = (b * act.rest).sum(axis=(1, 2)) * env_.dm_cost_rest
    tick['w_eat'] = (b * act.eat.sum(axis=1)).sum(axis=(1, 2)) * env_.dm_cost_eat
    tick['w_move'] = (b * act.move.sum(axis=1)).sum(axis=(1, 2)) * env_.dm_cost_move
    return act
decisions.actions_from_probabilities = p_dec

o_cap = predation._reserve_room_cap
def p_cap(env_, demand, predator_biomass, reserve_offset):
    out = o_cap(env_, demand, predator_biomass, reserve_offset)
    e = env_.energy_gain_mat[:, :, None, None]
    before = (demand * e).sum(axis=(1, 2, 3)); after = (out * e).sum(axis=(1, 2, 3))
    tick['pot'] = before.astype(float); tick['kept'] = after.astype(float)
    return out
predation._reserve_room_cap = p_cap

o_ig = movement.income_growth
def p_ig(env_, biomass, reserve, gains, cost, sb, sr):
    nb, nr = o_ig(env_, biomass, reserve, gains, cost, sb, sr)
    tick['I'] = gains.sum(axis=(1, 2)).astype(float); tick['C'] = cost.sum(axis=(1, 2)).astype(float)
    tick['G'] = (sr - nr).sum(axis=(1, 2)).astype(float); tick['dB'] = (nb - sb).sum(axis=(1, 2)).astype(float)
    return nb, nr
movement.income_growth = p_ig

tpd = ticks_per_day(float(env.daylight['tick_hours'])); tpy = int(round(365 * tpd))
ME = env.dm_max_energy_reserve.astype(float)
F = len(FISH)
out = []
names = ('I', 'C', 'G', 'dB', 'eaten', 'Bsum', 'fillw', 'pot', 'kept', 'w_rest', 'w_eat', 'w_move')
with evaluation_randomness(seed):
    for y in range(years):
        acc = {k: np.zeros((12, F)) for k in names}; nt = np.zeros(12)
        for k in range(tpy):
            month = daylight.month_of_day(int(interactions.light_index(env) / tpd))
            b = np.array([float(env.fgs[f].biomass.sum()) for f in FISH])
            r = np.array([float(env.fgs[f].energy_reserve.sum()) for f in FISH])
            book0 = np.array([sum(env.intake_by_pred_prey.get(f, {}).values()) for f in FISH])
            tick.clear()
            env.tick()
            book1 = np.array([sum(env.intake_by_pred_prey.get(f, {}).values()) for f in FISH])
            acc['eaten'][month] += book1 - book0
            acc['Bsum'][month] += b
            acc['fillw'][month] += r / ME[ix]          # sum of R/ME; / sum B = biomass-weighted fill
            for key in ('I', 'C', 'G', 'dB', 'pot', 'kept', 'w_rest', 'w_eat', 'w_move'):
                if key in tick:
                    acc[key][month] += np.asarray(tick[key])[ix]
            nt[month] += 1
        rec = dict(year=y + 1, fish=FISH, monthly={}, annual={})
        for f_i, f in enumerate(FISH):
            def per_ty(a, m=None):
                if m is None:
                    return float(a[:, f_i].sum() / max(acc['Bsum'][:, f_i].sum(), 1e-30) * tpy)
                return float(a[m, f_i] / max(acc['Bsum'][m, f_i], 1e-30) * tpy)
            def row(m=None):
                sel = slice(None) if m is None else m
                pot = acc['pot'][sel, f_i].sum(); kept = acc['kept'][sel, f_i].sum()
                wr, we, wm = (acc[k][sel, f_i].sum() for k in ('w_rest', 'w_eat', 'w_move'))
                wt = max(wr + we + wm, 1e-30)
                return dict(I=per_ty(acc['I'], m), C=per_ty(acc['C'], m), G=per_ty(acc['G'], m),
                            PB=per_ty(acc['dB'], m), ration=per_ty(acc['eaten'], m),
                            fill=float(acc['fillw'][sel, f_i].sum() / max(acc['Bsum'][sel, f_i].sum(), 1e-30)),
                            cut=float(1 - kept / pot) if pot > 0 else 0.0,
                            C_rest=float(wr / wt), C_eat=float(we / wt), C_move=float(wm / wt))
            rec['annual'][f] = row()
            rec['monthly'][f] = [row(m) for m in range(12)]
        out.append(rec)
        rec['arm'] = ARM
        rec['fish_biomass'] = {f: float(acc['Bsum'][:, i].sum() / max(nt.sum(), 1)) for i, f in enumerate(FISH)}
        print(f"=== {ARM} year {y+1}: fish mean biomass t/km2 per cell x cells:", {f: round(v, 0) for f, v in rec['fish_biomass'].items()})
        print(f"=== year {y+1}: per tonne-year (MJ); ration t/t/yr; fill; cap cut; cost split rest/eat/move ===", flush=True)
        for f in FISH:
            a = rec['annual'][f]
            print(f"{f[:20]:20s} I {a['I']:8.0f}  C {a['C']:8.0f}  G {a['G']:7.0f}  I-C-G {a['I']-a['C']-a['G']:7.0f}  "
                  f"P/B {a['PB']:5.2f}  ration {a['ration']:5.2f}  fill {a['fill']:.2f}  cut {a['cut']:.2f}  "
                  f"C split {a['C_rest']:.2f}/{a['C_eat']:.2f}/{a['C_move']:.2f}", flush=True)
        for key, lab in (('ration', 'ration'), ('fill', 'fill'), ('cut', 'cap cut'), ('C', 'C MJ/t/yr'), ('G', 'G MJ/t/yr')):
            print(f"monthly {lab}:")
            for f in FISH:
                p = 0 if key in ('C', 'G') else 2
                print(f"  {f[:18]:18s}" + ''.join(f"{rec['monthly'][f][m][key]:7.{p}f}" for m in range(12)), flush=True)
os.makedirs(f"{S}/../tables", exist_ok=True)
json.dump(out, open(f"{S}/../tables/ration_budget_{os.environ['RUN']}" + ("" if ARM == 'base' else "_" + ARM) + ".json", 'w'))
