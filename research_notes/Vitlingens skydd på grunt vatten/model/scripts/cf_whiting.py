"""Why do whiting stand at 100-250 m and away from the benthivores? Payoffs of whiting placements (frozen world).

At 4 snapshots (days 0/91/182/273 after the warm-up) copy the world and place the whiting
  trained       as the policies left them
  field_kept    GOV bed-class shares (0.769/0.204/0.027/0.001), the trained pattern kept within each class
  field_even    GOV class shares, even within each class
  uniform       even over the grid
  in_clumps     proportional to the benthivores (into their clumps)
  trained_noBE  trained, but every cell with benthivores > 8x their mean emptied of whiting (moved pro rata
                to the rest): the avoidance's own payoff
wcF+ = the water column also at GOV saithe class shares (trained pattern within class); H: = whiting
reserves at the month's field start_fill (hungry, as in 278.8); HA: = whiting, benthivores (their month) and the
water column (0.71) all at field fill (278.8's 'both'). Then run 4 ticks (one day, all light phases). Readouts from the books: whiting intake (t/t/yr, net MJ/t/yr,
diet shares), M2 on whiting by each predator (/yr), fill. Also the pair depth multipliers per bed class.
Usage: cf_whiting.py POLICY_DIR SEED LIBRARY
"""
import sys, json, copy, numpy as np, yaml
sys.path.insert(0, '/home/simon/eco-evo')
from lib.runners.training_progress import build_inference_env, evaluation_randomness
from lib.runners.warmup import settle
from lib.world import habitat
from lib.diagnostics.field_diet import bed_class, FIELD_BED_SHARES
import inference

D, seed, LIB = sys.argv[1], int(sys.argv[2]), sys.argv[3]
WH, BE = 'whiting', 'benthivorous_fish'
SNAP = [0, 91, 182, 273]; W = 4
cfg = json.load(open(D + '/config.json')); cfg['library'] = LIB
with evaluation_randomness(seed):
    env = build_inference_env(cfg, seed)
    pol, m, v = inference.load_policies_and_stats(env, D, verbose=False)
    env.policies = pol; env.rebuild_batched_weights(); env.obs_mean, env.obs_var = m, v
    settle(env, cfg['warmup_ticks'], seed=seed)
    doy0 = (env.light_start + env.tick_count) % env.light_period / 4
man = yaml.safe_load(open(cfg['project']))
cls = bed_class(habitat.load_depth_map(man, cfg['project'], (env.H, env.W)))
field = np.array(FIELD_BED_SHARES[WH][0])
order = list(env.global_fg_order); iw = env.dm_ids.index(WH); jw = order.index(WH)
fmt = lambda s: '/'.join(f'{x:.2f}' for x in s)
shares = lambda x: np.array([x[cls == k].sum() for k in range(4)]) / max(x.sum(), 1e-30)

# pair depth multipliers by bed class (equal-cell mean 1 per pair)
pdm = getattr(env, 'pair_depth_mult', None)
print(f"seed {seed}; cells per class {fmt([(cls == k).mean() for k in range(4)])}")
if pdm is not None:
    for p in env.dm_ids:
        ip = env.dm_ids.index(p)
        if env.max_intake_mat[ip, jw] > 0:
            print(f"  depth mult {p:>18s} -> whiting   : {fmt([float(np.asarray(pdm[ip, jw])[cls == k].mean()) for k in range(4)])}")
    for q in order:
        jq = order.index(q)
        if env.max_intake_mat[iw, jq] > 0:
            print(f"  depth mult whiting -> {q:>18s}: {fmt([float(np.asarray(pdm[iw, jq])[cls == k].mean()) for k in range(4)])}")

def set_wh(e, B_new):
    fg = e.fgs[WH]
    B = fg.biomass.astype(np.float64); R = fg.energy_reserve.astype(np.float64); Jv = fg.juvenile.astype(np.float64)
    fill = R.sum() / max(B.sum(), 1e-30); js = Jv.sum() / max(B.sum(), 1e-30)
    B_new = B_new * B.sum() / B_new.sum()
    fg.biomass = B_new.astype(fg.biomass.dtype)
    fg.energy_reserve = (B_new * fill).astype(fg.energy_reserve.dtype)
    fg.juvenile = np.minimum(B_new * js, B_new).astype(fg.juvenile.dtype)

def place(e, mode):
    B = e.fgs[WH].biomass.astype(np.float64); Bb = e.fgs[BE].biomass.astype(np.float64)
    if mode == 'trained': return
    if mode == 'uniform':
        set_wh(e, np.ones_like(B)); return
    if mode == 'in_clumps':
        set_wh(e, Bb + 1e-12); return
    if mode == 'trained_noBE':
        hot = Bb > 8 * Bb.mean(); x = np.where(hot, 0.0, B); set_wh(e, x); return
    new = np.zeros_like(B)
    for k in range(4):
        c = cls == k
        if not c.any(): continue
        if mode == 'field_kept' and B[c].sum() > 1e-9 * B.sum():
            new[c] = B[c] / B[c].sum() * field[k]
        else:
            new[c] = field[k] / c.sum()
    set_wh(e, new)

WC = 'predatory_fish'
wc_field = np.array(FIELD_BED_SHARES[WC][0])
def place_wc_field(e):
    fg = e.fgs[WC]
    B = fg.biomass.astype(np.float64); R = fg.energy_reserve.astype(np.float64)
    tot = B.sum(); fill = R.sum() / max(tot, 1e-30); newB = np.zeros_like(B); newR = np.zeros_like(R)
    for k in range(4):
        c = cls == k
        if not c.any(): continue
        bk = B[c].sum(); target = wc_field[k] * tot
        if bk > 1e-9 * tot:
            newB[c] = B[c] * target / bk; newR[c] = R[c] * target / bk
        else:
            newB[c] = target / c.sum(); newR[c] = newB[c] * fill
    s_ = tot / newB.sum()
    fg.biomass = (newB * s_).astype(fg.biomass.dtype); fg.energy_reserve = (newR * s_).astype(fg.energy_reserve.dtype)
    if getattr(fg, 'juvenile', None) is not None:
        fg.juvenile = np.minimum(fg.juvenile, fg.biomass)

def books(e):
    on = {p: float(e.intake_by_pred_prey.get(p, {}).get(WH, 0.0)) for p in e.dm_ids}
    own = {q: float(x) for q, x in e.intake_by_pred_prey.get(WH, {}).items()}
    return on, own

MODES = ['trained', 'field_kept', 'field_even', 'uniform', 'in_clumps', 'trained_noBE',
         'wcF+trained', 'wcF+field_even', 'wcF+uniform', 'H:trained', 'H:wcF+trained', 'H:wcF+field_even', 'HA:wcF+trained', 'HA:wcF+field_even']
def set_fill_g(e, g, f):
    fg = e.fgs[g]
    fg.energy_reserve = (f * fg.biomass.astype(np.float64) * fg.max_energy_reserve).astype(fg.energy_reserve.dtype)
def set_fill(e, month):
    fg = e.fgs[WH]; f = float(fg.params['start_fill'][month])
    fg.energy_reserve = (f * fg.biomass.astype(np.float64) * fg.max_energy_reserve).astype(fg.energy_reserve.dtype)
res = {mo: [] for mo in MODES}
day = 0.0
with evaluation_randomness(seed * 13 + 5):
    for k, sd in enumerate(SNAP):
        while day < sd:
            env.step(); day += 0.25
        Bw = env.fgs[WH].biomass.astype(np.float64); Bb = env.fgs[BE].biomass.astype(np.float64)
        print(f"\n=== day {sd}: whiting classes {fmt(shares(Bw))} (GOV {fmt(field)}), benthivores {fmt(shares(Bb))}, "
              f"WC {fmt(shares(env.fgs['predatory_fish'].biomass.astype(np.float64)))}, krill "
              f"{fmt(shares(env.fgs['krill'].biomass.astype(np.float64)))}", flush=True)
        for mo in MODES:
            e = copy.deepcopy(env)
            mm = mo.split(':', 1)[1] if ':' in mo else mo
            if mm.startswith('wcF+'): place_wc_field(e); place(e, mm[4:])
            else: place(e, mm)
            month = min(11, int(((doy0 + sd) % 365) / 30.42))
            if mo.startswith('H:') or mo.startswith('HA:'):
                set_fill(e, month)
            if mo.startswith('HA:'):
                set_fill_g(e, WC, 0.71); set_fill_g(e, BE, float(e.fgs[BE].params['start_fill'][month]))
            fg = e.fgs[WH]
            on0, own0 = books(e); Bs = 0.0
            with evaluation_randomness(1000 * seed + k):
                for t in range(W):
                    Bs += float(fg.biomass.sum()); e.step()
            on1, own1 = books(e); Bm = Bs / W; yr = 1460 / W
            m2 = {p: (on1[p] - on0[p]) / Bm * yr for p in on1}
            d = {q: own1.get(q, 0) - own0.get(q, 0) for q in own1}; tot = max(sum(d.values()), 1e-30)
            eat = tot / Bm * yr
            mj = sum(x * float(e.energy_gain_mat[iw, order.index(q)]) for q, x in d.items()) / Bm * yr
            fill = float(fg.energy_reserve.sum() / max(fg.biomass.sum(), 1e-30) / float(fg.max_energy_reserve))
            r = dict(eat=eat, mj=mj, m2=sum(m2.values()), m2p=m2, diet={q: x / tot for q, x in d.items()}, fill=fill)
            res[mo].append(r)
            top = ', '.join(f"{p[:5]} {x:.3f}" for p, x in sorted(m2.items(), key=lambda z: -z[1]) if x > 1e-4)
            dt = ', '.join(f"{q[:5]} {x:.2f}" for q, x in sorted(r['diet'].items(), key=lambda z: -z[1]) if x > 0.005)
            print(f"  {mo:12s} eat {eat:5.2f} t/t/yr MJ {mj:6.0f} fill {fill:.3f} | M2 {r['m2']:.3f} ({top}) | diet {dt}", flush=True)

print("\n=== means over the 4 snapshots")
for mo in MODES:
    R = res[mo]
    m2p = {p: np.mean([r['m2p'][p] for r in R]) for p in R[0]['m2p']}
    print(f"  {mo:12s} eat {np.mean([r['eat'] for r in R]):5.2f} MJ {np.mean([r['mj'] for r in R]):6.0f} "
          f"M2 {np.mean([r['m2'] for r in R]):.3f} (" + ', '.join(f"{p[:5]} {x:.3f}" for p, x in
          sorted(m2p.items(), key=lambda z: -z[1]) if x > 1e-4) + ")")
