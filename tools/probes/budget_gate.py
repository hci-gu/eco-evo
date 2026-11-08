"""Energy break-even per decision maker, over the DIET rather than per pair.

What the engine actually does
-----------------------------
``predation.apply_predation`` sums the intake over every prey on the
menu and hands the predator ONE energy gain; the hunger gate and the
feeding cost are applied once, to that total. The budget a DM has to
close is therefore a property of its *ration*, not of any single pair.

A DM that eats at 100 % of the time, at the reserve level where it just
breaks even (s_X = u_X), takes in

    ration = a * h(u_X)                         [t prey / t pred / tick]
    gain   = ration * quality                   [MJ / t pred / tick]

where ``quality`` is the ration-weighted mean assimilated energy content
of the prey it eats, and pays

    cost = resting_metabolism * feeding_cost    [MJ / t pred / tick]

so survival by eating alone requires

    quality >= need_quality = cost / ration,
    h(u_X)  = min(1, (1 - u_X) / (1 - u_X)) = 1 (section 133).

Why the per-pair form was wrong (Section 74.2b, superseded)
-----------------------------------------------------------
This probe used to evaluate the identity pair by pair and label a pair
INFEASIBLE when its own ``sat_min`` exceeded 1. For
``porpoises -> gadoids`` it reported 1.227 and Section 74.2b concluded
that no cod density whatsoever can pay for a porpoise, i.e. that the
pair was a calibration bug.

It is not a bug and the pair is not "pure loss". It is the junk-food
hypothesis (MacLeod et al. 2007; Spitz et al. 2012): a porpoise cannot
live on lean gadoid ALONE, and the literature says the same. What the
model must be checked against is the minimum share of the high-quality
prey in the ration - reported below as ``min share`` - and Kattegat /
Skagerrak stomach data (50-70 % clupeids by mass) clears it.

Printed next to each requirement is the density the live probe world
actually delivers, so feasible-but-unreachable stays distinguishable
from infeasible.
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import train as train_mod
from lib.world.energy_balance import hunger_at, intake_ceiling
from lib.world.tick_time import LIBRARY_TICK_HOURS, tick_label, ticks_per_day

PROJECT = 'mareld2.yaml'
# This probe reads fg_library.yaml's own numbers, and those mean
# LIBRARY_TICK_HOURS by definition - a run's --tick-length converts them
# in memory and never writes back, so it cannot change what is on disk.
# %bm/day is the only real-time figure here. Sections 97 and 106.
TICK_HOURS = LIBRARY_TICK_HOURS
TICKS_PER_DAY = ticks_per_day(TICK_HOURS)

H = W = 16
builder = train_mod._ProbeEnvBuilder(
    project_path=PROJECT, grid_size=(H, W),
    apply_natural_mortality=True, migration=True)
env = builder()
env._build_static_caches()


def _pred_costs(i, fg_id):
    """Resting metabolism and the eat-cost multiplier, from the caches.

    ``feeding_cost`` left ``FunctionalGroup`` when the action costs
    moved into the env caches (``dm_cost_eat``), which is what made this
    probe unrunnable - and Section 74's numbers unreproducible - for
    several arcs.
    """
    rm = float(env.dm_resting_metabolism[i])
    c_eat = float(env.dm_cost_eat[i])
    params = env.fgs[fg_id].params
    u_x = float(params.get('maintenance_level', 0.0) or 0.0)
    return rm, c_eat, u_x


print("=== per (predator, prey): what ONE prey alone would have to do ===")
print("A row that does not clear the bar on its own is low-quality food,")
print("not pure loss - the verdict is the per-predator block below.")
print()
print(f"{'predator':>14} {'prey':>18} {'u_x':>5} {'h(u)':>6} "
      f"{'cost':>7} {'a*gain':>8} {'sat_min':>8} {'need_vis':>9} "
      f"{'B/cell':>8} {'max':>8} {'alone':>12}")

for i, pid in enumerate(env.dm_ids):
    rm, c_eat, u_x = _pred_costs(i, pid)
    h_u = hunger_at(u_x, u_x)
    cost = rm * c_eat
    for j, prey_id in enumerate(env.global_fg_order):
        if not env.eat_static_mask[i, j]:
            continue
        a = float(env.max_intake_mat[i, j])
        ht = float(env.handling_time_mat[i, j])
        gain_t = float(env.energy_gain_mat[i, j])
        # Holling ceiling 1/h, not the attack rate (section 134).
        max_gain = intake_ceiling(a, ht) * gain_t
        if h_u <= 0.0 or max_gain <= 0.0:
            sat_min = float('inf')
        else:
            sat_min = cost / (max_gain * h_u)
        Bp = env.fgs[prey_id].biomass
        occ = Bp > 0
        mean_occ = float(Bp[occ].mean()) if occ.any() else 0.0
        mx = float(Bp.max())
        if sat_min >= 1.0:
            need = float('inf')
            verdict = "not alone"
        else:
            need = sat_min / ((1.0 - sat_min) * a * ht) if a * ht > 0 else 0.0
            verdict = "ok" if mx >= need else "unreachable"
        print(f"{pid:>14} {prey_id:>18} {u_x:>5.2f} "
              f"{h_u:>6.3f} {cost:>7.1f} {max_gain:>8.1f} {sat_min:>8.3f} "
              f"{need:>9.3f} {mean_occ:>8.2f} {mx:>8.2f} {verdict:>12}")

print()
print("=== per predator: the budget the engine actually closes ===")
print(f"{'predator':>14} {'ration':>9} {'%bm/day':>8} {'cost':>7} "
      f"{'need_q':>9} {'best_q':>9} {'worst_q':>9} {'min share':>10} "
      f"{'verdict':>12}")

for i, pid in enumerate(env.dm_ids):
    rm, c_eat, u_x = _pred_costs(i, pid)
    h_u = hunger_at(u_x, u_x)
    cost = rm * c_eat
    menu = [j for j in range(env.N_all) if env.eat_static_mask[i, j]]
    if not menu or h_u <= 0.0:
        print(f"{pid:>14} {'-':>9} {'-':>8} {cost:>7.1f} "
              f"{'-':>9} {'-':>9} {'-':>9} {'-':>10} {'NO INTAKE':>12}")
        continue

    # The physiological ration is a per-predator property; the library
    # convention is one max_intake_rate per predator, so take the row
    # maximum rather than assuming the pairs agree.
    a = max(intake_ceiling(float(env.max_intake_mat[i, j]),
                           float(env.handling_time_mat[i, j])) for j in menu)
    ration = a * h_u
    quality = {env.global_fg_order[j]: float(env.energy_gain_mat[i, j])
               for j in menu}
    need_q = cost / ration if ration > 0 else float('inf')
    best_id = max(quality, key=quality.get)
    worst_id = min(quality, key=quality.get)
    best_q, worst_q = quality[best_id], quality[worst_id]

    if best_q < need_q:
        share, verdict = float('nan'), "INFEASIBLE"
    elif worst_q >= need_q:
        share, verdict = 0.0, "any diet"
    else:
        share = (need_q - worst_q) / (best_q - worst_q)
        verdict = "mix needed"
    print(f"{pid:>14} {ration:>9.5f} {ration*TICKS_PER_DAY*100:>7.1f}% "
          f"{cost:>7.1f} {need_q:>9.0f} {best_q:>9.0f} {worst_q:>9.0f} "
          f"{share:>10.3f} {verdict:>12}")
    if verdict == "mix needed":
        print(f"{'':>14} at least {share*100:.1f} % {best_id} in the ration, "
              f"the rest {worst_id}")

print()
print(f"Units: ration in t prey / t predator / {tick_label(TICK_HOURS)}; "
      f"cost and quality in")
print("MJ / t. quality is (assimilation_factor - sda_fraction) * energy_content, i.e. the")
print("energy the predator keeps per tonne eaten.")

print()
print("Sensitivity for porpoises -> planktivorous_fish (the pair in question):")
i = env.dm_ids.index('porpoises')
j = env.global_fg_order.index('planktivorous_fish')
a = float(env.max_intake_mat[i, j])
ht = float(env.handling_time_mat[i, j])
gain_t = float(env.energy_gain_mat[i, j])
cost = float(env.dm_resting_metabolism[i]) * float(env.dm_cost_eat[i])
print(f"{'u_x':>6} {'h(u)':>7} {'sat_min':>8} "
      f"{'need_visible_t/cell':>20}")
for u_x in (0.5, 0.4, 0.3):
    h_u = hunger_at(u_x, u_x)
    sat_min = cost / (intake_ceiling(a, ht) * gain_t * h_u)
    need = (sat_min / ((1.0 - sat_min) * a * ht)
            if sat_min < 1.0 else float('inf'))
    print(f"{u_x:>6.2f} {h_u:>7.3f} {sat_min:>8.3f} "
          f"{need:>20.3f}")

# ---------------------------------------------------------------------------
# Diet gate (section 232): does the encounter side of each generalist's
# pairs reproduce its booked diet at today's densities?
#
# Under concurrent search (one predator meeting every prey on its menu in
# the same time) the mass taken of prey i is proportional to its
# encounter a_i * B_vis_i (Type III: a_i * B_vis_i**2), whatever the
# saturation. The booked diet is predation_mortality_i * B_i, the flow
# the growth budget (section 129) pays for. Both come from the library
# and the manifest alone - no policy, no rollout - so a pair whose shares
# disagree is inconsistent with the field diet it was booked from, not
# with any model outcome. Densities are the manifest's inference start
# over the reference grid; the visible part is the pair's
# accessible_fraction, its stage access at the prey's annual-mean start
# J/B (section 223.32) and a refuge prey's exposed share (223.12). All
# prey are taken as visible: hiding, darkness and the mixed-layer
# scaling of the plankton pairs are left out, so zooplankton (every pair
# scaled by M_ref/M(t)) is skipped. The factor-3 band is a judgement:
# wide enough for stomach-data spread, narrow enough to catch a pair
# that is an order of magnitude off. Since section 242 the engine's
# response is this concurrent search with one handling budget: a policy
# that attacks every prey it meets (p = 1, the optimum when E/h is equal)
# takes exactly these encounter shares, so an OFF pair is a diet the
# trained engine will now realise.
# ---------------------------------------------------------------------------
import yaml

DIET_BAND = 3.0
DIET_MIN_SHARE = 0.05


def _start_densities():
    with open(PROJECT, encoding='utf-8-sig') as fh:
        manifest = yaml.safe_load(fh)
    meta = manifest.get('project_metadata', {})
    area = float(meta['reference_grid_width']) * float(meta['reference_grid_height'])
    out = {}
    for entries in manifest.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and 'group_id' in entry \
                    and entry.get('inference_initial_biomass') is not None:
                out[entry['group_id']] = float(entry['inference_initial_biomass']) / area
    return out


def _depth_factor(i, j, prey_id):
    """Section 245: the pair's per-cell depth multiplier on a, averaged over
    the cells with the prey's own outside-density shape (an open pool;
    uniform otherwise) - the equal-cell a turned into the encounter the
    prey's spatial pattern meets. 1 without a depth table."""
    if not getattr(env, '_has_pair_depth', False):
        return 1.0
    shape = getattr(env.fgs[prey_id], 'outside_weight_map', None)
    mult = env.pair_depth_mult[i, j]
    if shape is None:
        return float(mult.mean())
    return float((mult * shape).mean())


def _visible_share(i, j, prey_id):
    if getattr(env, '_has_stages', False) and j in set(env.stage_rows.tolist()):
        js = float(env.stage_start[:, j].mean())
        f = js * float(env.stage_f_j[i, j]) + (1.0 - js) * float(env.stage_f_a[i, j])
    else:
        f = float(env.access_mat[i, j])
    phi = getattr(env.fgs[prey_id], 'exposed_share', None)
    if phi is not None and not env.fgs[prey_id].is_decision_maker:
        f *= float(phi)
    return f


print()
print("=== diet gate: encounter-implied diet vs booked diet (section 232) ===")
print("enc = a*B_vis share (concurrent search), booked = M2*B share; ratio")
print(f"outside 1/{DIET_BAND:g}-{DIET_BAND:g} on a pair with either share >= "
      f"{DIET_MIN_SHARE:g} is OFF. sat = a h B_vis / (1 + a h B_vis), the")
print("share of the ceiling the pair pays on its own. Densities t/km2.")
density = _start_densities()
gate_failures = []
# Section 249: per predator (booked flow sum, rows) for the prey-side check.
gate_rows = {}
for i, pid in enumerate(env.dm_ids):
    menu = [j for j in range(env.N_all) if env.eat_static_mask[i, j]]
    if len(menu) < 2:
        continue
    if pid == 'zooplankton':
        print(f"{pid:>14}: skipped (plankton pairs scale with the mixed layer)")
        continue
    pairs = env.fgs[pid].params.get('interaction', {}) or {}
    rows = []
    for j in menu:
        prey_id = env.global_fg_order[j]
        b = density.get(prey_id)
        if b is None:
            continue
        a = float(env.max_intake_mat[i, j]) * _depth_factor(i, j, prey_id)
        ht = float(env.handling_time_mat[i, j])
        bv = b * _visible_share(i, j, prey_id)
        type3 = float(env._type3_pred_mask[i, j, 0, 0]) > 0.0
        enc = a * bv * (bv if type3 else 1.0)
        x = enc * ht
        m2 = float((pairs.get(f"{pid}_preys_on_{prey_id}") or {})
                   .get('predation_mortality', 0.0) or 0.0)
        rows.append((prey_id, b, bv, enc, m2 * b, x / (1.0 + x), type3))
    enc_sum = sum(r[3] for r in rows)
    book_sum = sum(r[4] for r in rows)
    if enc_sum <= 0.0 or book_sum <= 0.0:
        continue
    gate_rows[pid] = (book_sum, {r[0]: r[3] / enc_sum for r in rows})
    print(f"{pid:>14} {'prey':>18} {'B':>7} {'B_vis':>7} {'enc':>6} "
          f"{'booked':>6} {'ratio':>7} {'sat':>5}  verdict")
    for prey_id, b, bv, enc, book, sat, type3 in rows:
        e, k = enc / enc_sum, book / book_sum
        ratio = e / k if k > 0.0 else float('inf')
        off = (max(e, k) >= DIET_MIN_SHARE
               and not (1.0 / DIET_BAND <= ratio <= DIET_BAND))
        verdict = "OFF" if off else "ok"
        if off:
            gate_failures.append((pid, prey_id, e, k))
        print(f"{'':>14} {prey_id:>18} {b:>7.2f} {bv:>7.3f} {e:>6.3f} "
              f"{k:>6.3f} {ratio:>7.2f} {sat:>5.2f}  {verdict}"
              f"{' (Type III)' if type3 else ''}")
print()
if gate_failures:
    print("Diet gate: OFF pairs - "
          + "; ".join(f"{p} -> {q} enc {e:.3f} vs booked {k:.3f}"
                      for p, q, e, k in gate_failures))
else:
    print("Diet gate: every generalist's encounters match its booked diet.")

# ---------------------------------------------------------------------------
# Diet gate at the field spread (section 248). The gate above spreads every
# predator evenly over the cells; a pair with a depth_access_table then
# meets the prey on beds where the field predator is not (the water column
# on the < 150 m beds that carry its krill a). Here each group with GOV
# bed-depth shares is spread over the cells as in the field and every pair's
# encounter a(c) B_vis(c) is averaged with those weights, the prey at their
# start maps. A like-for-like reading of the same booked diet; it sets
# nothing. research_notes/"Krill i Skagerrak" a_revisit/.
# ---------------------------------------------------------------------------
from lib.diagnostics.field_diet import FIELD_BED_SHARES, field_cell_weights  # noqa: E402
from lib.world import habitat  # noqa: E402

with open(PROJECT, encoding='utf-8-sig') as fh:
    _manifest = yaml.safe_load(fh)
_depth = habitat.load_depth_map(_manifest, PROJECT, (H, W))
_meta = _manifest.get('project_metadata', {})
_cell_km2 = (float(_meta['reference_grid_width']) * float(_meta['reference_grid_height'])
             / float(H * W))
print()
print("=== diet gate at the field spread (section 248) ===")
print("The predator over the bed-depth classes < 100 / 100-150 / 150-250 / > 250 m as in")
print("NS-IBTS GOV hauls inside the box; encounters averaged over its cells, prey at")
print("their start maps.")
if _depth is not None:
    from lib.diagnostics.field_diet import bed_class  # noqa: E402
    _cls = bed_class(_depth)
    print(f"cells per class at this {H}x{W} grid: "
          + " / ".join(f"{(_cls == k).mean():.3f}" for k in range(4)))
field_failures = []
field_rows = {}
for i, pid in enumerate(env.dm_ids):
    if pid not in FIELD_BED_SHARES or _depth is None:
        continue
    shares, label = FIELD_BED_SHARES[pid]
    w = field_cell_weights(_depth, shares)
    pairs = env.fgs[pid].params.get('interaction', {}) or {}
    rows = []
    for j in range(env.N_all):
        if not env.eat_static_mask[i, j]:
            continue
        prey_id = env.global_fg_order[j]
        if prey_id not in density:
            continue
        mult = (np.asarray(env.pair_depth_mult[i, j], dtype=np.float64)
                if getattr(env, '_has_pair_depth', False) else 1.0)
        bv = (np.asarray(env.fgs[prey_id].biomass, dtype=np.float64) / _cell_km2
              * _visible_share(i, j, prey_id))
        type3 = float(env._type3_pred_mask[i, j, 0, 0]) > 0.0
        enc = float((w * float(env.max_intake_mat[i, j]) * mult * bv
                     * (bv if type3 else 1.0)).sum())
        m2 = float((pairs.get(f"{pid}_preys_on_{prey_id}") or {})
                   .get('predation_mortality', 0.0) or 0.0)
        rows.append((prey_id, enc, m2 * density[prey_id]))
    enc_sum = sum(r[1] for r in rows)
    book_sum = sum(r[2] for r in rows)
    if enc_sum <= 0.0 or book_sum <= 0.0:
        continue
    field_rows[pid] = (book_sum, {r[0]: r[1] / enc_sum for r in rows})
    print(f"{pid:>14} ({label}: {' / '.join(f'{x:.3f}' for x in shares)})")
    for prey_id, enc, book in rows:
        e, k = enc / enc_sum, book / book_sum
        ratio = e / k if k > 0.0 else float('inf')
        off = (max(e, k) >= DIET_MIN_SHARE
               and not (1.0 / DIET_BAND <= ratio <= DIET_BAND))
        if off:
            field_failures.append((pid, prey_id, e, k))
        print(f"{'':>14} {prey_id:>18} enc {e:>6.3f} booked {k:>6.3f} "
              f"ratio {ratio:>6.2f}  {'OFF' if off else 'ok'}")
print()
if field_failures:
    print("Diet gate at the field spread: OFF pairs - "
          + "; ".join(f"{p} -> {q} enc {e:.3f} vs booked {k:.3f}"
                      for p, q, e, k in field_failures))
else:
    print("Diet gate at the field spread: every pair within the band.")

# ---------------------------------------------------------------------------
# Prey-side check (section 249). The diet gate judges a pair by its share of
# the PREDATOR's diet, so a pair that is minor for the predator and major for
# the prey (benthivores -> whiting: 0.012 of the benthivores' booked diet, 0.18
# of whiting's 0.37 /yr booked M2) passes unseen - yet the prey's derived g
# pays that M2 (growth_budget). Here each predator eats its booked ration
# split by the gate's encounter shares (shared handling), so the implied M2
# of pair (i, j) is booked_flow_i x enc_ij / B_j, i.e. booked M2 x the gate's
# ratio. A pair carrying >= PREY_MIN_SHARE of its prey's booked M2 with a
# ratio outside the band is OFF. Predators the gate skips (one prey,
# zooplankton) count at their booked value. Diagnostic only.
# ---------------------------------------------------------------------------
PREY_MIN_SHARE = 0.10
print()
print("=== prey-side check: encounter-implied M2 vs booked M2 per prey (section 249) ===")
print("implied = booked M2 x (encounter share / booked share) of the predator's diet;")
print(f"a pair carrying >= {PREY_MIN_SHARE:g} of its prey's booked M2 with a ratio outside "
      f"1/{DIET_BAND:g}-{DIET_BAND:g} is OFF.")
print("field = the same with the predator at its field bed-depth spread (where it has one).")
booked_m2 = {}
for i, pid in enumerate(env.dm_ids):
    pairs = env.fgs[pid].params.get('interaction', {}) or {}
    for j in range(env.N_all):
        if not env.eat_static_mask[i, j]:
            continue
        prey_id = env.global_fg_order[j]
        m2 = float((pairs.get(f"{pid}_preys_on_{prey_id}") or {})
                   .get('predation_mortality', 0.0) or 0.0)
        if m2 > 0.0 and prey_id in density:
            booked_m2.setdefault(prey_id, {})[pid] = m2


def _implied(table, pid, prey_id, m2):
    if pid not in table:
        return None
    book_sum, enc = table[pid]
    return book_sum * enc.get(prey_id, 0.0) / density[prey_id]


prey_failures = []
for prey_id in sorted(booked_m2):
    row = booked_m2[prey_id]
    total = sum(row.values())
    implied_total = sum((_implied(gate_rows, p, prey_id, m) if p in gate_rows else m)
                        for p, m in row.items())
    print(f"{prey_id:>20}: booked M2 {total:.4g} /yr, implied {implied_total:.4g} "
          f"(ratio {implied_total / total:.2f})")
    for p, m2 in sorted(row.items(), key=lambda kv: -kv[1]):
        imp = _implied(gate_rows, p, prey_id, m2)
        fld = _implied(field_rows, p, prey_id, m2)
        share = m2 / total
        if imp is None:
            print(f"{'':>22}{p:>20} booked {m2:8.4g} ({share:4.2f})  not judged (gate skips it)")
            continue
        ratio = imp / m2
        off = share >= PREY_MIN_SHARE and not (1.0 / DIET_BAND <= ratio <= DIET_BAND)
        if off:
            prey_failures.append((p, prey_id, m2, imp))
        extra = f"  field {fld:8.4g} ({fld / m2:5.2f})" if fld is not None else ""
        print(f"{'':>22}{p:>20} booked {m2:8.4g} ({share:4.2f})  implied {imp:8.4g} "
              f"ratio {ratio:6.2f}  {'OFF' if off else 'ok'}{extra}")
print()
if prey_failures:
    print("Prey-side check: OFF pairs - "
          + "; ".join(f"{p} -> {q} booked M2 {m:.4g} vs implied {x:.4g}"
                      for p, q, m, x in prey_failures))
else:
    print("Prey-side check: every pair that matters to its prey is within the band.")

# ---------------------------------------------------------------------------
# Total ration (section 245): the gate above compares SHARES and cannot see
# a booked ration that is short of the predator's field consumption (232.1
# cut whiting -> planktivores without booking the ration elsewhere; whiting
# then ate the unbooked 1.6 t/t/yr from the saturated bed pair). Booked
# ration = sum over the checked pairs of M2 x B_prey / B_pred at the start
# densities, against the group's field Q/B (research_notes/"Vitlingens
# ranson" summary.md, its sources). Diagnostic only; nothing is set from it.
# ---------------------------------------------------------------------------
from lib.diagnostics.field_diet import FIELD_QB  # noqa: E402
print()
print("=== total ration: booked flow vs field Q/B (section 245) ===")
print(f"{'predator':>20} {'booked':>7} {'field':>6} {'ratio':>6}  source")
for i, pid in enumerate(env.dm_ids):
    if pid not in FIELD_QB or pid not in density:
        continue
    pairs = env.fgs[pid].params.get('interaction', {}) or {}
    booked = 0.0
    for j in range(env.N_all):
        if not env.eat_static_mask[i, j]:
            continue
        prey_id = env.global_fg_order[j]
        m2 = float((pairs.get(f"{pid}_preys_on_{prey_id}") or {})
                   .get('predation_mortality', 0.0) or 0.0)
        booked += m2 * density.get(prey_id, 0.0) / density[pid]
    qb, source = FIELD_QB[pid]
    print(f"{pid:>20} {booked:>7.2f} {qb:>6.2f} {booked / qb:>6.2f}  {source}")
