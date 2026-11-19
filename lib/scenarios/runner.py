"""Run one scenario variant with one seed and write its results.

The world is built exactly as ``inference.py`` builds it (``build_env``,
the trained policies and frozen observation statistics, the run's
warm-up) and is stepped on the tensor engine (``lib/gpu``): CUDA graphs on
a GPU machine, eager on the CPU for small local tests.

Output, in ``<out_dir>/``:

* ``timeseries.npz`` - one row per ``record_every_days``:
  ``t_days`` (days since the recorded start), ``biomass``, ``reserve``,
  ``juvenile`` [T, G] tonnes / MJ, ``losses`` [T, 5, G] cumulative tonnes
  by cause (``loss_causes``), ``imports`` [T, G] cumulative tonnes in from
  outside the box, ``catch`` [T, I, G] cumulative tonnes removed by each
  impact (``impact_ids``), ``scale`` [T, I] the impact intensities in
  force over the interval ending at each row (row 0: the start), and
  ``group_ids``.
* ``spatial.npz`` (``spatial_every_days`` > 0) - ``t_days`` and
  ``biomass`` [S, G, H, W] float16.
* ``meta.json`` - the scenario, seed, world options, git revision,
  timings.

``catch`` splits the engine's booked impact loss (exact, per group) over
the impacts by each impact's share of the expected loss at the start of
the record interval (sum over cells of B x its biomass factor). With one
impact acting on a group the split is exact; with two it is exact to the
change in the spatial biomass within one record interval.
"""

import json
import os
import subprocess
import time
from argparse import Namespace

import numpy as np
import torch

from lib.scenarios.spec import MaterializedManifest, expand_impact_ids, scenario_manifest
from lib.world.impact_maps import load_impact_map_npz


DAY_HOURS = 24.0


def _git_revision():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"],
                                stderr=subprocess.DEVNULL) != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return None


# ------------------------------------------------------------------ world

def apply_warming(env, delta_c):
    """Raise every temperature layer by ``delta_c`` relative to today.

    The engine's temperature tables are Q10 multipliers. Metabolism with a
    fixed t_ref scales by q10^(dT/10) under a uniform warming; the tables
    normalised to an annual mean of 1 (metabolism with t_ref annual_mean,
    ``ingestion_q10``, ``growth_q10``) would renormalise a shifted table
    back to the same values, so a warming would vanish from them. Here
    every Q10 acts relative to TODAY's climate: each table is multiplied
    by its q10^(dT/10). Producers, nutrients and the mixed layer are left
    as they are. Must run before the tensor engine is built from ``env``.
    Returns ``{group: {table: factor}}``.
    """
    from lib.world import temperature
    delta = float(delta_c or 0.0)
    applied = {}
    if delta == 0.0:
        return applied
    for i, fid in enumerate(env.dm_ids):
        fg = env.fgs[fid]
        if not fg.params.get("temperature"):
            continue
        settings = temperature.species_settings(fg.params)
        if settings is not None and getattr(env, "metabolism_temp_table", None) is not None:
            factor = float(settings[0]) ** (delta / 10.0)
            env.metabolism_temp_table[:, i] *= factor
            applied.setdefault(fid, {})["metabolism"] = factor
        q10 = temperature.ingestion_q10(fg.params)
        if q10 is not None and getattr(env, "ingestion_temp_table", None) is not None:
            factor = float(q10) ** (delta / 10.0)
            env.ingestion_temp_table[:, i] *= factor
            applied.setdefault(fid, {})["ingestion"] = factor
        q10 = temperature.growth_q10(fg.params)
        if q10 is not None and getattr(env, "growth_temp_table", None) is not None:
            factor = float(q10) ** (delta / 10.0)
            env.growth_temp_table[:, i] *= factor
            applied.setdefault(fid, {})["growth"] = factor
    for entry in getattr(env, "ndm_grazers", []) or []:
        if entry.get("temperature") is None:
            continue
        q10 = float(env.fgs[entry["grazer"]].params.get("grazing_q10"))
        factor = q10 ** (delta / 10.0)
        entry["temperature"] = (entry["temperature"] * factor).astype(
            entry["temperature"].dtype)
        applied.setdefault(entry["grazer"], {})[f"grazing_{entry['prey']}"] = factor
    return applied


def _scaled_fields(base_maps, masks, scale, impact_ids):
    """``{impact_id: field}``: the loaded field x its intensity, zero on
    every mask that names it."""
    fields = {}
    for iid in impact_ids:
        base = base_maps.get(iid)
        if base is None:
            continue
        field = np.asarray(base, dtype=np.float64) * float(scale.get(iid, 1.0))
        for mask, names in masks:
            if iid in names:
                field = field * (1.0 - mask)
        fields[iid] = field.astype(np.float32)
    return fields


def _load_mask(path, key, H, W):
    with np.load(path, allow_pickle=False) as data:
        arr = np.asarray(data[key], dtype=np.float32)
    if arr.shape != (H, W):
        rows = (np.arange(H) * arr.shape[0] / H).astype(np.int64)
        cols = (np.arange(W) * arr.shape[1] / W).astype(np.int64)
        arr = arr[rows[:, None], cols[None, :]]
    return np.clip(arr, 0.0, 1.0)


# ------------------------------------------------------------------ run

def run_variant(variant, seed, out_dir, device="cpu", grid=None, years=None,
                ticks=None, run_name=None, warmup_days=None, verbose=True):
    """Run ``variant`` with ``seed``; write ``out_dir``; return its meta."""
    import inference as inf
    from lib.environments.ecosystem_env.currents import CurrentConfig
    from lib.gpu.inference import build_runner
    from lib.runners.warmup import settle, warmup_ticks

    started = time.time()
    run = variant.run
    if grid is not None:
        run["grid"] = list(grid)
    if warmup_days is not None:
        run["warmup_days"] = float(warmup_days)
    H, W = (int(v) for v in run["grid"])
    tick_hours = int(run["tick_length"])
    ticks_per_day = DAY_HOURS / tick_hours
    run_name = run_name or variant.run_name
    if not run_name:
        raise ValueError(f"{variant.scenario}: no run_name (scenario or --run-name)")
    checkpoints = os.path.join("results", run_name)
    if ticks is None:
        ticks = int(round((years if years is not None else variant.years)
                          * 365.0 * ticks_per_day))
    record_ticks = max(1, int(round(variant.record_every_days * ticks_per_day)))
    spatial_ticks = (int(round(variant.spatial_every_days * ticks_per_day))
                     if variant.spatial_every_days > 0 else 0)
    currents = None
    if run["currents"] == "on":
        currents = CurrentConfig(float(run["current_strength"]), int(run["current_period"]),
                                 int(run["current_seed"]), float(run["current_scale"]))

    with MaterializedManifest(variant) as manifest:
        env = inf.build_env(manifest, (H, W), seed=seed, verbose=False,
                            apply_natural_mortality=(run["mortality"] == "on"),
                            migration=(run["migration"] == "on"), currents=currents,
                            mass_balance=bool(run["mass_balance"]), tick_hours=tick_hours)
    policies, mean, var = inf.load_policies_and_stats(env, checkpoints, verbose=False)
    env.policies = dict(policies)
    env.obs_mean, env.obs_var = mean, var
    env.rebuild_batched_weights()

    # Impact fields: the manifest's inference maps, then the scenario's.
    impact_ids = active_impact_ids(scenario_manifest(variant))
    base_maps = {}
    for iid in impact_ids:
        field = env.grid.get_map(iid)
        if field is not None and np.any(field):
            base_maps[iid] = np.array(field, dtype=np.float32, copy=True)
    for name, cfg in (variant.spec.get("impacts") or {}).items():
        for iid in expand_impact_ids(name):
            if iid not in impact_ids:
                raise ValueError(f"impact '{iid}' is not in the world "
                                 f"(impact_variables, muted?)")
            path = variant.path((cfg or {}).get("map"))
            if path:
                field = load_impact_map_npz(path, iid, H, W, verbose=False)
                if field is None:
                    raise FileNotFoundError(f"no field for '{iid}' in {path}")
                base_maps[iid] = field
            if (cfg or {}).get("enabled") is False:
                base_maps.pop(iid, None)
    masks = []
    for m in variant.spec.get("masks") or []:
        if m.get("enabled") is False:
            continue
        names = set()
        for name in m.get("impacts", ["fishing"]):
            names.update(expand_impact_ids(name))
        masks.append((_load_mask(variant.path(m["map"]), m.get("key", "mask"), H, W), names))

    def set_fields(scale):
        fields = _scaled_fields(base_maps, masks, scale, impact_ids)
        for iid in impact_ids:
            env.grid.add_map(iid, fields.get(iid, np.zeros((H, W), np.float32)))
        return fields

    warming = apply_warming(env, variant.spec.get("warming_c"))

    # Warm-up (section 274) at the warm-up intensities.
    set_fields(variant.warmup_scale())
    warm_days = run["warmup_days"]
    if warm_days is None:
        warm_days = inf.run_warmup_days(checkpoints)
    warm = warmup_ticks(warm_days, tick_hours)
    engine = "gpu"  # the tensor engine, on ``device``
    t0 = time.time()
    settle(env, warm, seed=seed, policies=policies, obs=(mean, var), engine=engine,
           device=device)
    warmup_seconds = time.time() - t0

    schedule = variant.schedule()
    phase_ticks = [int(round(y * 365.0 * ticks_per_day)) for y, _ in schedule]
    set_fields(schedule[0][1])

    model, runner = build_runner(env, policies, mean, var, None, device=device,
                                 graph_ticks=record_ticks, diagnostics=True)
    G = model.G
    ids = list(model.ids)
    imp = [iid for iid in impact_ids if iid in base_maps]

    def impact_rates(scale):
        """[I, G, C] tensor: each impact's loss rate alone."""
        fields = _scaled_fields(base_maps, masks, scale, imp)
        rates = [model.impact_fields([{iid: fields[iid]}])[0][0, :G].double()
                 for iid in imp]
        return torch.stack(rates) if rates else None

    def upload(scale):
        set_fields(scale)
        if model.has_impacts:
            loss, cost = model.import_impacts([env])
            runner.impact_loss.copy_(loss)
            runner.impact_cost.copy_(cost)
        if model.n_obs_imp:
            runner.impact_obs.copy_(model.import_impact_observations([env]))

    n_rec = ticks // record_ticks
    t_days = np.zeros(n_rec + 1)
    biomass = np.zeros((n_rec + 1, G))
    reserve = np.zeros((n_rec + 1, G))
    juvenile = np.zeros((n_rec + 1, G))
    losses = np.zeros((n_rec + 1, len(runner.LOSS_ORDER), G))
    imports = np.zeros((n_rec + 1, G))
    catch = np.zeros((n_rec + 1, len(imp), G))
    scale_rec = np.zeros((n_rec + 1, len(imp)))
    spatial_t, spatial_b = [], []

    def read(row, tick, scale):
        b = runner.biomass[0].double()
        t_days[row] = tick / ticks_per_day
        biomass[row] = b.sum(-1).cpu().numpy()
        reserve[row] = runner.reserve[0].double().sum(-1).cpu().numpy()
        if model.stages_on:
            juvenile[row] = runner.juvenile[0].double().sum(-1).cpu().numpy()
        losses[row] = runner.loss_book[0].cpu().numpy()
        imports[row] = runner.import_book[0].cpu().numpy()
        scale_rec[row] = [float(scale.get(iid, 1.0)) for iid in imp]
        if spatial_ticks and tick % spatial_ticks == 0:
            spatial_t.append(tick / ticks_per_day)
            spatial_b.append(b.reshape(G, H, W).cpu().numpy().astype(np.float16))

    impact_row = list(runner.LOSS_ORDER).index("impact")
    phase = 0
    scale = schedule[0][1]
    rates = impact_rates(scale)
    read(0, 0, scale)
    tick = 0
    last_print = time.time()
    for row in range(1, n_rec + 1):
        # Expected loss shares at the start of the interval.
        shares = None
        if rates is not None:
            expected = (rates * runner.biomass[0].double()[None]).sum(-1)   # [I, G]
            total = expected.sum(0, keepdim=True)
            shares = torch.where(total > 0, expected / total.clamp_min(1e-300),
                                 torch.zeros_like(expected)).cpu().numpy()
        before = runner.loss_book[0, impact_row].cpu().numpy().copy()
        runner.run(record_ticks)
        tick += record_ticks
        read(row, tick, scale)
        if shares is not None:
            delta = losses[row, impact_row] - before
            catch[row] = catch[row - 1] + shares * delta[None]
        while phase + 1 < len(schedule) and tick >= phase_ticks[phase + 1]:
            phase += 1
            scale = schedule[phase][1]
            upload(scale)
            rates = impact_rates(scale)
        if verbose and (time.time() - last_print > 30 or row == n_rec):
            last_print = time.time()
            years_done = tick / ticks_per_day / 365.0
            print(f"    [{variant.scenario}/{variant.name} seed {seed}] "
                  f"year {years_done:.2f}/{ticks / ticks_per_day / 365.0:.2f} "
                  f"({time.time() - started:.0f} s)", flush=True)
    runner.close()

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "timeseries.npz"),
        t_days=t_days, biomass=biomass, reserve=reserve, juvenile=juvenile,
        losses=losses, imports=imports, catch=catch, scale=scale_rec,
        group_ids=np.array(ids), impact_ids=np.array(imp, dtype=str),
        loss_causes=np.array(runner.LOSS_ORDER),
        decision_makers=np.array([fid for fid in ids if env.fgs[fid].is_decision_maker]))
    if spatial_b:
        np.savez_compressed(os.path.join(out_dir, "spatial.npz"),
                            t_days=np.array(spatial_t), biomass=np.stack(spatial_b),
                            group_ids=np.array(ids))
    meta = variant.to_meta()
    meta.update({
        "seed": int(seed),
        "run_name": run_name,
        "grid": [H, W],
        "tick_hours": tick_hours,
        "ticks": int(tick),
        "years": tick / ticks_per_day / 365.0,
        "record_every_days": variant.record_every_days,
        "warmup_days": float(warm_days),
        "warming_factors": warming,
        "schedule": [{"from_year": y, "scale": s} for y, s in schedule],
        "warmup_scale": variant.warmup_scale(),
        "impact_means": {iid: float(np.mean(base_maps[iid])) for iid in imp},
        "device": str(device),
        "git": _git_revision(),
        "torch": torch.__version__,
        "warmup_seconds": warmup_seconds,
        "seconds": time.time() - started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta



def active_impact_ids(project):
    """The project's impacts that are not muted, in the loader's order."""
    return [iv["impact_id"] for iv in project.get("impact_variables") or []
            if isinstance(iv, dict) and iv.get("impact_id") and not iv.get("muted")]
