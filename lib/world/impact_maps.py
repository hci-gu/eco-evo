"""Impact fields for training worlds, inference worlds and baseline impacts.

An impact variable is either a scenario pressure (wind-farm noise, the
rotor) or a baseline pressure that belongs to the world as it is today
(fishing, ``baseline: true`` in the project; section 161). Training
samples a field for every impact the policy observes and for every
baseline impact; any other impact is an all-zero field during training
(per project decision, a pressure the policy cannot see is absent from
the training world). Inference worlds load the project's
``inference.impact_maps`` and fall back to a sampled field for a
baseline impact that has no map, so the baseline world is never silently
unfished.
"""
import zlib

import numpy as np

from lib.spawn import make_weights


def trained_impact_vars(impact_vars, observable_impact_vars=None,
                        baseline_impact_vars=None):
    """The impacts that get a sampled (not zero) field in a training world.

    ``observable_impact_vars`` None keeps the legacy behaviour (every
    impact sampled).
    """
    if observable_impact_vars is None:
        return list(impact_vars)
    keep = set(observable_impact_vars) | set(baseline_impact_vars or ())
    return [iv for iv in impact_vars if iv in keep]


def sample_impact_maps(impact_vars, impact_ranges, grid_size, seed=None,
                       impact_spawn_specs=None,
                       observable_impact_vars=None,
                       baseline_impact_vars=None):
    """Sample one impact field per active impact variable.

    Two code paths per impact:

    * Legacy (no ``spawn:`` block in the project): each cell is drawn
      i.i.d. uniformly from ``[value_min, value_max]``.
    * Spawn-strategy (when ``impact_spawn_specs[iv]`` is set): per-cell
      weights are produced by ``lib.spawn.make_weights`` (uniform / perlin /
      colony / env_driven, same set as for FG biomass spawn). The weights
      are normalised to ``[0, 1]`` (min->0, max->1) and then affinely
      mapped to ``[value_min, value_max]`` so the field still respects the
      configured value range. Constant weights collapse to ``value_min`` to
      preserve the legacy behaviour for ``vmax == vmin``.

    When ``observable_impact_vars`` is provided, any impact that is
    neither in that set nor in ``baseline_impact_vars`` is forced to an
    all-zero field (see the module docstring).

    Returns a dict ``{impact_id: np.ndarray}``.
    """
    H, W = grid_size
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    specs = impact_spawn_specs or {}
    sampled = set(trained_impact_vars(impact_vars, observable_impact_vars,
                                      baseline_impact_vars))
    maps = {}
    for iv in impact_vars:
        if iv not in sampled:
            # Absent from the training world: a zero field (ignored by the
            # policy network anyway; this also keeps any downstream env
            # logic that references the map from seeing stale values).
            maps[iv] = np.zeros((H, W), dtype=np.float32)
            continue
        vmin, vmax = impact_ranges.get(iv, (0.0, 0.0))
        spec = specs.get(iv)
        if spec is not None and vmax > vmin:
            # Per-impact sub-seed so two impacts with the same spec still
            # get different realisations within one snapshot.
            sub_seed = None
            if seed is not None:
                sub_seed = int(zlib.adler32(
                    f"{int(seed)}::impact::{iv}".encode("utf-8"))) & 0x7FFFFFFF
            try:
                w = make_weights(spec, (H, W), context=None,
                                 project_seed=sub_seed)
            except Exception:
                w = None
            if w is not None:
                w = np.asarray(w, dtype=np.float64)
                # Colony+jitter already returns the field on a [0, 1]
                # scale where the amplitudes encode relative source
                # strength (per-colony Uniform(amp_min, amp_max)). Bypass
                # the min-max stretch so heterogeneous source strengths
                # survive (otherwise the strongest cell would always be
                # pushed up to vmax).
                bypass_stretch = (
                    spec.mode == "colony"
                    and str(spec.params.get("amplitude_mode",
                                            "uniform")).lower() == "jitter"
                )
                if bypass_stretch:
                    w01 = np.clip(w, 0.0, 1.0)
                else:
                    w_min = float(w.min())
                    w_max = float(w.max())
                    if w_max > w_min:
                        w01 = (w - w_min) / (w_max - w_min)
                    else:
                        w01 = np.zeros_like(w)
                field = vmin + (vmax - vmin) * w01
                maps[iv] = field.astype(np.float32)
                continue
        if vmax > vmin:
            field = rng.uniform(vmin, vmax, size=(H, W))
        else:
            field = np.full((H, W), float(vmin))
        maps[iv] = field.astype(np.float32)
    return maps


def baseline_fallback_maps(project_path, missing, grid_size, seed=None):
    """Sampled fields for baseline impacts that have no inference map.

    ``missing`` lists the impact ids without a loaded field; only the
    baseline ones among them get a field. Returns ``{impact_id: field}``.
    """
    if not project_path or not missing:
        return {}
    from lib.config.config_loader import (
        load_config, load_impact_spawn_specs, project_baseline_impact_vars,
        project_impact_ranges)
    project = load_config(project_path)
    baseline = [iv for iv in project_baseline_impact_vars(project) if iv in missing]
    if not baseline:
        return {}
    return sample_impact_maps(
        baseline, project_impact_ranges(project), grid_size, seed=seed,
        impact_spawn_specs=load_impact_spawn_specs(project_path),
        observable_impact_vars=[], baseline_impact_vars=baseline)
