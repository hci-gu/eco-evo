"""Impact fields for training worlds and inference worlds (section 161).

A training world samples a field for every impact the policy observes,
so the groups can develop behaviour against it, and is otherwise without
impacts (per project decision, a pressure the policy cannot see is absent
from training) - except that an impact with a ``baseline_map`` (set in
the FG editor's Impact Editor, e.g. today's fishing effort per cell)
always gets that map. An inference world reads only the project's
``inference.impact_maps``; to keep today's fishing there, the same file
is added under the Inference tab. No map means no impact.
"""
import os
import zlib

import numpy as np

from lib.spawn import make_weights


def trained_impact_vars(impact_vars, observable_impact_vars=None):
    """The impacts that get a sampled (not zero) field in a training world.

    ``observable_impact_vars`` None keeps the legacy behaviour (every
    impact sampled). A baseline map overrides this (``sample_impact_maps``).
    """
    if observable_impact_vars is None:
        return list(impact_vars)
    keep = set(observable_impact_vars)
    return [iv for iv in impact_vars if iv in keep]


def load_impact_map_npz(path, impact_id, H, W, verbose=True):
    """Load a single (H, W) impact map from ``path[impact_id]`` with nearest-neighbor resampling.

    Returns a ``float32`` array of shape (H, W), or ``None`` on failure (caller
    should fall back to a zero field).
    """
    if not os.path.isfile(path):
        if verbose:
            print(f"  [warn] Impact map for '{impact_id}' not found at {path}; using zero field.")
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.files)
            arr = None
            # Priority 1: exact impact_id match (back-compat with archives
            # whose key happens to be named after the impact).
            if impact_id in keys:
                cand = np.asarray(data[impact_id])
                if cand.ndim == 2:
                    arr = cand
            # Priority 2: the first 2-D array in the archive. This makes
            # impact-map .npz files key-agnostic — any 2-D array works,
            # regardless of what it is called inside the archive. Mirrors
            # the loader used by fgconfig's Inference tab.
            if arr is None:
                for k in keys:
                    cand = np.asarray(data[k])
                    if cand.ndim == 2:
                        arr = cand
                        if verbose and k != impact_id:
                            print(f"  [info] Using array '{k}' from "
                                  f"{os.path.basename(path)} as impact map "
                                  f"for '{impact_id}' (key-agnostic load).")
                        break
            if arr is None:
                if verbose:
                    print(f"  [warn] No 2-D array found in "
                          f"{os.path.basename(path)} (keys: {keys}); "
                          f"using zero field for '{impact_id}'.")
                return None
    except Exception as e:
        if verbose:
            print(f"  [warn] Could not read '{impact_id}' from {path}: {e}; using zero field.")
        return None
    arr = arr.astype(np.float32, copy=False)
    Hs, Ws = arr.shape
    if (Hs, Ws) != (H, W):
        # Nearest-neighbor resample to the requested grid.
        if Hs == 0 or Ws == 0:
            return None
        row_idx = (np.arange(H) * Hs / H).astype(np.int64)
        col_idx = (np.arange(W) * Ws / W).astype(np.int64)
        arr = arr[row_idx[:, None], col_idx[None, :]]
        if verbose:
            print(f"  [info] Resampled '{impact_id}' from {(Hs, Ws)} to {(H, W)} (nearest).")
    return arr.astype(np.float32, copy=False)


def project_baseline_maps(project, project_path, grid_size, verbose=False):
    """Load the Impact Editor's baseline maps, ``{impact_id: (H, W) field}``.

    ``baseline_map`` on an active ``impact_variables`` entry is a .npz
    path, relative to the project file or absolute, read like an
    inference map (nearest-neighbour resampled to ``grid_size``). An
    impact whose file is missing or invalid has no baseline map.
    """
    base = os.path.dirname(os.path.abspath(project_path)) if project_path else os.getcwd()
    H, W = grid_size
    maps = {}
    for iv in (project or {}).get('impact_variables', []) or []:
        if not isinstance(iv, dict) or iv.get('muted') or not iv.get('impact_id'):
            continue
        stored = iv.get('baseline_map')
        if not isinstance(stored, str) or not stored:
            continue
        path = stored if os.path.isabs(stored) else os.path.normpath(os.path.join(base, stored))
        field = load_impact_map_npz(path, iv['impact_id'], H, W, verbose=verbose)
        if field is not None:
            maps[iv['impact_id']] = field
    return maps


def sample_impact_maps(impact_vars, impact_ranges, grid_size, seed=None,
                       impact_spawn_specs=None,
                       observable_impact_vars=None,
                       baseline_maps=None):
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

    When ``observable_impact_vars`` is provided, any impact not in that
    set is forced to an all-zero field. An impact in ``baseline_maps``
    gets its baseline map instead, observable or not (see the module
    docstring).

    Returns a dict ``{impact_id: np.ndarray}``.
    """
    H, W = grid_size
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    specs = impact_spawn_specs or {}
    sampled = set(trained_impact_vars(impact_vars, observable_impact_vars))
    baseline_maps = baseline_maps or {}
    maps = {}
    for iv in impact_vars:
        if iv in baseline_maps:
            maps[iv] = np.array(baseline_maps[iv], dtype=np.float32)
            continue
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
