"""Inference runner for trained Mareld policies.

Usage:
    python3 inference.py [flags]

This is a draft / scaffold meant to be extended. It implements the inference
pipeline sketched out previously and takes the known pitfalls into account:

  1. Loads policy checkpoints saved by ``train.py`` (`results/policy_<fg>.pth`),
     which carry both ``state_dict`` and (optionally) ``obs_stats`` per FG.
  2. Installs the frozen running mean/var on ``env.obs_mean`` / ``env.obs_var``
     so observations at inference time are normalised identically to training.
  3. Handles dm_ids ordering mismatch between checkpoint and current env via
     an explicit per-DM lookup (rather than positional indexing).
  4. Falls back to ``mean=0, var=1`` for DMs whose checkpoint has ``count==0``
     or is missing — and warns loudly so the user can tell something is off.
  5. Tolerates both the new checkpoint format ({'state_dict': ..., 'obs_stats': ...})
     and the legacy bare ``state_dict`` format.
  6. Does NOT update obs_stats during inference (stats are frozen).
  7. ``sigma_f`` reward normalisation is irrelevant at inference and is not
     touched — no ARS updates are performed.

Future flags can be added under the argparse block; this draft already
provides hooks for project file, grid size, number of ticks, checkpoint
directory, output path, and seed.
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

from lib.config.config_loader import (load_project_config, setup_full_mareld_mvp,
                                       compute_inference_b0_defaults,
                                       _build_spawn_spec,
                                       _spawn_biomass_distribution)
from lib.environments.ecosystem import EcosystemEnvironment
from lib.runners.policy import PolicyNetwork


class RandomPolicy:
    """Drop-in replacement for :class:`PolicyNetwork` that returns uniform
    pre-softmax logits (zeros) for every cell. After the env applies its
    action mask + softmax this yields a uniform distribution over the
    currently-legal actions — i.e. random actions sampled fresh every tick.

    Only the methods used by :meth:`EcosystemEnvironment.step` are
    implemented (no ``parameters()``, no ``state_dict()``); the batched
    forward path is intentionally bypassed by setting
    ``env._batched_ready = False`` on the rnd env, forcing the per-DM
    Python loop that calls ``get_action_logits_torch``.
    """

    def __init__(self, in_dim, out_dim):
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

    def get_action_logits_torch(self, flat_state):
        n = flat_state.shape[0]
        return torch.zeros((n, self.out_dim), dtype=torch.float32)

    def get_action_probs_torch(self, flat_state):
        n = flat_state.shape[0]
        return torch.full((n, self.out_dim), 1.0 / self.out_dim,
                          dtype=torch.float32)

    def eval(self):
        return self


def parse_grid_arg(value):
    m = re.fullmatch(r"\s*(\d+)\s*\*\s*(\d+)\s*", value)
    if not m:
        raise argparse.ArgumentTypeError(
            f"Invalid --grid format: '{value}'. Expected n*m (e.g. 60*60)."
        )
    n, k = int(m.group(1)), int(m.group(2))
    if n < 3 or k < 3:
        raise argparse.ArgumentTypeError(
            f"Invalid --grid '{value}': both dimensions must be >= 3."
        )
    return n, k


def _load_inference_map_paths(project_path):
    """Return ``{impact_id: absolute_path}`` from the project's ``inference.impact_maps``.

    Returns an empty dict when the project has no such section. Relative paths
    are resolved against the project YAML file's directory.
    """
    if not project_path:
        return {}
    try:
        import yaml as _yaml
        with open(project_path, 'r') as f:
            data = _yaml.safe_load(f) or {}
    except Exception:
        return {}
    section = (data.get('inference') or {}).get('impact_maps') or {}
    if not isinstance(section, dict):
        return {}
    base = os.path.dirname(os.path.abspath(project_path))
    resolved = {}
    for iid, p in section.items():
        if not isinstance(p, str) or not p:
            continue
        resolved[iid] = p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
    return resolved


def _load_impact_map_npz(path, impact_id, H, W, verbose=True):
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


def _load_spawn_templates(project_path):
    """Read the global ``spawn_templates`` block from the project YAML.

    Returns ``{<mode>: {<name>: {<params>}}}`` or ``{}`` when the project
    has no templates (or no project file at all). Used to populate the
    per-FG dropdowns under each heatmap in the live visualiser.
    """
    if not project_path:
        return {}
    try:
        import yaml as _yaml
        with open(project_path, 'r') as f:
            data = _yaml.safe_load(f) or {}
    except Exception:
        return {}
    tpls = data.get('spawn_templates')
    if not isinstance(tpls, dict):
        return {}
    out = {}
    for mode, by_name in tpls.items():
        if not isinstance(by_name, dict):
            continue
        cleaned = {}
        for name, params in by_name.items():
            if isinstance(params, dict):
                cleaned[str(name)] = dict(params)
        if cleaned:
            out[str(mode)] = cleaned
    return out


def _load_spawn_defaults(project_path):
    """Read each FG/impact's default spawn mode from the project YAML.

    Returns ``{fid: <mode_str>}`` mapping FG-id (decision_makers,
    non_decision_makers) and observable impact-id to the ``mode`` stored
    in their ``spawn:`` block in the project file. Used by the live
    visualiser to pre-fill the per-heatmap Mode dropdown so the user
    sees the strategy that is actually configured for each FG instead
    of a generic "(default)" placeholder.
    """
    if not project_path:
        return {}
    try:
        import yaml as _yaml
        with open(project_path, 'r') as f:
            data = _yaml.safe_load(f) or {}
    except Exception:
        return {}

    # Load the FG library (fg_library.yaml) for fallback. ``load_project_config``
    # mergar projektets per-FG ``spawn:``-override med library-specens
    # ``spawn:``-block; om projektfilen saknar block (vanligt) styrs spawn
    # alltså av library. Spegla samma resolution här så Mode-dropdownen
    # i visualiseraren visar den strategi som faktiskt körs.
    lib_specs = {}
    try:
        import os as _os
        candidates = []
        proj_dir = _os.path.dirname(_os.path.abspath(project_path))
        candidates.append(_os.path.join(proj_dir, 'fgconfig', 'fg_library.yaml'))
        candidates.append(_os.path.join(proj_dir, 'fg_library.yaml'))
        candidates.append('fgconfig/fg_library.yaml')
        for lib_path in candidates:
            if _os.path.isfile(lib_path):
                with open(lib_path, 'r') as f:
                    lib_data = _yaml.safe_load(f) or {}
                lib_specs = lib_data.get('species_definitions', {}) or {}
                break
    except Exception:
        lib_specs = {}

    out = {}
    def _extract(entries, id_key, use_library):
        if not isinstance(entries, list):
            return
        for e in entries:
            if not isinstance(e, dict):
                continue
            fid = e.get(id_key)
            if not fid:
                continue
            sp = e.get('spawn')
            mode = None
            if isinstance(sp, dict):
                mode = sp.get('mode')
            # Fallback 1: leta upp library-specens spawn-block (gäller bara
            # FGs — impacts har inget motsvarande library).
            if not mode and use_library:
                lib_sp = (lib_specs.get(str(fid), {}) or {}).get('spawn')
                if isinstance(lib_sp, dict):
                    mode = lib_sp.get('mode')
            # Fallback 2: ``_build_spawn_spec`` i config_loader använder
            # ``'uniform'`` när inget block finns alls. Spegla det.
            if not mode:
                mode = 'uniform'
            out[str(fid)] = str(mode)
    _extract(data.get('decision_makers'), 'group_id', True)
    _extract(data.get('non_decision_makers'), 'group_id', True)
    _extract(data.get('impact_variables'), 'impact_id', False)
    return out


def apply_spawn_overrides(env, spawn_overrides, spawn_templates, seed=None):
    """Respawn each FG's biomass field according to user-chosen templates.

    ``spawn_overrides`` is ``{fid: {"mode": <m>, "template": <name>}}``
    from the visualiser. For every FG in this dict we rebuild a
    :class:`StrategySpec` from the corresponding template parameters,
    recompute a weight field via the same code path used by
    :func:`load_project_config`, and overwrite ``fg.biomass`` while
    preserving the FG's current total biomass (so per-FG b0 sliders and
    project defaults still govern the magnitude).

    Called after :func:`build_env` (and any :func:`apply_b0_overrides`)
    so the spatial layout is the only thing that changes. FGs without
    an override entry — or with ``template == None`` — keep the spawn
    configuration loaded from the project file.
    """
    if not spawn_overrides:
        return
    H, W = None, None
    for fid, ov in spawn_overrides.items():
        if not isinstance(ov, dict):
            continue
        name = ov.get("template")
        mode = ov.get("mode")
        if not name or not mode:
            continue  # (default) → leave projektfilens spawn untouched
        params = (spawn_templates.get(mode, {}) or {}).get(name)
        if not isinstance(params, dict):
            continue
        fg = env.fgs.get(fid)
        if fg is None:
            continue
        arr = np.asarray(fg.biomass, dtype=np.float64)
        if H is None:
            H, W = arr.shape
        total_b = float(arr.sum())
        if total_b <= 0.0:
            continue
        spawn_cfg = {"mode": mode}
        spawn_cfg.update(params)
        spec = _build_spawn_spec(spawn_cfg, default_seed=seed)
        if spec is None:
            continue
        # Reuse the existing env_fields of all currently-spawned FGs as
        # the env_driven context, so refs (if any) can resolve.
        env_fields = {sid: np.asarray(f.biomass, dtype=np.float64)
                      for sid, f in env.fgs.items()}
        rng = np.random.default_rng(seed) if seed is not None else None
        min_per_cell = 5.0 * float(getattr(fg, 'min_split_biomass', 0.0))
        try:
            new_arr = _spawn_biomass_distribution(
                arr.shape, total_b, min_per_cell,
                allowed_mask=None, rng=rng,
                spawn_spec=spec, project_seed=seed,
                env_context={'env_fields': env_fields})
        except Exception as e:
            print(f"  [spawn override] {fid}: failed to apply "
                  f"template '{name}' ({mode}): {e!r}")
            continue
        fg.biomass[...] = np.asarray(new_arr, dtype=np.float32)


def apply_b0_overrides(env, b0_overrides):
    """Skala om varje FG:s spawnade biomass-fält så att totalsumman
    matchar det användardefinierade ``b0_overrides[fg_id]`` (ton).

    Bevarar den spatiala fördelningen (formen). FGs som saknas i
    ``b0_overrides`` eller har None lämnas orörda. Om en FG:s nuvarande
    totalsumma är 0 (tom karta) sprids målvärdet jämnt över alla celler.
    Anropas direkt efter att env är byggd och innan första
    ``env.step()`` så ``b0``-baseline blir det nya värdet.
    """
    if not b0_overrides:
        return
    for fid, target in b0_overrides.items():
        if target is None:
            continue
        fg = env.fgs.get(fid)
        if fg is None:
            continue
        try:
            target_t = float(target)
        except (TypeError, ValueError):
            continue
        if target_t < 0.0:
            target_t = 0.0
        arr = np.asarray(fg.biomass, dtype=np.float64)
        cur = float(arr.sum())
        if target_t == 0.0:
            new_arr = np.zeros_like(arr, dtype=np.float32)
        elif cur > 0.0:
            new_arr = (arr * (target_t / cur)).astype(np.float32, copy=False)
        else:
            # Tom karta: sprid jämnt över hela griden.
            H, W = arr.shape
            new_arr = np.full((H, W), target_t / float(H * W), dtype=np.float32)
        # Skriv tillbaka till FG. ``biomass`` är en ndarray; vi
        # ersätter innehållet in-place så övriga referenser i env
        # (caches osv) följer med.
        fg.biomass[...] = new_arr


def build_env(project_path, grid_size, seed=None, verbose=True,
              apply_natural_mortality=False):
    """Construct a fresh EcosystemEnvironment for inference."""
    H, W = grid_size
    impact_ranges = {}
    if project_path:
        fgs, impact_vars, impact_ranges, observable_impact_vars = load_project_config(
            project_path, grid_size=grid_size, seed=seed, mode='inference')
    else:
        fgs = setup_full_mareld_mvp(grid_size=grid_size, seed=seed)
        impact_vars = ['windfarm_noise']
        observable_impact_vars = ['windfarm_noise']

    grid_config = {
        'width': W,
        'height': H,
        'cell_size': 1000.0,
        'tick_duration': 6.0,
    }
    env = EcosystemEnvironment(grid_config, fgs, {},
                               observable_impact_vars=observable_impact_vars,
                               apply_natural_mortality=apply_natural_mortality)
    # Impact maps are read from .npz files configured in the project's
    # ``inference.impact_maps`` section (set via the fgconfig Inference tab).
    # Missing entries — or files that fail validation — are treated as
    # all-zero fields. Biomass / energy maps are still randomly spawned
    # from the configured initial biomass values.
    map_paths = _load_inference_map_paths(project_path)
    for iv in impact_vars:
        field = None
        if iv in map_paths:
            field = _load_impact_map_npz(map_paths[iv], iv, H, W, verbose=verbose)
        if field is None:
            field = np.zeros((H, W), dtype=np.float32)
        env.grid.add_map(iv, field)
    return env


class PolicyCheckpointMismatchError(Exception):
    """Raised when the checkpoints on disk do not match the project's DM set
    or have an incompatible architecture / input dimension."""


def _infer_arch_from_state_dict(sd):
    """Inspect a PolicyNetwork state_dict and return (in_dim, out_dim,
    hidden_dim, hidden_layers). The PolicyNetwork stores its layers inside
    an ``nn.Sequential`` named ``net`` with alternating Linear+Sigmoid and
    a final Linear; weight keys look like ``net.0.weight``, ``net.2.weight``,
    ..., ``net.<2k>.weight``. Returns None on unrecognised layouts.
    """
    try:
        linear_keys = sorted(
            (int(k.split('.')[1]), k) for k in sd.keys()
            if k.startswith('net.') and k.endswith('.weight')
        )
        if not linear_keys:
            return None
        shapes = [tuple(sd[k].shape) for _, k in linear_keys]
        in_dim = shapes[0][1]
        out_dim = shapes[-1][0]
        hidden_dim = shapes[0][0] if len(shapes) > 1 else None
        hidden_layers = len(shapes) - 1
        return in_dim, out_dim, hidden_dim, hidden_layers
    except Exception:
        return None


def load_policies_and_stats(env, checkpoint_dir, verbose=True):
    """Load all DM policy checkpoints from ``checkpoint_dir``.

    Returns (policies_dict, obs_mean, obs_var) where obs_mean/obs_var are
    aligned with env.dm_ids ordering. Falls back to (mean=0, var=1) for
    DMs without usable stats.

    Raises :class:`PolicyCheckpointMismatchError` if the set of
    ``policy_<fg>.pth`` files in ``checkpoint_dir`` does not match the
    project's decision makers, or if any checkpoint has an incompatible
    architecture (e.g. trained on a different grid / FG set / hidden size).
    """
    env._build_static_caches()
    dm_ids = list(env.dm_ids)
    N_all = env.N_all
    N_dm = env.N_dm
    # With the Observability matrix each DM has its own compact ``in_dim``
    # (see Section 4 / Section 41 in mareld_resume.txt). ``env.max_in_dim``
    # is the padded width used for the batched bmm tensor and is what
    # ``obs_stats`` arrays must match. Per-DM checkpoints are saved with
    # their own compact ``per_dm_in_dim[i]``, which we cross-check below.
    D = int(env.max_in_dim)
    per_dm_in_dim = list(env.per_dm_in_dim)
    out_dim = 5 + N_all

    # ---- Cross-check checkpoint files against project's DM set ----------
    # Find every ``policy_<id>.pth`` file currently in the checkpoint dir.
    found_ids = set()
    try:
        for name in os.listdir(checkpoint_dir):
            m = re.fullmatch(r"policy_(.+)\.pth", name)
            if m:
                found_ids.add(m.group(1))
    except OSError as e:
        raise PolicyCheckpointMismatchError(
            f"Could not list checkpoint directory '{checkpoint_dir}': {e}"
        )
    expected_ids = set(dm_ids)
    missing = sorted(expected_ids - found_ids)
    extra = sorted(found_ids - expected_ids)
    # ``extra`` = checkpoint files for FGs that are NOT active DMs in the
    # current project (either removed entirely, marked ``muted: true`` in
    # the YAML, or demoted from DM to NDM). These are silently ignored
    # rather than treated as an error so that a single run-name directory
    # can be reused across project variants (e.g. mareld2.yaml muting
    # most FGs while a sibling project trains them all). Only ``missing``
    # — an active DM without a checkpoint — is still a hard failure.
    if extra and verbose:
        print(f"  [inference] Ignoring {len(extra)} checkpoint(s) for "
              f"FGs not active in this project: {extra}")
    if missing:
        lines = [
            f"Policy checkpoints in '{checkpoint_dir}' do not match the "
            f"project's decision makers."
        ]
        lines.append(f"  Expected ({len(expected_ids)}): "
                     f"{sorted(expected_ids)}")
        lines.append(f"  Found    ({len(found_ids)}): "
                     f"{sorted(found_ids)}")
        lines.append(f"  Missing checkpoint(s): {missing}")
        if extra:
            lines.append(f"  Ignored (not active DMs): {extra}")
        lines.append("  Hint: make sure --run-name / --checkpoints points at "
                     "a run trained with the same project file, or retrain.")
        raise PolicyCheckpointMismatchError("\n".join(lines))

    mean = np.zeros((N_dm, D), dtype=np.float32)
    var = np.ones((N_dm, D), dtype=np.float32)

    policies = {}
    for i, fid in enumerate(dm_ids):
        ckpt_path = os.path.join(checkpoint_dir, f"policy_{fid}.pth")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # Support both formats: bare state_dict (legacy) and new dict payload.
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            sd = ckpt['state_dict']
            os_stats = ckpt.get('obs_stats')
        else:
            sd = ckpt
            os_stats = None

        # Cross-check architecture against the env that we're about to run
        # inference on. Mismatch usually means the checkpoint was trained
        # with a different project (different FG count -> different in_dim
        # / out_dim) or with a different --policynetwork.
        arch = _infer_arch_from_state_dict(sd)
        if arch is None:
            raise PolicyCheckpointMismatchError(
                f"Checkpoint '{ckpt_path}' has an unrecognised layout "
                f"(state_dict keys: {list(sd.keys())[:6]}...)."
            )
        ck_in, ck_out, ck_hidden, ck_layers = arch
        # Per-DM expected in_dim: the Observability matrix gives each DM
        # its own compact input layout (Section 4 / Section 41). The
        # checkpoint was saved with that compact dim, not with the
        # padded ``max_in_dim``.
        expected_in = int(per_dm_in_dim[i])
        if ck_in != expected_in or ck_out != out_dim:
            obs_list = env.fgs[fid].params.get('observes')
            if obs_list is None:
                obs_desc = ("legacy 'see all other FGs' (no _observes_ "
                            "entries for this DM)")
            else:
                others = [o for o in obs_list if o != fid and o in env.fgs]
                obs_desc = (f"{len(others)} other FG(s): {sorted(others)}")
            raise PolicyCheckpointMismatchError(
                f"Checkpoint '{ckpt_path}' has incompatible input/output "
                f"dimensions: file has (in={ck_in}, out={ck_out}) but the "
                f"current project expects (in={expected_in}, out={out_dim}).\n"
                f"  Current Observability config for '{fid}': observes "
                f"{obs_desc}.\n"
                f"  Hint: this checkpoint was trained against a different "
                f"Observability matrix, FG set, or observable impacts. "
                f"Adjust the Observability matrix in fgconfig so '{fid}' "
                f"yields in_dim={ck_in}, or retrain."
            )

        net = PolicyNetwork(expected_in, out_dim,
                            hidden_dim=ck_hidden if ck_hidden else 30,
                            hidden_layers=max(1, ck_layers))
        try:
            net.load_state_dict(sd)
        except Exception as e:
            raise PolicyCheckpointMismatchError(
                f"Could not load weights from '{ckpt_path}' into a "
                f"PolicyNetwork(in={expected_in}, out={out_dim}, "
                f"hidden_dim={ck_hidden}, hidden_layers={ck_layers}): {e}"
            )
        net.eval()
        policies[fid] = net

        if os_stats is not None and int(os_stats.get('count', 0)) > 0:
            m = np.asarray(os_stats['mean'], dtype=np.float32)
            v = np.asarray(os_stats['var'], dtype=np.float32)
            if m.shape == (D,) and v.shape == (D,):
                mean[i] = m
                var[i] = v
                if verbose:
                    print(f"  [ok]   Loaded '{fid}' (obs_stats count={int(os_stats['count'])}).")
            else:
                if verbose:
                    print(f"  [warn] '{fid}' obs_stats shape {m.shape} != ({D},); "
                          f"using mean=0/var=1.")
        else:
            if verbose:
                print(f"  [warn] '{fid}' has no usable obs_stats; using mean=0/var=1. "
                      f"Observations will be clipped to [-10, 10] without scaling.")

    return policies, mean, var


def _push_action_fracs(viz, env, step, suffix=""):
    """Push per-DM mean (over rollout-so-far) action fractions into the
    visualiser's move/rest/eat tabs.

    ``env._action_move_frac`` / ``_rest_frac`` / ``_eat_frac`` are running
    sums per DM (indexed by ``env.dm_ids``) over ticks where the DM has
    any biomass; ``env._action_active_ticks[i]`` is that DM's active-tick
    counter. The plotted value is the running mean = sum / max(1, count).
    """
    if viz is None:
        return
    mv = getattr(env, '_action_move_frac', None)
    rs = getattr(env, '_action_rest_frac', None)
    et = getattr(env, '_action_eat_frac', None)
    cnt = getattr(env, '_action_active_ticks', None)
    if mv is None or rs is None or et is None or cnt is None:
        return
    try:
        for i, fid in enumerate(env.dm_ids):
            c = float(cnt[i]) if cnt[i] > 0 else 0.0
            if c <= 0.0:
                continue
            key = fid + suffix
            _mv_pct = 100.0 * float(mv[i]) / c
            _rs_pct = 100.0 * float(rs[i]) / c
            _et_pct = 100.0 * float(et[i]) / c
            viz.update_series("move", key, _mv_pct, step=step)
            viz.update_series("rest", key, _rs_pct, step=step)
            viz.update_series("eat",  key, _et_pct, step=step)
            # Pusha även till heatmap-headerns dedikerade state så raden
            # ``mv/rs/et = …`` läser från ``_action_fracs`` (frikopplat
            # från plot-serierna, vilket är nödvändigt för att headern och
            # plot-flikarna ska kunna ha olika x-skalor under träning).
            try:
                viz.update_action_fracs(key, _mv_pct, _rs_pct, _et_pct)
            except Exception:
                pass
    except Exception:
        pass


def run_inference(env, policies, obs_mean, obs_var, n_ticks, verbose=True,
                  viz=None, rnd_env=None):
    """Install policies + frozen stats and step the environment ``n_ticks`` times.

    If ``viz`` is a :class:`lib.viz.LiveVisualizer`, biomass heatmaps and a
    rolling per-FG total-biomass plot are updated every tick. The visualiser
    is allowed to abort the run early by returning False from ``pump_events``;
    in that case the partial history collected so far is returned.
    """
    env.policies = dict(policies)
    env.obs_mean = obs_mean
    env.obs_var = obs_var
    # Rebuild batched weight tensors used by the fast inference path.
    env._rebuild_batched_weights()

    # Install uniform-random policies on the parallel baseline env, if any.
    if rnd_env is not None:
        rnd_env._build_static_caches()
        out_dim_rnd = 5 + rnd_env.N_all
        in_dim_rnd = env.obs_mean.shape[1] if env.obs_mean.ndim == 2 else 0
        rnd_env.policies = {
            fid: RandomPolicy(in_dim_rnd, out_dim_rnd) for fid in rnd_env.dm_ids
        }
        # Frozen mean=0 / var=1 (we want true uniform actions, no obs
        # normalisation pulling the masked logits anywhere). Logits are
        # constant zero anyway so this is pure bookkeeping.
        rnd_env.obs_mean = np.zeros_like(env.obs_mean)
        rnd_env.obs_var = np.ones_like(env.obs_var)
        # Force the per-DM Python path so RandomPolicy.get_action_logits_torch
        # is actually called (the batched path needs stacked Linear weights).
        rnd_env._batched_ready = False

    history = {fid: [] for fid in env.fgs}
    energy_history = {fid: [] for fid in env.fgs}
    rnd_history = {fid: [] for fid in (rnd_env.fgs if rnd_env is not None else {})}
    rnd_energy_history = {fid: [] for fid in (rnd_env.fgs if rnd_env is not None else {})}
    # Spela in hela inference-rollouten som en uppspelningsbar "film" i
    # viz. Frames capturas vid varje ``update_biomass``; vid loop-slut
    # kallas ``end_rollout_recording`` så användaren kan spela upp/stega.
    if viz is not None:
        try:
            viz.begin_rollout_recording()
        except Exception:
            pass
    for t in range(n_ticks):
        try:
            env.step()
            if rnd_env is not None:
                rnd_env.step()
        except KeyboardInterrupt:
            if verbose:
                print(f"\n    interrupted at tick {t+1}/{n_ticks}; "
                      f"returning partial history.")
            break
        for fid, fg in env.fgs.items():
            history[fid].append(float(fg.biomass.sum()))
            er = getattr(fg, 'energy_reserve', None)
            energy_history[fid].append(float(er.sum()) if er is not None else 0.0)
        if rnd_env is not None:
            for fid, fg in rnd_env.fgs.items():
                rnd_history[fid].append(float(fg.biomass.sum()))
                er = getattr(fg, 'energy_reserve', None)
                rnd_energy_history[fid].append(
                    float(er.sum()) if er is not None else 0.0)
        # Avbryt loopen tidigt om all biomassa har kollapsat till 0 i
        # samtliga FG (och, om rnd_env används, även där). Annars fortsätter
        # visualiseringen att uppdateras med b = 0.00 tick efter tick utan
        # att något kan hända i ekosystemet. Den sista frame:n som ritats
        # speglar redan kollapsen; vi gör en sista paint nedan och stannar.
        # Använd en absolut epsilon-tröskel i stället för == 0. När all
        # biomassa kollapsar lämnar upprepad multiplikation med decay-
        # faktorer kvar float32-subnormaler (ned mot ~1.4e-45) som aldrig
        # når exakt 0. Sådana värden saknar biologisk innebörd men gör att
        # heatmapens normalisering (max-värde per frame) fortsätter
        # fluktuera vilt mellan subnormaler.
        #
        # ``DEAD_EPS`` sätts till 50 % av minsta positiva ``min_split_biomass``
        # över alla FG i env (lagras i ton i ``FunctionalGroup``, dvs kg/1000).
        # Det är den minsta odelbara enheten i ekosystemet: när total biomass
        # underskrider halva den nivån finns inte ens en halv odelbar individ
        # kvar någonstans, och rollouten kan tryggt avbrytas. Om ingen FG har
        # ``min_split_biomass > 0`` (helt kontinuerligt läge) faller vi tillbaka
        # på en liten numerisk tröskel som bara fångar float32-subnormaler.
        _msb_values = [
            float(getattr(fg, 'min_split_biomass', 0.0))
            for fg in env.fgs.values()
        ]
        if rnd_env is not None:
            _msb_values.extend(
                float(getattr(fg, 'min_split_biomass', 0.0))
                for fg in rnd_env.fgs.values()
            )
        _msb_pos = [v for v in _msb_values if v > 0.0]
        DEAD_EPS = 0.5 * min(_msb_pos) if _msb_pos else 1e-9
        total_b = sum(float(fg.biomass.sum()) for fg in env.fgs.values())
        total_b_rnd = (
            sum(float(fg.biomass.sum()) for fg in rnd_env.fgs.values())
            if rnd_env is not None else 0.0
        )
        ecosystem_dead = (total_b <= DEAD_EPS) and (
            rnd_env is None or total_b_rnd <= DEAD_EPS
        )
        # Per-FG average-biomass / average-energy ratio over the rollout
        # so far, expressed as a percentage of the initial value. ``b0``
        # is the post-tick-0 baseline (history[fid][0]); the value plotted
        # is ``100 * mean(history[fid]) / b0`` (and analogously for energy).
        pct_bio = {}
        pct_eng = {}
        for fid in history.keys():
            b0 = history[fid][0] if history[fid] else 0.0
            if b0 > 0.0 and history[fid]:
                avg_b = sum(history[fid]) / len(history[fid])
                pct_bio[fid] = 100.0 * avg_b / b0
            else:
                pct_bio[fid] = 0.0
            e0 = energy_history[fid][0] if energy_history[fid] else 0.0
            if e0 > 0.0 and energy_history[fid]:
                avg_e = sum(energy_history[fid]) / len(energy_history[fid])
                pct_eng[fid] = 100.0 * avg_e / e0
            else:
                pct_eng[fid] = 0.0
        if verbose and (t % max(1, n_ticks // 10) == 0):
            print(f"    tick {t+1}/{n_ticks}")
        if viz is not None:
            try:
                viz.update_biomass(env.fgs, tick=t)
                for fid in history.keys():
                    viz.update_series("biomass", fid, pct_bio[fid], step=t)
                    viz.update_series("energy", fid, pct_eng[fid], step=t)
                _push_action_fracs(viz, env, step=t)
                # Per-FG biomass-loss breakdown (pr/st/im) so far,
                # computed from the env's running accumulators. Each
                # value is a fraction in [0, 1] summing to 1.0 when
                # there has been any loss for that FG.
                _lb = {}
                for fid in env.fgs:
                    ls = float(getattr(env, 'loss_starvation', {}).get(fid, 0.0))
                    lp = float(getattr(env, 'loss_predation', {}).get(fid, 0.0))
                    li = float(getattr(env, 'loss_impact', {}).get(fid, 0.0))
                    _tot = ls + lp + li
                    if _tot > 0.0:
                        _lb[fid] = {
                            'predation':  lp / _tot,
                            'starvation': ls / _tot,
                            'impact':     li / _tot,
                        }
                    else:
                        _lb[fid] = {'predation': 0.0,
                                    'starvation': 0.0,
                                    'impact': 0.0}
                viz.update_loss_breakdown(_lb)
                # Push the same fractions (×100%) to the dedicated plot
                # tabs ``predation``/``starvation``/``impacts`` per tick,
                # mirroring how biomass/energy is fed in inference. Detta
                # gör att kurvorna visar hur andelarna utvecklas över
                # rollouten — speglar headern ``pr/st/im=…`` över tid.
                for fid, _br in _lb.items():
                    viz.update_series("predation", fid,
                                      100.0 * float(_br.get('predation', 0.0)),
                                      step=t)
                    viz.update_series("starvation", fid,
                                      100.0 * float(_br.get('starvation', 0.0)),
                                      step=t)
                    viz.update_series("impacts", fid,
                                      100.0 * float(_br.get('impact', 0.0)),
                                      step=t)
                # Per-DM diet breakdown: läs env-ackumulatorn
                # ``intake_by_pred_prey`` (ton intagen prey-biomassa över
                # rollouten) och normalisera per predator. Heatmap-headern
                # ritar då en rad '<abbr>/… = X/…%' ovanför heatmapen.
                _diet = {}
                _ipp = getattr(env, 'intake_by_pred_prey', None) or {}
                for pred_id, prey_map in _ipp.items():
                    _tot_d = float(sum(prey_map.values()))
                    if _tot_d <= 0.0:
                        continue
                    _diet[pred_id] = {
                        pid: float(v) / _tot_d for pid, v in prey_map.items()
                        if float(v) > 0.0
                    }
                if _diet:
                    viz.update_diet_breakdown(_diet)
                if rnd_env is not None:
                    _push_action_fracs(viz, rnd_env, step=t, suffix="_rnd")
                if rnd_env is not None:
                    for fid in rnd_history.keys():
                        b0 = rnd_history[fid][0] if rnd_history[fid] else 0.0
                        e0 = rnd_energy_history[fid][0] if rnd_energy_history[fid] else 0.0
                        if b0 > 0.0 and rnd_history[fid]:
                            avg_b = sum(rnd_history[fid]) / len(rnd_history[fid])
                            pb = 100.0 * avg_b / b0
                        else:
                            pb = 0.0
                        if e0 > 0.0 and rnd_energy_history[fid]:
                            avg_e = sum(rnd_energy_history[fid]) / len(rnd_energy_history[fid])
                            pe = 100.0 * avg_e / e0
                        else:
                            pe = 0.0
                        viz.update_series("biomass", fid + "_rnd", pb, step=t)
                        viz.update_series("energy", fid + "_rnd", pe, step=t)
                if not viz.pump_events():
                    if verbose:
                        print("    [viz] window closed; stopping early.")
                    break
                if ecosystem_dead:
                    if verbose:
                        print(f"    [inference] all biomass has collapsed to 0 "
                              f"at tick {t+1}/{n_ticks}; stopping early.")
                    break
            except KeyboardInterrupt:
                if verbose:
                    print(f"\n    interrupted at tick {t+1}/{n_ticks} "
                          f"(during viz update); returning partial history.")
                break
    # Avsluta rollout-inspelningen så uppspelningsknapparna blir aktiva
    # i ``wait_for_close``-loopen. Säker att kalla även vid early-break
    # (KI eller ecosystem_dead) — vi får då en partiell film.
    if viz is not None:
        try:
            viz.end_rollout_recording()
        except Exception:
            pass
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Mareld Ecosystem Simulator - Inference Runner (draft)."
    )
    parser.add_argument("--project", type=str, default=None,
                        help="Path to project YAML file. If omitted, uses setup_full_mareld_mvp.")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Directory containing policy_<fg>.pth checkpoints. "
                             "If omitted, results/<run-name>/ is used (see --run-name).")
    parser.add_argument("--run-name", "--run_name", dest="run_name", type=str, default="default",
                        help="Name of the run whose checkpoints to load (results/<run-name>/). "
                             "Default: 'default'. Ignored if --checkpoints is given explicitly.")
    parser.add_argument("--grid", type=parse_grid_arg, default=(60, 60),
                        help="Grid dimensions as n*m (default: 60*60).")
    parser.add_argument("--ticks", type=int, default=100,
                        help="Number of simulation ticks to run (default: 100).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for deterministic initial biomass distribution.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save per-FG biomass history as .npz.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output.")
    parser.add_argument("--mortality", choices=["on", "off"], default="off",
                        help="Toggle the artificial (density-independent) natural "
                             "mortality term applied to decision-maker FGs each tick. "
                             "Default: off.")
    parser.add_argument("--rnd-baseline", "--rnd_baseline", dest="rnd_baseline",
                        action="store_true",
                        help="Run a parallel rollout where each DM acts uniformly "
                             "at random (mask-respecting) and overlay its biomass%% "
                             "/ energy%% in the live plot as a baseline. Legend "
                             "entries are suffixed with '_rnd'. No heatmaps for "
                             "the random agents.")
    parser.add_argument("--visual", action="store_true",
                        help="Open a live pygame window with per-FG biomass heatmaps "
                             "and a rolling total-biomass plot. Requires pygame; if "
                             "unavailable the flag is silently ignored.")

    args = parser.parse_args()
    verbose = not args.quiet
    if args.checkpoints is None:
        args.checkpoints = os.path.join("results", args.run_name)

    if verbose:
        print("==========================================")
        print("      MARELD INFERENCE SESSION            ")
        print("==========================================")
        print(f"Project:      {args.project or '(built-in MVP)'}")
        print(f"Run name:     {args.run_name}")
        print(f"Checkpoints:  {args.checkpoints}")
        print(f"Grid:         {args.grid[0]}x{args.grid[1]}")
        print(f"Ticks:        {args.ticks}")
        print(f"Seed:         {args.seed}")
        print(f"Mortality:    {args.mortality}")
        print("------------------------------------------")

    if not os.path.isdir(args.checkpoints):
        print(f"Error: checkpoint directory does not exist: {args.checkpoints}", file=sys.stderr)
        return 1

    env = build_env(args.project, args.grid, seed=args.seed, verbose=verbose,
                    apply_natural_mortality=(args.mortality == "on"))
    if verbose:
        print(f"Loading policies for DMs: {[fid for fid in env.fgs if env.fgs[fid].is_decision_maker]}")
    try:
        policies, mean, var = load_policies_and_stats(env, args.checkpoints, verbose=verbose)
    except PolicyCheckpointMismatchError as e:
        print("Error: policy checkpoint mismatch.", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 1

    viz = None
    if args.visual:
        try:
            from lib.viz import LiveVisualizer
            extra = ([fid + "_rnd" for fid in env.fgs.keys()]
                     if args.rnd_baseline else None)
            ndm_ids = [fid for fid, fg in env.fgs.items()
                       if not getattr(fg, 'is_decision_maker', False)]
            viz = LiveVisualizer(fg_ids=list(env.fgs.keys()),
                                 grid_shape=args.grid,
                                 mode="inference",
                                 extra_plot_ids=extra,
                                 ndm_ids=ndm_ids or None)
            # b0-slider defaults: gridskaleberäknat inference_initial_biomass
            # per FG, läst direkt från projekt-YAML. Slider-rangen blir
            # ``[0, 4 * default]`` per FG, mittposition = default.
            try:
                b0_defaults = compute_inference_b0_defaults(
                    args.project, args.grid)
                if b0_defaults:
                    viz.set_b0_defaults(b0_defaults)
            except Exception as _e:
                print(f"[viz] could not compute b0 defaults: {_e!r}",
                      file=sys.stderr)
            # Rollout-längd-slider: registrera CLI-värdet som default.
            try:
                viz.set_ticks_default(int(args.ticks))
            except Exception:
                pass
            # Spawn-strategi-templates till per-heatmap-dropdownen.
            try:
                viz.set_spawn_templates(_load_spawn_templates(args.project))
            except Exception as _e:
                print(f"[viz] could not load spawn templates: {_e!r}",
                      file=sys.stderr)
            # Per-FG default-spawn-mode från projektfilen — Mode-dropdownen
            # förinställs till den strategi som är sparad för respektive FG
            # istället för det generiska "(default)".
            try:
                viz.set_spawn_defaults(_load_spawn_defaults(args.project))
            except Exception as _e:
                print(f"[viz] could not load spawn defaults: {_e!r}",
                      file=sys.stderr)
        except Exception as e:
            print(f"[viz] failed to start visualiser: {e!r}", file=sys.stderr)
            viz = None

    if verbose:
        print(f"Running {args.ticks} ticks...")
    interrupted = False
    history = None
    # Rerun-loop: så länge användaren drar i en b0-slider efter rollouten
    # spelas en ny inspelning in med det nya värdet. Första iterationen
    # använder ``env``/``rnd_env`` som redan byggts ovan (med eventuella
    # initiala overrides från en sparad slider-state, som dock i praktiken
    # är tomma i runda 1). Efterföljande iterationer bygger om från grunden
    # så biomass/impact-fält återställs.
    first_iteration = True
    try:
        while True:
            if not first_iteration:
                env = build_env(args.project, args.grid, seed=args.seed,
                                verbose=False,
                                apply_natural_mortality=(args.mortality == "on"))
                # Återbygg de statiska caches som ``load_policies_and_stats``
                # satte upp i runda 1 (N_dm, N_all, dm_ids, per_dm_in_dim,
                # max_in_dim m.fl.). Utan dessa kraschar
                # ``_rebuild_batched_weights`` med ``AttributeError: N_dm``.
                env._build_static_caches()
            # b0-overrides från slidrarna (om viz finns) appliceras på env
            # innan första env.step() — total biomass per FG skalas så
            # totalsumman matchar slider-värdet. Bevarar spatial form.
            b0_overrides = (viz.get_b0_overrides() if viz is not None else {})
            if b0_overrides:
                apply_b0_overrides(env, b0_overrides)
                if verbose:
                    print(f"    [b0 override] Applied: "
                          + ", ".join(f"{k}={v:.1f}" for k, v in
                                       b0_overrides.items()))
            # Spawn-strategi-overrides från drop-downsen under varje
            # heatmap. Applicera EFTER b0-skalningen så totalvärdet är
            # det användaren förväntar sig och bara den spatiala
            # fördelningen byts ut.
            spawn_overrides = (viz.get_spawn_overrides()
                               if viz is not None else {})
            spawn_tpls_now = (_load_spawn_templates(args.project)
                              if spawn_overrides else {})
            if spawn_overrides:
                apply_spawn_overrides(env, spawn_overrides, spawn_tpls_now,
                                      seed=args.seed)
                if verbose:
                    print(f"    [spawn override] Applied: "
                          + ", ".join(
                              f"{k}={v.get('template')}({v.get('mode')})"
                              for k, v in spawn_overrides.items()))

            rnd_env = None
            if args.rnd_baseline:
                rnd_env = build_env(args.project, args.grid, seed=args.seed,
                                    verbose=False,
                                    apply_natural_mortality=(args.mortality == "on"))
                if b0_overrides:
                    apply_b0_overrides(rnd_env, b0_overrides)
                if spawn_overrides:
                    apply_spawn_overrides(rnd_env, spawn_overrides,
                                          spawn_tpls_now, seed=args.seed)

            # Konsumera ev. dirty-flaggor som råkade vara satta vid start
            # av ny iteration — vi vill bara reagera på drag som sker EFTER
            # denna rollout är klar.
            if viz is not None:
                viz.consume_b0_change()
                viz.consume_ticks_change()
                viz.consume_spawn_change()

            # Rollout-längd: använd slider-värdet om det är satt, annars
            # CLI-default. Slidern kan ändras efter rolloutens slut, vilket
            # triggar en ny inspelning via rerun-loopen.
            ticks_this_run = (viz.get_ticks() if viz is not None
                              else int(args.ticks))
            if verbose and ticks_this_run != int(args.ticks):
                print(f"    [ticks override] rollout length = {ticks_this_run}")
            history = run_inference(env, policies, mean, var, ticks_this_run,
                                    verbose=verbose, viz=viz,
                                    rnd_env=rnd_env)
            first_iteration = False

            if viz is None:
                break
            # wait_for_close returnerar tidigt om användaren dragit i en
            # slider (musen släpps -> dirty=True). Vid stängning av
            # fönstret (Q/ESC eller window-close) blir ``self._quit=True``
            # och loopen avslutas.
            try:
                viz.wait_for_close(
                    banner="inference finished — drag a slider (b0 or "
                           "rollout ticks) to re-record, or close window "
                           "to exit (Q/ESC)")
            except KeyboardInterrupt:
                if verbose:
                    print("\nInterrupted by user (Ctrl+C); closing window.")
                break
            b0_changed = viz.consume_b0_change()
            ticks_changed = viz.consume_ticks_change()
            spawn_changed = viz.consume_spawn_change()
            if not (b0_changed or ticks_changed or spawn_changed):
                # Inget ändrades — användaren stängde fönstret.
                break
            if verbose:
                reasons = []
                if b0_changed:
                    reasons.append("b0")
                if ticks_changed:
                    reasons.append("ticks")
                if spawn_changed:
                    reasons.append("spawn")
                print(f"\n[change] {'/'.join(reasons)} changed — "
                      f"re-recording rollout…")
    except KeyboardInterrupt:
        interrupted = True
        if verbose:
            print("\nInterrupted by user (Ctrl+C).")
        if history is None:
            history = {fid: [0.0] for fid in env.fgs}
    finally:
        if viz is not None:
            viz.close()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        np.savez(args.output, **{fid: np.asarray(h, dtype=np.float64) for fid, h in history.items()})
        if verbose:
            print(f"Saved biomass history to {args.output}")

    if verbose:
        print("\nFinal totals (tonnes):")
        for fid, h in history.items():
            print(f"  {fid:30s}  start={h[0]:12.3f}   end={h[-1]:12.3f}")
        print("==========================================")
        print("Inference complete.")
        print("==========================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
