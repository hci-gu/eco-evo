import yaml
import numpy as np
from lib.world.functional_group import FunctionalGroup
from lib.spawn import StrategySpec, distribute_with_floor, make_weights

def load_config(path):
    # The editor writes UTF-8 with a BOM. Using the host encoding on Windows
    # turns that BOM into "ï»¿" in the first key (e.g. species_definitions).
    # utf-8-sig accepts both editor output and ordinary UTF-8 YAML.
    with open(path, 'r', encoding='utf-8-sig') as f:
        return yaml.safe_load(f)


def _resolve_initial_biomass_range(*sources):
    """Resolve an (min, max) initial-biomass range from one or more dict sources.

    Sources are checked in the given order; the first source that provides any
    biomass information wins. Schema: ``initial_biomass_min`` and
    ``initial_biomass_max`` (both non-negative ints; max >= min).
    Returns ``(min, max)`` as floats, or ``(None, None)`` if no source provides
    a value.
    """
    def _as_num(v):
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    for src in sources:
        if not isinstance(src, dict):
            continue
        mn = _as_num(src.get("initial_biomass_min"))
        mx = _as_num(src.get("initial_biomass_max"))
        if mn is not None or mx is not None:
            if mn is None:
                mn = mx
            if mx is None:
                mx = mn
            if mx < mn:
                mn, mx = mx, mn
            return mn, mx
    return None, None


def _sample_total_biomass(min_b, max_b, rng):
    """Sample an initial total biomass uniformly from [min_b, max_b] integers.

    If both bounds are None, returns 1000.0 (legacy fallback).
    """
    if min_b is None and max_b is None:
        return 1000.0
    if min_b is None:
        min_b = max_b
    if max_b is None:
        max_b = min_b
    lo = int(round(min_b))
    hi = int(round(max_b))
    if lo == hi:
        return float(lo)
    if rng is not None and hasattr(rng, "integers"):
        return float(rng.integers(lo, hi + 1))
    return float(np.random.randint(lo, hi + 1))

def _derive_fg_spawn_seed(base_seed, fg_id):
    """Derive a deterministic, FG-specific spawn seed from a base seed.

    Used so that a generation-wide ``spawn_seed`` produces *different* but
    reproducible per-FG patterns (otherwise every FG would share the same
    Perlin/colony layout). Returns None if ``base_seed`` is None so legacy
    callers keep their previous behaviour.
    """
    if base_seed is None:
        return None
    # Mix base seed and FG id into a 32-bit positive integer. zlib.adler32
    # is cheap and deterministic across Python versions/platforms.
    import zlib
    key = f"{int(base_seed)}::{fg_id}".encode("utf-8")
    return int(zlib.adler32(key)) & 0x7FFFFFFF


def _build_spawn_spec(spawn_cfg, default_seed=None):
    """Bygg en `StrategySpec` från ett YAML `spawn:`-block (eller None).

    Returnerar None om ``spawn_cfg`` saknas / är felaktigt (då används
    legacy-vägen). Mode-fältet defaultar till 'uniform' om enbart `params`
    anges men inget mode — det ger samma utseende som legacy-vägen.
    """
    if not isinstance(spawn_cfg, dict):
        return None
    mode = str(spawn_cfg.get('mode', 'uniform')).strip().lower()
    if not mode:
        return None
    params = {k: v for k, v in spawn_cfg.items() if k not in ('mode', 'seed')}
    seed = spawn_cfg.get('seed', default_seed)
    try:
        return StrategySpec(mode=mode, params=params, seed=seed)
    except Exception:
        return None


def _spawn_biomass_distribution(grid_size, total_b, min_per_cell,
                                allowed_mask=None, rng=None,
                                spawn_spec=None, env_context=None,
                                project_seed=None):
    """Distribute ``total_b`` over the grid.

    Two code paths:

    1. **New strategy-driven path** (when ``spawn_spec`` is not None):
       compute per-cell weights via ``lib.spawn.make_weights`` and allocate
       biomass via ``lib.spawn.distribute_with_floor`` (greedy fill +
       per-cell floor). Keeps the same sum/floor contract as the legacy
       path but supports arbitrary weight strategies (uniform, perlin,
       colony, env_driven).

    2. **Legacy cluster-spawn path** (default, ``spawn_spec is None``):
       random pick of cells, biomass drawn uniformly in
       ``[min_per_cell, remaining]`` until the remainder drops below the
       floor. Preserved bit-for-bit so existing projects without a
       ``spawn:`` block produce identical output.

    Rules (legacy path):
      * Eligible cells are those where ``allowed_mask`` is truthy. If
        ``allowed_mask`` is None, every cell is eligible.
      * Each picked cell receives a biomass drawn uniformly from
        ``[min_per_cell, total_b]`` (or ``min_per_cell .. remaining`` if less
        biomass than ``total_b`` is left). The pick continues until the
        running remainder drops below ``min_per_cell``.
      * Any leftover (< ``min_per_cell``) is merged into one of the already
        seeded cells so total biomass is preserved exactly.
      * If the number of eligible cells is too small to fit
        ``ceil(total_b / total_b)``-many cells while honouring the floor,
        cells are allowed to receive more than ``total_b`` (i.e. the
        remainder is absorbed and the loop exits naturally).
      * When ``min_per_cell`` is 0 or non-positive the function falls back
        to the legacy uniform-Dirichlet spread over all eligible cells.
    """
    # ---- New strategy-driven path -----------------------------------------
    if spawn_spec is not None:
        H, W = grid_size
        if total_b <= 0:
            return np.zeros((H, W), dtype=np.float64)
        ctx = dict(env_context) if env_context else {}
        if allowed_mask is not None and 'allowed_mask' not in ctx:
            ctx['allowed_mask'] = np.asarray(allowed_mask).reshape(H, W).astype(bool)
        weights = make_weights(spawn_spec, (H, W), context=ctx,
                               project_seed=project_seed)
        return distribute_with_floor(weights, float(total_b),
                                     float(min_per_cell),
                                     allowed_mask=ctx.get('allowed_mask'))

    # ---- Legacy cluster-spawn path (unchanged) ----------------------------
    H, W = grid_size
    if rng is None:
        rng = np.random
    initial_b = np.zeros(grid_size, dtype=np.float64)
    if total_b <= 0:
        return initial_b

    # Determine the eligible cell pool.
    if allowed_mask is None:
        flat_idx = np.arange(H * W)
    else:
        mask = np.asarray(allowed_mask).reshape(H, W)
        flat_idx = np.flatnonzero(mask > 0)
        if flat_idx.size == 0:
            # No accessible cell — fall back to the full grid so that biomass
            # never silently vanishes when accessibility is not configured.
            flat_idx = np.arange(H * W)

    # Legacy path: no clustering threshold, spread biomass across the full
    # eligible pool with a uniform Dirichlet-like draw. Preserves prior
    # behaviour for FGs whose min_split_biomass == 0.
    if min_per_cell is None or min_per_cell <= 0:
        if hasattr(rng, "random"):
            draws = rng.random(flat_idx.size)
        else:
            draws = np.random.rand(flat_idx.size)
        draws = (draws / (draws.sum() + 1e-9)) * total_b
        flat = initial_b.reshape(-1)
        flat[flat_idx] = draws
        return initial_b

    # Cluster-spawn path: pick cells one at a time, assign biomass uniformly
    # in [min_per_cell, total_b] (capped at remainder), repeat until the
    # remainder is below the floor; merge the leftover into a seeded cell.
    flat = initial_b.reshape(-1)
    seeded = []
    remaining = float(total_b)
    available = list(flat_idx)
    # `available` is the pool of eligible cells that have not yet received
    # biomass. Picking without replacement keeps the seed cells distinct.
    while remaining >= min_per_cell and available:
        # Random pick from the available pool.
        if hasattr(rng, "integers"):
            i = int(rng.integers(0, len(available)))
        else:
            i = int(np.random.randint(0, len(available)))
        cell = available.pop(i)
        # Draw biomass uniformly in [min_per_cell, min(total_b, remaining)].
        hi = min(float(total_b), remaining)
        # `hi` can equal `min_per_cell` near the end; np.random.uniform is OK
        # with hi == lo, but guard against a tiny numerical underflow.
        if hi <= min_per_cell:
            amount = remaining  # absorb the remainder, exit next iteration
        else:
            if hasattr(rng, "uniform"):
                amount = float(rng.uniform(min_per_cell, hi))
            else:
                amount = float(np.random.uniform(min_per_cell, hi))
        amount = min(amount, remaining)
        flat[cell] = amount
        seeded.append(cell)
        remaining -= amount

    # Edge case: no seed could be placed (e.g. total_b < min_per_cell). Put
    # all biomass into a single random eligible cell so it isn't lost. This
    # also allows the cell to exceed ``total_b`` if needed — see spec point 3.
    if not seeded:
        if hasattr(rng, "integers"):
            i = int(rng.integers(0, flat_idx.size))
        else:
            i = int(np.random.randint(0, flat_idx.size))
        flat[flat_idx[i]] = float(total_b)
        return initial_b

    # Merge leftover (< min_per_cell) into a randomly chosen seeded cell.
    if remaining > 0:
        if hasattr(rng, "integers"):
            j = int(rng.integers(0, len(seeded)))
        else:
            j = int(np.random.randint(0, len(seeded)))
        flat[seeded[j]] += remaining

    return initial_b


def setup_full_mareld_mvp(library_path='fgconfig/fg_library.yaml', grid_size=(60, 60), seed=None, spawn_seed=None,
                          allowed_mask=None, library_config=None):
    lib = library_config if library_config is not None else load_config(library_path)
    spec_defs = lib['species_definitions']
    inter_defs = lib['interaction_definitions']

    rng = np.random.default_rng(seed) if seed is not None else np.random
    spawn_rng_base = spawn_seed if spawn_seed is not None else seed
    spawn_allowed_mask = None
    if allowed_mask is not None:
        spawn_allowed_mask = np.asarray(allowed_mask).reshape(grid_size).astype(bool)

    # Topo-sort FG ids by env_driven refs so dependencies spawn first.
    _all_ids = list(spec_defs.keys())
    _id_set = set(_all_ids)
    _deps = {sid: [] for sid in _all_ids}
    for sid in _all_ids:
        cfg = spec_defs[sid].get('spawn') if isinstance(spec_defs[sid], dict) else None
        if not isinstance(cfg, dict):
            continue
        if str(cfg.get('mode', '')).strip().lower() != 'env_driven':
            continue
        for ref in (cfg.get('refs') or []):
            if isinstance(ref, dict):
                name = ref.get('name')
                if name and name in _id_set and name != sid:
                    _deps[sid].append(name)
    _remaining = list(_all_ids)
    _ordered = []
    _done = set()
    _safety = 0
    while _remaining and _safety < len(_all_ids) + 5:
        _safety += 1
        progressed = False
        nxt = []
        for sid in _remaining:
            if all(d in _done for d in _deps[sid]):
                _ordered.append(sid); _done.add(sid); progressed = True
            else:
                nxt.append(sid)
        _remaining = nxt
        if not progressed:
            _ordered.extend(_remaining); break

    env_fields = {}
    fgs = {}
    for sid in _ordered:
        specs = spec_defs[sid]
        # Merge specs with interaction data
        params = specs.copy()
        params['interaction'] = {}
        params['menu'] = []
        
        # Find interactions for this species
        for iid, idef in inter_defs.items():
            if iid.startswith(f"{sid}_preys_on_"):
                prey_id = iid.replace(f"{sid}_preys_on_", "")
                if idef.get('preys_on', False):
                    params['menu'].append(prey_id)
                    params['interaction'][iid] = idef
        fg = FunctionalGroup(sid, params)

        # Initial total biomass: prefer library range (min/max or legacy scalar),
        # fallback to 1000. Sampled fresh on each call so spatial layouts vary.
        min_b, max_b = _resolve_initial_biomass_range(specs)
        sample_rng = rng if seed is not None else None
        total_b = _sample_total_biomass(min_b, max_b, sample_rng)

        # Cluster-aware spawn: enforce a per-cell floor of
        # 5 * min_split_biomass (kg -> tonnes already applied inside
        # FG.__init__) so newly spawned cells start safely above the
        # sub-threshold mask. For FGs with min_split == 0 this collapses
        # to the legacy uniform spread.
        min_per_cell = 5.0 * float(getattr(fg, 'min_split_biomass', 0.0))
        # Opt-in: if the species defines a ``spawn:`` block (mode + params),
        # use the new strategy-driven path; otherwise fall back to the
        # legacy cluster-spawn so existing projects are bit-for-bit
        # preserved.
        fg_spawn_seed = _derive_fg_spawn_seed(spawn_rng_base, sid)
        spawn_spec = _build_spawn_spec(params.get('spawn'), default_seed=fg_spawn_seed)
        spawn_rng = (np.random.default_rng(fg_spawn_seed)
                     if (spawn_seed is not None and fg_spawn_seed is not None)
                     else rng)
        initial_b = _spawn_biomass_distribution(
            grid_size, total_b, min_per_cell,
            allowed_mask=spawn_allowed_mask, rng=spawn_rng,
            spawn_spec=spawn_spec, project_seed=fg_spawn_seed,
            env_context={'env_fields': env_fields, 'allowed_mask': spawn_allowed_mask})
        env_fields[sid] = np.asarray(initial_b, dtype=np.float64)
        fg.initialize_state(grid_size, initial_biomass=initial_b)
        fgs[sid] = fg
        
    return fgs

def _resolve_inference_initial_biomass(*sources):
    """Read fixed inference initial biomass (ton) from the first source that has it.

    Schema: ``inference_initial_biomass`` (non-negative number). Returns float or None.
    """
    for src in sources:
        if not isinstance(src, dict):
            continue
        v = src.get("inference_initial_biomass")
        if v is None or v == "":
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f < 0:
            continue
        return f
    return None


def compute_inference_b0_defaults(project_path, grid_size):
    """Return ``{fg_id: b0_default_in_tonnes}`` for every active FG in the
    project, beräknat på exakt samma sätt som :func:`load_project_config`
    gör i ``mode='inference'``:

        b0_default = max(inference_initial_biomass * biomass_scale, 1.0)

    där ``biomass_scale = (H*W) / (ref_H * ref_W)`` med referens-griden
    läst från ``project_metadata.reference_grid_{width,height}``.

    Muted FGs och FGs utan ``inference_initial_biomass`` (eller 0) utelämnas.
    Används av visualiseraren (``--visual``) för att initiera b0-slidrarna
    så slider-rangen ``[0, 4 * b0_default]`` matchar exakt det b0-värde
    en probe/inference-rollout faktiskt startar på vid mittposition.
    """
    if not project_path:
        return {}
    project = load_config(project_path)
    pmeta = project.get('project_metadata', {}) or {}
    _MIN_REF = 3
    def _as_pos_int(v, default):
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return default
        return _MIN_REF if iv < _MIN_REF else iv
    ref_w = _as_pos_int(pmeta.get('reference_grid_width'), 60)
    ref_h = _as_pos_int(pmeta.get('reference_grid_height'), 60)
    H_act, W_act = int(grid_size[0]), int(grid_size[1])
    ref_cells = ref_w * ref_h
    act_cells = H_act * W_act
    biomass_scale = (act_cells / ref_cells) if ref_cells > 0 else 1.0
    out = {}
    for key in ('decision_makers', 'non_decision_makers', 'functional_groups'):
        for fg in project.get(key, []) or []:
            if not isinstance(fg, dict) or fg.get('muted'):
                continue
            gid = fg.get('group_id')
            if not gid:
                continue
            fixed = _resolve_inference_initial_biomass(fg)
            if fixed is None or fixed <= 0.0:
                continue
            out[gid] = max(float(fixed) * biomass_scale, 1.0)
    return out


def load_project_config(project_path, library_path='fgconfig/fg_library.yaml', grid_size=(60, 60), seed=None, mode='train',
                        spawn_seed=None, allowed_mask=None, project_config=None, library_config=None):
    """Load a project config and build its FunctionalGroups.

    ``spawn_seed`` (optional) controls the per-cell biomass distribution
    independently of ``seed`` (which drives total-biomass / energy sampling).
    When set, every FG's spawn map is fully determined by ``(spawn_seed,
    fg_id)`` — used by ``train.py`` to share *identical* biomass maps across
    all deltas/workers within one generation. When ``None``, falls back to
    ``seed`` so each rollout gets its own spawn layout.
    """
    project = project_config if project_config is not None else load_config(project_path)
    rng = np.random.default_rng(seed) if seed is not None else None
    # Spawn RNG base: spawn_seed overrides seed for the spatial layout, so
    # that a whole generation can share one map even though individual
    # rollouts still vary in total biomass / starting energy via ``seed``.
    spawn_rng_base = spawn_seed if spawn_seed is not None else seed
    spawn_allowed_mask = None
    if allowed_mask is not None:
        spawn_allowed_mask = np.asarray(allowed_mask).reshape(grid_size).astype(bool)
    lib = library_config if library_config is not None else load_config(library_path)
    spec_defs = lib['species_definitions']
    inter_defs = lib['interaction_definitions']
    
    # Support both the new split (decision_makers / non_decision_makers) and the
    # legacy unified functional_groups list for backward compatibility.
    # Also retain per-FG project overrides (e.g. initial_biomass).
    project_fg_ids = []
    project_fg_overrides = {}
    for key in ('decision_makers', 'non_decision_makers', 'functional_groups'):
        for fg in project.get(key, []) or []:
            if not isinstance(fg, dict):
                continue
            # Skip muted FGs: they're soft-deleted and must be invisible to
            # the simulation, while their YAML configuration is preserved so
            # the user can quickly unmute them later.
            if fg.get('muted'):
                continue
            gid = fg.get('group_id')
            if gid and gid not in project_fg_ids:
                project_fg_ids.append(gid)
                project_fg_overrides[gid] = fg
    # Keep empty metadata slots for callers that still unpack the legacy
    # four-value shape.
    inactive_metadata = ([], {}, [])

    # Reference-grid scaling: biomass values entered in the FG editor and the
    # Inference tab are defined relative to a reference grid (Rx x Ry cells)
    # stored under project_metadata.reference_grid_{width,height}. When the
    # actual runtime grid (grid_size = (H, W)) differs in cell count, all
    # initial biomass values (training range min/max and inference fixed
    # value) are scaled linearly by (H*W) / (Ry*Rx). A hard floor of 1 ton
    # applies to the scaled lower bound and to the scaled inference value so
    # tiny grids never produce zero-biomass spawns. The reference defaults
    # to 60 x 60 (the historical mareld2 grid) when unspecified.
    pmeta = project.get('project_metadata', {}) or {}
    # Runtime floor: reference grid must be at least 3x3 (matches the
    # live-validation rule in fgconfig). Sub-floor values from legacy /
    # hand-edited project files are rounded up so downstream scaling math
    # is always well-defined.
    _MIN_REF = 3
    def _as_pos_int(v, default):
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return default
        if iv < _MIN_REF:
            return _MIN_REF
        return iv
    ref_w = _as_pos_int(pmeta.get('reference_grid_width'), 60)
    ref_h = _as_pos_int(pmeta.get('reference_grid_height'), 60)
    H_act, W_act = int(grid_size[0]), int(grid_size[1])
    ref_cells = ref_w * ref_h
    act_cells = H_act * W_act
    biomass_scale = (act_cells / ref_cells) if ref_cells > 0 else 1.0

    # ------------------------------------------------------------------
    # Topological ordering of spawn for env_driven refs
    # ------------------------------------------------------------------
    # Some FGs may declare spawn mode=env_driven and reference other FGs
    # via ``refs: [{name: <other_fg_id>, ...}]`` (e.g. zooplankton following
    # phytoplankton). Those references are resolved at runtime by exposing
    # the dependency's *already spawned* biomass map under
    # ``context['env_fields'][<name>]``. To make this work we must spawn
    # the dependency before the dependant. Below we build the dependency
    # graph (only on currently active FGs) and topologically sort it.
    # Cycles and missing references fall back to the project's declared
    # order with the missing refs silently ignored by ``weights_env_driven``.
    def _spawn_cfg_for(sid):
        ovr = project_fg_overrides.get(sid, {}) if isinstance(
            project_fg_overrides.get(sid, {}), dict) else {}
        cfg = ovr.get('spawn') if isinstance(ovr, dict) else None
        if cfg is None:
            cfg = spec_defs.get(sid, {}).get('spawn')
        return cfg if isinstance(cfg, dict) else None

    _active_set = set(project_fg_ids)
    _deps = {sid: [] for sid in project_fg_ids}
    for sid in project_fg_ids:
        cfg = _spawn_cfg_for(sid)
        if not cfg:
            continue
        if str(cfg.get('mode', '')).strip().lower() != 'env_driven':
            continue
        for ref in (cfg.get('refs') or []):
            if not isinstance(ref, dict):
                continue
            name = ref.get('name')
            if name and name in _active_set and name != sid:
                _deps[sid].append(name)

    # Kahn's algorithm: stable topo-sort preserving the original FG order
    # for ties so non-env_driven FGs keep their legacy spawn order.
    _remaining = list(project_fg_ids)
    _spawn_order = []
    _spawned_set = set()
    _safety = 0
    while _remaining and _safety < len(project_fg_ids) + 5:
        _safety += 1
        progressed = False
        new_remaining = []
        for sid in _remaining:
            if all(d in _spawned_set for d in _deps.get(sid, [])):
                _spawn_order.append(sid)
                _spawned_set.add(sid)
                progressed = True
            else:
                new_remaining.append(sid)
        _remaining = new_remaining
        if not progressed:
            # Cycle detected: append the rest in declared order so the
            # simulation still runs; env_driven refs that aren't satisfied
            # yet will simply be ignored by weights_env_driven.
            _spawn_order.extend(_remaining)
            break

    # Accumulator: env_fields[<fg_id>] -> (H, W) initial biomass array.
    # Passed into _spawn_biomass_distribution via env_context so that
    # env_driven strategies can use it via context['env_fields'][name].
    env_fields = {}

    fgs = {}
    for sid in _spawn_order:
        if sid not in spec_defs:
            continue
            
        specs = spec_defs[sid]
        params = specs.copy()
        params['interaction'] = {}
        params['menu'] = []
        
        active_fg_set = set(project_fg_ids)
        # Observability: list of FG ids that this DM can see in its input
        # space. Built from interaction_definitions entries of the form
        # ``{sid}_observes_{other_id}`` with ``observes: True``. Observed FGs
        # that are muted are intentionally KEPT in the list (the policy still
        # gets an input slot for them, fed with 0 at runtime — see
        # EcosystemEnvironment.build_static_caches). If no observability
        # entries exist for this sid (legacy projects), we fall back to "see
        # everything" for backward compatibility.
        observes_list = []
        observes_seen_any = False
        for iid, idef in inter_defs.items():
            if iid.startswith(f"{sid}_preys_on_"):
                prey_id = iid.replace(f"{sid}_preys_on_", "")
                # Skip predation relations whose prey is muted (or absent).
                if prey_id not in active_fg_set:
                    continue
                if idef.get('preys_on', False):
                    params['menu'].append(prey_id)
                    params['interaction'][iid] = idef
            elif iid.startswith(f"{sid}_observes_"):
                obs_id = iid.replace(f"{sid}_observes_", "")
                # Track that at least one observability entry exists for sid,
                # so we know to use the explicit list instead of the
                # see-everything fallback.
                if 'observes' in idef:
                    observes_seen_any = True
                if idef.get('observes', False):
                    observes_list.append(obs_id)
        params['observes'] = observes_list if observes_seen_any else None
                
        fg = FunctionalGroup(sid, params)

        # Initial total biomass: per-project override range > library range >
        # 1000 fallback. The actual scalar `total_b` is sampled uniformly from
        # [min, max] on every call so each spatial layout varies even when the
        # configured range is unchanged.
        override = project_fg_overrides.get(sid, {})
        if mode == 'inference':
            # Inference uses the per-FG fixed value entered in the FG config
            # tool's "Inference" tab. No random sampling: the exact value is
            # spread spatially. Missing values fall back to 0.0. The value
            # is scaled by the reference-grid factor with a 1-ton floor.
            fixed = _resolve_inference_initial_biomass(override)
            if fixed is None:
                total_b = 0.0
            else:
                scaled = float(fixed) * biomass_scale
                total_b = max(scaled, 1.0) if float(fixed) > 0 else 0.0
        else:
            min_b, max_b = _resolve_initial_biomass_range(override, specs)
            # When a generation-wide ``spawn_seed`` is supplied, also draw
            # ``total_b`` from a deterministic per-FG RNG so the *whole*
            # biomass map (sum + shape) is identical across rollouts in
            # the generation. Otherwise use ``rng`` (legacy: varies per
            # rollout).
            _tb_rng = (np.random.default_rng(_derive_fg_spawn_seed(spawn_rng_base, sid))
                       if spawn_seed is not None else rng)
            # Scale the training range by the reference-grid factor. The
            # scaled lower bound is clamped to >= 1 ton so degenerate small
            # grids cannot produce zero-biomass spawns.
            if min_b is not None and max_b is not None:
                min_b_s = max(min_b * biomass_scale, 1.0)
                max_b_s = max(max_b * biomass_scale, min_b_s)
                total_b = _sample_total_biomass(min_b_s, max_b_s, _tb_rng)
            else:
                total_b = _sample_total_biomass(min_b, max_b, _tb_rng)

        # Cluster-aware spawn (see _spawn_biomass_distribution): per-cell
        # floor = 5 * min_split_biomass pushes spawned cells safely above
        # the sub-threshold mask (which lock-in previously affected
        # seals/porpoises). When an inference/API caller supplies an
        # allowed_mask, spawning is restricted to those cells.
        min_per_cell = 5.0 * float(getattr(fg, 'min_split_biomass', 0.0))
        # Opt-in strategy-driven spawn: project override > library spec.
        # Missing block on both sides -> legacy cluster-spawn (unchanged).
        spawn_cfg = override.get('spawn') if isinstance(override, dict) else None
        if spawn_cfg is None:
            spawn_cfg = specs.get('spawn')
        # Per-FG sub-seed derived deterministically from (spawn_rng_base,
        # sid). This guarantees that two FGs do not share the exact same
        # Perlin/colony pattern even when they all inherit the same
        # generation-wide ``spawn_seed``.
        fg_spawn_seed = _derive_fg_spawn_seed(spawn_rng_base, sid)
        spawn_spec = _build_spawn_spec(spawn_cfg, default_seed=fg_spawn_seed)
        # When ``spawn_seed`` was supplied we also drive the legacy
        # cluster-spawn RNG from the derived per-FG seed so its random
        # cell picks are identical across rollouts in the generation.
        # Otherwise we keep the shared ``rng`` (legacy behaviour).
        spawn_rng = (np.random.default_rng(fg_spawn_seed)
                     if (spawn_seed is not None and fg_spawn_seed is not None)
                     else rng)
        initial_b = _spawn_biomass_distribution(
            grid_size, total_b, min_per_cell,
            allowed_mask=spawn_allowed_mask, rng=spawn_rng,
            spawn_spec=spawn_spec, project_seed=fg_spawn_seed,
            env_context={'env_fields': env_fields, 'allowed_mask': spawn_allowed_mask})
        # Expose this FG's freshly spawned biomass map to subsequent FGs
        # (env_driven refs use the already-spawned dependencies).
        env_fields[sid] = np.asarray(initial_b, dtype=np.float64)
        # Training: E_X(c) ~ Uniform(0, ME_X) per cell so policies see varied
        # initial energy fill levels. Inference keeps the deterministic
        # 0.7 * ME_X default for reproducible scenario comparisons.
        randomize_energy = (mode == 'train')
        fg.initialize_state(grid_size, initial_biomass=initial_b,
                            randomize_energy=randomize_energy, rng=rng)
        fgs[sid] = fg

    return fgs, *inactive_metadata
