"""Scenario files for the paper runs (inference with trained policies).

A scenario is a YAML file that names a base project manifest and says how
an inference world differs from it. Nothing is written back to the
manifest or the library: the differences are applied to an in-memory copy
of the manifest (``manifest``, ``nutrient_reservoir_scale``) or to the built
world (impact fields, the impact schedule, ``warming_c``).

Keys (every key optional except ``base``)::

    name: current_fishing            # default: the file name
    description: ...
    base: ../../mareld2.yaml         # relative to the scenario file
    run_name: 2026-11-17_1           # results/<run_name>/policy_*.pth
    years: 100                       # recorded rollout after the warm-up
    seeds: [0, 1, 2]
    record_every_days: 1             # time series resolution
    spatial_every_days: 30           # 0 = no spatial snapshots
    run:                             # world options (inference.py's flags)
      grid: [60, 60]
      mortality: on
      migration: on
      currents: on
      warmup_days: null              # null = the run's training warm-up
    manifest: {...}                  # deep-merged into the manifest;
                                     # impact_variables merge by impact_id,
                                     # decision_makers / non_decision_makers
                                     # by group_id, other lists are replaced
    nutrient_reservoir_scale: 1.0    # x the 50 m nitrate reservoir
    warming_c: 0.0                   # +deg C on every temperature layer,
                                     # as Q10 relative to today's climate
    impacts:                         # impact fields (default: the manifest's
      windfarm_noise:                # inference.impact_maps)
        map: ../../maps/poseidon_placeholder.npz
    masks:                           # cells where impacts are set to 0
      - map: ../../maps/poseidon_placeholder.npz
        key: mask
        impacts: [fishing]
    schedule:                        # piecewise-constant intensity x field
      - {from_year: 0, scale: {fishing: 0.2}}
      - {from_year: 5, scale: {fishing: 1.0}}
    warmup_scale: {fishing: 1.0}     # intensities during the warm-up
                                     # (default: the first schedule phase)
    sweep:                           # one variant per value (a list of
      name: k                        # such axes: every combination)
      values: [0.0, 0.5, 1.0, 2.0]
      set: [schedule.1.scale.fishing]
    variants:                        # or named variants instead of a sweep
      - {name: noise, set: {impacts.windfarm_noise.enabled: true}}
    same_as:                         # variants that are another scenario's
      "k=1": current_fishing/base    # world: not run, the results point
                                     # at that run (resolve_same_as checks
                                     # that the worlds are the same)

``fishing`` is an alias for both trawl impacts wherever an impact id is
expected. Paths in ``impacts`` and ``masks`` are relative to the scenario
file.
"""

import copy
import os
from dataclasses import dataclass, field

import yaml

# Wherever an impact id is expected, these names stand for several.
IMPACT_ALIASES = {
    "fishing": ("bottom_trawling", "pelagic_trawling"),
}

RUN_DEFAULTS = {
    "grid": [60, 60],
    "mortality": "on",
    "migration": "on",
    "currents": "on",
    "current_strength": 0.1,
    "current_period": 20,
    "current_seed": 0,
    "current_scale": 12.0,
    "tick_length": 6,
    "mass_balance": True,
    "warmup_days": None,
}

TOP_LEVEL_KEYS = {
    "name", "description", "base", "run_name", "years", "seeds",
    "record_every_days", "spatial_every_days", "run", "manifest",
    "nutrient_reservoir_scale", "warming_c", "impacts", "masks",
    "schedule", "warmup_scale", "sweep", "variants", "same_as",
}


def expand_impact_ids(name):
    """``fishing`` -> both trawls; any other id -> itself."""
    return IMPACT_ALIASES.get(name, (name,))


def expand_scale(scale):
    """``{alias or id: k}`` -> ``{id: k}``; an explicit id wins over an alias."""
    out = {}
    for key, value in (scale or {}).items():
        if key in IMPACT_ALIASES:
            for iid in IMPACT_ALIASES[key]:
                out.setdefault(iid, float(value))
    for key, value in (scale or {}).items():
        if key not in IMPACT_ALIASES:
            out[key] = float(value)
    return out


@dataclass
class Variant:
    """One concrete world setup: a scenario with its sweep value set."""
    scenario: str
    name: str
    spec: dict
    source: str
    sweep_name: str = None
    sweep_value: object = None
    same_as: str = None

    @property
    def base_dir(self):
        return os.path.dirname(os.path.abspath(self.source))

    def path(self, value):
        """A path from the scenario file, made absolute."""
        if value is None or os.path.isabs(value):
            return value
        return os.path.normpath(os.path.join(self.base_dir, value))

    @property
    def base_manifest(self):
        return self.path(self.spec["base"])

    @property
    def run(self):
        merged = dict(RUN_DEFAULTS)
        merged.update(self.spec.get("run") or {})
        return merged

    @property
    def years(self):
        return float(self.spec.get("years", 100))

    @property
    def seeds(self):
        return [int(s) for s in (self.spec.get("seeds") or [0])]

    @property
    def run_name(self):
        return self.spec.get("run_name")

    @property
    def record_every_days(self):
        return float(self.spec.get("record_every_days", 1))

    @property
    def spatial_every_days(self):
        return float(self.spec.get("spatial_every_days", 0) or 0)

    def schedule(self):
        """``[(from_year, {impact_id: scale})]`` sorted, starting at year 0.

        Each phase states the intensities it changes; the others carry over
        from the phase before (1.0 before the first)."""
        phases = sorted(self.spec.get("schedule") or [],
                        key=lambda p: float(p.get("from_year", 0)))
        out, current = [], {}
        if not phases or float(phases[0].get("from_year", 0)) > 0:
            out.append((0.0, {}))
        for phase in phases:
            current = dict(current)
            current.update(expand_scale(phase.get("scale")))
            out.append((float(phase.get("from_year", 0)), current))
        return out

    def warmup_scale(self):
        if self.spec.get("warmup_scale") is not None:
            return expand_scale(self.spec["warmup_scale"])
        return dict(self.schedule()[0][1])

    def to_meta(self):
        return {
            "scenario": self.scenario,
            "variant": self.name,
            "sweep_name": self.sweep_name,
            "sweep_value": self.sweep_value,
            "source": os.path.relpath(self.source),
            "same_as": self.same_as,
            "spec": self.spec,
        }

    def world(self):
        """Everything that shapes the simulated world (not what is recorded
        or how often): two variants with equal worlds give the same run."""
        impacts = {k: {kk: (self.path(vv) if kk == "map" else vv) for kk, vv in v.items()
                       if kk != "enabled"}
                   for k, v in (self.spec.get("impacts") or {}).items()
                   if (v or {}).get("enabled", True)}
        masks = [{**{k: v for k, v in m.items() if k != "enabled"}, "map": self.path(m["map"])}
                 for m in (self.spec.get("masks") or []) if m.get("enabled", True)]
        return {
            "base": self.base_manifest,
            "run_name": self.run_name,
            "years": self.years,
            "record_every_days": self.record_every_days,
            "run": self.run,
            "manifest": self.spec.get("manifest") or {},
            "nutrient_reservoir_scale": float(self.spec.get("nutrient_reservoir_scale", 1.0)),
            "warming_c": float(self.spec.get("warming_c", 0.0) or 0.0),
            "impacts": impacts,
            "masks": masks,
            "schedule": self.schedule(),
            "warmup_scale": self.warmup_scale(),
        }


def _set_path(tree, dotted, value):
    """Set ``a.b.0.c`` in nested dicts/lists, creating dicts on the way."""
    keys = dotted.split(".")
    node = tree
    for i, key in enumerate(keys[:-1]):
        nxt = keys[i + 1]
        if isinstance(node, list):
            node = node[int(key)]
            continue
        if key not in node or node[key] is None:
            node[key] = [] if nxt.isdigit() else {}
        node = node[key]
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def _format_value(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _sweep_axes(sweep, path):
    """``[(name, values, targets)]`` of one sweep or a list of sweeps."""
    axes = []
    for item in (sweep if isinstance(sweep, list) else [sweep]):
        targets = item.get("set") or []
        if isinstance(targets, str):
            targets = [targets]
        if not targets:
            raise ValueError(f"{path}: sweep needs 'set' (dotted paths)")
        axes.append((str(item.get("name", "value")), list(item.get("values") or []), targets))
    return axes


def load_scenario(path):
    """The variants of one scenario file.

    ``sweep`` (one axis, or a list of axes whose values are combined)
    makes one variant per value / combination, named ``k=0.5`` or
    ``r=0,k=0.5``; ``variants`` is an explicit list of
    ``{name, set: {dotted.path: value}}``; neither gives one variant,
    ``base``. ``same_as: {variant: "scenario/variant"}`` marks variants
    whose world is another scenario's (see ``resolve_same_as``).
    """
    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    unknown = sorted(set(spec) - TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown scenario keys {unknown}")
    if "base" not in spec:
        raise ValueError(f"{path}: a scenario needs 'base' (the project manifest)")
    if spec.get("sweep") and spec.get("variants"):
        raise ValueError(f"{path}: use 'sweep' or 'variants', not both")
    name = spec.get("name") or os.path.splitext(os.path.basename(path))[0]
    variants = _variants(spec, path, name)
    same_as = {str(k): str(v) for k, v in (spec.get("same_as") or {}).items()}
    names = {v.name for v in variants}
    missing = sorted(set(same_as) - names)
    if missing:
        raise ValueError(f"{path}: same_as names no variant {missing} (variants: {sorted(names)})")
    for v in variants:
        v.same_as = same_as.get(v.name)
    return variants


def _variants(spec, path, name):
    import itertools
    common = {k: v for k, v in spec.items() if k not in ("sweep", "variants", "same_as")}
    if spec.get("variants"):
        variants = []
        for item in spec["variants"]:
            variant_spec = copy.deepcopy(common)
            for target, value in (item.get("set") or {}).items():
                _set_path(variant_spec, target, value)
            variants.append(Variant(name, str(item["name"]), variant_spec, path,
                                    "variant", str(item["name"])))
        return variants
    if not spec.get("sweep"):
        return [Variant(name, "base", spec, path)]
    axes = _sweep_axes(spec["sweep"], path)
    variants = []
    for combo in itertools.product(*[values for _, values, _ in axes]):
        variant_spec = copy.deepcopy(common)
        labels = []
        for (axis, _, targets), value in zip(axes, combo):
            for target in targets:
                _set_path(variant_spec, target, value)
            labels.append(f"{axis}={_format_value(value)}")
        if len(axes) == 1:
            sweep_name, sweep_value = axes[0][0], combo[0]
        else:
            sweep_name = [axis for axis, _, _ in axes]
            sweep_value = list(combo)
        variants.append(Variant(name, ",".join(labels), variant_spec, path,
                                sweep_name, sweep_value))
    return variants


def resolve_same_as(variant):
    """The variant ``variant.same_as`` names (``scenario/variant``, the
    scenario file next to this one), after checking that both build the
    same world; None for a variant that runs itself."""
    if not variant.same_as:
        return None
    scenario, _, name = variant.same_as.partition("/")
    path = os.path.join(variant.base_dir, scenario + ".yaml")
    if not os.path.exists(path):
        raise ValueError(f"{variant.scenario}/{variant.name}: same_as {variant.same_as}: "
                         f"no {path}")
    matches = [v for v in load_scenario(path) if v.name == (name or "base")]
    if not matches:
        raise ValueError(f"{variant.scenario}/{variant.name}: same_as {variant.same_as}: "
                         f"{scenario} has no variant {name!r}")
    source = matches[0]
    if source.same_as:
        raise ValueError(f"{variant.scenario}/{variant.name}: same_as {variant.same_as} "
                         f"is itself an alias")
    mine, theirs = variant.world(), source.world()
    differ = sorted(k for k in mine if mine[k] != theirs[k])
    if differ:
        raise ValueError(f"{variant.scenario}/{variant.name}: same_as {variant.same_as}, "
                         f"but the worlds differ in {differ}")
    missing = sorted(set(variant.seeds) - set(source.seeds))
    if missing:
        raise ValueError(f"{variant.scenario}/{variant.name}: seeds {missing} are not "
                         f"among {variant.same_as}'s")
    return source


# ---------------------------------------------------------------- manifest

_LIST_KEYS = {
    "impact_variables": "impact_id",
    "decision_makers": "group_id",
    "non_decision_makers": "group_id",
    "functional_groups": "group_id",
}


def deep_merge(base, override, key=None):
    """``override`` merged into a copy of ``base`` (see module docstring)."""
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = deep_merge(base.get(k), v, k) if k in base else copy.deepcopy(v)
        return out
    if (isinstance(base, list) and isinstance(override, list)
            and key in _LIST_KEYS):
        ident = _LIST_KEYS[key]
        out = [copy.deepcopy(item) for item in base]
        index = {item.get(ident): i for i, item in enumerate(out)
                 if isinstance(item, dict)}
        for item in override:
            if isinstance(item, dict) and item.get(ident) in index:
                i = index[item[ident]]
                out[i] = deep_merge(out[i], item)
            else:
                out.append(copy.deepcopy(item))
        return out
    return copy.deepcopy(override)


def scenario_manifest(variant):
    """The base manifest with the scenario's manifest-level changes."""
    with open(variant.base_manifest) as f:
        project = yaml.safe_load(f) or {}
    project = deep_merge(project, variant.spec.get("manifest") or {})
    scale = variant.spec.get("nutrient_reservoir_scale")
    if scale is not None and float(scale) != 1.0:
        # The new-nitrogen source (lib/world/nutrients.py): the 50 m
        # reservoir of the two-layer column, else the sub-mixed-layer
        # concentration of the one-layer column. The start concentrations
        # are left as they are.
        block = ((project.get("simulation_settings") or {}).get("nutrients") or {})
        sub = block.get("subsurface_layer")
        if isinstance(sub, dict) and sub.get("reservoir_umol_l") is not None:
            owner, key = sub, "reservoir_umol_l"
        elif block.get("sub_mixed_layer_umol_l") is not None:
            owner, key = block, "sub_mixed_layer_umol_l"
        else:
            raise ValueError("nutrient_reservoir_scale needs a nutrients block "
                             "with a reservoir")
        table = owner[key]
        if isinstance(table, list):
            owner[key] = [float(v) * float(scale) for v in table]
        else:
            owner[key] = float(table) * float(scale)
    # An impact muted in training (the wind farm noise) that the scenario
    # gives a field: present, never observed - the checkpoints'
    # observation layout stays the one they were trained on.
    wanted = set()
    for name in (variant.spec.get("impacts") or {}):
        wanted.update(expand_impact_ids(name))
    for iv in project.get("impact_variables") or []:
        if isinstance(iv, dict) and iv.get("impact_id") in wanted and iv.get("muted"):
            iv["muted"] = False
            iv["observable"] = False
    return project


class MaterializedManifest:
    """Context manager: the scenario manifest written next to the base
    manifest (its relative paths resolve the same), removed on exit."""

    def __init__(self, variant):
        self.variant = variant
        self.path = None

    def __enter__(self):
        import tempfile
        project = scenario_manifest(self.variant)
        directory = os.path.dirname(self.variant.base_manifest)
        fd, self.path = tempfile.mkstemp(prefix=".scenario_", suffix=".yaml",
                                         dir=directory)
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(project, f, sort_keys=False)
        return self.path

    def __exit__(self, *exc):
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
        return False
