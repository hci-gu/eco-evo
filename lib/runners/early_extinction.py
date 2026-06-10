import numpy as np


def normalize_early_extinction_config(config):
    """Return a small dict with validated early-stop settings."""
    cfg = dict(config or {})
    threshold = max(0.0, float(cfg.get("threshold", 0.0) or 0.0))
    penalty = max(0.0, float(cfg.get("penalty", 0.0) or 0.0))
    grace_ticks = max(0, int(cfg.get("grace_ticks", 0) or 0))
    monitor = str(cfg.get("monitor", "all") or "all").strip().lower()
    if monitor not in {"all", "decision_makers", "trained"}:
        monitor = "all"
    return {
        "threshold": threshold,
        "penalty": penalty,
        "grace_ticks": grace_ticks,
        "monitor": monitor,
        "enabled": threshold > 0.0 and penalty > 0.0,
    }


def init_early_extinction_state(env, trained_ids, config):
    """Capture per-species starting biomass for early-collapse detection."""
    cfg = normalize_early_extinction_config(config)
    if not cfg["enabled"]:
        return None

    trained_set = set(trained_ids or [])
    if cfg["monitor"] == "trained":
        monitor_ids = [fid for fid in trained_ids if fid in env.fgs]
    elif cfg["monitor"] == "decision_makers":
        monitor_ids = [
            fid for fid, fg in env.fgs.items()
            if getattr(fg, "is_decision_maker", False)
        ]
    else:
        monitor_ids = list(env.fgs.keys())

    b0 = {}
    for fid in monitor_ids:
        fg = env.fgs.get(fid)
        if fg is None or fg.biomass is None:
            continue
        total = float(np.sum(fg.biomass))
        if total > 1e-9:
            b0[fid] = total

    if not b0:
        return None
    cfg["b0"] = b0
    return cfg


def check_early_extinction(env, state, elapsed_ticks):
    """Return collapsed species ids once any monitored biomass is too low."""
    if state is None:
        return []
    if elapsed_ticks <= int(state["grace_ticks"]):
        return []
    threshold = float(state["threshold"])
    collapsed = []
    for fid, start_b in state["b0"].items():
        fg = env.fgs.get(fid)
        if fg is None or fg.biomass is None:
            continue
        current_b = float(np.sum(fg.biomass))
        if current_b <= threshold * start_b:
            collapsed.append(fid)
    return collapsed


def early_extinction_penalty(state, elapsed_ticks, n_ticks):
    """Penalty is strongest for early collapse and still nonzero near the end."""
    if state is None:
        return 0.0
    survived_frac = elapsed_ticks / max(1.0, float(n_ticks))
    survived_frac = min(1.0, max(0.0, survived_frac))
    return float(state["penalty"]) * (2.0 - survived_frac)
