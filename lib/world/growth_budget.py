"""Growth rate derived from a literature budget and the predation matrix.

``growth_rate`` used to be a free per-FG number. It is now DERIVED, so a
prey's growth follows the predators it actually has in the project:
adding a predator raises the prey's growth by that predator's predation
share, removing it lowers it by the same amount. The net low-density
growth of the prey (``r_max``) is invariant either way.

Budget (all rates per year; the result is per tick at the library
calibration, ``LIBRARY_TICK_HOURS``):

  decision maker
      gross = r_max + M1 + sum_j M2_ij
      g     = gross / ((1 - u) * TICKS_PER_YEAR)

  non decision maker (logistic, no separate mortality term)
      g     = (r_max + sum_j M2_ij) / TICKS_PER_YEAR

  r_max     ``species_definitions.<fg>.r_max``. For a DM the literature
            maximum NET population growth at low density (recruitment and
            individual growth included, all natural mortality removed).
            For an NDM the net growth after natural mortality and
            predation by groups the model does not resolve.
  M1        the FG's own ``natural_mortality`` (per tick, converted to a
            per-year rate). It holds every loss the model does not
            resolve as predation by another FG.
  M2_ij     ``interaction_definitions.<j>_preys_on_<i>.predation_mortality``,
            the literature predation mortality of prey i caused by
            predator j. Counted only while the pair is checked
            (``preys_on``), the predator is a decision maker (only DMs
            hunt in the tick) and the predator is active in the project.
  1 - u     the largest energy surplus, reached at a full reserve
            (s = 1, where the hunger gate closes). Ideal
            conditions mean a full reserve.

When ``--mortality off`` the tick never applies M1, so the loader passes
``include_m1=False`` and M1 is left out of the budget as well: removing a
loss removes its growth compensation, exactly as unchecking a predator
does.

An FG without ``r_max`` keeps its hand-set ``growth_rate`` (legacy
entries). ``r_max`` and ``predation_mortality`` are per year and
therefore tick-independent; the derived ``growth_rate`` is per tick at
the library calibration and goes through the normal "growth" rescale
rule for other tick lengths (``lib/world/tick_time.py``).

Shared by the loader (``lib/config/config_loader.py``), the FG editor
(``fgconfig/fgconfig.py``) and the tests, so all three compute the same
number.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from lib.world.tick_time import LIBRARY_TICK_HOURS

TICKS_PER_YEAR = 365.0 * 24.0 / LIBRARY_TICK_HOURS

PREYS_ON = "_preys_on_"


def _as_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def annual_rate_from_tick_loss(m_tick):
    """Per-tick loss fraction -> continuous per-year rate (the "loss" rule)."""
    m = _as_float(m_tick)
    if m <= 0.0:
        return 0.0
    if m >= 1.0:
        return math.inf
    return -TICKS_PER_YEAR * math.log(1.0 - m)


def max_gate(species):
    """(s - u)_max for a decision maker: 1 - maintenance_level."""
    return 1.0 - _as_float(species.get("maintenance_level"))


def has_budget(species):
    """True when the FG's growth_rate is derived rather than hand-set."""
    return isinstance(species, dict) and _as_float(species.get("r_max")) > 0.0


def active_predation(prey_id, species_defs, interactions,
                     active_predators: Optional[Iterable[str]] = None):
    """{predator_id: M2 per year} for every pair that currently counts."""
    active = None if active_predators is None else set(active_predators)
    suffix = PREYS_ON + prey_id
    out = {}
    for key, entry in (interactions or {}).items():
        if not key.endswith(suffix) or not isinstance(entry, dict):
            continue
        pred_id = key[: -len(suffix)]
        if not pred_id or PREYS_ON in pred_id:
            continue
        if not entry.get("preys_on", False):
            continue
        if active is not None and pred_id not in active:
            continue
        pred_def = (species_defs or {}).get(pred_id) or {}
        if not pred_def.get("is_decision_maker", False):
            continue
        m2 = _as_float(entry.get("predation_mortality"))
        if m2 > 0.0:
            out[pred_id] = m2
    return out


@dataclass
class GrowthBudget:
    prey_id: str
    is_decision_maker: bool
    r_max: float
    m1_annual: float
    predation: Dict[str, float] = field(default_factory=dict)
    gate: float = 1.0
    growth_rate: Optional[float] = None
    error: Optional[str] = None

    @property
    def predation_total(self):
        return sum(self.predation.values())

    @property
    def gross_annual(self):
        return self.r_max + self.m1_annual + self.predation_total

    def per_tick(self, annual):
        """Contribution of an annual term to the per-tick growth_rate."""
        if self.error:
            return 0.0
        return annual / (self.gate * TICKS_PER_YEAR)

    def describe(self):
        if self.error:
            return f"{self.prey_id}: {self.error}"
        parts = [f"r_max {self.r_max:g}"]
        if self.is_decision_maker:
            parts.append(f"M1 {self.m1_annual:.3g}")
        for pred, m2 in sorted(self.predation.items()):
            parts.append(f"{pred} {m2:g}")
        gate = f" / ({self.gate:g} x {TICKS_PER_YEAR:g})" if (
            self.is_decision_maker) else f" / {TICKS_PER_YEAR:g}"
        return (f"g = {self.growth_rate:.4g} = (" + " + ".join(parts)
                + f"){gate}")


def growth_budget(prey_id, species_defs, interactions,
                  active_predators: Optional[Iterable[str]] = None,
                  species=None, include_m1=True):
    """Full budget for one FG, or None when it has no ``r_max``.

    ``species`` overrides the FG's own entry in ``species_defs`` (the FG
    editor passes the values currently typed into its fields).
    ``include_m1=False`` leaves M1 out (``--mortality off``).
    """
    spec = species if species is not None else (species_defs or {}).get(prey_id)
    if not has_budget(spec):
        return None
    is_dm = bool(spec.get("is_decision_maker", False))
    budget = GrowthBudget(
        prey_id=prey_id,
        is_decision_maker=is_dm,
        r_max=_as_float(spec.get("r_max")),
        m1_annual=annual_rate_from_tick_loss(spec.get("natural_mortality"))
        if (is_dm and include_m1) else 0.0,
        predation=active_predation(prey_id, species_defs, interactions,
                                   active_predators),
    )
    if is_dm:
        budget.gate = max_gate(spec)
        if budget.gate <= 0.0:
            budget.error = "no growth window: maintenance_level must be below 1"
            return budget
        if not math.isfinite(budget.m1_annual):
            budget.error = "natural_mortality must be below 1"
            return budget
    else:
        budget.gate = 1.0
    budget.growth_rate = budget.gross_annual / (budget.gate * TICKS_PER_YEAR)
    return budget


def derived_growth_rate(prey_id, species_defs, interactions,
                        active_predators: Optional[Iterable[str]] = None,
                        species=None, include_m1=True):
    """The derived per-tick growth_rate, or None to keep the stored one."""
    budget = growth_budget(prey_id, species_defs, interactions,
                           active_predators, species=species,
                           include_m1=include_m1)
    if budget is None or budget.error:
        return None
    return budget.growth_rate
