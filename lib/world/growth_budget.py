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
      g     = gross / ((s* - u) * TICKS_PER_YEAR)

  non decision maker
      g     = (r_max + M1 + sum_j M2_ij) / TICKS_PER_YEAR

            M1 is 0 for an NDM without ``natural_mortality``, which
            keeps the old logistic form exactly. Section 173: a producer's
            own losses (respiration, sinking, lysis) are an explicit M1,
            so its growth rate is gross of them and the net growth at the
            light reference day is unchanged.

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
            (``preys_on``) and the predator is active in the project; a
            non decision maker predator grazes in the population step
            (section 177, microzooplankton).
  s* - u    the energy surplus at the reserve a DM settles on under
            ideal conditions (section 160). Growth per tick is
            g (s - u), and it used to be paid at a full reserve (1 - u),
            but the hunger gate h = (1 - s)/(1 - u) closes there, so no
            DM ever produced its budget. Ideal conditions: unlimited prey
            of the best checked, active pair, eaten every tick at the
            Holling ceiling, C = assimilation * prey energy / h_t (MJ per
            t per tick), paying resting_metabolism * feeding_cost (the
            library value, i.e. at the reference temperature). The
            growth is debited from the reserve (energy_content per t) and
            the new biomass dilutes it (s ME per t), so the reserve
            settles where h(s) C = cost + P (ec + s ME). Asking that the
            production there, P = g (s* - u), equals gross per tick gives

                s* = (C/(1-u) - cost - P ec) / (C/(1-u) + P ME)

            in closed form, with no policy and no rollout. The budget is
            infeasible when P >= Pmax = (C - cost) / (ec + u ME), and
            unrealisable under the mass balance when g > ME / ec (the
            growth is capped at the reserve above u). A DM without the
            energy data (no checked prey with an energy content, no
            energy_content, max_energy_reserve or resting_metabolism)
            falls back to the full reserve, 1 - u.

Income-funded growth (section 184)
----------------------------------
A decision maker with ``income_growth_kappa`` kappa grows out of this
tick's income, not out of its reserve: G = min(kappa I - C, B g (s - u)
ec), I the assimilated intake and C the action costs of the tick (MJ),
the rest of the income goes to the reserve, and with nothing to eat it
does not grow at all (the fat is kept for the winter; the reserve-funded
law of section 116 kept turning it into biomass after the food had
gone). Its g is paid over ``feeding_season_months`` n (default 12):

      g     = gross / ((s* - u) * TICKS_PER_YEAR) * 12 / n

because a group that can only grow while it is fed has to produce its
annual budget in those months. s* is unchanged (taken at the annual-mean
production, as before); the mass-balance ceiling is checked on the
season-sized g. The months are a stated biological parameter of the
budget, not a switch: when the group grows is decided by its food.
Herring: Apr - mid-Sep, 5.5 months (Bachiller et al. 2018),
research_notes/Sillens vinterbudget/growth_allocation.md.

Regional carrying capacity (section 185)
---------------------------------------
r_max is the r of a Schaefer (logistic) stock; a pool paid r_max at any
density grows at r_max - F until its food runs out and then crashes
(resume 184.13). ``regional_carrying_capacity`` K (t/km2, the stock's
effective capacity today - lowered by trawled nurseries and warming, not
the historical ceiling) closes it: every tick the engine removes the
fraction

      c * B_region,   c = r_max / (TICKS_PER_YEAR * K)

of the group's biomass and reserve in every cell, B_region its grid-mean
density (t per 1 km2 cell). The loss is NOT in the derived g, so net
growth at ideal food is r_max (1 - B/K) - F, as in the source model of
the r_max prior. It acts on the regional stock (recruitment compensation
at the nursery scale), not per cell, so it gives a policy no gradient.
The loader derives ``capacity_loss_rate`` = c at the library tick (a
"flux" for other tick lengths). Gadoids: r_max 0.37, K 1.02
(research_notes/Torskfiskarnas reglering/, resume 184.18-185).

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

from lib.world.energy_balance import intake_ceiling, prey_energy
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
    """(s - u) at a full reserve: 1 - maintenance_level (the fallback)."""
    return 1.0 - _as_float(species.get("maintenance_level"))


def ideal_intake(pred_id, species_defs, interactions, active_predators=None):
    """(C, best prey) at the Holling ceiling of the best checked pair.

    C in MJ per t predator per tick: max over checked pairs with an
    active prey of intake_ceiling(a, h_t) * assimilation * prey energy,
    the same numbers the runtime puts in ``max_intake_mat``,
    ``handling_time_mat`` and ``energy_gain_mat``. (0.0, None) when no
    pair has a usable prey energy.
    """
    active = None if active_predators is None else set(active_predators)
    species_defs = species_defs or {}
    pred_def = species_defs.get(pred_id) or {}
    prefix = pred_id + PREYS_ON
    best, best_id = 0.0, None
    for key, entry in (interactions or {}).items():
        if not key.startswith(prefix) or not isinstance(entry, dict):
            continue
        prey_id = key[len(prefix):]
        if not entry.get("preys_on", False) or PREYS_ON in prey_id:
            continue
        if active is not None and prey_id not in active:
            continue
        energy = prey_energy(species_defs.get(prey_id))
        if not energy:
            continue
        a = _as_float(entry.get("max_intake_rate"),
                      _as_float(pred_def.get("max_intake_rate")))
        assim = min(1.0, max(0.0, _as_float(entry.get("assimilation_factor"), 1.0)))
        rate = intake_ceiling(a, _as_float(entry.get("handling_time"))) * assim * energy
        if rate > best:
            best, best_id = rate, prey_id
    return best, best_id


def ideal_reserve(maintenance, intake, cost, energy_content, max_reserve,
                  production):
    """s* where h(s) C = cost + P (energy_content + s ME), or None.

    ``production`` P is the wanted g (s* - u) per tick. None when the
    energy cannot pay for it, i.e. P >= Pmax = (C - cost) / (ec + u ME).
    """
    u = maintenance
    k = intake / (1.0 - u)
    s = (k - cost - production * energy_content) / (k + production * max_reserve)
    return s if s > u else None


def income_growth(spec):
    """``(kappa, feeding_season_months)`` of an income-funded decision
    maker (section 184), or None for the reserve-funded law.

    ``income_growth_kappa`` empty or 0 is off; otherwise 0 < kappa <= 1.
    ``feeding_season_months`` (empty or 0 = 12) needs a kappa and lies in
    (0, 12].
    """
    spec = spec or {}

    def number(key):
        raw = spec.get(key, None)
        if raw in (None, ""):
            return 0.0
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number, got {raw!r}")
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite, got {raw!r}")
        return value

    kappa = number("income_growth_kappa")
    months = number("feeding_season_months")
    if kappa == 0.0:
        if months != 0.0:
            raise ValueError("feeding_season_months needs income_growth_kappa "
                             "(section 184): a season-sized g on the "
                             "reserve-funded law keeps growing on the fat")
        return None
    if not 0.0 < kappa <= 1.0:
        raise ValueError(f"income_growth_kappa must be in (0, 1], got {kappa}")
    if months == 0.0:
        months = 12.0
    if not 0.0 < months <= 12.0:
        raise ValueError(f"feeding_season_months must be in (0, 12], got {months}")
    return kappa, months


def regional_capacity(spec):
    """``regional_carrying_capacity`` K (t/km2) of a group, or None.

    Empty or 0 is off; otherwise a positive finite number (section 185).
    """
    raw = (spec or {}).get("regional_carrying_capacity", None)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"regional_carrying_capacity must be a number, got {raw!r}")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"regional_carrying_capacity must be >= 0 t/km2, got {raw!r}")
    return value if value > 0.0 else None


def capacity_loss_rate(spec):
    """c = r_max / (TICKS_PER_YEAR * K) per tick per (t/km2), or None.

    The regional logistic loss of section 185 at the library tick. K
    needs r_max: it closes the logistic whose r the budget pays.
    """
    capacity = regional_capacity(spec)
    if capacity is None:
        return None
    r_max = _as_float((spec or {}).get("r_max"))
    if not r_max > 0.0:
        raise ValueError("regional_carrying_capacity needs r_max (section "
                         "185): it closes the logistic whose r the derived "
                         "growth rate pays")
    return r_max / (TICKS_PER_YEAR * capacity)


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
        # Decision makers hunt in the predation step; a non decision maker
        # grazes in the population step only when its pair has an attack
        # rate (section 177, interactions.build_ndm_grazing).
        if not pred_def:
            continue
        if (not pred_def.get("is_decision_maker", False)
                and _as_float(entry.get("max_intake_rate")) <= 0.0):
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
    # Section 160: the ideal-conditions reserve the gate is taken at.
    maintenance: float = 0.0
    reference_fill: Optional[float] = None    # s*; None = full reserve
    ideal_intake: float = 0.0                 # C, MJ/t/tick
    ideal_cost: float = 0.0                   # MJ/t/tick
    best_prey: Optional[str] = None
    pmax_annual: Optional[float] = None       # (C - cost)/(ec + u ME), /yr
    season_factor: float = 1.0                # 12 / feeding_season_months (184)

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
        return annual / (self.gate * TICKS_PER_YEAR) * self.season_factor

    def describe_terms(self):
        """``describe()`` as (text, term) pieces.

        ``term`` names the budget quantity a piece shows - "growth_rate",
        "r_max", "m1", "m2:<predator>", "gate", "ticks_per_year" or
        "error" - and is None for the operators between them. The FG
        editor uses it to give each quantity its own hover help.
        """
        if self.error:
            return [(f"{self.prey_id}: {self.error}", "error")]
        terms = [(f"g = {self.growth_rate:.4g}", "growth_rate"),
                 (" = (", None), (f"r_max {self.r_max:g}", "r_max")]
        if self.is_decision_maker or self.m1_annual > 0.0:
            terms += [(" + ", None), (f"M1 {self.m1_annual:.3g}", "m1")]
        for pred, m2 in sorted(self.predation.items()):
            terms += [(" + ", None), (f"{pred} {m2:g}", f"m2:{pred}")]
        if self.is_decision_maker:
            terms += [(") / (", None), (f"{self.gate:g}", "gate"),
                      (" x ", None), (f"{TICKS_PER_YEAR:g}", "ticks_per_year"),
                      (")", None)]
            if self.season_factor != 1.0:
                terms += [(" x ", None),
                          (f"12/{12.0 / self.season_factor:g}", "season")]
        else:
            terms += [(") / ", None),
                      (f"{TICKS_PER_YEAR:g}", "ticks_per_year")]
        return terms

    def describe(self):
        return "".join(text for text, _term in self.describe_terms())


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
        if include_m1 else 0.0,
        predation=active_predation(prey_id, species_defs, interactions,
                                   active_predators),
    )
    if is_dm:
        budget.maintenance = _as_float(spec.get("maintenance_level"))
        budget.gate = max_gate(spec)
        if budget.gate <= 0.0:
            budget.error = "no growth window: maintenance_level must be below 1"
            return budget
        if not math.isfinite(budget.m1_annual):
            budget.error = "natural_mortality must be below 1"
            return budget
        _ideal_gate(budget, prey_id, spec, species_defs, interactions,
                    active_predators)
        if budget.error:
            return budget
        try:
            income = income_growth(spec)
        except ValueError as error:      # the FG editor, mid-keystroke
            budget.error = str(error)
            return budget
        if income is not None:
            budget.season_factor = 12.0 / income[1]
    else:
        budget.gate = 1.0
        if not math.isfinite(budget.m1_annual):
            budget.error = "natural_mortality must be below 1"
            return budget
    budget.growth_rate = (budget.gross_annual / (budget.gate * TICKS_PER_YEAR)
                          * budget.season_factor)
    if is_dm and budget.reference_fill is not None:
        ceiling = budget_mass_balance_ceiling(spec)
        if ceiling is not None and budget.growth_rate > ceiling:
            budget.error = (f"g = {budget.growth_rate:.4g} exceeds the "
                            f"mass-balance ceiling max_energy_reserve / "
                            f"energy_content = {ceiling:.4g}")
            budget.growth_rate = None
    return budget


def budget_mass_balance_ceiling(spec):
    """ME / energy_content: the growth a reserve above u can pay per tick."""
    ec = _as_float(spec.get("energy_content"))
    me = _as_float(spec.get("max_energy_reserve"))
    return me / ec if ec > 0.0 and me > 0.0 else None


def _ideal_gate(budget, prey_id, spec, species_defs, interactions,
                active_predators):
    """Set ``budget.gate`` to s* - u (section 160), or leave 1 - u."""
    u = budget.maintenance
    ec = _as_float(spec.get("energy_content"))
    me = _as_float(spec.get("max_energy_reserve"))
    cost = (_as_float(spec.get("resting_metabolism"))
            * _as_float(spec.get("feeding_cost"), 3.0))
    defs = dict(species_defs or {})
    defs[prey_id] = spec
    intake, best = ideal_intake(prey_id, defs, interactions, active_predators)
    if intake <= 0.0 or ec <= 0.0 or me <= 0.0 or cost <= 0.0:
        return
    budget.ideal_intake, budget.ideal_cost, budget.best_prey = intake, cost, best
    budget.pmax_annual = (intake - cost) / (ec + u * me) * TICKS_PER_YEAR
    production = budget.gross_annual / TICKS_PER_YEAR
    s_star = ideal_reserve(u, intake, cost, ec, me, production)
    if s_star is None:
        budget.error = (f"budget {budget.gross_annual:.4g} /yr exceeds the most "
                        f"the energy allows, Pmax {budget.pmax_annual:.4g} /yr "
                        f"(best prey {best})")
        return
    budget.reference_fill = s_star
    budget.gate = s_star - u


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
