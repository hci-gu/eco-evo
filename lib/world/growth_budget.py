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

Appetite and SDA (section 240)
------------------------------
A DM with ``appetite: reserve_room`` has h = 1 and its intake capped at
the room left in its reserve, so under ideal conditions the reserve
settles at s* = min(1, (C - cost - P ec)/(P ME)) - full whenever the
best pair can pay more than the budget (``ideal_reserve``). A DM's
``sda_fraction`` is netted from every pair's assimilation in C, as the
runtime nets it in ``energy_gain_mat``.

Shared handling (section 242)
-----------------------------
The runtime's multi-prey response (one handling budget for every
attacked prey) leaves s* unchanged: the ideal forager here attacks only
its best pair at an unlimited density, where that pair's rate is its
ceiling 1/h whatever else it might attack, so C is the same number.

Growth cost (section 241)
-------------------------
A DM's ``growth_cost`` c is the synthesis cost, J dissipated per J
deposited. The runtime buys a tonne for ec + c (ec + s ME) (its body
energy, structure plus the reserve it dilutes, at 1 + c, less the
diluted reserve that stays in the store), so the settling balance is
h(s) C = cost + (1 + c) P (ec + s ME), Pmax = (C - cost)/((1 + c)(ec +
u ME)) and the mass-balance ceiling ME / (ec + c (ec + s* ME)). It
costs nothing at maintenance, where nothing is deposited.

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

Material-capped growth (section 244)
------------------------------------
A decision maker with ``material_growth_share`` kappa_m builds its new
tonnes only out of this tick's food, while the energy may come from the
store: per cell G = min(B g (s - u) p, kappa_m I p / ec, max(I - C, 0),
R_settled - u ME B_settled) (MJ; p the price of a tonne, C the tick's
action costs). The material of a lean tonne (protein, N) cannot come
from the lipid reserve; the energy of activity and synthesis can, but
only while the tick's balance I - C is positive (section 244.10): fish
below the maintenance ration lose protein, they do not deposit it, so a
fish eating less than it spends never turns its fat into tissue
(research_notes/"Hostens tillvaxt och sommarens aptit", o-umlaut,
a-umlaut). kappa_m is the share of the assimilated intake (net of SDA,
as the engine's I) that can become lean tissue, from the protein route
(research_notes/"Tillvaxt ur dagens foda", a-umlaut, o-umlaut). No
intake, no growth, as for the income law; unlike it, the costs are not
taken off the top of a share of I. Energy is pooled, so s* and the
closure are those of the reserve law; the g is paid over
``feeding_season_months`` exactly as for the income law. The budget is
refused when the material cannot pay it under ideal conditions in the
feeding season: kappa_m h(s_n) C < P ec 12/n, s_n the reserve that
settles at the season's production P 12/n (h = 1 with ``appetite:
reserve_room``).
``income_growth_kappa`` and ``material_growth_share`` exclude each other.

Regional carrying capacity (section 185)
---------------------------------------
No library group carries one since section 243.4; the mechanism stays.
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

Open population (section 250; research_notes/"Oppen population",
O-umlaut). A DM with ``somatic_production`` P (/yr) is a window or a
nursery on a stock whose recruitment happens outside the box (recruits
imported into its juveniles, emigration, the migration of 246; stages.py,
open_pool.py). Its g pays the somatic production of the fish in the box
and nothing else:

      g = P / ((s* - u) * TICKS_PER_YEAR)

r_max (a whole-life-cycle rate, recruitment included) and the booked
M2 stay out of it: paying r_max and importing the recruits would count
recruitment twice, and the box's level is then I / (L - P) with L the
realised losses, so a booked M2 the predators do not take is a level
bias, not a growth rate (section 249). P is an annual field rate, so it
excludes ``r_max`` and ``feeding_season_months`` (the material law of 244
still caps the new tonnes on today's food).

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

from lib.world.energy_balance import (appetite_form, growth_cost,
                                     intake_ceiling, net_assimilation,
                                     prey_energy, sda_fraction, tonne_price)
from lib.world import set_point
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
    """(C, best prey) at the Holling ceilings of the checked pairs.

    C in MJ per t predator per tick: the whole eat share on the best
    checked pair with an active prey, ranked by intake_ceiling(a, h_t) *
    assimilation * prey energy - the same numbers the runtime puts in
    ``max_intake_mat``, ``handling_time_mat`` and ``energy_gain_mat``.
    (0.0, None) when no pair has a usable prey energy. Section 221: no
    pair caps the eat share any more (``predator_share`` removed).
    """
    active = None if active_predators is None else set(active_predators)
    species_defs = species_defs or {}
    pred_def = species_defs.get(pred_id) or {}
    sda = sda_fraction(pred_def)
    prefix = pred_id + PREYS_ON
    pairs = []
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
        # Section 240: the predator's SDA is netted, as in the runtime's
        # energy_gain_mat.
        assim = net_assimilation(assim, sda)
        rate = intake_ceiling(a, _as_float(entry.get("handling_time"))) * assim * energy
        if rate > 0.0:
            pairs.append((rate, prey_id))
    if not pairs:
        return 0.0, None
    best = max(pairs, key=lambda p: p[0])
    return best[0], best[1]


def ideal_reserve(maintenance, intake, cost, energy_content, max_reserve,
                  production, appetite="gate", growth_cost=0.0):
    """s* where h(s) C = cost + (1 + c) P (energy_content + s ME), or None.

    ``production`` P is the wanted g (s* - u) per tick. None when the
    energy cannot pay for it, i.e. P >= Pmax = (C - cost) / ((1 + c)
    (ec + u ME)).

    With the ``reserve_room`` appetite (section 240) h = 1 below a full
    reserve, so the balance C = cost + P (ec + s ME) has the root
    s = (C - cost - P ec) / (P ME); above 1 the reserve stays full (the
    room cap takes only what it needs), so s* = min(1, s). Same Pmax.

    ``growth_cost`` c (section 241) prices every deposited joule at
    1 + c: the new tonne's structure ec and its diluted reserve s ME are
    both bought at (1 + c), so P enters every branch as (1 + c) P.
    """
    u = maintenance
    production = production * (1.0 + growth_cost)
    if appetite == "reserve_room":
        if production <= 0.0:
            return 1.0 if intake > cost else None
        s = (intake - cost - production * energy_content) / (
            production * max_reserve)
        s = min(1.0, s)
        return s if s > u else None
    k = intake / (1.0 - u)
    s = (k - cost - production * energy_content) / (k + production * max_reserve)
    return s if s > u else None


INCOME_LAW = "income"
MATERIAL_LAW = "material"


def growth_law(spec):
    """``(law, share, feeding_season_months)`` of a decision maker that
    grows only while it is fed, or None for the reserve-funded law.

    ``law`` is ``"income"`` (``income_growth_kappa``, section 184) or
    ``"material"`` (``material_growth_share``, section 244); the two keys
    exclude each other. Each is off when empty or 0, otherwise in (0, 1].
    ``feeding_season_months`` (empty or 0 = 12) needs one of them and lies
    in (0, 12].
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
    share = number("material_growth_share")
    months = number("feeding_season_months")
    if kappa != 0.0 and share != 0.0:
        raise ValueError("income_growth_kappa and material_growth_share "
                         "exclude each other (sections 184, 244)")
    if kappa == 0.0 and share == 0.0:
        if months != 0.0:
            raise ValueError("feeding_season_months needs income_growth_kappa "
                             "or material_growth_share (sections 184, 244): a "
                             "season-sized g on the reserve-funded law keeps "
                             "growing on the fat")
        return None
    law, key, value = ((INCOME_LAW, "income_growth_kappa", kappa) if kappa
                       else (MATERIAL_LAW, "material_growth_share", share))
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{key} must be in (0, 1], got {value}")
    if months == 0.0:
        months = 12.0
    if not 0.0 < months <= 12.0:
        raise ValueError(f"feeding_season_months must be in (0, 12], got {months}")
    return law, value, months


def income_growth(spec):
    """``(kappa, feeding_season_months)`` of an income-funded decision
    maker (section 184), or None for any other law (reserve-funded, or
    material-capped: ``growth_law``).

    ``income_growth_kappa`` empty or 0 is off; otherwise 0 < kappa <= 1.
    ``feeding_season_months`` (empty or 0 = 12) needs a kappa or a
    ``material_growth_share`` and lies in (0, 12].
    """
    law = growth_law(spec)
    if law is None or law[0] != INCOME_LAW:
        return None
    return law[1], law[2]


def material_growth(spec):
    """``(kappa_m, feeding_season_months)`` of a material-capped decision
    maker (section 244), or None for any other law."""
    law = growth_law(spec)
    if law is None or law[0] != MATERIAL_LAW:
        return None
    return law[1], law[2]


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
    """True when the FG's growth_rate is derived rather than hand-set.

    An open pool (section 245, ``outside_density``) is derived with r_max 0:
    its local production replaces the booked losses at the outside density,
    and the exchange with the outside sits outside the budget.
    """
    if not isinstance(species, dict):
        return False
    if species.get("outside_density") not in (None, ""):
        return True
    try:
        if somatic_production(species) is not None:
            return True
    except ValueError:      # a bad P is the budget's error to report
        return True
    return _as_float(species.get("r_max")) > 0.0


SOMATIC_KEY = "somatic_production"


def somatic_production(spec):
    """P (/yr) of an open-population DM (section 250) or an open pool's
    local production P/B (section 268), or None."""
    raw = (spec or {}).get(SOMATIC_KEY)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float("nan")
    if value == 0.0:        # the FG editor's "0 = off"
        return None
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{SOMATIC_KEY} must be > 0 per year, got {raw!r}")
    return value


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
    window_share: Optional[float] = None      # reserve set point's growth window (271)
    material_need: Optional[float] = None     # share of I the material needs (244)
    somatic: Optional[float] = None           # P /yr of an open population (250)

    @property
    def predation_total(self):
        return sum(self.predation.values())

    @property
    def gross_annual(self):
        if self.somatic is not None:
            return self.somatic
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
        if self.somatic is not None:
            return [(f"g = {self.growth_rate:.4g}", "growth_rate"), (" = ", None),
                    (f"P {self.somatic:g}", "somatic"), (" / (", None),
                    (f"{self.gate:g}", "gate"), (" x ", None),
                    (f"{TICKS_PER_YEAR:g}", "ticks_per_year"), (")", None)]
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
                  species=None, include_m1=True, latitude_deg=None):
    """Full budget for one FG, or None when it has no ``r_max``.

    ``species`` overrides the FG's own entry in ``species_defs`` (the FG
    editor passes the values currently typed into its fields).
    ``include_m1=False`` leaves M1 out (``--mortality off``).
    ``latitude_deg`` is the site of the daylight calendar; with it a
    ``reserve_set_point`` (section 271) caps s* at the annual mean target,
    without it the block is inert, as on the engines.
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
        try:
            budget.somatic = somatic_production(spec)
        except ValueError as error:      # the FG editor, mid-keystroke
            budget.error = str(error)
            return budget
        if budget.somatic is not None:
            if _as_float(spec.get("r_max")) > 0.0:
                budget.error = (f"{SOMATIC_KEY} replaces r_max (section 250): r_max is "
                                "a whole-life-cycle rate whose recruitment is imported")
                return budget
            if _as_float(spec.get("feeding_season_months")) not in (0.0, 12.0):
                budget.error = (f"{SOMATIC_KEY} is an annual rate: feeding_season_months "
                                "does not apply (section 250)")
                return budget
        budget.maintenance = _as_float(spec.get("maintenance_level"))
        budget.gate = max_gate(spec)
        if budget.gate <= 0.0:
            budget.error = "no growth window: maintenance_level must be below 1"
            return budget
        if not math.isfinite(budget.m1_annual):
            budget.error = "natural_mortality must be below 1"
            return budget
        _ideal_gate(budget, prey_id, spec, species_defs, interactions,
                    active_predators, latitude_deg)
        if budget.error:
            return budget
        try:
            law = growth_law(spec)
        except ValueError as error:      # the FG editor, mid-keystroke
            budget.error = str(error)
            return budget
        if law is not None:
            budget.season_factor = 12.0 / law[2]
        if law is not None and law[0] == MATERIAL_LAW:
            _material_closure(budget, spec, law[1])
            if budget.error:
                return budget
    else:
        budget.gate = 1.0
        if spec.get("outside_density") not in (None, ""):
            # Section 268: an open pool may carry its field local production
            # P/B; the booked M2 above it is then paid by the exchange.
            try:
                budget.somatic = somatic_production(spec)
            except ValueError as error:      # the FG editor, mid-keystroke
                budget.error = str(error)
                return budget
            if budget.somatic is not None and _as_float(spec.get("r_max")) > 0.0:
                budget.error = (f"{SOMATIC_KEY} replaces r_max (section 268): an open "
                                "pool's local production is its whole growth")
                return budget
        if not math.isfinite(budget.m1_annual):
            budget.error = "natural_mortality must be below 1"
            return budget
    budget.growth_rate = (budget.gross_annual / (budget.gate * TICKS_PER_YEAR)
                          * budget.season_factor)
    if is_dm and budget.reference_fill is not None:
        ceiling = budget_mass_balance_ceiling(spec, budget.reference_fill)
        if ceiling is not None and budget.growth_rate > ceiling:
            budget.error = (f"g = {budget.growth_rate:.4g} exceeds the "
                            f"mass-balance ceiling max_energy_reserve / "
                            f"tonne price = {ceiling:.4g}")
            budget.growth_rate = None
    return budget


def budget_mass_balance_ceiling(spec, fill=0.0):
    """ME / price of a tonne: the growth a reserve above u can pay per tick.

    The price is ec + c (ec + s ME) at the fill s (section 241); ec
    without a growth cost."""
    ec = _as_float(spec.get("energy_content"))
    me = _as_float(spec.get("max_energy_reserve"))
    if not (ec > 0.0 and me > 0.0):
        return None
    return me / tonne_price(ec, growth_cost(spec), fill, me)


def _ideal_gate(budget, prey_id, spec, species_defs, interactions,
                active_predators, latitude_deg=None):
    """Set ``budget.gate`` to s* - u (section 160), or leave 1 - u.

    With a reserve set point (section 271) the reserve cannot settle above
    the defended level, so s* is at most the annual mean of s_T (the growth
    law pays g (s - u) every tick)."""
    u = budget.maintenance
    ec = _as_float(spec.get("energy_content"))
    me = _as_float(spec.get("max_energy_reserve"))
    try:
        c_growth = growth_cost(spec)
    except ValueError as error:      # the FG editor, mid-keystroke
        budget.error = str(error)
        return
    cost = (_as_float(spec.get("resting_metabolism"))
            * _as_float(spec.get("feeding_cost"), 3.0))
    defs = dict(species_defs or {})
    defs[prey_id] = spec
    intake, best = ideal_intake(prey_id, defs, interactions, active_predators)
    if intake <= 0.0 or ec <= 0.0 or me <= 0.0 or cost <= 0.0:
        return
    budget.ideal_intake, budget.ideal_cost, budget.best_prey = intake, cost, best
    budget.pmax_annual = ((intake - cost) / ((1.0 + c_growth) * (ec + u * me))
                          * TICKS_PER_YEAR)
    production = budget.gross_annual / TICKS_PER_YEAR
    s_star = ideal_reserve(u, intake, cost, ec, me, production,
                           appetite=appetite_form(spec), growth_cost=c_growth)
    if s_star is None:
        budget.error = (f"budget {budget.gross_annual:.4g} /yr exceeds the most "
                        f"the energy allows, Pmax {budget.pmax_annual:.4g} /yr "
                        f"(best prey {best})")
        return
    try:
        window = set_point.budget_gate(spec, latitude_deg, s_star, u)
    except ValueError as error:      # the FG editor, mid-keystroke
        budget.error = str(error)
        return
    if window is not None:
        # Section 271 (form 2): growth only in the clock's window, at the
        # target there; P is paid inside it.
        gate, target, share = window
        if gate <= 0.0:
            budget.error = (f"the reserve set point leaves no growth above "
                            f"maintenance_level {u:.4g} (section 271)")
            return
        budget.reference_fill = target
        budget.gate = gate
        budget.window_share = share
        return
    budget.reference_fill = s_star
    budget.gate = s_star - u


def _material_closure(budget, spec, share):
    """Refuse a material-capped budget (section 244) the food cannot build.

    While it feeds the DM must lay down P_n = P 12/n a tick (P = gross /
    TICKS_PER_YEAR), so under ideal conditions its reserve settles at the
    in-season s_n where h(s_n) C = cost + (1 + c) P_n (ec + s_n ME)
    (``ideal_reserve`` at P_n; h = 1 with ``appetite: reserve_room``) and
    it eats I = h(s_n) C. Only ``material_growth_share`` kappa_m of I can
    become lean material, so kappa_m h(s_n) C >= P_n ec, or the budget is
    unpayable; so is a season whose production the energy cannot pay
    (no s_n). The balance term max(I - C, 0) of section 244.10 needs no
    check of its own: at s_n, h(s_n) C - cost = (1 + c) P_n (ec + s_n ME)
    >= P_n p. g itself keeps the annual-mean s* (section 184). Without the
    energy data there is nothing to check. With ``reserve_room`` the check
    takes I = C, the ceiling; the realised intake at a full reserve is the
    room (about cost + P_n p), so for such a group the guard is looser
    than the material balance it stands for."""
    intake, cost = budget.ideal_intake, budget.ideal_cost
    ec = _as_float(spec.get("energy_content"))
    me = _as_float(spec.get("max_energy_reserve"))
    if budget.reference_fill is None or intake <= 0.0 or ec <= 0.0:
        return
    u = budget.maintenance
    production = budget.gross_annual / TICKS_PER_YEAR * budget.season_factor
    # Section 271: with a reserve set point P is laid down in its window.
    production = production / (getattr(budget, "window_share", None) or 1.0)
    appetite = appetite_form(spec)
    s_season = ideal_reserve(u, intake, cost, ec, me, production,
                             appetite=appetite, growth_cost=growth_cost(spec))
    if s_season is None:
        budget.error = (f"the energy cannot pay the feeding season's production "
                        f"{production * TICKS_PER_YEAR:.4g} /yr (section 244)")
        return
    if appetite == "reserve_room":
        hunger = 1.0
    else:
        hunger = min(1.0, max(0.0, (1.0 - s_season) / (1.0 - u)))
    need = production * ec
    budget.material_need = need / (hunger * intake) if hunger * intake > 0 else math.inf
    if share * hunger * intake < need:
        budget.error = (f"material_growth_share {share:.4g} cannot build the "
                        f"budget: it needs {budget.material_need:.4g} of the "
                        f"ideal intake (section 244)")


def derived_growth_rate(prey_id, species_defs, interactions,
                        active_predators: Optional[Iterable[str]] = None,
                        species=None, include_m1=True, latitude_deg=None):
    """The derived per-tick growth_rate, or None to keep the stored one."""
    budget = growth_budget(prey_id, species_defs, interactions,
                           active_predators, species=species,
                           include_m1=include_m1, latitude_deg=latitude_deg)
    if budget is None or budget.error:
        return None
    return budget.growth_rate
