import numpy as np

class FunctionalGroup:
    def __init__(self, group_id, params):
        self.group_id = group_id
        self.params = params
        
        # State variables (matrices)
        self.biomass = None  # B_X(c)
        self.energy_reserve = None  # R_X(c) = B_X(c) * E_X(c)
        self.temp_energy_gains = None # Intermediate predation gains
        
        # Metadata from params
        self.is_decision_maker = params.get('is_decision_maker', False)
        self.max_energy_reserve = params.get('max_energy_reserve', 1000.0)  # ME_X
        self.resting_metabolism = params.get('resting_metabolism', 0.0)  # Rest_X
        self.growth_rate = params.get('growth_rate', 0.0)  # MG_X
        # Starvation catabolism rate when q_x < 0. Kept separate from
        # growth_rate because biomass gain and starvation loss can operate
        # on different time scales.
        self.starve_rate = float(params.get('starve_rate', 0.0) or 0.0)
        # Visibility floor for rest/hide: with floor=f, the effective visible
        # fraction is 1 - rest * (1 - f), so a fully resting FG still exposes
        # f of its biomass per tick.
        self.visibility_floor = float(params.get('visibility_floor', 0.0) or 0.0)
        if self.visibility_floor < 0.0:
            self.visibility_floor = 0.0
        elif self.visibility_floor > 1.0:
            self.visibility_floor = 1.0
        # Density-independent natural mortality per tick.
        self.natural_mortality = float(params.get('natural_mortality', 0.0))
        # Recolonisation floor for non-decision makers: fraction of
        # max_carrying_capacity added per tick in every cell.
        self.seed_rate = float(params.get('seed_rate', 0.0))
        # Seasonal modulation of non-decision-maker growth rate:
        # r_eff(t) = r * (1 + AMP * sin(2*pi*(t+phase)/PERIOD)).
        self.seasonal_amplitude = float(params.get('seasonal_amplitude', 0.0) or 0.0)
        self.seasonal_period = float(params.get('seasonal_period', 0.0) or 0.0)
        # u_X: maintenance level. The relative energy fill ratio s_X required
        # to break even (q_X = s_X - u_X = 0). Below u_X the population
        # shrinks; above it, it grows. Method.pdf §6.
        self.maintenance_level = float(params.get('maintenance_level', 0.0))
        self.speed = params.get('movement_speed', 0.0)  # V_X
        # Minimum biomass (per cell) required to split via movement actions.
        # YAML/GUI value is in kg; internal biomass units are tonnes, so
        # convert kg -> tonnes here at the input boundary.
        # 0 = continuous biomass (no threshold).
        self.min_split_biomass = float(params.get('min_split_biomass', 0.0)) / 1000.0
        # Per-cell extinction threshold: cells with biomass below
        # extinction_threshold_factor * min_split_biomass are cleared and
        # recorded in starvation diagnostics. A value of 0 disables the
        # threshold; no threshold applies when min_split_biomass is 0.
        self.extinction_threshold_factor = float(
            params.get('extinction_threshold_factor', 0.5) or 0.0)
        if self.extinction_threshold_factor < 0.0:
            self.extinction_threshold_factor = 0.0

    def initialize_state(self, shape, initial_biomass=None, initial_energy_ratio=0.7,
                         randomize_energy=False, rng=None):
        if initial_biomass is not None:
            self.biomass = initial_biomass.copy()
        else:
            self.biomass = np.zeros(shape)

        if randomize_energy:
            # s_X(0) ~ Uniform[u_X - w, u_X + w] per cell, clippad till
            # [0, 1]. Centrerad på maintenance-nivån u_X så att
            # E[q_X(0)] = s_X(0) - u_X ≈ 0 (break-even) — det neutraliserar
            # ARS-bias från miljö-drift vid t=0 utan att smalna av
            # observationsstödet i s_X-dimensionen (som den gamla
            # *0.6-skalningen gjorde när u_X != 0.3). w = 0.3 ger ett
            # brett, symmetriskt stöd kring u_X. Per-FG: använder
            # self.maintenance_level så formeln auto-anpassar sig om
            # olika FGs får olika u_X.
            if rng is not None and hasattr(rng, "random"):
                u = rng.random(shape)
            else:
                u = np.random.rand(*shape)
            w = 0.3
            lo = max(0.0, float(self.maintenance_level) - w)
            hi = min(1.0, float(self.maintenance_level) + w)
            ratios = lo + (hi - lo) * u
            energy_per_ton = ratios * self.max_energy_reserve
        else:
            # E_X(c) = ratio * ME_X (uniform across the grid).
            energy_per_ton = np.full(shape, initial_energy_ratio * self.max_energy_reserve)
        self.energy_reserve = self.biomass * energy_per_ton
        self.temp_energy_gains = np.zeros(shape)

    @property
    def energy_level(self):
        """s_X(c) = E_X(c) / ME_X"""
        if self.biomass is None or self.max_energy_reserve == 0: 
            return np.zeros_like(self.biomass) if self.biomass is not None else None
        # Use np.divide with where to avoid warnings
        e_x = np.divide(self.energy_reserve, self.biomass, out=np.zeros_like(self.biomass), where=self.biomass > 1e-9)
        return e_x / self.max_energy_reserve

    def get_hunger(self):
        """h_X = max(0, 1 - s_X / 0.8)"""
        s_x = self.energy_level
        return np.maximum(0.0, 1.0 - s_x / 0.8)
