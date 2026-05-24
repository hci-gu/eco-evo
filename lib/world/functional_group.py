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
        
        # Costs
        self.movement_cost = params.get('movement_cost', 1.0)
        self.feeding_cost = params.get('feeding_cost', 1.0)
        self.resting_cost = params.get('resting_cost', 1.0)

    def initialize_state(self, shape, initial_biomass=None, initial_energy_ratio=0.7,
                         randomize_energy=False, rng=None):
        if initial_biomass is not None:
            self.biomass = initial_biomass.copy()
        else:
            self.biomass = np.zeros(shape)

        if randomize_energy:
            # E_X(c) ~ Uniform(0, ME_X) per cell. Used during training so
            # policies see varied initial energy fill levels.
            if rng is not None and hasattr(rng, "random"):
                ratios = rng.random(shape)
            else:
                ratios = np.random.rand(*shape)
            # Skala ner uniform-samplet så E[s_X(0)] = 0.3 (= typiskt u_X)
            # istället för 0.5. Bryter den triviala startgradienten där
            # zooplankton/växare annars får gratis positiv reward de
            # första tickarna oavsett policy.
            ratios = ratios * np.float64(0.6)
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

    def calculate_metabolism(self, actions_mask, noise_impact=None):
        """
        Calculates energy loss per tick.
        Rest_X * (Action_Cost) * (1 + Noise_Sens * Noise)
        """
        # For simplicity in MVP, we can assume a mean cost if actions are distributed,
        # or calculate per-action cost in the simulation loop.
        pass
