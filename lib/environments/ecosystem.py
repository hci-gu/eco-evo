import numpy as np
from lib.world.grid import Grid
from lib.world.functional_group import FunctionalGroup

class EcosystemEnvironment:
    def __init__(self, grid_config, functional_groups, interactions, policies=None):
        self.grid = Grid(**grid_config)
        self.fgs = functional_groups  # Dictionary: id -> FunctionalGroup
        self.interactions = interactions
        self.policies = policies or {} # Dictionary: id -> PolicyNetwork
        self.tick_count = 0

    def step(self):
        """
        Executes one tick of the simulation.
        """
        # Randomize order of functional groups for this tick to ensure fairness
        shuffled_ids = list(self.fgs.keys())
        np.random.shuffle(shuffled_ids)
        self.ordered_fg_ids = shuffled_ids

        self._calculate_decisions()
        self._apply_predation()
        self._apply_metabolism()
        self._apply_movement()
        self._apply_growth_and_impact()
        
        self.tick_count += 1

    def _get_observation(self, fg_id):
        # Gathering local state for each cell: [B_own, E_own, B_prey1, B_prey2, Noise, ...]
        fg = self.fgs[fg_id]
        H, W = self.grid.height, self.grid.width
        
        obs_layers = []
        obs_layers.append(fg.biomass)
        obs_layers.append(fg.energy_level)
        
        # Other FGs
        for other_id, other_fg in self.fgs.items():
            if other_id != fg_id:
                obs_layers.append(other_fg.biomass)
        
        # Noise
        noise = self.grid.get_map('windfarm_noise')
        if noise is None:
            noise = np.zeros((H, W))
        obs_layers.append(noise)
        
        # Stack and return as (H, W, D)
        return np.stack(obs_layers, axis=-1)

    def _calculate_decisions(self):
        # Dictionary to store policy outputs for the current tick
        self.pi = {}
        for fg_id in self.ordered_fg_ids:
            fg = self.fgs[fg_id]
            if fg.is_decision_maker:
                obs = self._get_observation(fg_id)
                import torch
                obs_tensor = torch.from_numpy(obs).float()
                
                if fg_id in self.policies:
                    probs = self.policies[fg_id].get_action_probs(obs_tensor)
                else:
                    # Uniform fallback
                    num_actions = 5 + len(fg.params.get('menu', []))
                    probs = np.full((num_actions, self.grid.height, self.grid.width), 1.0 / num_actions)
                
                self.pi[fg_id] = {
                    'move': probs[0:4],
                    'rest': probs[4],
                    'eat': {prey_id: probs[5+i] for i, prey_id in enumerate(fg.params.get('menu', []))}
                }
            else:
                self.pi[fg_id] = None

    def _apply_predation(self):
        # 3. Update biomass after predation.
        # 4. Update reserve energy with intake and costs.
        
        # Track total demand for each prey
        total_demand = {fg_id: np.zeros((self.grid.height, self.grid.width)) for fg_id in self.fgs}
        
        # Calculate desired feeding D_XY(c)
        desired_feeding = {} # (pred_id, prey_id) -> matrix
        
        for pred_id in self.ordered_fg_ids:
            pred_fg = self.fgs[pred_id]
            if not pred_fg.is_decision_maker: continue
            
            pi_eat = self.pi[pred_id]['eat']
            hunger = pred_fg.get_hunger()
            
            for prey_id, pi_val in pi_eat.items():
                # D_XY = B_X * pi_eatY * I_XY * hunger
                inter_id = f'{pred_id}_preys_on_{prey_id}'
                max_intake = pred_fg.params['interaction'][inter_id]['max_intake_rate']
                d_xy = pred_fg.biomass * pi_val * max_intake * hunger
                desired_feeding[(pred_id, prey_id)] = d_xy
                total_demand[prey_id] += d_xy
                
        # Calculate actual feeding and update biomass/energy
        for prey_id in self.ordered_fg_ids:
            prey_fg = self.fgs[prey_id]
            demand = total_demand[prey_id]
            if np.any(demand > 0):
                # scale = min(1, B_Y / demand)
                scale = np.where(demand > prey_fg.biomass, prey_fg.biomass / (demand + 1e-9), 1.0)
                
                for (pred_id, target_prey_id), d_xy in desired_feeding.items():
                    if target_prey_id == prey_id:
                        actual_intake = d_xy * scale
                        actual_predator = self.fgs[pred_id]
                        
                        # Update predator energy
                        inter_id = f'{pred_id}_preys_on_{prey_id}'
                        energy_gain = actual_predator.params['interaction'][inter_id]['energy_gain']
                        actual_predator.energy_reserve += actual_intake * energy_gain
                        
                        # Update prey biomass and energy (proportional reduction)
                        # R_new = R_old * (B_new / B_old)
                        reduction_factor = np.where(prey_fg.biomass > 1e-9, 
                                                   (prey_fg.biomass - actual_intake) / (prey_fg.biomass + 1e-9), 
                                                   0.0)
                        prey_fg.energy_reserve *= reduction_factor
                        prey_fg.biomass -= actual_intake
                
                # After all predators ate prey_id:
                # Actually, Strategi.pdf says "Predation losses ... are recorded in an intermediate stage before movement."
                # "Prey biomass is reduced before movement."
                pass # Already handled above by prey_fg.biomass -= actual_intake

    def _apply_metabolism(self):
        for fg_id in self.ordered_fg_ids:
            fg = self.fgs[fg_id]
            if not fg.is_decision_maker: continue
            
            # Metabolism = Rest_X * (Action_Cost) * (1 + Noise_Sens * Noise)
            # Action_Cost is a weighted average based on pi
            pi = self.pi[fg_id]
            
            # Costs: movement_cost, feeding_cost, resting_cost
            cost_move = fg.params.get('movement_cost', 3.0)
            cost_eat = fg.params.get('feeding_cost', 3.0)
            cost_rest = fg.params.get('resting_cost', 1.0)
            
            avg_action_cost = (
                np.sum(pi['move'], axis=0) * cost_move +
                np.sum(list(pi['eat'].values()), axis=0) * cost_eat +
                pi['rest'] * cost_rest
            )
            
            # Noise impact
            noise = self.grid.get_map('windfarm_noise')
            if noise is None:
                noise = np.zeros_like(fg.biomass)
            noise_sens = fg.params.get('impact', {}).get('windfarm_noise', {}).get('impact_sensitivity', 0.0)
            noise_factor = 1.0 + noise_sens * noise
            
            total_metabolism = fg.resting_metabolism * avg_action_cost * noise_factor * fg.biomass
            fg.energy_reserve -= total_metabolism
            
            # Clip energy
            fg.energy_reserve = np.clip(fg.energy_reserve, 0.0, fg.biomass * fg.max_energy_reserve)

    def _apply_movement(self):
        # 5. Move biomass and reserve energy between cells.
        # 6. Mix inflows in each cell.
        
        for fg_id in self.ordered_fg_ids:
            fg = self.fgs[fg_id]
            if not fg.is_decision_maker or fg.speed == 0:
                continue
            
            pi = self.pi[fg_id]
            v = fg.speed
            
            # B_out,d = B * pi_d * v * Access_d
            # R_out,d = R * pi_d * v * Access_d
            
            # Directions: 0=N, 1=E, 2=S, 3=W (assuming standard image coordinates: N is up/-y)
            # Actually, let's define: N: (y-1), E: (x+1), S: (y+1), W: (x-1)
            
            b_total_in = np.zeros_like(fg.biomass)
            r_total_in = np.zeros_like(fg.energy_reserve)
            
            b_stay = fg.biomass.copy()
            r_stay = fg.energy_reserve.copy()
            
            # For each direction, calculate out-flow and where it goes
            offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)] # (dy, dx)
            for d, (dy, dx) in enumerate(offsets):
                pi_d = pi['move'][d]
                # Accessibility (placeholder: 1 everywhere except edges)
                access = np.ones_like(fg.biomass)
                if dy == -1: access[0, :] = 0  # North edge
                if dy == 1: access[-1, :] = 0 # South edge
                if dx == -1: access[:, 0] = 0  # West edge
                if dx == 1: access[:, -1] = 0 # East edge
                
                b_out = fg.biomass * pi_d * v * access
                r_out = fg.energy_reserve * pi_d * v * access
                
                b_stay -= b_out
                r_stay -= r_out
                
                # Move to neighbor
                b_in = np.roll(b_out, (dy, dx), axis=(0, 1))
                r_in = np.roll(r_out, (dy, dx), axis=(0, 1))
                
                b_total_in += b_in
                r_total_in += r_in
                
            fg.biomass = b_stay + b_total_in
            fg.energy_reserve = r_stay + r_total_in

    def _apply_growth_and_impact(self):
        # 7. Calculate net growth, mortality and impact.
        for fg_id in self.ordered_fg_ids:
            fg = self.fgs[fg_id]
            if fg_id == 'phytoplankton' or fg_id == 'benthic_community':
                # Logistic growth
                cc = fg.params.get('max_carrying_capacity', 100.0)
                mg = fg.growth_rate
                growth = mg * fg.biomass * (1.0 - fg.biomass / (cc + 1e-9))
                fg.biomass += growth
                fg.biomass = np.clip(fg.biomass, 0.0, cc)
            else:
                # Energy-based growth: B_new = B* * (1 + MG_X * q_X)
                s_x = fg.energy_level
                u_x = 0.3 # maintainance level
                q_x = s_x - u_x
                
                # Mortality impact (e.g. trawling, rotor)
                total_mortality_impact = np.zeros_like(fg.biomass)
                if 'impact' in fg.params:
                    for impact_id, impact_def in fg.params['impact'].items():
                        # Noise is handled in metabolism, others in mortality
                        if impact_id == 'windfarm_noise':
                            continue
                        
                        map_data = self.grid.get_map(impact_id)
                        if map_data is not None:
                            sensitivity = impact_def.get('impact_sensitivity', 0.0)
                            total_mortality_impact += fg.biomass * map_data * sensitivity
                
                growth = fg.biomass * fg.growth_rate * q_x
                
                # Energy management during growth/starvation (consistency check)
                total_loss = total_mortality_impact
                if np.any(growth < 0):
                    total_loss -= np.minimum(0.0, growth) # Add loss from negative growth
                
                # For cells with loss, reduce R proportionally
                loss_mask = total_loss > 0
                reduction = np.ones_like(fg.biomass)
                reduction[loss_mask] = (fg.biomass[loss_mask] - total_loss[loss_mask]) / (fg.biomass[loss_mask] + 1e-9)
                reduction = np.clip(reduction, 0.0, 1.0)
                
                fg.energy_reserve *= reduction
                fg.biomass += (growth - total_mortality_impact)
                fg.biomass = np.maximum(0.0, fg.biomass)
