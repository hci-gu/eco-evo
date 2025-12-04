import os
from stable_baselines3 import PPO
from lib.environments.sb3_wrapper import SB3Wrapper
from lib.config.settings import Settings

def noop(a, b, c):
    pass

class RLRunner:
    def __init__(self, settings: Settings, species="cod", render_mode=None, n_steps=2048):
        self.settings = settings
        self.species = species
        self.render_mode = render_mode
        
        # Initialize the environment wrapper
        self.env = SB3Wrapper(settings, species=species, render_mode=render_mode)
        
        # Initialize PPO model
        # Use a MultiInputPolicy if we had dict observations, but we flattened them to Box
        # So MlpPolicy is appropriate for flattened vector inputs.
        # We might want CnnPolicy if we kept the 3x3 grid structure, but 3x3 is very small for CNN.
        # Flattened MLP is likely best for this local patch.
        self.model = PPO("MlpPolicy", self.env, verbose=1, tensorboard_log=f"{settings.folder}/tensorboard/", n_steps=n_steps)

    def train(self, total_timesteps=10000):
        print(f"Starting PPO training for {self.species}...")
        self.model.learn(total_timesteps=total_timesteps)
        print("Training complete.")

    def save(self, path):
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")

    def evaluate(self, steps=1000, callback=noop):
        """
        Evaluate the model for a given number of steps.
        
        Args:
            steps: Number of steps to run
            callback: Function(info_dict, cumulative_reward, is_done) called each step
                     info_dict contains population info in format compatible with PBMRunner
        """
        obs = self.env.reset()
        cumulative_reward = 0.0
        
        for step in range(steps):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)
            cumulative_reward += rewards.sum() if hasattr(rewards, 'sum') else rewards
            
            # Build info dict compatible with PBMRunner format
            # Access underlying petting_zoo env to get world state
            info_dict = self._build_info_dict()
            callback(info_dict, cumulative_reward, False)
            
            if self.render_mode == "human":
                self.env.render()
            
            # Check if all environments are done
            if dones.all() if hasattr(dones, 'all') else dones:
                break
        
        callback(None, cumulative_reward, True)
        return cumulative_reward, step + 1, None
    
    def _build_info_dict(self):
        """
        Build an info dictionary compatible with PBMRunner format.
        Maps species to functional groups: FG_0=plankton, FG_1=sprat, FG_2=herring, FG_3=cod
        """
        from lib.model import MODEL_OFFSETS
        
        info = {}
        world = self.env.env.world  # Access underlying petting_zoo world
        
        # Map species to functional group indices
        species_to_fg = {
            'plankton': 0,
            'sprat': 1, 
            'herring': 2,
            'cod': 3
        }
        
        # Get grid dimensions (excluding padding)
        pad = 1
        inner_world = world[pad:-pad, pad:-pad]
        n_cells = inner_world.shape[0] * inner_world.shape[1]
        
        for species, fg_idx in species_to_fg.items():
            if species in MODEL_OFFSETS:
                biomass_channel = MODEL_OFFSETS[species]["biomass"]
                total_biomass = inner_world[..., biomass_channel].sum()
                # Store as population density (per cell) to match PBMRunner format
                info[f"info--population/FG_{fg_idx}"] = total_biomass / n_cells
        
        return info
