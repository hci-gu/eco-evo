import os
from stable_baselines3 import PPO
from lib.environments.sb3_wrapper import SB3Wrapper
from lib.config.settings import Settings

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

    def evaluate(self, steps=1000):
        obs = self.env.reset()
        for _ in range(steps):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)
            if self.render_mode == "human":
                self.env.render()
