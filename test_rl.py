from lib.config.settings import Settings
from lib.runners.rl_runner import RLRunner
import os

def test_rl_training():
    # Create a test settings object
    # Use a small world size for faster testing
    # We can't easily modify the dataclass, so we'll rely on default or load a config
    # Let's just use default settings but maybe override if possible?
    # Settings is frozen, so we have to instantiate it with overrides.
    
    settings = Settings(
        world_size=10, # Small world
        num_agents=1, # Not used in RL runner directly but good to keep small
        max_steps=100,
        folder="results/test_rl"
    )
    
    if not os.path.exists(settings.folder):
        os.makedirs(settings.folder)
        
    runner = RLRunner(settings, species="cod", render_mode="none")
    
    print("Starting test training...")
    runner.train(total_timesteps=1000)
    print("Test training finished.")
    
    save_path = f"{settings.folder}/test_model"
    runner.save(save_path)
    
    print("Starting evaluation...")
    runner.evaluate(steps=50)
    print("Evaluation finished.")

if __name__ == "__main__":
    test_rl_training()
