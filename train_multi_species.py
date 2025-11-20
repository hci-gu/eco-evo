import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from lib.config.settings import Settings
from lib.runners.rl_runner import RLRunner

def evaluate_survival(runner, episodes=5):
    """
    Evaluates the average survival time (in steps) over a number of episodes.
    """
    survival_times = []
    for _ in range(episodes):
        obs = runner.env.reset()
        steps = 0
        # Run until done or max steps (to avoid infinite loops)
        max_steps = runner.settings.max_steps * 4 # Same as truncation limit
        
        for _ in range(max_steps):
            action, _ = runner.model.predict(obs)
            obs, rewards, dones, infos = runner.env.step(action)
            steps += 1
            
            # Check if ANY environment is done (since we broadcast done)
            # In our wrapper, dones is boolean array.
            if np.any(dones):
                break
        survival_times.append(steps)
    
    return np.mean(survival_times)

def train_multi_species():
    print("Setting up multi-species training...")
    settings = Settings(
        world_size=12,
        num_agents=1,
        max_steps=5000,
        folder="results/multi_species_train"
    )
    
    if not os.path.exists(settings.folder):
        os.makedirs(settings.folder)

    # Training parameters
    iterations = 50
    # PPO collects n_steps per environment before updating.
    # Total buffer size = n_steps * num_envs
    # We must run for at least this many steps to trigger an update.
    n_steps_per_env = 128
    num_envs = settings.world_size * settings.world_size
    steps_per_iteration = n_steps_per_env * num_envs
    eval_episodes = 10 # Increased from 5 to reduce variance
        
    # Initialize runners for each species
    species_list = ["cod", "herring", "sprat"]
    runners = {}
    for sp in species_list:
        print(f"Initializing runner for {sp} with n_steps={n_steps_per_env}...")
        runners[sp] = RLRunner(settings, species=sp, render_mode="none", n_steps=n_steps_per_env)
    
    # History tracking
    history = {sp: {"steps": [], "survival": []} for sp in species_list}
    best_survival = {sp: 0.0 for sp in species_list}
    best_model_paths = {sp: None for sp in species_list}
    
    print(f"Starting training loop: {iterations} iterations of {steps_per_iteration} steps per species.")
    
    # Initial evaluation
    for sp in species_list:
        avg_survival = evaluate_survival(runners[sp], episodes=eval_episodes)
        history[sp]["survival"].append(avg_survival)
        history[sp]["steps"].append(0)
        best_survival[sp] = avg_survival
        
        # Save initial as best (baseline)
        path = os.path.join(settings.folder, f"model_{sp}_best.zip")
        runners[sp].save(path)
        best_model_paths[sp] = path
        
        print(f"Initial {sp}: Avg Survival = {avg_survival:.2f}")

    for i in range(1, iterations + 1):
        print(f"\n--- Iteration {i} ---")
        
        for sp in species_list:
            runner = runners[sp]
            
            # 1. Update environment with CURRENT models of OTHER species
            # Reverting to latest models to ensure co-evolutionary arms race.
            # Using static "best" models caused stagnation.
            other_models = {}
            for other_sp in species_list:
                if other_sp != sp:
                    other_models[other_sp] = runners[other_sp].model
            
            runner.env.set_species_models(other_models)
            
            # 2. Train
            print(f"Training {sp}...")
            runner.model.learn(total_timesteps=steps_per_iteration, reset_num_timesteps=False)
            
            # 3. Evaluate
            avg_survival = evaluate_survival(runner, episodes=eval_episodes)
            
            # 4. Record
            current_step = i * steps_per_iteration
            history[sp]["survival"].append(avg_survival)
            history[sp]["steps"].append(current_step)
            
            print(f"{sp} Step {current_step}: Avg Survival = {avg_survival:.2f}")
            
            # Check if best
            if avg_survival > best_survival[sp]:
                print(f"New best for {sp}! ({avg_survival:.2f} > {best_survival[sp]:.2f})")
                best_survival[sp] = avg_survival
                best_path = os.path.join(settings.folder, f"model_{sp}_best.zip")
                runner.save(best_path)
                best_model_paths[sp] = best_path
            
            # Save regular checkpoint
            model_path = os.path.join(settings.folder, f"model_{sp}_{current_step}.zip")
            runner.save(model_path)
        
        # Plotting after each iteration
        plt.figure(figsize=(12, 8))
        for sp in species_list:
            plt.plot(history[sp]["steps"], history[sp]["survival"], marker='o', linestyle='-', label=sp)
            
        plt.title(f"Multi-Species RL Training ({settings.world_size}x{settings.world_size} Grid)")
        plt.xlabel("Training Steps (per species)")
        plt.ylabel("Average Survival Time (Steps)")
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(settings.folder, "multi_species_survival_plot.png")
        plt.savefig(plot_path)
        plt.close()
        
    print(f"Training complete. Plot saved to {plot_path}")

if __name__ == "__main__":
    train_multi_species()
