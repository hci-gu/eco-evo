import os
import numpy as np
import matplotlib.pyplot as plt
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

def train_and_plot():
    print("Setting up training and plotting...")
    settings = Settings(
        world_size=10,
        num_agents=1,
        max_steps=100,
        folder="results/train_plot"
    )
    
    if not os.path.exists(settings.folder):
        os.makedirs(settings.folder)
        
    runner = RLRunner(settings, species="cod", render_mode="none")
    
    # Training parameters
    iterations = 10
    steps_per_iteration = 2048 # Default PPO buffer size
    eval_episodes = 5
    
    survival_history = []
    training_steps = []
    
    print(f"Starting training loop: {iterations} iterations of {steps_per_iteration} steps.")
    
    # Initial evaluation
    avg_survival = evaluate_survival(runner, episodes=eval_episodes)
    survival_history.append(avg_survival)
    training_steps.append(0)
    print(f"Step 0: Avg Survival = {avg_survival:.2f}")
    
    for i in range(1, iterations + 1):
        # Train
        runner.model.learn(total_timesteps=steps_per_iteration, reset_num_timesteps=False)
        
        # Evaluate
        avg_survival = evaluate_survival(runner, episodes=eval_episodes)
        
        # Record
        current_step = i * steps_per_iteration
        survival_history.append(avg_survival)
        training_steps.append(current_step)
        
        print(f"Step {current_step}: Avg Survival = {avg_survival:.2f}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(training_steps, survival_history, marker='o', linestyle='-', color='b')
        plt.title(f"RL Agent Survival Time ({settings.world_size}x{settings.world_size} Grid)")
        plt.xlabel("Training Steps")
        plt.ylabel("Average Survival Time (Steps)")
        plt.grid(True)
        
        plot_path = os.path.join(settings.folder, "survival_plot.png")
        plt.savefig(plot_path)
        plt.close()
        
    print(f"Training complete. Plot saved to {plot_path}")

if __name__ == "__main__":
    train_and_plot()
