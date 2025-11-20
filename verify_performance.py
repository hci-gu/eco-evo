from lib.config.settings import Settings
from lib.runners.rl_runner import RLRunner
import os
import numpy as np

def evaluate_runner(runner, episodes=5):
    total_rewards = []
    for _ in range(episodes):
        obs = runner.env.reset()
        episode_reward = 0
        done = False
        # We need to track when the *episode* is done, not just individual agents
        # But VecEnv auto-resets.
        # So we run for a fixed number of steps or until we detect a reset?
        # Let's run for a fixed number of steps that is likely an episode length
        # Or better, track the rewards returned by step()
        
        # Since we are training one species, the reward is global for that species.
        # We can just sum up the rewards.
        # But VecEnv returns an array of rewards (one per env).
        # In our wrapper, we broadcast the global reward to all envs.
        # So we just take the first one.
        
        # Let's run for 100 steps per episode
        for _ in range(100):
            action, _ = runner.model.predict(obs)
            obs, rewards, dones, infos = runner.env.step(action)
            episode_reward += rewards[0] # Take reward from first env (they are all same)
            if any(dones):
                break
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def verify_performance():
    print("Setting up verification test...")
    settings = Settings(
        world_size=10,
        num_agents=1,
        max_steps=100,
        folder="results/verify_rl"
    )
    
    if not os.path.exists(settings.folder):
        os.makedirs(settings.folder)

    # 1. Initialize untrained agent
    runner = RLRunner(settings, species="cod", render_mode="none")
    
    print("Evaluating untrained agent...")
    initial_performance = evaluate_runner(runner, episodes=10)
    print(f"Untrained Performance (Avg Reward): {initial_performance:.4f}")
    
    # 2. Train
    print("Training for 20,000 steps...")
    runner.train(total_timesteps=20000)
    
    # 3. Evaluate trained agent
    print("Evaluating trained agent...")
    trained_performance = evaluate_runner(runner, episodes=10)
    print(f"Trained Performance (Avg Reward): {trained_performance:.4f}")
    
    # 4. Compare
    improvement = trained_performance - initial_performance
    print(f"Improvement: {improvement:.4f}")
    
    if improvement > 0:
        print("SUCCESS: Model performance improved.")
    else:
        print("WARNING: Model performance did not improve (might need more training or tuning).")

if __name__ == "__main__":
    verify_performance()
