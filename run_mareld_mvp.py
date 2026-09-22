import matplotlib.pyplot as plt
from lib.config.config_loader import library_tick_hours, setup_full_mareld_mvp
from lib.environments.ecosystem import EcosystemEnvironment
from lib.world.tick_time import tick_label

# The MVP runs off the library alone (no project file), so it adopts the
# tick length the library's numbers are calibrated at. Section 97.
TICK_HOURS = library_tick_hours('fgconfig/fg_library.yaml')

def main():
    print("Setting up Mareld Full MVP...")
    fgs = setup_full_mareld_mvp()

    grid_config = {
        'width': 60,
        'height': 60,
        'cell_size': 1000.0,
        'tick_duration': float(TICK_HOURS)
    }
    
    env = EcosystemEnvironment(grid_config, fgs)
    
    # Load optional context maps. The current environment transition does
    # not consume pressure maps, so only neutral spatial context is loaded.
    try:
        env.grid.load_map_from_png('djup', 'djup.png', scale=100.0)  # Depth in meters.
    except Exception as e:
        print(f"Warning: Could not load some maps: {e}")
    
    history = {fid: [] for fid in env.fgs}
    
    print("\nRunning simulation for 100 ticks (25 days)...")
    for t in range(100):
        if t % 10 == 0:
            print(f"Tick {t}...")
        observation = env.get_observation()
        actions = env.policy_controller.forward(observation)
        env.step(actions)
        for fid, fg in env.fgs.items():
            history[fid].append(fg.biomass.sum())
            
    print("\nSimulation finished.")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for fid, h in history.items():
        plt.plot(h, label=fid)
    plt.xlabel(f'Ticks ({tick_label(TICK_HOURS)})')
    plt.ylabel('Total Biomass (tons)')
    plt.title('Mareld Mini MVP Simulation')
    plt.legend()
    plt.yscale('log')
    plt.savefig('results/mvp_biomass_log.png')
    plt.show() # Note: show() might not work in this environment, but savefig will.

if __name__ == "__main__":
    main()
