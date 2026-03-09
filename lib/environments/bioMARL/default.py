import numpy as np

DEFAULT = {
    "n_fg": 4,
    "fg_init_k": [100, 20, 20, 4],  # Initial max density.
    "padding": 1,
    "grid_shape": [10, 10],
    "seed": None,
    "action_sigma_max": 6,
    "energy_in_obs": True,
    "fg_k": [100, np.inf, np.inf, np.inf],
    "reproduction_freq": [1, 5, 5, 5],
    "growth_rates": [1.4, 1.2, 1.2, 1.01],
    "feeding_energy_reward": [[0, 0, 0, 0], [3, 0, 0, 0], [2, 0, 0, 0], [1, 10, 5, 0]], 
    
    "base_energy_cost": [0, 1, 1, 1.5],
    "idling_energy_cost": [0, 0, 0, 0],  # cost of not eating
    "hiding_energy_cost": [1, 2, 2, 2],
    "migration_energy_cost": [0, 1, 1, 3],
    "mask": None,
}

# import yaml
# def custom_dump(data):
#     def represent_inline_list(dumper, value):
#         return dumper.represent_sequence('tag:yaml.org,2002:seq', value, flow_style=True)

#     # Override PyYAML's default list representation
#     yaml.add_representer(list, represent_inline_list)

#     return yaml.dump(data, f, default_flow_style=False)
# with open("config.yaml","w") as f:
#     # yaml.dump(DEFAULT, f, allow_unicode=True,default_flow_style=False)
#     custom_dump(DEFAULT)

# with open("config.yaml","r") as f:
#     settings = yaml.load(f, Loader=yaml.FullLoader)
#     for key, value in settings.items():
#         print(key, value)