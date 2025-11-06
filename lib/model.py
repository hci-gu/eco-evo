from enum import Enum
import numpy as np
import lib.config.const as const

HIDDEN_SIZE = 48
OUTPUT_SIZE = 5

MODEL_OFFSETS = {
    "terrain": {
        "land": 0,
        "water": 1,
        "out_of_bounds": 2,
    },
    # from to for the types
    "terrain_range": [0, 2],

}

for i, species in enumerate(const.SPECIES):
    print(i, species)
    biomass_start = 3
    energy_start = biomass_start + len(const.SPECIES)
    smell_start = energy_start + len(const.SPECIES)
    MODEL_OFFSETS[species] = {
        "biomass": 3 + i,
        "energy": energy_start + i,
        "smell": smell_start + i,
    }

MODEL_OFFSETS["biomass_range"] = [
    3, 3 + len(const.SPECIES) - 1
]
MODEL_OFFSETS["energy_range"] = [
    3 + len(const.SPECIES), 3 + len(const.SPECIES) * 2 - 1
]
MODEL_OFFSETS["smell_range"] = [
    3 + len(const.SPECIES) * 2, 3 + len(const.SPECIES) * 3 - 1
]

SINGLE_CELL_INPUT = 3 + len(const.SPECIES) * 3
INPUT_SIZE = SINGLE_CELL_INPUT * 9  # 3x3 grid of cells

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    EAT = 4

class Model:
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, chromosome=None):
        if chromosome is not None:
            self.set_weights(chromosome)
        else:
            # Random initialization of weights and biases.
            self.fc1_weight = np.ascontiguousarray(np.random.randn(input_size, hidden_size).astype(np.float32))
            self.fc1_bias = np.ascontiguousarray(np.random.randn(hidden_size).astype(np.float32))
            self.fc2_weight = np.ascontiguousarray(np.random.randn(hidden_size, output_size).astype(np.float32))
            self.fc2_bias = np.ascontiguousarray(np.random.randn(output_size).astype(np.float32))
    
    def forward(self, x):
        h = np.dot(x, self.fc1_weight) + self.fc1_bias
        h = np.maximum(0, h)

        out = np.dot(h, self.fc2_weight) + self.fc2_bias

        out_max = np.max(out, axis=1, keepdims=True)
        exp_vals = np.exp(out - out_max)
        softmax_output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        return softmax_output
    
    def hardcoded_single_action(self, x):
        """
        A hardcoded action function for testing purposes.
        Always returns the action '2' (which could represent 'stay still').
        """
        batch_size = x.shape[0]
        actions = np.full((batch_size, OUTPUT_SIZE), 0.0, dtype=np.float32)
        actions[:, Action.UP.value] = 0.9  # Set action '2' to have probability 1
        actions[:, Action.DOWN.value] = 0.1  # Set action '2' to have probability 1
        return actions

    def set_weights(self, chromosome):
        self.fc1_weight = np.ascontiguousarray(chromosome["fc1_weight"], dtype=np.float32)
        self.fc1_bias = np.ascontiguousarray(chromosome["fc1_bias"], dtype=np.float32)
        self.fc2_weight = np.ascontiguousarray(chromosome["fc2_weight"], dtype=np.float32)
        self.fc2_bias = np.ascontiguousarray(chromosome["fc2_bias"], dtype=np.float32)

    def state_dict(self):
        """
        Returns the model's parameters as a dictionary.
        """
        return {
            "fc1_weight": self.fc1_weight,
            "fc1_bias": self.fc1_bias,
            "fc2_weight": self.fc2_weight,
            "fc2_bias": self.fc2_bias,
        }

    def save(self, path):
        """
        Save the model's parameters to a file.
        """
        np.savez(path, **self.state_dict())