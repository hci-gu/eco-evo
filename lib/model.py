import numpy as np

class Model:
    def __init__(self, input_size=135, hidden_size=64, output_size=5, chromosome=None):
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