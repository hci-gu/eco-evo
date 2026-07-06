import numpy as np
from PIL import Image

class Grid:
    def __init__(self, width=60, height=60, cell_size=1000.0, tick_duration=6.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size  # meters
        self.tick_duration = tick_duration  # hours
        self.maps = {}

    def add_map(self, name, data):
        if data.shape != (self.height, self.width):
            raise ValueError(f"Map {name} has wrong shape: {data.shape}")
        self.maps[name] = data.astype(np.float64)

    def get_map(self, name):
        return self.maps.get(name)

    def load_map_from_png(self, name, path, scale=1.0, invert=False):
        img = Image.open(path).convert('L')
        data = np.array(img).astype(np.float64)
        if invert:
            data = 255.0 - data
        data = (data / 255.0) * scale
        self.add_map(name, data)
