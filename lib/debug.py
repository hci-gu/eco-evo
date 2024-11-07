
import lib.constants as const
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize the figure and axis outside the function
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(np.zeros((const.WORLD_SIZE, const.WORLD_SIZE)), cmap='gray', origin='upper')
scatter = ax.scatter([], [], facecolors='none', edgecolors='red', s=100)
ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
ax.set_xticks(np.arange(-0.5, const.WORLD_SIZE, 1))
ax.set_yticks(np.arange(-0.5, const.WORLD_SIZE, 1))
ax.set_xlim([-0.5, const.WORLD_SIZE - 0.5])
ax.set_ylim([const.WORLD_SIZE - 0.5, -0.5])
legend = ax.legend(loc='upper right')
title = ax.set_title('Selected Cells')

def visualize_selection(world, selected_positions, species_index=None):
    # Convert world tensor to NumPy array
    world_np = world[1:-1, 1:-1].cpu().numpy()
    image = np.sum(world_np[:, :, 1:], axis=2)  # Adjust as needed
    image = (image - image.min()) / (image.max() - image.min() + 1e-5)

    im.set_data(image)

    # Update scatter data
    selected_positions_np = selected_positions.cpu().numpy() - 1
    x_coords = selected_positions_np[:, 1]
    y_coords = selected_positions_np[:, 0]
    scatter.set_offsets(np.c_[x_coords, y_coords])

    # Update title
    title.set_text(f'Selected Cells for Species {species_index}' if species_index is not None else 'Selected Cells')

    plt.pause(0.001)  # Brief pause to update the plot