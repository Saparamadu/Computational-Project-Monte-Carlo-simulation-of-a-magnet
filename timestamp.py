# Author: Lakindu
# Date: 06/02/2025
# Description: Simulation of a 2D Ising model to calculate energy, magnetisation, and their fluctuations
#              across a temperature range using the Metropolis Monte Carlo algorithm.
# Inspirations from: https://rajeshrinet.github.io/blog/2014/ising-model/

import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

def metropolis_monte_carlo(configuration, ratio):
    """
    Perform a single Metropolis Monte Carlo step on a 2D Ising spin configuration.

    Parameters:
    - configuration (2D np.array): The lattice of spins (+1 or -1)
    - ratio (float): J / kT, where J is interaction strength and T is temperature
    """
    x, y = configuration.shape
    i, j = np.random.randint(0, x), np.random.randint(0, y)  # Random site
    S = configuration[i, j]  # Current spin

    # Sum of 4 nearest neighbors (periodic boundaries)
    neighbors = (
        configuration[(i + 1) % x, j] +
        configuration[i, (j + 1) % y] +
        configuration[(i - 1) % x, j] +
        configuration[i, (j - 1) % y]
    )

    # Energy change if spin is flipped
    delta_E = 2 * S * neighbors

    # Metropolis criterion
    if delta_E < 0 or np.random.rand() < math.exp(-delta_E * ratio):
        configuration[i, j] *= -1  # Flip spin


def configPlot(fig, config, step, L, index):
    """
    Plot a single spin configuration on the subplot grid.

    Parameters:
    - fig (matplotlib Figure): Main figure object
    - config (2D np.array): Spin configuration
    - step (int): Iteration count
    - L (int): Lattice size
    - index (int): Subplot index
    """
    X, Y = np.meshgrid(range(L), range(L))
    ax = fig.add_subplot(3, 3, index)
    ax.set_xticks([])
    ax.set_yticks([])
    mesh = ax.pcolormesh(X, Y, config, cmap=plt.cm.bwr, shading='auto', vmin=-1, vmax=1)
    ax.set_title(f'Iteration = {step}', fontsize=12)
    ax.axis('tight')


# -------------------- Simulation Parameters --------------------
L = 50  # Lattice size (LxL)
n_steps = 10**7  # Total number of Monte Carlo steps
temperature = 0.3  # Simulation temperature
J_over_kT = 1 / temperature  # Ratio of coupling strength to thermal energy
spins = [-1, 1]  # Possible spin values
configuration = np.random.choice(spins, size=(L, L))  # Initial random configuration

# Iteration steps at which to capture snapshots
snapshots = [10, 100, 10_000, 100_000, 10_000_000]
snapshot_set = set(snapshots)

# -------------------- Plot Setup --------------------
fig = plt.figure(figsize=(15, 15), dpi=100)
configPlot(fig, configuration, 0, L, 1)  # Plot initial configuration
plot_index = 2  # Track subplot index

# -------------------- Monte Carlo Loop --------------------
print("Running Metropolis Monte Carlo simulation...")
for step in tqdm(range(1, n_steps + 1)):
    metropolis_monte_carlo(configuration, J_over_kT)

    # Capture configuration at specified steps
    if step in snapshot_set:
        configPlot(fig, configuration, step, L, plot_index)
        plot_index += 1

# -------------------- Finalize Plot --------------------
plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.07, wspace=0.2, hspace=0.3)
plt.suptitle(f"Metropolis Simulation: Ising Model on {L}×{L} Lattice at T = {temperature}", fontsize=13)
plt.show()
