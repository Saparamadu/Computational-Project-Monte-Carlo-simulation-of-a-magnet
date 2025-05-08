# Author: Lakindu
# Date: 06/02/2025
# Description: Simulation of a 2D Ising model to calculate energy, magnetisation, and their fluctuations
#              across a temperature range using the Metropolis Monte Carlo algorithm.
# Inspirations from: https://rajeshrinet.github.io/blog/2014/ising-model/

import numpy as np
import math
from tqdm import tqdm

def metropolis_monte_carlo(configuration, ratio):
    """
    Perform one Metropolis Monte Carlo update on the 2D lattice.

    Parameters:
    - configuration (2D np.array): Spin configuration (+1 or -1)
    - ratio (float): J / kT, where J is the interaction constant (set to 1)
    """
    x, y = configuration.shape
    i, j = np.random.randint(0, x), np.random.randint(0, y)
    S = configuration[i, j]

    # Sum of 4 nearest neighbors with periodic boundary conditions
    neighbors = (
        configuration[(i + 1) % x, j] +
        configuration[i, (j + 1) % y] +
        configuration[(i - 1) % x, j] +
        configuration[i, (j - 1) % y]
    )

    delta_E = 2 * S * neighbors

    # Metropolis acceptance criterion
    if delta_E < 0 or np.random.rand() < math.exp(-delta_E * ratio):
        configuration[i, j] *= -1  # Flip spin


def energy_calculation(configuration):
    """
    Compute the total energy of the configuration.

    Parameters:
    - configuration (2D np.array): Spin configuration

    Returns:
    - float: Total energy of the system
    """
    neighbors = (
        np.roll(configuration, shift=1, axis=0) +
        np.roll(configuration, shift=-1, axis=0) +
        np.roll(configuration, shift=1, axis=1) +
        np.roll(configuration, shift=-1, axis=1)
    )
    energy = -np.sum(configuration * neighbors) / 4.0  # Avoid double counting
    return energy


def magnetisation_calculation(configuration):
    """
    Compute the total magnetisation of the system.

    Parameters:
    - configuration (2D np.array): Spin configuration

    Returns:
    - float: Total magnetisation
    """
    return np.sum(configuration)


# -------------------- Simulation Parameters --------------------
L = 5                       # Lattice size (LxL)
n = 10**6                  # Total Monte Carlo steps per temperature
steps = 161                # Number of temperature points
end_temperature = 2.6      # Final temperature
temperature = np.linspace(1, end_temperature, steps)
ratio_values = 1 / temperature  # J/kT
spins = [-1, 1]            # Possible spin values
measurement_interval = 1   # Interval between measurements
measurements_per_temp = n // measurement_interval

# -------------------- Storage Arrays --------------------
configuration_energy = np.zeros(steps)
configuration_magnetisation = np.zeros(steps)
configuration_energy_fluctuations = np.zeros(steps)
configuration_magnetisation_fluctuations = np.zeros(steps)


# -------------------- Monte Carlo Simulation --------------------
for i, ratio in tqdm(list(enumerate(ratio_values)), total=steps, desc="Simulating..."):
    E1 = M1 = E2 = M2 = 0  # Reset accumulators for this temperature
    configuration = np.random.choice(spins, size=(L, L))  # Fresh random spin configuration

    for j in range(n):
        metropolis_monte_carlo(configuration, ratio)

        # Measurement step
        if j % measurement_interval == 0:
            E0 = energy_calculation(configuration)
            M0 = magnetisation_calculation(configuration)

            E1 += E0
            M1 += M0
            E2 += E0 ** 2
            M2 += M0 ** 2

    # Normalize measurements and compute fluctuations
    norm = measurements_per_temp * L**2
    configuration_energy[i] = E1 / norm
    configuration_magnetisation[i] = M1 / norm

    configuration_energy_fluctuations[i] = ((E2 / norm) - (E1 / (measurements_per_temp * L))**2) * ratio**2
    configuration_magnetisation_fluctuations[i] = ((M2 / norm) - (M1 / (measurements_per_temp * L))**2) * ratio


# -------------------- Save Results to File --------------------
filename = f"Lattice{L}Data.dat"
combined_data = np.column_stack((
    temperature,
    configuration_energy,
    configuration_magnetisation,
    configuration_energy_fluctuations,
    configuration_magnetisation_fluctuations
))

header = (
    "Temperature  Energy_per_Spin  Magnetisation  Energy_Fluctuations  "
    "Magnetisation_Fluctuations\n"
    f"Lattice Size: {L}x{L}, MC Steps: {n}, Measurement Interval: {measurement_interval}"
)

np.savetxt(filename, combined_data, header=header, delimiter='  ', fmt='%0.4f')
