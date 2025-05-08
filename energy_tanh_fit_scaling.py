# File: energy_tanh_fit_scaling.py
# Author: Lakindu
# Date: 06/02/2025
# Description: This script estimates the critical temperature Tc from the energy-temperature curve
#              by fitting a hyperbolic tangent function. Tc is extracted from the inflection point
#              for each lattice size L, and a finite-size scaling fit is used to extrapolate Tc(∞).

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------- Fit Model for Energy --------------------

def tanh_energy(T, a, b, c, d):
    """
    Model energy per spin as a hyperbolic tangent function:
    E(T) = a * tanh(b*T + c) + d

    Inflection point (i.e., phase transition) occurs at T = -c/b.
    """
    return a * np.tanh(b * T + c) + d

# -------------------- Finite-Size Scaling Function --------------------

def scaling_function(L, Tc_inf, a, b):
    """
    Finite-size scaling function:
    Tc(L) = Tc_inf + a / L^b

    Parameters:
    - L : Lattice size
    - Tc_inf : Infinite-lattice critical temperature
    - a, b : Fit parameters
    """
    return Tc_inf + a / (L ** b)

# -------------------- Main Analysis --------------------

# Define lattice sizes
L_values = np.array([5, 10, 20, 30])
Tc_values = []      # Critical temperatures for each L
Tc_errors = []      # Associated uncertainties

plt.figure(figsize=(8, 5))

for L in L_values:
    filename = f"Lattice{L}Data1.dat"
    data = np.loadtxt(filename)
    
    # Extract data (skip initial points to avoid noise)
    T = data[30:, 0]
    E = data[30:, 1]
    
    # Fit the tanh model to the energy curve
    popt, pcov = curve_fit(tanh_energy, T, E, p0=[1, 2, -4, -1])
    a, b, c, d = popt
    
    # Estimate Tc from the inflection point of the tanh curve
    Tc = -c / b
    error_Tc = (1 / b) * np.sqrt((pcov[2, 2]) + (((c/b)**2) * pcov[1, 1]))
    
    # Store results
    Tc_values.append(Tc)
    Tc_errors.append(error_Tc)
    
    # Plot fitted curve
    plt.plot(T, tanh_energy(T, *popt), label=f"Lattice Size = {L} × {L}")
    print(f"L = {L}, Estimated Tc = {Tc:.3f} ± {error_Tc:.3f}")

plt.xlabel("Temperature")
plt.ylabel("Energy per Spin (J)")
plt.title("Tanh Fit to Energy Curves")
plt.xticks()
plt.yticks()
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------- Tc Extrapolation via Finite-Size Scaling --------------------

Tc_values = np.array(Tc_values)
Tc_errors = np.array(Tc_errors)

# Fit finite-size scaling law
params, pcov2 = curve_fit(
    scaling_function,
    L_values,
    Tc_values,
    sigma=Tc_errors,
    absolute_sigma=True
)

Tc_inf, a_fit, b_fit = params
Tc_inf_err = np.sqrt(pcov2[0, 0])

print(f"\nExtrapolated Critical Temperature (Tc_inf): {Tc_inf:.4f} ± {Tc_inf_err:.4f}")
print(f"Fit Parameters: a = {a_fit:.4f}, b = {b_fit:.4f}")

# -------------------- Plot Tc(L) Scaling --------------------

L_fit = np.linspace(5, 50, 100)
Tc_fit_curve = scaling_function(L_fit, *params)

plt.errorbar(L_values, Tc_values, yerr=Tc_errors, fmt='o', capsize=3, label="Simulation Data")
plt.plot(L_fit, Tc_fit_curve, label=f"Fit: Tc(L) = {Tc_inf:.2f} + {a_fit:.3f}/L^{b_fit:.3f}")
plt.xlabel("Linear Lattice Size (L)")
plt.ylabel("Critical Temperature (Tc)")
plt.title("Finite-Size Scaling of Critical Temperature from Energy Fit")
plt.xticks(L_values)
plt.yticks()
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
