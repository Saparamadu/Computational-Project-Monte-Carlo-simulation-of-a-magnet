# File: scaled_lorentzian_fit.py
# Author: Lakindu
# Date: 06/02/2025
# Description: This script fits a scaled Lorentzian function to the magnetic susceptibility peaks
#              of 2D Ising model simulations for various lattice sizes (L), and extrapolates the
#              critical temperature in the thermodynamic limit (L → ∞) using finite-size scaling.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------- Lorentzian Model --------------------

def scaled_lorentzian(T, A, Tc, gamma0, offset, L):
    """
    Lorentzian function scaled by lattice size L.

    Parameters:
    - T : Temperature
    - A : Amplitude
    - Tc : Critical temperature (peak center)
    - gamma0 : Characteristic width scale
    - offset : Background level
    - L : Lattice size

    Returns:
    - Lorentzian curve values
    """
    gamma_L = gamma0 / L
    return A * gamma_L**2 / ((T - Tc)**2 + gamma_L**2) + offset

def wrapper_fit(T, A, Tc, gamma0, offset):
    """
    Wrapper to pass lattice size L via global context for fitting.

    Parameters: Same as above except L is external.
    """
    return scaled_lorentzian(T, A, Tc, gamma0, offset, wrapper_fit.L_current)

# -------------------- Finite-Size Scaling Function --------------------

def scaling_function(L, Tc_inf, a, b):
    """
    Scaling function to extrapolate Tc as L → ∞:
    Tc(L) = Tc_inf + a / L^(1/b)

    Parameters:
    - L : Lattice size
    - Tc_inf : Extrapolated infinite lattice critical temperature
    - a, b : Fitting constants

    Returns:
    - Tc(L) values
    """
    return Tc_inf + (a / (L**(1/b)))


# -------------------- Fitting Across Lattice Sizes --------------------

L_values = np.array([5, 10, 20, 30])  # Lattice sizes to analyze
Tc_values, Tc_errors = [], []        # Store Tc and uncertainties

plt.figure(figsize=(8, 5))

for L in L_values:
    # Load susceptibility data: columns [T, Energy, Mag, E-fluct, M-fluct]
    data = np.loadtxt(f"Lattice{L}Data1.dat")
    T, chi = data[30:, 0], data[30:, 4]  # Skip low-T data (transient)

    # Estimate peak and define fitting window around it
    peak_idx = np.argmax(chi)
    T_peak = T[peak_idx]
    mask = (T > T_peak - 0.5) & (T < T_peak + 0.5)
    T_fit, chi_fit = T[mask], chi[mask]

    # Store lattice size globally for wrapper
    wrapper_fit.L_current = L

    # Fit Lorentzian to the peak region
    popt, pcov = curve_fit(
        wrapper_fit, T_fit, chi_fit,
        p0=[max(chi_fit), T_peak, 1.0, min(chi_fit)],  # Initial guess
        bounds=([0, T_peak - 0.5, 0, -np.inf], [np.inf, T_peak + 0.5, np.inf, np.inf])
    )

    # Extract fitted parameters
    A, Tc, gamma0, offset = popt
    Tc_err = np.sqrt(pcov[1, 1])  # Standard error on Tc

    print(f"L = {L}: Tc = {Tc:.3f} ± {Tc_err:.3f}")
    Tc_values.append(Tc)
    Tc_errors.append(Tc_err)

    # Plot fit
    plt.plot(T_fit, wrapper_fit(T_fit, *popt), label=f"Lattice Size = {L} × {L}")

# Plot fitted Lorentzians
plt.xlabel("Temperature")
plt.ylabel("Magnetic Susceptibility")
plt.title("Lorentzian Fit to Magnetic Susceptibility Peaks")
plt.xticks()
plt.yticks()
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------- Extrapolation of Tc (L → ∞) --------------------

Tc_values = np.array(Tc_values)
Tc_errors = np.array(Tc_errors)
valid = ~np.isnan(Tc_values)  # Filter any NaN fits

# Fit finite-size scaling law
params, pcov2 = curve_fit(
    scaling_function,
    L_values[valid], Tc_values[valid],
    sigma=Tc_errors[valid],
    absolute_sigma=True
)

Tc_inf, a, b = params
Tc_inf_err = np.sqrt(pcov2[0, 0])

print(f"\nExtrapolated Critical Temperature (Tc_inf): {Tc_inf:.6f} ± {Tc_inf_err:.6f}")

# -------------------- Plot Scaling Law --------------------

L_plot = np.linspace(5, 50, 100)
Tc_fit = scaling_function(L_plot, *params)

plt.errorbar(L_values, Tc_values, yerr=Tc_errors, fmt='o', label='Simulation Data')
plt.plot(L_plot, Tc_fit, label=f"Fit: $T_c(L) = {Tc_inf:.3f} + {a:.3f}/L^{{1/{b:.3f}}}$")
plt.xlabel("Lattice Size (L)")
plt.ylabel("Critical Temperature (Tc)")
plt.title("Finite-Size Scaling of Critical Temperature obtained using Magnetic Susceptibility")
plt.xticks(L_values)
plt.yticks()
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
