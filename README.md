# Computational Project: Monte Carlo Simulation of a Magnet (2D Ising Model)

This repository contains Python code and data for a computational physics project simulating the two-dimensional Ising model using the **Metropolis Monte Carlo algorithm**. The project explores the **ferromagnetic–paramagnetic phase transition**, investigates **finite-size effects**, and estimates the **critical temperature \( T_c \)** in the thermodynamic limit.

## 🔬 Project Overview

The Ising model is a classical lattice model of interacting magnetic spins, used extensively in statistical physics to study phase transitions. In this project, we:

- Simulate square lattices of spins (\( s_i = \pm 1 \)) with **periodic boundary conditions**
- Compute energy per spin, magnetisation, magnetic susceptibility, and their fluctuations
- Estimate finite-size critical temperatures using:
  - **Hyperbolic tangent fit** to the energy–temperature curve
  - **Scaled Lorentzian fit** to the magnetic susceptibility peak
- Apply **finite-size scaling** to extrapolate the infinite-lattice critical temperature \( T_c(\infty) \)
- Visualise spin configurations during evolution at different temperatures and Monte Carlo steps

## 📊 Key Results

- Estimated \( T_c(\infty) \approx 2.301 \pm 0.006 \), in agreement with Onsager’s exact solution
- Visualisation of phase transition dynamics for a \( 50 \times 50 \) lattice
- Identified limitations of the Metropolis algorithm near criticality and suggested **cluster algorithms** (e.g. Wolff) for future improvement

## 📁 Repository Structure

```plaintext
├── Magnet.py  # Ising model with Metropolis Monte Carlo algorithm. Save the data of Energy per spin, Magnetisation, Heat Capacity and Magnetic Susceptibility to a file
├── energy_tanh_fit_scaling.py  # Energy per spin curve fitting and scaling
├── scaled_lorentzian_fit.py    # Magnetic Susceptibility curve fitting and scaling
├── timestamp.py                # Snapshots of the Monte Carlo steps 
├── README.md
