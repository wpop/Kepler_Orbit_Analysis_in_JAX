"""
kepler_law3.py

Analyze and verify Kepler's Third Law from simulated or measured orbital trajectory data.

This module estimates the orbital period and semi-major axis for one or more
orbits, then checks whether the relation

    T^2 ∝ a^3

holds approximately.

Main goals
----------
1. Estimate the orbital period T of each orbit.
2. Estimate the semi-major axis a of each orbit.
3. Compare T^2 and a^3 across multiple trajectories.
4. Produce numerical diagnostics and optional plots.

Typical workflow
----------------
- Input trajectory data (2D or projected orbital-plane coordinates).
- Detect repeated orbital motion to estimate period.
- Fit or estimate ellipse size to obtain semi-major axis.
- Evaluate the Kepler-3 relation using ratios or regression.

Notes
-----
- This file is intended for educational / computational physics analysis.
- The implementation may use approximate numerical methods depending on
  sampling quality and orbit completeness.
- For best results, trajectories should cover at least one full orbit.

Author: William Popkov
Date: 2026-01-25
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp

# Define file paths for planetary trajectory data
PLANET_FILES = {
    "Mercury": "mercury_orbit.csv",
    "Venus":   "venus_orbit.csv",
    "Earth":   "earth_orbit.csv",
    "Mars":    "mars_orbit.csv",
    "Jupiter": "jupiter_orbit.csv",
    "Saturn":  "saturn_orbit.csv",
}

# Known orbital periods in Earth years for the planets in our solar system
KNOWN_PERIODS_YEARS = {
    "Mercury": 0.2408467,
    "Venus":   0.61519726,
    "Earth":   1.0000174,
    "Mars":    1.8808476,
    "Jupiter": 11.862615,
    "Saturn":  29.447498,
}


def load_positions(filename):
    """Load 3D orbital positions from a CSV file."""
    df = pd.read_csv(filename)
    R = df[["x", "y", "z"]].values.astype(np.float32)
    return jnp.array(R)


def project_to_orbital_plane(R):
    """Project 3D positions onto the orbital plane."""
    U, S, Vt = jnp.linalg.svd(R, full_matrices=False)
    plane_basis = Vt[:2]
    R2 = R @ plane_basis.T
    return R2


def estimate_semi_major_axis(R2):
    """
    Estimate the semi-major axis of a 2D orbit.

    Parameters
    ----------
    R2 : jax.numpy.ndarray
        Array of shape (N, 2) containing 2D orbital positions.

    Returns
    -------
    a : float
        Estimated semi-major axis of the orbit.
    """
    R2_centered = R2 - jnp.mean(R2, axis=0)  # Compute the geometric center of the orbit
    Cov2 = jnp.cov(R2_centered.T)            # Compute the covariance matrix of the centered positions
    eigvals, eigvecs = jnp.linalg.eigh(Cov2) # Eigen-decomposition to find principal axes

     # Project the orbit points onto the principal axes
    proj = R2_centered @ eigvecs             
    u = np.array(proj[:, 1])
    v = np.array(proj[:, 0])

     # Estimate semi-axis lengths from the projected coordinate ranges
    a_est = 0.5 * (u.max() - u.min())
    b_est = 0.5 * (v.max() - v.min())

    # Return the larger semi-axis as the semi-major axis
    return max(a_est, b_est)


def analyze_law3(save_plots=True):
    """
    Analyze Kepler's Third Law for a set of planets.

    This function loads planetary trajectory data, projects each orbit onto its
    orbital plane, estimates the semi-major axis, and compares orbital periods
    against the Kepler-3 relation T^2 ∝ a^3.

    Parameters
    ----------
    save_plots : bool, optional
        If True, save the generated plots to disk. Default is True.

    Returns
    -------
    results : dict
        Dictionary containing orbital parameters, Kepler ratios, and fit diagnostics.
    """
    planets = []
    a_values = []
    T_values = []

    for planet, filename in PLANET_FILES.items():
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing file: {filename}")

        R = load_positions(filename)         # Load 3D orbital positions from file
        R2 = project_to_orbital_plane(R)     # Project positions onto the orbital plane
        a_est = estimate_semi_major_axis(R2) # Estimate the semi-major axis from the projected orbit

        T = KNOWN_PERIODS_YEARS[planet]      # The known orbital period (in years) for this planet

        planets.append(planet)
        a_values.append(a_est)
        T_values.append(T)

    a_values = np.array(a_values)
    T_values = np.array(T_values)

    # Compute Kepler-3 transformed variables
    T2 = T_values**2
    a3 = a_values**3
    ratios = T2 / a3

    # Compute the Kepler ratio T^2 / a^3 for each planet
    ratio_mean = ratios.mean()
    ratio_std = ratios.std()
    ratio_rel_std = 100 * ratio_std / ratio_mean

    # Plot 1
    plt.figure(figsize=(8, 6))
    plt.scatter(a3, T2, s=110, label="Planets")

    for i, p in enumerate(planets):
        plt.annotate(p, (a3[i], T2[i]), textcoords="offset points", xytext=(6, 6))

    coef = np.polyfit(a3, T2, 1)
    xline = np.linspace(a3.min()*0.9, a3.max()*1.05, 300)
    yline = coef[0]*xline + coef[1]
    plt.plot(xline, yline, lw=2.2, label=f"Linear fit: y={coef[0]:.3f}x + {coef[1]:.3f}")

    plt.xlabel(r"$a^3$ (AU$^3$)")
    plt.ylabel(r"$T^2$ (yr$^2$)")
    plt.title("Kepler's Third Law: $T^2$ vs $a^3$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig("law3_T2_vs_a3.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 2
    log_a = np.log(a_values)
    log_T = np.log(T_values)

    slope, intercept = np.polyfit(log_a, log_T, 1)
    xfit = np.linspace(log_a.min()*0.95, log_a.max()*1.05, 300)
    yfit = slope * xfit + intercept

    plt.figure(figsize=(8, 6))
    plt.scatter(log_a, log_T, s=110, label="Planets")

    for i, p in enumerate(planets):
        plt.annotate(p, (log_a[i], log_T[i]), textcoords="offset points", xytext=(6, 6))

    plt.plot(xfit, yfit, lw=2.2, label=f"Fit slope = {slope:.4f}")
    plt.xlabel(r"$\log(a)$")
    plt.ylabel(r"$\log(T)$")
    plt.title("Log-Log Verification of Kepler's Third Law")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig("law3_loglog.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Return all key numerical diagnostics in a structured format
    return {
        "planets": planets,
        "a_values": a_values,
        "T_values": T_values,
        "ratios": ratios,
        "ratio_mean": float(ratio_mean),
        "ratio_std": float(ratio_std),
        "ratio_rel_std_percent": float(ratio_rel_std),
        "slope": float(slope),
        "slope_abs_error": float(abs(slope - 1.5)),
    }


def main():
    result = analyze_law3(save_plots=True)

    print("\n=== Kepler's Third Law Summary ===")
    print(f"Mean T^2/a^3 = {result['ratio_mean']}")
    print(f"Std T^2/a^3 = {result['ratio_std']}")
    print(f"Relative std (%) = {result['ratio_rel_std_percent']}")
    print(f"Log-log slope = {result['slope']}")
    print(f"Slope absolute error = {result['slope_abs_error']}")

if __name__ == "__main__":
    main()
