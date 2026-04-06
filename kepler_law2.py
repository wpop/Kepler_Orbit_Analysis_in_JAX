"""
kepler_law2.py

Kepler's Second Law verification using Mars orbital data.
Main goal: Demonstrate that Mars sweeps out equal areas in equal times.
Main idea: Kepler's Second Law -> conservation of angular momentum

This script:
1. Loads Mars position and velocity data from a CSV file.
2. Projects the 3D orbit onto its best-fit orbital plane.
3. Computes swept areas between consecutive position vectors.
4. Computes angular momentum vectors and their magnitudes.
5. Visualizes:
   - the orbit with equal-time swept sectors,
   - swept area versus time step,
   - angular momentum magnitude versus time step.
6. Prints a short numerical summary showing whether equal areas
   are swept in equal times.

Author: William Popkov
Date: 2026-01-18
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import jit

FILENAME = "mars_orbit.csv"             # (position: x, y, z and velocity: vx, vy, vz).
SEGMENT_LENGTH = 30                     # Number of consecutive time steps used to form one swept-area segment
STARTS = [0, 150, 300, 500, 800, 1100]  # Starting indices for highlighted equal-time orbital segments.

@jit
def vector_norm(x):
    """
    Compute the Euclidean norm (magnitude) of a vector or an array of vectors.

    Parameters
    ----------
    x : jax.numpy.ndarray
        Input array with the last axis representing vector components.

    Returns
    -------
    jax.numpy.ndarray
        Euclidean norm of the input vector(s).
    """
    return jnp.sqrt(jnp.sum(x**2, axis=-1))

@jit
def compute_swept_areas(R):
    """
    Compute the area swept between consecutive position vectors.

    Each pair of neighboring position vectors forms a triangle with the Sun
    located at the origin. The area of each triangle is used as an approximation
    of the area swept during one time step.

    Parameters
    ----------
    R : jax.numpy.ndarray
        Array of shape (N, 3) containing 3D position vectors.

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (N-1,) containing swept areas for consecutive time steps.
    """
    r1 = R[:-1]
    r2 = R[1:]
    cross = jnp.cross(r1, r2)
    area = 0.5 * vector_norm(cross)
    return area


@jit
def compute_angular_momentum(R, V):
    """
    Compute the angular momentum vectors for an orbiting body.

    Angular momentum is computed at each time step as the cross product
    between position and velocity vectors.

    Parameters
    ----------
    R : jax.numpy.ndarray
        Array of shape (N, 3) containing 3D position vectors.
    V : jax.numpy.ndarray
        Array of shape (N, 3) containing 3D velocity vectors.

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (N, 3) containing angular momentum vectors.
    """
    return jnp.cross(R, V)


def project_to_orbital_plane(R):
    """
    Project 3D position vectors onto the best-fit orbital plane.

    The orbital plane is estimated using singular value decomposition (SVD).
    The first two principal directions define a 2D basis for the orbit, and
    all 3D position vectors are projected onto that basis.

    Parameters
    ----------
    R : jax.numpy.ndarray
        Array of shape (N, 3) containing 3D position vectors.

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (N, 2) containing 2D projected coordinates in the
        orbital plane.
    """
    U, S, Vt = jnp.linalg.svd(R, full_matrices=False)
    plane_basis = Vt[:2]   # Take the first two principal directions as the orbital plane basis
    R2 = R @ plane_basis.T # Project onto the 2D orbital plane
    return R2


def analyze_law2(save_plots=True):
    """
    Analyze Kepler's Second Law using Mars orbital data.

    This function loads orbital position and velocity data, projects the orbit
    onto its best-fit plane, computes swept areas and angular momentum, and
    evaluates how constant these quantities remain over equal time steps.

    Parameters
    ----------
    save_plots : bool, optional
        If True, save generated plots as PNG files. Default is True.

    Returns
    -------
    dict
        Dictionary containing summary statistics for swept area variation
        and angular momentum variation.
    """
    df = pd.read_csv(FILENAME)

    # Convert position and velocity columns to JAX arrays for numerical computation.
    R = jnp.array(df[["x", "y", "z"]].values.astype(np.float32))
    V = jnp.array(df[["vx", "vy", "vz"]].values.astype(np.float32))

    # Project the 3D orbit onto its best-fit 2D orbital plane.
    R2 = np.array(project_to_orbital_plane(R))

    areas = np.array(compute_swept_areas(R)) # Compute swept areas between consecutive position vectors.
    L = compute_angular_momentum(R, V)       # Compute angular momentum vectors at each time step.
    Lmag = np.array(vector_norm(L))          # Compute the magnitude of angular momentum at each time step.

    # Compute summary statistics for swept area consistency.
    area_mean = areas.mean()
    area_std = areas.std()
    area_rel = 100 * area_std / area_mean
    
    # Compute relative variation of angular momentum magnitude.
    L_rel = 100 * Lmag.std() / Lmag.mean()

    # Compute radial distance from the Sun in the 2D orbital plane.
    r_norm = np.sqrt(np.sum(R2**2, axis=1))
    peri_idx = np.argmin(r_norm)
    aphe_idx = np.argmax(r_norm)
    peri = R2[peri_idx] # Position of perihelion (closest approach to the Sun)
    aphe = R2[aphe_idx] # Position of aphelion (farthest distance from the Sun)

    # Plot 1
    plt.figure(figsize=(9, 9))
    plt.plot(R2[:, 0], R2[:, 1], lw=2.5, label="Mars orbit")
    plt.scatter([0], [0], s=350, marker='o', label="Sun")
    plt.scatter(peri[0], peri[1], s=120, marker='x', label="Perihelion")
    plt.scatter(aphe[0], aphe[1], s=120, marker='x', label="Aphelion")

    for s in STARTS:
        if s + SEGMENT_LENGTH < len(R2):
            segment = R2[s:s+SEGMENT_LENGTH]
            poly_x = np.concatenate([[0], segment[:, 0], [0]])
            poly_y = np.concatenate([[0], segment[:, 1], [0]])
            plt.fill(poly_x, poly_y, alpha=0.30)

    plt.annotate("Sun", xy=(0, 0), xytext=(10, 10), textcoords="offset points")
    plt.annotate("Perihelion", xy=(peri[0], peri[1]), xytext=(10, 10), textcoords="offset points")
    plt.annotate("Aphelion", xy=(aphe[0], aphe[1]), xytext=(10, -15), textcoords="offset points")

    plt.axis("equal")
    plt.xlabel("Orbital plane x (AU)")
    plt.ylabel("Orbital plane y (AU)")
    plt.title("Kepler's Second Law: Equal Areas in Equal Times (Mars)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig("law2_equal_areas.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 2
    plt.figure(figsize=(10, 4))
    plt.plot(areas, lw=1.8, label="Swept area per time step")
    plt.axhline(area_mean, linestyle="--", lw=1.5, label="Mean area")
    plt.xlabel("Time step")
    plt.ylabel("Swept area")
    plt.title("Swept Area per Equal Time Step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig("law2_area_vs_time.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 3
    plt.figure(figsize=(10, 4))
    plt.plot(Lmag, lw=1.8, label="|L|")
    plt.axhline(Lmag.mean(), linestyle="--", lw=1.5, label="Mean |L|")
    plt.xlabel("Time step")
    plt.ylabel("|L|")
    plt.title("Angular Momentum Magnitude of Mars")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig("law2_angular_momentum.png", dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "area_mean": float(area_mean),
        "area_std": float(area_std),
        "area_rel_percent": float(area_rel),
        "L_rel_percent": float(L_rel),
    }

def main():
    result = analyze_law2(save_plots=True)

    print("\n=== Kepler's Second Law Summary ===")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
