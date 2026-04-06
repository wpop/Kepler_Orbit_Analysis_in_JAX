"""
kepler_law1.py

Kepler's First Law Verification from Planetary Orbit Data
---------------------------------------------------------

This script analyzes planetary orbit data and numerically demonstrates
Kepler's First Law: planets move around the Sun in approximately elliptical
orbits, with the Sun located near one focus of the ellipse.

What this script does:
- Loads 3D orbital position data for several planets from CSV files
- Projects each orbit onto its best-fit orbital plane
- Estimates ellipse geometry parameters:
    * semi-major axis (a)
    * semi-minor axis (b)
    * eccentricity (e)
- Measures how closely each orbit lies in a single plane
- Identifies perihelion and aphelion positions
- Produces visualizations of:
    * all planetary orbits
    * detailed orbital geometry for Mars
- Prints a summary table of estimated orbital properties

Expected input files:
- mercury_orbit.csv
- venus_orbit.csv
- earth_orbit.csv
- mars_orbit.csv
- jupiter_orbit.csv
- saturn_orbit.csv

Each CSV file should contain the columns:
- x
- y
- z

Units:
- Position coordinates are assumed to be in astronomical units (AU)

Outputs:
- law1_all_orbits.png
- law1_mars_geometry.png
- printed summary table in the terminal

Main scientific idea:
Kepler's First Law states that planetary orbits are ellipses rather than
perfect circles. This script provides a simple computational verification
of that law using orbital position data.

Author: William Popkov
Project: Kepler's Laws with JAX
Date: 2026-01-11
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import jit


# Dictionary that maps each planet name to its corresponding orbit data file.
# Each CSV file is expected to contain 3D Cartesian coordinates: x, y, z.
PLANET_FILES = {
    "Mercury": "mercury_orbit.csv",
    "Venus":   "venus_orbit.csv",
    "Earth":   "earth_orbit.csv",
    "Mars":    "mars_orbit.csv",
    "Jupiter": "jupiter_orbit.csv",
    "Saturn":  "saturn_orbit.csv",
}

@jit
def vector_norm(x):
    """
    Compute the Euclidean norm (length) of vectors along the last axis.

    Parameters
    ----------
    x : jax.numpy.ndarray
        Input array containing one or more vectors.

    Returns
    -------
    jax.numpy.ndarray
        Euclidean norm of each vector.
    """
    return jnp.sqrt(jnp.sum(x**2, axis=-1))



def load_positions(filename):
    """
    Load 3D planetary position data from a CSV file.

    Parameters
    ----------
    filename : str
        Path to the CSV file containing orbit coordinates.

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (N, 3) containing 3D positions.
    """
    df = pd.read_csv(filename)
    R = df[["x", "y", "z"]].values.astype(np.float32)
    return jnp.array(R)


def project_to_orbital_plane(R):
    """
    Project 3D positions onto their best-fit orbital plane.

    Parameters
    ----------
    R : jax.numpy.ndarray
        Array of shape (N, 3) containing 3D positions.

    Returns
    -------
    R2 : jax.numpy.ndarray
        Array of shape (N, 2) containing 2D projections onto the orbital plane.
    normal : jax.numpy.ndarray
        Normal vector to the orbital plane.
    """
    
	# Perform Singular Value Decomposition on the 3D orbit points.
    # This identifies the dominant geometric directions of the orbit.
    U, S, Vt = jnp.linalg.svd(R, full_matrices=False)
    
	# The first two right-singular vectors define the best-fit orbital plane.
    plane_basis = Vt[:2] 
    
    # The third right-singular vector is perpendicular to the orbital plane.
    normal = Vt[2]
    
	# Project the 3D orbit coordinates onto the 2D orbital-plane basis(x,y,z -> u,v).
    R2 = R @ plane_basis.T
    
    return R2, normal


def estimate_ellipse_geometry(R2):
    """
    Estimate the geometric parameters of a 2D orbital ellipse.

    This function approximates the orbit as an ellipse by:
    1. Centering the 2D orbit points
    2. Computing the covariance matrix
    3. Finding the principal directions via eigen-decomposition
    4. Estimating the semi-major and semi-minor axes
    5. Computing the eccentricity

    Parameters
    ----------
    R2 : jax.numpy.ndarray
        Array of shape (N, 2) containing 2D orbital coordinates
        projected onto the orbital plane.

    Returns
    -------
    tuple
        (
            R2_centered : numpy.ndarray
                Centered 2D orbit points,
            a_est : float
                Estimated semi-major axis,
            b_est : float
                Estimated semi-minor axis,
            e_est : float
                Estimated eccentricity,
            major_dir : numpy.ndarray
                Direction of the estimated major axis,
            center : numpy.ndarray
                Estimated geometric center of the ellipse
        )
    """
    center = jnp.mean(R2, axis=0)   # Estimate the geometric center of the orbit in 2D.
    R2_centered = R2 - center       # Shift the orbit so that its center is at the origin.

    # 2D covariance matrix of the centered orbit points to identify spread directions of the ellipse.
    Cov2 = jnp.cov(R2_centered.T)

    # The eigenvectors define the principal axes of the ellipse.
    eigvals, eigvecs = jnp.linalg.eigh(Cov2)

    # Project the centered orbit points onto the principal axis basis.
    proj = R2_centered @ eigvecs

    # Coordinates in the principal-axis frame.
    u = np.array(proj[:, 1])
    v = np.array(proj[:, 0])

    # Estimate the semi-axis lengths from the spread of the projected coordinates.
    a_est = 0.5 * (u.max() - u.min())
    b_est = 0.5 * (v.max() - v.min())

    # Ensure that a_est is always the semi-major axis and b_est is the semi-minor axis.
    if a_est < b_est:
        a_est, b_est = b_est, a_est
        major_dir = np.array(eigvecs[:, 0])
    else:
        major_dir = np.array(eigvecs[:, 1])

    # Compute the eccentricity of the ellipse:
    e_est = np.sqrt(max(0.0, 1.0 - (b_est**2 / a_est**2)))

    return np.array(R2_centered), a_est, b_est, e_est, major_dir, np.array(center)


def analyze_law1(save_plots=True):
    """
    Perform a full numerical analysis of Kepler's First Law
    using planetary orbit data.

    For each planet, this function:
    1. Loads the 3D orbit data
    2. Projects the orbit onto its best-fit orbital plane
    3. Measures how closely the orbit lies in a plane
    4. Estimates the ellipse geometry of the orbit
    5. Identifies perihelion and aphelion
    6. Stores the computed quantities

    It also generates:
    - a plot of all planetary orbits
    - a detailed geometry plot for Mars

    Parameters
    ----------
    save_plots : bool, optional
        If True, save the generated figures as image files.

    Returns
    -------
    dict
        Dictionary containing geometric and orbital properties
        for each planet.
    """
    results = {}

    for planet, filename in PLANET_FILES.items():
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing file: {filename}")

		# Load 3D orbital coordinates for the current planet.
        R = load_positions(filename)
        
		# Project the 3D orbit onto its best-fit 2D orbital plane.
        R2, normal = project_to_orbital_plane(R)

		# Measure how far the orbit points deviate from the orbital plane.
        # A smaller value indicates a thinner, more planar orbit.
        dist_from_plane = np.array(R @ normal)
        plane_thickness = np.std(dist_from_plane)

		# Estimate ellipse parameters from the 2D projected orbit.
        R2_centered, a_est, b_est, e_est, major_dir, center = estimate_ellipse_geometry(R2)

		# Compute the radial distance of each point from the Sun (origin).
        r_norm = np.sqrt(np.sum(np.array(R2)**2, axis=1))
        
        peri_idx = np.argmin(r_norm) # Perihelion: closest point to the Sun
        aphe_idx = np.argmax(r_norm) # Aphelion: farthest point from the Sun

		# Store all computed quantities for this planet.
        results[planet] = {
            "R2": np.array(R2),
            "a": a_est,
            "b": b_est,
            "e": e_est,
            "plane_thickness": plane_thickness,
            "peri_point": np.array(R2[peri_idx]),
            "aphe_point": np.array(R2[aphe_idx]),
            "major_dir": major_dir,
            "center": center
        }

    # Plot 1: all orbits
    plt.figure(figsize=(10, 10))
    for planet, d in results.items():
        R2 = d["R2"]
        plt.plot(R2[:, 0], R2[:, 1], lw=2, label=planet)

    plt.scatter([0], [0], s=350, marker='o', label="Sun")
    plt.axis("equal")
    plt.xlabel("Orbital plane x (AU)")
    plt.ylabel("Orbital plane y (AU)")
    plt.title("Kepler's First Law: Planetary Orbits Around the Sun")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # Save the Mars geometry figure if requested.
    if save_plots:
        plt.savefig("law1_all_orbits.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot 2: Mars geometry
    d = results["Mars"]
    R2 = d["R2"]
    peri = d["peri_point"]
    aphe = d["aphe_point"]
    major_dir = d["major_dir"]
    a_est = d["a"]
    center = d["center"]

    plt.figure(figsize=(9, 9))
    plt.plot(R2[:, 0], R2[:, 1], lw=2.5, label="Mars orbit")
    plt.scatter([0], [0], s=350, marker='o', label="Sun")
    plt.scatter(peri[0], peri[1], s=120, marker='x', label="Perihelion")
    plt.scatter(aphe[0], aphe[1], s=120, marker='x', label="Aphelion")

    p1 = center - a_est * major_dir
    p2 = center + a_est * major_dir
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", lw=2, label="Estimated major axis")

    plt.annotate("Perihelion", xy=(peri[0], peri[1]), xytext=(10, 10), textcoords="offset points")
    plt.annotate("Aphelion", xy=(aphe[0], aphe[1]), xytext=(10, -15), textcoords="offset points")
    plt.annotate("Sun", xy=(0, 0), xytext=(10, 10), textcoords="offset points")

    plt.axis("equal")
    plt.xlabel("Orbital plane x (AU)")
    plt.ylabel("Orbital plane y (AU)")
    plt.title("Kepler's First Law: Elliptical Orbit Geometry (Mars)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig("law1_mars_geometry.png", dpi=300, bbox_inches="tight")
    plt.show()

    return results

def main():
    results = analyze_law1(save_plots=True)

    print("\n=== Kepler's First Law Summary ===")
    for planet, d in results.items():
        print(
            f"{planet:8s} | a={d['a']:.4f} AU | b={d['b']:.4f} AU | "
            f"e={d['e']:.4f} | plane std={d['plane_thickness']:.6f}"
        )

if __name__ == "__main__":
    main()
