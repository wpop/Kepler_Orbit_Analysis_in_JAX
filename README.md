# Numerical Verification of Kepler’s Laws

A numerical orbital mechanics project that validates **Kepler’s First, Second, and Third Laws** from planetary trajectory data through ellipse fitting, swept-area analysis, and scaling relations.

---

## Project Overview

This project numerically verifies all three of **Kepler’s Laws of Planetary Motion** using planetary orbital trajectory data.

The analysis is implemented in **Python** and **JAX**, and combines:

- orbital-plane projection
- ellipse geometry estimation
- swept-area analysis
- angular momentum consistency checks
- period–semi-major axis scaling
- scientific visualization

The goal is to show how classical physical laws can be recovered directly from numerical orbit data using computational methods.

---

## Scientific Goal

The project validates the following laws of planetary motion:

### 1. Kepler’s First Law
Planetary orbits are **ellipses**, with the Sun located at one focus.

### 2. Kepler’s Second Law
A planet sweeps out **equal areas in equal times**.

### 3. Kepler’s Third Law
The square of the orbital period is proportional to the cube of the semi-major axis:

\[
T^2 \propto a^3
\]

---

## Methods

This project uses numerical orbital trajectory data and applies the following computational methods:

- projection of 3D trajectories onto the best-fit orbital plane
- covariance-based estimation of orbital principal axes
- semi-major and semi-minor axis estimation
- eccentricity estimation
- area-sweep computation over equal time intervals
- angular momentum consistency verification
- multi-planet comparison of \(T^2\) and \(a^3\)
- log-log regression analysis for power-law validation

---

## Law 1 — Elliptical Orbits

For each planet:

- the 3D orbit is projected onto its orbital plane
- the orbit is analyzed geometrically
- the semi-major axis \(a\), semi-minor axis \(b\), and eccentricity \(e\) are estimated

This verifies that the orbital paths are approximately elliptical.

### Output examples
- orbital projection plots
- estimated ellipse parameters
- plane thickness diagnostics

---

## Law 2 — Equal Areas in Equal Times

Using **Mars** as a test case:

- consecutive orbital segments are selected
- the swept area for equal time intervals is computed
- angular momentum consistency is checked

This verifies Kepler’s Second Law numerically.

### Output examples
- swept-area comparison plots
- relative area variation
- relative angular momentum variation

---

## Law 3 — \(T^2 \propto a^3\)

Across multiple planets:

- the orbital period \(T\) is assigned using known planetary periods
- the semi-major axis \(a\) is estimated from the orbital data
- the scaling law \(T^2 \propto a^3\) is tested numerically

Two checks are performed:

1. **Direct scaling plot:** \(T^2\) vs \(a^3\)
2. **Log-log verification:** expected slope \(\approx 1.5\)

This confirms Kepler’s Third Law.

### Output examples
- \(T^2\) vs \(a^3\) scatter plot
- linear fit
- log-log regression plot
- slope error relative to the theoretical value 1.5

---

## Project Structure

```text
numerical-kepler-laws/
├── kepler_law1.py
├── kepler_law2.py
├── kepler_law3.py
├── run_all.py
├── README.md
├── requirements.txt
├── data/
│   ├── raw_txt/
│   │   ├── mercury.txt
│   │   ├── venus.txt
│   │   ├── earth.txt
│   │   └── mars.txt
│   └── csv/
│       ├── mercury.csv
│       ├── venus.csv
│       ├── earth.csv
│       └── mars.csv
└── outputs/
    ├── law1_earth.png
    ├── law2_mars.png
    ├── law3_T2_vs_a3.png
    ├── kepler_summary.csv
    └── kepler_report.txt
