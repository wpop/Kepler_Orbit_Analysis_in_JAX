"""
run_all.py

Main execution script for the numerical verification of Kepler's laws.

This script runs the complete orbital analysis pipeline for:

    1. Kepler's First Law  - elliptical shape of planetary orbits
    2. Kepler's Second Law - equal areas swept in equal times
    3. Kepler's Third Law  - relation between orbital period and semi-major axis

Main tasks
----------
1. Execute all three Kepler-law analysis modules.
2. Generate and save diagnostic plots.
3. Build a summary table of key physical quantities.
4. Export results to CSV format.
5. Optionally create a plain-text mini report.

Outputs
-------
- PNG figures for each law
- kepler_summary.csv
- kepler_report.txt (optional)

Notes
-----
- This script serves as the main entry point for the project.
- It is intended for educational and computational physics purposes.
- All required planetary trajectory files must be available before execution.

Author: William Popkov
Date: 2026-01-25
"""

import pandas as pd

from kepler_law1 import analyze_law1
from kepler_law2 import analyze_law2
from kepler_law3 import analyze_law3

# ============================================================
# Settings
# ============================================================

GENERATE_REPORT = True

# ============================================================
# Run all analyses
# ============================================================

print("\nRunning Kepler Law 1 analysis...")
law1_results = analyze_law1(save_plots=True)

print("\nRunning Kepler Law 2 analysis...")
law2_result = analyze_law2(save_plots=True)

print("\nRunning Kepler Law 3 analysis...")
law3_result = analyze_law3(save_plots=True)

# ============================================================
# Create summary table
# ============================================================

summary_rows = []

# Law 1 per-planet summary
for planet, d in law1_results.items():
    summary_rows.append({
        "law": "Law 1",
        "planet": planet,
        "metric_1_name": "semi_major_axis_a_AU",
        "metric_1_value": d["a"],
        "metric_2_name": "eccentricity_e",
        "metric_2_value": d["e"],
        "metric_3_name": "plane_thickness_std",
        "metric_3_value": d["plane_thickness"],
    })

# Law 2 summary
summary_rows.append({
    "law": "Law 2",
    "planet": "Mars",
    "metric_1_name": "mean_swept_area",
    "metric_1_value": law2_result["area_mean"],
    "metric_2_name": "relative_area_variation_percent",
    "metric_2_value": law2_result["area_rel_percent"],
    "metric_3_name": "relative_angular_momentum_variation_percent",
    "metric_3_value": law2_result["L_rel_percent"],
})

# Law 3 per-planet summary
for planet, a, T, ratio in zip(
    law3_result["planets"],
    law3_result["a_values"],
    law3_result["T_values"],
    law3_result["ratios"]
):
    summary_rows.append({
        "law": "Law 3",
        "planet": planet,
        "metric_1_name": "semi_major_axis_a_AU",
        "metric_1_value": a,
        "metric_2_name": "orbital_period_T_years",
        "metric_2_value": T,
        "metric_3_name": "T2_over_a3",
        "metric_3_value": ratio,
    })

# Law 3 global summary
summary_rows.append({
    "law": "Law 3",
    "planet": "ALL",
    "metric_1_name": "mean_T2_over_a3",
    "metric_1_value": law3_result["ratio_mean"],
    "metric_2_name": "relative_std_percent",
    "metric_2_value": law3_result["ratio_rel_std_percent"],
    "metric_3_name": "loglog_slope",
    "metric_3_value": law3_result["slope"],
})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("kepler_summary.csv", index=False)

print("\nSaved summary table: kepler_summary.csv")
print(summary_df)

# ============================================================
# Optional mini report
# ============================================================

if GENERATE_REPORT:
    with open("kepler_report.txt", "w", encoding="utf-8") as f:
        f.write("KEPLER'S LAWS PROJECT REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. Kepler's First Law\n")
        f.write("-" * 60 + "\n")
        f.write("Projected planetary trajectories form approximately elliptical orbits.\n")
        f.write("Estimated orbital parameters:\n")
        for planet, d in law1_results.items():
            f.write(
                f"  {planet}: a={d['a']:.4f} AU, b={d['b']:.4f} AU, "
                f"e={d['e']:.4f}, plane_std={d['plane_thickness']:.6f}\n"
            )
        f.write("\n")

        f.write("2. Kepler's Second Law\n")
        f.write("-" * 60 + "\n")
        f.write("Mars was used to verify that equal areas are swept in equal times.\n")
        f.write(f"Mean swept area per step: {law2_result['area_mean']:.8f}\n")
        f.write(f"Relative swept area variation: {law2_result['area_rel_percent']:.6f}%\n")
        f.write(f"Relative angular momentum variation: {law2_result['L_rel_percent']:.6f}%\n\n")

        f.write("3. Kepler's Third Law\n")
        f.write("-" * 60 + "\n")
        f.write("The relation T^2 ∝ a^3 was verified across multiple planets.\n")
        f.write(f"Mean T^2/a^3: {law3_result['ratio_mean']:.6f}\n")
        f.write(f"Relative std of T^2/a^3: {law3_result['ratio_rel_std_percent']:.6f}%\n")
        f.write(f"Log-log slope: {law3_result['slope']:.6f}\n")
        f.write(f"Absolute slope error from theoretical 1.5: {law3_result['slope_abs_error']:.6f}\n\n")

        f.write("Final Conclusion\n")
        f.write("-" * 60 + "\n")
        f.write("The planetary orbital dataset is consistent with all three of Kepler's laws.\n")

    print("Saved mini report: kepler_report.txt")
