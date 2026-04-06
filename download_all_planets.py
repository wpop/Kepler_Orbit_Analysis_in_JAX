import requests
import re
import pandas as pd

# ============================================================
# Planet IDs in JPL Horizons
# ============================================================

PLANETS = {
    "mercury": "199",
    "venus":   "299",
    "earth":   "399",
    "mars":    "499",
    "jupiter": "599",
    "saturn":  "699",
}

START_TIME = "2022-01-01"
STOP_TIME  = "2026-12-31"
STEP_SIZE  = "1 d"

URL = "https://ssd.jpl.nasa.gov/api/horizons.api"


def download_planet(name, command_id):
    # Parameters for JPL Horizons API
    params = {
        "format": "text",
        "COMMAND": f"'{command_id}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@10'",      # Sun-centered
        "START_TIME": f"'{START_TIME}'",
        "STOP_TIME":  f"'{STOP_TIME}'",
        "STEP_SIZE":  f"'{STEP_SIZE}'",
        "CSV_FORMAT": "'YES'",
        "OUT_UNITS": "'AU-D'",
        "VEC_TABLE": "'2'",
        "VEC_LABELS": "'NO'"
    }

    print(f"\nDownloading {name}...")
    response = requests.get(URL, params=params)
    response.raise_for_status()
    text = response.text

    # Save raw file for debugging if needed
    raw_filename = f"{name}_orbit_raw.txt"
    with open(raw_filename, "w", encoding="utf-8") as f:
        f.write(text)

    # Extract data block between $$SOE and $$EOE
    match = re.search(r"\$\$SOE(.*?)\$\$EOE", text, re.DOTALL)
    if not match:
        print(f"Could not parse data block for {name}")
        return

    block = match.group(1).strip()

    rows = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]

        # Expected format:
        # JD, date, x, y, z, vx, vy, vz, ...
        if len(parts) >= 8:
            try:
                date = parts[1]
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                vx = float(parts[5])
                vy = float(parts[6])
                vz = float(parts[7])

                rows.append([date, x, y, z, vx, vy, vz])
            except ValueError:
                continue

    df = pd.DataFrame(rows, columns=["date", "x", "y", "z", "vx", "vy", "vz"])

    csv_filename = f"{name}_orbit.csv"
    df.to_csv(csv_filename, index=False)

    print(f"Saved: {csv_filename} ({len(df)} rows)")


# ============================================================
# Main
# ============================================================

for planet_name, planet_id in PLANETS.items():
    download_planet(planet_name, planet_id)

print("\nDone. All 6 planetary datasets downloaded.")
