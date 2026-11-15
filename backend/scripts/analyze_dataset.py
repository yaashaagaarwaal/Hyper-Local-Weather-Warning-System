"""One-off dataset analysis script for dataset/delhi_ncr_labeled_weather.csv.

Not part of the running application — stdlib only (csv, statistics, math,
collections), run manually to produce docs/dataset_analysis.md. Kept here
(rather than deleted after running) so the analysis is reproducible and the
report's numbers can be independently re-verified.

Usage: python backend/scripts/analyze_dataset.py
"""

import csv
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[2] / "dataset" / "delhi_ncr_labeled_weather.csv"

NUMERIC_COLUMNS = [
    "temperature_c", "apparent_temperature_c", "relative_humidity_pct", "dew_point_c",
    "precipitation_mm_hour", "rainfall_mm_hour", "showers_mm_hour", "weather_code",
    "wind_speed_kmh", "wind_gust_kmh", "wind_direction_deg", "visibility_m",
    "pressure_hpa", "surface_pressure_hpa", "cloud_cover_pct",
    "soil_moisture_0_7cm", "soil_moisture_7_28cm", "soil_moisture_28_100cm",
    "cape_j_kg", "lifted_index", "boundary_layer_height_m", "wet_bulb_temperature_c",
    "elevation_m_api", "latitude", "longitude",
    "rain_3h_mm", "rain_6h_mm", "rain_12h_mm", "rain_24h_mm", "rain_48h_mm", "rain_72h_mm",
    "rainfall_daily_total_mm", "soil_moisture_mean_available",
]
CATEGORICAL_COLUMNS = [
    "area_name", "state", "wind_class", "rain_class", "heat_class", "visibility_class",
    "fog_risk_proxy_class", "thunderstorm_proxy_class", "flood_risk_rain_soil_proxy_class",
    "label_source", "label_quality",
]


def is_missing(v):
    return v is None or v == "" or v.strip() == ""


def try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main():
    total_rows = 0
    missing_counts = Counter()
    numeric_values = defaultdict(list)
    categorical_counts = defaultdict(Counter)
    area_rows = defaultdict(int)
    area_coords = {}
    timestamps_by_area = defaultdict(list)
    seen_exact_rows = set()
    exact_duplicate_count = 0
    seen_area_ts = Counter()
    row_hash_sample_limit = 2_000_000  # dedup check only needs the tuple; cheap for this row count
    implausible = Counter()
    columns = None

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for row in reader:
            total_rows += 1

            for col in columns:
                if is_missing(row.get(col)):
                    missing_counts[col] += 1

            for col in NUMERIC_COLUMNS:
                val = try_float(row.get(col))
                if val is not None:
                    numeric_values[col].append(val)

            for col in CATEGORICAL_COLUMNS:
                val = row.get(col)
                if not is_missing(val):
                    categorical_counts[col][val] += 1

            area = row.get("area_name")
            ts = row.get("timestamp_local")
            area_rows[area] += 1
            timestamps_by_area[area].append(ts)
            area_coords[area] = (row.get("latitude"), row.get("longitude"), row.get("state"))

            key_exact = tuple(row.get(c) for c in columns)
            if total_rows <= row_hash_sample_limit:
                if key_exact in seen_exact_rows:
                    exact_duplicate_count += 1
                else:
                    seen_exact_rows.add(key_exact)

            seen_area_ts[(area, ts)] += 1

            # Physical-plausibility sanity checks
            t = try_float(row.get("temperature_c"))
            if t is not None and not (-10 <= t <= 55):
                implausible["temperature_c out of [-10, 55]"] += 1
            rh = try_float(row.get("relative_humidity_pct"))
            if rh is not None and not (0 <= rh <= 100):
                implausible["relative_humidity_pct out of [0, 100]"] += 1
            rain = try_float(row.get("rainfall_mm_hour"))
            if rain is not None and rain < 0:
                implausible["rainfall_mm_hour negative"] += 1
            wind = try_float(row.get("wind_speed_kmh"))
            if wind is not None and wind < 0:
                implausible["wind_speed_kmh negative"] += 1
            pres = try_float(row.get("pressure_hpa"))
            if pres is not None and not (850 <= pres <= 1100):
                implausible["pressure_hpa out of [850, 1100]"] += 1

    logical_duplicates = sum(c - 1 for c in seen_area_ts.values() if c > 1)

    print("=" * 70)
    print(f"TOTAL ROWS: {total_rows}")
    print(f"TOTAL COLUMNS: {len(columns)}")
    print(f"EXACT DUPLICATE ROWS (first {row_hash_sample_limit:,} rows checked): {exact_duplicate_count}")
    print(f"LOGICAL DUPLICATES (same area_name + timestamp_local appearing >1x): {logical_duplicates}")
    print()

    print("=" * 70)
    print("MISSING VALUES BY COLUMN (count, % of total rows)")
    for col in columns:
        n = missing_counts.get(col, 0)
        if n > 0:
            print(f"  {col:45s} {n:8d}  ({100*n/total_rows:5.2f}%)")
    never_missing = [c for c in columns if missing_counts.get(c, 0) == 0]
    print(f"  ({len(never_missing)} of {len(columns)} columns have ZERO missing values)")
    print()

    print("=" * 70)
    print("GEOGRAPHIC COVERAGE")
    print(f"  Unique area_name values: {len(area_rows)}")
    for area, (lat, lon, state) in area_coords.items():
        print(f"    {area:35s} lat={lat:>10s} lon={lon:>10s} state={state} rows={area_rows[area]}")
    lats = [float(lat) for lat, lon, state in area_coords.values()]
    lons = [float(lon) for lat, lon, state in area_coords.values()]
    print(f"  Latitude range:  {min(lats):.4f} .. {max(lats):.4f}  (span {max(lats)-min(lats):.4f}°, ~{(max(lats)-min(lats))*111:.1f} km)")
    print(f"  Longitude range: {min(lons):.4f} .. {max(lons):.4f}  (span {max(lons)-min(lons):.4f}°, ~{(max(lons)-min(lons))*111*math.cos(math.radians(sum(lats)/len(lats))):.1f} km)")
    print()

    print("=" * 70)
    print("TIMESTAMP COVERAGE PER AREA")
    for area, ts_list in timestamps_by_area.items():
        ts_sorted = sorted(ts_list)
        n = len(ts_sorted)
        first, last = ts_sorted[0], ts_sorted[-1]
        # Expected hourly count between first and last
        fmt = "%Y-%m-%d %H:%M:%S"
        try:
            dt_first = datetime.strptime(first, fmt)
            dt_last = datetime.strptime(last, fmt)
            expected = int((dt_last - dt_first).total_seconds() // 3600) + 1
            gap = expected - n
        except ValueError:
            expected, gap = None, None
        print(f"  {area:35s} n={n:6d}  first={first}  last={last}  expected_hourly={expected}  missing_hours={gap}")
    print()

    print("=" * 70)
    print("NUMERIC COLUMN SUMMARY (non-missing values only)")
    print(f"  {'column':35s} {'n':>8s} {'min':>12s} {'max':>12s} {'mean':>12s} {'stdev':>12s}")
    for col in NUMERIC_COLUMNS:
        vals = numeric_values.get(col, [])
        if not vals:
            print(f"  {col:35s} {'(no numeric data)':>8s}")
            continue
        mean = statistics.fmean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"  {col:35s} {len(vals):8d} {min(vals):12.3f} {max(vals):12.3f} {mean:12.3f} {sd:12.3f}")
    print()

    print("=" * 70)
    print("PHYSICAL PLAUSIBILITY CHECKS")
    if implausible:
        for k, v in implausible.items():
            print(f"  {k}: {v} rows ({100*v/total_rows:.3f}%)")
    else:
        print("  No implausible values detected for the checked fields/ranges.")
    print()

    print("=" * 70)
    print("CATEGORICAL / LABEL DISTRIBUTIONS")
    for col in CATEGORICAL_COLUMNS:
        counts = categorical_counts.get(col, Counter())
        print(f"  --- {col} ---")
        for val, n in counts.most_common(15):
            print(f"      {val:55s} {n:8d}  ({100*n/total_rows:5.2f}%)")
        print()


if __name__ == "__main__":
    main()
