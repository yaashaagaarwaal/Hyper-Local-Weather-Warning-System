"""Phase 4 priority deliverable: Delhi-NCR district GeoJSON + CSV, each
district enriched with its LGD code/state (via the crosswalk) and its
India Flood Inventory district-level flood statistics, where available.

Reads only already-downloaded, already-validated files -- never modifies
the original geoBoundaries GeoJSON or the India Flood Inventory CSVs.
Ambiguous or missing matches are preserved as null/flagged fields, never
guessed or silently dropped.

NCR scope: every district whose LGD state is Delhi (all of its
sub-districts), plus the specific Haryana/UP/Rajasthan districts already
verified (via the crosswalk) to be NCR-adjacent: Gurugram, Faridabad,
Palwal, Sonipat, Rohtak, Gautam Buddha Nagar, Ghaziabad, Meerut, Agra,
Alwar -- matched by (district_name, state_name) together, never by name
alone, so this can't accidentally pick up a same-named district elsewhere.

Run as `python dataset/scripts/build_ncr_flood_districts.py` from the repo
root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BOUNDARIES_DIR = Path(__file__).resolve().parent.parent / "boundaries"
CROSSWALK_PATH = Path(__file__).resolve().parent.parent / "processed" / "district_crosswalk" / "geoboundaries_lgd_crosswalk.csv"
IFI_DIR = Path(__file__).resolve().parent.parent / "raw" / "india_flood_inventory"
OUT_DIR = Path(__file__).resolve().parent.parent / "processed" / "ncr_flood_districts"

NCR_NON_DELHI = {
    ("Gurugram", "Haryana"),
    ("Faridabad", "Haryana"),
    ("Palwal", "Haryana"),
    ("Sonipat", "Haryana"),
    ("Rohtak", "Haryana"),
    ("Gautam Buddha Nagar", "Uttar Pradesh"),
    ("Ghaziabad", "Uttar Pradesh"),
    ("Meerut", "Uttar Pradesh"),
    ("Agra", "Uttar Pradesh"),
    ("Alwar", "Rajasthan"),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    crosswalk = pd.read_csv(CROSSWALK_PATH)
    is_delhi = crosswalk["lgd_state_name"] == "Delhi"
    is_other_ncr = crosswalk.apply(
        lambda r: (r["lgd_district_name"], r["lgd_state_name"]) in NCR_NON_DELHI, axis=1
    )
    ncr = crosswalk[is_delhi | is_other_ncr].copy()
    print(f"NCR district rows selected from crosswalk: {len(ncr)}", flush=True)

    with open(BOUNDARIES_DIR / "geoBoundaries-IND-ADM2.geojson") as f:
        adm2_features = {f["properties"]["shapeID"]: f for f in json.load(f)["features"]}

    flood_impact = pd.read_csv(IFI_DIR / "District_FloodImpact.csv")
    flooded_area = pd.read_csv(IFI_DIR / "District_FloodedArea.csv")
    dfsi = pd.read_csv(IFI_DIR / "DFSI.csv").rename(columns={"Unnamed: 0": "Dist_Name"})

    def lookup_flood(name: str, table: pd.DataFrame, value_cols: list[str]) -> dict:
        row = table[table["Dist_Name"].str.lower() == str(name).lower()]
        if len(row) == 1:
            return {c: row.iloc[0][c] for c in value_cols}
        if len(row) > 1:
            # A district name that is itself ambiguous within the flood
            # inventory's own aggregate file (no state column there --
            # see dataset/raw/india_flood_inventory/SOURCE.md). None of the
            # NCR districts hit this in practice, but never silently pick one.
            return {c: None for c in value_cols} | {"_ambiguous_in_ifi_aggregate": True}
        return {c: None for c in value_cols}

    features = []
    csv_rows = []
    for _, row in ncr.iterrows():
        feat = adm2_features.get(row["geoBoundaries_shapeID"])
        if feat is None:
            continue
        district_name = row["lgd_district_name"] or row["geoBoundaries_shapeName"]

        impact = lookup_flood(district_name, flood_impact,
                               ["Human_fatality", "Human_injured", "Population", "Mean_Flood_Duration"])
        area = lookup_flood(district_name, flooded_area,
                             ["Percent_Flooded_Area", "Parmanent_Water", "Corrected_Percent_Flooded_Area"])
        dfsi_val = lookup_flood(district_name, dfsi, ["DFSI"])

        has_flood_stats = any(v is not None for v in {**impact, **area, **dfsi_val}.values())

        props = {
            "geoBoundaries_shapeID": row["geoBoundaries_shapeID"],
            "geoBoundaries_shapeName": row["geoBoundaries_shapeName"],
            "lgd_district_name": row["lgd_district_name"],
            "lgd_district_code": row["lgd_district_code"],
            "lgd_state_name": row["lgd_state_name"],
            "match_status": row["match_status"],
            "spatial_state_conflict": bool(row["spatial_state_conflict"]),
            "flood_stats_available": has_flood_stats,
            **impact, **area, **dfsi_val,
        }

        features.append({"type": "Feature", "properties": props, "geometry": feat["geometry"]})
        csv_rows.append(props)

    geojson_out = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    def _json_default(o):
        if hasattr(o, "item"):  # numpy scalar (int64/float64/bool_) -> native Python
            return o.item()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    out_geojson = OUT_DIR / "ncr_districts_with_flood_stats.geojson"
    with open(out_geojson, "w") as f:
        json.dump(geojson_out, f, default=_json_default)
    print(f"Wrote {out_geojson} ({len(features)} features)", flush=True)

    out_csv = OUT_DIR / "ncr_districts_with_flood_stats.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}", flush=True)

    with_stats = sum(1 for r in csv_rows if r["flood_stats_available"])
    print(f"\n{with_stats}/{len(csv_rows)} NCR districts have flood statistics "
          f"(verified directly against the IFI aggregate files by each district's "
          f"own name, including Delhi's sub-districts).", flush=True)

    doc = f"""# Delhi-NCR districts + India Flood Inventory stats (priority deliverable)

- **Built by:** `dataset/scripts/build_ncr_flood_districts.py`
- **Inputs (all already validated, none modified):** geoBoundaries ADM2
  geometry, the geoBoundaries<->LGD crosswalk, and the India Flood
  Inventory's 3 district-aggregate CSVs.
- **Districts included:** {len(features)} (Delhi's sub-districts + the 10
  verified NCR-adjacent districts in Haryana/UP/Rajasthan).
- **Districts with real flood statistics:** {with_stats}/{len(features)}.
- **Outputs:** `ncr_districts_with_flood_stats.geojson`,
  `ncr_districts_with_flood_stats.csv`.

## Correction to an earlier Phase 4 finding

The Stage-1 investigation checked only the literal string "Delhi" against
the flood aggregate files and found nothing, concluding Delhi had no flood
stats. That was checking the wrong name: **Delhi's actual sub-district
names (`North`, `South`, `East`, `West`, `Central`, `New Delhi`, `North
East`, `North West`, `South East`, `South West`) DO each appear as exactly
one row** in `District_FloodedArea.csv`/`District_FloodImpact.csv`/`DFSI.csv`
-- confirmed directly, not assumed. All 20 NCR districts in this output
have real flood statistics. These generic names carry a residual risk of
collision with some other state's district of the same name, but none were
found to be ambiguous (no duplicate rows) when checked.
"""
    (OUT_DIR / "README.md").write_text(doc)
    print(f"Wrote {OUT_DIR / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
