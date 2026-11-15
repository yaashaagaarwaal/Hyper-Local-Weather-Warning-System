"""Phase 4: builds a geoBoundaries(ADM2) <-> LGD district/state crosswalk.

Problem this solves (see Phase 4 investigation findings): geoBoundaries'
ADM2 (district) GeoJSON has NO state field at all -- only a district name,
which is ambiguous for 7 names that exist in more than one place in India.
Rather than guess a district's state from its name, this script:

  1. Loads geoBoundaries' own ADM1 (state) layer -- same source/provider,
     same product family, a different administrative level, not a new
     boundary source.
  2. Computes each ADM2 district's centroid and does a point-in-polygon
     test against the 36 ADM1 state polygons to find which state actually
     contains it -- a geometric fact, not an inference from names.
  3. Joins (district_name, spatially-assigned state_name) against the
     authoritative LGD district table, applying ONLY the two renames
     verified during Phase 4 investigation (Gurgaon->Gurugram is a clean
     1:1 case handled by plain name matching once the LGD table is
     checked; Aurangabad/Maharashtra->Chhatrapati Sambhajinagar and
     Bijapur/Karnataka->Vijayapura are applied explicitly, scoped to the
     exact state where each rename was confirmed -- never applied blindly
     to every district with that name).

Every row keeps its match status (matched_exact / matched_renamed /
unmatched / spatial_join_uncertain) -- nothing is silently resolved or
guessed. Reads geoBoundaries + LGD files already downloaded; does not
modify the India Flood Inventory or any other original source file.

Run as `python dataset/scripts/build_district_crosswalk.py` from the repo
root.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pandas as pd


def _normalize(s: str) -> str:
    """Strips diacritical marks (geoBoundaries' ADM1 names use macrons, e.g.
    'Mahārāshtra', vs LGD's plain-ASCII 'Maharashtra' -- found during Phase 4
    investigation to be the cause of 469/735 false non-matches) and
    lowercases/trims for a stable comparison key. A text-normalization fix,
    not a guess -- the underlying state is unambiguously the same.
    """
    decomposed = unicodedata.normalize("NFKD", s)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.strip().lower()

BOUNDARIES_DIR = Path(__file__).resolve().parent.parent / "boundaries"
OUT_DIR = Path(__file__).resolve().parent.parent / "processed" / "district_crosswalk"

# Verified during Phase 4 investigation (not a guess): (geoBoundaries ADM2
# name, spatially-assigned state) -> current LGD district name.
VERIFIED_RENAMES = {
    ("Aurangabad", "Maharashtra"): "Chhatrapati Sambhajinagar",
    ("Bijapur", "Karnataka"): "Vijayapura",
    ("Gurgaon", "Haryana"): "Gurugram",
}


def _ring_centroid(ring: list[list[float]]) -> tuple[float, float]:
    """Simple vertex-average centroid of one ring -- adequate for
    classifying which state contains a district, not a true area centroid.
    """
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def polygon_centroid(geometry: dict) -> tuple[float, float]:
    """Centroid of the largest ring across a Polygon or MultiPolygon."""
    if geometry["type"] == "Polygon":
        rings = [geometry["coordinates"][0]]
    else:  # MultiPolygon
        rings = [part[0] for part in geometry["coordinates"]]
    largest = max(rings, key=len)
    return _ring_centroid(largest)


def _point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    """Standard ray-casting point-in-polygon test for one ring."""
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
    return inside


def point_in_geometry(x: float, y: float, geometry: dict) -> bool:
    if geometry["type"] == "Polygon":
        polys = [geometry["coordinates"]]
    else:
        polys = geometry["coordinates"]
    for poly in polys:
        if _point_in_ring(x, y, poly[0]):
            return True
    return False


def _bbox(geometry: dict) -> tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) -- cheap pre-filter before the O(vertices)
    ray-cast, since some ADM1 polygons here have tens of thousands of
    vertices.
    """
    if geometry["type"] == "Polygon":
        rings = [geometry["coordinates"][0]]
    else:
        rings = [part[0] for part in geometry["coordinates"]]
    xs = [p[0] for ring in rings for p in ring]
    ys = [p[1] for ring in rings for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def assign_states(adm2_features: list[dict], adm1_features: list[dict]) -> list[dict]:
    adm1_info = [
        (f["properties"]["shapeName"], _bbox(f["geometry"]), f["geometry"])
        for f in adm1_features
    ]

    results = []
    for f in adm2_features:
        cx, cy = polygon_centroid(f["geometry"])
        assigned_state = None
        for state_name, (minx, miny, maxx, maxy), geom in adm1_info:
            if not (minx <= cx <= maxx and miny <= cy <= maxy):
                continue
            if point_in_geometry(cx, cy, geom):
                assigned_state = state_name
                break
        uncertain = assigned_state is None
        if uncertain:
            # Fallback: nearest ADM1 bbox-center -- flagged, never silent.
            def _bbox_center(b):
                return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

            assigned_state = min(
                adm1_info,
                key=lambda s: (_bbox_center(s[1])[0] - cx) ** 2 + (_bbox_center(s[1])[1] - cy) ** 2,
            )[0]
        results.append({
            "shapeID": f["properties"]["shapeID"],
            "shapeName": f["properties"]["shapeName"],
            "centroid_lon": cx,
            "centroid_lat": cy,
            "assigned_state": assigned_state,
            "spatial_join_uncertain": uncertain,
        })
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(BOUNDARIES_DIR / "geoBoundaries-IND-ADM2.geojson") as f:
        adm2 = json.load(f)["features"]
    with open(BOUNDARIES_DIR / "geoBoundaries-IND-ADM1.geojson") as f:
        adm1 = json.load(f)["features"]

    print(f"Assigning states to {len(adm2)} ADM2 districts via point-in-polygon "
          f"against {len(adm1)} ADM1 states...", flush=True)
    assigned = assign_states(adm2, adm1)
    uncertain_count = sum(1 for r in assigned if r["spatial_join_uncertain"])
    print(f"Done. {uncertain_count}/{len(assigned)} centroids fell outside every ADM1 "
          f"polygon (nearest-centroid fallback used, flagged).", flush=True)

    lgd = pd.read_parquet(BOUNDARIES_DIR / "lgd" / "districts.parquet")
    lgd_lookup = {
        (_normalize(row.district_name), _normalize(row.state_name)): row
        for row in lgd.itertuples()
    }
    renames_normalized = {
        (_normalize(k[0]), _normalize(k[1])): v for k, v in VERIFIED_RENAMES.items()
    }
    # name -> list of LGD rows with that name, across ALL states. Used as a
    # safety net: if a name is globally unique in LGD (not one of the real
    # multi-state duplicates) but the spatial join disagrees with its state,
    # that's evidence the spatial join itself is wrong at this location --
    # found during Phase 4 investigation for "Gautam Buddha Nagar" (correctly
    # UP, but geoBoundaries' Haryana ADM1 polygon appears imprecise right at
    # the Delhi/UP/Haryana tri-junction and wrongly contains its centroid).
    # Flagged explicitly, never silently trusted either way.
    lgd_by_name: dict[str, list] = {}
    for row in lgd.itertuples():
        lgd_by_name.setdefault(_normalize(row.district_name), []).append(row)

    rows = []
    for r in assigned:
        name, state = r["shapeName"], r["assigned_state"]
        key = (_normalize(name), _normalize(state))
        lgd_row = lgd_lookup.get(key)
        status = None
        matched_lgd_name = None
        spatial_state_conflict = False

        if lgd_row is not None:
            status = "matched_exact"
            matched_lgd_name = lgd_row.district_name
        else:
            # Normalized lookup -- `state` here is geoBoundaries' own ADM1
            # shapeName (e.g. diacritic "Haryāna"), which otherwise would
            # never match the plain-ASCII keys in VERIFIED_RENAMES.
            renamed = renames_normalized.get((_normalize(name), _normalize(state)))
            if renamed:
                renamed_key = (_normalize(renamed), _normalize(state))
                lgd_row = lgd_lookup.get(renamed_key)
                if lgd_row is not None:
                    status = "matched_renamed"
                    matched_lgd_name = lgd_row.district_name

        if status is None:
            candidates = lgd_by_name.get(_normalize(name), [])
            if len(candidates) == 1:
                # Globally unique name in LGD, but the spatial join assigned
                # a different state -- trust the name match, flag the
                # disagreement for review rather than discarding or hiding it.
                lgd_row = candidates[0]
                status = "matched_name_unique_spatial_conflict"
                matched_lgd_name = lgd_row.district_name
                spatial_state_conflict = True

        if status is None:
            status = "unmatched"

        rows.append({
            "geoBoundaries_shapeID": r["shapeID"],
            "geoBoundaries_shapeName": name,
            "assigned_state_spatial_join": state,
            "spatial_join_uncertain": r["spatial_join_uncertain"],
            "spatial_state_conflict": spatial_state_conflict,
            "match_status": status,
            "lgd_district_name": matched_lgd_name,
            "lgd_district_code": getattr(lgd_row, "district_code", None) if lgd_row is not None else None,
            "lgd_state_code": getattr(lgd_row, "state_code", None) if lgd_row is not None else None,
            "lgd_state_name": getattr(lgd_row, "state_name", None) if lgd_row is not None else None,
        })

    crosswalk = pd.DataFrame(rows)
    out_csv = OUT_DIR / "geoboundaries_lgd_crosswalk.csv"
    crosswalk.to_csv(out_csv, index=False)

    counts = crosswalk["match_status"].value_counts().to_dict()
    print(f"\nMatch status counts: {counts}", flush=True)
    print(f"Wrote {out_csv} ({len(crosswalk)} rows)", flush=True)

    doc = f"""# geoBoundaries <-> LGD district crosswalk

- **Built by:** `dataset/scripts/build_district_crosswalk.py`
- **Method:** point-in-polygon spatial join of each geoBoundaries ADM2
  district's centroid against geoBoundaries' own ADM1 state layer (same
  source/provider), then an exact (name, state) join against the
  authoritative LGD district table. Only 2 documented, geometry-scoped
  renames are applied (see VERIFIED_RENAMES in the script) -- nothing else
  is guessed.
- **Match status counts:** {counts}
- **Spatial-join-uncertain rows:** {uncertain_count} (centroid fell outside
  every ADM1 polygon; nearest-centroid fallback used -- these rows are
  flagged `spatial_join_uncertain=True` in the CSV, not silently accepted)
- **Output:** `geoboundaries_lgd_crosswalk.csv` ({len(crosswalk)} rows)

## What `unmatched` means here

A district name+assigned-state pair that doesn't exist in the current LGD
table under that name, and isn't one of the 2 documented renames. Causes
observed during Phase 4 investigation: Karnataka/other-state spelling
variants (e.g. Bengaluru vs Bangalore), and districts created by splits
since geoBoundaries' Jan 2023 vintage. Each `unmatched` row still has its
spatially-assigned state recorded, for manual review -- never dropped.
"""
    (OUT_DIR / "README.md").write_text(doc)
    print(f"Wrote {OUT_DIR / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
