"""Hyper-Local Grid geometry.

Ports the *geometry only* of `buildRiskGrid()` from the frontend's
src/data/riskData.js: a 7x7 lat/lng lattice over the same Delhi/NCR bounding
box, so the backend-served grid lines up exactly with what the frontend map
already renders. Deliberately does NOT port the mock risk/weather fields that
function also generates — those are fabricated demo values and have no place
in the database.

A regular lat/lng lattice was chosen over an H3/S2 hex index for Phase 3: it
requires no new dependency, it's what the frontend map already renders
(`RISK_GRID` in riskData.js draws rectangular Leaflet `Polygon`s, not
hexagons), and resolution is already configurable (see `build_grid_geometry`
below) without one. H3 remains a reasonable future upgrade once the GIS
Engine needs real neighbor/containment queries at scale (docs/architecture.md
§6) — nothing here forecloses that.
"""

DEFAULT_ROWS = 7
DEFAULT_COLS = 7
DEFAULT_BOUNDS = {"south": 28.32, "north": 28.88, "west": 76.82, "east": 77.56}
DEFAULT_RESOLUTION = "7x7"


def build_grid_geometry(
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    bounds: dict | None = None,
    resolution: str | None = None,
) -> list[dict]:
    """Returns one dict per cell: id, row, col, resolution, bounds (4
    [lat,lng] corners), center_lat, center_lng — everything GridCell needs
    except timestamps.

    Resolution is configurable via `rows`/`cols` (and optionally a different
    `bounds` box). Calling this with no arguments reproduces the exact
    original 49-cell "7x7" grid and its original "GRID-001".."GRID-049" ids
    unchanged — existing seeded rows and anything that references them by id
    are unaffected. A non-default resolution gets ids of the form
    "GRID-<resolution>-NNN" instead, so multiple resolutions can coexist in
    the same `grid_cells` table without id collisions.
    """
    is_default = rows == DEFAULT_ROWS and cols == DEFAULT_COLS and bounds is None
    bounds = bounds or DEFAULT_BOUNDS
    resolution = resolution or (DEFAULT_RESOLUTION if is_default else f"{rows}x{cols}")
    id_prefix = "GRID" if is_default else f"GRID-{resolution}"

    lat_step = (bounds["north"] - bounds["south"]) / rows
    lng_step = (bounds["east"] - bounds["west"]) / cols

    cells = []
    for r in range(rows):
        for c in range(cols):
            south = bounds["south"] + r * lat_step
            north = south + lat_step
            west = bounds["west"] + c * lng_step
            east = west + lng_step
            cell_id = f"{id_prefix}-{r * cols + c + 1:03d}"
            cells.append(
                {
                    "id": cell_id,
                    "row": r,
                    "col": c,
                    "resolution": resolution,
                    "bounds": [
                        [south, west],
                        [south, east],
                        [north, east],
                        [north, west],
                    ],
                    "center_lat": round((south + north) / 2, 6),
                    "center_lng": round((west + east) / 2, 6),
                }
            )
    return cells
