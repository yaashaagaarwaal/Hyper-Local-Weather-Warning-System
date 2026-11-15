"""Seeds reference data only: Location rows (1:1 port of the frontend's
src/data/locations.js) and GridCell rows (geometry from app/utils/grid.py).

Deliberately does NOT seed WeatherObservation, WeatherForecast,
RiskAssessment, Alert, or ModelPrediction — those are real/time-series data
that must come from an actual API call, a real risk computation, or a real
model, never fabricated (see Phase 1 instructions).

Run with: python -m app.db.seed
"""

import logging

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models import GridCell, Location
from app.utils.grid import build_grid_geometry

logger = logging.getLogger(__name__)

# 1:1 port of LOCATIONS in src/data/locations.js — keep these two lists in sync.
LOCATIONS = [
    {"id": "loc-ndls", "name": "New Delhi", "region": "Delhi", "lat": 28.6139, "lng": 77.209, "is_primary": True, "saved_as": None},
    {"id": "loc-sec62", "name": "Sector 62", "region": "Noida", "lat": 28.628, "lng": 77.3649, "is_primary": False, "saved_as": "home"},
    {"id": "loc-ggn", "name": "Cyber Hub", "region": "Gurugram", "lat": 28.4949, "lng": 77.0891, "is_primary": False, "saved_as": "work"},
    {"id": "loc-du", "name": "North Campus", "region": "Delhi University", "lat": 28.6874, "lng": 77.2098, "is_primary": False, "saved_as": "college"},
    {"id": "loc-noida", "name": "Noida", "region": "Uttar Pradesh", "lat": 28.5355, "lng": 77.391, "is_primary": False, "saved_as": None},
    {"id": "loc-fbd", "name": "Faridabad", "region": "Haryana", "lat": 28.4089, "lng": 77.3178, "is_primary": False, "saved_as": None},
    {"id": "loc-ghz", "name": "Ghaziabad", "region": "Uttar Pradesh", "lat": 28.6692, "lng": 77.4538, "is_primary": False, "saved_as": None},
    {"id": "loc-dwk", "name": "Dwarka", "region": "Delhi", "lat": 28.5921, "lng": 77.046, "is_primary": False, "saved_as": None},
    {"id": "loc-rkp", "name": "Rohini", "region": "Delhi", "lat": 28.7495, "lng": 77.0565, "is_primary": False, "saved_as": None},
    {"id": "loc-saket", "name": "Saket", "region": "Delhi", "lat": 28.5245, "lng": 77.2066, "is_primary": False, "saved_as": None},
    {"id": "loc-agra", "name": "Agra", "region": "Uttar Pradesh", "lat": 27.1767, "lng": 78.0081, "is_primary": False, "saved_as": None},
]


def seed_locations(db: Session) -> int:
    created = 0
    for loc in LOCATIONS:
        if db.get(Location, loc["id"]):
            continue
        db.add(
            Location(
                id=loc["id"],
                name=loc["name"],
                region=loc["region"],
                latitude=loc["lat"],
                longitude=loc["lng"],
                is_primary=loc["is_primary"],
                saved_as=loc["saved_as"],
            )
        )
        created += 1
    db.commit()
    return created


def seed_grid_cells(db: Session) -> int:
    created = 0
    for cell in build_grid_geometry():
        if db.get(GridCell, cell["id"]):
            continue
        db.add(GridCell(**cell))
        created += 1
    db.commit()
    return created


def run() -> None:
    db = SessionLocal()
    try:
        n_locations = seed_locations(db)
        n_grid = seed_grid_cells(db)
        logger.info("Seeded %d locations, %d grid cells", n_locations, n_grid)
        print(f"Seeded {n_locations} new location(s), {n_grid} new grid cell(s).")
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
