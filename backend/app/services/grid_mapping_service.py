"""Weather-to-grid mapping — nearest-coordinate association between a
GridCell and the real monitored `Location`s, built entirely on Phase 2's
real `WeatherObservation` data.

IMPORTANT (per Phase 3 instructions): this does NOT claim street-level
accuracy. The system has only 11 real monitored point locations (see
src/data/locations.js / app/db/seed.py) spread across Delhi/NCR — a grid
cell's weather is always the nearest real station's reading (or an average
of the nearest few, when more than one is genuinely close), explicitly
tagged with the distance and a staleness/low-confidence flag, never an
independent per-cell measurement. It also never copies one single city-wide
observation across every grid cell unexamined — each grid's nearest
station(s) are computed by real distance, so different parts of the grid
legitimately read different real stations' data, exactly as far as the
actual station density supports and no further.
"""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import GridCell, Location, WeatherObservation
from app.services.risk_engine.thresholds import CLOSE_RADIUS_KM, MAX_USABLE_RADIUS_KM, STALE_THRESHOLD_MINUTES
from app.services.risk_engine.types import WeatherInput
from app.utils.geo import haversine_km


def _latest_observation(
    db: Session, location_id: str, cache: dict[str, WeatherObservation | None] | None = None
) -> WeatherObservation | None:
    """`cache` (optional, shared across a batch of map_grid_weather calls,
    e.g. assessing all 49 grids in one request) avoids re-querying the same
    location's latest observation once per grid that happens to reference
    it — at most len(locations) queries per request instead of up to
    len(grid_cells) * contributing_locations.
    """
    if cache is not None and location_id in cache:
        return cache[location_id]
    obs = db.execute(
        select(WeatherObservation)
        .where(WeatherObservation.location_id == location_id)
        .order_by(WeatherObservation.recorded_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if cache is not None:
        cache[location_id] = obs
    return obs


def _is_stale(observed_at: datetime | None) -> bool:
    if observed_at is None:
        return True
    age = datetime.now(timezone.utc) - observed_at
    return age.total_seconds() / 60 > STALE_THRESHOLD_MINUTES


class GridWeatherMapping:
    """The result of mapping one grid cell to the real weather data nearest
    it — everything a risk assessment or an API response needs, including
    the transparency fields (distance, staleness, how many stations
    contributed) that keep this honest about its own spatial resolution.
    """

    def __init__(
        self,
        grid_cell_id: str,
        data_available: bool,
        observation_count: int,
        nearest_location: Location | None,
        nearest_distance_km: float | None,
        weather_input: WeatherInput,
        nearest_observation: WeatherObservation | None = None,
    ):
        self.grid_cell_id = grid_cell_id
        self.data_available = data_available
        self.observation_count = observation_count
        self.nearest_location = nearest_location
        self.nearest_distance_km = nearest_distance_km
        self.weather_input = weather_input
        self.nearest_observation = nearest_observation


def map_grid_weather(
    db: Session,
    grid_cell: GridCell,
    locations: list[Location],
    observation_cache: dict[str, WeatherObservation | None] | None = None,
) -> GridWeatherMapping:
    """Nearest-coordinate mapping with support for averaging across multiple
    nearby stations (when >=2 are within CLOSE_RADIUS_KM of the grid center),
    missing-data detection (no station within MAX_USABLE_RADIUS_KM), and an
    observation-freshness check.
    """
    if not locations:
        return GridWeatherMapping(
            grid_cell.id, False, 0, None, None,
            _empty_weather_input(), None,
        )

    center_lat, center_lng = float(grid_cell.center_lat), float(grid_cell.center_lng)
    distances = sorted(
        ((loc, haversine_km(center_lat, center_lng, float(loc.latitude), float(loc.longitude))) for loc in locations),
        key=lambda pair: pair[1],
    )
    nearest_loc, nearest_dist = distances[0]

    if nearest_dist > MAX_USABLE_RADIUS_KM:
        return GridWeatherMapping(
            grid_cell.id, False, 0, nearest_loc, round(nearest_dist, 2),
            _empty_weather_input(distance_km=nearest_dist), None,
        )

    close = [(loc, d) for loc, d in distances if d <= CLOSE_RADIUS_KM]
    contributing = close if len(close) >= 2 else [(nearest_loc, nearest_dist)]

    obs_pairs = []
    for loc, dist in contributing:
        obs = _latest_observation(db, loc.id, observation_cache)
        if obs is not None:
            obs_pairs.append((loc, dist, obs))

    if not obs_pairs:
        return GridWeatherMapping(
            grid_cell.id, False, 0, nearest_loc, round(nearest_dist, 2),
            _empty_weather_input(distance_km=nearest_dist), None,
        )

    def avg(attr: str) -> float | None:
        vals = [getattr(o, attr) for _, _, o in obs_pairs if getattr(o, attr) is not None]
        return sum(vals) / len(vals) if vals else None

    most_recent = max(o.recorded_at for _, _, o in obs_pairs)
    is_stale = _is_stale(most_recent)
    # Use the single nearest observation's text/categorical fields (can't
    # meaningfully average "Clear Sky" with "Overcast"), but numeric fields
    # average across every contributing station.
    nearest_obs = next(o for loc, _, o in obs_pairs if loc.id == nearest_loc.id) if any(
        loc.id == nearest_loc.id for loc, _, _ in obs_pairs
    ) else obs_pairs[0][2]

    weather_input = WeatherInput(
        temperature_c=avg("temperature_c"),
        humidity_pct=avg("humidity_pct"),
        wind_speed_kmh=avg("wind_speed_kmh"),
        rainfall_mm=avg("rainfall_mm"),
        rain_probability_pct=avg("rain_probability_pct"),
        pressure_hpa=avg("pressure_hpa"),
        condition=nearest_obs.condition,
        weather_code=nearest_obs.weather_code,
        cape_j_kg=nearest_obs.cape_j_kg,
        lifted_index=nearest_obs.lifted_index,
        observed_at=most_recent,
        is_stale=is_stale,
        distance_km=round(nearest_dist, 2),
        data_available=True,
    )

    return GridWeatherMapping(
        grid_cell.id, True, len(obs_pairs), nearest_loc, round(nearest_dist, 2),
        weather_input, nearest_obs,
    )


def _empty_weather_input(distance_km: float | None = None) -> WeatherInput:
    return WeatherInput(
        temperature_c=None, humidity_pct=None, wind_speed_kmh=None, rainfall_mm=None,
        rain_probability_pct=None, pressure_hpa=None, condition=None, weather_code=None,
        cape_j_kg=None, lifted_index=None, observed_at=None, is_stale=True,
        distance_km=distance_km, data_available=False,
    )
