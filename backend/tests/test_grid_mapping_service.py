from datetime import datetime, timedelta, timezone

from app.models import GridCell, Location, WeatherObservation
from app.services import grid_mapping_service

NOW = datetime.now(timezone.utc)


def _location(db_session, suffix, lat, lng):
    loc = Location(id=f"loc-map-{suffix}", name=f"Test {suffix}", region="Test", latitude=lat, longitude=lng)
    db_session.add(loc)
    db_session.flush()
    return loc


def _grid_cell(db_session, suffix, center_lat, center_lng):
    cell = GridCell(
        id=f"GRID-MAP-{suffix}", row=0, col=0, resolution="test",
        bounds=[[center_lat - 0.05, center_lng - 0.05], [center_lat - 0.05, center_lng + 0.05],
                [center_lat + 0.05, center_lng + 0.05], [center_lat + 0.05, center_lng - 0.05]],
        center_lat=center_lat, center_lng=center_lng,
    )
    db_session.add(cell)
    db_session.flush()
    return cell


def _observation(db_session, location_id, recorded_at=NOW, **overrides):
    defaults = dict(
        temperature_c=25.0, humidity_pct=60.0, wind_speed_kmh=10.0, rainfall_mm=0.0,
        rain_probability_pct=10.0, pressure_hpa=1013.0, condition="Clear Sky",
        weather_code=0, cape_j_kg=100.0, lifted_index=5.0, source="open-meteo",
    )
    defaults.update(overrides)
    obs = WeatherObservation(location_id=location_id, recorded_at=recorded_at, **defaults)
    db_session.add(obs)
    db_session.flush()
    return obs


def test_nearest_location_is_used_when_only_one_is_close(db_session):
    near = _location(db_session, "near", 28.60, 77.20)
    far = _location(db_session, "far", 28.90, 77.60)  # >30km away
    cell = _grid_cell(db_session, "A", 28.605, 77.205)
    _observation(db_session, near.id, temperature_c=30.0)
    _observation(db_session, far.id, temperature_c=10.0)

    mapping = grid_mapping_service.map_grid_weather(db_session, cell, [near, far])

    assert mapping.data_available is True
    assert mapping.nearest_location.id == near.id
    assert mapping.observation_count == 1
    assert mapping.weather_input.temperature_c == 30.0


def test_multiple_nearby_locations_are_averaged(db_session):
    loc_a = _location(db_session, "avgA", 28.60, 77.20)
    loc_b = _location(db_session, "avgB", 28.61, 77.21)  # ~1.5km from loc_a
    cell = _grid_cell(db_session, "B", 28.605, 77.205)
    _observation(db_session, loc_a.id, temperature_c=20.0, rainfall_mm=0.0)
    _observation(db_session, loc_b.id, temperature_c=30.0, rainfall_mm=10.0)

    mapping = grid_mapping_service.map_grid_weather(db_session, cell, [loc_a, loc_b])

    assert mapping.observation_count == 2
    assert mapping.weather_input.temperature_c == 25.0  # average of 20 and 30
    assert mapping.weather_input.rainfall_mm == 5.0


def test_single_distant_location_is_used_as_fallback_not_averaged(db_session):
    loc = _location(db_session, "lonely", 28.75, 77.35)  # ~22km away: beyond CLOSE_RADIUS_KM (15) but within MAX_USABLE_RADIUS_KM (45)
    cell = _grid_cell(db_session, "C", 28.60, 77.20)
    _observation(db_session, loc.id, temperature_c=22.0)

    mapping = grid_mapping_service.map_grid_weather(db_session, cell, [loc])

    assert mapping.data_available is True
    assert mapping.observation_count == 1
    assert mapping.nearest_distance_km > 15  # outside CLOSE_RADIUS_KM
    assert mapping.weather_input.temperature_c == 22.0


def test_missing_data_when_no_location_within_usable_radius(db_session):
    loc = _location(db_session, "toofar", 10.0, 70.0)  # thousands of km away
    cell = _grid_cell(db_session, "D", 28.60, 77.20)
    _observation(db_session, loc.id)

    mapping = grid_mapping_service.map_grid_weather(db_session, cell, [loc])

    assert mapping.data_available is False
    assert mapping.weather_input.data_available is False


def test_missing_data_when_no_locations_given(db_session):
    cell = _grid_cell(db_session, "E", 28.60, 77.20)
    mapping = grid_mapping_service.map_grid_weather(db_session, cell, [])
    assert mapping.data_available is False


def test_missing_data_when_location_exists_but_has_no_observations(db_session):
    loc = _location(db_session, "noobs", 28.60, 77.20)
    cell = _grid_cell(db_session, "F", 28.605, 77.205)
    mapping = grid_mapping_service.map_grid_weather(db_session, cell, [loc])
    assert mapping.data_available is False
    assert mapping.nearest_location.id == loc.id  # we know it's nearest, just has no readings


def test_staleness_check_flags_old_observations(db_session):
    loc = _location(db_session, "stale", 28.60, 77.20)
    cell = _grid_cell(db_session, "G", 28.605, 77.205)
    _observation(db_session, loc.id, recorded_at=NOW - timedelta(hours=3))

    mapping = grid_mapping_service.map_grid_weather(db_session, cell, [loc])

    assert mapping.data_available is True
    assert mapping.weather_input.is_stale is True


def test_fresh_observation_is_not_stale(db_session):
    loc = _location(db_session, "fresh", 28.60, 77.20)
    cell = _grid_cell(db_session, "H", 28.605, 77.205)
    _observation(db_session, loc.id, recorded_at=NOW - timedelta(minutes=5))

    mapping = grid_mapping_service.map_grid_weather(db_session, cell, [loc])

    assert mapping.weather_input.is_stale is False


def test_latest_observation_is_used_when_location_has_several(db_session):
    loc = _location(db_session, "history", 28.60, 77.20)
    cell = _grid_cell(db_session, "I", 28.605, 77.205)
    _observation(db_session, loc.id, recorded_at=NOW - timedelta(hours=2), temperature_c=15.0)
    _observation(db_session, loc.id, recorded_at=NOW, temperature_c=28.0)

    mapping = grid_mapping_service.map_grid_weather(db_session, cell, [loc])

    assert mapping.weather_input.temperature_c == 28.0


def test_observation_cache_is_reused_across_calls(db_session):
    loc = _location(db_session, "cached", 28.60, 77.20)
    cell_a = _grid_cell(db_session, "J1", 28.605, 77.205)
    cell_b = _grid_cell(db_session, "J2", 28.606, 77.206)
    _observation(db_session, loc.id, temperature_c=27.0)

    cache = {}
    mapping_a = grid_mapping_service.map_grid_weather(db_session, cell_a, [loc], cache)
    assert loc.id in cache
    # Mutate the cache to prove the second call actually reads from it
    # rather than re-querying.
    cache[loc.id].temperature_c = 99.0
    mapping_b = grid_mapping_service.map_grid_weather(db_session, cell_b, [loc], cache)

    assert mapping_a.weather_input.temperature_c == 27.0
    assert mapping_b.weather_input.temperature_c == 99.0
