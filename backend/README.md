# AtmoGuard Backend

FastAPI backend for AtmoGuard (SIH Problem Statement 26077).

- **Phase 1**: app skeleton, configuration, logging, centralized error handling, health
  checks, and the full database schema.
- **Phase 2**: a real, working weather ingestion service — async Open-Meteo integration
  (current/hourly/daily, single + batched locations), caching, retry/backoff, database
  storage with dedup, the `/weather/*` endpoints, and the frontend wired to call this
  backend first (with an automatic fallback to Open-Meteo directly if unreachable).
- **Phase 3**: a configurable spatial grid engine (`/grids/*`) and a transparent, rule-based
  weather risk engine (`/risk/*`) — 4 hazard modules scored against cited meteorological
  thresholds, built entirely on Phase 2's real weather data.
- **Phase 4**: a real, trained machine-learning weather prediction pipeline
  (`/predictions/*`) — baseline models (persistence/linear/random forest) plus XGBoost,
  trained on the historical CSV dataset for temperature/rainfall regression and
  heavy-rain/thunderstorm classification at 1h/3h/6h horizons. See
  [`docs/model_report.md`](../docs/model_report.md) for the full report.
- **Phase 5 (this phase, final integration)**: an automated alert-generation engine
  (`/alerts/*`) wired to the real rule-based risk engine and, where genuinely validated, the
  real ML predictions — with real duplicate prevention, a real active/resolved/expired
  lifecycle, a scheduled end-to-end processing pipeline (configurable interval, verified
  duplicate-run prevention), and an in-app notification feed. The frontend is now wired to
  every one of these real endpoints (weather, grids/risk, predictions, alerts) instead of its
  original mock data.

No GIS heatmap/safe-zones, grid-clustering, real-time push, or mobile/SMS dispatch yet — see
[`docs/architecture.md`](../docs/architecture.md) for the full roadmap,
[`docs/dataset_analysis.md`](../docs/dataset_analysis.md) for the historical CSV dataset's
general analysis, and [`docs/model_report.md`](../docs/model_report.md) for the Phase 4 model
development report.

## Stack

Python 3.12 · FastAPI · Uvicorn · PostgreSQL · SQLAlchemy 2.x (sync) · Alembic ·
Pydantic v2 / pydantic-settings · HTTPX (async) · Pytest / pytest-asyncio / respx ·
pandas / scikit-learn / XGBoost / joblib (Phase 4)

## Project layout

```
backend/
  app/
    main.py                 FastAPI app, CORS, lifespan (shared httpx.AsyncClient), routing
    core/
      config.py              Settings — DB, CORS, HTTP timeout/retry, cache TTL, IMD key
      logging.py               console logging setup
      exceptions.py             centralized exception handlers (AppError/404/422/500/503)
    api/
      deps.py                   get_weather_service(), resolve_location() — shared deps
      v1/router.py                aggregates all v1 endpoint routers
      v1/endpoints/
        health.py                  GET /health, /health/db
        weather.py                  GET /weather/current, /current/batch, /hourly, /daily, /locations
        grids.py                    GET /grids, /grids/{id}
        risk.py                     GET /risk/summary, /risk/grids, /risk/grids/{id}
        predictions.py               GET /predictions/{grid_id}, /predictions/summary (Phase 4)
        alerts.py                    /alerts/* (Phase 5)
    models/                   SQLAlchemy models (Phase 1, +constraints/columns added in Phase 2/3,
                                 alerts redesigned in Phase 5)
    schemas/
      weather.py                CurrentWeather, HourlyForecast, DailyForecast, WeatherLocation
      grid.py                    GridCellOut
      risk.py                    RiskAssessmentOut, HazardAssessmentOut, RiskSummaryOut
      prediction.py              GridPredictionOut, RegressionPredictionOut, ClassificationPredictionOut (Phase 4)
      alert.py                   AlertOut, PipelineRunOut, NotificationOut (Phase 5)
                                 — all camelCase wire format, matching the frontend's own field names
    services/
      weather_service.py         orchestrates: cache → provider → persistence → response
      cache.py                    in-memory TTL cache + request de-duplication
      grid_mapping_service.py     nearest-coordinate weather-to-grid mapping (Phase 3)
      risk_service.py             orchestrates: mapping → risk_engine → persistence (Phase 3)
      prediction_service.py       orchestrates: nearest location → live history → features → model (Phase 4)
      alert_service.py            real hazard evidence (rule-based + validated ML) → alert
                                   create/update/resolve, duplicate prevention, lifecycle (Phase 5)
      pipeline.py                  the scheduled end-to-end processing pipeline (Phase 5)
      notification_service.py     NotificationChannel abstraction — in-app real, WS/email/SMS
                                   refuse to send (Phase 5)
      providers/
        base.py                    WeatherProvider ABC — the provider abstraction
        open_meteo.py               the only working provider (async, retried, logged);
                                     Phase 4 added get_feature_history() for ML lag features
        imd.py                      stub — NOT implemented, see its docstring for why
      risk_engine/                (Phase 3 — transparent, rule-based, no ML)
        types.py                    WeatherInput, HazardAssessment, RiskResult, EvidenceQuality
        thresholds.py                every cited meteorological threshold, in one place
        scoring_utils.py              piecewise_score() — linear interpolation between bands
        heavy_rainfall.py, thunderstorm.py, lightning.py, flood.py   — the 4 hazard modules
        aggregator.py                combines the 4 into one RiskResult (overall = max)
    ml/                       (Phase 4 — inference ONLY, see backend/ml/ below for training)
      model_registry.py          loads backend/ml/artifacts/*.joblib, caches in memory
      feature_builder.py          live weather history → the named feature values a model expects
    repositories/
      location_repository.py
      weather_repository.py        upserts (dedup via DB unique constraints)
      grid_repository.py           grid CRUD + weather-association/risk-assessment persistence
      alert_repository.py          alert CRUD + lifecycle transitions (Phase 5)
    utils/
      grid.py                   Hyper-Local Grid geometry — configurable resolution (Phase 3)
      geo.py                     haversine_km() (Phase 3)
      weather_codes.py           WMO code → text, degrees → compass (ports weatherCodes.js)
    db/
      session.py, seed.py
  ml/                        (Phase 4 — TRAINING ONLY, never imported by the running app
                               except for its constants/config — see its own docstrings)
    config.py                   feature lists, horizons, thresholds, split dates, paths
    preprocessing.py             load/clean/engineer features/chronological split
    targets.py                   forward-looking regression + classification target construction
    models.py                    persistence/linear/random_forest/xgboost/mlp factories
    evaluate.py                   regression + classification metrics
    train.py                      the training/comparison/artifact-saving orchestrator
    artifacts/                    {target}_h{horizon}_{model,scaler,metadata}.{joblib,json}
                                   + model_comparison.json (every algorithm's metrics)
  scripts/
    analyze_dataset.py         one-off dataset analysis (produces docs/dataset_analysis.md)
  alembic/                   migrations
  tests/                     176 tests, see "Testing" below
  requirements.txt / .env.example / Dockerfile / docker-compose.yml
```

## Setup

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env — at minimum set DATABASE_URL for your local Postgres
```

## Database

Requires a running PostgreSQL instance (no extensions required — vanilla Postgres,
no PostGIS needed yet; see `app/models/grid_cell.py` for why).

```bash
createdb atmoguard          # dev database
createdb atmoguard_test     # used automatically by the test suite

alembic upgrade head        # applies the schema (Phase 1 tables + Phase 2/3 additions)
python -m app.db.seed       # seeds the 11 demo Location rows + 49 GridCell rows (the
                             # default "7x7" resolution) — reference data only, NOT
                             # weather/risk data

# Optional: generate an additional grid at a different resolution (coexists with 7x7):
python -c "
from app.db.session import SessionLocal
from app.repositories.grid_repository import save_grid_cells
from app.utils.grid import build_grid_geometry
db = SessionLocal()
print(save_grid_cells(db, build_grid_geometry(rows=10, cols=10)), 'cells created')
"
```

To create a new migration after changing a model:

```bash
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```

## Environment variables

See [`.env.example`](.env.example) for the full list.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg2://localhost:5432/atmoguard` | SQLAlchemy connection string |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated list of allowed frontend origins |
| `API_V1_PREFIX` | `/api/v1` | Prefix for the versioned API |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `APP_ENV` | `development` | `development` \| `test` \| `production` |
| `OPEN_METEO_BASE_URL` | `https://api.open-meteo.com/v1/forecast` | Weather provider endpoint |
| `HTTP_TIMEOUT_SECONDS` | `10.0` | Per-request timeout to Open-Meteo |
| `HTTP_RETRY_ATTEMPTS` | `2` | Retries after the first attempt (3 total) |
| `HTTP_RETRY_BACKOFF_SECONDS` | `0.8` | Backoff multiplier between retries |
| `WEATHER_CACHE_TTL_SECONDS` | `300` | In-memory cache TTL (matches the frontend's own cache) |
| `IMD_API_KEY` | unset | IMD provider stays disabled until a real key is obtained — see `app/services/providers/imd.py` |
| `PIPELINE_ENABLED` | `true` | Phase 5: starts the scheduled alert-processing loop in `main.py`'s lifespan. Always force-disabled under `APP_ENV=test` regardless of this value |
| `PIPELINE_INTERVAL_SECONDS` | `900` | How often the scheduled pipeline runs (15 min default) |

## Running the API

```bash
uvicorn app.main:app --reload --port 8000
```

This also starts the Phase 5 scheduler (unless `PIPELINE_ENABLED=false`), which runs the full
ingestion → risk → prediction → alert pipeline once immediately and then every
`PIPELINE_INTERVAL_SECONDS`. Watch it run in the server logs (`pipeline.run_complete {...}`),
or trigger one cycle on demand without waiting: `curl -X POST
http://localhost:8000/api/v1/alerts/process`.

- Liveness: `GET /health` — always 200 if the process is up, never touches the DB.
- Readiness: `GET /health/db` — runs `SELECT 1`; returns 503 if the database is unreachable.
- Weather: `GET /api/v1/weather/current?locationId=loc-ndls` (or `?lat=..&lon=..`),
  `/current/batch?locationIds=loc-ndls,loc-agra`, `/hourly?locationId=..&hours=24`,
  `/daily?locationId=..&days=7`, `/locations`.
- Grids: `GET /api/v1/grids?resolution=7x7` (optionally `&south=&north=&west=&east=` for a
  bbox filter), `/grids/{grid_id}`.
- Risk: `GET /api/v1/risk/summary?resolution=7x7`, `/risk/grids?resolution=7x7`,
  `/risk/grids/{grid_id}` — **fetch current weather for the 11 locations at least once
  first** (any `/weather/current*` call), so there's a `WeatherObservation` row for the risk
  engine's grid mapping to find; otherwise every grid reports `dataAvailable: false`.
- Predictions (Phase 4): `GET /api/v1/predictions/{grid_id}`,
  `/predictions/summary?resolution=7x7&horizon=3` — needs no prior `/weather/*` call (it
  fetches its own live history from Open-Meteo per request), but does need a `Location`
  within 45km of the grid cell; otherwise returns `dataAvailable: false` with empty lists.
- Alerts (Phase 5): `GET /api/v1/alerts`, `/alerts/active`, `/alerts/history`, `/alerts/{id}`
  (accepts `ALT-0001` or a bare integer), `POST /alerts/{id}/acknowledge`,
  `POST /alerts/{id}/resolve`, `GET /alerts/notifications`, `POST /alerts/process` (runs one
  pipeline cycle immediately — the easiest way to populate real alerts for a demo without
  waiting for the scheduler or for real severe weather).
- Interactive docs: `http://localhost:8000/docs` (Swagger) or `/redoc`.

### Example requests

```bash
curl "http://localhost:8000/api/v1/weather/current?locationId=loc-ndls"
curl "http://localhost:8000/api/v1/weather/current?lat=27.1767&lon=78.0081"   # Agra, by raw coords
curl "http://localhost:8000/api/v1/weather/current/batch?locationIds=loc-ndls,loc-noida,loc-sec62,loc-ghz,loc-agra"
curl "http://localhost:8000/api/v1/weather/hourly?locationId=loc-sec62&hours=6"
curl "http://localhost:8000/api/v1/weather/daily?locationId=loc-ghz&days=3"
curl "http://localhost:8000/api/v1/weather/locations"

curl "http://localhost:8000/api/v1/grids?resolution=7x7" | python3 -m json.tool | head -20
curl "http://localhost:8000/api/v1/grids/GRID-025"

# Seed real weather first (see note above), then:
curl "http://localhost:8000/api/v1/risk/summary"
curl "http://localhost:8000/api/v1/risk/grids/GRID-025" | python3 -m json.tool

# Predictions (Phase 4) — no prior weather seeding needed:
curl "http://localhost:8000/api/v1/predictions/GRID-025" | python3 -m json.tool
curl "http://localhost:8000/api/v1/predictions/summary?horizon=3" | python3 -m json.tool

# Alerts (Phase 5) — run the pipeline once to generate real alerts from current conditions,
# then list them (an empty list on a calm day is the correct, honest result):
curl -X POST "http://localhost:8000/api/v1/alerts/process" | python3 -m json.tool
curl "http://localhost:8000/api/v1/alerts/active" | python3 -m json.tool
curl "http://localhost:8000/api/v1/alerts/history" | python3 -m json.tool
curl -X POST "http://localhost:8000/api/v1/alerts/ALT-0001/acknowledge"
curl -X POST "http://localhost:8000/api/v1/alerts/ALT-0001/resolve"
```

## Testing

```bash
pytest -v                   # everything, including real Open-Meteo calls (needs internet)
pytest -m "not network" -v  # skip the real-API tests (offline-friendly, mocked-only)
```

The suite (176 tests) runs against a **separate** `atmoguard_test` database —
`tests/conftest.py` sets `DATABASE_URL` before any app module is imported, and seeds the
default 49-cell "7x7" grid once per test session. The schema is dropped and recreated at the
start of the test session (not just created-then-dropped — this avoids a stale
`alembic_version` row if you run `alembic` commands against the same DB between pytest runs).

Covers everything from Phases 1–2 (config, health/readiness, model relationships, the
Open-Meteo provider against the **real API** for all 5 required locations, retry/backoff
against a mocked transport, caching, every `/weather/*` endpoint end-to-end) plus, for
Phase 3:

- **Grid generation** (`test_grid_utils.py`): the default grid reproduces the exact original
  49 `GRID-001`..`GRID-049` ids; custom resolutions get distinct, non-colliding ids; cells
  tile the bounding box edge-to-edge with no gaps/overlaps; center points fall inside their
  own bounds; custom row/col counts and custom bounding boxes are respected; haversine
  distance is checked against a known real-world value (Delhi↔Agra).
- **Weather-to-grid mapping** (`test_grid_mapping_service.py`): nearest-single-location
  selection, averaging across ≥2 nearby locations, a single distant location used as
  fallback (not averaged) when it's the only one in range, missing-data detection (no
  location at all / nothing within the usable radius / a location with zero observations),
  staleness detection (fresh vs. >1h old), "most recent observation wins" when a location has
  several, and the shared observation cache actually being reused across calls.
- **Risk engine** (`test_risk_engine.py`, 37 tests): every score↔level threshold boundary;
  each of the 4 hazard modules independently (rainfall rate drives the rainfall/flood score,
  humidity modifies flood, weather code/CAPE/Lifted Index each independently drive the
  thunderstorm score, lightning never exceeds thunderstorm's score for the same input and
  its notes/indicators never claim "measured" or detection); the max-of-hazards aggregation
  rule (a severe single hazard isn't diluted by calm co-hazards); confidence/evidence-quality
  derived correctly from distance and staleness; the full "no data available" path.
- **API + persistence** (`test_grid_risk_api.py`): `/grids` resolution and bbox filtering,
  404s, `/risk/grids/{id}` reflecting real seeded severe weather with the correct dominant
  hazard, every request persisting a `RiskAssessment` **and** a `GridWeatherAssociation` row,
  repeated requests accumulating history rather than overwriting, and `/risk/summary`'s
  counts always summing to the total grid count.

Plus, for Phase 4 (33 tests — see `docs/model_report.md` §10 for a longer narrative):

- **Preprocessing pipeline** (`test_ml_preprocessing.py`, 10 tests): the two leakage columns
  found during this phase's data review (`rainfall_daily_total_mm`, the proxy-label columns)
  are excluded from the feature set; the high-missingness column is dropped, not imputed,
  while trailing-rain NaNs are zero-filled; lag features never cross a station boundary; the
  chronological split never overlaps and stays time-ordered; forward-value and window-max
  target construction only ever use strictly later rows, and return NaN (never a guessed
  value) when the forward window runs past the end of a station's series; the thunderstorm
  target's weather-code-or-CAPE/LI rule.
- **Serving layer** (`test_ml_serving.py`, 19 tests): live-history → feature-row construction
  succeeds with a complete history, refuses when too much is missing, and handles empty
  history; trailing-rain windows sum only genuinely past hours; safe column reordering with a
  missing key; and — parametrized across all 4 targets × 3 horizons — every artifact
  `train.py` actually produced loads correctly and its `.predict()`/`.predict_proba()` output
  shape matches its own declared feature count.
- **Predictions API** (`test_predictions_api.py`, 4 tests, through the real app + real Open-Meteo):
  unknown grid → 404; a grid with no station within 45km → `dataAvailable: false` with empty
  lists (200, not a crash); a grid with a real nearby station → a real prediction response
  with all 4 targets × all trained horizons, including the documented `lowConfidence: true`
  on every heavy-rain-risk prediction; the summary endpoint's counts.

Plus, for Phase 5 (28 tests):

- **Alert evidence & lifecycle** (`test_alert_service.py`, 13 tests): heavy-rainfall and flood
  evidence is always `"rule-based-v1"` only; thunderstorm evidence combines the rule-based
  score with a confident ML prediction (`source="combined"`) but ignores a low-confidence one;
  lightning is asserted to never be in the alertable-hazards set; a HIGH/SEVERE hazard creates
  a real `Alert` row with the right fields; LOW/MODERATE never creates one; a repeated HIGH
  condition updates the same alert rather than duplicating it (checked both via the service's
  return value and by querying `list_active()` directly); conditions subsiding resolves the
  active alert; an overdue alert is swept to EXPIRED; acknowledge/resolve helpers; the
  `ALT-0042` public-id format.
- **Notification channels** (`test_notification_service.py`, 7 tests): the in-app channel
  records and returns most-recent-first and respects a max size; WebSocket/email/SMS each
  raise `NotConfiguredError` rather than silently no-op; the service delivers to configured
  channels and skips unconfigured ones without raising; the default service uses in-app.
- **Alerts API** (`test_alerts_api.py`, 8 tests, through the real app + real database):
  active/history listing, lookup by both the public `ALT-0001` id and a bare integer, 404 on
  an unknown id, the full acknowledge-then-resolve lifecycle through the API (including that
  acknowledgement survives resolution), status/hazard-type filtering, the notifications
  endpoint, and — marked `network` — a real end-to-end pipeline trigger through
  `POST /alerts/process` against live Open-Meteo data, verified to run to completion and
  leave `/alerts/active` queryable afterwards regardless of what the real weather turned out
  to be.
- **Live, hand-verified** (not part of the automated suite, but run against a real `uvicorn`
  process before considering this phase complete): the scheduler starting and completing a
  real pipeline run on startup; two concurrent manual `/alerts/process` triggers, one of which
  correctly reports `"skipped": true` (duplicate-run prevention); every `/alerts/*` endpoint;
  CORS preflight headers.

## Docker

```bash
docker compose up --build
```

Starts Postgres 17 + the API together (`docker-compose.yml`). Not run/verified in this
environment (Docker was not available on the development machine) — review before relying
on it for a real deployment.

## Weather ingestion architecture (Phase 2)

```
Endpoint (weather.py)
  → WeatherService (weather_service.py) — orchestration only
      → TTLCache.get_or_fetch()  (cache.py — in-memory, per-process, request-deduplicating)
          → WeatherProvider.get_current/hourly/daily()  (providers/base.py — the abstraction)
              → OpenMeteoProvider  (providers/open_meteo.py — the only concrete one today)
                  → httpx.AsyncClient (shared, created in main.py's lifespan)
      → weather_repository.save_*()  (only when the request is for a known Location —
         upserts via Postgres ON CONFLICT, keyed by the Phase-2 unique constraints, so
         re-fetching the same reading never creates a duplicate row)
  ← CurrentWeather / HourlyForecast / DailyForecast  (schemas/weather.py — camelCase,
     carries observedAt vs. retrievedAt vs. servedFromCache so a cached response never
     hides how stale its underlying reading actually is)
```

**Caching note**: the cache is in-process (a dict + asyncio locks), not Redis — fine for a
single-worker deployment, but **not shared across multiple worker processes**
(`uvicorn --workers N`). See `cache.py`'s docstring.

**IMD**: researched, not integrated. IMD's API (`api.imd.gov.in`) is documented but gated
behind an API key this project doesn't have (confirmed by actually calling it —
`{"error":"API key missing"}`, HTTP 401). `providers/imd.py` implements the same
`WeatherProvider` interface as a disabled stub, so a real integration later is a one-file
change, not a redesign.

## Spatial grid & risk engine architecture (Phase 3)

```
Endpoint (grids.py / risk.py)
  → risk_service.assess_and_persist() / assess_many()
      → grid_mapping_service.map_grid_weather()
          — nearest-coordinate: finds the Location(s) nearest a GridCell's center,
            averages ≥2 if within CLOSE_RADIUS_KM (15km), else uses the single
            nearest (if within MAX_USABLE_RADIUS_KM, 45km), else data_available=False
          → reads WeatherObservation (real Phase 2 data) — never fabricates one
      → risk_engine.assess_grid(WeatherInput)
          → heavy_rainfall.assess() + thunderstorm.assess() + lightning.assess()
            + flood.assess()  (each independent, each cites its own thresholds)
          → aggregator: overall = max(hazard scores); confidence/evidence_quality
            from distance+staleness (never from the scores themselves)
      → grid_repository.save_weather_association() + save_risk_assessment()
         (both append-only — every call adds a history row)
  ← RiskAssessmentOut (schemas/risk.py — score, level, dominant hazard, per-hazard
     indicators with a measured/forecast/official_warning category, a weatherBasis
     block showing exactly what the score is and isn't based on)
```

### Grid resolution

A plain lat/lng lattice (`app/utils/grid.py`), not H3/S2 hexagons — needs no new dependency
and matches what the frontend map already renders as rectangular Leaflet polygons. Default
`rows=7, cols=7` over the same Delhi/NCR bounding box as the frontend's mock grid, producing
the exact original `GRID-001`..`GRID-049` ids (nothing already referencing them breaks).
Any other `rows`/`cols` (optionally a different `bounds` box) produces `GRID-<res>-NNN` ids
that coexist in the same table without collision — pass `resolution` explicitly or it's
derived as `"<rows>x<cols>"`.

### Risk scoring formulas (all thresholds cited in `app/services/risk_engine/thresholds.py`)

| Hazard | Primary signal(s) | Formula |
|---|---|---|
| **Heavy rainfall** | Current rainfall rate (mm/hr, measured) + forecast rain probability | `score = piecewise(rate, WMO/IMD hourly bands) × 0.7 + probability × 0.3` |
| **Thunderstorm** | WMO weather code 95/96/99 (forecast) + CAPE (forecast) + Lifted Index (forecast) | `score = max(code_score, cape_score, lifted_index_score)` — any one strong signal is enough |
| **Lightning** | *Derived from the Thunderstorm score* — **not** an independent measurement | `score = thunderstorm_score × 0.9` (a documented, deliberately conservative dampening factor — lightning is never scored higher than thunderstorm for the same input) |
| **Flood** | Same rainfall rate as above + relative humidity (saturated-air proxy) | `score = piecewise(rate, same bands) + humidity_bonus` (up to +15 pts when humidity > 80%) |
| **Overall** | — | `overall = max(the 4 hazard scores)`; `dominant_hazard` = whichever hazard produced it. **Not an average** — a severe single hazard is never diluted by calm co-hazards. |

Score → category uses the same `low <30 / moderate <55 / high <80 / severe ≥80` bands as the
frontend's `frontend/src/utils/risk.js` and the `RiskLevel` enum already on `RiskAssessment` —
`score_to_level()` is the single place this is defined for the backend.

**Confidence** (0.0–1.0) and **evidence quality** (`measured` / `regional_estimate` /
`insufficient_data`) come entirely from the grid-mapping step's own distance/staleness facts,
never from how high a hazard score happens to be — see `aggregator.py`'s `_confidence()` /
`_evidence_quality()` for the exact formula.

**Honesty constraints enforced in code, not just docs:**
- Lightning's indicators/notes can never claim `"measured"` or `"official_warning"` category,
  and its score can never exceed the Thunderstorm score for the same input (tested directly).
- `weatherBasis.dataAvailable=false` + `evidenceQuality="insufficient_data"` whenever no
  Location is within 45km of a grid (never silently substitutes a distant/fabricated value).
- `distanceKm` and `observationCount` are always present in the API response, so "this is a
  coarse, station-based estimate" is never hidden behind a clean-looking score.

### Known limitations (see docs/architecture.md §7 for more)

- Rainfall scoring uses the *current hourly rate* only — no rolling 3h/6h/24h accumulation in
  the live pipeline yet (unlike the historical CSV dataset, which has it).
- Flood scoring is a rainfall+humidity proxy, not a hydrological/drainage/terrain model.
- Risk is computed **on-demand per request**, not on a schedule — there is no background job
  yet (explicitly out of scope for Phase 3).

## ML prediction pipeline architecture (Phase 4)

Full detail, every formula, every metric: [`docs/model_report.md`](../docs/model_report.md).
This section is the short version.

```
Training (offline, run by hand — backend/ml/, never imported by the running app's logic)
  ml/preprocessing.py  — load CSV → clean → lag/calendar/spatial features → chronological split
  ml/targets.py         — forward-looking regression + classification targets
  ml/train.py            — for each (target, horizon): fit persistence/linear/RF/XGBoost/(MLP
                            at the flagship horizon) → select best by validation score → save

Serving (online, inside the FastAPI app — app/ml/)
  Endpoint (predictions.py)
    → prediction_service.predict_for_grid()
        → nearest Location by distance (same MAX_USABLE_RADIUS_KM as the risk engine)
        → weather_service.get_feature_history()  — LIVE Open-Meteo call, past_hours=73
            (not the database — see docs/model_report.md §8 for why)
        → feature_builder.build_feature_values()  — nearest-in-time match per lag/window,
            refuses (None) if too much of the row is missing
        → model_registry.get_model(target, horizon)  — loads the saved .joblib + metadata
        → scaler.transform() → model.predict() / predict_proba()
    ← GridPredictionOut (schemas/prediction.py — temperature/rainfall/heavyRainRisk/
       thunderstormRisk lists, a basis block mirroring risk.py's weatherBasis, and
       unsupportedHazards for flood/lightning)
```

### Targets, horizons, and the winning algorithm (validation-selected; full table in model_report.md §6)

| Target | Type | Horizons | Best algorithm (3h) | Test metric (3h) |
|---|---|---|---|---|
| `temperature_c` | regression | 1h, 3h, 6h | XGBoost | R² 0.977 |
| `rainfall_mm_hour` | regression | 1h, 3h, 6h | Linear | R² 0.027 (honestly weak — see below) |
| `heavy_rain_risk` (≥7.6mm/hr in the window) | classification | 1h, 3h, 6h | Linear | PR-AUC 0.034, precision ~1% (honestly weak) |
| `thunderstorm_risk` (WMO code or CAPE/LI) | classification | 1h, 3h, 6h | XGBoost | PR-AUC 0.878, precision 66%, recall 90% |

**Why XGBoost over a deep spatio-temporal model** (ConvLSTM/CNN-LSTM/Transformer, the
long-term spec in `docs/architecture.md` §5): 5 point stations in one city is not an
independent spatial raster a convolution should learn over, and the severe class imbalance in
the hazard targets (0.18% heavy-rain rows in the raw dataset) doesn't support a heavier
sequence model without synthetically balancing it — which would itself be fabricating weather
events. A small MLP was tried as a genuine neural-network comparison point; it never won
against XGBoost on any target. Full reasoning: `docs/model_report.md` §5.

**Live feature parity**: `OpenMeteoProvider.get_feature_history()` (Phase 4 addition) was
verified, by directly calling the Open-Meteo API, to return every one of this pipeline's 21
base weather columns — including the three soil-moisture depth bins — with no special access,
so serving-time features are built from the same columns training used, not a reduced subset.

### Known limitations (see `docs/model_report.md` §9 for the full, honest list)

- **Heavy-rain risk is not actionable alone** (~1–3% precision at every horizon) — a direct
  consequence of how rare heavy rain is in this dataset (309 / 175,320 hours), not a bug.
  Every response flags it `lowConfidence: true`; keep relying on the Phase 3 rule-based
  `heavy_rainfall.py` module for this hazard in the meantime.
- **Rainfall-amount regression explains very little variance** (test R² between −0.74 and
  0.03 depending on horizon) — the same zero-inflation problem.
- **No flood or lightning ML model** — no suitable ground truth/sensor data exists for either;
  both stay on the Phase 3 rule-based engine (`unsupportedHazards` in every response).
- **Predictions are not persisted** — `ModelPrediction`'s fixed 3-column shape (from the
  original spec) doesn't match what this pipeline outputs; see model_report.md §8.

## Alert-generation engine & processing pipeline architecture (Phase 5)

```
Scheduled (asyncio loop, main.py lifespan) or on-demand (POST /alerts/process):
  app/services/pipeline.py  run_pipeline_once()
    1-2. WeatherService.get_current_batch()      — real ingestion + Pydantic validation
    3-4. risk_service.assess_many()               — real grid mapping + rule-based risk
    5.   prediction_service.predict_for_many()     — real ML, best-effort (never blocks 3-4)
    6-7. alert_service.sync_alerts_for_grid()       — real alert create/update/resolve
         + alert_service.expire_due_alerts()
    8.   NotificationService.notify()                — in-app feed (notification_service.py)
```

- **Evidence, per hazard** (full reasoning in `alert_service.py`'s module docstring):
  - **Heavy rainfall** — the rule-based risk engine's score only (`source="rule-based-v1"`).
    The Phase 4 ML `heavy_rain_risk` classifier is deliberately *not* used here — it has
    ~1-3% precision at every horizon (model_report.md §9), so using it to trigger warnings
    would mean ~97-99% false alarms.
  - **Thunderstorm** — the rule-based score, OR the real, validated ML `thunderstorm_risk`
    classifier (PR-AUC ~0.88 at its 3h horizon), whichever is more severe
    (`source="rule-based-v1"` / `"ml-v1"` / `"combined"`, recorded exactly).
  - **Lightning** — **never generates an alert.** No lightning sensor or authoritative
    warning feed exists; the rule-based "lightning" score is a derived thunderstorm-
    favorability indicator, not independent evidence (`ALERTABLE_HAZARDS` excludes it, and a
    dedicated test asserts this).
  - **Flood** — the rule-based score only (a rainfall+humidity proxy, not a hydrological
    model). No ML flood model exists (no suitable ground truth).
- **Thresholds**: only HIGH or SEVERE (`score_to_level()`'s existing bands — Phase 3) creates
  or keeps an alert.
- **Severity**: `Alert.severity` reuses `RiskLevel` directly (low/moderate/high/severe) —
  the same enum `RiskAssessment.risk_level` already uses, not a separate vocabulary.
- **Duplicate prevention**: a real DB constraint, not just application logic — a partial
  unique index (`ux_alerts_active_key`) allows at most one ACTIVE row per
  `hazard_type:grid_cell_id` key. `sync_alerts_for_grid()` looks up the existing active alert
  for a key first and updates it (risk score, evidence, `expires_at`, `last_seen_at`) instead
  of inserting — verified by running the same conditions through the pipeline twice and
  checking only one row exists.
- **Lifecycle**: ACTIVE → RESOLVED (the next pipeline run finds the hazard no longer HIGH/
  SEVERE) or → EXPIRED (`expires_at` passes — 3h for heavy-rain/thunderstorm, 6h for flood —
  without being re-confirmed by a later run). Acknowledgement (`acknowledged`,
  `acknowledged_at`) is orthogonal to status — an alert can be ACTIVE and acknowledged at the
  same time.
- **Scheduling**: a single `asyncio` task in `main.py`'s lifespan (not a separate worker
  process/queue — consistent with this project's "avoid unnecessary dependencies/
  infrastructure" default), interval configurable via `PIPELINE_INTERVAL_SECONDS`, disabled
  under `APP_ENV=test` so the test suite never starts a real, repeating polling loop. An
  `asyncio.Lock` makes a tick that fires mid-run skip itself (logged) rather than overlap —
  verified live by firing two concurrent manual triggers.
- **Notifications**: `NotificationChannel` abstraction (`notification_service.py`) — in-app
  is a real in-process feed (`GET /alerts/notifications`); WebSocket/email/SMS are stubs that
  raise `NotConfiguredError` on `send()` rather than silently no-op, per the explicit
  instruction not to send real external notifications without a configured, tested provider.

### Known limitations

- **One alert per grid cell per hazard**, not clustered — if 5 adjacent cells all cross HIGH
  for the same hazard at once, that's 5 alerts, not one bulletin covering the area. Clustering
  adjacent severe grids is future work (docs/architecture.md §12).
- **In-app notifications only** — no WebSocket push (the frontend polls `/alerts/active`
  instead) and no email/SMS (no provider configured — see above).
- **No background-job framework** — the scheduler is a single `asyncio` loop in one process;
  it does not survive a process restart mid-cycle and isn't coordinated across multiple
  `uvicorn` workers (same caveat as the in-memory weather cache).
- **Heavy-rain and flood alerts inherit the rule-based engine's own limitations** (a rainfall+
  humidity proxy for flood, current-hourly-rate-only for rainfall — see Phase 3's "Known
  limitations" above) — Phase 5 did not change the underlying evidence, only how it's turned
  into a bulletin.

## Frontend integration (Phase 2 weather; Phase 5 risk/predictions/alerts)

`frontend/src/services/weatherService.js` tries this backend first for every weather call
(`fetchCurrentWeather`, `fetchHourlyTrend`, `fetchWeeklyOutlook`, `fetchAllCurrentWeather`,
`fetchLocations`), with a 3-second timeout, and falls back to calling Open-Meteo directly —
completely unchanged from before this backend existed — on any failure. The two paths are
never blended within one response (each call is fully backend or fully direct), and the
returned object's `source` field says which (`"open-meteo-backend"` vs. `"Open-Meteo"`), so
it's always auditable which one actually answered.

As of Phase 5, the same backend-first/fallback contract covers everything else too:
- `frontend/src/services/riskService.js` (new) — `/grids` + `/risk/grids`, falling back to the mock
  `RISK_GRID`; `RiskMap.jsx` shows a "Live Rule-Based Engine" vs. "Simulated" badge depending
  on which answered.
- `frontend/src/services/predictionService.js` — `fetchGridMlForecast()` calls `/predictions/{id}`
  with **no mock fallback** (resolves to `null` on failure, so the UI shows "model
  unavailable" instead of a fabricated number); the Predictions page's new "Validated Model
  Forecast" panel is deliberately a separate section from the rule-based "Hazard Probability"
  panel, so a rule-based estimate is never presented as an ML prediction.
- `frontend/src/services/alertService.js` (new) — `/alerts/active` + `/alerts/history`, falling back
  to the mock feed; `acknowledgeAlert()`/`resolveAlert()` call the real
  acknowledge/resolve endpoints. The navbar's notification badge and `NotificationPanel` now
  reflect real active-alert counts.

Set `VITE_USE_BACKEND=false` in the frontend's `.env` to force every one of the above onto
its direct/mock path only. See the root `CLAUDE.md` for the full real-vs-simulated table.

## What's NOT in Phase 5

- No `/gis/*` endpoints, no grid-clustering for alerts, no real-time WebSocket/SSE push (the
  frontend polls `/alerts/active`), no mobile app or SMS gateway — the `NotificationChannel`
  abstraction is ready for email/SMS, but neither has a configured provider (see above).
- No `/analytics/*` 1h/6h backend equivalent — those ranges stay on the frontend's direct
  Open-Meteo path.
- No IMD integration (researched and stubbed only).
- No deep-learning spatio-temporal model (ConvLSTM/CNN-LSTM/Transformer) and no flood/lightning
  ML model — see "ML prediction pipeline architecture" above and `docs/model_report.md` §5/§9
  for the checked, data-driven reasons.
- No persisted prediction history — `/predictions/*` is on-demand only (see §8 of the model
  report for why `model_predictions` still isn't written to).
- The rule-based risk engine and the ML predictions are still not fused into one score — the
  real alert engine (Phase 5) consumes both independently per hazard, which is not the same
  as the vulnerability-aware fusion docs/architecture.md §7/§12 describes as future work.
- No static vulnerability layer (terrain, drainage, historical hazard frequency) feeding into
  flood/risk scoring — purely weather-driven for now.
- No auth.
- No multi-process-safe caching or scheduling (Redis / a real job queue) — both the weather
  cache and the Phase 5 pipeline scheduler are single-process, in-memory/in-process.

## Trained image models (added after Phase 5)

Three PyTorch models, all served from one isolated worker process
(`app/services/imagery/nowcast_worker.py`) because PyTorch and XGBoost cannot share a process
on macOS. The API process never imports torch; torch tests live in `tests_torch/` and are run
by `tests/test_nowcast_isolation.py` in a subprocess, so a plain `pytest` still covers them.

| Model | Endpoints | Trained on | Report |
|---|---|---|---|
| INSAT-3DR infrared nowcast (ConvLSTM) | `/api/v1/imagery/ml/*`, `/imagery/insat/*` | 59 real INSAT-3DR frames | `docs/imagery_model_report.md` |
| Flood mapping (U-Net, Sentinel-1 radar) | `/api/v1/hazard-models/flood/*` | 80 real Sen1Floods11 chips | `docs/flood_lightning_model_report.md` |
| Lightning detection (U-Net, infrared) | `/api/v1/hazard-models/lightning/*` | GOES-16 + GLM, Americas, 2 hours — **not validated for India** | same |

Also `GET /api/v1/hazard-models/status`, `/{model}/status`, `/{model}/metrics`,
`/{model}/figures/{name}`. Every prediction response states `engine` (`trained-ml` or
`classical-fallback`) and whether the input was a real observation or a synthetic simulation.
None of these models feed the risk engine or alerts.

```bash
python -m ml.hazards.synth_flood && python -m ml.hazards.flood_train --all && python -m ml.hazards.flood_evaluate
python -m ml.hazards.lightning_prepare && python -m ml.hazards.synth_lightning
python -m ml.hazards.lightning_train --all && python -m ml.hazards.lightning_evaluate
```

Environment: `HAZARD_MODEL_DIR` (artifact folder), `NOWCAST_DEVICE=cpu|mps|cuda` (force a
device), `FLOOD_DATA_DIR`, `GOES_DATA_DIR`, `FLOOD_SYNTHETIC_DIR`, `LIGHTNING_SYNTHETIC_DIR`.
