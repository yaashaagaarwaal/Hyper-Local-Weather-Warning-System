# AtmoGuard — Hyper-Local Weather Warning System

**AtmoGuard** is a hyper-local weather intelligence and early-warning system built for
SIH Problem Statement 26077. It divides Delhi-NCR into a configurable spatial grid and
combines **real weather data, a transparent rule-based risk engine, 12 trained ML
prediction models, real historical flood-vulnerability data, an automated alert
pipeline, and a classical computer-vision satellite/video imagery module** into one
working dashboard.

> This README describes what is **actually implemented and running** today, not a
> future plan. Every claim below has a corresponding test or was verified against a
> live server — see [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) for the detailed,
> chronological build log and [CLAUDE.md](CLAUDE.md) for the full technical reference.

## What's real vs. what's a documented prototype

| Capability | Status |
|---|---|
| Current/hourly/daily weather, 11 Delhi-NCR locations | **Real** — live Open-Meteo, backend-first with a direct fallback |
| Hyper-local risk grid (7×7, 49 cells) + 4-hazard rule-based risk engine | **Real** — cited WMO/IMD thresholds, computed on demand |
| Temperature/rainfall/heavy-rain/thunderstorm predictions (1h/3h/6h) | **Real, trained ML** — XGBoost/linear, validation-selected, honestly weak on rainfall (documented) |
| Flood/lightning hazards | **Rule-based only** — no suitable training data exists; never presented as ML |
| Historical flood vulnerability (20 Delhi-NCR districts) | **Real** — India Flood Inventory data, feeds a bounded risk-score modifier |
| Automated alerts (heavy rain/thunderstorm/flood) + notification feed | **Real** — scheduled pipeline, DB-enforced duplicate prevention |
| Satellite/precipitation imagery, image & video processing | **Real data, classical CV** — 18 real NASA GIBS images, OpenCV pipeline, **no trained CV model** (documented, not fabricated) |
| Ground radar | **Not available** — no accessible public source found; the UI says so explicitly |
| 0–60min "nowcast" prediction timeline | **Prototype/simulated** — labeled "Prototype Prediction" in the UI; no model exists at 15-min granularity |

## Architecture

```
NASA GIBS (satellite/precip)   Open-Meteo (weather)      India Flood Inventory
          │                           │                      + geoBoundaries/LGD
          ▼                           ▼                           │
  dataset/imagery/ (18 real    WeatherService (cache,              ▼
   images, metadata)            retry, dedup, DB persist)   20-district flood
          │                           │                      vulnerability layer
          ▼                           ▼                           │
 OpenCV pipeline (denoise,     Hyper-Local Grid (7×7)              │
  contrast, cloud/storm        + grid↔weather mapping              │
  detection, optical flow,            │                            │
  segmentation, video)                ▼                            │
          │                 Rule-based Risk Engine ◄────────────────┘
          │              (heavy rain / thunderstorm /
          │               lightning / flood, 0-100 each)
          │                           │              ┌─ ML Prediction Pipeline
          │                           │              │  (12 trained models:
          │                           ▼              ▼   4 targets × 3 horizons)
          │                 Alert-Generation Pipeline (scheduled + on-demand)
          │                           │
          ▼                           ▼
        FastAPI backend (/api/v1/*) ──┴── React frontend (9 pages, one dashboard)
```

Every indicator the frontend shows is tagged by **which of these four subsystems
produced it** — rule-based (`measured`/`forecast`/`official_warning`/
`historical_vulnerability` categories), trained ML (a separate "Validated Model
Forecast" panel, never mixed with the rule-based score), or classical image-processing
heuristic (its own "Prototype"-style banner + per-operation note). See
[CLAUDE.md](CLAUDE.md)'s "Architecture: real data vs. simulated data" section for the
full table.

## Project layout

```
backend/    FastAPI + PostgreSQL — see backend/README.md
frontend/   React 19 + Vite + Tailwind — see CLAUDE.md
dataset/    Acquired datasets: ERA5/GPM scripts, India Flood Inventory,
            geoBoundaries/LGD crosswalk, dataset/imagery/ (NASA GIBS samples)
docs/       architecture.md, model_report.md, dataset_analysis.md
```

## Installation & setup

Requires **Python 3.12**, **Node 18+**, and a local **PostgreSQL** instance.

### 1. Backend

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # includes opencv-python-headless for the imagery module

cp .env.example .env             # edit DATABASE_URL if your Postgres isn't on localhost:5432

createdb atmoguard
createdb atmoguard_test          # used automatically by the test suite

alembic upgrade head
python -m app.db.seed            # seeds 11 demo locations + the 49-cell 7×7 grid (reference
                                  # data only — no weather/risk/alert rows are seeded)
```

### 2. Frontend

```bash
cd frontend
npm install
# Works with zero config — defaults to http://localhost:8000/api/v1. To override,
# copy frontend/.env.example to .env.local (Vite auto-loads it).
```

### 3. Imagery dataset (optional — already included)

The 18 real NASA GIBS sample images already ship in `dataset/imagery/`. To re-fetch a
fresh set (no API key needed — verified empirically, see
`dataset/imagery/metadata/SOURCES.md`):

```bash
python3 dataset/scripts/fetch_gibs_imagery.py
```

## Running it

```bash
# Terminal 1 — backend (also starts the Phase 5 alert scheduler every 15 min;
# add PIPELINE_ENABLED=false before uvicorn to disable it during development)
cd backend && source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev
```

Open **http://localhost:5173**. Interactive API docs: **http://localhost:8000/docs**.

To see real alerts immediately rather than waiting for the 15-minute scheduler:

```bash
curl -X POST http://localhost:8000/api/v1/alerts/process
```

### Demo flow

`Dashboard` → click a grid cell on the live risk map → `Risk Map` for the full view
(toggle the real flood-vulnerability district overlay) → `Predictions` to compare the
rule-based hazard score against the real trained-model forecast for the same location
→ `Imagery` to run the classical CV pipeline (cloud detection, optical flow, a real
time-lapse) on real satellite/precipitation samples → `Alerts` for any active bulletins
→ `Locations` to see every monitored station's live weather, risk level, and historical
flood vulnerability together.

## Testing

```bash
# Backend — 222 tests (also exercises some real Open-Meteo calls)
cd backend && source .venv/bin/activate
pytest -v                    # everything
pytest -m "not network" -v   # skip real-API tests, offline-friendly

# Frontend
cd frontend
npm run build                # production build
npm run lint                 # oxlint
```

There is no automated frontend test suite — frontend changes are verified by running
the app and checking the browser console/network tab directly (see PROJECT_HANDOFF.md
for what was checked and when).

## Known limitations (see PROJECT_HANDOFF.md for the full, current list)

- **Heavy-rain-amount ML prediction is honestly weak** (~1-3% precision) — flagged
  `lowConfidence: true` in every API response; alerts use the rule-based score instead.
- **No flood or lightning ML model** — no suitable ground-truth data exists; both stay
  rule-based, reported under `unsupportedHazards` in every prediction response.
- **No ground radar imagery anywhere in this project** — no accessible public source was
  found for Delhi-NCR; the Imagery page says so explicitly rather than faking one.
- **No trained computer-vision model** in the imagery module (cloud/storm detection is
  classical OpenCV thresholding/clustering) — 18 images is nowhere near enough to train
  one, and none is claimed.
- **ERA5/GPM raw-swath download is blocked** on NASA/CDS credentials that don't exist in
  this environment — the acquisition scripts exist and are unit-tested, just not run.
- **117 districts outside Delhi-NCR remain unmatched** in the geoBoundaries↔LGD
  crosswalk — Delhi-NCR itself (the project's actual scope) is fully resolved.
- Single-process only: the in-memory weather cache and the alert scheduler are not
  coordinated across multiple `uvicorn` workers.
- No authentication, no persisted ML prediction history (see `docs/model_report.md` §8
  for why), no mobile app, no email/SMS delivery (the notification abstraction exists
  but has no configured provider).

## Disclaimer

AtmoGuard is a research/prototype system built for a hackathon (SIH). It is not a
certified weather service — treat every "rule-based", "prototype", or "low-confidence"
label in the UI as a hard boundary on what the number actually means.
