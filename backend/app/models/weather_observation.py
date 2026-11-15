from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, CreatedAtMixin


class WeatherObservation(Base, CreatedAtMixin):
    """A single real current-weather reading for a location.

    Field names mirror `normalizeCurrent()` in the frontend's
    src/services/weatherService.js exactly, so a future `/weather/current`
    endpoint can serialize a row from this table with (almost) no reshaping.

    Append-only: observations are never edited, only inserted, so this model
    has no `updated_at`.
    """

    __tablename__ = "weather_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    location_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("locations.id", ondelete="CASCADE"), nullable=False
    )
    # The weather provider's own observation timestamp (not ingestion time).
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    feels_like_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_direction_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure_hpa: Mapped[float | None] = mapped_column(Float, nullable=True)
    visibility_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    rainfall_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    rain_probability_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    condition: Mapped[str | None] = mapped_column(String(50), nullable=True)
    uv_index: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Phase 3: raw WMO code + standard convective-instability indicators,
    # used by the rule-based risk engine (app/services/risk_engine/).
    weather_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cape_j_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    lifted_index: Mapped[float | None] = mapped_column(Float, nullable=True)
    # e.g. "open-meteo" — which provider this reading came from.
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="open-meteo")

    location: Mapped["Location"] = relationship(back_populates="weather_observations")

    __table_args__ = (
        Index("ix_weather_observations_location_recorded", "location_id", "recorded_at"),
        # Phase 2: prevents duplicate rows when the same reading is fetched
        # more than once (e.g. two requests landing in the same provider
        # timestep) — ingestion upserts on this instead of blind INSERT.
        UniqueConstraint(
            "location_id", "recorded_at", "source", name="uq_weather_observations_location_recorded_source"
        ),
    )

    def __repr__(self) -> str:
        return f"<WeatherObservation location={self.location_id} at={self.recorded_at}>"
