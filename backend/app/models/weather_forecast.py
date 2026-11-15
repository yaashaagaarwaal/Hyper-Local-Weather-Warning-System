import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, CreatedAtMixin


class ForecastHorizon(str, enum.Enum):
    HOURLY = "hourly"
    DAILY = "daily"


class WeatherForecast(Base, CreatedAtMixin):
    """A single forecast point for a location — one row per (location, horizon,
    forecast_for) combination.

    Covers both `getHourlyForecast` (horizon=HOURLY, one row per hour, ~24h out)
    and `getDailyForecast` (horizon=DAILY, one row per day, ~7d out) from
    src/services/weatherService.js in one flexible table rather than two near-
    identical ones. `temperature_high_c` / `temperature_low_c` are only
    populated for daily rows; `temperature_c` is only populated for hourly rows.
    """

    __tablename__ = "weather_forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    location_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("locations.id", ondelete="CASCADE"), nullable=False
    )
    horizon: Mapped[ForecastHorizon] = mapped_column(
        Enum(ForecastHorizon, name="forecast_horizon"), nullable=False
    )
    # The timestamp (hourly) or date-at-midnight (daily) this row forecasts.
    forecast_for: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    temperature_high_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    temperature_low_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    rainfall_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    rain_probability_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    condition: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="open-meteo")

    location: Mapped["Location"] = relationship(back_populates="weather_forecasts")

    __table_args__ = (
        Index(
            "ix_weather_forecasts_location_horizon_for",
            "location_id",
            "horizon",
            "forecast_for",
        ),
        # Phase 2: re-fetching the same forecast point upserts (refreshing
        # generated_at + values) instead of inserting a duplicate row.
        UniqueConstraint(
            "location_id", "horizon", "forecast_for", "source",
            name="uq_weather_forecasts_location_horizon_for_source",
        ),
    )

    def __repr__(self) -> str:
        return f"<WeatherForecast location={self.location_id} horizon={self.horizon} for={self.forecast_for}>"
