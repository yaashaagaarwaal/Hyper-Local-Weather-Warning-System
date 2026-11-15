from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, CreatedAtMixin


class GridWeatherAssociation(Base, CreatedAtMixin):
    """One record of "this grid cell's weather, as of this computation" —
    the persisted result of nearest-coordinate weather-to-grid mapping
    (see app/services/grid_mapping_service.py).

    Append-only, like WeatherObservation/RiskAssessment: each time a grid's
    weather is (re)computed, a new row is written rather than updating one in
    place, so `grid_weather_associations` doubles as a history of how the
    mapping evolved (which station(s) were nearest, how far, how stale).

    Does NOT duplicate the actual weather values (temperature, rainfall, …) —
    those stay in `WeatherObservation`, referenced by `weather_observation_id`
    where a single nearest station was used, or left null and summarized only
    via the `*_summary` columns when multiple nearby observations were
    averaged (see the mapping service for when each path is taken).
    """

    __tablename__ = "grid_weather_associations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    grid_cell_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("grid_cells.id", ondelete="CASCADE"), nullable=False
    )
    # The single nearest location, if any was within range. Null means no
    # location was close enough to use at all (missing-data case).
    location_id: Mapped[str | None] = mapped_column(
        String(50), ForeignKey("locations.id", ondelete="SET NULL"), nullable=True
    )
    weather_observation_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("weather_observations.id", ondelete="SET NULL"), nullable=True
    )
    distance_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    # How many locations' observations contributed (0 = no data available,
    # 1 = single nearest station, >1 = averaged across nearby stations).
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Grid-level weather summary (averaged across `observation_count`
    # contributing stations when >1; equal to the single station's reading
    # when ==1; null when 0).
    temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    rainfall_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure_hpa: Mapped[float | None] = mapped_column(Float, nullable=True)
    condition: Mapped[str | None] = mapped_column(String(50), nullable=True)

    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_stale: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    data_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    grid_cell: Mapped["GridCell"] = relationship(back_populates="weather_associations")

    __table_args__ = (
        Index("ix_grid_weather_associations_grid_created", "grid_cell_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<GridWeatherAssociation grid={self.grid_cell_id} n={self.observation_count}>"
