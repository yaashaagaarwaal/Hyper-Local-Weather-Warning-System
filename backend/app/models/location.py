from sqlalchemy import Boolean, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Location(Base, TimestampMixin):
    """A monitored point location (mirrors src/data/locations.js on the frontend).

    `id` intentionally matches the frontend's string ids (e.g. "loc-ndls") so
    seed data can be a 1:1 port of locations.js, and so a future endpoint can
    serve this table directly as the `/locations` response.
    """

    __tablename__ = "locations"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    region: Mapped[str] = mapped_column(String(100), nullable=False)
    latitude: Mapped[float] = mapped_column(Numeric(9, 6), nullable=False)
    longitude: Mapped[float] = mapped_column(Numeric(9, 6), nullable=False)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # "home" | "work" | "college" | NULL — matches SAVED_LOCATION_META keys in locations.js
    saved_as: Mapped[str | None] = mapped_column(String(20), nullable=True)

    weather_observations: Mapped[list["WeatherObservation"]] = relationship(
        back_populates="location", cascade="all, delete-orphan"
    )
    weather_forecasts: Mapped[list["WeatherForecast"]] = relationship(
        back_populates="location", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Location {self.id} {self.name!r}>"
