from sqlalchemy import Index, Integer, JSON, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class GridCell(Base, TimestampMixin):
    """One cell of the Hyper-Local Grid (mirrors RISK_GRID in src/data/riskData.js).

    Geometry is stored as plain JSON (`bounds`: a list of [lat, lng] pairs) rather
    than a PostGIS geometry column, to keep Phase 1 running on vanilla PostgreSQL
    with no extra extensions. Migrating `bounds`/`center` to PostGIS geometry
    columns is a natural later upgrade (see docs/architecture.md §6) once the GIS
    Engine needs real spatial queries (nearest-neighbor, polygon intersection).
    """

    __tablename__ = "grid_cells"

    # e.g. "GRID-001" for the default 7x7 resolution, matching the frontend's
    # id format exactly; "GRID-<resolution>-001" for any other resolution
    # (see app/utils/grid.py) so multiple resolutions can coexist without id
    # collisions.
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    row: Mapped[int] = mapped_column(Integer, nullable=False)
    col: Mapped[int] = mapped_column(Integer, nullable=False)
    # Phase 3: e.g. "7x7" — lets more than one grid resolution exist at once
    # (see app/utils/grid.py's configurable rows/cols) without needing a
    # separate table per resolution.
    resolution: Mapped[str] = mapped_column(String(20), nullable=False, server_default="7x7")
    # [[south, west], [south, east], [north, east], [north, west]]
    bounds: Mapped[list] = mapped_column(JSON, nullable=False)
    center_lat: Mapped[float] = mapped_column(Numeric(9, 6), nullable=False)
    center_lng: Mapped[float] = mapped_column(Numeric(9, 6), nullable=False)

    risk_assessments: Mapped[list["RiskAssessment"]] = relationship(
        back_populates="grid_cell", cascade="all, delete-orphan"
    )
    model_predictions: Mapped[list["ModelPrediction"]] = relationship(
        back_populates="grid_cell", cascade="all, delete-orphan"
    )
    weather_associations: Mapped[list["GridWeatherAssociation"]] = relationship(
        back_populates="grid_cell", cascade="all, delete-orphan"
    )
    alerts: Mapped[list["Alert"]] = relationship(
        back_populates="grid_cell", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_grid_cells_row_col", "row", "col"),
        Index("ix_grid_cells_resolution", "resolution"),
    )

    def __repr__(self) -> str:
        return f"<GridCell {self.id}>"
