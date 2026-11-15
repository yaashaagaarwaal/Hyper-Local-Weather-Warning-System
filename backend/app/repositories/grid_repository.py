from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import GridCell, GridWeatherAssociation, RiskAssessment


def list_grid_cells(
    db: Session,
    resolution: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> list[GridCell]:
    """`bbox` is (south, west, north, east) in degrees, filtering by each
    cell's center point falling inside it."""
    stmt = select(GridCell)
    if resolution:
        stmt = stmt.where(GridCell.resolution == resolution)
    if bbox:
        south, west, north, east = bbox
        stmt = stmt.where(
            GridCell.center_lat >= south,
            GridCell.center_lat <= north,
            GridCell.center_lng >= west,
            GridCell.center_lng <= east,
        )
    stmt = stmt.order_by(GridCell.row, GridCell.col)
    return list(db.execute(stmt).scalars())


def get_grid_cell(db: Session, grid_id: str) -> GridCell | None:
    return db.get(GridCell, grid_id)


def list_resolutions(db: Session) -> list[str]:
    rows = db.execute(select(GridCell.resolution).distinct()).scalars()
    return sorted(set(rows))


def save_grid_cells(db: Session, cells: list[dict]) -> int:
    """Inserts any cells (by dict, as returned by app.utils.grid.build_grid_geometry)
    not already present. Used for generating a new resolution on demand.
    """
    created = 0
    for cell in cells:
        if db.get(GridCell, cell["id"]):
            continue
        db.add(GridCell(**cell))
        created += 1
    db.commit()
    return created


def save_weather_association(db: Session, association: GridWeatherAssociation) -> GridWeatherAssociation:
    db.add(association)
    db.commit()
    db.refresh(association)
    return association


def latest_weather_association(db: Session, grid_cell_id: str) -> GridWeatherAssociation | None:
    return db.execute(
        select(GridWeatherAssociation)
        .where(GridWeatherAssociation.grid_cell_id == grid_cell_id)
        .order_by(GridWeatherAssociation.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def save_risk_assessment(db: Session, assessment: RiskAssessment) -> RiskAssessment:
    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    return assessment


def latest_risk_assessment(db: Session, grid_cell_id: str) -> RiskAssessment | None:
    return db.execute(
        select(RiskAssessment)
        .where(RiskAssessment.grid_cell_id == grid_cell_id)
        .order_by(RiskAssessment.assessed_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def risk_history(
    db: Session, grid_cell_id: str, since: datetime | None = None, limit: int = 100
) -> list[RiskAssessment]:
    stmt = select(RiskAssessment).where(RiskAssessment.grid_cell_id == grid_cell_id)
    if since:
        stmt = stmt.where(RiskAssessment.assessed_at >= since)
    stmt = stmt.order_by(RiskAssessment.assessed_at.desc()).limit(limit)
    return list(db.execute(stmt).scalars())
