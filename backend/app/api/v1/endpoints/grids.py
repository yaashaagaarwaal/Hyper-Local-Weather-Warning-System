from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.db.session import get_db
from app.repositories import grid_repository
from app.schemas.grid import GridCellOut

router = APIRouter(prefix="/grids", tags=["grids"])


@router.get("", response_model=list[GridCellOut])
def list_grids(
    resolution: str = Query("7x7", description="Grid resolution label, e.g. '7x7'"),
    south: float | None = Query(None, ge=-90, le=90, description="Bounding box south latitude"),
    north: float | None = Query(None, ge=-90, le=90, description="Bounding box north latitude"),
    west: float | None = Query(None, ge=-180, le=180, description="Bounding box west longitude"),
    east: float | None = Query(None, ge=-180, le=180, description="Bounding box east longitude"),
    db: Session = Depends(get_db),
) -> list[GridCellOut]:
    """All grid cells at the given resolution, optionally filtered to a
    geographic bounding box (south/west/north/east — all 4 required together
    to filter; omit all 4 for the whole grid).
    """
    bbox = None
    if None not in (south, north, west, east):
        bbox = (south, west, north, east)
    cells = grid_repository.list_grid_cells(db, resolution=resolution, bbox=bbox)
    return cells


@router.get("/{grid_id}", response_model=GridCellOut)
def get_grid(grid_id: str, db: Session = Depends(get_db)) -> GridCellOut:
    cell = grid_repository.get_grid_cell(db, grid_id)
    if cell is None:
        raise NotFoundError(f"Unknown grid id: {grid_id!r}")
    return cell
