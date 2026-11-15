from datetime import datetime

from app.schemas.weather import CamelModel


class GridCellOut(CamelModel):
    id: str
    row: int
    col: int
    resolution: str
    bounds: list[list[float]]
    center_lat: float
    center_lng: float
    created_at: datetime
