from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Location


def get_by_id(db: Session, location_id: str) -> Location | None:
    return db.get(Location, location_id)


def list_all(db: Session) -> list[Location]:
    return list(db.execute(select(Location).order_by(Location.name)).scalars())
