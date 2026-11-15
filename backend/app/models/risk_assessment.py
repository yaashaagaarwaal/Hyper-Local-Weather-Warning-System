import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, CreatedAtMixin


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class RiskAssessment(Base, CreatedAtMixin):
    """One fused risk result for one grid cell at one point in time — the
    Hyper-Local Grid record described in docs/architecture.md §6/§7.

    `hazard_scores` is a JSON dict (e.g. {"heavyRain": 22, "thunderstorm": 46,
    "lightning": 12, "flood": 61}) rather than fixed columns, because the
    frontend's current mock taxonomy (4 categories) and the fixed ML
    pipeline's Multi-Task Head (3 categories: thunderstorm/cloudburst/
    flash-flood) haven't been reconciled yet (architecture.md §12) — this
    keeps the schema correct under either outcome without a migration.

    Append-only: a new row is written each time risk is reassessed for a
    cell, instead of updating one row in place, so history/trends are
    queryable later.
    """

    __tablename__ = "risk_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    grid_cell_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("grid_cells.id", ondelete="CASCADE"), nullable=False
    )
    assessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    overall_score: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_level: Mapped[RiskLevel] = mapped_column(Enum(RiskLevel, name="risk_level"), nullable=False)
    dominant_hazard: Mapped[str | None] = mapped_column(String(50), nullable=True)
    hazard_scores: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Phase 3: "measured" (fresh, nearby real observation backing this),
    # "regional_estimate" (real observation, but distant and/or stale), or
    # "insufficient_data" (no usable observation at all) — see
    # app/services/risk_engine/ for exactly how this is derived.
    evidence_quality: Mapped[str | None] = mapped_column(String(30), nullable=True)
    # e.g. "mock" (Phase 1 placeholder) or "rule-based-v1" (Phase 3 engine,
    # see docs/architecture.md §7) today; "risk-fusion-v1" once a real
    # ML-informed fusion engine exists.
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="mock")

    grid_cell: Mapped["GridCell"] = relationship(back_populates="risk_assessments")

    __table_args__ = (
        Index("ix_risk_assessments_grid_cell_assessed", "grid_cell_id", "assessed_at"),
    )

    def __repr__(self) -> str:
        return f"<RiskAssessment grid={self.grid_cell_id} score={self.overall_score}>"
