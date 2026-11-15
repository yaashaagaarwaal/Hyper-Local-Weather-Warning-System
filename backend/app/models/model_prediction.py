from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, CreatedAtMixin


class ModelPrediction(Base, CreatedAtMixin):
    """Raw output of the (future) Multi-Task Head for one grid cell, at one
    forecast horizon — see docs/architecture.md §5.

    This is deliberately separate from RiskAssessment: this table holds the
    fixed, 3-head model output (thunderstorm / cloudburst / flash-flood
    probabilities) exactly as specified by the pipeline, while RiskAssessment
    holds the *fused* result after the Risk Fusion Engine combines this with
    static vulnerability data. Nothing writes to this table yet — Phase 1
    only defines the shape the ML pipeline will write into.
    """

    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    grid_cell_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("grid_cells.id", ondelete="CASCADE"), nullable=False
    )
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # The timestamp this prediction is FOR (predicted_at + horizon_minutes).
    valid_for: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    horizon_minutes: Mapped[int] = mapped_column(Integer, nullable=False)

    thunderstorm_probability: Mapped[float] = mapped_column(Float, nullable=False)
    cloudburst_probability: Mapped[float] = mapped_column(Float, nullable=False)
    flash_flood_probability: Mapped[float] = mapped_column(Float, nullable=False)

    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    grid_cell: Mapped["GridCell"] = relationship(back_populates="model_predictions")

    __table_args__ = (
        Index("ix_model_predictions_grid_cell_valid_for", "grid_cell_id", "valid_for"),
    )

    def __repr__(self) -> str:
        return f"<ModelPrediction grid={self.grid_cell_id} valid_for={self.valid_for}>"
