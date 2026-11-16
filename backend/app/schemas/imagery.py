from datetime import datetime

from app.schemas.weather import CamelModel


class UploadedImageOut(CamelModel):
    asset_id: str
    original_filename: str
    size_bytes: int
    width: int
    height: int
    url: str
    uploaded_at: datetime


class UploadedVideoOut(CamelModel):
    asset_id: str
    original_filename: str
    size_bytes: int
    frame_count: int
    fps: float | None
    duration_seconds: float | None
    width: int
    height: int
    url: str
    uploaded_at: datetime


class ProcessedOutputOut(CamelModel):
    operation: str
    url: str
    note: str | None = None


class ProcessingResultOut(CamelModel):
    result_id: str
    source_image_id: str
    outputs: list[ProcessedOutputOut]
    stats: dict
    generated_at: datetime


class CompareResultOut(CamelModel):
    result_id: str
    before_image_id: str
    after_image_id: str
    heatmap_url: str
    mean_abs_diff: float
    changed_pixel_pct: float
    generated_at: datetime


class MotionResultOut(CamelModel):
    result_id: str
    before_image_id: str
    after_image_id: str
    visualization_url: str
    mean_magnitude_px: float
    dominant_direction_deg: float
    dominant_direction_label: str
    note: str
    generated_at: datetime


class ExtractedFrameOut(CamelModel):
    index: int
    timestamp_seconds: float
    image_id: str
    url: str


class FrameExtractionResultOut(CamelModel):
    video_id: str
    frames: list[ExtractedFrameOut]
    frame_count: int


class VideoGenerateRequest(CamelModel):
    image_ids: list[str]
    fps: float = 2.0


class GeneratedVideoOut(CamelModel):
    result_id: str
    url: str
    frame_count: int
    fps: float
    source_image_ids: list[str]
    note: str
    generated_at: datetime


class NowcastRequest(CamelModel):
    """Run the INSAT-3DR infrared nowcast from `anchor_frame_id` (the latest
    observed frame; the two half-hourly frames before it are used as well)."""
    anchor_frame_id: str
    region: str = "north_india"
