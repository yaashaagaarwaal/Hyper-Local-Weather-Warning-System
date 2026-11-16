"""Image & video processing endpoints -- satellite/precipitation imagery
viewer, upload+process, video upload+frame-extraction+motion, and
time-lapse generation from genuinely sequential frames.

Two clearly separated families of endpoints live here:

1. The classical OpenCV pipeline (`/samples`, `/upload`, `/process`,
   `/compare`, `/motion`, `/video/*`) -- see processing.py / detection.py.
   No trained model is involved; every heuristic result carries a `note`
   saying so.
2. The trained deep-learning nowcast (`/ml/*`, `/insat/*`) -- a ConvLSTM
   trained on real INSAT-3DR infrared frames (see nowcast_service.py and
   docs/imagery_model_report.md). Its responses always say which engine
   produced each image ("trained-ml" vs "classical-fallback") and fall back
   to the classical optical-flow extrapolation if the model is unavailable.

Neither is a weather forecast, and neither feeds the risk engine or alerts.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, File, Query, UploadFile

from app.core.exceptions import AppError
from app.schemas.imagery import (
    CompareResultOut,
    ExtractedFrameOut,
    FrameExtractionResultOut,
    GeneratedVideoOut,
    MotionResultOut,
    NowcastRequest,
    ProcessedOutputOut,
    ProcessingResultOut,
    UploadedImageOut,
    UploadedVideoOut,
    VideoGenerateRequest,
)
from app.services.imagery import detection, nowcast_client, nowcast_service, processing, storage

router = APIRouter(prefix="/imagery", tags=["imagery"])

VALID_OPERATIONS = {"denoise", "contrast", "cloud_detection", "storm_regions", "segment_otsu", "segment_kmeans"}


@router.get("/samples")
def list_samples() -> list[dict]:
    """The small, real, pre-acquired NASA GIBS sample set (see
    dataset/imagery/metadata/SOURCES.md) -- lets the frontend show a
    satellite/precipitation viewer and run the processing pipeline with
    zero uploads required.
    """
    return storage.list_dataset_samples()


@router.post("/upload", response_model=UploadedImageOut)
async def upload_image(file: UploadFile = File(...)) -> UploadedImageOut:
    data = await file.read()
    image = processing.decode_image(data)  # validates it's a real, decodable image
    asset = storage.save_upload(data, file.filename or "upload.png", kind="image")
    info = processing.basic_info(image)
    return UploadedImageOut(
        asset_id=asset.asset_id,
        original_filename=asset.original_filename,
        size_bytes=asset.size_bytes,
        width=info["width"],
        height=info["height"],
        url=storage.runtime_media_url(asset.path),
        uploaded_at=datetime.now(timezone.utc),
    )


@router.post("/video/upload", response_model=UploadedVideoOut)
async def upload_video(file: UploadFile = File(...)) -> UploadedVideoOut:
    data = await file.read()
    asset = storage.save_upload(data, file.filename or "upload.mp4", kind="video")
    try:
        info = processing.probe_video(asset.path)
    except Exception:
        asset.path.unlink(missing_ok=True)
        (storage.UPLOADS_DIR / f"{asset.asset_id}.json").unlink(missing_ok=True)
        raise
    storage.update_upload_metadata(asset.asset_id, info)
    return UploadedVideoOut(
        asset_id=asset.asset_id,
        original_filename=asset.original_filename,
        size_bytes=asset.size_bytes,
        frame_count=info["frameCount"],
        fps=info["fps"],
        duration_seconds=info["durationSeconds"],
        width=info["width"],
        height=info["height"],
        url=storage.runtime_media_url(asset.path),
        uploaded_at=datetime.now(timezone.utc),
    )


@router.post("/process/{image_id}", response_model=ProcessingResultOut)
def process_image(
    image_id: str,
    operations: list[str] = Query(default=["denoise", "contrast", "cloud_detection"]),
) -> ProcessingResultOut:
    unknown = set(operations) - VALID_OPERATIONS
    if unknown:
        raise AppError(f"Unknown operation(s): {sorted(unknown)}. Valid: {sorted(VALID_OPERATIONS)}", status_code=422)

    data, _meta = storage.load_image_bytes_by_id(image_id)
    image = processing.resize_max_dim(processing.decode_image(data))

    result_id, out_dir = storage.new_processed_dir("process")
    outputs: list[ProcessedOutputOut] = []
    stats: dict = {"source": processing.basic_info(image), **processing.brightness_stats(image)}

    cloud_mask = None
    if "denoise" in operations:
        out = processing.denoise(image)
        path = storage.save_processed_image(processing.encode_png(out), out_dir, "denoise")
        outputs.append(ProcessedOutputOut(operation="denoise", url=storage.runtime_media_url(path)))

    if "contrast" in operations:
        out = processing.enhance_contrast(image)
        path = storage.save_processed_image(processing.encode_png(out), out_dir, "contrast")
        outputs.append(ProcessedOutputOut(operation="contrast", url=storage.runtime_media_url(path)))

    if "cloud_detection" in operations or "storm_regions" in operations:
        cloud = detection.detect_clouds(image)
        cloud_mask = cloud["mask"]
        if "cloud_detection" in operations:
            path = storage.save_processed_image(processing.encode_png(cloud["overlay"]), out_dir, "cloud_detection")
            outputs.append(ProcessedOutputOut(operation="cloud_detection", url=storage.runtime_media_url(path), note=cloud["note"]))
            stats["cloudCoveragePct"] = cloud["cloudCoveragePct"]
            stats["otsuThreshold"] = cloud["otsuThreshold"]

    if "storm_regions" in operations:
        storm = detection.identify_storm_candidate_regions(image, cloud_mask)
        path = storage.save_processed_image(processing.encode_png(storm["overlay"]), out_dir, "storm_regions")
        outputs.append(ProcessedOutputOut(operation="storm_regions", url=storage.runtime_media_url(path), note=storm["note"]))
        stats["stormCandidateRegions"] = storm["regions"]

    if "segment_otsu" in operations:
        seg = detection.segment_otsu(image)
        path = storage.save_processed_image(processing.encode_png(seg["overlay"]), out_dir, "segment_otsu")
        outputs.append(ProcessedOutputOut(operation="segment_otsu", url=storage.runtime_media_url(path), note=seg["note"]))
        stats["otsuForegroundPct"] = seg["foregroundPct"]

    if "segment_kmeans" in operations:
        seg = detection.segment_kmeans(image)
        path = storage.save_processed_image(processing.encode_png(seg["overlay"]), out_dir, "segment_kmeans")
        outputs.append(ProcessedOutputOut(operation="segment_kmeans", url=storage.runtime_media_url(path), note=seg["note"]))
        stats["kmeansClusterPct"] = seg["clusterPct"]

    generated_at = datetime.now(timezone.utc)
    result = ProcessingResultOut(
        result_id=result_id, source_image_id=image_id, outputs=outputs, stats=stats, generated_at=generated_at,
    )
    storage.save_result(out_dir, result.model_dump(by_alias=True, mode="json"))
    return result


@router.post("/compare", response_model=CompareResultOut)
def compare_images(before_id: str = Query(..., alias="beforeId"), after_id: str = Query(..., alias="afterId")) -> CompareResultOut:
    before_bytes, _ = storage.load_image_bytes_by_id(before_id)
    after_bytes, _ = storage.load_image_bytes_by_id(after_id)
    before = processing.resize_max_dim(processing.decode_image(before_bytes))
    after = processing.resize_max_dim(processing.decode_image(after_bytes))

    comparison = processing.compare_images(before, after)
    result_id, out_dir = storage.new_processed_dir("compare")
    path = storage.save_processed_image(processing.encode_png(comparison["heatmap"]), out_dir, "heatmap")

    generated_at = datetime.now(timezone.utc)
    result = CompareResultOut(
        result_id=result_id, before_image_id=before_id, after_image_id=after_id,
        heatmap_url=storage.runtime_media_url(path), mean_abs_diff=comparison["meanAbsDiff"],
        changed_pixel_pct=comparison["changedPixelPct"], generated_at=generated_at,
    )
    storage.save_result(out_dir, result.model_dump(by_alias=True, mode="json"))
    return result


@router.post("/motion", response_model=MotionResultOut)
def motion_analysis(before_id: str = Query(..., alias="beforeId"), after_id: str = Query(..., alias="afterId")) -> MotionResultOut:
    before_bytes, _ = storage.load_image_bytes_by_id(before_id)
    after_bytes, _ = storage.load_image_bytes_by_id(after_id)
    before = processing.resize_max_dim(processing.decode_image(before_bytes))
    after = processing.resize_max_dim(processing.decode_image(after_bytes))

    flow = processing.optical_flow(before, after)
    result_id, out_dir = storage.new_processed_dir("motion")
    path = storage.save_processed_image(processing.encode_png(flow["visualization"]), out_dir, "flow")

    generated_at = datetime.now(timezone.utc)
    result = MotionResultOut(
        result_id=result_id, before_image_id=before_id, after_image_id=after_id,
        visualization_url=storage.runtime_media_url(path), mean_magnitude_px=flow["meanMagnitudePx"],
        dominant_direction_deg=flow["dominantDirectionDeg"], dominant_direction_label=flow["dominantDirectionLabel"],
        note=flow["note"], generated_at=generated_at,
    )
    storage.save_result(out_dir, result.model_dump(by_alias=True, mode="json"))
    return result


@router.post("/video/{video_id}/frames", response_model=FrameExtractionResultOut)
def extract_video_frames(video_id: str, max_frames: int = Query(12, ge=1, le=60, alias="maxFrames")) -> FrameExtractionResultOut:
    path, _meta = storage.get_upload_path(video_id)
    extracted = processing.extract_frames(path, max_frames=max_frames)

    frames_out = []
    for idx, timestamp, frame_array in extracted:
        asset = storage.register_derived_image(
            processing.encode_png(frame_array),
            f"frame_{idx:04d}_from_{video_id}.png",
            {"sourceVideoId": video_id, "frameIndex": idx, "timestampSeconds": timestamp},
        )
        frames_out.append(ExtractedFrameOut(
            index=idx, timestamp_seconds=timestamp, image_id=asset.asset_id, url=storage.runtime_media_url(asset.path),
        ))

    return FrameExtractionResultOut(video_id=video_id, frames=frames_out, frame_count=len(frames_out))


@router.post("/video/generate", response_model=GeneratedVideoOut)
def generate_video(body: VideoGenerateRequest) -> GeneratedVideoOut:
    if len(body.image_ids) < 2:
        raise AppError("Need at least 2 image ids, in time order, to generate a video.", status_code=422)

    frames = []
    for image_id in body.image_ids:
        data, _meta = storage.load_image_bytes_by_id(image_id)
        frames.append(processing.resize_max_dim(processing.decode_image(data), max_dim=720))

    result_id, out_dir = storage.new_processed_dir("video")
    out_path = out_dir / "timelapse.mp4"
    info = processing.write_video(frames, out_path, fps=body.fps)

    generated_at = datetime.now(timezone.utc)
    result = GeneratedVideoOut(
        result_id=result_id, url=storage.runtime_media_url(out_path), frame_count=info["frameCount"],
        fps=info["fps"], source_image_ids=body.image_ids, generated_at=generated_at,
        note=(
            "Time-lapse encoded directly from the given image ids, in the order given. "
            "The caller is responsible for passing genuinely time-ordered frames -- "
            "this endpoint does not verify or infer timestamps."
        ),
    )
    storage.save_result(out_dir, result.model_dump(by_alias=True, mode="json"))
    return result


@router.get("/results/{result_id}")
def get_result(result_id: str) -> dict:
    return storage.read_result(result_id)


# --------------------------------------------------------------------------
# Trained deep-learning nowcast (INSAT-3DR infrared) -- see nowcast_service.py
# --------------------------------------------------------------------------
@router.get("/ml/status")
def ml_model_status() -> dict:
    """Is a trained model loaded, which one, on what device, and how did it
    score on its held-out test set. Never errors: an unavailable model is
    reported as `{"available": false, "reason": ...}`."""
    return nowcast_client.model_status()


@router.get("/insat/frames")
def list_insat_frames() -> dict:
    """Catalog of the real INSAT-3DR L1C frames on disk: which are valid,
    which are corrupt, which can anchor a nowcast, and whether each was in
    the model's training, validation or held-out test split."""
    return nowcast_service.list_frames()


@router.get("/insat/frames/{frame_id}/preview")
def preview_insat_frame(
    frame_id: str,
    region: str = Query("north_india"),
    channel: str = Query("TIR1"),
) -> dict:
    """Render one observed frame (a real measurement, no processing beyond
    calibration to brightness temperature and a grey scale)."""
    return nowcast_service.frame_preview(frame_id, region, channel)


@router.post("/ml/nowcast")
def run_ml_nowcast(body: NowcastRequest) -> dict:
    """Predict the infrared image 30/60/90 minutes after the anchor frame
    with the trained model, alongside the classical optical-flow
    extrapolation and -- where the later frame exists -- the actual
    observation and error metrics for both."""
    return nowcast_service.run_nowcast(body.anchor_frame_id, body.region)
