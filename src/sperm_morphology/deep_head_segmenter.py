from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
    from .segment_head import extract_candidate_regions, score_region_candidate
    from .utils import normalize_gray_uint8
except ImportError:
    from segment_head import extract_candidate_regions, score_region_candidate
    from utils import normalize_gray_uint8


def _deep_config(config: dict | None) -> dict:
    if config is None:
        return {}
    return config.get("deep_segmentation", {})


def _resolve_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[2]
    for base in (Path.cwd(), project_root, project_root.parent):
        candidate = base / path
        if candidate.exists():
            return candidate
    return project_root / path


@lru_cache(maxsize=2)
def _load_yolo_model(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def _mask_candidates_from_result(result, roi_shape: tuple[int, int], min_confidence: float) -> list[dict]:
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or getattr(masks, "data", None) is None:
        return []

    mask_data = masks.data
    try:
        mask_array = mask_data.detach().cpu().numpy()
    except AttributeError:
        mask_array = np.asarray(mask_data)

    confidences = []
    if boxes is not None and getattr(boxes, "conf", None) is not None:
        try:
            confidences = boxes.conf.detach().cpu().numpy().astype(float).tolist()
        except AttributeError:
            confidences = np.asarray(boxes.conf, dtype=float).tolist()

    roi_h, roi_w = roi_shape[:2]
    candidates = []
    for index, raw_mask in enumerate(mask_array):
        confidence = float(confidences[index]) if index < len(confidences) else 1.0
        if confidence < min_confidence:
            continue

        mask = (raw_mask > 0.5).astype(np.uint8) * 255
        if mask.shape[:2] != (roi_h, roi_w):
            mask = cv2.resize(mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        if int(np.count_nonzero(mask)) == 0:
            continue
        candidates.append({"mask": mask, "confidence": confidence})

    return candidates


def segment_head_with_yolo(roi_pre, target_bbox_local=None, config: dict | None = None):
    """
    Segment one ROI with YOLOv8-Seg and return a binary 0/255 head mask.

    The output contract matches segment_head.py so existing feature extraction,
    scoring, mask saving, and visualization can be reused directly.
    """
    deep_config = _deep_config(config)
    model_path = _resolve_path(deep_config.get("model_path"))
    if model_path is None or not model_path.exists():
        return None, {
            "success": False,
            "method": "yolov8_seg",
            "reason": "deep_model_not_found",
            "mask_path": "",
        }

    try:
        model = _load_yolo_model(str(model_path))
    except Exception as exc:
        return None, {
            "success": False,
            "method": "yolov8_seg",
            "reason": f"deep_model_load_failed: {exc}",
            "mask_path": "",
        }

    gray = normalize_gray_uint8(roi_pre, name="roi_pre")
    input_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    min_confidence = float(deep_config.get("conf", 0.15))
    imgsz = int(deep_config.get("imgsz", 640))
    max_det = int(deep_config.get("max_det", 8))

    try:
        result = model(
            input_image,
            conf=min_confidence,
            imgsz=imgsz,
            max_det=max_det,
            stream=False,
            verbose=False,
        )[0]
    except Exception as exc:
        return None, {
            "success": False,
            "method": "yolov8_seg",
            "reason": f"deep_predict_failed: {exc}",
            "mask_path": "",
        }

    mask_candidates = _mask_candidates_from_result(result, gray.shape, min_confidence)
    if not mask_candidates:
        return None, {
            "success": False,
            "method": "yolov8_seg",
            "reason": "deep_no_mask",
            "mask_path": "",
        }

    best = None
    all_regions = []
    for candidate in mask_candidates:
        regions = extract_candidate_regions(candidate["mask"], target_bbox_local, config)
        all_regions.extend(regions)
        for region in regions:
            shape_score = score_region_candidate(region, target_bbox_local, config)
            score = 0.75 * shape_score + 0.25 * candidate["confidence"]
            if score <= 0:
                continue
            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
                    "confidence": float(candidate["confidence"]),
                    "region": region,
                }

    if best is None:
        return None, {
            "success": False,
            "method": "yolov8_seg",
            "reason": "deep_no_valid_region",
            "area": max((region.get("area", 0) for region in all_regions), default=0),
            "mask_path": "",
        }

    region = best["region"]
    return region["mask"], {
        "success": True,
        "method": "yolov8_seg",
        "region_score": float(best["score"]),
        "model_confidence": float(best["confidence"]),
        "area": int(region["area"]),
        "reason": "",
        "bbox": region["bbox"],
        "center": region["center"],
        "axis_ratio": region["axis_ratio"],
        "mask_path": "",
    }
