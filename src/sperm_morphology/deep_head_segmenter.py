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


def resolve_deep_model_path(config: dict | None = None) -> Path | None:
    """Return the configured YOLOv8-Seg head model path, resolved from project root."""
    return _resolve_path(_deep_config(config).get("model_path"))


@lru_cache(maxsize=2)
def _load_yolo_model(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def _as_bgr_image(image) -> np.ndarray:
    if image is None:
        raise ValueError("image cannot be None")
    if image.ndim == 2:
        return cv2.cvtColor(normalize_gray_uint8(image, name="image"), cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError("image must be grayscale, BGR, or BGRA")


def _bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    return [float(x), float(y), float(x + w), float(y + h)]


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


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
        bbox = _bbox_from_mask(mask)
        if bbox is None:
            continue
        candidates.append({"mask": mask, "confidence": confidence, "bbox": bbox})

    return candidates


def _place_tile_mask(mask: np.ndarray, image_shape: tuple[int, int], x0: int, y0: int) -> np.ndarray:
    image_h, image_w = image_shape[:2]
    full_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    tile_h, tile_w = mask.shape[:2]
    x1 = min(x0 + tile_w, image_w)
    y1 = min(y0 + tile_h, image_h)
    full_mask[y0:y1, x0:x1] = mask[: y1 - y0, : x1 - x0]
    return full_mask


def _suppress_duplicate_heads(heads: list[dict], iou_threshold: float = 0.45) -> list[dict]:
    if not heads:
        return []

    boxes_xywh = []
    scores = []
    valid_heads = []
    for head in heads:
        bbox = head.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        boxes_xywh.append(
            [
                int(round(x1)),
                int(round(y1)),
                int(round(max(x2 - x1, 1.0))),
                int(round(max(y2 - y1, 1.0))),
            ]
        )
        scores.append(float(head.get("confidence", 0.0)))
        valid_heads.append(head)

    if not valid_heads:
        return []

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=0.0,
        nms_threshold=float(iou_threshold),
    )
    if len(indices) == 0:
        return []

    picked = [valid_heads[int(index)] for index in indices.flatten()]
    picked.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    for head_id, head in enumerate(picked):
        head["head_id"] = head_id
    return picked


def segment_heads_in_image(image, config: dict | None = None) -> tuple[list[dict], dict]:
    """
    Run the trained YOLOv8-Seg head model on the whole image.

    This is the path used by the DeepLearning app: first find all heads in the
    original image or image tiles, then match those masks back to sperm
    detection boxes for scoring.
    """
    deep_config = _deep_config(config)
    model_path = resolve_deep_model_path(config)
    if model_path is None or not model_path.exists():
        return [], {
            "success": False,
            "method": "yolov8_seg_full_image",
            "reason": "deep_model_not_found",
            "model_path": str(model_path) if model_path is not None else "",
        }

    try:
        model = _load_yolo_model(str(model_path))
    except Exception as exc:
        return [], {
            "success": False,
            "method": "yolov8_seg_full_image",
            "reason": f"deep_model_load_failed: {exc}",
            "model_path": str(model_path),
        }

    input_image = _as_bgr_image(image)
    min_confidence = float(deep_config.get("conf", 0.15))
    imgsz = int(deep_config.get("imgsz", 640))
    max_det = int(deep_config.get("max_det", 300))
    use_tiling = bool(deep_config.get("full_image_tiling", True))
    tile_size = int(deep_config.get("tile_size", 320))
    overlap = float(deep_config.get("overlap", 0.25))
    nms_iou = float(deep_config.get("head_nms_iou", 0.45))

    heads: list[dict] = []
    try:
        if not use_tiling:
            result = model(
                input_image,
                conf=min_confidence,
                imgsz=imgsz,
                max_det=max_det,
                stream=False,
                verbose=False,
            )[0]
            heads.extend(_mask_candidates_from_result(result, input_image.shape, min_confidence))
        else:
            image_h, image_w = input_image.shape[:2]
            stride = max(1, int(round(tile_size * (1.0 - overlap))))
            for y0 in _tile_starts(image_h, tile_size, stride):
                for x0 in _tile_starts(image_w, tile_size, stride):
                    x1 = min(x0 + tile_size, image_w)
                    y1 = min(y0 + tile_size, image_h)
                    tile = input_image[y0:y1, x0:x1]
                    result = model(
                        tile,
                        conf=min_confidence,
                        imgsz=imgsz,
                        max_det=max_det,
                        stream=False,
                        verbose=False,
                    )[0]
                    for head in _mask_candidates_from_result(result, tile.shape, min_confidence):
                        full_mask = _place_tile_mask(head["mask"], input_image.shape, x0, y0)
                        bbox = _bbox_from_mask(full_mask)
                        if bbox is None:
                            continue
                        heads.append(
                            {
                                "mask": full_mask,
                                "confidence": float(head["confidence"]),
                                "bbox": bbox,
                            }
                        )
    except Exception as exc:
        return [], {
            "success": False,
            "method": "yolov8_seg_full_image",
            "reason": f"deep_predict_failed: {exc}",
            "model_path": str(model_path),
        }

    heads = _suppress_duplicate_heads(heads, iou_threshold=nms_iou)
    return heads, {
        "success": True,
        "method": "yolov8_seg_full_image",
        "reason": "",
        "model_path": str(model_path),
        "head_count": len(heads),
        "full_image_tiling": use_tiling,
        "tile_size": tile_size,
        "overlap": overlap,
    }


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
