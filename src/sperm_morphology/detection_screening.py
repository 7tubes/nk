from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from sperm_morphology.crop import crop_roi
from sperm_morphology.features import compute_features
from sperm_morphology.preprocess import preprocess_roi
from sperm_morphology.scoring import score_features
from sperm_morphology.segment_head import segment_head
from sperm_morphology.utils import normalize_gray_uint8, write_image_unicode


# OpenCV uses BGR channel order. The names below describe the color that the
# user sees in the saved image, while the tuple values are what cv2 expects.
GRADE_COLORS_BGR = {
    "green": (0, 190, 0),
    "yellow": (0, 215, 255),
    "red": (0, 0, 255),
}


@dataclass
class DetectionBox:
    """A single sperm detection box produced by YOLO or supplied for testing."""

    xyxy: list[float]
    confidence: float = 1.0
    source: str = "manual"


def normalize_detection(raw_detection: dict | DetectionBox, default_source: str = "yolo") -> DetectionBox:
    """
    Convert different detection formats into one stable DetectionBox object.

    The deep learning app stores boxes under ``xyxy`` and confidence under
    ``conf``. Tests or future callers may already pass a DetectionBox. Keeping
    the conversion here prevents the screening code from knowing about every
    possible caller-specific dictionary shape.
    """
    if isinstance(raw_detection, DetectionBox):
        return raw_detection

    if "xyxy" not in raw_detection:
        raise KeyError("detection must contain an 'xyxy' field")

    confidence = raw_detection.get("confidence", raw_detection.get("conf", 1.0))
    source = raw_detection.get("source", default_source)
    return DetectionBox(
        xyxy=[float(value) for value in raw_detection["xyxy"]],
        confidence=float(confidence),
        source=str(source),
    )


def grade_to_traffic_color(scores: dict, config: dict | None = None) -> str:
    """
    Map morphology grades into the requested red/yellow/green result colors.

    By default, A/B are treated as green, C is yellow, and D/Reject are red.
    The score thresholds are also read from config so the mapping still works
    if a future scoring setup changes grade names or only returns scores.
    """
    if config is None:
        config = {}

    grade = str(scores.get("grade", "Reject"))
    if grade in {"A", "B"}:
        return "green"
    if grade == "C":
        return "yellow"
    if grade in {"D", "Reject"}:
        return "red"

    thresholds = config.get("scoring", {}).get("grade_thresholds", {})
    total_score = float(scores.get("total_score", 0.0))
    if total_score >= float(thresholds.get("B", 70)):
        return "green"
    if total_score >= float(thresholds.get("C", 55)):
        return "yellow"
    return "red"


def _failed_scores(reason: str) -> dict:
    """Build a score-like dictionary for detections that cannot be screened."""
    return {
        "success": False,
        "fit_score": 0.0,
        "axis_score": 0.0,
        "uniformity_score": 0.0,
        "total_score": 0.0,
        "grade": "Reject",
        "reject_reason": reason,
    }


def _mask_bbox_global(mask: np.ndarray | None, roi_info: dict | None) -> list[float] | None:
    if mask is None:
        return None

    mask_gray = normalize_gray_uint8(mask, name="head_mask")
    points = cv2.findNonZero((mask_gray > 0).astype(np.uint8))
    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)
    offset_x = 0
    offset_y = 0
    if roi_info is not None:
        roi_bbox_global = roi_info.get("roi_bbox_global")
        if roi_bbox_global is not None and len(roi_bbox_global) == 4:
            offset_x = int(round(float(roi_bbox_global[0])))
            offset_y = int(round(float(roi_bbox_global[1])))

    return [
        float(x + offset_x),
        float(y + offset_y),
        float(x + w + offset_x),
        float(y + h + offset_y),
    ]


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def _bbox_intersection_area(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def _point_inside_bbox(point: tuple[float, float], bbox: list[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return x1 <= x <= x2 and y1 <= y <= y2


def _candidate_match_score(head: dict, detection_bbox: list[float], roi_bbox: list[float]) -> float:
    head_bbox = head.get("bbox")
    if not head_bbox:
        return -1.0

    center = _bbox_center(head_bbox)
    if not _point_inside_bbox(center, roi_bbox):
        return -1.0

    det_center = _bbox_center(detection_bbox)
    det_w = max(float(detection_bbox[2]) - float(detection_bbox[0]), 1.0)
    det_h = max(float(detection_bbox[3]) - float(detection_bbox[1]), 1.0)
    distance = float(np.hypot(center[0] - det_center[0], center[1] - det_center[1]))
    distance_score = 1.0 - min(distance / max(det_w, det_h), 1.0)

    head_area = max(_bbox_area(head_bbox), 1.0)
    overlap_det = _bbox_intersection_area(head_bbox, detection_bbox) / head_area
    overlap_roi = _bbox_intersection_area(head_bbox, roi_bbox) / head_area
    center_bonus = 1.0 if _point_inside_bbox(center, detection_bbox) else 0.0
    confidence = float(head.get("confidence", 0.0))

    return 2.0 * center_bonus + 1.5 * overlap_det + 0.75 * overlap_roi + 0.5 * distance_score + 0.25 * confidence


def _select_head_candidate(
    head_candidates: list[dict],
    detection_bbox: list[float],
    roi_info: dict,
) -> tuple[dict | None, float]:
    if not head_candidates:
        return None, 0.0

    roi_bbox = roi_info.get("roi_bbox_global")
    if roi_bbox is None:
        roi_bbox = detection_bbox

    best = None
    best_score = -1.0
    for head in head_candidates:
        score = _candidate_match_score(head, detection_bbox, roi_bbox)
        if score > best_score:
            best = head
            best_score = score

    if best is None or best_score < 0:
        return None, 0.0
    return best, float(best_score)


def _crop_global_mask_to_roi(global_mask: np.ndarray, roi_info: dict) -> np.ndarray:
    rx1, ry1, rx2, ry2 = [int(round(value)) for value in roi_info["roi_bbox_global"]]
    return (global_mask[ry1:ry2, rx1:rx2] > 0).astype(np.uint8) * 255


def screen_detection(
    image: np.ndarray,
    detection: dict | DetectionBox,
    config: dict,
    image_id: str = "image",
    target_id: int = 0,
    head_candidates: list[dict] | None = None,
    head_segmentation_info: dict | None = None,
) -> dict:
    """
    Run the morphology screening pipeline on one recognized sperm bbox.

    The deep learning detector only knows where a sperm candidate is. This
    function crops that candidate, segments the head, computes morphology
    features, scores the candidate, and finally assigns one of three display
    colors. Failures are intentionally returned as red Reject results so the
    final image still shows where the recognition part found a candidate.
    """
    detection_box = normalize_detection(detection)
    result = {
        "image_id": image_id,
        "target_id": target_id,
        "bbox": detection_box.xyxy,
        "confidence": detection_box.confidence,
        "source": detection_box.source,
        "head_bbox": None,
    }

    try:
        roi_info = crop_roi(
            image,
            detection_box.xyxy,
            config,
            image_id=image_id,
            target_id=target_id,
        )
        roi_pre = preprocess_roi(roi_info["roi"], config)
        if head_candidates is not None:
            matched_head, match_score = _select_head_candidate(head_candidates, detection_box.xyxy, roi_info)
            if matched_head is None:
                quality_info = {
                    "success": False,
                    "method": "yolov8_seg_full_image",
                    "reason": "global_head_not_matched",
                    "model_path": (head_segmentation_info or {}).get("model_path", ""),
                    "match_score": 0.0,
                    "mask_path": "",
                }
                head_mask = None
            else:
                head_mask = _crop_global_mask_to_roi(matched_head["mask"], roi_info)
                head_bbox = _mask_bbox_global(head_mask, roi_info)
                quality_info = {
                    "success": True,
                    "method": "yolov8_seg_full_image",
                    "reason": "",
                    "model_path": (head_segmentation_info or {}).get("model_path", ""),
                    "head_id": matched_head.get("head_id", ""),
                    "model_confidence": float(matched_head.get("confidence", 0.0)),
                    "match_score": float(match_score),
                    "bbox": matched_head.get("bbox"),
                    "bbox_global": head_bbox,
                    "mask_path": "",
                }
        else:
            head_mask, quality_info = segment_head(roi_pre, roi_info["target_bbox_local"], config)

        if head_mask is None:
            scores = _failed_scores(quality_info.get("reason", "segment_failed"))
            result.update(
                {
                    "roi_info": roi_info,
                    "roi_pre": roi_pre,
                    "mask": None,
                    "head_bbox": None,
                    "quality_info": quality_info,
                    "features": {"success": False, "reason": scores["reject_reason"]},
                    "scores": scores,
                    "traffic_color": "red",
                }
            )
            return result

        features = compute_features(head_mask, roi_pre, config)
        scores = score_features(features, config)
        color_name = grade_to_traffic_color(scores, config)
        head_bbox = _mask_bbox_global(head_mask, roi_info)
        if quality_info is not None:
            quality_info["bbox_global"] = head_bbox
        result.update(
            {
                "roi_info": roi_info,
                "roi_pre": roi_pre,
                "mask": head_mask,
                "head_bbox": head_bbox,
                "quality_info": quality_info,
                "features": features,
                "scores": scores,
                "traffic_color": color_name,
            }
        )
        return result

    except Exception as exc:
        scores = _failed_scores(str(exc))
        result.update(
            {
                "roi_info": None,
                "roi_pre": None,
                "mask": None,
                "head_bbox": None,
                "quality_info": {"success": False, "reason": str(exc)},
                "features": {"success": False, "reason": str(exc)},
                "scores": scores,
                "traffic_color": "red",
            }
        )
        return result


def screen_detections(
    image: np.ndarray,
    detections: Iterable[dict | DetectionBox],
    config: dict,
    image_id: str = "image",
) -> list[dict]:
    """Screen all detection boxes from one image and keep target ids stable."""
    head_candidates = None
    head_segmentation_info = None
    screening_config = config
    deep_config = config.get("deep_segmentation", {}) if config else {}
    if deep_config.get("enabled", False):
        try:
            from sperm_morphology.deep_head_segmenter import segment_heads_in_image

            head_candidates, head_segmentation_info = segment_heads_in_image(image, config)
            if not head_segmentation_info.get("success", False):
                head_candidates = None
                if not deep_config.get("fallback_to_traditional", True):
                    head_candidates = []
                else:
                    screening_config = copy.deepcopy(config)
                    screening_config.setdefault("deep_segmentation", {})["enabled"] = False
        except Exception as exc:
            head_segmentation_info = {
                "success": False,
                "method": "yolov8_seg_full_image",
                "reason": str(exc),
                "model_path": "",
            }
            head_candidates = None
            if not deep_config.get("fallback_to_traditional", True):
                head_candidates = []
            else:
                screening_config = copy.deepcopy(config)
                screening_config.setdefault("deep_segmentation", {})["enabled"] = False

    results = []
    for target_id, detection in enumerate(detections):
        results.append(
            screen_detection(
                image=image,
                detection=detection,
                config=screening_config,
                image_id=image_id,
                target_id=target_id,
                head_candidates=head_candidates,
                head_segmentation_info=head_segmentation_info,
            )
        )
    return results


def _draw_head_mask(canvas: np.ndarray, screening_result: dict, color: tuple[int, int, int]) -> None:
    """
    Draw the segmented head contour back on the full image.

    If segmentation failed, there is no mask to draw. The caller still draws
    the detection bbox in red, which is enough to show that this recognized
    candidate did not pass screening.
    """
    mask = screening_result.get("mask")
    roi_info = screening_result.get("roi_info")
    if mask is None or roi_info is None:
        return

    mask_gray = normalize_gray_uint8(mask, name="head_mask")
    contours, _ = cv2.findContours(
        (mask_gray > 0).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return

    rx1, ry1, _, _ = [int(round(value)) for value in roi_info["roi_bbox_global"]]
    offset = np.array([[[rx1, ry1]]], dtype=np.int32)
    shifted = [contour.astype(np.int32) + offset for contour in contours]
    cv2.drawContours(canvas, shifted, -1, color, 1)


def draw_screening_results(image: np.ndarray, screening_results: Iterable[dict]) -> np.ndarray:
    """
    Draw red/yellow/green screening results on the original image.

    The detection bbox gets the traffic-light color, the segmented head contour
    is drawn in yellow, and a short label includes the grade, score, and
    detector confidence.
    """
    canvas = image.copy()
    for result in screening_results:
        color = GRADE_COLORS_BGR.get(result.get("traffic_color", "red"), GRADE_COLORS_BGR["red"])
        x1, y1, x2, y2 = [int(round(value)) for value in result["bbox"]]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        _draw_head_mask(canvas, result, (0, 255, 255))

        scores = result.get("scores", {})
        grade = scores.get("grade", "Reject")
        total_score = float(scores.get("total_score", 0.0))
        confidence = float(result.get("confidence", 0.0))
        label = f"{grade} {total_score:.1f} conf {confidence:.2f}"

        label_y = max(16, y1 - 6)
        cv2.putText(
            canvas,
            label,
            (max(0, x1), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return canvas


def save_screening_overlay(
    image: np.ndarray,
    screening_results: Iterable[dict],
    output_path: str | Path,
) -> str:
    """Save the combined recognition + morphology screening visualization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = draw_screening_results(image, screening_results)
    if not write_image_unicode(output_path, overlay):
        raise OSError(f"failed to write overlay: {output_path}")
    return str(output_path)
