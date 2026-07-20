from __future__ import annotations

from dataclasses import dataclass
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


def screen_detection(
    image: np.ndarray,
    detection: dict | DetectionBox,
    config: dict,
    image_id: str = "image",
    target_id: int = 0,
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
        head_mask, quality_info = segment_head(roi_pre, roi_info["target_bbox_local"], config)

        if head_mask is None:
            scores = _failed_scores(quality_info.get("reason", "segment_failed"))
            result.update(
                {
                    "roi_info": roi_info,
                    "roi_pre": roi_pre,
                    "mask": None,
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
        result.update(
            {
                "roi_info": roi_info,
                "roi_pre": roi_pre,
                "mask": head_mask,
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
    results = []
    for target_id, detection in enumerate(detections):
        results.append(
            screen_detection(
                image=image,
                detection=detection,
                config=config,
                image_id=image_id,
                target_id=target_id,
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
    cv2.drawContours(canvas, shifted, -1, color, 2)


def draw_screening_results(image: np.ndarray, screening_results: Iterable[dict]) -> np.ndarray:
    """
    Draw red/yellow/green screening results on the original image.

    The detection bbox gets the traffic-light color, and a short label includes
    the grade, score, and detector confidence. The segmented head contour is
    intentionally not drawn here, because noisy masks can make the review image
    hard to read. Mask contours are still available through the saved mask and
    debug overlay outputs when segmentation debugging is needed.
    """
    canvas = image.copy()
    for result in screening_results:
        color = GRADE_COLORS_BGR.get(result.get("traffic_color", "red"), GRADE_COLORS_BGR["red"])
        x1, y1, x2, y2 = [int(round(value)) for value in result["bbox"]]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

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
