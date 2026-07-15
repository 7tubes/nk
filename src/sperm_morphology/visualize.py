from pathlib import Path

import cv2
import numpy as np

try:
    from .utils import normalize_gray_uint8
except ImportError:
    from utils import normalize_gray_uint8


def _output_dir(config):
    data_config = (config or {}).get("data", {})
    output_root = Path(data_config.get("output_dir", "outputs"))
    overlay_dir = output_root / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    return overlay_dir


def _as_bgr(image):
    if image is None:
        raise ValueError("image 不能为 None")
    if image.ndim == 2:
        return cv2.cvtColor(normalize_gray_uint8(image), cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError("image 必须是灰度图、BGR 图或 BGRA 图")


def _offset_contours(contours, offset_x, offset_y):
    shifted = []
    offset = np.array([[[offset_x, offset_y]]], dtype=np.int32)
    for contour in contours:
        shifted.append(contour.astype(np.int32) + offset)
    return shifted


def _draw_axes(canvas, ellipse_info, offset_x, offset_y):
    cx, cy = ellipse_info["center"]
    cx += offset_x
    cy += offset_y
    angle = np.deg2rad(float(ellipse_info["angle"]))
    major = float(ellipse_info["major_axis"]) / 2.0
    minor = float(ellipse_info["minor_axis"]) / 2.0

    dx_major = np.cos(angle) * major
    dy_major = np.sin(angle) * major
    dx_minor = -np.sin(angle) * minor
    dy_minor = np.cos(angle) * minor

    cv2.line(
        canvas,
        (int(round(cx - dx_major)), int(round(cy - dy_major))),
        (int(round(cx + dx_major)), int(round(cy + dy_major))),
        (0, 255, 255),
        1,
    )
    cv2.line(
        canvas,
        (int(round(cx - dx_minor)), int(round(cy - dy_minor))),
        (int(round(cx + dx_minor)), int(round(cy + dy_minor))),
        (255, 0, 255),
        1,
    )


def _safe_name(value):
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_overlay(image, roi_info, head_mask, features, scores, config: dict | None = None) -> str:
    """保存原图 overlay，并返回 overlay_path。"""
    if config is None:
        config = {}

    canvas = _as_bgr(image)
    roi_bbox = roi_info.get("roi_bbox_global", [0, 0, head_mask.shape[1], head_mask.shape[0]])
    rx1, ry1, rx2, ry2 = [int(round(v)) for v in roi_bbox]

    cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (255, 180, 0), 1)

    target_bbox = roi_info.get("target_bbox_local")
    if target_bbox is not None:
        tx1, ty1, tx2, ty2 = [int(round(v)) for v in target_bbox]
        cv2.rectangle(
            canvas,
            (rx1 + tx1, ry1 + ty1),
            (rx1 + tx2, ry1 + ty2),
            (0, 180, 255),
            1,
        )

    if head_mask.ndim == 2:
        mask_gray = head_mask
    elif head_mask.ndim == 3 and head_mask.shape[2] == 1:
        mask_gray = head_mask[:, :, 0]
    elif head_mask.ndim == 3 and head_mask.shape[2] == 3:
        mask_gray = cv2.cvtColor(head_mask, cv2.COLOR_BGR2GRAY)
    elif head_mask.ndim == 3 and head_mask.shape[2] == 4:
        mask_gray = cv2.cvtColor(head_mask, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("head_mask 必须是灰度图、BGR 图或 BGRA 图")

    mask = (mask_gray > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    shifted = _offset_contours(contours, rx1, ry1)
    cv2.drawContours(canvas, shifted, -1, (0, 255, 0), 1)

    ellipse_info = features.get("ellipse")
    if ellipse_info:
        center = (
            int(round(ellipse_info["center"][0] + rx1)),
            int(round(ellipse_info["center"][1] + ry1)),
        )
        axes = (
            max(1, int(round(ellipse_info["major_axis"] / 2.0))),
            max(1, int(round(ellipse_info["minor_axis"] / 2.0))),
        )
        cv2.ellipse(canvas, center, axes, float(ellipse_info["angle"]), 0, 360, (0, 0, 255), 1)
        _draw_axes(canvas, ellipse_info, rx1, ry1)

    grade = scores.get("grade", "NA")
    total_score = float(scores.get("total_score", 0.0))
    label = f"{grade} {total_score:.1f}"
    cv2.putText(
        canvas,
        label,
        (max(0, rx1), max(14, ry1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    output_dir = _output_dir(config)
    image_id = _safe_name(roi_info.get("image_id", "sample"))
    target_id = _safe_name(roi_info.get("target_id", "target"))

    if grade == "Reject":
        reason = _safe_name(scores.get("reject_reason", "reject"))
        filename = f"{image_id}_target{target_id}_Reject_{reason}.png"
    else:
        filename = f"{image_id}_target{target_id}_{grade}_{total_score:.1f}.png"

    overlay_path = output_dir / filename
    cv2.imwrite(str(overlay_path), canvas)
    return str(overlay_path)
