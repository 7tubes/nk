import math

import cv2
import numpy as np

try:
    from .utils import normalize_gray_uint8
except ImportError:
    from utils import normalize_gray_uint8


EPS = 1e-6


def _clamp01(value):
    return float(max(0.0, min(1.0, value)))


def _binary_mask(head_mask):
    if head_mask is None:
        raise ValueError("输入的 head_mask 不能为 None")
    if not isinstance(head_mask, np.ndarray):
        raise TypeError("head_mask 必须是 numpy.ndarray 类型")
    if head_mask.size == 0:
        raise ValueError("输入的 head_mask 不能为空图像")

    if head_mask.ndim == 2:
        gray = head_mask
    elif head_mask.ndim == 3 and head_mask.shape[2] == 1:
        gray = head_mask[:, :, 0]
    elif head_mask.ndim == 3 and head_mask.shape[2] == 3:
        gray = cv2.cvtColor(head_mask, cv2.COLOR_BGR2GRAY)
    elif head_mask.ndim == 3 and head_mask.shape[2] == 4:
        gray = cv2.cvtColor(head_mask, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("head_mask 必须是灰度图、BGR 图或 BGRA 图")

    if np.issubdtype(gray.dtype, np.floating):
        gray = np.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)

    return (gray > 0).astype(np.uint8) * 255


def find_head_contour(head_mask):
    """找到头部 mask 的最大外轮廓。"""
    mask = _binary_mask(head_mask)

    if int(np.count_nonzero(mask)) == 0:
        return {
            "success": False,
            "reason": "empty_mask",
            "contour": None,
        }

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return {
            "success": False,
            "reason": "no_contour",
            "contour": None,
        }

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return {
            "success": False,
            "reason": "not_enough_contour_points",
            "contour": contour,
        }

    return {
        "success": True,
        "reason": "",
        "contour": contour,
    }


def fit_head_ellipse(contour):
    """拟合头部椭圆，并保证 major_axis >= minor_axis。"""
    if contour is None or len(contour) < 5:
        return {
            "success": False,
            "reason": "not_enough_contour_points",
        }

    try:
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    except cv2.error:
        return {
            "success": False,
            "reason": "ellipse_fit_failed",
        }

    if axis_a <= EPS or axis_b <= EPS:
        return {
            "success": False,
            "reason": "ellipse_fit_failed",
        }

    if axis_a >= axis_b:
        major_axis = float(axis_a)
        minor_axis = float(axis_b)
        major_angle = float(angle)
    else:
        major_axis = float(axis_b)
        minor_axis = float(axis_a)
        major_angle = float((angle + 90.0) % 180.0)

    return {
        "success": True,
        "reason": "",
        "ellipse": {
            "center": [float(cx), float(cy)],
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "angle": major_angle,
        },
    }


def compute_basic_features(head_mask, ellipse_info, config: dict | None = None) -> dict:
    """计算 L、W、R、HA、HD 等基础指标。"""
    if config is None:
        config = {}

    mask = _binary_mask(head_mask)
    l_px = float(ellipse_info["major_axis"])
    w_px = float(ellipse_info["minor_axis"])

    if w_px <= EPS:
        return {
            "success": False,
            "reason": "minor_axis_too_small",
        }

    ha_px2 = int(np.count_nonzero(mask))
    hd_px = math.sqrt(4.0 * ha_px2 / math.pi)

    features = {
        "L_px": l_px,
        "W_px": w_px,
        "R": l_px / w_px,
        "HA_px2": ha_px2,
        "HD_px": hd_px,
    }

    um_per_pixel = config.get("calibration", {}).get("um_per_pixel")
    if um_per_pixel is not None:
        scale = float(um_per_pixel)
        features.update(
            {
                "L_um": l_px * scale,
                "W_um": w_px * scale,
                "HA_um2": ha_px2 * scale * scale,
                "HD_um": hd_px * scale,
            }
        )

    return features


def _rotate_to_ellipse_axes(mask, ellipse_info):
    center = tuple(float(v) for v in ellipse_info["center"])
    angle = float(ellipse_info["angle"])
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        mask,
        rotation,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def compute_symmetry_features(head_mask, ellipse_info, config: dict | None = None) -> dict:
    """计算 SAS 和 LAS。"""
    del config
    mask = _binary_mask(head_mask)
    aligned = _rotate_to_ellipse_axes(mask, ellipse_info)

    cx, cy = ellipse_info["center"]
    cx = int(round(cx))
    cy = int(round(cy))
    height, width = aligned.shape[:2]
    cx = max(0, min(width - 1, cx))
    cy = max(0, min(height - 1, cy))

    upper = int(np.count_nonzero(aligned[:cy, :]))
    lower = int(np.count_nonzero(aligned[cy:, :]))
    left = int(np.count_nonzero(aligned[:, :cx]))
    right = int(np.count_nonzero(aligned[:, cx:]))

    las = 1.0 - abs(upper - lower) / (upper + lower + EPS)
    sas = 1.0 - abs(left - right) / (left + right + EPS)

    return {
        "SAS": _clamp01(sas),
        "LAS": _clamp01(las),
    }


def _ellipse_mask(shape, ellipse_info):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    center = tuple(int(round(v)) for v in ellipse_info["center"])
    axes = (
        max(1, int(round(ellipse_info["major_axis"] / 2.0))),
        max(1, int(round(ellipse_info["minor_axis"] / 2.0))),
    )
    cv2.ellipse(
        mask,
        center,
        axes,
        float(ellipse_info["angle"]),
        0,
        360,
        255,
        thickness=-1,
    )
    return mask


def compute_fit_goodness(head_mask, ellipse_info, config: dict | None = None) -> dict:
    """计算 head mask 和拟合椭圆 mask 的 IoU。"""
    del config
    mask = _binary_mask(head_mask)
    ellipse = _ellipse_mask(mask.shape, ellipse_info)

    intersection = np.logical_and(mask > 0, ellipse > 0).sum()
    union = np.logical_or(mask > 0, ellipse > 0).sum()

    if union == 0:
        return {
            "success": False,
            "reason": "empty_union",
        }

    return {
        "fit_iou": _clamp01(intersection / (union + EPS)),
    }


def compute_uniformity(
    head_mask,
    roi_pre,
    symmetry_features,
    config: dict | None = None,
) -> dict:
    """计算灰度均匀度和综合均匀度。"""
    del config
    mask = _binary_mask(head_mask)
    roi_gray = normalize_gray_uint8(roi_pre, name="roi_pre")

    if roi_gray.shape[:2] != mask.shape[:2]:
        return {
            "success": False,
            "reason": "shape_mismatch",
        }

    values = roi_gray[mask > 0].astype(np.float32)
    if values.size == 0:
        return {
            "success": False,
            "reason": "empty_head_pixels",
        }

    gray_uniformity = 1.0 - float(np.std(values)) / (float(np.mean(values)) + EPS)
    gray_uniformity = _clamp01(gray_uniformity)
    shape_uniformity = min(
        float(symmetry_features.get("SAS", 0.0)),
        float(symmetry_features.get("LAS", 0.0)),
    )
    uniformity = 0.6 * gray_uniformity + 0.4 * shape_uniformity

    return {
        "gray_uniformity": gray_uniformity,
        "uniformity": _clamp01(uniformity),
    }


def compute_features(head_mask, roi_pre, config: dict | None = None) -> dict:
    """
    输入头部 mask 和预处理 ROI，输出全部形态指标。
    """
    if config is None:
        config = {}

    try:
        contour_result = find_head_contour(head_mask)
    except (TypeError, ValueError) as exc:
        return {
            "success": False,
            "reason": str(exc),
        }

    if not contour_result["success"]:
        return {
            "success": False,
            "reason": contour_result["reason"],
        }

    ellipse_result = fit_head_ellipse(contour_result["contour"])
    if not ellipse_result["success"]:
        return {
            "success": False,
            "reason": ellipse_result["reason"],
        }

    ellipse_info = ellipse_result["ellipse"]
    basic = compute_basic_features(head_mask, ellipse_info, config)
    if basic.get("success") is False:
        return basic

    symmetry = compute_symmetry_features(head_mask, ellipse_info, config)
    fit = compute_fit_goodness(head_mask, ellipse_info, config)
    if fit.get("success") is False:
        return fit

    uniformity = compute_uniformity(head_mask, roi_pre, symmetry, config)
    if uniformity.get("success") is False:
        return uniformity

    return {
        "success": True,
        "reason": "",
        "ellipse": ellipse_info,
        **basic,
        **symmetry,
        **fit,
        **uniformity,
    }
