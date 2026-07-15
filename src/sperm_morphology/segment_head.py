from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    from .utils import ensure_odd_kernel, normalize_gray_uint8
except ImportError:
    from utils import ensure_odd_kernel, normalize_gray_uint8


EPS = 1e-6


def _segmentation_config(config: dict | None) -> dict:
    if config is None:
        return {}
    return config.get("segmentation", config)


def _binary_mask(mask):
    gray = normalize_gray_uint8(mask, name="mask")
    return (gray > 0).astype(np.uint8) * 255


def _target_center(target_bbox_local):
    if target_bbox_local is None:
        return None
    if len(target_bbox_local) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in target_bbox_local]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def threshold_candidates(roi_pre, config: dict | None = None) -> list[dict]:
    """
    对预处理后的 ROI 生成四种二值分割候选：
    Otsu 正向、Otsu 反向、自适应正向、自适应反向。
    """
    seg_config = _segmentation_config(config)
    gray = normalize_gray_uint8(roi_pre, name="roi_pre")

    blur_kernel = ensure_odd_kernel(
        seg_config.get("pre_blur_kernel", 1),
        default_value=1,
        minimum=1,
    )
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), sigmaX=0)

    _, otsu_binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    _, otsu_binary_inv = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    adaptive_method_name = str(seg_config.get("adaptive_method", "gaussian")).lower()
    if adaptive_method_name == "gaussian":
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    elif adaptive_method_name == "mean":
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        raise ValueError("adaptive_method 只能是 'gaussian' 或 'mean'")

    min_side = min(gray.shape[:2])
    if min_side < 3:
        raise ValueError("ROI 尺寸过小，无法进行自适应阈值分割")

    block_size = ensure_odd_kernel(
        seg_config.get("adaptive_block_size", 21),
        default_value=21,
        minimum=3,
    )
    if block_size > min_side:
        block_size = min_side if min_side % 2 == 1 else min_side - 1
        block_size = max(block_size, 3)

    adaptive_c = float(seg_config.get("adaptive_c", 3))

    adaptive_binary = cv2.adaptiveThreshold(
        gray,
        255,
        adaptive_method,
        cv2.THRESH_BINARY,
        block_size,
        adaptive_c,
    )
    adaptive_binary_inv = cv2.adaptiveThreshold(
        gray,
        255,
        adaptive_method,
        cv2.THRESH_BINARY_INV,
        block_size,
        adaptive_c,
    )

    return [
        {"method": "otsu_binary", "mask": otsu_binary},
        {"method": "otsu_binary_inv", "mask": otsu_binary_inv},
        {"method": "adaptive_binary", "mask": adaptive_binary},
        {"method": "adaptive_binary_inv", "mask": adaptive_binary_inv},
    ]


def clean_binary_mask(mask, config: dict | None = None):
    """对二值图做开运算、闭运算、轻微腐蚀和膨胀。"""
    seg_config = _segmentation_config(config)
    cleaned = _binary_mask(mask)
    min_area = int(seg_config.get("min_head_area_px", 8))

    kernel_size = ensure_odd_kernel(
        seg_config.get("morphology_kernel", 3),
        default_value=3,
        minimum=1,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    open_iterations = max(int(seg_config.get("morphology_open_iterations", 1)), 0)
    close_iterations = max(int(seg_config.get("morphology_close_iterations", 1)), 0)
    erode_iterations = max(int(seg_config.get("morphology_erode_iterations", 1)), 0)
    dilate_iterations = max(int(seg_config.get("morphology_dilate_iterations", 1)), 0)

    if open_iterations > 0:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            kernel,
            iterations=open_iterations,
        )

    if close_iterations > 0:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iterations,
        )

    before_erode = cleaned.copy()
    if erode_iterations > 0:
        eroded = cv2.erode(cleaned, kernel, iterations=erode_iterations)
        if int(np.count_nonzero(eroded)) >= min_area:
            cleaned = eroded
        else:
            cleaned = before_erode

    if dilate_iterations > 0:
        cleaned = cv2.dilate(cleaned, kernel, iterations=dilate_iterations)

    return _binary_mask(cleaned)


def extract_candidate_regions(mask, target_bbox_local=None, config: dict | None = None) -> list[dict]:
    """找连通域或轮廓，返回候选头部区域。"""
    seg_config = _segmentation_config(config)
    binary = _binary_mask(mask)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    regions = []
    target_center = _target_center(target_bbox_local)
    min_area = float(seg_config.get("min_head_area_px", 8))
    max_area = float(seg_config.get("max_head_area_px", 800))
    min_contour_points = int(seg_config.get("min_contour_points", 5))
    min_axis_ratio = float(seg_config.get("min_head_axis_ratio", 1.0))
    max_axis_ratio = float(seg_config.get("max_head_axis_ratio", 4.0))

    for contour in contours:
        region_mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=-1)

        area = int(np.count_nonzero(region_mask))
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w / 2.0, y + h / 2.0)

        distance_to_target = None
        if target_center is not None:
            distance_to_target = float(np.hypot(center[0] - target_center[0], center[1] - target_center[1]))

        ellipse = None
        axis_ratio = None
        can_fit_ellipse = len(contour) >= max(5, min_contour_points)
        if can_fit_ellipse:
            try:
                (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
                major_axis = max(float(axis_a), float(axis_b))
                minor_axis = max(min(float(axis_a), float(axis_b)), EPS)
                axis_ratio = major_axis / minor_axis
                ellipse = {
                    "center": [float(cx), float(cy)],
                    "major_axis": major_axis,
                    "minor_axis": minor_axis,
                    "angle": float(angle if axis_a >= axis_b else (angle + 90.0) % 180.0),
                }
            except cv2.error:
                can_fit_ellipse = False

        reject_reason = ""
        if area < min_area:
            reject_reason = "area_too_small"
        elif area > max_area * 1.5:
            reject_reason = "overlap_or_debris"
        elif area > max_area:
            reject_reason = "area_too_large"
        elif not can_fit_ellipse:
            reject_reason = "cannot_fit_ellipse"
        elif axis_ratio is not None and axis_ratio > max_axis_ratio:
            reject_reason = "tail_connected"
        elif axis_ratio is not None and axis_ratio < min_axis_ratio:
            reject_reason = "cannot_fit_ellipse"

        regions.append(
            {
                "mask": region_mask,
                "contour": contour,
                "area": area,
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "center": [float(center[0]), float(center[1])],
                "distance_to_target": distance_to_target,
                "can_fit_ellipse": can_fit_ellipse,
                "ellipse": ellipse,
                "axis_ratio": axis_ratio,
                "contour_points": int(len(contour)),
                "reject_reason": reject_reason,
            }
        )

    return regions


def score_region_candidate(region, target_bbox_local=None, config: dict | None = None) -> float:
    """给单个候选区域打质量分，分数范围为 0-1。"""
    if region.get("reject_reason"):
        return 0.0

    seg_config = _segmentation_config(config)
    min_area = float(seg_config.get("min_head_area_px", 8))
    max_area = float(seg_config.get("max_head_area_px", 800))

    area = float(region.get("area", 0))
    if area <= 0:
        return 0.0

    area_mid = (min_area + max_area) / 2.0
    area_range = max((max_area - min_area) / 2.0, EPS)
    area_score = 1.0 - min(abs(area - area_mid) / area_range, 1.0)

    bbox = region.get("bbox", [0, 0, 1, 1])
    width = max(float(bbox[2] - bbox[0]), 1.0)
    height = max(float(bbox[3] - bbox[1]), 1.0)
    fill_ratio = area / (width * height + EPS)
    fill_score = max(0.0, min(fill_ratio / 0.65, 1.0))

    axis_ratio = float(region.get("axis_ratio") or 0.0)
    axis_score = 0.0
    if axis_ratio > 0:
        target_ratio = float(seg_config.get("target_head_axis_ratio", 1.7))
        tolerance = float(seg_config.get("head_axis_ratio_tolerance", 1.2))
        axis_score = 1.0 - min(abs(axis_ratio - target_ratio) / tolerance, 1.0)

    distance = region.get("distance_to_target")
    if distance is None:
        distance_score = 1.0
    else:
        target_center = _target_center(target_bbox_local)
        if target_center is None:
            distance_score = 1.0
        else:
            max_distance = float(seg_config.get("max_center_distance_px", max(width, height) * 3.0 + 10.0))
            distance_score = 1.0 - min(float(distance) / max(max_distance, EPS), 1.0)

    score = (
        0.30 * area_score
        + 0.25 * axis_score
        + 0.30 * distance_score
        + 0.15 * fill_score
    )
    return float(max(0.0, min(score, 1.0)))


def _failure_info(reason, method="", area=0, region_score=0.0):
    return {
        "success": False,
        "method": method,
        "region_score": float(region_score),
        "area": int(area),
        "reason": reason,
        "mask_path": "",
    }


def _best_failure_reason(regions):
    reasons = [region.get("reject_reason") for region in regions if region.get("reject_reason")]
    if not reasons:
        return "no_valid_contour"
    return Counter(reasons).most_common(1)[0][0]


def segment_head(roi_pre, target_bbox_local=None, config: dict | None = None):
    """
    输入预处理 ROI 和局部 bbox，输出最可靠的精子头部 mask 及质量信息。
    """
    try:
        candidates = threshold_candidates(roi_pre, config)
    except (TypeError, ValueError) as exc:
        return None, _failure_info(str(exc))

    best = None
    rejected_regions = []
    has_foreground = False

    for candidate in candidates:
        method = candidate["method"]
        cleaned = clean_binary_mask(candidate["mask"], config)
        foreground = int(np.count_nonzero(cleaned))
        if foreground == 0:
            continue

        has_foreground = True
        regions = extract_candidate_regions(cleaned, target_bbox_local, config)
        rejected_regions.extend(regions)

        for region in regions:
            score = score_region_candidate(region, target_bbox_local, config)
            if score <= 0:
                continue
            if best is None or score > best["score"]:
                best = {
                    "method": method,
                    "score": score,
                    "region": region,
                }

    if not has_foreground:
        return None, _failure_info("no_foreground")

    if best is None:
        reason = _best_failure_reason(rejected_regions)
        area = max((region.get("area", 0) for region in rejected_regions), default=0)
        return None, _failure_info(reason, area=area)

    region = best["region"]
    quality_info = {
        "success": True,
        "method": best["method"],
        "region_score": float(best["score"]),
        "area": int(region["area"]),
        "reason": "",
        "bbox": region["bbox"],
        "center": region["center"],
        "axis_ratio": region["axis_ratio"],
        "mask_path": "",
    }

    return region["mask"], quality_info


def save_mask_result(
    head_mask,
    quality_info,
    config: dict | None = None,
    image_id="sample",
    target_id="target",
):
    """保存成功或失败的 mask 调试图，并返回保存路径。"""
    if config is None:
        config = {}

    output_root = Path(config.get("data", {}).get("output_dir", "outputs"))
    mask_dir = output_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    success = bool((quality_info or {}).get("success", False))
    if success:
        filename = f"{image_id}_target{target_id}_mask.png"
    else:
        reason = str((quality_info or {}).get("reason", "failed")).replace("/", "_").replace("\\", "_")
        filename = f"{image_id}_target{target_id}_failed_{reason}.png"

    path = mask_dir / filename
    if head_mask is None:
        head_mask = np.zeros((32, 32), dtype=np.uint8)
    cv2.imwrite(str(path), _binary_mask(head_mask))

    if quality_info is not None:
        quality_info["mask_path"] = str(path)

    return str(path)
