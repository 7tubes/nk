def clamp(value, lower=0.0, upper=1.0):
    return max(lower, min(upper, float(value)))


def _scoring_config(config: dict | None) -> dict:
    if config is None:
        config = {}
    return config.get("scoring", config)


def score_fit_goodness(fit_iou, config: dict | None = None) -> float:
    """拟合优度评分，0-100。"""
    scoring = _scoring_config(config)
    good = float(scoring.get("fit_iou_good", 0.85))
    bad = float(scoring.get("fit_iou_bad", 0.60))
    return clamp((float(fit_iou) - bad) / (good - bad)) * 100.0


def score_axis_ratio(ratio, config: dict | None = None) -> float:
    """长轴比评分，0-100。"""
    scoring = _scoring_config(config)
    target = float(scoring.get("axis_ratio_target", 1.62))
    tolerance = float(scoring.get("axis_ratio_tolerance", 0.35))
    return clamp(1.0 - abs(float(ratio) - target) / tolerance) * 100.0


def score_uniformity(uniformity, config: dict | None = None) -> float:
    """均匀度评分，0-100。"""
    scoring = _scoring_config(config)
    good = float(scoring.get("uniformity_good", 0.85))
    bad = float(scoring.get("uniformity_bad", 0.60))
    return clamp((float(uniformity) - bad) / (good - bad)) * 100.0


def _reject_reason(features: dict, config: dict | None = None):
    scoring = _scoring_config(config)
    segmentation = (config or {}).get("segmentation", {})

    if not features.get("success", False):
        return features.get("reason", "feature_failed")

    area = float(features.get("HA_px2", 0))
    min_area = float(scoring.get("min_head_area_px", segmentation.get("min_head_area_px", 8)))
    max_area = float(scoring.get("max_head_area_px", segmentation.get("max_head_area_px", 800)))
    if area < min_area:
        return "area_too_small"
    if area > max_area:
        return "area_too_large"

    ratio = float(features.get("R", 0.0))
    min_ratio = float(scoring.get("min_axis_ratio", 1.0))
    max_ratio = float(scoring.get("max_axis_ratio", 4.0))
    if ratio < min_ratio or ratio > max_ratio:
        return "axis_ratio_extreme"

    if min(float(features.get("SAS", 0.0)), float(features.get("LAS", 0.0))) < float(
        scoring.get("min_symmetry", 0.55)
    ):
        return "low_symmetry"

    if float(features.get("fit_iou", 0.0)) < float(scoring.get("min_fit_iou", 0.50)):
        return "bad_fit"

    return ""


def _grade(total_score, config: dict | None = None):
    scoring = _scoring_config(config)
    thresholds = scoring.get("grade_thresholds", {})
    if total_score >= float(thresholds.get("A", 85)):
        return "A"
    if total_score >= float(thresholds.get("B", 70)):
        return "B"
    if total_score >= float(thresholds.get("C", 55)):
        return "C"
    return "D"


def score_features(features: dict, config: dict | None = None) -> dict:
    """计算总分和等级。"""
    reject_reason = _reject_reason(features, config)
    if reject_reason:
        return {
            "success": False,
            "fit_score": 0.0,
            "axis_score": 0.0,
            "uniformity_score": 0.0,
            "total_score": 0.0,
            "grade": "Reject",
            "reject_reason": reject_reason,
        }

    scoring = _scoring_config(config)
    weights = scoring.get("weights", {})
    fit_weight = float(weights.get("fit_goodness", 0.40))
    axis_weight = float(weights.get("axis_ratio", 0.30))
    uniformity_weight = float(weights.get("uniformity", 0.30))

    fit_score = score_fit_goodness(features["fit_iou"], config)
    axis_score = score_axis_ratio(features["R"], config)
    uniformity_score = score_uniformity(features["uniformity"], config)
    total_score = (
        fit_weight * fit_score
        + axis_weight * axis_score
        + uniformity_weight * uniformity_score
    )

    return {
        "success": True,
        "fit_score": fit_score,
        "axis_score": axis_score,
        "uniformity_score": uniformity_score,
        "total_score": total_score,
        "grade": _grade(total_score, config),
        "reject_reason": "",
    }
