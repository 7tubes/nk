import cv2

try:
    from .utils import ensure_odd_kernel, normalize_gray_uint8
except ImportError:
    from utils import ensure_odd_kernel, normalize_gray_uint8


def threshold_candidates(roi_pre, config: dict | None = None) -> list[dict]:
    """
    对预处理后的 ROI 生成四种二值分割候选：
    Otsu 正向、Otsu 反向、自适应正向、自适应反向。
    """
    if config is None:
        config = {}

    gray = normalize_gray_uint8(roi_pre, name="roi_pre")

    blur_kernel = ensure_odd_kernel(
        config.get("pre_blur_kernel", 1),
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

    adaptive_method_name = str(config.get("adaptive_method", "gaussian")).lower()
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
        config.get("adaptive_block_size", 21),
        default_value=21,
        minimum=3,
    )
    if block_size > min_side:
        block_size = min_side if min_side % 2 == 1 else min_side - 1
        block_size = max(block_size, 3)

    adaptive_c = float(config.get("adaptive_c", 3))

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
