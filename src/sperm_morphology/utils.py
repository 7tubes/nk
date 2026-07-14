import cv2
import numpy as np


def ensure_odd_kernel(value, default_value=3, minimum=1):
    """Return a positive odd kernel size for OpenCV operations."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default_value

    value = max(value, minimum)

    if value % 2 == 0:
        value += 1

    return value


def normalize_gray_uint8(image, name="image"):
    """
    Validate an image, convert it to one-channel grayscale, and normalize to uint8.

    This is shared by preprocessing and segmentation. It is intentionally small:
    denoising, background correction, and CLAHE still belong in preprocess.py.
    """
    if image is None:
        raise ValueError(f"输入的 {name} 不能为 None")

    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} 必须是 numpy.ndarray 类型")

    if image.size == 0:
        raise ValueError(f"输入的 {name} 不能为空图像")

    if image.ndim == 2:
        gray = image.copy()
    elif image.ndim == 3:
        channel_num = image.shape[2]

        if channel_num == 1:
            gray = image[:, :, 0].copy()
        elif channel_num == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channel_num == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"不支持的图像通道数：{channel_num}")
    else:
        raise ValueError(f"{name} 的维度必须为 2 或 3，当前维度为：{image.ndim}")

    if np.issubdtype(gray.dtype, np.floating):
        gray = np.nan_to_num(gray, nan=0.0, posinf=255.0, neginf=0.0)

    gray = cv2.normalize(
        gray,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )

    return np.clip(gray, 0, 255).astype(np.uint8)
