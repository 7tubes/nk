import cv2
import numpy as np

try:
    from .utils import ensure_odd_kernel, normalize_gray_uint8
except ImportError:  # 允许直接运行单个文件做调试
    from utils import ensure_odd_kernel, normalize_gray_uint8


def preprocess_roi(roi_gray, config: dict):
    """
    输入 dzh 裁剪出的灰度 ROI。
    输出增强后的灰度 ROI。

    参数：
        roi_gray:输入图像
        config:
            图像预处理参数字典，例如：
            {
                "noise_filter": "gaussian",
                "noise_kernel": 3,
                "background_kernel": 31,
                "background_method": "divide",
                "clahe_clip_limit": 2.0,
                "clahe_grid_size": (8, 8)
            }

    返回：
        enhanced_roi:
            经过灰度归一化、去噪、背景校正和 CLAHE 增强后的
            uint8 单通道灰度图
    """

    if config is None:
        config = {}
    preprocess_config = config.get("preprocess", config)

    # 1. 确保输入图像为单通道 uint8 灰度图
    gray = normalize_gray_uint8(roi_gray, name="roi_gray")

    # 3. 图像去噪
    # 去噪方式：
    # gaussian：高斯噪声，图像更加平滑
    # median：椒盐噪声，边缘保留
    noise_filter = str(
        preprocess_config.get("noise_filter", "gaussian")
    ).lower()

    # 去噪卷积核大小，默认使用 3×3
    noise_kernel = ensure_odd_kernel(
        preprocess_config.get("noise_kernel", preprocess_config.get("gaussian_kernel", 3)),
        default_value=3
    )

    if noise_filter == "gaussian":
        # Gaussian 标准差，设置为 0 时由 OpenCV 自动计算
        noise_sigma = float(
            preprocess_config.get("noise_sigma", 0)
        )

        denoised = cv2.GaussianBlur(
            gray,
            (noise_kernel, noise_kernel),
            sigmaX=noise_sigma,
            sigmaY=noise_sigma
        )

    elif noise_filter == "median":
        denoised = cv2.medianBlur(
            gray,
            noise_kernel
        )

    else:
        raise ValueError(
            "noise_filter 只能设置为 'gaussian' 或 'median'"
        )

    # 4. 背景校正
    # 使用较大的 Gaussian 卷积核得到缓慢变化的背景亮度图
    background_kernel = ensure_odd_kernel(
        preprocess_config.get("background_kernel", preprocess_config.get("background_blur_kernel", 31)),
        default_value=31
    )

    # 获取 ROI 的最短边长度
    min_side = min(
        denoised.shape[0],
        denoised.shape[1]
    )

    # 防止背景卷积核明显大于 ROI 尺寸
    if background_kernel > min_side:
        background_kernel = min_side

        # Gaussian 核必须是奇数
        if background_kernel % 2 == 0:
            background_kernel -= 1

        background_kernel = max(
            background_kernel,
            1
        )

    # 背景模糊的标准差，0 表示由 OpenCV 自动计算
    background_sigma = float(
        preprocess_config.get("background_sigma", 0)
    )

    # 得到背景亮度图
    background = cv2.GaussianBlur(
        denoised,
        (background_kernel, background_kernel),
        sigmaX=background_sigma,
        sigmaY=background_sigma
    )

    # 背景校正方法
    background_method = str(
        preprocess_config.get("background_method", "divide")
    ).lower()

    # 转换为 float32，避免计算时发生 uint8 溢出
    denoised_float = denoised.astype(np.float32)
    background_float = background.astype(np.float32)

    if background_method == "divide":
        # 除法背景校正：
        # 用原图除以背景图，可以减轻由于光照不均造成的亮度变化。
        # +1.0 是为了防止除数为 0。
        corrected = cv2.divide(
            denoised_float,
            background_float + 1.0,
            scale=128.0
        )

    elif background_method == "subtract":
        # 减法背景校正：
        # 从原图中减去背景亮度，再加 128 防止出现大量负数。
        corrected = (
            denoised_float
            - background_float
            + 128.0
        )

    else:
        raise ValueError(
            "background_method 只能设置为 'divide' 或 'subtract'"
        )

    # 背景校正后再次归一化到 0～255
    corrected = cv2.normalize(
        corrected,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    corrected = np.clip(
        corrected,
        0,
        255
    ).astype(np.uint8)

    # 5. 使用 CLAHE 增强局部对比度
    # clipLimit 越大，对比度增强越明显，但过大可能同时放大噪声。
    clahe_clip_limit = float(
        preprocess_config.get("clahe_clip_limit", 2.0)
    )

    # CLAHE 将图像划分为多个小网格分别增强
    clahe_grid_size = preprocess_config.get(
        "clahe_grid_size",
        preprocess_config.get("clahe_tile_grid_size", (8, 8))
    )

    # 如果只传入一个整数，例如 8，则自动转换成 (8, 8)
    if isinstance(clahe_grid_size, int):
        clahe_grid_size = (
            clahe_grid_size,
            clahe_grid_size
        )

    # 检查网格参数是否合法
    if (
        not isinstance(clahe_grid_size, (tuple, list))
        or len(clahe_grid_size) != 2
    ):
        raise ValueError(
            "clahe_grid_size 必须是整数，或者包含两个整数的元组"
        )

    clahe_grid_size = (
        max(int(clahe_grid_size[0]), 1),
        max(int(clahe_grid_size[1]), 1)
    )

    # 创建 CLAHE 对象
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_grid_size
    )

    # 对背景校正后的图像进行局部对比度增强
    enhanced_roi = clahe.apply(corrected)

    # 返回最终增强后的单通道灰度 ROI
    return enhanced_roi

