import math


def clamp_bbox(bbox, image_width: int, image_height: int) -> list:
    """
    将bbox限制在图像边界内

    bbox:
    [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    return [x1, y1, x2, y2]


def expand_bbox(bbox, margin_px: int, image_width: int, image_height: int) -> list:
    """
    根据margin扩展bbox，并限制在图像范围内
    """
    x1, y1, x2, y2 = bbox

    expanded_bbox = [
        x1 - margin_px,
        y1 - margin_px,
        x2 + margin_px,
        y2 + margin_px
    ]

    return clamp_bbox(
        expanded_bbox,
        image_width,
        image_height
    )


def crop_roi(image, bbox, config: dict, image_id: str | None = None, target_id: int | None = None) -> dict:
    """
    根据目标bbox裁剪ROI

    返回:
    {
        "roi": roi图像,
        "roi_bbox_global": ROI在原图中的坐标,
        "target_bbox_local": 目标框在ROI中的坐标
    }
    """
    height, width = image.shape[:2]

    # 保留边界信息，向外取整
    x1 = math.floor(bbox[0])
    y1 = math.floor(bbox[1])
    x2 = math.ceil(bbox[2])
    y2 = math.ceil(bbox[3])

    # 检查目标框尺寸
    if (
        x2 - x1 < config["crop"]["min_box_width"]
        or y2 - y1 < config["crop"]["min_box_height"]
    ):
        raise ValueError("bbox_too_small")

    target_bbox = [x1, y1, x2, y2]

    roi_bbox = expand_bbox(
        target_bbox,
        config["crop"]["margin_px"],
        width,
        height
    )

    rx1, ry1, rx2, ry2 = roi_bbox

    roi = image[ry1:ry2, rx1:rx2]

    target_bbox_local = [
        x1 - rx1,
        y1 - ry1,
        x2 - rx1,
        y2 - ry1
    ]

    return {
        "roi": roi,
        "roi_bbox_global": [rx1, ry1, rx2, ry2],
        "target_bbox_local": target_bbox_local,
        "image_id": image_id,
        "target_id": target_id,
    }