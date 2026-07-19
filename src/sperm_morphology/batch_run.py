import os
import csv
from collections import defaultdict
import cv2

from sperm_morphology.dataset import load_config, load_dataset
from sperm_morphology.crop import crop_roi
from sperm_morphology.segment_head import save_mask_result

from sperm_morphology.preprocess import preprocess_roi
from sperm_morphology.segment_head import segment_head

from sperm_morphology.features import compute_features
from sperm_morphology.scoring import score_features
from sperm_morphology.visualize import save_overlay, save_image_overlay


def ensure_output_dirs(config: dict):
    """
    创建输出目录
    """
    output_dir = config["data"]["output_dir"]

    dirs = [
        output_dir,
        os.path.join(output_dir, "rois"),
        os.path.join(output_dir, "masks"),
        os.path.join(output_dir, "overlays")
    ]

    for path in dirs:
        os.makedirs(path, exist_ok=True)


def read_image(image_path: str):
    """
    读取图像
    """
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"cannot read image: {image_path}"
        )

    return image


def save_csv(rows: list, path: str):
    """
    保存csv结果
    """
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)


def save_roi_result(roi_image, roi_info, config: dict):
    """保存 ROI 图像，并返回保存路径。"""
    if roi_image is None:
        return ""

    output_root = config.get("data", {}).get("output_dir", "outputs")
    roi_dir = os.path.join(output_root, "rois")
    os.makedirs(roi_dir, exist_ok=True)

    image_id = str(roi_info.get("image_id", "sample")).replace("/", "_").replace("\\", "_")
    target_id = str(roi_info.get("target_id", "target")).replace("/", "_").replace("\\", "_")
    path = os.path.join(roi_dir, f"{image_id}_target{target_id}_roi.png")
    cv2.imwrite(path, roi_image)
    return path


def run_batch(config_path: str):
    """
    批量运行精子形态筛选流程
    """
    config = load_config(config_path)

    ensure_output_dirs(config)

    samples = load_dataset(config)

    image_groups = defaultdict(list)
    for sample in samples:
        image_groups[sample["image_id"]].append(sample)

    rows = []
    failed_rows = []

    total_images = len(image_groups)
    image_ids = sorted(image_groups.keys())

    for index, image_id in enumerate(image_ids, start=1):
        image_samples = image_groups[image_id]
        print(f"[{index}/{total_images}] {image_id} ({len(image_samples)} targets)")

        image_path = image_samples[0]["image_path"]
        image = read_image(image_path)

        detections = []

        for sample in image_samples:
            try:
                roi_info = crop_roi(
                    image,
                    sample["bbox"],
                    config,
                    image_id=sample["image_id"],
                    target_id=sample["target_id"],
                )

                roi_path = save_roi_result(
                    roi_info["roi"],
                    roi_info,
                    config
                )

                roi_pre = preprocess_roi(
                    roi_info["roi"],
                    config
                )

                mask, quality_info = segment_head(
                    roi_pre,
                    roi_info["target_bbox_local"],
                    config
                )

                if mask is None:
                    mask_path = save_mask_result(
                        None,
                        quality_info,
                        config,
                        image_id=sample["image_id"],
                        target_id=sample["target_id"],
                    )
                    failed_rows.append({
                        "image_id": sample["image_id"],
                        "target_id": sample["target_id"],
                        "reason": quality_info.get(
                            "reason",
                            "segment_failed"
                        ),
                        "mask_path": mask_path,
                    })
                    continue

                features = compute_features(
                    mask,
                    roi_pre,
                    config
                )

                scores = score_features(
                    features,
                    config
                )

                mask_path = save_mask_result(
                    mask,
                    quality_info,
                    config,
                    image_id=sample["image_id"],
                    target_id=sample["target_id"],
                )

                overlay_path = save_overlay(
                    image,
                    roi_info,
                    mask,
                    features,
                    scores,
                    config
                )

                detections.append({
                    "roi_info": roi_info,
                    "mask": mask,
                    "features": features,
                    "scores": scores,
                    "overlay_path": overlay_path,
                    "mask_path": mask_path,
                    "roi_path": roi_path,
                    "sample": sample,
                })

                rows.append({
                    "image_id": sample["image_id"],
                    "image_path": sample["image_path"],
                    "target_id": sample["target_id"],
                    "bbox": sample["bbox"],
                    **features,
                    **scores,
                    "mask_path": mask_path,
                    "overlay_path": overlay_path,
                    "roi_path": roi_path,
                })

            except Exception as e:
                failed_rows.append({
                    "image_id": sample["image_id"],
                    "target_id": sample["target_id"],
                    "reason": str(e)
                })

        image_overlay_path = save_image_overlay(
            image,
            detections,
            config,
            image_id=image_id,
        )

        for row in rows:
            if row.get("image_id") == image_id and "image_overlay_path" not in row:
                row["image_overlay_path"] = image_overlay_path

        print(f"saved image overlay: {image_overlay_path}")


    output_dir = config["data"]["output_dir"]

    save_csv(
        rows,
        os.path.join(output_dir, "morphology_scores.csv")
    )

    save_csv(
        failed_rows,
        os.path.join(output_dir, "failed_cases.csv")
    )

    print("\nFinished!")
    print(f"success: {len(rows)}")
    print(f"failed: {len(failed_rows)}")