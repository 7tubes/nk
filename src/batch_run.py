import os
import csv
import cv2

from sperm_morphology.dataset import load_config, load_dataset
from sperm_morphology.crop import crop_roi

from sperm_morphology.preprocess import preprocess_roi
from sperm_morphology.segment_head import segment_head

from sperm_morphology.features import compute_features
from sperm_morphology.scoring import score_features
from sperm_morphology.visualize import save_overlay


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


def run_batch(config_path: str):
    """
    批量运行精子形态筛选流程
    """
    config = load_config(config_path)

    ensure_output_dirs(config)

    samples = load_dataset(config)

    rows = []
    failed_rows = []

    total = len(samples)

    for index, sample in enumerate(samples, start=1):

        try:
            print(
                f"[{index}/{total}] "
                f"{sample['image_id']} "
                f"target={sample['target_id']}"
            )

            image = read_image(sample["image_path"])

            roi_info = crop_roi(
                image,
                sample["bbox"],
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
                failed_rows.append({
                    "image_id": sample["image_id"],
                    "target_id": sample["target_id"],
                    "reason": quality_info.get(
                        "reason",
                        "segment_failed"
                    )
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

            overlay_path = save_overlay(
                image,
                roi_info,
                mask,
                features,
                scores,
                config
            )

            rows.append({
                "image_id": sample["image_id"],
                "image_path": sample["image_path"],
                "target_id": sample["target_id"],
                "bbox": sample["bbox"],
                **features,
                **scores,
                "mask_path": "",
                "overlay_path": overlay_path
            })

            print("success")

        except Exception as e:
            failed_rows.append({
                "image_id": sample["image_id"],
                "target_id": sample["target_id"],
                "reason": str(e)
            })


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