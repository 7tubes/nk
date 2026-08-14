from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from sperm_morphology.features import compute_features
from sperm_morphology.dataset import load_config
from sperm_morphology.scoring import score_features
from sperm_morphology.utils import normalize_gray_uint8, read_image_unicode, write_image_unicode


DEFAULT_MODEL = ROOT / "runs" / "sperm_head_seg" / "weights" / "best.pt"
DEFAULT_INPUT = ROOT / "datasets"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "morphology.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
YOLO_CONFIG_DIR = ROOT


def iter_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def result_masks(result, image_shape: tuple[int, int], min_conf: float, offset_x: int = 0, offset_y: int = 0) -> list[dict]:
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or getattr(masks, "data", None) is None:
        return []

    try:
        mask_array = masks.data.detach().cpu().numpy()
    except AttributeError:
        mask_array = np.asarray(masks.data)

    confidences = []
    xyxy_values = []
    if boxes is not None:
        if getattr(boxes, "conf", None) is not None:
            try:
                confidences = boxes.conf.detach().cpu().numpy().astype(float).tolist()
            except AttributeError:
                confidences = np.asarray(boxes.conf, dtype=float).tolist()
        if getattr(boxes, "xyxy", None) is not None:
            try:
                xyxy_values = boxes.xyxy.detach().cpu().numpy().astype(float).tolist()
            except AttributeError:
                xyxy_values = np.asarray(boxes.xyxy, dtype=float).tolist()

    image_h, image_w = image_shape[:2]
    output = []
    for index, raw_mask in enumerate(mask_array):
        confidence = float(confidences[index]) if index < len(confidences) else 1.0
        if confidence < min_conf:
            continue
        mask = (raw_mask > 0.5).astype(np.uint8) * 255
        if mask.shape[:2] != (image_h, image_w):
            mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
        if int(np.count_nonzero(mask)) == 0:
            continue
        if index < len(xyxy_values):
            x1, y1, x2, y2 = xyxy_values[index]
            bbox = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
        else:
            bbox = ""
        output.append({"mask": mask, "confidence": confidence, "bbox": bbox})
    return output


def bbox_from_mask(mask) -> list[float]:
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return [0.0, 0.0, 1.0, 1.0]
    x, y, w, h = cv2.boundingRect(points)
    return [float(x), float(y), float(x + w), float(y + h)]


def suppress_duplicate_masks(mask_results: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    if not mask_results:
        return []

    boxes_xywh = []
    scores = []
    for item in mask_results:
        bbox = item["bbox"] if item.get("bbox") else bbox_from_mask(item["mask"])
        x1, y1, x2, y2 = [float(v) for v in bbox]
        boxes_xywh.append([int(round(x1)), int(round(y1)), int(round(max(x2 - x1, 1))), int(round(max(y2 - y1, 1)))])
        scores.append(float(item["confidence"]))

    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.0, nms_threshold=float(iou_threshold))
    if len(indices) == 0:
        return []
    picked = [mask_results[int(index)] for index in indices.flatten()]
    picked.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return picked


def predict_masks(
    model,
    image,
    conf: float,
    imgsz: int,
    max_det: int,
    use_tiling: bool,
    tile_size: int,
    overlap: float,
) -> list[dict]:
    if not use_tiling:
        result = model(
            image,
            conf=float(conf),
            imgsz=int(imgsz),
            max_det=int(max_det),
            stream=False,
            verbose=False,
        )[0]
        return result_masks(result, image.shape, float(conf))

    image_h, image_w = image.shape[:2]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    merged_masks = []
    for y0 in tile_starts(image_h, tile_size, stride):
        for x0 in tile_starts(image_w, tile_size, stride):
            x1 = min(x0 + tile_size, image_w)
            y1 = min(y0 + tile_size, image_h)
            tile = image[y0:y1, x0:x1]
            result = model(
                tile,
                conf=float(conf),
                imgsz=int(imgsz),
                max_det=int(max_det),
                stream=False,
                verbose=False,
            )[0]

            for mask_result in result_masks(result, tile.shape, float(conf), x0, y0):
                full_mask = np.zeros((image_h, image_w), dtype=np.uint8)
                tile_mask = mask_result["mask"]
                full_mask[y0:y1, x0:x1] = np.maximum(full_mask[y0:y1, x0:x1], tile_mask[: y1 - y0, : x1 - x0])
                mask_result["mask"] = full_mask
                merged_masks.append(mask_result)

    return suppress_duplicate_masks(merged_masks)


def save_overlay(image, mask, output_path: Path) -> None:
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours, -1, (0, 255, 255), 2)
    overlay = canvas.copy()
    overlay[mask > 0] = (0.45 * overlay[mask > 0] + 0.55 * np.array([0, 220, 255])).astype(np.uint8)
    write_image_unicode(output_path, overlay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sperm head masks, features, and morphology grades with YOLOv8-Seg.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="image file or image folder")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLOv8-Seg weights")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="morphology scoring config yaml")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"), help="mask/overlay/csv output directory")
    parser.add_argument("--conf", type=float, default=0.15, help="minimum segmentation confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--max-det", type=int, default=300, help="maximum masks per image")
    parser.add_argument("--tile-size", type=int, default=320, help="tile size for full-image inference")
    parser.add_argument("--overlap", type=float, default=0.25, help="tile overlap ratio")
    parser.add_argument("--no-tiling", action="store_true", help="disable tiled inference")
    args = parser.parse_args()

    os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
    from ultralytics import YOLO

    input_path = Path(args.input)
    model_path = Path(args.model)
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"YOLOv8-Seg model not found: {model_path}")

    model = YOLO(str(model_path))
    rows = []
    for image_path in iter_images(input_path):
        image = read_image_unicode(image_path)
        if image is None:
            continue

        gray = normalize_gray_uint8(image, name="image")
        mask_results = predict_masks(
            model=model,
            image=image,
            conf=float(args.conf),
            imgsz=int(args.imgsz),
            max_det=int(args.max_det),
            use_tiling=not args.no_tiling,
            tile_size=int(args.tile_size),
            overlap=float(args.overlap),
        )
        print(f"{image_path.name}: {len(mask_results)} masks")

        for index, mask_result in enumerate(mask_results):
            mask = mask_result["mask"]
            stem = f"{image_path.stem}_head{index:03d}"
            mask_path = mask_dir / f"{stem}_mask.png"
            overlay_path = overlay_dir / f"{stem}_overlay.png"
            write_image_unicode(mask_path, mask)
            save_overlay(image, mask, overlay_path)
            features = compute_features(mask, gray, config=config)
            scores = score_features(features, config)

            rows.append(
                {
                    "image_id": image_path.stem,
                    "head_id": index,
                    "image_path": str(image_path),
                    "confidence": mask_result["confidence"],
                    "bbox": mask_result["bbox"],
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                    **features,
                    **scores,
                }
            )

    csv_path = output_dir / "head_segmentation_features.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["image_id"]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"images: {len(iter_images(input_path))}")
    print(f"masks: {len(rows)}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
