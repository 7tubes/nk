import argparse
import csv
import os
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sperm_morphology.dataset import load_config
from sperm_morphology.detection_screening import (
    save_screening_overlay,
    screen_detections,
)
from sperm_morphology.utils import read_image_unicode


DEFAULT_MODEL = ROOT / "deep_learning_app" / "runs" / "sperm_detection" / "weights" / "best.pt"
YOLO_CONFIG_DIR = ROOT


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """
    Return tile start positions that cover the full image.

    Sperm targets are very small in microscope images. Tiled inference lets the
    detector see each sperm at a larger relative scale, while this helper makes
    sure the final tile also reaches the image edge instead of leaving a strip
    unprocessed.
    """
    if length <= tile_size:
        return [0]

    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def detections_from_yolo_result(result, offset_x: int = 0, offset_y: int = 0) -> list[dict]:
    """Convert one Ultralytics result into the dictionary format used downstream."""
    detections = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        detections.append(
            {
                "xyxy": [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
                "conf": float(box.conf[0]),
                "source": "yolo",
            }
        )
    return detections


def nms_detections(detections: list[dict], iou_threshold: float, max_det: int) -> list[dict]:
    """
    Remove duplicate detections after tiled inference.

    Adjacent tiles overlap, so the same sperm can be detected more than once.
    OpenCV NMS keeps the highest-confidence box and removes boxes that overlap
    it too much.
    """
    if not detections:
        return []

    boxes_xywh = []
    scores = []
    for detection in detections:
        x1, y1, x2, y2 = detection["xyxy"]
        boxes_xywh.append([int(x1), int(y1), int(max(x2 - x1, 1)), int(max(y2 - y1, 1))])
        scores.append(float(detection["conf"]))

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=0.0,
        nms_threshold=float(iou_threshold),
        top_k=int(max_det),
    )
    if len(indices) == 0:
        return []

    picked = [detections[int(index)] for index in indices.flatten()]
    picked.sort(key=lambda item: item["conf"], reverse=True)
    return picked[: int(max_det)]


def predict_with_yolo(
    image,
    model_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    tile_size: int,
    overlap: float,
    use_tiling: bool,
) -> list[dict]:
    """
    Run YOLO recognition and return sperm candidate boxes.

    The import stays inside the function so tests that only use manual boxes do
    not require importing Ultralytics or loading PyTorch.
    """
    os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if not use_tiling:
        result = model(
            image,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            max_det=int(max_det),
            stream=False,
            verbose=False,
        )[0]
        return detections_from_yolo_result(result)

    height, width = image.shape[:2]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    detections = []

    for y0 in tile_starts(height, tile_size, stride):
        for x0 in tile_starts(width, tile_size, stride):
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            tile = image[y0:y1, x0:x1]
            result = model(
                tile,
                conf=float(conf),
                iou=float(iou),
                imgsz=int(imgsz),
                max_det=int(max_det),
                stream=False,
                verbose=False,
            )[0]
            detections.extend(detections_from_yolo_result(result, x0, y0))

    return nms_detections(detections, iou_threshold=iou, max_det=max_det)


def parse_manual_boxes(raw_boxes: list[str]) -> list[dict]:
    """Parse repeated --bbox values into detection dictionaries."""
    detections = []
    for index, raw_box in enumerate(raw_boxes):
        values = [float(part.strip()) for part in raw_box.split(",")]
        if len(values) != 4:
            raise ValueError("--bbox must be formatted as x1,y1,x2,y2")
        detections.append(
            {
                "xyxy": values,
                "conf": 1.0,
                "source": f"manual_{index}",
            }
        )
    return detections


def save_results_csv(screening_results: list[dict], output_csv: Path) -> None:
    """Save the key recognition and screening fields for later review."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in screening_results:
        scores = result.get("scores", {})
        features = result.get("features", {})
        rows.append(
            {
                "image_id": result.get("image_id", ""),
                "target_id": result.get("target_id", ""),
                "source": result.get("source", ""),
                "confidence": result.get("confidence", 0.0),
                "bbox": result.get("bbox", ""),
                "traffic_color": result.get("traffic_color", ""),
                "grade": scores.get("grade", "Reject"),
                "total_score": scores.get("total_score", 0.0),
                "reject_reason": scores.get("reject_reason", ""),
                "fit_iou": features.get("fit_iou", ""),
                "R": features.get("R", ""),
                "uniformity": features.get("uniformity", ""),
            }
        )

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys() if rows else ["image_id"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLO sperm recognition and morphology red/yellow/green screening on one image."
    )
    parser.add_argument("--image", required=True, help="path to a jpg/png image")
    parser.add_argument("--config", default="configs/morphology.yaml", help="morphology config yaml")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO sperm detector weights")
    parser.add_argument("--bbox", action="append", default=[], help="manual test box: x1,y1,x2,y2")
    parser.add_argument("--output", default="outputs/detection_screening/combined_overlay.png")
    parser.add_argument("--csv", default="outputs/detection_screening/combined_results.csv")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--no-tiling", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = read_image_unicode(image_path)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    config = load_config(args.config)

    if args.bbox:
        detections = parse_manual_boxes(args.bbox)
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            parser.error(
                f"YOLO model not found: {model_path}. Train the detector first or pass --bbox for integration testing."
            )
        detections = predict_with_yolo(
            image=image,
            model_path=model_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            tile_size=args.tile_size,
            overlap=args.overlap,
            use_tiling=not args.no_tiling,
        )

    image_id = image_path.stem
    screening_results = screen_detections(image, detections, config, image_id=image_id)
    overlay_path = save_screening_overlay(image, screening_results, args.output)
    save_results_csv(screening_results, Path(args.csv))

    counts = {"green": 0, "yellow": 0, "red": 0}
    for result in screening_results:
        counts[result["traffic_color"]] = counts.get(result["traffic_color"], 0) + 1

    print(f"recognized targets: {len(detections)}")
    print(f"green: {counts.get('green', 0)}")
    print(f"yellow: {counts.get('yellow', 0)}")
    print(f"red: {counts.get('red', 0)}")
    print(f"overlay: {overlay_path}")
    print(f"csv: {args.csv}")


if __name__ == "__main__":
    main()
