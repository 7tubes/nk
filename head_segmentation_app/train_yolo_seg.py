from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

from src.labelme_to_yolo_seg import DEFAULT_DATASET_DIR, DEFAULT_SOURCE_DIR
from src.labelme_to_yolo_seg import convert_labelme_dataset, print_summary


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "runs"
YOLO_CONFIG_DIR = ROOT


def resolve_model_path(model_name: str) -> str:
    local_model = ROOT / model_name
    if local_model.exists():
        return str(local_model)
    return model_name


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8-Seg for sperm head segmentation.")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="YOLOv8 segmentation checkpoint")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="LabelMe image/json directory")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="output YOLO-Seg dataset directory")
    parser.add_argument("--epochs", type=int, default=120, help="training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="training image size")
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    parser.add_argument("--name", default="sperm_head_seg", help="run name under head_segmentation_app/runs")
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device id such as 0")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="validation image ratio")
    parser.add_argument("--seed", type=int, default=20260813, help="random split/training seed")
    parser.add_argument("--full-image", action="store_true", help="disable tile dataset conversion")
    parser.add_argument("--tile-size", type=int, default=320, help="tile size for small head segmentation")
    parser.add_argument("--overlap", type=float, default=0.25, help="tile overlap ratio")
    parser.add_argument("--min-area", type=float, default=6.0, help="minimum clipped polygon area in pixels")
    parser.add_argument("--prepare-only", action="store_true", help="only convert dataset and exit")
    parser.add_argument("--no-prepare", action="store_true", help="reuse existing dataset.yaml without conversion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

    if args.no_prepare:
        dataset_yaml = ROOT / "dataset.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"dataset yaml not found: {dataset_yaml}")
    else:
        dataset_yaml, stats = convert_labelme_dataset(
            source_dir=args.source_dir,
            dataset_dir=args.dataset_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            tile=not args.full_image,
            tile_size=args.tile_size,
            overlap=args.overlap,
            min_area=args.min_area,
        )
        print_summary(stats, tile=not args.full_image, tile_size=args.tile_size, overlap=args.overlap)

    if args.prepare_only:
        print(f"dataset prepared: {dataset_yaml}")
        return

    model = YOLO(resolve_model_path(args.model))
    train_kwargs = {
        "data": str(dataset_yaml),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "project": str(OUTPUT_DIR),
        "name": args.name,
        "workers": 0,
        "exist_ok": True,
        "pretrained": True,
        "single_cls": True,
        "cache": True,
        "patience": 30,
        "optimizer": "AdamW",
        "lr0": 0.002,
        "lrf": 0.01,
        "cos_lr": True,
        "close_mosaic": 15,
        "mosaic": 0.2,
        "mixup": 0.0,
        "copy_paste": 0.1,
        "scale": 0.20,
        "translate": 0.05,
        "degrees": 8.0,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.01,
        "hsv_s": 0.25,
        "hsv_v": 0.25,
        "plots": True,
        "seed": args.seed,
        "deterministic": True,
    }
    if args.device != "auto":
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)
    print("training finished")
    print(f"best weights: {OUTPUT_DIR / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
