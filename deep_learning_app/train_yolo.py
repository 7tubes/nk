import argparse
from pathlib import Path

from ultralytics import YOLO

from src.dataset_utils import convert_label_files

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "runs"


def resolve_model_path(model_name):
    local_model = ROOT / model_name
    if local_model.exists():
        return str(local_model)
    return model_name


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for sperm detection.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO checkpoint, e.g. yolov8n.pt or yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--imgsz", type=int, default=768, help="training image size")
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    parser.add_argument("--name", default="sperm_detection", help="run name under runs/")
    parser.add_argument("--seed", type=int, default=20260719, help="random split/training seed")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="validation image ratio")
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device id such as 0")
    parser.add_argument("--full-image", action="store_true", help="disable tile training")
    parser.add_argument("--tile-size", type=int, default=512, help="tile size for tiny-object training")
    parser.add_argument("--overlap", type=float, default=0.25, help="tile overlap ratio")
    parser.add_argument("--prepare-only", action="store_true", help="only convert dataset and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_yaml = convert_label_files(
        val_ratio=args.val_ratio,
        seed=args.seed,
        tile=not args.full_image,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )

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
        "copy_paste": 0.0,
        "scale": 0.25,
        "translate": 0.05,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.01,
        "hsv_s": 0.4,
        "hsv_v": 0.3,
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
