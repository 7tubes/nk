from __future__ import annotations

import argparse

from src.labelme_to_yolo_seg import DEFAULT_DATASET_DIR, DEFAULT_SOURCE_DIR
from src.labelme_to_yolo_seg import convert_labelme_dataset, print_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LabelMe head polygons to YOLOv8 segmentation format.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="LabelMe image/json directory")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="output YOLO-Seg dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="validation image ratio")
    parser.add_argument("--seed", type=int, default=20260813, help="random split seed")
    parser.add_argument("--full-image", action="store_true", help="do not tile images")
    parser.add_argument("--tile-size", type=int, default=320, help="tile size for small head segmentation")
    parser.add_argument("--overlap", type=float, default=0.25, help="tile overlap ratio")
    parser.add_argument("--min-area", type=float, default=6.0, help="minimum clipped polygon area in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    print(f"dataset prepared: {dataset_yaml}")


if __name__ == "__main__":
    main()
