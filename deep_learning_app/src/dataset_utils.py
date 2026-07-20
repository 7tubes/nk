import random
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DATA_ROOT = PROJECT_ROOT.parent / "SpermTracking" / "ImagesWithLabels"
IMAGE_DIR = DATA_ROOT / "images"
LABEL_DIR = DATA_ROOT / "labels"
DATASET_ROOT = ROOT / "dataset"


def read_image_unicode(image_path):
    """Read an image reliably on Windows paths that contain Chinese characters."""
    import cv2

    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def convert_bbox_to_yolo(box, image_w, image_h):
    x1, y1, x2, y2 = map(float, box)
    cx = (x1 + x2) / 2.0 / image_w
    cy = (y1 + y2) / 2.0 / image_h
    width = (x2 - x1) / image_w
    height = (y2 - y1) / image_h
    return [cx, cy, width, height]


def read_boxes(label_path):
    boxes = []
    if not label_path.exists():
        return boxes

    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            x1, y1, x2, y2 = map(float, parts[1:5])
        except ValueError:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])

    return boxes


def clip_box(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0.0, min(x1, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def write_yolo_label(boxes, dst_label, image_w, image_h):
    lines = []
    for box in boxes:
        clipped = clip_box(box, image_w, image_h)
        if clipped is None:
            continue
        yolo_bbox = convert_bbox_to_yolo(clipped, image_w, image_h)
        if all(0.0 <= value <= 1.0 for value in yolo_bbox):
            lines.append("0 " + " ".join(f"{value:.8f}" for value in yolo_bbox))

    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def convert_single_label(src_label, dst_label, image_path):
    img = read_image_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    h, w = img.shape[:2]
    boxes = read_boxes(src_label)
    return write_yolo_label(boxes, dst_label, w, h)


def reset_dataset_dirs():
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            path = DATASET_ROOT / kind / split
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)


def split_labeled_images(val_ratio=0.2, seed=20260719):
    image_files = sorted(IMAGE_DIR.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError("no jpg images found")

    labeled = []
    missing = []
    empty = []

    for image_path in image_files:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing.append(image_path)
            continue
        if not read_boxes(label_path):
            empty.append(image_path)
            continue
        labeled.append(image_path)

    if len(labeled) < 2:
        raise ValueError("not enough labeled images for train/val split")

    rng = random.Random(seed)
    rng.shuffle(labeled)

    val_count = max(1, int(round(len(labeled) * float(val_ratio))))
    val_count = min(val_count, len(labeled) - 1)

    return labeled[val_count:], labeled[:val_count], missing, empty


def tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]

    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def boxes_in_tile(boxes, x0, y0, tile_w, tile_h):
    selected = []
    x1_tile = float(x0)
    y1_tile = float(y0)
    x2_tile = float(x0 + tile_w)
    y2_tile = float(y0 + tile_h)

    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if not (x1_tile <= cx < x2_tile and y1_tile <= cy < y2_tile):
            continue

        clipped = [
            max(x1, x1_tile) - x1_tile,
            max(y1, y1_tile) - y1_tile,
            min(x2, x2_tile) - x1_tile,
            min(y2, y2_tile) - y1_tile,
        ]
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            selected.append(clipped)

    return selected


def convert_full_image(image_path, split):
    label_path = LABEL_DIR / f"{image_path.stem}.txt"
    target_image = DATASET_ROOT / "images" / split / image_path.name
    target_label = DATASET_ROOT / "labels" / split / label_path.name
    shutil.copy2(image_path, target_image)
    count = convert_single_label(label_path, target_label, image_path)
    if count == 0:
        target_image.unlink(missing_ok=True)
        target_label.unlink(missing_ok=True)
    return count


def convert_tiled_image(image_path, split, tile_size=512, overlap=0.25):
    import cv2

    image = read_image_unicode(image_path)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    height, width = image.shape[:2]
    boxes = read_boxes(LABEL_DIR / f"{image_path.stem}.txt")
    stride = max(1, int(round(tile_size * (1.0 - float(overlap)))))
    x_starts = tile_starts(width, tile_size, stride)
    y_starts = tile_starts(height, tile_size, stride)

    written_tiles = 0
    written_boxes = 0

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            tile = image[y0:y1, x0:x1]
            tile_h, tile_w = tile.shape[:2]
            tile_boxes = boxes_in_tile(boxes, x0, y0, tile_w, tile_h)
            if not tile_boxes:
                continue

            tile_name = f"{image_path.stem}_x{x0:04d}_y{y0:04d}.jpg"
            label_name = f"{Path(tile_name).stem}.txt"
            target_image = DATASET_ROOT / "images" / split / tile_name
            target_label = DATASET_ROOT / "labels" / split / label_name

            ok = cv2.imwrite(str(target_image), tile)
            if not ok:
                raise OSError(f"failed to write tile: {target_image}")

            count = write_yolo_label(tile_boxes, target_label, tile_w, tile_h)
            if count == 0:
                target_image.unlink(missing_ok=True)
                target_label.unlink(missing_ok=True)
                continue

            written_tiles += 1
            written_boxes += count

    return written_tiles, written_boxes


def write_dataset_yaml():
    dataset_path = DATASET_ROOT.resolve().as_posix()
    dataset_path = dataset_path.encode("unicode_escape").decode("ascii")

    with open(ROOT / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(f'path: "{dataset_path}"\n')
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  - sperm\n")

    return ROOT / "dataset.yaml"


def convert_label_files(
    val_ratio=0.2,
    seed=20260719,
    tile=True,
    tile_size=512,
    overlap=0.25,
):
    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        raise FileNotFoundError("image/label directory not found")

    reset_dataset_dirs()
    train_images, val_images, missing, empty = split_labeled_images(val_ratio, seed)

    summary = {
        "train_images": len(train_images),
        "val_images": len(val_images),
        "missing_label_images": len(missing),
        "empty_label_images": len(empty),
        "train_items": 0,
        "val_items": 0,
        "train_boxes": 0,
        "val_boxes": 0,
    }

    for split, images in (("train", train_images), ("val", val_images)):
        for image_path in images:
            if tile:
                item_count, box_count = convert_tiled_image(
                    image_path,
                    split,
                    tile_size=tile_size,
                    overlap=overlap,
                )
            else:
                box_count = convert_full_image(image_path, split)
                item_count = 1 if box_count > 0 else 0

            summary[f"{split}_items"] += item_count
            summary[f"{split}_boxes"] += box_count

    dataset_yaml = write_dataset_yaml()
    print_dataset_summary(summary, tile, tile_size, overlap)
    return dataset_yaml


def print_dataset_summary(summary, tile, tile_size, overlap):
    mode = "tiles" if tile else "full images"
    print(f"dataset mode: {mode}")
    if tile:
        print(f"tile_size: {tile_size}, overlap: {overlap}")
    print(f"train source images: {summary['train_images']}")
    print(f"val source images: {summary['val_images']}")
    print(f"skipped missing labels: {summary['missing_label_images']}")
    print(f"skipped empty labels: {summary['empty_label_images']}")
    print(f"train dataset items: {summary['train_items']}")
    print(f"val dataset items: {summary['val_items']}")
    print(f"train boxes: {summary['train_boxes']}")
    print(f"val boxes: {summary['val_boxes']}")
