import random
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_IMAGE_ROOT = WORKSPACE_ROOT / "\u56fe\u2014\u2014\u7cbe\u5b50"
DEFAULT_LABEL_ROOT = WORKSPACE_ROOT / "\u7cbe\u5b50\u5305\u56f4\u6846_\u5f20\u73ee\u7476"
DATASET_ROOT = ROOT / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_image_unicode(image_path):
    """Read an image reliably on Windows paths that contain Chinese characters."""
    import cv2

    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image_unicode(image_path, image):
    import cv2

    suffix = Path(image_path).suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(str(image_path))
    return True


def resolve_source_path(path):
    path = Path(path)
    if path.is_absolute():
        return path

    for base in (Path.cwd(), PROJECT_ROOT, WORKSPACE_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate

    return PROJECT_ROOT / path


def iter_image_files(image_dir):
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def convert_bbox_to_yolo(box, image_w, image_h):
    x1, y1, x2, y2 = map(float, box)
    cx = (x1 + x2) / 2.0 / image_w
    cy = (y1 + y2) / 2.0 / image_h
    width = (x2 - x1) / image_w
    height = (y2 - y1) / image_h
    return [cx, cy, width, height]


def yolo_to_xyxy(values, image_w, image_h):
    if image_w is None or image_h is None:
        return None

    cx, cy, width, height = values
    if width <= 0 or height <= 0:
        return None

    x1 = (cx - width / 2.0) * image_w
    y1 = (cy - height / 2.0) * image_h
    x2 = (cx + width / 2.0) * image_w
    y2 = (cy + height / 2.0) * image_h
    return [x1, y1, x2, y2]


def xyxy_from_values(values):
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def parse_box_values(values, image_w=None, image_h=None, label_format="auto"):
    if label_format not in {"auto", "xyxy", "yolo"}:
        raise ValueError("label_format must be auto, xyxy, or yolo")

    is_normalized = all(0.0 <= value <= 1.0 for value in values)
    if label_format == "yolo" or (label_format == "auto" and is_normalized):
        return yolo_to_xyxy(values, image_w, image_h)

    return xyxy_from_values(values)


def read_boxes(label_path, image_w=None, image_h=None, label_format="auto"):
    boxes = []
    if not label_path.exists():
        return boxes

    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            values = list(map(float, parts[1:5]))
        except ValueError:
            continue

        box = parse_box_values(values, image_w, image_h, label_format)
        if box is None:
            continue
        boxes.append(box)

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


def convert_single_label(src_label, dst_label, image_path, label_format="auto"):
    img = read_image_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    h, w = img.shape[:2]
    boxes = read_boxes(src_label, w, h, label_format)
    return write_yolo_label(boxes, dst_label, w, h)


def reset_dataset_dirs():
    if DATASET_ROOT.exists():
        for cache_path in DATASET_ROOT.rglob("*.cache"):
            cache_path.unlink(missing_ok=True)

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            path = DATASET_ROOT / kind / split
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)


def collect_labeled_images(image_dir, label_dir, label_format="auto"):
    image_files = iter_image_files(image_dir)
    if not image_files:
        raise FileNotFoundError(f"no images found in: {image_dir}")

    labeled = []
    missing = []
    empty = []
    unreadable = []
    label_files = sorted(label_dir.glob("*.txt"))
    image_stems = {path.stem for path in image_files}
    extra_labels = [path for path in label_files if path.stem not in image_stems]

    for image_path in image_files:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing.append(image_path)
            continue

        image = read_image_unicode(image_path)
        if image is None:
            unreadable.append(image_path)
            continue

        h, w = image.shape[:2]
        if not read_boxes(label_path, w, h, label_format):
            empty.append(image_path)
            continue
        labeled.append(image_path)

    stats = {
        "source_images": len(image_files),
        "label_files": len(label_files),
        "labeled_images": len(labeled),
        "missing_label_images": len(missing),
        "empty_label_images": len(empty),
        "unreadable_images": len(unreadable),
        "extra_label_files": len(extra_labels),
    }

    return labeled, stats


def split_labeled_images(image_dir, label_dir, val_ratio=0.2, seed=20260719, label_format="auto"):
    labeled, stats = collect_labeled_images(image_dir, label_dir, label_format)

    if len(labeled) < 2:
        raise ValueError("not enough labeled images for train/val split")

    rng = random.Random(seed)
    rng.shuffle(labeled)

    val_count = max(1, int(round(len(labeled) * float(val_ratio))))
    val_count = min(val_count, len(labeled) - 1)

    train_stats = stats.copy()
    val_stats = stats.copy()
    train_stats["labeled_images"] = len(labeled) - val_count
    val_stats["labeled_images"] = val_count

    return labeled[val_count:], labeled[:val_count], train_stats, val_stats


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


def convert_full_image(image_path, split, label_dir, label_format="auto"):
    label_path = label_dir / f"{image_path.stem}.txt"
    target_image = DATASET_ROOT / "images" / split / image_path.name
    target_label = DATASET_ROOT / "labels" / split / label_path.name
    shutil.copy2(image_path, target_image)
    count = convert_single_label(label_path, target_label, image_path, label_format)
    if count == 0:
        target_image.unlink(missing_ok=True)
        target_label.unlink(missing_ok=True)
    return count


def convert_tiled_image(image_path, split, label_dir, tile_size=512, overlap=0.25, label_format="auto"):
    image = read_image_unicode(image_path)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    height, width = image.shape[:2]
    boxes = read_boxes(label_dir / f"{image_path.stem}.txt", width, height, label_format)
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

            if not write_image_unicode(target_image, tile):
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
        f.write("test: images/val\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  - sperm\n")

    return ROOT / "dataset.yaml"


def convert_label_files(
    image_root=DEFAULT_IMAGE_ROOT,
    label_root=DEFAULT_LABEL_ROOT,
    train_split="1",
    val_split="2",
    val_ratio=0.2,
    seed=20260719,
    tile=True,
    tile_size=512,
    overlap=0.25,
    label_format="auto",
    random_split=False,
):
    image_root = resolve_source_path(image_root)
    label_root = resolve_source_path(label_root)

    if random_split:
        train_image_dir = image_root
        val_image_dir = image_root
        train_label_dir = label_root
        val_label_dir = label_root
    else:
        train_image_dir = image_root / str(train_split)
        val_image_dir = image_root / str(val_split)
        train_label_dir = label_root / str(train_split)
        val_label_dir = label_root / str(val_split)

    for path in (train_image_dir, val_image_dir, train_label_dir, val_label_dir):
        if not path.exists():
            raise FileNotFoundError(f"directory not found: {path}")

    reset_dataset_dirs()
    if random_split:
        train_images, val_images, train_stats, val_stats = split_labeled_images(
            train_image_dir,
            train_label_dir,
            val_ratio=val_ratio,
            seed=seed,
            label_format=label_format,
        )
    else:
        train_images, train_stats = collect_labeled_images(
            train_image_dir,
            train_label_dir,
            label_format=label_format,
        )
        val_images, val_stats = collect_labeled_images(
            val_image_dir,
            val_label_dir,
            label_format=label_format,
        )

    if not train_images:
        raise ValueError(f"no usable training images found in: {train_image_dir}")
    if not val_images:
        raise ValueError(f"no usable validation images found in: {val_image_dir}")

    summary = {
        "image_root": image_root,
        "label_root": label_root,
        "train_image_dir": train_image_dir,
        "val_image_dir": val_image_dir,
        "train_label_dir": train_label_dir,
        "val_label_dir": val_label_dir,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "train_items": 0,
        "val_items": 0,
        "train_boxes": 0,
        "val_boxes": 0,
    }

    for split, images, label_dir in (
        ("train", train_images, train_label_dir),
        ("val", val_images, val_label_dir),
    ):
        for image_path in images:
            if tile:
                item_count, box_count = convert_tiled_image(
                    image_path,
                    split,
                    label_dir,
                    tile_size=tile_size,
                    overlap=overlap,
                    label_format=label_format,
                )
            else:
                box_count = convert_full_image(image_path, split, label_dir, label_format)
                item_count = 1 if box_count > 0 else 0

            summary[f"{split}_items"] += item_count
            summary[f"{split}_boxes"] += box_count

    dataset_yaml = write_dataset_yaml()
    print_dataset_summary(summary, tile, tile_size, overlap)
    return dataset_yaml


def print_dataset_summary(summary, tile, tile_size, overlap):
    mode = "tiles" if tile else "full images"
    print(f"image root: {summary['image_root']}")
    print(f"label root: {summary['label_root']}")
    print(f"train images dir: {summary['train_image_dir']}")
    print(f"train labels dir: {summary['train_label_dir']}")
    print(f"val images dir: {summary['val_image_dir']}")
    print(f"val labels dir: {summary['val_label_dir']}")
    print(f"dataset mode: {mode}")
    if tile:
        print(f"tile_size: {tile_size}, overlap: {overlap}")
    print_split_summary("train", summary["train_stats"])
    print_split_summary("val", summary["val_stats"])
    print(f"train dataset items: {summary['train_items']}")
    print(f"val dataset items: {summary['val_items']}")
    print(f"train boxes: {summary['train_boxes']}")
    print(f"val boxes: {summary['val_boxes']}")


def print_split_summary(name, stats):
    print(f"{name} source images: {stats['source_images']}")
    print(f"{name} label files: {stats['label_files']}")
    print(f"{name} usable labeled images: {stats['labeled_images']}")
    print(f"{name} skipped missing labels: {stats['missing_label_images']}")
    print(f"{name} skipped empty labels: {stats['empty_label_images']}")
    print(f"{name} skipped unreadable images: {stats['unreadable_images']}")
    print(f"{name} extra label files without images: {stats['extra_label_files']}")
