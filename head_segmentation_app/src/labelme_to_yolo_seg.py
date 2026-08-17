from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_SOURCE_DIR = ROOT / "datasets"
DEFAULT_DATASET_DIR = ROOT / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class PolygonObject:
    points: np.ndarray
    label: str


def read_image_unicode(path: str | Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image_unicode(path: str | Path, image) -> bool:
    suffix = Path(path).suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), PROJECT_ROOT, WORKSPACE_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / path


def reset_dataset_dirs(dataset_dir: Path) -> None:
    for cache_path in dataset_dir.rglob("*.cache") if dataset_dir.exists() else []:
        cache_path.unlink(missing_ok=True)
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            path = dataset_dir / kind / split
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)


def find_image_for_json(json_path: Path, labelme_data: dict, source_dir: Path) -> Path | None:
    image_path_value = labelme_data.get("imagePath")
    candidates = []
    if image_path_value:
        candidates.append(json_path.parent / image_path_value)
        candidates.append(source_dir / image_path_value)
    for extension in IMAGE_EXTENSIONS:
        candidates.append(json_path.with_suffix(extension))
        candidates.append(source_dir / f"{json_path.stem}{extension}")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_labelme_polygons(
    json_path: Path,
    source_dir: Path,
    class_names: tuple[str, ...] = ("head",),
) -> tuple[Path | None, int, int, list[PolygonObject]]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    image_path = find_image_for_json(json_path, data, source_dir)
    image_width = int(data.get("imageWidth") or 0)
    image_height = int(data.get("imageHeight") or 0)

    if (image_width <= 0 or image_height <= 0) and image_path is not None:
        image = read_image_unicode(image_path)
        if image is not None:
            image_height, image_width = image.shape[:2]

    if image_width <= 0 or image_height <= 0:
        return image_path, image_width, image_height, []

    allowed_labels = {name.lower() for name in class_names}
    objects: list[PolygonObject] = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        if label.lower() not in allowed_labels:
            continue
        if str(shape.get("shape_type", "polygon")).lower() != "polygon":
            continue

        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            continue
        points[:, 0] = np.clip(points[:, 0], 0, image_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, image_height - 1)
        if abs(cv2.contourArea(points.astype(np.float32))) <= 1.0:
            continue
        objects.append(PolygonObject(points=points, label=label))

    return image_path, image_width, image_height, objects


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def polygon_center(points: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(points.astype(np.float32))
    if abs(moments["m00"]) > 1e-6:
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    return float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))


def contours_for_tile(
    obj: PolygonObject,
    x0: int,
    y0: int,
    tile_w: int,
    tile_h: int,
    min_area: float,
) -> list[np.ndarray]:
    cx, cy = polygon_center(obj.points)
    if not (x0 <= cx < x0 + tile_w and y0 <= cy < y0 + tile_h):
        return []

    local_points = obj.points.copy()
    local_points[:, 0] -= x0
    local_points[:, 1] -= y0

    mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(local_points).astype(np.int32)], 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = max(0.5, 0.002 * cv2.arcLength(contour, closed=True))
        approx = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
        if approx.shape[0] >= 3:
            output.append(approx.astype(np.float32))
    return output


def polygon_to_yolo_line(points: np.ndarray, image_w: int, image_h: int, class_id: int = 0) -> str | None:
    if points.ndim != 2 or points.shape[0] < 3:
        return None
    points = points.astype(np.float32).copy()
    points[:, 0] = np.clip(points[:, 0] / float(image_w), 0.0, 1.0)
    points[:, 1] = np.clip(points[:, 1] / float(image_h), 0.0, 1.0)
    flattened = points.reshape(-1)
    if flattened.size < 6:
        return None
    return f"{class_id} " + " ".join(f"{value:.8f}" for value in flattened)


def write_labels(label_path: Path, polygons: list[np.ndarray], image_w: int, image_h: int) -> int:
    lines = []
    for polygon in polygons:
        line = polygon_to_yolo_line(polygon, image_w, image_h, class_id=0)
        if line is not None:
            lines.append(line)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def convert_full_image(
    image_path: Path,
    image,
    objects: list[PolygonObject],
    split: str,
    dataset_dir: Path,
) -> tuple[int, int]:
    target_image = dataset_dir / "images" / split / image_path.name
    target_label = dataset_dir / "labels" / split / f"{image_path.stem}.txt"
    if not write_image_unicode(target_image, image):
        raise OSError(f"failed to write image: {target_image}")
    count = write_labels(target_label, [obj.points for obj in objects], image.shape[1], image.shape[0])
    if count == 0:
        target_image.unlink(missing_ok=True)
        target_label.unlink(missing_ok=True)
        return 0, 0
    return 1, count


def convert_tiled_image(
    image_path: Path,
    image,
    objects: list[PolygonObject],
    split: str,
    dataset_dir: Path,
    tile_size: int,
    overlap: float,
    min_area: float,
) -> tuple[int, int]:
    height, width = image.shape[:2]
    stride = max(1, int(round(tile_size * (1.0 - float(overlap)))))
    written_tiles = 0
    written_masks = 0

    for y0 in tile_starts(height, tile_size, stride):
        for x0 in tile_starts(width, tile_size, stride):
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            tile = image[y0:y1, x0:x1]
            tile_h, tile_w = tile.shape[:2]

            polygons = []
            for obj in objects:
                polygons.extend(contours_for_tile(obj, x0, y0, tile_w, tile_h, min_area))
            if not polygons:
                continue

            tile_name = f"{image_path.stem}_x{x0:04d}_y{y0:04d}.jpg"
            target_image = dataset_dir / "images" / split / tile_name
            target_label = dataset_dir / "labels" / split / f"{Path(tile_name).stem}.txt"
            if not write_image_unicode(target_image, tile):
                raise OSError(f"failed to write tile: {target_image}")

            count = write_labels(target_label, polygons, tile_w, tile_h)
            if count == 0:
                target_image.unlink(missing_ok=True)
                target_label.unlink(missing_ok=True)
                continue

            written_tiles += 1
            written_masks += count

    return written_tiles, written_masks


def write_dataset_yaml(dataset_dir: Path, class_names: tuple[str, ...]) -> Path:
    yaml_path = ROOT / "dataset.yaml"
    dataset_path = dataset_dir.resolve().as_posix()
    with yaml_path.open("w", encoding="utf-8") as file:
        file.write(f'path: "{dataset_path}"\n')
        file.write("train: images/train\n")
        file.write("val: images/val\n")
        file.write("test: images/val\n")
        file.write(f"nc: {len(class_names)}\n")
        file.write("names:\n")
        for name in class_names:
            file.write(f"  - {name}\n")
    return yaml_path


def convert_labelme_dataset(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    class_names: tuple[str, ...] = ("head",),
    val_ratio: float = 0.2,
    seed: int = 20260813,
    tile: bool = True,
    tile_size: int = 320,
    overlap: float = 0.25,
    min_area: float = 6.0,
) -> tuple[Path, dict]:
    source_dir = resolve_path(source_dir)
    dataset_dir = resolve_path(dataset_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"source dir not found: {source_dir}")

    json_files = sorted(path for path in source_dir.glob("*.json") if path.is_file())
    samples = []
    skipped = {"missing_image": 0, "empty_head": 0, "unreadable": 0, "bad_size": 0}
    for json_path in json_files:
        image_path, image_w, image_h, objects = load_labelme_polygons(json_path, source_dir, class_names)
        if image_path is None:
            skipped["missing_image"] += 1
            continue
        if image_w <= 0 or image_h <= 0:
            skipped["bad_size"] += 1
            continue
        if not objects:
            skipped["empty_head"] += 1
            continue
        image = read_image_unicode(image_path)
        if image is None:
            skipped["unreadable"] += 1
            continue
        samples.append((image_path, image, objects))

    if len(samples) < 2:
        raise ValueError("not enough LabelMe files with head polygons")

    rng = random.Random(seed)
    rng.shuffle(samples)
    val_count = max(1, int(round(len(samples) * float(val_ratio))))
    val_count = min(val_count, len(samples) - 1)

    reset_dataset_dirs(dataset_dir)
    split_samples = {"train": samples[val_count:], "val": samples[:val_count]}
    stats = {
        "source_dir": str(source_dir),
        "json_files": len(json_files),
        "usable_images": len(samples),
        "train_images": len(split_samples["train"]),
        "val_images": len(split_samples["val"]),
        "train_items": 0,
        "val_items": 0,
        "train_masks": 0,
        "val_masks": 0,
        **{f"skipped_{key}": value for key, value in skipped.items()},
    }

    for split, items in split_samples.items():
        for image_path, image, objects in items:
            if tile:
                item_count, mask_count = convert_tiled_image(
                    image_path=image_path,
                    image=image,
                    objects=objects,
                    split=split,
                    dataset_dir=dataset_dir,
                    tile_size=tile_size,
                    overlap=overlap,
                    min_area=min_area,
                )
            else:
                item_count, mask_count = convert_full_image(
                    image_path=image_path,
                    image=image,
                    objects=objects,
                    split=split,
                    dataset_dir=dataset_dir,
                )
            stats[f"{split}_items"] += item_count
            stats[f"{split}_masks"] += mask_count

    dataset_yaml = write_dataset_yaml(dataset_dir, class_names)
    stats["dataset_yaml"] = str(dataset_yaml)
    stats["dataset_dir"] = str(dataset_dir)
    return dataset_yaml, stats


def print_summary(stats: dict, tile: bool, tile_size: int, overlap: float) -> None:
    mode = "tiles" if tile else "full images"
    print(f"source dir: {stats['source_dir']}")
    print(f"dataset dir: {stats['dataset_dir']}")
    print(f"dataset yaml: {stats['dataset_yaml']}")
    print(f"dataset mode: {mode}")
    if tile:
        print(f"tile_size: {tile_size}, overlap: {overlap}")
    print(f"json files: {stats['json_files']}")
    print(f"usable labeled images: {stats['usable_images']}")
    print(f"train source images: {stats['train_images']}")
    print(f"val source images: {stats['val_images']}")
    print(f"train dataset items: {stats['train_items']}")
    print(f"val dataset items: {stats['val_items']}")
    print(f"train head masks: {stats['train_masks']}")
    print(f"val head masks: {stats['val_masks']}")
    print(f"skipped missing image: {stats['skipped_missing_image']}")
    print(f"skipped empty head: {stats['skipped_empty_head']}")
    print(f"skipped unreadable: {stats['skipped_unreadable']}")
    print(f"skipped bad size: {stats['skipped_bad_size']}")
