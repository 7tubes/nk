import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ROOT = Path(__file__).resolve().parent


def list_images(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def normalize_u8(score):
    score = score.astype(np.float32)
    low, high = np.percentile(score, (1, 99))
    if high <= low:
        return np.zeros_like(score, dtype=np.uint8)
    score = (score - low) * 255.0 / (high - low)
    return np.clip(score, 0, 255).astype(np.uint8)


def gray_world_balance(image_bgr):
    image = image_bgr.astype(np.float32)
    channel_means = image.reshape(-1, 3).mean(axis=0)
    gray_mean = channel_means.mean()
    scale = gray_mean / np.maximum(channel_means, 1.0)
    balanced = image * scale.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


def correct_background(image_bgr, blur_sigma):
    if blur_sigma <= 0:
        return image_bgr
    image = image_bgr.astype(np.float32)
    background = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    corrected = image / np.maximum(background, 1.0) * 210.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def preprocess_image(image_bgr, args):
    image = image_bgr
    if args.gray_world:
        image = gray_world_balance(image)
    image = correct_background(image, args.background_sigma)
    if args.denoise > 0:
        image = cv2.fastNlMeansDenoisingColored(image, None, args.denoise, args.denoise, 7, 21)
    if args.median > 1:
        ksize = args.median if args.median % 2 == 1 else args.median + 1
        image = cv2.medianBlur(image, ksize)
    return image


def fill_holes(mask):
    if mask.max() == 0:
        return mask
    flood = mask.copy()
    h, w = flood.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def keep_seeded_weak_regions(weak_mask, seed_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weak_mask, connectivity=8)
    kept = np.zeros_like(weak_mask)
    for label_id in range(1, num_labels):
        x, y, w, h, _ = stats[label_id]
        region = labels[y : y + h, x : x + w] == label_id
        seed_region = seed_mask[y : y + h, x : x + w] > 0
        if np.any(region & seed_region):
            kept[labels == label_id] = 255
    return kept


def repair_mask_shape(mask, mode):
    if mode == "none" or mask.max() == 0:
        return mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    repaired = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) < 3:
            continue
        if mode == "ellipse" and len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(repaired, ellipse, 255, thickness=-1)
        else:
            hull = cv2.convexHull(contour)
            cv2.drawContours(repaired, [hull], -1, 255, thickness=-1)
    return repaired


def make_stain_mask(image_bgr, args):
    image = preprocess_image(image_bgr, args)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(image)
    _, sat, val = cv2.split(hsv)
    _, lab_a, lab_b = cv2.split(lab)

    hue = hsv[:, :, 0]
    hue_ok = (hue >= args.hue_min) & (hue <= args.hue_max)
    chroma = np.sqrt((lab_a.astype(np.float32) - 128.0) ** 2 + (lab_b.astype(np.float32) - 128.0) ** 2)
    purple_excess = np.maximum(((b.astype(np.float32) + r.astype(np.float32)) * 0.5) - g.astype(np.float32), 0)
    stain_score = normalize_u8(0.55 * sat.astype(np.float32) + 0.25 * chroma + 0.20 * purple_excess)

    weak_mask = (
        hue_ok
        & (sat >= args.weak_sat_min)
        & (val >= args.val_min)
        & (stain_score >= args.weak_score_min)
    ).astype(np.uint8) * 255
    seed_mask = (
        hue_ok
        & (sat >= args.seed_sat_min)
        & (val >= args.val_min)
        & (stain_score >= args.seed_score_min)
    ).astype(np.uint8) * 255

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    weak_mask = cv2.morphologyEx(weak_mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    weak_mask = cv2.morphologyEx(weak_mask, cv2.MORPH_CLOSE, kernel5, iterations=2)

    mask = keep_seeded_weak_regions(weak_mask, seed_mask)
    mask = fill_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)
    mask = repair_mask_shape(mask, args.shape_repair)
    if args.final_erode > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (args.final_erode * 2 + 1, args.final_erode * 2 + 1),
        )
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask, seed_mask, weak_mask, stain_score


def component_candidates(mask, min_area, max_area, min_aspect, max_aspect, min_extent):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < min_area or area > max_area:
            continue
        aspect = w / max(h, 1)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        extent = area / max(w * h, 1)
        if extent < min_extent:
            continue
        candidates.append((label_id, x, y, w, h, area))
    candidates.sort(key=lambda item: item[5], reverse=True)
    return labels, candidates


def clear_directory(path):
    path = Path(path)
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def square_crop_bounds(x, y, w, h, image_w, image_h, scale):
    side = int(round(max(w, h) * scale))
    side = max(side, max(w, h) + 8)
    cx = x + w / 2
    cy = y + h / 2
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    right = left + side
    bottom = top + side

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_w:
        left -= right - image_w
        right = image_w
    if bottom > image_h:
        top -= bottom - image_h
        bottom = image_h

    left = max(left, 0)
    top = max(top, 0)
    right = min(right, image_w)
    bottom = min(bottom, image_h)
    return left, top, right, bottom


def estimate_background(crop_bgr, crop_mask):
    outside = crop_bgr[crop_mask == 0]
    if outside.size == 0:
        return np.array([225, 225, 225], dtype=np.uint8)
    return np.median(outside.reshape(-1, 3), axis=0).astype(np.uint8)


def save_yolo_label(path, mask, class_id=0):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return
    h, w = mask.shape[:2]
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")


def remove_existing_image_outputs(image_path, output_images, output_masks, output_labels, debug_dir=None):
    stem = image_path.stem
    for path in output_images.glob(f"{stem}_head_*.png"):
        path.unlink()
    for path in output_masks.glob(f"{stem}_head_*.png"):
        path.unlink()
    for path in output_labels.glob(f"{stem}_head_*.txt"):
        path.unlink()
    if debug_dir is not None:
        for path in debug_dir.glob(f"{stem}_*"):
            path.unlink()


def extract_from_image(
    image_path,
    args,
    output_images,
    output_masks,
    output_labels,
    debug_dir=None,
    remaining=None,
):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return 0

    raw_mask, seed_mask, weak_mask, stain_score = make_stain_mask(image, args)
    labels, candidates = component_candidates(
        raw_mask,
        min_area=args.min_area,
        max_area=args.max_area,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        min_extent=args.min_extent,
    )
    if args.max_heads_per_image > 0:
        candidates = candidates[: args.max_heads_per_image]
    if remaining is not None:
        candidates = candidates[:remaining]

    image_h, image_w = image.shape[:2]
    kept = 0
    for label_id, x, y, w, h, area in candidates:
        instance_mask = (labels == label_id).astype(np.uint8) * 255
        if args.dilate > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (args.dilate * 2 + 1, args.dilate * 2 + 1),
            )
            instance_mask = cv2.dilate(instance_mask, kernel, iterations=1)

        left, top, right, bottom = square_crop_bounds(
            x, y, w, h, image_w, image_h, scale=args.crop_scale
        )
        crop = image[top:bottom, left:right]
        crop_mask = instance_mask[top:bottom, left:right]
        background = estimate_background(crop, crop_mask)

        cleaned = crop.copy()
        cleaned[crop_mask == 0] = background
        if args.blur_background > 0:
            blurred = cv2.GaussianBlur(cleaned, (0, 0), args.blur_background)
            cleaned[crop_mask == 0] = blurred[crop_mask == 0]

        cleaned = cv2.resize(
            cleaned,
            (args.patch_size, args.patch_size),
            interpolation=cv2.INTER_CUBIC,
        )
        resized_mask = cv2.resize(
            crop_mask,
            (args.patch_size, args.patch_size),
            interpolation=cv2.INTER_NEAREST,
        )
        _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

        out_name = f"{image_path.stem}_head_{kept:03d}.png"
        cv2.imwrite(str(output_images / out_name), cleaned)
        cv2.imwrite(str(output_masks / out_name), resized_mask)
        save_yolo_label(output_labels / f"{Path(out_name).stem}.txt", resized_mask)
        kept += 1

    if debug_dir is not None:
        overlay = image.copy()
        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
        seed_contours, _ = cv2.findContours(seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, seed_contours, -1, (0, 0, 255), 1)
        for _, x, y, w, h, _ in candidates:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_debug.jpg"), overlay)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_weak.png"), weak_mask)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_seed.png"), seed_mask)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_score.png"), stain_score)

    return kept


def parse_args():
    parser = argparse.ArgumentParser(description="Extract stained sperm head patches without manual labels.")
    parser.add_argument("--input", type=str, default=str(ROOT / "datasets" / "stain"))
    parser.add_argument("--output-images", type=str, default=str(ROOT / "datasets" / "stain_heads"))
    parser.add_argument("--output-masks", type=str, default=str(ROOT / "datasets" / "stain_head_masks"))
    parser.add_argument("--output-labels", type=str, default=str(ROOT / "datasets" / "stain_head_labels"))
    parser.add_argument("--debug-dir", type=str, default="")
    parser.add_argument("--clean-output", action="store_true", help="clear output folders before extraction")
    parser.add_argument("--clean-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-heads-per-image", type=int, default=0, help="0 means no per-image limit")
    parser.add_argument("--max-total-heads", type=int, default=0, help="0 means no total limit")
    parser.add_argument("--shuffle", action="store_true", help="shuffle input images before applying total limit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--crop-scale", type=float, default=3.0)
    parser.add_argument("--background-sigma", type=float, default=35.0)
    parser.add_argument("--denoise", type=float, default=3.0)
    parser.add_argument("--median", type=int, default=3)
    parser.add_argument("--gray-world", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hue-min", type=int, default=105)
    parser.add_argument("--hue-max", type=int, default=170)
    parser.add_argument("--weak-sat-min", type=int, default=12)
    parser.add_argument("--seed-sat-min", type=int, default=42)
    parser.add_argument("--weak-score-min", type=int, default=28)
    parser.add_argument("--seed-score-min", type=int, default=58)
    parser.add_argument("--val-min", type=int, default=30)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--max-area", type=int, default=12000)
    parser.add_argument("--min-aspect", type=float, default=0.30)
    parser.add_argument("--max-aspect", type=float, default=3.2)
    parser.add_argument("--min-extent", type=float, default=0.20)
    parser.add_argument("--shape-repair", choices=["none", "hull", "ellipse"], default="hull")
    parser.add_argument("--final-erode", type=int, default=0)
    parser.add_argument("--dilate", type=int, default=2)
    parser.add_argument("--blur-background", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = list_images(args.input)
    if not input_paths:
        raise FileNotFoundError(f"No images found: {args.input}")

    output_images = Path(args.output_images)
    output_masks = Path(args.output_masks)
    output_labels = Path(args.output_labels)
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_output:
        clear_directory(output_images)
        clear_directory(output_masks)
        clear_directory(output_labels)
        if debug_dir is not None:
            clear_directory(debug_dir)

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(input_paths)

    total = 0
    for image_path in tqdm(input_paths, desc="extract stain heads"):
        if args.max_total_heads > 0 and total >= args.max_total_heads:
            break
        if args.clean_existing:
            remove_existing_image_outputs(
                image_path,
                output_images=output_images,
                output_masks=output_masks,
                output_labels=output_labels,
                debug_dir=debug_dir,
            )
        remaining = None
        if args.max_total_heads > 0:
            remaining = args.max_total_heads - total
        total += extract_from_image(
            image_path,
            args,
            output_images=output_images,
            output_masks=output_masks,
            output_labels=output_labels,
            debug_dir=debug_dir,
            remaining=remaining,
        )

    print(f"Images scanned: {len(input_paths)}")
    print(f"Head patches saved: {total}")
    print(f"Patch images: {output_images}")
    print(f"Patch masks: {output_masks}")
    print(f"Patch labels: {output_labels}")
    if args.max_total_heads > 0:
        print(f"Total limit: {args.max_total_heads}")


if __name__ == "__main__":
    main()
