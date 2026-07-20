import tempfile
import os
import sys
from pathlib import Path

import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sperm_morphology.dataset import load_config
from sperm_morphology.detection_screening import draw_screening_results, screen_detections
from sperm_morphology.utils import read_image_unicode

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))

from ultralytics import YOLO

MODEL_PATH = ROOT / "runs" / "sperm_detection" / "weights" / "best.pt"
MORPHOLOGY_CONFIG_PATH = PROJECT_ROOT / "configs" / "morphology.yaml"


def tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def detections_from_result(result, offset_x=0, offset_y=0):
    detections = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        detections.append(
            {
                "xyxy": [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
                "conf": conf,
            }
        )
    return detections


def nms_detections(detections, iou_threshold=0.45, max_det=500):
    if not detections:
        return []

    boxes_xywh = []
    scores = []
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        boxes_xywh.append([int(x1), int(y1), int(max(x2 - x1, 1)), int(max(y2 - y1, 1))])
        scores.append(float(det["conf"]))

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=0.0,
        nms_threshold=float(iou_threshold),
        top_k=int(max_det),
    )
    if len(indices) == 0:
        return []

    flat = [int(i) for i in indices.flatten()]
    picked = [detections[i] for i in flat]
    picked.sort(key=lambda item: item["conf"], reverse=True)
    return picked[: int(max_det)]


def predict_full(model, image, conf, iou, imgsz, max_det):
    result = model(
        image,
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        max_det=int(max_det),
        stream=False,
        verbose=False,
    )[0]
    return detections_from_result(result)


def predict_tiled(model, image, conf, iou, imgsz, max_det, tile_size, overlap):
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
            detections.extend(detections_from_result(result, x0, y0))

    return nms_detections(detections, iou_threshold=iou, max_det=max_det)


def draw_detections(image, detections):
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        conf = float(det["conf"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            canvas,
            f"{conf:.2f}",
            (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def run_detection(model, image, use_tiling, conf, iou, imgsz, max_det, tile_size, overlap):
    if use_tiling:
        return predict_tiled(model, image, conf, iou, imgsz, max_det, tile_size, overlap)
    return predict_full(model, image, conf, iou, imgsz, max_det)


if not MODEL_PATH.exists():
    st.error("No trained model found. Run train_yolo.py first.")
    st.stop()

model = YOLO(str(MODEL_PATH))

st.set_page_config(page_title="Sperm Detection", layout="wide")
st.title("Sperm Detection")

with st.sidebar:
    st.header("Detection Settings")
    conf_threshold = st.slider("Confidence", 0.01, 0.90, 0.10, 0.01)
    iou_threshold = st.slider("NMS IoU", 0.10, 0.90, 0.45, 0.01)
    imgsz = st.select_slider("Image size", options=[640, 768, 896, 960, 1024], value=768)
    max_det = st.number_input("Max detections", min_value=1, max_value=2000, value=500, step=50)
    use_tiling = st.checkbox("Tiled inference", value=True)
    tile_size = st.select_slider("Tile size", options=[384, 512, 640], value=512)
    overlap = st.slider("Tile overlap", 0.00, 0.60, 0.25, 0.05)
    run_morphology_screening = st.checkbox("Morphology screening colors", value=True)

uploaded_file = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / uploaded_file.name
        tmp_path.write_bytes(uploaded_file.read())

        if suffix in {".jpg", ".jpeg", ".png"}:
            image = read_image_unicode(tmp_path)
            if image is None:
                st.error("Failed to read image.")
                st.stop()

            detections = run_detection(
                model,
                image,
                use_tiling,
                conf_threshold,
                iou_threshold,
                imgsz,
                max_det,
                tile_size,
                overlap,
            )
            if run_morphology_screening:
                morphology_config = load_config(str(MORPHOLOGY_CONFIG_PATH))
                screening_results = screen_detections(
                    image,
                    detections,
                    morphology_config,
                    image_id=Path(uploaded_file.name).stem,
                )
                annotated = draw_screening_results(image, screening_results)
            else:
                screening_results = []
                annotated = draw_detections(image, detections)
            st.image(annotated, channels="BGR", use_container_width=True)
            st.write(f"Detected {len(detections)} targets")

            if run_morphology_screening:
                color_counts = {"green": 0, "yellow": 0, "red": 0}
                for result in screening_results:
                    color_counts[result["traffic_color"]] += 1
                st.write(
                    f"Green {color_counts['green']} | Yellow {color_counts['yellow']} | Red {color_counts['red']}"
                )

                for result in screening_results:
                    x1, y1, x2, y2 = result["bbox"]
                    scores = result["scores"]
                    st.write(
                        f"box ({x1:.1f}, {y1:.1f})-({x2:.1f}, {y2:.1f}), "
                        f"conf {result['confidence']:.2f}, "
                        f"{result['traffic_color']}, {scores['grade']} {scores['total_score']:.1f}"
                    )
            else:
                for det in detections:
                    x1, y1, x2, y2 = det["xyxy"]
                    st.write(f"box ({x1:.1f}, {y1:.1f})-({x2:.1f}, {y2:.1f}), conf {det['conf']:.2f}")

        elif suffix in {".mp4", ".avi", ".mov"}:
            st.video(tmp_path.open("rb"))
            st.info("Video detection is available frame by frame. Tiled inference can be slow on CPU.")

            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                st.error("Failed to open video.")
                st.stop()

            frame_index = 0
            preview_every = 30
            previews = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % preview_every == 0:
                    detections = run_detection(
                        model,
                        frame,
                        use_tiling,
                        conf_threshold,
                        iou_threshold,
                        imgsz,
                        max_det,
                        tile_size,
                        overlap,
                    )
                    previews.append(draw_detections(frame, detections))
                frame_index += 1
            cap.release()

            st.write(f"Read {frame_index} frames; showing every {preview_every}th frame.")
            for preview in previews[:10]:
                st.image(preview, channels="BGR", use_container_width=True)
