import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sperm_morphology.detection_screening import (
    grade_to_traffic_color,
    draw_screening_results,
    save_screening_overlay,
    screen_detections,
)


class DetectionScreeningTests(unittest.TestCase):
    def test_grade_to_traffic_color_maps_screening_levels(self):
        self.assertEqual(grade_to_traffic_color({"grade": "A", "total_score": 90}), "green")
        self.assertEqual(grade_to_traffic_color({"grade": "B", "total_score": 72}), "green")
        self.assertEqual(grade_to_traffic_color({"grade": "C", "total_score": 60}), "yellow")
        self.assertEqual(grade_to_traffic_color({"grade": "D", "total_score": 40}), "red")
        self.assertEqual(grade_to_traffic_color({"grade": "Reject", "total_score": 0}), "red")

    def test_manual_detection_box_can_be_screened_and_drawn(self):
        image = np.full((96, 96, 3), 220, dtype=np.uint8)
        cv2.ellipse(image, (48, 48), (12, 7), 20, 0, 360, (40, 40, 40), -1)

        config = {
            "crop": {
                "margin_px": 16,
                "min_box_width": 4,
                "min_box_height": 4,
            },
            "preprocess": {
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": 8,
                "gaussian_kernel": 3,
                "background_blur_kernel": 31,
            },
            "segmentation": {
                "min_head_area_px": 8,
                "max_head_area_px": 800,
                "min_contour_points": 5,
            },
            "scoring": {
                "weights": {
                    "fit_goodness": 0.40,
                    "axis_ratio": 0.30,
                    "uniformity": 0.30,
                },
                "axis_ratio_target": 1.62,
                "axis_ratio_tolerance": 0.60,
                "fit_iou_good": 0.85,
                "fit_iou_bad": 0.60,
                "uniformity_good": 0.85,
                "uniformity_bad": 0.60,
                "grade_thresholds": {
                    "A": 85,
                    "B": 70,
                    "C": 55,
                },
            },
        }

        results = screen_detections(
            image,
            [{"xyxy": [35, 36, 61, 60], "conf": 0.88}],
            config,
            image_id="synthetic",
        )

        self.assertEqual(len(results), 1)
        self.assertIn(results[0]["traffic_color"], {"green", "yellow", "red"})
        self.assertIn("grade", results[0]["scores"])
        self.assertIsNotNone(results[0].get("head_bbox"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "overlay.png"
            saved_path = save_screening_overlay(image, results, output_path)
            self.assertTrue(Path(saved_path).exists())

    def test_draw_screening_results_renders_yellow_head_outline(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        cv2.rectangle(mask, (4, 5), (11, 11), 255, -1)

        result = {
            "bbox": [20, 18, 44, 40],
            "head_bbox": [24, 23, 32, 29],
            "confidence": 0.91,
            "traffic_color": "green",
            "scores": {"grade": "A", "total_score": 92.5},
            "mask": mask,
            "roi_info": {"roi_bbox_global": [20, 18, 36, 34]},
        }

        annotated = draw_screening_results(image, [result])
        self.assertTrue(np.array_equal(annotated[23, 24], np.array([0, 255, 255], dtype=np.uint8)))

    def test_draw_screening_results_renders_fitted_ellipse_axes(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        cv2.ellipse(mask, (8, 8), (5, 3), 0, 0, 360, 255, -1)

        result = {
            "bbox": [20, 18, 44, 40],
            "head_bbox": [23, 23, 33, 29],
            "confidence": 0.91,
            "traffic_color": "green",
            "scores": {"grade": "A", "total_score": 92.5},
            "mask": mask,
            "roi_info": {"roi_bbox_global": [20, 18, 36, 34]},
            "features": {
                "ellipse": {
                    "center": [8.0, 8.0],
                    "major_axis": 10.0,
                    "minor_axis": 6.0,
                    "angle": 0.0,
                },
                "L_px": 10.0,
                "W_px": 6.0,
            },
        }

        annotated = draw_screening_results(image, [result])
        self.assertTrue(np.array_equal(annotated[26, 25], np.array([255, 255, 0], dtype=np.uint8)))
        self.assertTrue(np.array_equal(annotated[23, 28], np.array([255, 0, 255], dtype=np.uint8)))

        hidden = draw_screening_results(image, [result], show_ellipse_axes=False)
        self.assertFalse(np.array_equal(hidden[26, 25], np.array([255, 255, 0], dtype=np.uint8)))
        self.assertFalse(np.array_equal(hidden[23, 28], np.array([255, 0, 255], dtype=np.uint8)))

    def test_deep_segmentation_uses_full_image_head_candidates(self):
        image = np.full((96, 96, 3), 220, dtype=np.uint8)
        cv2.ellipse(image, (48, 48), (12, 7), 20, 0, 360, (40, 40, 40), -1)
        full_mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.ellipse(full_mask, (48, 48), (12, 7), 20, 0, 360, 255, -1)

        config = {
            "crop": {
                "margin_px": 16,
                "min_box_width": 4,
                "min_box_height": 4,
            },
            "preprocess": {
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": 8,
                "gaussian_kernel": 3,
                "background_blur_kernel": 31,
            },
            "segmentation": {
                "min_head_area_px": 8,
                "max_head_area_px": 800,
                "min_contour_points": 5,
            },
            "deep_segmentation": {
                "enabled": True,
                "model_path": "fake_full_image_model.pt",
                "fallback_to_traditional": False,
            },
            "scoring": {
                "weights": {
                    "fit_goodness": 0.40,
                    "axis_ratio": 0.30,
                    "uniformity": 0.30,
                },
                "axis_ratio_target": 1.62,
                "axis_ratio_tolerance": 0.60,
                "fit_iou_good": 0.85,
                "fit_iou_bad": 0.60,
                "uniformity_good": 0.85,
                "uniformity_bad": 0.60,
                "grade_thresholds": {
                    "A": 85,
                    "B": 70,
                    "C": 55,
                },
            },
        }

        fake_heads = [
            {
                "head_id": 0,
                "mask": full_mask,
                "confidence": 0.93,
                "bbox": [36, 41, 61, 56],
            }
        ]
        fake_info = {
            "success": True,
            "method": "yolov8_seg_full_image",
            "model_path": "fake_full_image_model.pt",
            "head_count": 1,
        }

        with patch(
            "sperm_morphology.deep_head_segmenter.segment_heads_in_image",
            return_value=(fake_heads, fake_info),
        ):
            results = screen_detections(
                image,
                [{"xyxy": [35, 36, 61, 60], "conf": 0.88}],
                config,
                image_id="synthetic",
            )

        self.assertEqual(results[0]["quality_info"].get("method"), "yolov8_seg_full_image")
        self.assertEqual(results[0]["quality_info"].get("model_path"), "fake_full_image_model.pt")
        self.assertEqual(results[0]["quality_info"].get("head_id"), 0)
        self.assertIsNotNone(results[0].get("head_bbox"))

    def test_missing_deep_segmentation_model_falls_back_to_traditional(self):
        image = np.full((96, 96, 3), 220, dtype=np.uint8)
        cv2.ellipse(image, (48, 48), (12, 7), 20, 0, 360, (40, 40, 40), -1)

        config = {
            "crop": {
                "margin_px": 16,
                "min_box_width": 4,
                "min_box_height": 4,
            },
            "preprocess": {
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": 8,
                "gaussian_kernel": 3,
                "background_blur_kernel": 31,
            },
            "segmentation": {
                "min_head_area_px": 8,
                "max_head_area_px": 800,
                "min_contour_points": 5,
            },
            "deep_segmentation": {
                "enabled": True,
                "model_path": "missing_head_segmentation_weights.pt",
                "fallback_to_traditional": True,
            },
            "scoring": {
                "weights": {
                    "fit_goodness": 0.40,
                    "axis_ratio": 0.30,
                    "uniformity": 0.30,
                },
                "axis_ratio_target": 1.62,
                "axis_ratio_tolerance": 0.60,
                "fit_iou_good": 0.85,
                "fit_iou_bad": 0.60,
                "uniformity_good": 0.85,
                "uniformity_bad": 0.60,
                "grade_thresholds": {
                    "A": 85,
                    "B": 70,
                    "C": 55,
                },
            },
        }

        results = screen_detections(
            image,
            [{"xyxy": [35, 36, 61, 60], "conf": 0.88}],
            config,
            image_id="synthetic",
        )

        self.assertEqual(len(results), 1)
        self.assertNotEqual(results[0]["quality_info"].get("method"), "yolov8_seg")
        self.assertIn("grade", results[0]["scores"])


if __name__ == "__main__":
    unittest.main()
