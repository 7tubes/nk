import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sperm_morphology.detection_screening import (
    grade_to_traffic_color,
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

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "overlay.png"
            saved_path = save_screening_overlay(image, results, output_path)
            self.assertTrue(Path(saved_path).exists())


if __name__ == "__main__":
    unittest.main()
