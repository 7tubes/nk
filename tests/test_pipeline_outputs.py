import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sperm_morphology.crop import crop_roi


class PipelineOutputTests(unittest.TestCase):
    def test_crop_roi_propagates_image_context(self):
        image = np.zeros((64, 64), dtype=np.uint8)
        config = {
            "crop": {
                "margin_px": 8,
                "min_box_width": 4,
                "min_box_height": 4,
            }
        }

        info = crop_roi(
            image,
            [10, 12, 20, 24],
            config,
            image_id="0001",
            target_id=3,
        )

        self.assertEqual(info["image_id"], "0001")
        self.assertEqual(info["target_id"], 3)


if __name__ == "__main__":
    unittest.main()
