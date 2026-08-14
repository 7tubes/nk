import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from head_segmentation_app.src.labelme_to_yolo_seg import convert_labelme_dataset
from head_segmentation_app.src.labelme_to_yolo_seg import write_image_unicode


class HeadSegmentationDatasetTests(unittest.TestCase):
    def _write_labelme_pair(self, folder: Path, stem: str, x_offset: int) -> None:
        image = np.full((64, 64, 3), 220, dtype=np.uint8)
        cv2.ellipse(image, (24 + x_offset, 32), (8, 5), 0, 0, 360, (40, 40, 40), -1)
        image_path = folder / f"{stem}.jpg"
        write_image_unicode(image_path, image)

        data = {
            "version": "5.5.0",
            "imagePath": image_path.name,
            "imageHeight": 64,
            "imageWidth": 64,
            "shapes": [
                {
                    "label": "head",
                    "shape_type": "polygon",
                    "points": [
                        [16 + x_offset, 28],
                        [24 + x_offset, 24],
                        [32 + x_offset, 28],
                        [32 + x_offset, 36],
                        [24 + x_offset, 40],
                        [16 + x_offset, 36],
                    ],
                },
                {
                    "label": "tail",
                    "shape_type": "polygon",
                    "points": [[32 + x_offset, 32], [50, 33], [50, 35], [32 + x_offset, 34]],
                },
            ],
        }
        (folder / f"{stem}.json").write_text(json.dumps(data), encoding="utf-8")

    def test_labelme_head_polygons_convert_to_yolo_seg_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            dataset_dir = Path(tmpdir) / "dataset"
            source_dir.mkdir()
            self._write_labelme_pair(source_dir, "sample_a", 0)
            self._write_labelme_pair(source_dir, "sample_b", 4)

            dataset_yaml, stats = convert_labelme_dataset(
                source_dir=source_dir,
                dataset_dir=dataset_dir,
                val_ratio=0.5,
                seed=1,
                tile=False,
            )

            self.assertTrue(dataset_yaml.exists())
            self.assertEqual(stats["usable_images"], 2)
            labels = list((dataset_dir / "labels").rglob("*.txt"))
            self.assertEqual(len(labels), 2)

            for label_path in labels:
                line = label_path.read_text(encoding="utf-8").strip()
                values = [float(value) for value in line.split()]
                self.assertEqual(values[0], 0.0)
                self.assertGreaterEqual(len(values), 7)
                self.assertTrue(all(0.0 <= value <= 1.0 for value in values[1:]))


if __name__ == "__main__":
    unittest.main()
