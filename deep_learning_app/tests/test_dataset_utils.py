import tempfile
import unittest
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_utils import convert_single_label


class DatasetUtilsTest(unittest.TestCase):
    def test_convert_single_label_uses_class_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            src_label = tmpdir_path / "src.txt"
            dst_label = tmpdir_path / "dst.txt"
            image_path = tmpdir_path / "image.jpg"
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            encoded.tofile(str(image_path))
            src_label.write_text("7 10 20 30 40\n12 1 2 3 4\n", encoding="utf-8")

            convert_single_label(src_label, dst_label, image_path)

            lines = dst_label.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(all(line.startswith("0 ") for line in lines))
            for line in lines:
                values = [float(v) for v in line.split()[1:]]
                self.assertTrue(all(0.0 <= value <= 1.0 for value in values))

    def test_convert_single_label_accepts_yolo_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            src_label = tmpdir_path / "src.txt"
            dst_label = tmpdir_path / "dst.txt"
            image_path = tmpdir_path / "image.jpg"
            image = np.zeros((100, 200, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            encoded.tofile(str(image_path))
            src_label.write_text("3 0.500000 0.250000 0.200000 0.100000\n", encoding="utf-8")

            convert_single_label(src_label, dst_label, image_path)

            line = dst_label.read_text(encoding="utf-8").strip()
            self.assertEqual(line, "0 0.50000000 0.25000000 0.20000000 0.10000000")


if __name__ == "__main__":
    unittest.main()
