"""Regression coverage for angle-bin OBB polygon reconstruction."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_boxes():
    path = ROOT / "yolox" / "utils" / "boxes.py"
    spec = importlib.util.spec_from_file_location("yolox.utils.boxes_angle_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    original_modules = {
        name: sys.modules.get(name)
        for name in ("yolox", "yolox.utils", "DOTA_devkit_YOLO", "torchvision")
    }
    yolox_stub = types.ModuleType("yolox")
    yolox_stub.__path__ = [str(ROOT / "yolox")]
    utils_stub = types.ModuleType("yolox.utils")
    utils_stub.__path__ = [str(ROOT / "yolox" / "utils")]
    dota_stub = types.ModuleType("DOTA_devkit_YOLO")
    dota_stub.polyiou = types.SimpleNamespace()
    torchvision_stub = types.ModuleType("torchvision")
    torchvision_stub.ops = types.SimpleNamespace()
    sys.modules.update(
        {
            "yolox": yolox_stub,
            "yolox.utils": utils_stub,
            "DOTA_devkit_YOLO": dota_stub,
            "torchvision": torchvision_stub,
        }
    )
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


def polygon_iou(left, right):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    left_area = abs(float(cv2.contourArea(left)))
    right_area = abs(float(cv2.contourArea(right)))
    intersection, _ = cv2.intersectConvexConvex(left, right)
    return float(intersection) / (left_area + right_area - float(intersection) + 1e-12)


class AngleBinPostprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boxes = load_boxes()
        cls.boxes.py_cpu_nms_poly = lambda dets, threshold: list(range(len(dets)))

    def test_negative_angle_reconstruction_preserves_polygon(self):
        prediction = torch.zeros((1, 1, 186), dtype=torch.float32)
        prediction[0, 0, :4] = torch.tensor([100.0, 100.0, 40.0, 10.0])
        prediction[0, 0, 5 + 45] = 1.0  # angle index 45 -> -45 degrees
        prediction[0, 0, 185] = 0.99
        prediction[0, 0, 4] = 0.99

        output = self.boxes.postprocessobb(
            prediction, num_classes=1, conf_thre=0.01, nms_thre=0.65
        )
        self.assertIsNotNone(output[0])
        actual = output[0][0, :8].numpy().reshape(4, 2)
        expected = np.intp(cv2.boxPoints(((100.0, 100.0), (40.0, 10.0), -45.0)))
        self.assertGreater(polygon_iou(actual, expected), 0.90)


if __name__ == "__main__":
    unittest.main()
