"""Regression coverage for rotated postprocess shape and angle contracts."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_boxes():
    path = ROOT / "yolox" / "utils" / "boxes.py"
    spec = importlib.util.spec_from_file_location("yolox.utils.boxes_postprocess_test", path)
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


class OBBPostprocessContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boxes = load_boxes()
        cls.boxes.py_cpu_nms_poly = lambda dets, threshold: list(range(len(dets)))
        cls.boxes.torchvision.ops.batched_nms = lambda boxes, scores, classes, threshold: torch.arange(
            boxes.shape[0]
        )

    def test_axis_aligned_postprocess_accepts_one_surviving_detection(self):
        prediction = torch.tensor([[[100.0, 100.0, 40.0, 10.0, 0.99, 0.99]]])
        output = self.boxes.postprocess(
            prediction, num_classes=1, conf_thre=0.01, nms_thre=0.65
        )
        self.assertIsNotNone(output[0])
        self.assertEqual(tuple(output[0].shape), (1, 7))

    def test_kld_postprocess_accepts_one_surviving_detection(self):
        prediction = torch.tensor(
            [[[100.0, 100.0, 40.0, 10.0, -45.0, 0.99, 0.99]]]
        )
        output = self.boxes.postprocessobb_kld(
            prediction, num_classes=1, conf_thre=0.01, nms_thre=0.65
        )
        self.assertIsNotNone(output[0])
        self.assertEqual(tuple(output[0].shape), (1, 10))

    def test_kld_postprocess_offsets_polygon_coordinates_per_class(self):
        observed = []

        def capture_nms(dets, threshold):
            observed.append(dets.copy())
            return list(range(len(dets)))

        self.boxes.py_cpu_nms_poly = capture_nms
        prediction = torch.tensor(
            [[
                [100.0, 100.0, 40.0, 10.0, 25.0, 0.99, 0.99, 0.001],
                [100.0, 100.0, 40.0, 10.0, 25.0, 0.99, 0.001, 0.99],
            ]]
        )
        output = self.boxes.postprocessobb_kld(
            prediction, num_classes=2, conf_thre=0.01, nms_thre=0.65
        )

        self.assertIsNotNone(output[0])
        self.assertEqual(tuple(output[0].shape), (2, 10))
        self.assertEqual(sorted(output[0][:, 9].tolist()), [0.0, 1.0])
        self.assertEqual(float(observed[0][0, 0] - observed[0][1, 0]), 4000.0)

if __name__ == "__main__":
    unittest.main()
