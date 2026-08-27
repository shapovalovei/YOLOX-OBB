"""Regression coverage for OBB labels through optional mixup."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_mosaic_obb():
    path = ROOT / "yolox" / "data" / "datasets" / "mosaicdetection_obb.py"
    spec = importlib.util.spec_from_file_location(
        "yolox.data.datasets.mosaicdetection_obb_mixup_test", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    original_modules = {
        name: sys.modules.get(name)
        for name in (
            "yolox",
            "yolox.utils",
            "yolox.data",
            "yolox.data.data_augment_obb",
            "yolox.data.datasets",
            "yolox.data.datasets.datasets_wrapper",
        )
    }

    yolox_stub = types.ModuleType("yolox")
    yolox_stub.__path__ = [str(ROOT / "yolox")]
    utils_stub = types.ModuleType("yolox.utils")

    def adjust_box_anns(boxes, scale, padw, padh, w_max, h_max):
        boxes = boxes.copy()
        boxes[:, 0::2] = boxes[:, 0::2] * scale + padw
        boxes[:, 1::2] = boxes[:, 1::2] * scale + padh
        return boxes

    utils_stub.adjust_box_anns = adjust_box_anns
    data_stub = types.ModuleType("yolox.data")
    data_stub.__path__ = [str(ROOT / "yolox" / "data")]
    augment_stub = types.ModuleType("yolox.data.data_augment_obb")
    augment_stub.box_candidates = lambda before, after, wh_thr: np.ones(
        before.shape[1], dtype=bool
    )
    augment_stub.random_perspective = lambda *args, **kwargs: args[:2]
    datasets_stub = types.ModuleType("yolox.data.datasets")
    datasets_stub.__path__ = [str(ROOT / "yolox" / "data" / "datasets")]
    wrapper_stub = types.ModuleType("yolox.data.datasets.datasets_wrapper")

    class Dataset:
        def __init__(self, input_dim, mosaic=True):
            self.input_dim = input_dim

        @staticmethod
        def resize_getitem(function):
            return function

    wrapper_stub.Dataset = Dataset
    sys.modules.update(
        {
            "yolox": yolox_stub,
            "yolox.utils": utils_stub,
            "yolox.data": data_stub,
            "yolox.data.data_augment_obb": augment_stub,
            "yolox.data.datasets": datasets_stub,
            "yolox.data.datasets.datasets_wrapper": wrapper_stub,
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


class OBBMixupContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_mosaic_obb()

    def test_mixup_preserves_angle_and_class_columns(self):
        labels = np.array([[5.0, 5.0, 35.0, 20.0, 17.0, 2.0]], dtype=np.float32)
        image = np.zeros((40, 40, 3), dtype=np.uint8)

        class SourceDataset:
            def load_anno(self, index):
                return labels.copy()

            def pull_item(self, index):
                return image.copy(), labels.copy(), None, None

            def __len__(self):
                return 1

        wrapper = object.__new__(self.module.MosaicDetectionOBB)
        wrapper._dataset = SourceDataset()
        wrapper.mixup_scale = (1.0, 1.0)

        with mock.patch.object(self.module.random, "uniform", side_effect=[1.0, 0.0]):
            with mock.patch.object(self.module.random, "randint", return_value=0):
                _, mixed_labels = wrapper.mixup(
                    image.copy(), np.empty((0, 6), dtype=np.float32), (40, 40)
                )

        self.assertEqual(tuple(mixed_labels.shape), (1, 6))
        np.testing.assert_allclose(mixed_labels[0], labels[0])

        with mock.patch.object(self.module.random, "uniform", side_effect=[1.0, 1.0]):
            with mock.patch.object(self.module.random, "randint", return_value=0):
                _, flipped_labels = wrapper.mixup(
                    image.copy(), np.empty((0, 6), dtype=np.float32), (40, 40)
                )

        self.assertEqual(tuple(flipped_labels.shape), (1, 6))
        self.assertAlmostEqual(float(flipped_labels[0, 4]), -17.0)
        self.assertAlmostEqual(float(flipped_labels[0, 5]), 2.0)


if __name__ == "__main__":
    unittest.main()
