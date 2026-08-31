"""Regression tests for invalid OBB target rejection before assignment."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_augmentation():
    path = ROOT / "yolox" / "data" / "data_augment_obb.py"
    spec = importlib.util.spec_from_file_location(
        "maintained_data_augment_obb_target_validity_test", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)

    original_utils = sys.modules.get("yolox.utils")
    utility_stub = types.ModuleType("yolox.utils")
    utility_stub.xyxy2cxcywh = lambda boxes: np.column_stack(
        (
            (boxes[:, 0] + boxes[:, 2]) * 0.5,
            (boxes[:, 1] + boxes[:, 3]) * 0.5,
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        )
    )
    sys.modules["yolox.utils"] = utility_stub
    try:
        spec.loader.exec_module(module)
    finally:
        if original_utils is None:
            sys.modules.pop("yolox.utils", None)
        else:
            sys.modules["yolox.utils"] = original_utils
    return module


def load_head():
    path = ROOT / "yolox" / "models" / "yolo_head_obb_kld.py"
    spec = importlib.util.spec_from_file_location(
        "yolox.models.yolo_head_obb_kld_target_validity_test", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)

    original_modules = {
        name: sys.modules.get(name)
        for name in ("yolox", "yolox.models", "yolox.models.network_blocks")
    }
    yolox_stub = types.ModuleType("yolox")
    yolox_stub.__path__ = [str(ROOT / "yolox")]
    models_stub = types.ModuleType("yolox.models")
    models_stub.__path__ = [str(ROOT / "yolox" / "models")]
    models_stub.compute_kld_loss = lambda *args, **kwargs: None
    models_stub.KLDloss = object
    blocks_stub = types.ModuleType("yolox.models.network_blocks")
    blocks_stub.BaseConv = object
    blocks_stub.DWConv = object
    sys.modules.update(
        {
            "yolox": yolox_stub,
            "yolox.models": models_stub,
            "yolox.models.network_blocks": blocks_stub,
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
    return module.YOLOXHeadOBB_KLD


def run_transform(augment, targets, input_dim=(100, 100), flip_h=None):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    identity_flip_h = lambda image, boxes, angles: (image, boxes, angles)
    with mock.patch.object(augment, "_distort", side_effect=lambda value: value), \
            mock.patch.object(augment, "_flip_h", side_effect=flip_h or identity_flip_h), \
            mock.patch.object(augment, "_flip_v", side_effect=identity_flip_h):
        return augment.TrainTransformOBB(max_labels=8)(
            image, np.asarray(targets, dtype=np.float64), input_dim
        )


class TrainTransformOBBTargetValidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.augment = load_augmentation()

    def test_positive_tiny_targets_are_not_resurrected_by_fallback(self):
        for short_side in (1.0, 0.1):
            with self.subTest(short_side=short_side):
                image, labels = run_transform(
                    self.augment,
                    [[0.0, 50.0 - short_side / 2, 100.0, 50.0 + short_side / 2,
                      0.0, 0.0]],
                )
                self.assertEqual(image.shape, (3, 100, 100))
                self.assertEqual(labels.shape, (8, 6))
                self.assertEqual(np.count_nonzero(labels), 0)

    def test_post_resize_minimum_size_remains_strictly_greater_than_four(self):
        for short_side, expected_valid in ((4.0, False), (4.01, True)):
            with self.subTest(short_side=short_side):
                _, labels = run_transform(
                    self.augment,
                    [[40.0, 50.0 - short_side / 2, 60.0,
                      50.0 + short_side / 2, 0.0, 0.0]],
                )
                self.assertEqual(np.count_nonzero(labels) > 0, expected_valid)

    def test_zero_and_negative_dimensions_are_not_resurrected(self):
        cases = {
            "zero_width": [50.0, 40.0, 50.0, 60.0],
            "negative_width": [60.0, 40.0, 50.0, 60.0],
            "zero_height": [40.0, 50.0, 60.0, 50.0],
            "negative_height": [40.0, 60.0, 60.0, 50.0],
        }
        for name, box in cases.items():
            with self.subTest(name=name):
                _, labels = run_transform(
                    self.augment, [box + [0.0, 0.0]]
                )
                self.assertEqual(np.count_nonzero(labels), 0)

    def test_nonfinite_geometry_is_not_resurrected(self):
        cases = {
            "nan_angle": [40.0, 40.0, 60.0, 60.0, np.nan, 0.0],
            "inf_width": [40.0, 40.0, np.inf, 60.0, 0.0, 0.0],
            "negative_inf_height": [40.0, 40.0, 60.0, -np.inf, 0.0, 0.0],
        }
        for name, target in cases.items():
            with self.subTest(name=name):
                _, labels = run_transform(self.augment, [target])
                self.assertEqual(np.count_nonzero(labels), 0)

    def test_mixed_valid_and_invalid_targets_preserve_valid_class_zero(self):
        _, labels = run_transform(
            self.augment,
            [
                [40.0, 40.0, 60.0, 60.0, 0.0, 0.0],
                [50.0, 40.0, 50.0, 60.0, 25.0, 3.0],
            ],
        )
        np.testing.assert_allclose(labels[0], [0.0, 50.0, 50.0, 20.0, 20.0, 0.0])
        self.assertEqual(np.count_nonzero(labels[1:]), 0)

    def test_valid_original_fallback_is_preserved_after_transform_filtering(self):
        def shrink_transformed_box(image, boxes, angles):
            tiny = boxes.copy()
            tiny[0] = [49.5, 49.5, 50.5, 50.5]
            return image, tiny, angles

        _, labels = run_transform(
            self.augment,
            [[40.0, 40.0, 60.0, 60.0, 12.0, 0.0]],
            input_dim=(200, 200),
            flip_h=shrink_transformed_box,
        )
        np.testing.assert_allclose(labels[0], [0.0, 100.0, 100.0, 40.0, 40.0, 12.0])
        self.assertEqual(np.count_nonzero(labels[1:]), 0)

    def test_all_invalid_fallback_returns_standard_empty_targets(self):
        image, labels = run_transform(
            self.augment,
            [[50.0, 40.0, 50.0, 60.0, 0.0, 2.0]],
            input_dim=(200, 200),
        )
        self.assertEqual(image.shape, (3, 200, 200))
        self.assertEqual(image.dtype, np.float32)
        self.assertEqual(labels.shape, (8, 6))
        self.assertEqual(labels.dtype, np.float32)
        self.assertEqual(np.count_nonzero(labels), 0)

    def test_random_perspective_does_not_repair_negative_source_geometry(
        self
    ):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        target = np.asarray([[60.0, 40.0, 50.0, 60.0, 0.0, 0.0]])
        with mock.patch.object(
            self.augment.random,
            "uniform",
            side_effect=[0.0, 1.0, 0.0, 0.0, 0.5, 0.5],
        ):
            _, output = self.augment.random_perspective(
                image,
                target,
                degrees=0.0,
                translate=0.0,
                scale=(1.0, 1.0),
                shear=0.0,
                perspective=0.0,
            )
        self.assertEqual(output.shape, (0, 6))


class HeadTargetValidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.head_class = load_head()

    def make_head(self, calls):
        head = self.head_class.__new__(self.head_class)
        torch.nn.Module.__init__(head)
        head.use_l1 = False
        head.num_classes = 1
        head.iou_loss = lambda pred, target: pred.sum(dim=1) * 0.0
        head.bcewithlog_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        def fake_assign(*args):
            calls.append(
                {
                    "num_gt": args[1],
                    "boxes": args[3].detach().clone(),
                    "classes": args[4].detach().clone(),
                    "angles": args[5].detach().clone(),
                }
            )
            total_num_anchors = args[2]
            return (
                torch.empty(0, dtype=torch.long),
                torch.zeros(total_num_anchors, dtype=torch.bool),
                torch.empty(0),
                torch.empty(0, dtype=torch.long),
                0,
            )

        head.get_assignments = fake_assign
        return head

    def run_head(self, labels, calls):
        head = self.make_head(calls)
        outputs = torch.zeros(1, 3, 7)
        return head.get_losses(
            None,
            [torch.zeros(1, 3)],
            [torch.zeros(1, 3)],
            [torch.full((1, 3), 8.0)],
            torch.tensor([labels], dtype=torch.float32),
            outputs,
            [],
            torch.float32,
        )

    def test_malformed_rows_are_not_passed_to_assignment(self):
        malformed_rows = [
            [7.0, 100.0, 80.0, 0.0, 10.0, 15.0],
            [7.0, 100.0, 80.0, -1.0, 10.0, 15.0],
            [7.0, 100.0, 80.0, 10.0, 0.0, 15.0],
            [7.0, 100.0, 80.0, 10.0, -1.0, 15.0],
        ]
        for row in malformed_rows:
            with self.subTest(row=row):
                calls = []
                losses = self.run_head([row, [0.0] * 6], calls)
                self.assertEqual(calls, [])
                self.assertTrue(torch.isfinite(losses[0]))

    def test_valid_class_zero_is_counted_and_malformed_middle_row_is_skipped(self):
        calls = []
        losses = self.run_head(
            [
                [0.0, 100.0, 80.0, 40.0, 10.0, 0.0],
                [3.0, 100.0, 80.0, -1.0, 10.0, 15.0],
                [0.0] * 6,
            ],
            calls,
        )
        self.assertTrue(torch.isfinite(losses[0]))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["num_gt"], 1)
        torch.testing.assert_close(
            calls[0]["boxes"], torch.tensor([[100.0, 80.0, 40.0, 10.0]])
        )
        torch.testing.assert_close(calls[0]["classes"], torch.tensor([0.0]))
        torch.testing.assert_close(calls[0]["angles"], torch.tensor([0.0]))

    def test_nonfinite_padded_geometry_is_not_counted(self):
        calls = []
        losses = self.run_head(
            [[0.0, np.nan, 80.0, 40.0, 10.0, 0.0], [0.0] * 6], calls
        )
        self.assertTrue(torch.isfinite(losses[0]))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
