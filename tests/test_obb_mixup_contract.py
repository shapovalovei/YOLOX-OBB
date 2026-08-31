"""Regression coverage for OBB labels through optional mixup."""

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_augmentation():
    path = ROOT / "yolox" / "data" / "data_augment_obb.py"
    spec = importlib.util.spec_from_file_location(
        "maintained_data_augment_obb_mixup_test", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    original_utils = sys.modules.get("yolox.utils")
    utility_stub = types.ModuleType("yolox.utils")
    utility_stub.xyxy2cxcywh = lambda boxes: boxes
    sys.modules["yolox.utils"] = utility_stub
    try:
        spec.loader.exec_module(module)
    finally:
        if original_utils is None:
            sys.modules.pop("yolox.utils", None)
        else:
            sys.modules["yolox.utils"] = original_utils
    return module


def load_mosaic_obb(augment_module):
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
    augment_stub.box_candidates = augment_module.box_candidates
    augment_stub.random_perspective = lambda *args, **kwargs: args[:2]
    for name in (
        "_clip_polygon_to_rect",
        "_obb_target_to_corners",
        "_obb_to_target",
        "_polygon_area",
        "corners_to_obb",
    ):
        setattr(augment_stub, name, getattr(augment_module, name))
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


def encode_obb(center_x, center_y, width, height, angle, class_value):
    return np.asarray(
        [
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
            angle,
            class_value,
        ],
        dtype=np.float64,
    )


def corners_from_target(target):
    center_x = (target[0] + target[2]) * 0.5
    center_y = (target[1] + target[3]) * 0.5
    width = target[2] - target[0]
    height = target[3] - target[1]
    radians = math.radians(float(target[4]))
    ux, uy = math.cos(radians), math.sin(radians)
    vx, vy = -uy, ux
    return np.asarray(
        [
            (center_x - width * ux / 2.0 - height * vx / 2.0,
             center_y - width * uy / 2.0 - height * vy / 2.0),
            (center_x + width * ux / 2.0 - height * vx / 2.0,
             center_y + width * uy / 2.0 - height * vy / 2.0),
            (center_x + width * ux / 2.0 + height * vx / 2.0,
             center_y + width * uy / 2.0 + height * vy / 2.0),
            (center_x - width * ux / 2.0 + height * vx / 2.0,
             center_y - width * uy / 2.0 + height * vy / 2.0),
        ],
        dtype=np.float64,
    )


def intersect_with_canvas(points, width, height):
    canvas = np.asarray(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float32,
    )
    area, intersection = cv2.intersectConvexConvex(
        np.asarray(points, dtype=np.float32), canvas
    )
    if intersection is None or area <= 1e-6:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(intersection, dtype=np.float64).reshape(-1, 2)


def canonical_min_area_obb(points):
    """Independent edge-orientation minimum-area fit for the test oracle."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("polygon is not finite")

    candidates = []
    for index in range(len(points)):
        edge = points[(index + 1) % len(points)] - points[index]
        edge_length = np.linalg.norm(edge)
        if edge_length <= 1e-12:
            continue
        unit = edge / edge_length
        normal = np.asarray([-unit[1], unit[0]])
        along = points @ unit
        across = points @ normal
        min_along, max_along = along.min(), along.max()
        min_across, max_across = across.min(), across.max()
        width = max_along - min_along
        height = max_across - min_across
        if min(width, height) <= 1e-6:
            continue
        center = (
            unit * ((min_along + max_along) * 0.5)
            + normal * ((min_across + max_across) * 0.5)
        )
        if width >= height:
            long_edge, short_edge, direction = width, height, unit
        else:
            long_edge, short_edge, direction = height, width, normal
        angle = math.degrees(math.atan2(direction[1], direction[0]))
        while angle >= 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        candidates.append((width * height, center[0], center[1], long_edge, short_edge, angle))

    if not candidates:
        raise ValueError("polygon is degenerate")
    _, center_x, center_y, long_edge, short_edge, angle = min(candidates)
    return np.asarray(
        [center_x, center_y, long_edge, short_edge, angle], dtype=np.float64
    )


def transformed_visible_target(source, jit_factor, flip, x_offset, y_offset):
    points = corners_from_target(source) * jit_factor
    origin_width = int(100 * jit_factor)
    if flip:
        points[:, 0] = origin_width - points[:, 0]
    points -= np.asarray([x_offset, y_offset], dtype=np.float64)
    visible = intersect_with_canvas(points, 100, 100)
    if len(visible) == 0:
        return None, visible
    obb = canonical_min_area_obb(visible)
    return encode_obb(*obb, source[5]), visible


def legacy_mixup_target(source, jit_factor, flip, x_offset, y_offset):
    """Reproduce only the old envelope-clipping result for before/after evidence."""
    origin_width = int(100 * jit_factor)
    box = source[:4].copy() * jit_factor
    if flip:
        box[0::2] = origin_width - box[0::2][::-1]
    box[0::2] = np.clip(box[0::2] - x_offset, 0, 100)
    box[1::2] = np.clip(box[1::2] - y_offset, 0, 100)
    angle = -source[4] if flip else source[4]
    return np.concatenate((box, [angle, source[5]])).astype(np.float64)


def cyclic_corner_error(left, right):
    candidates = []
    for reverse in (False, True):
        ordered = right[::-1] if reverse else right
        for shift in range(4):
            distances = np.linalg.norm(left - np.roll(ordered, shift, axis=0), axis=1)
            candidates.append(float(np.sqrt(np.mean(distances ** 2))))
    return min(candidates)


class OBBMixupContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.augment = load_augmentation()
        cls.module = load_mosaic_obb(cls.augment)

    def run_mixup(self, source, jit_factor, flip, x_offset=0, y_offset=0):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        source = np.asarray([source], dtype=np.float64)

        class SourceDataset:
            def load_anno(self, index):
                return source.copy()

            def pull_item(self, index):
                return image.copy(), source.copy(), None, None

            def __len__(self):
                return 1

        wrapper = object.__new__(self.module.MosaicDetectionOBB)
        wrapper._dataset = SourceDataset()
        wrapper.mixup_scale = (jit_factor, jit_factor)
        randint_values = [0]
        if jit_factor > 1.0:
            randint_values.extend((y_offset, x_offset))
        with mock.patch.object(
            self.module.random,
            "uniform",
            side_effect=[jit_factor, 1.0 if flip else 0.0],
        ), mock.patch.object(self.module.random, "randint", side_effect=randint_values):
            mixed_image, mixed_labels = wrapper.mixup(
                image.copy(), np.empty((0, 6), dtype=np.float64), (100, 100)
            )
        return mixed_image, mixed_labels

    def assert_matches_reference(self, actual, expected, class_value, check_angle=True):
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(actual[2] - actual[0], 0.0)
        self.assertGreater(actual[3] - actual[1], 0.0)
        self.assertGreaterEqual(actual[2] - actual[0], actual[3] - actual[1])
        self.assertEqual(actual[5], class_value)
        self.assertLess(
            cyclic_corner_error(corners_from_target(actual), corners_from_target(expected)),
            1e-3,
        )
        if check_angle:
            angle_error = abs(
                (float(actual[4]) - float(expected[4]) + 90.0) % 180.0 - 90.0
            )
            self.assertLess(angle_error, 1e-3)

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
        np.testing.assert_allclose(mixed_labels[0], labels[0], atol=1e-5)

        with mock.patch.object(self.module.random, "uniform", side_effect=[1.0, 1.0]):
            with mock.patch.object(self.module.random, "randint", return_value=0):
                _, flipped_labels = wrapper.mixup(
                    image.copy(), np.empty((0, 6), dtype=np.float32), (40, 40)
                )

        self.assertEqual(tuple(flipped_labels.shape), (1, 6))
        self.assertAlmostEqual(float(flipped_labels[0, 4]), -17.0, places=5)
        self.assertAlmostEqual(float(flipped_labels[0, 5]), 2.0, places=5)

    def test_mixup_no_crop_matrix_preserves_transformed_obb(self):
        for angle in (0.0, 5.0, 20.0, 45.0, 80.0, -80.0):
            for jit_factor in (0.75, 1.0, 1.25):
                for flip in (False, True):
                    source = encode_obb(50.0, 50.0, 30.0, 15.0, angle, 4.0)
                    with self.subTest(angle=angle, jit_factor=jit_factor, flip=flip):
                        _, labels = self.run_mixup(source, jit_factor, flip)
                        self.assertEqual(len(labels), 1)
                        expected, visible = transformed_visible_target(
                            source, jit_factor, flip, 0, 0
                        )
                        self.assertEqual(len(visible), 4)
                        self.assert_matches_reference(labels[0], expected, 4.0)

    def test_mixup_one_edge_uses_visible_polygon_geometry(self):
        source = encode_obb(50.0, 35.0, 60.0, 20.0, 20.0, 7.0)
        _, labels = self.run_mixup(source, 1.6, False, x_offset=50, y_offset=0)
        self.assertEqual(len(labels), 1)
        expected, visible = transformed_visible_target(source, 1.6, False, 50, 0)
        legacy = legacy_mixup_target(source, 1.6, False, 50, 0)
        self.assertEqual(len(visible), 4)
        self.assertGreater(
            cyclic_corner_error(corners_from_target(legacy), corners_from_target(expected)),
            1.0,
        )
        self.assert_matches_reference(labels[0], expected, 7.0)

    def test_mixup_corner_uses_visible_polygon_geometry(self):
        source = encode_obb(50.0, 50.0, 60.0, 20.0, 45.0, 8.0)
        _, labels = self.run_mixup(source, 1.6, False, x_offset=50, y_offset=50)
        self.assertEqual(len(labels), 1)
        expected, visible = transformed_visible_target(source, 1.6, False, 50, 50)
        legacy = legacy_mixup_target(source, 1.6, False, 50, 50)
        self.assertEqual(len(visible), 5)
        self.assertGreater(
            cyclic_corner_error(corners_from_target(legacy), corners_from_target(expected)),
            1.0,
        )
        self.assert_matches_reference(labels[0], expected, 8.0)

    def test_mixup_boundary_matrix_matches_visible_polygon(self):
        # Scaled representatives of the #24 240x40, 120x80, and 80x80
        # aspect-ratio families, with both one-edge and corner crops.
        for width, height in ((60.0, 10.0), (40.0, 27.0), (20.0, 20.0)):
            for angle in (-80.0, -70.0, -45.0, -20.0, -5.0,
                          5.0, 20.0, 45.0, 70.0, 80.0):
                for crop_kind, x_offset, y_offset in (
                    ("one-edge", 50, 0),
                    ("corner", 50, 50),
                ):
                    source = encode_obb(50.0, 40.0, width, height, angle, 12.0)
                    with self.subTest(
                        width=width,
                        height=height,
                        angle=angle,
                        crop_kind=crop_kind,
                    ):
                        _, labels = self.run_mixup(
                            source, 1.6, False, x_offset=x_offset, y_offset=y_offset
                        )
                        expected, visible = transformed_visible_target(
                            source, 1.6, False, x_offset, y_offset
                        )
                        self.assertGreaterEqual(len(visible), 3)
                        self.assertEqual(len(labels), 1)
                        # Square fits have a 90-degree orientation tie; their
                        # fitted corners remain uniquely testable.
                        self.assert_matches_reference(
                            labels[0],
                            expected,
                            12.0,
                            check_angle=width != height,
                        )

    def test_mixup_accepts_triangle_visible_polygon(self):
        source = encode_obb(50.0, 50.0, 60.0, 20.0, 45.0, 9.0)
        _, labels = self.run_mixup(source, 2.0, False, x_offset=90, y_offset=10)
        self.assertEqual(len(labels), 1)
        expected, visible = transformed_visible_target(source, 2.0, False, 90, 10)
        self.assertEqual(len(visible), 3)
        # The right-triangle fit is square-tied: 0 and -90 degrees describe
        # the same geometry, so compare its corners rather than that scalar.
        self.assert_matches_reference(labels[0], expected, 9.0, check_angle=False)

    def test_mixup_rejects_fully_outside_and_zero_area_touch(self):
        for center_x in (10.0, 26.875):
            source = encode_obb(center_x, 50.0, 20.0, 20.0, 0.0, 10.0)
            with self.subTest(center_x=center_x):
                _, labels = self.run_mixup(source, 1.6, False, x_offset=59, y_offset=0)
                expected, visible = transformed_visible_target(source, 1.6, False, 59, 0)
                self.assertIsNone(expected)
                self.assertEqual(len(visible), 0)
                self.assertEqual(len(labels), 0)

    def test_maintained_dota_recipe_keeps_mixup_disabled(self):
        config = (
            ROOT / "exps" / "example" / "yolox_voc" / "yolox_dota_s_obb_kld.py"
        ).read_text()
        self.assertIn("self.enable_mixup = False", config)


if __name__ == "__main__":
    unittest.main()
