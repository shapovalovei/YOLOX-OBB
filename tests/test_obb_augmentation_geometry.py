"""Regression tests for OBB geometry through augmentation and Mosaic."""

import ast
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
    spec = importlib.util.spec_from_file_location("maintained_data_augment_obb_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)

    # The package-level utility import also loads optional native extensions.
    # Keep this test focused on the augmentation module itself.
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


def transform_points(points, matrix, perspective):
    homogeneous = np.ones((len(points), 3), dtype=np.float64)
    homogeneous[:, :2] = points
    transformed = homogeneous @ matrix.T
    if perspective:
        transformed = transformed[:, :2] / transformed[:, 2:3]
    return transformed[:, :2]


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
    """Independently scan polygon edge orientations for a minimum-area OBB."""
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
        area = width * height
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
        candidates.append((area, center[0], center[1], long_edge, short_edge, angle))

    if not candidates:
        raise ValueError("polygon is degenerate")
    _, center_x, center_y, long_edge, short_edge, angle = min(candidates)
    return np.asarray(
        [center_x, center_y, long_edge, short_edge, angle], dtype=np.float64
    )


def corners_from_obb(obb):
    center_x, center_y, width, height, angle = obb
    return corners_from_target(encode_obb(center_x, center_y, width, height, angle, 0.0))


def cyclic_corner_error(left, right):
    candidates = []
    for reverse in (False, True):
        ordered = right[::-1] if reverse else right
        for shift in range(4):
            distances = np.linalg.norm(left - np.roll(ordered, shift, axis=0), axis=1)
            candidates.append(float(np.sqrt(np.mean(distances ** 2))))
    return min(candidates)


def polygon_iou(left, right):
    left_area = abs(float(cv2.contourArea(left.astype(np.float32))))
    right_area = abs(float(cv2.contourArea(right.astype(np.float32))))
    intersection, _ = cv2.intersectConvexConvex(
        left.astype(np.float32), right.astype(np.float32)
    )
    return float(intersection) / (left_area + right_area - float(intersection) + 1e-12)


class OBBAugmentationGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.augment = load_augmentation()

    def run_case(self, angle, mode):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        target = np.asarray([encode_obb(320.0, 240.0, 240.0, 40.0, angle, 7.0)])
        params = {
            "identity": {
                "degrees": 0.0, "translate": 0.0, "scale": (1.0, 1.0),
                "shear": 0.0, "perspective": 0.0,
                "values": [0.0, 1.0, 0.0, 0.0, 0.5, 0.5],
            },
            "translation": {
                "degrees": 0.0, "translate": 0.2, "scale": (1.0, 1.0),
                "shear": 0.0, "perspective": 0.0,
                "values": [0.0, 1.0, 0.0, 0.0, 0.62, 0.38],
            },
            "scale": {
                "degrees": 0.0, "translate": 0.0, "scale": (1.2, 1.2),
                "shear": 0.0, "perspective": 0.0,
                "values": [0.0, 1.2, 0.0, 0.0, 0.5, 0.5],
            },
            "rotation": {
                "degrees": 17.0, "translate": 0.0, "scale": (1.0, 1.0),
                "shear": 0.0, "perspective": 0.0,
                "values": [17.0, 1.0, 0.0, 0.0, 0.5, 0.5],
            },
            "shear": {
                "degrees": 0.0, "translate": 0.0, "scale": (1.0, 1.0),
                "shear": 5.0, "perspective": 0.0,
                "values": [0.0, 1.0, 5.0, -3.0, 0.5, 0.5],
            },
            "perspective": {
                "degrees": 13.0, "translate": 0.0, "scale": (1.0, 1.0),
                "shear": 2.0, "perspective": 1.0,
                "values": [13.0, 1.0, 2.0, -2.0, 0.5, 0.5],
            },
        }[mode]
        matrix = {"value": np.eye(3, dtype=np.float64)}

        if params["perspective"]:
            def capture_perspective(source, transform, **kwargs):
                matrix["value"] = np.asarray(transform, dtype=np.float64)
                return source

            warp_patch = mock.patch.object(
                self.augment.cv2, "warpPerspective", side_effect=capture_perspective
            )
        else:
            def capture_affine(source, transform, **kwargs):
                full = np.eye(3, dtype=np.float64)
                full[:2] = transform
                matrix["value"] = full
                return source

            warp_patch = mock.patch.object(
                self.augment.cv2, "warpAffine", side_effect=capture_affine
            )

        with mock.patch.object(
            self.augment.random, "uniform", side_effect=params["values"]
        ), warp_patch:
            _, output = self.augment.random_perspective(
                image,
                target.copy(),
                **{
                    key: params[key]
                    for key in ("degrees", "translate", "scale", "shear", "perspective")
                }
            )

        self.assertEqual(len(output), 1)
        transformed = transform_points(
            corners_from_target(target[0]), matrix["value"], bool(params["perspective"])
        )
        expected_obb = canonical_min_area_obb(transformed)
        reconstructed = corners_from_target(output[0])
        expected_corners = corners_from_obb(expected_obb)
        return output[0], transformed, expected_corners, reconstructed

    def run_translated_boundary_case(self, center_x, center_y, width, height, angle):
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        target = np.asarray([encode_obb(200.0, 200.0, width, height, angle, 11.0)])
        matrix = {"value": np.eye(3, dtype=np.float64)}

        def capture_affine(source, transform, **kwargs):
            full = np.eye(3, dtype=np.float64)
            full[:2] = transform
            matrix["value"] = full
            return source

        with mock.patch.object(
            self.augment.random,
            "uniform",
            side_effect=[0.0, 1.0, 0.0, 0.0, center_x / 400.0, center_y / 400.0],
        ), mock.patch.object(
            self.augment.cv2, "warpAffine", side_effect=capture_affine
        ):
            _, output = self.augment.random_perspective(
                image,
                target.copy(),
                degrees=0.0,
                translate=1.0,
                scale=(1.0, 1.0),
                shear=0.0,
                perspective=0.0,
            )

        transformed = transform_points(corners_from_target(target[0]), matrix["value"], False)
        visible = intersect_with_canvas(transformed, 400, 400)
        self.assertEqual(len(output), 1)
        self.assertGreater(float(cv2.contourArea(visible.astype(np.float32))), 1e-6)
        expected_obb = canonical_min_area_obb(visible)
        return output[0], expected_obb, visible

    def test_identity_translation_scale_rotation_shear_and_perspective(self):
        for mode in ("identity", "translation", "scale", "rotation", "shear", "perspective"):
            for angle in (0.0, 10.0, -10.0, 20.0, -20.0, 45.0, -45.0, 80.0, -80.0):
                with self.subTest(mode=mode, angle=angle):
                    output, transformed, expected, reconstructed = self.run_case(angle, mode)
                    self.assertTrue(np.isfinite(output).all())
                    self.assertGreater(output[2] - output[0], 0.0)
                    self.assertGreater(output[3] - output[1], 0.0)
                    self.assertEqual(output[5], 7.0)
                    self.assertLess(cyclic_corner_error(expected, reconstructed), 1e-3)
                    self.assertGreater(polygon_iou(transformed, reconstructed), 0.90)

    def test_nonzero_transform_reconstructs_a_new_orientation(self):
        output, transformed, _, _ = self.run_case(45.0, "rotation")
        transformed_orientation = canonical_min_area_obb(transformed)[4]
        orientation_error = abs((float(output[4]) - transformed_orientation + 90.0) % 180.0 - 90.0)
        self.assertLess(orientation_error, 1e-3)
        self.assertGreater(abs(float(output[4])), 1.0)

    def test_mosaic_boundary_geometry_is_clipped_after_corner_transform(self):
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        # The encoded HBB extends beyond the left image edge. Mosaic must pass
        # these dimension fields through unchanged until OBB corners are made.
        target = np.asarray([encode_obb(192.0, 256.0, 512.0, 64.0, 20.0, 3.0)])
        matrix = {"value": np.eye(3, dtype=np.float64)}

        def capture_affine(source, transform, **kwargs):
            full = np.eye(3, dtype=np.float64)
            full[:2] = transform
            matrix["value"] = full
            return source

        with mock.patch.object(
            self.augment.random, "uniform", side_effect=[0.0, 1.0, 0.0, 0.0, 0.5, 0.5]
        ), mock.patch.object(
            self.augment.cv2, "warpAffine", side_effect=capture_affine
        ):
            _, output = self.augment.random_perspective(
                image,
                target.copy(),
                degrees=0.0,
                translate=0.0,
                scale=(1.0, 1.0),
                shear=0.0,
                perspective=0.0,
                border=(-128, -128),
            )

        self.assertEqual(len(output), 1)
        transformed = transform_points(corners_from_target(target[0]), matrix["value"], False)
        visible = intersect_with_canvas(transformed, 256, 256)
        self.assertGreater(float(cv2.contourArea(visible.astype(np.float32))), 1e-6)
        expected_obb = canonical_min_area_obb(visible)
        self.assertLess(
            cyclic_corner_error(corners_from_obb(expected_obb), corners_from_target(output[0])),
            1e-3,
        )
        self.assertLess(
            abs((float(output[0][4]) - expected_obb[4] + 90.0) % 180.0 - 90.0),
            1e-3,
        )
        self.assertEqual(output[0][5], 3.0)

    def test_boundary_intersection_matrix_matches_visible_polygon(self):
        cases = (
            (70.0, 200.0, 240.0, 40.0, 20.0),
            (330.0, 200.0, 240.0, 40.0, -20.0),
            (200.0, 70.0, 160.0, 60.0, 45.0),
            (70.0, 70.0, 160.0, 60.0, -45.0),
            (330.0, 70.0, 120.0, 80.0, 30.0),
        )
        for case in cases:
            with self.subTest(case=case):
                output, expected_obb, visible = self.run_translated_boundary_case(*case)
                actual_corners = corners_from_target(output)
                expected_corners = corners_from_obb(expected_obb)
                self.assertTrue(np.isfinite(output).all())
                self.assertGreater(output[2] - output[0], 0.0)
                self.assertGreater(output[3] - output[1], 0.0)
                self.assertLess(cyclic_corner_error(expected_corners, actual_corners), 1e-3)
                self.assertLess(
                    abs((float(output[4]) - expected_obb[4] + 90.0) % 180.0 - 90.0),
                    1e-3,
                )
                self.assertEqual(output[5], 11.0)
                self.assertGreater(float(cv2.contourArea(visible.astype(np.float32))), 1e-6)

    def test_boundary_touch_preserves_the_visible_polygon(self):
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        for width, height, angle in ((240.0, 40.0, 0.0), (240.0, 40.0, 20.0)):
            with self.subTest(width=width, height=height, angle=angle):
                radians = math.radians(angle)
                center_x = abs(width * math.cos(radians) / 2.0) + abs(
                    height * math.sin(radians) / 2.0
                )
                target = np.asarray(
                    [encode_obb(center_x, 200.0, width, height, angle, 13.0)]
                )
                with mock.patch.object(
                    self.augment.random,
                    "uniform",
                    side_effect=[0.0, 1.0, 0.0, 0.0, 0.5, 0.5],
                ):
                    _, output = self.augment.random_perspective(
                        image, target.copy(), degrees=0.0, translate=0.0,
                        scale=(1.0, 1.0), shear=0.0, perspective=0.0,
                    )

                self.assertEqual(len(output), 1)
                visible = intersect_with_canvas(corners_from_target(target[0]), 400, 400)
                expected = canonical_min_area_obb(visible)
                self.assertLess(
                    cyclic_corner_error(
                        corners_from_obb(expected), corners_from_target(output[0])
                    ),
                    1e-3,
                )
                self.assertEqual(output[0][5], 13.0)

    def test_polygon_intersection_handles_touch_empty_and_triangle(self):
        touch = self.augment._clip_polygon_to_rect(
            np.asarray([[-20.0, 100.0], [0.0, 100.0], [0.0, 200.0], [-20.0, 200.0]]),
            400,
            400,
        )
        self.assertEqual(self.augment._polygon_area(touch), 0.0)

        empty = self.augment._clip_polygon_to_rect(
            np.asarray([[-40.0, 80.0], [-20.0, 80.0], [-20.0, 100.0], [-40.0, 100.0]]),
            400,
            400,
        )
        self.assertEqual(len(empty), 0)

        triangle = np.asarray([[10.0, 10.0], [100.0, 10.0], [10.0, 100.0]])
        obb = self.augment.corners_to_obb(triangle)
        self.assertTrue(np.isfinite(obb).all())
        self.assertGreater(obb[2], 0.0)
        self.assertGreater(obb[3], 0.0)

    def test_mosaic_code_does_not_clip_encoded_dimensions_as_hbb(self):
        source = (ROOT / "yolox" / "data" / "datasets" / "mosaicdetection_obb.py").read_text()
        tree = ast.parse(source)
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "clip" and any(
                    isinstance(argument, ast.Subscript)
                    and isinstance(argument.value, ast.Name)
                    and argument.value.id == "mosaic_labels"
                    for argument in node.args
                ):
                    forbidden.append(node.lineno)
        self.assertEqual(forbidden, [])

    def test_obb_representations_are_periodic_and_swap_equivalent(self):
        base = corners_from_obb(np.asarray([320.0, 240.0, 240.0, 40.0, 20.0]))
        for equivalent in (
            np.asarray([320.0, 240.0, 240.0, 40.0, 200.0]),
            np.asarray([320.0, 240.0, 40.0, 240.0, 110.0]),
            np.asarray([320.0, 240.0, 40.0, 240.0, -70.0]),
        ):
            with self.subTest(angle=equivalent[4], width=equivalent[2], height=equivalent[3]):
                self.assertLess(cyclic_corner_error(base, corners_from_obb(equivalent)), 1e-9)

        axis_aligned = corners_from_obb(np.asarray([320.0, 240.0, 240.0, 40.0, 0.0]))
        for angle in (90.0, -90.0):
            equivalent = np.asarray([320.0, 240.0, 40.0, 240.0, angle])
            self.assertLess(cyclic_corner_error(axis_aligned, corners_from_obb(equivalent)), 1e-9)

    def test_invalid_or_degenerate_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            self.augment.corners_to_obb(np.full((4, 2), np.nan))
        with self.assertRaises(ValueError):
            self.augment.corners_to_obb(np.zeros((4, 2)))


if __name__ == "__main__":
    unittest.main()
