"""Regression coverage for the packaged native rotated-IoU capability."""

import unittest

import numpy as np

from DOTA_devkit_YOLO import polyiou
from yolox.utils.boxes import py_cpu_nms_poly


def polygon_iou(first, second):
    first_vector = polyiou.VectorDouble(first)
    second_vector = polyiou.VectorDouble(second)
    return polyiou.iou_poly(first_vector, second_vector)


class PolyiouNumericalTests(unittest.TestCase):
    square = [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]

    def test_identical_rectangles(self):
        self.assertAlmostEqual(polygon_iou(self.square, self.square), 1.0)

    def test_disjoint_rectangles(self):
        disjoint = [3.0, 0.0, 5.0, 0.0, 5.0, 2.0, 3.0, 2.0]
        self.assertAlmostEqual(polygon_iou(self.square, disjoint), 0.0)

    def test_partial_overlap(self):
        partial = [1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 1.0, 2.0]
        self.assertAlmostEqual(polygon_iou(self.square, partial), 1.0 / 3.0)

    def test_containment(self):
        contained = [0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5]
        self.assertAlmostEqual(polygon_iou(self.square, contained), 0.25)

    def test_equal_squares_rotated_45_degrees(self):
        rotated = [
            1.0,
            -0.4142135623730951,
            2.414213562373095,
            1.0,
            1.0,
            2.414213562373095,
            -0.4142135623730951,
            1.0,
        ]
        self.assertAlmostEqual(
            polygon_iou(self.square, rotated),
            0.707106781186547,
            places=12,
        )

    def test_reversed_point_order(self):
        reversed_square = [0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(polygon_iou(self.square, reversed_square), 1.0)

    def test_touching_edge(self):
        edge_touching = [2.0, 0.0, 4.0, 0.0, 4.0, 2.0, 2.0, 2.0]
        self.assertAlmostEqual(polygon_iou(self.square, edge_touching), 0.0)

    def test_touching_corner(self):
        corner_touching = [2.0, 2.0, 4.0, 2.0, 4.0, 4.0, 2.0, 4.0]
        self.assertAlmostEqual(polygon_iou(self.square, corner_touching), 0.0)


class RotatedNmsNativeTests(unittest.TestCase):
    def test_overlapping_lower_priority_duplicate_is_suppressed(self):
        rotated = [
            1.0,
            -0.4142135623730951,
            2.414213562373095,
            1.0,
            1.0,
            2.414213562373095,
            -0.4142135623730951,
            1.0,
        ]
        detections = np.array(
            [
                [*PolyiouNumericalTests.square, 0.95, 0.0],
                [*rotated, 0.50, 0.0],
            ],
            dtype=np.float64,
        )

        keep = py_cpu_nms_poly(detections, thresh=0.65)

        self.assertEqual(keep, [0])


if __name__ == "__main__":
    unittest.main()
