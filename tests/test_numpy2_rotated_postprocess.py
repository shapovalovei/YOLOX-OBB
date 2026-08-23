"""Regression coverage for rotated postprocess on modern NumPy."""

import unittest
from unittest import mock

import numpy as np
import torch

from yolox.utils import boxes as boxes_module


class NumPy2RotatedPostprocessTests(unittest.TestCase):
    def test_postprocessobb_kld_converts_polygon_to_integer_coordinates(self):
        observed = {}
        original_intp = boxes_module.np.intp

        def capture_intp(value, *args, **kwargs):
            converted = original_intp(value, *args, **kwargs)
            observed["dtype"] = converted.dtype
            observed["finite"] = bool(np.isfinite(converted).all())
            observed["shape"] = converted.shape
            return converted

        prediction = torch.tensor(
            [[
                [32.0, 32.0, 20.0, 10.0, 25.0, 0.99, 0.99],
                [96.0, 96.0, 20.0, 10.0, 25.0, 0.99, 0.001],
            ]],
            dtype=torch.float32,
        )

        with mock.patch.object(boxes_module.np, "intp", side_effect=capture_intp):
            output = boxes_module.postprocessobb_kld(
                prediction, num_classes=1, conf_thre=0.01, nms_thre=0.65
            )

        self.assertIn("dtype", observed)
        self.assertTrue(np.issubdtype(observed["dtype"], np.integer))
        self.assertTrue(observed["finite"])
        self.assertEqual(observed["shape"], (4, 2))
        self.assertIsNotNone(output[0])
        self.assertTrue(torch.isfinite(output[0]).all())
        self.assertEqual(tuple(output[0].shape), (1, 10))


if __name__ == "__main__":
    unittest.main()
