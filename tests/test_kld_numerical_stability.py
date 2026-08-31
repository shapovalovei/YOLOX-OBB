"""Regression tests for valid-positive KLD numerical stability."""

import importlib.util
import math
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_kld():
    path = ROOT / "yolox" / "models" / "KLD_loss.py"
    spec = importlib.util.spec_from_file_location("maintained_kld_numerical_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_kld_loss(pred, target, taf=1.0):
    """The pre-fix expression, retained only as an ordinary-value reference."""
    pred = pred.view(-1, 5)
    target = target.view(-1, 5)

    delta_x = pred[:, 0] - target[:, 0]
    delta_y = pred[:, 1] - target[:, 1]
    pred_angle = math.pi * pred[:, 4] / 180.0
    target_angle = math.pi * target[:, 4] / 180.0
    delta_angle = pred_angle - target_angle

    kld = 0.5 * (
        4 * torch.pow(
            delta_x * torch.cos(target_angle) + delta_y * torch.sin(target_angle), 2
        )
        / torch.pow(target[:, 2], 2)
        + 4
        * torch.pow(
            delta_y * torch.cos(target_angle) - delta_x * torch.sin(target_angle), 2
        )
        / torch.pow(target[:, 3], 2)
    ) + 0.5 * (
        torch.pow(pred[:, 3], 2)
        / torch.pow(target[:, 2], 2)
        * torch.pow(torch.sin(delta_angle), 2)
        + torch.pow(pred[:, 2], 2)
        / torch.pow(target[:, 3], 2)
        * torch.pow(torch.sin(delta_angle), 2)
        + torch.pow(pred[:, 3], 2)
        / torch.pow(target[:, 3], 2)
        * torch.pow(torch.cos(delta_angle), 2)
        + torch.pow(pred[:, 2], 2)
        / torch.pow(target[:, 2], 2)
        * torch.pow(torch.cos(delta_angle), 2)
    ) + 0.5 * (
        torch.log(torch.pow(target[:, 3], 2) / torch.pow(pred[:, 3], 2))
        + torch.log(torch.pow(target[:, 2], 2) / torch.pow(pred[:, 2], 2))
    ) - 1.0

    return 1 - 1 / (taf + torch.log(kld + 1))


class KLDNumericalStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kld = load_kld()

    def evaluate(self, prediction_values, target_values, dtype=torch.float32):
        prediction = torch.tensor(
            [prediction_values], dtype=dtype, requires_grad=True
        )
        target = torch.tensor([target_values], dtype=dtype)
        output = self.kld.KLDloss()(prediction, target)
        output.sum().backward()
        return output.detach(), prediction.grad.detach()

    def test_fp32_tiny_prediction_has_finite_forward_and_gradient(self):
        output, gradient = self.evaluate(
            [100.0, 80.0, 9.313225746154785e-10, 10.0, 15.0],
            [100.0, 80.0, 40.0, 10.0, 15.0],
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(gradient).all())

    def test_fp32_tiny_target_has_finite_forward_and_gradient(self):
        output, gradient = self.evaluate(
            [100.0, 80.0, 40.0, 10.0, 15.0],
            [100.0, 80.0, 2.0 ** -61, 10.0, 15.0],
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(gradient).all())

    def test_fp32_tiny_target_height_has_finite_forward_and_gradient(self):
        output, gradient = self.evaluate(
            [100.0, 80.0, 40.0, 10.0, 15.0],
            [100.0, 80.0, 40.0, 2.0 ** -61, 15.0],
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(gradient).all())

    def test_identical_tiny_positive_boxes_remain_finite(self):
        output, gradient = self.evaluate(
            [100.0, 80.0, 2.0 ** -75, 2.0 ** -75, 15.0],
            [100.0, 80.0, 2.0 ** -75, 2.0 ** -75, 15.0],
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertAlmostEqual(float(output), 0.0, places=6)

    def test_decoded_zero_prediction_is_conditioned_without_repairing_negative(self):
        output, gradient = self.evaluate(
            [100.0, 80.0, 0.0, 10.0, 15.0],
            [100.0, 80.0, 40.0, 10.0, 15.0],
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(gradient).all())

        negative_prediction = torch.tensor(
            [[100.0, 80.0, -1.0, 10.0, 15.0]], dtype=torch.float32
        )
        target = torch.tensor(
            [[100.0, 80.0, 40.0, 10.0, 15.0]], dtype=torch.float32
        )
        self.assertFalse(torch.isfinite(self.kld.kld_loss(negative_prediction, target)).all())

    def test_extreme_aspect_ratio_with_angle_and_center_offset_is_finite(self):
        output, gradient = self.evaluate(
            [106.0, 77.0, 2.0 ** -30, 512.0, -25.0],
            [100.0, 80.0, 512.0, 2.0 ** -30, 15.0],
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(gradient).all())

    def test_large_finite_prediction_dimensions_are_finite(self):
        output, gradient = self.evaluate(
            [100.0, 80.0, 1.0e20, 1.0e20, 15.0],
            [100.0, 80.0, 40.0, 10.0, 15.0],
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(gradient).all())

    def test_cpu_fp16_ordinary_large_identical_boxes_are_finite(self):
        try:
            output, gradient = self.evaluate(
                [100.0, 80.0, 256.0, 256.0, 15.0],
                [100.0, 80.0, 256.0, 256.0, 15.0],
                dtype=torch.float16,
            )
        except RuntimeError as error:
            self.skipTest("CPU FP16 KLD operations unavailable: {}".format(error))
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(gradient).all())

    def test_cpu_fp16_subpixel_positive_boxes_are_finite(self):
        try:
            output, gradient = self.evaluate(
                [100.0, 80.0, 2.0 ** -13, 2.0 ** -13, 15.0],
                [100.0, 80.0, 2.0 ** -13, 2.0 ** -13, 15.0],
                dtype=torch.float16,
            )
        except RuntimeError as error:
            self.skipTest("CPU FP16 KLD operations unavailable: {}".format(error))
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(gradient).all())

    def test_assignment_returns_finite_matrix_for_positive_small_target(self):
        predictions = torch.tensor(
            [
                [100.0, 80.0, 40.0, 10.0, 15.0],
                [106.0, 77.0, 8.0, 4.0, -25.0],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [
                [100.0, 80.0, 2.0 ** -61, 10.0, 15.0],
                [100.0, 80.0, 40.0, 10.0, 15.0],
            ],
            dtype=torch.float32,
        )
        matrix = self.kld.compute_kld_loss(targets, predictions)
        self.assertEqual(tuple(matrix.shape), (2, 2))
        self.assertEqual(matrix.device, predictions.device)
        self.assertEqual(matrix.dtype, torch.float32)
        self.assertFalse(matrix.requires_grad)
        self.assertTrue(torch.isfinite(matrix).all())

    def test_module_and_function_paths_have_identical_stable_values(self):
        prediction = torch.tensor(
            [[100.0, 80.0, 9.313225746154785e-10, 10.0, 15.0]],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [[100.0, 80.0, 2.0 ** -61, 10.0, 15.0]], dtype=torch.float32
        )
        module_output = self.kld.KLDloss()(prediction, target)
        function_output = self.kld.kld_loss(prediction, target)
        self.assertTrue(torch.equal(module_output, function_output))

    def test_taf_mapping_and_identical_box_zero_distance_are_unchanged(self):
        prediction = torch.tensor(
            [[100.0, 80.0, 40.0, 10.0, 15.0]], dtype=torch.float32
        )
        target = prediction.clone()
        self.assertAlmostEqual(
            float(self.kld.KLDloss(taf=1.0)(prediction, target)), 0.0, places=6
        )
        self.assertAlmostEqual(
            float(self.kld.KLDloss(taf=2.0)(prediction, target)), 0.5, places=6
        )

    def test_ordinary_forward_and_prediction_gradient_match_legacy_expression(self):
        cases = [
            ([100.0, 80.0, 40.0, 10.0, 15.0], [100.0, 80.0, 40.0, 10.0, 15.0]),
            ([106.0, 77.0, 40.0, 10.0, 15.0], [100.0, 80.0, 40.0, 10.0, 15.0]),
            ([100.0, 80.0, 52.0, 8.0, -25.0], [100.0, 80.0, 40.0, 10.0, 15.0]),
            ([100.0, 80.0, 8.0, 4.0, 90.0], [100.0, 80.0, 40.0, 10.0, -89.0]),
            ([106.0, 77.0, 256.0, 256.0, 0.0], [100.0, 80.0, 40.0, 10.0, 15.0]),
            ([106.0, 77.0, 500.0, 3.0, -89.0], [100.0, 80.0, 40.0, 10.0, 15.0]),
        ]

        for dtype in (torch.float64, torch.float32):
            for prediction_values, target_values in cases:
                prediction = torch.tensor(
                    [prediction_values], dtype=dtype, requires_grad=True
                )
                target = torch.tensor([target_values], dtype=dtype)
                reference = legacy_kld_loss(prediction, target)

                candidate_prediction = torch.tensor(
                    [prediction_values], dtype=dtype, requires_grad=True
                )
                candidate = self.kld.kld_loss(candidate_prediction, target)
                reference.sum().backward()
                candidate.sum().backward()

                self.assertTrue(torch.isfinite(reference).all())
                self.assertTrue(torch.isfinite(candidate).all())
                self.assertTrue(
                    torch.allclose(candidate, reference, rtol=3e-6, atol=3e-7),
                    (dtype, prediction_values, reference, candidate),
                )
                self.assertTrue(
                    torch.allclose(
                        candidate_prediction.grad,
                        prediction.grad,
                        rtol=3e-5,
                        atol=1e-7,
                    ),
                    (
                        dtype,
                        prediction_values,
                        prediction.grad,
                        candidate_prediction.grad,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
