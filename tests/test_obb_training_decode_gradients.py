"""Regression coverage for the OBB training decode gradient chain."""

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_head():
    """Load only the head so the test does not require native polyiou."""
    path = ROOT / "yolox" / "models" / "yolo_head_obb_kld.py"
    spec = importlib.util.spec_from_file_location(
        "yolox.models.yolo_head_obb_kld_decode_gradient_test", path
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


def load_kld():
    path = ROOT / "yolox" / "models" / "KLD_loss.py"
    spec = importlib.util.spec_from_file_location(
        "maintained_kld_decode_gradient_test", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OBBTrainingDecodeGradientTests(unittest.TestCase):
    TARGET = (0.0, 0.0, 40.0, 10.0, 0.0)

    @classmethod
    def setUpClass(cls):
        cls.head_class = load_head()
        cls.kld = load_kld()

    def make_head(self):
        head = self.head_class.__new__(self.head_class)
        torch.nn.Module.__init__(head)
        head.n_anchors = 1
        head.num_classes = 1
        head.grids = [torch.zeros(1)]
        return head

    def evaluate_training_chain(self, raw_value, stride, dimension, dtype=torch.float32):
        """Run raw tensor -> maintained training decode -> merged KLD -> backward."""
        raw_values = torch.zeros((1, 7, 1, 1), dtype=dtype)
        raw_values[0, dimension, 0, 0] = raw_value
        raw = raw_values.requires_grad_()

        head = self.make_head()
        decoded, _ = head.get_output_and_grid(raw * 1.0, 0, stride, raw.dtype)
        prediction = decoded[..., :5].reshape(1, 5)
        target = torch.tensor([self.TARGET], dtype=dtype)
        loss = self.kld.KLDloss()(prediction, target).sum()
        loss.backward()
        return decoded.detach(), loss.detach(), raw.grad.detach()

    def evaluate_old_decode_chain(self, raw_value, stride, dimension):
        """Evaluate the unchanged mathematical exp(raw) * stride chain."""
        raw = torch.tensor(raw_value, dtype=torch.float32, requires_grad=True)
        decoded_dimension = torch.exp(raw) * stride
        ordinary_dimension = torch.tensor(float(stride), dtype=torch.float32)
        if dimension == 2:
            width, height = decoded_dimension, ordinary_dimension
        else:
            width, height = ordinary_dimension, decoded_dimension
        prediction = torch.stack(
            (
                torch.tensor(0.0),
                torch.tensor(0.0),
                width,
                height,
                torch.tensor(0.0),
            )
        ).unsqueeze(0)
        target = torch.tensor([self.TARGET])
        loss = self.kld.KLDloss()(prediction, target).sum()
        loss.backward()
        return loss.detach(), raw.grad.detach()

    def evaluate_fp64_reference(self, raw_value, stride, dimension):
        """Use the pre-rewrite prediction-first KLD expression in FP64."""
        raw = torch.tensor(raw_value, dtype=torch.float64, requires_grad=True)
        decoded_dimension = torch.exp(raw) * float(stride)
        ordinary_dimension = torch.tensor(float(stride), dtype=torch.float64)
        if dimension == 2:
            width, height = decoded_dimension, ordinary_dimension
        else:
            width, height = ordinary_dimension, decoded_dimension
        prediction = torch.stack(
            (
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(0.0, dtype=torch.float64),
                width,
                height,
                torch.tensor(0.0, dtype=torch.float64),
            )
        ).unsqueeze(0)
        target = torch.tensor([self.TARGET], dtype=torch.float64)
        pred = prediction.view(-1, 5)
        target = target.view(-1, 5)
        delta_x = pred[:, 0] - target[:, 0]
        delta_y = pred[:, 1] - target[:, 1]
        pred_angle = math.pi * pred[:, 4] / 180.0
        target_angle = math.pi * target[:, 4] / 180.0
        delta_angle = pred_angle - target_angle
        kld = 0.5 * (
            4 * (delta_x * torch.cos(target_angle) + delta_y * torch.sin(target_angle)) ** 2
            / target[:, 2] ** 2
            + 4 * (delta_y * torch.cos(target_angle) - delta_x * torch.sin(target_angle)) ** 2
            / target[:, 3] ** 2
        ) + 0.5 * (
            pred[:, 3] ** 2 / target[:, 2] ** 2 * torch.sin(delta_angle) ** 2
            + pred[:, 2] ** 2 / target[:, 3] ** 2 * torch.sin(delta_angle) ** 2
            + pred[:, 3] ** 2 / target[:, 3] ** 2 * torch.cos(delta_angle) ** 2
            + pred[:, 2] ** 2 / target[:, 2] ** 2 * torch.cos(delta_angle) ** 2
        ) + 0.5 * (
            torch.log(target[:, 3] ** 2 / pred[:, 3] ** 2)
            + torch.log(target[:, 2] ** 2 / pred[:, 2] ** 2)
        ) - 1.0
        output = 1.0 - 1.0 / (1.0 + torch.log(kld + 1.0))
        output.sum().backward()
        return decoded_dimension.detach(), output.detach(), raw.grad.detach()

    def training_decoded_dimensions(self, raw_values, stride):
        raw_values = torch.as_tensor(raw_values, dtype=torch.float32)
        raw = torch.zeros((raw_values.numel(), 7, 1, 1), dtype=torch.float32)
        raw[:, 2, 0, 0] = raw_values
        raw[:, 3, 0, 0] = raw_values
        head = self.make_head()
        decoded, _ = head.get_output_and_grid(raw * 1.0, 0, stride, raw.dtype)
        return decoded[:, 0, 2:4].detach()

    def test_subnormal_training_decode_gradients_are_finite(self):
        for stride in (8, 16, 32):
            for dimension, name in ((2, "width"), (3, "height")):
                with self.subTest(stride=stride, dimension=name):
                    decoded, loss, raw_gradient = self.evaluate_training_chain(
                        -97.0, stride, dimension
                    )
                    self.assertTrue(torch.isfinite(decoded).all())
                    self.assertGreater(float(decoded[0, 0, dimension]), 0.0)
                    self.assertTrue(torch.isfinite(loss).all())
                    self.assertTrue(torch.isfinite(raw_gradient).all())
                    self.assertNotEqual(float(raw_gradient[0, dimension, 0, 0]), 0.0)

    def test_subnormal_gradient_matrix_matches_fp64_reference(self):
        for stride in (8, 16, 32):
            for dimension, name in ((2, "width"), (3, "height")):
                with self.subTest(stride=stride, dimension=name):
                    _, _, raw_gradient = self.evaluate_training_chain(
                        -97.0, stride, dimension
                    )
                    _, reference_output, reference_gradient = self.evaluate_fp64_reference(
                        -97.0, stride, dimension
                    )
                    self.assertTrue(torch.isfinite(reference_output).all())
                    self.assertTrue(torch.isfinite(reference_gradient).all())
                    self.assertTrue(
                        torch.allclose(
                            raw_gradient[0, dimension, 0, 0].double(),
                            reference_gradient,
                            rtol=5e-5,
                            atol=2e-8,
                        ),
                        (stride, name, raw_gradient, reference_gradient),
                    )

    def test_subnormal_failure_matrix_remains_positive_and_finite(self):
        for raw_value in (-96.5, -96.75, -97.0):
            for stride in (8, 16, 32):
                for dimension, name in ((2, "width"), (3, "height")):
                    with self.subTest(raw=raw_value, stride=stride, dimension=name):
                        decoded, loss, raw_gradient = self.evaluate_training_chain(
                            raw_value, stride, dimension
                        )
                        self.assertGreater(float(decoded[0, 0, dimension]), 0.0)
                        self.assertTrue(torch.isfinite(loss).all())
                        self.assertTrue(torch.isfinite(raw_gradient).all())

    def test_ordinary_training_forward_parity(self):
        raw_values = torch.linspace(-4.0, 4.0, 161)
        max_absolute = 0.0
        max_relative = 0.0
        for stride in (8, 16, 32):
            actual = self.training_decoded_dimensions(raw_values, stride)
            expected = torch.exp(raw_values[:, None]) * stride
            difference = (actual - expected).abs()
            max_absolute = max(max_absolute, float(difference.max()))
            max_relative = max(
                max_relative, float((difference / expected.abs()).max())
            )
        self.assertLess(max_absolute, 5e-4)
        self.assertLess(max_relative, 5e-6)

    def test_ordinary_training_gradient_parity(self):
        max_difference = 0.0
        raw_values = (
            -4.0, -3.1, -2.25, -1.7, -1.0, -0.37, -0.1, 0.123,
            0.7, 1.1, 1.8, 2.37, 3.1, 3.77, 4.0,
        )
        for raw_value in raw_values:
            for stride in (8, 16, 32):
                for dimension, name in ((2, "width"), (3, "height")):
                    with self.subTest(raw=raw_value, stride=stride, dimension=name):
                        _, _, new_gradient = self.evaluate_training_chain(
                            raw_value, stride, dimension
                        )
                        _, old_gradient = self.evaluate_old_decode_chain(
                            raw_value, stride, dimension
                        )
                        difference = abs(
                            float(new_gradient[0, dimension, 0, 0] - old_gradient)
                        )
                        max_difference = max(max_difference, difference)
        self.assertLess(max_difference, 5e-7)

    def test_training_decode_preserves_input_dtype(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                raw_values = torch.zeros((1, 7, 1, 1), dtype=dtype)
                head = self.make_head()
                decoded, _ = head.get_output_and_grid(raw_values * 1.0, 0, 16, dtype)
                self.assertEqual(decoded.dtype, dtype)

    def test_training_exact_zero_has_only_the_expected_zero_gradient(self):
        for stride in (8, 16, 32):
            for dimension, name in ((2, "width"), (3, "height")):
                with self.subTest(stride=stride, dimension=name):
                    decoded, loss, raw_gradient = self.evaluate_training_chain(
                        -108.0, stride, dimension
                    )
                    self.assertEqual(float(decoded[0, 0, dimension]), 0.0)
                    self.assertTrue(torch.isfinite(loss).all())
                    self.assertEqual(float(raw_gradient[0, dimension, 0, 0]), 0.0)

    def test_training_decode_boundary_movement_is_explicit(self):
        old_zero = -103.972084045
        new_zero = {8: -106.051528931, 16: -106.744674683, 32: -107.437820435}
        old_overflow = {8: 86.643402100, 16: 85.950256348, 32: 85.257110596}
        new_overflow = {8: 86.643394470, 16: 85.950248718, 32: 85.257102966}
        positive_infinity = torch.tensor(float("inf"), dtype=torch.float32)
        negative_infinity = torch.tensor(float("-inf"), dtype=torch.float32)

        for stride in (8, 16, 32):
            old_zero_value = torch.tensor(old_zero, dtype=torch.float32)
            old_zero_next = torch.nextafter(old_zero_value, positive_infinity)
            self.assertEqual(float(torch.exp(old_zero_value) * stride), 0.0)
            self.assertGreater(float(torch.exp(old_zero_next) * stride), 0.0)

            new_zero_value = torch.tensor(new_zero[stride], dtype=torch.float32)
            new_zero_next = torch.nextafter(new_zero_value, positive_infinity)
            new_decoded = self.training_decoded_dimensions([new_zero[stride]], stride)
            next_decoded = self.training_decoded_dimensions(
                [float(new_zero_next)], stride
            )
            self.assertEqual(float(new_decoded[0, 0]), 0.0)
            self.assertGreater(float(next_decoded[0, 0]), 0.0)

            old_overflow_value = torch.tensor(old_overflow[stride], dtype=torch.float32)
            old_overflow_previous = torch.nextafter(old_overflow_value, negative_infinity)
            self.assertTrue(torch.isinf(torch.exp(old_overflow_value) * stride))
            self.assertTrue(torch.isfinite(torch.exp(old_overflow_previous) * stride))

            new_overflow_value = torch.tensor(new_overflow[stride], dtype=torch.float32)
            new_overflow_previous = torch.nextafter(
                new_overflow_value, negative_infinity
            )
            new_decoded = self.training_decoded_dimensions(
                [new_overflow[stride]], stride
            )
            previous_decoded = self.training_decoded_dimensions(
                [float(new_overflow_previous)], stride
            )
            self.assertTrue(torch.isinf(new_decoded[0, 0]))
            self.assertTrue(torch.isfinite(previous_decoded[0, 0]))

    def test_eager_decode_keeps_exp_stride_forward_contract(self):
        head = self.make_head()
        head.hw = [(1, 1)]
        head.strides = [8]
        outputs = torch.tensor(
            [[[0.25, -0.25, -2.0, 1.0, 0.3, 0.4, 0.5]]], dtype=torch.float32
        )
        expected = outputs.clone()
        grid = torch.zeros((1, 1, 2), dtype=torch.float32)
        strides = torch.full((1, 1, 1), 8.0, dtype=torch.float32)
        expected[..., :2] = (expected[..., :2] + grid) * strides
        expected[..., 2:4] = torch.exp(expected[..., 2:4]) * strides
        expected[..., 4] = (expected[..., 4].sigmoid() - 0.5) * 180.0

        actual = head.decode_outputs(outputs.clone(), torch.float32)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
