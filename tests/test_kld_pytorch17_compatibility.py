"""Regression tests for the maintained KLD PyTorch 1.7 compatibility boundary."""

import contextlib
import importlib.util
import unittest
from pathlib import Path

import torch

from test_kld_numerical_stability import legacy_kld_loss


ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


def load_kld():
    path = ROOT / "yolox" / "models" / "KLD_loss.py"
    spec = importlib.util.spec_from_file_location(
        "maintained_kld_pytorch17_compatibility_test", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def temporary_torch_attributes(**replacements):
    """Apply narrow torch-module replacements and restore every attribute."""
    originals = {}
    try:
        for name, replacement in replacements.items():
            originals[name] = getattr(torch, name, _MISSING)
            if replacement is _MISSING:
                if originals[name] is not _MISSING:
                    delattr(torch, name)
            else:
                setattr(torch, name, replacement)
        yield
    finally:
        for name, original in reversed(list(originals.items())):
            if original is _MISSING:
                if hasattr(torch, name):
                    delattr(torch, name)
            else:
                setattr(torch, name, original)


def old_autograd_unsafe_predicate(original):
    """Model the PyTorch 1.7 is*inf failure for grad-bearing tensors."""
    def predicate(value):
        if isinstance(value, torch.Tensor) and value.requires_grad:
            raise RuntimeError(
                "simulated PyTorch 1.7 {} autograd incompatibility".format(
                    getattr(original, "__name__", "is*inf")
                )
            )
        return original(value)

    return predicate


DIRECT_CASES = (
    (
        "identical",
        [100.0, 80.0, 40.0, 10.0, 15.0],
        [100.0, 80.0, 40.0, 10.0, 15.0],
    ),
    (
        "center_offset",
        [106.0, 77.0, 40.0, 10.0, 15.0],
        [100.0, 80.0, 40.0, 10.0, 15.0],
    ),
    (
        "width_height_scale_difference",
        [100.0, 80.0, 52.0, 8.0, 15.0],
        [100.0, 80.0, 40.0, 10.0, 15.0],
    ),
    (
        "positive_angle_difference",
        [100.0, 80.0, 40.0, 10.0, 45.0],
        [100.0, 80.0, 40.0, 10.0, 15.0],
    ),
    (
        "negative_angle_difference",
        [100.0, 80.0, 40.0, 10.0, -25.0],
        [100.0, 80.0, 40.0, 10.0, 15.0],
    ),
    (
        "non_square_geometry",
        [106.0, 77.0, 500.0, 3.0, -89.0],
        [100.0, 80.0, 40.0, 10.0, 15.0],
    ),
    (
        "normal_image_scale_geometry",
        [106.0, 77.0, 256.0, 256.0, 0.0],
        [100.0, 80.0, 40.0, 10.0, 15.0],
    ),
    (
        "tiny_prediction_width",
        [100.0, 80.0, 9.313225746154785e-10, 10.0, 15.0],
        [100.0, 80.0, 40.0, 10.0, 15.0],
    ),
    (
        "tiny_prediction_width_and_height",
        [100.0, 80.0, 2.0 ** -30, 2.0 ** -30, 15.0],
        [100.0, 80.0, 40.0, 10.0, 15.0],
    ),
    (
        "tiny_target_width",
        [100.0, 80.0, 40.0, 10.0, 15.0],
        [100.0, 80.0, 2.0 ** -61, 10.0, 15.0],
    ),
    (
        "tiny_target_height",
        [100.0, 80.0, 40.0, 10.0, 15.0],
        [100.0, 80.0, 40.0, 2.0 ** -61, 15.0],
    ),
    (
        "decoded_zero_prediction_width",
        [100.0, 80.0, 0.0, 10.0, 15.0],
        [100.0, 80.0, 40.0, 10.0, 15.0],
    ),
    (
        "extreme_finite_aspect_angle_center",
        [106.0, 77.0, 2.0 ** -30, 512.0, -25.0],
        [100.0, 80.0, 512.0, 2.0 ** -30, 15.0],
    ),
    (
        "large_finite_dimensions",
        [100.0, 80.0, 1.0e20, 1.0e20, 15.0],
        [100.0, 80.0, 40.0, 10.0, 15.0],
    ),
)


class KLDPyTorch17CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kld = load_kld()

    def test_missing_scalar_aliases_support_actual_direct_paths(self):
        prediction = torch.tensor(
            [[106.0, 77.0, 40.0, 10.0, 15.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        target = torch.tensor(
            [[100.0, 80.0, 40.0, 10.0, 15.0]], dtype=torch.float32
        )
        prediction_before = prediction.detach().clone()
        target_before = target.clone()

        with temporary_torch_attributes(inf=_MISSING, nan=_MISSING):
            self.assertFalse(hasattr(torch, "inf"))
            self.assertFalse(hasattr(torch, "nan"))
            function_output = self.kld.kld_loss(prediction, target)
            module_output = self.kld.KLDloss().forward(prediction, target)
            (function_output.sum() + module_output.sum()).backward()

        self.assertEqual(tuple(function_output.shape), (1,))
        self.assertEqual(function_output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(function_output).all())
        self.assertTrue(torch.equal(function_output, module_output))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(torch.equal(prediction.detach(), prediction_before))
        self.assertTrue(torch.equal(target, target_before))
        self.assertTrue(hasattr(torch, "inf"))
        self.assertTrue(hasattr(torch, "nan"))

    def test_isneginf_old_autograd_behavior_is_not_required(self):
        prediction = torch.tensor(
            [[106.0, 77.0, 40.0, 10.0, 15.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        target = torch.tensor(
            [[100.0, 80.0, 40.0, 10.0, 15.0]], dtype=torch.float32
        )
        original = torch.isneginf
        with temporary_torch_attributes(
            isneginf=old_autograd_unsafe_predicate(original)
        ):
            output = self.kld.kld_loss(prediction, target)
            output.sum().backward()

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_isposinf_old_autograd_behavior_is_not_required(self):
        prediction = torch.tensor(
            [[106.0, 77.0, 40.0, 10.0, 15.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        target = torch.tensor(
            [[100.0, 80.0, 40.0, 10.0, 15.0]], dtype=torch.float32
        )
        original = torch.isposinf
        with temporary_torch_attributes(
            isposinf=old_autograd_unsafe_predicate(original)
        ):
            output = self.kld.KLDloss().forward(prediction, target)
            output.sum().backward()

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_direct_paths_preserve_reference_values_and_gradients(self):
        for name, prediction_values, target_values in DIRECT_CASES:
            with self.subTest(case=name):
                prediction = torch.tensor(
                    [prediction_values], dtype=torch.float64, requires_grad=True
                )
                target = torch.tensor([target_values], dtype=torch.float64)
                prediction_before = prediction.detach().clone()
                target_before = target.clone()

                function_output = self.kld.kld_loss(prediction, target)
                module_prediction = prediction.detach().clone().requires_grad_()
                module_output = self.kld.KLDloss().forward(module_prediction, target)
                function_output.sum().backward()
                module_output.sum().backward()

                self.assertEqual(tuple(function_output.shape), (1,))
                self.assertEqual(function_output.dtype, torch.float64)
                self.assertTrue(torch.isfinite(function_output).all())
                self.assertTrue(torch.isfinite(module_output).all())
                self.assertTrue(
                    torch.allclose(function_output, module_output, rtol=0.0, atol=0.0)
                )
                self.assertTrue(torch.isfinite(prediction.grad).all())
                self.assertTrue(torch.isfinite(module_prediction.grad).all())
                self.assertTrue(
                    torch.allclose(
                        prediction.grad,
                        module_prediction.grad,
                        rtol=0.0,
                        atol=0.0,
                    )
                )
                self.assertTrue(torch.equal(prediction.detach(), prediction_before))
                self.assertTrue(torch.equal(target, target_before))

                if name != "decoded_zero_prediction_width":
                    reference_prediction = torch.tensor(
                        [prediction_values], dtype=torch.float64, requires_grad=True
                    )
                    reference_target = torch.tensor(
                        [target_values], dtype=torch.float64
                    )
                    reference = legacy_kld_loss(
                        reference_prediction, reference_target
                    )
                    reference.sum().backward()
                    self.assertTrue(torch.isfinite(reference).all())
                    self.assertTrue(
                        torch.allclose(
                            function_output, reference, rtol=3e-6, atol=3e-7
                        ),
                        (name, function_output, reference),
                    )
                    self.assertTrue(
                        torch.allclose(
                            prediction.grad,
                            reference_prediction.grad,
                            rtol=3e-5,
                            atol=1e-7,
                        ),
                        (name, prediction.grad, reference_prediction.grad),
                    )

    def test_pairwise_assignment_path_preserves_reference_order_and_contract(self):
        predictions = torch.tensor(
            [
                [100.0, 80.0, 40.0, 10.0, 15.0],
                [106.0, 77.0, 40.0, 10.0, 15.0],
                [100.0, 80.0, 52.0, 8.0, 15.0],
                [100.0, 80.0, 40.0, 10.0, -25.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.tensor(
            [
                [100.0, 80.0, 40.0, 10.0, 15.0],
                [500.0, 400.0, 200.0, 100.0, 15.0],
                [320.0, 200.0, 500.0, 3.0, 89.0],
            ],
            dtype=torch.float32,
        )
        predictions_before = predictions.detach().clone()
        targets_before = targets.clone()

        matrix = self.kld.compute_kld_loss(targets, predictions)
        expected = torch.stack(
            [
                legacy_kld_loss(predictions.detach(), target.repeat(4, 1))
                for target in targets
            ]
        )

        self.assertEqual(tuple(matrix.shape), (3, 4))
        self.assertEqual(matrix.dtype, torch.float32)
        self.assertTrue(torch.isfinite(matrix).all())
        self.assertFalse(matrix.requires_grad)
        self.assertTrue(
            torch.allclose(matrix, expected, rtol=3e-6, atol=3e-7),
            (matrix, expected),
        )
        self.assertTrue(torch.equal(predictions.detach(), predictions_before))
        self.assertTrue(torch.equal(targets, targets_before))


if __name__ == "__main__":
    unittest.main()
