"""Regression tests for the KLD prediction/target argument contract."""

import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_kld():
    path = ROOT / "yolox" / "models" / "KLD_loss.py"
    spec = importlib.util.spec_from_file_location("maintained_kld_loss_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KLDPredictionTargetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kld = load_kld()
        cls.loss = cls.kld.KLDloss()

    def evaluate(self, prediction_values, target_values):
        prediction = torch.tensor(
            [prediction_values], dtype=torch.float64, requires_grad=True
        )
        target = torch.tensor([target_values], dtype=torch.float64)
        forward = self.loss(prediction, target).sum()
        forward.backward()
        forward_gradient = prediction.grad.detach().clone()

        reversed_prediction = torch.tensor(
            [prediction_values], dtype=torch.float64, requires_grad=True
        )
        reversed_target = torch.tensor([target_values], dtype=torch.float64)
        reverse = self.loss(reversed_target, reversed_prediction).sum()
        reverse.backward()
        reverse_gradient_wrt_prediction = reversed_prediction.grad.detach().clone()
        return {
            "forward": float(forward.detach()),
            "reverse": float(reverse.detach()),
            "forward_gradient": forward_gradient,
            "reverse_gradient_wrt_prediction": reverse_gradient_wrt_prediction,
        }

    def test_identical_center_width_height_and_angle_boxes_have_zero_loss(self):
        values = [100.0, 80.0, 40.0, 10.0, 15.0]
        result = self.evaluate(values, values)
        self.assertAlmostEqual(result["forward"], 0.0, places=12)
        self.assertTrue(torch.isfinite(result["forward_gradient"]).all())

    def test_center_offset_produces_a_finite_prediction_gradient(self):
        result = self.evaluate(
            [106.0, 77.0, 40.0, 10.0, 15.0],
            [100.0, 80.0, 40.0, 10.0, 15.0],
        )
        gradient = result["forward_gradient"]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient[0, 0]), 0.0)
        self.assertLess(float(gradient[0, 1]), 0.0)

    def test_width_height_mismatch_is_asymmetric(self):
        result = self.evaluate(
            [100.0, 80.0, 52.0, 8.0, 15.0],
            [100.0, 80.0, 40.0, 10.0, 15.0],
        )
        self.assertGreater(abs(result["forward"] - result["reverse"]), 1e-8)
        self.assertGreater(
            float(
                torch.max(
                    torch.abs(
                        result["forward_gradient"]
                        - result["reverse_gradient_wrt_prediction"]
                    )
                )
            ),
            1e-8,
        )

    def test_angle_mismatch_changes_prediction_gradient(self):
        result = self.evaluate(
            [100.0, 80.0, 40.0, 10.0, -25.0],
            [100.0, 80.0, 40.0, 10.0, 15.0],
        )
        self.assertTrue(torch.isfinite(result["forward_gradient"]).all())
        self.assertGreater(
            float(
                torch.max(
                    torch.abs(
                        result["forward_gradient"]
                        - result["reverse_gradient_wrt_prediction"]
                    )
                )
            ),
            1e-8,
        )

    def test_training_head_passes_prediction_before_target(self):
        source = (ROOT / "yolox" / "models" / "yolo_head_obb_kld.py").read_text()
        expected = (
            "self.iou_loss(bbox_preds_with_angle.view(-1, 5)[fg_masks], "
            "reg_targets_with_angle)"
        )
        self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
