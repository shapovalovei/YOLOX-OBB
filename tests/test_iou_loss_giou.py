"""Regression tests for the optional generic GIoU loss."""

import unittest

import torch

from yolox.models.losses import IOUloss


FORWARD_CASES = (
    ("identical", [0.0, 0.0, 4.0, 2.0], [0.0, 0.0, 4.0, 2.0]),
    ("partial_overlap", [0.0, 0.0, 4.0, 2.0], [0.7, 0.3, 3.0, 1.2]),
    ("containment", [0.0, 0.0, 6.0, 4.0], [0.4, 0.3, 2.0, 1.2]),
    ("edge_contact", [0.0, 0.0, 2.0, 2.0], [2.0, 0.2, 2.0, 1.2]),
    ("fully_disjoint", [-1.0, -1.0, 2.0, 2.0], [4.0, 3.0, 1.0, 1.0]),
    ("unequal_size", [1.0, -0.5, 7.0, 1.4], [0.2, 0.3, 1.1, 3.2]),
    ("strong_aspect_ratio", [0.0, 0.0, 12.0, 0.6], [0.3, 0.2, 0.8, 8.0]),
)


def to_xyxy(box):
    cx, cy, width, height = box
    return (
        cx - width / 2.0,
        cy - height / 2.0,
        cx + width / 2.0,
        cy + height / 2.0,
    )


def giou_oracle(prediction, target):
    """Compute scalar GIoU from the definition, independently of IOUloss."""
    pred = to_xyxy(prediction)
    truth = to_xyxy(target)
    intersection_width = max(0.0, min(pred[2], truth[2]) - max(pred[0], truth[0]))
    intersection_height = max(0.0, min(pred[3], truth[3]) - max(pred[1], truth[1]))
    intersection = intersection_width * intersection_height
    area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
    area_truth = (truth[2] - truth[0]) * (truth[3] - truth[1])
    union = area_pred + area_truth - intersection
    enclosing_width = max(pred[2], truth[2]) - min(pred[0], truth[0])
    enclosing_height = max(pred[3], truth[3]) - min(pred[1], truth[1])
    enclosing = enclosing_width * enclosing_height
    iou = intersection / (union + 1e-16)
    giou = iou - (enclosing - union) / max(enclosing, 1e-16)
    return 1.0 - min(max(giou, -1.0), 1.0)


def iou_loss_oracle(prediction, target):
    pred = to_xyxy(prediction)
    truth = to_xyxy(target)
    intersection_width = max(0.0, min(pred[2], truth[2]) - max(pred[0], truth[0]))
    intersection_height = max(0.0, min(pred[3], truth[3]) - max(pred[1], truth[1]))
    intersection = intersection_width * intersection_height
    area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
    area_truth = (truth[2] - truth[0]) * (truth[3] - truth[1])
    union = area_pred + area_truth - intersection
    iou = intersection / (union + 1e-16)
    return 1.0 - iou ** 2


def finite_difference_gradient(prediction, target, step=1e-5):
    gradient = []
    for index in range(4):
        plus = list(prediction)
        minus = list(prediction)
        plus[index] += step
        minus[index] -= step
        gradient.append(
            (giou_oracle(plus, target) - giou_oracle(minus, target)) / (2.0 * step)
        )
    return gradient


class GIoULossTests(unittest.TestCase):
    def evaluate(self, prediction, target, reduction="none", loss_type="giou", dtype=torch.float64):
        prediction_tensor = torch.tensor([prediction], dtype=dtype, requires_grad=True)
        target_tensor = torch.tensor([target], dtype=dtype)
        loss = IOUloss(reduction=reduction, loss_type=loss_type)
        return loss(prediction_tensor, target_tensor), prediction_tensor

    def test_forward_matches_independent_oracle_for_required_matrix(self):
        for name, prediction, target in FORWARD_CASES:
            with self.subTest(case=name, dtype="float64"):
                observed, _ = self.evaluate(prediction, target)
                expected = giou_oracle(prediction, target)
                self.assertTrue(
                    torch.allclose(
                        observed,
                        torch.tensor([expected], dtype=torch.float64),
                        rtol=1e-12,
                        atol=1e-12,
                    ),
                    (name, observed, expected),
                )
            with self.subTest(case=name, dtype="float32"):
                observed, _ = self.evaluate(
                    prediction, target, dtype=torch.float32
                )
                expected = giou_oracle(prediction, target)
                self.assertTrue(
                    torch.allclose(
                        observed,
                        torch.tensor([expected], dtype=torch.float32),
                        rtol=1e-6,
                        atol=1e-6,
                    ),
                    (name, observed, expected),
                )

    def test_batched_mixed_inputs_match_independent_oracle(self):
        cases = FORWARD_CASES[1:]
        predictions = [prediction for _, prediction, _ in cases]
        targets = [target for _, _, target in cases]
        prediction_tensor = torch.tensor(predictions, dtype=torch.float64)
        target_tensor = torch.tensor(targets, dtype=torch.float64)
        observed = IOUloss(reduction="none", loss_type="giou")(
            prediction_tensor, target_tensor
        )
        expected = torch.tensor(
            [giou_oracle(prediction, target) for prediction, target in zip(predictions, targets)],
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(observed, expected, rtol=1e-12, atol=1e-12))

    def test_reduction_semantics_match_none_output(self):
        cases = FORWARD_CASES[1:]
        predictions = [prediction for _, prediction, _ in cases]
        targets = [target for _, _, target in cases]
        prediction_tensor = torch.tensor(predictions, dtype=torch.float64)
        target_tensor = torch.tensor(targets, dtype=torch.float64)
        none = IOUloss(reduction="none", loss_type="giou")(
            prediction_tensor, target_tensor
        )
        mean = IOUloss(reduction="mean", loss_type="giou")(
            prediction_tensor, target_tensor
        )
        total = IOUloss(reduction="sum", loss_type="giou")(
            prediction_tensor, target_tensor
        )
        self.assertTrue(torch.allclose(mean, none.mean(), rtol=0.0, atol=0.0))
        self.assertTrue(torch.allclose(total, none.sum(), rtol=0.0, atol=0.0))

    def test_default_iou_matches_maintained_convention(self):
        for name, prediction, target in FORWARD_CASES:
            with self.subTest(case=name):
                observed, _ = self.evaluate(
                    prediction, target, loss_type="iou"
                )
                expected = iou_loss_oracle(prediction, target)
                self.assertTrue(
                    torch.allclose(
                        observed,
                        torch.tensor([expected], dtype=torch.float64),
                        rtol=1e-12,
                        atol=1e-12,
                    ),
                    (name, observed, expected),
                )

    def test_prediction_gradients_match_independent_finite_difference(self):
        gradient_cases = (
            ("partial", [0.15, -0.2, 4.2, 2.3], [0.7, 0.4, 3.0, 1.2]),
            ("containment", [0.2, -0.1, 6.4, 4.4], [0.4, 0.3, 2.0, 1.2]),
            ("aspect_ratio", [0.0, 0.0, 12.0, 0.8], [0.3, 0.2, 0.8, 8.0]),
        )
        for name, prediction, target in gradient_cases:
            with self.subTest(case=name):
                observed, prediction_tensor = self.evaluate(prediction, target)
                observed.backward()
                expected = torch.tensor(
                    finite_difference_gradient(prediction, target), dtype=torch.float64
                )
                self.assertTrue(
                    torch.allclose(
                        prediction_tensor.grad[0], expected, rtol=3e-5, atol=3e-6
                    ),
                    (name, prediction_tensor.grad[0], expected),
                )
                self.assertTrue(torch.isfinite(prediction_tensor.grad).all())


if __name__ == "__main__":
    unittest.main()
