"""Regression coverage for angle-aware OBB candidate gating."""

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
        "yolox.models.yolo_head_obb_kld_assignment_test", path
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


class OBBAssignmentGeometryTests(unittest.TestCase):
    def test_empty_targets_have_finite_loss_and_objectness_gradient(self):
        head_class = load_head()
        head = head_class.__new__(head_class)
        torch.nn.Module.__init__(head)
        head.use_l1 = False
        head.num_classes = 1
        head.iou_loss = lambda pred, target: pred.sum(dim=1) * 0.0
        head.bcewithlog_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        outputs = torch.randn(2, 3, 7, requires_grad=True)
        labels = torch.zeros(2, 4, 6)
        x_shifts = [torch.zeros(1, 3)]
        y_shifts = [torch.zeros(1, 3)]
        expanded_strides = [torch.full((1, 3), 8.0)]

        losses = head.get_losses(
            None,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            [],
            outputs.dtype,
        )
        losses[0].backward()

        self.assertTrue(torch.isfinite(losses[0]))
        self.assertTrue(torch.isfinite(outputs.grad).all())
        self.assertGreater(float(outputs.grad[:, :, 5].abs().sum()), 0.0)

    def test_short_side_rejects_anchor_outside_rotated_card_strip(self):
        """A point in the center-radius square can still miss a thin OBB."""
        head_class = load_head()
        head = head_class.__new__(head_class)

        # Card-like strip: its 45-degree short-side boundary is only 20 px
        # from the center.  The stride-8 anchor is 16 px in both image axes,
        # which is inside the old axis-aligned center square but outside the
        # rotated strip.
        gt_box = torch.tensor([[320.0, 240.0, 250.0, 40.0]])
        gt_angle = torch.tensor([45.0])
        expanded_strides = torch.tensor([[8.0]])
        x_shifts = torch.tensor([[37.5]])  # center = (304, 256)
        y_shifts = torch.tensor([[31.5]])

        try:
            # The expected OBB-aware contract includes the target angle.
            _, is_in_boxes_and_center = head.get_in_boxes_info(
                gt_box,
                gt_angle,
                expanded_strides,
                x_shifts,
                y_shifts,
                total_num_anchors=1,
                num_gt=1,
            )
        except TypeError:
            # Pre-fix compatibility: the maintained head has no angle
            # argument, so call its existing angle-blind implementation to
            # expose the behavioral failure below.
            _, is_in_boxes_and_center = head.get_in_boxes_info(
                gt_box,
                expanded_strides,
                x_shifts,
                y_shifts,
                total_num_anchors=1,
                num_gt=1,
            )

        self.assertFalse(bool(is_in_boxes_and_center[0, 0]))

    def test_card_aspect_ratios_and_angle_boundaries_match_obb_reference(self):
        head_class = load_head()
        head = head_class.__new__(head_class)
        angles = (0.0, 1.0, -1.0, 10.0, -10.0, 44.0, 45.0, 46.0,
                  -44.0, -45.0, -46.0, 89.0, -89.0)
        dimensions = ((240.0, 150.0), (300.0, 180.0),
                      (300.0, 60.0), (250.0, 40.0))

        for width, height in dimensions:
            for angle in angles:
                with self.subTest(width=width, height=height, angle=angle):
                    radians = math.radians(angle)
                    cos_angle, sin_angle = math.cos(radians), math.sin(radians)
                    local_offsets = torch.tensor([
                        [0.0, 0.0],
                        [0.0, min(0.25 * height, 16.0)],
                        [0.0, height / 2.0 + 1.0],
                        [width / 2.0 + 1.0, 0.0],
                    ])
                    world_offsets = torch.stack((
                        local_offsets[:, 0] * cos_angle - local_offsets[:, 1] * sin_angle,
                        local_offsets[:, 0] * sin_angle + local_offsets[:, 1] * cos_angle,
                    ), dim=1)
                    centers = torch.tensor([320.0, 240.0]) + world_offsets
                    shifts = centers / 8.0 - 0.5
                    gt_box = torch.tensor([[320.0, 240.0, width, height]])
                    gt_angle = torch.tensor([angle])
                    expanded_strides = torch.full((1, len(centers)), 8.0)
                    x_shifts = shifts[:, 0].unsqueeze(0)
                    y_shifts = shifts[:, 1].unsqueeze(0)

                    fg_mask, selected = head.get_in_boxes_info(
                        gt_box,
                        gt_angle,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        total_num_anchors=len(centers),
                        num_gt=1,
                    )
                    actual = torch.zeros(len(centers), dtype=torch.bool)
                    actual[fg_mask] = selected[0]
                    expected = (
                        (local_offsets[:, 0].abs() < width / 2.0)
                        & (local_offsets[:, 1].abs() < height / 2.0)
                        & (world_offsets[:, 0].abs() < 20.0)
                        & (world_offsets[:, 1].abs() < 20.0)
                    )
                    self.assertTrue(torch.equal(actual, expected))

    def test_width_height_swap_and_periodic_angle_have_same_gate(self):
        head_class = load_head()
        head = head_class.__new__(head_class)
        centers = torch.tensor([
            [320.0, 240.0], [320.0, 250.0], [330.0, 240.0],
            [350.0, 250.0], [300.0, 220.0],
        ])
        expanded_strides = torch.full((1, len(centers)), 8.0)
        shifts = centers / 8.0 - 0.5

        def gate(width, height, angle):
            return head.get_in_boxes_info(
                torch.tensor([[320.0, 240.0, width, height]]),
                torch.tensor([angle]),
                expanded_strides,
                shifts[:, 0].unsqueeze(0),
                shifts[:, 1].unsqueeze(0),
                total_num_anchors=len(centers),
                num_gt=1,
            )

        canonical = gate(240.0, 40.0, 10.0)
        swapped = gate(40.0, 240.0, 100.0)
        periodic = gate(240.0, 40.0, 190.0)
        for expected, equivalent in ((canonical, swapped), (canonical, periodic)):
            self.assertTrue(torch.equal(expected[0], equivalent[0]))
            self.assertTrue(torch.equal(expected[1], equivalent[1]))


if __name__ == "__main__":
    unittest.main()
