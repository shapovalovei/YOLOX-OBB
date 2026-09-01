"""Regression coverage for FP16 logits in CPU assignment classification cost."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def _hbb_iou(bboxes_a, bboxes_b, xyxy=True):
    """Provide the maintained HBB IoU arithmetic without native imports."""
    if xyxy:
        raise AssertionError("the assignment path must use center-width boxes")
    top_left = torch.max(
        bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2,
        bboxes_b[:, :2] - bboxes_b[:, 2:] / 2,
    )
    bottom_right = torch.min(
        bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2,
        bboxes_b[:, :2] + bboxes_b[:, 2:] / 2,
    )
    intersection = torch.prod(bottom_right - top_left, dim=2)
    intersection = intersection * (top_left < bottom_right).type(top_left.type()).prod(dim=2)
    area_a = torch.prod(bboxes_a[:, 2:], dim=1)
    area_b = torch.prod(bboxes_b[:, 2:], dim=1)
    return intersection / (area_a[:, None] + area_b - intersection)


def _load_head(path, module_name, models_attributes, utils_attributes=None):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)

    module_names = (
        "yolox",
        "yolox.utils",
        "yolox.models",
        "yolox.models.losses",
        "yolox.models.network_blocks",
    )
    original_modules = {name: sys.modules.get(name) for name in module_names}
    yolox_stub = types.ModuleType("yolox")
    yolox_stub.__path__ = [str(ROOT / "yolox")]
    utils_stub = types.ModuleType("yolox.utils")
    for name, value in (utils_attributes or {}).items():
        setattr(utils_stub, name, value)
    models_stub = types.ModuleType("yolox.models")
    models_stub.__path__ = [str(ROOT / "yolox" / "models")]
    for name, value in models_attributes.items():
        setattr(models_stub, name, value)
    losses_stub = types.ModuleType("yolox.models.losses")
    losses_stub.IOUloss = object
    blocks_stub = types.ModuleType("yolox.models.network_blocks")
    blocks_stub.BaseConv = object
    blocks_stub.DWConv = object
    sys.modules.update(
        {
            "yolox": yolox_stub,
            "yolox.utils": utils_stub,
            "yolox.models": models_stub,
            "yolox.models.losses": losses_stub,
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
    return module


def _load_kld():
    path = ROOT / "yolox" / "models" / "KLD_loss.py"
    spec = importlib.util.spec_from_file_location(
        "yolox.models.fp16_cpu_assignment_kld_test", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FP16CPUAssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hbb_module = _load_head(
            ROOT / "yolox" / "models" / "yolo_head.py",
            "yolox.models.fp16_cpu_assignment_hbb_test",
            {"__path__": [str(ROOT / "yolox" / "models")]},
            {"bboxes_iou": _hbb_iou},
        )
        kld = _load_kld()
        cls.obb_module = _load_head(
            ROOT / "yolox" / "models" / "yolo_head_obb_kld.py",
            "yolox.models.fp16_cpu_assignment_obb_test",
            {
                "__path__": [str(ROOT / "yolox" / "models")],
                "compute_kld_loss": kld.compute_kld_loss,
                "KLDloss": kld.KLDloss,
            },
        )

    def make_fixture(self, dtype):
        centers = torch.tensor(
            [
                [36.0, 36.0],
                [40.0, 36.0],
                [44.0, 36.0],
                [36.0, 40.0],
                [40.0, 40.0],
                [44.0, 40.0],
                [36.0, 44.0],
                [40.0, 44.0],
                [44.0, 44.0],
            ]
        )
        bbox_preds = torch.tensor(
            [
                [38.0, 38.0, 38.0, 30.0],
                [40.0, 38.0, 40.0, 31.0],
                [42.0, 38.0, 39.0, 32.0],
                [38.0, 40.0, 41.0, 30.0],
                [40.0, 40.0, 40.0, 32.0],
                [42.0, 40.0, 39.0, 31.0],
                [38.0, 42.0, 40.0, 31.0],
                [40.0, 42.0, 39.0, 32.0],
                [42.0, 42.0, 41.0, 30.0],
            ]
        )
        cls_logits = torch.tensor(
            [
                [1.7, -0.8],
                [1.2, -0.2],
                [0.8, 0.1],
                [1.4, -0.5],
                [2.1, 0.4],
                [0.9, -0.1],
                [1.1, -0.6],
                [1.8, 0.3],
                [0.6, -0.4],
            ],
            dtype=dtype,
        ).unsqueeze(0)
        obj_logits = torch.tensor(
            [1.3, 0.6, -0.2, 1.0, 1.8, 0.2, 0.8, 1.1, -0.4],
            dtype=dtype,
        ).view(1, -1, 1)
        return {
            "gt_bboxes": torch.tensor([[40.0, 40.0, 40.0, 32.0]]),
            "gt_classes": torch.tensor([1.0]),
            "gt_angles": torch.tensor([30.0]),
            "bbox_preds": bbox_preds,
            "angle_preds": torch.tensor(
                [28.0, 29.0, 30.0, 30.0, 30.0, 31.0, 30.0, 31.0, 32.0]
            ),
            "expanded_strides": torch.full((1, 9), 8.0),
            "x_shifts": (centers[:, 0] / 8.0 - 0.5).unsqueeze(0),
            "y_shifts": (centers[:, 1] / 8.0 - 0.5).unsqueeze(0),
            "cls_preds": cls_logits,
            "obj_preds": obj_logits,
            "labels": torch.tensor(
                [[[1.0, 40.0, 40.0, 40.0, 32.0, 30.0], [0.0] * 6]]
            ),
        }

    def run_assignment(self, kind, dtype):
        module = self.hbb_module if kind == "hbb" else self.obb_module
        head_class = module.YOLOXHead if kind == "hbb" else module.YOLOXHeadOBB_KLD
        head = head_class.__new__(head_class)
        torch.nn.Module.__init__(head)
        head.num_classes = 2

        fixture = self.make_fixture(dtype)
        before = {name: value.clone() for name, value in fixture.items()}
        sigmoid_trace = []
        bce_trace = []
        original_sigmoid_ = torch.Tensor.sigmoid_
        original_bce = module.F.binary_cross_entropy

        def trace_sigmoid_(tensor):
            if tensor.device.type == "cpu":
                sigmoid_trace.append((tuple(tensor.shape), tensor.dtype, tensor.device.type))
            return original_sigmoid_(tensor)

        def trace_bce(input_tensor, target, *args, **kwargs):
            bce_trace.append((input_tensor.dtype, target.dtype, input_tensor.device.type))
            return original_bce(input_tensor, target, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor), \
                mock.patch.object(torch.Tensor, "sigmoid_", trace_sigmoid_), \
                mock.patch.object(module.F, "binary_cross_entropy", trace_bce):
            if kind == "hbb":
                result = head.get_assignments(
                    0,
                    1,
                    9,
                    fixture["gt_bboxes"],
                    fixture["gt_classes"],
                    fixture["bbox_preds"],
                    fixture["expanded_strides"],
                    fixture["x_shifts"],
                    fixture["y_shifts"],
                    fixture["cls_preds"],
                    fixture["bbox_preds"],
                    fixture["obj_preds"],
                    fixture["labels"],
                    None,
                    "cpu",
                )
            else:
                result = head.get_assignments(
                    0,
                    1,
                    9,
                    fixture["gt_bboxes"],
                    fixture["gt_classes"],
                    fixture["gt_angles"],
                    fixture["bbox_preds"],
                    fixture["angle_preds"],
                    fixture["expanded_strides"],
                    fixture["x_shifts"],
                    fixture["y_shifts"],
                    fixture["cls_preds"],
                    fixture["bbox_preds"],
                    fixture["obj_preds"],
                    fixture["labels"],
                    None,
                    "cpu",
                )

        for name, value in fixture.items():
            self.assertTrue(torch.equal(value, before[name]), name)
        return result, sigmoid_trace, bce_trace

    def assert_valid_assignment(self, result):
        gt_matched_classes, fg_mask, pred_ious, matched_gt_inds, num_fg = result
        self.assertGreater(num_fg, 0)
        self.assertEqual(int(fg_mask.sum()), num_fg)
        self.assertTrue(torch.equal(gt_matched_classes, torch.ones_like(gt_matched_classes)))
        self.assertTrue(torch.equal(matched_gt_inds, torch.zeros_like(matched_gt_inds)))
        self.assertTrue(torch.isfinite(gt_matched_classes).all())
        self.assertTrue(torch.isfinite(pred_ious).all())
        self.assertTrue(torch.isfinite(matched_gt_inds).all())

    def assert_same_assignment(self, left, right):
        for index in (0, 1, 3):
            self.assertTrue(torch.equal(left[index], right[index]), index)
        self.assertEqual(left[4], right[4])
        self.assertTrue(torch.allclose(left[2], right[2], rtol=1e-5, atol=1e-6))

    def run_real_fallback(self, kind):
        module = self.hbb_module if kind == "hbb" else self.obb_module
        head_class = module.YOLOXHead if kind == "hbb" else module.YOLOXHeadOBB_KLD
        head = head_class.__new__(head_class)
        torch.nn.Module.__init__(head)
        head.num_classes = 2
        head.use_l1 = False
        head.iou_loss = lambda pred, target: pred.sum(dim=1) * 0.0
        head.bcewithlog_loss = lambda input_tensor, target: (
            module.F.binary_cross_entropy_with_logits(
                input_tensor.float(), target.float(), reduction="none"
            )
        )

        fixture = self.make_fixture(torch.float16)
        bbox_preds = fixture["bbox_preds"].to(torch.float16)
        if kind == "hbb":
            outputs = torch.cat(
                (bbox_preds, fixture["obj_preds"][0], fixture["cls_preds"][0]),
                dim=1,
            ).unsqueeze(0)
            labels = fixture["labels"][..., :5]
            cls_index, obj_index = 9, 11
        else:
            outputs = torch.cat(
                (
                    bbox_preds,
                    fixture["angle_preds"].to(torch.float16).unsqueeze(1),
                    fixture["obj_preds"][0],
                    fixture["cls_preds"][0],
                ),
                dim=1,
            ).unsqueeze(0)
            labels = fixture["labels"]
            cls_index, obj_index = 11, 13

        modes = []
        retry_logits = []
        real_get_assignments = head_class.get_assignments.__get__(head)
        cuda_oom_error = getattr(module.torch.cuda, "OutOfMemoryError", None)
        first_error = (
            cuda_oom_error("CUDA out of memory")
            if cuda_oom_error is not None
            else RuntimeError("CUDA out of memory. Tried to allocate 2.00 MiB")
        )

        def assignment_with_first_oom(*args, **kwargs):
            mode = kwargs.get("mode")
            if mode is None and args and args[-1] in ("cpu", "gpu"):
                mode = args[-1]
            if mode is None:
                mode = "gpu"
            modes.append(mode)
            if mode == "gpu":
                raise first_error
            retry_logits.append((args[cls_index], args[obj_index]))
            return real_get_assignments(*args, **kwargs)

        head.get_assignments = assignment_with_first_oom
        with mock.patch.object(module.logger, "error") as log_error, \
                mock.patch.object(module.torch.cuda, "empty_cache") as empty_cache, \
                mock.patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor):
            result = head.get_losses(
                None,
                [fixture["x_shifts"]],
                [fixture["y_shifts"]],
                [fixture["expanded_strides"]],
                labels,
                outputs,
                [],
                outputs.dtype,
            )

        self.assertTrue(torch.isfinite(result[0]))
        self.assertEqual(modes, ["gpu", "cpu"])
        self.assertEqual(len(retry_logits), 1)
        self.assertEqual(retry_logits[0][0].dtype, torch.float16)
        self.assertEqual(retry_logits[0][1].dtype, torch.float16)
        log_error.assert_called_once()
        self.assertEqual(empty_cache.call_count, 2)

    def test_accepted_cuda_oom_retries_real_fp16_cpu_assignment(self):
        self.run_real_fallback("hbb")
        self.run_real_fallback("obb")

    def test_hbb_cpu_assignment_promotes_fp16_objectness_before_sigmoid(self):
        self.assert_dtype_invariant("hbb")

    def test_obb_cpu_assignment_promotes_fp16_objectness_before_sigmoid(self):
        self.assert_dtype_invariant("obb")

    def assert_dtype_invariant(self, kind):
        fp32_result, fp32_sigmoid, fp32_bce = self.run_assignment(kind, torch.float32)
        fp16_result, fp16_sigmoid, fp16_bce = self.run_assignment(kind, torch.float16)
        repeat_result, repeat_sigmoid, repeat_bce = self.run_assignment(kind, torch.float16)

        for result in (fp32_result, fp16_result, repeat_result):
            self.assert_valid_assignment(result)
        self.assert_same_assignment(fp32_result, fp16_result)
        self.assert_same_assignment(fp16_result, repeat_result)

        for trace in (fp32_sigmoid, fp16_sigmoid, repeat_sigmoid):
            self.assertEqual(len(trace), 2)
            self.assertEqual([shape[-1] for shape, _, _ in trace], [2, 1])
            self.assertEqual([shape[1] for shape, _, _ in trace], [9, 9])
            self.assertTrue(all(device == "cpu" for _, _, device in trace))
            self.assertTrue(
                all(dtype is torch.float32 for _, dtype, _ in trace), trace
            )
        for trace in (fp32_bce, fp16_bce, repeat_bce):
            self.assertEqual(trace, [(torch.float32, torch.float32, "cpu")])


if __name__ == "__main__":
    unittest.main()
