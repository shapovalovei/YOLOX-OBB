"""Regression coverage for the generic HBB assignment OOM fallback."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_head_module():
    """Load the maintained HBB head without requiring native polyiou."""
    path = ROOT / "yolox" / "models" / "yolo_head.py"
    spec = importlib.util.spec_from_file_location(
        "yolox.models.yolo_head_hbb_assignment_fallback_test", path
    )
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
    utils_stub.bboxes_iou = lambda *args, **kwargs: None
    models_stub = types.ModuleType("yolox.models")
    models_stub.__path__ = [str(ROOT / "yolox" / "models")]
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


class AssignmentPlan:
    """Record HBB assignment modes and return or raise deterministic actions."""

    def __init__(self, actions):
        self.actions = iter(actions)
        self.modes = []

    def __call__(self, *args, **kwargs):
        mode = kwargs.get("mode", args[14] if len(args) > 14 else "gpu")
        self.modes.append(mode)
        action = next(self.actions)
        if isinstance(action, BaseException):
            raise action
        return action


class HBBAssignmentFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_head_module()

    def make_assignment_result(self):
        return (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([True, False, False]),
            torch.tensor([0.5]),
            torch.tensor([0], dtype=torch.long),
            1,
        )

    def make_head(self, plan):
        head_class = self.module.YOLOXHead
        head = head_class.__new__(head_class)
        torch.nn.Module.__init__(head)
        head.use_l1 = False
        head.num_classes = 1
        head.iou_loss = lambda pred, target: pred.sum(dim=1) * 0.0
        head.bcewithlog_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        head.get_assignments = plan
        return head

    def run_losses(self, actions):
        plan = AssignmentPlan(actions)
        head = self.make_head(plan)
        outputs = torch.zeros(1, 3, 6)
        labels = torch.tensor(
            [[[0.0, 50.0, 50.0, 20.0, 20.0], [0.0] * 5]]
        )
        with mock.patch.object(self.module.logger, "error") as log_error, \
                mock.patch.object(
                    self.module.torch.cuda, "empty_cache"
                ) as empty_cache:
            self.last_plan = plan
            self.last_log_error = log_error
            self.last_empty_cache = empty_cache
            result = head.get_losses(
                None,
                [torch.zeros(1, 3)],
                [torch.zeros(1, 3)],
                [torch.full((1, 3), 8.0)],
                labels,
                outputs,
                [],
                outputs.dtype,
            )
        return result, plan, log_error, empty_cache

    def test_ordinary_assignment_success_is_unchanged(self):
        result, plan, log_error, empty_cache = self.run_losses(
            [self.make_assignment_result()]
        )

        self.assertTrue(torch.isfinite(result[0]))
        self.assertEqual(plan.modes, ["gpu"])
        log_error.assert_not_called()
        empty_cache.assert_called_once_with()

    def test_non_oom_runtime_error_propagates_unchanged_without_fallback(self):
        original = RuntimeError("ORIGINAL_SENTINEL")
        with self.assertRaises(RuntimeError) as raised:
            self.run_losses([original, self.make_assignment_result()])

        self.assertIs(raised.exception, original)
        self.assertEqual(self.last_plan.modes, ["gpu"])
        self.last_log_error.assert_not_called()
        self.last_empty_cache.assert_not_called()

    def test_near_miss_runtime_error_does_not_retry(self):
        original = RuntimeError("CUDA allocator out of memory")
        with self.assertRaises(RuntimeError) as raised:
            self.run_losses([original, self.make_assignment_result()])

        self.assertIs(raised.exception, original)
        self.assertEqual(self.last_plan.modes, ["gpu"])
        self.last_log_error.assert_not_called()
        self.last_empty_cache.assert_not_called()

    def test_modern_cuda_oom_retries_on_cpu(self):
        cuda_oom_error = getattr(self.module.torch.cuda, "OutOfMemoryError", None)
        if cuda_oom_error is None:
            self.skipTest("torch.cuda.OutOfMemoryError is unavailable")

        result, plan, log_error, empty_cache = self.run_losses(
            [cuda_oom_error("CUDA out of memory"), self.make_assignment_result()]
        )

        self.assertTrue(torch.isfinite(result[0]))
        self.assertEqual(plan.modes, ["gpu", "cpu"])
        log_error.assert_called_once()
        self.assertIn("OOM RuntimeError", log_error.call_args.args[0])
        empty_cache.assert_has_calls([mock.call(), mock.call()])
        self.assertEqual(empty_cache.call_count, 2)

    def test_legacy_cuda_oom_messages_retry_on_cpu(self):
        for message in (
            "CUDA out of memory. Tried to allocate 2.00 MiB",
            "CUDA error: out of memory",
        ):
            with self.subTest(message=message):
                result, plan, log_error, empty_cache = self.run_losses(
                    [RuntimeError(message), self.make_assignment_result()]
                )

                self.assertTrue(torch.isfinite(result[0]))
                self.assertEqual(plan.modes, ["gpu", "cpu"])
                log_error.assert_called_once()
                self.assertIn("OOM RuntimeError", log_error.call_args.args[0])
                empty_cache.assert_has_calls([mock.call(), mock.call()])
                self.assertEqual(empty_cache.call_count, 2)


if __name__ == "__main__":
    unittest.main()
