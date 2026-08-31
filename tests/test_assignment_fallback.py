"""Executable regression coverage for the assignment OOM fallback."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_head_module():
    """Load only the head so the tests do not require native polyiou."""
    path = ROOT / "yolox" / "models" / "yolo_head_obb_kld.py"
    spec = importlib.util.spec_from_file_location(
        "yolox.models.yolo_head_obb_kld_assignment_fallback_test", path
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
    return module


class AssignmentPlan:
    """Record assignment modes and return or raise deterministic actions."""

    def __init__(self, actions):
        self.actions = iter(actions)
        self.modes = []

    def __call__(self, *args, **kwargs):
        mode = kwargs.get("mode", args[16] if len(args) > 16 else "gpu")
        self.modes.append(mode)
        action = next(self.actions)
        if isinstance(action, BaseException):
            raise action
        return action


class AssignmentFallbackTests(unittest.TestCase):
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
        head_class = self.module.YOLOXHeadOBB_KLD
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
        outputs = torch.zeros(1, 3, 7)
        labels = torch.tensor(
            [[[0.0, 50.0, 50.0, 20.0, 20.0, 0.0], [0.0] * 6]]
        )
        self.last_plan = plan
        with mock.patch.object(self.module.logger, "error") as log_error, \
                mock.patch.object(
                    self.module.torch.cuda, "empty_cache"
                ) as empty_cache:
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
        self.assertIn("CUDA", log_error.call_args.args[0])
        empty_cache.assert_has_calls([mock.call(), mock.call()])
        self.assertEqual(empty_cache.call_count, 2)

    def test_legacy_cuda_oom_messages_retry_on_cpu(self):
        for message in (
            "CUDA out of memory. Tried to allocate 2.00 MiB",
            "CUDA error: out of memory",
        ):
            with self.subTest(message=message):
                error = RuntimeError(message)
                self.assertIs(type(error), RuntimeError)
                result, plan, log_error, empty_cache = self.run_losses(
                    [error, self.make_assignment_result()]
                )

                self.assertTrue(torch.isfinite(result[0]))
                self.assertEqual(plan.modes, ["gpu", "cpu"])
                log_error.assert_called_once()
                empty_cache.assert_has_calls([mock.call(), mock.call()])
                self.assertEqual(empty_cache.call_count, 2)

    def test_legacy_classifier_does_not_require_modern_exception_attribute(self):
        error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 MiB")
        with mock.patch.object(
            self.module.torch.cuda, "OutOfMemoryError", None, create=True
        ):
            self.assertTrue(self.module._is_cuda_oom(error))

    def test_non_oom_runtime_error_is_reraised_without_retry(self):
        original = RuntimeError("synthetic assignment geometry failure")
        with self.assertRaises(RuntimeError) as raised:
            self.run_losses(
                [original, self.make_assignment_result()]
            )

        self.assertIs(raised.exception, original)
        self.assertEqual(self.last_plan.modes, ["gpu"])
        self.last_log_error.assert_not_called()
        self.last_empty_cache.assert_not_called()

    def test_non_oom_does_not_hide_error_when_cpu_would_fail_differently(self):
        original = RuntimeError("ORIGINAL_SENTINEL")
        cpu_retry = RuntimeError("CPU_RETRY_SENTINEL")
        with self.assertRaises(RuntimeError) as raised:
            self.run_losses(
                [original, cpu_retry]
            )

        self.assertIs(raised.exception, original)
        self.assertEqual(str(raised.exception), "ORIGINAL_SENTINEL")
        self.assertEqual(self.last_plan.modes, ["gpu"])
        self.last_log_error.assert_not_called()
        self.last_empty_cache.assert_not_called()

    def test_cpu_mps_and_memory_related_errors_do_not_retry(self):
        messages = (
            "DefaultCPUAllocator: cannot allocate memory",
            "MPS backend out of memory",
            "memory format mismatch",
            "memory-related sentinel error",
        )
        for message in messages:
            with self.subTest(message=message):
                original = RuntimeError(message)
                with self.assertRaises(RuntimeError) as raised:
                    self.run_losses(
                        [original, RuntimeError("CPU_RETRY_SENTINEL")]
                    )

                self.assertIs(raised.exception, original)
                self.assertEqual(self.last_plan.modes, ["gpu"])
                self.last_log_error.assert_not_called()
                self.last_empty_cache.assert_not_called()

    def test_cuda_oom_with_failed_cpu_retry_preserves_exception_context(self):
        cuda_oom_error = getattr(self.module.torch.cuda, "OutOfMemoryError", None)
        original = (
            cuda_oom_error("CUDA out of memory")
            if cuda_oom_error is not None
            else RuntimeError("CUDA out of memory. Tried to allocate 2.00 MiB")
        )
        cpu_retry = RuntimeError("CPU_RETRY_SENTINEL")
        with self.assertRaises(RuntimeError) as raised:
            self.run_losses(
                [original, cpu_retry]
            )

        self.assertIs(raised.exception, cpu_retry)
        self.assertIs(raised.exception.__context__, original)
        self.assertEqual(self.last_plan.modes, ["gpu", "cpu"])
        self.last_log_error.assert_called_once()
        self.last_empty_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
