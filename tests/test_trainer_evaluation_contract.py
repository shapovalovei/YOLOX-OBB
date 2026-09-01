"""Regression coverage for trainer evaluation and checkpoint lifecycle contracts."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def _load_trainer():
    yolox_stub = types.ModuleType("yolox")
    yolox_stub.__path__ = [str(ROOT / "yolox")]
    data_stub = types.ModuleType("yolox.data")
    data_stub.DataPrefetcher = object
    utils_stub = types.ModuleType("yolox.utils")
    utils_stub.MeterBuffer = object
    utils_stub.ModelEMA = object
    utils_stub.all_reduce_norm = lambda *args: None
    utils_stub.get_async_norm_states = lambda *args: {}
    utils_stub.get_local_rank = lambda: 0
    utils_stub.get_model_info = lambda *args: ""
    utils_stub.get_rank = lambda: 0
    utils_stub.get_world_size = lambda: 1
    utils_stub.gpu_mem_usage = lambda: 0
    utils_stub.is_parallel = lambda model: False
    utils_stub.load_ckpt = lambda model, ckpt: model
    utils_stub.occupy_mem = lambda: None
    utils_stub.save_checkpoint = lambda *args: None
    utils_stub.setup_logger = lambda *args, **kwargs: None
    utils_stub.synchronize = lambda: None
    apex_stub = types.ModuleType("apex")
    apex_stub.amp = types.SimpleNamespace()
    apex_amp_stub = types.ModuleType("apex.amp")
    tensorboard_stub = types.ModuleType("torch.utils.tensorboard")
    tensorboard_stub.SummaryWriter = object
    loguru_stub = types.ModuleType("loguru")
    loguru_stub.logger = _Logger()
    replacements = {
        "yolox": yolox_stub,
        "yolox.data": data_stub,
        "yolox.utils": utils_stub,
        "apex": apex_stub,
        "apex.amp": apex_amp_stub,
        "torch.utils.tensorboard": tensorboard_stub,
        "loguru": loguru_stub,
    }
    original_modules = {
        module_name: sys.modules.get(module_name) for module_name in replacements
    }
    sys.modules.update(replacements)
    try:
        path = ROOT / "yolox" / "core" / "trainer.py"
        spec = importlib.util.spec_from_file_location(
            "trainer_evaluation_contract_test_module", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load {}".format(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for module_name, original in original_modules.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original


class _Model:
    def __init__(self):
        self.train_calls = 0

    def train(self):
        self.train_calls += 1
        return self

class _TensorBoard:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, value, step))


def _trainer(module, evaluation_result):
    trainer = object.__new__(module.Trainer)
    trainer.use_model_ema = False
    trainer.model = _Model()
    trainer.rank = 0
    trainer.epoch = 2
    trainer.evaluator = object()
    trainer.is_distributed = False
    trainer.tblogger = _TensorBoard()
    trainer.best_ap = 0.5
    trainer.best_ap_available = False
    trainer.save_ckpt = mock.Mock()
    trainer.exp = SimpleNamespace(
        eval=lambda model, evaluator, distributed: evaluation_result
    )
    return trainer


def _lifecycle_trainer(module, exp, epoch):
    trainer = object.__new__(module.Trainer)
    trainer.use_model_ema = False
    trainer.model = _Model()
    trainer.rank = 0
    trainer.epoch = epoch
    trainer.evaluator = object()
    trainer.is_distributed = False
    trainer.tblogger = _TensorBoard()
    trainer.best_ap = 0.5
    trainer.best_ap_available = False
    trainer.save_ckpt = mock.Mock()
    trainer.exp = exp
    return trainer


def _hbb_exp(eval_interval):
    from yolox.exp.yolox_base import Exp

    exp = Exp()
    exp.eval_interval = eval_interval
    exp.eval = mock.Mock(return_value=(0.75, 0.8, "numeric timing"))
    return exp


class TrainerEvaluationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer_module = _load_trainer()

    def test_numeric_evaluator_behavior_is_unchanged(self):
        trainer = _trainer(self.trainer_module, (0.75, 0.8, "numeric timing"))

        trainer.evaluate_and_save_model()

        self.assertEqual(
            trainer.tblogger.scalars,
            [
                ("val/COCOAP50", 0.8, 3),
                ("val/COCOAP50_95", 0.75, 3),
            ],
        )
        trainer.save_ckpt.assert_called_once_with("last_epoch", True)
        self.assertEqual(trainer.best_ap, 0.75)
        self.assertTrue(trainer.best_ap_available)

    def test_writer_only_evaluator_saves_last_epoch_without_fake_metrics(self):
        trainer = _trainer(self.trainer_module, (None, None, "external timing"))
        self.trainer_module.logger.messages.clear()

        trainer.evaluate_and_save_model()
        trainer.after_train()

        self.assertEqual(trainer.tblogger.scalars, [])
        trainer.save_ckpt.assert_called_once_with("last_epoch", False)
        self.assertEqual(trainer.best_ap, 0.5)
        self.assertFalse(trainer.best_ap_available)
        self.assertEqual(trainer.model.train_calls, 1)
        self.assertTrue(
            any(
                "internal AP was not computed" in message
                for message in self.trainer_module.logger.messages
            )
        )


class TrainerCheckpointLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer_module = _load_trainer()

    def test_missing_save_interval_keeps_latest_and_skips_periodic_save(self):
        exp = _hbb_exp(eval_interval=6)
        trainer = _lifecycle_trainer(self.trainer_module, exp, epoch=1)

        trainer.after_epoch()

        self.assertEqual(trainer.save_ckpt.call_args_list, [mock.call(ckpt_name="latest")])
        exp.eval.assert_not_called()

    def test_missing_save_interval_preserves_evaluation_save(self):
        exp = _hbb_exp(eval_interval=3)
        trainer = _lifecycle_trainer(self.trainer_module, exp, epoch=2)

        trainer.after_epoch()

        self.assertEqual(
            trainer.save_ckpt.call_args_list,
            [mock.call(ckpt_name="latest"), mock.call("last_epoch", True)],
        )
        exp.eval.assert_called_once()

    def test_missing_save_interval_preserves_final_epoch_evaluation(self):
        exp = _hbb_exp(eval_interval=3)
        trainer = _lifecycle_trainer(self.trainer_module, exp, epoch=2)
        trainer.max_epoch = 3

        trainer.after_epoch()

        self.assertEqual(
            trainer.save_ckpt.call_args_list,
            [mock.call(ckpt_name="latest"), mock.call("last_epoch", True)],
        )
        exp.eval.assert_called_once()

    def test_explicit_save_interval_preserves_nonmatching_and_matching_epochs(self):
        nonmatching_exp = _hbb_exp(eval_interval=6)
        nonmatching_exp.save_interval = 3
        nonmatching_trainer = _lifecycle_trainer(
            self.trainer_module, nonmatching_exp, epoch=1
        )

        nonmatching_trainer.after_epoch()

        self.assertEqual(
            nonmatching_trainer.save_ckpt.call_args_list,
            [mock.call(ckpt_name="latest")],
        )
        nonmatching_exp.eval.assert_not_called()

        matching_exp = _hbb_exp(eval_interval=6)
        matching_exp.save_interval = 3
        matching_trainer = _lifecycle_trainer(self.trainer_module, matching_exp, epoch=2)

        matching_trainer.after_epoch()

        self.assertEqual(
            matching_trainer.save_ckpt.call_args_list,
            [mock.call(ckpt_name="latest"), mock.call(ckpt_name="3_epoch")],
        )
        matching_exp.eval.assert_not_called()

    def test_explicit_save_interval_preserves_evaluation_and_periodic_save(self):
        exp = _hbb_exp(eval_interval=3)
        exp.save_interval = 3
        trainer = _lifecycle_trainer(self.trainer_module, exp, epoch=2)

        trainer.after_epoch()

        self.assertEqual(
            trainer.save_ckpt.call_args_list,
            [
                mock.call(ckpt_name="latest"),
                mock.call(ckpt_name="3_epoch"),
                mock.call("last_epoch", True),
            ],
        )
        exp.eval.assert_called_once()

    def test_obb_save_interval_cadence_is_unchanged(self):
        from yolox.exp.yolox_base_obb_kld import ExpOBB_KLD

        exp = ExpOBB_KLD()
        exp.eval_interval = 999
        exp.eval = mock.Mock(return_value=(0.75, 0.8, "numeric timing"))
        trainer = _lifecycle_trainer(self.trainer_module, exp, epoch=9)

        trainer.after_epoch()

        self.assertEqual(exp.save_interval, 10)
        self.assertEqual(
            trainer.save_ckpt.call_args_list,
            [mock.call(ckpt_name="latest"), mock.call(ckpt_name="10_epoch")],
        )
        exp.eval.assert_not_called()


if __name__ == "__main__":
    unittest.main()
