"""Regression coverage for trainer best-checkpoint state persistence."""

import hashlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


class _Logger:
    def info(self, message):
        pass


class _TensorBoard:
    def add_scalar(self, name, value, step):
        pass


class _CheckpointModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))
        self.register_buffer("norm", torch.tensor([value], dtype=torch.float32))


def _load_maintained_save_checkpoint():
    path = ROOT / "yolox" / "utils" / "checkpoint.py"
    spec = importlib.util.spec_from_file_location(
        "trainer_checkpoint_state_checkpoint_module", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    loguru_stub = types.ModuleType("loguru")
    loguru_stub.logger = _Logger()
    original_loguru = sys.modules.get("loguru")
    sys.modules["loguru"] = loguru_stub
    try:
        spec.loader.exec_module(module)
    finally:
        if original_loguru is None:
            sys.modules.pop("loguru", None)
        else:
            sys.modules["loguru"] = original_loguru
    return module.save_checkpoint


def _load_trainer():
    yolox_stub = types.ModuleType("yolox")
    yolox_stub.__path__ = [str(ROOT / "yolox")]
    data_stub = types.ModuleType("yolox.data")
    data_stub.DataPrefetcher = object
    utils_stub = types.ModuleType("yolox.utils")
    utils_stub.MeterBuffer = object
    utils_stub.ModelEMA = object
    utils_stub.all_reduce_norm = lambda *args: None
    utils_stub.get_async_norm_states = lambda model: {"norm": model.norm}
    utils_stub.get_local_rank = lambda: 0
    utils_stub.get_model_info = lambda *args: ""
    utils_stub.get_rank = lambda: 0
    utils_stub.get_world_size = lambda: 1
    utils_stub.gpu_mem_usage = lambda: 0
    utils_stub.is_parallel = lambda model: False
    utils_stub.load_ckpt = lambda model, ckpt: model
    utils_stub.occupy_mem = lambda: None
    utils_stub.save_checkpoint = _load_maintained_save_checkpoint()
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
            "trainer_checkpoint_state_trainer_module", path
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


def _trainer(module, directory, best_ap=0, best_ap_available=False, epoch=7):
    trainer = object.__new__(module.Trainer)
    trainer.use_model_ema = False
    trainer.model = _CheckpointModel(2.0)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.amp_training = False
    trainer.rank = 0
    trainer.device = "cpu"
    trainer.file_name = str(directory)
    trainer.epoch = epoch
    trainer.best_ap = best_ap
    trainer.best_ap_available = best_ap_available
    return trainer


def _resume(module, directory, checkpoint_path, best_ap=42, best_ap_available=True):
    trainer = _trainer(
        module,
        directory,
        best_ap=best_ap,
        best_ap_available=best_ap_available,
    )
    trainer.args = SimpleNamespace(resume=True, ckpt=str(checkpoint_path), start_epoch=None)
    trainer.resume_train(trainer.model)
    return trainer


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrainerCheckpointStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer_module = _load_trainer()

    def _historical_checkpoint(self, directory):
        source = _trainer(
            self.trainer_module,
            directory,
            best_ap=0.8,
            best_ap_available=True,
        )
        source.save_ckpt("historical", update_best_ckpt=True)
        source.save_ckpt("latest")
        return directory / "latest_ckpt.pth"

    def test_new_checkpoint_serializes_and_resume_restores_best_state(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = _trainer(
                self.trainer_module,
                directory,
                best_ap=0.8,
                best_ap_available=True,
            )
            source.save_ckpt("latest")

            state = torch.load(directory / "latest_ckpt.pth", map_location="cpu")
            self.assertEqual(
                set(state),
                {"start_epoch", "model", "optimizer", "best_ap", "best_ap_available"},
            )
            self.assertEqual(state["best_ap"], 0.8)
            self.assertTrue(state["best_ap_available"])

            resumed = _resume(
                self.trainer_module,
                directory,
                directory / "latest_ckpt.pth",
            )

            self.assertEqual(resumed.start_epoch, 8)
            self.assertEqual(resumed.best_ap, 0.8)
            self.assertTrue(resumed.best_ap_available)

    def test_lower_metric_after_resume_keeps_historical_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            latest_path = self._historical_checkpoint(directory)
            historical_best_digest = _digest(directory / "best_ckpt.pth")
            resumed = _resume(self.trainer_module, directory, latest_path)
            resumed.epoch = 8
            resumed.exp = SimpleNamespace(
                eval=mock.Mock(return_value=(0.7, 0.8, "numeric timing"))
            )
            resumed.evaluator = object()
            resumed.is_distributed = False
            resumed.tblogger = _TensorBoard()

            resumed.evaluate_and_save_model()

            self.assertEqual(_digest(directory / "best_ckpt.pth"), historical_best_digest)
            self.assertEqual(resumed.best_ap, 0.8)
            self.assertTrue(resumed.best_ap_available)
            state = torch.load(directory / "last_epoch_ckpt.pth", map_location="cpu")
            self.assertEqual(state["best_ap"], 0.8)
            self.assertTrue(state["best_ap_available"])

    def test_higher_metric_after_resume_updates_best_checkpoint_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            latest_path = self._historical_checkpoint(directory)
            historical_best_digest = _digest(directory / "best_ckpt.pth")
            resumed = _resume(self.trainer_module, directory, latest_path)
            resumed.epoch = 8
            resumed.exp = SimpleNamespace(
                eval=mock.Mock(return_value=(0.9, 0.8, "numeric timing"))
            )
            resumed.evaluator = object()
            resumed.is_distributed = False
            resumed.tblogger = _TensorBoard()

            resumed.evaluate_and_save_model()

            self.assertNotEqual(_digest(directory / "best_ckpt.pth"), historical_best_digest)
            self.assertEqual(resumed.best_ap, 0.9)
            self.assertTrue(resumed.best_ap_available)
            for name in ("last_epoch_ckpt.pth", "best_ckpt.pth"):
                state = torch.load(directory / name, map_location="cpu")
                self.assertEqual(state["best_ap"], 0.9)
                self.assertTrue(state["best_ap_available"])

    def test_after_epoch_keeps_default_latest_resume_state_current_after_best_eval(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            trainer = _trainer(
                self.trainer_module,
                directory,
                best_ap=0.8,
                best_ap_available=True,
            )
            trainer.exp = SimpleNamespace(
                eval_interval=1,
                save_interval=99,
                eval=mock.Mock(return_value=(0.9, 0.8, "numeric timing")),
            )
            trainer.evaluator = object()
            trainer.is_distributed = False
            trainer.tblogger = _TensorBoard()
            save_ckpt = mock.Mock(wraps=trainer.save_ckpt)
            trainer.save_ckpt = save_ckpt

            trainer.after_epoch()

            self.assertEqual(
                save_ckpt.call_args_list,
                [mock.call(ckpt_name="latest"), mock.call("last_epoch", True)],
            )
            latest_state = torch.load(
                directory / "latest_ckpt.pth", map_location="cpu"
            )
            self.assertEqual(latest_state["best_ap"], 0.9)
            self.assertTrue(latest_state["best_ap_available"])

            resumed = _trainer(
                self.trainer_module,
                directory,
                best_ap=42,
                best_ap_available=True,
            )
            resumed.args = SimpleNamespace(
                resume=True,
                ckpt=None,
                start_epoch=None,
            )
            resumed.resume_train(resumed.model)

            self.assertEqual(resumed.start_epoch, 8)
            self.assertEqual(resumed.best_ap, 0.9)
            self.assertTrue(resumed.best_ap_available)

    def test_after_epoch_preserves_pre_evaluation_model_snapshots_with_current_latest_metadata(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            trainer = _trainer(
                self.trainer_module,
                directory,
                best_ap=0.8,
                best_ap_available=True,
            )

            trainer.exp = SimpleNamespace(
                eval_interval=1,
                save_interval=1,
                eval=mock.Mock(return_value=(0.9, 0.8, "numeric timing")),
            )
            trainer.evaluator = object()
            trainer.is_distributed = False
            trainer.tblogger = _TensorBoard()
            save_ckpt = mock.Mock(wraps=trainer.save_ckpt)
            trainer.save_ckpt = save_ckpt
            trainer.optimizer.state[trainer.model.weight]["momentum_buffer"] = torch.tensor(
                [3.0], dtype=torch.float32
            )
            optimizer_states = []
            original_optimizer_state_dict = trainer.optimizer.state_dict

            def capture_optimizer_state():
                state = original_optimizer_state_dict()
                optimizer_states.append(state)
                return state

            trainer.optimizer.state_dict = capture_optimizer_state

            with mock.patch.object(
                self.trainer_module,
                "all_reduce_norm",
                side_effect=lambda model: model.norm.fill_(9.0),
            ) as all_reduce_norm:
                trainer.after_epoch()

            all_reduce_norm.assert_called_once_with(trainer.model)
            trainer.exp.eval.assert_called_once()
            self.assertEqual(
                save_ckpt.call_args_list,
                [
                    mock.call(ckpt_name="latest"),
                    mock.call(ckpt_name="8_epoch"),
                    mock.call("last_epoch", True),
                ],
            )
            latest_state = torch.load(
                directory / "latest_ckpt.pth", map_location="cpu"
            )
            periodic_state = torch.load(
                directory / "8_epoch_ckpt.pth", map_location="cpu"
            )
            last_epoch_state = torch.load(
                directory / "last_epoch_ckpt.pth", map_location="cpu"
            )
            best_state = torch.load(directory / "best_ckpt.pth", map_location="cpu")

            self.assertEqual(latest_state["model"]["norm"].item(), 2.0)
            self.assertEqual(periodic_state["model"]["norm"].item(), 2.0)
            self.assertEqual(last_epoch_state["model"]["norm"].item(), 9.0)
            self.assertEqual(best_state["model"]["norm"].item(), 9.0)
            self.assertEqual(latest_state["best_ap"], 0.9)
            self.assertTrue(latest_state["best_ap_available"])
            optimizer_checkpoint_states = [
                torch.load(directory / name, map_location="cpu")["optimizer"]
                for name in (
                    "latest_ckpt.pth",
                    "8_epoch_ckpt.pth",
                    "last_epoch_ckpt.pth",
                )
            ]
            first_checkpoint_optimizer_state = optimizer_checkpoint_states[0]
            for optimizer_state in optimizer_checkpoint_states[1:]:
                self.assertEqual(
                    optimizer_state["param_groups"],
                    first_checkpoint_optimizer_state["param_groups"],
                )
                self.assertEqual(
                    optimizer_state["state"].keys(),
                    first_checkpoint_optimizer_state["state"].keys(),
                )
            momentum_buffer = next(
                iter(first_checkpoint_optimizer_state["state"].values())
            )["momentum_buffer"]
            self.assertTrue(torch.equal(momentum_buffer, torch.tensor([3.0])))
            self.assertEqual(len(optimizer_states), 3)
            first_optimizer_state = optimizer_states[0]
            for optimizer_state in optimizer_states[1:]:
                self.assertEqual(
                    optimizer_state["state"].keys(), first_optimizer_state["state"].keys()
                )
                for parameter_id in first_optimizer_state["state"]:
                    self.assertTrue(
                        torch.equal(
                            optimizer_state["state"][parameter_id]["momentum_buffer"],
                            first_optimizer_state["state"][parameter_id]["momentum_buffer"],
                        )
                    )
            self.assertFalse(hasattr(trainer, "_checkpoint_model_state"))
            self.assertFalse(hasattr(trainer, "_checkpoint_optimizer_state"))

            resumed = _trainer(
                self.trainer_module,
                directory,
                best_ap=42,
                best_ap_available=True,
            )
            resumed.args = SimpleNamespace(
                resume=True,
                ckpt=None,
                start_epoch=None,
            )
            resumed.resume_train(resumed.model)

            self.assertEqual(resumed.model.norm.item(), 2.0)
            self.assertEqual(resumed.best_ap, 0.9)
            self.assertTrue(resumed.best_ap_available)
            resumed_optimizer_state = resumed.optimizer.state_dict()
            resumed_momentum_buffer = next(
                iter(resumed_optimizer_state["state"].values())
            )["momentum_buffer"]
            self.assertTrue(torch.equal(resumed_momentum_buffer, torch.tensor([3.0])))

    def test_after_epoch_does_not_deepcopy_full_checkpoint_state(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            trainer = _trainer(
                self.trainer_module,
                directory,
                best_ap=0.8,
                best_ap_available=True,
            )
            trainer.exp = SimpleNamespace(
                eval_interval=1,
                save_interval=99,
                eval=mock.Mock(return_value=(0.9, 0.8, "numeric timing")),
            )
            trainer.evaluator = object()
            trainer.is_distributed = False
            trainer.tblogger = _TensorBoard()

            with mock.patch(
                "copy.deepcopy",
                side_effect=AssertionError("full checkpoint state deepcopy is forbidden"),
            ):
                trainer.after_epoch()

    def test_legacy_checkpoint_missing_either_field_uses_safe_defaults(self):
        for present_fields in ((), ("best_ap",), ("best_ap_available",)):
            with self.subTest(present_fields=present_fields):
                with tempfile.TemporaryDirectory() as directory_name:
                    directory = Path(directory_name)
                    source = _trainer(self.trainer_module, directory)
                    state = {
                        "start_epoch": 8,
                        "model": source.model.state_dict(),
                        "optimizer": source.optimizer.state_dict(),
                    }
                    if "best_ap" in present_fields:
                        state["best_ap"] = 0.8
                    if "best_ap_available" in present_fields:
                        state["best_ap_available"] = True
                    checkpoint_path = directory / "legacy_ckpt.pth"
                    torch.save(state, checkpoint_path)

                    resumed = _resume(
                        self.trainer_module,
                        directory,
                        checkpoint_path,
                    )

                    self.assertEqual(resumed.start_epoch, 8)
                    self.assertEqual(resumed.best_ap, 0)
                    self.assertFalse(resumed.best_ap_available)

                    resumed.exp = SimpleNamespace(
                        eval=mock.Mock(return_value=(0.7, 0.8, "numeric timing"))
                    )
                    resumed.evaluator = object()
                    resumed.is_distributed = False
                    resumed.tblogger = _TensorBoard()
                    resumed.epoch = 8
                    resumed.evaluate_and_save_model()

                    self.assertEqual(resumed.best_ap, 0.7)
                    self.assertTrue(resumed.best_ap_available)
                    state = torch.load(
                        directory / "last_epoch_ckpt.pth", map_location="cpu"
                    )
                    self.assertEqual(state["best_ap"], 0.7)
                    self.assertTrue(state["best_ap_available"])


if __name__ == "__main__":
    unittest.main()
