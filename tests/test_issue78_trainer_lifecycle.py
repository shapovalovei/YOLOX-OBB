"""Behavioral regressions for Issue #78 Trainer lifecycle contracts."""

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


class _SummaryWriter:
    def __init__(self, *args, **kwargs):
        pass


class _TinyModel(torch.nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value]))
        self.head = SimpleNamespace(use_l1=False)

    def to(self, device):
        return self


class _CheckpointModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))
        self.register_buffer("norm", torch.tensor([value], dtype=torch.float32))
        self.head = SimpleNamespace(use_l1=False)


class _Loader:
    def __init__(self, length, events=None):
        self.length = length
        self.events = events if events is not None else []
        self.schedule = None
        self.close_count = 0

    def __len__(self):
        return self.length

    def configure_mosaic_schedule(self, start_ordinal, cutover_ordinal):
        self.schedule = (start_ordinal, cutover_ordinal)
        self.events.append(("configure", self.schedule))

    def close_mosaic(self):
        self.close_count += 1


class _Prefetcher:
    events = None

    def __init__(self, loader):
        self.events.append(("prefetch", loader.schedule))


def _load_save_checkpoint():
    path = ROOT / "yolox" / "utils" / "checkpoint.py"
    spec = importlib.util.spec_from_file_location(
        "issue78_trainer_checkpoint_module", path
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
    utils_stub.save_checkpoint = _load_save_checkpoint()
    utils_stub.setup_logger = lambda *args, **kwargs: None
    utils_stub.synchronize = lambda: None
    apex_stub = types.ModuleType("apex")
    apex_stub.amp = types.SimpleNamespace()
    apex_amp_stub = types.ModuleType("apex.amp")
    tensorboard_stub = types.ModuleType("torch.utils.tensorboard")
    tensorboard_stub.SummaryWriter = _SummaryWriter
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
            "issue78_trainer_module", path
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


def _epoch_trainer(module, max_epoch, no_aug_epochs, epoch, no_aug=False):
    trainer = object.__new__(module.Trainer)
    trainer.epoch = epoch
    trainer.max_epoch = max_epoch
    trainer.no_aug = no_aug
    trainer.exp = SimpleNamespace(
        no_aug_epochs=no_aug_epochs,
        eval_interval=5,
    )
    trainer.train_loader = _Loader(length=4)
    trainer.is_distributed = False
    trainer.model = _TinyModel()
    trainer.save_ckpt = mock.Mock()
    return trainer


class Issue78TrainerLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer_module = _load_trainer()

    def test_before_train_configures_schedule_before_prefetch(self):
        events = []
        loader = _Loader(length=4, events=events)
        model = _TinyModel()
        exp = SimpleNamespace(
            max_epoch=100,
            no_aug_epochs=15,
            ema=False,
            input_size=(32, 32),
            test_size=(32, 32),
            output_dir="/tmp/issue78",
            print_interval=10,
            basic_lr_per_img=0.01,
            random_size=None,
            eval_interval=10,
            get_model=lambda: model,
            get_optimizer=lambda batch_size: torch.optim.SGD(
                model.parameters(), lr=0.1
            ),
            get_data_loader=lambda **kwargs: loader,
            get_lr_scheduler=lambda *args: object(),
            get_evaluator=lambda **kwargs: object(),
        )
        args = SimpleNamespace(
            fp16=False,
            batch_size=2,
            occupy=False,
            resume=False,
            ckpt=None,
            experiment_name="issue78",
        )
        trainer = object.__new__(self.trainer_module.Trainer)
        trainer.exp = exp
        trainer.args = args
        trainer.max_epoch = exp.max_epoch
        trainer.amp_training = False
        trainer.is_distributed = False
        trainer.rank = 0
        trainer.local_rank = 0
        trainer.device = "cuda:0"
        trainer.use_model_ema = False
        trainer.file_name = "/tmp/issue78"
        trainer.best_ap = 0
        trainer.best_ap_available = False
        trainer.resume_train = lambda current_model: (
            setattr(trainer, "start_epoch", 0) or current_model
        )
        _Prefetcher.events = events

        with mock.patch.object(self.trainer_module.torch.cuda, "set_device"):
            with mock.patch.object(self.trainer_module, "DataPrefetcher", _Prefetcher):
                trainer.before_train()

        self.assertEqual(loader.schedule, (0, 85 * 4))
        self.assertEqual(trainer.max_iter, 4)
        self.assertEqual(
            events,
            [("configure", (0, 340)), ("prefetch", (0, 340))],
        )

    def test_small_epoch_matrix_enables_l1_only_in_no_aug_phase(self):
        observed_l1 = []
        observed_close = []
        for epoch in range(5):
            trainer = _epoch_trainer(self.trainer_module, 5, 2, epoch)
            trainer.before_epoch()
            observed_l1.append(trainer.model.head.use_l1)
            observed_close.append(trainer.train_loader.close_count)

        self.assertEqual(observed_l1, [False, False, False, True, True])
        self.assertEqual(observed_close, [0, 0, 0, 1, 1])

    def test_direct_no_aug_start_has_l1_and_no_mosaic_immediately(self):
        trainer = _epoch_trainer(
            self.trainer_module, max_epoch=5, no_aug_epochs=2, epoch=3, no_aug=True
        )
        trainer.before_epoch()

        self.assertTrue(trainer.model.head.use_l1)
        self.assertEqual(trainer.train_loader.close_count, 1)
        trainer.save_ckpt.assert_not_called()

    def test_last_mosaic_checkpoint_is_after_final_epoch_and_resumes_at_cutover(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            trainer = object.__new__(self.trainer_module.Trainer)
            trainer.use_model_ema = True
            trainer.model = _CheckpointModel(84.0)
            trainer.ema_model = SimpleNamespace(ema=_CheckpointModel(84.0))
            trainer.optimizer = torch.optim.SGD(
                trainer.model.parameters(), lr=0.1, momentum=0.9
            )
            trainer.optimizer.state[trainer.model.weight]["momentum_buffer"] = torch.tensor(
                [84.0]
            )
            trainer.amp_training = False
            trainer.rank = 0
            trainer.device = "cpu"
            trainer.file_name = str(directory)
            trainer.epoch = 84
            trainer.max_epoch = 100
            trainer.no_aug = False
            trainer.train_loader = _Loader(length=4)
            trainer.best_ap = 0
            trainer.best_ap_available = False
            trainer.is_distributed = False
            trainer.exp = SimpleNamespace(
                no_aug_epochs=15,
                eval_interval=1,
                save_interval=99,
                eval=mock.Mock(return_value=(None, None, "external timing")),
            )
            trainer.evaluator = object()
            trainer.tblogger = _SummaryWriter()

            # before_epoch is intentionally called before the synthetic final
            # epoch mutation to expose the old lifecycle ordering.
            trainer.before_epoch()
            trainer.model.weight.data.fill_(85.0)
            trainer.model.norm.data.fill_(85.0)
            trainer.ema_model.ema.weight.data.fill_(85.0)
            trainer.ema_model.ema.norm.data.fill_(85.0)
            trainer.optimizer.state[trainer.model.weight]["momentum_buffer"].fill_(85.0)
            trainer.after_epoch()

            checkpoint_path = directory / "last_mosaic_epoch_ckpt.pth"
            self.assertTrue(checkpoint_path.exists())
            state = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(state["start_epoch"], 85)
            self.assertEqual(state["model"]["weight"].item(), 85.0)
            self.assertEqual(state["model"]["norm"].item(), 85.0)
            momentum = next(iter(state["optimizer"]["state"].values()))[
                "momentum_buffer"
            ]
            self.assertEqual(momentum.item(), 85.0)

            resumed = object.__new__(self.trainer_module.Trainer)
            resumed.use_model_ema = False
            resumed.model = _CheckpointModel(0.0)
            resumed.optimizer = torch.optim.SGD(
                resumed.model.parameters(), lr=0.1, momentum=0.9
            )
            resumed.amp_training = False
            resumed.device = "cpu"
            resumed.best_ap = 0
            resumed.best_ap_available = False
            resumed.args = SimpleNamespace(
                resume=True,
                ckpt=str(checkpoint_path),
                start_epoch=None,
            )
            resumed.resume_train(resumed.model)

            resumed_epochs = list(range(resumed.start_epoch, 100))
            self.assertEqual(resumed.start_epoch, 85)
            self.assertEqual(resumed_epochs, list(range(85, 100)))


if __name__ == "__main__":
    unittest.main()
