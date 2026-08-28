"""Deterministic tests for the DOTA result-generation contract."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def _load_module(path, name, replacements):
    original_modules = {
        module_name: sys.modules.get(module_name) for module_name in replacements
    }
    sys.modules.update(replacements)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
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


def _load_dota_evaluator():
    yolox_stub = types.ModuleType("yolox")
    yolox_stub.__path__ = [str(ROOT / "yolox")]
    utils_stub = types.ModuleType("yolox.utils")
    utils_stub.gather = lambda value, dst=0: [value]
    utils_stub.is_main_process = lambda: True
    utils_stub.postprocessobb_kld = lambda outputs, *args: [None] * len(outputs)
    utils_stub.synchronize = lambda: None
    utils_stub.time_synchronized = lambda: 0.0
    loguru_stub = types.ModuleType("loguru")
    loguru_stub.logger = _Logger()
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable: iterable
    return _load_module(
        ROOT / "yolox" / "evaluators" / "dota_evaluator.py",
        "dota_evaluator_contract_test_module",
        {
            "yolox": yolox_stub,
            "yolox.utils": utils_stub,
            "loguru": loguru_stub,
            "tqdm": tqdm_stub,
        },
    )


def _load_dota_dataset():
    yolox_stub = types.ModuleType("yolox")
    yolox_stub.__path__ = [str(ROOT / "yolox")]
    data_stub = types.ModuleType("yolox.data")
    data_stub.__path__ = [str(ROOT / "yolox" / "data")]
    datasets_stub = types.ModuleType("yolox.data.datasets")
    datasets_stub.__path__ = [str(ROOT / "yolox" / "data" / "datasets")]
    evaluators_stub = types.ModuleType("yolox.evaluators")
    evaluators_stub.__path__ = [str(ROOT / "yolox" / "evaluators")]

    class Dataset:
        def __init__(self, img_size):
            self.input_dim = img_size

        @staticmethod
        def resize_getitem(function):
            return function

    wrapper_stub = types.ModuleType("yolox.data.datasets.datasets_wrapper")
    wrapper_stub.Dataset = Dataset
    classes_stub = types.ModuleType("yolox.data.datasets.dota_classes")
    classes_stub.VOC_CLASSES = (
        "plane",
        "ship",
        "storage-tank",
        "baseball-diamond",
        "tennis-court",
        "basketball-court",
        "ground-track-field",
        "harbor",
        "bridge",
        "large-vehicle",
        "small-vehicle",
        "helicopter",
        "roundabout",
        "soccer-ball-field",
        "swimming-pool",
        "container-crane",
    )
    voc_eval_stub = types.ModuleType("yolox.evaluators.voc_eval")
    voc_eval_stub.voc_eval = object()
    return _load_module(
        ROOT / "yolox" / "data" / "datasets" / "dota_obb.py",
        "yolox.data.datasets.dota_obb_contract_test_module",
        {
            "yolox": yolox_stub,
            "yolox.data": data_stub,
            "yolox.data.datasets": datasets_stub,
            "yolox.data.datasets.datasets_wrapper": wrapper_stub,
            "yolox.data.datasets.dota_classes": classes_stub,
            "yolox.evaluators": evaluators_stub,
            "yolox.evaluators.voc_eval": voc_eval_stub,
        },
    )


class _FakeDataset:
    def __init__(self, size):
        self.size = size
        self.received = None

    def __len__(self):
        return self.size

    def evaluate_detections(self, all_boxes, output_dir=None):
        self.received = all_boxes
        return None, None


class _FakeDataLoader:
    def __init__(self, dataset, batches, batch_size):
        self.dataset = dataset
        self.batches = batches
        self.batch_size = batch_size

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.seen = []

    def forward(self, images):
        self.seen.append((images.device, images.dtype))
        return torch.zeros(
            (images.shape[0], 1, 6), dtype=images.dtype, device=images.device
        )


def _batch(image_ids):
    image_ids = list(image_ids)
    return (
        torch.zeros((len(image_ids), 3, 4, 4), dtype=torch.float32),
        None,
        (
            torch.tensor([10] * len(image_ids)),
            torch.tensor([20] * len(image_ids)),
        ),
        torch.tensor(image_ids),
    )


class DotaEvaluatorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator_module = _load_dota_evaluator()
        cls.dataset_module = _load_dota_dataset()

    def _evaluator(self, dataset, batches, batch_size=1, img_size=(20, 40)):
        loader = _FakeDataLoader(dataset, batches, batch_size)
        return self.evaluator_module.DOTAEvaluator(
            loader, img_size, confthre=0.01, nmsthre=0.65, num_classes=16
        )

    def test_cpu_float32_one_batch_returns_no_metrics_and_uses_cpu_statistics(self):
        dataset = _FakeDataset(1)
        evaluator = self._evaluator(dataset, [_batch([0])])
        model = _FakeModel()
        self.evaluator_module.postprocessobb_kld = lambda outputs, *args: [None] * len(
            outputs
        )
        observed = {}
        original_evaluate_prediction = evaluator.evaluate_prediction

        def capture_statistics(data_dict, statistics):
            observed["device"] = statistics.device
            observed["timed_batches"] = statistics[2].item()
            return original_evaluate_prediction(data_dict, statistics)

        evaluator.evaluate_prediction = capture_statistics
        result = evaluator.evaluate(model)

        self.assertEqual(model.seen, [(torch.device("cpu"), torch.float32)])
        self.assertEqual(observed["device"], torch.device("cpu"))
        self.assertEqual(observed["timed_batches"], 0)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        self.assertIn("Timing unavailable", result[2])
        self.assertIn("Internal AP was not computed", result[2])

    def test_non_empty_polygon_is_scaled_and_separated_without_metrics(self):
        dataset = _FakeDataset(1)
        evaluator = self._evaluator(dataset, [_batch([0])])
        model = _FakeModel()

        def polygon_postprocess(outputs, *args):
            row = torch.tensor(
                [10, 20, 30, 40, 50, 60, 70, 80, 0.75, 0],
                dtype=outputs.dtype,
                device=outputs.device,
            )
            return [row.unsqueeze(0) for _ in range(outputs.shape[0])]

        self.evaluator_module.postprocessobb_kld = polygon_postprocess
        result = evaluator.evaluate(model)

        self.assertEqual(result[:2], (None, None))
        detections = dataset.received[0][0]
        np.testing.assert_allclose(
            detections,
            np.array([[5, 10, 15, 20, 25, 30, 35, 40, 0.75]], dtype=np.float32),
        )

    def test_cpu_half_precision_is_rejected(self):
        dataset = _FakeDataset(1)
        evaluator = self._evaluator(dataset, [_batch([0])])

        with self.assertRaisesRegex(ValueError, "CPU half precision is not supported"):
            evaluator.evaluate(_FakeModel(), half=True)

    def test_multiple_batches_time_only_eligible_batches_and_accept_partial_final_batch(
        self,
    ):
        dataset = _FakeDataset(3)
        evaluator = self._evaluator(
            dataset, [_batch([0, 1]), _batch([2])], batch_size=2
        )
        model = _FakeModel()
        self.evaluator_module.postprocessobb_kld = lambda outputs, *args: [None] * len(
            outputs
        )
        observed = {}
        original_evaluate_prediction = evaluator.evaluate_prediction

        def capture_statistics(data_dict, statistics):
            observed["timed_batches"] = statistics[2].item()
            result = original_evaluate_prediction(data_dict, statistics)
            observed["info"] = result[2]
            return result

        evaluator.evaluate_prediction = capture_statistics
        result = evaluator.evaluate(model)

        self.assertEqual(observed["timed_batches"], 1)
        self.assertIn("Average forward time", observed["info"])
        self.assertEqual(result[:2], (None, None))
        self.assertEqual(len(model.seen), 2)

    def test_non_main_process_has_truthful_empty_metric_slots(self):
        evaluator = self._evaluator(_FakeDataset(0), [])
        original_is_main_process = self.evaluator_module.is_main_process
        self.evaluator_module.is_main_process = lambda: False
        try:
            result = evaluator.evaluate_prediction(
                {}, torch.zeros(3, dtype=torch.float32)
            )
        finally:
            self.evaluator_module.is_main_process = original_is_main_process

        self.assertEqual(result, (None, None, None))


class DotaWriterContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset_module = _load_dota_dataset()
        cls.dataset_class = dataset_module.DOTAOBBDetection
        cls.classes = dataset_module.VOC_CLASSES

    def _dataset(self, root):
        dataset = self.dataset_class.__new__(self.dataset_class)
        dataset.root = root
        dataset._year = "2012"
        dataset.ids = [("unused", "image_a")]
        return dataset

    def test_empty_writer_returns_no_metrics_and_creates_empty_class_files(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = self._dataset(root)
            all_boxes = [[np.empty((0, 9), dtype=np.float32)] for _ in self.classes]

            result = dataset.evaluate_detections(
                all_boxes, output_dir=root + "/ignored"
            )

            self.assertEqual(result, (None, None))
            result_dir = Path(root) / "results" / "VOC2012" / "Main"
            self.assertTrue((result_dir / "plane.txt").exists())
            self.assertEqual((result_dir / "plane.txt").read_text(), "")
            self.assertFalse((Path(root) / "ignored").exists())

    def test_non_empty_writer_preserves_polygon_order_and_current_plus_one_format(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = self._dataset(root)
            all_boxes = [[[]] for _ in self.classes]
            all_boxes[0] = [
                np.array(
                    [[10, 20, 30, 40, 50, 60, 70, 80, 0.9876]], dtype=np.float32
                )
            ]

            result = dataset.evaluate_detections(all_boxes)

            self.assertEqual(result, (None, None))
            result_file = Path(root) / "results" / "VOC2012" / "Main" / "plane.txt"
            self.assertEqual(
                result_file.read_text(),
                "image_a 0.988 11.0 21.0 31.0 41.0 51.0 61.0 71.0 81.0\n",
            )


if __name__ == "__main__":
    unittest.main()
