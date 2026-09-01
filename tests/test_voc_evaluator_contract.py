"""Regression coverage for the maintained VOC evaluator timing contract."""

import unittest
from unittest import mock

import torch
from torch.utils.data import DataLoader, Dataset

from yolox.evaluators import voc_evaluator


class _SyntheticDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.eval_calls = 0
        self.received_history = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return (
            torch.zeros((3, 4, 4), dtype=torch.float32),
            torch.zeros((0, 5), dtype=torch.float32),
            (10, 20),
            index,
        )

    def evaluate_detections(self, all_boxes, output_dir):
        self.eval_calls += 1
        self.received_history.append(all_boxes)
        return 0.25, 0.50


class _FakeModel(torch.nn.Module):
    def __init__(self, output_factory):
        super().__init__()
        self.output_factory = output_factory
        self.eval_calls = 0
        self.forward_calls = 0

    def eval(self):
        self.eval_calls += 1
        return super().eval()

    def forward(self, images):
        self.forward_calls += 1
        return self.output_factory(self.forward_calls, images)


def _ordinary_output(_call_number, images):
    row = torch.tensor(
        [5, 5, 2, 2, 0.9, 0.9], dtype=images.dtype, device=images.device
    )
    return row.view(1, 1, 6).expand(images.shape[0], -1, -1).clone()


def _empty_output(_call_number, images):
    return torch.empty(
        (images.shape[0], 0, 6), dtype=images.dtype, device=images.device
    )


def _below_threshold_output(_call_number, images):
    row = torch.tensor(
        [5, 5, 2, 2, 0.1, 0.1], dtype=images.dtype, device=images.device
    )
    return row.view(1, 1, 6).expand(images.shape[0], -1, -1).clone()


class VOCEvaluatorTimingContractTests(unittest.TestCase):
    def _make_evaluator(self, dataset_size, batch_size=1):
        dataset = _SyntheticDataset(dataset_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        evaluator = voc_evaluator.VOCEvaluator(
            dataloader,
            img_size=(20, 40),
            confthre=0.5,
            nmsthre=0.45,
            num_classes=1,
        )
        return evaluator, dataset

    def _evaluate(self, evaluator, model):
        observed = {}
        original_evaluate_prediction = evaluator.evaluate_prediction

        def capture_statistics(data_dict, statistics):
            observed["data_dict"] = data_dict
            observed["statistics"] = statistics.clone()
            return original_evaluate_prediction(data_dict, statistics)

        evaluator.evaluate_prediction = capture_statistics
        with (
            mock.patch.object(
                voc_evaluator.torch.cuda, "FloatTensor", torch.FloatTensor
            ),
            mock.patch.object(voc_evaluator, "tqdm", lambda iterable: iterable),
            mock.patch.object(voc_evaluator, "synchronize", lambda: None),
            mock.patch.object(voc_evaluator.time, "time", return_value=0.0),
            mock.patch.object(
                voc_evaluator,
                "time_synchronized",
                side_effect=[0.01, 0.03] * 10,
            ),
        ):
            result = evaluator.evaluate(model)
        return result, observed

    def test_empty_loader_fails_before_model_and_dataset_side_effects(self):
        evaluator, dataset = self._make_evaluator(0)
        model = _FakeModel(_ordinary_output)

        with self.assertRaisesRegex(ValueError, "non-empty"):
            self._evaluate(evaluator, model)

        self.assertEqual(model.eval_calls, 0)
        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(dataset.eval_calls, 0)

    def test_one_batch_ordinary_predictions_reach_evaluation_with_unavailable_timing(
        self,
    ):
        evaluator, dataset = self._make_evaluator(1)
        model = _FakeModel(_ordinary_output)

        result, observed = self._evaluate(evaluator, model)

        self.assertEqual(result[:2], (0.25, 0.50))
        self.assertEqual(len(result), 3)
        self.assertIn("Timing unavailable", result[2])
        self.assertNotIn("Average forward time", result[2])
        self.assertNotRegex(result[2].lower(), r"(?:nan|inf)")
        self.assertEqual(observed["statistics"][2].item(), 0)
        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(dataset.eval_calls, 1)
        self.assertEqual(dataset.received_history[-1][0][0].shape, (1, 5))

    def test_one_batch_empty_prediction_reaches_evaluation_with_empty_boxes(self):
        evaluator, dataset = self._make_evaluator(1)
        model = _FakeModel(_empty_output)

        result, observed = self._evaluate(evaluator, model)

        self.assertEqual(result[:2], (0.25, 0.50))
        self.assertIn("Timing unavailable", result[2])
        self.assertNotIn("0.00 ms", result[2])
        self.assertNotRegex(result[2].lower(), r"(?:nan|inf)")
        self.assertEqual(observed["data_dict"][0], (None, None, None))
        self.assertEqual(dataset.eval_calls, 1)
        self.assertEqual(dataset.received_history[-1][0][0].shape, (0, 5))

    def test_one_batch_below_threshold_prediction_reaches_evaluation_as_none(self):
        evaluator, dataset = self._make_evaluator(1)
        model = _FakeModel(_below_threshold_output)

        result, observed = self._evaluate(evaluator, model)

        self.assertEqual(result[:2], (0.25, 0.50))
        self.assertIn("Timing unavailable", result[2])
        self.assertNotIn("0.00 ms", result[2])
        self.assertNotRegex(result[2].lower(), r"(?:nan|inf)")
        self.assertEqual(observed["data_dict"][0], (None, None, None))
        self.assertEqual(dataset.eval_calls, 1)
        self.assertEqual(dataset.received_history[-1][0][0].shape, (0, 5))

    def test_two_batches_preserve_finite_timing_and_final_batch_exclusion(self):
        evaluator, dataset = self._make_evaluator(2)
        model = _FakeModel(_ordinary_output)

        result, observed = self._evaluate(evaluator, model)

        self.assertEqual(
            result[2],
            "Average forward time: 10.00 ms, Average NMS time: 20.00 ms, "
            "Average inference time: 30.00 ms\n",
        )
        self.assertEqual(observed["statistics"][2].item(), 1)
        self.assertEqual(model.forward_calls, 2)
        self.assertEqual(dataset.eval_calls, 1)
        self.assertEqual(len(dataset.received_history[-1][0]), 2)

    def test_multi_batch_partial_final_batch_preserves_timing_denominator(self):
        evaluator, dataset = self._make_evaluator(5, batch_size=2)
        model = _FakeModel(_ordinary_output)

        result, observed = self._evaluate(evaluator, model)

        self.assertEqual(
            result[2],
            "Average forward time: 5.00 ms, Average NMS time: 10.00 ms, "
            "Average inference time: 15.00 ms\n",
        )
        self.assertEqual(observed["statistics"][2].item(), 2)
        self.assertEqual(model.forward_calls, 3)
        self.assertEqual(dataset.eval_calls, 1)
        self.assertEqual(len(dataset.received_history[-1][0]), 5)

    def test_repeated_evaluator_use_does_not_reuse_timing_or_result_state(self):
        evaluator, dataset = self._make_evaluator(2)
        model = _FakeModel(_ordinary_output)

        first_result, first_observed = self._evaluate(evaluator, model)
        second_result, second_observed = self._evaluate(evaluator, model)

        self.assertEqual(second_result, first_result)
        self.assertEqual(first_observed["statistics"][2].item(), 1)
        self.assertEqual(second_observed["statistics"][2].item(), 1)
        self.assertEqual(model.forward_calls, 4)
        self.assertEqual(dataset.eval_calls, 2)
        self.assertEqual(len(dataset.received_history), 2)
        self.assertEqual(len(dataset.received_history[0][0]), 2)
        self.assertEqual(len(dataset.received_history[1][0]), 2)

    def test_none_prediction_content_remains_independent_of_timing(self):
        def first_none_then_ordinary(call_number, images):
            if call_number == 1:
                return _below_threshold_output(call_number, images)
            return _ordinary_output(call_number, images)

        evaluator, dataset = self._make_evaluator(2)
        model = _FakeModel(first_none_then_ordinary)

        result, observed = self._evaluate(evaluator, model)

        self.assertEqual(
            result[2],
            "Average forward time: 10.00 ms, Average NMS time: 20.00 ms, "
            "Average inference time: 30.00 ms\n",
        )
        self.assertEqual(observed["statistics"][2].item(), 1)
        self.assertEqual(dataset.eval_calls, 1)
        self.assertEqual(dataset.received_history[-1][0][0].shape, (0, 5))
        self.assertEqual(dataset.received_history[-1][0][1].shape, (1, 5))


if __name__ == "__main__":
    unittest.main()
