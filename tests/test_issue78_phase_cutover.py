"""Behavioral regressions for Issue #78 logical Mosaic phase scheduling."""

import itertools
import os
import random
import unittest

import numpy as np
import torch
from torch.utils.data import SequentialSampler
from torch.utils.data import get_worker_info

from yolox.data import DataLoader, YoloBatchSampler
from yolox.data.datasets.datasets_wrapper import Dataset
from yolox.data.datasets.mosaicdetection import MosaicDetection
from yolox.data.datasets.mosaicdetection_obb import MosaicDetectionOBB
from yolox.data.samplers import InfiniteSampler


INPUT_DIM = (32, 32)
SOURCE_HW = (20, 24)


class SyntheticSource(Dataset):
    """Small maintained-path source dataset with HBB or OBB labels."""

    def __init__(self, obb, size=16):
        super().__init__(INPUT_DIM, mosaic=False)
        self.obb = obb
        self.size = size
        self.image = np.zeros((*SOURCE_HW, 3), dtype=np.uint8)
        if obb:
            self.target = np.asarray(
                [[4.0, 4.0, 18.0, 14.0, 15.0, 0.0]], dtype=np.float32
            )
        else:
            self.target = np.asarray(
                [[4.0, 4.0, 18.0, 14.0, 0.0]], dtype=np.float32
            )

    def __len__(self):
        return self.size

    def pull_item(self, index):
        return self.image.copy(), self.target.copy(), SOURCE_HW, index

    def load_anno(self, index):
        return self.target.copy()


class PhaseRecordingPreproc:
    """Encode phase, worker identity, call count, and RNG probes in the image."""

    def __init__(self, obb):
        self.obb = obb
        self.call_count = 0

    def __call__(self, image, targets, input_dim):
        self.call_count += 1
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        phase_is_mosaic = tuple(image.shape[:2]) == tuple(input_dim)

        # These draws are intentionally after the wrapper's phase-specific
        # augmentation.  They make worker RNG continuation observable without
        # treating the expected Mosaic/no-Mosaic draw-count difference as a
        # regression.
        python_token = int(random.random() * 1000000)
        numpy_token = int(np.random.randint(0, 1000000))
        torch_token = int(torch.randint(0, 1000000, (1,)).item())

        image_out = np.zeros((3, input_dim[0], input_dim[1]), dtype=np.float32)
        image_out[0, 0, 0] = float(phase_is_mosaic)
        image_out[0, 0, 1] = float(os.getpid())
        image_out[0, 0, 2] = float(worker_id)
        image_out[0, 0, 3] = float(self.call_count)
        image_out[0, 1, 0] = float(python_token)
        image_out[0, 1, 1] = float(numpy_token)
        image_out[0, 1, 2] = float(torch_token)

        label_width = 6 if self.obb else 5
        labels_out = np.zeros((2, label_width), dtype=np.float32)
        return image_out, labels_out


def make_phase_loader(obb, num_workers, prefetch_factor):
    source = SyntheticSource(obb)
    preproc = PhaseRecordingPreproc(obb)
    wrapper_type = MosaicDetectionOBB if obb else MosaicDetection
    wrapper = wrapper_type(
        source,
        INPUT_DIM,
        mosaic=True,
        preproc=preproc,
        degrees=0.0,
        translate=0.0,
        scale=(1.0, 1.0),
        mscale=(1.0, 1.0),
        shear=0.0,
        perspective=0.0,
        enable_mixup=False,
    )
    batch_sampler = YoloBatchSampler(
        sampler=SequentialSampler(wrapper),
        batch_size=1,
        drop_last=False,
        input_dimension=INPUT_DIM,
        mosaic=True,
    )
    loader_kwargs = {"num_workers": num_workers}
    if num_workers:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(wrapper, batch_sampler=batch_sampler, **loader_kwargs)
    return loader


def decode_phase_batch(batch):
    image = batch[0][0]
    return {
        "mosaic": bool(round(float(image[0, 0, 0].item()))),
        "pid": int(round(float(image[0, 0, 1].item()))),
        "worker": int(round(float(image[0, 0, 2].item()))),
        "call": int(round(float(image[0, 0, 3].item()))),
        "rng": tuple(int(round(float(image[0, 1, i].item()))) for i in range(3)),
    }


def consume_phase_matrix(obb, num_workers, prefetch_factor, count=16):
    loader = make_phase_loader(obb, num_workers, prefetch_factor)
    loader.configure_mosaic_schedule(start_ordinal=0, cutover_ordinal=8)
    iterator = iter(loader)
    try:
        return [decode_phase_batch(next(iterator)) for _ in range(count)]
    finally:
        if num_workers:
            try:
                next(iterator)
            except StopIteration:
                pass


class Issue78PhaseCutoverTests(unittest.TestCase):
    def test_hbb_phase_matrix_is_independent_of_prefetch_depth(self):
        for num_workers, prefetch_factor in ((0, None), (4, 2), (2, 3)):
            with self.subTest(num_workers=num_workers, prefetch_factor=prefetch_factor):
                observations = consume_phase_matrix(
                    False, num_workers, prefetch_factor
                )
                self.assertEqual(
                    [item["mosaic"] for item in observations],
                    [ordinal < 8 for ordinal in range(16)],
                )

    def test_obb_phase_matrix_is_independent_of_prefetch_depth(self):
        for num_workers, prefetch_factor in ((0, None), (4, 2), (2, 3)):
            with self.subTest(num_workers=num_workers, prefetch_factor=prefetch_factor):
                observations = consume_phase_matrix(
                    True, num_workers, prefetch_factor
                )
                self.assertEqual(
                    [item["mosaic"] for item in observations],
                    [ordinal < 8 for ordinal in range(16)],
                )

    def test_positive_workers_keep_pid_and_rng_stream_continuous(self):
        for obb in (False, True):
            with self.subTest(obb=obb):
                observations = consume_phase_matrix(obb, 4, 2)
                by_worker = {}
                for ordinal, item in enumerate(observations):
                    if item["worker"] < 0:
                        self.fail("positive-worker path returned no worker id")
                    by_worker.setdefault(item["worker"], []).append((ordinal, item))

                self.assertGreaterEqual(len(by_worker), 2)
                crossed = 0
                for entries in by_worker.values():
                    pids = {item["pid"] for _, item in entries}
                    self.assertEqual(len(pids), 1)
                    calls_before = [item["call"] for o, item in entries if o < 8]
                    calls_after = [item["call"] for o, item in entries if o >= 8]
                    if calls_before and calls_after:
                        crossed += 1
                        self.assertGreater(min(calls_after), max(calls_before))
                        self.assertTrue(any(item["rng"] for _, item in entries))
                self.assertGreater(crossed, 0)

    def test_unscheduled_close_mosaic_only_affects_future_emission(self):
        loader = make_phase_loader(False, 0, None)
        iterator = iter(loader)
        held = next(iterator)
        loader.close_mosaic()
        future = next(iterator)

        self.assertTrue(decode_phase_batch(held)["mosaic"])
        self.assertFalse(decode_phase_batch(future)["mosaic"])

    def test_scheduled_cutover_is_assigned_before_a_held_batch_is_consumed(self):
        loader = make_phase_loader(False, 0, None)
        loader.configure_mosaic_schedule(start_ordinal=0, cutover_ordinal=1)
        iterator = iter(loader)
        held = next(iterator)
        future = next(iterator)

        self.assertTrue(decode_phase_batch(held)["mosaic"])
        self.assertFalse(decode_phase_batch(future)["mosaic"])

    def test_direct_no_aug_start_emits_no_mosaic_on_first_batch(self):
        for obb in (False, True):
            with self.subTest(obb=obb):
                loader = make_phase_loader(obb, 0, None)
                loader.configure_mosaic_schedule(start_ordinal=8, cutover_ordinal=8)
                observation = decode_phase_batch(next(iter(loader)))
                self.assertFalse(observation["mosaic"])

    def test_infinite_sampler_sample_order_is_unchanged(self):
        candidate_sampler = YoloBatchSampler(
            sampler=InfiniteSampler(size=10, shuffle=True, seed=17),
            batch_size=3,
            drop_last=False,
            input_dimension=INPUT_DIM,
            mosaic=True,
        )
        control_sampler = YoloBatchSampler(
            sampler=InfiniteSampler(size=10, shuffle=True, seed=17),
            batch_size=3,
            drop_last=False,
            input_dimension=INPUT_DIM,
            mosaic=True,
        )
        candidate_sampler.configure_mosaic_schedule(0, 8)

        candidate = list(itertools.islice(iter(candidate_sampler), 13))
        control = list(itertools.islice(iter(control_sampler), 13))
        self.assertEqual(
            [[item[1] for item in batch] for batch in candidate],
            [[item[1] for item in batch] for batch in control],
        )
        self.assertEqual(
            [batch[0][2] for batch in candidate],
            [ordinal < 8 for ordinal in range(13)],
        )

    def test_rank_ordinals_are_per_rank_not_world_interleaved(self):
        observations = []
        for rank in (0, 1):
            sampler = YoloBatchSampler(
                sampler=InfiniteSampler(
                    size=10,
                    shuffle=False,
                    seed=0,
                    rank=rank,
                    world_size=2,
                ),
                batch_size=2,
                drop_last=False,
                input_dimension=INPUT_DIM,
                mosaic=True,
            )
            sampler.configure_mosaic_schedule(0, 2)
            batches = list(itertools.islice(iter(sampler), 3))
            observations.append(
                ([item[1] for item in batches[0]], [batch[0][2] for batch in batches])
            )

        self.assertEqual(observations[0][0], [0, 2])
        self.assertEqual(observations[1][0], [1, 3])
        self.assertEqual(observations[0][1], [True, True, False])
        self.assertEqual(observations[1][1], [True, True, False])

    def test_partial_batch_consumes_one_ordinal_and_drop_last_is_unchanged(self):
        for drop_last, expected_ids, expected_flags, expected_len in (
            (
                False,
                [[0, 1], [2, 3], [4]],
                [True, True, False],
                3,
            ),
            (
                True,
                [[0, 1], [2, 3]],
                [True, True],
                2,
            ),
        ):
            with self.subTest(drop_last=drop_last):
                sampler = YoloBatchSampler(
                    sampler=SequentialSampler(range(5)),
                    batch_size=2,
                    drop_last=drop_last,
                    input_dimension=INPUT_DIM,
                    mosaic=True,
                )
                sampler.configure_mosaic_schedule(0, 2)
                batches = list(iter(sampler))
                self.assertEqual(len(sampler), expected_len)
                self.assertEqual(
                    [[item[1] for item in batch] for batch in batches], expected_ids
                )
                self.assertEqual(
                    [batch[0][2] for batch in batches], expected_flags
                )


if __name__ == "__main__":
    unittest.main()
