"""Regression coverage for runtime dimensions through enabled Mosaic."""

import unittest

import numpy as np
from torch.utils.data import SequentialSampler

from yolox.data import DataLoader, TrainTransform, TrainTransformOBB, YoloBatchSampler
from yolox.data.datasets.datasets_wrapper import Dataset
from yolox.data.datasets.mosaicdetection import MosaicDetection
from yolox.data.datasets.mosaicdetection_obb import MosaicDetectionOBB


INITIAL_DIM = (416, 416)
SOURCE_HW = (240, 360)
SOURCE_CENTER = (300.0, 180.0)
SOURCE_SIZE = (100.0, 60.0)
SOURCE_ANGLE = 25.0
SOURCE_CLASS = 3.0
TRANSITIONS = (
    (512, 512),
    (320, 512),
    (512, 320),
    (384, 640),
    (640, 384),
)
REPEATED_TRANSITIONS = ((384, 640), (640, 384), (512, 512))


class SyntheticDataset(Dataset):
    """Small source dataset with deterministic HBB or OBB geometry."""

    def __init__(self, obb):
        super().__init__(INITIAL_DIM, mosaic=False)
        self.obb = obb
        self.image = np.zeros((*SOURCE_HW, 3), dtype=np.uint8)
        center_x, center_y = SOURCE_CENTER
        width, height = SOURCE_SIZE
        x1, x2 = center_x - width / 2.0, center_x + width / 2.0
        y1, y2 = center_y - height / 2.0, center_y + height / 2.0
        if obb:
            self.target = np.asarray(
                [[x1, y1, x2, y2, SOURCE_ANGLE, SOURCE_CLASS]],
                dtype=np.float32,
            )
        else:
            self.target = np.asarray(
                [[x1, y1, x2, y2, SOURCE_CLASS]],
                dtype=np.float32,
            )

    def __len__(self):
        return 8

    def pull_item(self, index):
        return self.image.copy(), self.target.copy(), SOURCE_HW, index

    def load_anno(self, index):
        return self.target.copy()


class RecordingPreproc:
    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = []

    def __call__(self, image, targets, input_dim):
        self.calls.append(
            {
                "image_shape": tuple(image.shape[:2]),
                "input_dim": tuple(input_dim),
                "targets": targets.copy(),
            }
        )
        return self.delegate(image, targets, input_dim)


class MosaicTrace:
    def __init__(self):
        self.resize_calls = []
        self.random_perspective_calls = []
        self.uniform_calls = []
        self.mixup_dims = []


def make_loader(obb, trace, mosaic=True, enable_mixup=False):
    dataset = SyntheticDataset(obb)
    if obb:
        delegate = TrainTransformOBB(p=0.0, max_labels=8)
        wrapper_class = MosaicDetectionOBB
    else:
        delegate = TrainTransform(p=0.0, max_labels=8)
        wrapper_class = MosaicDetection
    preproc = RecordingPreproc(delegate)
    wrapper = wrapper_class(
        dataset,
        INITIAL_DIM,
        mosaic=mosaic,
        preproc=preproc,
        degrees=0.0,
        translate=0.0,
        scale=(1.0, 1.0),
        mscale=(1.0, 1.0),
        shear=0.0,
        perspective=0.0,
        enable_mixup=enable_mixup,
    )
    sampler = SequentialSampler(wrapper)
    batch_sampler = YoloBatchSampler(
        sampler=sampler,
        batch_size=1,
        drop_last=False,
        input_dimension=INITIAL_DIM,
        mosaic=mosaic,
    )
    loader = DataLoader(wrapper, batch_sampler=batch_sampler, num_workers=0)

    module = __import__(
        "yolox.data.datasets.mosaicdetection_obb" if obb
        else "yolox.data.datasets.mosaicdetection",
        fromlist=["random_perspective"],
    )
    original_resize = module.cv2.resize
    original_random_perspective = module.random_perspective

    def resize(image, size, *args, **kwargs):
        trace.resize_calls.append(
            {"source_shape": tuple(image.shape[:2]), "size": tuple(size)}
        )
        return original_resize(image, size, *args, **kwargs)

    def random_perspective(image, targets=(), **kwargs):
        event = {
            "input_shape": tuple(image.shape[:2]),
            "input_targets": targets.copy(),
            "border": tuple(kwargs["border"]),
        }
        output_image, output_targets = original_random_perspective(
            image, targets, **kwargs
        )
        event["output_shape"] = tuple(output_image.shape[:2])
        event["output_targets"] = output_targets.copy()
        trace.random_perspective_calls.append(event)
        return output_image, output_targets

    module.random_perspective = random_perspective
    module.cv2.resize = resize

    original_uniform = module.random.uniform
    original_randint = module.random.randint
    original_randrange = module.random.randrange

    def uniform(a, b):
        trace.uniform_calls.append((a, b))
        return (a + b) / 2.0

    module.random.uniform = uniform
    module.random.randint = lambda *args, **kwargs: 0
    module.random.randrange = lambda *args, **kwargs: 0

    def close():
        module.random.uniform = original_uniform
        module.random.randint = original_randint
        module.random.randrange = original_randrange
        module.cv2.resize = original_resize
        module.random_perspective = original_random_perspective

    return loader, dataset, wrapper, preproc, close


def expected_first_target_center(input_dim):
    input_h, input_w = input_dim
    source_h, source_w = SOURCE_HW
    scale = min(input_h / source_h, input_w / source_w)
    resized_w = int(source_w * scale)
    resized_h = int(source_h * scale)
    center_x = input_w - resized_w + scale * SOURCE_CENTER[0] - input_w / 2.0
    center_y = input_h - resized_h + scale * SOURCE_CENTER[1] - input_h / 2.0
    return np.asarray((center_x, center_y), dtype=np.float64)


class MosaicRuntimeDimensionTests(unittest.TestCase):
    def assert_batch_shape(self, batch, input_dim):
        image = batch[0]
        self.assertEqual(tuple(image.shape), (1, 3, input_dim[0], input_dim[1]))

    def assert_transition(self, obb, input_dim):
        trace = MosaicTrace()
        loader, dataset, wrapper, preproc, close = make_loader(obb, trace)
        try:
            iterator = iter(loader)
            self.assert_batch_shape(next(iterator), INITIAL_DIM)
            self.assertEqual(loader.change_input_dim(input_dim, random_range=None), input_dim)
            batch = next(iterator)
        finally:
            close()

        self.assert_batch_shape(batch, input_dim)
        self.assertFalse(hasattr(wrapper, "_input_dim"))
        self.assertEqual(wrapper.input_dim, INITIAL_DIM)
        self.assertEqual(dataset.input_dim, INITIAL_DIM)

        self.assertEqual(len(trace.random_perspective_calls), 2)
        event = trace.random_perspective_calls[-1]
        input_h, input_w = input_dim
        self.assertEqual(
            trace.uniform_calls[-8:-6],
            [
                (0.5 * input_h, 1.5 * input_h),
                (0.5 * input_w, 1.5 * input_w),
            ],
        )
        self.assertEqual(event["input_shape"], (2 * input_h, 2 * input_w))
        self.assertEqual(event["border"], (-input_h // 2, -input_w // 2))
        self.assertEqual(event["output_shape"], input_dim)

        source_resizes = [
            call for call in trace.resize_calls if call["source_shape"] == SOURCE_HW
        ]
        self.assertEqual(len(source_resizes), 8)
        scale = min(input_h / SOURCE_HW[0], input_w / SOURCE_HW[1])
        expected_source_size = (
            int(SOURCE_HW[1] * scale),
            int(SOURCE_HW[0] * scale),
        )
        self.assertEqual(
            [call["size"] for call in source_resizes[-4:]],
            [expected_source_size] * 4,
        )

        preproc_call = preproc.calls[-1]
        self.assertEqual(preproc_call["input_dim"], input_dim)
        self.assertEqual(preproc_call["image_shape"], input_dim)
        target = preproc_call["targets"][0]
        target_center = np.asarray(((target[0] + target[2]) / 2.0, (target[1] + target[3]) / 2.0))
        np.testing.assert_allclose(
            target_center,
            expected_first_target_center(input_dim),
            rtol=0.0,
            atol=1e-4,
        )
        if obb:
            self.assertAlmostEqual(float(target[4]), SOURCE_ANGLE, places=4)
            self.assertAlmostEqual(float(target[5]), SOURCE_CLASS, places=4)
        else:
            self.assertAlmostEqual(float(target[4]), SOURCE_CLASS, places=4)

    def test_hbb_enabled_mosaic_runtime_dimension_matrix(self):
        for input_dim in TRANSITIONS:
            with self.subTest(input_dim=input_dim):
                self.assert_transition(False, input_dim)

    def test_obb_enabled_mosaic_runtime_dimension_matrix(self):
        for input_dim in TRANSITIONS:
            with self.subTest(input_dim=input_dim):
                self.assert_transition(True, input_dim)

    def assert_repeated_transitions(self, obb):
        trace = MosaicTrace()
        loader, dataset, wrapper, preproc, close = make_loader(obb, trace)
        try:
            iterator = iter(loader)
            self.assert_batch_shape(next(iterator), INITIAL_DIM)
            for input_dim in REPEATED_TRANSITIONS:
                loader.change_input_dim(input_dim, random_range=None)
                batch = next(iterator)
                self.assert_batch_shape(batch, input_dim)
                event = trace.random_perspective_calls[-1]
                self.assertEqual(event["input_shape"], (2 * input_dim[0], 2 * input_dim[1]))
                self.assertEqual(event["output_shape"], input_dim)
                self.assertEqual(preproc.calls[-1]["input_dim"], input_dim)
                self.assertEqual(dataset.input_dim, INITIAL_DIM)
                self.assertFalse(hasattr(wrapper, "_input_dim"))
        finally:
            close()

    def test_hbb_enabled_mosaic_repeated_transitions(self):
        self.assert_repeated_transitions(False)

    def test_obb_enabled_mosaic_repeated_transitions(self):
        self.assert_repeated_transitions(True)

    def assert_close_mosaic_boundary(self, obb, schedule_before_close):
        trace = MosaicTrace()
        loader, dataset, wrapper, preproc, close = make_loader(obb, trace)
        try:
            iterator = iter(loader)
            self.assert_batch_shape(next(iterator), INITIAL_DIM)
            input_dim = (384, 640) if schedule_before_close else (512, 320)
            if schedule_before_close:
                loader.change_input_dim(input_dim, random_range=None)
                loader.close_mosaic()
            else:
                loader.close_mosaic()
                loader.change_input_dim(input_dim, random_range=None)
            batch = next(iterator)
        finally:
            close()

        self.assert_batch_shape(batch, input_dim)
        self.assertFalse(loader.batch_sampler.mosaic)
        self.assertEqual(dataset.input_dim, input_dim)
        self.assertFalse(hasattr(wrapper, "_input_dim"))
        self.assertEqual(len(trace.random_perspective_calls), 1)
        self.assertEqual(preproc.calls[-1]["input_dim"], input_dim)
        self.assertEqual(preproc.calls[-1]["image_shape"], SOURCE_HW)

    def test_hbb_close_mosaic_phase_boundaries(self):
        self.assert_close_mosaic_boundary(False, schedule_before_close=False)
        self.assert_close_mosaic_boundary(False, schedule_before_close=True)

    def test_obb_close_mosaic_phase_boundaries(self):
        self.assert_close_mosaic_boundary(True, schedule_before_close=False)
        self.assert_close_mosaic_boundary(True, schedule_before_close=True)

    def assert_no_mosaic_transition(self, obb):
        trace = MosaicTrace()
        loader, dataset, wrapper, preproc, close = make_loader(obb, trace, mosaic=False)
        try:
            iterator = iter(loader)
            self.assert_batch_shape(next(iterator), INITIAL_DIM)
            input_dim = (512, 320)
            loader.change_input_dim(input_dim, random_range=None)
            batch = next(iterator)
        finally:
            close()

        self.assert_batch_shape(batch, input_dim)
        self.assertEqual(dataset.input_dim, input_dim)
        self.assertEqual(len(trace.random_perspective_calls), 0)
        self.assertEqual(preproc.calls[-1]["input_dim"], input_dim)
        self.assertFalse(hasattr(wrapper, "_input_dim"))

        labels = batch[1][0, 0].numpy()
        scale = min(input_dim[0] / SOURCE_HW[0], input_dim[1] / SOURCE_HW[1])
        np.testing.assert_allclose(
            labels[1:3], np.asarray(SOURCE_CENTER) * scale, rtol=0.0, atol=1e-4
        )
        if obb:
            self.assertAlmostEqual(float(labels[5]), SOURCE_ANGLE, places=4)
        else:
            self.assertAlmostEqual(float(labels[0]), SOURCE_CLASS, places=4)

    def test_hbb_no_mosaic_transition_remains_current(self):
        self.assert_no_mosaic_transition(False)

    def test_obb_no_mosaic_transition_remains_current(self):
        self.assert_no_mosaic_transition(True)

    def assert_mixup_dimension(self, obb):
        trace = MosaicTrace()
        loader, dataset, wrapper, preproc, close = make_loader(
            obb, trace, enable_mixup=True
        )

        def record_mixup(origin_image, origin_labels, input_dim):
            trace.mixup_dims.append(tuple(input_dim))
            return origin_image, origin_labels

        wrapper.mixup = record_mixup
        try:
            iterator = iter(loader)
            self.assert_batch_shape(next(iterator), INITIAL_DIM)
            input_dim = (384, 640)
            loader.change_input_dim(input_dim, random_range=None)
            batch = next(iterator)
        finally:
            close()

        self.assert_batch_shape(batch, input_dim)
        self.assertEqual(trace.mixup_dims, [INITIAL_DIM, input_dim])
        self.assertEqual(preproc.calls[-1]["input_dim"], input_dim)

    def test_hbb_mixup_receives_active_runtime_dimension(self):
        self.assert_mixup_dimension(False)

    def test_obb_mixup_receives_active_runtime_dimension(self):
        self.assert_mixup_dimension(True)


if __name__ == "__main__":
    unittest.main()
