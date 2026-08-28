"""Regression coverage for the opt-in dynamic OBB ONNX export contract."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_demo_utils():
    path = ROOT / "yolox" / "utils" / "demo_utils.py"
    spec = importlib.util.spec_from_file_location(
        "yolox.utils.demo_utils_dynamic_export_test", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demo_utils = load_demo_utils()
decode_obb_raw_outputs = demo_utils.decode_obb_raw_outputs

try:
    import onnx
    import onnxruntime
    import torch

    from tools.export_onnx import (
        get_export_kwargs,
        get_output_field_count,
        make_parser,
        set_static_output_field_dim,
    )
    from yolox.exp.yolox_base_obb_kld import ExpOBB_KLD
    from yolox.models.network_blocks import SiLU
    from yolox.utils import replace_module

    ONNX_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - exercised by dependency absence
    onnx = None
    onnxruntime = None
    torch = None
    ONNX_IMPORT_ERROR = error

try:
    from onnxsim import simplify

    ONNXSIM_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - exercised by dependency absence
    simplify = None
    ONNXSIM_IMPORT_ERROR = error


class OBBRawDecoderTests(unittest.TestCase):
    EXPECTED = {
        320: (40, 20, 10, 2100),
        416: (52, 26, 13, 3549),
        512: (64, 32, 16, 5376),
    }

    def raw_output(self, size, batch=1, fields=7):
        return np.zeros((batch, self.EXPECTED[size][3], fields), dtype=np.float32)

    def test_supported_square_sizes_derive_prediction_counts(self):
        for size, (_, _, _, predictions) in self.EXPECTED.items():
            with self.subTest(size=size):
                decoded = decode_obb_raw_outputs(
                    self.raw_output(size), (size, size), num_classes=1
                )
                self.assertEqual(tuple(decoded.shape), (1, predictions, 7))
                self.assertTrue(np.isfinite(decoded).all())

    def test_grid_order_and_stride_follow_p3_p4_p5(self):
        raw = self.raw_output(320)
        decoded = decode_obb_raw_outputs(raw, (320, 320), num_classes=1)

        np.testing.assert_allclose(decoded[0, 0, :5], [0.0, 0.0, 8.0, 8.0, 0.0])
        np.testing.assert_allclose(decoded[0, 1, :5], [8.0, 0.0, 8.0, 8.0, 0.0])
        np.testing.assert_allclose(decoded[0, 40, :5], [0.0, 8.0, 8.0, 8.0, 0.0])

        p4_start = 40 * 40
        p5_start = p4_start + 20 * 20
        np.testing.assert_allclose(
            decoded[0, p4_start, :5], [0.0, 0.0, 16.0, 16.0, 0.0]
        )
        np.testing.assert_allclose(
            decoded[0, p5_start, :5], [0.0, 0.0, 32.0, 32.0, 0.0]
        )

    def test_xy_wh_angle_and_probability_fields_preserve_contract(self):
        raw = self.raw_output(320)
        raw[0, 0, :7] = [0.5, -0.25, np.log(2.0), np.log(3.0), np.log(3.0), 0.2, 0.8]

        decoded = decode_obb_raw_outputs(raw, (320, 320), num_classes=1)

        np.testing.assert_allclose(
            decoded[0, 0], [4.0, -2.0, 16.0, 24.0, 45.0, 0.2, 0.8],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_batch_dimension_is_preserved_without_redecoding_probabilities(self):
        raw = self.raw_output(416, batch=2)
        raw[0, :, 5:] = [0.2, 0.3]
        raw[1, :, 5:] = [0.7, 0.8]

        decoded = decode_obb_raw_outputs(raw, (416, 416), num_classes=1)

        self.assertEqual(tuple(decoded.shape), (2, 3549, 7))
        np.testing.assert_allclose(
            decoded[0, :, 5:], np.tile([0.2, 0.3], (3549, 1))
        )
        np.testing.assert_allclose(
            decoded[1, :, 5:], np.tile([0.7, 0.8], (3549, 1))
        )

    def test_malformed_field_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "field"):
            decode_obb_raw_outputs(self.raw_output(320, fields=6), (320, 320))

        with self.assertRaisesRegex(ValueError, "field count"):
            decode_obb_raw_outputs(self.raw_output(320), (320, 320), num_classes=2)

    def test_prediction_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prediction count"):
            decode_obb_raw_outputs(
                np.zeros((1, 1, 7), dtype=np.float32), (320, 320), num_classes=1
            )

    def test_non_square_and_non_multiple_shapes_are_rejected(self):
        raw = np.zeros((1, 2730, 7), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "square"):
            decode_obb_raw_outputs(raw, (320, 416), num_classes=1)

        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            decode_obb_raw_outputs(raw, (400, 400), num_classes=1)


@unittest.skipUnless(
    ONNX_IMPORT_ERROR is None,
    "ONNX/ONNX Runtime/model dependencies unavailable: {}".format(ONNX_IMPORT_ERROR),
)
class ONNXDynamicExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        exp = ExpOBB_KLD()
        exp.num_classes = 1
        exp.depth = 0.33
        exp.width = 0.125
        model = exp.get_model().eval()
        model = replace_module(model, torch.nn.SiLU, SiLU)
        model.head.decode_in_inference = False
        cls.model = model
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.static_path = os.path.join(cls.tempdir.name, "static.onnx")
        cls.dynamic_path = os.path.join(cls.tempdir.name, "dynamic.onnx")
        dummy = torch.randn(1, 3, 416, 416)

        with torch.no_grad():
            torch.onnx._export(
                model,
                dummy,
                cls.static_path,
                **get_export_kwargs("images", "output", 11),
            )
            torch.onnx._export(
                model,
                dummy,
                cls.dynamic_path,
                **get_export_kwargs("images", "output", 11, dynamic_shape=True),
            )

        if simplify is not None:
            dynamic_model = onnx.load(cls.dynamic_path)
            dynamic_model, check = simplify(dynamic_model)
            if not check:
                raise AssertionError("Simplified dynamic ONNX model could not be validated")
            onnx.save(dynamic_model, cls.dynamic_path)

        set_static_output_field_dim(
            cls.dynamic_path,
            "output",
            get_output_field_count(model.head),
        )
        cls.static_graph = onnx.load(cls.static_path)
        cls.dynamic_graph = onnx.load(cls.dynamic_path)
        cls.static_session = onnxruntime.InferenceSession(
            cls.static_path, providers=["CPUExecutionProvider"]
        )
        cls.dynamic_session = onnxruntime.InferenceSession(
            cls.dynamic_path, providers=["CPUExecutionProvider"]
        )

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    @staticmethod
    def shape_of(value_info):
        dimensions = []
        for dimension in value_info.type.tensor_type.shape.dim:
            if dimension.dim_param:
                dimensions.append(dimension.dim_param)
            else:
                dimensions.append(dimension.dim_value)
        return dimensions

    @staticmethod
    def output_for(graph):
        return graph.graph.output[0]

    def test_static_export_defaults_remain_static(self):
        parser_defaults = make_parser().parse_args([])
        self.assertFalse(parser_defaults.dynamic_shape)
        self.assertNotIn("dynamic_axes", get_export_kwargs("images", "output", 11))

        static_input = self.shape_of(self.static_graph.graph.input[0])
        static_output = self.shape_of(self.output_for(self.static_graph))
        self.assertEqual(static_input, [1, 3, 416, 416])
        self.assertEqual(static_output, [1, 3549, 7])

    def test_dynamic_graph_exposes_only_supported_symbolic_axes(self):
        dynamic_args = make_parser().parse_args(["--dynamic-shape"])
        self.assertTrue(dynamic_args.dynamic_shape)
        self.assertEqual(
            get_export_kwargs("images", "output", 11, dynamic_shape=True)["dynamic_axes"],
            {
                "images": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 1: "predictions"},
            },
        )

        dynamic_input = self.shape_of(self.dynamic_graph.graph.input[0])
        dynamic_output = self.shape_of(self.output_for(self.dynamic_graph))
        self.assertEqual(dynamic_input, ["batch", 3, "height", "width"])
        self.assertEqual(dynamic_output, ["batch", "predictions", 7])

    def test_dynamic_runtime_shapes_and_external_decode(self):
        cases = (
            (1, 320, 2100),
            (1, 416, 3549),
            (1, 512, 5376),
            (2, 416, 3549),
        )
        for batch, size, predictions in cases:
            with self.subTest(batch=batch, size=size):
                values = np.random.default_rng(batch + size).standard_normal(
                    (batch, 3, size, size), dtype=np.float32
                )
                output = self.dynamic_session.run(None, {"images": values})[0]
                self.assertEqual(tuple(output.shape), (batch, predictions, 7))
                self.assertTrue(np.isfinite(output).all())

                decoded = decode_obb_raw_outputs(
                    output, (size, size), num_classes=1
                )
                self.assertEqual(tuple(decoded.shape), (batch, predictions, 7))
                self.assertTrue(np.isfinite(decoded).all())

    def test_static_and_dynamic_416_match_eager_decode(self):
        values = np.random.default_rng(416).standard_normal(
            (1, 3, 416, 416), dtype=np.float32
        )
        with torch.no_grad():
            eager_raw = self.model(torch.from_numpy(values))
            eager_decoded = self.model.head.decode_outputs(
                eager_raw.clone(), dtype=torch.float32
            ).numpy()

        static_raw = self.static_session.run(None, {"images": values})[0]
        dynamic_raw = self.dynamic_session.run(None, {"images": values})[0]
        static_decoded = decode_obb_raw_outputs(
            static_raw, (416, 416), num_classes=1
        )
        dynamic_decoded = decode_obb_raw_outputs(
            dynamic_raw, (416, 416), num_classes=1
        )

        self.assertEqual(tuple(eager_raw.shape), (1, 3549, 7))
        np.testing.assert_allclose(static_raw, eager_raw.numpy(), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(dynamic_raw, static_raw, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(static_decoded, eager_decoded, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(dynamic_decoded, static_decoded, rtol=1e-3, atol=1e-4)
    @unittest.skipUnless(
        ONNXSIM_IMPORT_ERROR is None,
        "ONNX Simplifier unavailable: {}".format(ONNXSIM_IMPORT_ERROR),
    )
    def test_final_dynamic_artifact_is_runtime_dynamic(self):
        values = np.random.default_rng(320).standard_normal(
            (1, 3, 320, 320), dtype=np.float32
        )
        output = self.dynamic_session.run(None, {"images": values})[0]
        self.assertEqual(tuple(output.shape), (1, 2100, 7))
        decoded = decode_obb_raw_outputs(output, (320, 320), num_classes=1)
        self.assertTrue(np.isfinite(decoded).all())


if __name__ == "__main__":
    unittest.main()
