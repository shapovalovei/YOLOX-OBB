#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "--dynamic-shape",
        action="store_true",
        help="export with dynamic batch and square spatial dimensions",
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


def get_export_kwargs(input_name, output_name, opset, dynamic_shape=False):
    """Build the legacy exporter arguments without changing static defaults."""
    export_kwargs = {
        "input_names": [input_name],
        "output_names": [output_name],
        "opset_version": opset,
    }
    if dynamic_shape:
        export_kwargs["dynamic_axes"] = {
            input_name: {0: "batch", 2: "height", 3: "width"},
            output_name: {0: "batch", 1: "predictions"},
        }
    return export_kwargs


def get_output_field_count(head):
    """Return the static last-axis width for the maintained YOLOX head."""
    base_fields = 6 if hasattr(head, "angle_preds") else 5
    return base_fields + head.num_classes


def set_static_output_field_dim(model_path, output_tensor_name, field_count):
    """Make the known field axis explicit after legacy dynamic export."""
    import onnx

    onnx_model = onnx.load(model_path)
    matching_outputs = [
        value_info
        for value_info in onnx_model.graph.output
        if value_info.name == output_tensor_name
    ]
    if len(matching_outputs) != 1:
        raise ValueError("could not identify the exported output tensor")

    output_shape = matching_outputs[0].type.tensor_type.shape
    if len(output_shape.dim) != 3:
        raise ValueError("expected an exported detection tensor with rank 3")
    field_dim = output_shape.dim[2]
    if field_dim.dim_value not in (0, int(field_count)):
        raise ValueError(
            "exported output field dimension {} does not match {}".format(
                field_dim.dim_value, field_count
            )
        )
    field_dim.ClearField("dim_param")
    field_dim.dim_value = int(field_count)
    onnx.save(onnx_model, model_path)


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1])
    torch.onnx._export(
        model,
        dummy_input,
        args.output_name,
        **get_export_kwargs(
            args.input,
            args.output,
            args.opset,
            dynamic_shape=args.dynamic_shape,
        )
    )
    logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))

    if args.dynamic_shape:
        set_static_output_field_dim(
            args.output_name,
            args.output,
            get_output_field_count(model.head),
        )
        logger.info("preserved static output field dimension in {}".format(args.output_name))


if __name__ == "__main__":
    main()
