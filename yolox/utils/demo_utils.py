#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os

import numpy as np

__all__ = [
    "mkdir",
    "nms",
    "multiclass_nms",
    "demo_postprocess",
    "decode_obb_raw_outputs",
]


OBB_EXPORT_STRIDES = (8, 16, 32)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def _sigmoid(values):
    """Compute sigmoid without overflow warnings for large negative values."""
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _validate_obb_export_input_shape(input_shape):
    try:
        if len(input_shape) != 2:
            raise ValueError
        height, width = input_shape
    except (TypeError, ValueError):
        raise ValueError(
            "input_shape must contain exactly two integer dimensions: (height, width)"
        )

    if not all(isinstance(dimension, (int, np.integer)) for dimension in (height, width)):
        raise ValueError(
            "input_shape must contain exactly two integer dimensions: (height, width)"
        )
    if height <= 0 or width <= 0:
        raise ValueError("input_shape dimensions must be positive")
    if height != width:
        raise ValueError("OBB raw decoder supports square input shapes only")
    if height % max(OBB_EXPORT_STRIDES) != 0:
        raise ValueError("OBB input size must be divisible by 32")
    return int(height), int(width)


def decode_obb_raw_outputs(outputs, input_shape, num_classes=None):
    """Decode an exported OBB tensor while preserving its external boundary.

    The exported OBB tensor has shape ``[batch, predictions, 6 + classes]`` and
    contains ``[tx, ty, tw, th, angle_logit, objectness, classes...]``.  The
    objectness and class fields are already probabilities; only the box and
    angle fields are transformed here.  This first-slice decoder intentionally
    accepts square input shapes divisible by 32 only.
    """
    height, width = _validate_obb_export_input_shape(input_shape)
    outputs = np.asarray(outputs)
    if outputs.ndim != 3:
        raise ValueError(
            "OBB raw output must have shape [batch, predictions, fields]"
        )
    if outputs.shape[2] < 7:
        raise ValueError("OBB raw output must contain 6 + num_classes fields")
    if num_classes is not None:
        if not isinstance(num_classes, (int, np.integer)) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        expected_fields = 6 + int(num_classes)
        if outputs.shape[2] != expected_fields:
            raise ValueError(
                "OBB raw output field count {} does not match 6 + num_classes ({})".format(
                    outputs.shape[2], expected_fields
                )
            )

    expected_predictions = sum(
        (height // stride) * (width // stride) for stride in OBB_EXPORT_STRIDES
    )
    if outputs.shape[1] != expected_predictions:
        raise ValueError(
            "OBB raw output prediction count {} does not match input_shape {} (expected {})".format(
                outputs.shape[1], tuple(input_shape), expected_predictions
            )
        )

    decoded = np.array(outputs, copy=True)
    if not np.issubdtype(decoded.dtype, np.floating):
        decoded = decoded.astype(np.float32)

    offset = 0
    for stride in OBB_EXPORT_STRIDES:
        hsize = height // stride
        wsize = width // stride
        cells = hsize * wsize
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), axis=2).reshape(1, cells, 2)
        expanded_strides = np.full(
            (1, cells, 1), stride, dtype=decoded.dtype
        )
        level = slice(offset, offset + cells)
        decoded[:, level, :2] = (
            decoded[:, level, :2] + grid
        ) * expanded_strides
        decoded[:, level, 2:4] = np.exp(decoded[:, level, 2:4]) * expanded_strides
        decoded[:, level, 4] = (_sigmoid(decoded[:, level, 4]) - 0.5) * 180.0
        offset += cells

    return decoded
