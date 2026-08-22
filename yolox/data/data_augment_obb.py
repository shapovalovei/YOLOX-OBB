#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates


def obb_to_corners(center_x, center_y, width, height, angle):
    """Decode the fork's (cx, cy, long_w, short_h, angle_deg) contract."""
    radians = math.radians(float(angle))
    ux, uy = math.cos(radians), math.sin(radians)
    vx, vy = -uy, ux
    half_width, half_height = float(width) / 2.0, float(height) / 2.0
    return np.asarray(
        [
            (center_x - half_width * ux - half_height * vx,
             center_y - half_width * uy - half_height * vy),
            (center_x + half_width * ux - half_height * vx,
             center_y + half_width * uy - half_height * vy),
            (center_x + half_width * ux + half_height * vx,
             center_y + half_width * uy + half_height * vy),
            (center_x - half_width * ux + half_height * vx,
             center_y - half_width * uy + half_height * vy),
        ],
        dtype=np.float64,
    )


def corners_to_obb(corners):
    """Fit the canonical fork OBB to a transformed quadrilateral."""
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 4 or not np.isfinite(points).all():
        raise ValueError("OBB corners are not finite")
    rect = cv2.minAreaRect(points)
    rectangle = cv2.boxPoints(rect).astype(np.float64)
    edges = np.roll(rectangle, -1, axis=0) - rectangle
    lengths = np.linalg.norm(edges, axis=1)
    long_index = int(np.argmax(lengths))
    long_edge = float(lengths[long_index])
    short_edge = float(
        (lengths[(long_index - 1) % 4] + lengths[(long_index + 1) % 4]) / 2.0
    )
    if not np.isfinite([long_edge, short_edge]).all() or min(long_edge, short_edge) <= 1e-6:
        raise ValueError("OBB corners are degenerate")
    direction = edges[long_index]
    angle = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    center_x, center_y = rectangle.mean(axis=0)
    return np.asarray([center_x, center_y, long_edge, short_edge, angle], dtype=np.float64)


def _obb_target_to_corners(target):
    center_x = (target[0] + target[2]) * 0.5
    center_y = (target[1] + target[3]) * 0.5
    width = target[2] - target[0]
    height = target[3] - target[1]
    return obb_to_corners(center_x, center_y, width, height, target[4])


def _obb_to_target(obb, class_value):
    center_x, center_y, width, height, angle = obb
    return np.asarray(
        [
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
            angle,
            class_value,
        ],
        dtype=np.float64,
    )


def random_perspective(
    img,
    targets=(),
    degrees=0.0,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Transform label coordinates. The first four target values encode the
    # center and long/short dimensions that TrainTransformOBB later converts
    # with xyxy2cxcywh; they are not axis-aligned polygon corners.
    n = len(targets)
    if n:
        source_corners = np.asarray(
            [_obb_target_to_corners(target) for target in targets], dtype=np.float64
        )
        xy = np.ones((n * 4, 3), dtype=np.float64)
        xy[:, :2] = source_corners.reshape(n * 4, 2)
        # Apply the exact image transform to the OBB corners.
        xy = xy @ M.T  # transform
        if perspective:
            homogeneous_w = xy[:, 2].reshape(n, 4)
            valid_w = (
                np.isfinite(homogeneous_w).all(axis=1)
                & (np.abs(homogeneous_w) > 1e-8).all(axis=1)
                & (
                    (homogeneous_w > 1e-8).all(axis=1)
                    | (homogeneous_w < -1e-8).all(axis=1)
                )
            )
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 4, 2)  # rescale
        else:  # affine
            valid_w = np.isfinite(xy).all(axis=1).reshape(n, 4).all(axis=1)
            xy = xy[:, :2].reshape(n, 4, 2)

        # Retain HBB candidate filtering, but reconstruct the final OBB from
        # the transformed polygon rather than retaining the source angle.
        xy_hbb = np.concatenate(
            (
                xy[:, :, 0].min(1),
                xy[:, :, 1].min(1),
                xy[:, :, 0].max(1),
                xy[:, :, 1].max(1),
            )
        ).reshape(4, n).T

        # Clip the polygon used for reconstruction consistently with the
        # existing augmentation's clipped candidate-box behavior.
        clipped_xy = xy.copy()
        clipped_xy[:, :, 0] = clipped_xy[:, :, 0].clip(0, width)
        clipped_xy[:, :, 1] = clipped_xy[:, :, 1].clip(0, height)
        xy_hbb[:, [0, 2]] = xy_hbb[:, [0, 2]].clip(0, width)
        xy_hbb[:, [1, 3]] = xy_hbb[:, [1, 3]].clip(0, height)

        # filter candidates
        keep = valid_w & box_candidates(box1=targets[:, :4].T * s, box2=xy_hbb.T)
        updated_targets = []
        for index in np.flatnonzero(keep):
            try:
                obb = corners_to_obb(clipped_xy[index])
            except ValueError:
                continue
            if not np.isfinite(obb).all() or min(obb[2], obb[3]) <= 1e-6:
                continue
            updated_targets.append(_obb_to_target(obb, targets[index, 5]))
        if updated_targets:
            targets = np.asarray(updated_targets, dtype=targets.dtype)
        else:
            targets = np.empty((0, targets.shape[1]), dtype=targets.dtype)

    return img, targets


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _mirror(image, boxes): #水平翻转
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes

def _flip_h(image, boxes, angles): #水平翻转
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        angles = angles.copy()
        angles[:] = 0.0 - angles[:]
        angles[angles[:]==90.0] = -90.0
    return image, boxes, angles

def _flip_v(image, boxes, angles): #垂直翻转
    height, _, _ = image.shape
    if random.randrange(2):
        image = image[::-1, :]
        boxes = boxes.copy()
        boxes[:, 1::2] = height - boxes[:, 3::-2]
        angles = angles.copy()
        angles[:] = 0.0 - angles[:]
        angles[angles[:] == 90.0] = -90.0
    return image, boxes, angles



def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1] # 将bgr转化为rgb
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap) # 将rgb转化为bgr
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransformOBB:
    def __init__(self, p=0.5, rgb_means=None, std=None, max_labels=50):
        self.means = rgb_means
        self.std = std
        self.p = p
        self.max_labels = max_labels

    def __call__(self, image, targets, input_dim): # target  [[xmin, ymin, xmax, ymax, angle, label_ind], ... ]
        boxes = targets[:, :4].copy()
        labels = targets[:, 5].copy() # modify  labels = targets[:, 4].copy()
        angles = targets[:, 4].copy()  #add
        if len(boxes) == 0:
            #targets = np.zeros((self.max_labels, 5), dtype=np.float32) # delete
            targets = np.zeros((self.max_labels, 6), dtype=np.float32) #add
            image, r_o = preproc(image, input_dim, self.means, self.std)
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 5] # modify labels_o = targets_o[:, 4]
        angles_o = targets_o[:, 4] #add
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o) #[c_x,c_y,w,h]

        image_t = _distort(image) # distort
        #image_t, boxes = _mirror(image_t, boxes) # 50%概率水平翻转
        image_t, boxes, angles = _flip_h(image_t, boxes, angles) # 50%概率水平翻转 , 翻转后角度要变
        image_t, boxes, angles = _flip_v(image_t, boxes, angles) # 50%概率垂直翻转 , 翻转后角度要变
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim, self.means, self.std)
        # image_t：resize且padding后的图片, r_:resize的比例
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes) #[c_x,c_y,w,h]
        boxes *= r_ #缩放boxes

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 4 #如果bbox的长或者宽小于8，那么忽略这个bbox
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        angles_t = angles[mask_b] #add

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim, self.means, self.std)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            angles_t = angles_o #add

        labels_t = np.expand_dims(labels_t, 1)
        angles_t = np.expand_dims(angles_t, 1) #add

        # targets_t = np.hstack((labels_t, boxes_t)) #delete
        targets_t = np.hstack((labels_t, boxes_t, angles_t))
        # padded_labels = np.zeros((self.max_labels, 5)) #delete
        padded_labels = np.zeros((self.max_labels, 6)) #add
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ] #保留前self.max_labels个标注
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        image_t = np.ascontiguousarray(image_t, dtype=np.float32)
        return image_t, padded_labels
        # padded_labels：  [[label_id, c_x, c_y, w, h, angle], ...]


class ValTransformOBB:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1)):
        self.means = rgb_means
        self.swap = swap
        self.std = std

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.means, self.std, self.swap)
        return img, np.zeros((1, 6))
