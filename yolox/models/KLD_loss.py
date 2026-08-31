import math

import torch
import torch.nn as nn


def _kld_compute_dtype(dtype):
    """Use at least FP32 for half-precision KLD arithmetic."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _condition_prediction_dimensions(dimensions, floor):
    """Protect decoded zero/subnormal predictions without repairing negatives."""
    conditioned = torch.clamp_min(dimensions, floor)
    return torch.where(dimensions < 0.0, dimensions, conditioned)


def _squared_ratio(numerator, denominator, ratio_limit):
    """Square a ratio while keeping an unrepresentable ratio finite."""
    ratio = torch.clamp(numerator / denominator, -ratio_limit, ratio_limit)
    return torch.square(ratio)


def _compute_kld_loss(pred, target, taf):
    """Compute the maintained prediction-first KLD expression tensorwise.

    The dimension ratios and logarithms are algebraically equivalent to the
    original expression for positive finite dimensions.  The ratio limit is a
    defensive conditioning rule only for squared ratios that cannot be
    represented by the compute dtype; it is inactive for ordinary boxes.
    """
    result_dtype = torch.promote_types(pred.dtype, target.dtype)
    compute_dtype = _kld_compute_dtype(result_dtype)
    pred = pred.to(compute_dtype).view(-1, 5)
    target = target.to(compute_dtype).view(-1, 5)

    dimension_floor = torch.finfo(compute_dtype).tiny
    pred_dimensions = _condition_prediction_dimensions(
        pred[:, 2:4], dimension_floor
    )
    pred_width = pred_dimensions[:, 0]
    pred_height = pred_dimensions[:, 1]
    target_width = target[:, 2]
    target_height = target[:, 3]

    delta_x = pred[:, 0] - target[:, 0]
    delta_y = pred[:, 1] - target[:, 1]
    pred_angle_radian = math.pi * pred[:, 4] / 180.0
    target_angle_radian = math.pi * target[:, 4] / 180.0
    delta_angle_radian = pred_angle_radian - target_angle_radian

    rotated_delta_x = delta_x * torch.cos(target_angle_radian) + delta_y * torch.sin(
        target_angle_radian
    )
    rotated_delta_y = delta_y * torch.cos(target_angle_radian) - delta_x * torch.sin(
        target_angle_radian
    )
    sin_delta_angle = torch.sin(delta_angle_radian)
    cos_delta_angle = torch.cos(delta_angle_radian)

    # In exact arithmetic, a^2 / b^2 == (a / b)^2.  Keep enough headroom
    # for the weighted sum of all squared terms when the ratio is extreme.
    ratio_limit = math.sqrt(torch.finfo(compute_dtype).max / 16.0)
    kld = 0.5 * (
        4 * _squared_ratio(rotated_delta_x, target_width, ratio_limit)
        + 4 * _squared_ratio(rotated_delta_y, target_height, ratio_limit)
    ) + 0.5 * (
        _squared_ratio(pred_height, target_width, ratio_limit)
        * torch.square(sin_delta_angle)
        + _squared_ratio(pred_width, target_height, ratio_limit)
        * torch.square(sin_delta_angle)
        + _squared_ratio(pred_height, target_height, ratio_limit)
        * torch.square(cos_delta_angle)
        + _squared_ratio(pred_width, target_width, ratio_limit)
        * torch.square(cos_delta_angle)
    ) + (
        torch.log(target_height)
        - torch.log(pred_height)
        + torch.log(target_width)
        - torch.log(pred_width)
    ) - 1.0

    # In exact arithmetic, log(b^2 / a^2) == 2 * (log(b) - log(a)).
    # log1p is the same outer mapping as log(kld + 1), with better behavior
    # when ordinary boxes make kld close to zero.
    result = 1 - 1 / (taf + torch.log1p(kld))
    # Preserve the dtype callers received before the internal promotion.
    return result.to(result_dtype)


class KLDloss(nn.Module):
    def __init__(self, taf=1.0, reduction="none"):
        super(KLDloss, self).__init__()
        self.reduction = reduction
        self.taf = taf

    def forward(self, pred, target): # pred [[x,y,w,h,angle], ...]
        assert pred.shape[0] == target.shape[0]
        return _compute_kld_loss(pred, target, self.taf)


def compute_kld_loss(targets, preds):
    with torch.no_grad():
        kld_loss_ts_ps = torch.zeros(0, preds.shape[0], device=targets.device)
        for target in targets:
            target = target.unsqueeze(0).repeat(preds.shape[0], 1)
            kld_loss_t_p = kld_loss(preds, target)
            kld_loss_ts_ps = torch.cat((kld_loss_ts_ps, kld_loss_t_p.unsqueeze(0)), dim=0)
    return kld_loss_ts_ps


def kld_loss(pred, target, taf=1.0):  # pred [[x,y,w,h,angle], ...]
    assert pred.shape[0] == target.shape[0]
    return _compute_kld_loss(pred, target, taf)

# loss = KLDloss()
# pred = torch.tensor([[20, 20, 10, 10, -90], [20, 20, 20, 10, 90], [1, 0.5, 2, 1, 0]], dtype=torch.float32)
# target = torch.tensor([[20, 20, 10, 10, -90], [20, 20, 20, 10, 0], [0.5, 1, 2, 1, -90]], dtype=torch.float32)
# kld = kld_loss(pred, target)
# print(kld)


# pred = torch.tensor([[20, 20, 10, 10, -90], [20, 20, 20, 10, 90], [1, 0.5, 2, 1, 0]], dtype=torch.float32)
# target = torch.tensor([[20, 20, 10, 10, -90], [20, 20, 20, 10, 0]], dtype=torch.float32)
# kld = compute_kld_loss(target, pred)
# print(kld)
#
# print(torch.floor(torch.tensor(-9.9)))
