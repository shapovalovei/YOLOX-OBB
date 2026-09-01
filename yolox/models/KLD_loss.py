import math

import torch
import torch.nn as nn


def _kld_compute_dtype(dtype):
    """Use at least FP32 for half-precision KLD arithmetic."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _log_positive(value):
    """Log positive values without creating an invalid zero-value derivative."""
    positive = value > 0.0
    zero = value == 0.0
    safe_value = torch.where(positive, value, torch.ones_like(value))
    logged = torch.log(safe_value)
    return torch.where(
        positive,
        logged,
        torch.where(
            zero,
            torch.full_like(value, float("-inf")),
            torch.full_like(value, float("nan")),
        ),
    )


def _log_abs_nonzero(value):
    """Log abs(value), with an exact zero represented as a zero term."""
    absolute = torch.abs(value)
    nonzero = absolute > 0.0
    safe_absolute = torch.where(nonzero, absolute, torch.ones_like(absolute))
    logged = torch.log(safe_absolute)
    return torch.where(nonzero, logged, torch.full_like(value, float("-inf")))


def _log_squared_abs(value):
    return 2.0 * _log_abs_nonzero(value)


def _log_squared_abs_ratio(numerator, denominator):
    """Log((numerator / denominator)^2) without forming the square."""
    return _log_squared_abs(numerator) - 2.0 * _log_positive(denominator)


def _log_positive_sum_with_signed_term(log_positive_sum, signed_term):
    """Compute log(exp(log_positive_sum) + signed_term) in the valid domain."""
    positive_signed_term = signed_term > 0.0
    safe_signed_term = torch.where(
        positive_signed_term, signed_term, torch.ones_like(signed_term)
    )
    log_signed_term = torch.where(
        positive_signed_term,
        torch.log(safe_signed_term),
        torch.full_like(signed_term, float("-inf")),
    )
    negative_signed_term = signed_term < 0.0
    safe_abs_signed_term = torch.where(
        negative_signed_term, torch.abs(signed_term), torch.ones_like(signed_term)
    )
    log_abs_signed_term = torch.where(
        negative_signed_term,
        torch.log(safe_abs_signed_term),
        torch.full_like(signed_term, float("-inf")),
    )

    # Around ordinary values, retain the derivative at signed_term == 0 by
    # evaluating log(P + L) as log(P) + log1p(L / P).  The guarded tensors
    # keep the unused branch finite when P is outside the compute dtype range.
    finite_log_sum = torch.isfinite(log_positive_sum)
    max_log = math.log(torch.finfo(log_positive_sum.dtype).max)
    materializable = finite_log_sum & torch.isfinite(signed_term) & (
        (log_positive_sum >= 0.0)
        & (log_positive_sum <= max_log)
    )
    direct_log_sum = torch.where(
        materializable, log_positive_sum, torch.zeros_like(log_positive_sum)
    )
    direct_positive_sum = torch.exp(direct_log_sum)
    direct_signed_term = torch.where(
        materializable, signed_term, torch.zeros_like(signed_term)
    )
    direct = direct_log_sum + torch.log1p(
        direct_signed_term / direct_positive_sum
    )

    positive_infinite_limit = (log_positive_sum == float("-inf")) & (
        signed_term == float("inf")
    )
    safe_log_positive_sum = torch.where(
        positive_infinite_limit, torch.zeros_like(log_positive_sum), log_positive_sum
    )
    safe_log_signed_term = torch.where(
        positive_infinite_limit, torch.zeros_like(log_signed_term), log_signed_term
    )
    log_plus_raw = torch.logaddexp(safe_log_positive_sum, safe_log_signed_term)
    log_plus = torch.where(
        positive_infinite_limit,
        torch.full_like(log_plus_raw, float("inf")),
        log_plus_raw,
    )
    # For valid positive dimensions, P + L = KLD + 1 >= 1.  Therefore when
    # L is negative, exp(log_abs(L) - log(P)) is strictly below one and this
    # signed log-domain subtraction is well-defined without a ratio cap.
    safe_log_positive_sum = torch.where(
        positive_infinite_limit, torch.zeros_like(log_positive_sum), log_positive_sum
    )
    safe_log_abs_signed_term = torch.where(
        positive_infinite_limit,
        torch.full_like(log_abs_signed_term, float("-inf")),
        log_abs_signed_term,
    )
    log_minus_raw = safe_log_positive_sum + torch.log1p(
        -torch.exp(safe_log_abs_signed_term - safe_log_positive_sum)
    )
    log_minus = torch.where(
        positive_infinite_limit, torch.zeros_like(log_minus_raw), log_minus_raw
    )
    log_domain = torch.where(signed_term >= 0.0, log_plus, log_minus)
    result = torch.where(materializable, direct, log_domain)
    return torch.where(
        torch.isnan(signed_term), torch.full_like(result, float("nan")), result
    )


def _compute_kld_loss(pred, target, taf):
    """Compute the maintained prediction-first KLD expression tensorwise.

    The maintained expression is KLD = P + L - 1, where P is the sum of the
    non-negative center/covariance terms and L is the sum of the logarithmic
    dimension terms.  Consequently log1p(KLD) = log(P + L).  Compute log(P)
    and combine it with L in the log domain so P never has to be materialized
    as an overflowing squared ratio sum.
    """
    result_dtype = torch.promote_types(pred.dtype, target.dtype)
    compute_dtype = _kld_compute_dtype(result_dtype)
    pred = pred.to(compute_dtype).view(-1, 5)
    target = target.to(compute_dtype).view(-1, 5)

    pred_width = pred[:, 2]
    pred_height = pred[:, 3]
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

    log_p_terms = torch.stack(
        (
            math.log(2.0)
            + _log_squared_abs_ratio(rotated_delta_x, target_width),
            math.log(2.0)
            + _log_squared_abs_ratio(rotated_delta_y, target_height),
            math.log(0.5)
            + _log_squared_abs_ratio(pred_height, target_width)
            + _log_squared_abs(sin_delta_angle),
            math.log(0.5)
            + _log_squared_abs_ratio(pred_width, target_height)
            + _log_squared_abs(sin_delta_angle),
            math.log(0.5)
            + _log_squared_abs_ratio(pred_height, target_height)
            + _log_squared_abs(cos_delta_angle),
            math.log(0.5)
            + _log_squared_abs_ratio(pred_width, target_width)
            + _log_squared_abs(cos_delta_angle),
        ),
        dim=0,
    )
    all_terms_zero = (log_p_terms == float("-inf")).all(dim=0)
    safe_log_p_terms = torch.where(
        all_terms_zero.unsqueeze(0), torch.zeros_like(log_p_terms), log_p_terms
    )
    log_p_raw = torch.logsumexp(safe_log_p_terms, dim=0)
    log_p = torch.where(
        all_terms_zero, torch.full_like(log_p_raw, float("-inf")), log_p_raw
    )

    # In exact arithmetic, the old log-ratio expression is exactly L below.
    log_l = (
        _log_positive(target_height)
        - _log_positive(pred_height)
        + _log_positive(target_width)
        - _log_positive(pred_width)
    )
    log1p_kld = _log_positive_sum_with_signed_term(log_p, log_l)
    result = 1 - 1 / (taf + log1p_kld)
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
