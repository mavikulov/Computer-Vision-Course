from typing import List, Tuple

import numpy as np
import torch


# Task 1 (1 point)
class QuantizationParameters:
    def __init__(
        self,
        scale: np.float64,
        zero_point: np.int32,
        q_min: np.int32,
        q_max: np.int32,
    ):
        self.scale = scale
        self.zero_point = zero_point
        self.q_min = q_min
        self.q_max = q_max

    def __repr__(self):
        return f"scale: {self.scale}, zero_point: {self.zero_point}"


def compute_quantization_params(
    r_min: np.float64,
    r_max: np.float64,
    q_min: np.int32,
    q_max: np.int32,
) -> QuantizationParameters:
    scale = ((r_max - r_min) / (q_max - q_min)).astype(np.float64)
    zero_point = np.round((r_max * q_min - r_min * q_max) / (r_max - r_min)).astype(np.int32)
    return QuantizationParameters(scale, zero_point, q_min, q_max)


# Task 2 (0.5 + 0.5 = 1 point)
def quantize(r: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    return np.clip(np.round(r / qp.scale + qp.zero_point), a_min=-128, a_max=127).astype(np.int8)


def dequantize(q: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    return (qp.scale * (q - qp.zero_point)).astype(np.float64)


# Task 3 (1 point)
class MinMaxObserver:
    def __init__(self):
        self.min = np.finfo(np.float64).max
        self.max = np.finfo(np.float64).min

    def __call__(self, x: torch.Tensor):
        self.min = min(self.min, torch.min(x).numpy().astype(np.float64))
        self.max = max(self.max, torch.max(x).numpy().astype(np.float64))


# Task 4 (1 + 1 = 2 points)
def quantize_weights_per_tensor(
    weights: np.ndarray,
) -> Tuple[np.array, QuantizationParameters]:
    r_min = np.min(weights)
    r_max = np.max(weights)
    r = max(np.abs(r_min), np.abs(r_max))
    r_min_sym, r_max_sym = -r, r 
    q_min, q_max = -127, 127
    qp = compute_quantization_params(r_min_sym, r_max_sym, q_min, q_max)
    quantized = quantize(weights, qp)
    return quantized, qp


def quantize_weights_per_channel(
    weights: np.ndarray,
) -> Tuple[np.array, List[QuantizationParameters]]:
    C_out = weights.shape[0]
    q_min = -127
    q_max = 127
    quantized_weights = np.empty_like(weights, dtype=np.int8)
    qp_list = []

    for i in range(C_out):
        w_i = weights[i]
        r_min = float(np.min(w_i))
        r_max = float(np.max(w_i))
        r = max(abs(r_min), abs(r_max)) 
        r_min_sym, r_max_sym = np.float64(-r), np.float64(r)
        qp = compute_quantization_params(r_min_sym, r_max_sym, q_min, q_max)
        quantized_channel = quantize(w_i, qp)
        quantized_weights[i] = quantized_channel
        qp_list.append(qp)

    return quantized_weights, qp_list


# Task 5 (1 point)
def quantize_bias(
    bias: np.float64,
    scale_w: np.float64,
    scale_x: np.float64,
) -> np.int32:
    return np.round(bias / (float(scale_w) * float(scale_x))).astype(np.int32)


# Task 6 (2 points)
def quantize_multiplier(m: np.float64) -> [np.int32, np.int32]:
    m_val = float(m)
    M0 = m_val
    n = 0

    while M0 < 0.5:
        M0 *= 2.0
        n += 1

    while M0 >= 1.0:
        M0 /= 2.0
        n -= 1

    M0_int = int(round(M0 * (1 << 31)))
    return np.int32(n), np.int32(M0_int)


# Task 7 (2 points)
def multiply_by_quantized_multiplier(
    accum: np.int32,
    n: np.int32,
    m0: np.int32,
) -> np.int32:
    accum_int64 = np.int64(accum)
    m0_int64 = np.int64(m0)
    P = accum_int64 * m0_int64 
    shift = 31 + n

    if shift > 0:
        offset = np.int64(1) << (shift - 1)
        P = P + offset
        res = P >> shift
    elif shift == 0:
        res = P
        res = P << (-shift)

    return np.int32(res)
