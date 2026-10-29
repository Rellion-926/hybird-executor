from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def fused_qknorm_rope(
    q_ptr, k_ptr, gamma_q_ptr, gamma_k_ptr, sin_ptr, cos_ptr, combined_out_ptr,
    Q_HEADS: tl.constexpr, K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, ROTARY_HALF: tl.constexpr, EPSILON: tl.constexpr,
):
    token = tl.program_id(0)
    combined_head = tl.program_id(1)
    is_q = combined_head < Q_HEADS
    q_head = tl.minimum(combined_head, Q_HEADS - 1)
    k_head = tl.maximum(combined_head - Q_HEADS, 0)
    lane = tl.arange(0, HEAD_DIM)
    partner_lane = tl.where(lane < ROTARY_HALF, lane + ROTARY_HALF, lane - ROTARY_HALF)
    q_base = (token * Q_HEADS + q_head) * HEAD_DIM
    k_base = (token * K_HEADS + k_head) * HEAD_DIM
    q = tl.load(q_ptr + q_base + lane).to(tl.float32)
    k = tl.load(k_ptr + k_base + lane).to(tl.float32)
    q_partner = tl.load(q_ptr + q_base + partner_lane).to(tl.float32)
    k_partner = tl.load(k_ptr + k_base + partner_lane).to(tl.float32)
    x = tl.where(is_q, q, k)
    x_partner = tl.where(is_q, q_partner, k_partner)
    gamma = tl.where(is_q, tl.load(gamma_q_ptr + lane), tl.load(gamma_k_ptr + lane)).to(tl.float32)
    gamma_partner = tl.where(is_q, tl.load(gamma_q_ptr + partner_lane), tl.load(gamma_k_ptr + partner_lane)).to(tl.float32)
    inv_rms = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / HEAD_DIM + EPSILON)
    norm = (x * inv_rms * gamma).to(tl.float16).to(tl.float32)
    norm_partner = (x_partner * inv_rms * gamma_partner).to(tl.float16).to(tl.float32)
    trig_lane = lane % ROTARY_HALF
    sin_value = tl.load(sin_ptr + token * ROTARY_HALF + trig_lane)
    cos_value = tl.load(cos_ptr + token * ROTARY_HALF + trig_lane)
    rotated = tl.where(lane < ROTARY_HALF, -norm_partner, norm_partner)
    result = (norm * cos_value + rotated * sin_value).to(tl.float16)
    combined_base = (token * (Q_HEADS + K_HEADS) + combined_head) * HEAD_DIM
    tl.store(combined_out_ptr + combined_base + lane, result)
