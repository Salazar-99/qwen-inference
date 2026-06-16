"""CUDA/Triton kernels for the custom inference backend.

The kernel in this module targets the decode-time shape that shows up as the
remaining hot spot after the body projections are served by Marlin: a small
batch of hidden states multiplied by a very tall output matrix, such as
``lm_head``.  We keep weights packed as row-major uint32 INT4 values and
dequantize only the tile currently being reduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only/dev installs
    triton = None
    tl = None


@dataclass(frozen=True)
class Int4Weight:
    """Row-major packed INT4 weight metadata.

    Attributes:
        qweight: ``uint32`` tensor shaped ``[out_features, ceil(in_features / 8)]``.
            Element ``k`` of a row is stored in bits ``4 * (k % 8)`` of word
            ``k // 8``.
        scales: floating tensor shaped ``[out_features, ceil(in_features / group_size)]``.
        zeros: optional zero-points with the same shape as ``scales``.  For
            unsigned GPTQ/AWQ-style quantization, dequantization is
            ``(q - zero) * scale``.  For signed symmetric quantization set
            ``signed=True`` and leave ``zeros=None``.
        in_features: logical input dimension before padding to a multiple of 8.
        group_size: number of input columns sharing one scale/zero.
        signed: interpret nibbles as signed 4-bit values in ``[-8, 7]``.
    """

    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor | None
    in_features: int
    group_size: int = 128
    signed: bool = False


def int4_gemv(
    x: torch.Tensor,
    weight: Int4Weight,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute ``x @ dequant(weight).T`` with fused INT4 dequantization.

    ``x`` may be ``[K]`` or ``[B, K]`` and should be CUDA FP16/BF16 for the
    Triton path.  CPU tensors and unavailable Triton fall back to a reference
    implementation, which is useful for tests and shape validation.
    """

    if x.ndim == 1:
        x_2d = x[None, :]
        squeeze = True
    elif x.ndim == 2:
        x_2d = x
        squeeze = False
    else:
        raise ValueError(f"x must be rank 1 or 2, got shape {tuple(x.shape)}")

    _validate_int4_inputs(x_2d, weight)
    output_dtype = output_dtype or x_2d.dtype

    if (
        triton is None
        or not x_2d.is_cuda
        or not weight.qweight.is_cuda
        or not weight.scales.is_cuda
    ):
        out = _int4_gemv_reference(x_2d, weight).to(output_dtype)
        return out[0] if squeeze else out

    qweight = weight.qweight.contiguous()
    scales = weight.scales.contiguous()
    zeros = (
        weight.zeros.contiguous()
        if weight.zeros is not None
        else torch.empty((1, 1), device=x_2d.device, dtype=weight.scales.dtype)
    )
    out_features = qweight.shape[0]
    out = torch.empty(
        (x_2d.shape[0], out_features),
        device=x_2d.device,
        dtype=output_dtype,
    )

    block_n = 32
    block_k = 256
    grid = (x_2d.shape[0], triton.cdiv(out_features, block_n))
    _int4_gemv_kernel[grid](
        x_2d,
        qweight,
        scales,
        zeros,
        out,
        x_2d.shape[0],
        weight.in_features,
        out_features,
        qweight.shape[1],
        scales.shape[1],
        weight.group_size,
        x_2d.stride(0),
        x_2d.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_ZEROS=weight.zeros is not None,
        SIGNED=weight.signed,
        num_warps=4,
        num_stages=4,
    )
    return out[0] if squeeze else out


def _validate_int4_inputs(x: torch.Tensor, weight: Int4Weight) -> None:
    if x.shape[1] != weight.in_features:
        raise ValueError(
            f"x has K={x.shape[1]} but weight.in_features={weight.in_features}"
        )
    if weight.qweight.ndim != 2:
        raise ValueError("qweight must be shaped [out_features, packed_k]")
    if weight.qweight.dtype != torch.uint32:
        raise TypeError(f"qweight must be torch.uint32, got {weight.qweight.dtype}")
    expected_packed_k = (weight.in_features + 7) // 8
    if weight.qweight.shape[1] != expected_packed_k:
        raise ValueError(
            f"qweight packed K mismatch: expected {expected_packed_k}, "
            f"got {weight.qweight.shape[1]}"
        )
    if weight.group_size <= 0:
        raise ValueError("group_size must be positive")
    expected_groups = (weight.in_features + weight.group_size - 1) // weight.group_size
    if weight.scales.shape != (weight.qweight.shape[0], expected_groups):
        raise ValueError(
            "scales must be shaped "
            f"({weight.qweight.shape[0]}, {expected_groups}), "
            f"got {tuple(weight.scales.shape)}"
        )
    if weight.zeros is not None and weight.zeros.shape != weight.scales.shape:
        raise ValueError(
            f"zeros shape {tuple(weight.zeros.shape)} does not match scales "
            f"{tuple(weight.scales.shape)}"
        )
    if weight.signed and weight.zeros is not None:
        raise ValueError("signed quantization should not provide zero-points")


def _int4_gemv_reference(x: torch.Tensor, weight: Int4Weight) -> torch.Tensor:
    qweight = weight.qweight.to(device=x.device).contiguous()
    scales = weight.scales.to(device=x.device, dtype=torch.float32).contiguous()
    zeros = (
        weight.zeros.to(device=x.device, dtype=torch.float32).contiguous()
        if weight.zeros is not None
        else None
    )
    out_features = qweight.shape[0]
    k = weight.in_features
    packed = qweight[:, torch.arange((k + 7) // 8, device=x.device)]
    shifts = (torch.arange(k, device=x.device, dtype=torch.int64) % 8) * 4
    words = packed[:, torch.arange(k, device=x.device) // 8]
    q = ((words >> shifts) & 0xF).to(torch.float32)
    if weight.signed:
        q = torch.where(q >= 8, q - 16, q)
        dequant = q
    else:
        groups = torch.arange(k, device=x.device) // weight.group_size
        zero_values = 8.0 if zeros is None else zeros[:, groups]
        dequant = q - zero_values
    groups = torch.arange(k, device=x.device) // weight.group_size
    dequant = dequant * scales[:, groups]
    return torch.matmul(x.to(torch.float32), dequant.T).reshape(x.shape[0], out_features)


if tl is not None:

    @triton.jit
    def _int4_gemv_kernel(
        x_ptr,
        qweight_ptr,
        scales_ptr,
        zeros_ptr,
        out_ptr,
        batch: tl.constexpr,
        in_features: tl.constexpr,
        out_features: tl.constexpr,
        packed_k: tl.constexpr,
        num_groups: tl.constexpr,
        group_size: tl.constexpr,
        stride_xb: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_on: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_ZEROS: tl.constexpr,
        SIGNED: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        n_block = tl.program_id(1)
        n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)
        accum = tl.zeros((BLOCK_N,), tl.float32)

        for k_start in range(0, in_features, BLOCK_K):
            k = k_start + k_offsets
            k_mask = k < in_features
            packed_offsets = k // 8
            shifts = (k % 8) * 4
            q_ptrs = qweight_ptr + n_offsets[:, None] * packed_k + packed_offsets[None, :]
            q_words = tl.load(
                q_ptrs,
                mask=(n_offsets[:, None] < out_features) & k_mask[None, :],
                other=0,
            )
            q_vals = ((q_words >> shifts[None, :]) & 0xF).to(tl.float32)

            group_offsets = k // group_size
            scale_ptrs = scales_ptr + n_offsets[:, None] * num_groups + group_offsets[None, :]
            scales = tl.load(
                scale_ptrs,
                mask=(n_offsets[:, None] < out_features) & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            if SIGNED:
                q_vals = tl.where(q_vals >= 8.0, q_vals - 16.0, q_vals)
                w_vals = q_vals * scales
            else:
                if HAS_ZEROS:
                    zero_ptrs = zeros_ptr + n_offsets[:, None] * num_groups + group_offsets[None, :]
                    zeros = tl.load(
                        zero_ptrs,
                        mask=(n_offsets[:, None] < out_features) & k_mask[None, :],
                        other=8.0,
                    ).to(tl.float32)
                else:
                    zeros = tl.full((BLOCK_N, BLOCK_K), 8.0, tl.float32)
                w_vals = (q_vals - zeros) * scales

            x_vals = tl.load(
                x_ptr + batch_id * stride_xb + k * stride_xk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            accum += tl.sum(w_vals * x_vals[None, :], axis=1)

        tl.store(
            out_ptr + batch_id * stride_ob + n_offsets * stride_on,
            accum,
            mask=n_offsets < out_features,
        )
