from __future__ import annotations

import importlib.util

import pytest

torch = pytest.importorskip("torch")

from qwen_inference.backends.custom.kernels import Int4Weight, int4_gemv


def _pack_rows(q: torch.Tensor) -> torch.Tensor:
    assert q.ndim == 2
    rows, cols = q.shape
    packed_cols = (cols + 7) // 8
    packed = torch.zeros((rows, packed_cols), dtype=torch.uint32, device=q.device)
    for bit in range(8):
        idx = torch.arange(bit, cols, 8, device=q.device)
        if idx.numel() == 0:
            continue
        values = q[:, idx].to(torch.uint32)
        packed[:, : values.shape[1]] |= values << (4 * bit)
    return packed


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="Triton missing")
def test_int4_gemv_matches_asymmetric_dequant_reference() -> None:
    torch.manual_seed(0)
    batch = 3
    in_features = 257
    out_features = 96
    group_size = 64
    groups = (in_features + group_size - 1) // group_size

    x = torch.randn(batch, in_features, device="cuda", dtype=torch.bfloat16)
    q = torch.randint(0, 16, (out_features, in_features), device="cuda")
    scales = torch.rand(out_features, groups, device="cuda", dtype=torch.float16) * 0.2
    zeros = torch.randint(0, 16, (out_features, groups), device="cuda").to(torch.float16)

    packed = _pack_rows(q)
    weight = Int4Weight(
        qweight=packed,
        scales=scales,
        zeros=zeros,
        in_features=in_features,
        group_size=group_size,
    )

    groups_for_k = torch.arange(in_features, device="cuda") // group_size
    dequant = (q.float() - zeros.float()[:, groups_for_k]) * scales.float()[:, groups_for_k]
    expected = x.float() @ dequant.T

    actual = int4_gemv(x, weight, output_dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="Triton missing")
def test_int4_gemv_matches_signed_symmetric_reference() -> None:
    torch.manual_seed(1)
    in_features = 192
    out_features = 64
    group_size = 128
    groups = (in_features + group_size - 1) // group_size

    x = torch.randn(in_features, device="cuda", dtype=torch.float16)
    signed_q = torch.randint(-8, 8, (out_features, in_features), device="cuda")
    packed = _pack_rows(signed_q & 0xF)
    scales = torch.rand(out_features, groups, device="cuda", dtype=torch.float16) * 0.1
    weight = Int4Weight(
        qweight=packed,
        scales=scales,
        zeros=None,
        in_features=in_features,
        group_size=group_size,
        signed=True,
    )

    groups_for_k = torch.arange(in_features, device="cuda") // group_size
    dequant = signed_q.float() * scales.float()[:, groups_for_k]
    expected = x.float() @ dequant.T

    actual = int4_gemv(x, weight, output_dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-2)
