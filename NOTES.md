# Optimization Notes — Qwen3.5-4B on vLLM (A10)

Source: `profiles/traces/vllm-short-cuda-forward-20260615-020343.trace.json`
GPU: NVIDIA A10 (sm86, 72 SMs, ~600 GB/s). Hybrid model: gated-delta-net (GDN /
fla linear-attention) layers + some full-attention layers.

## Cold start (first request) — already addressed
- First forward `execute_context_1(61)_generation_0(0)` = ~78 ms, **67% GPU idle**.
- Gaps were one-time **Triton JIT + autotune** of the GDN/fla kernels, plus
  torch._dynamo AOT compile and CUDA-graph capture. Confirmed: steady-state
  decode is only **1.4% idle** (same kernels, gap-free).
- Fix shipped: warmup in `VllmBackend.__init__` (before the server is reachable,
  so it's NOT counted in benchmark timing). Sweeps a generic geometric range of
  prompt lengths (64/256/1024/4096/8192) so per-shape compile/capture happens at
  startup. Env: `VLLM_WARMUP`, `VLLM_WARMUP_PROMPT_TOKENS`,
  `VLLM_WARMUP_DECODE_STEPS`.

## Where steady-state time goes
Decode step (~19.6 ms, batch≈1, 98.6% GPU busy):
| Item | ms | % |
|---|---|---|
| Weight-projection matmuls (q/k/v/o, gate/up/down × layers) | 15.3 | 79% |
| **LM-head / vocab GEMV** (single `gemv2T`, grid 31040) | 2.6 | **13.5%** |
| GDN linear-attn (`fused_recurrent_gated_delta_rule`) | 0.25 | 1.3% |
| norm / silu / conv / memcpy / flash | ~1.2 | ~6% |

Prefill (61 tok, 25.5 ms busy): 83% matmul, GDN attn 3.3%, elementwise 5.5%,
norm 3.9%. **44% of launches are <3 µs** (~1 ms total) — hidden today by CUDA
graphs.

Interpretation: **decode = memory-bandwidth-bound** (batch-1 GEMVs re-reading
bf16 weights per token); **prefill = compute-bound GEMM**. Latency benchmark is
decode-heavy (short 64→128 out, medium 2048→256, long 8192→256).

## Opportunities beyond quantizing the body GEMMs (ranked)
1. **Quantize the LM-head** (2.6 ms, 13.5% of decode). It's a ~152k-vocab
   projection run as a standalone bf16 cuBLAS GEMV — bigger than the whole
   attention path, easy to miss. Pure memory-bound; int4/int8 or fused dequant.
2. **Speculative decoding** (allowed by rules). Root cause of decode cost: every
   token reads all ~4B weights once. EAGLE/Medusa/draft/n-gram → near-K× fewer
   weight reads at high acceptance. Stacks with quantization. ~1.5–2.5×.
3. **W8A8 int8, not just weight-only int4.** int4 helps decode (memory) but not
   prefill compute; medium/long prompts are prefill-heavy (83% GEMM). A10 has
   INT8 tensor cores (no FP8), so W8A8 ~2× prefill GEMM throughput too.
4. **Batching / arithmetic intensity.** Quality evals use NUM_CONCURRENT=8. If
   co-scheduled into one decode batch, GEMVs become batch-8 GEMMs (~8× compute
   per weight read). Verify `max_num_seqs` / continuous batching actually batch.
5. **Fuse the tiny-kernel tail + keep everything in CUDA graphs.** 44% of
   launches are sub-3 µs (RMSNorm, SwiGLU, rotary, GDN gating, copies). Free
   today, but the GDN/fla path runs **eager** — at 8192 prefill its many small
   kernels + Python launch cost can re-expose idle. Fuse + ensure graph capture
   covers served decode batch sizes.
6. **Long-context specifics.** At 8192 ctx, full-attention KV traffic grows →
   **fp8 KV cache** (only touches attention layers; GDN uses fixed recurrent
   state). Confirm chunked-prefill GDN kernels are warmed/autotuned.

Suggested order: **LM-head quant → speculative decoding → W8A8 prefill**, then
fusion / batching / fp8-KV as polish.

## TODO / open questions
- [ ] Re-profile with `record_shapes=True` to get per-layer matmul dims and
      confirm LM-head shape (size int4-vs-W8A8 trade-off).
- [ ] Measure prefill-dominant runs (medium 2048 / long 8192) — this trace is
      the "short" case; attention/KV cost grows with context.
- [ ] Validate quality gates after each quant step (MMLU-Pro ≥0.621,
      IFEval ≥0.814, GPQA-Diamond ≥0.630).
