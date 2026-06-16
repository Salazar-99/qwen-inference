# Optimization Notes — Qwen3.5-4B on vLLM (A10)

GPU: NVIDIA A10 (sm86, 72 SMs, ~600 GB/s, 24 GB GDDR6). Ampere architecture.
Model: hybrid gated-delta-net (GDN / fla linear-attention) + full-attention layers.
Current best: **W4A16 GPTQ (Marlin), 2.04× average speedup** (see LOG.md).
Traces: `profiles/traces/vllm-{short,medium,long}-cuda-forward-20260615-*.trace.json`

---

## GPU utilization by category (W4A16 traces, DECODE_STEPS=4)

| | Short (64→128) | Medium (2048→256) | Long (8192→256) |
|---|---|---|---|
| Wall (kernel span) | 139.9 ms | 306.9 ms | 1192.7 ms |
| GPU busy | 39.4 ms (**28.2%**) | 296.0 ms (**96.5%**) | 1168.9 ms (**98.0%**) |
| GPU idle | 100.5 ms (**71.8%**) | 10.9 ms (3.5%) | 23.7 ms (2.0%) |
| Gaps >100 µs | 305 | 17 | 28 |

Short is an outlier — 71.8% idle despite the same model. Root cause: the prefill path
runs eager dispatch (305 inter-kernel gaps >100 µs); only decode is CUDA-graph-captured.
Medium and long prefills are large enough that Marlin GEMMs dominate and scheduling
overhead shrinks to noise.

## Kernel breakdown (% of total kernel time per trace)

| Kernel | Short | Medium | Long |
|---|---|---|---|
| marlin_w4a16 | 23.7 ms **(60%)** | 235.5 ms **(79.5%)** | 893.8 ms **(76.5%)** |
| lm_head_gemv (bf16) | 10.4 ms **(26.4%)** | 10.4 ms (3.5%) | 18.2 ms (1.6%) |
| flash_attn | 0.65 ms (1.6%) | 7.8 ms (2.6%) | 92.8 ms **(7.9%)** |
| gdn_* combined | 2.0 ms (5.0%) | 24.6 ms (8.3%) | 96.1 ms (8.2%) |
| norm / fused triton | 1.0 ms (2.5%) | 8.4 ms (2.8%) | 31.1 ms (2.7%) |
| memcpy / other | 0.9 ms (2.3%) | 4.3 ms (1.4%) | 18.7 ms (1.6%) |

lm_head is bf16 (vLLM VocabParallelEmbedding cannot load a quantized lm_head — see LOG.md).
Long lm_head is higher than medium because chunked prefill (8192 split into 2048+2047+...
chunks) runs an lm_head forward per chunk as well as per decode step.

---

## Opportunities (ranked by expected competition impact)

### 1. CUDA-graph the prefill path — short category

**Why:** Short trace is 71.8% idle. The 305 inter-kernel gaps >100 µs are Python dispatch
stalls in the eager prefill path — every RMSNorm, rotary, GDN gate, and silu kernel waits
for the CPU to queue the next op. Decode is already CUDA-graph-captured and runs gap-free.

**Fix:** Enable chunked prefill with fixed chunk sizes (64/128/256/512/1024/2048). Each
chunk is a static shape that can be graph-captured. vLLM has `--enable-chunked-prefill`;
the extra requirement is warming the GDN/fla kernels per chunk shape at startup (already
done by the geometric warmup sweep in `VllmBackend._warmup`).

**Estimate:** Could recover most of the 100 ms idle → short wall time drops from ~1167 ms
toward ~1000 ms. Stacks with every other optimization.

### 2. Quantize lm_head — all categories, highest per-step payoff

**Why:** lm_head (gemv2T, 152k vocab × 2560 hidden, bf16) is 2.6 ms/decode step. At full
decode depth this dominates short and competes with prefill on medium:
- Short: 128 steps × 2.6 ms ≈ **333 ms (~28% of 1167 ms total)**
- Medium: 256 steps × 2.6 ms ≈ **666 ms (~26% of 2587 ms total)**

With W4A16 body quantization, the weight projections are now faster, so lm_head's share
has grown from the 13.5% seen in the original bf16 baseline.

**Blocker:** vLLM's `VocabParallelEmbedding` (lm_head in Qwen3_5ForCausalLM) has no
`weight_scale` slot. Options to unblock:
- Patch `VocabParallelEmbedding` to accept W4A16 or W8A16 weight + scale, dispatching
  to a dequant+gemv fused kernel
- Write a standalone int4-dequant+gemv Triton kernel that replaces the lm_head forward

### 3. Speculative decoding — all categories

**Why:** Decode is the dominant cost for short (~87% of wall time after prefill). Every
decode step reads all ~4B weights once through Marlin. Speculative decoding avoids many
of those reads by accepting k draft tokens per step instead of generating 1.

EAGLE, n-gram, or a small draft model at k=4 and ~0.7 acceptance rate → effective
throughput ~2–3× on decode. Stacks multiplicatively with W4A16 quantization. Explicitly
allowed by competition rules.

**Estimate:** 1.5–2× on short (decode-dominant), 1.2–1.5× on medium/long. Combined with
W4A16 this could push average speedup toward 3–4×.

### 4. 2:4 structured sparsity + sparse Marlin — all categories

**Why:** Marlin W4A16 is 60–80% of all kernel time across every category. 2:4 sparsity
(2 of every 4 weights zeroed, enforced at training/PTQ time) is accelerated by A10 Sparse
Tensor Cores and halves the effective weight bytes, multiplying on top of INT4.
vLLM ships a `SparseMarlin` kernel; llm-compressor supports 2:4 sparsification before GPTQ.

**Estimate:** ~20–30% reduction in Marlin time → ~15–20% total speedup per category.
Quality risk: GPQA-Diamond (90% threshold) is the sensitive gate. Validate carefully.

### 5. KV cache quantization for long context — requires custom kernel

**Why:** flash_attn grows from 1.6% (short) → 7.9% (long, 92.8 ms across 4 decode steps).
At full 256-step decode with 8192-token KV cache, per-step decode attention is memory-
bandwidth-bound: each step reads ~8 full-attention layers × 2 × 8192 tokens × 128 head_dim
≈ 33 MB from DRAM. Quantizing KV to INT8 halves that, to INT4 quarters it.

**FP8 KV cache is NOT viable on A10.** FP8 requires SM9.0+ (H100/Hopper). A10 is SM8.6
(Ampere) — no native FP8 compute. The vLLM blog post (April 2026) explicitly calls out
AWS g5 / A10G as incompatible. Using fp8 on Ampere emulates in software and degrades
performance 10–20%.

**INT8 KV cache — viable on A10, not in stock vLLM.** INT8 tensor cores ARE present on
A10. vLLM's KV cache quantization only supports FP8 (GitHub issue #33480 open as of 2026).
lmdeploy implements INT4/INT8 KV cache natively including for Ampere, using a custom
attention kernel that fuses dequant + flash attention. Using lmdeploy as the serving
backend instead of vLLM is the lowest-effort path to INT8 KV cache on A10.

**INT4 KV cache (KIVI / custom Triton kernel):** The KIVI paper shows 2-bit KV with a
custom Triton decode attention kernel (quantize after prefill attention to avoid prefill
error, use quantized K/V for all decode steps). An INT4 attention kernel can be ~2.88×
faster than FlashAttention at 128k context. Engineering cost is high but the building
blocks are open-source.

**Practical recommendation:** For remaining competition time, fp8 KV is off the table.
INT8 KV via lmdeploy is worth a spike if the serving backend can be swapped. Custom
INT4/Triton attention is a longer-horizon item.

### 6. GDN kernel improvement — medium/long, low priority

**Why:** GDN combined is 8.2–8.3% of medium/long kernel time (chunk_gated_delta_rule_fwd,
_causal_conv1d_fwd, fused_recurrent_gated_delta_rule). These are Triton kernels; less
mature than FlashAttention. Custom CUDA could give 20–30% reduction in GDN time (~2%
total). High engineering cost for modest gain.

---

## Cold start — already addressed

Warmup in `VllmBackend.__init__` sweeps geometric prompt lengths (64/256/1024/4096/8192)
before the server is reachable. Triton JIT + autotune + CUDA-graph capture all happen at
startup. Confirmed: medium/long steady-state GPU busy >96%.

## Open questions

- [ ] Validate W4A16 quality gates (MMLU-Pro ≥0.621, IFEval ≥0.814, GPQA-Diamond ≥0.630)
- [ ] Re-profile short after `--enable-chunked-prefill` to confirm idle reduction
- [ ] Measure lm_head cost at full 128/256-step decode (4-step profile understates it)
- [ ] Check SparseMarlin availability in vllm==0.19.1 before committing to sparsity path
- [ ] Spike lmdeploy INT8 KV cache on A10 — measure flash_attn decode step speedup at 8192 ctx
