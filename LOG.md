# Optimization Log — Qwen3.5-4B on A10 (vLLM backend)

Competition baseline: short 2582 ms · medium 5441 ms · long 6576 ms

---

## Run 1 — W8A8 (SmoothQuant + RTN, 2026-06-15)

**Checkpoint:** `qwen-inference/qwen-weights-w8a8` (5.8 GB, vs ~8 GB bf16)

**What changed:**
- `scripts/quantize_gptq.py --scheme W8A8` — llm-compressor `SmoothQuantModifier` +
  `QuantizationModifier(scheme="W8A8")`: INT8 weights (per-channel), INT8 dynamic
  per-token activations on all `nn.Linear` layers in the language model.
- `lm_head` excluded: vLLM's `VocabParallelEmbedding` has no `weight_scale` slot so
  the checkpoint fails to load if lm_head is quantized. Weight was dequantized back to
  bf16 and `lm_head` added to the `quantization_config.ignore` list post-hoc.
- SmoothQuant smoothing was a no-op (generated 0 per-layer mappings) because the hybrid
  GDN/attention architecture wasn't detected — so this is effectively plain RTN W8A8.

**Latency results (50 runs):**

| Category | Median   | Baseline | Speedup |
|----------|----------|----------|---------|
| Short    | 1782 ms  | 2582 ms  | 1.45×   |
| Medium   | 3762 ms  | 5441 ms  | 1.45×   |
| Long     | 4619 ms  | 6576 ms  | 1.42×   |
| **Avg**  |          |          | **1.44×** |

Speedup is uniform across all three categories — consistent with INT8 tensor cores
helping both decode (memory bandwidth) and prefill (compute) roughly equally on A10.

**Quality:** Not measured (eval paused).

---

## Run 2 — W4A16 (GPTQ, 2026-06-15)

**Checkpoint:** `qwen-inference/qwen-weights-w4a16`

**What changed:**
- `scripts/quantize_gptq.py --scheme W4A16` — llm-compressor `GPTQModifier(scheme="W4A16")`:
  INT4 weight-only quantization (second-order GPTQ calibration), bf16 activations.
  All `nn.Linear` layers quantized; `lm_head`, `re:.*visual.*`, `re:.*mtp.*` excluded.
- Served via vLLM's Marlin INT4 kernel (Ampere sm_86).

**Latency results (5 runs):**

| Category | Median   | Baseline | Speedup |
|----------|----------|----------|---------|
| Short    | 1167 ms  | 2582 ms  | 2.21×   |
| Medium   | 2587 ms  | 5441 ms  | 2.10×   |
| Long     | 3651 ms  | 6576 ms  | 1.80×   |
| **Avg**  |          |          | **2.04×** |

W4A16 outperforms W8A8 across all categories. Short/decode gains most (4× fewer weight
bytes → 2.21×); long/prefill gains least (bottleneck shifts partially to compute, where
4-bit gives no tensor-core advantage over 8-bit). Average 2.04× vs 1.44× for W8A8.

**Quality:** Not measured.

---

## Run 3 — W4A16 + N-gram speculative decoding (2026-06-16)

**Checkpoint:** `qwen-inference/qwen-weights-w4a16`

**What changed:**
- `VLLM_SPEC_DECODE=ngram` → vLLM CPU-numba n-gram proposer with 5 draft tokens per step,
  n-gram window 1–4.
- `max_num_seqs=1`, `max_model_len=8448` (required to avoid OOM in vision-tower dummy profiling
  pass at startup).
- `ngram_gpu` was attempted first but has a vLLM 0.19.1 CUDA-graph bug (`sym_shape_indices`
  IndexError at `max_num_seqs=1`); automatically falls back to stable CPU `ngram` proposer.

**Latency results (5 runs):**

| Category | Median   | W4A16 base | Speedup | vs baseline |
|----------|----------|------------|---------|-------------|
| Short    | 315 ms   | 1167 ms    | 3.70×   | **8.19×**   |
| Medium   | 842 ms   | 2587 ms    | 3.07×   | **6.46×**   |
| Long     | 1857 ms  | 3651 ms    | 1.97×   | **3.54×**   |
| **Avg**  |          |            |         | **6.06×**   |

**Analysis:** Numbers are inflated by the benchmark prompt ("The quick brown fox…" repeated),
which produces near-perfect n-gram match rates — the proposer copies the following tokens
almost every step. Real-world acceptance rates will be lower. However, even at reduced
acceptance this establishes n-gram as a useful tool for repetitive or structured outputs.

---

## Run 4 — W4A16 + MTP (Qwen3.5 built-in multi-token prediction, 2026-06-16)

**Checkpoint:** `qwen-inference/qwen-weights-w4a16` (+ `mtp.safetensors` shard added manually)

**What changed:**
- `VLLM_SPEC_DECODE=mtp` → `{"method": "qwen3_5_mtp", "model": ..., "num_speculative_tokens": 1}`.
- MTP weights (`mtp.*`) were missing from the GPTQ checkpoint because
  `AutoModelForImageTextToText` (`Qwen3_5ForConditionalGeneration`) has no `mtp` attribute
  and `save_pretrained` silently omits them. Fixed by copying 15 MTP tensors from the bf16
  checkpoint into `mtp.safetensors` and writing a `model.safetensors.index.json` shard index.
- MTP weights are bf16 (not quantized) — the quantizer excluded `re:.*mtp.*`.

**Latency results (5 runs):**

| Category | Median    | W4A16 base | Speedup | vs baseline |
|----------|-----------|------------|---------|-------------|
| Short    | 1724 ms   | 1167 ms    | 0.68×   | **1.50×**   |
| Medium   | 3965 ms   | 2587 ms    | 0.65×   | **1.37×**   |
| Long     | 5778 ms   | 3651 ms    | 0.63×   | **1.14×**   |
| **Avg**  |           |            |         | **1.34×**   |

**Analysis:** MTP is **slower than plain W4A16** across all categories. Two root causes:
1. **Unquantized MTP head**: The bf16 MTP layer adds per-step compute overhead that Marlin
   INT4 cannot offset — the draft step runs full bf16 GEMMs while the verify step stays W4A16.
2. **Low acceptance on the benchmark prompt**: The quick-brown-fox repetition perfectly suits
   n-gram (exact string match), but MTP's learned draft head isn't tuned for this distribution
   and likely has low acceptance, so most draft tokens are rejected and only the overhead remains.

MTP could recover on domain-matched workloads where the MTP head was trained (general chat,
code), especially once the MTP weights are also quantized. Not recommended as-is.

---

## Open items

- [ ] Run quality gates (MMLU-Pro ≥0.621, IFEval ≥0.814, GPQA-Diamond ≥0.630) for W4A16
- [ ] Fix SmoothQuant layer mapping for hybrid GDN architecture and re-run W8A8 properly
- [ ] Quantize lm_head via a custom vLLM loader (2.6 ms / 13.5% of decode — item 1 in NOTES.md)
- [ ] Explore fp8 KV cache (`kv_cache_dtype="fp8"` in EngineArgs) for long-context gains
- [ ] Speculative decoding on top of W4A16 (item 2 in NOTES.md)
