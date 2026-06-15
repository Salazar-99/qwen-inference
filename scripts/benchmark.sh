#!/usr/bin/env bash
set -Eeuo pipefail

# Run the competition quality and/or latency benchmarks from evals/ against a
# locally served set of model weights (via the qwen-inference server).
#
# Usage:
#   ./scripts/benchmark.sh [quality|latency|both] [MODEL_WEIGHTS_DIR]
#
#   arg 1  benchmark kind: quality | latency | both          (default: both)
#   arg 2  path to the model weights to serve                (default: qwen-inference/qwen-weights)
#
# The weights are served with the qwen-inference server (INFERENCE_MODE=vllm by
# default) by exporting MODEL_DIR; the evals in evals/ are pointed at it through
# CONTAINER_URL.
#
# Examples:
#   ./scripts/benchmark.sh both qwen-inference/qwen-weights-quantized
#   ./scripts/benchmark.sh quality qwen-inference/qwen-weights-quantized
#   QUALITY_LIMIT=0.1 NUM_CONCURRENT=8 ./scripts/benchmark.sh quality
#   LATENCY_RUNS=50 ./scripts/benchmark.sh latency qwen-inference/qwen-weights
#
# Requirements: the eval harness deps must be importable by EVAL_PYTHON
#   pip install lm-eval==0.4.11 langdetect immutabledict
# (run_eval_local.py imports lm_eval at module load, so this is needed for every
# kind, including latency-only.)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

KIND="${1:-both}"
MODEL_WEIGHTS="${2:-${MODEL_DIR:-${REPO_ROOT}/qwen-inference/qwen-weights}}"

MODE="${INFERENCE_MODE:-${MODE:-vllm}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CONTAINER_URL="http://${HOST}:${PORT}"

# Eval harness knobs (passed through to evals/run_eval_local.py).
QUALITY_LIMIT="${QUALITY_LIMIT:-0.1}"
NUM_CONCURRENT="${NUM_CONCURRENT:-8}"
LATENCY_RUNS="${LATENCY_RUNS:-50}"
EVAL_PYTHON="${EVAL_PYTHON:-python3}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-600}"

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/profiles/benchmarks}"
LOGS_DIR="${REPO_ROOT}/profiles/logs"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
WEIGHTS_TAG="$(basename "${MODEL_WEIGHTS}")"
OUTPUT_STEM="${MODE}-${WEIGHTS_TAG}-${KIND}-${TIMESTAMP}"
SERVER_LOG="${LOGS_DIR}/${OUTPUT_STEM}.log"
RESULTS_OUTPUT="${RESULTS_DIR}/${OUTPUT_STEM}.json"

case "${KIND}" in
  quality)  EVAL_MODE="quality" ;;
  latency)  EVAL_MODE="latency" ;;
  both)     EVAL_MODE="full" ;;
  *)
    echo "Usage: $0 [quality|latency|both] [MODEL_WEIGHTS_DIR]" >&2
    exit 2
    ;;
esac

case "${MODE}" in
  baseline|custom|vllm) ;;
  *)
    echo "Invalid INFERENCE_MODE/MODE='${MODE}' (expected baseline|custom|vllm)." >&2
    exit 2
    ;;
esac

if [[ ! -d "${MODEL_WEIGHTS}" ]]; then
  echo "Model weights directory not found: ${MODEL_WEIGHTS}" >&2
  echo "Pass a valid path as the second argument or via MODEL_DIR." >&2
  exit 1
fi

for command in uv curl "${EVAL_PYTHON}"; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing required command: ${command}" >&2
    exit 1
  fi
done

# Resolve to an absolute path so the server finds it regardless of cwd.
MODEL_WEIGHTS="$(cd "${MODEL_WEIGHTS}" && pwd)"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

wait_for_server() {
  echo "Waiting for ${CONTAINER_URL}/ping (up to ${SERVER_WAIT_SECONDS}s)..."
  for _ in $(seq 1 "${SERVER_WAIT_SECONDS}"); do
    if curl -fsS "${CONTAINER_URL}/ping" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "Server exited before becoming ready. Last log lines:" >&2
      tail -n 80 "${SERVER_LOG}" >&2
      exit 1
    fi
    sleep 1
  done
  echo "Timed out waiting for the server to become ready. Last log lines:" >&2
  tail -n 80 "${SERVER_LOG}" >&2
  exit 1
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "Stopping server..."
    kill -INT "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT

echo "=== Benchmark: ${KIND} (EVAL_MODE=${EVAL_MODE}) ==="
echo "Serving '${MODEL_WEIGHTS}' with mode='${MODE}'"
echo "Server log: ${SERVER_LOG}"

INFERENCE_MODE="${MODE}" MODEL_DIR="${MODEL_WEIGHTS}" HOST="${HOST}" PORT="${PORT}" \
  uv run --package qwen-inference qwen-serve --mode "${MODE}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"
wait_for_server

echo "Running evals/run_eval_local.py (EVAL_MODE=${EVAL_MODE})..."
CONTAINER_URL="${CONTAINER_URL}" \
  EVAL_MODE="${EVAL_MODE}" \
  QUALITY_LIMIT="${QUALITY_LIMIT}" \
  NUM_CONCURRENT="${NUM_CONCURRENT}" \
  NUM_RUNS="${LATENCY_RUNS}" \
  "${EVAL_PYTHON}" "${REPO_ROOT}/evals/run_eval_local.py"

# run_eval_local.py always writes here; copy to a timestamped, weights-tagged file.
if [[ -f /tmp/local_eval_results.json ]]; then
  cp /tmp/local_eval_results.json "${RESULTS_OUTPUT}"
fi

cleanup
trap - EXIT

echo
echo "Benchmark complete:"
echo "  results: ${RESULTS_OUTPUT}"
echo "  server log: ${SERVER_LOG}"
