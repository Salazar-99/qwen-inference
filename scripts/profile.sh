#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-baseline}"
PROMPT_SIZE="${2:-${PROMPT_SIZE:-short}}"
PROFILE_KIND="${3:-${PROFILE_KIND:-latency}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
PROFILE_DIR="${PROFILE_DIR:-profiles}"
LATENCY_DIR="${PROFILE_DIR}/latency"
LOGS_DIR="${PROFILE_DIR}/logs"
TRACES_DIR="${PROFILE_DIR}/traces"
WARMUP_RUNS="${WARMUP_RUNS:-3}"
LATENCY_RUNS="${LATENCY_RUNS:-50}"
DECODE_STEPS="${DECODE_STEPS:-4}"
FILLER="${FILLER:-The quick brown fox jumps over the lazy dog. }"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_STEM="${MODE}-${PROMPT_SIZE}-${PROFILE_KIND}-${TIMESTAMP}"
SERVER_LOG="${LOGS_DIR}/${OUTPUT_STEM}.log"
CONTAINER_URL="http://${HOST}:${PORT}"
export CONTAINER_URL DECODE_STEPS FILLER LATENCY_RUNS PROMPT_SIZE WARMUP_RUNS

if [[ "${MODE}" == "all" ]]; then
  MODES=(baseline custom vllm)
  for mode in "${MODES[@]}"; do
    echo
    echo "=== Profiling ${mode} (${PROFILE_KIND}, warmup=${WARMUP_RUNS}) ==="
    WARMUP_RUNS="${WARMUP_RUNS}" LATENCY_RUNS="${LATENCY_RUNS}" DECODE_STEPS="${DECODE_STEPS}" \
      PROFILE_DIR="${PROFILE_DIR}" HOST="${HOST}" PORT="${PORT}" FILLER="${FILLER}" \
      "$0" "${mode}" "${PROMPT_SIZE}" "${PROFILE_KIND}"
  done
  exit 0
fi

case "${MODE}" in
  baseline|custom|vllm) ;;
  *)
    echo "Usage: $0 [baseline|custom|vllm|all] [short|medium|long|all] [latency|cuda-forward|nsys-forward]" >&2
    exit 2
    ;;
esac

case "${PROMPT_SIZE}" in
  short|medium|long|all) ;;
  *)
    echo "Usage: $0 [baseline|custom|vllm|all] [short|medium|long|all] [latency|cuda-forward|nsys-forward]" >&2
    exit 2
    ;;
esac

case "${PROFILE_KIND}" in
  latency|cuda-forward|nsys-forward) ;;
  *)
    echo "Usage: $0 [baseline|custom|vllm|all] [short|medium|long|all] [latency|cuda-forward|nsys-forward]" >&2
    exit 2
    ;;
esac

if [[ "${PROFILE_KIND}" != "latency" && "${PROMPT_SIZE}" == "all" ]]; then
  echo "CUDA forward profiling captures one prompt size at a time; run short, medium, and long separately." >&2
  exit 2
fi

for command in uv curl python3; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing required command: ${command}" >&2
    exit 1
  fi
done

run_as_root() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "Need root privileges to install Nsight Systems, but sudo is not available." >&2
    return 1
  fi
}

install_nsys() {
  if command -v nsys >/dev/null 2>&1; then
    return 0
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "nsys is missing and automatic install currently supports apt-get only." >&2
    return 1
  fi

  echo "nsys is missing; attempting to install Nsight Systems with apt-get..."
  run_as_root env DEBIAN_FRONTEND=noninteractive apt-get update

  local packages=(
    nsight-systems
    nvidia-nsight-systems
    nsight-systems-cli
  )

  for package in "${packages[@]}"; do
    echo "Trying apt-get install -y ${package}..."
    if run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y "${package}"; then
      if command -v nsys >/dev/null 2>&1; then
        echo "Installed Nsight Systems: $(command -v nsys)"
        return 0
      fi
    fi
  done

  echo "Could not install a package that provides nsys." >&2
  echo "If your NVIDIA repository is not configured, install Nsight Systems from NVIDIA and re-run this script." >&2
  return 1
}

check_nsys_importer() {
  local importer="/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter"
  if [[ ! -x "${importer}" ]]; then
    echo "Nsight Systems importer not found at ${importer}." >&2
    echo "Install the host Nsight Systems package, not only the target CLI." >&2
    return 1
  fi

  local importer_output
  if importer_output="$("${importer}" --version 2>&1)"; then
    return 0
  fi

  echo "Nsight Systems is installed, but the QDSTRM importer cannot run:" >&2
  echo "${importer_output}" >&2
  echo >&2
  echo "This machine cannot convert .qdstrm captures into .nsys-rep until the importer dependency issue is fixed." >&2
  echo "For the current Lambda Labs package this commonly means libssh is too old; the importer requires LIBSSH_4_9_0." >&2
  echo "Install a compatible Nsight Systems host package/libssh combination, then rerun this script." >&2
  return 1
}

make_payload() {
  local prompt_size="$1"
  local endpoint_kind="${2:-invocations}"
  local profile="${3:-false}"
  local trace_path="${4:-}"
  python3 - "${prompt_size}" "${endpoint_kind}" "${profile}" "${trace_path}" <<'PY'
import json
import os
import sys

prompt_configs = {
    "short": {"num_tokens": 64, "max_new_tokens": 128},
    "medium": {"num_tokens": 2048, "max_new_tokens": 256},
    "long": {"num_tokens": 8192, "max_new_tokens": 256},
}

cfg = prompt_configs[sys.argv[1]]
prompt = os.environ["FILLER"] * max(1, cfg["num_tokens"] // 10)
if sys.argv[2] == "forward":
    payload = {
        "prompt": prompt,
        "decode_steps": int(os.environ["DECODE_STEPS"]),
        "profile": sys.argv[3] == "true",
    }
    if sys.argv[4]:
        payload["trace_path"] = sys.argv[4]
else:
    payload = {
        "prompt": prompt,
        "max_tokens": cfg["max_new_tokens"],
        "temperature": 0.0,
    }
print(json.dumps(payload))
PY
}

SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-600}"

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
    echo "Stopping profiled server..."
    kill -INT "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT

mkdir -p "${LATENCY_DIR}" "${LOGS_DIR}" "${TRACES_DIR}"

if [[ "${PROMPT_SIZE}" == "all" ]]; then
  PROMPT_SIZES=(short medium long)
else
  PROMPT_SIZES=("${PROMPT_SIZE}")
fi

if [[ "${PROFILE_KIND}" == "latency" ]]; then
  LATENCY_OUTPUT="${LATENCY_DIR}/${OUTPUT_STEM}.json"
  echo "Starting ${MODE} server for end-to-end latency profiling..."
  echo "Using latency-eval prompt size(s): ${PROMPT_SIZES[*]}"
  echo "Warmup runs: ${WARMUP_RUNS}; measured runs: ${LATENCY_RUNS}"
  echo "Writing latency results to ${LATENCY_OUTPUT}"
  echo "Writing server log to ${SERVER_LOG}"

  INFERENCE_MODE="${MODE}" HOST="${HOST}" PORT="${PORT}" \
    uv run --package qwen-inference qwen-serve --mode "${MODE}" \
    >"${SERVER_LOG}" 2>&1 &
  SERVER_PID="$!"
  wait_for_server

  python3 - "${LATENCY_OUTPUT}" "${PROMPT_SIZES[@]}" <<'PY'
import json
import os
import statistics
import sys
import time
import urllib.request

output_path = sys.argv[1]
prompt_sizes = sys.argv[2:]
container_url = os.environ["CONTAINER_URL"]
warmup_runs = int(os.environ["WARMUP_RUNS"])
latency_runs = int(os.environ["LATENCY_RUNS"])
filler = os.environ["FILLER"]
prompt_configs = {
    "short": {"num_tokens": 64, "max_new_tokens": 128},
    "medium": {"num_tokens": 2048, "max_new_tokens": 256},
    "long": {"num_tokens": 8192, "max_new_tokens": 256},
}
competition_baseline_ms = {
    "short": 2582,
    "medium": 5441,
    "long": 6576,
    "average": 4866,
}

def make_payload(prompt_size: str) -> dict:
    cfg = prompt_configs[prompt_size]
    return {
        "prompt": filler * max(1, cfg["num_tokens"] // 10),
        "max_tokens": cfg["max_new_tokens"],
        "temperature": 0.0,
    }

def invoke(payload: dict) -> float:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{container_url}/invocations",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as response:
        response.read()
    return (time.perf_counter() - start) * 1000

results = {}
for prompt_size in prompt_sizes:
    payload = make_payload(prompt_size)
    for _ in range(warmup_runs):
        invoke(payload)

    latencies = [invoke(payload) for _ in range(latency_runs)]
    mean_ms = statistics.mean(latencies)
    median_ms = statistics.median(latencies)
    baseline_ms = competition_baseline_ms[prompt_size]
    results[prompt_size] = {
        "warmup_runs": warmup_runs,
        "runs": latency_runs,
        "competition_baseline_ms": baseline_ms,
        "mean_ms": round(mean_ms, 2),
        "mean_delta_vs_competition_ms": round(mean_ms - baseline_ms, 2),
        "mean_speedup_vs_competition": round(baseline_ms / mean_ms, 4),
        "median_ms": round(median_ms, 2),
        "median_delta_vs_competition_ms": round(median_ms - baseline_ms, 2),
        "median_speedup_vs_competition": round(baseline_ms / median_ms, 4),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "latencies_ms": [round(value, 2) for value in latencies],
    }

    median_delta = results[prompt_size]["median_delta_vs_competition_ms"]
    median_speedup = results[prompt_size]["median_speedup_vs_competition"]
    print(
        f"{prompt_size}: mean={results[prompt_size]['mean_ms']} ms, "
        f"median={results[prompt_size]['median_ms']} ms, "
        f"baseline={baseline_ms} ms, "
        f"delta={median_delta:+.2f} ms, "
        f"speedup={median_speedup:.4f}x"
    )

if results:
    average_median_ms = statistics.mean(
        result["median_ms"] for result in results.values()
    )
    results["average"] = {
        "competition_baseline_ms": competition_baseline_ms["average"],
        "median_ms": round(average_median_ms, 2),
        "median_delta_vs_competition_ms": round(
            average_median_ms - competition_baseline_ms["average"],
            2,
        ),
        "median_speedup_vs_competition": round(
            competition_baseline_ms["average"] / average_median_ms,
            4,
        ),
    }

with open(output_path, "w") as output_file:
    json.dump(results, output_file, indent=2)
PY

  cleanup
  trap - EXIT

  echo
  echo "Latency profile complete:"
  echo "  ${LATENCY_OUTPUT}"
  echo "  ${SERVER_LOG}"
  exit 0
fi

if [[ "${PROFILE_KIND}" == "cuda-forward" ]]; then
  TRACE_OUTPUT="${TRACES_DIR}/${OUTPUT_STEM}.trace.json"
  echo "Starting ${MODE} server for torch CUDA forward-pass profiling..."
  echo "Using prompt size: ${PROMPT_SIZE}; decode forwards: ${DECODE_STEPS}"
  echo "Warmup runs: ${WARMUP_RUNS}"
  echo "Writing Chrome trace to ${TRACE_OUTPUT}"
  echo "Writing server log to ${SERVER_LOG}"

  INFERENCE_MODE="${MODE}" HOST="${HOST}" PORT="${PORT}" \
    uv run --package qwen-inference qwen-serve --mode "${MODE}" \
    >"${SERVER_LOG}" 2>&1 &
  SERVER_PID="$!"
  wait_for_server

  warmup_json="$(make_payload "${PROMPT_SIZE}" forward false)"
  measured_json="$(make_payload "${PROMPT_SIZE}" forward true "${TRACE_OUTPUT}")"

  echo "Sending ${WARMUP_RUNS} forward warmup request(s)..."
  for _ in $(seq 1 "${WARMUP_RUNS}"); do
    curl -fsS "${CONTAINER_URL}/profile/forward" \
      -H "Content-Type: application/json" \
      -d "${warmup_json}" \
      >/dev/null
  done

  echo "Sending measured forward request..."
  curl -fsS "${CONTAINER_URL}/profile/forward" \
    -H "Content-Type: application/json" \
    -d "${measured_json}" \
    | python3 -m json.tool

  cleanup
  trap - EXIT

  echo
  echo "CUDA forward profile complete:"
  echo "  ${TRACE_OUTPUT}"
  echo "  ${SERVER_LOG}"
  echo
  echo "Open the trace in chrome://tracing or Perfetto."
  exit 0
fi

install_nsys
check_nsys_importer

NSYS_OUTPUT="${TRACES_DIR}/${OUTPUT_STEM}"
echo "Starting ${MODE} server under Nsight Systems for CUDA forward-pass profiling..."
echo "Using prompt size: ${PROMPT_SIZE}; decode forwards: ${DECODE_STEPS}"
echo "Warmup runs: ${WARMUP_RUNS}"
echo "Writing profile to ${NSYS_OUTPUT}.nsys-rep"
echo "Writing server log to ${SERVER_LOG}"

INFERENCE_MODE="${MODE}" HOST="${HOST}" PORT="${PORT}" \
  nsys profile \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --trace=cuda,nvtx,cublas,cudnn,osrt \
    --output="${NSYS_OUTPUT}" \
    uv run --package qwen-inference qwen-serve --mode "${MODE}" \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"
wait_for_server

warmup_json="$(make_payload "${PROMPT_SIZE}" forward false)"
measured_json="$(make_payload "${PROMPT_SIZE}" forward true)"

echo "Sending ${WARMUP_RUNS} forward warmup request(s)..."
for _ in $(seq 1 "${WARMUP_RUNS}"); do
  curl -fsS "${CONTAINER_URL}/profile/forward" \
    -H "Content-Type: application/json" \
    -d "${warmup_json}" \
    >/dev/null
done

echo "Sending measured forward request..."
curl -fsS "${CONTAINER_URL}/profile/forward" \
  -H "Content-Type: application/json" \
  -d "${measured_json}" \
  | python3 -m json.tool

cleanup
trap - EXIT

echo
if [[ -f "${NSYS_OUTPUT}.nsys-rep" ]]; then
  echo "CUDA forward profile complete:"
  echo "  ${NSYS_OUTPUT}.nsys-rep"
  echo "  ${SERVER_LOG}"
  echo
  echo "Summarize with:"
  echo "  nsys stats ${NSYS_OUTPUT}.nsys-rep"
else
  echo "Nsight finished, but ${NSYS_OUTPUT}.nsys-rep was not created." >&2
  echo "Check the server log:" >&2
  echo "  ${SERVER_LOG}" >&2
  exit 1
fi
