# Efficient Qwen: Minimizing Inference Latency for Qwen3.5-4B on A10G

Source: <https://adaptfm.gitlab.io/call-for-competition/>

Join the AdaptFM Slack for announcements, Q&A, and updates: [adaptfm.slack.com](https://join.slack.com/t/adaptfm/shared_invite/zt-3vuvm0rdx-mpkxxnqOBm6Xy8oUFadg5g)

---

## The Challenge

How fast can you make Qwen3.5-4B run on a single NVIDIA A10G GPU without breaking it?

We provide a base Docker image with the unoptimised Qwen3.5-4B model serving on an AWS `g5.xlarge` instance. Your job: make inference as fast as possible while keeping model quality above minimum thresholds on standard benchmarks.

Quantize it. Prune it. Rewrite the kernels. Swap the inference engine. Anything goes, as long as the model still works.

---

## Evaluation & Ranking

Rankings are by average speedup over the unoptimized baseline:

| Category | Prompt tokens | Output tokens | Baseline latency |
| --- | --- | --- | --- |
| Short | 64 | 128 | 2,582 ms |
| Medium | 2,048 | 256 | 5,441 ms |
| Long | 8,192 | 256 | 6,576 ms |
| Average | — | — | 4,866 ms |

### Quality Gates

Your optimized model must still pass these benchmarks. Fail any and your submission is invalid.

| Benchmark | Baseline | Minimum Threshold |
| --- | --- | --- |
| MMLU-Pro | 0.690 | ≥ 0.621 (90%) |
| IFEval | 0.857 | ≥ 0.814 (95%) |
| GPQA-Diamond | 0.700 | ≥ 0.630 (90%) |

Evaluation mode: Quality benchmarks use `/v1/chat/completions` with the Qwen3.5-4B chat template. GPQA-Diamond uses thinking enabled; MMLU-Pro and IFEval use thinking disabled.

You can submit multiple times. Only your best valid submission appears on the leaderboard.

Live leaderboard: <https://d1krc5fcnf73gi.cloudfront.net>

## What's Allowed

- Quantization (AWQ, GPTQ, GGUF, etc. at any bit-width)
- Pruning (structured, unstructured, semi-structured)
- Knowledge distillation (student must init from Qwen3.5-4B)
- Architecture modifications (layer removal, head pruning)
- Custom CUDA/Triton kernels (source code required)
- Custom inference engines (TensorRT-LLM, llama.cpp, anything)
- KV-cache optimization, operator fusion, speculative decoding
- `torch.compile`, Flash Attention variants
- Memory offloading to system RAM

## What's NOT Allowed

- Hard-coded or cached benchmark answers
- Benchmark detection/routing logic
- Obfuscated code
- Training on evaluation benchmark data
- Proprietary/non-distributable software
- Parameters exceeding original Qwen3.5-4B count

---

## Getting Started

### 1. Pull the Base Image

```bash
docker pull adaptfm/adaptfm-base:latest
```

### 2. Download Model Weights

```bash
pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B', local_dir='./qwen-weights')"
```

### 3. Test Locally

```bash
docker run -d --gpus all -p 8080:8080 --name test-run \
  -v hf_cache:/root/.cache/huggingface --shm-size=4g \
  adaptfm/adaptfm-base:latest

# Wait ~3 min for model to load, then test
curl http://localhost:8080/ping

# Test with thinking disabled (MMLU-Pro/IFEval mode)
curl -s -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":32}' | python3 -m json.tool

# Test with thinking enabled (GPQA mode)
curl -s -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":512,"thinking":true}' | python3 -m json.tool

docker rm -f test-run
```

### 4. Build Your Submission

Important: Your submission image must include model weights baked in at `/opt/ml/model/`. The evaluation environment has no internet access.

```dockerfile
FROM adaptfm/adaptfm-base:latest

# Copy model weights into the image (REQUIRED)
COPY qwen-weights/ /opt/ml/model/

# Add your optimizations / custom serve script
COPY my_serve.py /opt/program/my_serve.py

# Disable network access for model loading
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# Must serve /ping + /invocations + /v1/completions + /v1/chat/completions on port 8080
ENTRYPOINT ["python3", "/opt/program/my_serve.py"]
```

```bash
docker build -t my-submission:latest .
```

## API Contract

API update (May 14): The `/v1/chat/completions` endpoint is now required. Submissions that only expose `/ping`, `/invocations`, and `/v1/completions` will fail quality evaluation. Please update your container accordingly.

Your container must serve on port 8080:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/ping` | GET | Return `200` when model is loaded and ready |
| `/invocations` | POST | Accept inference request, return output |
| `/v1/completions` | POST | Raw text completions (used for latency benchmark) |
| `/v1/chat/completions` | POST | Chat completions with template (used for quality benchmarks) |

Request format (`/invocations` and `/v1/completions`):

```json
{"prompt": "...", "max_tokens": 128, "temperature": 0.0}
```

Request format (`/v1/chat/completions`):

```json
{"model": "Qwen/Qwen3.5-4B", "messages": [{"role": "user", "content": "..."}], "max_tokens": 128, "temperature": 0.0}
```

Response format (OpenAI-compatible):

```json
{
  "choices": [{"text": "generated text here"}]
}
```

---

## Thinking Mode

Qwen3.5-4B generates `<think>...</think>` tokens by default. For the competition:

- Latency benchmark: Uses `/v1/completions` (raw text, no chat template, no thinking)
- Quality (MMLU-Pro, IFEval): Uses `/v1/chat/completions` with thinking disabled
- Quality (GPQA-Diamond): Uses `/v1/chat/completions` with thinking enabled + streaming

To disable thinking, use a chat template that outputs an empty think block. They provide [qwen_no_think.jinja](https://adaptfm.gitlab.io/assets/other/qwen_no_think.jinja):

```bash
# vLLM example
--chat-template /path/to/qwen_no_think.jinja
```

---

## Local Evaluation

### Prerequisites

- NVIDIA GPU with ≥24 GB VRAM (A10G or better)
- Docker with GPU support (`nvidia-container-toolkit`)
- Python 3.10+ with: `pip install lm-eval==0.4.11 langdetect immutabledict`

### Download Eval Scripts

- [run_eval_local.py](https://adaptfm.gitlab.io/assets/other/run_eval_local.py) — Full eval (latency + quality)
- [run_quality_local.py](https://adaptfm.gitlab.io/assets/other/run_quality_local.py) — Quality eval only
- [qwen_no_think.jinja](https://adaptfm.gitlab.io/assets/other/qwen_no_think.jinja) — Chat template with thinking disabled

### Run

```bash
# Start your container
docker run -d --gpus all -p 8080:8080 --name test-submission my-submission:latest
watch -n5 'curl -s http://localhost:8080/ping'  # wait for 200

# Quality only (~20 min with 10% sample)
HF_HOME=/path/to/hf_cache QUALITY_LIMIT=0.1 NUM_CONCURRENT=8 \
  python3 run_quality_local.py 2>&1 | tee /tmp/quality.log

# Full eval — latency + quality (~60 min with 10% sample)
HF_HOME=/path/to/hf_cache EVAL_MODE=full QUALITY_LIMIT=0.1 NUM_CONCURRENT=8 \
  python3 run_eval_local.py 2>&1 | tee /tmp/eval.log
```

### Eval Harness Configuration

| Task | Mode | Concurrency | Max tokens |
| --- | --- | --- | --- |
| MMLU-Pro (5-shot) | Chat completions, thinking OFF | 8 | 512 |
| IFEval (0-shot) | Chat completions, thinking OFF | 8 | 512 |
| GPQA-Diamond (0-shot) | Chat completions, thinking ON + streaming | 8 | 12288 |

Latency: 5 warmup + 50 measurement runs × 3 categories.

---

## Submission Guide

### API Details

| Item | Value |
| --- | --- |
| API Base URL | `https://79x0as8g44.execute-api.us-east-1.amazonaws.com/prod` |
| API Key | `qoXdZQNYbX1s4wnhmcBJG2APABlqjNVSao8CdM3j` |
| Max image size | 20 GB |
| Tarball filename | Must be `image.tar.gz` |
| Upload URL expiry | 2 hours |

### Step 1 — Register Your Team

```bash
curl -s -X POST \
  https://79x0as8g44.execute-api.us-east-1.amazonaws.com/prod/register \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: qoXdZQNYbX1s4wnhmcBJG2APABlqjNVSao8CdM3j' \
  -d '{"team_id": "AFM-xxxxxxxx"}' | python3 -m json.tool
```

Replace `AFM-xxxxxxxx` with the team ID you received upon registration.

### Step 2 — Save Image as Tarball

The tarball must be named `image.tar.gz`. Any other filename will cause the submission to fail.

```bash
docker save my-submission:latest | gzip > image.tar.gz
du -sh image.tar.gz  # must be under 20 GB
```

### Step 3 — Get Upload URL

```bash
FILE_SIZE=$(stat -c%s image.tar.gz 2>/dev/null || stat -f%z image.tar.gz)

curl -s -X POST \
  https://79x0as8g44.execute-api.us-east-1.amazonaws.com/prod/upload-url \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: qoXdZQNYbX1s4wnhmcBJG2APABlqjNVSao8CdM3j' \
  -d "{\"team_id\": \"AFM-xxxxxxxx\", \"file_size_bytes\": $FILE_SIZE}" \
  | tee /tmp/upload_resp.json | python3 -m json.tool
```

The response will contain `upload_type`:

- `single` — file ≤ 5 GB → follow Step 4A
- `multipart` — file > 5 GB → follow Step 4B

### Step 4A — Upload (Single, ≤ 5 GB)

```bash
UPLOAD_URL=$(python3 -c "import json; print(json.load(open('/tmp/upload_resp.json'))['upload_url'])")

curl -X PUT "$UPLOAD_URL" \
  --upload-file image.tar.gz \
  --progress-bar \
  -w "\nHTTP Status: %{http_code}\n"
```

Expected: `HTTP Status: 200` — then go to Step 5.

### Step 4B — Upload (Multipart, > 5 GB)

Supports resume — if interrupted, re-run and it skips already-uploaded parts.

```bash
python3 - << 'EOF'
import json, os, subprocess, sys

resp    = json.load(open('/tmp/upload_resp.json'))
team_id = resp['s3_key'].split('/')[1]

etags_file = '/tmp/upload_etags.json'
if os.path.exists(etags_file):
    etags = json.load(open(etags_file))
    done  = {e['part_number'] for e in etags}
    print(f"Resuming — {len(done)} parts already done: {sorted(done)}")
else:
    etags = []
    done  = set()

print(f"Uploading {resp['num_parts']} parts ({resp['part_size']//1024//1024} MB each)...")

with open('image.tar.gz', 'rb') as f:
    for part in resp['part_urls']:
        n = part['part_number']
        if n in done:
            f.seek(resp['part_size'], 1)
            continue
        chunk = f.read(resp['part_size'])
        if not chunk:
            break
        tmp = f'/tmp/part_{n}.bin'
        with open(tmp, 'wb') as pf:
            pf.write(chunk)
        result = subprocess.run(
            ['curl', '-s', '-X', 'PUT', part['upload_url'],
             '--upload-file', tmp, '-D', '-', '-o', '/dev/null'],
            capture_output=True, text=True
        )
        os.remove(tmp)
        etag = ''
        for line in result.stdout.splitlines():
            if line.lower().startswith('etag:'):
                etag = line.split(':', 1)[1].strip().strip('\r').strip('"')
                break
        if not etag:
            print(f'  ERROR: No ETag for part {n}. Re-run to resume.')
            sys.exit(1)
        etags.append({'part_number': n, 'etag': etag})
        done.add(n)
        with open(etags_file, 'w') as ef:
            json.dump(etags, ef)
        print(f'  Part {n}/{resp["num_parts"]} done')

print(f'\nAll parts uploaded. Completing...')
body = json.dumps({
    'team_id':   team_id,
    's3_key':    resp['s3_key'],
    'upload_id': resp['upload_id'],
    'parts':     etags,
})
result = subprocess.run(
    ['curl', '-s', '-X', 'POST',
     'https://79x0as8g44.execute-api.us-east-1.amazonaws.com/prod/complete-upload',
     '-H', 'Content-Type: application/json',
     '-H', 'x-api-key: qoXdZQNYbX1s4wnhmcBJG2APABlqjNVSao8CdM3j',
     '-d', body],
    capture_output=True, text=True
)
print(json.dumps(json.loads(result.stdout), indent=2))
os.remove(etags_file)
EOF
```

### Step 5 — Submit for Evaluation

```bash
S3_KEY=$(python3 -c "import json; print(json.load(open('/tmp/upload_resp.json'))['s3_key'])")

curl -s -X POST \
  https://79x0as8g44.execute-api.us-east-1.amazonaws.com/prod/submit \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: qoXdZQNYbX1s4wnhmcBJG2APABlqjNVSao8CdM3j' \
  -d "{\"team_id\": \"AFM-xxxxxxxx\", \"s3_key\": \"$S3_KEY\"}" \
  | python3 -m json.tool
```

Save your `submission_id` from the response — share it with organizers if you face any issues.

### Resubmission

To resubmit, repeat Steps 2–5 with your updated image. Evaluation takes ~90–100 minutes.

---

## Timeline

Registration and submissions are now open. See Contact Details below for the portal link.

| Date | Milestone |
| --- | --- |
| April 20 | Competition launches — rules + base Docker published |
| May 8 | Registration opens |
| May 14 | Submission portal opens - Submissions begin |
| May 30 | Registration deadline (teams of 1–4) |
| June 15 | Submissions close (23:59 AoE) |
| June 19 | Final leaderboard announced |
| July 11 | Top teams present at ICML 2026, Seoul |

---

## Prizes

| Place | Prize | Presentation |
| --- | --- | --- |
| 1st | $3,000 | Oral + poster |
| 2nd | $2,000 | Oral + poster |
| 3rd | $1,000 | Poster |

Top 10 teams must open-source their code and model weights under a permissive license (BSD, MIT, Apache, etc.).

---

## Who Can Participate

- Open to individuals and teams worldwide (1–4 members per team)
- Amazon employees may not submit solutions
- Academic and industry participants welcome
- No ICML registration required to submit (only to present)
- Max 1 submission per team per day

---

## Contact

- Email: [adaptfmworkshop@gmail.com](mailto:adaptfmworkshop@gmail.com)
- Slack: [AdaptFM Slack](https://join.slack.com/t/adaptfm/shared_invite/zt-3vuvm0rdx-mpkxxnqOBm6Xy8oUFadg5g) — `#efficient-qwen-competition` channel
- Registration: [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSecrr-1kELwqzJiMg2PBcrSl-2jpK_TYGFRmNZO4SaPZy6Fbw/viewform)
- Leaderboard: <https://d1krc5fcnf73gi.cloudfront.net>
