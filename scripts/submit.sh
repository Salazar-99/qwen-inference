#!/usr/bin/env bash
# Submit a Docker image to the AdaptFM Efficient Qwen competition.
#
# Usage:
#   bash scripts/submit.sh <team_id>
#
# Example:
#   bash scripts/submit.sh AFM-xxxxxxxx
#
# The script:
#   1. Builds the submission image from qwen-inference/Dockerfile
#   2. Saves it as image.tar.gz (must be ≤ 20 GB)
#   3. Registers the team (no-op if already registered)
#   4. Requests an upload URL and uploads the tarball
#   5. Submits for evaluation
#
# Requirements: docker, curl, python3, jq (optional, for prettier output)

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_CTX="${REPO_ROOT}/qwen-inference"

API_BASE="https://79x0as8g44.execute-api.us-east-1.amazonaws.com/prod"
API_KEY="qoXdZQNYbX1s4wnhmcBJG2APABlqjNVSao8CdM3j"
IMAGE_TAG="qwen-submission:latest"
TARBALL="image.tar.gz"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <team_id>" >&2
  echo "  team_id: your AFM-xxxxxxxx team identifier" >&2
  exit 2
fi
TEAM_ID="$1"

pjson() {
  python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin), indent=2))" 2>/dev/null || cat
}

# ---------------------------------------------------------------------------
# Step 1 — Build Docker image
# ---------------------------------------------------------------------------
echo "=== Step 1: Building Docker image ==="
echo "Context: ${DOCKER_CTX}"
docker build -t "${IMAGE_TAG}" "${DOCKER_CTX}"
echo "Build complete: ${IMAGE_TAG}"

# ---------------------------------------------------------------------------
# Step 2 — Save as tarball
# ---------------------------------------------------------------------------
echo
echo "=== Step 2: Saving image to ${TARBALL} ==="
docker save "${IMAGE_TAG}" | gzip > "${TARBALL}"
FILE_SIZE=$(stat -c%s "${TARBALL}" 2>/dev/null || stat -f%z "${TARBALL}")
FILE_SIZE_GB=$(python3 -c "print(f'{${FILE_SIZE}/1024**3:.2f}')")
echo "Tarball size: ${FILE_SIZE_GB} GB (${FILE_SIZE} bytes)"
if [[ "${FILE_SIZE}" -gt $((20 * 1024 * 1024 * 1024)) ]]; then
  echo "ERROR: Image exceeds 20 GB limit." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 3 — Register team (idempotent)
# ---------------------------------------------------------------------------
echo
echo "=== Step 3: Registering team ${TEAM_ID} ==="
curl -s -X POST "${API_BASE}/register" \
  -H 'Content-Type: application/json' \
  -H "x-api-key: ${API_KEY}" \
  -d "{\"team_id\": \"${TEAM_ID}\"}" | pjson

# ---------------------------------------------------------------------------
# Step 4 — Get upload URL and upload
# ---------------------------------------------------------------------------
echo
echo "=== Step 4: Getting upload URL ==="
curl -s -X POST "${API_BASE}/upload-url" \
  -H 'Content-Type: application/json' \
  -H "x-api-key: ${API_KEY}" \
  -d "{\"team_id\": \"${TEAM_ID}\", \"file_size_bytes\": ${FILE_SIZE}}" \
  | tee /tmp/upload_resp.json | pjson

UPLOAD_TYPE=$(python3 -c "import json; print(json.load(open('/tmp/upload_resp.json'))['upload_type'])")
echo "Upload type: ${UPLOAD_TYPE}"

if [[ "${UPLOAD_TYPE}" == "single" ]]; then
  echo
  echo "=== Step 4A: Single-part upload ==="
  UPLOAD_URL=$(python3 -c "import json; print(json.load(open('/tmp/upload_resp.json'))['upload_url'])")
  curl -X PUT "${UPLOAD_URL}" \
    --upload-file "${TARBALL}" \
    --progress-bar \
    -w "\nHTTP Status: %{http_code}\n"
else
  echo
  echo "=== Step 4B: Multipart upload ==="
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
if os.path.exists(etags_file):
    os.remove(etags_file)
EOF
fi

# ---------------------------------------------------------------------------
# Step 5 — Submit for evaluation
# ---------------------------------------------------------------------------
echo
echo "=== Step 5: Submitting for evaluation ==="
S3_KEY=$(python3 -c "import json; print(json.load(open('/tmp/upload_resp.json'))['s3_key'])")
SUBMIT_RESP=$(curl -s -X POST "${API_BASE}/submit" \
  -H 'Content-Type: application/json' \
  -H "x-api-key: ${API_KEY}" \
  -d "{\"team_id\": \"${TEAM_ID}\", \"s3_key\": \"${S3_KEY}\"}")
echo "${SUBMIT_RESP}" | pjson

SUBMISSION_ID=$(python3 -c "import json,sys; d=json.loads('${SUBMIT_RESP}'); print(d.get('submission_id','(not found)'))" 2>/dev/null || echo "(parse error)")
echo
echo "==================================================================="
echo "Submission complete!"
echo "  Team:          ${TEAM_ID}"
echo "  Submission ID: ${SUBMISSION_ID}"
echo "  Leaderboard:   https://d1krc5fcnf73gi.cloudfront.net"
echo "  Evaluation:    ~90-100 minutes"
echo "==================================================================="
