#!/usr/bin/env bash
set -euo pipefail

# ── Required ───────────────────────────────────────────────────────────────────
INSTANCE_ID="${INSTANCE_ID:-i-0b72fe63ba947f187}"  # ollama (g6e.2xlarge)
KEY_PATH="${KEY_PATH:-$(dirname "$0")/../ollama.pem}"

# ── Optional ───────────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/ollama-benchmark}"
RESULTS_LOCAL_DIR="${RESULTS_LOCAL_DIR:-./results}"
MODEL_OVERRIDE="${MODEL_OVERRIDE:-}"
CONFIG_FILE="${CONFIG_FILE:-config/experiment.yaml}"
KEEP_INSTANCE="${KEEP_INSTANCE:-false}"

SSH_OPTS=(-i "${KEY_PATH}" -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5)
SCP_OPTS=(-i "${KEY_PATH}" -o StrictHostKeyChecking=no)

# ── Lifecycle ──────────────────────────────────────────────────────────────────
INSTANCE_WAS_RUNNING=false

stop_if_we_started() {
  if [[ "${INSTANCE_WAS_RUNNING}" == "false" && "${KEEP_INSTANCE}" != "true" ]]; then
    echo "Stopping instance ${INSTANCE_ID}..."
    aws ec2 stop-instances --instance-ids "${INSTANCE_ID}" --region "${AWS_REGION}" >/dev/null
    echo "Stop requested (instance will stop shortly)."
  else
    echo "Leaving instance ${INSTANCE_ID} as-is."
  fi
}
trap stop_if_we_started EXIT

# ── Start instance ─────────────────────────────────────────────────────────────
INITIAL_STATE="$(aws ec2 describe-instances \
  --instance-ids "${INSTANCE_ID}" \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text \
  --region "${AWS_REGION}")"

if [[ "${INITIAL_STATE}" == "running" ]]; then
  INSTANCE_WAS_RUNNING=true
  echo "Instance ${INSTANCE_ID} already running."
else
  # If the instance is still stopping, wait for it to fully stop first
  if [[ "${INITIAL_STATE}" == "stopping" ]]; then
    echo "Instance is stopping, waiting for it to fully stop..."
    aws ec2 wait instance-stopped --instance-ids "${INSTANCE_ID}" --region "${AWS_REGION}"
    echo "Instance stopped."
  fi
  echo "Starting instance ${INSTANCE_ID} (was: ${INITIAL_STATE})..."
  aws ec2 start-instances --instance-ids "${INSTANCE_ID}" --region "${AWS_REGION}" >/dev/null
  aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}" --region "${AWS_REGION}"
  echo "Instance running."
fi

PUBLIC_IP="$(aws ec2 describe-instances \
  --instance-ids "${INSTANCE_ID}" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region "${AWS_REGION}")"
echo "Public IP: ${PUBLIC_IP}"

# ── Wait for SSH ───────────────────────────────────────────────────────────────
echo "Waiting for SSH..."
SSH_READY=false
for _ in {1..60}; do
  if ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${PUBLIC_IP}" true 2>/dev/null; then
    SSH_READY=true
    break
  fi
  sleep 5
done

if [[ "${SSH_READY}" != "true" ]]; then
  echo "ERROR: SSH did not become available in time." >&2
  exit 1
fi
echo "SSH ready."

# ── Prepare config ─────────────────────────────────────────────────────────────
EFFECTIVE_CONFIG="${CONFIG_FILE}"
if [[ -n "${MODEL_OVERRIDE}" ]]; then
  EFFECTIVE_CONFIG="$(mktemp /tmp/experiment-XXXXXX.yaml)"
  sed "s|__MODEL__|${MODEL_OVERRIDE}|g" config/experiment.single-model.template.yaml > "${EFFECTIVE_CONFIG}"
  echo "Config generated for model: ${MODEL_OVERRIDE}"
fi

# ── Copy files ─────────────────────────────────────────────────────────────────
echo "Copying files to ${REMOTE_USER}@${PUBLIC_IP}:${REMOTE_DIR}..."
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${PUBLIC_IP}" \
  "mkdir -p '${REMOTE_DIR}'/{config,results,runner,utils,agent}"

scp "${SCP_OPTS[@]}" \
  "${EFFECTIVE_CONFIG}" \
  "${REMOTE_USER}@${PUBLIC_IP}:${REMOTE_DIR}/config/experiment.yaml"

scp "${SCP_OPTS[@]}" -r \
  runner \
  utils \
  agent \
  "${REMOTE_USER}@${PUBLIC_IP}:${REMOTE_DIR}/runner/"

echo "Files copied."

# ── Install Python deps (virtualenv, reused across runs) ──────────────────────
echo "Installing Python dependencies..."
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${PUBLIC_IP}" "
  sudo apt-get install -y -q python3-venv &&
  python3 -m venv /home/${REMOTE_USER}/ollama-venv &&
  /home/${REMOTE_USER}/ollama-venv/bin/pip install -q -r '${REMOTE_DIR}/runner/requirements.txt'
"

# ── Pull models ────────────────────────────────────────────────────────────────
MODELS_TO_PULL=()
if [[ -n "${MODEL_OVERRIDE}" ]]; then
  MODELS_TO_PULL=("${MODEL_OVERRIDE}")
else
  while IFS= read -r model; do
    MODELS_TO_PULL+=("${model}")
  done < <(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${EFFECTIVE_CONFIG}'))
for m in cfg.get('models', []):
    print(m)
")
fi

for model in "${MODELS_TO_PULL[@]}"; do
  echo "Pulling model: ${model}..."
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${PUBLIC_IP}" "ollama pull '${model}'"
done

# ── Run benchmark ──────────────────────────────────────────────────────────────
echo "Running benchmark..."
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${PUBLIC_IP}" \
  "cd '${REMOTE_DIR}' && \
   OLLAMA_BASE_URL=http://127.0.0.1:11434 \
   CONFIG_PATH='${REMOTE_DIR}/config/experiment.yaml' \
   RESULTS_DIR='${REMOTE_DIR}/results' \
   /home/${REMOTE_USER}/ollama-venv/bin/python3 runner/main.py"
echo "Benchmark complete."

# ── Copy results back ──────────────────────────────────────────────────────────
mkdir -p "${RESULTS_LOCAL_DIR}"
echo "Downloading results to ${RESULTS_LOCAL_DIR}/ ..."
scp "${SCP_OPTS[@]}" -r \
  "${REMOTE_USER}@${PUBLIC_IP}:${REMOTE_DIR}/results/." \
  "${RESULTS_LOCAL_DIR}/"
echo "Results saved to ${RESULTS_LOCAL_DIR}/"