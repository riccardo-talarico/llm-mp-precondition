#!/usr/bin/env bash
set -euo pipefail
[[ "${DEBUG:-0}" == "1" ]] && set -x

# ── Required ───────────────────────────────────────────────────────────────────
INSTANCE_ID="${INSTANCE_ID:-i-00df26dabf73bd992}"
KEY_PATH="${KEY_PATH:-~/.ssh/ollama.pem}"
# ── Optional ───────────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/ollama-benchmark}"
RESULTS_LOCAL_DIR="${RESULTS_LOCAL_DIR:-./results}"
MODEL_OVERRIDE="${MODEL_OVERRIDE:-}"
CONFIG_FILE="${CONFIG_FILE:-config/experiment.yaml}"
KEEP_INSTANCE="${KEEP_INSTANCE:-false}"   # ← false = stoppa sempre alla fine
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-/opt/dlami/nvme/ollama/models}"  # ← NVMe

SSH_OPTS=(-i "${KEY_PATH}" -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5)
SCP_OPTS=(-i "${KEY_PATH}" -o StrictHostKeyChecking=no)

# ── Lifecycle ──────────────────────────────────────────────────────────────────
INSTANCE_WAS_RUNNING=false

stop_if_we_started() {
  if [[ "${KEEP_INSTANCE}" != "true" ]]; then
    echo "Stopping instance ${INSTANCE_ID}..."
    aws ec2 stop-instances --instance-ids "${INSTANCE_ID}" --region "${AWS_REGION}" >/dev/null
    echo "Stop requested (instance will stop shortly)."
  else
    echo "Leaving instance ${INSTANCE_ID} as-is (KEEP_INSTANCE=true)."
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

# ── Install Ollama + configure NVMe ───────────────────────────────────────────
echo "Installing Ollama and configuring NVMe storage..."
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${PUBLIC_IP}" "
  # Installa Ollama se non presente
  if ! command -v ollama &>/dev/null; then
    echo 'Installing Ollama...'
    curl -fsSL https://ollama.com/install.sh | sh
  else
    echo 'Ollama already installed, skipping.'
  fi

  # Crea cartella modelli sull'NVMe
  #DEBUG:
  systemctl status ollama

  sudo mkdir -p '${OLLAMA_MODELS_DIR}'
  sudo chmod +x /opt /opt/dlami /opt/dlami/nvme
  #sudo chown -R ${REMOTE_USER}:${REMOTE_USER} '${OLLAMA_MODELS_DIR}'
  sudo chown -R ollama:ollama '${OLLAMA_MODELS_DIR}'

  # Configura Ollama per usare NVMe
  sudo mkdir -p /etc/systemd/system/ollama.service.d
  sudo tee /etc/systemd/system/ollama.service.d/override.conf <<EOF
[Service]
Environment=\"OLLAMA_MODELS=${OLLAMA_MODELS_DIR}\" \"OLLAMA_NUM_PARALLEL=1\"
EOF

  # Riavvia Ollama con la nuova configurazione
  sudo systemctl daemon-reload
  sudo systemctl enable ollama
  sudo systemctl restart ollama

  # Riavvia Ollama e aspetta che sia pronto
  echo 'Waiting for Ollama to be ready...'
  OLLAMA_READY=false
  for i in {1..30}; do
    if curl -s http://127.0.0.1:11434/api/tags &>/dev/null; then
      echo 'Ollama ready.'
      OLLAMA_READY=true
      break
    fi
    sleep 2
  done
  if [[ \"\${OLLAMA_READY}\" != \"true\" ]]; then
    echo 'Ollama not responding, trying ollama serve...'
    nohup ollama serve &>/tmp/ollama.log &
    sleep 5
  fi
"

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
  "mkdir -p '${REMOTE_DIR}'/{config,utils,results,agent,runner,benchmarks,benchmarks_paths}"

scp "${SCP_OPTS[@]}" \
  "${EFFECTIVE_CONFIG}" \
  "${REMOTE_USER}@${PUBLIC_IP}:${REMOTE_DIR}/config/experiment.yaml"

scp "${SCP_OPTS[@]}" -r utils \
  "${REMOTE_USER}@${PUBLIC_IP}:${REMOTE_DIR}/"

scp "${SCP_OPTS[@]}" -r runner\
  "${REMOTE_USER}@${PUBLIC_IP}:${REMOTE_DIR}/"

scp "${SCP_OPTS[@]}" -r agent\
  "${REMOTE_USER}@${PUBLIC_IP}:${REMOTE_DIR}/"

## Benchmarks
#scp "${SCP_OPTS[@]}" -r benchmarks \
#  "${REMOTE_USER}@${PUBLIC_IP}:${REMOTE_DIR}/"
#
# Benchmark paths
scp "${SCP_OPTS[@]}" -r benchmarks_paths \
  "${REMOTE_USER}@${PUBLIC_IP}:${REMOTE_DIR}/"

echo "Files copied."





# ── Install Python deps ────────────────────────────────────────────────────────
echo "Installing Python dependencies..."
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${PUBLIC_IP}" "
  sudo apt-get install -y -q python3-venv &&
  python3 -m venv /home/${REMOTE_USER}/ollama-venv &&
  /home/${REMOTE_USER}/ollama-venv/bin/pip install -r '${REMOTE_DIR}/runner/requirements.txt'
"

# ── Pull models ────────────────────────────────────────────────────────────────
# Note: for this part of the script to work you must have yaml in your local pyhton environment
MODELS_TO_PULL=()
if [[ -n "${MODEL_OVERRIDE}" ]]; then
  MODELS_TO_PULL=("${MODEL_OVERRIDE}")
else
  while IFS= read -r model; do
    MODELS_TO_PULL+=("${model}")
  done < <(python3 -c "
import yaml, sys
try:
    with open('${EFFECTIVE_CONFIG}') as f:
        cfg = yaml.safe_load(f)
    # Check if 'models' is a list, otherwise fallback to single 'model'
    models = cfg.get('models', None)
    if not isinstance(models, list):
        models = [cfg.get('model')]
    for m in models:
        if m: print(m)
except Exception as e:
    pass
")
fi

# Debug: Check if we actually found anything
if [ ${#MODELS_TO_PULL[@]} -eq 0 ]; then
    echo "Warning: No models found in config. Check ${EFFECTIVE_CONFIG}"
fi

for model in "${MODELS_TO_PULL[@]}"; do
  echo "Pulling model: ${model}..."
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${PUBLIC_IP}" \
    "OLLAMA_MODELS='${OLLAMA_MODELS_DIR}' ollama pull '${model}'"
done

# DEBUG:
#ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${PUBLIC_IP}" "ollama pull llama3.1:8b"

# ── Run benchmark ──────────────────────────────────────────────────────────────
echo "Running benchmark..."
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${PUBLIC_IP}" \
  "cd '${REMOTE_DIR}' && \
   PYTHONPATH='${REMOTE_DIR}' \
   OLLAMA_BASE_URL=http://127.0.0.1:11434 \
   OLLAMA_MODELS='${OLLAMA_MODELS_DIR}' \
   CONFIG_PATH='${REMOTE_DIR}/config/experiment.yaml' \
   RESULTS_DIR='${REMOTE_DIR}/results' \
   /home/${REMOTE_USER}/ollama-venv/bin/python3 -u -m runner.main"
echo "Benchmark complete."

# ── Copy results back ──────────────────────────────────────────────────────────
mkdir -p "${RESULTS_LOCAL_DIR}"
echo "Downloading results to ${RESULTS_LOCAL_DIR}/ ..."
scp "${SCP_OPTS[@]}" -r \
  "${REMOTE_USER}@${PUBLIC_IP}:${REMOTE_DIR}/results/*" \
  "${RESULTS_LOCAL_DIR}/"
echo "Results saved to ${RESULTS_LOCAL_DIR}/"