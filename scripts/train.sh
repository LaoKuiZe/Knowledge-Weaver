#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/train.sh alfworld|webshop [--dry-run] [CONFIG_OVERRIDES...]

Examples:
  bash scripts/train.sh alfworld --dry-run
  bash scripts/train.sh alfworld total_train_steps=10
  bash scripts/serve_webshop.sh                 # Separate terminal
  bash scripts/train.sh webshop

Environment overrides:
  PYTHON_BIN            Python in the installed training environment (python3)
  CONFIG                YAML file (configs/train_<task>.yaml)
  ACTOR_MODEL_PATH      Model ID or local directory (Qwen/Qwen3.5-4B)
  ACTOR_SERVER_GPUS     Six physical executor GPU indices (0,1,2,3,4,5)
  AREAL_GPUS            Two disjoint physical training GPU indices (6,7)
  ALFWORLD_DATA_ROOT    ALFWorld json_2.1.1 directory
  WEBSHOP_DATA_ROOT     Full WebShop data directory, including search_engine/indexes
  WEBSHOP_ENV_URL       Running environment service (http://127.0.0.1:31080)
  ALFWORLD_ISOLATE_ENV_PROCESS  1 runs each TextWorld environment in a subprocess (1)
  AREAL_OUTPUT_ROOT, AREAL_CHECKPOINT_ROOT, CACHE_ROOT, HF_HOME
  EXPERIMENT_NAME, TRIAL_NAME, WANDB_MODE (disabled by default)

Overrides are passed to the training config last. --dry-run prints the command
without importing training dependencies, checking data, or allocating GPUs.
HELP
}
fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  alfworld|webshop) TASK="$1"; shift ;;
  *) usage >&2; exit 2 ;;
esac
DRY_RUN=0
OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; OVERRIDES+=("$@"); break ;;
    *=*) OVERRIDES+=("$1") ;;
    *) fail "Unknown argument: $1" ;;
  esac
  shift
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"
export AREAL_ROOT="${AREAL_ROOT:-${PROJECT_ROOT}/AReaL}"
export ALFWORLD_ROOT="${ALFWORLD_ROOT:-${PROJECT_ROOT}}"
export WEBSHOP_ROOT="${WEBSHOP_ROOT:-${WEBSHOP_REPO_ROOT:-${PROJECT_ROOT}/.benchmark-runtime/webshop}}"
export WEBSHOP_REPO_ROOT="${WEBSHOP_REPO_ROOT:-${WEBSHOP_ROOT}}"
export ALFWORLD_DATA_ROOT="${ALFWORLD_DATA_ROOT:-${PROJECT_ROOT}/data/alfworld/json_2.1.1}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$(dirname -- "${ALFWORLD_DATA_ROOT}")}"
export WEBSHOP_DATA_ROOT="${WEBSHOP_DATA_ROOT:-${PROJECT_ROOT}/data/webshop}"
export WEBSHOP_ENV_URL="${WEBSHOP_ENV_URL:-http://127.0.0.1:${WEBSHOP_PORT:-31080}}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-Qwen/Qwen3.5-4B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-knowledgeweaver-${TASK}}"
export TRIAL_NAME="${TRIAL_NAME:-train_$(date +%Y%m%d_%H%M%S)}"
export AREAL_OUTPUT_ROOT="${AREAL_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
export AREAL_CHECKPOINT_ROOT="${AREAL_CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
export AREAL_NAME_RESOLVE_ROOT="${AREAL_NAME_RESOLVE_ROOT:-${AREAL_OUTPUT_ROOT}/${EXPERIMENT_NAME}/${TRIAL_NAME}/name_resolve}"
export CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export AREAL_CACHE_DIR="${AREAL_CACHE_DIR:-${CACHE_ROOT}/areal/${TRIAL_NAME}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
export SGLANG_ACTOR_STORAGE_ROOT="${SGLANG_ACTOR_STORAGE_ROOT:-${CACHE_ROOT}/sglang/${TRIAL_NAME}/executor}"
export SGLANG_ROLLOUT_STORAGE_ROOT="${SGLANG_ROLLOUT_STORAGE_ROOT:-${CACHE_ROOT}/sglang/${TRIAL_NAME}/rollout}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DIR="${WANDB_DIR:-${AREAL_OUTPUT_ROOT}/wandb}"
export PYTHONPATH="${AREAL_ROOT}:${PROJECT_ROOT}:${WEBSHOP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
# Run each native TextWorld environment in its own short-lived process so a
# crash fails one episode instead of the shared rollout pool.
if [[ "${TASK}" == alfworld ]]; then
  export ALFWORLD_ISOLATE_ENV_PROCESS="${ALFWORLD_ISOLATE_ENV_PROCESS:-1}"
fi
export PYTHONFAULTHANDLER=1
export ACTOR_SERVER_CUDA_VISIBLE_DEVICES="${ACTOR_SERVER_GPUS:-0,1,2,3,4,5}"
export CUDA_VISIBLE_DEVICES="${AREAL_GPUS:-6,7}"
# TorchMemorySaver offload requires the standard CUDA allocator.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export PYTORCH_ALLOC_CONF="expandable_segments:False"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/train_${TASK}.yaml}"
[[ "${CONFIG}" == /* ]] || CONFIG="${PWD}/${CONFIG}"

# Reject overlapping/duplicate pools before importing a CUDA library.
[[ "${ACTOR_SERVER_CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || fail 'ACTOR_SERVER_GPUS must contain six comma-separated GPU indices.'
[[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+,[0-9]+$ ]] || fail 'AREAL_GPUS must contain two comma-separated GPU indices.'
IFS=, read -r -a GPU_IDS <<< "${ACTOR_SERVER_CUDA_VISIBLE_DEVICES},${CUDA_VISIBLE_DEVICES}"
SEEN=,
for GPU in "${GPU_IDS[@]}"; do
  [[ "${SEEN}" != *",${GPU},"* ]] || fail "Duplicate GPU ${GPU}: executor and training pools must be disjoint."
  SEEN+="${GPU},"
done

CMD=("${PYTHON_BIN}" "${AREAL_ROOT}/examples/${TASK}_skill/train.py" --config "${CONFIG}"
  "experiment_name=${EXPERIMENT_NAME}" "trial_name=${TRIAL_NAME}"
  scheduler.type=local cluster.n_nodes=1 cluster.n_gpus_per_node=2
  actor.backend=fsdp:d2 ref.backend=fsdp:d2 rollout.backend=sglang:d2
  "actor.path=${ACTOR_MODEL_PATH}" "actor_server.model_path=${ACTOR_MODEL_PATH}"
  actor_server.enabled=true actor_server.tensor_parallel_size=1 actor_server.data_parallel_size=6
  "actor_server.cuda_visible_devices='${ACTOR_SERVER_CUDA_VISIBLE_DEVICES}'"
  "actor_server.host=${ACTOR_SERVER_HOST:-127.0.0.1}" "actor_server.port=${ACTOR_SERVER_PORT:-30080}"
  actor.weight_update_mode=disk enable_offload=true actor.offload=true ref.offload=true
  actor.scheduling_strategy.type=separation
  ref.scheduling_strategy.type=colocation ref.scheduling_strategy.target=actor ref.scheduling_strategy.fork=true
  rollout.scheduling_strategy.type=colocation rollout.scheduling_strategy.target=actor rollout.scheduling_strategy.fork=true
  sglang.enable_memory_saver=true
  "actor.mb_spec.max_tokens_per_mb=${MAX_TOKENS_PER_MB:-6000}"
  "ref.mb_spec.max_tokens_per_mb=${MAX_TOKENS_PER_MB:-6000}"
  "gconfig.max_new_tokens=${GENERATION_MAX_NEW_TOKENS:-256}"
  "gconfig.max_tokens=${GENERATION_MAX_TOKENS:-6000}"
  "eval_gconfig.max_new_tokens=${GENERATION_MAX_NEW_TOKENS:-256}"
  "eval_gconfig.max_tokens=${GENERATION_MAX_TOKENS:-6000}"
  "actor_server.context_length=${SGLANG_CONTEXT_LENGTH:-16384}"
  "actor_server.mem_fraction_static=${SGLANG_ACTOR_MEM_FRACTION:-0.86}"
  'actor_server.extra_args=["--max-running-requests","256","--disable-cuda-graph","--file-storage-path","${oc.env:SGLANG_ACTOR_STORAGE_ROOT}"]'
  "sglang.context_length=${SGLANG_CONTEXT_LENGTH:-16384}"
  "sglang.mem_fraction_static=${SGLANG_ROLLOUT_MEM_FRACTION:-0.55}"
  ++sglang.disable_cuda_graph=true
  '++sglang.file_storage_path=${oc.env:SGLANG_ROLLOUT_STORAGE_ROOT}'
  "stats_logger.wandb.mode=${WANDB_MODE}")
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then CMD+=("${OVERRIDES[@]}"); fi
printf 'Executor GPUs: %s; AReaL GPUs: %s\n' "${ACTOR_SERVER_CUDA_VISIBLE_DEVICES}" "${CUDA_VISIBLE_DEVICES}"
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "${DRY_RUN}" == 0 ]] || exit 0

[[ -f "${CONFIG}" ]] || fail "Config not found: ${CONFIG}"
command -v "${PYTHON_BIN}" >/dev/null || fail "Python not found: ${PYTHON_BIN}"
command -v nvidia-smi >/dev/null || fail 'Training requires an NVIDIA GPU environment.'
nvidia-smi -i "${ACTOR_SERVER_CUDA_VISIBLE_DEVICES},${CUDA_VISIBLE_DEVICES}" --query-gpu=index,name --format=csv,noheader
if [[ "${TASK}" == alfworld ]]; then
  for SPLIT in train valid_seen valid_unseen; do
    [[ -d "${ALFWORLD_DATA_ROOT}/${SPLIT}" ]] || fail "Missing ALFWorld split: ${ALFWORLD_DATA_ROOT}/${SPLIT}"
  done
else
  "${PYTHON_BIN}" - <<'PY'
import os
from urllib.request import urlopen
url = os.environ["WEBSHOP_ENV_URL"].rstrip("/") + "/health"
try:
    with urlopen(url, timeout=10) as response:
        assert response.status == 200
except Exception as exc:
    raise SystemExit(f"WebShop service unavailable at {url}. Start scripts/serve_webshop.sh first. ({exc})")
PY
fi
mkdir -p "${AREAL_OUTPUT_ROOT}" "${AREAL_CHECKPOINT_ROOT}" "${AREAL_NAME_RESOLVE_ROOT}" \
  "${HF_HOME}" "${AREAL_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${WANDB_DIR}" "${SGLANG_ACTOR_STORAGE_ROOT}" "${SGLANG_ROLLOUT_STORAGE_ROOT}"
PYTHON_PATH="$(command -v "${PYTHON_BIN}")"
PYTHON_DIR="$(cd -- "$(dirname -- "${PYTHON_PATH}")" && pwd -P)"
CMD[0]="${PYTHON_DIR}/$(basename -- "${PYTHON_PATH}")"
export PATH="${PYTHON_DIR}:${PATH}"
cd "${AREAL_ROOT}"
exec "${CMD[@]}"
