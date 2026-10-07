#!/usr/bin/env bash
# Run the CPU/JVM WebShop simulator separately from GPU training.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/serve_webshop.sh [--dry-run] [CONFIG_OVERRIDES...]

Uses the full WebShop catalog and search index. Set WEBSHOP_DATA_ROOT to a
folder containing items_shuffle.json, items_ins_v2.json, items_human_ins.json,
and search_engine/indexes. Default: <repo>/data/webshop.

Optional: PYTHON_BIN, CONFIG, WEBSHOP_ROOT, WEBSHOP_HOST (127.0.0.1),
WEBSHOP_PORT (31080), WEBSHOP_SPACY_MODEL_PATH, JAVA_HOME.
Pass matching webshop.* overrides to this service and scripts/train.sh.
--dry-run prints the command without importing dependencies or loading data.
HELP
}
fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
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
export WEBSHOP_DATA_ROOT="${WEBSHOP_DATA_ROOT:-${PROJECT_ROOT}/data/webshop}"
export WEBSHOP_ENV_URL="${WEBSHOP_ENV_URL:-http://127.0.0.1:${WEBSHOP_PORT:-31080}}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-Qwen/Qwen3.5-4B}"
export AREAL_OUTPUT_ROOT="${AREAL_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
export AREAL_CHECKPOINT_ROOT="${AREAL_CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
export AREAL_NAME_RESOLVE_ROOT="${AREAL_NAME_RESOLVE_ROOT:-${AREAL_OUTPUT_ROOT}/webshop_service/name_resolve}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${AREAL_ROOT}:${PROJECT_ROOT}:${WEBSHOP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/train_webshop.yaml}"
[[ "${CONFIG}" == /* ]] || CONFIG="${PWD}/${CONFIG}"
CMD=("${PYTHON_BIN}" "${AREAL_ROOT}/examples/webshop_skill/env_service.py"
  --config "${CONFIG}" --host "${WEBSHOP_HOST:-127.0.0.1}" --port "${WEBSHOP_PORT:-31080}"
  --repo-root "${WEBSHOP_ROOT}"
  --products-file "${WEBSHOP_DATA_ROOT}/items_shuffle.json"
  --attributes-file "${WEBSHOP_DATA_ROOT}/items_ins_v2.json"
  --human-attributes-file "${WEBSHOP_DATA_ROOT}/items_human_ins.json"
  --search-index "${WEBSHOP_DATA_ROOT}/search_engine/indexes"
  --training-overrides)
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then CMD+=("${OVERRIDES[@]}"); fi
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "${DRY_RUN}" == 0 ]] || exit 0

[[ -f "${CONFIG}" ]] || fail "Config not found: ${CONFIG}"
command -v "${PYTHON_BIN}" >/dev/null || fail "Python not found: ${PYTHON_BIN}"
command -v java >/dev/null || fail 'Java is required for the WebShop search index.'
for DATA_FILE in items_shuffle.json items_ins_v2.json items_human_ins.json; do
  [[ -s "${WEBSHOP_DATA_ROOT}/${DATA_FILE}" ]] || fail "Missing WebShop data: ${WEBSHOP_DATA_ROOT}/${DATA_FILE}"
done
[[ -d "${WEBSHOP_DATA_ROOT}/search_engine/indexes" ]] || fail 'Missing full WebShop search index.'
PYTHON_PATH="$(command -v "${PYTHON_BIN}")"
PYTHON_DIR="$(cd -- "$(dirname -- "${PYTHON_PATH}")" && pwd -P)"
CMD[0]="${PYTHON_DIR}/$(basename -- "${PYTHON_PATH}")"
export PATH="${PYTHON_DIR}:${PATH}"
cd "${AREAL_ROOT}"
exec "${CMD[@]}"
