#!/usr/bin/env bash
# Evaluate an existing bank with the frozen executor and semantic top-k retrieval.
set -euo pipefail
usage() {
  cat <<'HELP'
Usage: bash scripts/eval.sh alfworld|webshop [--dry-run] --skillbank PATH --output-dir PATH [--top-k K] [EVALUATOR_OPTIONS...]

Defaults: Qwen/Qwen3.5-4B frozen executor on four GPUs; one run at k=10.
Rollout workers are capped at 384 (ALFWorld) and 512 (WebShop), but each task
runs one episode at a time, so a single k uses at most 134 ALFWorld worker
processes or 100 WebShop threads. ALFWorld workers are separate processes;
set ROLLOUT_WORKERS lower on hosts with limited memory.
Use --top-k K to set the number of retrieved entries (0 disables retrieval).
ALFWorld: all 134 valid_unseen tasks. WebShop: frozen official test100 subset.
Use --base-url to attach to an existing executor server. The curator is never
loaded during evaluation. Use --help after the task for evaluator options.
Environment: PYTHON_BIN, ALFWORLD_DATA_ROOT, WEBSHOP_DATA_ROOT,
WEBSHOP_REPO_ROOT, ACTOR_MODEL_PATH, GPU_IDS, ROLLOUT_WORKERS.
HELP
}
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  alfworld|webshop) TASK="$1"; shift ;;
  *) usage >&2; exit 2 ;;
esac
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export ALFWORLD_ROOT="${ROOT}"
export ALFWORLD_DATA_ROOT="${ALFWORLD_DATA_ROOT:-${ROOT}/eval_data/alfworld/json_2.1.1}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$(dirname -- "${ALFWORLD_DATA_ROOT}")}"
export WEBSHOP_REPO_ROOT="${WEBSHOP_REPO_ROOT:-${WEBSHOP_ROOT:-${ROOT}/.benchmark-runtime/webshop}}"
export PYTHONPATH="${ROOT}/generality_eval/src:${ROOT}/AReaL:${ROOT}:${WEBSHOP_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WEBSHOP_DATA_ROOT="${WEBSHOP_DATA_ROOT:-${ROOT}/data/webshop}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
DRY_RUN=0
ARGS=()
for ARG in "$@"; do
  if [[ "${ARG}" == --dry-run ]]; then DRY_RUN=1; else ARGS+=("${ARG}"); fi
done
CMD=("${PYTHON_BIN:-python3}" -m "${TASK}_generality.run_semantic_skillbank_eval")
CMD+=(--config "${ROOT}/generality_eval/config/${TASK}_semantic_skillbank_eval.yaml")
if [[ "${TASK}" == alfworld ]]; then
  CMD+=(--repo-root "${ROOT}" --data-root "${ALFWORLD_DATA_ROOT}")
  [[ -z "${ACTOR_MODEL_PATH:-}" ]] || CMD+=(--checkpoint-path "${ACTOR_MODEL_PATH}")
else
  [[ -z "${ACTOR_MODEL_PATH:-}" ]] || CMD+=(--model-path "${ACTOR_MODEL_PATH}")
fi
[[ -z "${GPU_IDS:-}" ]] || CMD+=(--gpus "${GPU_IDS}")
[[ -z "${ROLLOUT_WORKERS:-}" ]] || CMD+=(--rollout-workers "${ROLLOUT_WORKERS}")
if [[ ${#ARGS[@]} -gt 0 ]]; then CMD+=("${ARGS[@]}"); fi
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
[[ "${DRY_RUN}" == 0 ]] || exit 0
exec "${CMD[@]}"
