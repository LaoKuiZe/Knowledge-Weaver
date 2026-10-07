#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec "${PYTHON_BIN:-python3}" "${ROOT}/scripts/build_knowledge_bank.py" "$@"
