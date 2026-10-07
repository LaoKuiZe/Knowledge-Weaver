#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/setup_env.sh

Install the training environment and pinned benchmark dependencies.
No datasets are downloaded and no training jobs are started.
Prerequisites: Linux x86_64, Python 3.12, uv, NVIDIA CUDA-compatible drivers,
and Java 11+ on PATH for the WebShop Lucene search engine.
Optional overrides: PYTHON_BIN (default python3.12), VENV (default .venv).
EOF
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "Unknown argument; use --help." >&2
  exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "Training setup requires Linux x86_64." >&2
  exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
for tool in "${PYTHON_BIN}" uv java; do
  command -v "${tool}" >/dev/null || { echo "Missing prerequisite: ${tool}" >&2; exit 2; }
done
"${PYTHON_BIN}" - <<'PY'
import re
import subprocess
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit("Python 3.12 is required.")
result = subprocess.run(["java", "-version"], capture_output=True, text=True)
match = re.search(r'version "(\d+)(?:\.(\d+))?', result.stderr + result.stdout)
major = int(match.group(2) if match and match.group(1) == "1" else match.group(1)) if match else 0
if result.returncode or major < 11:
    raise SystemExit("Java 11+ is required for WebShop; configure JAVA_HOME and PATH.")
PY
VENV="$("${PYTHON_BIN}" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "${VENV:-${ROOT}/.venv}")"
export UV_PROJECT_ENVIRONMENT="${VENV}"
export UV_LINK_MODE=copy

cd "${ROOT}/AReaL"
uv sync --frozen --no-dev --inexact --python "${PYTHON_BIN}" --extra cuda
export PATH="${VENV}/bin:${PATH}"
uv pip install --python "${VENV}/bin/python" --reinstall-package alfworld -r "${ROOT}/requirements-env.txt"
"${VENV}/bin/python" "${ROOT}/scripts/install_webshop.py" --register-pth
"${VENV}/bin/python" - "${ROOT}" <<'PY'
import pathlib
import site
import sys
root = pathlib.Path(sys.argv[1])
site_root = pathlib.Path(site.getsitepackages()[0])
(site_root / "knowledgeweaver_eval.pth").write_text(str(root / "generality_eval" / "src") + "\n")
PY
"${VENV}/bin/python" "${ROOT}/AReaL/examples/patch_textworld_runtime.py"
"${VENV}/bin/python" "${ROOT}/scripts/check_webshop_runtime.py"
echo "Environment ready. Activate with: source ${VENV}/bin/activate"
