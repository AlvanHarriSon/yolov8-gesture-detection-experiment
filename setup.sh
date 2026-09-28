#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

PY=""
for candidate in python3 python3.14 python3.13 python3.12 python3.11 python3.10; do
  if command -v "$candidate" >/dev/null 2>&1 && \
     "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    PY="$candidate"
    break
  fi
done

if [[ -z "$PY" ]]; then
  echo "Python 3.10 or newer is required."
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  "$PY" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "Environment ready: ./.venv/bin/python"
