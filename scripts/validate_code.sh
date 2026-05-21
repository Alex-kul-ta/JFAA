#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPYCACHEPREFIX="$ROOT_DIR/.pycache" python3 -m py_compile \
  train_jfaa_probe.py \
  evaluate_probe_heads.py \
  export_submission.py \
  ensemble_submissions.py \
  jfaa/*.py \
  evals/main.py \
  evals/scaffold.py \
  evals/action_anticipation_frozen/*.py \
  evals/action_anticipation_frozen/modelcustom/*.py
