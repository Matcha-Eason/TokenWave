#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
DATASET_ID="${DATASET_ID:-togethercomputer/CoderForge-Preview}"
DATA_SUBDIR="${DATA_SUBDIR:-trajectories-tokenized_qwencoder}"
if [[ -n "${DATA_DIR:-}" ]]; then
  DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$(dirname "${DATA_DIR}")}"
else
  DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-data/CoderForge-Preview}"
  DATA_DIR="${DOWNLOAD_ROOT}/${DATA_SUBDIR}"
fi
OUT_DIR="${OUT_DIR:-outputs/coderforge_full}"
MODEL_SCOPE_REVISION="${MODEL_SCOPE_REVISION:-master}"
MODEL_SCOPE_MAX_WORKERS="${MODEL_SCOPE_MAX_WORKERS:-8}"
MODELSCOPE_TOKEN="${MODELSCOPE_TOKEN:-}"
MAX_FILES="${MAX_FILES:-0}"
WRITE_CSV="${WRITE_CSV:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

echo "[1/4] Preparing Python environment: ${VENV_DIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt

echo "[2/4] Preparing data directory: ${DOWNLOAD_ROOT}"
mkdir -p "${DOWNLOAD_ROOT}"

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  echo "[3/4] Downloading ModelScope dataset shard directory"
  echo "      dataset=${DATASET_ID}"
  echo "      include=${DATA_SUBDIR}/**"
  echo "      local_dir=${DOWNLOAD_ROOT}"
  DOWNLOAD_ARGS=(
    download
    --dataset "${DATASET_ID}"
    --revision "${MODEL_SCOPE_REVISION}"
    --include "${DATA_SUBDIR}/**"
    --local_dir "${DOWNLOAD_ROOT}"
    --max-workers "${MODEL_SCOPE_MAX_WORKERS}"
  )
  if [[ -n "${MODELSCOPE_TOKEN}" ]]; then
    DOWNLOAD_ARGS+=(--token "${MODELSCOPE_TOKEN}")
  fi
  "${VENV_DIR}/bin/modelscope" "${DOWNLOAD_ARGS[@]}"
else
  echo "[3/4] SKIP_DOWNLOAD=1, using existing files in ${DATA_DIR}"
fi

echo "[4/4] Extracting turn time-series metrics"
ARGS=(
  scripts/build_full_timeseries.py
  --input "${DATA_DIR}"
  --out-dir "${OUT_DIR}"
)

if [[ "${MAX_FILES}" != "0" ]]; then
  ARGS+=(--max-files "${MAX_FILES}")
fi

if [[ "${WRITE_CSV}" == "1" ]]; then
  ARGS+=(--write-csv)
fi

"${VENV_DIR}/bin/python" "${ARGS[@]}"

cat <<EOF

Done.
Metrics parquet: ${OUT_DIR}/turn_timeseries.parquet
Summary:         ${OUT_DIR}/summary.json
Interactive UI: ${OUT_DIR}/turn_timeseries_visualization.html
Static figures: ${OUT_DIR}/longest_trajectory_timeseries.png
                ${OUT_DIR}/aggregate_envelope.png

Useful overrides:
  SKIP_DOWNLOAD=1 ./scripts/run_coderforge_pipeline.sh
  MAX_FILES=3 ./scripts/run_coderforge_pipeline.sh
  DATA_DIR=/path/to/parquets SKIP_DOWNLOAD=1 ./scripts/run_coderforge_pipeline.sh
  OUT_DIR=/path/to/output ./scripts/run_coderforge_pipeline.sh
EOF
