#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORKDIR="${WORKDIR:-${SCRIPT_DIR}/workdir}"
CHECKPOINT="${CHECKPOINT:-${WORKDIR}/checkpoints/stage3/stage3_best.pth}"
DEVICE="${DEVICE:-auto}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
EUVP_ROOT="${EUVP_ROOT:-${PROJECT_ROOT}/data/raw_underwater/EUVP}"

python "${SCRIPT_DIR}/test_three_stage.py" \
  --workdir "${WORKDIR}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${WORKDIR}/test_results" \
  --image-size "${IMAGE_SIZE}" \
  --device "${DEVICE}"

python "${SCRIPT_DIR}/test_euvp.py" \
  --checkpoint "${CHECKPOINT}" \
  --euvp-root "${EUVP_ROOT}" \
  --output-dir "${WORKDIR}/euvp_test_results" \
  --image-size "${IMAGE_SIZE}" \
  --device "${DEVICE}"

echo "UIEB split test results: ${WORKDIR}/test_results"
echo "EUVP external test results: ${WORKDIR}/euvp_test_results"
