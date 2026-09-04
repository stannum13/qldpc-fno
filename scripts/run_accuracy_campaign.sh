#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

usage() {
  echo "usage: bash scripts/run_accuracy_campaign.sh [--resume] [--disconfirm]" >&2
  echo "" >&2
  echo "CAMPAIGN_OUTPUT selects a fresh local artifact directory." >&2
  echo "CAMPAIGN_REDUCED=1 enables test-only, non-scientific size overrides:" >&2
  echo "  CAMPAIGN_PILOT_SHOTS, CAMPAIGN_TRAIN_SHOTS," >&2
  echo "  CAMPAIGN_CALIBRATION_SHOTS, CAMPAIGN_TEST_SHOTS, CAMPAIGN_EPOCHS," >&2
  echo "  and CAMPAIGN_CALIBRATION_CANDIDATES." >&2
}

resume=0
disconfirm=0
while (( $# )); do
  case "$1" in
    --resume)
      if (( resume )); then
        usage
        exit 2
      fi
      resume=1
      ;;
    --disconfirm)
      if (( disconfirm )); then
        usage
        exit 2
      fi
      disconfirm=1
      ;;
    *)
      usage
      exit 2
      ;;
  esac
  shift
done

if [[ -n "${CAMPAIGN_CONFIG:-}" ]]; then
  echo "CAMPAIGN_CONFIG is not supported; canonical mode is pinned to the committed config" >&2
  exit 2
fi
reduced="${CAMPAIGN_REDUCED:-0}"
if [[ "$reduced" != "0" && "$reduced" != "1" ]]; then
  echo "CAMPAIGN_REDUCED must be 0 or 1" >&2
  exit 2
fi
if (( disconfirm )) && [[ "$reduced" == "1" ]]; then
  echo "--disconfirm cannot be combined with CAMPAIGN_REDUCED=1" >&2
  exit 2
fi
if (( disconfirm )); then
  canonical_config="$repo_root/configs/accuracy_disconfirm_p0375.json"
  output="${CAMPAIGN_OUTPUT:-artifacts/accuracy-disconfirm-p0375}"
else
  canonical_config="$repo_root/configs/accuracy_campaign.json"
  output="${CAMPAIGN_OUTPUT:-artifacts/accuracy-campaign}"
fi
if [[ ! -f "$canonical_config" || -L "$canonical_config" ]]; then
  echo "committed campaign configuration is unavailable: $canonical_config" >&2
  exit 2
fi
if [[ -L "$output" ]]; then
  echo "campaign output must not be a symlink: $output" >&2
  exit 2
fi
if (( resume )); then
  if [[ ! -d "$output" ]]; then
    echo "cannot resume missing campaign output: $output" >&2
    exit 2
  fi
elif [[ -e "$output" ]]; then
  echo "refusing to overwrite existing campaign output: $output" >&2
  exit 2
fi

pilot_shots="${CAMPAIGN_PILOT_SHOTS:-8}"
train_shots="${CAMPAIGN_TRAIN_SHOTS:-24}"
calibration_shots="${CAMPAIGN_CALIBRATION_SHOTS:-8}"
test_shots="${CAMPAIGN_TEST_SHOTS:-8}"
epochs="${CAMPAIGN_EPOCHS:-1}"
calibration_candidates="${CAMPAIGN_CALIBRATION_CANDIDATES:-1}"
stage_execution_guard="${CAMPAIGN_FAIL_ON_STAGE_EXECUTION:-0}"
stop_after_inputs="${CAMPAIGN_STOP_AFTER_INPUTS:-0}"

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer" >&2
    exit 2
  fi
}

if [[ "$reduced" == "1" ]]; then
  require_positive_integer "CAMPAIGN_PILOT_SHOTS" "$pilot_shots"
  require_positive_integer "CAMPAIGN_TRAIN_SHOTS" "$train_shots"
  require_positive_integer "CAMPAIGN_CALIBRATION_SHOTS" "$calibration_shots"
  require_positive_integer "CAMPAIGN_TEST_SHOTS" "$test_shots"
  require_positive_integer "CAMPAIGN_EPOCHS" "$epochs"
  require_positive_integer "CAMPAIGN_CALIBRATION_CANDIDATES" "$calibration_candidates"
elif [[ -n "${CAMPAIGN_PILOT_SHOTS:-}${CAMPAIGN_TRAIN_SHOTS:-}${CAMPAIGN_CALIBRATION_SHOTS:-}${CAMPAIGN_TEST_SHOTS:-}${CAMPAIGN_EPOCHS:-}${CAMPAIGN_CALIBRATION_CANDIDATES:-}" ]]; then
  echo "reduced campaign overrides require CAMPAIGN_REDUCED=1" >&2
  exit 2
fi
if [[ "$stage_execution_guard" != "0" && "$stage_execution_guard" != "1" ]]; then
  echo "CAMPAIGN_FAIL_ON_STAGE_EXECUTION must be 0 or 1" >&2
  exit 2
fi
if [[ "$stage_execution_guard" == "1" && "$reduced" != "1" && "$disconfirm" != "1" ]]; then
  echo "stage execution guard is available only for reduced or disconfirm integration tests" >&2
  exit 2
fi
if [[ "$stop_after_inputs" != "0" && "$stop_after_inputs" != "1" ]]; then
  echo "CAMPAIGN_STOP_AFTER_INPUTS must be 0 or 1" >&2
  exit 2
fi
if [[ "$stop_after_inputs" == "1" && "$reduced" != "1" ]]; then
  echo "input bootstrap stop is available only for reduced integration tests" >&2
  exit 2
fi

temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/qldpc-fno-accuracy-campaign.XXXXXX")"
cleanup() {
  if [[ -n "$temporary_root" && -d "$temporary_root" ]]; then
    rm -rf -- "$temporary_root"
  fi
}
trap cleanup EXIT

requested_mode="canonical"
if [[ "$reduced" == "1" ]]; then
  git_commit="$(git rev-parse HEAD)"
  requested_mode="reduced_non_scientific"
  echo "NON-SCIENTIFIC REDUCED CAMPAIGN: execution coverage only; do not report these measurements." >&2
else
  git_commit="$(uv run python -c '
import sys
from pathlib import Path

from qldpc_fno.campaign.local import verify_canonical_checkout

print(verify_canonical_checkout(Path(sys.argv[1]), Path(sys.argv[2])))
' "$repo_root" "$canonical_config")"
fi

bootstrap_root="$temporary_root/bootstrap"
mkdir -p -- "$bootstrap_root"
effective_config="$bootstrap_root/config.json"
code="$bootstrap_root/code"
if [[ "$reduced" == "1" ]]; then
  uv run python -c '
import json
import sys
from pathlib import Path

from qldpc_fno.artifacts import write_canonical_json

payload = json.loads(Path(sys.argv[1]).read_text())
pilot, train, calibration, test, epochs, candidates = map(int, sys.argv[3:])
payload.update(
    {
        "pilot_shots_per_point": pilot,
        "train_shots_cap": train,
        "calibration_shots_cap": calibration,
        "calibration_decode_shots_cap": calibration,
        "calibration_shortlist_per_method": min(candidates, 4),
        "test_batch_shots": test,
        "max_test_shots_per_point": test,
        "target_failures": test,
        "training_epochs": epochs,
    }
)
write_canonical_json(Path(sys.argv[2]), payload)
' "$canonical_config" "$effective_config" "$pilot_shots" "$train_shots" "$calibration_shots" "$test_shots" "$epochs" "$calibration_candidates"
else
  /bin/cp -- "$canonical_config" "$effective_config"
fi
uv run python experiments/01_build_lp_codes.py --out "$code"
uv run python experiments/02_validate_lp_codes.py --code "$code"

workdir="$temporary_root/work"
runner=(
  uv run python -c 'from qldpc_fno.campaign.runner import main; main()'
  --config "$effective_config"
  --canonical-config "$canonical_config"
  --code "$code"
  --git-commit "$git_commit"
  --workdir "$workdir"
  --store "$output"
  --campaign-mode "$requested_mode"
)
if [[ "$reduced" == "1" ]]; then
  runner+=(
    --calibration-grid-limit "$calibration_candidates"
  )
fi
if [[ "$stage_execution_guard" == "1" ]]; then
  runner+=(--fail-on-stage-execution)
fi
if [[ "$stop_after_inputs" == "1" ]]; then
  runner+=(--stop-after-inputs)
fi
runner_stdout="$temporary_root/runner.stdout"
if ! "${runner[@]}" >"$runner_stdout"; then
  exit 2
fi
runner_output="$(<"$runner_stdout")"
echo "$runner_output"
runner_status="$(printf '%s\n' "$runner_output" | uv run python -c '
import json
import sys

lines = [line for line in sys.stdin.read().splitlines() if line]
print(json.loads(lines[-1])["status"])
')"
if [[ "$runner_status" == "inputs_complete" ]]; then
  echo "campaign inputs published and verified; stopping before stage execution"
elif [[ "$runner_status" == "complete" ]]; then
  echo "accuracy campaign complete: $output"
elif [[ "$runner_status" == "partial_deadline" ]]; then
  echo "accuracy campaign paused at deadline and is resumable: $output"
else
  echo "campaign runner returned an invalid status: $runner_status" >&2
  exit 1
fi
