#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

usage() {
  echo "usage: bash scripts/run_accuracy_campaign.sh [--resume]" >&2
  echo "" >&2
  echo "CAMPAIGN_OUTPUT selects a fresh local artifact directory." >&2
  echo "CAMPAIGN_REDUCED=1 enables test-only, non-scientific size overrides:" >&2
  echo "  CAMPAIGN_PILOT_SHOTS, CAMPAIGN_TRAIN_SHOTS," >&2
  echo "  CAMPAIGN_CALIBRATION_SHOTS, CAMPAIGN_TEST_SHOTS, CAMPAIGN_EPOCHS," >&2
  echo "  CAMPAIGN_CALIBRATION_CANDIDATES, and CAMPAIGN_BOOTSTRAP_SAMPLES." >&2
}

resume=0
if (( $# > 1 )); then
  usage
  exit 2
fi
if (( $# == 1 )); then
  if [[ "$1" != "--resume" ]]; then
    usage
    exit 2
  fi
  resume=1
fi

if [[ -n "${CAMPAIGN_CONFIG:-}" ]]; then
  echo "CAMPAIGN_CONFIG is not supported; canonical mode is pinned to the committed config" >&2
  exit 2
fi
canonical_config="$repo_root/configs/accuracy_campaign.json"
output="${CAMPAIGN_OUTPUT:-artifacts/accuracy-campaign}"
reduced="${CAMPAIGN_REDUCED:-0}"
if [[ "$reduced" != "0" && "$reduced" != "1" ]]; then
  echo "CAMPAIGN_REDUCED must be 0 or 1" >&2
  exit 2
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
bootstrap_samples="${CAMPAIGN_BOOTSTRAP_SAMPLES:-100}"
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
  require_positive_integer "CAMPAIGN_BOOTSTRAP_SAMPLES" "$bootstrap_samples"
elif [[ -n "${CAMPAIGN_PILOT_SHOTS:-}${CAMPAIGN_TRAIN_SHOTS:-}${CAMPAIGN_CALIBRATION_SHOTS:-}${CAMPAIGN_TEST_SHOTS:-}${CAMPAIGN_EPOCHS:-}${CAMPAIGN_CALIBRATION_CANDIDATES:-}${CAMPAIGN_BOOTSTRAP_SAMPLES:-}" ]]; then
  echo "reduced campaign overrides require CAMPAIGN_REDUCED=1" >&2
  exit 2
fi
if [[ "$stage_execution_guard" != "0" && "$stage_execution_guard" != "1" ]]; then
  echo "CAMPAIGN_FAIL_ON_STAGE_EXECUTION must be 0 or 1" >&2
  exit 2
fi
if [[ "$stage_execution_guard" == "1" && "$reduced" != "1" ]]; then
  echo "stage execution guard is available only for reduced integration tests" >&2
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
requested_bootstrap=10000
if [[ "$reduced" == "1" ]]; then
  git_commit="$(git rev-parse HEAD)"
  requested_mode="reduced_non_scientific"
  requested_bootstrap="$bootstrap_samples"
  echo "NON-SCIENTIFIC REDUCED CAMPAIGN: execution coverage only; do not report these measurements." >&2
else
  git_commit="$(uv run python -c '
import sys
from pathlib import Path

from qldpc_fno.campaign.local import verify_canonical_checkout

print(verify_canonical_checkout(Path(sys.argv[1]), Path(sys.argv[2])))
' "$repo_root" "$canonical_config")"
fi

input_state="$(uv run python -c '
import sys
from pathlib import Path

from qldpc_fno.campaign.storage import LocalArtifactStore

try:
    store = LocalArtifactStore(Path(sys.argv[1]))
    if store.verify_completion("inputs"):
        print("verified")
    else:
        established = store.exists("inputs/_COMPLETE.json")
        for index in range(1_000_000):
            key = f"inputs/.recovery/{index:08d}/_COMPLETE.json"
            if not store.exists(key):
                break
            established = True
        execution_state = any(path.name != "inputs" for path in store.root.iterdir())
        print("corrupt" if established or execution_state else "unpublished")
except (OSError, TypeError, ValueError):
    print("corrupt")
' "$output")"

if [[ "$input_state" == "corrupt" ]]; then
  echo "established campaign input publication is corrupt; refusing unsafe recovery" >&2
  exit 2
fi
if [[ "$input_state" == "unpublished" ]]; then
  input_staging="$temporary_root/inputs"
  mkdir -p -- "$input_staging"
  effective_staging="$input_staging/config.json"
  code_staging="$input_staging/code"
  mode_staging="$input_staging/run-mode.json"
  if [[ "$reduced" == "1" ]]; then
    uv run python -c '
import hashlib
import json
import sys
from pathlib import Path

from qldpc_fno.artifacts import write_canonical_json

source = Path(sys.argv[1])
effective = Path(sys.argv[2])
mode_path = Path(sys.argv[3])
commit = sys.argv[4]
pilot, train, calibration, test, epochs, candidates, bootstrap = map(int, sys.argv[5:])
payload = json.loads(source.read_text())
overrides = {
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
payload.update(overrides)
write_canonical_json(effective, payload)
write_canonical_json(
    mode_path,
    {
        "canonical_config": str(source.resolve()),
        "canonical_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "effective_config_sha256": hashlib.sha256(effective.read_bytes()).hexdigest(),
        "execution_controls": {
            "bootstrap_samples": bootstrap,
            "calibration_candidates": candidates,
        },
        "git_commit": commit,
        "mode": "reduced_non_scientific",
        "overrides": overrides,
        "scientific_claims_permitted": False,
    },
)
' "$canonical_config" "$effective_staging" "$mode_staging" "$git_commit" "$pilot_shots" "$train_shots" "$calibration_shots" "$test_shots" "$epochs" "$calibration_candidates" "$bootstrap_samples"
  else
    /bin/cp -- "$canonical_config" "$effective_staging"
    uv run python -c '
import hashlib
import sys
from pathlib import Path

from qldpc_fno.artifacts import write_canonical_json

source = Path(sys.argv[1])
effective = Path(sys.argv[2])
write_canonical_json(
    Path(sys.argv[3]),
    {
        "canonical_config": str(source.resolve()),
        "canonical_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "effective_config_sha256": hashlib.sha256(effective.read_bytes()).hexdigest(),
        "execution_controls": {"bootstrap_samples": 10000},
        "git_commit": sys.argv[4],
        "mode": "canonical",
        "overrides": {},
        "scientific_claims_permitted": True,
    },
)
' "$canonical_config" "$effective_staging" "$mode_staging" "$git_commit"
  fi
  uv run python experiments/01_build_lp_codes.py --out "$code_staging"
  uv run python experiments/02_validate_lp_codes.py --code "$code_staging"
  uv run python -c '
import sys
from pathlib import Path

from qldpc_fno.campaign.storage import LocalArtifactStore

LocalArtifactStore(Path(sys.argv[1])).publish_directory(Path(sys.argv[2]), "inputs")
' "$output" "$input_staging"
fi

verified_inputs="$temporary_root/verified-inputs"
uv run python -c '
import sys
from pathlib import Path

from qldpc_fno.campaign.storage import LocalArtifactStore, materialize_completion

materialize_completion(
    LocalArtifactStore(Path(sys.argv[1])),
    "inputs",
    Path(sys.argv[2]),
)
' "$output" "$verified_inputs"

effective_config="$verified_inputs/config.json"
code="$verified_inputs/code"
uv run python -c '
import hashlib
import json
import sys
from pathlib import Path

from qldpc_fno.campaign.config import CampaignConfig
root = Path(sys.argv[1])
canonical = Path(sys.argv[2])
requested_mode = sys.argv[3]
commit = sys.argv[4]
pilot, train, calibration, test, epochs, candidates, bootstrap = map(int, sys.argv[5:])
effective = root / "config.json"
mode_path = root / "run-mode.json"
mode = json.loads(mode_path.read_text())
expected_keys = {
    "canonical_config",
    "canonical_config_sha256",
    "effective_config_sha256",
    "execution_controls",
    "git_commit",
    "mode",
    "overrides",
    "scientific_claims_permitted",
}
if set(mode) != expected_keys:
    raise ValueError("campaign mode manifest schema is invalid")
if mode["mode"] != requested_mode or mode["git_commit"] != commit:
    raise ValueError("resume mode or Git commit does not match the original campaign")
canonical_digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
effective_digest = hashlib.sha256(effective.read_bytes()).hexdigest()
if mode["canonical_config"] != str(canonical.resolve()):
    raise ValueError("campaign is not bound to the committed canonical configuration")
if mode["canonical_config_sha256"] != canonical_digest:
    raise ValueError("committed canonical configuration changed since campaign creation")
if mode["effective_config_sha256"] != effective_digest:
    raise ValueError("effective campaign configuration failed SHA-256 verification")
if requested_mode == "canonical":
    expected_overrides = {}
    expected_controls = {"bootstrap_samples": 10000}
    if effective_digest != canonical_digest or mode["scientific_claims_permitted"] is not True:
        raise ValueError("canonical campaign inputs differ from committed policy")
else:
    expected_overrides = {
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
    expected_controls = {
        "bootstrap_samples": bootstrap,
        "calibration_candidates": candidates,
    }
    if mode["scientific_claims_permitted"] is not False:
        raise ValueError("reduced campaign cannot permit scientific claims")
if mode["overrides"] != expected_overrides or mode["execution_controls"] != expected_controls:
    raise ValueError("resume controls do not match the original campaign")
CampaignConfig.from_json(effective)
' "$verified_inputs" "$canonical_config" "$requested_mode" "$git_commit" "$pilot_shots" "$train_shots" "$calibration_shots" "$test_shots" "$epochs" "$calibration_candidates" "$requested_bootstrap"

if [[ "$stop_after_inputs" == "1" ]]; then
  echo "campaign inputs published and verified; stopping before stage execution"
  exit 0
fi

workdir="$temporary_root/work"
runner=(
  uv run python -c 'from qldpc_fno.campaign.runner import main; main()'
  --config "$effective_config"
  --code "$code"
  --workdir "$workdir"
  --store "$output"
  --campaign-mode "$requested_mode"
)
if [[ "$reduced" == "1" ]]; then
  runner+=(
    --calibration-grid-limit "$calibration_candidates"
    --bootstrap-samples "$bootstrap_samples"
  )
fi
if [[ "$stage_execution_guard" == "1" ]]; then
  runner+=(--fail-on-stage-execution)
fi
runner_output="$("${runner[@]}")"
echo "$runner_output"
runner_status="$(printf '%s\n' "$runner_output" | uv run python -c '
import json
import sys

lines = [line for line in sys.stdin.read().splitlines() if line]
print(json.loads(lines[-1])["status"])
')"
if [[ "$runner_status" == "complete" ]]; then
  echo "accuracy campaign complete: $output"
elif [[ "$runner_status" == "partial_deadline" ]]; then
  echo "accuracy campaign paused at deadline and is resumable: $output"
else
  echo "campaign runner returned an invalid status: $runner_status" >&2
  exit 1
fi
