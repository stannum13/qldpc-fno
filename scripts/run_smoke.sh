#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="configs/smoke_lp_3_7_16.json"
config_value() {
  uv run python -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
    "$config" "$1"
}

output="${SMOKE_OUTPUT:-artifacts/smoke}"
shots="${SMOKE_SHOTS:-$(config_value shots)}"
steps="${SMOKE_STEPS:-$(config_value training_steps)}"
ell="$(config_value ell)"
error_rate="$(config_value error_rate)"
sample_seed="$(config_value sample_seed)"
training_seed="$(config_value training_seed)"
canonical_overfit_shots="$(config_value overfit_shots)"
require_gates=1
if [[ -n "${SMOKE_STEPS:-}" ]]; then
  require_gates=0
  echo "SMOKE_STEPS override: running non-gating execution check" >&2
fi

if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing smoke output: $output" >&2
  exit 2
fi
if (( shots < 4 )); then
  echo "SMOKE_SHOTS must be at least 4" >&2
  exit 2
fi
if (( steps < 1 )); then
  echo "SMOKE_STEPS must be positive" >&2
  exit 2
fi

train_shots=$(( shots * 3 / 4 ))
overfit_shots="$canonical_overfit_shots"
if (( overfit_shots > train_shots )); then
  overfit_shots="$train_shots"
fi

mkdir -p "$output"
uv run python experiments/00_lock_sources.py --out "$output/source-lock.json"
uv run python experiments/01_build_lp_codes.py --out "$output/code"
uv run python experiments/02_validate_lp_codes.py --code "$output/code"
uv run python experiments/05_build_code_capacity_dem.py \
  --code "$output/code" \
  --error-rate "$error_rate" \
  --out "$output/dem"
uv run python experiments/06_sample_code_capacity.py \
  --dem "$output/dem/model.dem" \
  --shots "$shots" \
  --seed "$sample_seed" \
  --out "$output/samples"
uv run python experiments/07_decode_bplsd.py \
  --code "$output/code" \
  --dem "$output/dem" \
  --samples "$output/samples" \
  --error-rate "$error_rate" \
  --out "$output/bplsd"
uv run python experiments/08_tensorize_ring_fields.py \
  --samples "$output/samples" \
  --corrections "$output/bplsd/corrections.b8" \
  --ell "$ell" \
  --out "$output/tensors"
training_command=(
  uv run python experiments/10_overfit_tiny_models.py
  --tensors "$output/tensors"
  --code "$output/code"
  --shots "$overfit_shots"
  --steps "$steps"
  --seed "$training_seed"
)
if (( require_gates )); then
  training_command+=(--require-gates)
fi
training_command+=(--out "$output/fno")
"${training_command[@]}"
uv run python experiments/12_evaluate_in_size.py \
  --code "$output/code" \
  --dem "$output/dem" \
  --samples "$output/samples" \
  --tensors "$output/tensors" \
  --model "$output/fno/model.pt" \
  --out "$output/evaluation"

echo "smoke experiment complete: $output"
