#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: bash scripts/launch_cloud_campaign.sh [--execute] [--reduced] [--multi-execution] [--resume]" >&2
  echo "Dry-run is the default. Set CLOUD_REGION to override us-central1." >&2
}

die() {
  echo "$1" >&2
  exit 2
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

require_absent() {
  local label="$1"
  local resource="$2"
  shift 2
  local output
  if output="$("$@" 2>&1)"; then
    die "campaign $label already exists: $resource"
  fi
  case "$output" in
    *NOT_FOUND* | *"not found"* | *"does not exist"* | *"Cannot find"*) return ;;
  esac
  [[ -z "$output" ]] || echo "$output" >&2
  die "cannot verify campaign $label absence"
}

require_present() {
  local label="$1"
  local resource="$2"
  shift 2
  local output
  if ! output="$("$@" 2>&1)"; then
    [[ -z "$output" ]] || echo "$output" >&2
    die "cannot verify existing campaign $label: $resource"
  fi
}

execute=0
reduced=0
multi_execution=0
resume=0
for argument in "$@"; do
  case "$argument" in
    --execute) execute=1 ;;
    --multi-execution) multi_execution=1 ;;
    --reduced) reduced=1 ;;
    --resume) resume=1 ;;
    *)
      usage
      exit 2
      ;;
  esac
done

(( ! resume || execute )) || die "--resume requires the explicit --execute flag"
(( ! resume || ! reduced )) || die "--resume targets canonical campaigns; omit --reduced"
(( ! resume || ! multi_execution )) || die "--resume and --multi-execution are mutually exclusive"
if (( execute && ! reduced && ! resume && ! multi_execution )); then
  die "canonical execution requires explicit --multi-execution acknowledgement before benchmark-gate evaluation"
fi

command -v git >/dev/null 2>&1 || die "git is required"
command -v gcloud >/dev/null 2>&1 || die "gcloud is required"
command -v uv >/dev/null 2>&1 || die "uv is required"

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" \
  || die "launcher must run from a Git checkout"
[[ -n "$repo_root" && -d "$repo_root" ]] || die "Git repository root is invalid"
cd "$repo_root"

git_commit="$(git rev-parse HEAD 2>/dev/null)" || die "cannot resolve the Git commit"
[[ "$git_commit" =~ ^[0-9a-f]{40}$ ]] || die "Git commit must be a full lowercase SHA-1"
git_status="$(git status --porcelain=v1 --untracked-files=all)" \
  || die "cannot inspect the Git checkout"
[[ -z "$git_status" ]] || die "cloud campaigns require a clean Git checkout"

region="${CLOUD_REGION:-us-central1}"
[[ "$region" =~ ^[a-z][a-z0-9-]{0,30}[a-z0-9]$ ]] \
  || die "CLOUD_REGION is not a valid Google Cloud region"

campaign_id="${CAMPAIGN_ID:-}"
if (( resume )) && [[ -z "$campaign_id" ]]; then
  die "CAMPAIGN_ID is required to resume the exact existing campaign"
fi
if [[ -z "$campaign_id" ]]; then
  random_hex="$(LC_ALL=C od -An -N3 -tx1 /dev/urandom | tr -d ' \n')"
  campaign_id="accuracy-$(date -u +%Y%m%d-%H%M%S)-${random_hex}"
fi
[[ "$campaign_id" =~ ^[a-z][a-z0-9-]{0,30}[a-z0-9]$ ]] \
  || die "CAMPAIGN_ID must be a lowercase Google Cloud name of at most 32 characters"

project="$(gcloud config get-value project 2>/dev/null)" \
  || die "cannot resolve the active gcloud project"
[[ -n "$project" && "$project" != "(unset)" ]] || die "active gcloud project is empty"
[[ "$project" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] \
  || die "active gcloud project is not a valid project ID"
if [[ -n "${CLOUD_PROJECT:-}" && "$project" != "$CLOUD_PROJECT" ]]; then
  die "active gcloud project does not match CLOUD_PROJECT"
fi

normalize_service_account() {
  local value="$1"
  value="${value##*/}"
  printf '%s' "$value"
}

build_service_account_output=""
build_service_account_fallback=0
if build_service_account_output="$(
  gcloud builds get-default-service-account "--project=$project" \
    "--format=value(serviceAccountEmail)" 2>&1
)"; then
  build_service_account="$(normalize_service_account "$build_service_account_output")"
else
  case "$build_service_account_output" in
    *"Invalid choice"* | *"invalid choice"* | *"Unknown command"* | \
      *"unknown command"* | *"unrecognized command"*)
      build_service_account_fallback=1
      ;;
    *)
      [[ -z "$build_service_account_output" ]] || echo "$build_service_account_output" >&2
      die "cannot resolve the default Cloud Build service account"
      ;;
  esac
fi
project_number="$(
  gcloud projects describe "$project" "--format=value(projectNumber)" 2>/dev/null
)" || die "cannot resolve the project number for the Cloud Build service account"
[[ "$project_number" =~ ^[0-9]+$ ]] \
  || die "Cloud Build service account project number is invalid"
if (( build_service_account_fallback )); then
  build_service_account="${project_number}-compute@developer.gserviceaccount.com"
fi
if [[ ! "$build_service_account" =~ ^[0-9]+@cloudbuild\.gserviceaccount\.com$ \
  && ! "$build_service_account" =~ ^[0-9]+-compute@developer\.gserviceaccount\.com$ ]]; then
  die "resolved Cloud Build service account is invalid"
fi
case "$build_service_account" in
  *-compute@developer.gserviceaccount.com)
    build_service_account_kind="compute"
    build_service_account_project_number="${build_service_account%-compute@developer.gserviceaccount.com}"
    ;;
  *@cloudbuild.gserviceaccount.com)
    build_service_account_kind="legacy"
    build_service_account_project_number="${build_service_account%@cloudbuild.gserviceaccount.com}"
    ;;
esac
[[ "$build_service_account_project_number" == "$project_number" ]] \
  || die "Cloud Build service account does not belong to the active project"
if [[ "$build_service_account_kind" == "compute" ]]; then
  described_build_service_account="$(
    gcloud iam service-accounts describe "$build_service_account" \
      "--project=$project" "--format=value(email)" 2>/dev/null
  )" || die "cannot verify the resolved Cloud Build service account"
  described_build_service_account="$(
    normalize_service_account "$described_build_service_account"
  )"
  [[ "$described_build_service_account" == "$build_service_account" ]] \
    || die "Cloud Build service account verification differs from the resolved principal"
fi

repository="qldpc-fno-${campaign_id}"
job="qldpc-fno-${campaign_id}"
bucket="${project}-${campaign_id}"
prefix="campaigns/${campaign_id}/${git_commit}"
store="gs://${bucket}/${prefix}"
image="${region}-docker.pkg.dev/${project}/${repository}/accuracy-campaign:${git_commit}"
resource_digest="$(printf '%s' "${campaign_id}:${git_commit}" | git hash-object --stdin)" \
  || die "cannot derive unique campaign resource names"
[[ "$resource_digest" =~ ^[0-9a-f]{40}$ ]] || die "campaign resource digest is invalid"
service_account_id="qfno-${resource_digest:0:24}"
service_account="${service_account_id}@${project}.iam.gserviceaccount.com"
config_path="/app/configs/accuracy_campaign.json"
campaign_mode="canonical"
execution_completion=--async
if (( reduced )); then
  config_path="/app/configs/accuracy_campaign_cloud_reduced.json"
  campaign_mode="reduced_non_scientific"
  execution_completion=--wait
fi

if (( resume )); then
  campaign_mode="canonical"
  execution_completion=--async
fi

repo_describe=(
  gcloud artifacts repositories describe "$repository"
  "--location=$region" "--project=$project"
)
bucket_describe=(
  gcloud storage buckets describe "gs://$bucket" "--project=$project"
)
job_describe=(
  gcloud run jobs describe "$job" "--region=$region" "--project=$project"
)
service_account_describe=(
  gcloud iam service-accounts describe "$service_account" "--project=$project"
)

context_archive=""
temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/qldpc-fno-cloud-context.XXXXXX")"
cleanup_temporary() {
  if [[ -n "$temporary_root" && -d "$temporary_root" ]]; then
    rm -rf -- "$temporary_root"
  fi
}
trap cleanup_temporary EXIT
if (( ! resume )); then
  context_archive="$temporary_root/qldpc-fno-${git_commit}.tar.gz"
  build_context_paths=(
    .dockerignore
    Dockerfile
    README.md
    pyproject.toml
    uv.lock
    configs/accuracy_campaign.json
    configs/accuracy_campaign_cloud_reduced.json
    experiments/01_build_lp_codes.py
    experiments/02_validate_lp_codes.py
    experiments/13_pilot_noise_grid.py
    experiments/14_generate_campaign_shards.py
    experiments/15_train_conditional_fno.py
    experiments/16_calibrate_hybrid_priors.py
    experiments/17_evaluate_hybrid_decoders.py
    src/qldpc_fno
  )
  git archive --format=tar.gz --output="$context_archive" "$git_commit" \
    -- "${build_context_paths[@]}" \
    || die "cannot create exact tracked cloud build context"
fi
repo_create=(
  gcloud artifacts repositories create "$repository"
  "--repository-format=docker" "--location=$region" "--project=$project"
  "--description=qLDPC FNO campaign ${campaign_id}" --quiet
)
grant_build_push_access=(
  gcloud artifacts repositories add-iam-policy-binding "$repository"
  "--location=$region" "--member=serviceAccount:$build_service_account"
  --role=roles/artifactregistry.writer "--project=$project"
)
bucket_create=(
  gcloud storage buckets create "gs://$bucket"
  "--location=$region" "--project=$project" --uniform-bucket-level-access
)
service_account_create=(
  gcloud iam service-accounts create "$service_account_id"
  "--display-name=qLDPC FNO campaign ${campaign_id}" "--project=$project"
)
grant_bucket_read_access=(
  gcloud storage buckets add-iam-policy-binding "gs://$bucket"
  "--member=serviceAccount:$service_account" --role=roles/storage.objectViewer
  "--project=$project"
)
grant_bucket_create_access=(
  gcloud storage buckets add-iam-policy-binding "gs://$bucket"
  "--member=serviceAccount:$service_account" --role=roles/storage.objectCreator
  "--project=$project"
)
build_image=(
  gcloud builds submit --tag "$image" "--project=$project" "$context_archive"
)
describe_image=(
  gcloud artifacts docker images describe "$image"
  "--project=$project" "--format=value(image_summary.digest)"
)
image_digest="sha256:<resolved-after-build>"
pinned_image=""
environment_variables=""
create_job=()
configure_job_contract() {
  pinned_image="${image%:*}@${image_digest}"
  calibration_grid_limit=""
  bootstrap_samples="10000"
  if (( reduced )); then
    calibration_grid_limit="1"
    bootstrap_samples="100"
  fi
  environment_variables="CAMPAIGN_BOOTSTRAP_SAMPLES=${bootstrap_samples},CAMPAIGN_BUCKET=${bucket},CAMPAIGN_CALIBRATION_GRID_LIMIT=${calibration_grid_limit},CAMPAIGN_CANONICAL_CONFIG=/app/configs/accuracy_campaign.json,CAMPAIGN_CLOUD_JOB=${job},CAMPAIGN_CLOUD_PROJECT=${project},CAMPAIGN_CLOUD_REGION=${region},CAMPAIGN_CODE=/app/campaign-code,CAMPAIGN_CONFIG=${config_path},CAMPAIGN_FINALIZATION_RESERVE_SECONDS=2700,CAMPAIGN_GIT_COMMIT=${git_commit},CAMPAIGN_IMAGE=${pinned_image},CAMPAIGN_IMAGE_DIGEST=${image_digest},CAMPAIGN_MODE=${campaign_mode},CAMPAIGN_OUTER_TIMEOUT_SECONDS=28800,CAMPAIGN_PREFIX=${prefix},CAMPAIGN_SERVICE_ACCOUNT=${service_account},CAMPAIGN_STORE=${store},CAMPAIGN_WORKDIR=/tmp/qldpc-fno-work,CAMPAIGN_WORK_CUTOFF_SECONDS=26100"
  create_job=(
    gcloud run jobs create "$job" "--image=$pinned_image"
    --cpu=8 --memory=32Gi --task-timeout=8h --max-retries=0 --tasks=1 --parallelism=1
    "--service-account=$service_account"
    "--set-env-vars=$environment_variables"
    "--labels=qldpc-fno-identity=${resource_digest},qldpc-fno-mode=${campaign_mode}"
  )
  if (( reduced )); then
    create_job+=(
      "--args=--campaign-mode=reduced_non_scientific,--calibration-grid-limit=1,--bootstrap-samples=100"
    )
  fi
  create_job+=("--region=$region" "--project=$project" --quiet)
}
configure_job_contract
execute_job=(
  gcloud run jobs execute "$job" "--region=$region" "--project=$project"
  "$execution_completion"
)
cleanup_job=(
  gcloud run jobs delete "$job" "--region=$region" "--project=$project"
)
cleanup_bucket_objects=(
  gcloud storage rm --recursive "gs://$bucket/**" "--project=$project"
)
cleanup_bucket=(
  gcloud storage buckets delete "gs://$bucket" "--project=$project"
)
cleanup_repository=(
  gcloud artifacts repositories delete "$repository"
  "--location=$region" "--project=$project"
)
cleanup_service_account=(
  gcloud iam service-accounts delete "$service_account" "--project=$project"
)

if (( resume )); then
  mode="resume"
elif (( execute )); then
  mode="execute"
else
  mode="dry-run"
fi
echo "mode=$mode"
echo "campaign_mode=$campaign_mode"
echo "project=$project"
echo "region=$region"
echo "repository=$repository"
echo "image=$image"
echo "pinned_image=$pinned_image"
echo "bucket=$bucket"
echo "prefix=$prefix"
echo "store=$store"
echo "job=$job"
echo "service_account=$service_account"
echo "build_service_account=$build_service_account"
echo "cpu=8"
echo "memory=32Gi"
echo "timeout=8h"
echo "work_cutoff=7h15m"
echo "finalization_reserve=45m"
echo "multi_execution_required=$(( ! reduced ))"
if (( reduced )); then
  echo "canonical_execution_gate=not_applicable_reduced_non_scientific"
else
  echo "canonical_execution_gate=blocked_representative_decoder_benchmark"
fi
echo "retries=0"
echo "tasks=1"
echo "git_commit=$git_commit"
echo "mutation commands:"
if (( ! resume )); then
  print_command "${repo_create[@]}"
  print_command "${grant_build_push_access[@]}"
  print_command "${bucket_create[@]}"
  print_command "${service_account_create[@]}"
  print_command "${grant_bucket_read_access[@]}"
  print_command "${grant_bucket_create_access[@]}"
  print_command "${build_image[@]}"
  print_command "${describe_image[@]}"
  print_command "${create_job[@]}"
fi
print_command "${execute_job[@]}"
echo "verified resume command:"
print_command env "CLOUD_PROJECT=${project}" "CLOUD_REGION=${region}" \
  "CAMPAIGN_ID=${campaign_id}" bash scripts/launch_cloud_campaign.sh --execute --resume
echo "cleanup commands (not executed):"
print_command "${cleanup_job[@]}"
print_command "${cleanup_bucket_objects[@]}"
print_command "${cleanup_bucket[@]}"
print_command "${cleanup_repository[@]}"
print_command "${cleanup_service_account[@]}"

if (( ! execute )); then
  if (( reduced )); then
    echo "dry-run only; pass --execute to create and run the reduced non-scientific job"
  else
    echo "dry-run only; canonical cloud execution is blocked by the representative decoder benchmark gate"
  fi
  exit 0
fi

if (( resume )); then
  require_present repository "$repository" "${repo_describe[@]}"
  require_present bucket "$bucket" "${bucket_describe[@]}"
  require_present job "$job" "${job_describe[@]}"
  require_present service-account "$service_account" "${service_account_describe[@]}"
  downloaded_store="$temporary_root/downloaded-store"
  mkdir -p -- "$downloaded_store"
  gcloud storage cp --recursive "${store}/inputs" "$downloaded_store" \
    || die "cannot download existing campaign input publication"
  stored_identity="$(uv run python -c '
import json
import sys
from pathlib import Path

from qldpc_fno.campaign.inputs import verify_downloaded_cloud_inputs

expected = {
    "bucket": sys.argv[4],
    "finalization_reserve_seconds": 2700,
    "job": sys.argv[5],
    "kind": "cloud",
    "outer_timeout_seconds": 28800,
    "prefix": sys.argv[6],
    "project": sys.argv[7],
    "region": sys.argv[8],
    "service_account": sys.argv[9],
    "store": sys.argv[10],
    "work_cutoff_seconds": 26100,
}
identity = verify_downloaded_cloud_inputs(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    git_commit=sys.argv[3],
    campaign_mode="canonical",
    calibration_grid_limit=None,
    bootstrap_samples=10000,
    expected_execution_identity=expected,
)
print(json.dumps(identity, sort_keys=True))
' "$downloaded_store" "$repo_root/configs/accuracy_campaign.json" "$git_commit" "$bucket" "$job" "$prefix" "$project" "$region" "$service_account" "$store")" \
    || die "existing campaign input publication failed provenance verification"
  image_digest="$(printf '%s' "$stored_identity" | uv run python -c 'import json,sys; print(json.load(sys.stdin)["image_digest"])')"
  configure_job_contract
  job_contract="$(gcloud run jobs describe "$job" "--region=$region" "--project=$project" --format=json)" \
    || die "cannot read existing campaign job contract"
  printf '%s' "$job_contract" | uv run python -c '
import json
import sys

from qldpc_fno.campaign.cloud_contract import verify_cloud_job_contract

environment = dict(item.split("=", 1) for item in sys.argv[6].split(","))
expected = {
    "args": [],
    "command": [],
    "cpu": "8",
    "env": environment,
    "execution_environment": "gen2",
    "identity_label": sys.argv[3],
    "image": sys.argv[1],
    "max_retries": 0,
    "memory": "32Gi",
    "mode_label": "canonical",
    "parallelism": 1,
    "service_account": sys.argv[2],
    "task_count": 1,
    "timeout_seconds": 28800,
}
verify_cloud_job_contract(json.load(sys.stdin), expected)
' "$pinned_image" "$service_account" "$resource_digest" "$job" "$store" "$environment_variables" \
    || die "existing Cloud Run job contract failed exact provenance verification"
  die "canonical Cloud execution is blocked by the representative decoder benchmark gate"
fi

if (( ! reduced )); then
  die "canonical Cloud execution is blocked by the representative decoder benchmark gate"
fi

require_absent repository "$repository" "${repo_describe[@]}"
require_absent bucket "$bucket" "${bucket_describe[@]}"
require_absent job "$job" "${job_describe[@]}"
require_absent service-account "$service_account" "${service_account_describe[@]}"

"${repo_create[@]}"
"${grant_build_push_access[@]}" \
  || die "cannot grant repository-scoped Artifact Registry writer to Cloud Build"
"${bucket_create[@]}"
"${service_account_create[@]}"
"${grant_bucket_read_access[@]}"
"${grant_bucket_create_access[@]}"
"${build_image[@]}"
image_digest="$("${describe_image[@]}")" || die "cannot resolve built image digest"
[[ "$image_digest" =~ ^sha256:[0-9a-f]{64}$ ]] || die "built image digest is invalid"
configure_job_contract
"${create_job[@]}"
"${execute_job[@]}"
