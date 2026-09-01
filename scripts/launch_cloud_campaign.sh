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
  die "canonical execution cannot finish safely in one allocation; pass --multi-execution and plan to resume"
fi

command -v git >/dev/null 2>&1 || die "git is required"
command -v gcloud >/dev/null 2>&1 || die "gcloud is required"

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
temporary_root=""
if (( ! resume )); then
  temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/qldpc-fno-cloud-context.XXXXXX")"
  cleanup_temporary() {
    if [[ -n "$temporary_root" && -d "$temporary_root" ]]; then
      rm -rf -- "$temporary_root"
    fi
  }
  trap cleanup_temporary EXIT
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
environment_variables="CAMPAIGN_BUCKET=${bucket},CAMPAIGN_PREFIX=${prefix},CAMPAIGN_STORE=${store},CAMPAIGN_CONFIG=${config_path},CAMPAIGN_CODE=/app/campaign-code,CAMPAIGN_WORKDIR=/tmp/qldpc-fno-work,CAMPAIGN_GIT_COMMIT=${git_commit},CAMPAIGN_CLOUD_JOB=${job},CAMPAIGN_CLOUD_REGION=${region},CAMPAIGN_CLOUD_PROJECT=${project}"
create_job=(
  gcloud run jobs create "$job" "--image=$image"
  --cpu=8 --memory=32Gi --task-timeout=8h --max-retries=0 --tasks=1
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
echo "bucket=$bucket"
echo "prefix=$prefix"
echo "store=$store"
echo "job=$job"
echo "service_account=$service_account"
echo "cpu=8"
echo "memory=32Gi"
echo "timeout=8h"
echo "work_cutoff=7h15m"
echo "finalization_reserve=45m"
echo "multi_execution_required=$(( ! reduced ))"
echo "retries=0"
echo "tasks=1"
echo "git_commit=$git_commit"
echo "mutation commands:"
if (( ! resume )); then
  print_command "${repo_create[@]}"
  print_command "${bucket_create[@]}"
  print_command "${service_account_create[@]}"
  print_command "${grant_bucket_read_access[@]}"
  print_command "${grant_bucket_create_access[@]}"
  print_command "${build_image[@]}"
  print_command "${create_job[@]}"
fi
print_command "${execute_job[@]}"
echo "verified resume command:"
print_command "${execute_job[@]}"
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
    echo "dry-run only; canonical creation requires --execute --multi-execution"
  fi
  exit 0
fi

if (( resume )); then
  require_present repository "$repository" "${repo_describe[@]}"
  require_present bucket "$bucket" "${bucket_describe[@]}"
  require_present job "$job" "${job_describe[@]}"
  require_present service-account "$service_account" "${service_account_describe[@]}"
  job_identity="$(
    gcloud run jobs describe "$job" "--region=$region" "--project=$project" \
      "--format=value(metadata.labels.qldpc-fno-identity)"
  )" || die "cannot read existing campaign job identity"
  [[ "$job_identity" == "$resource_digest" ]] \
    || die "existing campaign job identity does not match CAMPAIGN_ID and Git commit"
  job_image="$(
    gcloud run jobs describe "$job" "--region=$region" "--project=$project" \
      "--format=value(spec.template.spec.template.spec.containers[0].image)"
  )" || die "cannot read existing campaign job image"
  [[ "$job_image" == "$image" ]] \
    || die "existing campaign job image does not match the exact Git commit"
  "${execute_job[@]}"
  exit 0
fi

require_absent repository "$repository" "${repo_describe[@]}"
require_absent bucket "$bucket" "${bucket_describe[@]}"
require_absent job "$job" "${job_describe[@]}"
require_absent service-account "$service_account" "${service_account_describe[@]}"

"${repo_create[@]}"
"${bucket_create[@]}"
"${service_account_create[@]}"
"${grant_bucket_read_access[@]}"
"${grant_bucket_create_access[@]}"
"${build_image[@]}"
"${create_job[@]}"
"${execute_job[@]}"
