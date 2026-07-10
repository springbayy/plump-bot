#!/bin/zsh
set -u

repo_dir="/Users/erehmax/Documents/side_projects/plump-bot"
run_dir="${PLUMP_V4_RUN_DIR:-$repo_dir/checkpoints/v4_50m_sinkhorn_20260608_204818_seed1}"
log_file="$run_dir/metrics-updater.log"
updates=1440
interval_seconds=60

cd "$repo_dir" || exit 1
mkdir -p "$run_dir"

for (( update = 1; update <= updates; update++ )); do
  timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  print -r -- "[$timestamp] refresh $update/$updates" >> "$log_file"
  if "$repo_dir/.venv/bin/python" examples/plot_training_metrics.py \
    "$run_dir" \
    --smooth 5 \
    >> "$log_file" 2>&1; then
    exit_status=0
  else
    exit_status=$?
  fi
  if (( exit_status != 0 )); then
    print -r -- "[$timestamp] refresh failed with status $exit_status" >> "$log_file"
  fi
  if (( update < updates )); then
    sleep "$interval_seconds"
  fi
done

print -r -- "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] updater complete" >> "$log_file"
