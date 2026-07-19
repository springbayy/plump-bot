#!/bin/zsh
set -u

repo_dir="/Users/erehmax/Documents/Projects/Personal/plump-bot"
run_dir="${PLUMP_V8_RUN_DIR:-$repo_dir/checkpoints/v8_10m_fastppo_seed1}"
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
