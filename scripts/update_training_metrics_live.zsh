#!/bin/zsh
set -u

repo_dir="/Users/erehmax/Documents/Projects/Personal/plump-bot"
run_dir="${1:-${PLUMP_RUN_DIR:-$repo_dir/checkpoints/v9_8m_wideppo_seed1}}"
interval_seconds="${PLOT_INTERVAL_SECONDS:-20}"
log_file="$run_dir/metrics-updater.log"

cd "$repo_dir" || exit 1
mkdir -p "$run_dir"

while true; do
  timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  if "$repo_dir/.venv/bin/python" examples/plot_training_metrics.py \
    "$run_dir" --smooth 20 --diagnostic-smooth 3 \
    >> "$log_file" 2>&1; then
    print -r -- "[$timestamp] refreshed" >> "$log_file"
  else
    status=$?
    print -r -- "[$timestamp] refresh failed with status $status" >> "$log_file"
  fi

  if [[ -r "$run_dir/train.pid" ]]; then
    pid="$(< "$run_dir/train.pid")"
  else
    pid=""
  fi
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    print -r -- "[$timestamp] trainer stopped; updater exiting" >> "$log_file"
    break
  fi
  sleep "$interval_seconds"
done
