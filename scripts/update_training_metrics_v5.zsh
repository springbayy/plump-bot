#!/bin/zsh
set -euo pipefail

repo_dir="/Users/erehmax/Documents/side_projects/plump-bot"
run_dir="${PLUMP_V5_RUN_DIR:-$repo_dir/checkpoints/v5_50m_expert_seed1}"

cd "$repo_dir"
while true; do
  if [[ -f "$run_dir/metrics.csv" ]]; then
    "$repo_dir/.venv/bin/python" examples/plot_search_metrics.py \
      "$run_dir" --smooth 5 \
      >> "$run_dir/metrics_update.log" 2>&1 || true
  fi
  sleep 60
done
