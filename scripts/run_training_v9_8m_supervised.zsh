#!/bin/zsh
set -u

repo_dir="/Users/erehmax/Documents/Projects/Personal/plump-bot"
run_dir="${PLUMP_V9_PPO_RUN_DIR:-$repo_dir/checkpoints/v9_8m_wideppo_seed1}"
max_restarts=50

mkdir -p "$run_dir"
for (( attempt = 1; attempt <= max_restarts; attempt++ )); do
  timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  print -r -- "[$timestamp] supervisor: launch attempt $attempt/$max_restarts" >> "$run_dir/supervisor.log"
  if zsh "$repo_dir/scripts/run_training_v9_8m_seed1.zsh"; then
    print -r -- "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] supervisor: training completed" >> "$run_dir/supervisor.log"
    break
  fi
  print -r -- "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] supervisor: training exited with failure; restarting in 15s" >> "$run_dir/supervisor.log"
  sleep 15
done
