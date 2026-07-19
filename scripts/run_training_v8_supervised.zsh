#!/bin/zsh
set -u

repo_dir="/Users/erehmax/Documents/Projects/Personal/plump-bot"
run_dir="${PLUMP_V8_RUN_DIR:-$repo_dir/checkpoints/v8_10m_fastppo_seed1}"
max_restarts=50

mkdir -p "$run_dir"
for (( attempt = 1; attempt <= max_restarts; attempt++ )); do
  timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  print -r -- "[$timestamp] supervisor: launch attempt $attempt/$max_restarts" >> "$run_dir/supervisor.log"
  if zsh "$repo_dir/scripts/run_training_v8_10m_seed1.zsh"; then
    print -r -- "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] supervisor: training completed" >> "$run_dir/supervisor.log"
    break
  fi
  print -r -- "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] supervisor: training exited with failure; restarting in 15s" >> "$run_dir/supervisor.log"
  sleep 15
done
