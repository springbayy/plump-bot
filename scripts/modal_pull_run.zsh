#!/bin/zsh
# Continuously mirror a Modal training run's dashboard files to this machine.
#
#   zsh scripts/modal_pull_run.zsh [run_name] [--with-checkpoints]
#
# Pulls metrics.csv / latest.json / supervisor.log every PULL_INTERVAL_SEC
# (default 60), then renders metrics.png locally with the current dashboard
# code. This keeps the mirrored dashboard current without an older deployed
# trainer overwriting local presentation changes.
# --with-checkpoints additionally downloads any remote plump_v4_iter_*.pt
# that is not already present locally.
set -u

repo_dir="${0:A:h:h}"
volume_name="plump-checkpoints"
run_name="v9_8m_wideppo_seed1"
with_checkpoints=0
interval="${PULL_INTERVAL_SEC:-60}"

for arg in "$@"; do
  case "$arg" in
    --with-checkpoints) with_checkpoints=1 ;;
    *) run_name="$arg" ;;
  esac
done

dest="$repo_dir/checkpoints/modal/$run_name"
mkdir -p "$dest"
modal_cli=("$repo_dir/.venv/bin/python" -m modal)

print "pulling '$run_name' -> $dest every ${interval}s (ctrl-c to stop)"
while true; do
  pulled=0
  for file in metrics.csv latest.json supervisor.log; do
    if "${modal_cli[@]}" volume get --force "$volume_name" \
        "$run_name/$file" "$dest/$file" >/dev/null 2>&1; then
      (( pulled++ ))
    fi
  done
  if "$repo_dir/.venv/bin/python" "$repo_dir/examples/plot_training_metrics.py" \
      "$dest" --smooth 50 --diagnostic-smooth 50 >/dev/null 2>&1; then
    (( pulled++ ))
  else
    print "[$(date '+%H:%M:%S')] warning: metrics plot refresh failed"
  fi
  # Restart-only view: same dashboard, rows from the clean/explore soft
  # restart onward, lighter smoothing so recent movement shows.
  if "$repo_dir/.venv/bin/python" "$repo_dir/examples/plot_training_metrics.py" \
      "$dest" --output "$dest/metrics_new.png" --since-restart \
      --smooth 15 --diagnostic-smooth 5 >/dev/null 2>&1; then
    (( pulled++ ))
  else
    print "[$(date '+%H:%M:%S')] warning: metrics_new plot refresh failed"
  fi
  if (( with_checkpoints )); then
    remote_files=$("${modal_cli[@]}" volume ls "$volume_name" "$run_name" 2>/dev/null \
      | grep -o 'plump_v4_iter_[0-9]*\.pt' | sort -u)
    for file in ${(f)remote_files}; do
      [[ -n "$file" && ! -f "$dest/$file" ]] || continue
      print "[$(date '+%H:%M:%S')] downloading $file"
      "${modal_cli[@]}" volume get "$volume_name" \
        "$run_name/$file" "$dest/$file" >/dev/null 2>&1
    done
  fi
  print "[$(date '+%H:%M:%S')] pulled $pulled dashboard files for $run_name"
  sleep "$interval"
done
