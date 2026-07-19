#!/bin/zsh
# Pull the newest training checkpoint from the Modal volume and launch the
# local play GUI against it. Training keeps running remotely; this only
# downloads a file. Re-run to refresh to the newest checkpoint.
set -euo pipefail

repo_dir="/Users/erehmax/Documents/Projects/Personal/plump-bot"
run_name="${1:-v9_8m_wideppo_seed1}"
port="${2:-8765}"
dest_dir="$repo_dir/checkpoints/modal/$run_name"
dest="$dest_dir/play_latest.pt"

close_listening_port() {
  local selected_port="$1"
  local listeners remaining
  local -a pids

  listeners="$(lsof -nP -tiTCP:"$selected_port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$listeners" ]]; then
    return
  fi

  pids=(${(f)listeners})
  print -r -- "Closing existing listener on port $selected_port (PID ${pids[*]})"
  kill -TERM -- $pids 2>/dev/null || true

  for _ in {1..20}; do
    remaining="$(lsof -nP -tiTCP:"$selected_port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -z "$remaining" ]]; then
      return
    fi
    sleep 0.1
  done

  pids=(${(f)remaining})
  print -r -- "Listener did not exit cleanly; force-closing PID ${pids[*]}"
  kill -KILL -- $pids 2>/dev/null || true
}

mkdir -p "$dest_dir"
newest=$("$repo_dir/.venv/bin/python" -m modal volume ls plump-checkpoints "$run_name/" \
  | grep -o 'plump_v4_iter_[0-9]*\.pt' | sort | tail -1)
if [[ -z "$newest" ]]; then
  print -r -- "No checkpoints found on volume for run '$run_name'" >&2
  exit 1
fi

remote_digits="${${newest#plump_v4_iter_}%.pt}"
remote_iteration=$((10#$remote_digits))
local_iteration=""
if [[ -f "$dest" ]]; then
  local_iteration=$("$repo_dir/.venv/bin/python" - "$dest" 2>/dev/null <<'PY' || true
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
iteration = payload.get("iteration")
if iteration is not None:
    print(int(iteration))
PY
  )
fi

if [[ "$local_iteration" == "$remote_iteration" ]]; then
  print -r -- "Using already-downloaded latest checkpoint: $newest"
else
  print -r -- "Pulling $newest -> $dest"
  "$repo_dir/.venv/bin/python" -m modal volume get --force \
    plump-checkpoints "$run_name/$newest" "$dest"
fi

close_listening_port "$port"

exec "$repo_dir/.venv/bin/python" "$repo_dir/examples/play_gui.py" \
  --checkpoint "$dest" --port "$port"
