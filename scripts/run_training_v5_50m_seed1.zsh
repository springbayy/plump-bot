#!/bin/zsh
set -euo pipefail

repo_dir="/Users/erehmax/Documents/side_projects/plump-bot"
run_dir="${PLUMP_V5_RUN_DIR:-$repo_dir/checkpoints/v5_50m_expert_seed1}"

cd "$repo_dir"
mkdir -p "$run_dir"
print -r -- "$$" > "$run_dir/train.pid"

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.95
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.85
export PYTORCH_MPS_FAST_MATH=1

args=(
  --cycles 2500
  --seed 1
  --training-mode round
  --rounds-per-configuration 16
  --minibatch-size 1440
  --microbatch-size 576
  --lr 2e-5
  --self-play-fraction 0.3
  --heuristic-fraction 0.3
  --mixed-fraction 0.3
  --historical-fraction 0.1
  --historical-max-snapshots 4
  --search-min-worlds 4
  --search-max-worlds 32
  --search-node-budget 65536
  --search-maximum-js 0.05
  --replay-capacity 50000
  --replay-max-age 100
  --diag-samples 2048
  --diag-batch-size 256
  --eval-every 25
  --eval-batch-size 320
  --teacher-eval-every 100
  --save-every-minutes 30
  --checkpoint-dir "$run_dir"
  --log-dir "$run_dir"
  --precision bf16
  --max-seq-len 64
  --d-model 704
  --n-layers 8
  --n-heads 11
  --d-ff 2560
  --context-hidden-dim 512
)

checkpoints=("$run_dir"/plump_v5_cycle_*.pt(N))
if (( ${#checkpoints} > 0 )); then
  newest="${checkpoints[-1]}"
  args+=(--resume-from "$newest")
  print -r -- "Resuming schema v5 from $newest" >> "$run_dir/train.log"
else
  print -r -- "Starting schema v5 from random weights" >> "$run_dir/train.log"
fi

exec "$repo_dir/.venv/bin/python" -u examples/train_search.py "${args[@]}" \
  >> "$run_dir/train.log" 2>> "$run_dir/train.err.log"
