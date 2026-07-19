#!/bin/zsh
set -euo pipefail

repo_dir="/Users/erehmax/Documents/Projects/Personal/plump-bot"
run_dir="${PLUMP_V9_RUN_DIR:-$repo_dir/checkpoints/v9_10m_expert_seed1}"
warm_start="${PLUMP_V9_WARM_START:-$repo_dir/checkpoints/v8_10m_fastppo_seed1/best.pt}"

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
  --concurrent-episodes 32
  --minibatch-size 1440
  --microbatch-size 1440
  --lr 1e-4
  --self-play-fraction 0.3
  --heuristic-fraction 0.3
  --mixed-fraction 0.3
  --historical-fraction 0.1
  --historical-max-snapshots 4
  --play-search-fraction 0.25
  --search-min-worlds 4
  --search-max-worlds 12
  --search-node-budget 2048
  --search-maximum-js 0.05
  --replay-capacity 50000
  --replay-max-age 100
  --diag-samples 2048
  --diag-batch-size 512
  --eval-every 25
  --eval-batch-size 320
  --teacher-eval-every 100
  --save-every-minutes 30
  --checkpoint-dir "$run_dir"
  --log-dir "$run_dir"
  --precision bf16
  --max-seq-len 100
  --d-model 256
  --n-layers 12
  --n-heads 8
  --d-ff 1024
  --context-hidden-dim 256
)

checkpoints=("$run_dir"/plump_v5_cycle_*.pt(N))
if (( ${#checkpoints} > 0 )); then
  newest="${checkpoints[-1]}"
  args+=(--resume-from "$newest")
  print -r -- "Resuming v9 expert iteration from $newest" >> "$run_dir/train.log"
else
  args+=(--initialize-from-v4 "$warm_start")
  print -r -- "Warm-starting v9 expert iteration from v8 checkpoint $warm_start (fresh Q heads, oracle head dropped)" \
    >> "$run_dir/train.log"
fi

exec "$repo_dir/.venv/bin/python" -u examples/train_search.py "${args[@]}" \
  >> "$run_dir/train.log" 2>> "$run_dir/train.err.log"
