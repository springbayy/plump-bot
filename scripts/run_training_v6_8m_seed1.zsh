#!/bin/zsh
set -euo pipefail

repo_dir="/Users/erehmax/Documents/side_projects/plump-bot"
run_dir="${PLUMP_V6_RUN_DIR:-$repo_dir/checkpoints/v6_8m_fastppo_seed1}"

cd "$repo_dir"
mkdir -p "$run_dir"
print -r -- "$$" > "$run_dir/train.pid"

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.95
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.85
export PYTORCH_MPS_FAST_MATH=1

args=(
  --iterations 2500
  --seed 1
  --training-mode round
  --rounds-per-configuration 16
  --num-envs 384
  --ppo-epochs 4
  --minibatch-size 1440
  --microbatch-size 1440
  --lr 3e-4
  --trick-baseline
  --owner-coef 0.05
  --owner-capacity-coef 0.1
  --owner-sinkhorn-iterations 16
  --self-play-fraction 0.3
  --heuristic-fraction 0.3
  --mixed-fraction 0.3
  --historical-fraction 0.1
  --historical-max-snapshots 4
  --historical-current-snapshots
  --no-counterfactual-search
  --diag-every 5
  --diag-samples 2048
  --diag-batch-size 512
  --eval-every 25
  --eval-batch-size 320
  --save-every-minutes 30
  --checkpoint-dir "$run_dir"
  --log-dir "$run_dir"
  --precision bf16
  --max-seq-len 64
  --d-model 256
  --n-layers 6
  --n-heads 8
  --d-ff 1024
  --context-hidden-dim 256
)

checkpoints=("$run_dir"/plump_v4_iter_*.pt(N))
if (( ${#checkpoints} > 0 )); then
  newest="${checkpoints[-1]}"
  args+=(--resume-from "$newest" --resume-optimizer)
  print -r -- "Resuming v6 (schema v4, 6M params) from $newest" >> "$run_dir/train.log"
else
  print -r -- "Starting v6 (schema v4, 6M params) from random weights" >> "$run_dir/train.log"
fi

exec "$repo_dir/.venv/bin/python" -u examples/train_ppo.py "${args[@]}" \
  >> "$run_dir/train.log" 2>> "$run_dir/train.err.log"
