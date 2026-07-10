#!/bin/zsh
set -euo pipefail

repo_dir="/Users/erehmax/Documents/side_projects/plump-bot"
run_dir="${PLUMP_V4_RUN_DIR:-$repo_dir/checkpoints/v4_50m_sinkhorn_20260608_204818_seed1}"
warm_start="$repo_dir/checkpoints/v3_50m_search_20260608_174712_seed1/best.pt"

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
  --microbatch-size 576
  --lr 2e-5
  --owner-coef 0.05
  --owner-capacity-coef 0.1
  --owner-sinkhorn-iterations 16
  --self-play-fraction 0.3
  --heuristic-fraction 0.3
  --mixed-fraction 0.3
  --historical-fraction 0.1
  --historical-checkpoint "$warm_start"
  --historical-max-snapshots 4
  --historical-current-snapshots
  --counterfactual-search
  --search-min-iteration 250
  --search-ev-threshold 0.30
  --search-states-per-phase 24
  --search-replay-capacity 50000
  --search-replay-max-age 250
  --search-lr 1e-4
  --search-minibatch-size 256
  --search-entropy-floor-coef 0.002
  --diag-every 5
  --diag-samples 2048
  --diag-batch-size 256
  --eval-every 25
  --eval-batch-size 320
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

checkpoints=("$run_dir"/plump_v4_iter_*.pt(N))
if (( ${#checkpoints} > 0 )); then
  newest="${checkpoints[-1]}"
  args+=(--resume-from "$newest" --resume-optimizer)
  print -r -- "Resuming schema v4 from $newest" >> "$run_dir/train.log"
else
  args+=(--warm-start-v3 "$warm_start")
  print -r -- "Warm-starting schema v4 from $warm_start with a fresh owner head and optimizer" \
    >> "$run_dir/train.log"
fi

exec "$repo_dir/.venv/bin/python" -u examples/train_ppo.py "${args[@]}" \
  >> "$run_dir/train.log" 2>> "$run_dir/train.err.log"
