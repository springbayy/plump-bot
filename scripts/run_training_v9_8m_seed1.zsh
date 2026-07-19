#!/bin/zsh
set -euo pipefail

repo_dir="/Users/erehmax/Documents/Projects/Personal/plump-bot"
run_dir="${PLUMP_V9_PPO_RUN_DIR:-$repo_dir/checkpoints/v9_8m_wideppo_seed1}"

cd "$repo_dir"
mkdir -p "$run_dir"
print -r -- "$$" > "$run_dir/train.pid"

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.95
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.85
export PYTORCH_MPS_FAST_MATH=0

args=(
  --iterations 2500
  --seed 1
  --training-mode round
  --rounds-per-configuration 16
  --num-envs 384
  --event-length-buckets 8,16,32,64
  --batch-packing numpy
  --lean-rollout-forward
  --ppo-epochs 4
  --target-kl 0.02
  --minibatch-size 1440
  --microbatch-size 1440
  --lr 2e-4
  --trick-baseline
  --oracle-critic
  --oracle-value-coef 0.5
  --suit-presence-head
  --suit-coef 0.1
  --owner-coef 0.0
  --owner-capacity-coef 0.1
  --owner-sinkhorn-iterations 16
  --self-play-fraction 0.35
  --heuristic-fraction 0.10
  --mixed-fraction 0.30
  --historical-fraction 0.25
  --explore-eps-bid 0.10
  --explore-eps-play 0.02
  --historical-max-snapshots 8
  --historical-current-snapshots
  --league-temperature 2.0
  --league-reward-decay 0.95
  --league-meta-solver regret_matching
  --batched-league-sampling
  --league-probe-fraction 0.10
  --league-eval-every 50
  --league-eval-deals-per-configuration 2
  --mmd-enabled
  --mmd-coef 0.05
  --no-counterfactual-search
  --diag-every 5
  --diag-samples 2048
  --diag-batch-size 512
  --eval-every 25
  --eval-batch-size 320
  --save-every-minutes 30
  # The detached atomic refresher owns metrics.png for this run.
  --plot-every 0
  --checkpoint-dir "$run_dir"
  --log-dir "$run_dir"
  --precision bf16
  --max-seq-len 100
  --d-model 320
  --n-layers 6
  --n-heads 10
  --d-ff 896
  --context-hidden-dim 256
)

checkpoints=("$run_dir"/plump_v4_iter_*.pt(N))
if (( ${#checkpoints} > 0 )); then
  newest="${checkpoints[-1]}"
  args+=(--resume-from "$newest" --resume-optimizer)
  print -r -- "Resuming v9 PPO (schema v4, 8.2M wide model) from $newest" >> "$run_dir/train.log"
else
  print -r -- "Starting v9 PPO (schema v4, 8.2M wide model) from random weights" >> "$run_dir/train.log"
fi

exec "$repo_dir/.venv/bin/python" -u examples/train_ppo.py "${args[@]}" \
  >> "$run_dir/train.log" 2>> "$run_dir/train.err.log"
