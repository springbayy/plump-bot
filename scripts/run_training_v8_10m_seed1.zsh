#!/bin/zsh
set -euo pipefail

repo_dir="/Users/erehmax/Documents/Projects/Personal/plump-bot"
run_dir="${PLUMP_V8_RUN_DIR:-$repo_dir/checkpoints/v8_10m_fastppo_seed1}"

cd "$repo_dir"
mkdir -p "$run_dir"
print -r -- "$$" > "$run_dir/train.pid"

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.95
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.85
# Keep fast-math disabled for a clean comparison with the failed deep run.
export PYTORCH_MPS_FAST_MATH=0

args=(
  --iterations 2500
  --seed 1
  --training-mode round
  --rounds-per-configuration 16
  --num-envs 384
  --ppo-epochs 4
  --target-kl 0.02
  --minibatch-size 1440
  --microbatch-size 480
  --lr 2e-4
  --trick-baseline
  --oracle-critic
  --oracle-value-coef 0.5
  --suit-presence-head
  --suit-coef 0.1
  --owner-coef 0.0
  --owner-capacity-coef 0.1
  --owner-sinkhorn-iterations 16
  --self-play-fraction 0.3
  --heuristic-fraction 0.3
  --mixed-fraction 0.3
  --historical-fraction 0.1
  --historical-max-snapshots 8
  --historical-current-snapshots
  --league-temperature 2.0
  --league-reward-decay 0.95
  --league-meta-solver regret_matching
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
  --plot-every 5
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
  print -r -- "Resuming v8 (schema v4, 8.2M wide model, KL control + MMD + PSRO-lite league) from $newest" >> "$run_dir/train.log"
else
  print -r -- "Starting v8 (schema v4, 8.2M wide model, KL control + MMD + PSRO-lite league) from random weights" >> "$run_dir/train.log"
fi

exec "$repo_dir/.venv/bin/python" -u examples/train_ppo.py "${args[@]}" \
  >> "$run_dir/train.log" 2>> "$run_dir/train.err.log"
