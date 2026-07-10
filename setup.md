# Plump Bot Setup

## Environment

The project targets Python 3.11-3.12 and uses `uv`:

```bash
uv sync
uv run pytest
```

PyTorch device selection prefers MPS, then CUDA, then CPU.

## Rules

The engine remains the source of truth:

- standard 52-card deck;
- players must follow suit when able;
- no trump by default;
- the final bidder may not make total bids equal the hand size;
- highest bid leads first, with bidding order breaking ties;
- successful nonzero bids score `10 + bid`;
- successful zero bids score 5;
- missed bids score 0.

Every v4/v5 run stores a hash of these rules. A mismatched checkpoint or
replay is rejected.

## Supported Deployment Space

Schema-v4 observations, schema-v5 training, and controlled evaluation cover:

```text
players: 3, 4, 5
cards:   3, 4, 5, 6, 7, 8, 9, 10
```

The engine itself supports a wider range for tests and interactive play, but
those states are outside the trained deployment contract.

## Commands

Architecture smoke check:

```bash
uv run python examples/model_forward.py
```

Default schema-v5 expert-iteration training:

```bash
scripts/run_training_v5_50m_seed1.zsh
```

This starts from random weights unless a schema-v5 checkpoint already exists
in the run directory. Search controls every non-forced focal decision and
distills split-half-stable policy and Q targets. The v5 replay, optimizer,
league, RNG, baseline, and search schedules resume together.

Legacy schema-v4 PPO training:

```bash
PYTORCH_MPS_FAST_MATH=1 \
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.95 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.85 \
uv run python examples/train_ppo.py --iterations 2500
```

The production command warm-starts schema v4 from the best compatible v3
checkpoint, drops the old owner head, and starts a fresh optimizer and history:

```bash
uv run python examples/train_ppo.py \
  --iterations 2500 \
  --warm-start-v3 checkpoints/v3_50m_search_20260608_174712_seed1/best.pt \
  --historical-checkpoint checkpoints/v3_50m_search_20260608_174712_seed1/best.pt
```

Defaults use four PPO epochs, 1,440-action logical minibatches, 576-action
physical microbatches, a 50.7M-parameter transformer, and 30-minute checkpoints.

## Legacy V4 Counterfactual Search

Each iteration first collects ordinary focal-player rollouts under the
`30/30/30/10` self-play, heuristic, mixed, and historical opponent objective.
Counterfactual search runs afterward on saved rollout states, before PPO:

1. Bid and play candidates must have at least two legal actions.
2. Up to 24 states per eligible phase are selected by stratified shuffling over
   player count, hand size, bidding position, trick position, and opponent arm.
3. Forced plays are excluded. This includes the literal last card and any
   earlier state where following suit leaves only one legal card.
4. Every legal root action is evaluated over common public-consistent hidden
   worlds and random tapes.
5. Later focal decisions use the observation-only current policy; opponents
   receive only their legitimate private observations.

Hidden worlds preserve exact opponent capacities, known cards, public voids,
and the undealt pool. With at most three tricks and at most 4,096 projected
nodes, continuations are enumerated; otherwise search uses paired Monte Carlo.

For root action `a`, search estimates `Q(a)` and computes:

```text
baseline = sum(old_policy(a) * Q(a))
regret(a) = Q(a) - baseline
target(a) proportional to old_policy(a) * exp(regret(a) / 2)
```

Bid and play search become eligible independently after update 250 and three
diagnostics with direct-head phase EV at least `0.30`. Accepted stable labels
replace PPO policy and entropy supervision for those states; PPO value,
trick-count, and hidden-card losses remain active. Rejected and unsearched
states retain normal PPO policy learning.

Search policy updates use a separate stateless SGD step with reverse-KL
backtracking. Once positive-regret matching begins, a `0.002` entropy-floor
hinge prevents cumulative collapse. PPO and search entropy are both calculated
only over post-mask legal actions.

Diagnostics include:

- direct value-head EV and trick-distribution-implied EV, including bid/play
  splits;
- exact-capacity sampler infeasible-candidate rejection and failed-draw rates;
- search acceptance, agreement, divergence, KL, entropy floor, and routing.

The search gate currently uses direct-head EV. The trick-implied EV gauge is
logged for comparison before changing that gate.

Controlled and full-game evaluation:

```bash
uv run python examples/evaluate_policy.py checkpoints/v4/best.pt
```

Three-seed acceptance:

```bash
uv run python examples/check_acceptance.py \
  --candidate checkpoints/v4/seed1/best.pt \
  --candidate checkpoints/v4/seed2/best.pt \
  --candidate checkpoints/v4/seed3/best.pt \
  --legacy checkpoints/legacy.pt
```

Root-search gate:

```bash
uv run python examples/run_search_gate.py checkpoints/v4/best.pt
```

Training metrics dashboard:

```bash
uv run python examples/plot_training_metrics.py checkpoints/v4
```

## Migration Boundary

Schema-v1 models are retained only by
`plump/modeling/legacy_encoding_v1.py` and
`plump/modeling/legacy_torch_model_v1.py`. `LegacyCheckpointPolicy` can evaluate
old checkpoints. Schema-v2/v3 checkpoints can initialize v4 weights and act as
frozen opponents, but cannot resume a v4 optimizer. Point-head metrics,
independent hand BCE, persistent standalone distillation, and superseded
training entrypoints have been removed.

Schema-v5 keeps the v4 observation encoding but has a separate model,
checkpoint, replay, optimizer, logger, and entrypoint. V1-v4 checkpoints are
evaluation or legacy-PPO artifacts only and cannot initialize or resume v5.
