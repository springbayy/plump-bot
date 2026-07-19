# Position-Robust Training

## Recommended Pipeline: Fast PPO Baseline, Then Expert Iteration

Stage 1 trains an 8.2M-parameter wide, six-layer schema-v4 PPO baseline
(`d_model=320`, `d_ff=896`) with KL early stopping, the MMD magnet, oracle
critic, PSRO-lite league sampling, and the suit-presence belief head enabled
(per-card owner loss disabled); stage 2 warm-starts schema-v5 expert iteration
from it:

```bash
zsh scripts/run_training_v8_10m_seed1.zsh        # stage 1: PPO from random weights
zsh scripts/run_training_v9_10m_expert_seed1.zsh # stage 2: expert iteration from v8 best.pt
zsh scripts/run_training_v8_then_v9.zsh          # or both stages back to back
```

The earlier ~6M v6/v7 scripts remain for reproducibility.

## Oracle (Asymmetric) Critic

`--oracle-critic` adds a privileged value head that receives the ground-truth
hidden-card ownership (opponent hands plus the undealt pool) alongside the
shared trunk state. It is training-only: privileged inputs feed only this
head, never the trunk or policy, so acting and evaluation are unchanged. When
enabled, rollout GAE bootstraps on the oracle value instead of the plain
value head, which removes deal luck from advantages; the plain head keeps
training on the same residual returns (weight `--oracle-value-coef`, default
0.5) and remains the head used at inference. The v4-to-v5 warm start drops
oracle weights (expert iteration does not use GAE). Diagnostics report
`pred_oracle_value_explained_variance` next to the plain value EV.

## Suit-Presence Belief Head

Predicting the exact owner of all 52 hidden cards proved too noisy a target.
`--suit-presence-head` adds a compact alternative: for every relative
opponent and suit, a binary logit for "does that player currently hold at
least one card of this suit" (void tracking), trained with masked binary
cross-entropy (`--suit-coef`, default 0.1) against ground-truth current
hands. The observer's own slot and padding players are masked. The v8 run
enables this head and disables the per-card owner objective
(`--owner-coef 0.0`, which also skips the owner/Sinkhorn forward during
updates). The oracle critic still receives per-card ground truth as
privileged *input* — that is not a prediction target. Diagnostics report
`pred_suit_presence_accuracy` and `pred_suit_presence_brier`; the v4-to-v5
warm start drops this head like the oracle head.

## PSRO-Lite League Opponent Sampling

The historical arm is a small PSRO-lite league containing frozen checkpoints,
a heuristic anchor, and the current policy. A fixed deal bank estimates the
focal-vs-table payoff matrix. Frozen/frozen cells are cached permanently;
cells involving `current` refresh every `--league-eval-every` iterations
(v8 uses 50). Regret matching turns the matrix into a meta-mixture, preserving
mass across non-transitive strategy cycles. Historical and mixed arms sample
the snapshot portion of that mixture with light uniform smoothing. Until a
matrix is available, sampling falls back to the per-snapshot difficulty EMA.

When the pool exceeds `--historical-max-snapshots` (v8 uses 8), the lowest
meta-mass snapshot is evicted while the newest remains protected. Payoffs,
mixture, snapshot paths, and fallback EMAs persist in checkpoints. The
heuristic arm remains a separate 30% training anchor.

## PPO KL Control

`--target-kl` checks mean approximate KL after every PPO epoch and skips the
remaining epochs once the target is exceeded. V8 uses `0.02`; `epochs_run`
is logged so oversized updates are visible rather than silently applying all
four epochs.

## MMD Magnet-KL Regularizer

`--mmd-enabled` adds a magnet-KL term (Magnetic Mirror Descent style): a
frozen magnet copy of the policy is updated as an EMA of the live weights
(`--mmd-magnet-decay`, default 0.995) after every PPO update, and
`KL(pi_theta || pi_magnet)` over legal actions is added to the loss with
weight `--mmd-coef` (default 0.05). This gives self-play last-iterate
stability if strategy cycling appears. V8 enables it with coefficient `0.05`.
The magnet state persists in checkpoints.

## Diagnostics Dashboard

Training refreshes `metrics.png` in-process every five iterations; a separate
metrics-updater process is no longer required. The former throughput panel
shows suit-presence BCE and accuracy for early, middle, and late round states,
where later states should become easier as public card information accumulates.

## Schema-v5 Expert Iteration

`examples/train_search.py` replaces PPO policy gradients with information-set
tree targets. A round run may start from random weights or warm-start its
trunk and non-Q heads from a schema-v4 PPO checkpoint via
`--initialize-from-v4`; the two Q heads always start fresh:

```bash
uv run python examples/train_search.py --cycles 2500 \
  --initialize-from-v4 checkpoints/v6_8m_fastppo_seed1/best.pt
```

Collection advances `--concurrent-episodes` episodes in lockstep and batches
all frozen-model turns per policy. Every non-forced focal bid decision is
searched; play decisions are searched with probability
`--play-search-fraction` and otherwise sampled from the frozen policy
(rejected searches also fall back to the frozen policy). Skipped decisions
still train value, trick-count, and ownership heads.

Every non-forced focal turn searches common public-consistent hidden worlds.
Focal choices are shared across worlds with the same information set, while
worlds partition after observable opponent actions. Accepted split-half-stable
trees train policy cross-entropy and legal-action Q regression. Every focal
state trains terminal value, final trick count, and hidden-card assignment.

Each cycle freezes the behavior model, completes 16 rounds for every one of
the 24 player/card configurations, retains the 30/30/30/10 opponent objective,
then runs `ceil(4 * new_states / 1440)` replay updates. Checkpoints are schema
v5 and contain optimizer, replay, RNG, league references, position baseline,
and search-ramp state. V1-v4 weights cannot initialize or resume a v5 round
run.

Full-game mode may initialize from a v5 round checkpoint:

```bash
uv run python examples/train_search.py \
  --training-mode game \
  --initialize-game-from checkpoints/v5/best.pt
```

Game mode trains terminal game value and auxiliary heads but intentionally
does not apply round-local policy/Q search losses.

## Legacy PPO Objective

Phase 1 optimizes terminal round-relative score:

```text
reward_i = score_i - mean(score_j for j != i)
```

This is a strong whole-game proxy, not a proof of game-optimal play. Full-game
standings and risk preferences are nonlinear. Schema v4 retains masked
standings and ordered schedule inputs for a later game-aware finetune.

## Legacy Balanced PPO

The deployment prior is uniform over all 24 combinations of 3-5 players and
3-10 cards. Every rollout batch completes the same number of rounds for every
configuration.

Within each round, bid and card PPO terms are summed. The loss first averages
over rounds for each configuration, then averages the 24 configuration losses.
Bid and play decisions are not separately reweighted.

The value target is residualized with a lagged EMA intercept keyed by:

```text
(players, hand_size, bidding_position)
```

Trick-count and hidden-owner losses are normalized over valid labels and
controlled only by `trick_coef` and `owner_coef`. The hidden-owner objective is
projected assignment cross-entropy plus `owner_capacity_coef` times raw
pre-Sinkhorn normalized capacity MSE.

Training uses four explicit opponent arms: 30% current-policy self-play, 30%
one focal player against heuristic opponents, 30% independently mixed current,
heuristic, and historical opponents, and 10% historical opponents. Every round
stores PPO actions for one focal current-policy player only. Per-arm round
weights make these percentages exact in the gradient objective. Before a
historical checkpoint exists, the historical 10% falls back to self-play.
Within mixed rounds, every non-focal seat independently selects one of the
three opponent categories with equal probability.

## Run Legacy PPO

```bash
PYTORCH_MPS_FAST_MATH=1 \
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.95 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.85 \
uv run python examples/train_ppo.py \
  --iterations 2500 \
  --warm-start-v3 checkpoints/v3_50m_search_20260608_174712_seed1/best.pt
```

`examples/train_ppo.py` remains available for v4 reproducibility. Its
optimizer, replay, checkpoints, and metrics are not shared with v5.

The default command covers all 24 configurations. Checkpoints contain schema
and rules fingerprints. V2 checkpoints are initialization/frozen-opponent
inputs only; only v4 checkpoints can resume training.

The hardware-oriented defaults use a 50.7M-parameter, eight-layer transformer,
64 event slots, 384 concurrent rounds, BF16 kernels, four PPO epochs,
1,440-sample logical minibatches, and 576-sample physical microbatches with
exact gradient accumulation. Checkpoints are written every 30 minutes. Frozen historical turns
are grouped by checkpoint and inferred in batches. Controlled evaluation and
diagnostics run every 25 and 5 iterations respectively; diagnostics use
256-state physical batches to avoid post-update MPS memory spikes.

## Evaluation

```bash
uv run python examples/evaluate_policy.py checkpoints/v4/best.pt \
  --comparison legacy=checkpoints/legacy.pt \
  --comparison recent=checkpoints/recent.pt
```

Evaluation uses deterministic deal banks, rotates each deal over every focal
hand and bidding position, and reports raw score, relative score, bid accuracy,
first-leader rate, compute, bootstrap confidence intervals, and position cells.
It separately reports complete multi-round schedules. Merely running a round
policy inside a full game is compatibility evidence, not strength evidence.

## Legacy Search Gate

```bash
uv run python examples/run_search_gate.py checkpoints/v4/best.pt \
  --output checkpoints/v4/search_gate.json
```

Root search samples exact-capacity hidden worlds under public void constraints.
Every legal root action uses common worlds and seeds. All later focal decisions
use the observation-only policy, and opponents see only their private
observations. Any deeper implementation must share focal-node statistics by
information set.

The gate passes only when paired macro relative reward has a positive 95%
confidence lower bound and no confirmed configuration-position regression is
worse than 0.5 points. Three-card and bidding-position results are written
explicitly. Breadth is swept first; the smallest budget within 5% of peak
controlled performance is selected.

## Legacy Online Search Routing

Bid and play search activate independently only after update 250 and three
held-out diagnostics with phase value explained variance at least 0.30.
Accepted labels replace PPO policy and entropy terms for those states; value,
trick-count, and owner losses remain active. Search replay is balanced, capped
at 50,000 labels, and expires after 250 updates. Separate bid/play policy steps
use stateless SGD with reverse-KL backtracking. Once positive-regret matching
starts, a `0.002` hinge penalty keeps the model's legal-action entropy at least
as high as the original policy-anchored search target.

Diagnostics log direct-head and trick-distribution-implied value explained
variance separately using `1 - Var(target - prediction) / Var(target)`. The
search eligibility gate remains on direct-head EV until measured runs justify
changing it. Search also logs infeasible candidate rejections and failed
complete determinizations from the exact-capacity sampler.

Round-local search is disabled by default in `--training-mode game`.

## Legacy PPO Full-Game Mode

```bash
uv run python examples/train_ppo.py \
  --training-mode game \
  --game-schedule 10,9,8,7,6,5,4,3,4,5,6,7,8,9,10
```

Full-game mode enables standings, ordered schedule tokens, and the game-value
head. It gives zero reward at round boundaries and terminal relative cumulative
score at game completion.

## Three-Seed Acceptance

```bash
uv run python examples/check_acceptance.py \
  --candidate checkpoints/v4/seed1/best.pt \
  --candidate checkpoints/v4/seed2/best.pt \
  --candidate checkpoints/v4/seed3/best.pt \
  --legacy checkpoints/legacy.pt
```

Acceptance requires a positive paired 95% confidence lower bound against the
legacy checkpoint for every independently trained seed.
