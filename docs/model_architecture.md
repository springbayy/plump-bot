# Schema-v4 Observation / Schema-v5 Search Model

Schema v5 deliberately keeps the schema-v4 observation contract unchanged.
Its checkpoint schema is separate and adds masked bid-Q and card-Q heads to
the existing policy, round/game value, trick-count, and ownership heads. Q
heads are trained from complete legal-action tree values and are used for tree
move ordering and diagnostics, never as a tree backup substitute.

Schema v4 is a shared observation-only policy for 3-5 players and 3-10 cards.
It does not change the game rules.

## Input Contract

Encode only the acting player's public observation:

```python
observation = env.get_observation(env.current_player())
encoded = encode_observation(observation, model_config)
```

All players are cyclically relative to the observer. Relative player 0 is the
observer, so rotating every absolute seat leaves the encoded decision problem
unchanged.

## Round-Local And Reserved Context

Event tokens contain only events from the current round. Each token has eight
integer fields:

```text
event type, relative player, rank, suit, card, bid, trick, trick position
```

Dense context contains the hand, legal actions, bids, tricks won, cards played,
voids, bidding position, bidding starter, first leader, current leader, and
position in the current trick.

Separate game context contains cumulative standings, focal-relative score
differences, current round, rounds remaining, schedule bounds and phase, plus
the complete ordered hand-size schedule. Schedule tokens carry hand size,
absolute schedule position, and past/current/future status. Phase-1 PPO zeros
these values and sets the explicit game-context mask false.

## Heads

The transformer emits:

- masked bid logits over `0..hand_size`;
- masked card logits over the 52-card deck;
- schema-v5 masked bid-Q values over `0..hand_size`;
- schema-v5 masked card-Q values over the 52-card deck;
- separate round-relative and full-game-relative value predictions;
- final trick-count distributions for every relative player;
- a capacity-conditioned hidden-card assignment over relative opponents plus
  an undealt class.

There is no point head. A player's probability of scoring is the probability
that the trick-count distribution equals that player's bid.

Final trick-count masks enforce:

```text
tricks already won <= final tricks <= tricks won + unresolved tricks
```

The owner head receives exact public remaining capacities for each relative
opponent and the undealt pool. It scores every hidden-card/owner pair, masks
observer cards, played cards, public voids, inactive opponents, and unavailable
undealt classes, then applies differentiable Sinkhorn normalization. Projected
rows sum to one and projected owner columns sum to their public capacities.

Training uses categorical cross-entropy against each hidden card's true owner.
A second loss penalizes capacity error in the raw pre-Sinkhorn marginals, so
the underlying scores learn feasibility instead of relying entirely on the
projection. Raw count errors are normalized by the state's hidden-card count
before squaring so this regularizer stays comparable across configurations.
Diagnostics report opponent-only accuracy and true-owner probability
separately from the easier undealt assignments. Search consumes only the
projected marginals and still performs exact-capacity determinization sampling.

## Checkpoints

Schema-v5 checkpoints retain schema-v4 encoded observations but require the
two Q heads and expert-iteration metadata. Round-mode v5 starts from random
weights or imports a schema-v4 PPO checkpoint's trunk and non-Q heads via
`--initialize-from-v4` (Q heads always start fresh). Its compressed replay
sidecar, optimizer, RNG, frozen-league
references, position baseline, and search ramps resume as one training state.
Older schemas cannot initialize or resume v5.

Schema-v4 checkpoints include the model config, rules fingerprint, context-mask
mode, and position-baseline state. Compatible v3 checkpoints import the trunk,
policy, value, trick-count, game-context, schedule, and position-baseline
weights. The old independent owner head is dropped; the capacity-conditioned
assignment head and optimizer are fresh.
