# How broader exploration silently broke PPO — and the fix

*(v9_8m_wideppo_seed1, iterations 9583–9899, 2026-07-18. Detected by the checkpoint
arena the same day, rolled back to iter 9574, fixed in `plump/training/ppo.py`.)*

## 1. What we are doing: exploration as a behavior policy

PPO is normally on-policy: the policy π that we train is also the policy that
picks every rollout action. Our training deliberately breaks that in a
controlled way. Actions are sampled from a **behavior policy** b that is a
distorted copy of the current policy:

```
b(a|s) = (1 − ε) · softmax(logits(s) / T)(a)  +  ε · uniform_over_legal(a)
```

Two distortion knobs, both applied only during rollout collection (the model
itself is never changed):

- **ε-mixture** (long-standing): with probability ε the action is drawn
  uniformly over legal actions. Guarantees rare strategies (bid 0, bid max)
  keep a floor trial rate forever. ε is small where learning seats sit
  (0.08 bid / 0.02 play) and large only on frozen-opponent tables.
- **Temperature** (new): on 50% of self-play and mixed rounds, the current
  policy's seats sample from `softmax(logits / T)` with T = 2.0 for bids and
  1.5 for plays. Raising T flattens the distribution: an action at
  probability p moves toward p^(1/T) (renormalized). At T = 2 a 1% bid
  becomes ≈ 4–5%, which was the design target — visit meaningfully more
  diverse trajectories, then train normally on them.

## 2. The math that makes off-policy sampling legitimate

Sampling from b instead of π is fine **if** the objective re-weights each
sample by how over- or under-represented it was. For any function f:

```
E_b [ (π(a)/b(a)) · f(a) ]  =  Σ_a b(a) · (π(a)/b(a)) · f(a)  =  Σ_a π(a) · f(a)  =  E_π [ f(a) ]
```

So the *unclipped* policy-gradient surrogate stays unbiased as long as we
record `log b(a)` at collection time and use the ratio π_new/b. This is why
each rollout sample stores two log-probs:

- `old_logprob`        = log b(a)      — the mixture that actually sampled the action
- `old_policy_logprob` = log π_old(a)  — the raw policy at collection time

The KL early-stop gate already used both correctly (it measures movement of
π_new away from π_old, importance-weighted back through b).

## 3. Where the bug lived: what PPO's clip is *centered on*

PPO's loss is not the plain importance-weighted gradient. It is

```
L = − E [ min( r · A ,  clip(r, 1−δ, 1+δ) · A ) ]        δ = 0.2 here
```

The clip implements a trust region: "don't move the sampled action's
probability more than ±20% from *where it started*." The construction
assumes **r = 1 at the start of the update** — the min() is symmetric around
the starting point only if the starting point is 1.

The code computed:

```
r = exp(new_logprob − old_logprob)          # π_new / b   ← centered on b !
```

On the first epoch π_new = π_old, so r starts at **π_old/b — not 1**. With
ε-only exploration that gap is a few percent and nobody notices. With T = 2
it is enormous. For temperature sampling, b ∝ π^(1/T), so

```
r₀ = π_old(a) / b(a) = π_old(a)^(1 − 1/T) · Z_T        Z_T = Σ_a π_old(a)^(1/T)
```

Concrete bid distribution (0.5, 0.3, 0.1, 0.05, 0.03, 0.01, …), T = 2,
Z ≈ 2.21:

| action prob π | tempered b | starting ratio r₀ |
|---|---|---|
| 0.50 | 0.32 | **1.56** |
| 0.30 | 0.25 | 1.21 |
| 0.10 | 0.14 | 0.70 |
| 0.01 | 0.045 | **0.22** |

Every confident action starts *above* the clip band [0.8, 1.2]; every rare
boosted action starts *below* it.

## 4. The asymmetry: a ratchet that only flattens

Walk through `min(r·A, clip(r)·A)` for a favored action with r₀ = 1.56:

- **A > 0** (action was good): the clipped term 1.2·A is smaller, min picks
  it, and the clipped term has **zero gradient**. The good, confident action
  **cannot be reinforced** from this sample. Ever.
- **A < 0** (action was bad): r·A = 1.56·A is more negative, min picks the
  *unclipped* term, gradient flows at 1.56× strength. The confident action
  **can be punished, extra hard**.

And the mirror case for a rare boosted action with r₀ = 0.22:

- **A > 0**: unclipped side selected → it **can be reinforced** (and the
  trust region lets its probability grow to 1.2·b ≈ 5.4% — a 5× jump from
  π = 1% in a single update, because the "region" is centered on the
  inflated b, not on π).
- **A < 0**: clipped side (constant) selected → it **cannot be punished**.

Put together, on every tempered round, for every action, independent of what
the advantages actually say on average:

> confident actions: only pushable **down** · rare actions: only pushable **up**

That is a systematic entropy-injection ratchet built out of pure clipping
asymmetry. It transfers probability mass from the policy's best bids to its
long tail, update after update, ~14k tempered decisions per iteration.

## 5. What it did to the model (all measured)

- **Bid entropy** climbed steadily after the switch (0.62 → 1.12 and rising —
  beyond the one-time composition step from the new cell weights).
- **Average bid inflated** on *identical* probe deals: 1.50 → 1.73. Flattening
  a distribution whose support extends far above its mean (bids 0..10,
  mass concentrated at 1–2) necessarily raises the mean.
- **Playing strength collapsed**: in the 30-checkpoint arena (3507–9819,
  uniform bank, raw un-tempered policies), the two post-change checkpoints
  rated **−0.33** (iter 9602) and **−0.49** (iter 9819) pts/round vs a
  plateau wobbling around 0.0–0.24 with SE ≈ 0.04 — the two worst
  checkpoints in the field, and getting worse with training.
- **clip_fraction doubled** (0.07 → 0.14) the moment the regime started —
  in hindsight the direct fingerprint of ratios starting outside the band.

The irony: the diagnostic we watched for safety (approx_kl) stayed *low*,
because the KL gate was importance-corrected properly. The bias was not in
how far the policy moved per step, but in the *direction* it was allowed to
move.

## 6. The fix: separate the trust region from the importance weight

Factor the behavior ratio into two parts with different jobs:

```
r = π_new/b = w · ρ        w = π_old / b       (fixed constant per sample)
                           ρ = π_new / π_old   (starts at exactly 1)
```

- **ρ** is what the trust region should constrain — clip *it*.
- **w** is the exploration correction — a constant importance weight that
  scales the sample's contribution but must sit **outside** the min().

```
L = − E_b [ w · min( ρ · A ,  clip(ρ, 1−δ, 1+δ) · A ) ]
```

Now every sample starts at ρ = 1 (nothing pre-clipped), good tempered actions
can be reinforced again, rare boosted ones can be punished again, and the
unclipped branch w·ρ·A = (π_new/b)·A keeps the exact unbiasedness identity
from §2. w is bounded for temperature sampling
(w ≤ Z_T/(1−ε) ≈ 2.2 at T = 2), so no variance blow-up.

Two numerical-stability details (the reason the code looks the way it does in
`ppo.py`):

- `w·ρ` is computed directly as `exp(new_logprob − old_logprob)` — finite by
  construction because b actually sampled the action. Computing w and ρ
  separately would produce `0 · inf` for ε-uniform picks of actions with
  π_old ≈ e^(−1000).
- the clipped term bounds the log-ratio (clamp ±20) before exponentiating
  for the same reason; the clamp is invisible inside the clip band.
- `clip_fraction` now tests ρ against the band in log space, so it again
  measures real trust-region pressure; it should return to its ~0.07
  baseline under the fix (ratios start at 1 again).

When no exploration distortion is active, w = 1 and ρ = r: the objective is
bit-for-bit the classic one. All 155 tests pass, including the pre-existing
rare-action overflow test.

## 7. Cleanup

The ~320 iterations trained under the flawed objective (9583–9899) carried
measurably distorted weights, so rather than pay an unknown recovery cost we
rolled back: the 12 affected checkpoints were deleted from the Modal volume
(plus 2 pulled local copies), metrics.csv / events.jsonl were truncated to
iteration ≤ 9574, and training resumed from the last healthy checkpoint
(iter 9574) with the corrected objective and the same sampling regime —
weighted cells, temperatures, no heuristic arm, 30% historical — all of
which were and remain sound.

## 8. The general lesson

Importance sampling makes the *estimator* unbiased, but PPO's clip is not an
estimator — it is a constraint, and a constraint is defined relative to a
reference point. Any exploration scheme that moves the behavior policy far
from the trained policy must keep the clip centered on π_old and push the
behavior correction outside the clip. "Record the behavior log-prob" is
necessary but not sufficient: the mixture being *close to* the policy was a
hidden assumption of the classic ratio, and turning up the temperature is
exactly the act that voids it.
