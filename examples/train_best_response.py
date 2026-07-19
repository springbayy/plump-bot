"""Train an approximate best response against one frozen candidate checkpoint.

Every opponent seat is pinned to the candidate; a fresh (smaller) model is
trained with PPO purely to beat it. The final macro relative reward of the
best response versus the candidate is an exploitability proxy: around zero
means this training budget found no reliable counter-strategy, clearly
positive means the candidate is exploitable by at least that margin.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plump.evaluation import DealBank, evaluate_policy
from plump.modeling import ModelConfig
from plump.modeling.torch_model import PlumpTransformerModel
from plump.policies import ModelPolicy
from plump.training import PPOTrainer, TrainingConfig, format_update_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approximate best-response exploitability probe.",
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--player-counts", default="3,4,5")
    parser.add_argument("--hand-sizes", default="3,4,5,6,7,8,9,10")
    parser.add_argument("--rounds-per-configuration", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=384)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--minibatch-size", type=int, default=1440)
    parser.add_argument("--microbatch-size", type=int, default=480)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--trick-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--max-seq-len", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--context-hidden-dim", type=int, default=256)
    parser.add_argument("--eval-deals-per-configuration", type=int, default=8)
    parser.add_argument("--eval-bootstrap-samples", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=320)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Where to write the JSON report (default: <candidate dir>/best_response_report.json).",
    )
    return parser.parse_args()


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    args = parse_args()
    if not args.candidate.exists():
        raise SystemExit(f"Candidate checkpoint not found: {args.candidate}")
    player_counts = _csv_ints(args.player_counts)
    hand_sizes = _csv_ints(args.hand_sizes)
    model_config = ModelConfig(
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        context_hidden_dim=args.context_hidden_dim,
    )
    config = TrainingConfig(
        player_counts=player_counts,
        hand_sizes=hand_sizes,
        rounds_per_configuration=args.rounds_per_configuration,
        num_envs=args.num_envs,
        ppo_epochs=args.ppo_epochs,
        target_kl=args.target_kl,
        minibatch_size=args.minibatch_size,
        microbatch_size=args.microbatch_size or None,
        learning_rate=args.lr,
        # Every opponent seat is the frozen candidate: no self-play, no
        # heuristic, no mixing, and a pool capped at the single snapshot.
        self_play_fraction=0.0,
        heuristic_fraction=0.0,
        mixed_fraction=0.0,
        historical_fraction=1.0,
        historical_max_snapshots=1,
        league_meta_solver="softmax_ema",
        trick_baseline=args.trick_baseline,
        precision=args.precision,
        device=args.device,
        seed=args.seed,
        model_config=model_config,
    )
    trainer = PPOTrainer(PlumpTransformerModel(model_config), config)
    trainer.add_historical_checkpoint(args.candidate)
    candidate_id = args.candidate.stem
    parameter_count = sum(p.numel() for p in trainer.model.parameters())
    print(
        f"best-response probe candidate={args.candidate} device={trainer.device} "
        f"br_parameters={parameter_count} iterations={args.iterations}"
    )

    started_at = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        buffer = trainer.collect_rollouts(iteration=iteration)
        update_stats = trainer.update(buffer)
        if iteration % args.log_every == 0:
            rewards = [
                outcome.focal_reward
                for outcome in buffer.round_outcomes
                if outcome.focal_reward is not None
            ]
            batch_reward = sum(rewards) / len(rewards) if rewards else 0.0
            ema = trainer.league_reward_ema.get(candidate_id, batch_reward)
            print(
                f"iter={iteration} vs_candidate={batch_reward:.4f} "
                f"vs_candidate_ema={ema:.4f} {format_update_stats(update_stats)}"
            )

    bank = DealBank.generate(
        player_counts=player_counts,
        hand_sizes=hand_sizes,
        deals_per_configuration=args.eval_deals_per_configuration,
        seed=args.seed + 20_000,
    )
    best_response = ModelPolicy(
        trainer.model,
        device=trainer.device,
        greedy=True,
        precision=args.precision,
        name="best-response",
    )
    candidate_policy = ModelPolicy.from_checkpoint(
        args.candidate,
        device=trainer.device,
        greedy=True,
    )
    evaluation = evaluate_policy(
        best_response,
        candidate_policy,
        bank,
        bootstrap_samples=args.eval_bootstrap_samples,
        seed=args.seed + 30_000,
        batch_size=args.eval_batch_size,
    )
    print(
        f"exploitability_proxy={evaluation.macro_relative_reward:.4f} "
        f"ci=[{evaluation.relative_reward_ci_low:.4f},"
        f"{evaluation.relative_reward_ci_high:.4f}] "
        f"rounds={evaluation.rounds} "
        f"elapsed_sec={time.perf_counter() - started_at:.0f}"
    )
    report_path = args.report or (
        args.candidate.parent / "best_response_report.json"
    )
    report_path.write_text(
        json.dumps(
            {
                "candidate": str(args.candidate),
                "br_iterations": args.iterations,
                "br_parameters": parameter_count,
                "exploitability_proxy": evaluation.macro_relative_reward,
                "ci_low": evaluation.relative_reward_ci_low,
                "ci_high": evaluation.relative_reward_ci_high,
                "rounds": evaluation.rounds,
                "final_ema_vs_candidate": trainer.league_reward_ema.get(candidate_id),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
