"""Train a schema-v5 Plump agent with information-set expert iteration."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plump.evaluation import DealBank, evaluate_policy
from plump.information_search import (
    InformationSearchConfig,
    InformationSearchPolicy,
)
from plump.modeling import ModelConfig, PlumpSearchModel
from plump.policies import HeuristicPolicy, ModelPolicy
from plump.training import (
    ExpertIterationConfig,
    ExpertIterationTrainer,
    ExpertRunLogger,
    OpponentMix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the schema-v5 information-set expert-iteration agent."
        )
    )
    parser.add_argument("--cycles", type=int, default=2_500)
    parser.add_argument("--player-counts", default="3,4,5")
    parser.add_argument(
        "--hand-sizes",
        default="3,4,5,6,7,8,9,10",
    )
    parser.add_argument("--rounds-per-configuration", type=int, default=16)
    parser.add_argument("--games-per-player-seat", type=int, default=4)
    parser.add_argument(
        "--training-mode",
        choices=("round", "game"),
        default="round",
    )
    parser.add_argument("--game-schedule", default="")
    parser.add_argument("--min-cards", type=int, default=3)
    parser.add_argument("--max-cards", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--minibatch-size", type=int, default=1_440)
    parser.add_argument("--microbatch-size", type=int, default=576)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--replay-max-age", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--self-play-fraction", type=float, default=0.3)
    parser.add_argument("--heuristic-fraction", type=float, default=0.3)
    parser.add_argument("--mixed-fraction", type=float, default=0.3)
    parser.add_argument("--historical-fraction", type=float, default=0.1)
    parser.add_argument("--historical-checkpoint", action="append", default=[])
    parser.add_argument("--historical-max-snapshots", type=int, default=4)
    parser.add_argument("--concurrent-episodes", type=int, default=32)
    parser.add_argument("--play-search-fraction", type=float, default=1.0)
    parser.add_argument("--search-min-worlds", type=int, default=4)
    parser.add_argument("--search-max-worlds", type=int, default=32)
    parser.add_argument("--search-node-budget", type=int, default=65_536)
    parser.add_argument("--search-maximum-js", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--initialize-game-from",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--initialize-from-v4",
        type=Path,
        default=None,
    )
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument(
        "--eval-deals-per-configuration",
        type=int,
        default=4,
    )
    parser.add_argument("--eval-bootstrap-samples", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=320)
    parser.add_argument("--teacher-eval-every", type=int, default=100)
    parser.add_argument(
        "--teacher-eval-deals-per-configuration",
        type=int,
        default=1,
    )
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-every-minutes", type=float, default=30.0)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/v5"),
    )
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--diag-samples", type=int, default=2_048)
    parser.add_argument("--diag-batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=704)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=11)
    parser.add_argument("--d-ff", type=int, default=2_560)
    parser.add_argument("--context-hidden-dim", type=int, default=512)
    parser.add_argument("--owner-sinkhorn-iterations", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    )


def main() -> None:
    args = parse_args()
    initialization_flags = [
        args.resume_from,
        args.initialize_game_from,
        args.initialize_from_v4,
    ]
    if sum(flag is not None for flag in initialization_flags) > 1:
        raise ValueError(
            "--resume-from, --initialize-game-from, and --initialize-from-v4 "
            "are mutually exclusive."
        )
    if (
        args.initialize_game_from
        and args.training_mode != "game"
    ):
        raise ValueError(
            "--initialize-game-from requires --training-mode game."
        )
    model_config = ModelConfig(
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        context_hidden_dim=args.context_hidden_dim,
        owner_sinkhorn_iterations=args.owner_sinkhorn_iterations,
        dropout=args.dropout,
    )
    search_config = InformationSearchConfig(
        min_determinizations=args.search_min_worlds,
        max_determinizations=args.search_max_worlds,
        node_budget=args.search_node_budget,
        maximum_target_js=args.search_maximum_js,
        seed=args.seed + 10_000,
    )
    config = ExpertIterationConfig(
        player_counts=_csv_ints(args.player_counts),
        hand_sizes=_csv_ints(args.hand_sizes),
        rounds_per_configuration=args.rounds_per_configuration,
        games_per_player_seat=args.games_per_player_seat,
        concurrent_episodes=args.concurrent_episodes,
        play_search_fraction=args.play_search_fraction,
        training_mode=args.training_mode,
        game_schedule=_csv_ints(args.game_schedule),
        min_cards=args.min_cards,
        max_cards=args.max_cards,
        learning_rate=args.lr,
        minibatch_size=args.minibatch_size,
        microbatch_size=args.microbatch_size,
        replay_capacity=args.replay_capacity,
        replay_max_age=args.replay_max_age,
        max_grad_norm=args.max_grad_norm,
        opponent_mix=OpponentMix(
            args.self_play_fraction,
            args.heuristic_fraction,
            args.mixed_fraction,
            args.historical_fraction,
        ),
        historical_checkpoint_paths=tuple(
            args.historical_checkpoint
        ),
        historical_max_snapshots=args.historical_max_snapshots,
        precision=args.precision,
        seed=args.seed,
        device=args.device,
        model_config=model_config,
        search_config=search_config,
    )
    trainer = ExpertIterationTrainer(
        PlumpSearchModel(model_config),
        config,
    )
    resume = (
        trainer.load_checkpoint(args.resume_from)
        if args.resume_from
        else None
    )
    if (
        args.resume_from
        and str(args.resume_from) not in trainer.historical_paths
    ):
        trainer.add_historical_checkpoint(args.resume_from)
    initialization = (
        trainer.initialize_game_from_round_checkpoint(
            args.initialize_game_from
        )
        if args.initialize_game_from
        else (
            trainer.initialize_from_v4_checkpoint(args.initialize_from_v4)
            if args.initialize_from_v4
            else None
        )
    )
    start_cycle = int(resume["cycle"]) if resume else 0
    if start_cycle >= args.cycles:
        raise ValueError(
            f"Checkpoint cycle {start_cycle} already reached "
            f"--cycles {args.cycles}."
        )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or args.checkpoint_dir
    logger = ExpertRunLogger(log_dir)
    logger.write_config(
        {
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
            "args": vars(args),
            "device": str(trainer.device),
            "parameter_count": sum(
                parameter.numel()
                for parameter in trainer.model.parameters()
            ),
            "random_initialization": (
                resume is None and initialization is None
            ),
            "resume": resume,
            "game_initialization": initialization,
            "training_config": asdict(config),
            "mps_environment": {
                name: os.environ.get(name)
                for name in (
                    "PYTORCH_MPS_FAST_MATH",
                    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
                    "PYTORCH_MPS_PREFER_METAL",
                )
            },
        }
    )
    print(
        f"device={trainer.device} schema=5 observation_schema=4 "
        f"parameters={sum(p.numel() for p in trainer.model.parameters())} "
        f"start_cycle={start_cycle} mode={args.training_mode} "
        f"log_dir={log_dir}",
        flush=True,
    )

    eval_bank = DealBank.generate(
        player_counts=config.player_counts,
        hand_sizes=config.hand_sizes,
        deals_per_configuration=args.eval_deals_per_configuration,
        seed=args.seed + 20_000,
    )
    teacher_bank = DealBank.generate(
        player_counts=config.player_counts,
        hand_sizes=config.hand_sizes,
        deals_per_configuration=(
            args.teacher_eval_deals_per_configuration
        ),
        seed=args.seed + 30_000,
    )
    eval_opponent = HeuristicPolicy()
    best_raw = float("-inf")
    last_checkpoint_at = time.monotonic()

    for cycle_index in range(start_cycle + 1, args.cycles + 1):
        print(f"cycle={cycle_index} generation_start", flush=True)

        def progress(completed: int, total: int) -> None:
            if completed == total or completed % 16 == 0:
                print(
                    f"cycle={cycle_index} generated_rounds="
                    f"{completed}/{total}",
                    flush=True,
                )

        cycle = trainer.collect_cycle(
            cycle=cycle_index,
            progress_callback=progress,
        )
        trainer.add_cycle(cycle, cycle_index=cycle_index)
        update = trainer.update(
            new_state_count=len(cycle.samples)
        )
        diagnostics = trainer.diagnostics(
            cycle,
            max_samples=args.diag_samples,
            batch_size=args.diag_batch_size,
        )
        trainer.search_schedule.update(diagnostics)
        print(
            f"cycle={cycle_index} states={len(cycle.samples)} "
            f"accepted={diagnostics.accepted_rate:.3f} "
            f"policy_ce={update.policy_loss:.4f} "
            f"q_loss={update.q_loss:.4f} "
            f"value_loss={update.value_loss:.4f} "
            f"q_ev={diagnostics.q_explained_variance:.3f} "
            f"value_ev={diagnostics.value_explained_variance:.3f}",
            flush=True,
        )

        raw_evaluation = None
        if (
            args.eval_every > 0
            and cycle_index % args.eval_every == 0
        ):
            raw_policy = ModelPolicy(
                trainer.model,
                device=trainer.device,
                greedy=True,
                include_game_context=(
                    args.training_mode == "game"
                ),
                precision=args.precision,
                name=f"v5-raw-{cycle_index}",
            )
            raw_evaluation = evaluate_policy(
                raw_policy,
                eval_opponent,
                eval_bank,
                bootstrap_samples=args.eval_bootstrap_samples,
                seed=args.seed + cycle_index,
                batch_size=args.eval_batch_size,
            )
            print(
                f"cycle={cycle_index} raw_eval="
                f"{raw_evaluation.macro_relative_reward:.4f} "
                f"ci=[{raw_evaluation.relative_reward_ci_low:.4f},"
                f"{raw_evaluation.relative_reward_ci_high:.4f}]",
                flush=True,
            )

        teacher_evaluation = None
        if (
            args.training_mode == "round"
            and args.teacher_eval_every > 0
            and cycle_index % args.teacher_eval_every == 0
        ):
            raw_for_search = ModelPolicy(
                trainer.model,
                device=trainer.device,
                greedy=False,
                precision=args.precision,
                name=f"v5-teacher-base-{cycle_index}",
            )
            teacher = InformationSearchPolicy(
                raw_for_search,
                eval_opponent,
                config=replace_search_schedule(
                    config.search_config,
                    trainer,
                ),
                belief_weight=trainer.search_schedule.owner_weight(),
                leaf_value_weight=max(
                    trainer.search_schedule.leaf_weight("bid"),
                    trainer.search_schedule.leaf_weight("play"),
                ),
                name=f"v5-search-teacher-{cycle_index}",
            )
            teacher_evaluation = evaluate_policy(
                teacher,
                eval_opponent,
                teacher_bank,
                bootstrap_samples=args.eval_bootstrap_samples,
                seed=args.seed + 50_000 + cycle_index,
                batch_size=1,
            )
            print(
                f"cycle={cycle_index} teacher_eval="
                f"{teacher_evaluation.macro_relative_reward:.4f}",
                flush=True,
            )

        initial_due = cycle_index == start_cycle + 1
        cycle_due = (
            args.save_every > 0
            and cycle_index % args.save_every == 0
        )
        wall_due = (
            args.save_every_minutes > 0
            and time.monotonic() - last_checkpoint_at
            >= args.save_every_minutes * 60.0
        )
        checkpoint_path = None
        if initial_due or cycle_due or wall_due:
            checkpoint_path = (
                args.checkpoint_dir
                / f"plump_v5_cycle_{cycle_index:05d}.pt"
            )
            trainer.save_checkpoint(
                checkpoint_path,
                cycle=cycle_index,
                extra={
                    "raw_evaluation": raw_evaluation,
                    "teacher_evaluation": teacher_evaluation,
                },
            )
            trainer.add_current_historical_snapshot(checkpoint_path)
            last_checkpoint_at = time.monotonic()
        if (
            raw_evaluation is not None
            and raw_evaluation.macro_relative_reward > best_raw
        ):
            best_raw = raw_evaluation.macro_relative_reward
            trainer.save_checkpoint(
                args.checkpoint_dir / "best.pt",
                cycle=cycle_index,
                extra={"raw_evaluation": raw_evaluation},
            )
        logger.log_cycle(
            cycle_index=cycle_index,
            cycle=cycle,
            replay_states=len(trainer.replay),
            update=update,
            diagnostics=diagnostics,
            raw_evaluation=raw_evaluation,
            teacher_evaluation=teacher_evaluation,
            checkpoint_path=checkpoint_path,
        )
        print(f"cycle={cycle_index} complete", flush=True)


def replace_search_schedule(
    config: InformationSearchConfig,
    trainer: ExpertIterationTrainer,
) -> InformationSearchConfig:
    from dataclasses import replace
    return replace(
        config,
        internal_temperature=min(
            trainer.search_schedule.temperature("bid"),
            trainer.search_schedule.temperature("play"),
        ),
        target_temperature=min(
            trainer.search_schedule.temperature("bid"),
            trainer.search_schedule.temperature("play"),
        ),
    )


if __name__ == "__main__":
    main()
