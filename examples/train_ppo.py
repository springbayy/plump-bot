"""Train the schema-v4 position-robust Plump policy."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plump.evaluation import DealBank, evaluate_policy
from plump.modeling import ModelConfig
from plump.modeling.torch_model import PlumpTransformerModel
from plump.policies import HeuristicPolicy, ModelPolicy, RandomPolicy
from plump.training import (
    CounterfactualSearchRouter,
    PPOTrainer,
    SearchTrustRegionUpdater,
    TrainingConfig,
    TrainingRunLogger,
    format_update_stats,
    training_config_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the schema-v4 balanced Plump agent.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--player-counts", default="3,4,5")
    parser.add_argument("--hand-sizes", default="3,4,5,6,7,8,9,10")
    parser.add_argument("--rounds-per-configuration", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=384)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1440)
    parser.add_argument("--microbatch-size", type=int, default=576)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--ppo-clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--trick-coef", type=float, default=0.1)
    parser.add_argument("--owner-coef", type=float, default=0.05)
    parser.add_argument("--owner-capacity-coef", type=float, default=0.1)
    parser.add_argument("--position-baseline-decay", type=float, default=0.98)
    parser.add_argument("--self-play-fraction", type=float, default=0.3)
    parser.add_argument("--heuristic-fraction", type=float, default=0.3)
    parser.add_argument("--mixed-fraction", type=float, default=0.3)
    parser.add_argument("--historical-fraction", type=float, default=0.1)
    parser.add_argument("--historical-checkpoint", action="append", default=[])
    parser.add_argument("--historical-max-snapshots", type=int, default=4)
    parser.add_argument(
        "--historical-current-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--warm-start-v2", type=Path, default=None)
    parser.add_argument("--warm-start-v3", type=Path, default=None)
    parser.add_argument("--trick-baseline", action="store_true")
    parser.add_argument("--training-mode", choices=("round", "game"), default="round")
    parser.add_argument("--game-schedule", default="")
    parser.add_argument("--min-cards", type=int, default=3)
    parser.add_argument("--max-cards", type=int, default=10)
    parser.add_argument("--games-per-player-seat", type=int, default=4)
    parser.add_argument(
        "--counterfactual-search",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--search-min-iteration", type=int, default=250)
    parser.add_argument("--search-ev-threshold", type=float, default=0.30)
    parser.add_argument("--search-states-per-phase", type=int, default=24)
    parser.add_argument("--search-replay-capacity", type=int, default=50_000)
    parser.add_argument("--search-replay-max-age", type=int, default=250)
    parser.add_argument("--search-lr", type=float, default=1e-4)
    parser.add_argument("--search-minibatch-size", type=int, default=256)
    parser.add_argument("--search-entropy-floor-coef", type=float, default=0.002)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-deals-per-configuration", type=int, default=4)
    parser.add_argument("--eval-bootstrap-samples", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=384)
    parser.add_argument("--eval-opponent", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-every-minutes", type=float, default=30.0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/v4"))
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=704)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=11)
    parser.add_argument("--d-ff", type=int, default=2560)
    parser.add_argument("--context-hidden-dim", type=int, default=512)
    parser.add_argument("--owner-sinkhorn-iterations", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--diag-every", type=int, default=5)
    parser.add_argument("--diag-samples", type=int, default=2048)
    parser.add_argument("--diag-batch-size", type=int, default=256)
    return parser.parse_args()


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    args = parse_args()
    player_counts = _csv_ints(args.player_counts)
    hand_sizes = _csv_ints(args.hand_sizes)
    game_schedule = _csv_ints(args.game_schedule)
    initialization_paths = [
        args.resume_from,
        args.warm_start_v2,
        args.warm_start_v3,
    ]
    if sum(path is not None for path in initialization_paths) > 1:
        raise ValueError(
            "--resume-from, --warm-start-v2, and --warm-start-v3 "
            "are mutually exclusive."
        )
    search_enabled = (
        args.counterfactual_search
        and args.training_mode == "round"
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
    train_config = TrainingConfig(
        player_counts=player_counts,
        hand_sizes=hand_sizes,
        rounds_per_configuration=args.rounds_per_configuration,
        games_per_player_seat=args.games_per_player_seat,
        num_envs=args.num_envs,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        microbatch_size=args.microbatch_size or None,
        learning_rate=args.lr,
        ppo_clip_eps=args.ppo_clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        trick_coef=args.trick_coef,
        owner_coef=args.owner_coef,
        owner_capacity_coef=args.owner_capacity_coef,
        max_grad_norm=args.max_grad_norm,
        position_baseline_decay=args.position_baseline_decay,
        self_play_fraction=args.self_play_fraction,
        heuristic_fraction=args.heuristic_fraction,
        mixed_fraction=args.mixed_fraction,
        historical_fraction=args.historical_fraction,
        historical_checkpoint_paths=tuple(args.historical_checkpoint),
        historical_max_snapshots=args.historical_max_snapshots,
        trick_baseline=args.trick_baseline,
        training_mode=args.training_mode,
        game_schedule=game_schedule,
        min_cards=args.min_cards,
        max_cards=args.max_cards,
        precision=args.precision,
        seed=args.seed,
        device=args.device,
        model_config=model_config,
    )
    trainer = PPOTrainer(PlumpTransformerModel(model_config), train_config)
    resume_info = (
        trainer.load_checkpoint(args.resume_from, load_optimizer=args.resume_optimizer)
        if args.resume_from
        else None
    )
    warm_start_info = (
        trainer.warm_start_v2(args.warm_start_v2)
        if args.warm_start_v2
        else (
            trainer.warm_start_v3(args.warm_start_v3)
            if args.warm_start_v3
            else None
        )
    )
    start_iteration = int(resume_info["iteration"] or 0) if resume_info else 0
    if start_iteration >= args.iterations:
        raise ValueError(
            f"Checkpoint iteration {start_iteration} has already reached "
            f"the requested total of {args.iterations} iterations."
        )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or args.checkpoint_dir
    logger = TrainingRunLogger(log_dir)
    search_router = (
        CounterfactualSearchRouter(
            trainer.model,
            device=trainer.device,
            precision=args.precision,
            minimum_iteration=args.search_min_iteration,
            explained_variance_threshold=args.search_ev_threshold,
            states_per_phase=args.search_states_per_phase,
            replay_capacity=args.search_replay_capacity,
            replay_max_age=args.search_replay_max_age,
            seed=args.seed + 40_000,
        )
        if search_enabled
        else None
    )
    search_updater = (
        SearchTrustRegionUpdater(
            trainer.model,
            device=trainer.device,
            learning_rate=args.search_lr,
            minibatch_size=args.search_minibatch_size,
            max_grad_norm=args.max_grad_norm,
            entropy_floor_coef=args.search_entropy_floor_coef,
            seed=args.seed + 50_000,
        )
        if search_enabled
        else None
    )
    logger.write_config(
        {
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
            "args": vars(args),
            "device": str(trainer.device),
            "parameter_count": sum(parameter.numel() for parameter in trainer.model.parameters()),
            "resume": resume_info,
            "warm_start": warm_start_info,
            "counterfactual_search_enabled": search_enabled,
            "mps_environment": {
                key: os.environ.get(key)
                for key in (
                    "PYTORCH_MPS_FAST_MATH",
                    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
                    "PYTORCH_MPS_PREFER_METAL",
                )
            },
            "training_config": training_config_snapshot(train_config),
        }
    )
    print(
        f"device={trainer.device} configurations={len(train_config.specs)} "
        f"rounds_per_batch={train_config.rounds_per_batch} "
        f"precision={train_config.precision} start_iteration={start_iteration} "
        f"search={search_enabled} log_dir={log_dir}"
    )

    eval_bank = DealBank.generate(
        player_counts=player_counts,
        hand_sizes=hand_sizes,
        deals_per_configuration=args.eval_deals_per_configuration,
        seed=args.seed + 20_000,
    )
    eval_opponent = HeuristicPolicy() if args.eval_opponent == "heuristic" else RandomPolicy(args.seed + 30_000)
    best_eval = float("-inf")
    started_at = time.perf_counter()
    last_checkpoint_at = started_at

    for iteration in range(start_iteration + 1, args.iterations + 1):
        iteration_start = time.perf_counter()
        collect_start = time.perf_counter()
        buffer = trainer.collect_rollouts(iteration=iteration)
        collect_sec = time.perf_counter() - collect_start
        rollout_stats = trainer.summarize_rollout(buffer)

        search_routing = (
            search_router.route(buffer, iteration=iteration)
            if search_router is not None
            else []
        )
        update_start = time.perf_counter()
        update_stats = trainer.update(buffer)
        search_updates = []
        if search_router is not None and search_updater is not None:
            for phase in ("bid", "play"):
                search_updates.append(
                    search_updater.update(
                        search_router.replay,
                        phase=phase,
                        iteration=iteration,
                        regret_matching_fraction=(
                            search_router.regret_matching_fraction(phase)
                        ),
                    )
                )
        update_sec = time.perf_counter() - update_start

        diagnostics_sec = 0.0
        prediction_stats = None
        if args.diag_every > 0 and iteration % args.diag_every == 0:
            diagnostics_start = time.perf_counter()
            if trainer.device.type == "mps":
                torch.mps.empty_cache()
            prediction_stats = trainer.compute_prediction_stats(
                buffer,
                max_samples=args.diag_samples,
                minibatch_size=args.diag_batch_size,
            )
            if search_router is not None:
                search_router.update_diagnostics(
                    bid_explained_variance=(
                        prediction_stats.bid_value_explained_variance
                    ),
                    play_explained_variance=(
                        prediction_stats.play_value_explained_variance
                    ),
                )
            diagnostics_sec = time.perf_counter() - diagnostics_start

        print(
            f"iter={iteration} bid_hit={rollout_stats.bid_hit_rate:.4f} "
            f"{format_update_stats(update_stats)}"
        )
        for routing in search_routing:
            if routing.eligible or routing.gate_passed:
                print(
                    f"search phase={routing.phase} probed={routing.probed} "
                    f"accepted={routing.accepted_rate:.3f} "
                    f"ci_low={routing.paired_ci_low:.4f} "
                    f"sampler_reject={routing.sampler_infeasible_rejection_rate:.4f} "
                    f"gate={routing.gate_passed} routed={routing.routed}"
                )

        eval_sec = 0.0
        evaluation = None
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            eval_start = time.perf_counter()
            if trainer.device.type == "mps":
                torch.mps.empty_cache()
            candidate = ModelPolicy(
                trainer.model,
                device=trainer.device,
                greedy=True,
                precision=args.precision,
                name=f"iteration-{iteration}",
            )
            evaluation = evaluate_policy(
                candidate,
                eval_opponent,
                eval_bank,
                bootstrap_samples=args.eval_bootstrap_samples,
                seed=args.seed + iteration,
                batch_size=args.eval_batch_size,
            )
            eval_sec = time.perf_counter() - eval_start
            print(
                f"eval rounds={evaluation.rounds} macro_rel={evaluation.macro_relative_reward:.4f} "
                f"ci=[{evaluation.relative_reward_ci_low:.4f},{evaluation.relative_reward_ci_high:.4f}] "
                f"bid_hit={evaluation.macro_bid_hit_rate:.4f}"
            )
            if evaluation.macro_relative_reward > best_eval:
                best_eval = evaluation.macro_relative_reward
                best_path = args.checkpoint_dir / "best.pt"
                trainer.save_checkpoint(best_path, iteration=iteration, extra={"evaluation": evaluation})

        checkpoint_sec = 0.0
        checkpoint_path = None
        initial_checkpoint_due = iteration == start_iteration + 1
        iteration_checkpoint_due = args.save_every > 0 and iteration % args.save_every == 0
        wall_checkpoint_due = (
            args.save_every_minutes > 0
            and time.perf_counter() - last_checkpoint_at >= args.save_every_minutes * 60.0
        )
        if initial_checkpoint_due or iteration_checkpoint_due or wall_checkpoint_due:
            checkpoint_start = time.perf_counter()
            checkpoint_path = args.checkpoint_dir / f"plump_v4_iter_{iteration:05d}.pt"
            trainer.save_checkpoint(checkpoint_path, iteration=iteration, extra={"evaluation": evaluation})
            if search_router is not None:
                search_router.replay.save(
                    args.checkpoint_dir / "search_replay.pt",
                    gate_report_path=logger.events_path,
                )
            if args.historical_current_snapshots:
                trainer.add_historical_checkpoint(checkpoint_path)
            checkpoint_sec = time.perf_counter() - checkpoint_start
            last_checkpoint_at = time.perf_counter()

        timings = {
            "collect_sec": collect_sec,
            "update_sec": update_sec,
            "diagnostics_sec": diagnostics_sec,
            "eval_sec": eval_sec,
            "checkpoint_sec": checkpoint_sec,
            "iteration_sec": time.perf_counter() - iteration_start,
        }
        logger.log_iteration(
            iteration=iteration,
            elapsed_sec=time.perf_counter() - started_at,
            timings=timings,
            update=update_stats,
            rollout=rollout_stats,
            prediction=prediction_stats,
            evaluation=evaluation,
            checkpoint_path=checkpoint_path,
            search_routing=search_routing,
            search_updates=search_updates,
        )


if __name__ == "__main__":
    main()
