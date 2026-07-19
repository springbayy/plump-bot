"""Checkpoint-backed PPO throughput benchmark with no training side effects.

The script intentionally bypasses evaluation, diagnostics, plotting, league
payoff refreshes, and checkpoint writes. Each process loads the same schema-v4
checkpoint (including optimizer and league state), performs optional warm-up
cycles, and reports synchronized collect/update wall times plus MPS memory.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plump.modeling import ModelConfig
from plump.modeling.torch_model import PlumpTransformerModel
from plump.training import PPOTrainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--packing", choices=("torch", "numpy"), default="numpy")
    parser.add_argument("--microbatch-size", type=int, default=480)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--memory-watermark", type=float, default=0.95)
    parser.add_argument("--require-memory-watermark", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


class MemoryMonitor:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.peak_current = 0
        self.peak_driver = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "MemoryMonitor":
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            self.sample()

    def sample(self) -> None:
        if self.device.type == "mps":
            self.peak_current = max(
                self.peak_current,
                int(torch.mps.current_allocated_memory()),
            )
            self.peak_driver = max(
                self.peak_driver,
                int(torch.mps.driver_allocated_memory()),
            )
        elif self.device.type == "cuda":
            self.peak_current = max(
                self.peak_current,
                int(torch.cuda.max_memory_allocated(self.device)),
            )
            self.peak_driver = self.peak_current


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def checkpoint_config(
    checkpoint: Path,
    *,
    args: argparse.Namespace,
) -> tuple[ModelConfig, TrainingConfig]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**payload["model_config"])
    stored = payload.get("training_config", {})
    defaults = TrainingConfig(model_config=model_config)
    values = {
        field.name: stored.get(field.name, getattr(defaults, field.name))
        for field in fields(TrainingConfig)
        if field.name != "model_config"
    }
    optimized = args.mode == "optimized"
    values.update(
        {
            "player_counts": (3, 4, 5),
            "hand_sizes": tuple(range(3, 11)),
            "rounds_per_configuration": 16,
            "num_envs": 384,
            "ppo_epochs": 4,
            "target_kl": 0.02,
            "pipeline_rollouts": False,
            "env_workers": 0,
            "event_length_buckets": (8, 16, 32, 64) if optimized else (),
            "batch_packing": args.packing if optimized else "torch",
            "lean_rollout_forward": optimized,
            "batched_league_sampling": optimized,
            "league_probe_fraction": 0.10,
            "minibatch_size": 1440,
            "microbatch_size": args.microbatch_size,
            "league_eval_every": 0,
            "seed": args.seed,
            "device": args.device,
            "model_config": model_config,
        }
    )
    del payload
    return model_config, TrainingConfig(**values)


def finite_update(update) -> bool:
    return all(
        math.isfinite(float(value))
        for value in asdict(update).values()
        if isinstance(value, float)
    )


def main() -> None:
    args = parse_args()
    model_config, training_config = checkpoint_config(args.checkpoint, args=args)
    trainer = PPOTrainer(
        PlumpTransformerModel(model_config),
        training_config,
    )
    resume = trainer.load_checkpoint(args.checkpoint, load_optimizer=True)
    if not resume["optimizer_loaded"]:
        raise RuntimeError("Optimizer state did not resume.")
    if resume["league_snapshots_missing"]:
        raise RuntimeError(
            f"Missing league snapshots: {resume['league_snapshots_missing']}"
        )

    device = trainer.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_cycles = args.warmup + args.measured
    measured_rows = []
    all_rows = []
    with MemoryMonitor(device) as memory:
        for cycle in range(total_cycles):
            iteration = int(resume["iteration"] or 0) + cycle + 1
            synchronize(device)
            started = time.perf_counter()
            buffer = trainer.collect_rollouts(iteration=iteration)
            collect_sec = trainer.last_collect_sec
            update = trainer.update(buffer)
            synchronize(device)
            wall_sec = time.perf_counter() - started
            memory.sample()
            row = {
                "cycle": cycle + 1,
                "iteration": iteration,
                "warmup": cycle < args.warmup,
                "wall_sec": wall_sec,
                "collect_sec": collect_sec,
                "update_sec": wall_sec - collect_sec,
                "rollout_rounds": len(buffer.round_outcomes),
                "samples": len(buffer.ready_samples()),
                "finite": finite_update(update),
                "skipped_steps": update.skipped_steps,
                "epochs_run": update.epochs_run,
                "approx_kl": update.approx_kl,
                "total_loss": update.total_loss,
                "collection": asdict(trainer.last_collection_stats),
                "peak_current_memory_bytes": memory.peak_current,
                "peak_driver_memory_bytes": memory.peak_driver,
            }
            if row["rollout_rounds"] != training_config.rounds_per_batch:
                raise RuntimeError(
                    "Rollout count changed: "
                    f"{row['rollout_rounds']} != {training_config.rounds_per_batch}"
                )
            if not row["finite"] or row["skipped_steps"] != 0:
                raise RuntimeError(f"Invalid update: {row}")
            all_rows.append(row)
            if not row["warmup"]:
                measured_rows.append(row)
            print(json.dumps({"type": "cycle", **row}, sort_keys=True), flush=True)

    recommended = (
        int(torch.mps.recommended_max_memory())
        if device.type == "mps"
        else 0
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "resume": resume,
        "mode": args.mode,
        "packing": training_config.batch_packing,
        "microbatch_size": args.microbatch_size,
        "warmup_cycles": args.warmup,
        "measured_cycles": args.measured,
        "median_wall_sec": statistics.median(
            row["wall_sec"] for row in measured_rows
        ),
        "median_collect_sec": statistics.median(
            row["collect_sec"] for row in measured_rows
        ),
        "median_update_sec": statistics.median(
            row["update_sec"] for row in measured_rows
        ),
        "peak_current_memory_bytes": memory.peak_current,
        "peak_driver_memory_bytes": memory.peak_driver,
        "recommended_max_memory_bytes": recommended,
        "memory_watermark": args.memory_watermark,
        "under_memory_watermark": (
            not recommended
            or memory.peak_driver <= args.memory_watermark * recommended
        ),
        "rows": all_rows,
    }
    print(json.dumps({"type": "result", **result}, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.require_memory_watermark and not result["under_memory_watermark"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
