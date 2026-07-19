"""Modal app: continue Plump PPO training on a cloud GPU, fire-and-forget.

Workflow:
    modal deploy modal_train.py                     # once, and after edits
    modal run modal_train.py::kickoff               # spawn detached training
    modal run modal_train.py::status                # latest.json summary
    modal run modal_train.py::stop                  # graceful stop sentinel
    modal app logs plump-training                   # live stdout

The `train` function supervises `examples/train_ppo.py` as a subprocess
(restart-on-crash, like scripts/run_training_v9_8m_supervised.zsh), always
resuming from the newest checkpoint in the run directory on the shared
volume. Modal caps one function call at 24h, so shortly before the budget
expires the supervisor checkpoints via SIGTERM and respawns itself.
"""

from __future__ import annotations

import io
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import modal

APP_NAME = "plump-training"
VOLUME_NAME = "plump-checkpoints"
VOLUME_MOUNT = "/vol"
REMOTE_REPO = "/root/plump-bot"
DEFAULT_RUN = "v9_8m_wideppo_seed1"

# 24h Modal limit; leave headroom for a graceful SIGTERM + respawn.
FUNCTION_TIMEOUT_SEC = 24 * 60 * 60
SUPERVISOR_BUDGET_SEC = 23 * 60 * 60
MAX_RESTARTS = 50
# A run that dies this quickly is misconfigured, not unlucky.
FAST_FAILURE_SEC = 120.0
MAX_CONSECUTIVE_FAST_FAILURES = 3

# The local v9 command with data/batch sizes scaled 3x for an L40S
# (microbatch == minibatch: no gradient accumulation on 48GB). Model dims
# must stay identical to the checkpoint being resumed.
BASE_TRAIN_ARGS = [
    "--iterations", "12000",
    "--seed", "1",
    "--training-mode", "round",
    "--rounds-per-configuration", "48",
    "--num-envs", "1152",
    "--event-length-buckets", "8,16,32,64",
    "--batch-packing", "numpy",
    "--lean-rollout-forward",
    # Pipeline/env-worker flags exist but measured slower on L40S (GIL eats
    # the overlap; pickle-heavy IPC beats the engine win) — keep sequential.
    "--ppo-epochs", "4",
    "--target-kl", "0.02",
    # Minibatch is a learning-dynamics knob, not a memory knob: 17280 (full
    # batch) cut optimizer steps 20 -> 8 per iteration and collapsed approx_kl
    # from ~0.008 to ~0.0015 at lr 2e-4, stalling eval from iter ~3643 on.
    "--minibatch-size", "4320",
    "--microbatch-size", "4320",
    "--lr", "2e-4",
    # Entropy floor, not an exploration driver (the explore arms handle
    # diversity). The bonus was off from iter ~7.5k; the pre-agreed tripwire
    # (clean-state bid entropy < 0.15) fired at iter ~10.4k with 0.115, so
    # the small floor dose came back. The old 0.01 inflated entropy
    # monotonically — do not raise this without a new reason.
    "--entropy-coef", "0.002",
    "--trick-baseline",
    "--oracle-critic",
    "--oracle-value-coef", "0.5",
    "--suit-presence-head",
    "--suit-coef", "0.1",
    # Owner head retired (iter ~10.4k): activated at 9.7k with a 200-iter
    # trunk-detached warmup, it sat at chance-level opponent-card accuracy
    # (~0.10) for 700 iterations — CE never left the uniform-over-valid
    # level, so the gradient was pure noise. Capacity/void structure already
    # explains everything it managed to learn.
    "--owner-coef", "0.0",
    "--owner-warmup-iters", "200",
    "--owner-capacity-coef", "0.1",
    "--owner-sinkhorn-iterations", "16",
    # Cell weights (since iter ~9590): players 3/4/5 at 2:3:4 and hand sizes
    # 3..10 on a linear 1..8 ramp, so training concentrates on the larger
    # action spaces (more players, more cards) while keeping small rounds in
    # the mix. Joint cell shares range from ~0.6% (3p,3c) to ~9.9% (5p,10c).
    "--player-count-weights", "2,3,4",
    "--hand-size-weights", "1,2,3,4,5,6,7,8",
    # Clean/explore split (since ~iter 9.7k). Clean 60%: pure self-play and
    # vs-historical tables with ZERO noise anywhere — optimization and belief
    # targets come only from non-random play. Explore 40%: the focal seat
    # samples from the tempered+eps behavior policy while every opponent is
    # frozen weight playing raw (explore_self = frozen current weights,
    # explore_historical = league snapshots), so diverse trajectories never
    # mean optimizing against noise. Mixed and heuristic arms parked at 0.
    "--self-play-fraction", "0.30",
    "--heuristic-fraction", "0.0",
    "--mixed-fraction", "0.0",
    "--historical-fraction", "0.30",
    "--explore-self-fraction", "0.20",
    "--explore-historical-fraction", "0.20",
    # Exploration noise lives ONLY on the explore arms' focal seat, and is
    # bounded per round: every focal decision samples tempered (never i.i.d.
    # eps — one round is one deliberate deviation inside otherwise-plausible
    # play, not a random walk), plus AT MOST ONE uniform-random action per
    # round (probability 0.30 at 3 cards, uniformly placed). Both the
    # temperature and the uniform probability shrink by (min_hand+1)/(h+1)
    # on longer rounds so total per-round distortion stays roughly constant:
    # bid T 3.0 -> ~1.73 at 10 cards, play T 2.0 -> ~1.36, uniform prob
    # 0.30 -> ~0.11. Clean arms are exactly on-policy (w = 1, the classic
    # PPO path, bit-for-bit); the decoupled objective (clip rho =
    # pi_new/pi_old, importance weight w = pi_old/b outside the min) keeps
    # the noised rounds unbiased.
    "--explore-eps-bid", "0.0",
    "--explore-eps-play", "0.0",
    "--explore-uniform-round-prob", "0.30",
    "--explore-noise-hand-normalized",
    "--explore-temp-fraction", "1.0",
    "--explore-temp-bid", "3.0",
    "--explore-temp-play", "2.0",
    "--explore-temp-arms", "explore_self,explore_historical",
    "--historical-max-snapshots", "8",
    # Uniform league: pool of 8 resampled uniformly from every checkpoint at
    # or after iter 3000 on each save (~26 iters), so the reward remembers
    # the whole run instead of a ~200-iteration sliding window. Payoff
    # bookkeeping (regret matching, admission evals) is inert under
    # "uniform" and current checkpoints no longer auto-join the pool.
    "--league-meta-solver", "uniform",
    "--league-uniform-min-iteration", "3000",
    "--no-historical-current-snapshots",
    "--league-temperature", "2.0",
    "--league-reward-decay", "0.95",
    "--batched-league-sampling",
    "--league-probe-fraction", "0.10",
    "--league-eval-every", "50",
    "--league-eval-deals-per-configuration", "4",
    "--mmd-enabled",
    "--mmd-coef", "0.05",
    "--no-counterfactual-search",
    "--diag-every", "5",
    "--diag-samples", "2048",
    "--diag-batch-size", "2048",
    # No in-training eval: ~17s every 25 iters bought a metric the ladder
    # measures better offline. Evaluate at your discretion (checkpoint
    # tournament / head-to-head); best.pt tracking is inert without it.
    "--eval-every", "0",
    "--eval-batch-size", "1536",
    "--save-every-minutes", "20",
    "--plot-every", "5",
    "--precision", "bf16",
    "--max-seq-len", "100",
    "--d-model", "320",
    "--n-layers", "6",
    "--n-heads", "10",
    "--d-ff", "896",
    "--context-hidden-dim", "256",
]

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "numpy>=2", "matplotlib>=3.10")
    .add_local_dir(
        "plump",
        remote_path=f"{REMOTE_REPO}/plump",
        ignore=["**/__pycache__"],
    )
    .add_local_dir(
        "examples",
        remote_path=f"{REMOTE_REPO}/examples",
        ignore=["**/__pycache__"],
    )
)


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(
        run_dir.glob("plump_v4_iter_*.pt"),
        key=lambda file: int(file.stem.rsplit("_", 1)[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def _supervisor_log(run_dir: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{timestamp}] supervisor: {message}"
    print(line, flush=True)
    with (run_dir / "supervisor.log").open("a") as handle:
        handle.write(line + "\n")


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=16384,
    timeout=FUNCTION_TIMEOUT_SEC,
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
    volumes={VOLUME_MOUNT: volume},
)
def train(run_name: str = DEFAULT_RUN, extra_args: list[str] | None = None) -> str:
    run_dir = Path(VOLUME_MOUNT) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    stop_file = run_dir / "STOP"
    started = time.monotonic()
    fast_failures = 0
    status = "failed"

    for attempt in range(1, MAX_RESTARTS + 1):
        # Volume mounts only see external writes (e.g. the stop entrypoint's
        # sentinel) after an explicit reload.
        volume.reload()
        if stop_file.exists():
            status = "stopped"
            _supervisor_log(run_dir, "STOP sentinel found; exiting")
            break
        remaining = SUPERVISOR_BUDGET_SEC - (time.monotonic() - started)
        if remaining <= 0:
            status = "respawn"
            break

        latest = _latest_checkpoint(run_dir)
        command = [
            sys.executable,
            "-u",
            "examples/train_ppo.py",
            *BASE_TRAIN_ARGS,
            "--checkpoint-dir", str(run_dir),
            "--log-dir", str(run_dir),
            *(extra_args or []),
        ]
        if latest is not None:
            command += ["--resume-from", str(latest), "--resume-optimizer"]
        _supervisor_log(
            run_dir,
            f"attempt {attempt}/{MAX_RESTARTS} resume={latest}",
        )

        attempt_started = time.monotonic()
        process = subprocess.Popen(command, cwd=REMOTE_REPO)
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _supervisor_log(run_dir, "wall-clock budget reached; SIGTERM")
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            status = "respawn"
            break

        if code == 0:
            status = "completed"
            _supervisor_log(run_dir, "training completed")
            break
        if time.monotonic() - attempt_started < FAST_FAILURE_SEC:
            fast_failures += 1
            if fast_failures >= MAX_CONSECUTIVE_FAST_FAILURES:
                status = "failed"
                _supervisor_log(
                    run_dir,
                    f"{fast_failures} consecutive fast failures; giving up",
                )
                break
        else:
            fast_failures = 0
        _supervisor_log(run_dir, f"exit code {code}; restarting in 15s")
        time.sleep(15)

    volume.commit()
    volume.reload()
    if status == "respawn" and not stop_file.exists():
        _supervisor_log(run_dir, "respawning follow-up training call")
        handle = modal.Function.from_name(APP_NAME, "train").spawn(
            run_name,
            extra_args,
        )
        _supervisor_log(run_dir, f"respawned as {handle.object_id}")
    return status


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=600)
def verify_checkpoint(run_name: str = DEFAULT_RUN) -> str:
    """CPU-only model, optimizer, and league-reference resume sanity check."""

    sys.path.insert(0, REMOTE_REPO)
    import torch
    from plump.modeling import ModelConfig
    from plump.modeling.torch_model import PlumpTransformerModel

    run_dir = Path(VOLUME_MOUNT) / run_name
    latest = _latest_checkpoint(run_dir)
    if latest is None:
        return f"no checkpoints under {run_dir}"
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    model = PlumpTransformerModel(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    league = payload.get("league", {})
    missing_snapshots = [
        stored
        for stored in league.get("snapshot_paths", [])
        if not (run_dir / Path(stored).name).exists()
    ]
    if missing_snapshots:
        raise FileNotFoundError(
            f"checkpoint references missing league snapshots: {missing_snapshots}"
        )
    summary = (
        f"loaded {latest.name}: torch={torch.__version__} "
        f"iteration={payload.get('iteration')} "
        f"schema={payload.get('schema_version')} "
        "model=strict optimizer=loaded "
        f"league_snapshots={len(league.get('snapshot_paths', []))} "
        "league_missing=0 "
        f"payoff_cells={len(league.get('payoffs', []))}"
    )
    print(summary, flush=True)
    return summary


@app.local_entrypoint()
def kickoff(run_name: str = DEFAULT_RUN, extra: str = ""):
    """Spawn a detached training call on the deployed app."""

    extra_args = extra.split() if extra else None
    function = modal.Function.from_name(APP_NAME, "train")
    handle = function.spawn(run_name, extra_args)
    print(f"spawned train call {handle.object_id} for run '{run_name}'")
    print(f"logs: modal app logs {APP_NAME}")


@app.local_entrypoint()
def stop(run_name: str = DEFAULT_RUN):
    """Write the STOP sentinel; the supervisor exits at its next check."""

    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(b"stop\n"), f"/{run_name}/STOP")
    print(f"STOP sentinel written for run '{run_name}'")


@app.local_entrypoint()
def resume(run_name: str = DEFAULT_RUN, extra: str = ""):
    """Clear the STOP sentinel and spawn training again."""

    try:
        volume.remove_file(f"/{run_name}/STOP")
        print("STOP sentinel removed")
    except FileNotFoundError:
        pass
    except modal.exception.InvalidError as error:
        # modal surfaces a missing sentinel as InvalidError, not FileNotFound
        if "No such file" not in str(error):
            raise
    kickoff(run_name, extra)


@app.local_entrypoint()
def status(run_name: str = DEFAULT_RUN):
    """Print the latest.json summary for a run."""

    data = b"".join(volume.read_file(f"{run_name}/latest.json"))
    latest = json.loads(data)
    keys = ("iteration", "elapsed_sec", "checkpoint_path")
    print({key: latest.get(key) for key in keys})
