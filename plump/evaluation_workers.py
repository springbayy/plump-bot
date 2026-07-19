"""Multiprocess arena evaluation: resident models, cross-matchup batched forwards.

The tournament scripts evaluate one matchup at a time in a single process, so
the GPU idles while Python steps engines and encodes observations. This module
mirrors the proven training split (``plump/training/env_workers.py``): CPU
workers own engine stepping and observation encoding for thousands of
concurrent rounds spanning many matchups, while the parent process keeps every
checkpoint's model resident and answers each wave with one batched forward per
model, pooling rows across all concurrent matchups.

Only stdlib imports at module level — the heavy plump/torch imports happen
inside the spawned worker (and inside parent-side functions) so importing this
module stays cheap and workers never touch the parent's GPU state.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Iterator, Sequence


@dataclass(frozen=True)
class ArenaMatchup:
    """One directed matchup: focal_ref plays seat 0, table_ref all other seats."""

    matchup_id: int
    kind: str  # "matchup" | "probe" (probe: focal_ref == table_ref, behavior stats)
    focal_ref: str
    table_ref: str
    bank_ref: str
    rng_seed: int  # base seed for per-scenario action-sampling rngs


@dataclass(frozen=True)
class ScenarioAssignment:
    scenario_id: int
    matchup_id: int
    bank_ref: str
    deal_index: int
    focal_hand: int
    bidding_position: int
    focal_ref: str
    table_ref: str


@dataclass
class PackedRequestGroup:
    """One wave's decisions for one (model, event-length bucket) pair.

    Workers trim each row's event fields to the smallest bucket that fits it
    before packing, so short-history rows (the majority) ship small and run in
    short-sequence forwards instead of being padded to the wave's longest row.
    """

    policy_ref: str
    event_length: int  # bucket the event fields are trimmed to
    scenario_ids: list[int]
    players: list[int]
    bidding: list[bool]  # per row: True in the bidding phase, False in play
    arrays: dict  # pack_encoded_rows output


@dataclass
class ScenarioDone:
    scenario_id: int
    matchup_id: int
    round_scores: dict[int, int]
    focal_bid: int
    focal_tricks: int
    first_leader: bool


@dataclass
class ArenaWave:
    request_groups: list[PackedRequestGroup] = field(default_factory=list)
    dones: list[ScenarioDone] = field(default_factory=list)
    active: int = 0
    queued: int = 0


def subsample_uniform(items: Sequence, cap: int) -> list:
    """Uniformly spaced subsequence with both endpoints, at most ``cap`` items."""

    if cap < 2:
        raise ValueError("cap must be at least 2.")
    if len(items) <= cap:
        return list(items)
    indices = sorted({round(i * (len(items) - 1) / (cap - 1)) for i in range(cap)})
    return [items[index] for index in indices]


def _worker_main(
    connection,
    banks,
    model_config,
    include_game_context,
    max_active,
    event_length_buckets,
) -> None:
    # Imported after spawn: workers are CPU-only engine/encoding processes.
    import gc

    from plump.env import PlumpEnv
    from plump.modeling.encoding import encode_observation, pack_encoded_rows
    from plump.rounds import round_game_config
    from plump.state import Phase

    # Cyclic GC repeatedly rescans thousands of live envs and encoded rows
    # during the per-wave allocation storm (measured ~30% of worker CPU).
    # Collect once per chunk instead.
    gc.disable()

    class _Scenario:
        __slots__ = ("assignment", "env")

        def __init__(self, assignment: ScenarioAssignment):
            deal = banks[assignment.bank_ref].deals[assignment.deal_index]
            n = deal.spec.num_players
            manual_hands = {
                player: list(deal.hands[(assignment.focal_hand + player) % n])
                for player in range(n)
            }
            self.assignment = assignment
            self.env = PlumpEnv(
                round_game_config(
                    deal.spec,
                    bidding_start_player=(-assignment.bidding_position) % n,
                    manual_hands=manual_hands,
                )
            )
            self.env.reset()

    def request_for(scenario: _Scenario) -> tuple | None:
        """(policy_ref, scenario_id, player, bidding, encoded) or None when done."""

        env = scenario.env
        if env.is_done():
            return None
        player = env.current_player()
        assignment = scenario.assignment
        return (
            assignment.focal_ref if player == 0 else assignment.table_ref,
            assignment.scenario_id,
            player,
            env.phase() == Phase.BIDDING,
            encode_observation(
                env.get_observation(player),
                model_config,
                include_game_context=include_game_context,
            ),
        )

    def done_for(scenario: _Scenario) -> ScenarioDone:
        round_state = scenario.env.state.current_round
        assignment = scenario.assignment
        bid = next(item.value for item in round_state.bids if item.player == 0)
        return ScenarioDone(
            scenario_id=assignment.scenario_id,
            matchup_id=assignment.matchup_id,
            round_scores=dict(round_state.round_scores),
            focal_bid=int(bid),
            focal_tricks=int(round_state.tricks_won[0]),
            first_leader=round_state.play_start_player == 0,
        )

    while True:
        message = connection.recv()
        if message[0] == "shutdown":
            connection.close()
            return
        assert message[0] == "run"
        gc.collect()
        queue: deque[ScenarioAssignment] = deque(message[1])
        active: dict[int, _Scenario] = {}
        needs_advance: list[int] = []

        while True:
            while queue and len(active) < max_active:
                assignment = queue.popleft()
                active[assignment.scenario_id] = _Scenario(assignment)
                needs_advance.append(assignment.scenario_id)

            wave = ArenaWave()
            # (policy_ref, bucket) -> ([scenario_ids], [players], [bidding], [encoded])
            buffers: dict[tuple[str, int], tuple[list, list, list, list]] = {}
            while needs_advance:
                scenario_id = needs_advance.pop()
                scenario = active[scenario_id]
                request = request_for(scenario)
                if request is None:
                    del active[scenario_id]
                    wave.dones.append(done_for(scenario))
                    while queue and len(active) < max_active:
                        assignment = queue.popleft()
                        active[assignment.scenario_id] = _Scenario(assignment)
                        needs_advance.append(assignment.scenario_id)
                else:
                    ref, sid, player, bidding, encoded = request
                    valid = sum(encoded.event_valid_mask)
                    bucket = next(
                        (edge for edge in event_length_buckets if edge >= valid),
                        len(encoded.event_tokens),
                    )
                    buffer = buffers.setdefault((ref, bucket), ([], [], [], []))
                    buffer[0].append(sid)
                    buffer[1].append(player)
                    buffer[2].append(bidding)
                    buffer[3].append(encoded)
            for (ref, bucket), (sids, players, bidding, encoded) in buffers.items():
                wave.request_groups.append(
                    PackedRequestGroup(
                        policy_ref=ref,
                        event_length=bucket,
                        scenario_ids=sids,
                        players=players,
                        bidding=bidding,
                        arrays=pack_encoded_rows(encoded, bucket),
                    )
                )
            wave.active = len(active)
            wave.queued = len(queue)
            connection.send(wave)
            if not active and not queue:
                break

            message = connection.recv()
            assert message[0] == "actions"
            for scenario_id, action in message[1].items():
                active[scenario_id].env.step(action)
                needs_advance.append(scenario_id)


class ArenaWorkerPool:
    """Persistent spawn-based worker pool driven wave-by-wave by the parent."""

    def __init__(
        self,
        *,
        num_workers: int,
        banks: dict,
        model_config,
        include_game_context: bool,
        max_active_per_worker: int | None = None,
        event_length_buckets: Sequence[int] = (),
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be positive.")
        context = get_context("spawn")
        max_active = max_active_per_worker or 1_000_000_000
        self._connections = []
        self._processes = []
        for _ in range(num_workers):
            parent_connection, child_connection = context.Pipe()
            process = context.Process(
                target=_worker_main,
                args=(
                    child_connection,
                    banks,
                    model_config,
                    include_game_context,
                    max_active,
                    tuple(event_length_buckets),
                ),
                daemon=True,
            )
            process.start()
            child_connection.close()
            self._connections.append(parent_connection)
            self._processes.append(process)
        self._scenario_owner: dict[int, int] = {}
        self._finished: set[int] = set()

    @property
    def num_workers(self) -> int:
        return len(self._processes)

    def worker_groups(self, count: int) -> list[list[int]]:
        """Split worker ids into ``count`` round-robin groups (empties dropped)."""

        groups = [
            [index for index in range(len(self._processes)) if index % count == offset]
            for offset in range(count)
        ]
        return [group for group in groups if group]

    def begin(self, assignments: Sequence[ScenarioAssignment]) -> None:
        self._scenario_owner.clear()
        self._finished.clear()
        shards: list[list[ScenarioAssignment]] = [[] for _ in self._connections]
        for index, assignment in enumerate(assignments):
            worker_id = index % len(self._connections)
            shards[worker_id].append(assignment)
            self._scenario_owner[assignment.scenario_id] = worker_id
        for connection, shard in zip(self._connections, shards):
            connection.send(("run", shard))

    def gather_wave(
        self,
        worker_ids: Sequence[int] | None = None,
    ) -> tuple[list[PackedRequestGroup], list[ScenarioDone], bool]:
        """Receive one wave from the given workers (all by default).

        The returned flag is True when every polled worker is finished.
        """

        if worker_ids is None:
            worker_ids = range(len(self._connections))
        request_groups: list[PackedRequestGroup] = []
        dones: list[ScenarioDone] = []
        for worker_id in worker_ids:
            if worker_id in self._finished:
                continue
            wave: ArenaWave = self._connections[worker_id].recv()
            request_groups.extend(wave.request_groups)
            dones.extend(wave.dones)
            if wave.active == 0 and wave.queued == 0:
                self._finished.add(worker_id)
        done = all(worker_id in self._finished for worker_id in worker_ids)
        return request_groups, dones, done

    def send_actions(
        self,
        actions: dict[int, object],
        worker_ids: Sequence[int] | None = None,
    ) -> None:
        if worker_ids is None:
            worker_ids = range(len(self._connections))
        by_worker: dict[int, dict[int, object]] = defaultdict(dict)
        for scenario_id, action in actions.items():
            by_worker[self._scenario_owner[scenario_id]][scenario_id] = action
        for worker_id in worker_ids:
            if worker_id in self._finished:
                continue
            self._connections[worker_id].send(("actions", by_worker.get(worker_id, {})))

    def close(self) -> None:
        for connection in self._connections:
            try:
                connection.send(("shutdown",))
                connection.close()
            except (BrokenPipeError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

    def __enter__(self) -> "ArenaWorkerPool":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _macro_cell_mean(results, value) -> float:
    """Uniform mean over (players, hand size) cells: fair across configurations."""

    by_cell: dict[tuple[int, int], list[float]] = defaultdict(list)
    for result in results:
        by_cell[(result.spec.num_players, result.spec.hand_size)].append(value(result))
    return sum(
        sum(values) / len(values) for values in by_cell.values()
    ) / len(by_cell)


def _matchup_row(matchup: ArenaMatchup, results: list, elapsed_sec: float) -> dict:
    if matchup.kind == "probe":
        return {
            "kind": "behavior",
            "key": matchup.focal_ref,
            "rounds": len(results),
            "avg_bid": _macro_cell_mean(results, lambda r: r.bid_value),
            "bid_hit_rate": _macro_cell_mean(results, lambda r: r.bid_hit),
            "zero_bid_rate": _macro_cell_mean(
                results, lambda r: float(r.bid_value == 0)
            ),
            "all_bid_rate": _macro_cell_mean(
                results, lambda r: float(r.bid_value == r.spec.hand_size)
            ),
            "elapsed_sec": elapsed_sec,
        }
    return {
        "kind": "matchup",
        "focal": matchup.focal_ref,
        "table": matchup.table_ref,
        "rounds": len(results),
        "macro_relative_reward": _macro_cell_mean(
            results, lambda r: r.relative_reward
        ),
        "elapsed_sec": elapsed_sec,
    }


def run_matchups_pooled(
    chunks: Sequence[Sequence[ArenaMatchup]],
    banks: dict,
    policies: dict,
    *,
    num_workers: int = 8,
    forward_batch: int = 8192,
    max_active_per_worker: int | None = 3072,
    stats: dict | None = None,
) -> Iterator[dict]:
    """Yield one result-row dict per completed matchup, chunk by chunk.

    Each chunk's matchups run concurrently (workers keep up to
    ``max_active_per_worker`` rounds each in flight, refilling as rounds
    finish). Every wave the parent concatenates pending decisions per
    (model, event-length bucket) — workers pre-trim rows to their bucket —
    and answers each group with one homogeneous batched forward, so rows from
    every concurrent matchup involving the same model share forwards. Chunks
    that pack few distinct models (e.g. all-pairs blocks) get the largest
    per-forward batches. ``stats`` (if given) is updated in place with
    cumulative ``decisions``, ``waves``, ``rounds``, and phase-second
    counters.
    """

    from plump.evaluation import ScenarioResult, _relative_rewards, _scenario_seed
    import gc

    from plump.modeling.torch_model import concat_packed_arrays, packed_arrays_to_batch
    from plump.state import Phase

    if stats is None:
        stats = {}
    stats.setdefault("decisions", 0)
    stats.setdefault("waves", 0)
    stats.setdefault("rounds", 0)
    stats.setdefault("gather_sec", 0.0)
    stats.setdefault("forward_sec", 0.0)
    stats.setdefault("send_sec", 0.0)

    first = next(iter(policies.values()))
    # Uploads must match the resident weights (fp16 when models were .half()ed).
    import torch

    param_dtype = next(first.model.parameters()).dtype
    float_dtype = param_dtype if param_dtype != torch.float32 else None
    pool = ArenaWorkerPool(
        num_workers=num_workers,
        banks=banks,
        model_config=first.model_config,
        include_game_context=first.include_game_context,
        max_active_per_worker=max_active_per_worker,
        event_length_buckets=first.event_length_buckets,
    )
    scenario_counter = 0
    gc.disable()
    try:
        for chunk in chunks:
            gc.collect()
            chunk_by_id = {matchup.matchup_id: matchup for matchup in chunk}
            assignments: list[ScenarioAssignment] = []
            rngs: dict[int, random.Random] = {}
            # scenario_id -> (matchup, deal, focal_hand, bidding_position)
            meta: dict[int, tuple] = {}
            pending: dict[int, int] = {matchup.matchup_id: 0 for matchup in chunk}
            results: dict[int, list] = {matchup.matchup_id: [] for matchup in chunk}
            for matchup in chunk:
                bank = banks[matchup.bank_ref]
                for deal_index, deal in enumerate(bank.deals):
                    n = deal.spec.num_players
                    for focal_hand in range(n):
                        for bidding_position in range(n):
                            scenario_id = scenario_counter
                            scenario_counter += 1
                            assignments.append(
                                ScenarioAssignment(
                                    scenario_id=scenario_id,
                                    matchup_id=matchup.matchup_id,
                                    bank_ref=matchup.bank_ref,
                                    deal_index=deal_index,
                                    focal_hand=focal_hand,
                                    bidding_position=bidding_position,
                                    focal_ref=matchup.focal_ref,
                                    table_ref=matchup.table_ref,
                                )
                            )
                            rngs[scenario_id] = random.Random(
                                _scenario_seed(
                                    matchup.rng_seed,
                                    deal.deal_id,
                                    focal_hand,
                                    bidding_position,
                                )
                            )
                            meta[scenario_id] = (
                                matchup,
                                deal,
                                focal_hand,
                                bidding_position,
                            )
                            pending[matchup.matchup_id] += 1

            chunk_started = time.monotonic()
            pool.begin(assignments)
            # Two worker groups take turns: while the parent runs one group's
            # forwards on the GPU, the other group's workers step engines and
            # encode observations, overlapping CPU and GPU work.
            turns = deque(pool.worker_groups(2))
            while turns:
                worker_group = turns.popleft()
                phase_started = time.monotonic()
                request_groups, dones, group_finished = pool.gather_wave(worker_group)
                stats["gather_sec"] += time.monotonic() - phase_started
                for done in dones:
                    matchup, deal, focal_hand, bidding_position = meta.pop(
                        done.scenario_id
                    )
                    rngs.pop(done.scenario_id, None)
                    relative = _relative_rewards(done.round_scores)
                    results[matchup.matchup_id].append(
                        ScenarioResult(
                            deal_id=deal.deal_id,
                            spec=deal.spec,
                            focal_hand=focal_hand,
                            bidding_position=bidding_position,
                            raw_score=float(done.round_scores[0]),
                            relative_reward=relative[0],
                            bid_hit=float(done.focal_tricks == done.focal_bid),
                            bid_abs_error=float(
                                abs(done.focal_tricks - done.focal_bid)
                            ),
                            bid_value=float(done.focal_bid),
                            first_leader=float(done.first_leader),
                            forward_passes=1,
                        )
                    )
                    stats["rounds"] += 1
                    pending[matchup.matchup_id] -= 1
                    if pending[matchup.matchup_id] == 0:
                        elapsed = round(time.monotonic() - chunk_started, 1)
                        yield _matchup_row(
                            chunk_by_id[matchup.matchup_id],
                            results.pop(matchup.matchup_id),
                            elapsed,
                        )
                if group_finished:
                    continue
                turns.append(worker_group)

                phase_started = time.monotonic()
                actions: dict[int, object] = {}
                wave_rows = 0
                if request_groups:
                    # Workers pre-trim rows to their event-length bucket, so
                    # grouping by (model, bucket) yields homogeneous batches:
                    # short-history rows (the majority) run short forwards.
                    by_key: dict[tuple[str, int], list[PackedRequestGroup]] = (
                        defaultdict(list)
                    )
                    for request_group in request_groups:
                        by_key[
                            (request_group.policy_ref, request_group.event_length)
                        ].append(request_group)
                    for ref, event_length in sorted(by_key):
                        groups = by_key[(ref, event_length)]
                        policy = policies[ref]
                        merged = concat_packed_arrays(
                            [group.arrays for group in groups]
                        )
                        scenario_ids = [
                            sid for group in groups for sid in group.scenario_ids
                        ]
                        players = [
                            player for group in groups for player in group.players
                        ]
                        phases = [
                            Phase.BIDDING if bidding else Phase.PLAYING
                            for group in groups
                            for bidding in group.bidding
                        ]
                        wave_rows += len(scenario_ids)
                        for start in range(0, len(scenario_ids), forward_batch):
                            stop = min(start + forward_batch, len(scenario_ids))
                            batch = packed_arrays_to_batch(
                                {
                                    key: value[start:stop]
                                    for key, value in merged.items()
                                },
                                device=first.device,
                                event_length_buckets=first.event_length_buckets,
                                float_dtype=float_dtype,
                            )
                            batch_actions = policy.act_batch(
                                batch,
                                phases=phases[start:stop],
                                players=players[start:stop],
                                rngs=[rngs[sid] for sid in scenario_ids[start:stop]],
                            )
                            for sid, action in zip(
                                scenario_ids[start:stop], batch_actions
                            ):
                                actions[sid] = action
                stats["forward_sec"] += time.monotonic() - phase_started
                stats["decisions"] += wave_rows
                stats["waves"] += 1
                phase_started = time.monotonic()
                pool.send_actions(actions, worker_group)
                stats["send_sec"] += time.monotonic() - phase_started
    finally:
        gc.enable()
        pool.close()
