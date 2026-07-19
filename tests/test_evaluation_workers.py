import math

import torch

from plump.evaluation import DealBank
from plump.evaluation_workers import (
    ArenaMatchup,
    run_matchups_pooled,
    subsample_uniform,
)
from plump.modeling import ModelConfig
from plump.modeling.torch_model import PlumpTransformerModel
from plump.policies import ModelPolicy


def test_subsample_uniform_keeps_endpoints_and_even_spacing():
    items = list(range(331))
    picked = subsample_uniform(items, 100)
    assert len(picked) == 100
    assert picked[0] == 0
    assert picked[-1] == 330
    gaps = {b - a for a, b in zip(picked, picked[1:])}
    assert gaps <= {3, 4}


def test_subsample_uniform_returns_everything_under_cap():
    items = list(range(42))
    assert subsample_uniform(items, 100) == items


def _tiny_policy(seed: int) -> ModelPolicy:
    config = ModelConfig(
        max_seq_len=100,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        context_hidden_dim=64,
        game_hidden_dim=32,
    )
    torch.manual_seed(seed)
    model = PlumpTransformerModel(config)
    return ModelPolicy(
        model,
        device="cpu",
        greedy=False,
        event_length_buckets=(8, 16, 32, 64),
        batch_packing="numpy",
        lean_action_forward=True,
        name=f"tiny_{seed}",
    )


def test_run_matchups_pooled_completes_matchups_and_probes():
    policies = {"a": _tiny_policy(1), "b": _tiny_policy(2)}
    banks = {
        "match": DealBank.generate(
            player_counts=(3,),
            hand_sizes=(3, 4),
            deals_per_configuration=1,
            seed=11,
        ),
        "probe": DealBank.generate(
            player_counts=(3,),
            hand_sizes=(3,),
            deals_per_configuration=1,
            seed=12,
        ),
    }
    matchups = [
        ArenaMatchup(0, "matchup", "a", "b", "match", rng_seed=5),
        ArenaMatchup(1, "matchup", "b", "a", "match", rng_seed=5),
        ArenaMatchup(2, "probe", "a", "a", "probe", rng_seed=6),
    ]
    stats: dict = {}
    rows = list(
        run_matchups_pooled(
            [matchups[:2], matchups[2:]],
            banks,
            policies,
            num_workers=2,
            forward_batch=64,
            stats=stats,
        )
    )

    match_rows = [row for row in rows if row["kind"] == "matchup"]
    probe_rows = [row for row in rows if row["kind"] == "behavior"]
    assert len(match_rows) == 2
    assert len(probe_rows) == 1
    # match bank: 2 deals x 3 focal hands x 3 bidding positions = 18 rounds
    assert all(row["rounds"] == 18 for row in match_rows)
    assert probe_rows[0]["rounds"] == 9
    assert all(
        math.isfinite(row["macro_relative_reward"]) for row in match_rows
    )
    assert 0.0 <= probe_rows[0]["avg_bid"] <= 4.0
    assert 0.0 <= probe_rows[0]["bid_hit_rate"] <= 1.0
    assert stats["rounds"] == 45
    assert stats["decisions"] > 0
    assert stats["waves"] > 0


def test_build_arena_chunks_layers_and_degrees():
    import random
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from checkpoint_arena import build_arena_chunks

    keys = [f"iter_{i:05d}" for i in range(9)]
    chunks = build_arena_chunks(
        keys, matchings=3, tree_chunk=16, seed=11, rng=random.Random(11)
    )
    items = [item for chunk in chunks for item in chunk]
    probes = [item for item in items if item.kind == "probe"]
    matchups = [item for item in items if item.kind == "matchup"]
    assert sorted(item.focal_ref for item in probes) == sorted(keys)

    # every edge is played from both seats, equally often (layers may
    # re-draw a pair the tree or chain already played; that only reweights)
    from collections import Counter

    directed = Counter((item.focal_ref, item.table_ref) for item in matchups)
    assert all(directed[(b, a)] == count for (a, b), count in directed.items())

    # tree (n-1) + chain (n-1) + matchings, all both directions; odd field
    # size gives the leftover node an extra partner each matching layer
    per_matching = 2 * (len(keys) // 2 + 1)
    assert len(matchups) == 2 * (len(keys) - 1) * 2 + 3 * per_matching

    # chain edges present: adjacent keys meet in both directions
    for a, b in zip(keys, keys[1:]):
        assert (a, b) in directed and (b, a) in directed

    # matchup ids unique
    ids = [item.matchup_id for item in items]
    assert len(ids) == len(set(ids))
