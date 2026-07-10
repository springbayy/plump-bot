"""Self-play training pipeline for Plump agents."""

from .ppo import (
    PredictionStats,
    PPOTrainer,
    PositionBaseline,
    RolloutBuffer,
    RolloutSample,
    RolloutStats,
    TrainingConfig,
    UpdateStats,
    compute_relative_rewards,
    format_update_stats,
)
from .common import OpponentMix, allocate_opponent_arms
from .run_logger import TrainingRunLogger, training_config_snapshot
from .search_distill import (
    CounterfactualSearchRouter,
    SearchReplaySample,
    SearchRoutingStats,
    SearchTrustRegionUpdater,
    SearchUpdateStats,
    StratifiedReplayWindow,
)
from .expert_iteration import (
    CHECKPOINT_SCHEMA_VERSION,
    ExpertCycle,
    ExpertDiagnostics,
    ExpertIterationConfig,
    ExpertIterationTrainer,
    ExpertReplay,
    ExpertSample,
    ExpertUpdateStats,
)
from .expert_logger import ExpertRunLogger

__all__ = [
    "CounterfactualSearchRouter",
    "CHECKPOINT_SCHEMA_VERSION",
    "ExpertCycle",
    "ExpertDiagnostics",
    "ExpertIterationConfig",
    "ExpertIterationTrainer",
    "ExpertReplay",
    "ExpertRunLogger",
    "ExpertSample",
    "ExpertUpdateStats",
    "OpponentMix",
    "PredictionStats",
    "PPOTrainer",
    "PositionBaseline",
    "RolloutBuffer",
    "RolloutSample",
    "RolloutStats",
    "SearchReplaySample",
    "SearchRoutingStats",
    "SearchTrustRegionUpdater",
    "SearchUpdateStats",
    "StratifiedReplayWindow",
    "TrainingConfig",
    "TrainingRunLogger",
    "UpdateStats",
    "compute_relative_rewards",
    "allocate_opponent_arms",
    "format_update_stats",
    "training_config_snapshot",
]
