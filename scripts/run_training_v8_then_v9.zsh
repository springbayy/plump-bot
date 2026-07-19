#!/bin/zsh
set -euo pipefail

repo_dir="/Users/erehmax/Documents/Projects/Personal/plump-bot"

zsh "$repo_dir/scripts/run_training_v8_10m_seed1.zsh"
zsh "$repo_dir/scripts/run_training_v9_10m_expert_seed1.zsh"
