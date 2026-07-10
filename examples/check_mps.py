"""Check whether PyTorch can use Apple Silicon MPS in this environment."""

from __future__ import annotations

import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"mps built: {torch.backends.mps.is_built()}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    print(f"mps device count: {torch.mps.device_count()}")

    x = torch.ones(2, device="mps")
    print(f"tensor device: {x.device}")
    print(f"tensor value: {x + 1}")


if __name__ == "__main__":
    main()
