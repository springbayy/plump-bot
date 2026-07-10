"""Launch the local Plump browser GUI."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plump.gui.app import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local Plump browser GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(host=args.host, port=args.port, checkpoint_path=args.checkpoint, device=args.device)
