import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sperm_morphology.batch_run import run_batch


def main():
    parser = argparse.ArgumentParser(
        description="Run sperm morphology screening"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/morphology.yaml",
        help="path to config yaml"
    )

    args = parser.parse_args()

    run_batch(args.config)


if __name__ == "__main__":
    main()