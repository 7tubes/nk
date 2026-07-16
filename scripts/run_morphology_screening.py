import argparse

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