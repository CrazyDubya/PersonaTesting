import argparse
from .runners import run_full_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persona evaluation experiment runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    run_full_experiment(args.config)


if __name__ == "__main__":
    main()
