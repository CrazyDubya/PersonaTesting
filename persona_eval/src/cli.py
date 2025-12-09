"""
CLI entrypoint for the persona evaluation framework.

Usage:
    python -m src.cli --config config/default_config.yaml
    python -m src.cli --config config/default_config.yaml --models gpt-4o --conditions baseline_mc
    python -m src.cli --config config/default_config.yaml --test --model gpt-4o --condition baseline_mc
"""

import argparse
import sys
from typing import List, Optional
from .runners import run_full_experiment, run_quick_test


def parse_list_arg(value: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated list argument."""
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Persona evaluation experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment with default config
  python -m src.cli --config config/default_config.yaml

  # Run only specific models
  python -m src.cli --config config/default_config.yaml --models gpt-4o,gpt-4o-mini

  # Run only specific conditions
  python -m src.cli --config config/default_config.yaml --conditions baseline_mc,shallow_persona_mc

  # Run quick test with 5 questions
  python -m src.cli --config config/default_config.yaml --test --model gpt-4o --condition baseline_mc

  # Skip sampling and only run scoring + metrics
  python -m src.cli --config config/default_config.yaml --skip-sampling

  # Force re-run even if output files exist
  python -m src.cli --config config/default_config.yaml --no-skip-existing
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model IDs to run (default: all)",
    )

    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated list of condition IDs to run (default: all)",
    )

    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Skip the sampling phase (use existing raw responses)",
    )

    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip the scoring phase (use existing scored files)",
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Don't skip existing output files (re-run everything)",
    )

    # Quick test mode
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick test with limited questions",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID for quick test (used with --test)",
    )

    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Condition ID for quick test (used with --test)",
    )

    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Number of questions for quick test (default: 5)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples per question for quick test (default: 1)",
    )

    args = parser.parse_args()

    # Quick test mode
    if args.test:
        if not args.model:
            print("Error: --model is required for --test mode")
            sys.exit(1)
        if not args.condition:
            print("Error: --condition is required for --test mode")
            sys.exit(1)

        print(f"Running quick test: model={args.model}, condition={args.condition}")
        print(f"  Questions: {args.num_questions}, Samples: {args.num_samples}")

        results = run_quick_test(
            config_path=args.config,
            model_id=args.model,
            condition_id=args.condition,
            num_questions=args.num_questions,
            num_samples=args.num_samples,
        )

        if "error" in results:
            print(f"Error: {results['error']}")
            sys.exit(1)

        print(f"\nResults:")
        print(f"  Accuracy: {results['accuracy']:.2%}")
        print(f"\nDetailed results:")
        for r in results["results"]:
            correct_str = "CORRECT" if r.get("is_correct") else "WRONG"
            print(f"  Q{r['question_id']}: predicted={r['predicted_option_letter']}, "
                  f"correct={r['correct_letter']}, {correct_str}")
            if r.get("raw_response"):
                # Show first 200 chars of response
                snippet = r["raw_response"][:200].replace("\n", " ")
                print(f"    Response: {snippet}...")
        return

    # Full experiment mode
    models_filter = parse_list_arg(args.models)
    conditions_filter = parse_list_arg(args.conditions)

    print(f"Running full experiment")
    print(f"  Config: {args.config}")
    if models_filter:
        print(f"  Models: {models_filter}")
    if conditions_filter:
        print(f"  Conditions: {conditions_filter}")

    run_full_experiment(
        config_path=args.config,
        models_filter=models_filter,
        conditions_filter=conditions_filter,
        skip_sampling=args.skip_sampling,
        skip_scoring=args.skip_scoring,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
