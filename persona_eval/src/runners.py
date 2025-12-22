"""
Experiment runner module for orchestrating the full evaluation pipeline.

Handles:
- Loading configurations and data
- Running sampling across models and conditions
- Scoring responses
- Computing and saving metrics
"""

import json
import os
from typing import List, Dict, Any, Optional
from .config import load_yaml_config, build_experiment_config
from .data_models import Question, ConditionConfig, ModelConfig, ExperimentConfig
from .utils.io_utils import ensure_dir, load_questions_from_jsonl, write_jsonl, read_jsonl
from .utils.logging_utils import setup_logger, get_logger
from .model_api import build_clients, LLMClient
from .sampling import run_sampling_for_condition_and_model
from .scoring import score_raw_responses
from .metrics import compute_metrics, format_summary_table


def load_all_questions(datasets_config: List[Dict[str, Any]]) -> List[Question]:
    """
    Load questions from all configured datasets.

    Args:
        datasets_config: List of dataset configuration dicts.

    Returns:
        List of all questions from all datasets.
        
    Raises:
        ValueError: If no questions could be loaded from any dataset.
    """
    if not datasets_config:
        raise ValueError("No datasets configured")
    
    all_questions: List[Question] = []
    for ds in datasets_config:
        dataset_name = ds.get("name")
        path = ds.get("path")
        
        if not dataset_name:
            print(f"⚠ Warning: Dataset missing 'name' field, skipping")
            continue
        if not path:
            print(f"⚠ Warning: Dataset '{dataset_name}' missing 'path' field, skipping")
            continue
            
        if not os.path.exists(path):
            print(f"⚠ Warning: Dataset file not found: {path}")
            continue
            
        try:
            questions = load_questions_from_jsonl(path, dataset_name)
            print(f"✓ Loaded {len(questions)} questions from {dataset_name}")
            all_questions.extend(questions)
        except Exception as e:
            print(f"⚠ Warning: Failed to load dataset '{dataset_name}': {e}")
            continue
    
    if not all_questions:
        raise ValueError("No questions could be loaded from any dataset. Check dataset paths and format.")
    
    return all_questions


def run_sampling_phase(
    exp_cfg: ExperimentConfig,
    all_questions: List[Question],
    model_clients: Dict[str, LLMClient],
    logger: Any,
    models_filter: Optional[List[str]] = None,
    conditions_filter: Optional[List[str]] = None,
    skip_existing: bool = True,
) -> None:
    """
    Run the sampling phase: generate responses for all model/condition pairs.

    Args:
        exp_cfg: Experiment configuration.
        all_questions: List of questions to sample.
        model_clients: Dict of model_id -> LLMClient.
        logger: Logger instance.
        models_filter: Optional list of model IDs to run (None = all).
        conditions_filter: Optional list of condition IDs to run (None = all).
        skip_existing: If True, skip sampling if output file exists.
    """
    models_to_run = exp_cfg.models
    if models_filter:
        models_to_run = [m for m in exp_cfg.models if m.id in models_filter]

    conditions_to_run = exp_cfg.conditions
    if conditions_filter:
        conditions_to_run = [c for c in exp_cfg.conditions if c.id in conditions_filter]

    for model_config in models_to_run:
        if model_config.id not in model_clients:
            logger.warning(f"Skipping model {model_config.id}: client not available")
            continue

        client = model_clients[model_config.id]

        for condition in conditions_to_run:
            raw_path = os.path.join(
                exp_cfg.output["raw_responses_dir"],
                f"raw_{model_config.id}_{condition.id}.jsonl",
            )

            if skip_existing and os.path.exists(raw_path):
                logger.info(f"Skipping existing: model={model_config.id}, condition={condition.id}")
                continue

            logger.info(f"Sampling for model={model_config.id}, condition={condition.id}")

            def progress_callback(current: int, total: int) -> None:
                if current % 10 == 0 or current == total:
                    logger.info(f"  Progress: {current}/{total}")

            run_sampling_for_condition_and_model(
                questions=all_questions,
                condition=condition,
                model_config=model_config,
                client=client,
                num_samples_per_question=exp_cfg.sampling.num_samples_per_question,
                temperature=exp_cfg.sampling.temperature,
                max_tokens_reasoning=exp_cfg.sampling.max_tokens_reasoning,
                raw_output_path=raw_path,
                progress_callback=progress_callback,
            )

            logger.info(f"Completed sampling: model={model_config.id}, condition={condition.id}")


def run_scoring_phase(
    exp_cfg: ExperimentConfig,
    all_questions: List[Question],
    model_clients: Dict[str, LLMClient],
    logger: Any,
    models_filter: Optional[List[str]] = None,
    conditions_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run the scoring phase: score all raw responses.

    Args:
        exp_cfg: Experiment configuration.
        all_questions: List of questions (for lookup).
        model_clients: Dict of model_id -> LLMClient (for judge).
        logger: Logger instance.
        models_filter: Optional list of model IDs to score (None = all).
        conditions_filter: Optional list of condition IDs to score (None = all).

    Returns:
        List of all scored rows.
    """
    # Build judge clients
    judge_clients: Dict[str, LLMClient] = {}
    if exp_cfg.judge.enabled:
        if exp_cfg.judge.judge_model_id in model_clients:
            judge_clients[exp_cfg.judge.judge_model_id] = model_clients[exp_cfg.judge.judge_model_id]

    all_scored_rows: List[Dict[str, Any]] = []

    models_to_score = exp_cfg.models
    if models_filter:
        models_to_score = [m for m in exp_cfg.models if m.id in models_filter]

    conditions_to_score = exp_cfg.conditions
    if conditions_filter:
        conditions_to_score = [c for c in exp_cfg.conditions if c.id in conditions_filter]

    for model_config in models_to_score:
        for condition in conditions_to_score:
            raw_path = os.path.join(
                exp_cfg.output["raw_responses_dir"],
                f"raw_{model_config.id}_{condition.id}.jsonl",
            )

            if not os.path.exists(raw_path):
                logger.warning(f"Raw file not found: {raw_path}")
                continue

            logger.info(f"Scoring for model={model_config.id}, condition={condition.id}")

            raw_rows = read_jsonl(raw_path)

            def progress_callback(current: int, total: int) -> None:
                if current % 100 == 0 or current == total:
                    logger.info(f"  Scoring progress: {current}/{total}")

            scored_rows = score_raw_responses(
                raw_rows=raw_rows,
                questions=all_questions,
                judge_config=exp_cfg.judge,
                judge_clients=judge_clients,
                progress_callback=progress_callback,
            )

            scored_path = os.path.join(
                exp_cfg.output["scored_dir"],
                f"scored_{model_config.id}_{condition.id}.jsonl",
            )
            write_jsonl(scored_path, scored_rows)
            all_scored_rows.extend(scored_rows)

            logger.info(f"Completed scoring: model={model_config.id}, condition={condition.id}")

    return all_scored_rows


def run_metrics_phase(
    exp_cfg: ExperimentConfig,
    all_scored_rows: List[Dict[str, Any]],
    logger: Any,
) -> List[Dict[str, Any]]:
    """
    Run the metrics phase: compute and save aggregate metrics.

    Args:
        exp_cfg: Experiment configuration.
        all_scored_rows: List of all scored rows.
        logger: Logger instance.

    Returns:
        List of summary metric rows.
    """
    logger.info("Computing aggregate metrics")

    summary_rows = compute_metrics(
        scored_rows=all_scored_rows,
        num_samples_per_question=exp_cfg.sampling.num_samples_per_question,
    )

    summary_path = os.path.join(
        exp_cfg.output["summaries_dir"],
        "summary_metrics.jsonl",
    )
    write_jsonl(summary_path, summary_rows)

    # Also write a human-readable summary
    summary_table = format_summary_table(summary_rows)
    summary_txt_path = os.path.join(
        exp_cfg.output["summaries_dir"],
        "summary_metrics.txt",
    )
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_table)

    logger.info(f"Summary metrics written to {summary_path}")
    logger.info("\n" + summary_table)

    return summary_rows


def run_full_experiment(
    config_path: str,
    models_filter: Optional[List[str]] = None,
    conditions_filter: Optional[List[str]] = None,
    skip_sampling: bool = False,
    skip_scoring: bool = False,
    skip_existing: bool = True,
) -> None:
    """
    Run the full experiment pipeline.

    Args:
        config_path: Path to YAML configuration file.
        models_filter: Optional list of model IDs to run.
        conditions_filter: Optional list of condition IDs to run.
        skip_sampling: If True, skip the sampling phase.
        skip_scoring: If True, skip the scoring phase.
        skip_existing: If True, skip sampling for existing output files.
    """
    raw_cfg = load_yaml_config(config_path)
    exp_cfg = build_experiment_config(raw_cfg)

    # Ensure output directories exist
    ensure_dir(exp_cfg.output["base_dir"])
    ensure_dir(exp_cfg.output["logs_dir"])
    ensure_dir(exp_cfg.output["raw_responses_dir"])
    ensure_dir(exp_cfg.output["scored_dir"])
    ensure_dir(exp_cfg.output["summaries_dir"])

    logger = setup_logger(exp_cfg.output["logs_dir"])
    logger.info("Starting full experiment")
    logger.info(f"Config: {config_path}")

    # Load questions
    all_questions = load_all_questions(exp_cfg.datasets)
    logger.info(f"Loaded {len(all_questions)} total questions")

    if len(all_questions) == 0:
        logger.error("No questions loaded. Check dataset paths.")
        return

    # Build model clients
    model_clients = build_clients(exp_cfg.models)
    logger.info(f"Initialized {len(model_clients)} model clients")

    # Phase 1: Sampling
    if not skip_sampling:
        run_sampling_phase(
            exp_cfg=exp_cfg,
            all_questions=all_questions,
            model_clients=model_clients,
            logger=logger,
            models_filter=models_filter,
            conditions_filter=conditions_filter,
            skip_existing=skip_existing,
        )
    else:
        logger.info("Skipping sampling phase")

    # Phase 2: Scoring
    all_scored_rows: List[Dict[str, Any]] = []
    if not skip_scoring:
        all_scored_rows = run_scoring_phase(
            exp_cfg=exp_cfg,
            all_questions=all_questions,
            model_clients=model_clients,
            logger=logger,
            models_filter=models_filter,
            conditions_filter=conditions_filter,
        )
    else:
        logger.info("Skipping scoring phase - loading existing scored files")
        # Load existing scored files
        for model_config in exp_cfg.models:
            for condition in exp_cfg.conditions:
                scored_path = os.path.join(
                    exp_cfg.output["scored_dir"],
                    f"scored_{model_config.id}_{condition.id}.jsonl",
                )
                if os.path.exists(scored_path):
                    rows = read_jsonl(scored_path)
                    all_scored_rows.extend(rows)

    # Phase 3: Metrics
    if all_scored_rows:
        run_metrics_phase(
            exp_cfg=exp_cfg,
            all_scored_rows=all_scored_rows,
            logger=logger,
        )
    else:
        logger.warning("No scored rows to compute metrics from")

    logger.info("Experiment completed")


def run_quick_test(
    config_path: str,
    model_id: str,
    condition_id: str,
    num_questions: int = 5,
    num_samples: int = 1,
) -> Dict[str, Any]:
    """
    Run a quick test with limited questions and samples.

    Args:
        config_path: Path to YAML configuration file.
        model_id: ID of the model to test.
        condition_id: ID of the condition to test.
        num_questions: Number of questions to test.
        num_samples: Number of samples per question.

    Returns:
        Dict with test results.
    """
    raw_cfg = load_yaml_config(config_path)
    exp_cfg = build_experiment_config(raw_cfg)

    # Load questions
    all_questions = load_all_questions(exp_cfg.datasets)
    test_questions = all_questions[:num_questions]

    # Get model and condition configs
    model_config = next((m for m in exp_cfg.models if m.id == model_id), None)
    condition_config = next((c for c in exp_cfg.conditions if c.id == condition_id), None)

    if not model_config:
        return {"error": f"Model not found: {model_id}"}
    if not condition_config:
        return {"error": f"Condition not found: {condition_id}"}

    # Build client
    clients = build_clients([model_config])
    if model_id not in clients:
        return {"error": f"Could not initialize client for {model_id}"}

    client = clients[model_id]

    # Run sampling
    from .sampling import run_single_question_sample

    results: List[Dict[str, Any]] = []
    for q in test_questions:
        for _ in range(num_samples):
            result = run_single_question_sample(
                question=q,
                condition=condition_config,
                client=client,
                temperature=exp_cfg.sampling.temperature,
                max_tokens=exp_cfg.sampling.max_tokens_reasoning,
            )
            result["question_id"] = q.id
            result["correct_letter"] = q.correct_option_letter
            result["is_correct"] = (
                result["predicted_option_letter"] == q.correct_option_letter
                if q.correct_option_letter
                else None
            )
            results.append(result)

    # Compute quick stats
    correct_count = sum(1 for r in results if r.get("is_correct"))
    total_count = len([r for r in results if r.get("is_correct") is not None])

    return {
        "model_id": model_id,
        "condition_id": condition_id,
        "num_questions": num_questions,
        "num_samples": num_samples,
        "accuracy": correct_count / total_count if total_count > 0 else 0.0,
        "results": results,
    }
