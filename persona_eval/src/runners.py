import json
import os
from typing import List, Dict, Any
from .config import load_yaml_config, build_experiment_config
from .data_models import Question, ConditionConfig, ModelConfig
from .utils.io_utils import ensure_dir, load_questions_from_jsonl, write_jsonl
from .utils.logging_utils import setup_logger
from .model_api import build_clients, LLMClient
from .sampling import run_sampling_for_condition_and_model
from .scoring import score_raw_responses
from .metrics import compute_metrics


def load_all_questions(datasets_config: List[Dict[str, Any]]) -> List[Question]:
    all_questions: List[Question] = []
    for ds in datasets_config:
        dataset_name = ds["name"]
        path = ds["path"]
        questions = load_questions_from_jsonl(path, dataset_name)
        all_questions.extend(questions)
    return all_questions


def run_full_experiment(config_path: str) -> None:
    raw_cfg = load_yaml_config(config_path)
    exp_cfg = build_experiment_config(raw_cfg)

    ensure_dir(exp_cfg.output["base_dir"])
    ensure_dir(exp_cfg.output["logs_dir"])
    ensure_dir(exp_cfg.output["raw_responses_dir"])
    ensure_dir(exp_cfg.output["scored_dir"])
    ensure_dir(exp_cfg.output["summaries_dir"])

    logger = setup_logger(exp_cfg.output["logs_dir"])
    logger.info("Starting full experiment")

    all_questions = load_all_questions(exp_cfg.datasets)
    logger.info(f"Loaded {len(all_questions)} total questions.")

    model_clients = build_clients(exp_cfg.models)

    # 1. Sampling
    for model_config in exp_cfg.models:
        for condition in exp_cfg.conditions:
            logger.info(
                f"Sampling for model={model_config.id}, condition={condition.id}"
            )
            raw_path = os.path.join(
                exp_cfg.output["raw_responses_dir"],
                f"raw_{model_config.id}_{condition.id}.jsonl",
            )
            client = model_clients[model_config.id]

            run_sampling_for_condition_and_model(
                questions=all_questions,
                condition=condition,
                model_config=model_config,
                client=client,
                num_samples_per_question=exp_cfg.sampling.num_samples_per_question,
                temperature=exp_cfg.sampling.temperature,
                max_tokens_reasoning=exp_cfg.sampling.max_tokens_reasoning,
                raw_output_path=raw_path,
            )

    # 2. Scoring + Metrics
    judge_clients: Dict[str, LLMClient] = {}
    if exp_cfg.judge.enabled:
        for m in exp_cfg.models:
            judge_clients[m.id] = model_clients[m.id]

    all_scored_rows: List[Dict[str, Any]] = []

    for model_config in exp_cfg.models:
        for condition in exp_cfg.conditions:
            raw_path = os.path.join(
                exp_cfg.output["raw_responses_dir"],
                f"raw_{model_config.id}_{condition.id}.jsonl",
            )
            if not os.path.exists(raw_path):
                continue

            logger.info(
                f"Scoring for model={model_config.id}, condition={condition.id}"
            )

            raw_rows: List[Dict[str, Any]] = []
            with open(raw_path, "r", encoding="utf-8") as f:
                for line in f:
                    raw_rows.append(json.loads(line))

            scored_rows = score_raw_responses(
                raw_rows=raw_rows,
                questions=all_questions,
                judge_config=exp_cfg.judge,
                judge_clients=judge_clients,
            )

            scored_path = os.path.join(
                exp_cfg.output["scored_dir"],
                f"scored_{model_config.id}_{condition.id}.jsonl",
            )
            write_jsonl(scored_path, scored_rows)
            all_scored_rows.extend(scored_rows)

    # 3. Aggregate metrics
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

    logger.info("Experiment completed")
