from typing import List, Dict, Any, Tuple
from collections import defaultdict
import math
from .utils.io_utils import write_jsonl


def compute_metrics(
    scored_rows: List[Dict[str, Any]],
    num_samples_per_question: int,
) -> List[Dict[str, Any]]:
    """
    Compute per (dataset, model, condition) metrics:
    - overall accuracy
    - question-level robust correctness thresholds:
      * fraction of questions with all samples correct (N/N)
      * fraction with >= 23/25
      * fraction with >= 13/25
    - average refusal rate
    - average reasoning token length
    """

    # group by dataset, model_id, condition_id, question_id
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        key = (
            row["dataset"],
            row["model_id"],
            row["condition_id"],
            str(row["question_id"]),
        )
        groups[key].append(row)

    # question-level stats
    question_stats: Dict[Tuple[str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "num_questions": 0,
            "num_correct_answers": 0,
            "num_samples": 0,
            "num_refusals": 0,
            "sum_reasoning_tokens": 0,
            "num_with_all_correct": 0,
            "num_with_ge_23_correct": 0,
            "num_with_ge_13_correct": 0,
        }
    )

    for key, rows in groups.items():
        dataset, model_id, condition_id, question_id = key
        aggregate_key = (dataset, model_id, condition_id)

        correct_count = sum(1 for r in rows if r["is_correct"])
        refusal_count = sum(1 for r in rows if r["is_refusal"])
        token_sum = sum(r["reasoning_token_estimate"] for r in rows)
        total_samples = len(rows)

        q_stats = question_stats[aggregate_key]
        q_stats["num_questions"] += 1
        q_stats["num_samples"] += total_samples
        q_stats["num_correct_answers"] += correct_count
        q_stats["num_refusals"] += refusal_count
        q_stats["sum_reasoning_tokens"] += token_sum

        if correct_count == total_samples:
            q_stats["num_with_all_correct"] += 1
        if correct_count >= min(23, total_samples):
            q_stats["num_with_ge_23_correct"] += 1
        if correct_count >= min(13, total_samples):
            q_stats["num_with_ge_13_correct"] += 1

    # summary rows
    summary_rows: List[Dict[str, Any]] = []

    for aggregate_key, q_stats in question_stats.items():
        dataset, model_id, condition_id = aggregate_key
        num_questions = q_stats["num_questions"]
        num_samples = q_stats["num_samples"]

        mean_accuracy = q_stats["num_correct_answers"] / float(num_samples)
        mean_refusal_rate = q_stats["num_refusals"] / float(num_samples)
        mean_reasoning_tokens = (
            q_stats["sum_reasoning_tokens"] / float(num_samples)
            if num_samples > 0
            else 0.0
        )

        frac_all_correct = (
            q_stats["num_with_all_correct"] / float(num_questions)
            if num_questions > 0
            else 0.0
        )
        frac_ge_23_correct = (
            q_stats["num_with_ge_23_correct"] / float(num_questions)
            if num_questions > 0
            else 0.0
        )
        frac_ge_13_correct = (
            q_stats["num_with_ge_13_correct"] / float(num_questions)
            if num_questions > 0
            else 0.0
        )

        summary_rows.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "condition_id": condition_id,
                "num_questions": num_questions,
                "num_samples": num_samples,
                "mean_accuracy": mean_accuracy,
                "mean_refusal_rate": mean_refusal_rate,
                "mean_reasoning_tokens": mean_reasoning_tokens,
                "frac_questions_all_samples_correct": frac_all_correct,
                "frac_questions_ge_23_correct": frac_ge_23_correct,
                "frac_questions_ge_13_correct": frac_ge_13_correct,
            }
        )

    return summary_rows
