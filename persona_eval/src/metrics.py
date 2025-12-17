"""
Metrics module for computing aggregate statistics.

Computes per (dataset, model, condition) metrics:
- Overall accuracy
- Question-level robust correctness thresholds
- Refusal rate
- Average reasoning token length
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import math


def compute_confidence_interval(
    accuracy: float,
    n_samples: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.

    Args:
        accuracy: Observed accuracy (proportion).
        n_samples: Number of samples.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if n_samples == 0:
        return (0.0, 0.0)
    
    # Clamp accuracy to valid range
    accuracy = max(0.0, min(1.0, accuracy))

    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    # Wilson score interval
    denominator = 1 + z * z / n_samples
    center = (accuracy + z * z / (2 * n_samples)) / denominator
    spread = z * math.sqrt(accuracy * (1 - accuracy) / n_samples + z * z / (4 * n_samples * n_samples)) / denominator

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return (lower, upper)


def compute_metrics(
    scored_rows: List[Dict[str, Any]],
    num_samples_per_question: int,
) -> List[Dict[str, Any]]:
    """
    Compute per (dataset, model, condition) metrics.

    Metrics include:
    - overall accuracy
    - question-level robust correctness thresholds:
      * fraction of questions with all samples correct (N/N)
      * fraction with >= 23/25
      * fraction with >= 13/25
    - average refusal rate
    - average reasoning token length

    Args:
        scored_rows: List of scored response dicts.
        num_samples_per_question: Number of samples per question.

    Returns:
        List of summary metric dicts.
    """
    # Group by dataset, model_id, condition_id, question_id
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        key = (
            row["dataset"],
            row["model_id"],
            row["condition_id"],
            str(row["question_id"]),
        )
        groups[key].append(row)

    # Question-level stats aggregated by (dataset, model, condition)
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
            "subjects": defaultdict(lambda: {"correct": 0, "total": 0}),
            "difficulties": defaultdict(lambda: {"correct": 0, "total": 0}),
        }
    )

    for key, rows in groups.items():
        dataset, model_id, condition_id, question_id = key
        aggregate_key = (dataset, model_id, condition_id)

        correct_count = sum(1 for r in rows if r.get("is_correct", False))
        refusal_count = sum(1 for r in rows if r.get("is_refusal", False))
        token_sum = sum(r.get("reasoning_token_estimate", 0) for r in rows)
        total_samples = len(rows)

        # Get subject and difficulty from first row
        subject = rows[0].get("subject") if rows else None
        difficulty = rows[0].get("difficulty") if rows else None

        q_stats = question_stats[aggregate_key]
        q_stats["num_questions"] += 1
        q_stats["num_samples"] += total_samples
        q_stats["num_correct_answers"] += correct_count
        q_stats["num_refusals"] += refusal_count
        q_stats["sum_reasoning_tokens"] += token_sum

        # Track by subject
        if subject:
            q_stats["subjects"][subject]["correct"] += correct_count
            q_stats["subjects"][subject]["total"] += total_samples

        # Track by difficulty
        if difficulty:
            q_stats["difficulties"][difficulty]["correct"] += correct_count
            q_stats["difficulties"][difficulty]["total"] += total_samples

        # Robust correctness thresholds
        if correct_count == total_samples:
            q_stats["num_with_all_correct"] += 1
        if correct_count >= min(23, total_samples):
            q_stats["num_with_ge_23_correct"] += 1
        if correct_count >= min(13, total_samples):
            q_stats["num_with_ge_13_correct"] += 1

    # Build summary rows
    summary_rows: List[Dict[str, Any]] = []

    for aggregate_key, q_stats in question_stats.items():
        dataset, model_id, condition_id = aggregate_key
        num_questions = q_stats["num_questions"]
        num_samples = q_stats["num_samples"]

        if num_samples == 0:
            continue

        mean_accuracy = q_stats["num_correct_answers"] / float(num_samples)
        mean_refusal_rate = q_stats["num_refusals"] / float(num_samples)
        mean_reasoning_tokens = q_stats["sum_reasoning_tokens"] / float(num_samples)

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

        # Compute confidence intervals
        ci_lower, ci_upper = compute_confidence_interval(mean_accuracy, num_samples)

        # Compute per-subject accuracy
        subject_accuracies = {}
        for subject, stats in q_stats["subjects"].items():
            if stats["total"] > 0:
                subject_accuracies[subject] = stats["correct"] / stats["total"]

        # Compute per-difficulty accuracy
        difficulty_accuracies = {}
        for difficulty, stats in q_stats["difficulties"].items():
            if stats["total"] > 0:
                difficulty_accuracies[difficulty] = stats["correct"] / stats["total"]

        summary_rows.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "condition_id": condition_id,
                "num_questions": num_questions,
                "num_samples": num_samples,
                "mean_accuracy": mean_accuracy,
                "accuracy_ci_lower": ci_lower,
                "accuracy_ci_upper": ci_upper,
                "mean_refusal_rate": mean_refusal_rate,
                "mean_reasoning_tokens": mean_reasoning_tokens,
                "frac_questions_all_samples_correct": frac_all_correct,
                "frac_questions_ge_23_correct": frac_ge_23_correct,
                "frac_questions_ge_13_correct": frac_ge_13_correct,
                "subject_accuracies": subject_accuracies,
                "difficulty_accuracies": difficulty_accuracies,
            }
        )

    return summary_rows


def compute_condition_comparison(
    summary_rows: List[Dict[str, Any]],
    baseline_condition_id: str = "baseline_mc",
) -> List[Dict[str, Any]]:
    """
    Compute relative performance compared to baseline condition.

    Args:
        summary_rows: List of summary metric dicts.
        baseline_condition_id: ID of the baseline condition to compare against.

    Returns:
        List of comparison dicts with delta values.
    """
    # Index baseline metrics by (dataset, model_id)
    baseline_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in summary_rows:
        if row["condition_id"] == baseline_condition_id:
            key = (row["dataset"], row["model_id"])
            baseline_metrics[key] = row

    comparison_rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        key = (row["dataset"], row["model_id"])
        baseline = baseline_metrics.get(key)

        comparison = {
            "dataset": row["dataset"],
            "model_id": row["model_id"],
            "condition_id": row["condition_id"],
            "mean_accuracy": row["mean_accuracy"],
        }

        if baseline:
            comparison["delta_accuracy"] = row["mean_accuracy"] - baseline["mean_accuracy"]
            comparison["relative_accuracy"] = (
                row["mean_accuracy"] / baseline["mean_accuracy"]
                if baseline["mean_accuracy"] > 0
                else float('inf')
            )
            comparison["baseline_accuracy"] = baseline["mean_accuracy"]
        else:
            comparison["delta_accuracy"] = 0.0
            comparison["relative_accuracy"] = 1.0
            comparison["baseline_accuracy"] = row["mean_accuracy"]

        comparison_rows.append(comparison)

    return comparison_rows


def format_summary_table(summary_rows: List[Dict[str, Any]]) -> str:
    """
    Format summary metrics as a readable table.

    Args:
        summary_rows: List of summary metric dicts.

    Returns:
        Formatted table string.
    """
    if not summary_rows:
        return "No data available."

    # Sort by dataset, model, condition
    sorted_rows = sorted(
        summary_rows,
        key=lambda r: (r["dataset"], r["model_id"], r["condition_id"])
    )

    lines = [
        "=" * 100,
        f"{'Dataset':<12} {'Model':<15} {'Condition':<25} {'Accuracy':>10} {'CI 95%':>15} {'Refusal':>8}",
        "=" * 100,
    ]

    for row in sorted_rows:
        ci_str = f"[{row['accuracy_ci_lower']:.3f}, {row['accuracy_ci_upper']:.3f}]"
        line = (
            f"{row['dataset']:<12} "
            f"{row['model_id']:<15} "
            f"{row['condition_id']:<25} "
            f"{row['mean_accuracy']:>10.4f} "
            f"{ci_str:>15} "
            f"{row['mean_refusal_rate']:>8.4f}"
        )
        lines.append(line)

    lines.append("=" * 100)
    return "\n".join(lines)
