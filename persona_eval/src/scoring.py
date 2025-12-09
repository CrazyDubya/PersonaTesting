"""
Scoring module for evaluating model responses.

Handles:
- MCQ scoring (letter matching)
- Open-ended scoring (using judge model)
- Refusal detection
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from .data_models import Question, JudgeConfig, ModelConfig
from .model_api import LLMClient
from .utils.io_utils import write_jsonl


# Keywords that indicate a refusal or inability to answer
REFUSAL_KEYWORDS = [
    "i am not able to",
    "i cannot answer",
    "i can't answer",
    "i'm not able to",
    "i am unable to",
    "i'm unable to",
    "as a language model",
    "as an ai",
    "as an artificial intelligence",
    "i do not have enough information",
    "i don't have enough information",
    "i'm not an expert",
    "i am not an expert",
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "i cannot provide",
    "i can't provide",
    "i'm not qualified",
    "i am not qualified",
    "beyond my capabilities",
    "outside my knowledge",
    "i refuse to",
    "i will not answer",
    "i won't answer",
]


def detect_refusal(text: str) -> bool:
    """
    Detect if the response indicates a refusal or inability to answer.

    Args:
        text: Raw model response.

    Returns:
        True if the response appears to be a refusal.
    """
    lower = text.lower()
    for kw in REFUSAL_KEYWORDS:
        if kw in lower:
            return True
    return False


def build_open_judge_prompt(
    question_text: str,
    gold_answer_text: str,
    model_answer_text: str,
) -> List[Dict[str, str]]:
    """
    Build messages for the judge model to decide correctness.

    Args:
        question_text: The original question.
        gold_answer_text: The correct/reference answer.
        model_answer_text: The model's predicted answer.

    Returns:
        List of message dicts for the judge model.
    """
    system = {
        "role": "system",
        "content": (
            "You are a strict grading assistant. "
            "You decide whether a student's final answer is correct "
            "given the question and the official correct answer. "
            "Be fair but strict: minor wording differences are acceptable "
            "if the meaning is the same, but incorrect facts or significant "
            "omissions should be marked incorrect."
        ),
    }

    user_lines: List[str] = [
        "Question:",
        question_text,
        "",
        "Official correct answer:",
        gold_answer_text,
        "",
        "Student's final answer:",
        model_answer_text,
        "",
        "Respond with exactly one word: 'correct' if the student's answer "
        "is essentially correct, or 'incorrect' otherwise. Do not explain.",
    ]

    user = {
        "role": "user",
        "content": "\n".join(user_lines),
    }

    return [system, user]


def judge_open_answer_correct(
    question: Question,
    predicted_answer_text: str,
    judge_config: JudgeConfig,
    judge_client: LLMClient,
) -> bool:
    """
    Use a judge model to decide correctness of an open-ended answer.

    Args:
        question: The question being judged.
        predicted_answer_text: The model's predicted answer.
        judge_config: Configuration for the judge model.
        judge_client: LLM client for the judge model.

    Returns:
        True if the judge determines the answer is correct.
    """
    if question.correct_answer_text is None:
        # No gold text; cannot judge
        return False

    if not predicted_answer_text.strip():
        # Empty answer is incorrect
        return False

    messages = build_open_judge_prompt(
        question_text=question.question_text,
        gold_answer_text=question.correct_answer_text,
        model_answer_text=predicted_answer_text,
    )

    response_text = judge_client.generate(
        messages=messages,
        temperature=judge_config.temperature,
        max_tokens=judge_config.max_tokens,
    )

    cleaned = response_text.strip().lower()

    # Check for explicit correct/incorrect
    if cleaned == "correct":
        return True
    if cleaned == "incorrect":
        return False

    # Handle variations
    if "correct" in cleaned and "incorrect" not in cleaned:
        return True

    return False


def build_question_lookup(
    questions: List[Question],
) -> Dict[Tuple[str, str], Question]:
    """
    Build a lookup dictionary mapping (dataset, id) to Question.

    Args:
        questions: List of questions.

    Returns:
        Dictionary for fast question lookup.
    """
    lookup: Dict[Tuple[str, str], Question] = {}
    for q in questions:
        key = (q.dataset, q.id)
        lookup[key] = q
    return lookup


def score_mcq_answer(
    predicted_letter: str,
    correct_letter: Optional[str],
) -> bool:
    """
    Score an MCQ answer by comparing letters.

    Args:
        predicted_letter: The predicted letter (A-J).
        correct_letter: The correct letter.

    Returns:
        True if the answer is correct.
    """
    if not correct_letter:
        return False
    if not predicted_letter:
        return False
    return predicted_letter.upper() == correct_letter.upper()


def score_raw_responses(
    raw_rows: List[Dict[str, Any]],
    questions: List[Question],
    judge_config: JudgeConfig,
    judge_clients: Dict[str, LLMClient],
    progress_callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """
    Add correctness labels to raw response rows.

    Args:
        raw_rows: List of raw response dicts.
        questions: List of all questions (for lookup).
        judge_config: Configuration for judge model.
        judge_clients: Dict of model_id -> LLMClient for judging.
        progress_callback: Optional callback(current, total) for progress.

    Returns:
        List of scored response dicts with is_correct and is_refusal added.
    """
    lookup = build_question_lookup(questions)
    scored_rows: List[Dict[str, Any]] = []
    total_rows = len(raw_rows)

    for i, row in enumerate(raw_rows):
        key = (row["dataset"], str(row["question_id"]))
        question = lookup.get(key)

        if question is None:
            # Question not found; skip
            print(f"Warning: Question not found: {key}")
            continue

        predicted_letter = row.get("predicted_option_letter", "")
        predicted_text = row.get("predicted_answer_text", "")
        raw_response = row.get("raw_response", "")

        refusal_flag = detect_refusal(raw_response)
        is_correct = False

        if question.correct_option_letter:
            # MCQ scoring
            is_correct = score_mcq_answer(
                predicted_letter=predicted_letter,
                correct_letter=question.correct_option_letter,
            )
        else:
            # Open-ended scoring using judge
            if judge_config.enabled and judge_config.judge_model_id in judge_clients:
                judge_client = judge_clients[judge_config.judge_model_id]
                is_correct = judge_open_answer_correct(
                    question=question,
                    predicted_answer_text=predicted_text,
                    judge_config=judge_config,
                    judge_client=judge_client,
                )

        scored_row = dict(row)
        scored_row["is_correct"] = is_correct
        scored_row["is_refusal"] = refusal_flag
        scored_row["correct_option_letter"] = question.correct_option_letter
        scored_row["correct_answer_text"] = question.correct_answer_text
        scored_rows.append(scored_row)

        if progress_callback:
            progress_callback(i + 1, total_rows)

    return scored_rows


def compute_accuracy(scored_rows: List[Dict[str, Any]]) -> float:
    """
    Compute simple accuracy from scored rows.

    Args:
        scored_rows: List of scored response dicts.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    if not scored_rows:
        return 0.0
    correct_count = sum(1 for row in scored_rows if row.get("is_correct", False))
    return correct_count / len(scored_rows)


def compute_refusal_rate(scored_rows: List[Dict[str, Any]]) -> float:
    """
    Compute refusal rate from scored rows.

    Args:
        scored_rows: List of scored response dicts.

    Returns:
        Refusal rate as a float between 0 and 1.
    """
    if not scored_rows:
        return 0.0
    refusal_count = sum(1 for row in scored_rows if row.get("is_refusal", False))
    return refusal_count / len(scored_rows)
