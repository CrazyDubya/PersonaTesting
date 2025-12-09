import re
from typing import List, Dict, Any, Tuple
from .data_models import Question, JudgeConfig, ModelConfig
from .model_api import LLMClient
from .utils.io_utils import write_jsonl


def detect_refusal(text: str) -> bool:
    """
    Heuristic refusal detection.
    """
    lower = text.lower()
    keywords = [
        "i am not able to",
        "i cannot answer",
        "i can't answer",
        "as a language model",
        "as an ai",
        "i do not have enough information",
        "i'm not an expert",
        "i am not an expert",
    ]
    for kw in keywords:
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
    """
    system = {
        "role": "system",
        "content": (
            "You are a strict grading assistant. "
            "You decide whether a student's final answer is correct "
            "given the question and the official correct answer."
        ),
    }

    user_lines: List[str] = []
    user_lines.append("Question:")
    user_lines.append(question_text)
    user_lines.append("")
    user_lines.append("Official correct answer:")
    user_lines.append(gold_answer_text)
    user_lines.append("")
    user_lines.append("Student's final answer:")
    user_lines.append(model_answer_text)
    user_lines.append("")
    user_lines.append(
        "Respond with exactly one word: 'correct' if the student's answer "
        "is essentially correct, or 'incorrect' otherwise."
    )

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
    """
    if question.correct_answer_text is None:
        # No gold text; cannot judge
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
    if "correct" in cleaned and "incorrect" not in cleaned:
        return True
    if cleaned == "correct":
        return True
    return False


def build_question_lookup(
    questions: List[Question],
) -> Dict[Tuple[str, str], Question]:
    """
    Map (dataset, id) to Question.
    """
    lookup: Dict[Tuple[str, str], Question] = {}
    for q in questions:
        key = (q.dataset, q.id)
        lookup[key] = q
    return lookup


def score_raw_responses(
    raw_rows: List[Dict[str, Any]],
    questions: List[Question],
    judge_config: JudgeConfig,
    judge_clients: Dict[str, LLMClient],
) -> List[Dict[str, Any]]:
    """
    Add correctness labels to raw response rows.
    """
    lookup = build_question_lookup(questions)
    scored_rows: List[Dict[str, Any]] = []

    for row in raw_rows:
        key = (row["dataset"], str(row["question_id"]))
        question = lookup[key]

        condition_type = row["condition_id"].split("_")[0]  # "baseline", etc. (not used heavily)
        predicted_letter = row["predicted_option_letter"]
        predicted_text = row["predicted_answer_text"]
        raw_response = row["raw_response"]

        refusal_flag = detect_refusal(raw_response)

        is_correct = False

        if question.correct_option_letter:
            # MCQ scoring
            if predicted_letter and predicted_letter == question.correct_option_letter:
                is_correct = True
        else:
            # open-ended scoring
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
        scored_rows.append(scored_row)

    return scored_rows
