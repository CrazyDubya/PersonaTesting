import re
import time
from typing import List, Dict, Any
from .data_models import Question, ConditionConfig, ModelConfig
from .prompts import build_chat_messages
from .model_api import LLMClient
from .utils.io_utils import write_jsonl


def parse_answer_mcq_short_format(text: str) -> str:
    """
    For 'The correct answer is X' style.
    Returns the inferred letter, uppercase.
    """
    # Strip whitespace
    cleaned = text.strip()

    # Regex patterns to cover variations
    patterns = [
        r"[Tt]he correct answer is\s*[:\-]?\s*([A-J])\b",
        r"[Tt]he correct answer is\s*[:\-]?\s*\"?([A-J])\"?\b",
    ]

    for pattern in patterns:
        m = re.search(pattern, cleaned)
        if m:
            return m.group(1).upper()

    # Fallback: extract first standalone letter from A-J
    fallback_match = re.search(r"\b([A-J])\b", cleaned)
    if fallback_match:
        return fallback_match.group(1).upper()

    return ""


def parse_answer_mcq_final_answer(text: str) -> str:
    """
    For 'Final answer: X' style, where X is a letter.
    """
    cleaned = text.strip()
    m = re.search(r"[Ff]inal answer\s*[:\-]\s*([A-J])\b", cleaned)
    if m:
        return m.group(1).upper()
    # Fallback: search for [A-J] near the end
    last_line = cleaned.splitlines()[-1]
    fallback_match = re.search(r"\b([A-J])\b", last_line)
    if fallback_match:
        return fallback_match.group(1).upper()
    return ""


def parse_answer_open_final_answer(text: str) -> str:
    """
    For open-ended 'Final answer: ...' style.
    Returns the substring after 'Final answer:' on that line.
    """
    cleaned = text.strip()
    lines = cleaned.splitlines()
    for line in lines:
        m = re.search(r"[Ff]inal answer\s*[:\-]\s*(.+)", line)
        if m:
            return m.group(1).strip()
    # Fallback: use the last line
    if lines:
        return lines[-1].strip()
    return ""


def approximate_token_count(text: str) -> int:
    """
    Crude token count approximation: 1 token ~ 0.75 words.
    """
    words = text.split()
    word_count = len(words)
    approx_tokens = int(word_count / 0.75)
    return approx_tokens


def split_reasoning_and_final_answer(text: str) -> Dict[str, str]:
    """
    Split the model's output into reasoning part and final answer line.
    """
    lines = text.strip().splitlines()
    reasoning_lines: List[str] = []
    final_line = ""

    for line in lines:
        if re.search(r"[Ff]inal answer\s*[:\-]", line):
            final_line = line
            break
        reasoning_lines.append(line)

    reasoning_text = "\n".join(reasoning_lines).strip()
    return {"reasoning": reasoning_text, "final_line": final_line}


def run_sampling_for_condition_and_model(
    questions: List[Question],
    condition: ConditionConfig,
    model_config: ModelConfig,
    client: LLMClient,
    num_samples_per_question: int,
    temperature: float,
    max_tokens_reasoning: int,
    raw_output_path: str,
) -> None:
    """
    Generate responses for every question under a given condition and model.
    Writes JSONL containing all responses.
    """
    rows: List[Dict[str, Any]] = []

    for q in questions:
        messages = build_chat_messages(q, condition)

        for sample_idx in range(num_samples_per_question):
            # Call model
            response_text = client.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens_reasoning,
            )

            reasoning_and_final = split_reasoning_and_final_answer(response_text)
            reasoning_text = reasoning_and_final["reasoning"]
            final_line = reasoning_and_final["final_line"]

            reasoning_tokens = approximate_token_count(reasoning_text)

            if condition.type == "mcq":
                if condition.answer_format == "the_correct_answer_is":
                    predicted_letter = parse_answer_mcq_short_format(response_text)
                else:
                    predicted_letter = parse_answer_mcq_final_answer(response_text)
                predicted_text = predicted_letter
            else:
                predicted_text = parse_answer_open_final_answer(response_text)
                predicted_letter = ""

            row: Dict[str, Any] = {
                "dataset": q.dataset,
                "question_id": q.id,
                "subject": q.subject,
                "difficulty": q.difficulty,
                "condition_id": condition.id,
                "model_id": model_config.id,
                "sample_idx": sample_idx,
                "prompt_messages": messages,
                "raw_response": response_text,
                "parsed_reasoning": reasoning_text,
                "final_line": final_line,
                "predicted_option_letter": predicted_letter,
                "predicted_answer_text": predicted_text,
                "reasoning_token_estimate": reasoning_tokens,
                "timestamp": time.time(),
            }
            rows.append(row)

    write_jsonl(raw_output_path, rows)
