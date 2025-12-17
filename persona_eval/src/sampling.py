"""
Sampling module for generating and parsing model responses.

Handles:
- Running multiple samples per question
- Parsing different answer formats
- Estimating reasoning token counts
- Saving raw responses to JSONL
"""

import re
import time
from typing import List, Dict, Any, Optional
from .data_models import Question, ConditionConfig, ModelConfig
from .prompts import build_chat_messages
from .model_api import LLMClient
from .utils.io_utils import write_jsonl


def parse_answer_mcq_short_format(text: str) -> str:
    """
    Parse answer from 'The correct answer is X' style format.

    Args:
        text: Raw model response text.

    Returns:
        Uppercase letter (A-J) or empty string if not found.
    """
    cleaned = text.strip()

    # Regex patterns to cover variations
    patterns = [
        r"[Tt]he correct answer is\s*[:\-]?\s*\(?([A-J])\)?\b",
        r"[Tt]he correct answer is\s*[:\-]?\s*\"?\(?([A-J])\)?\"?\b",
        r"[Cc]orrect answer\s*[:\-]?\s*\(?([A-J])\)?\b",
        r"[Aa]nswer\s*[:\-]?\s*\(?([A-J])\)?\b",
    ]

    for pattern in patterns:
        m = re.search(pattern, cleaned)
        if m:
            return m.group(1).upper()

    # Fallback: extract first standalone letter from A-J at the end
    last_lines = cleaned.splitlines()[-3:] if len(cleaned.splitlines()) >= 3 else cleaned.splitlines()
    for line in reversed(last_lines):
        fallback_match = re.search(r"\b([A-J])\b", line)
        if fallback_match:
            return fallback_match.group(1).upper()

    return ""


def parse_answer_mcq_final_answer(text: str) -> str:
    """
    Parse answer from 'Final answer: X' style format, where X is a letter.

    Args:
        text: Raw model response text.

    Returns:
        Uppercase letter (A-J) or empty string if not found.
    """
    cleaned = text.strip()

    # Primary pattern
    m = re.search(r"[Ff]inal answer\s*[:\-]\s*\(?([A-J])\)?\b", cleaned)
    if m:
        return m.group(1).upper()

    # Alternative patterns
    m = re.search(r"[Ff]inal [Aa]nswer\s*[:\-]\s*\(?([A-J])\)?\b", cleaned)
    if m:
        return m.group(1).upper()

    # Fallback: search for [A-J] near the end
    lines = cleaned.splitlines()
    for line in reversed(lines[-5:] if len(lines) >= 5 else lines):
        if re.search(r"[Ff]inal", line, re.IGNORECASE):
            fallback_match = re.search(r"\b([A-J])\b", line)
            if fallback_match:
                return fallback_match.group(1).upper()

    # Last resort: last standalone letter
    if lines:
        last_line = lines[-1]
        fallback_match = re.search(r"\b([A-J])\b", last_line)
        if fallback_match:
            return fallback_match.group(1).upper()

    return ""


def parse_answer_open_final_answer(text: str) -> str:
    """
    Parse answer from open-ended 'Final answer: ...' style format.

    Args:
        text: Raw model response text.

    Returns:
        The text after 'Final answer:' or the last line as fallback.
    """
    cleaned = text.strip()
    lines = cleaned.splitlines()

    for line in lines:
        m = re.search(r"[Ff]inal answer\s*[:\-]\s*(.+)", line)
        if m:
            answer = m.group(1).strip()
            # Remove trailing punctuation that might be formatting
            answer = re.sub(r'[.!?]+$', '', answer).strip()
            return answer

    # Fallback: use the last non-empty line
    for line in reversed(lines):
        if line.strip():
            return line.strip()

    return ""


def approximate_token_count(text: str) -> int:
    """
    Approximate token count based on word count.
    Rough estimate: 1 token ~ 0.75 words (or ~4 characters).

    Args:
        text: Text to estimate tokens for.

    Returns:
        Approximate token count.
    """
    if not text:
        return 0
    
    words = text.split()
    word_count = len(words)
    # More accurate estimate using both words and characters
    char_count = len(text)
    word_based = int(word_count / 0.75) if word_count > 0 else 0
    char_based = int(char_count / 4)
    # Average the two estimates
    return max(1, (word_based + char_based) // 2)


def split_reasoning_and_final_answer(text: str) -> Dict[str, str]:
    """
    Split the model's output into reasoning part and final answer line.

    Args:
        text: Raw model response.

    Returns:
        Dict with 'reasoning' and 'final_line' keys.
    """
    lines = text.strip().splitlines()
    reasoning_lines: List[str] = []
    final_line = ""

    for i, line in enumerate(lines):
        if re.search(r"[Ff]inal answer\s*[:\-]", line):
            final_line = line
            # Everything before this is reasoning
            reasoning_lines = lines[:i]
            break
        reasoning_lines.append(line)

    if not final_line and lines:
        # No explicit final answer line found
        # Check for "The correct answer is" pattern
        for i, line in enumerate(lines):
            if re.search(r"[Tt]he correct answer is", line):
                final_line = line
                reasoning_lines = lines[:i]
                break

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
    progress_callback: Optional[callable] = None,
) -> None:
    """
    Generate responses for every question under a given condition and model.
    Writes JSONL containing all responses.

    Args:
        questions: List of questions to sample.
        condition: Experimental condition configuration.
        model_config: Model configuration.
        client: LLM client to use.
        num_samples_per_question: Number of samples per question.
        temperature: Sampling temperature.
        max_tokens_reasoning: Max tokens for model output.
        raw_output_path: Path to write raw responses JSONL.
        progress_callback: Optional callback(current, total) for progress.
    """
    rows: List[Dict[str, Any]] = []
    total_samples = len(questions) * num_samples_per_question
    current_sample = 0

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

            # Parse based on condition type and answer format
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

            current_sample += 1
            if progress_callback:
                progress_callback(current_sample, total_samples)

    write_jsonl(raw_output_path, rows)


def run_single_question_sample(
    question: Question,
    condition: ConditionConfig,
    client: LLMClient,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """
    Run a single sample for one question.

    Args:
        question: The question to sample.
        condition: Experimental condition.
        client: LLM client.
        temperature: Sampling temperature.
        max_tokens: Max tokens for response.

    Returns:
        Dict containing response data.
    """
    messages = build_chat_messages(question, condition)
    response_text = client.generate(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
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

    return {
        "raw_response": response_text,
        "parsed_reasoning": reasoning_text,
        "final_line": final_line,
        "predicted_option_letter": predicted_letter,
        "predicted_answer_text": predicted_text,
        "reasoning_token_estimate": reasoning_tokens,
    }
