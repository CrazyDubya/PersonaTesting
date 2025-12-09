from typing import List, Dict
from .data_models import Question, ConditionConfig


BASE_SYSTEM_PROMPT = (
    "You are a very intelligent assistant, who follows instructions directly."
)


def get_persona_text(persona_type: str, domain: str) -> str:
    if persona_type == "none":
        return ""

    if persona_type == "expert_shallow":
        return (
            f"You are a world-class expert in {domain}. "
            f"You have deep knowledge of {domain} and always choose the most accurate answer."
        )

    if persona_type == "expert_deep":
        return (
            f"You are Dr. Alex Rivera, a world-class expert in {domain}. "
            f"You hold multiple advanced degrees and have spent over 20 years "
            f"researching and teaching {domain}. You are meticulous, analytical, and calm. "
            f"You always approach problems by carefully unpacking the question, "
            f"enumerating possibilities, checking each step, and only then committing to an answer. "
            f"You dislike hand-waving and insist on sound reasoning grounded in {domain} principles."
        )

    if persona_type == "toddler":
        return (
            "You are a 4-year-old child. You speak in very simple, short sentences. "
            "You often say that you do not know things. You do not understand advanced science. "
            "You may guess, but your guesses are not very reliable."
        )

    # default fallback
    return ""


def build_mcq_content_with_short_answer(
    question: Question,
    persona_type: str,
) -> str:
    domain = question.subject or "the relevant subject"
    persona_text = get_persona_text(persona_type, domain)

    options_block_lines: List[str] = []
    for opt in question.options:
        options_block_lines.append(opt)
    options_block = "\n".join(options_block_lines)

    # This matches the original paper-style “The correct answer is X”
    content_parts: List[str] = []
    if persona_text:
        content_parts.append(persona_text)
        content_parts.append("")
    content_parts.append("What is the correct answer to this question:")
    content_parts.append(question.question_text)
    content_parts.append("")
    content_parts.append("Choices:")
    content_parts.append(options_block)
    content_parts.append("")
    content_parts.append(
        'Format your response as follows: "The correct answer is (insert answer here)"'
    )

    return "\n".join(content_parts)


def build_mcq_content_with_preamble(
    question: Question,
    persona_type: str,
    min_reasoning_tokens: int,
) -> str:
    domain = question.subject or "the relevant subject"
    persona_text = get_persona_text(persona_type, domain)
    options_block_lines: List[str] = []
    for opt in question.options:
        options_block_lines.append(opt)
    options_block = "\n".join(options_block_lines)

    content_parts: List[str] = []
    if persona_text:
        content_parts.append(persona_text)
        content_parts.append("")

    content_parts.append(
        f"You will now answer a multiple-choice question in {domain}."
    )
    content_parts.append(
        "First, think out loud in detail as this persona. "
        f"Your reasoning should be at least {min_reasoning_tokens} tokens long. "
        "Do not rush to an answer. Examine the question, the options, and any relevant principles."
    )
    content_parts.append(
        "After you finish reasoning, on a new line write exactly: "
        '"Final answer: X" where X is the letter of the correct option (A, B, C, etc.).'
    )
    content_parts.append(
        "Do not include explanations after the final answer line."
    )
    content_parts.append("")
    content_parts.append("Question:")
    content_parts.append(question.question_text)
    content_parts.append("")
    content_parts.append("Choices:")
    content_parts.append(options_block)

    return "\n".join(content_parts)


def build_open_content_persona_long(
    question: Question,
    persona_type: str,
    min_reasoning_tokens: int,
) -> str:
    domain = question.subject or "the relevant subject"
    persona_text = get_persona_text(persona_type, domain)

    content_parts: List[str] = []
    if persona_text:
        content_parts.append(persona_text)
        content_parts.append("")

    content_parts.append(
        f"You will now answer a difficult open-ended question in {domain}."
    )
    content_parts.append(
        "As this persona, think through the problem carefully and in detail. "
        f"Your reasoning should be at least {min_reasoning_tokens} tokens long. "
        "Unpack the question, consider relevant concepts, and work step by step."
    )
    content_parts.append(
        "After you finish reasoning, on a new line write exactly: "
        '"Final answer: [your short final answer]" where the final answer is concise."'
    )
    content_parts.append(
        "Do not add any text after the final answer line."
    )
    content_parts.append("")
    content_parts.append("Question:")
    content_parts.append(question.question_text)

    return "\n".join(content_parts)


def build_open_content_process_only(
    question: Question,
    min_reasoning_tokens: int,
) -> str:
    content_parts: List[str] = []
    content_parts.append(
        "You will answer a difficult question. Think step by step."
    )
    content_parts.append(
        f"First, reason in detail for at least {min_reasoning_tokens} tokens. "
        "Break the problem into parts and solve each part carefully."
    )
    content_parts.append(
        "After you finish reasoning, on a new line write exactly: "
        '"Final answer: [your short final answer]".'
    )
    content_parts.append(
        "Do not add any text after the final answer line."
    )
    content_parts.append("")
    content_parts.append("Question:")
    content_parts.append(question.question_text)

    return "\n".join(content_parts)


def build_user_content_for_condition(
    question: Question,
    condition: ConditionConfig,
) -> str:
    if condition.type == "mcq":
        if condition.reasoning_mode == "short":
            return build_mcq_content_with_short_answer(
                question=question,
                persona_type=condition.persona_type,
            )
        else:
            return build_mcq_content_with_preamble(
                question=question,
                persona_type=condition.persona_type,
                min_reasoning_tokens=condition.min_reasoning_tokens or 200,
            )

    # open-ended
    if condition.type == "open":
        if condition.persona_type == "none":
            return build_open_content_process_only(
                question=question,
                min_reasoning_tokens=condition.min_reasoning_tokens or 200,
            )
        else:
            return build_open_content_persona_long(
                question=question,
                persona_type=condition.persona_type,
                min_reasoning_tokens=condition.min_reasoning_tokens or 200,
            )

    # fallback
    return question.question_text


def build_chat_messages(
    question: Question,
    condition: ConditionConfig,
) -> List[Dict[str, str]]:
    """
    Returns messages suitable for a ChatCompletion-style API.
    """
    user_content = build_user_content_for_condition(question, condition)
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
