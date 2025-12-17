from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class Question:
    dataset: str           # "gpqa" or "mmlu_pro"
    id: str                # unique question identifier
    question_text: str
    options: List[str]     # ["A. ...", "B. ...", ...] if MCQ; [] if open
    correct_option_letter: Optional[str]  # "A", "B", ... for MCQ
    correct_answer_text: Optional[str]    # canonical text answer (for open grading)
    subject: Optional[str]                # e.g. "physics", "law"
    difficulty: Optional[str]             # e.g. "easy", "post-grad"
    metadata: Dict[str, Any]              # arbitrary extra info


@dataclass
class ConditionConfig:
    id: str
    type: str                # "mcq" or "open"
    persona_type: str        # "none", "expert_shallow", "expert_deep", "toddler"
    reasoning_mode: str      # "short" or "long"
    min_reasoning_tokens: Optional[int]
    use_multiple_choice: bool
    answer_format: str       # "the_correct_answer_is" | "final_answer_letter" | "final_answer_text"


@dataclass
class ModelConfig:
    id: str
    provider: str
    model_name: str
    max_output_tokens: int
    default_temperature: float


@dataclass
class JudgeConfig:
    enabled: bool
    judge_model_id: str
    temperature: float
    max_tokens: int


@dataclass
class SamplingConfig:
    num_samples_per_question: int
    temperature: float
    max_tokens_reasoning: int


@dataclass
class ExperimentConfig:
    datasets: List[Dict[str, Any]]
    models: List[ModelConfig]
    conditions: List[ConditionConfig]
    sampling: SamplingConfig
    judge: JudgeConfig
    output: Dict[str, str]
