import yaml
from typing import Dict, Any, List
from .data_models import (
    ExperimentConfig, ModelConfig, ConditionConfig,
    SamplingConfig, JudgeConfig
)


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_experiment_config(raw: Dict[str, Any]) -> ExperimentConfig:
    """Build an ExperimentConfig from raw YAML data."""
    models = [
        ModelConfig(
            id=m["id"],
            provider=m["provider"],
            model_name=m["model_name"],
            max_output_tokens=m["max_output_tokens"],
            default_temperature=m["default_temperature"],
            api_base=m.get("api_base"),
        )
        for m in raw["models"]
    ]

    conditions = [
        ConditionConfig(
            id=c["id"],
            type=c["type"],
            persona_type=c["persona_type"],
            reasoning_mode=c["reasoning_mode"],
            min_reasoning_tokens=c.get("min_reasoning_tokens"),
            use_multiple_choice=c["use_multiple_choice"],
            answer_format=c["answer_format"],
        )
        for c in raw["conditions"]
    ]

    sampling_raw = raw["sampling"]
    sampling = SamplingConfig(
        num_samples_per_question=sampling_raw["num_samples_per_question"],
        temperature=sampling_raw["temperature"],
        max_tokens_reasoning=sampling_raw["max_tokens_reasoning"],
    )

    judge_raw = raw["judge"]
    judge = JudgeConfig(
        enabled=judge_raw["enabled"],
        judge_model_id=judge_raw["judge_model_id"],
        temperature=judge_raw["temperature"],
        max_tokens=judge_raw["max_tokens"],
    )

    return ExperimentConfig(
        datasets=raw["datasets"],
        models=models,
        conditions=conditions,
        sampling=sampling,
        judge=judge,
        output=raw["output"],
    )
