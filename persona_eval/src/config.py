import yaml
from typing import Dict, Any, List
from .data_models import (
    ExperimentConfig, ModelConfig, ConditionConfig,
    SamplingConfig, JudgeConfig
)


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError(f"Config file {path} is empty or invalid")
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file {path}: {e}")


def build_experiment_config(raw: Dict[str, Any]) -> ExperimentConfig:
    """Build an ExperimentConfig from raw YAML data."""
    # Validate required top-level keys
    required_keys = ["models", "conditions", "sampling", "judge", "datasets", "output"]
    missing_keys = [key for key in required_keys if key not in raw]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")
    
    # Validate models
    if not raw["models"]:
        raise ValueError("At least one model must be configured")
    
    models = []
    for m in raw["models"]:
        required_model_fields = ["id", "provider", "model_name", "max_output_tokens", "default_temperature"]
        missing_fields = [field for field in required_model_fields if field not in m]
        if missing_fields:
            raise ValueError(f"Model configuration missing fields: {', '.join(missing_fields)}")
        models.append(ModelConfig(
            id=m["id"],
            provider=m["provider"],
            model_name=m["model_name"],
            max_output_tokens=m["max_output_tokens"],
            default_temperature=m["default_temperature"],
            api_base=m.get("api_base"),
        ))

    # Validate conditions
    if not raw["conditions"]:
        raise ValueError("At least one condition must be configured")
    
    conditions = []
    for c in raw["conditions"]:
        required_condition_fields = ["id", "type", "persona_type", "reasoning_mode", "use_multiple_choice", "answer_format"]
        missing_fields = [field for field in required_condition_fields if field not in c]
        if missing_fields:
            raise ValueError(f"Condition configuration missing fields: {', '.join(missing_fields)}")
        conditions.append(ConditionConfig(
            id=c["id"],
            type=c["type"],
            persona_type=c["persona_type"],
            reasoning_mode=c["reasoning_mode"],
            min_reasoning_tokens=c.get("min_reasoning_tokens"),
            use_multiple_choice=c["use_multiple_choice"],
            answer_format=c["answer_format"],
        ))

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
