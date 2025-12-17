# Persona Evaluation Framework

A framework for evaluating the effect of expert and low-knowledge personas on LLM performance under different experimental conditions.

## Overview

This project evaluates how different persona prompts affect LLM accuracy on challenging benchmarks like GPQA and MMLU-Pro. It tests six experimental conditions:

1. **Baseline-MC**: No persona, multiple choice, short answer format
2. **Shallow-Persona-MC**: Paper-style expert tag, multiple choice, short answer
3. **Deep-Persona-MC-Preamble**: Rich expert persona, long reasoning, multiple choice
4. **Deep-Persona-Open**: Rich expert persona, long reasoning, no multiple choice
5. **Process-Only-Open**: No persona, long reasoning, no multiple choice
6. **Low-Knowledge-Persona-Deep**: Toddler persona, long reasoning, multiple choice

## Installation

```bash
# Clone the repository
git clone https://github.com/CrazyDubya/PersonaTesting.git
cd PersonaTesting/persona_eval

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development (includes linting, testing tools)
pip install -r requirements-dev.txt

# Or install as a package
pip install -e .
```

### Requirements

- Python 3.8 or higher
- At least one API key for OpenAI, Anthropic, or OpenRouter
- Sufficient API quota for your chosen models

## Configuration

### Environment Variables

Set API keys for the providers you want to use:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenRouter (optional - for accessing many models via single API)
export OPENROUTER_API_KEY="sk-or-..."
```

### Configuration File

Edit `config/default_config.yaml` to customize:
- Datasets and their paths
- Models to evaluate (supports OpenAI, Anthropic, OpenRouter)
- Experimental conditions
- Sampling parameters (num_samples, temperature)
- Output directories

## Data Format

Each dataset must be a JSONL file where each line is a JSON object:

```json
{
  "id": "unique_question_id",
  "question_text": "The question text here...",
  "options": [
    "A. option text here",
    "B. option text here",
    "C. option text here",
    "D. option text here"
  ],
  "correct_option_letter": "C",
  "correct_answer_text": "Canonical correct answer text (optional for MCQ)",
  "subject": "physics",
  "difficulty": "post-grad",
  "metadata": {}
}
```

For open-ended-only datasets:
- `options` may be `[]`
- `correct_option_letter` may be `null`
- `correct_answer_text` should be provided for judge-based grading

## Usage

### Run Full Experiment

```bash
# Run with default config
python -m src.cli --config config/default_config.yaml

# Run specific models only
python -m src.cli --config config/default_config.yaml --models gpt-4o,claude-3-5-sonnet

# Run specific conditions only
python -m src.cli --config config/default_config.yaml --conditions baseline_mc,shallow_persona_mc

# Skip sampling (use existing raw responses)
python -m src.cli --config config/default_config.yaml --skip-sampling

# Force re-run even if outputs exist
python -m src.cli --config config/default_config.yaml --no-skip-existing
```

### Quick Test

Test a single model/condition with limited questions:

```bash
python -m src.cli --config config/default_config.yaml \
    --test \
    --model gpt-4o \
    --condition baseline_mc \
    --num-questions 5 \
    --num-samples 1
```

## Output Structure

```
outputs/
  raw_responses/          # Raw model outputs (JSONL)
    raw_{model}_{condition}.jsonl
  scored/                 # Scored responses with correctness labels
    scored_{model}_{condition}.jsonl
  summaries/              # Aggregate metrics
    summary_metrics.jsonl
    summary_metrics.txt
  logs/                   # Execution logs
    persona_eval.log
```

### Metrics Computed

- **Mean accuracy**: Overall accuracy across all samples
- **95% Confidence interval**: Wilson score confidence interval
- **Refusal rate**: Fraction of responses that indicate refusal/inability
- **Mean reasoning tokens**: Average length of reasoning (for long conditions)
- **Robust correctness thresholds**:
  - Fraction of questions with all N/N samples correct
  - Fraction with ≥23/25 correct
  - Fraction with ≥13/25 correct
- **Per-subject accuracy**: Breakdown by subject area
- **Per-difficulty accuracy**: Breakdown by difficulty level

## Supported Models

### OpenAI
- `gpt-4o`
- `gpt-4o-mini`
- `o3-mini`

### Anthropic
- `claude-3-5-sonnet` (claude-3-5-sonnet-20241022)
- `claude-3-5-haiku` (claude-3-5-haiku-20241022)
- `claude-3-opus` (claude-3-opus-20240229)

### OpenRouter (many models via unified API)
- `openrouter-gpt-4o`
- `openrouter-claude-3-5-sonnet`
- `openrouter-llama-3-70b`
- `openrouter-mixtral-8x22b`
- `openrouter-gemini-pro`
- And many more (edit config to add)

## Project Structure

```
persona_eval/
  README.md
  pyproject.toml
  requirements.txt

  config/
    default_config.yaml

  data/
    gpqa/
      questions.jsonl
    mmlu_pro/
      questions.jsonl

  src/
    __init__.py
    config.py          # YAML config loading
    data_models.py     # Data classes (Question, Config, etc.)
    prompts.py         # Prompt templates for all conditions
    model_api.py       # LLM client abstraction (OpenAI, Anthropic, OpenRouter)
    sampling.py        # Response generation and parsing
    scoring.py         # Answer evaluation (MCQ and judge-based)
    metrics.py         # Aggregate statistics computation
    runners.py         # Experiment orchestration
    cli.py             # Command-line interface

    utils/
      __init__.py
      logging_utils.py
      io_utils.py

  outputs/
    logs/
    raw_responses/
    scored/
    summaries/
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests (when available)
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Lint code
ruff check src/

# Type checking
mypy src/
```

## Troubleshooting

### Common Issues

1. **API Key Not Set**: Make sure environment variables are exported in your current shell
   ```bash
   echo $OPENAI_API_KEY  # Should print your key
   ```

2. **Config File Not Found**: Run from the `persona_eval` directory or provide absolute path
   ```bash
   python -m src.cli --config /absolute/path/to/config.yaml
   ```

3. **Empty Responses**: Check API quota and model availability. Enable verbose logging.

4. **JSONL Parse Errors**: Validate your question files have proper JSON formatting (one object per line)

### Performance Tips

- Use `--skip-existing` to avoid re-running completed samples
- Start with `--test` mode to validate config before full runs
- Filter to specific models/conditions during development
- Monitor API costs, especially with large num_samples_per_question

## Notes

- **Judge model**: For open-ended answers, a judge model (default: gpt-4o) evaluates correctness by comparing the model's answer to the gold answer.
- **Retry logic**: API calls include retry logic with exponential backoff for robustness.
- **Parallel execution**: Currently runs sequentially; can be extended for parallel sampling.
- **Token estimation**: Reasoning token counts are approximate (based on word/character heuristics).
- **Error handling**: The framework includes comprehensive error handling and validation to prevent crashes.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run linting and tests before submitting
5. Submit a pull request with a clear description

## License

MIT License
