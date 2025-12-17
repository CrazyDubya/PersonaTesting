# Persona Evaluation Experiment

This project evaluates how different personas and reasoning styles affect model performance on datasets such as GPQA and MMLU-Pro.

## Experiment Conditions

1. **Baseline-MC:** No persona, multiple choice, short answer format.
2. **Shallow-Persona-MC:** Paper-style expert tag, multiple choice, short answer.
3. **Deep-Persona-MC-Preamble:** Rich expert persona, long reasoning, multiple choice.
4. **Deep-Persona-Open:** Rich expert persona, long reasoning, no multiple choice.
5. **Process-Only-Open:** No persona, long reasoning, no multiple choice.
6. **Low-Knowledge-Persona-Deep:** Toddler persona, long reasoning, multiple choice.

## Data Format

Each dataset (`data/gpqa/questions.jsonl`, `data/mmlu_pro/questions.jsonl`) must be a JSONL file where each line is a JSON object:

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

For open-ended-only datasets, `options` may be empty and `correct_option_letter` may be null, but `correct_answer_text` should be provided for judge-based grading.

Sample GPQA and MMLU-Pro question files are included in `data/gpqa/questions.jsonl` and `data/mmlu_pro/questions.jsonl` so the pipeline can run end-to-end out of the box. Replace or expand these stubs with your full datasets for real evaluations.

## Configuration

Default settings live in `config/default_config.yaml` and include dataset paths, model definitions, sampling parameters, judge settings, and the six experiment conditions. Update the file to point to your data locations or adjust sampling counts.

## Providers and API Keys

`src/model_api.py` supports three providers:

- **OpenAI:** Requires `OPENAI_API_KEY`.
- **OpenRouter:** Requires `OPENROUTER_API_KEY`; optional `OPENROUTER_BASE_URL` (defaults to `https://openrouter.ai/api/v1`).
- **Anthropic:** Requires `ANTHROPIC_API_KEY`.

Set the appropriate environment variables before running experiments.

## Running the Experiment

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure models and datasets in `config/default_config.yaml`.

3. Run the experiment:
   ```bash
   python -m src.cli --config config/default_config.yaml
   ```

Outputs are written to:
- `outputs/raw_responses/` – all raw model outputs.
- `outputs/scored/` – row-wise scored results.
- `outputs/summaries/summary_metrics.jsonl` – aggregate metrics per (dataset, model, condition).

## Notes

- `model_api.py` wires the `LLMClient` to OpenAI-compatible chat completions and Anthropic's Messages API.
- The judge model is used for open-ended answers to decide correctness when a gold text answer exists.
- Metrics include accuracy, refusal rate, reasoning length, and robust correctness thresholds (all correct, ≥23/25, ≥13/25).
