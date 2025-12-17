import json
import os
from typing import List, Dict, Any
from ..data_models import Question


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_questions_from_jsonl(path: str, dataset_name: str) -> List[Question]:
    """Load questions from a JSONL file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Question file not found: {path}")
    
    questions: List[Question] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    # Validate required fields
                    if "id" not in row:
                        print(f"Warning: Skipping line {line_num} in {path}: missing 'id' field")
                        continue
                    if "question_text" not in row:
                        print(f"Warning: Skipping line {line_num} in {path}: missing 'question_text' field")
                        continue
                    
                    q = Question(
                        dataset=dataset_name,
                        id=str(row["id"]),
                        question_text=row["question_text"],
                        options=row.get("options", []),
                        correct_option_letter=row.get("correct_option_letter"),
                        correct_answer_text=row.get("correct_answer_text"),
                        subject=row.get("subject"),
                        difficulty=row.get("difficulty"),
                        metadata=row.get("metadata", {}),
                    )
                    questions.append(q)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num} in {path}: {e}")
                    continue
    except Exception as e:
        raise IOError(f"Error reading question file {path}: {e}")
    
    if not questions:
        print(f"Warning: No valid questions loaded from {path}")
    
    return questions


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write rows to a JSONL file."""
    # Ensure parent directory exists
    parent_dir = os.path.dirname(path)
    if parent_dir:
        ensure_dir(parent_dir)
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        raise IOError(f"Error writing to file {path}: {e}")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read rows from a JSONL file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num} in {path}: {e}")
                    continue
    except Exception as e:
        raise IOError(f"Error reading file {path}: {e}")
    
    return rows
