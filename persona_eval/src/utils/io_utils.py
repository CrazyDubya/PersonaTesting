import json
import os
from typing import List, Dict, Any
from ..data_models import Question


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_questions_from_jsonl(path: str, dataset_name: str) -> List[Question]:
    questions: List[Question] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
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
    return questions


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
