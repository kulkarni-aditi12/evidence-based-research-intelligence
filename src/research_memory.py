import json
import os
from datetime import datetime


MEMORY_FILE = "research_memory.json"


def load_memory():

    if not os.path.exists(MEMORY_FILE):
        return []

    try:

        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return []


def save_memory(
    question,
    research_plan,
    final_answer
):

    memory = load_memory()

    entry = {

        "timestamp":
            datetime.now().isoformat(),

        "question":
            question,

        "research_plan":
            research_plan,

        "final_answer":
            final_answer
    }

    memory.append(entry)

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            memory,
            file,
            indent=4,
            ensure_ascii=False
        )


def search_memory(question):

    memory = load_memory()

    question_words = set(
        question.lower().split()
    )

    matches = []

    for entry in memory:

        previous_question = entry.get(
            "question",
            ""
        ).lower()

        previous_words = set(
            previous_question.split()
        )

        overlap = len(
            question_words.intersection(
                previous_words
            )
        )

        if overlap > 0:

            matches.append(
                (overlap, entry)
            )

    matches.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        entry
        for _, entry
        in matches[:3]
    ]