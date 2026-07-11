from grounded_rag.generation.prompts import build_question_prompt


def test_build_question_prompt_without_sub_questions_is_unchanged():
    prompt = build_question_prompt("What is X known for?", chunks=[])

    assert "Question: What is X known for?" in prompt
    assert "independent parts" not in prompt


def test_build_question_prompt_enumerates_sub_questions_when_present():
    prompt = build_question_prompt(
        "What is X known for, and where is Y located?",
        chunks=[],
        sub_questions=["What is X known for?", "Where is Y located?"],
    )

    assert "Question: What is X known for, and where is Y located?" in prompt
    assert "1. What is X known for?" in prompt
    assert "2. Where is Y located?" in prompt
    assert "address EVERY one of them" in prompt
