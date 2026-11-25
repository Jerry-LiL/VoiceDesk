from app.intent import classify_intent


def test_greeting():
    match = classify_intent("good morning")
    assert match.name == "greeting"
    assert match.confidence > 0.55


def test_farewell():
    assert classify_intent("bye").name == "farewell"


def test_task_beats_a_question_word():
    match = classify_intent("can you remind me to call mom")
    assert match.name == "task"


def test_smalltalk_beats_a_question_word():
    assert classify_intent("how are you").name == "smalltalk"


def test_question():
    assert classify_intent("what time is it?").name == "question"


def test_hello_plus_a_question_stays_a_question():
    assert classify_intent("hello, what time is it?").name == "question"


def test_unknown_when_nothing_matches():
    match = classify_intent("purple bicycle warranty")
    assert match.name == "unknown"
    assert match.confidence < 0.55


def test_empty_text():
    match = classify_intent("   ")
    assert match.name == "unknown"
    assert match.confidence == 0.0
