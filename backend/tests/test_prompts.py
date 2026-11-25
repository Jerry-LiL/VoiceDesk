from app.memory import ConversationMemory
from app.prompts import INTENT_INSTRUCTIONS, SPOKEN_STYLE, build_prompt


def test_first_turn_has_no_memory_and_a_spoken_style():
    built = build_prompt("hello", ConversationMemory(4), threshold=0.55)
    system = built.messages[0]["content"]
    assert built.intent == "greeting"
    assert built.used_fallback is False
    assert built.messages[-1] == {"role": "user", "content": "hello"}
    assert len(built.messages) == 2
    assert SPOKEN_STYLE in system
    assert INTENT_INSTRUCTIONS["greeting"] in system
    assert "no earlier conversation" in system


def test_unknown_request_uses_the_fallback_template():
    built = build_prompt("purple bicycle warranty", ConversationMemory(4), threshold=0.55)
    system = built.messages[0]["content"]
    assert built.used_fallback is True
    assert built.fallback_reason == "unknown_intent"
    assert built.intent == "unknown"
    assert INTENT_INSTRUCTIONS["unknown"] in system
    assert "Do not guess an answer." in system


def test_low_confidence_also_falls_back():
    built = build_prompt("hello", ConversationMemory(4), threshold=0.95)
    assert built.matched_intent == "greeting"
    assert built.intent == "unknown"
    assert built.used_fallback is True
    assert built.fallback_reason == "low_confidence"
    assert INTENT_INSTRUCTIONS["unknown"] in built.messages[0]["content"]
    assert INTENT_INSTRUCTIONS["greeting"] not in built.messages[0]["content"]


def test_memory_is_written_into_the_system_prompt():
    memory = ConversationMemory(max_turns=8)
    memory.add("remind me to buy milk", "I'll remember milk.", "task")
    built = build_prompt("what about tomorrow?", memory, threshold=0.55)
    system = built.messages[0]["content"]
    assert built.intent == "question"
    assert "Earlier turns, oldest first:" in system
    assert "User: remind me to buy milk" in system
    assert "Assistant (task): I'll remember milk." in system
    assert built.messages[-1]["content"] == "what about tomorrow?"


def test_memory_window_drops_the_oldest_turn():
    memory = ConversationMemory(max_turns=2)
    memory.add("one", "a", "question")
    memory.add("two", "b", "question")
    memory.add("three", "c", "question")
    text = memory.render()
    assert len(memory) == 2
    assert "one" not in text
    assert "two" in text
    assert "three" in text
