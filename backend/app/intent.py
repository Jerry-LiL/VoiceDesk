"""Rule-based intent router for the voice prompts.

The rules are ordered on purpose. A task phrase such as "can you remind me"
is a task even though it also looks like a question. Anything that does not
match, or that matches below the confidence threshold, uses the fallback
template instead of a guess.
"""

import re
from dataclasses import dataclass

_FAREWELL = re.compile(r"\b(goodbye|bye|see you|good night)\b", re.IGNORECASE)
_TASK = re.compile(
    r"\b(remind me|set a timer|set an alarm|take a note|make a note|"
    r"schedule|translate|summarize)\b",
    re.IGNORECASE,
)
_SMALLTALK = re.compile(
    r"\b(how are you|how's it going|whats up|what's up|thank you|thanks|"
    r"nice to meet you)\b",
    re.IGNORECASE,
)
_QUESTION = re.compile(
    r"(\?|(^|\s)(what|why|how|when|where|who|which|can you|could you|"
    r"would you|is there|are there)\b)",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"^(hi|hello|hey|good morning|good afternoon|good evening)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IntentMatch:
    name: str
    confidence: float
    reason: str


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def classify_intent(text: str) -> IntentMatch:
    cleaned = normalize(text)
    if not cleaned:
        return IntentMatch("unknown", 0.0, "empty")

    words = cleaned.split()
    if _FAREWELL.search(cleaned) and len(words) <= 8:
        return IntentMatch("farewell", 0.93, "farewell_phrase")
    if _TASK.search(cleaned):
        return IntentMatch("task", 0.88, "task_phrase")
    if _SMALLTALK.search(cleaned) and len(words) <= 8:
        return IntentMatch("smalltalk", 0.84, "smalltalk_phrase")
    if _QUESTION.search(cleaned):
        return IntentMatch("question", 0.8, "question_phrase")
    if _GREETING.search(cleaned) and len(words) <= 6:
        return IntentMatch("greeting", 0.9, "greeting_phrase")
    return IntentMatch("unknown", 0.25, "no_rule")
