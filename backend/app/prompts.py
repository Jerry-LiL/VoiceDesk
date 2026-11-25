from dataclasses import dataclass

from app.intent import classify_intent
from app.memory import ConversationMemory

SPOKEN_STYLE = (
    "You are VoiceDesk, a voice assistant. The user just spoke, and your "
    "reply will be read aloud.\n"
    "Speak in one to three short sentences. Use plain words. Do not use "
    "markdown, bullet lists, or emojis.\n"
    "Do not invent appointments, reminders, or actions you did not take.\n"
    'When earlier turns are included, use them to resolve words like "that", '
    '"it", and "tomorrow". Do not repeat an answer you already gave.'
)

INTENT_INSTRUCTIONS = {
    "greeting": ("The user is greeting you. Greet them back and offer help in one sentence."),
    "farewell": "The user is ending the conversation. Say a brief goodbye.",
    "question": ("Answer the question directly. If you are unsure, say so in one sentence."),
    "task": (
        "The user is asking you to do a small task. Confirm what you understood "
        "and state the outcome in plain speech. If you cannot do it, say what "
        "is missing."
    ),
    "smalltalk": "Respond warmly and briefly, then leave room for a real request.",
    "unknown": (
        "You are not sure what the user wants. Ask one short clarifying question. "
        "Do not guess an answer."
    ),
}

STATIC_FALLBACK = {
    "greeting": "Hi. What can I help you with?",
    "farewell": "Goodbye.",
    "question": "I couldn't reach the language model. Could you ask that once more?",
    "task": "I couldn't complete that just now. Could you repeat the request?",
    "smalltalk": "I'm here. What would you like to do?",
    "unknown": "I didn't quite catch what you need. Could you say that another way?",
}


@dataclass(frozen=True)
class BuiltPrompt:
    messages: list[dict[str, str]]
    intent: str
    matched_intent: str
    confidence: float
    used_fallback: bool
    fallback_reason: str


def static_fallback(intent: str) -> str:
    return STATIC_FALLBACK.get(intent, STATIC_FALLBACK["unknown"])


def render_system(intent: str, memory_text: str) -> str:
    instruction = INTENT_INSTRUCTIONS[intent]
    if memory_text.strip():
        memory_block = (
            "Earlier turns, oldest first:\n"
            f"{memory_text}\n"
            "Use these turns as memory for this reply."
        )
    else:
        memory_block = "This is the first turn. There is no earlier conversation."
    return "\n\n".join([SPOKEN_STYLE, instruction, memory_block])


def build_prompt(
    user_text: str,
    memory: ConversationMemory,
    threshold: float,
) -> BuiltPrompt:
    text = " ".join(user_text.strip().split())
    match = classify_intent(text)
    low_confidence = match.confidence < threshold
    used_fallback = match.name == "unknown" or low_confidence
    if match.name == "unknown":
        reason = "unknown_intent"
    elif low_confidence:
        reason = "low_confidence"
    else:
        reason = ""
    template_intent = "unknown" if used_fallback else match.name
    messages = [
        {"role": "system", "content": render_system(template_intent, memory.render())},
        {"role": "user", "content": text},
    ]
    return BuiltPrompt(
        messages=messages,
        intent=template_intent,
        matched_intent=match.name,
        confidence=match.confidence,
        used_fallback=used_fallback,
        fallback_reason=reason,
    )
