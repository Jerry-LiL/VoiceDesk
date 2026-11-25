from dataclasses import dataclass


@dataclass
class Turn:
    user: str
    assistant: str
    intent: str


class ConversationMemory:
    """A short window of spoken turns, oldest first."""

    def __init__(self, max_turns: int = 8) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.max_turns = max_turns
        self._turns: list[Turn] = []

    def add(self, user: str, assistant: str, intent: str) -> None:
        self._turns.append(Turn(user=user.strip(), assistant=assistant.strip(), intent=intent))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

    def render(self) -> str:
        if not self._turns:
            return ""
        lines: list[str] = []
        for turn in self._turns:
            lines.append(f"User: {turn.user}")
            lines.append(f"Assistant ({turn.intent}): {turn.assistant}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
