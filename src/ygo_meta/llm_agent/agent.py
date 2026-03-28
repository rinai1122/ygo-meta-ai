"""
LLM game-playing agent using Anthropic Claude.

Given a game state (Input), it formats it as text, calls Claude,
parses the response as an action index, and returns that index.

Falls back to action 0 on any parsing failure.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import anthropic

from ygo_meta.engine.types import Input
from ygo_meta.llm_agent.formatter import format_prompt

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


class LLMAgent:
    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_tokens: int = 16,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._system_prompt = _load_prompt("system.txt")
        self._decision_template = _load_prompt("decision.txt")

    def choose_action(self, input_: Input) -> int:
        """Return the index of the chosen action (0-based)."""
        user_message = format_prompt(input_, self._decision_template)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        text = response.content[0].text.strip() if response.content else "0"
        return self._parse_action(text, input_)

    def _parse_action(self, text: str, input_: Input) -> int:
        if re.match(r"^\d+$", text):
            idx = int(text)
            n_actions = self._count_actions(input_)
            if 0 <= idx < n_actions:
                return idx
        return 0

    @staticmethod
    def _count_actions(input_: Input) -> int:
        msg = input_.action_msg
        # Most action messages have an 'actions' field
        actions = getattr(msg, "actions", None)
        if actions is not None:
            return len(actions)
        # Yes/no prompts have exactly 2 choices
        return 2
