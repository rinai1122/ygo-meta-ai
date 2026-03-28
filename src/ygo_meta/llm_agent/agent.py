"""
LLM game-playing agent supporting Anthropic Claude and Google Gemini.

Given a game state (Input), it formats it as text, calls the LLM,
parses the response as an action index, and returns that index.

Falls back to action 0 on any parsing failure.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ygoinf.features import get_legal_actions

from ygo_meta.engine.types import Input
from ygo_meta.llm_agent.formatter import format_prompt

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


class LLMAgent:
    def __init__(
        self,
        model: str = "claude-opus-4-6",
        provider: str = "anthropic",
        max_tokens: int = 16,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._provider = provider
        self._max_tokens = max_tokens
        self._system_prompt = _load_prompt("system.txt")
        self._decision_template = _load_prompt("decision.txt")

        if provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key or os.environ["GOOGLE_API_KEY"])
            self._client = genai.GenerativeModel(
                model_name=model,
                system_instruction=self._system_prompt,
                generation_config=genai.GenerationConfig(max_output_tokens=max_tokens),
            )
        else:
            raise ValueError(f"Unknown provider '{provider}'. Use 'anthropic' or 'gemini'.")

    def _call_llm(self, prompt_text: str) -> str:
        if self._provider == "anthropic":
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                messages=[{"role": "user", "content": prompt_text}],
            )
            return response.content[0].text.strip() if response.content else "0"
        else:  # gemini
            response = self._client.generate_content(prompt_text)
            return response.text.strip() if response.text else "0"

    def choose_action(self, input_: Input) -> int:
        """Return the index of the chosen action (0-based)."""
        user_message = format_prompt(input_, self._decision_template)
        text = self._call_llm(user_message)
        return self._parse_action(text, input_)

    def choose_action_from_text(self, prompt_text: str, num_options: int) -> int:
        """Choose action from a pre-formatted prompt string (no Input object needed)."""
        text = self._call_llm(prompt_text)
        if re.match(r"^\d+$", text):
            idx = int(text)
            if 0 <= idx < num_options:
                return idx
        return 0

    def _parse_action(self, text: str, input_: Input) -> int:
        if re.match(r"^\d+$", text):
            idx = int(text)
            try:
                n_actions = len(get_legal_actions(input_.action_msg))
            except Exception:
                n_actions = 2  # fallback for select_yesno/effectyn with desc=0
            if 0 <= idx < n_actions:
                return idx
        return 0
