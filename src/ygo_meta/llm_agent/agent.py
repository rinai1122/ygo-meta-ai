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
        verbose: bool = False,
    ) -> None:
        self._verbose = verbose
        self._model = model
        self._provider = provider
        self._max_tokens = max_tokens
        self._system_prompt = _load_prompt("system.txt")
        self._decision_template = _load_prompt("decision.txt")

        if provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        elif provider == "gemini":
            import httpx
            from google import genai
            from google.genai import types as genai_types
            self._genai_types = genai_types
            # Force IPv4 transport: WSL's default network stack resolves Google APIs
            # to IPv6 addresses but has no IPv6 route, causing ConnectError ENETUNREACH.
            _transport = httpx.HTTPTransport(local_address="0.0.0.0")
            _httpx_client = httpx.Client(transport=_transport)
            self._client = genai.Client(
                api_key=api_key or os.environ["GOOGLE_API_KEY"],
                http_options=genai_types.HttpOptions(
                    httpx_client=_httpx_client,
                    timeout=30000,
                ),
            )
        else:
            raise ValueError(f"Unknown provider '{provider}'. Use 'anthropic' or 'gemini'.")

    def _call_llm(self, prompt_text: str) -> str:
        if self._verbose:
            print(f"\n{'='*60}\n[LLM PROMPT]\n{prompt_text}\n{'='*60}", flush=True)
        if self._provider == "anthropic":
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                messages=[{"role": "user", "content": prompt_text}],
            )
            result = response.content[0].text.strip() if response.content else "0"
        else:  # gemini
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt_text,
                    config=self._genai_types.GenerateContentConfig(
                        system_instruction=self._system_prompt,
                        max_output_tokens=self._max_tokens,
                    ),
                )
                result = response.text.strip() if response.text else "0"
            except Exception as e:
                print(f"[WARN] Gemini API error: {e}", flush=True)
                result = "0"
        if self._verbose:
            print(f"[LLM RESPONSE] {result}", flush=True)
        return result

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
