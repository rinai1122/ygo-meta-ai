"""
LLM-based and random battle evaluators.

Mirror BattleRunner's interface but use two agents (LLM or random) instead of
the RL model. Works with any cards — no code_list constraint beyond data/code_list.txt.

game_runner.py handles the ygoenv game loop and sends step/done JSON lines.
"""

from __future__ import annotations

import json
import os
import random as _random
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path, PureWindowsPath

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.deck_builder.ydk_parser import write_ydk
from ygo_meta.simulation.battle_runner import BattleResult

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_GAME_RUNNER  = _PROJECT_ROOT / "scripts" / "game_runner.py"
_VENDOR_CODE_LIST = _PROJECT_ROOT / "vendor" / "ygo-agent" / "scripts" / "code_list.txt"
_CUSTOM_CODE_LIST = _PROJECT_ROOT / "data" / "code_list.txt"
_CODE_LIST    = _CUSTOM_CODE_LIST if _CUSTOM_CODE_LIST.exists() else _VENDOR_CODE_LIST
_CARD_NAMES   = _PROJECT_ROOT / "data" / "card_names.json"


def _win_to_wsl(path: str) -> str:
    p = PureWindowsPath(path)
    if p.drive:
        drive = p.drive.rstrip(":").lower()
        rest = "/".join(p.parts[1:])
        return f"/mnt/{drive}/{rest}"
    return path.replace("\\", "/")


def _python_exe() -> list[str]:
    venv = os.environ.get("YGOAGENT_VENV", "")
    if venv:
        if sys.platform == "win32" and (venv.startswith("/") or venv.startswith("\\\\wsl")):
            return ["wsl", f"{venv}/bin/python3"]
        for candidate in [Path(venv) / "bin" / "python", Path(venv) / "Scripts" / "python.exe"]:
            if candidate.exists():
                return [str(candidate)]
    return [sys.executable]


def _is_engine_crash(rc: int) -> bool:
    # Unix signal: rc < 0; Linux exit code convention: rc >= 128 (e.g. 134 = 128+SIGABRT)
    return rc < 0 or rc >= 128


def _run_game_subprocess(
    deck1: Deck,
    deck2: Deck,
    num_episodes: int,
    seed: int,
    action_fn: Callable[[str, int], int],
    verbose: bool = False,
) -> BattleResult:
    """Spawn game_runner.py, feed actions via action_fn(prompt, num_options), return result."""
    is_wsl = sys.platform == "win32" and os.environ.get("YGOAGENT_VENV", "").startswith("/")
    _path = _win_to_wsl if is_wsl else str

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        write_ydk(deck1, tmp / "deck1.ydk")
        write_ydk(deck2, tmp / "deck2.ydk")

        cmd = _python_exe() + [
            _path(str(_GAME_RUNNER)),
            "--deck", _path(str(tmp)),
            "--deck1", "deck1",
            "--deck2", "deck2",
            "--num-episodes", str(num_episodes),
            "--seed", str(seed),
            "--code-list", _path(str(_CODE_LIST)),
            "--card-names", _path(str(_CARD_NAMES)),
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        wins = {0: 0, 1: 0}
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg["type"] == "step":
                    if verbose:
                        print(
                            f"\n{'='*60}\n[GAME PROMPT]\n{msg['prompt']}\n{'='*60}",
                            flush=True,
                        )
                    action = action_fn(msg["prompt"], msg["num_options"])
                    if verbose:
                        print(f"[ACTION] {action} / {msg['num_options']}", flush=True)
                    proc.stdin.write(json.dumps({"action": action}) + "\n")
                    proc.stdin.flush()
                elif msg["type"] == "done":
                    wins[msg["winner"]] += 1
        finally:
            proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read()
        total = wins[0] + wins[1]
        if _is_engine_crash(proc.returncode):
            # C++ engine abort (SIGABRT) — warn and use partial/0.5.
            stderr_tail = stderr.strip()
            # Show last 2000 chars so the actual error isn't hidden behind
            # the gym deprecation warning that always appears first.
            if len(stderr_tail) > 2000:
                stderr_tail = "...(truncated)...\n" + stderr_tail[-2000:]
            print(
                f"  [WARN] game engine crashed (rc={proc.returncode}) after {total} episodes. "
                f"Defaulting to 0.5.\n  stderr: {stderr_tail}",
                flush=True,
            )
        else:
            raise RuntimeError(
                f"game_runner.py failed (rc={proc.returncode}):\n{stderr}"
            )

    total = wins[0] + wins[1]
    wr1 = wins[0] / total if total else 0.5
    return BattleResult(
        win_rate_d1=wr1,
        win_rate_d2=1.0 - wr1,
        episodes=total,
        deck1_id=deck1.variant_id,
        deck2_id=deck2.variant_id,
    )


class RandomBattleRunner:
    """Battle evaluator that picks actions uniformly at random. Zero API cost."""

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    def run(
        self,
        deck1: Deck,
        deck2: Deck,
        num_episodes: int = 128,
        seed: int = 0,
    ) -> BattleResult:
        rng = _random.Random(seed)
        return _run_game_subprocess(
            deck1, deck2, num_episodes, seed,
            action_fn=lambda _prompt, n: rng.randrange(n),
            verbose=self._verbose,
        )


class LLMBattleRunner:
    def __init__(
        self,
        provider: str = "anthropic",
        model: str | None = None,
        api_key: str | None = None,
        verbose: bool = False,
    ) -> None:
        self._provider = provider
        _defaults = {"anthropic": "claude-opus-4-6", "gemini": "gemini-2.0-flash"}
        self._model = model or _defaults.get(provider, "claude-opus-4-6")
        self._api_key = api_key
        self._verbose = verbose
        self._agent: object | None = None  # lazy-init

    def _get_agent(self) -> object:
        if self._agent is None:
            from ygo_meta.llm_agent.agent import LLMAgent
            self._agent = LLMAgent(
                model=self._model,
                provider=self._provider,
                api_key=self._api_key,
                verbose=self._verbose,
            )
        return self._agent

    def run(
        self,
        deck1: Deck,
        deck2: Deck,
        num_episodes: int = 128,
        seed: int = 0,
    ) -> BattleResult:
        agent = self._get_agent()
        return _run_game_subprocess(
            deck1, deck2, num_episodes, seed,
            action_fn=agent.choose_action_from_text,
            verbose=self._verbose,
        )
