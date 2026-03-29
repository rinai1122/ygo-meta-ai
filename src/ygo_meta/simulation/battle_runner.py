"""
Subprocess wrapper for vendor/ygo-agent/scripts/battle.py.

Both decks use the same pre-trained RL agent. This isolates deck quality
from agent quality — the RL agent is held constant across all matchups.

Usage:
    runner = BattleRunner()
    result = runner.run(deck1, deck2, num_episodes=128, seed=0)
    print(result.win_rate_d1)  # win rate for deck1
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.deck_builder.ydk_parser import write_ydk

_BATTLE_SCRIPT = (
    Path(__file__).parent.parent.parent.parent
    / "vendor" / "ygo-agent" / "scripts" / "battle.py"
)


@dataclass
class BattleResult:
    win_rate_d1: float   # fraction of episodes won by deck1
    win_rate_d2: float   # fraction of episodes won by deck2
    episodes: int
    deck1_id: str
    deck2_id: str


class BattleRunner:
    def __init__(self, python_exe: str | None = None, xla_device: str = "cpu") -> None:
        if python_exe:
            self._python = python_exe
        elif venv := os.environ.get("YGOAGENT_VENV"):
            # Prefer the venv's python which has JAX/ygoenv installed
            candidate = Path(venv) / "bin" / "python"
            if not candidate.exists():
                candidate = Path(venv) / "Scripts" / "python.exe"
            self._python = str(candidate)
        else:
            self._python = sys.executable
        self._xla_device = xla_device

    def run(
        self,
        deck1: Deck,
        deck2: Deck,
        num_episodes: int = 128,
        seed: int = 0,
        checkpoint: str | None = None,
    ) -> BattleResult:
        if not _BATTLE_SCRIPT.exists():
            raise FileNotFoundError(
                f"battle.py not found at {_BATTLE_SCRIPT}. "
                "Run: git submodule update --init --recursive"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ydk1 = tmp / "deck1.ydk"
            ydk2 = tmp / "deck2.ydk"
            write_ydk(deck1, ydk1)
            write_ydk(deck2, ydk2)

            cmd = [
                self._python, "-P", str(_BATTLE_SCRIPT),
                "--xla_device", self._xla_device,
                "--deck", str(tmp),   # LF temp dir avoids CRLF in vendor assets/deck
                "--deck1", str(ydk1),
                "--deck2", str(ydk2),
                "--num-episodes", str(num_episodes),
                "--seed", str(seed),
            ]
            if checkpoint:
                cmd += ["--checkpoint1", checkpoint, "--checkpoint2", checkpoint]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(_BATTLE_SCRIPT.parent),
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"battle.py failed (rc={result.returncode}):\n{result.stderr}"
            )

        return self._parse_output(result.stdout, deck1.variant_id, deck2.variant_id, num_episodes)

    @staticmethod
    def _parse_output(stdout: str, d1_id: str, d2_id: str, episodes: int) -> BattleResult:
        # battle.py prints something like:
        #   win_rates: [0.53 0.47]
        # or a payoff matrix block. Try to extract the first player's win rate.
        match = re.search(r"win_rates?[:\s]+\[?\s*([\d.]+)", stdout, re.IGNORECASE)
        if match:
            wr1 = float(match.group(1))
        else:
            # Fallback: look for a plain fraction
            match2 = re.search(r"([\d.]+)\s*/\s*\d+", stdout)
            wr1 = float(match2.group(1)) / episodes if match2 else 0.5

        wr1 = max(0.0, min(1.0, wr1))
        return BattleResult(
            win_rate_d1=wr1,
            win_rate_d2=1.0 - wr1,
            episodes=episodes,
            deck1_id=d1_id,
            deck2_id=d2_id,
        )
