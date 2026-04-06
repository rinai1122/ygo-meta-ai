"""
Subprocess wrapper that runs RL vs RL battles via game_runner.py.

Both decks use the same pre-trained RL agent. This isolates deck quality
from agent quality — the RL agent is held constant across all matchups.

Uses game_runner.py (not battle.py) to avoid JAX API incompatibilities
in the vendor battle script.

Usage:
    runner = BattleRunner()                         # auto-detects checkpoints/agent.flax_model
    runner = BattleRunner(checkpoint="path/to.flax_model")
    result = runner.run(deck1, deck2, num_episodes=128, seed=0)
    print(result.win_rate_d1)  # win rate for deck1
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.deck_builder.ydk_parser import write_ydk

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_GAME_RUNNER = _PROJECT_ROOT / "scripts" / "game_runner.py"
_DEFAULT_CHECKPOINT = _PROJECT_ROOT / "checkpoints" / "agent.flax_model"
_DEFAULT_MODEL_ARGS = _PROJECT_ROOT / "checkpoints" / "model_args.json"
_FULL_MODEL = dict(num_layers=2, num_channels=128, rnn_channels=512, critic_width=128, critic_depth=3)
_VENDOR_CODE_LIST = _PROJECT_ROOT / "vendor" / "ygo-agent" / "scripts" / "code_list.txt"
_CUSTOM_CODE_LIST = _PROJECT_ROOT / "data" / "code_list.txt"
_CDB_PATH = _PROJECT_ROOT / "vendor" / "ygo-agent" / "assets" / "locale" / "en" / "cards.cdb"
_CARD_NAMES = _PROJECT_ROOT / "data" / "card_names.json"


def _ensure_code_list() -> Path:
    """Return path to an up-to-date code_list.txt, regenerating if the CDB is newer."""
    if _CDB_PATH.exists():
        need_regen = (
            not _CUSTOM_CODE_LIST.exists()
            or _CDB_PATH.stat().st_mtime > _CUSTOM_CODE_LIST.stat().st_mtime
        )
        if need_regen:
            import sqlite3
            con = sqlite3.connect(str(_CDB_PATH))
            ids = [r[0] for r in con.execute("SELECT id FROM datas ORDER BY id").fetchall()]
            con.close()
            _CUSTOM_CODE_LIST.parent.mkdir(parents=True, exist_ok=True)
            with open(_CUSTOM_CODE_LIST, "w", encoding="ascii") as f:
                for card_id in ids:
                    f.write(f"{card_id} 1\n")
            print(f"[battle_runner] Regenerated {_CUSTOM_CODE_LIST} ({len(ids)} cards from CDB)", flush=True)
    if _CUSTOM_CODE_LIST.exists():
        return _CUSTOM_CODE_LIST
    return _VENDOR_CODE_LIST


_CODE_LIST = _ensure_code_list()


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
    return rc < 0 or rc >= 128


@dataclass
class BattleResult:
    win_rate_d1: float   # fraction of episodes won by deck1
    win_rate_d2: float   # fraction of episodes won by deck2
    episodes: int
    deck1_id: str
    deck2_id: str


class BattleRunner:
    def __init__(
        self,
        checkpoint: str | None = None,
    ) -> None:
        if checkpoint is not None:
            self._checkpoint = checkpoint
        elif _DEFAULT_CHECKPOINT.exists():
            self._checkpoint = str(_DEFAULT_CHECKPOINT)
        else:
            self._checkpoint = None

        # Load saved model architecture (written by ygo-train).
        # Falls back to full-size defaults (matches downloaded pre-trained checkpoint).
        model_args_path = Path(self._checkpoint).parent / "model_args.json" if self._checkpoint else _DEFAULT_MODEL_ARGS
        if model_args_path.exists():
            self._model_args = json.loads(model_args_path.read_text())
        else:
            self._model_args = _FULL_MODEL

    def run(
        self,
        deck1: Deck,
        deck2: Deck,
        num_episodes: int = 128,
        seed: int = 0,
    ) -> BattleResult:
        if self._checkpoint is None:
            raise FileNotFoundError(
                "No RL checkpoint found. Download one or train with: ygo-train --archetypes ..."
            )

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
                "--checkpoint", _path(str(self._checkpoint)),
                "--llm-player", "-1",   # both players use RL; no step messages sent
                "--num-layers", str(self._model_args["num_layers"]),
                "--num-channels", str(self._model_args["num_channels"]),
                "--rnn-channels", str(self._model_args["rnn_channels"]),
                "--critic-width", str(self._model_args["critic_width"]),
                "--critic-depth", str(self._model_args["critic_depth"]),
            ]

            # Always use CPU for battle subprocesses. With num_envs=1 the game
            # engine is CPU-bound anyway, and GPU adds massive overhead: each
            # subprocess must init CUDA, JIT-compile the model, and allocate
            # device memory.  Multiple subprocesses fighting over one GPU causes
            # hangs and OOM.
            env = {
                **os.environ,
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
                "JAX_PLATFORMS": "cpu",
            }
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
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
                    if msg["type"] == "done":
                        wins[msg["winner"]] += 1
            except Exception:
                proc.kill()
                raise
            finally:
                proc.wait()

        if proc.returncode != 0:
            stderr = proc.stderr.read()
            total = wins[0] + wins[1]
            if _is_engine_crash(proc.returncode):
                stderr_tail = stderr.strip()
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
