"""
CLI entry point: play games with the LLM agent vs random or another LLM agent.

Decks are built on-the-fly from a fixed engine + random main-deck staples
(handtraps / backrow).  The extra deck is taken directly from the engine YDK.

    # LLM (BrandedDracotail engine) vs random, 3 games:
    python -m ygo_meta.cli.play

    # Specify a different engine and more episodes:
    python -m ygo_meta.cli.play --engine K9Vanquishsoul --episodes 10

    # LLM vs LLM:
    python -m ygo_meta.cli.play --opponent llm

Requires ygoenv + ygoai in the YGOAGENT_VENV (or current venv if installed there).
Set YGOAGENT_VENV to point at the venv containing ygoenv/ygoai/ygoinf.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path, PureWindowsPath

import typer
from dotenv import load_dotenv

load_dotenv()
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer()

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_GAME_RUNNER  = _PROJECT_ROOT / "scripts" / "game_runner.py"
_VENDOR_ROOT  = _PROJECT_ROOT / "vendor" / "ygo-agent"
_CODE_LIST    = _VENDOR_ROOT / "scripts" / "code_list.txt"
_CARD_NAMES   = _PROJECT_ROOT / "data" / "card_names.json"
_ENGINES_DIR  = _PROJECT_ROOT / "data" / "engines"
_STAPLES_DIR  = _PROJECT_ROOT / "data" / "staples"
_VENDOR_DECK  = _VENDOR_ROOT / "assets" / "deck"


def _load_valid_codes() -> set[int]:
    """Return the set of card codes known to the vendor game engine (code_list.txt)."""
    codes: set[int] = set()
    with open(_CODE_LIST) as f:
        for line in f:
            parts = line.split()
            if parts:
                try:
                    codes.add(int(parts[0]))
                except ValueError:
                    pass
    return codes


def _validate_deck_codes(deck_path: Path, valid_codes: set[int]) -> list[int]:
    """Return any card codes in the YDK that are absent from the vendor database."""
    missing: list[int] = []
    section = None
    for line in deck_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#main"):
            section = "main"
        elif line.startswith("#extra"):
            section = "extra"
        elif line.startswith("!side"):
            section = "side"
        elif line and not line.startswith("#") and section:
            try:
                code = int(line)
                if code not in valid_codes:
                    missing.append(code)
            except ValueError:
                pass
    return missing


def _load_dotenv() -> None:
    """Load key=value pairs from .env into os.environ (missing keys only)."""
    env_file = _PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def _win_to_wsl(path: str) -> str:
    """Convert a Windows path like C:\\foo\\bar to /mnt/c/foo/bar."""
    p = PureWindowsPath(path)
    if p.drive:
        drive = p.drive.rstrip(":").lower()
        rest = "/".join(p.parts[1:])
        return f"/mnt/{drive}/{rest}"
    return path.replace("\\", "/")


def _python_exe() -> list[str]:
    """Return the command prefix to invoke Python for the game runner.

    On Windows with a WSL venv path (starts with '/'), use 'wsl python3'.
    Otherwise look for bin/python or Scripts/python.exe under the venv.
    Returns a list so callers can do: cmd = _python_exe() + [script, ...]
    """
    venv = os.environ.get("YGOAGENT_VENV", "")
    if venv:
        if sys.platform == "win32" and (venv.startswith("/") or venv.startswith("\\\\wsl")):
            return ["wsl", f"{venv}/bin/python3"]
        for candidate in [
            Path(venv) / "bin" / "python",
            Path(venv) / "Scripts" / "python.exe",
        ]:
            if candidate.exists():
                return [str(candidate)]
        console.print(f"[yellow]Warning: YGOAGENT_VENV={venv} set but Python not found there; using current Python[/yellow]")
    return [sys.executable]


def _build_deck(engine: str, staples_dir: Path, seed: int) -> Path:
    """Generate one deck from engine + main-deck staples, write to a temp YDK, return its path."""
    from ygo_meta.deck_builder.generator import generate_decks
    from ygo_meta.deck_builder.ydk_parser import write_ydk

    engine_path = _ENGINES_DIR / engine / "engine.ydk"
    if not engine_path.exists():
        console.print(f"[red]Engine not found: {engine_path}[/red]")
        raise typer.Exit(1)

    # Validate that all engine cards are known to the vendor game database.
    valid_codes = _load_valid_codes()
    missing = _validate_deck_codes(engine_path, valid_codes)
    if missing:
        import json
        try:
            with open(_CARD_NAMES, encoding="utf-8") as f:
                names = json.load(f)
        except Exception:
            names = {}
        unique_missing = sorted(set(missing))
        console.print(f"[red]Engine '{engine}' has {len(unique_missing)} card(s) not in the vendor game database:[/red]")
        for c in unique_missing[:10]:
            console.print(f"  {c}  {names.get(str(c), '??')}")
        if len(unique_missing) > 10:
            console.print(f"  ... and {len(unique_missing) - 10} more")
        console.print(
            "\n[yellow]The vendor game database is too old to support these cards.[/yellow]\n"
            "[yellow]Fix: run  git submodule update --init --recursive  to pull the latest database.[/yellow]\n"
            "[yellow]Workaround: pass --deck <path-to-ydk>  to use a vendor deck directly, e.g.:[/yellow]\n"
            f"[yellow]  --deck {_VENDOR_DECK / 'Branded.ydk'}[/yellow]"
        )
        raise typer.Exit(1)

    decks = generate_decks(
        engine_path=engine_path,
        staples_dir=staples_dir,
        n_variants=1,
        target_size=40,
        seed=seed,
        main_only=True,
    )
    tmp_dir = Path(tempfile.mkdtemp())
    deck_path = tmp_dir / f"{engine}.ydk"
    write_ydk(decks[0], deck_path)
    return deck_path


@app.command()
def main(
    engine: str = typer.Option("BrandedDracotail", help="Archetype engine name (subdirectory of data/engines/)"),
    deck: Path = typer.Option(None, help="Use a pre-built YDK file directly, bypassing engine generation"),
    staples_dir: Path = typer.Option(_STAPLES_DIR, help="Directory containing staple YAML files"),
    opponent: str = typer.Option("random", help="Opponent type: random, llm, or rl"),
    checkpoint: Path = typer.Option(None, help="Path to RL agent .flax_model checkpoint (required for --opponent rl)"),
    num_embeddings: int = typer.Option(None, help="RL model embedding size (must match checkpoint)"),
    llm_player: int = typer.Option(0, help="Which player index is the LLM (0 or 1)"),
    episodes: int = typer.Option(3, help="Number of games to play"),
    seed: int = typer.Option(42, help="Random seed (also used for deck generation)"),
    provider: str = typer.Option("anthropic", help="LLM provider: anthropic or gemini"),
    model: str = typer.Option(None, help="Model name (default: claude-opus-4-6 for anthropic, gemini-2.5-flash for gemini)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print each prompt and response"),
) -> None:
    """Play games with the LLM agent against a random or another LLM opponent."""
    _load_dotenv()

    # --- Pre-flight checks ---
    for label, path in [
        ("game_runner.py", _GAME_RUNNER),
        ("code_list.txt", _CODE_LIST),
        ("card_names.json", _CARD_NAMES),
        ("staples dir", staples_dir),
    ]:
        if not path.exists():
            console.print(f"[red]{label} not found: {path}[/red]")
            if label == "card_names.json":
                console.print("Run: python scripts/build_card_db.py")
            raise typer.Exit(1)

    if opponent not in ("random", "llm", "rl"):
        console.print(f"[red]Unknown opponent '{opponent}'. Use 'random', 'llm', or 'rl'.[/red]")
        raise typer.Exit(1)

    if opponent == "rl" and checkpoint is None:
        console.print("[red]--checkpoint is required when --opponent rl[/red]")
        raise typer.Exit(1)

    if opponent == "rl" and not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    if provider not in ("anthropic", "gemini"):
        console.print(f"[red]Unknown provider '{provider}'. Use 'anthropic' or 'gemini'.[/red]")
        raise typer.Exit(1)

    _defaults = {"anthropic": "claude-opus-4-6", "gemini": "gemini-2.5-flash"}
    model = model or _defaults[provider]

    # --- Resolve deck path ---
    if deck is not None:
        # Direct deck override: validate it exists and its codes are in the vendor DB.
        if not deck.exists():
            console.print(f"[red]Deck file not found: {deck}[/red]")
            raise typer.Exit(1)
        valid_codes = _load_valid_codes()
        missing = _validate_deck_codes(deck, valid_codes)
        if missing:
            console.print(f"[yellow]Warning: {len(set(missing))} card code(s) in {deck.name} are not in the vendor database.[/yellow]")
        deck_path = deck
        console.print(f"[green]Using deck: {deck_path.name}[/green]")
    else:
        # Generate from engine + main-deck staples.
        deck_path = _build_deck(engine, staples_dir, seed)
        console.print(f"[green]Deck built: {engine} (main-deck staples only) → {deck_path.name}[/green]")

    # --- Initialize LLM agent(s) ---
    from ygo_meta.llm_agent.agent import LLMAgent

    llm0 = LLMAgent(model=model, provider=provider)
    llm1 = LLMAgent(model=model, provider=provider) if opponent == "llm" else None

    console.print(f"[green]LLM agent ready (provider={provider}, model={model})[/green]")
    console.print(
        f"Player 0: {'LLM' if llm_player == 0 else opponent} | "
        f"Player 1: {'LLM' if llm_player == 1 else opponent} | "
        f"Episodes: {episodes}"
    )

    # --- Build subprocess command ---
    _path = _win_to_wsl if sys.platform == "win32" and os.environ.get("YGOAGENT_VENV", "").startswith("/") else str
    cmd = _python_exe() + [
        _path(str(_GAME_RUNNER)),
        "--deck", _path(str(deck_path)),
        "--num-episodes", str(episodes),
        "--seed", str(seed),
        "--code-list", _path(str(_CODE_LIST)),
        "--card-names", _path(str(_CARD_NAMES)),
        "--llm-player", str(llm_player),
    ]
    if opponent == "rl":
        cmd += ["--checkpoint", _path(str(checkpoint))]
        if num_embeddings is not None:
            cmd += ["--num-embeddings", str(num_embeddings)]

    # --- Run game loop ---
    wins = {0: 0, 1: 0}

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

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
                if verbose:
                    console.print(f"[dim]{line}[/dim]")
                continue

            if msg["type"] == "step":
                action = _pick_action(msg, llm_player, opponent, llm0, llm1, verbose)
                proc.stdin.write(json.dumps({"action": action}) + "\n")
                proc.stdin.flush()

            elif msg["type"] == "done":
                ep_idx = msg["ep_idx"]
                winner = msg["winner"]
                turns = msg["turns"]
                wins[winner] += 1
                _who = _player_label(winner, llm_player, opponent, model)
                console.print(f"Game {ep_idx + 1}: {_who} wins (turns={turns})")

        proc.wait()

        if proc.returncode not in (0, None):
            stderr_out = proc.stderr.read()
            console.print(f"[red]Game runner exited with code {proc.returncode}[/red]")
            if stderr_out:
                console.print(stderr_out)
            raise typer.Exit(1)

    except KeyboardInterrupt:
        proc.terminate()
        console.print("\n[yellow]Interrupted.[/yellow]")

    # --- Results table ---
    total = sum(wins.values())
    if total == 0:
        console.print("[yellow]No games completed.[/yellow]")
        return

    table = Table(title=f"Results ({total} game(s))")
    table.add_column("Player")
    table.add_column("Role")
    table.add_column("Wins")
    table.add_column("Win Rate")

    for p in (0, 1):
        role = _player_label(p, llm_player, opponent, model)
        table.add_row(f"Player {p}", role, str(wins[p]), f"{wins[p] / total:.1%}")

    console.print(table)


def _pick_action(
    msg: dict,
    llm_player: int,
    opponent: str,
    llm0: "LLMAgent",
    llm1: "LLMAgent | None",
    verbose: bool,
) -> int:
    to_play: int = msg["to_play"]
    num_options: int = msg["num_options"]
    prompt: str = msg["prompt"]

    use_llm = (to_play == llm_player) or (opponent == "llm" and to_play != llm_player)
    agent = llm0 if to_play == llm_player else llm1

    if use_llm and agent is not None:
        if verbose:
            console.print(f"\n[cyan]--- Player {to_play} (LLM) decision ---[/cyan]")
            console.print(prompt)
        action = agent.choose_action_from_text(prompt, num_options)
        if verbose:
            console.print(f"[green]→ Chose action {action}[/green]")
    else:
        action = random.randint(0, num_options - 1)
        if verbose:
            # Find the description for the chosen action so chain/activate decisions
            # are visible (e.g. "Player 1 (random) → [0] Activate [Maxx \"C\"]").
            chosen_desc = f"action {action}"
            for line in prompt.splitlines():
                if line.startswith(f"[{action}]"):
                    chosen_desc = line
                    break
            console.print(f"[dim]Player {to_play} (random) → {chosen_desc}[/dim]")

    return action


def _player_label(player: int, llm_player: int, opponent: str, model: str) -> str:
    if player == llm_player:
        return f"LLM ({model})"
    if opponent == "llm":
        return f"LLM ({model})"
    if opponent == "rl":
        return "RL Agent"
    return "Random"


if __name__ == "__main__":
    app()
