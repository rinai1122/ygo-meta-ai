"""
CLI: ygo-train

Trains an RL agent via self-play PPO on the given deck pool.
Wraps vendor/ygo-agent/scripts/cleanba.py with single-machine defaults.
Saves checkpoint to checkpoints/agent.flax_model when done.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_VENDOR_SCRIPTS = _PROJECT_ROOT / "vendor" / "ygo-agent" / "scripts"
_CLEANBA = _VENDOR_SCRIPTS / "cleanba.py"
_CODE_LIST = (
    (_PROJECT_ROOT / "data" / "code_list.txt")
    if (_PROJECT_ROOT / "data" / "code_list.txt").exists()
    else _VENDOR_SCRIPTS / "code_list.txt"
)
_TOKENS_YDK = _PROJECT_ROOT / "vendor" / "ygo-agent" / "assets" / "deck" / "unsupported" / "_tokens.ydk"


def _python_exe() -> list[str]:
    venv = os.environ.get("YGOAGENT_VENV", "")
    if venv:
        for candidate in [Path(venv) / "bin" / "python", Path(venv) / "Scripts" / "python.exe"]:
            if candidate.exists():
                return [str(candidate)]
    return [sys.executable]


@app.command()
def main(
    archetypes: list[str] = typer.Option(..., help="Archetype names (subdirs of engines-dir) to train on"),
    engines_dir: Path = typer.Option(Path("data/engines"), help="Directory containing engine YDK subdirs"),
    staples_dir: Path = typer.Option(Path("data/staples"), help="Directory with staple YAML files"),
    checkpoints_dir: Path = typer.Option(Path("checkpoints"), help="Where to save checkpoints"),
    n_variants: int = typer.Option(8, help="Deck variants per archetype to populate the training pool"),
    timesteps: int = typer.Option(5_000_000, help="Total training timesteps"),
    num_envs: int = typer.Option(32, help="Parallel environments (8–64 for CPU)"),
    resume: Path = typer.Option(None, help="Resume from this .flax_model checkpoint"),
) -> None:
    """Train RL agent via PPO self-play on the given deck pool."""
    if not _CLEANBA.exists():
        console.print(
            f"[red]cleanba.py not found at {_CLEANBA}.\n"
            "Run: git submodule update --init --recursive[/red]"
        )
        raise typer.Exit(1)

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    abs_ckpt_dir = checkpoints_dir.resolve()

    # Round num_envs down to nearest multiple of 8 (cleanba batch-size constraint)
    num_envs = max(8, (num_envs // 8) * 8)
    num_minibatches = num_envs // 8

    from ygo_meta.deck_builder.generator import generate_all_decks
    from ygo_meta.deck_builder.ydk_parser import write_ydk

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        deck_dir = Path(tmpdir)

        # Generate full decks (engine + staple combos) — same as the simulation does
        decks = generate_all_decks(
            engines_dir=engines_dir,
            staples_dir=staples_dir,
            archetypes=archetypes,
            n_variants=n_variants,
            seed=0,
        )
        if not decks:
            console.print("[red]No decks generated — check archetypes and staples-dir.[/red]")
            raise typer.Exit(1)

        for deck in decks:
            write_ydk(deck, deck_dir / f"{deck.variant_id}.ydk")

        # Copy token deck so the compiled ygopro binary preloads token card data.
        # init_ygopro reads ALL *.ydk files in deck_dir; names starting with '_'
        # are excluded from the game pool but still passed to init_module, which
        # calls preload_deck() for them — loading token cards_data_ entries that
        # would otherwise be missing (causing card_reader_callback SIGABRT).
        if _TOKENS_YDK.exists():
            shutil.copy2(_TOKENS_YDK, deck_dir / "_tokens.ydk")

        console.print(f"[bold green]YGO RL Training[/bold green]")
        console.print(f"Archetypes : {archetypes}  ({len(decks)} deck variants in pool)")
        console.print(f"Steps      : {timesteps:,}  |  Envs: {num_envs}  |  Minibatches: {num_minibatches}")
        console.print(f"Ckpt dir   : {abs_ckpt_dir}\n")

        cmd = _python_exe() + [
            str(_CLEANBA),
            "--deck", str(deck_dir),
            "--code_list_file", str(_CODE_LIST),
            "--ckpt_dir", str(abs_ckpt_dir),
            "--total_timesteps", str(timesteps),
            "--local_num_envs", str(num_envs),
            "--num_actor_threads", "1",
            "--num_minibatches", str(num_minibatches),
            "--actor_device_ids", "0",
            "--learner_device_ids", "0",
            "--no-concurrency",
            "--save_interval", "100",
            "--tb_dir", "None",
        ]
        if resume:
            cmd += ["--checkpoint", str(resume.resolve())]

        env = {
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            # Disable AOT-compiled kernels; avoids SIGILL on CPUs that lack
            # prefer-no-gather / prefer-no-scatter LLVM pseudo-features that
            # jaxlib was compiled with but the WSL host CPU doesn't expose.
            "XLA_FLAGS": "--xla_cpu_use_thunk_runtime=false",
        }
        result = subprocess.run(cmd, cwd=str(_VENDOR_SCRIPTS), env=env)

    if result.returncode != 0:
        console.print(f"[red]Training failed (rc={result.returncode})[/red]")
        raise typer.Exit(result.returncode)

    latest = max(
        (p for p in abs_ckpt_dir.glob("*.flax_model") if p.name != "agent.flax_model"),
        key=lambda p: p.stat().st_mtime,
        default=None,
    )
    if latest:
        dest = abs_ckpt_dir / "agent.flax_model"
        shutil.copy2(latest, dest)
        console.print(f"\n[bold green]Training complete. Checkpoint saved: {dest}[/bold green]")
        console.print("Run evolution with: ygo-simulate --archetypes ...")
    else:
        console.print("[yellow]Training ended but no checkpoint was written.[/yellow]")


if __name__ == "__main__":
    app()
