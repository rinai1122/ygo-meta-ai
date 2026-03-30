"""
CLI: ygo-train

Trains an RL agent via self-play PPO on the given deck pool.
Wraps vendor/ygo-agent/scripts/cleanba.py with single-machine defaults.
Saves checkpoint to checkpoints/agent.flax_model when done.
Also saves checkpoints/model_args.json with the architecture so BattleRunner
can load the right model shape.
"""
from __future__ import annotations

import json
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

# Small model preset for CPU / low-memory machines (fits in ~2GB WSL).
_SMALL_MODEL = dict(num_layers=1, num_channels=32, rnn_channels=64, critic_width=32, critic_depth=1)
# Full model (default cleanba settings)
_FULL_MODEL  = dict(num_layers=2, num_channels=128, rnn_channels=512, critic_width=128, critic_depth=3)


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
    num_envs: int = typer.Option(32, help="Parallel environments"),
    resume: Path = typer.Option(None, help="Resume from this .flax_model checkpoint"),
    save_interval: int = typer.Option(100, help="Save checkpoint every N updates (lower for short debug runs)"),
    small: bool = typer.Option(False, "--small", help="Use tiny model + minimal envs to fit in ~2GB RAM (CPU/low-memory machines)"),
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

    model = _SMALL_MODEL if small else _FULL_MODEL

    if small:
        # Override to absolute minimums for low-memory runs
        num_envs = 2
        num_steps = 32
    else:
        # Round to nearest multiple of 2; minimum 2
        num_envs = max(2, (num_envs // 2) * 2)
        num_steps = 128

    num_minibatches = max(1, num_envs // 8)

    from ygo_meta.deck_builder.generator import generate_all_decks
    from ygo_meta.deck_builder.ydk_parser import write_ydk

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        deck_dir = Path(tmpdir)

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

        mode_label = "small (low-memory)" if small else "full"
        console.print(f"[bold green]YGO RL Training[/bold green]  [{mode_label}]")
        console.print(f"Archetypes : {archetypes}  ({len(decks)} deck variants in pool)")
        console.print(f"Steps      : {timesteps:,}  |  Envs: {num_envs}  |  Minibatches: {num_minibatches}")
        console.print(f"Model      : ch={model['num_channels']} rnn={model['rnn_channels']} layers={model['num_layers']}")
        console.print(f"Ckpt dir   : {abs_ckpt_dir}\n")

        cmd = _python_exe() + [
            str(_CLEANBA),
            "--deck", str(deck_dir),
            "--code_list_file", str(_CODE_LIST),
            "--ckpt_dir", str(abs_ckpt_dir),
            "--total_timesteps", str(timesteps),
            "--local_num_envs", str(num_envs),
            "--local_env_threads", "1",
            "--num_steps", str(num_steps),
            "--num_actor_threads", "1",
            "--num_minibatches", str(num_minibatches),
            "--actor_device_ids", "0",
            "--learner_device_ids", "0",
            "--no-concurrency",
            "--save_interval", str(save_interval),
            "--tb_dir", str(abs_ckpt_dir / "tb"),
            "--timeout", "3600",
            f"--m1.num-layers", str(model["num_layers"]),
            f"--m1.num-channels", str(model["num_channels"]),
            f"--m1.rnn-channels", str(model["rnn_channels"]),
            f"--m1.critic-width", str(model["critic_width"]),
            f"--m1.critic-depth", str(model["critic_depth"]),
            f"--m2.num-layers", str(model["num_layers"]),
            f"--m2.num-channels", str(model["num_channels"]),
            f"--m2.rnn-channels", str(model["rnn_channels"]),
            f"--m2.critic-width", str(model["critic_width"]),
            f"--m2.critic-depth", str(model["critic_depth"]),
        ]
        if resume:
            cmd += ["--checkpoint", str(resume.resolve())]

        env = {
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_cpu_use_thunk_runtime=false",
            # Force JAX to use platform (glibc) allocator instead of its custom BFC
            # allocator. Without this, JAX's allocator and ygoenv's glibc allocator
            # corrupt each other's heap metadata → SIGABRT in ygoenv.make().
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
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
        # Save model architecture so BattleRunner can load the right shape
        args_path = abs_ckpt_dir / "model_args.json"
        args_path.write_text(json.dumps(model))
        console.print(f"\n[bold green]Training complete. Checkpoint: {dest}[/bold green]")
        console.print(f"Model args : {args_path}")
        console.print("Run simulation: ygo-simulate --evaluator rl --archetypes ...")
    else:
        console.print("[yellow]Training ended but no checkpoint was written.[/yellow]")


if __name__ == "__main__":
    app()
