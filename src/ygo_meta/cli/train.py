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
import threading
import time
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


def _write_tokens_ydk(deck_dir: Path) -> None:
    """Write _tokens.ydk containing all token cards from the CDB.

    The compiled ygopro_ygoenv.so uses deck-based preloading: only cards that
    appear in a loaded deck file end up in cards_data_.  Tokens like the Primal
    Being Token (27204312) are never in player decks, so they would be absent at
    runtime and crash card_reader_callback when Nibiru activates.
    """
    import sqlite3
    cdb = _PROJECT_ROOT / "vendor" / "ygo-agent" / "assets" / "locale" / "en" / "cards.cdb"
    if not cdb.exists():
        return
    con = sqlite3.connect(str(cdb))
    TYPE_TOKEN = 0x4000
    token_ids = [r[0] for r in con.execute(
        "SELECT id FROM datas WHERE (type & ?) != 0", (TYPE_TOKEN,)
    ).fetchall()]
    con.close()
    tokens_ydk = deck_dir / "_tokens.ydk"
    with open(tokens_ydk, "w") as f:
        f.write("#main\n")
        for tid in token_ids:
            f.write(f"{tid}\n")
        f.write("#extra\n#side\n")


def _run_with_heartbeat(cmd: list, cwd: str, env: dict, interval: int = 60) -> subprocess.CompletedProcess:
    """Run a subprocess, streaming its output and printing a heartbeat every `interval`
    seconds of silence so the user knows training is still alive during long JIT phases."""
    last_output = [time.monotonic()]

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _stream() -> None:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            last_output[0] = time.monotonic()
        proc.stdout.close()

    reader = threading.Thread(target=_stream, daemon=True)
    reader.start()

    start = time.monotonic()
    while proc.poll() is None:
        time.sleep(5)
        silence = time.monotonic() - last_output[0]
        if silence >= interval:
            elapsed = int(time.monotonic() - start)
            console.print(
                f"[dim]  ... still running (no output for {int(silence)}s, "
                f"total elapsed {elapsed // 60}m{elapsed % 60:02d}s — "
                f"JAX JIT compilation can take several minutes)[/dim]"
            )
            last_output[0] = time.monotonic()  # reset so we don't spam

    reader.join()
    return subprocess.CompletedProcess(cmd, proc.returncode)


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
        # Reduce eval env pool: default 128 is far too large for a 2-env run and
        # causes unnecessary memory pressure.  Must be divisible by 4 (thread count
        # = local_eval_episodes // 4).
        local_eval_episodes = 8
    else:
        # Round to nearest multiple of 2; minimum 2
        num_envs = max(2, (num_envs // 2) * 2)
        num_steps = 128
        local_eval_episodes = 128  # cleanba default

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

        # Write _tokens.ydk so the compiled ygoenv.so can preload token card data
        # (e.g. Primal Being Token 27204312 summoned by Nibiru).  The compiled
        # binary uses deck-based preloading: tokens not in a deck file are absent
        # from cards_data_ at runtime → [card_reader_callback] Card not found crash.
        _write_tokens_ydk(deck_dir)

        mode_label = "small (low-memory)" if small else "full"
        console.print(f"[bold green]YGO RL Training[/bold green]  [{mode_label}]")
        console.print(f"Archetypes : {archetypes}  ({len(decks)} deck variants in pool)")
        console.print(f"Steps      : {timesteps:,}  |  Envs: {num_envs}  |  Minibatches: {num_minibatches}")
        console.print(f"Model      : ch={model['num_channels']} rnn={model['rnn_channels']} layers={model['num_layers']}")
        console.print(f"Ckpt dir   : {abs_ckpt_dir}\n")

        # Write a thin wrapper that enables preload_tokens=True in init_ygopro
        # before delegating to cleanba.  We cannot modify vendor directly.
        #
        # CRITICAL: cleanba.py line 35 contains:
        #   os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        # This is module-level code that overwrites XLA_FLAGS before any JAX call.
        # XLA initializes lazily (on the first jax.local_devices()), so our
        # --xla_cpu_use_thunk_runtime=false flag from the subprocess env would be lost.
        # Fix: combine all flags here and force-initialize JAX before runpy executes
        # cleanba.py, so cleanba's os.environ overwrite is a no-op (XLA is already up).
        _CLEANBA_XLA_FLAGS = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        wrapper = deck_dir / "_cleanba_wrapper.py"
        wrapper.write_text(
            "import os\n"
            # Combine cleanba's own flags with ours before JAX initializes.
            f"os.environ['XLA_FLAGS'] = {(_CLEANBA_XLA_FLAGS + ' --xla_cpu_use_thunk_runtime=false')!r}\n"
            # Set allocator flags explicitly in case the inherited env is stale.
            # JAX on CPU must use the platform (glibc) allocator; otherwise its
            # custom allocator and ygoenv's glibc heap corrupt each other → SIGABRT.
            "os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'\n"
            "os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'\n"
            # Force XLA to initialize NOW so cleanba.py's module-level os.environ
            # overwrite (line 35) cannot remove --xla_cpu_use_thunk_runtime=false.
            "import jax as _jax; _jax.local_devices()\n"
            "import ygoai.utils as _u\n"
            "_orig = _u.init_ygopro\n"
            "def _patched(env_id, lang, deck, code_list_file, preload_tokens=False, return_deck_names=False):\n"
            "    return _orig(env_id, lang, deck, code_list_file, preload_tokens=True, return_deck_names=return_deck_names)\n"
            "_u.init_ygopro = _patched\n"
            # cleanba.py uses SimpleNamespace() as a dummy TensorBoard writer but
            # never adds close() to it, then calls writer.close() at the end.
            # Patch SimpleNamespace so all instances have a no-op close().
            "import types as _t; _orig_SN = _t.SimpleNamespace\n"
            "class _NS(_orig_SN):\n"
            "    def close(self): pass\n"
            "_t.SimpleNamespace = _NS\n"
            f"import runpy; runpy.run_path({str(_CLEANBA)!r}, run_name='__main__')\n"
        )

        cmd = _python_exe() + [
            str(wrapper),
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
            # Disable TensorBoard for --small: TensorboardX's C++ backend crashes
            # when it receives NaN scalars (which occur during early small-batch
            # training).  Full runs have enough samples to avoid early NaN.
            "--tb_dir", "None" if small else str(abs_ckpt_dir / "tb"),
            "--local_eval_episodes", str(local_eval_episodes),
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
            # Unbuffered stdout so crash messages and Python output appear in order.
            # Without this, Python's block-buffered stdout through the pipe means
            # crash messages (written directly to stderr fd) appear before buffered
            # Python output, making it look like the crash happens earlier than it does.
            "PYTHONUNBUFFERED": "1",
        }
        result = _run_with_heartbeat(cmd, cwd=str(_VENDOR_SCRIPTS), env=env)

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
