"""CLI: run the round-robin gauntlet and save the empirical payoff matrix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from ygo_meta.meta_solver.gauntlet import load_decks, run_gauntlet

app = typer.Typer()
console = Console()


@app.command()
def main(
    decks_dir: Path = typer.Option(..., help="Directory containing one .ydk per deck"),
    output_dir: Path = typer.Option(Path("results/gauntlet"), help="Output dir"),
    episodes: int = typer.Option(500, help="BO1 games per matchup"),
    seed: int = typer.Option(0),
    lflist: Path | None = typer.Option(
        Path("data/lflist/master_duel.lflist.conf"),
        help="Banlist file used to validate decks before running",
    ),
    one_seat: bool = typer.Option(
        False,
        "--one-seat",
        help="Run only A(P0) vs B(P1). Faster, but keeps player-slot bias.",
    ),
    skip_validation: bool = typer.Option(False, "--skip-validation"),
) -> None:
    decks = load_decks(decks_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_npy = output_dir / "payoff.npy"
    cache_ids = output_dir / "deck_ids.json"

    console.print(f"  gauntlet: {len(decks)} decks, {episodes} episodes/pair")
    result = run_gauntlet(
        decks,
        num_episodes=episodes,
        seed=seed,
        cache_path=cache_npy,
        cache_ids_path=cache_ids,
        seat_balanced=not one_seat,
        validate_decks=not skip_validation,
        lflist_path=lflist,
    )

    np.save(output_dir / "payoff.npy", result.payoff)
    (output_dir / "deck_ids.json").write_text(
        json.dumps(result.deck_ids, indent=2), encoding="utf-8"
    )
    console.print(f"  saved payoff matrix → {cache_npy}")


if __name__ == "__main__":
    app()
