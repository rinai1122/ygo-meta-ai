"""
CLI entry point: run the evolutionary meta simulation.

    python -m ygo_meta.cli.simulate \\
        --archetypes snake_eye blue_eyes hero \\
        --staples-dir data/staples/ \\
        --episodes 128 \\
        --generations 10
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer()


@app.command()
def main(
    archetypes: list[str] = typer.Option(..., help="Archetype names (subdirs of engines-dir)", metavar="ARCHETYPE"),
    engines_dir: Path = typer.Option(Path("data/engines"), help="Directory containing engine YDK subdirs"),
    staples_dir: Path = typer.Option(Path("data/staples"), help="Directory with staple YAML files"),
    results_dir: Path = typer.Option(Path("results"), help="Directory to save results"),
    n_variants: int = typer.Option(4, help="Staple combo variants per archetype"),
    episodes: int = typer.Option(128, help="Battle episodes per matchup"),
    generations: int = typer.Option(10, help="Max evolutionary generations"),
    epsilon: float = typer.Option(0.01, help="Convergence threshold"),
    seed: int = typer.Option(0, help="Random seed"),
) -> None:
    """Run evolutionary Nash equilibrium meta simulation."""
    from ygo_meta.simulation.evolution import run_evolution

    console.print(f"[bold green]YGO Meta Simulation[/bold green]")
    console.print(f"Archetypes: {archetypes}")
    console.print(f"Variants per archetype: {n_variants} → {len(archetypes) * n_variants} total decks")
    console.print(f"Episodes per matchup: {episodes}")
    console.print(f"Max generations: {generations}\n")

    history = run_evolution(
        archetypes=archetypes,
        engines_dir=engines_dir,
        staples_dir=staples_dir,
        results_dir=results_dir,
        n_variants=n_variants,
        num_episodes=episodes,
        max_generations=generations,
        epsilon=epsilon,
        seed=seed,
    )

    # Summary table
    table = Table(title="Meta Simulation Summary")
    table.add_column("Gen", justify="right")
    table.add_column("Top Deck")
    table.add_column("Weight", justify="right")
    table.add_column("Converged")

    for r in history:
        top = r.top_decks[0] if r.top_decks else {}
        table.add_row(
            str(r.generation),
            top.get("deck_id", "-"),
            f"{top.get('weight', 0):.3f}",
            "YES" if r.converged else "no",
        )

    console.print(table)
    console.print(f"\nResults saved to: {results_dir}/nash_solutions/")


if __name__ == "__main__":
    app()
