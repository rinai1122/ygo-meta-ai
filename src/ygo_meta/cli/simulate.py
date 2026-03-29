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
from dotenv import load_dotenv

load_dotenv()
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
    evaluator: str = typer.Option("rl", help="Battle evaluator: 'rl' (pre-trained agent), 'llm' (Claude/Gemini), or 'random' (free, for debugging)"),
    checkpoint: Path = typer.Option(None, help="RL checkpoint (.flax_model). Auto-detects checkpoints/agent.flax_model if omitted."),
    provider: str = typer.Option("anthropic", help="LLM provider (llm evaluator only): anthropic or gemini"),
    model: str = typer.Option(None, help="LLM model name (llm evaluator only; default: claude-opus-4-6 / gemini-2.5-flash)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print LLM prompts and responses to stdout"),
) -> None:
    """Run evolutionary Nash equilibrium meta simulation."""
    from ygo_meta.simulation.evolution import run_evolution

    if evaluator not in ("rl", "llm", "random"):
        console.print(f"[red]Unknown evaluator '{evaluator}'. Use 'rl', 'llm', or 'random'.[/red]")
        raise typer.Exit(1)

    runner = None
    if evaluator == "rl":
        from ygo_meta.simulation.battle_runner import BattleRunner
        runner = BattleRunner(checkpoint=str(checkpoint) if checkpoint else None)
    elif evaluator == "llm":
        from ygo_meta.simulation.llm_battle_runner import LLMBattleRunner
        runner = LLMBattleRunner(provider=provider, model=model, verbose=verbose)
    elif evaluator == "random":
        from ygo_meta.simulation.llm_battle_runner import RandomBattleRunner
        runner = RandomBattleRunner(verbose=verbose)

    console.print(f"[bold green]YGO Meta Simulation[/bold green]")
    console.print(f"Archetypes: {archetypes}")
    console.print(f"Variants per archetype: {n_variants} → {len(archetypes) * n_variants} total decks")
    console.print(f"Episodes per matchup: {episodes}")
    console.print(f"Max generations: {generations}")
    evaluator_label = evaluator
    if evaluator == "llm":
        evaluator_label += f" ({provider} / {model or 'default'})"
    console.print(f"Evaluator: {evaluator_label}\n")

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
        runner=runner,
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
