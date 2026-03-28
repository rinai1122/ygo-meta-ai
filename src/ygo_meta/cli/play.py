"""
CLI entry point: play a game using the LLM agent.

    python -m ygo_meta.cli.play --deck data/engines/snake_eye/engine.ydk

The LLM agent plays against the RL agent hosted by ygoinf.
The ygoinf server must already be running (python -m ygo_meta.engine.runner start).
"""

from __future__ import annotations

import typer
from pathlib import Path
from rich.console import Console

console = Console()
app = typer.Typer()


@app.command()
def main(
    deck: Path = typer.Option(..., help="Path to deck YDK file"),
    model: str = typer.Option("claude-opus-4-6", help="Claude model to use"),
    episodes: int = typer.Option(1, help="Number of games to play"),
) -> None:
    """Play games using the LLM agent against the RL agent."""
    from ygo_meta.llm_agent.agent import LLMAgent
    from ygo_meta.engine.client import AsyncYgoInfClient
    import anyio

    if not deck.exists():
        console.print(f"[red]Deck file not found: {deck}[/red]")
        raise typer.Exit(1)

    agent = LLMAgent(model=model)
    console.print(f"[green]LLM agent ready (model={model})[/green]")
    console.print(f"[yellow]Note: Full game loop requires ygoinf server integration.[/yellow]")
    console.print(f"[yellow]Deck: {deck}[/yellow]")
    console.print(
        "\nTo run full games, start the ygoinf server and integrate the game loop. "
        "The LLMAgent.choose_action(input_) method is ready to use."
    )


if __name__ == "__main__":
    app()
