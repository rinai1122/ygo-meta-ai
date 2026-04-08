"""
CLI: evaluate tech-card deltas via the human web UI.

Usage::

    ygo-eval-tech-delta \\
        --baseline data/engines/K9Vanquishsoul/baseline.ydk \\
        --opponent data/engines/BrandedDracotail/baseline.ydk \\
        --tech 23434538:Maxx_C \\
        --tech 14558127:Ash_Blossom \\
        --tech 24224830:Called_by_the_Grave \\
        --n-baseline 36 --n-tech 12

Then in another shell::

    ygo-eval-server --store-dir results/judgments

Open http://127.0.0.1:8000/ and start judging. The CLI blocks until every
required judgment is in, then prints a ranked delta table and writes JSON
to ``results/tech_deltas/<baseline>_vs_<opponent>.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.deck_builder.ydk_parser import parse_ydk
from ygo_meta.evaluator.delta import (
    DEFAULT_N_BASELINE,
    DEFAULT_N_TECH,
    TechVariant,
    compute_deltas,
    enqueue_delta_queries,
    wait_for_completion,
)
from ygo_meta.evaluator.judgment_store import JudgmentStore

console = Console()
app = typer.Typer()


def _parse_tech_spec(spec: str) -> tuple[int, str]:
    """Parse 'CODE:Name' or just 'CODE' (used by --tech overrides)."""
    if ":" in spec:
        code_str, name = spec.split(":", 1)
        return int(code_str), name.replace("_", " ")
    return int(spec), spec


def _load_tech_pool(pool_path: Path) -> list[tuple[int, str]]:
    """Load (code, name) tuples from a staples-style YAML file.

    Accepts the same shape as ``data/staples/main_staple.yaml``: top-level
    keys map to lists of ``{code, name, copies_max}`` entries. ``name`` is
    optional and defaults to ``str(code)``.
    """
    data = yaml.safe_load(pool_path.read_text(encoding="utf-8"))
    if not data:
        return []
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for entries in data.values():
        if not isinstance(entries, list):
            continue
        for e in entries:
            code = int(e["code"])
            if code in seen:
                continue
            seen.add(code)
            out.append((code, e.get("name", str(code))))
    return out


def _make_tech_variant(baseline: Deck, code: int, name: str) -> TechVariant:
    """Build a tech variant by REPLACING the baseline's last main-deck slot
    with `code`. The baseline keeps its full 40 cards as the reference; the
    tech variant is also 40 cards, with the existing tech swapped for the
    candidate. Hand sampling pins the candidate and fills the remaining 4
    slots from the other 39 cards (handled by sampler.force_in_hand).
    """
    if not baseline.main:
        raise ValueError("baseline deck has empty main")
    new_main = list(baseline.main)
    new_main[-1] = code
    variant = Deck(
        archetype=baseline.archetype,
        variant_id=f"{baseline.variant_id}+{name.replace(' ', '_')}",
        main=new_main,
        extra=list(baseline.extra),
        side=list(baseline.side),
    )
    return TechVariant(name=name, code=code, deck=variant)


@app.command()
def main(
    baseline: Path = typer.Option(..., help="YDK file: baseline deck (last main slot is the flex slot)"),
    opponent: Path = typer.Option(..., help="YDK file: opponent deck"),
    tech_pool: Path = typer.Option(
        Path("data/staples/main_staple.yaml"),
        help="YAML staples file: every card in here is evaluated as a tech candidate",
    ),
    tech: list[str] = typer.Option(
        None,
        help="Optional override: 'CODE' or 'CODE:Name' (repeatable). If given, ignores --tech-pool.",
    ),
    n_baseline: int = typer.Option(DEFAULT_N_BASELINE, help="Baseline judgments (recommended ~3√K)"),
    n_tech: int = typer.Option(DEFAULT_N_TECH, help="Per-tech-card judgments"),
    store_dir: Path = typer.Option(Path("results/judgments"), help="JudgmentStore directory"),
    out_dir: Path = typer.Option(Path("results/tech_deltas"), help="Where to save the ranked JSON output"),
    seed: int = typer.Option(0),
    timeout: float = typer.Option(0.0, help="Seconds to wait for completion (0 = forever)"),
    eval_server_url: str = typer.Option("http://127.0.0.1:8000"),
    banlist_version: str = typer.Option("unknown"),
) -> None:
    """Enqueue tech-card delta queries and block until a human resolves them."""
    base_deck = parse_ydk(baseline)
    opp_deck = parse_ydk(opponent)

    if tech:
        tech_specs = [_parse_tech_spec(spec) for spec in tech]
    else:
        if not tech_pool.exists():
            console.print(f"[red]Tech pool not found: {tech_pool}[/red]")
            raise typer.Exit(1)
        tech_specs = _load_tech_pool(tech_pool)
        if not tech_specs:
            console.print(f"[red]Tech pool {tech_pool} is empty.[/red]")
            raise typer.Exit(1)
        console.print(f"Loaded {len(tech_specs)} tech candidates from {tech_pool}")
    tech_variants = [_make_tech_variant(base_deck, c, n) for c, n in tech_specs]
    K = len(tech_variants)
    total = n_baseline + n_tech * K

    store = JudgmentStore(store_dir)
    added = enqueue_delta_queries(
        store=store,
        baseline=base_deck,
        opponent=opp_deck,
        tech_variants=tech_variants,
        n_baseline=n_baseline,
        n_tech=n_tech,
        seed=seed,
    )

    console.print(f"[bold green]Tech-delta evaluation[/bold green]")
    console.print(f"  baseline: {baseline}")
    console.print(f"  opponent: {opponent}")
    console.print(f"  K={K} tech cards, n_baseline={n_baseline}, n_tech={n_tech}")
    console.print(f"  total queries: {total} (newly added: {added})")
    console.print(
        f"\n[yellow]Open {eval_server_url}/ in a browser, then run "
        f"`ygo-eval-server --store-dir {store_dir}` in another shell.[/yellow]\n"
    )

    def _progress(n_b: int, per_tech: list[int]) -> None:
        done = n_b + sum(per_tech)
        console.print(
            f"  [dim]progress: baseline {n_b}/{n_baseline}, "
            f"tech min/max {min(per_tech, default=0)}/{max(per_tech, default=0)} "
            f"of {n_tech} ({done}/{total} total)[/dim]"
        )

    try:
        wait_for_completion(
            store=store,
            baseline=base_deck,
            opponent=opp_deck,
            tech_variants=tech_variants,
            n_baseline=n_baseline,
            n_tech=n_tech,
            poll_interval=5.0,
            timeout=None if timeout <= 0 else timeout,
            on_progress=_progress,
        )
    except TimeoutError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    p_b, n_b, results = compute_deltas(store, base_deck, opp_deck, tech_variants)

    table = Table(title=f"Tech deltas (baseline winrate {p_b:.1%}, n={n_b})")
    table.add_column("Tech")
    table.add_column("WR with", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("SE(Δ)", justify="right")
    table.add_column("n", justify="right")
    for r in results:
        sign = "[green]" if r.delta > 0 else "[red]"
        table.add_row(
            r.tech_name,
            f"{r.tech_winrate:.1%}",
            f"{sign}{r.delta:+.1%}[/]",
            f"{r.se_delta:.1%}",
            str(r.n_tech),
        )
    console.print(table)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{baseline.stem}_vs_{opponent.stem}.json"
    out_path.write_text(json.dumps({
        "baseline_ydk": str(baseline),
        "opponent_ydk": str(opponent),
        "banlist_version": banlist_version,
        "n_baseline": n_b,
        "baseline_winrate": p_b,
        "results": [r.to_dict() for r in results],
    }, indent=2), encoding="utf-8")
    console.print(f"\n[green]Saved:[/green] {out_path}")


if __name__ == "__main__":
    app()
