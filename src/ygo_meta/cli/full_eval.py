"""
CLI: ``ygo-eval-full`` — run baseline + tech-delta evaluation across every
unordered pair of archetypes, in two phases.

  Phase 1: enqueue baseline queries for every pair, wait for all baselines.
  Phase 2: enqueue tech-card queries for every pair, wait for all tech.

Resumability is automatic. ``JudgmentStore`` dedupes pending queries against
already-answered judgments, so killing this script and re-running picks up
exactly where you left off — every previously-judged hand is reused, every
unjudged query goes back into the queue.

Usage::

    # Terminal 1
    ygo-eval-server --store-dir results/judgments

    # Terminal 2
    ygo-eval-full \\
        --archetypes K9Vanquishsoul \\
        --archetypes BrandedDracotail \\
        --archetypes RyzealMitsurugi \\
        --archetypes SolfachordYummy
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.evaluator.archetype_loader import load_archetype_deck
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


def _load_tech_pool(pool_path: Path) -> list[tuple[int, str]]:
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
    if not baseline.main:
        raise ValueError("baseline deck has empty main")
    new_main = list(baseline.main)
    new_main[-1] = code
    return TechVariant(
        name=name,
        code=code,
        deck=Deck(
            archetype=baseline.archetype,
            variant_id=f"{baseline.variant_id}+{name.replace(' ', '_')}",
            main=new_main,
            extra=list(baseline.extra),
            side=list(baseline.side),
        ),
    )


@app.command()
def main(
    archetypes: list[str] = typer.Option(..., help="Archetype names; resolves to <engines-dir>/<name>/baseline.ydk"),
    engines_dir: Path = typer.Option(Path("data/engines")),
    tech_pool: Path = typer.Option(Path("data/staples/main_staple.yaml")),
    n_baseline: int = typer.Option(DEFAULT_N_BASELINE),
    n_tech: int = typer.Option(DEFAULT_N_TECH),
    store_dir: Path = typer.Option(Path("results/judgments")),
    out_dir: Path = typer.Option(Path("results/tech_deltas")),
    seed: int = typer.Option(0),
    eval_server_url: str = typer.Option("http://127.0.0.1:8000"),
    banlist_version: str = typer.Option("unknown"),
    skip_phase1: bool = typer.Option(False, "--skip-phase1", help="Jump straight to phase 2 (assumes baselines are already answered)"),
    clear: bool = typer.Option(False, "--clear", help="Wipe pending.jsonl before starting (keeps answered judgments). Use this to drop stale queries from previous runs."),
    clear_all: bool = typer.Option(False, "--clear-all", help="Wipe BOTH pending.jsonl and judgments.jsonl. Destroys all human work — confirm before using."),
) -> None:
    """Two-phase full meta evaluation. Resume by simply re-running."""
    if len(archetypes) < 2:
        console.print("[red]Need at least 2 archetypes.[/red]")
        raise typer.Exit(1)

    if clear_all:
        store_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("pending.jsonl", "judgments.jsonl"):
            p = store_dir / fname
            if p.exists():
                p.unlink()
                console.print(f"[yellow]--clear-all:[/yellow] removed {p}")
    elif clear:
        store_dir.mkdir(parents=True, exist_ok=True)
        p = store_dir / "pending.jsonl"
        if p.exists():
            p.unlink()
            console.print(f"[yellow]--clear:[/yellow] removed {p} (answered judgments preserved)")

    # Load every archetype baseline once.
    deck_by_name: dict[str, Deck] = {}
    for name in archetypes:
        deck_by_name[name] = load_archetype_deck(name, engines_dir)
    console.print(
        f"Loaded {len(deck_by_name)} archetype decks (engine + baseline): "
        + ", ".join(f"{n}({len(d.main)})" for n, d in deck_by_name.items())
    )

    tech_specs = _load_tech_pool(tech_pool)
    if not tech_specs:
        console.print(f"[red]Tech pool {tech_pool} is empty.[/red]")
        raise typer.Exit(1)
    console.print(f"Loaded {len(tech_specs)} tech candidates from {tech_pool}")

    # Per-archetype tech variants (baseline → list[TechVariant]).
    tech_variants_by_arch: dict[str, list[TechVariant]] = {
        name: [_make_tech_variant(deck, c, n) for c, n in tech_specs]
        for name, deck in deck_by_name.items()
    }

    pairs = list(combinations(archetypes, 2))
    K = len(tech_specs)
    per_pair = n_baseline + K * n_tech
    total = per_pair * len(pairs)
    console.print(
        f"\n{len(pairs)} unordered pairs × ({n_baseline} baseline + "
        f"{K}×{n_tech} tech) = {total} judgments total\n"
        f"[yellow]Open {eval_server_url}/ in a browser; "
        f"start `ygo-eval-server --store-dir {store_dir}` if not running.[/yellow]\n"
    )

    store = JudgmentStore(store_dir)

    # ---------- Phase 1: all baselines ----------
    if not skip_phase1:
        console.print("[bold cyan]Phase 1 — baseline evaluations[/bold cyan]")
        for a, b in pairs:
            added = enqueue_delta_queries(
                store=store,
                baseline=deck_by_name[a],
                opponent=deck_by_name[b],
                tech_variants=[],
                n_baseline=n_baseline,
                n_tech=0,
                seed=seed,
                include_baseline=True,
                include_tech=False,
            )
            console.print(f"  enqueued {a} vs {b}: +{added} new baseline queries")

        for a, b in pairs:
            console.print(f"  waiting for baseline {a} vs {b}…")
            wait_for_completion(
                store=store,
                baseline=deck_by_name[a],
                opponent=deck_by_name[b],
                tech_variants=[],
                n_baseline=n_baseline,
                n_tech=0,
                poll_interval=5.0,
                timeout=None,
            )
        console.print("[green]Phase 1 complete.[/green]\n")
    else:
        console.print("[yellow]--skip-phase1 set; assuming baselines are answered.[/yellow]\n")

    # ---------- Phase 2: all tech variants ----------
    console.print("[bold cyan]Phase 2 — tech-card delta evaluations[/bold cyan]")
    for a, b in pairs:
        added = enqueue_delta_queries(
            store=store,
            baseline=deck_by_name[a],
            opponent=deck_by_name[b],
            tech_variants=tech_variants_by_arch[a],
            n_baseline=n_baseline,
            n_tech=n_tech,
            seed=seed,
            include_baseline=False,
            include_tech=True,
        )
        console.print(f"  enqueued tech for {a} vs {b}: +{added} new queries")

    for a, b in pairs:
        console.print(f"  waiting for tech {a} vs {b}…")
        wait_for_completion(
            store=store,
            baseline=deck_by_name[a],
            opponent=deck_by_name[b],
            tech_variants=tech_variants_by_arch[a],
            n_baseline=n_baseline,
            n_tech=n_tech,
            poll_interval=5.0,
            timeout=None,
        )
    console.print("[green]Phase 2 complete.[/green]\n")

    # ---------- Compute & save per-pair results ----------
    out_dir.mkdir(parents=True, exist_ok=True)
    for a, b in pairs:
        p_b, n_b, results = compute_deltas(
            store=store,
            baseline=deck_by_name[a],
            opponent=deck_by_name[b],
            tech_variants=tech_variants_by_arch[a],
        )
        table = Table(title=f"{a} vs {b}  (baseline winrate {p_b:.1%}, n={n_b})")
        table.add_column("Tech")
        table.add_column("Δ", justify="right")
        table.add_column("SE(Δ)", justify="right")
        for r in results[:10]:
            table.add_row(r.tech_name, f"{r.delta:+.1%}", f"{r.se_delta:.1%}")
        console.print(table)

        out_path = out_dir / f"{a}_vs_{b}.json"
        out_path.write_text(json.dumps({
            "baseline": a,
            "opponent": b,
            "banlist_version": banlist_version,
            "n_baseline": n_b,
            "baseline_winrate": p_b,
            "results": [r.to_dict() for r in results],
        }, indent=2), encoding="utf-8")
    console.print(f"\n[green]All results saved to {out_dir}/[/green]")


if __name__ == "__main__":
    app()
