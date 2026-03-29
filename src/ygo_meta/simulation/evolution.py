"""
Evolutionary meta simulation loop.

Each generation:
  1. Generate M×N decks (engine + staple combos)
  2. Run pairwise battles → complete payoff matrix
  3. Solve Nash equilibrium → sigma (distribution over decks)
  4. Dominant decks (sigma > 1/D) spawn variants with small staple mutations
  5. Extinct decks (sigma ≈ 0) are replaced with new random combos
  6. Repeat until convergence or max_generations reached

Convergence: max|sigma_{k+1} - sigma_k| < epsilon AND identical support.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ygo_meta.deck_builder.deck_model import Deck
from ygo_meta.deck_builder.generator import generate_all_decks
from ygo_meta.simulation.battle_runner import BattleRunner
from ygo_meta.simulation.nash import NashSolution, solve_nash
from ygo_meta.simulation.payoff_matrix import PayoffMatrix, build_matrix


@dataclass
class GenerationResult:
    generation: int
    deck_ids: list[str]
    nash_sigma: list[float]
    top_decks: list[dict]
    converged: bool
    method: str


def _mutate_staple_combo(
    deck: Deck,
    pool: list[dict],
    rng: random.Random,
    n_swaps: int = 1,
) -> Deck:
    """Return a new deck variant with 1–2 staple entries swapped."""
    if not pool or not deck.staple_combo:
        return deck

    combo = dict(deck.staple_combo)
    # Remove one existing staple entry
    if combo:
        victim = rng.choice(list(combo.keys()))
        removed_copies = combo.pop(victim)
    else:
        removed_copies = 0

    # Add one new staple entry from the pool not already in combo
    candidates = [e for e in pool if e.get("name", str(e["code"])) not in combo]
    if candidates:
        entry = rng.choice(candidates)
        name = entry.get("name", str(entry["code"]))
        copies = min(entry.get("copies_max", 3), removed_copies or 3)
        combo[name] = copies

    # Rebuild main deck: engine cards + new staples
    # Re-parse engine from original (engine cards are deck.main minus staple cards)
    # We approximate by keeping the engine portion (non-staple cards)
    # Since we store staple_combo separately, reconstruct:
    staple_codes_old = []
    # This approximation rebuilds staples from the updated combo
    staple_codes_new: list[int] = []
    code_by_name = {e.get("name", str(e["code"])): e["code"] for e in pool}
    for name_, copies in combo.items():
        code = code_by_name.get(name_)
        if code:
            staple_codes_new.extend([code] * copies)

    # Engine cards = first (40 - sum of old staples) entries of original main
    old_staple_total = sum(deck.staple_combo.values())
    engine_main = deck.main[: len(deck.main) - old_staple_total]
    new_main = engine_main + staple_codes_new

    # Pad or trim to original length
    target = len(deck.main)
    if len(new_main) < target and engine_main:
        while len(new_main) < target:
            new_main.append(engine_main[len(new_main) % len(engine_main)])
    new_main = new_main[:60]

    # Create new variant_id
    base = deck.variant_id.rsplit("_", 1)[0]
    new_id = f"{base}_m{rng.randint(0, 9999):04d}"

    return Deck(
        archetype=deck.archetype,
        variant_id=new_id,
        main=new_main,
        extra=deck.extra,
        side=deck.side,
        staple_combo=combo,
    )


def _load_staple_pool(staples_dir: Path) -> list[dict]:
    import yaml
    pool: list[dict] = []
    for yaml_file in sorted(staples_dir.glob("*.yaml")):
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data:
            continue
        for entries in data.values():
            if isinstance(entries, list):
                pool.extend(entries)
    return pool


def run_evolution(
    archetypes: list[str],
    engines_dir: Path,
    staples_dir: Path,
    results_dir: Path,
    n_variants: int = 4,
    num_episodes: int = 128,
    max_generations: int = 10,
    epsilon: float = 0.01,
    seed: int = 0,
    checkpoint: str | None = None,
    runner: object | None = None,
) -> list[GenerationResult]:
    rng = random.Random(seed)
    if runner is None:
        runner = BattleRunner(checkpoint=checkpoint)
    results_dir.mkdir(parents=True, exist_ok=True)
    pm_dir = results_dir / "payoff_matrices"
    ns_dir = results_dir / "nash_solutions"
    pm_dir.mkdir(exist_ok=True)
    ns_dir.mkdir(exist_ok=True)

    pool = _load_staple_pool(staples_dir)
    history: list[GenerationResult] = []
    prev_sigma: np.ndarray | None = None

    decks = generate_all_decks(
        engines_dir=engines_dir,
        staples_dir=staples_dir,
        archetypes=archetypes,
        n_variants=n_variants,
        seed=seed,
    )

    for gen in range(max_generations):
        print(f"\n=== Generation {gen} ({len(decks)} decks) ===")

        # Load cached matrix if it exists
        npy_path = pm_dir / f"gen_{gen:03d}.npy"
        ids_path = pm_dir / f"gen_{gen:03d}_deck_ids.json"
        if npy_path.exists() and ids_path.exists():
            pm = PayoffMatrix.load(npy_path, ids_path)
            # Add any new decks not in the cached matrix
            existing_ids = set(pm.deck_ids)
            new_deck_ids = [d.variant_id for d in decks if d.variant_id not in existing_ids]
            if new_deck_ids:
                # Rebuild with all decks, preserving existing results
                all_ids = pm.deck_ids + new_deck_ids
                new_pm = PayoffMatrix(all_ids)
                for i, id_i in enumerate(pm.deck_ids):
                    for j, id_j in enumerate(pm.deck_ids):
                        new_pm.matrix[i, j] = pm.matrix[i, j]
                pm = new_pm
        else:
            pm = PayoffMatrix([d.variant_id for d in decks])

        pm = build_matrix(decks, runner, num_episodes=num_episodes, seed=seed + gen)
        pm.save(npy_path, ids_path)

        solution = solve_nash(pm.matrix)
        sigma = solution.sigma
        top = solution.top_decks(pm.deck_ids, top_n=3)

        converged = False
        if prev_sigma is not None and len(prev_sigma) == len(sigma):
            diff = float(np.max(np.abs(sigma - prev_sigma)))
            prev_support = set(np.where(prev_sigma > 1e-4)[0])
            curr_support = set(np.where(sigma > 1e-4)[0])
            if diff < epsilon and prev_support == curr_support:
                converged = True

        gen_result = GenerationResult(
            generation=gen,
            deck_ids=pm.deck_ids,
            nash_sigma=sigma.tolist(),
            top_decks=top,
            converged=converged,
            method=solution.method,
        )
        history.append(gen_result)

        ns_path = ns_dir / f"gen_{gen:03d}.json"
        ns_path.write_text(json.dumps(asdict(gen_result), indent=2), encoding="utf-8")
        print(f"  Nash ({solution.method}): top decks = {[t['deck_id'] for t in top]}")

        if converged:
            print(f"\nConverged at generation {gen}.")
            break

        prev_sigma = sigma

        # --- Evolve deck pool ---
        n_total = len(decks)
        dominant_threshold = 1.0 / n_total
        deck_map = {d.variant_id: d for d in decks}
        new_decks: list[Deck] = []

        for i, did in enumerate(pm.deck_ids):
            w = sigma[i]
            deck = deck_map.get(did)
            if deck is None:
                continue
            if w > dominant_threshold:
                # Keep deck + 2 mutant variants
                new_decks.append(deck)
                for _ in range(2):
                    new_decks.append(_mutate_staple_combo(deck, pool, rng))
            elif w < 1e-4:
                # Replace with a fresh random combo
                from ygo_meta.deck_builder.generator import generate_decks
                fresh = generate_decks(
                    engine_path=engines_dir / deck.archetype / "engine.ydk",
                    staples_dir=staples_dir,
                    n_variants=1,
                    seed=rng.randint(0, 10**6),
                )
                if fresh:
                    new_decks.extend(fresh)
            else:
                # Borderline: keep as-is
                new_decks.append(deck)

        # Ensure at least n_variants per archetype
        arch_counts: dict[str, int] = {}
        for d in new_decks:
            arch_counts[d.archetype] = arch_counts.get(d.archetype, 0) + 1

        for arch in archetypes:
            if arch_counts.get(arch, 0) < n_variants:
                from ygo_meta.deck_builder.generator import generate_decks
                fill = generate_decks(
                    engine_path=engines_dir / arch / "engine.ydk",
                    staples_dir=staples_dir,
                    n_variants=n_variants - arch_counts.get(arch, 0),
                    seed=rng.randint(0, 10**6),
                )
                new_decks.extend(fill)

        decks = new_decks

    return history
