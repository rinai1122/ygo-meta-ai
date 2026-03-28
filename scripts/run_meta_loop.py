"""
Convenience wrapper: run meta simulation with sensible defaults.
Edit variables below or use the CLI (python -m ygo_meta.cli.simulate).
"""

from pathlib import Path

from ygo_meta.simulation.evolution import run_evolution

ARCHETYPES = ["snake_eye", "blue_eyes", "hero"]
ENGINES_DIR = Path("data/engines")
STAPLES_DIR = Path("data/staples")
RESULTS_DIR = Path("results")
N_VARIANTS = 4
EPISODES = 128
GENERATIONS = 10

if __name__ == "__main__":
    history = run_evolution(
        archetypes=ARCHETYPES,
        engines_dir=ENGINES_DIR,
        staples_dir=STAPLES_DIR,
        results_dir=RESULTS_DIR,
        n_variants=N_VARIANTS,
        num_episodes=EPISODES,
        max_generations=GENERATIONS,
    )
    for r in history:
        print(f"Gen {r.generation}: top={r.top_decks[0]['deck_id'] if r.top_decks else '-'} converged={r.converged}")
