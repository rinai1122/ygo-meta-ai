# YGO Meta AI

A Yu-Gi-Oh! AI system with two components:

1. **LLM Game-Playing Agent** — Claude receives the current game state and valid options at every decision point, then picks one.
2. **Evolutionary Deck-Building Meta-Solver** — M archetypes × N staple combinations play round-robin battles; Nash equilibrium identifies dominant strategies; the process repeats until the meta stabilizes.

The game engine is [sbl1996/ygo-agent](https://github.com/sbl1996/ygo-agent), integrated as a git submodule. Adding new cards requires only `git submodule update` + `python scripts/build_card_db.py`.

---

## Quick Start

```bash
# 1. Clone with submodule
git clone --recurse-submodules https://github.com/<your-username>/ygo-meta-ai.git
cd ygo-meta-ai

# 2. Install project deps
pip install -e ".[dev]"

# 3. Install ygo-agent deps (separate venv recommended due to JAX version pinning)
pip install -e ".[ygoagent]"

# 4. Build card name database
python scripts/build_card_db.py

# 5. Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# 6. Start ygoinf inference server
python -m ygo_meta.engine.runner start

# 7. Play (LLM agent vs RL agent)
python -m ygo_meta.cli.play --deck data/engines/snake_eye/engine.ydk

# 8. Run meta simulation
python -m ygo_meta.cli.simulate \
  --archetypes snake_eye blue_eyes hero \
  --staples-dir data/staples/ \
  --episodes 128 --generations 10
```

---

## Architecture

```
vendor/ygo-agent/          ← game engine (READ-ONLY submodule)
src/ygo_meta/
  engine/                  ← HTTP adapter to ygoinf + subprocess runner
  llm_agent/               ← game state text → Claude API → action index
  deck_builder/            ← YDK I/O, Deck dataclass, M×N generator
  simulation/              ← battles → payoff matrix → Nash eq → evolve
  cli/                     ← ygo-play, ygo-simulate entry points
data/
  engines/<arch>/engine.ydk   ← main engine cards (you provide these)
  staples/*.yaml              ← staple card pools
results/
  payoff_matrices/            ← gen_NNN.npy win-rate matrices
  nash_solutions/             ← gen_NNN.json Nash equilibrium results
```

---

## Adding a New Archetype

1. Create `data/engines/<archetype>/engine.ydk` with your main engine cards (YDK format, no staples).
2. Pass `--archetypes <archetype>` to `ygo-simulate`.

## Updating the Game Engine

```bash
git submodule update --remote vendor/ygo-agent
python scripts/build_card_db.py   # refresh card name database
```

---

## Meta Simulation

The simulation runs M×N decks (M archetypes, N staple combinations each) in round-robin battles.
Win rates form a D×D payoff matrix (D = M×N). Nash equilibrium is solved with [nashpy](https://github.com/drvinceknight/Nashpy).

Each generation:
- Dominant decks (high Nash weight) spawn variants with small staple mutations
- Extinct decks (Nash weight ≈ 0) are replaced with new random staple combos
- Only new matchups are run (results are cached)

Convergence: `max|sigma_{k+1} − sigma_k| < 0.01` with identical support across two generations.

Results are saved to `results/nash_solutions/gen_{k:03d}.json`.
