# CLAUDE.md

## Project: YGO Meta AI

Two subsystems built on top of `sbl1996/ygo-agent` (vendored as a git submodule):

1. **LLM game-playing agent** — receives valid options at each decision point, picks one via Claude API
2. **Evolutionary deck-building meta-solver** — M archetypes × N staple combos, Nash equilibrium, repeat until stable

---

## Architecture

```
vendor/ygo-agent/       ← git submodule, READ-ONLY, never modify
src/ygo_meta/
  engine/               ← HTTP adapter to ygoinf API + subprocess runner
  llm_agent/            ← game state → text → Claude API → action index
  deck_builder/         ← YDK I/O, deck generation (engine + staples)
  simulation/           ← pairwise battles → payoff matrix → Nash eq → evolve
  cli/                  ← entry points: ygo-play, ygo-simulate
data/
  engines/<archetype>/engine.ydk   ← user-provided main engine cards
  staples/*.yaml                   ← curated staple card pools
scripts/
  build_card_db.py      ← one-time: code_list.txt → data/card_names.json
```

**Update flow for new cards**: `git submodule update` → `python scripts/build_card_db.py`. Zero other changes needed.

---

## Essential Commands

```bash
# First-time setup
git submodule update --init --recursive
pip install -e ".[dev]"
python scripts/build_card_db.py          # build data/card_names.json

# Start ygoinf inference server (required before playing)
python -m ygo_meta.engine.runner start   # listens on localhost:3000

# Play a game (LLM agent vs RL agent)
python -m ygo_meta.cli.play --deck data/engines/snake_eye/engine.ydk

# Run meta simulation
python -m ygo_meta.cli.simulate \
  --archetypes snake_eye blue_eyes hero \
  --staples-dir data/staples/ \
  --episodes 128 --generations 10

# Tests
pytest tests/ -v

# Lint
ruff check src/ tests/
```

---

## NEVER Hardcode Card Data

**Never hardcode ATK, DEF, level, attribute, or card type for any card code.**
`data/card_info.json` has the real stats for every card. Always use `card_from_db()` /
`load_card_info()` to build Card objects. This applies to demos, tests, scripts — everywhere.
Wrong hardcoded stats have already caused bugs (Rahu Dracotail shown as ATK 1600 monster,
Fallen of Albaz with DEF 2000, wrong attributes everywhere). Do not repeat this.

---

## Behavioral Guidelines (adapted from Karpathy / andrej-karpathy-skills)

### 1. Think Before Coding

The ygoinf API schema in `vendor/ygo-agent/ygoinf/ygoinf/models.py` is ground truth for all data shapes.
Check it before modifying `engine/types.py`, `engine/client.py`, or `llm_agent/formatter.py`.

State assumptions explicitly. If a card code is missing from `data/card_names.json`, surface it as a
warning — do not silently fall back to the raw integer.

For non-trivial tasks, write a brief plan and verification checklist before writing code.

### 2. Simplicity First

- LLM prompt = minimum text required to make a decision. No flavor text, no strategy coaching.
- Nash solver defaults to `nashpy`. Do not add a custom LP solver unless nashpy provably fails.
- No speculative features, no abstractions for code used only once, no error handling for
  impossible scenarios, no backwards-compatibility shims.

### 3. Surgical Changes

- `vendor/` is READ-ONLY. All adaptations live in `src/ygo_meta/engine/`.
- When editing any file, touch only what is required. Don't reformat adjacent code.
- Every changed line must trace directly to the requirement.

### 4. Goal-Driven Execution

Define success before starting. A simulation run succeeds when:
- Payoff matrix has no `NaN` entries
- Nash solver returns a valid mixed strategy (probabilities sum to 1.0 ± 1e-6)
- Top-3 decks from Nash written to `results/nash_solutions/gen_{k:03d}.json`

Transform tasks into verifiable goals with tests before implementation for non-trivial work.

---

## Key Invariants

| Rule | Source |
|---|---|
| Deck: 40–60 main, 0–15 extra, 0–15 side | `deck_builder/validator.py` |
| Card codes are ints from `vendor/ygo-agent/scripts/code_list.txt` | `scripts/build_card_db.py` |
| ygoinf server must be running before any LLM agent action | `engine/runner.py` |
| `vendor/` is READ-ONLY — never commit changes inside it | `.gitmodules` |
| Nash solution probabilities must sum to 1.0 ± 1e-6 | `simulation/nash.py` |

---

## File Ownership

| Module | What it does | Touches vendor? |
|---|---|---|
| `engine/` | HTTP + subprocess adapter | Calls only |
| `llm_agent/` | Prompt build + Claude API | No |
| `deck_builder/` | YDK I/O + deck generation | No |
| `simulation/` | Payoff matrix + Nash + evolution | Calls `battle.py` |
| `cli/` | Entry points | No |
| `vendor/` | ygo-agent submodule | READ-ONLY |

---

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude API key | required for play |
| `YGOAGENT_VENV` | Path to venv containing ygo-agent deps (JAX etc.) | uses current venv |
| `YGOAGENT_PORT` | Port for ygoinf server | `3000` |
| `YGOAGENT_CHECKPOINT` | Path to `.flax_model` or `.tflite` checkpoint | auto-detect in vendor/ |
