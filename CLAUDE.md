# CLAUDE.md

# General

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

---

## RULE: Always Test Before Saying Done

**Never tell the user a task is complete without first running tests (and the relevant script if applicable) and confirming zero errors.**

This means:
- Run `pytest tests/ -v` after every code change.
- If the task involves a runnable script, run it and verify it succeeds.
- If tests fail, fix them. Do not report done with failing tests.
- If the script errors, debug and fix before responding.

Violating this rule has already caused multiple wasted debugging sessions.

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

**Always run tests and the relevant script before reporting done. Never tell the user work is complete without first confirming it runs with zero errors.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


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

## NEVER Reimplement Vendor Game-Playing Logic

**All game-playing logic lives in `vendor/ygo-agent`. Do not reimplement it.**

- Use `ygoinf.features.get_legal_actions(action_msg)` to count/enumerate valid actions — never write your own message-type switch.
- Use `ygoinf.models` enum string values directly (`"hand"`, `"main1"`, `"faceup_attack"`, etc.) — never map raw YGOPro integer flags to strings in application code.
- The one place that converts raw `card_info.json` integer attributes to enum strings is `ygo_meta.llm_agent.card_db._ATTR_INT_TO_STR`. Do not duplicate this mapping elsewhere.
- `card_from_db()` lives in `ygo_meta.llm_agent.card_db`. Import it from there — never re-define it in scripts, tests, or any other module.

**Never construct synthetic game states with hand-coded move lists.** Any code that constructs `MsgSelectIdleCmd`, `MsgSelectBattleCmd`, etc. with hardcoded `idle_cmds`/`battle_cmds` is reimplementing the game engine's move-generation logic. Valid move lists come only from the running game engine via the ygoinf server.

Violations that have already caused rewrites: integer location/phase/position maps in `demo_agent.py`, duplicate `_ATTR_MAP` in tests, `_count_actions()` switch in `agent.py`, synthetic `idle_cmds` in demo scenarios offering illegal normal summons of level 5 monsters.

---

## NEVER Reimplement Vendor Engine Types

**Never redefine types that already exist in `vendor/ygo-agent/ygoinf/ygoinf/models.py`.**
`src/ygo_meta/engine/types.py` must only re-export from `ygoinf.models` — zero local
class definitions. Any divergence (different field names, int instead of enum, wrong
discriminator key) causes the inference server to receive malformed data and produce
invalid moves. If `ygoinf.models` changes, update the re-exports; do not duplicate.

Key vendor schema facts every developer must know:
- `ActionMsg.data` holds the actual message variant (discriminator key: `msg_type`, not `msg`)
- `Global` fields: `my_lp`, `op_lp`, `phase: Phase` (enum), `is_first: bool`, `is_my_turn: bool`
- `Card` fields: `attack`, `defense` (not `atk`/`def_`); `attribute/location/position/controller/race` are enums
- `MsgSelectIdleCmd.idle_cmds` (not `.actions`); `MsgSelectBattleCmd.battle_cmds`; `MsgSelectChain.chains`
- `MsgSelectCard/Tribute/Sum` use `CardLocation` for selection items — no card code, look up from `Input.cards`

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
