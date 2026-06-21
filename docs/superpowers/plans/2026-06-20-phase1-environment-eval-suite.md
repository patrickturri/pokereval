# Phase 1: Environment + Eval Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a poker eval suite that scores any LLM on Kuhn/Leduc (exact, verifiable grading) and curated Hold'em spots (reference-match + LLM-judge), and produces a reproducible leaderboard.

**Architecture:** One typed interface — `State → Decision → Score` — that every component shares. An engine wraps OpenSpiel (Kuhn/Leduc) to enumerate decision spots; a CFR solver produces the exact Nash policy used by the verifiable grader; an LLM-judge grades reasoning quality; a runner executes any `LLMClient` over a spot set and a leaderboard aggregates results (including whole-policy exploitability).

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, OpenSpiel (`pyspiel`), PokerKit (Hold'em, Phase 1 uses only curated data), Anthropic/OpenAI SDKs (real model clients).

## Global Constraints

- Python **3.11+** (uses `X | None` and `list[...]` builtin generics).
- Package uses **src layout**: importable root is `pokereval` under `src/`.
- Pydantic **v2** API (`model_validate`, `model_dump`, `Field(default_factory=...)`).
- All public functions are **pure and side-effect-free** except `models/`, `eval/runner.py`, and `eval/leaderboard.py` (I/O allowed there).
- **TDD throughout:** write the failing test first, watch it fail, implement, watch it pass, commit.
- Heavy/external deps are **optional extras** so the pure-Python core installs and tests without OpenSpiel or network access. Tests for `engine`/`solver` are marked `@pytest.mark.openspiel` and skipped when `pyspiel` is absent.
- **Before implementing Task 4's real `AnthropicClient`/`OpenAIClient`**, confirm current model IDs and SDK call shapes via the `claude-api` skill / live provider docs. Do not hardcode model IDs elsewhere — they are constructor args.
- **OpenSpiel API note:** exact `pyspiel` / `open_spiel.python` symbol names and per-game action strings can vary by version. Where a step is marked **⚠ verify**, run the snippet in a REPL first and pin the observed values. Tests for those tasks assert on *behavioral invariants* (e.g. "exploitability of the Nash policy ≈ 0"), which hold regardless of minor API differences.

## File Structure

```
src/pokereval/
  __init__.py
  interface/
    __init__.py
    types.py          # State, Decision, Action, GameVariant, Score (Pydantic)
    prompts.py        # render_state(), parse_decision(), ParseError
  models/
    __init__.py
    client.py         # LLMClient protocol, FakeClient, Anthropic/OpenAI clients
  engine/
    __init__.py
    kuhn_leduc.py     # OpenSpiel wrapper: load_game, iter_nodes, normalize_os_action
  solver/
    __init__.py
    nash.py           # solve_nash, nash_probs_by_key, exploitability helpers
  graders/
    __init__.py
    verifiable.py     # VerifiableGrader
    judge.py          # JudgeGrader (LLM-judge rubric)
  eval/
    __init__.py
    spots.py          # SpotSet, build_kuhn_leduc_spotset, load_holdem_spotset
    runner.py         # run_eval
    leaderboard.py    # aggregate, render_markdown
  cli.py              # `python -m pokereval.cli run ...`
data/
  holdem_spots.jsonl  # curated Hold'em spots with reference actions
tests/
  interface/ test_types.py, test_prompts.py
  models/    test_client.py
  engine/    test_kuhn_leduc.py
  solver/    test_nash.py
  graders/   test_verifiable.py, test_judge.py
  eval/      test_spots.py, test_runner.py, test_leaderboard.py
```

---

### Task 1: Project scaffolding & dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `src/pokereval/__init__.py`
- Create: `tests/test_smoke.py`
- Create: `pytest.ini` (or `[tool.pytest.ini_options]` in pyproject — use pyproject)

**Interfaces:**
- Consumes: nothing.
- Produces: importable `pokereval` package (`pokereval.__version__: str`); pytest configured with an `openspiel` marker.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
import pokereval

def test_package_imports_and_has_version():
    assert isinstance(pokereval.__version__, str)
    assert pokereval.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pokereval'`.

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pokereval"
version = "0.1.0"
description = "A poker testbed for frontier-model research engineering"
requires-python = ">=3.11"
dependencies = ["pydantic>=2.6"]

[project.optional-dependencies]
openspiel = ["open_spiel"]
llm = ["anthropic", "openai"]
holdem = ["pokerkit"]
dev = ["pytest>=8.0"]

[tool.hatch.build.targets.wheel]
packages = ["src/pokereval"]

[tool.pytest.ini_options]
pythonpath = ["src"]
markers = ["openspiel: requires the open_spiel package"]
```

```python
# src/pokereval/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/pokereval/__init__.py tests/test_smoke.py
git commit -m "feat: project scaffolding and pytest setup"
```

---

### Task 2: Core interface types

**Files:**
- Create: `src/pokereval/interface/__init__.py` (empty)
- Create: `src/pokereval/interface/types.py`
- Test: `tests/interface/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Action(str, Enum)`: `FOLD, CHECK, CALL, BET, RAISE` (values are lowercase names).
  - `GameVariant(str, Enum)`: `KUHN="kuhn", LEDUC="leduc", HOLDEM="holdem"`.
  - `State(BaseModel)`: `variant: GameVariant`, `hero_cards: list[str]`, `board: list[str]=[]`, `pot: float`, `to_call: float=0.0`, `position: str=""`, `history: str=""`, `legal_actions: list[Action]`, `info_state_key: str|None=None`, `meta: dict=...`.
  - `Decision(BaseModel)`: `action: Action`, `size: float|None=None`, `reasoning: str=""`.
  - `Score(BaseModel)`: `value: float`, `detail: dict=...`.

- [ ] **Step 1: Write the failing test**

```python
# tests/interface/test_types.py
from pokereval.interface.types import Action, GameVariant, State, Decision, Score

def test_state_roundtrips_through_json():
    s = State(
        variant=GameVariant.LEDUC,
        hero_cards=["Js"],
        pot=4.0,
        to_call=2.0,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        info_state_key="J:cr",
        meta={"os_legal": {"fold": 0, "call": 1, "raise": 2}},
    )
    again = State.model_validate(s.model_dump())
    assert again == s
    assert again.legal_actions[0] is Action.FOLD

def test_decision_defaults():
    d = Decision(action=Action.BET, size=2.0)
    assert d.reasoning == ""
    assert d.action is Action.BET

def test_score_holds_detail():
    sc = Score(value=0.75, detail={"nash_prob": 0.75})
    assert sc.value == 0.75
    assert sc.detail["nash_prob"] == 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/interface/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pokereval.interface'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pokereval/interface/types.py
from enum import Enum
from pydantic import BaseModel, Field


class Action(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


class GameVariant(str, Enum):
    KUHN = "kuhn"
    LEDUC = "leduc"
    HOLDEM = "holdem"


class State(BaseModel):
    variant: GameVariant
    hero_cards: list[str]
    board: list[str] = Field(default_factory=list)
    pot: float
    to_call: float = 0.0
    position: str = ""
    history: str = ""
    legal_actions: list[Action]
    info_state_key: str | None = None
    meta: dict = Field(default_factory=dict)


class Decision(BaseModel):
    action: Action
    size: float | None = None
    reasoning: str = ""


class Score(BaseModel):
    value: float
    detail: dict = Field(default_factory=dict)
```

Also create the empty package marker:

```python
# src/pokereval/interface/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/interface/test_types.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pokereval/interface/__init__.py src/pokereval/interface/types.py tests/interface/test_types.py
git commit -m "feat: core interface types (State/Decision/Action/Score)"
```

---

### Task 3: Prompt rendering & decision parsing

**Files:**
- Create: `src/pokereval/interface/prompts.py`
- Test: `tests/interface/test_prompts.py`

**Interfaces:**
- Consumes: `State`, `Decision`, `Action` from `interface.types`.
- Produces:
  - `class ParseError(ValueError)`.
  - `render_state(state: State) -> str`.
  - `parse_decision(text: str, legal: list[Action]) -> Decision` — parses the **last** `ACTION: <action> [size]` line; raises `ParseError` if missing, unknown, or illegal.

- [ ] **Step 1: Write the failing test**

```python
# tests/interface/test_prompts.py
import pytest
from pokereval.interface.types import State, GameVariant, Action
from pokereval.interface.prompts import render_state, parse_decision, ParseError

def _state():
    return State(
        variant=GameVariant.LEDUC, hero_cards=["Js"], pot=4.0, to_call=2.0,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
    )

def test_render_mentions_cards_pot_and_legal_actions():
    text = render_state(_state())
    assert "Js" in text
    assert "4.0" in text or "4" in text
    assert "fold" in text and "call" in text and "raise" in text
    assert "ACTION:" in text

def test_parse_picks_last_action_line_and_reasoning():
    text = "I think about it.\nACTION: raise 4"
    d = parse_decision(text, [Action.FOLD, Action.CALL, Action.RAISE])
    assert d.action is Action.RAISE
    assert d.size == 4.0
    assert d.reasoning == "I think about it."

def test_parse_uses_final_action_when_multiple():
    text = "ACTION: call\n...changed my mind...\nACTION: fold"
    d = parse_decision(text, [Action.FOLD, Action.CALL])
    assert d.action is Action.FOLD

def test_parse_rejects_illegal_action():
    with pytest.raises(ParseError):
        parse_decision("ACTION: raise", [Action.FOLD, Action.CALL])

def test_parse_rejects_missing_action():
    with pytest.raises(ParseError):
        parse_decision("no action here", [Action.FOLD])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/interface/test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: ... prompts`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pokereval/interface/prompts.py
import re
from .types import State, Decision, Action


class ParseError(ValueError):
    pass


def render_state(state: State) -> str:
    lines = [
        f"You are playing {state.variant.value} poker.",
        f"Your cards: {' '.join(state.hero_cards)}",
    ]
    if state.board:
        lines.append(f"Board: {' '.join(state.board)}")
    lines.append(f"Pot: {state.pot}")
    if state.to_call:
        lines.append(f"To call: {state.to_call}")
    if state.position:
        lines.append(f"Position: {state.position}")
    if state.history:
        lines.append(f"Action history: {state.history}")
    lines.append("Legal actions: " + ", ".join(a.value for a in state.legal_actions))
    lines.append(
        "Think step by step, then end with a final line exactly:\n"
        "ACTION: <action> [size]\n"
        "where <action> is one of the legal actions and [size] is a number "
        "(only for bet/raise)."
    )
    return "\n".join(lines)


_ACTION_RE = re.compile(r"ACTION:\s*([a-zA-Z]+)\s*([0-9]*\.?[0-9]+)?", re.IGNORECASE)


def parse_decision(text: str, legal: list[Action]) -> Decision:
    matches = list(_ACTION_RE.finditer(text))
    if not matches:
        raise ParseError(f"No ACTION line found in: {text!r}")
    m = matches[-1]
    raw = m.group(1).lower()
    try:
        action = Action(raw)
    except ValueError as exc:
        raise ParseError(f"Unknown action {raw!r}") from exc
    if action not in legal:
        raise ParseError(f"Illegal action {action.value} (legal: {[a.value for a in legal]})")
    size = float(m.group(2)) if m.group(2) else None
    reasoning = text[: m.start()].strip()
    return Decision(action=action, size=size, reasoning=reasoning)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/interface/test_prompts.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pokereval/interface/prompts.py tests/interface/test_prompts.py
git commit -m "feat: prompt rendering and robust decision parsing"
```

---

### Task 4: LLM client abstraction

**Files:**
- Create: `src/pokereval/models/__init__.py` (empty)
- Create: `src/pokereval/models/client.py`
- Test: `tests/models/test_client.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces:
  - `class LLMClient(Protocol)`: attribute `name: str`; method `complete(self, prompt: str, system: str | None = None) -> str`.
  - `class FakeClient`: `__init__(self, name: str, response: str | Callable[[str], str] = "ACTION: fold")`; deterministic, used by all downstream tests.
  - `class AnthropicClient` and `class OpenAIClient`: thin wrappers; `__init__(self, name: str, model: str, **kwargs)`. (Network code — not unit-tested; see Global Constraints re: model IDs.)

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_client.py
from pokereval.models.client import LLMClient, FakeClient

def test_fakeclient_returns_static_string():
    c = FakeClient("fixed", "ACTION: call")
    assert c.complete("anything") == "ACTION: call"
    assert c.name == "fixed"

def test_fakeclient_supports_callable():
    c = FakeClient("dyn", lambda prompt: "ACTION: raise" if "Js" in prompt else "ACTION: fold")
    assert c.complete("Your cards: Js") == "ACTION: raise"
    assert c.complete("Your cards: 2c") == "ACTION: fold"

def test_fakeclient_satisfies_protocol():
    assert isinstance(FakeClient("x"), LLMClient)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_client.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pokereval/models/client.py
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str, system: str | None = None) -> str: ...


class FakeClient:
    """Deterministic client for tests and dry runs."""

    def __init__(self, name: str, response: str | Callable[[str], str] = "ACTION: fold"):
        self.name = name
        self._response = response

    def complete(self, prompt: str, system: str | None = None) -> str:
        if callable(self._response):
            return self._response(prompt)
        return self._response


class AnthropicClient:
    """⚠ Confirm model IDs and SDK call via the claude-api skill before use."""

    def __init__(self, name: str, model: str, max_tokens: int = 1024):
        from anthropic import Anthropic  # lazy import; optional dep

        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self._client = Anthropic()

    def complete(self, prompt: str, system: str | None = None) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAIClient:
    """⚠ Confirm model IDs and SDK call against live docs before use.
    Also serves OpenAI-compatible endpoints (vLLM) via base_url."""

    def __init__(self, name: str, model: str, base_url: str | None = None):
        from openai import OpenAI  # lazy import; optional dep

        self.name = name
        self.model = model
        self._client = OpenAI(base_url=base_url) if base_url else OpenAI()

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content or ""
```

```python
# src/pokereval/models/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_client.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pokereval/models/__init__.py src/pokereval/models/client.py tests/models/test_client.py
git commit -m "feat: LLM client protocol with FakeClient and provider wrappers"
```

---

### Task 5: OpenSpiel engine wrapper (Kuhn/Leduc)

**Files:**
- Create: `src/pokereval/engine/__init__.py` (empty)
- Create: `src/pokereval/engine/kuhn_leduc.py`
- Test: `tests/engine/test_kuhn_leduc.py`

**Interfaces:**
- Consumes: `Action`, `GameVariant` from `interface.types`.
- Produces:
  - `load_game(variant: GameVariant)` — returns a `pyspiel` game for `KUHN`/`LEDUC`; raises `ValueError` for `HOLDEM`.
  - `normalize_os_action(action_string: str, facing_bet: bool) -> Action` — maps OpenSpiel action strings to `Action` (handles Kuhn "Pass"→`CHECK`/`FOLD`, "Bet"→`BET`; Leduc "Fold/Call/Raise/Check").
  - `@dataclass NodeInfo`: `info_key: str`, `player: int`, `os_legal: dict[str, int]` (our action value → OpenSpiel action id), `facing_bet: bool`, `os_state` (the live `pyspiel.State`).
  - `iter_nodes(variant: GameVariant)` — generator over **every player decision node** of the full game tree, yielding `NodeInfo`. (Chance nodes are expanded; terminal nodes skipped.)

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_kuhn_leduc.py
import pytest
pyspiel = pytest.importorskip("pyspiel")
pytestmark = pytest.mark.openspiel

from pokereval.interface.types import GameVariant, Action
from pokereval.engine.kuhn_leduc import load_game, iter_nodes, normalize_os_action

def test_load_holdem_raises():
    with pytest.raises(ValueError):
        load_game(GameVariant.HOLDEM)

def test_kuhn_nodes_are_nonempty_with_unique_keys():
    nodes = list(iter_nodes(GameVariant.KUHN))
    keys = [n.info_key for n in nodes]
    assert len(keys) > 0
    assert len(set(keys)) == len(keys)              # info keys are unique
    for n in nodes:
        assert n.os_legal                            # every node has >=1 legal action
        assert all(v in {a.value for a in Action} for v in n.os_legal)

def test_leduc_has_more_decision_nodes_than_kuhn():
    assert len(list(iter_nodes(GameVariant.LEDUC))) > len(list(iter_nodes(GameVariant.KUHN)))

def test_normalize_pass_depends_on_facing_bet():
    assert normalize_os_action("Pass", facing_bet=False) is Action.CHECK
    assert normalize_os_action("Pass", facing_bet=True) is Action.FOLD
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_kuhn_leduc.py -v`
Expected: FAIL — module missing (or skipped if `pyspiel` not installed; install with `pip install -e ".[openspiel]"`, **⚠ verify wheel availability for your platform**).

- [ ] **Step 3: Write minimal implementation**

```python
# src/pokereval/engine/kuhn_leduc.py
from dataclasses import dataclass
import pyspiel
from ..interface.types import Action, GameVariant

_GAME_NAME = {GameVariant.KUHN: "kuhn_poker", GameVariant.LEDUC: "leduc_poker"}


def load_game(variant: GameVariant):
    if variant not in _GAME_NAME:
        raise ValueError(f"No OpenSpiel game for {variant}")
    return pyspiel.load_game(_GAME_NAME[variant])


def normalize_os_action(action_string: str, facing_bet: bool) -> Action:
    s = action_string.strip().lower()
    # ⚠ verify exact strings via state.action_to_string(...) in your OpenSpiel build.
    if s in ("fold",):
        return Action.FOLD
    if s in ("call",):
        return Action.CALL
    if s in ("raise",):
        return Action.RAISE
    if s in ("check",):
        return Action.CHECK
    if s in ("bet",):
        return Action.BET
    if s in ("pass",):  # Kuhn: pass = fold if facing a bet, else check
        return Action.FOLD if facing_bet else Action.CHECK
    raise ValueError(f"Unrecognized OpenSpiel action string: {action_string!r}")


@dataclass
class NodeInfo:
    info_key: str
    player: int
    os_legal: dict[str, int]
    facing_bet: bool
    os_state: "pyspiel.State"


def iter_nodes(variant: GameVariant):
    game = load_game(variant)

    def _walk(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _prob in state.chance_outcomes():
                _walk(state.child(action))
            return
        player = state.current_player()
        info_key = state.information_state_string(player)
        # facing_bet heuristic: a "Call"/"Fold" option implies a bet is pending.
        # ⚠ verify against action_to_string outputs for your build.
        labels = {a: state.action_to_string(player, a) for a in state.legal_actions()}
        facing_bet = any(s.strip().lower() in ("call", "fold") for s in labels.values())
        os_legal: dict[str, int] = {}
        for a, label in labels.items():
            act = normalize_os_action(label, facing_bet)
            os_legal[act.value] = a
        yield NodeInfo(info_key, player, os_legal, facing_bet, state)
        for a in state.legal_actions():
            yield from _walk(state.child(a))

    seen: set[str] = set()
    for node in _walk(game.new_initial_state()):
        if node.info_key in seen:
            continue
        seen.add(node.info_key)
        yield node
```

```python
# src/pokereval/engine/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/engine/test_kuhn_leduc.py -v`
Expected: PASS (4 tests). If `normalize_os_action` raises on an unexpected string, **⚠** print the labels you observe and extend the mapping, then re-run.

- [ ] **Step 5: Commit**

```bash
git add src/pokereval/engine/__init__.py src/pokereval/engine/kuhn_leduc.py tests/engine/test_kuhn_leduc.py
git commit -m "feat: OpenSpiel engine wrapper enumerating Kuhn/Leduc decision nodes"
```

---

### Task 6: CFR solver, Nash policy & exploitability

**Files:**
- Create: `src/pokereval/solver/__init__.py` (empty)
- Create: `src/pokereval/solver/nash.py`
- Test: `tests/solver/test_nash.py`

**Interfaces:**
- Consumes: `GameVariant`; `load_game`, `iter_nodes` from `engine.kuhn_leduc`.
- Produces:
  - `solve_nash(game, iterations: int = 2000)` — returns an OpenSpiel average policy object.
  - `nash_probs_by_key(variant: GameVariant, iterations: int = 2000) -> dict[str, dict[str, float]]` — `{info_key: {os_action_id_as_str: prob}}` over all decision nodes.
  - `nash_exploitability(game, avg_policy) -> float`.
  - `exploitability_of_choices(variant: GameVariant, choices: dict[str, int]) -> float` — builds a deterministic `TabularPolicy` from `{info_key: os_action_id}` and returns its exploitability (missing keys default to uniform).

- [ ] **Step 1: Write the failing test**

```python
# tests/solver/test_nash.py
import pytest
pyspiel = pytest.importorskip("pyspiel")
pytestmark = pytest.mark.openspiel

from pokereval.interface.types import GameVariant
from pokereval.solver.nash import (
    solve_nash, nash_probs_by_key, nash_exploitability, exploitability_of_choices,
)
from pokereval.engine.kuhn_leduc import load_game

def test_nash_policy_is_near_optimal():
    game = load_game(GameVariant.KUHN)
    avg = solve_nash(game, iterations=2000)
    assert nash_exploitability(game, avg) < 0.05   # converges toward 0

def test_nash_probs_cover_all_decision_keys_and_sum_to_one():
    probs = nash_probs_by_key(GameVariant.KUHN, iterations=500)
    assert probs
    for key, dist in probs.items():
        assert abs(sum(dist.values()) - 1.0) < 1e-6

def test_choices_policy_returns_nonnegative_float():
    probs = nash_probs_by_key(GameVariant.KUHN, iterations=500)
    choices = {k: int(max(dist, key=dist.get)) for k, dist in probs.items()}
    expl = exploitability_of_choices(GameVariant.KUHN, choices)
    assert isinstance(expl, float) and expl >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/solver/test_nash.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pokereval/solver/nash.py
from open_spiel.python.algorithms import cfr, exploitability
from open_spiel.python import policy as os_policy
from ..interface.types import GameVariant
from ..engine.kuhn_leduc import load_game, iter_nodes


def solve_nash(game, iterations: int = 2000):
    solver = cfr.CFRSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    return solver.average_policy()


def nash_exploitability(game, avg_policy) -> float:
    return float(exploitability.exploitability(game, avg_policy))


def nash_probs_by_key(variant: GameVariant, iterations: int = 2000) -> dict[str, dict[str, float]]:
    game = load_game(variant)
    avg = solve_nash(game, iterations)
    out: dict[str, dict[str, float]] = {}
    for node in iter_nodes(variant):
        # action_probabilities returns {os_action_id: prob}; ⚠ verify method name.
        dist = avg.action_probabilities(node.os_state)
        out[node.info_key] = {str(int(a)): float(p) for a, p in dist.items()}
    return out


def exploitability_of_choices(variant: GameVariant, choices: dict[str, int]) -> float:
    game = load_game(variant)
    tab = os_policy.TabularPolicy(game)
    for node in iter_nodes(variant):
        if node.info_key not in choices:
            continue
        chosen = choices[node.info_key]
        legal = list(node.os_state.legal_actions())
        row = tab.policy_for_key(node.info_key)  # mutable list over legal-action order
        for i, a in enumerate(legal):
            row[i] = 1.0 if a == chosen else 0.0
    return float(exploitability.exploitability(game, tab))
```

```python
# src/pokereval/solver/__init__.py
```

**⚠ verify:** `avg.action_probabilities(state)` and `TabularPolicy.policy_for_key` exist in your OpenSpiel version. If `policy_for_key` orders rows differently, align indices to `state.legal_actions()` order. The `test_nash_policy_is_near_optimal` test is your ground-truth check that the path is wired correctly.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/solver/test_nash.py -v`
Expected: PASS (3 tests). Exploitability of the Nash policy should be small and shrink with more iterations.

- [ ] **Step 5: Commit**

```bash
git add src/pokereval/solver/__init__.py src/pokereval/solver/nash.py tests/solver/test_nash.py
git commit -m "feat: CFR solver, Nash action probabilities, exploitability helpers"
```

---

### Task 7: Verifiable grader

**Files:**
- Create: `src/pokereval/graders/__init__.py` (empty)
- Create: `src/pokereval/graders/verifiable.py`
- Test: `tests/graders/test_verifiable.py`

**Interfaces:**
- Consumes: `State`, `Decision`, `Score`, `Action`.
- Produces:
  - `class VerifiableGrader`: `__init__(self, nash_probs: dict[str, dict[str, float]])`.
  - `grade(self, state: State, decision: Decision) -> Score`:
    - Kuhn/Leduc (state has `info_state_key` in `nash_probs`): map `decision.action` → os id via `state.meta["os_legal"]`; `value = nash_probs[key][str(os_id)]` (probability mass Nash assigns the chosen action); `detail = {"nash_prob": value, "exact_match": value == max(dist.values()), "kind": "nash"}`.
    - Hold'em (no nash entry): `reference = state.meta["reference_action"]`; `value = 1.0 if decision.action.value == reference else 0.0`; `detail = {"kind": "reference", "reference": reference}`.
    - Unknown action mapping or missing data → `Score(value=0.0, detail={"kind": "ungradeable"})`.

- [ ] **Step 1: Write the failing test**

```python
# tests/graders/test_verifiable.py
from pokereval.interface.types import State, Decision, Action, GameVariant
from pokereval.graders.verifiable import VerifiableGrader

def _leduc_state():
    return State(
        variant=GameVariant.LEDUC, hero_cards=["Js"], pot=4.0, to_call=2.0,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        info_state_key="K1",
        meta={"os_legal": {"fold": 0, "call": 1, "raise": 2}},
    )

def test_nash_grade_returns_probability_mass_and_exact_match():
    nash = {"K1": {"0": 0.1, "1": 0.2, "2": 0.7}}
    g = VerifiableGrader(nash)
    sc = g.grade(_leduc_state(), Decision(action=Action.RAISE))
    assert sc.value == 0.7
    assert sc.detail["exact_match"] is True
    sc2 = g.grade(_leduc_state(), Decision(action=Action.FOLD))
    assert sc2.value == 0.1
    assert sc2.detail["exact_match"] is False

def test_reference_grade_for_holdem():
    g = VerifiableGrader({})
    st = State(
        variant=GameVariant.HOLDEM, hero_cards=["As", "Ks"], board=["Ah", "7d", "2c"],
        pot=10.0, legal_actions=[Action.CHECK, Action.BET],
        meta={"reference_action": "bet"},
    )
    assert g.grade(st, Decision(action=Action.BET)).value == 1.0
    assert g.grade(st, Decision(action=Action.CHECK)).value == 0.0

def test_ungradeable_when_no_data():
    g = VerifiableGrader({})
    st = State(variant=GameVariant.HOLDEM, hero_cards=["As"], pot=1.0,
               legal_actions=[Action.CHECK], meta={})
    assert g.grade(st, Decision(action=Action.CHECK)).detail["kind"] == "ungradeable"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/graders/test_verifiable.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pokereval/graders/verifiable.py
from ..interface.types import State, Decision, Score


class VerifiableGrader:
    def __init__(self, nash_probs: dict[str, dict[str, float]]):
        self._nash = nash_probs

    def grade(self, state: State, decision: Decision) -> Score:
        key = state.info_state_key
        if key is not None and key in self._nash:
            os_legal = state.meta.get("os_legal", {})
            os_id = os_legal.get(decision.action.value)
            dist = self._nash[key]
            if os_id is None or str(os_id) not in dist:
                return Score(value=0.0, detail={"kind": "ungradeable"})
            value = dist[str(os_id)]
            return Score(
                value=value,
                detail={"kind": "nash", "nash_prob": value,
                        "exact_match": value == max(dist.values())},
            )
        ref = state.meta.get("reference_action")
        if ref is not None:
            value = 1.0 if decision.action.value == ref else 0.0
            return Score(value=value, detail={"kind": "reference", "reference": ref})
        return Score(value=0.0, detail={"kind": "ungradeable"})
```

```python
# src/pokereval/graders/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/graders/test_verifiable.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pokereval/graders/__init__.py src/pokereval/graders/verifiable.py tests/graders/test_verifiable.py
git commit -m "feat: verifiable grader (Nash probability + reference match)"
```

---

### Task 8: LLM-judge grader (unverifiable)

**Files:**
- Create: `src/pokereval/graders/judge.py`
- Test: `tests/graders/test_judge.py`

**Interfaces:**
- Consumes: `State`, `Decision`, `Score`; `LLMClient` from `models.client`; `render_state`.
- Produces:
  - `RUBRIC: list[str]` — the reasoning-quality criteria.
  - `class JudgeGrader`: `__init__(self, client: LLMClient, rubric: list[str] = RUBRIC)`.
  - `grade(self, state: State, decision: Decision) -> Score`: builds a judging prompt, calls `client.complete`, parses a JSON object `{criterion: 0|1, ...}`; `value = mean(scores)`; `detail = {"kind": "judge", "scores": {...}, "raw": text}`. On parse failure → `Score(value=0.0, detail={"kind": "judge_error", "raw": text})`.

- [ ] **Step 1: Write the failing test**

```python
# tests/graders/test_judge.py
from pokereval.interface.types import State, Decision, Action, GameVariant
from pokereval.models.client import FakeClient
from pokereval.graders.judge import JudgeGrader, RUBRIC

def _state():
    return State(variant=GameVariant.HOLDEM, hero_cards=["As", "Ks"],
                 board=["Ah", "7d", "2c"], pot=10.0,
                 legal_actions=[Action.CHECK, Action.BET])

def test_judge_averages_rubric_scores():
    payload = "{" + ", ".join(f'"{c}": 1' for c in RUBRIC) + "}"
    g = JudgeGrader(FakeClient("judge", payload))
    sc = g.grade(_state(), Decision(action=Action.BET, reasoning="Top pair, bet for value."))
    assert sc.value == 1.0
    assert sc.detail["kind"] == "judge"

def test_judge_partial_credit():
    scores = {c: (1 if i == 0 else 0) for i, c in enumerate(RUBRIC)}
    import json
    g = JudgeGrader(FakeClient("judge", json.dumps(scores)))
    sc = g.grade(_state(), Decision(action=Action.BET))
    assert 0.0 < sc.value < 1.0

def test_judge_handles_unparseable_output():
    g = JudgeGrader(FakeClient("judge", "I cannot produce JSON"))
    sc = g.grade(_state(), Decision(action=Action.BET))
    assert sc.detail["kind"] == "judge_error"
    assert sc.value == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/graders/test_judge.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pokereval/graders/judge.py
import json
import re
from ..interface.types import State, Decision, Score
from ..interface.prompts import render_state
from ..models.client import LLMClient

RUBRIC = [
    "identifies_pot_odds_or_equity",
    "reasons_about_opponent_range",
    "sound_bet_sizing_logic",
    "decision_consistent_with_reasoning",
]

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class JudgeGrader:
    def __init__(self, client: LLMClient, rubric: list[str] = RUBRIC):
        self.client = client
        self.rubric = rubric

    def _prompt(self, state: State, decision: Decision) -> str:
        criteria = "\n".join(f"- {c}" for c in self.rubric)
        return (
            "You are grading a poker player's reasoning. The situation:\n"
            f"{render_state(state)}\n\n"
            f"The player chose: {decision.action.value}"
            f"{' ' + str(decision.size) if decision.size else ''}\n"
            f"Their reasoning:\n{decision.reasoning or '(none given)'}\n\n"
            "Score each criterion 1 (met) or 0 (not met). "
            "Respond with ONLY a JSON object mapping each criterion to 0 or 1:\n"
            f"{criteria}"
        )

    def grade(self, state: State, decision: Decision) -> Score:
        text = self.client.complete(self._prompt(state, decision))
        match = _JSON_RE.search(text)
        if not match:
            return Score(value=0.0, detail={"kind": "judge_error", "raw": text})
        try:
            scores = json.loads(match.group(0))
            values = [float(scores[c]) for c in self.rubric]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return Score(value=0.0, detail={"kind": "judge_error", "raw": text})
        value = sum(values) / len(values)
        return Score(value=value, detail={"kind": "judge", "scores": scores, "raw": text})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/graders/test_judge.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pokereval/graders/judge.py tests/graders/test_judge.py
git commit -m "feat: LLM-judge grader for reasoning quality (rubric-based)"
```

---

### Task 9: Spot sets (Kuhn/Leduc enumeration + curated Hold'em)

**Files:**
- Create: `src/pokereval/eval/__init__.py` (empty)
- Create: `src/pokereval/eval/spots.py`
- Create: `data/holdem_spots.jsonl`
- Test: `tests/eval/test_spots.py`

**Interfaces:**
- Consumes: `State`, `Action`, `GameVariant`; `iter_nodes` (engine); `nash_probs_by_key` (solver).
- Produces:
  - `class SpotSet(BaseModel)`: `spots: list[State]`, `nash_probs: dict[str, dict[str, float]] = {}`.
  - `build_kuhn_leduc_spotset(variant: GameVariant, iterations: int = 2000) -> SpotSet` — one `State` per decision node, with `info_state_key`, `legal_actions`, and `meta["os_legal"]`; plus the Nash probs.
  - `load_holdem_spotset(path: str | Path) -> SpotSet` — reads JSONL of `State`-shaped dicts (each carrying `meta.reference_action`).

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_spots.py
import json
from pathlib import Path
import pytest
from pokereval.interface.types import GameVariant, Action
from pokereval.eval.spots import SpotSet, load_holdem_spotset

def test_load_holdem_spotset(tmp_path):
    rows = [{
        "variant": "holdem", "hero_cards": ["As", "Ks"], "board": ["Ah", "7d", "2c"],
        "pot": 10.0, "to_call": 0.0,
        "legal_actions": ["check", "bet"], "meta": {"reference_action": "bet"},
    }]
    p = tmp_path / "h.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    ss = load_holdem_spotset(p)
    assert len(ss.spots) == 1
    assert ss.spots[0].legal_actions == [Action.CHECK, Action.BET]
    assert ss.spots[0].meta["reference_action"] == "bet"

def test_shipped_holdem_file_is_valid():
    ss = load_holdem_spotset(Path("data/holdem_spots.jsonl"))
    assert len(ss.spots) >= 5
    for s in ss.spots:
        assert s.variant is GameVariant.HOLDEM
        assert "reference_action" in s.meta

@pytest.mark.openspiel
def test_build_kuhn_spotset_has_keys_and_nash():
    pytest.importorskip("pyspiel")
    ss = build = __import__(
        "pokereval.eval.spots", fromlist=["build_kuhn_leduc_spotset"]
    ).build_kuhn_leduc_spotset(GameVariant.KUHN, iterations=300)
    assert len(ss.spots) > 0
    for s in ss.spots:
        assert s.info_state_key in ss.nash_probs
        assert "os_legal" in s.meta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_spots.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pokereval/eval/spots.py
import json
from pathlib import Path
from pydantic import BaseModel, Field
from ..interface.types import State, Action, GameVariant
from ..engine.kuhn_leduc import iter_nodes
from ..solver.nash import nash_probs_by_key


class SpotSet(BaseModel):
    spots: list[State] = Field(default_factory=list)
    nash_probs: dict[str, dict[str, float]] = Field(default_factory=dict)


def build_kuhn_leduc_spotset(variant: GameVariant, iterations: int = 2000) -> SpotSet:
    nash = nash_probs_by_key(variant, iterations)
    spots: list[State] = []
    for node in iter_nodes(variant):
        legal = [Action(v) for v in node.os_legal]
        spots.append(State(
            variant=variant,
            hero_cards=[node.info_key],          # info string encodes the private card+history
            pot=0.0,
            history=node.info_key,
            legal_actions=legal,
            info_state_key=node.info_key,
            meta={"os_legal": node.os_legal},
        ))
    return SpotSet(spots=spots, nash_probs=nash)


def load_holdem_spotset(path: str | Path) -> SpotSet:
    spots: list[State] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            spots.append(State.model_validate(json.loads(line)))
    return SpotSet(spots=spots, nash_probs={})
```

```python
# src/pokereval/eval/__init__.py
```

Create the curated data file (hand-authored "obvious" spots; reference actions are the clearly-correct play — keep them uncontroversial so the verifiable signal is sound):

```jsonl
{"variant": "holdem", "hero_cards": ["As", "Ah"], "board": [], "pot": 3.0, "to_call": 2.0, "position": "BTN", "history": "raise to 3", "legal_actions": ["fold", "call", "raise"], "meta": {"reference_action": "raise"}}
{"variant": "holdem", "hero_cards": ["7c", "2d"], "board": [], "pot": 3.0, "to_call": 2.0, "position": "UTG", "history": "raise to 3", "legal_actions": ["fold", "call", "raise"], "meta": {"reference_action": "fold"}}
{"variant": "holdem", "hero_cards": ["As", "Ks"], "board": ["Ah", "7d", "2c"], "pot": 6.0, "to_call": 0.0, "position": "BB", "history": "checked to hero", "legal_actions": ["check", "bet"], "meta": {"reference_action": "bet"}}
{"variant": "holdem", "hero_cards": ["Qh", "Qd"], "board": ["2s", "5c", "9d"], "pot": 8.0, "to_call": 0.0, "position": "BB", "history": "checked to hero", "legal_actions": ["check", "bet"], "meta": {"reference_action": "bet"}}
{"variant": "holdem", "hero_cards": ["6h", "5h"], "board": ["As", "Kd", "Qc"], "pot": 8.0, "to_call": 6.0, "position": "BB", "history": "villain bets pot", "legal_actions": ["fold", "call", "raise"], "meta": {"reference_action": "fold"}}
{"variant": "holdem", "hero_cards": ["Ks", "Kd"], "board": ["Kh", "7d", "2c"], "pot": 8.0, "to_call": 0.0, "position": "BB", "history": "checked to hero", "legal_actions": ["check", "bet"], "meta": {"reference_action": "bet"}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_spots.py -v`
Expected: PASS (3 tests; the OpenSpiel one skips without `pyspiel`).

- [ ] **Step 5: Commit**

```bash
git add src/pokereval/eval/__init__.py src/pokereval/eval/spots.py data/holdem_spots.jsonl tests/eval/test_spots.py
git commit -m "feat: spot sets — Kuhn/Leduc enumeration and curated Hold'em data"
```

---

### Task 10: Eval runner

**Files:**
- Create: `src/pokereval/eval/runner.py`
- Test: `tests/eval/test_runner.py`

**Interfaces:**
- Consumes: `State`, `Decision`, `Score`; `LLMClient`; `render_state`, `parse_decision`, `ParseError`; `VerifiableGrader`, `JudgeGrader`; `SpotSet`.
- Produces:
  - `class SpotResult(BaseModel)`: `info_state_key: str | None`, `variant: str`, `action: str | None`, `verifiable: Score | None`, `judge: Score | None`, `error: str | None`.
  - `run_eval(client, spotset, verifiable, judge=None) -> list[SpotResult]` — for each spot: render → complete → parse; on `ParseError` record `error` and skip grading; else grade with `verifiable` and (if provided) `judge`.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_runner.py
from pokereval.interface.types import State, Action, GameVariant
from pokereval.models.client import FakeClient
from pokereval.graders.verifiable import VerifiableGrader
from pokereval.eval.spots import SpotSet
from pokereval.eval.runner import run_eval, SpotResult

def _spotset():
    s = State(variant=GameVariant.LEDUC, hero_cards=["K"], pot=4.0, to_call=2.0,
              legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
              info_state_key="K1", meta={"os_legal": {"fold": 0, "call": 1, "raise": 2}})
    return SpotSet(spots=[s], nash_probs={"K1": {"0": 0.1, "1": 0.2, "2": 0.7}})

def test_run_eval_grades_a_valid_decision():
    ss = _spotset()
    results = run_eval(FakeClient("m", "ACTION: raise"), ss, VerifiableGrader(ss.nash_probs))
    assert len(results) == 1
    assert results[0].action == "raise"
    assert results[0].verifiable.value == 0.7
    assert results[0].error is None

def test_run_eval_records_parse_errors():
    ss = _spotset()
    results = run_eval(FakeClient("m", "no action"), ss, VerifiableGrader(ss.nash_probs))
    assert results[0].error is not None
    assert results[0].verifiable is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_runner.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pokereval/eval/runner.py
from pydantic import BaseModel
from ..interface.types import Score
from ..interface.prompts import render_state, parse_decision, ParseError
from ..models.client import LLMClient
from ..graders.verifiable import VerifiableGrader
from ..graders.judge import JudgeGrader
from .spots import SpotSet


class SpotResult(BaseModel):
    info_state_key: str | None = None
    variant: str
    action: str | None = None
    verifiable: Score | None = None
    judge: Score | None = None
    error: str | None = None


def run_eval(
    client: LLMClient,
    spotset: SpotSet,
    verifiable: VerifiableGrader,
    judge: JudgeGrader | None = None,
) -> list[SpotResult]:
    results: list[SpotResult] = []
    for state in spotset.spots:
        prompt = render_state(state)
        text = client.complete(prompt)
        try:
            decision = parse_decision(text, state.legal_actions)
        except ParseError as exc:
            results.append(SpotResult(
                info_state_key=state.info_state_key,
                variant=state.variant.value, error=str(exc),
            ))
            continue
        results.append(SpotResult(
            info_state_key=state.info_state_key,
            variant=state.variant.value,
            action=decision.action.value,
            verifiable=verifiable.grade(state, decision),
            judge=judge.grade(state, decision) if judge else None,
        ))
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_runner.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pokereval/eval/runner.py tests/eval/test_runner.py
git commit -m "feat: eval runner (render -> complete -> parse -> grade)"
```

---

### Task 11: Leaderboard aggregation

**Files:**
- Create: `src/pokereval/eval/leaderboard.py`
- Test: `tests/eval/test_leaderboard.py`

**Interfaces:**
- Consumes: `SpotResult` (runner); `exploitability_of_choices` (solver, optional/lazy).
- Produces:
  - `class ModelSummary(BaseModel)`: `model: str`, `n: int`, `n_errors: int`, `mean_verifiable: float`, `exact_match_rate: float`, `mean_judge: float | None`, `exploitability: float | None`.
  - `summarize(model: str, results: list[SpotResult]) -> ModelSummary` — aggregates; `exploitability` left `None` here (filled by CLI when `pyspiel` available).
  - `render_markdown(summaries: list[ModelSummary]) -> str` — a Markdown leaderboard table sorted by `mean_verifiable` descending.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_leaderboard.py
from pokereval.interface.types import Score
from pokereval.eval.runner import SpotResult
from pokereval.eval.leaderboard import summarize, render_markdown, ModelSummary

def _results():
    return [
        SpotResult(variant="leduc", action="raise",
                   verifiable=Score(value=0.7, detail={"exact_match": True})),
        SpotResult(variant="leduc", action="fold",
                   verifiable=Score(value=0.1, detail={"exact_match": False})),
        SpotResult(variant="leduc", error="no action"),
    ]

def test_summarize_computes_means_and_error_count():
    s = summarize("modelA", _results())
    assert s.n == 3
    assert s.n_errors == 1
    assert abs(s.mean_verifiable - 0.4) < 1e-9      # (0.7 + 0.1) / 2 graded
    assert abs(s.exact_match_rate - 0.5) < 1e-9
    assert s.mean_judge is None

def test_render_markdown_sorts_and_tabulates():
    a = ModelSummary(model="A", n=2, n_errors=0, mean_verifiable=0.3,
                     exact_match_rate=0.5, mean_judge=None, exploitability=None)
    b = ModelSummary(model="B", n=2, n_errors=0, mean_verifiable=0.8,
                     exact_match_rate=1.0, mean_judge=None, exploitability=None)
    md = render_markdown([a, b])
    assert md.index("B") < md.index("A")            # higher score first
    assert "| Model |" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_leaderboard.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pokereval/eval/leaderboard.py
from pydantic import BaseModel
from .runner import SpotResult


class ModelSummary(BaseModel):
    model: str
    n: int
    n_errors: int
    mean_verifiable: float
    exact_match_rate: float
    mean_judge: float | None = None
    exploitability: float | None = None


def summarize(model: str, results: list[SpotResult]) -> ModelSummary:
    graded = [r for r in results if r.verifiable is not None]
    judged = [r for r in results if r.judge is not None]
    n_errors = sum(1 for r in results if r.error is not None)
    mean_verifiable = (
        sum(r.verifiable.value for r in graded) / len(graded) if graded else 0.0
    )
    exact = [r for r in graded if r.verifiable.detail.get("exact_match") is not None]
    exact_match_rate = (
        sum(1 for r in exact if r.verifiable.detail["exact_match"]) / len(exact)
        if exact else 0.0
    )
    mean_judge = (
        sum(r.judge.value for r in judged) / len(judged) if judged else None
    )
    return ModelSummary(
        model=model, n=len(results), n_errors=n_errors,
        mean_verifiable=mean_verifiable, exact_match_rate=exact_match_rate,
        mean_judge=mean_judge,
    )


def render_markdown(summaries: list[ModelSummary]) -> str:
    rows = sorted(summaries, key=lambda s: s.mean_verifiable, reverse=True)
    header = "| Model | N | Errors | Mean Verifiable | Exact-Match | Mean Judge | Exploitability |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for s in rows:
        judge = f"{s.mean_judge:.3f}" if s.mean_judge is not None else "—"
        expl = f"{s.exploitability:.4f}" if s.exploitability is not None else "—"
        lines.append(
            f"| {s.model} | {s.n} | {s.n_errors} | {s.mean_verifiable:.3f} | "
            f"{s.exact_match_rate:.3f} | {judge} | {expl} |"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_leaderboard.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pokereval/eval/leaderboard.py tests/eval/test_leaderboard.py
git commit -m "feat: leaderboard aggregation and markdown rendering"
```

---

### Task 12: CLI entrypoint & full-suite smoke test

**Files:**
- Create: `src/pokereval/cli.py`
- Test: `tests/eval/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `build_spotset(variant: GameVariant, holdem_path: str, iterations: int) -> SpotSet` — dispatches to Kuhn/Leduc builder or Hold'em loader.
  - `run_for_client(client, spotset, verifiable, judge) -> ModelSummary` — runs eval, summarizes, and (Kuhn/Leduc only, when `pyspiel` present) fills `exploitability` via `exploitability_of_choices` using each result's chosen os-action.
  - `main(argv: list[str] | None = None) -> int` — argparse CLI: `run --variant {kuhn,leduc,holdem} [--iterations N] [--holdem-path P]`; uses a `FakeClient` by default so the CLI runs with zero credentials; prints the Markdown leaderboard.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_cli.py
from pokereval.interface.types import GameVariant, Action, State
from pokereval.models.client import FakeClient
from pokereval.graders.verifiable import VerifiableGrader
from pokereval.eval.spots import SpotSet
from pokereval.cli import run_for_client, main

def test_run_for_client_returns_summary():
    s = State(variant=GameVariant.LEDUC, hero_cards=["K"], pot=4.0, to_call=2.0,
              legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
              info_state_key="K1", meta={"os_legal": {"fold": 0, "call": 1, "raise": 2}})
    ss = SpotSet(spots=[s], nash_probs={"K1": {"0": 0.1, "1": 0.2, "2": 0.7}})
    summary = run_for_client(FakeClient("m", "ACTION: raise"), ss,
                             VerifiableGrader(ss.nash_probs), None)
    assert summary.model == "m"
    assert summary.mean_verifiable == 0.7

def test_main_runs_holdem_endtoend(capsys):
    code = main(["run", "--variant", "holdem", "--holdem-path", "data/holdem_spots.jsonl"])
    assert code == 0
    out = capsys.readouterr().out
    assert "| Model |" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_cli.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/pokereval/cli.py
import argparse
from .interface.types import GameVariant
from .models.client import FakeClient, LLMClient
from .graders.verifiable import VerifiableGrader
from .graders.judge import JudgeGrader
from .eval.spots import SpotSet, build_kuhn_leduc_spotset, load_holdem_spotset
from .eval.runner import run_eval
from .eval.leaderboard import ModelSummary, summarize, render_markdown


def build_spotset(variant: GameVariant, holdem_path: str, iterations: int) -> SpotSet:
    if variant is GameVariant.HOLDEM:
        return load_holdem_spotset(holdem_path)
    return build_kuhn_leduc_spotset(variant, iterations)


def run_for_client(
    client: LLMClient,
    spotset: SpotSet,
    verifiable: VerifiableGrader,
    judge: JudgeGrader | None,
) -> ModelSummary:
    results = run_eval(client, spotset, verifiable, judge)
    summary = summarize(client.name, results)
    if spotset.nash_probs:  # Kuhn/Leduc → compute whole-policy exploitability
        try:
            from .solver.nash import exploitability_of_choices
            variant = GameVariant(spotset.spots[0].variant)
            choices = {
                r.info_state_key: s.meta["os_legal"][r.action]
                for r, s in zip(results, spotset.spots)
                if r.action is not None and r.info_state_key is not None
            }
            summary.exploitability = exploitability_of_choices(variant, choices)
        except Exception:  # pyspiel missing or version mismatch → leave None
            pass
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokereval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--variant", choices=[v.value for v in GameVariant], required=True)
    run.add_argument("--iterations", type=int, default=2000)
    run.add_argument("--holdem-path", default="data/holdem_spots.jsonl")
    args = parser.parse_args(argv)

    variant = GameVariant(args.variant)
    spotset = build_spotset(variant, args.holdem_path, args.iterations)
    verifiable = VerifiableGrader(spotset.nash_probs)
    # Default to a FakeClient so the CLI runs without credentials.
    client = FakeClient("fake-fold", "ACTION: fold")
    summary = run_for_client(client, spotset, verifiable, None)
    print(render_markdown([summary]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Add `build_kuhn_leduc_spotset` and `load_holdem_spotset` to spots' import surface — already defined in Task 9.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_cli.py -v && pytest -q`
Expected: both CLI tests PASS; full suite green (OpenSpiel-marked tests skip if `pyspiel` absent).

- [ ] **Step 5: Commit**

```bash
git add src/pokereval/cli.py tests/eval/test_cli.py
git commit -m "feat: CLI entrypoint and end-to-end eval run"
```

---

## Definition of Done (Phase 1)

- `pytest -q` is green (pure-Python tests always; OpenSpiel tests pass when `pyspiel` is installed).
- `python -m pokereval.cli run --variant holdem` prints a leaderboard with no credentials.
- With `pyspiel` installed: `python -m pokereval.cli run --variant leduc` prints a leaderboard including an **exploitability** column.
- Swapping the default `FakeClient` for `AnthropicClient`/`OpenAIClient` (with confirmed model IDs) produces a real frontier-model leaderboard.
- **Follow-on (not blocking):** run ≥3 real models, write the Phase-1 findings note, and publish the environment to the Prime Intellect Environments Hub (per spec §4, Phase 1).

## Self-Review Notes

- **Spec coverage:** environment wrapper (Tasks 5–6, 9), verifiable + unverifiable graders (Tasks 7–8), eval pipeline + leaderboard (Tasks 10–12), frontier-model clients (Task 4), exploitability headline metric (Tasks 6, 12). Synthetic data / RLVR / agent / serving are deferred to Phases 2–3 per the spec's phasing.
- **Type consistency:** `os_legal` is `{action_value: os_id}` everywhere; `nash_probs` keys are `str(os_id)` everywhere (engine → solver → spots → grader → runner → CLI all agree).
- **No placeholders:** every code step is complete; OpenSpiel-internal calls that depend on version are marked **⚠ verify** with a behavioral test (exploitability ≈ 0) that catches wiring errors regardless of symbol names.
