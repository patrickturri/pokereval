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
