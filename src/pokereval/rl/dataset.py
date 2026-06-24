from __future__ import annotations
import hashlib
from pokereval.synth.types import LabeledSpot


def _bucket(key: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_spots(
    spots: list[LabeledSpot], eval_frac: float, seed: int = 0
) -> tuple[list[LabeledSpot], list[LabeledSpot]]:
    """Deterministic split by info_state_key. Lowest-bucket eval_frac go to eval."""
    ordered = sorted(spots, key=lambda s: _bucket(s.state.info_state_key or "", seed))
    n_eval = round(len(ordered) * eval_frac)
    ev = ordered[:n_eval]
    train = ordered[n_eval:]
    return train, ev
