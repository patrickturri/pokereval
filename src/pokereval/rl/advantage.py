from __future__ import annotations


def group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    n = len(rewards)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    if std < eps:
        return [0.0] * n
    return [(r - mean) / (std + eps) for r in rewards]
