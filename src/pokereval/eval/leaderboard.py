"""Aggregate per-spot results into a per-model summary and a Markdown table."""

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
    """Reduce a model's per-spot results to means and rates.

    ``exploitability`` is left ``None`` here and filled in by the CLI when
    ``pyspiel`` is available, since it needs the whole-policy view of the game.
    """
    graded = [r for r in results if r.verifiable is not None]
    judged = [r for r in results if r.judge is not None]
    n_errors = sum(1 for r in results if r.error is not None)
    mean_verifiable = (
        sum(r.verifiable.value for r in graded) / len(graded) if graded else 0.0
    )
    exact = [r for r in graded if r.verifiable.detail.get("exact_match") is not None]
    exact_match_rate = (
        sum(1 for r in exact if r.verifiable.detail["exact_match"]) / len(exact)
        if exact
        else 0.0
    )
    mean_judge = sum(r.judge.value for r in judged) / len(judged) if judged else None
    return ModelSummary(
        model=model,
        n=len(results),
        n_errors=n_errors,
        mean_verifiable=mean_verifiable,
        exact_match_rate=exact_match_rate,
        mean_judge=mean_judge,
    )


def render_markdown(summaries: list[ModelSummary]) -> str:
    """Render summaries as a Markdown table sorted by mean verifiable score."""
    rows = sorted(summaries, key=lambda s: s.mean_verifiable, reverse=True)
    header = (
        "| Model | N | Errors | Mean Verifiable | Exact-Match | "
        "Mean Judge | Exploitability |"
    )
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
