from pokereval.eval.leaderboard import ModelSummary, render_markdown, summarize
from pokereval.eval.runner import SpotResult
from pokereval.interface.types import Score


def _results():
    return [
        SpotResult(
            variant="leduc",
            action="raise",
            verifiable=Score(value=0.7, detail={"exact_match": True}),
        ),
        SpotResult(
            variant="leduc",
            action="fold",
            verifiable=Score(value=0.1, detail={"exact_match": False}),
        ),
        SpotResult(variant="leduc", error="no action"),
    ]


def test_summarize_computes_means_and_error_count():
    s = summarize("modelA", _results())
    assert s.n == 3
    assert s.n_errors == 1
    assert abs(s.mean_verifiable - 0.4) < 1e-9
    assert abs(s.exact_match_rate - 0.5) < 1e-9
    assert s.mean_judge is None


def test_render_markdown_sorts_and_tabulates():
    a = ModelSummary(
        model="A",
        n=2,
        n_errors=0,
        mean_verifiable=0.3,
        exact_match_rate=0.5,
        mean_judge=None,
        exploitability=None,
    )
    b = ModelSummary(
        model="B",
        n=2,
        n_errors=0,
        mean_verifiable=0.8,
        exact_match_rate=1.0,
        mean_judge=None,
        exploitability=None,
    )
    md = render_markdown([a, b])
    assert md.index("B") < md.index("A")
    assert "| Model |" in md
