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


def test_exact_match_rate_is_none_when_not_applicable():
    # Hold'em reference grading carries no Nash 'exact_match' detail, so the
    # rate must be None (rendered as "—"), not a misleading 0.0.
    results = [
        SpotResult(variant="holdem", action="bet",
                   verifiable=Score(value=1.0, detail={})),
        SpotResult(variant="holdem", action="check",
                   verifiable=Score(value=0.0, detail={})),
    ]
    s = summarize("modelA", results)
    assert s.mean_verifiable == 0.5
    assert s.exact_match_rate is None


def test_render_shows_dash_for_none_exact_match():
    s = ModelSummary(model="H", n=2, n_errors=0, mean_verifiable=1.0,
                     exact_match_rate=None, mean_judge=None, exploitability=None)
    md = render_markdown([s])
    # The exact-match cell renders as an em dash, not 0.000.
    assert "| 1.000 | — |" in md

def test_render_markdown_sorts_and_tabulates():
    a = ModelSummary(model="A", n=2, n_errors=0, mean_verifiable=0.3,
                     exact_match_rate=0.5, mean_judge=None, exploitability=None)
    b = ModelSummary(model="B", n=2, n_errors=0, mean_verifiable=0.8,
                     exact_match_rate=1.0, mean_judge=None, exploitability=None)
    md = render_markdown([a, b])
    assert md.index("B") < md.index("A")            # higher score first
    assert "| Model |" in md
