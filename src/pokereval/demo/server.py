"""FastAPI app for the Phase-3 demo: the Phase-2 finding, served live.

The app is deliberately self-contained — one FastAPI process serving a single
static page plus two JSON endpoints. No node build, no API keys, no Tinker. The
expensive solver work is done once by `load_or_build_analysis` and cached to a
committed JSON artifact, so `make demo` starts instantly.

`create_app` takes a prebuilt `FindingAnalysis`, which keeps the HTTP layer
solver-free and instant to test.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from ..interface.types import GameVariant
from .data import RecordedRun, load_recorded_run
from .policies import FindingAnalysis, build_analysis

_HERE = Path(__file__).resolve().parent
_INDEX = _HERE / "static" / "index.html"
_CACHE = _HERE / "analysis_leduc.json"


def load_or_build_analysis(
    rebuild: bool = False,
    iterations: int = 2000,
    cache: Path = _CACHE,
) -> FindingAnalysis:
    """Return the cached analysis, building (and caching) it if needed."""
    if not rebuild and cache.exists():
        return FindingAnalysis.from_dict(json.loads(cache.read_text()))
    analysis = build_analysis(GameVariant.LEDUC, iterations=iterations)
    cache.write_text(json.dumps(analysis.to_dict(), indent=2))
    return analysis


def create_app(
    analysis: FindingAnalysis,
    recorded: RecordedRun | None = None,
) -> FastAPI:
    """Build the FastAPI app around a prebuilt analysis."""
    recorded = recorded or load_recorded_run()
    app = FastAPI(title="PokerEval — Phase 2 finding, live")

    @app.get("/api/summary")
    def summary() -> dict:
        return {
            "variant": analysis.variant.value,
            "n_info_states": analysis.n_info_states,
            "n_display_spots": len(analysis.spots),
            "collapsed_exploitability": analysis.collapsed_exploitability,
            "nash_exploitability": analysis.nash_exploitability,
            "recorded": recorded.to_dict(),
        }

    @app.get("/api/spots")
    def spots() -> list[dict]:
        return [
            {
                "info_key": s.info_key,
                "hero": s.hero,
                "board": s.board,
                "pot": s.pot,
                "round_num": s.round_num,
                "legal_actions": s.legal_actions,
                "nash_probs": s.nash_probs,
                "collapsed_action": s.collapsed_action,
                "nash_prob_of_collapsed": s.nash_prob_of_collapsed,
                "nash_mass_missed": s.nash_mass_missed,
                "mixedness": s.mixedness,
                "tags": s.tags,
            }
            for s in analysis.spots
        ]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX.read_text()

    return app
