"""HTTP-layer tests for the demo app.

These inject a hand-built analysis, so they need neither OpenSpiel nor Tinker
and run instantly — the slow solver path is covered separately in
test_policies.py.
"""
import pytest
from fastapi.testclient import TestClient

from pokereval.interface.types import GameVariant
from pokereval.demo.policies import FindingAnalysis, SpotView
from pokereval.demo.server import create_app


@pytest.fixture
def client():
    analysis = FindingAnalysis(
        variant=GameVariant.LEDUC,
        n_info_states=936,
        collapsed_exploitability=1.75,
        nash_exploitability=0.0,
        spots=[
            SpotView(
                info_key="Q:cr",
                hero="Q",
                board="K",
                pot=8,
                round_num=2,
                legal_actions=["fold", "call", "raise"],
                nash_probs={"fold": 0.33, "call": 0.67, "raise": 0.0},
                collapsed_action="fold",
                nash_prob_of_collapsed=0.33,
                nash_mass_missed=0.67,
                mixedness=0.33,
                tags=["mixed", "facing_bet"],
            )
        ],
    )
    return TestClient(create_app(analysis))


def test_summary_reports_live_and_recorded(client):
    body = client.get("/api/summary").json()
    assert body["collapsed_exploitability"] == 1.75
    assert body["nash_exploitability"] == 0.0
    assert body["n_info_states"] == 936
    # Real recorded numbers ride along for the headline.
    assert body["recorded"]["rlvr"]["exploitability"] == pytest.approx(1.75, abs=1e-9)
    assert body["recorded"]["base"]["exploitability"] == pytest.approx(1.0458, abs=1e-3)


def test_spots_endpoint_shape(client):
    spots = client.get("/api/spots").json()
    assert len(spots) == 1
    spot = spots[0]
    assert spot["collapsed_action"] == "fold"
    assert spot["nash_mass_missed"] == pytest.approx(0.67)
    assert set(spot["nash_probs"]) == {"fold", "call", "raise"}
    assert spot["hero"] == "Q"
    assert spot["board"] == "K"


def test_index_served_as_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "PokerEval" in resp.text
