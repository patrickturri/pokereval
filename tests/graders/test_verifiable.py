"""Tests for VerifiableGrader."""
import pytest
from pokereval.interface.types import State, Decision, Action, GameVariant
from pokereval.graders.verifiable import VerifiableGrader


def _leduc_state():
    return State(
        variant=GameVariant.LEDUC,
        hero_cards=["Js"],
        pot=4.0,
        to_call=2.0,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        info_state_key="K1",
        meta={"os_legal": {"fold": 0, "call": 1, "raise": 2}},
    )


def _holdem_state():
    return State(
        variant=GameVariant.HOLDEM,
        hero_cards=["As", "Kd"],
        board=["2h", "3d", "4c"],
        pot=100.0,
        to_call=10.0,
        legal_actions=[Action.FOLD, Action.CALL, Action.RAISE],
        meta={"reference_action": "call"},
    )


class TestVerifiableGraderNash:
    """Test Nash probability grading (Kuhn/Leduc with info_state_key)."""

    def test_exact_match_best_action(self):
        """When chosen action matches the max Nash probability."""
        nash_probs = {
            "K1": {"0": 0.3, "1": 0.7, "2": 0.0},
        }
        grader = VerifiableGrader(nash_probs)
        state = _leduc_state()
        decision = Decision(action=Action.CALL)
        score = grader.grade(state, decision)
        assert score.value == 0.7
        assert score.detail["kind"] == "nash"
        assert score.detail["nash_prob"] == 0.7
        assert score.detail["exact_match"] is True

    def test_non_exact_match_suboptimal_action(self):
        """When chosen action does not match the max Nash probability."""
        nash_probs = {
            "K1": {"0": 0.3, "1": 0.7, "2": 0.0},
        }
        grader = VerifiableGrader(nash_probs)
        state = _leduc_state()
        decision = Decision(action=Action.FOLD)
        score = grader.grade(state, decision)
        assert score.value == 0.3
        assert score.detail["kind"] == "nash"
        assert score.detail["nash_prob"] == 0.3
        assert score.detail["exact_match"] is False

    def test_ungradeable_missing_action_mapping(self):
        """When action cannot be mapped to os_id."""
        nash_probs = {
            "K1": {"0": 0.3, "1": 0.7, "2": 0.0},
        }
        grader = VerifiableGrader(nash_probs)
        state = _leduc_state()
        state.meta["os_legal"] = {}  # Missing mapping
        decision = Decision(action=Action.CALL)
        score = grader.grade(state, decision)
        assert score.value == 0.0
        assert score.detail["kind"] == "ungradeable"


class TestVerifiableGraderReference:
    """Test reference action grading (Hold'em without info_state_key)."""

    def test_reference_match(self):
        """When decision matches reference action."""
        grader = VerifiableGrader({})
        state = _holdem_state()
        decision = Decision(action=Action.CALL)
        score = grader.grade(state, decision)
        assert score.value == 1.0
        assert score.detail["kind"] == "reference"
        assert score.detail["reference"] == "call"

    def test_reference_mismatch(self):
        """When decision does not match reference action."""
        grader = VerifiableGrader({})
        state = _holdem_state()
        decision = Decision(action=Action.FOLD)
        score = grader.grade(state, decision)
        assert score.value == 0.0
        assert score.detail["kind"] == "reference"
        assert score.detail["reference"] == "call"
