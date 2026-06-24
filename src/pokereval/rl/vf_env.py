from __future__ import annotations
from pokereval.interface.prompts import render_state
from pokereval.rl.reward import spot_reward
from pokereval.synth.types import LabeledSpot


def build_vf_env(spots: list[LabeledSpot]):
    """Build a verifiers SingleTurnEnv whose reward is the Nash-prob verifiable reward."""
    try:
        import verifiers as vf
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            "verifiers (+datasets) required for build_vf_env; install the 'rl' extra"
        ) from e

    rows = [{"question": render_state(s.state), "info": {"idx": i}} for i, s in enumerate(spots)]
    dataset = Dataset.from_list(rows)

    def poker_reward(completion, info, **kwargs) -> float:
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        return spot_reward(spots[info["idx"]], text)[0]

    rubric = vf.Rubric(funcs=[poker_reward], weights=[1.0])
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
