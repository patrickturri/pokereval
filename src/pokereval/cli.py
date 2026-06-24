"""Command-line entrypoint: ``python -m pokereval.cli run --variant ...``.

By default the CLI evaluates a credential-free :class:`FakeClient` so the whole
pipeline runs end-to-end with no API keys. Swap in ``AnthropicClient`` /
``OpenAIClient`` (with confirmed model IDs) to produce a real leaderboard.
"""

import argparse

from .eval.leaderboard import ModelSummary, render_markdown, summarize
from .eval.runner import run_eval
from .eval.spots import SpotSet, build_kuhn_leduc_spotset, load_holdem_spotset
from .graders.judge import JudgeGrader
from .graders.verifiable import VerifiableGrader
from .interface.types import GameVariant
from .models.client import FakeClient, LLMClient


def build_spotset(variant: GameVariant, holdem_path: str, iterations: int) -> SpotSet:
    if variant is GameVariant.HOLDEM:
        return load_holdem_spotset(holdem_path)
    return build_kuhn_leduc_spotset(variant, iterations)


def run_for_client(
    client: LLMClient,
    spotset: SpotSet,
    verifiable: VerifiableGrader,
    judge: JudgeGrader | None,
) -> ModelSummary:
    """Run the eval, summarize, and (Kuhn/Leduc) fill in whole-policy exploitability."""
    results = run_eval(client, spotset, verifiable, judge)
    summary = summarize(client.name, results)
    if spotset.nash_probs:  # Kuhn/Leduc → compute whole-policy exploitability
        try:
            from .solver.nash import exploitability_of_choices

            variant = GameVariant(spotset.spots[0].variant)
            choices = {
                r.info_state_key: s.meta["os_legal"][r.action]
                for r, s in zip(results, spotset.spots)
                if r.action is not None and r.info_state_key is not None
            }
            summary.exploitability = exploitability_of_choices(variant, choices)
        except Exception:  # pyspiel missing or version mismatch → leave None
            pass
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokereval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run the eval suite and print a leaderboard")
    run.add_argument(
        "--variant", choices=[v.value for v in GameVariant], required=True
    )
    run.add_argument("--iterations", type=int, default=2000)
    run.add_argument("--holdem-path", default="data/holdem_spots.jsonl")
    args = parser.parse_args(argv)

    variant = GameVariant(args.variant)
    spotset = build_spotset(variant, args.holdem_path, args.iterations)
    verifiable = VerifiableGrader(spotset.nash_probs)
    # Default to a FakeClient so the CLI runs without credentials.
    client = FakeClient("fake-fold", "ACTION: fold")
    summary = run_for_client(client, spotset, verifiable, None)
    print(render_markdown([summary]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
