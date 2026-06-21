import argparse
from .interface.types import GameVariant
from .models.client import FakeClient, AnthropicClient, OpenAIClient, LLMClient
from .graders.verifiable import VerifiableGrader
from .graders.judge import JudgeGrader
from .eval.spots import SpotSet, build_kuhn_leduc_spotset, load_holdem_spotset
from .eval.runner import run_eval
from .eval.leaderboard import ModelSummary, summarize, render_markdown

# Default model ids per provider (override with --model). Anthropic default is
# the most capable current Opus; OpenAI has no single obvious default, so it is
# required when --provider openai is selected.
_DEFAULT_MODELS = {"anthropic": "claude-opus-4-8"}


def build_client(provider: str, model: str, name: str | None = None) -> LLMClient:
    """Construct an LLMClient for the given provider.

    `fake` is credential-free and deterministic (the default), so the CLI runs
    without any API keys. `anthropic`/`openai` lazily construct real SDK clients.
    """
    provider = provider.lower()
    if provider == "fake":
        return FakeClient(name or "fake-fold", "ACTION: fold")
    if provider == "anthropic":
        return AnthropicClient(name or model, model)
    if provider == "openai":
        return OpenAIClient(name or model, model)
    raise ValueError(
        f"Unknown provider {provider!r} (expected one of: fake, anthropic, openai)"
    )


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
    run = sub.add_parser("run")
    run.add_argument("--variant", choices=[v.value for v in GameVariant], required=True)
    run.add_argument("--iterations", type=int, default=2000)
    run.add_argument("--holdem-path", default="data/holdem_spots.jsonl")
    run.add_argument(
        "--provider",
        choices=["fake", "anthropic", "openai"],
        default="fake",
        help="Model provider. 'fake' (default) runs credential-free.",
    )
    run.add_argument(
        "--model",
        default=None,
        help="Model id. Defaults to claude-opus-4-8 for anthropic; required for openai.",
    )
    run.add_argument(
        "--name",
        default=None,
        help="Label for this model in the leaderboard (defaults to the model id).",
    )
    run.add_argument(
        "--judge",
        action="store_true",
        help="Also score reasoning quality with an LLM judge.",
    )
    run.add_argument(
        "--judge-provider",
        choices=["fake", "anthropic", "openai"],
        default="anthropic",
        help="Provider for the judge model (default: anthropic).",
    )
    run.add_argument(
        "--judge-model",
        default="claude-opus-4-8",
        help="Judge model id (default: claude-opus-4-8).",
    )
    run.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format. 'json' emits the raw summaries for archival.",
    )
    args = parser.parse_args(argv)

    variant = GameVariant(args.variant)
    spotset = build_spotset(variant, args.holdem_path, args.iterations)
    verifiable = VerifiableGrader(spotset.nash_probs)

    model = args.model or _DEFAULT_MODELS.get(args.provider)
    if args.provider != "fake" and not model:
        parser.error(f"--model is required for provider {args.provider!r}")
    client = build_client(args.provider, model or "", args.name)

    judge = None
    if args.judge:
        judge_client = build_client(args.judge_provider, args.judge_model)
        judge = JudgeGrader(judge_client)

    summary = run_for_client(client, spotset, verifiable, judge)
    if args.format == "json":
        import json

        print(json.dumps([summary.model_dump() for summary in [summary]], indent=2))
    else:
        print(render_markdown([summary]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
