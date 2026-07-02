import argparse
from .interface.types import GameVariant
from .models.client import (
    FakeClient,
    AnthropicClient,
    OpenAIClient,
    GeminiClient,
    LLMClient,
)
from .graders.verifiable import VerifiableGrader
from .graders.judge import JudgeGrader
from .eval.spots import SpotSet, build_kuhn_leduc_spotset, load_holdem_spotset
from .eval.runner import run_eval
from .eval.leaderboard import ModelSummary, summarize, render_markdown

# Default model ids per provider (override with --model). Anthropic default is
# the most capable current Opus; OpenAI has no single obvious default, so it is
# required when --provider openai is selected.
_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "gemini": "gemini-3.1-pro-preview",
}


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
    if provider == "gemini":
        return GeminiClient(name or model, model)
    raise ValueError(
        f"Unknown provider {provider!r} (expected: fake, anthropic, openai, gemini)"
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
    max_workers: int = 1,
) -> ModelSummary:
    results = run_eval(client, spotset, verifiable, judge, max_workers=max_workers)
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
        choices=["fake", "anthropic", "openai", "gemini"],
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
        choices=["fake", "anthropic", "openai", "gemini"],
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
    run.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent model calls (thread pool). >1 speeds up real-model runs.",
    )

    synth = sub.add_parser(
        "synth", help="Generate CFR/Nash-labeled synthetic spots (Kuhn/Leduc)."
    )
    synth.add_argument("--variant", choices=["kuhn", "leduc"], required=True)
    synth.add_argument("--iterations", type=int, default=2000)
    synth.add_argument("--out", required=True, help="Output .jsonl path.")
    synth.add_argument(
        "--tag",
        default=None,
        help="Only emit spots carrying this failure-mode tag (e.g. 'mixed').",
    )

    rl = sub.add_parser("rl-train", help="RLVR train a model on Kuhn/Leduc via Tinker.")
    rl.add_argument("--variant", choices=["kuhn", "leduc"], default="leduc")
    rl.add_argument("--preset", choices=["smoke", "real"], default="smoke")
    rl.add_argument("--live", action="store_true", help="Use the live Tinker backend.")
    rl.add_argument("--base-model", default=None)
    rl.add_argument("--wandb", action="store_true")
    # Optional overrides on top of the preset (None = keep preset value).
    rl.add_argument("--steps", type=int, default=None)
    rl.add_argument("--lr", type=float, default=None)
    rl.add_argument("--temperature", type=float, default=None)
    rl.add_argument("--group-size", type=int, default=None)
    rl.add_argument("--reward-mode", choices=["kl_nash", "nash_prob"], default=None)
    rl.add_argument("--eval-samples", type=int, default=None)

    sp = sub.add_parser("rl-selfplay", help="Self-play RL on Leduc via Tinker (Phase 3).")
    sp.add_argument("--variant", choices=["kuhn", "leduc"], default="leduc")
    sp.add_argument("--preset", choices=["smoke", "real"], default="smoke")
    sp.add_argument("--iterations", type=int, default=2000, help="CFR iterations for eval spots.")
    sp.add_argument("--live", action="store_true", help="Use the live Tinker backend.")
    sp.add_argument(
        "--tabular",
        action="store_true",
        help="Credential-free tabular self-play: a real offline learning result "
        "(exploitability drops), no Tinker/mock. Ignores --live.",
    )
    sp.add_argument("--base-model", default=None)
    sp.add_argument("--wandb", action="store_true")
    sp.add_argument("--steps", type=int, default=None)
    sp.add_argument("--deals", type=int, default=None)
    sp.add_argument("--hands", type=int, default=None)
    sp.add_argument("--refresh-every", type=int, default=None)
    sp.add_argument("--lr", type=float, default=None)
    sp.add_argument("--temperature", type=float, default=None)

    md = sub.add_parser(
        "metric-divergence",
        help="Credential-free: action-match vs. exploitability across a policy panel.",
    )
    md.add_argument("--variant", choices=["kuhn", "leduc"], default="leduc")
    md.add_argument("--iterations", type=int, default=2000)
    md.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format. 'json' emits the raw panel rows for archival.",
    )

    demo = sub.add_parser(
        "demo", help="Serve the Phase-2 finding as a local web demo (offline, no keys)."
    )
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8000)
    demo.add_argument("--iterations", type=int, default=2000, help="CFR iterations when (re)building.")
    demo.add_argument("--rebuild", action="store_true", help="Recompute the cached solver analysis.")

    args = parser.parse_args(argv)
    if args.cmd == "synth":
        return _cmd_synth(args)
    if args.cmd == "rl-train":
        return _cmd_rl_train(args)
    if args.cmd == "rl-selfplay":
        return _cmd_rl_selfplay(args)
    if args.cmd == "demo":
        return _cmd_demo(args)
    if args.cmd == "metric-divergence":
        return _cmd_metric_divergence(args)
    return _cmd_run(args, parser)


def _preflight_billing(timeout: float = 30.0) -> bool:
    """Quick check that Tinker is reachable and not billing-blocked.

    The SDK retries 402s for ~1h, so we run the probe in a daemon thread and give
    up after ``timeout`` seconds — a slow/blocked balance reads as not-ok rather
    than hanging the whole run.
    """
    import concurrent.futures

    def _probe() -> bool:
        import tinker

        tinker.ServiceClient().get_server_capabilities()
        return True

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_probe).result(timeout=timeout)
    except Exception:
        return False


def _first_legal_action(prompt: str) -> str:
    """Pick the first action listed on the prompt's 'Legal actions:' line.

    Used by the offline (mock) backend so the heuristic policy always emits a
    legal action. (Leduc actions are check/raise/fold/call — there is no 'bet',
    and the word 'bet' appears in the instruction text, so substring checks fail.)
    """
    line = next(l for l in prompt.splitlines() if l.startswith("Legal actions:"))
    return line.split(":", 1)[1].split(",")[0].strip()


def _cmd_rl_train(args) -> int:
    import json
    import dataclasses
    from .interface.types import GameVariant
    from .synth.generator import build_labeled_spots
    from .rl.config import get_preset
    from .rl.dataset import split_spots
    from .rl.train import train

    cfg = get_preset(args.preset)
    overrides = {"variant": args.variant}
    if args.base_model:
        overrides["base_model"] = args.base_model
    if args.steps is not None:
        overrides["num_steps"] = args.steps
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.group_size is not None:
        overrides["group_size"] = args.group_size
    if args.reward_mode is not None:
        overrides["reward_mode"] = args.reward_mode
    if args.eval_samples is not None:
        overrides["eval_samples"] = args.eval_samples
    cfg = dataclasses.replace(cfg, **overrides)

    variant = GameVariant(args.variant)
    spots = build_labeled_spots(variant, cfg.iterations)
    train_spots, eval_spots = split_spots(spots, cfg.eval_split, cfg.seed)

    logger = None
    if args.wandb:
        try:
            import wandb

            wandb.init(project="pokereval-rlvr", config=dataclasses.asdict(cfg))
            logger = lambda d: wandb.log(d)  # noqa: E731
        except Exception as e:  # noqa: BLE001
            print(f"[wandb disabled: {e}]")

    if args.live:
        if not _preflight_billing():
            print(
                "[abort] Tinker is unreachable or billing-blocked (HTTP 402). "
                "Add funds at https://tinker-console.thinkingmachines.ai/billing/balance "
                "and retry. (Skipping to avoid a ~1h silent retry loop on a dead balance.)"
            )
            return 2
        from .rl.backend import TinkerBackend

        backend = TinkerBackend(cfg)
    else:
        from .rl.backend import FakeBackend
        from .rl.sampler import MockSamplingClient

        backend = FakeBackend(MockSamplingClient(action_for=_first_legal_action))

    result = train(cfg, train_spots, eval_spots, backend, logger=logger)
    payload = {
        "preset": args.preset,
        "variant": args.variant,
        "live": args.live,
        "step_rewards": result.step_rewards,
        "before": dataclasses.asdict(result.before) if result.before else None,
        "after": dataclasses.asdict(result.after) if result.after else None,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_rl_selfplay(args) -> int:
    import json
    import dataclasses
    from .interface.types import GameVariant
    from .synth.generator import build_labeled_spots
    from .rl.config import get_preset
    from .rl.dataset import split_spots
    from .rl.selfplay.train import selfplay_train

    # The tabular path is a credential-free, genuinely-learning self-play loop;
    # it picks tabular-tuned presets and never touches a backend.
    preset_name = args.preset
    if args.tabular:
        preset_name = "tabular" if args.preset == "real" else "tabular_smoke"
    cfg = get_preset(preset_name)
    overrides = {"variant": args.variant}
    if args.base_model:
        overrides["base_model"] = args.base_model
    if args.steps is not None:
        overrides["num_steps"] = args.steps
    if args.deals is not None:
        overrides["deals_per_step"] = args.deals
    if args.hands is not None:
        overrides["hands_per_deal"] = args.hands
    if args.refresh_every is not None:
        overrides["opponent_refresh_every"] = args.refresh_every
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    cfg = dataclasses.replace(cfg, **overrides)

    variant = GameVariant(args.variant)
    spots = build_labeled_spots(variant, args.iterations)
    train_spots, eval_spots = split_spots(spots, cfg.eval_split, cfg.seed)
    all_spots = list(train_spots) + list(eval_spots)

    logger = None
    if args.wandb:
        try:
            import wandb

            wandb.init(project="pokereval-rlvr-selfplay", config=dataclasses.asdict(cfg))
            logger = lambda d: wandb.log(d)  # noqa: E731
        except Exception as e:  # noqa: BLE001
            print(f"[wandb disabled: {e}]")

    if args.tabular:
        from .rl.selfplay.tabular import tabular_selfplay_train

        result = tabular_selfplay_train(cfg, all_spots, eval_spots, logger=logger)
        payload = {
            "preset": preset_name,
            "variant": args.variant,
            "mode": "tabular",
            "step_returns": result.step_returns,
            "before": dataclasses.asdict(result.before) if result.before else None,
            "after": dataclasses.asdict(result.after) if result.after else None,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.live:
        if not _preflight_billing():
            print(
                "[abort] Tinker is unreachable or billing-blocked (HTTP 402). "
                "Add funds and retry. (Skipping to avoid a ~1h silent retry loop.)"
            )
            return 2
        from .rl.backend import TinkerBackend

        backend = TinkerBackend(cfg)
    else:
        from .rl.backend import FakeBackend
        from .rl.sampler import MockSamplingClient

        backend = FakeBackend(MockSamplingClient(action_for=_first_legal_action))

    result = selfplay_train(cfg, all_spots, eval_spots, backend, logger=logger)
    payload = {
        "preset": args.preset,
        "variant": args.variant,
        "live": args.live,
        "step_returns": result.step_returns,
        "before": dataclasses.asdict(result.before) if result.before else None,
        "after": dataclasses.asdict(result.after) if result.after else None,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_demo(args) -> int:
    try:
        import uvicorn
    except ModuleNotFoundError:
        print("[demo] needs the demo extras: pip install -e '.[demo,openspiel]'")
        return 2
    from .demo.server import create_app, load_or_build_analysis

    if args.rebuild:
        print(f"[demo] computing CFR analysis ({args.iterations} iters); cached after first run…")
    analysis = load_or_build_analysis(rebuild=args.rebuild, iterations=args.iterations)
    app = create_app(analysis)
    print(f"[demo] serving the Phase-2 finding at http://{args.host}:{args.port}  (Ctrl-C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _cmd_metric_divergence(args) -> int:
    from .eval.metric_divergence import (
        evaluate_panel,
        render_panel_markdown,
        verdict,
    )

    try:
        rows = evaluate_panel(args.variant, iterations=args.iterations)
    except Exception as exc:  # pyspiel missing or version mismatch
        print(f"[metric-divergence] needs OpenSpiel: pip install -e '.[openspiel]' ({exc})")
        return 2

    if args.format == "json":
        import json
        print(json.dumps([r.model_dump() for r in rows], indent=2))
    else:
        print(render_panel_markdown(rows))
        print()
        print(verdict(rows))
    return 0


def _cmd_run(args, parser) -> int:
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

    summary = run_for_client(client, spotset, verifiable, judge, max_workers=args.workers)
    if args.format == "json":
        import json

        print(json.dumps([summary.model_dump()], indent=2))
    else:
        print(render_markdown([summary]))
    return 0


def _cmd_synth(args) -> int:
    import json
    from pathlib import Path
    from .synth.generator import build_labeled_spots, select_by_tag, to_jsonl_records

    variant = GameVariant(args.variant)
    spots = build_labeled_spots(variant, args.iterations)
    if args.tag:
        spots = select_by_tag(spots, args.tag)
    records = to_jsonl_records(spots)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"Wrote {len(records)} labeled spots to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
