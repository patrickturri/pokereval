"""PokerEval demo: the Phase-2 finding as a shareable before/after artifact.

Two modes:

  Offline (default, credential-free — no API keys, no Tinker account):
      python scripts/demo.py                       # print the before/after write-up
      python scripts/demo.py --out demo.md         # ...and save it as Markdown

    Reconstructs the collapsed RLVR policy from the CFR solver, recomputes its
    exploitability *live* (reproducing the recorded 1.75), and renders it next to
    the real base-vs-RLVR run metrics — decision by decision, showing where a
    deterministic policy departs from the mixed Nash equilibrium Leduc requires.

  Live (needs a funded Tinker account; the base model sampled directly):
      set -a; . ./.env; set +a
      python scripts/demo.py --live --n-spots 6 --samples 24

    For a handful of mixed Leduc spots, prints the solver's Nash action
    distribution beside the model's sampled distribution.
"""
import argparse

from pokereval.demo.artifact import build_offline_artifact


def _fmt(dist):
    return ", ".join(f"{a}={p:.0%}" for a, p in sorted(dist.items(), key=lambda kv: -kv[1]))


def run_offline(args) -> None:
    artifact = build_offline_artifact(
        variant=args.variant,
        iterations=args.iterations,
        n_spots=args.n_spots,
    )
    if args.out:
        with open(args.out, "w") as f:
            f.write(artifact)
        print(f"wrote before/after artifact → {args.out}")
    else:
        print(artifact)


def run_live(args) -> None:
    # Heavy/credentialed imports are deferred so the offline path needs neither
    # Tinker nor a network — importing this script offline must not fail.
    import tinker  # noqa: F401
    from pokereval.interface.types import GameVariant
    from pokereval.interface.prompts import render_state, parse_decision, ParseError
    from pokereval.synth.generator import build_labeled_spots
    from pokereval.rl.config import RLConfig
    from pokereval.rl.backend import TinkerBackend

    def model_distribution(sampler, spot, samples):
        comps = sampler.sample(render_state(spot.state), n=samples, temperature=1.0)
        counts = {}
        for c in comps:
            try:
                dec = parse_decision(c.text, spot.state.legal_actions)
            except ParseError:
                counts["<unparsed>"] = counts.get("<unparsed>", 0) + 1
                continue
            counts[dec.action.value] = counts.get(dec.action.value, 0) + 1
        return {a: n / samples for a, n in counts.items()}

    spots = build_labeled_spots(GameVariant(args.variant), iterations=args.iterations)
    shown = [s for s in spots if args.tag in s.tags][: args.n_spots]

    backend = TinkerBackend(RLConfig(base_model=args.base_model, max_tokens=32))
    sampler = backend.sampler()

    agree = 0
    for i, spot in enumerate(shown, 1):
        nash = spot.nash_action_probs
        model = model_distribution(sampler, spot, args.samples)
        nash_top = max(nash, key=nash.get)
        model_top = max((k for k in model if k != "<unparsed>"), key=model.get, default=None)
        match = nash_top == model_top
        agree += match
        print(f"\n── Spot {i}  [tags: {', '.join(spot.tags)}] ──")
        print(f"   {render_state(spot.state).splitlines()[3]}")  # history line
        print(f"   legal: {[a.value for a in spot.state.legal_actions]}")
        print(f"   SOLVER (Nash): {_fmt(nash)}")
        print(f"   MODEL        : {_fmt(model)}")
        print(f"   top action match: {'YES' if match else 'NO'} "
              f"(solver={nash_top}, model={model_top})")
    print(f"\nTop-action agreement: {agree}/{len(shown)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true",
                    help="Sample a real model on Tinker (needs a funded account).")
    ap.add_argument("--variant", default="leduc")
    ap.add_argument("--iterations", type=int, default=2000,
                    help="CFR iterations for the solver reference.")
    ap.add_argument("--n-spots", type=int, default=6)
    ap.add_argument("--out", default=None,
                    help="Offline mode: write the Markdown artifact to this path.")
    # Live-only options:
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--tag", default="mixed", help="Live: show spots with this tag.")
    args = ap.parse_args()

    if args.live:
        run_live(args)
    else:
        run_offline(args)


if __name__ == "__main__":
    main()
