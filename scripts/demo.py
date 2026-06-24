"""Human-readable demo: model's poker strategy vs the solver's, spot by spot.

For a handful of representative Leduc spots it prints the situation, the solver's
Nash action distribution, and the model's sampled distribution — so you can see
*where* the model agrees with game-theory-optimal play and where it doesn't.

Usage (live, base model):
    set -a; . ./.env; set +a; python scripts/demo.py --n-spots 6 --samples 24
"""
import argparse
import tinker
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


def fmt(dist):
    return ", ".join(f"{a}={p:.0%}" for a, p in sorted(dist.items(), key=lambda kv: -kv[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n-spots", type=int, default=6)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--tag", default="mixed", help="Show spots with this tag (mixed/facing_bet).")
    args = ap.parse_args()

    spots = build_labeled_spots(GameVariant.LEDUC, iterations=2000)
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
        print(f"   SOLVER (Nash): {fmt(nash)}")
        print(f"   MODEL        : {fmt(model)}")
        print(f"   top action match: {'YES' if match else 'NO'} "
              f"(solver={nash_top}, model={model_top})")
    print(f"\nTop-action agreement: {agree}/{len(shown)}")


if __name__ == "__main__":
    main()
